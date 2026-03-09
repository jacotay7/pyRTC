"""Adaptive optics loop control kernels and the main loop component.

This module contains the numerical update kernels and the high-level
``Loop`` component that turn measured residuals into new correction commands.
It is the control-plane heart of pyRTC: interaction matrices, control matrices,
integrators, and command dispatch all come together here.
"""

import os 
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ['NUMBA_NUM_THREADS'] = '1'

import math
import matplotlib.pyplot as plt
import numpy as np
import time
from typing import Any
from numba import jit

from pyRTC.logging_utils import get_logger
from pyRTC.Pipeline import gpu_torch_available, initExistingShm, launchComponent
from pyRTC.pyRTCComponent import pyRTCComponent
from pyRTC.utils import add_to_buffer, get_tmp_filepath, setFromConfig

logger = get_logger(__name__)

COMMON_CONDITIONING_LINES = (10.0, 100.0, 1e3, 1e4, 1e5, 1e6)

@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def leakyIntegratorNumba(slopes: np.ndarray, 
                         resconstructionMatrix: np.ndarray, 
                         oldCorrection: np.ndarray,
                         correction: np.ndarray,
                         leak: np.float32,
                         numActiveModes: int) -> np.ndarray:
    
    # Perform the matrix-vector multiplication using np.dot
    correction = np.dot(resconstructionMatrix, slopes)
    
    # Apply the leaky integrator formula with an unrolled loop
    for i in range(numActiveModes + 1):
        correction[i] = (1 - leak) * oldCorrection[i] - correction[i]
    
    # Zero out the rest of the correction vector
    for i in range(numActiveModes + 1, correction.size):
        correction[i] = 0.0
    
    return correction

def leakIntegratorGPU(slopes:np.ndarray, 
                                resconstructionMatrix:Any,
                                oldCorrection:np.ndarray,
                                leak:float,
                                numActiveModes:int
                                ):
    """Run the leaky-integrator control update on a CUDA-backed torch matrix."""

    if not gpu_torch_available():
        raise ImportError("leakIntegratorGPU requires PyTorch. Install with 'pip install pyRTC[gpu]' or 'pip install torch'.")

    import torch

    slopes_GPU = torch.tensor(slopes, device='cuda')
    correctionGPU = torch.matmul(resconstructionMatrix, slopes_GPU) 
    correctionGPU[numActiveModes:] = 0
    return np.subtract((1-leak)*oldCorrection, correctionGPU.cpu().numpy())

@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def compCorrection(CM=np.array([[]], dtype=np.float32),  
                    slopes=np.array([], dtype=np.float32)):
    """Apply a control matrix to a slope vector and return the correction."""

    return np.dot(CM,slopes)

@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def updateCorrection(correction=np.array([], dtype=np.float32), 
                     gCM=np.array([[]], dtype=np.float32),  
                     slopes=np.array([], dtype=np.float32)):
    """Update an existing correction using a pre-scaled control matrix."""

    return correction - np.dot(gCM,slopes)

# @jit(nopython=True)
# def updateCorrectionPerturb(correction=np.array([], dtype=np.float32),
#                             pertub=np.array([], dtype=np.float32),  
#                      gCM=np.array([[]], dtype=np.float32),  
#                      slopes=np.array([], dtype=np.float32)):
#     return correction - np.dot(gCM,slopes) + pertub

class Loop(pyRTCComponent):
    """
    Real-time controller that closes the adaptive optics loop.

    ``Loop`` reads the current residual signal from the slopes pipeline,
    combines that signal with the calibrated control model, and writes the next
    correction vector to the wavefront-corrector stream. It also owns the
    operator-facing calibration state used to load or build interaction and
    control matrices and to tune classical integrator settings.

    In day-to-day use, this is the component that embodies the chosen control
    law for the system.

    Config
    ------
    numDroppedModes : int, optional
        Number of modes to drop. Default is 0.
    gain : float, optional
        Gain for the integrator. Default is 0.1.
    leakyGain : float, optional
        Leaky integrator gain. Default is 0.0.
    hardwareDelay : float, optional
        Delay for the hardware. Default is 0.0.
    pokeAmp : float, optional
        Amplitude for poking. Default is 0.01.
    numItersIM : int, optional
        Number of iterations for interaction matrix computation. Default is 100.
    delay : int, optional
        Delay for corrections. Default is 0.
    IMMethod : str, optional
        Method for interaction matrix computation. Default is "push-pull".
    IMFile : str, optional
        File to save the interaction matrix. Default is "".
    pGain : float, optional
        Proportional gain for PID integrator. Default is 0.1.
    iGain : float, optional
        Integral gain for PID integrator. Default is 0.0.
    dGain : float, optional
        Derivative gain for PID integrator. Default is 0.0.
    controlLimits : list, optional
        Control limits for PID integrator. Default is [-inf, inf].
    integralLimits : list, optional
        Integral limits for PID integrator. Default is [-inf, inf].
    absoluteLimits : list, optional
        Absolute limits for corrections. Default is [-inf, inf].
    derivativeFilter : float, optional
        Filter for the derivative term. Default is 0.1.

    Attributes
    ----------
    conf : dict
        Loop configuration.
    name : str
        Name of the loop.
    signalDType : type
        Data type of the wavefront sensor signal.
    signalSize : int
        Size of the wavefront sensor signal.
    signalShm : ImageSHM
        Shared memory object for the wavefront sensor signal.
    nullSignal : numpy.ndarray
        Null signal.
    signal2DDType : type
        Data type of the 2D wavefront sensor signal.
    signal2DSize : int
        Size of the 2D wavefront sensor signal.
    signal2D_width : int
        Width of the 2D wavefront sensor signal.
    signal2D_height : int
        Height of the 2D wavefront sensor signal.
    wfcDType : type
        Data type of the wavefront corrector.
    numModes : int
        Number of modes in the wavefront corrector.
    wfcShm : ImageSHM
        Shared memory object for the wavefront corrector.
    numDroppedModes : int
        Number of dropped modes.
    numActiveModes : int
        Number of active modes.
    flat : numpy.ndarray
        Flat correction vector.
    IM : numpy.ndarray
        Interaction matrix.
    CM : numpy.ndarray
        Control matrix.
    gain : float
        Gain for the integrator.
    leakyGain : float
        Leaky integrator gain.
    perturbAmp : float
        Perturbation amplitude.
    hardwareDelay : float
        Delay for the hardware.
    pokeAmp : float
        Amplitude for poking.
    numItersIM : int
        Number of iterations for interaction matrix computation.
    delay : int
        Delay for corrections.
    IMMethod : str
        Method for interaction matrix computation.
    IMFile : str
        File to save the interaction matrix.
    pGain : float
        Proportional gain for PID integrator.
    iGain : float
        Integral gain for PID integrator.
    dGain : float
        Derivative gain for PID integrator.
    controlLimits : list
        Control limits for PID integrator.
    integralLimits : list
        Integral limits for PID integrator.
    absoluteLimits : list
        Absolute limits for corrections.
    derivativeFilter : float
        Filter for the derivative term.
    CMMethod : str
        Control-matrix inversion method. Supported values are ``svd`` and
        ``tikhonov``.
    conditioning : float or None
        Optional target conditioning number used to truncate small singular
        values when computing the control matrix.
    tikhonovReg : float
        Tikhonov regularization strength used when ``CMMethod`` is
        ``tikhonov``.
    integral : numpy.ndarray
        Integral term for PID integrator.
    previousWfError : numpy.ndarray
        Previous wavefront error.
    previousDerivative : numpy.ndarray
        Previous derivative term.
    controlOutput : numpy.ndarray
        Control output.
    """
    def __init__(self, conf) -> None:
        """
        Constructs all the necessary attributes for the Loop object.

        Parameters
        ----------
        conf : dict
            Configuration dictionary with the following keys
            wfs : dict
                Wavefront sensor configuration.
            wfc : dict
                Wavefront corrector configuration.
            loop : dict
                Loop configuration containing
                numDroppedModes : int, optional
                    Number of modes to drop. Default is 0.
                gain : float, optional
                    Gain for the integrator. Default is 0.1.
                leakyGain : float, optional
                    Leaky integrator gain. Default is 0.0.
                hardwareDelay : float, optional
                    Delay for the hardware. Default is 0.0.
                pokeAmp : float, optional
                    Amplitude for poking. Default is 0.01.
                numItersIM : int, optional
                    Number of iterations for interaction matrix computation. Default is 100.
                delay : int, optional
                    Delay for corrections. Default is 0.
                IMMethod : str, optional
                    Method for interaction matrix computation. Default is "push-pull".
                IMFile : str, optional
                    File to save the interaction matrix. Default is "".
                pGain : float, optional
                    Proportional gain for PID integrator. Default is 0.1.
                iGain : float, optional
                    Integral gain for PID integrator. Default is 0.0.
                dGain : float, optional
                    Derivative gain for PID integrator. Default is 0.0.
                controlLimits : list, optional
                    Control limits for PID integrator. Default is [-inf, inf].
                integralLimits : list, optional
                    Integral limits for PID integrator. Default is [-inf, inf].
                absoluteLimits : list, optional
                    Absolute limits for corrections. Default is [-inf, inf].
                derivativeFilter : float, optional
                    Filter for the derivative term. Default is 0.1.
        """
        try:
            super().__init__(conf)
            self.name = "Loop"
            self.conf = conf
        
        #Read wfs signal's metadata and open a stream to the shared memory
            self.signalShm, self.signalShape, self.signalDType = initExistingShm("signal", gpuDevice=self.gpuDevice)
            self.signalSize = int(np.prod(self.signalShape))
            self.nullSignal = np.zeros(self.signalShape, dtype=self.signalDType)

        #Read wfc metadata and open a stream to the shared memory
            self.wfcShm, self.wfcShape, self.wfcDType = initExistingShm("wfc", gpuDevice=self.gpuDevice)
            self.numModes = int(np.prod(self.wfcShape))

            self.numDroppedModes = setFromConfig(self.conf, "numDroppedModes", 0)
            self.numActiveModes = self.numModes - self.numDroppedModes
            self.flat = np.zeros(self.numModes, dtype=self.wfcDType)
            self.nullCorrection = np.zeros_like(self.flat)

            self.IM = np.zeros((self.signalSize, self.numModes),dtype=self.signalDType)
            self.CM = np.zeros((self.numModes, self.signalSize),dtype=self.signalDType)
            self.gain = setFromConfig(self.conf, "gain", 0.1)
            self.leakyGain = setFromConfig(self.conf, "leakyGain", 0.0)
            self.perturbAmp = 0
            self.hardwareDelay = setFromConfig(self.conf, "hardwareDelay", 0.0)
            self.pokeAmp = setFromConfig(self.conf, "pokeAmp", 1e-2)
            self.numItersIM = setFromConfig(self.conf, "numItersIM", 100)
            self.delay = setFromConfig(self.conf, "delay", 0)
            self.IMMethod = setFromConfig(self.conf, "IMMethod", "push-pull")
            self.IMFile = setFromConfig(self.conf, "IMFile", "")
            self.CMMethod = str(setFromConfig(self.conf, "CMMethod", "svd")).lower()
            conditioning = setFromConfig(self.conf, "conditioning", None)
            self.conditioning = None if conditioning is None else float(conditioning)
            self.tikhonovReg = float(setFromConfig(self.conf, "tikhonovReg", 0.0))
            self.lastSingularValues = np.array([], dtype=np.float64)
            self.lastRetainedSingularMask = np.array([], dtype=bool)
            self.lastSuggestedConditioning = None
            self.lastSingularValueFit = None
        
            self.clDocrime = False
            self.numItersDC = 0
            tmp2 = self.flat.copy().reshape(self.flat.size, 1)
            tmp = self.nullSignal.copy().reshape(self.nullSignal.size, 1)
            self.docrimeCross = np.zeros_like(tmp @ tmp2.T)
            self.docrimeAuto = np.zeros_like(tmp2 @ tmp2.T)
            self.docrimeBuffer = np.zeros((1 + self.delay, *tmp2.shape), dtype=self.wfcDType)
        
            self.pGain = setFromConfig(self.conf, "pGain", 0.1)
            self.iGain = setFromConfig(self.conf, "iGain", 0.0)
            self.dGain = setFromConfig(self.conf, "dGain", 0.0)
            self.controlLimits = setFromConfig(self.conf, "controlLimits", [-np.inf, np.inf])
            self.integralLimits = setFromConfig(self.conf, "integralLimits", [-np.inf, np.inf])
            self.absoluteLimits = setFromConfig(self.conf, "absoluteLimits", [-np.inf, np.inf])
            self.derivativeFilter = setFromConfig(self.conf, "derivativeFilter", 0.1)
            self.integral = 0

            self.previousWfError = np.zeros_like(self.wfcShm.read_noblock())
            self.previousDerivative = np.zeros_like(self.previousWfError)
            self.controlOutput = np.zeros_like(self.previousWfError)

            self.loadIM()
            self.logger.info("Initialized loop signalShape=%s wfcShape=%s numModes=%s", self.signalShape, self.wfcShape, self.numModes)
        except Exception:
            logger.exception("Failed to initialize loop")
            raise

        return

    @property
    def gain(self):
        return getattr(self, "_gain", 0.0)

    @gain.setter
    def gain(self, gain):
        self._gain = float(gain)
        if hasattr(self, "CM"):
            self.gCM = self._gain * self.CM

    def setGain(self, gain):
        """
        Set the integrator gain. Only needed for certain integrators.

        Parameters
        ----------
        gain : float
            Gain to set.
        """
        component_logger = getattr(self, "logger", logger)
        try:
            self.gain = gain
            component_logger.info("Set loop gain to %s", gain)
        except Exception:
            component_logger.exception("Failed to set loop gain to %s", gain)
            raise
        return

    def setPeturbAmp(self, amp):
        """
        Set the perturbation amplitude.

        Parameters
        ----------
        amp : float
            Amplitude to set.
        """
        component_logger = getattr(self, "logger", logger)
        try:
            self.perturbAmp = amp
            component_logger.info("Set perturbation amplitude to %s", amp)
        except Exception:
            component_logger.exception("Failed to set perturbation amplitude to %s", amp)
            raise
        return

    def pushPullIM(self):
        """
        Compute the interaction matrix using the push-pull method.
        """
        #For each mode
        for i in range(self.numModes):
            #Reset the correction
            correction = self.flat.copy()
            #Plus amplitude
            correction[i] = self.pokeAmp
            #Post a new shape to be made
            self.sendToWfc(correction)
            #Add some delay to ensure one-to-one
            time.sleep(self.hardwareDelay)
            #Burn the first new image since we were moving the DM during the exposure
            self.signalShm.read(RELEASE_GIL = True)
            #Average out N new WFS frames
            tmp_plus = np.zeros_like(self.IM[:,i])
            for n in range(self.numItersIM):
                tmp_plus += self.signalShm.read(RELEASE_GIL = True)
            tmp_plus /= self.numItersIM

            #Minus amplitude
            correction[i] = -self.pokeAmp
            #Post a new shape to be made
            self.sendToWfc(correction)
            #Add some delay to ensure one-to-one
            time.sleep(self.hardwareDelay)
            #Burn the first new image since we were moving the DM during the exposure
            self.signalShm.read(RELEASE_GIL = True)
            #Average out N new WFS frames
            tmp_minus = np.zeros_like(self.IM[:,i])
            for n in range(self.numItersIM):
                tmp_minus += self.signalShm.read(RELEASE_GIL = True)
            tmp_minus /= self.numItersIM

            #Compute the normalized difference
            self.IM[:,i] = (tmp_plus-tmp_minus)/(2*self.pokeAmp)

        return
    
    def docrimeIM(self):
        """
        Compute the interaction matrix using the DOCRIME method.
        """        
        #Send the flat command to the WFC
        self.flatten()

        #Get a correction to set the shape
        correction = self.flat.copy()
        correction = correction.reshape(correction.size,1)

        #Have a history of corrections
        # corrections = np.zeros((1+self.delay, *correction.shape), dtype=correction.dtype)

        #Get an initial slope reading to set shapes
        slopes = self.nullSignal.copy()
        slopes = slopes.reshape(slopes.size,1)
        self.docrimeCross = np.zeros_like(self.docrimeCross)
        self.docrimeAuto = np.zeros_like(self.docrimeAuto)


        for i in range(self.numItersIM):
            #Compute new random shape
            correction = np.random.uniform(-self.pokeAmp,self.pokeAmp,correction.size).astype(correction.dtype).reshape(correction.shape)
            
            #Get current WFS response
            #I put this first to match CL case
            slopes = self.signalShm.read(RELEASE_GIL = True).reshape(slopes.shape)

            #Send random shape to mirror
            self.sendToWfc(correction)

            add_to_buffer(self.docrimeBuffer, correction)

            #Correlate Current response with old correction by delay time
            self.docrimeCross += slopes@self.docrimeBuffer[0].T
            self.docrimeAuto += self.docrimeBuffer[0]@self.docrimeBuffer[0].T

        self.docrimeCross /= self.numItersIM 
        self.docrimeAuto /= self.numItersIM
        self.IM = self.docrimeCross @np.linalg.inv(self.docrimeAuto)

        self.docrimeCross = np.zeros_like(self.docrimeCross)
        self.docrimeAuto = np.zeros_like(self.docrimeAuto)

        return

    def computeIM(self):
        """
        Compute the interaction matrix using the specified method. Method specified using IMMethod, default is push-pull.
        """
        component_logger = getattr(self, "logger", logger)
        try:
            component_logger.info("Computing interaction matrix using method=%s", self.IMMethod)
            if self.IMMethod == 'docrime':
                self.docrimeIM()
            else:
                self.pushPullIM()

            self.computeCM()
        except Exception:
            component_logger.exception("Failed to compute interaction matrix using method=%s", getattr(self, "IMMethod", None))
            raise
        return
    
    def saveIM(self,filename=''):
        """
        Save the interaction matrix to a file.

        Parameters
        ----------
        filename : str, optional
            File to save the interaction matrix to. If not specified, uses the configured IMFile.
        """
        component_logger = getattr(self, "logger", logger)
        try:
            if filename == '':
                filename = self.IMFile
            if filename == '':
                raise ValueError("No interaction matrix filename provided")
            np.save(filename, self.IM)
            component_logger.info("Saved interaction matrix to %s", filename)
        except Exception:
            component_logger.exception("Failed to save interaction matrix to %s", filename or getattr(self, "IMFile", ""))
            raise

    def loadIM(self,filename=''):
        """
        Load the interaction matrix from a file.

        Parameters
        ----------
        filename : str, optional
            File to load the interaction matrix from. If not specified, uses the configured IMFile.
        """
        component_logger = getattr(self, "logger", logger)
        try:
            if filename == '':
                filename = self.IMFile
            if filename == '':
                self.IM = np.zeros_like(self.IM)
                component_logger.info("No interaction matrix file configured; using zeros")
            else:
                self.IM = np.load(filename)
                component_logger.info("Loaded interaction matrix from %s", filename)
            self.computeCM()
        except Exception:
            component_logger.exception("Failed to load interaction matrix from %s", filename or getattr(self, "IMFile", ""))
            raise

    def flatten(self):
        """
        Send the flat correction to the wavefront corrector.
        """
        component_logger = getattr(self, "logger", logger)
        try:
            self.sendToWfc(self.flat)
            component_logger.info("Flattened loop correction")
        except Exception:
            component_logger.exception("Failed to flatten loop correction")
            raise
        return

    @staticmethod
    def _validate_cm_method(method: str) -> str:
        normalized = str(method).lower()
        if normalized not in {"svd", "tikhonov"}:
            raise ValueError(f"Unsupported CM inversion method: {method}")
        return normalized

    @staticmethod
    def _suggest_conditioning_from_singular_values(singular_values: np.ndarray):
        singular_values = np.asarray(singular_values, dtype=np.float64)
        singular_values = singular_values[np.isfinite(singular_values) & (singular_values > 0)]
        if singular_values.size < 4:
            return None, None

        normalized = singular_values / singular_values[0]
        indices = np.arange(normalized.size, dtype=np.float64)
        log_values = np.log10(np.clip(normalized, np.finfo(np.float64).tiny, None))

        min_leading_points = max(3, normalized.size // 8)
        best_score = -np.inf
        best_fit = None

        for knee_index in range(min_leading_points - 1, normalized.size - 1):
            leading_x = indices[: knee_index + 1]
            leading_y = log_values[: knee_index + 1]

            sample_count = float(leading_x.size)
            x_mean = math.fsum(float(value) for value in leading_x) / sample_count
            y_mean = math.fsum(float(value) for value in leading_y) / sample_count
            centered_x = leading_x - x_mean
            centered_y = leading_y - y_mean
            variance_x = float(np.dot(centered_x, centered_x))
            if variance_x <= 0:
                continue

            slope = float(np.dot(centered_x, centered_y) / variance_x)
            intercept = float(y_mean - slope * x_mean)
            fit_y = slope * leading_x + intercept
            fit_residual = leading_y - fit_y
            rmse = math.sqrt(
                math.fsum(float(value) * float(value) for value in fit_residual) / sample_count
            )

            predicted_next = slope * indices[knee_index + 1] + intercept
            downward_departure = predicted_next - log_values[knee_index + 1]
            if downward_departure <= 0:
                continue

            score = downward_departure / (rmse + 1e-6)
            if score > best_score:
                threshold = normalized[knee_index + 1]
                if threshold <= 0:
                    continue
                best_score = score
                best_fit = {
                    "knee_index": int(knee_index),
                    "suggested_index": int(knee_index + 1),
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "rmse": rmse,
                    "normalized_threshold": float(threshold),
                    "conditioning": float(1.0 / threshold),
                    "score": float(score),
                    "indices": indices.copy(),
                    "normalized_singular_values": normalized.copy(),
                    "fit_curve": np.power(10.0, slope * indices + intercept),
                }

        if best_fit is None:
            return None, None
        return best_fit["conditioning"], best_fit

    def getSingularValues(self) -> np.ndarray:
        if self.IM.size == 0:
            return np.array([], dtype=np.float64)
        return np.linalg.svd(self.IM, compute_uv=False)

    def suggestConditioningNumber(self):
        singular_values = self.getSingularValues()
        suggestion, fit = self._suggest_conditioning_from_singular_values(singular_values)
        self.lastSuggestedConditioning = suggestion
        self.lastSingularValueFit = fit
        return suggestion

    def plotSingularValues(self, conditioning_lines=COMMON_CONDITIONING_LINES, ax=None):
        singular_values = self.getSingularValues()
        self.lastSingularValues = singular_values
        suggestion, fit = self._suggest_conditioning_from_singular_values(singular_values)
        self.lastSuggestedConditioning = suggestion
        self.lastSingularValueFit = fit

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4.5))

        if singular_values.size == 0 or np.max(singular_values) <= 0:
            ax.set_title("Singular values unavailable")
            ax.set_xlabel("Singular value index")
            ax.set_ylabel("Singular value")
            return suggestion

        normalized = singular_values / singular_values[0]
        indices = np.arange(1, singular_values.size + 1)
        ax.semilogy(indices, normalized, marker="o", linewidth=1.5, label="Normalized singular values")

        for cond in conditioning_lines:
            if cond is None or cond <= 0:
                continue
            ax.axhline(1.0 / cond, linestyle="--", linewidth=0.8, alpha=0.5, label=f"cond={cond:.0e}")

        if fit is not None:
            ax.semilogy(indices, fit["fit_curve"], color="tab:green", linestyle="-.", linewidth=1.2, label="Leading log-fit")
            ax.axvline(fit["suggested_index"] + 1, color="tab:red", linestyle=":", linewidth=1.2, label=f"turnoff idx={fit['suggested_index'] + 1}")

        if suggestion is not None and suggestion > 0:
            ax.axhline(1.0 / suggestion, color="black", linestyle=":", linewidth=1.5, label=f"suggested={suggestion:.2e}")

        ax.set_title("Normalized IM singular values")
        ax.set_xlabel("Singular value index")
        ax.set_ylabel("Singular value / max singular value")
        ax.legend(loc="best", fontsize="small")
        return suggestion

    def _compute_inverse_from_svd(self, matrix, method: str, conditioning, tikhonovReg: float):
        matrix = np.asarray(matrix, dtype=np.float64)
        num_modes = matrix.shape[1]

        if matrix.size == 0:
            return np.zeros((num_modes, matrix.shape[0]), dtype=self.CM.dtype), np.array([], dtype=np.float64), np.array([], dtype=bool)

        singular_values = np.linalg.svd(matrix, compute_uv=False)
        if singular_values.size == 0 or singular_values[0] <= 0:
            return np.zeros((num_modes, matrix.shape[0]), dtype=self.CM.dtype), singular_values, np.zeros_like(singular_values, dtype=bool)

        U, singular_values, Vh = np.linalg.svd(matrix, full_matrices=False)
        retained = singular_values > 0
        if conditioning is not None:
            retained &= singular_values >= (singular_values[0] / conditioning)

        inverse_singular_values = np.zeros_like(singular_values)
        if method == "svd":
            inverse_singular_values[retained] = 1.0 / singular_values[retained]
        else:
            if tikhonovReg < 0:
                raise ValueError("tikhonovReg must be non-negative")
            inverse_singular_values[retained] = singular_values[retained] / (singular_values[retained] ** 2 + tikhonovReg ** 2)

        inverse = (Vh.T * inverse_singular_values) @ U.T
        return inverse.astype(self.CM.dtype, copy=False), singular_values, retained
    
    def computeCM(self, method=None, numDroppedModes=None, conditioning=None, tikhonovReg=None):
        """
        Compute the control matrix from the interaction matrix.

        Parameters
        ----------
        method : str, optional
            Inversion method to use. Supported values are ``svd`` and
            ``tikhonov``. Defaults to the configured ``CMMethod``.
        numDroppedModes : int, optional
            Number of modal commands to suppress before inversion. Defaults to
            the configured ``numDroppedModes``.
        conditioning : float, optional
            Optional target conditioning number. Singular values below
            ``max(s) / conditioning`` are discarded.
        tikhonovReg : float, optional
            Tikhonov regularization strength used when ``method`` is
            ``tikhonov``. Defaults to the configured ``tikhonovReg``.
        """
        component_logger = getattr(self, "logger", logger)
        try:
            method = self._validate_cm_method(self.CMMethod if method is None else method)
            requested_dropped_modes = self.numDroppedModes if numDroppedModes is None else int(numDroppedModes)
            requested_conditioning = self.conditioning if conditioning is None else conditioning
            requested_tikhonov = self.tikhonovReg if tikhonovReg is None else float(tikhonovReg)

            if requested_conditioning is not None:
                requested_conditioning = float(requested_conditioning)
                if requested_conditioning <= 1:
                    raise ValueError("conditioning must be greater than 1 when provided")

            self.numDroppedModes = requested_dropped_modes
            self.CMMethod = method
            self.conditioning = requested_conditioning
            self.tikhonovReg = requested_tikhonov
            self.numActiveModes = self.numModes - self.numDroppedModes
            if self.numActiveModes < 0:
                raise ValueError("Invalid number of modes used in CM. Check numDroppedModes")
            active_im = self.IM[:, :self.numActiveModes]
            inverse, singular_values, retained = self._compute_inverse_from_svd(
                active_im,
                method=self.CMMethod,
                conditioning=self.conditioning,
                tikhonovReg=self.tikhonovReg,
            )

            self.CM[:, :] = 0
            self.CM[:self.numActiveModes, :] = inverse
            self.CM[self.numActiveModes:, :] = 0
            self.gCM = self.gain * self.CM
            self.fIM = np.copy(self.IM)
            self.fIM[:, self.numActiveModes:] = 0
            self.lastSingularValues = singular_values
            self.lastRetainedSingularMask = retained
            suggestion, fit = self._suggest_conditioning_from_singular_values(singular_values)
            self.lastSuggestedConditioning = suggestion
            self.lastSingularValueFit = fit
            component_logger.info(
                "Computed control matrix method=%s activeModes=%s droppedModes=%s conditioning=%s retainedSingularValues=%s tikhonovReg=%s",
                self.CMMethod,
                self.numActiveModes,
                self.numDroppedModes,
                self.conditioning,
                int(np.count_nonzero(retained)),
                self.tikhonovReg,
            )
        except Exception:
            component_logger.exception("Failed to compute control matrix")
            raise
        return 
        
    # @jit(nopython=True)
    def updateCorrectionPOL(self, correction=np.array([], dtype=np.float32), slopes=np.array([], dtype=np.float32)):
        """
        Update the correction using pseudo open loop slopes.

        Parameters
        ----------
        correction : numpy.ndarray
            Current correction vector.
        slopes : numpy.ndarray
            Current slopes vector.

        Returns
        -------
        numpy.ndarray
            Updated correction vector.
        """   
        # Compute POL Slopes s_{POL} = s_{RES} + IM*c_{n-1}
        # print(f'slopes: {slopes.shape}, IM: {self.IM.shape}, corr: {correction.shape}')
        s_pol = slopes - self.fIM@correction

        # Update Command Vector c_n = g*CM*s_{POL} + (1 − g) c_{n-1}  https://arxiv.org/pdf/1903.12124.pdf Eq 3
        return (1-self.gain)*correction - np.dot(self.gCM,s_pol)

    def standardIntegratorPOL(self):
        """
        Standard integrator using the pseudo open loop slopes.
        """
        residual_slopes = self.signalShm.read(RELEASE_GIL = self.RELEASE_GIL)
        currentCorrection = self.wfcShm.read(RELEASE_GIL = self.RELEASE_GIL)
        # print(f'slopes: {residual_slopes.shape}, IM: {self.IM.shape}, corr: {currentCorrection.shape}')

        newCorrection = self.updateCorrectionPOL(correction=currentCorrection, 
                                                 slopes=residual_slopes)
        newCorrection[self.numActiveModes:] = 0
        self.sendToWfc(newCorrection)

        return

    
    def standardIntegrator(self):
        """
        Standard integrator.
        """
        slopes = self.signalShm.read(SAFE=False, RELEASE_GIL = self.RELEASE_GIL)
        newCorrection = leakyIntegratorNumba(slopes, 
                         self.gCM, 
                         self.wfcShm.read_noblock(SAFE=False).squeeze(),
                         self.nullCorrection,
                         np.float32(0),#No leak
                         self.numActiveModes)
        self.sendToWfc(newCorrection, slopes=slopes)
        return
    
    def leakyIntegrator(self):
        """
        Leaky integrator.
        """
        slopes = self.signalShm.read(SAFE=False, RELEASE_GIL = self.RELEASE_GIL)
        newCorrection = leakyIntegratorNumba(slopes, 
                         self.gCM, 
                         self.wfcShm.read_noblock(SAFE=False).squeeze(),
                         self.nullCorrection,
                         np.float32(self.leakyGain),
                         self.numActiveModes)
        self.sendToWfc(newCorrection, slopes=slopes)
        return

    def pidIntegratorPOL(self):
        """
        PID integrator using the pseudo-open loop slopes.
        """
        slopes = self.signalShm.read(RELEASE_GIL = self.RELEASE_GIL)
        correction = self.wfcShm.read(RELEASE_GIL = self.RELEASE_GIL)
        polSlopes = slopes - self.fIM@correction
        return self.pidIntegrator(slopes=polSlopes, correction=correction)

    def pidIntegrator(self, slopes = None, correction = None):
        """
        PID integrator.

        Parameters
        ----------
        slopes : numpy.ndarray, optional
            Current slopes vector. If not provided, reads from shared memory.
        correction : numpy.ndarray, optional
            Current correction vector. If not provided, reads from shared memory.
        """
        if slopes is None:
            slopes = self.signalShm.read(RELEASE_GIL = self.RELEASE_GIL)
        if correction is None:
            correction = self.wfcShm.read(RELEASE_GIL = self.RELEASE_GIL)

        #Compute raw error term (numba accelerated)
        wfError = compCorrection(CM=self.CM, 
                                    slopes=slopes)
        derivative = (wfError - self.previousWfError) 
        
        # Apply low-pass filter to the derivative to reduce noise
        derivative = self.derivativeFilter * derivative + (1 - self.derivativeFilter) * self.previousDerivative
        
        # Update integral (anti-windup: conditional integration)
        # notOutputLimiting = self.controlLimits[0] is None or self.controlLimits[1] is None
        isClipped = np.any(self.controlOutput == self.controlLimits[0]) or np.any(self.controlOutput == self.controlLimits[1])
        #Check to make sure we aren't actively clipping the correction
        if not isClipped:
            #Add to integral
            self.integral += wfError 
            #Clip integral term
            self.integral = np.clip(self.integral, *self.integralLimits)

        # Calculate PID output
        controlOutput = self.pGain * wfError + self.iGain * self.integral + self.dGain * derivative

        controlOutput = np.clip(controlOutput, *self.controlLimits)

        #Get new correction vector from the control output
        newCorrection = (1-self.leakyGain)*correction - controlOutput #Negative control direction is convention for pyRTC

        #Remove anything in non-corrected modes (might be redundant)
        newCorrection[self.numActiveModes:] = 0
        
        # Clip correction (force the loop to not over correct a mode)
        newCorrection = np.clip(newCorrection, *self.absoluteLimits)
        
        #Apply new correction to mirror
        self.sendToWfc(newCorrection, slopes = slopes)

        # Save state for next iteration
        self.previousWfError = wfError
        self.previousDerivative = derivative
        self.controlOutput = controlOutput
        
        return

    def sendToWfc(self, correction, slopes=None):
        #Get an initial slope reading to set shapes
        correction = correction.reshape(self.flat.shape)
        if self.clDocrime and isinstance(slopes, np.ndarray):

            slopes = slopes.reshape(slopes.size, 1)
            #Compute new random shape
            randShape = np.random.uniform(-self.pokeAmp,
                                          self.pokeAmp,
                                          correction.size).astype(self.docrimeBuffer[0].dtype).reshape(self.docrimeBuffer[0].shape)

            #Adds to end of buffer (i.e. pos -1)
            add_to_buffer(self.docrimeBuffer,randShape)

            randShape = randShape.astype(correction.dtype).reshape(correction.shape)

            #Only add randomness to active modes, otherwise it will build up
            if self.numActiveModes > 0:
                correction[:self.numActiveModes] += randShape[:self.numActiveModes]
                correction[self.numActiveModes:] = randShape[self.numActiveModes:]
            else:
                correction = randShape

            #Send our new pertubation to the WFC
            self.wfcShm.write(correction)

            #Correlate Current response with old correction by delay time
            self.docrimeCross += slopes@self.docrimeBuffer[0].T
            self.docrimeAuto += self.docrimeBuffer[0]@self.docrimeBuffer[0].T

            self.numItersDC += 1

        else:
            self.wfcShm.write(correction)
        return

    def solveDocrime(self):

        component_logger = getattr(self, "logger", logger)
        try:
            self.clDCIM = (self.docrimeCross / self.numItersDC) @ np.linalg.inv(self.docrimeAuto / self.numItersDC)
            tmpFilePath = get_tmp_filepath(self.IMFile, uniqueStr="CL_docrime")
            component_logger.info("Saving DOCRIME matrix to %s", tmpFilePath)
            np.save(tmpFilePath, self.clDCIM)
        except Exception:
            component_logger.exception("Failed to solve DOCRIME interaction matrix")
            raise

        return


    def plotIM(self, row=None):

        plt.imshow(self.IM, cmap = 'inferno', aspect='auto')
        plt.show()

if __name__ == "__main__":

    launchComponent(Loop, "loop", start = False)
