"""Adaptive optics loop control kernels and the main loop component.

This module contains the numerical update kernels and the high-level
``Loop`` component that turn measured residuals into new correction commands.
It is the control-plane heart of pyrtc: interaction matrices, control matrices,
integrators, and command dispatch all come together here.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import math
import matplotlib.pyplot as plt
import numpy as np
import time
from typing import Any
from numba import jit

from pyrtc.logging_utils import get_logger
from pyrtc.manager import launch_component
from pyrtc.streams import gpu_torch_available, open_stream
from pyrtc.component import Component
from pyrtc.utils import add_to_buffer, get_tmp_filepath, set_from_config

logger = get_logger(__name__)

COMMON_CONDITIONING_LINES = (10.0, 100.0, 1e3, 1e4, 1e5, 1e6)


@jit(nopython=True, nogil=True, cache=False, fastmath=True)
def leaky_integrator_numba(
    slopes: np.ndarray,
    reconstruction_matrix: np.ndarray,
    old_correction: np.ndarray,
    correction: np.ndarray,
    leak: np.float32,
    num_active_modes: int,
) -> np.ndarray:

    # Perform the matrix-vector multiplication using np.dot
    correction = np.dot(reconstruction_matrix, slopes)

    # Apply the leaky integrator formula with an unrolled loop
    for i in range(num_active_modes + 1):
        correction[i] = (1 - leak) * old_correction[i] - correction[i]

    # Zero out the rest of the correction vector
    for i in range(num_active_modes + 1, correction.size):
        correction[i] = 0.0

    return correction


def leak_integrator_gpu(
    slopes: np.ndarray,
    reconstruction_matrix: Any,
    old_correction: np.ndarray,
    leak: float,
    num_active_modes: int,
):
    """Run the leaky-integrator control update on a CUDA-backed torch matrix."""

    if not gpu_torch_available():
        raise ImportError(
            "leak_integrator_gpu requires PyTorch. Install with 'pip install pyrtc[gpu]' or 'pip install torch'."
        )

    import torch

    slopes_gpu = torch.tensor(slopes, device="cuda")
    correction_gpu = torch.matmul(reconstruction_matrix, slopes_gpu)
    correction_gpu[num_active_modes:] = 0
    return np.subtract((1 - leak) * old_correction, correction_gpu.cpu().numpy())


@jit(nopython=True, nogil=True, cache=False, fastmath=True)
def comp_correction(cm=np.array([[]], dtype=np.float32), slopes=np.array([], dtype=np.float32)):
    """Apply a control matrix to a slope vector and return the correction."""

    return np.dot(cm, slopes)


@jit(nopython=True, nogil=True, cache=False, fastmath=True)
def update_correction(
    correction=np.array([], dtype=np.float32),
    g_cm=np.array([[]], dtype=np.float32),
    slopes=np.array([], dtype=np.float32),
):
    """Update an existing correction using a pre-scaled control matrix."""

    return correction - np.dot(g_cm, slopes)


# @jit(nopython=True)


class Loop(Component):
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
    num_dropped_modes : int, optional
        Number of modes to drop. Default is 0.
    gain : float, optional
        Gain for the integrator. Default is 0.1.
    leaky_gain : float, optional
        Leaky integrator gain. Default is 0.0.
    hardware_delay : float, optional
        Delay for the hardware. Default is 0.0.
    poke_amp : float, optional
        Amplitude for poking. Default is 0.01.
    num_iters_im : int, optional
        Number of iterations for interaction matrix computation. Default is 100.
    delay : int, optional
        Delay for corrections. Default is 0.
    im_method : str, optional
        Method for interaction matrix computation. Default is "push-pull".
    im_file : str, optional
        File to save the interaction matrix. Default is "".
    p_gain : float, optional
        Proportional gain for PID integrator. Default is 0.1.
    i_gain : float, optional
        Integral gain for PID integrator. Default is 0.0.
    d_gain : float, optional
        Derivative gain for PID integrator. Default is 0.0.
    control_limits : list, optional
        Control limits for PID integrator. Default is [-inf, inf].
    integral_limits : list, optional
        Integral limits for PID integrator. Default is [-inf, inf].
    absolute_limits : list, optional
        Absolute limits for corrections. Default is [-inf, inf].
    derivative_filter : float, optional
        Filter for the derivative term. Default is 0.1.

    Attributes
    ----------
    conf : dict
        Loop configuration.
    name : str
        Name of the loop.
    signal_dtype : type
        Data type of the wavefront sensor signal.
    signal_size : int
        Size of the wavefront sensor signal.
    signal_shm : pyshmem.SharedMemory
        Shared memory object for the wavefront sensor signal.
    null_signal : numpy.ndarray
        Null signal.
    signal_2d_dtype : type
        Data type of the 2D wavefront sensor signal.
    signal_2d_size : int
        Size of the 2D wavefront sensor signal.
    signal_2d_width : int
        Width of the 2D wavefront sensor signal.
    signal_2d_height : int
        Height of the 2D wavefront sensor signal.
    wfc_dtype : type
        Data type of the wavefront corrector.
    num_modes : int
        Number of modes in the wavefront corrector.
    wfc_shm : pyshmem.SharedMemory
        Shared memory object for the wavefront corrector.
    num_dropped_modes : int
        Number of dropped modes.
    num_active_modes : int
        Number of active modes.
    flat : numpy.ndarray
        Flat correction vector.
    im : numpy.ndarray
        Interaction matrix.
    cm : numpy.ndarray
        Control matrix.
    gain : float
        Gain for the integrator.
    leaky_gain : float
        Leaky integrator gain.
    perturb_amp : float
        Perturbation amplitude.
    hardware_delay : float
        Delay for the hardware.
    poke_amp : float
        Amplitude for poking.
    num_iters_im : int
        Number of iterations for interaction matrix computation.
    delay : int
        Delay for corrections.
    im_method : str
        Method for interaction matrix computation.
    im_file : str
        File to save the interaction matrix.
    p_gain : float
        Proportional gain for PID integrator.
    i_gain : float
        Integral gain for PID integrator.
    d_gain : float
        Derivative gain for PID integrator.
    control_limits : list
        Control limits for PID integrator.
    integral_limits : list
        Integral limits for PID integrator.
    absolute_limits : list
        Absolute limits for corrections.
    derivative_filter : float
        Filter for the derivative term.
    cm_method : str
        Control-matrix inversion method. Supported values are ``svd`` and
        ``tikhonov``.
    conditioning : float or None
        Optional target conditioning number used to truncate small singular
        values when computing the control matrix.
    tikhonov_reg : float
        Tikhonov regularization strength used when ``cm_method`` is
        ``tikhonov``.
    integral : numpy.ndarray
        Integral term for PID integrator.
    previous_wf_error : numpy.ndarray
        Previous wavefront error.
    previous_derivative : numpy.ndarray
        Previous derivative term.
    control_output : numpy.ndarray
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
                num_dropped_modes : int, optional
                    Number of modes to drop. Default is 0.
                gain : float, optional
                    Gain for the integrator. Default is 0.1.
                leaky_gain : float, optional
                    Leaky integrator gain. Default is 0.0.
                hardware_delay : float, optional
                    Delay for the hardware. Default is 0.0.
                poke_amp : float, optional
                    Amplitude for poking. Default is 0.01.
                num_iters_im : int, optional
                    Number of iterations for interaction matrix computation. Default is 100.
                delay : int, optional
                    Delay for corrections. Default is 0.
                im_method : str, optional
                    Method for interaction matrix computation. Default is "push-pull".
                im_file : str, optional
                    File to save the interaction matrix. Default is "".
                p_gain : float, optional
                    Proportional gain for PID integrator. Default is 0.1.
                i_gain : float, optional
                    Integral gain for PID integrator. Default is 0.0.
                d_gain : float, optional
                    Derivative gain for PID integrator. Default is 0.0.
                control_limits : list, optional
                    Control limits for PID integrator. Default is [-inf, inf].
                integral_limits : list, optional
                    Integral limits for PID integrator. Default is [-inf, inf].
                absolute_limits : list, optional
                    Absolute limits for corrections. Default is [-inf, inf].
                derivative_filter : float, optional
                    Filter for the derivative term. Default is 0.1.
        """
        try:
            super().__init__(conf)
            self.name = "Loop"
            self.conf = conf

            # Read wfs signal's metadata and open a stream to the shared memory
            self.signal_shm = open_stream(
                self.input_stream_name("signal"), gpu_device=self.gpu_device
            )
            self.signal_shape = tuple(self.signal_shm.shape)
            self.signal_dtype = np.dtype(self.signal_shm.dtype)
            self.register_input_stream("signal", self.signal_shm)
            self.signal_size = int(np.prod(self.signal_shape))
            self.null_signal = np.zeros(self.signal_shape, dtype=self.signal_dtype)

            # Read wfc metadata and open a stream to the shared memory
            self.wfc_shm = open_stream(self.output_stream_name("wfc"), gpu_device=self.gpu_device)
            self.wfc_shape = tuple(self.wfc_shm.shape)
            self.wfc_dtype = np.dtype(self.wfc_shm.dtype)
            self.register_output_stream("wfc", self.wfc_shm)
            self.num_modes = int(np.prod(self.wfc_shape))

            self.num_dropped_modes = set_from_config(self.conf, "num_dropped_modes", 0)
            self.num_active_modes = self.num_modes - self.num_dropped_modes
            self.flat = np.zeros(self.num_modes, dtype=self.wfc_dtype)
            self.null_correction = np.zeros_like(self.flat)

            self.im = np.zeros((self.signal_size, self.num_modes), dtype=self.signal_dtype)
            self.cm = np.zeros((self.num_modes, self.signal_size), dtype=self.signal_dtype)
            self.gain = set_from_config(self.conf, "gain", 0.1)
            self.leaky_gain = set_from_config(self.conf, "leaky_gain", 0.0)
            self.perturb_amp = 0
            self.hardware_delay = set_from_config(self.conf, "hardware_delay", 0.0)
            self.poke_amp = set_from_config(self.conf, "poke_amp", 1e-2)
            self.num_iters_im = set_from_config(self.conf, "num_iters_im", 100)
            self.delay = set_from_config(self.conf, "delay", 0)
            self.im_method = set_from_config(self.conf, "im_method", "push-pull")
            self.im_file = set_from_config(self.conf, "im_file", "")
            self.cm_method = str(set_from_config(self.conf, "cm_method", "svd")).lower()
            conditioning = set_from_config(self.conf, "conditioning", None)
            self.conditioning = None if conditioning is None else float(conditioning)
            self.tikhonov_reg = float(set_from_config(self.conf, "tikhonov_reg", 0.0))
            self.last_singular_values = np.array([], dtype=np.float64)
            self.last_retained_singular_mask = np.array([], dtype=bool)
            self.last_suggested_conditioning = None
            self.last_singular_value_fit = None

            self.cl_docrime = False
            self.num_iters_dc = 0
            tmp2 = self.flat.copy().reshape(self.flat.size, 1)
            tmp = self.null_signal.copy().reshape(self.null_signal.size, 1)
            self.docrime_cross = np.zeros_like(tmp @ tmp2.T)
            self.docrime_auto = np.zeros_like(tmp2 @ tmp2.T)
            self.docrime_buffer = np.zeros((1 + self.delay, *tmp2.shape), dtype=self.wfc_dtype)

            self.p_gain = set_from_config(self.conf, "p_gain", 0.1)
            self.i_gain = set_from_config(self.conf, "i_gain", 0.0)
            self.d_gain = set_from_config(self.conf, "d_gain", 0.0)
            self.control_limits = set_from_config(self.conf, "control_limits", [-np.inf, np.inf])
            self.integral_limits = set_from_config(self.conf, "integral_limits", [-np.inf, np.inf])
            self.absolute_limits = set_from_config(self.conf, "absolute_limits", [-np.inf, np.inf])
            self.derivative_filter = set_from_config(self.conf, "derivative_filter", 0.1)
            self.integral = 0

            self.previous_wf_error = np.zeros_like(self.read_stream("wfc", block=False))
            self.previous_derivative = np.zeros_like(self.previous_wf_error)
            self.control_output = np.zeros_like(self.previous_wf_error)

            # Pre-allocated hot-path read buffers (ignored for GPU streams).
            self._signal_buffer = np.empty(self.signal_shape, dtype=self.signal_dtype)
            self._wfc_buffer = np.empty(self.wfc_shape, dtype=self.wfc_dtype)

            self.load_im()
            self.logger.info(
                "Initialized loop signal_shape=%s wfc_shape=%s num_modes=%s",
                self.signal_shape,
                self.wfc_shape,
                self.num_modes,
            )
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
        if hasattr(self, "cm"):
            self.g_cm = self._gain * self.cm

    def set_gain(self, gain):
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

    def set_peturb_amp(self, amp):
        """
        Set the perturbation amplitude.

        Parameters
        ----------
        amp : float
            Amplitude to set.
        """
        component_logger = getattr(self, "logger", logger)
        try:
            self.perturb_amp = amp
            component_logger.info("Set perturbation amplitude to %s", amp)
        except Exception:
            component_logger.exception("Failed to set perturbation amplitude to %s", amp)
            raise
        return

    def push_pull_im(self):
        """
        Compute the interaction matrix using the push-pull method.
        """
        # For each mode
        for i in range(self.num_modes):
            # Reset the correction
            correction = self.flat.copy()
            # Plus amplitude
            correction[i] = self.poke_amp
            # Post a new shape to be made
            self.send_to_wfc(correction)
            # Add some delay to ensure one-to-one
            time.sleep(self.hardware_delay)
            # Burn the first new image since we were moving the DM during the exposure
            self.read_stream("signal")
            # Average out N new WFS frames
            tmp_plus = np.zeros_like(self.im[:, i])
            for n in range(self.num_iters_im):
                tmp_plus += self.read_stream("signal")
            tmp_plus /= self.num_iters_im

            # Minus amplitude
            correction[i] = -self.poke_amp
            # Post a new shape to be made
            self.send_to_wfc(correction)
            # Add some delay to ensure one-to-one
            time.sleep(self.hardware_delay)
            # Burn the first new image since we were moving the DM during the exposure
            self.read_stream("signal")
            # Average out N new WFS frames
            tmp_minus = np.zeros_like(self.im[:, i])
            for n in range(self.num_iters_im):
                tmp_minus += self.read_stream("signal")
            tmp_minus /= self.num_iters_im

            # Compute the normalized difference
            self.im[:, i] = (tmp_plus - tmp_minus) / (2 * self.poke_amp)

        return

    def docrime_im(self):
        """
        Compute the interaction matrix using the DOCRIME method.
        """
        # Send the flat command to the WFC
        self.flatten()

        # Get a correction to set the shape
        correction = self.flat.copy()
        correction = correction.reshape(correction.size, 1)

        # Have a history of corrections
        # corrections = np.zeros((1+self.delay, *correction.shape), dtype=correction.dtype)

        # Get an initial slope reading to set shapes
        slopes = self.null_signal.copy()
        slopes = slopes.reshape(slopes.size, 1)
        self.docrime_cross = np.zeros_like(self.docrime_cross)
        self.docrime_auto = np.zeros_like(self.docrime_auto)

        for i in range(self.num_iters_im):
            # Compute new random shape
            correction = (
                np.random.uniform(-self.poke_amp, self.poke_amp, correction.size)
                .astype(correction.dtype)
                .reshape(correction.shape)
            )

            # Get current WFS response
            # I put this first to match CL case
            slopes = self.read_stream("signal").reshape(slopes.shape)

            # Send random shape to mirror
            self.send_to_wfc(correction)

            add_to_buffer(self.docrime_buffer, correction)

            # Correlate Current response with old correction by delay time
            self.docrime_cross += slopes @ self.docrime_buffer[0].T
            self.docrime_auto += self.docrime_buffer[0] @ self.docrime_buffer[0].T

        self.docrime_cross /= self.num_iters_im
        self.docrime_auto /= self.num_iters_im
        self.im = self.docrime_cross @ np.linalg.inv(self.docrime_auto)

        self.docrime_cross = np.zeros_like(self.docrime_cross)
        self.docrime_auto = np.zeros_like(self.docrime_auto)

        return

    def compute_im(self):
        """
        Compute the interaction matrix using the specified method. Method specified using im_method, default is push-pull.
        """
        component_logger = getattr(self, "logger", logger)
        try:
            component_logger.info("Computing interaction matrix using method=%s", self.im_method)
            if self.im_method == "docrime":
                self.docrime_im()
            else:
                self.push_pull_im()

            self.compute_cm()
        except Exception:
            component_logger.exception(
                "Failed to compute interaction matrix using method=%s",
                getattr(self, "im_method", None),
            )
            raise
        return

    def save_im(self, filename=""):
        """
        Save the interaction matrix to a file.

        Parameters
        ----------
        filename : str, optional
            File to save the interaction matrix to. If not specified, uses the configured im_file.
        """
        component_logger = getattr(self, "logger", logger)
        try:
            if filename == "":
                filename = self.im_file
            if filename == "":
                raise ValueError("No interaction matrix filename provided")
            np.save(filename, self.im)
            component_logger.info("Saved interaction matrix to %s", filename)
        except Exception:
            component_logger.exception(
                "Failed to save interaction matrix to %s", filename or getattr(self, "im_file", "")
            )
            raise

    def load_im(self, filename=""):
        """
        Load the interaction matrix from a file.

        Parameters
        ----------
        filename : str, optional
            File to load the interaction matrix from. If not specified, uses the configured im_file.
        """
        component_logger = getattr(self, "logger", logger)
        try:
            if filename == "":
                filename = self.im_file
            if filename == "":
                self.im = np.zeros_like(self.im)
                component_logger.info("No interaction matrix file configured; using zeros")
            else:
                self.im = np.load(filename)
                component_logger.info("Loaded interaction matrix from %s", filename)
            self.compute_cm()
        except Exception:
            component_logger.exception(
                "Failed to load interaction matrix from %s",
                filename or getattr(self, "im_file", ""),
            )
            raise

    def flatten(self):
        """
        Send the flat correction to the wavefront corrector.
        """
        component_logger = getattr(self, "logger", logger)
        try:
            self.send_to_wfc(self.flat)
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

    def get_singular_values(self) -> np.ndarray:
        if self.im.size == 0:
            return np.array([], dtype=np.float64)
        return np.linalg.svd(self.im, compute_uv=False)

    def suggest_conditioning_number(self):
        singular_values = self.get_singular_values()
        suggestion, fit = self._suggest_conditioning_from_singular_values(singular_values)
        self.last_suggested_conditioning = suggestion
        self.last_singular_value_fit = fit
        return suggestion

    def plot_singular_values(self, conditioning_lines=COMMON_CONDITIONING_LINES, ax=None):
        singular_values = self.get_singular_values()
        self.last_singular_values = singular_values
        suggestion, fit = self._suggest_conditioning_from_singular_values(singular_values)
        self.last_suggested_conditioning = suggestion
        self.last_singular_value_fit = fit

        if ax is None:
            fig = plt.figure(figsize=(8, 4.5))
            ax = fig.add_axes((0.12, 0.15, 0.83, 0.78))

        if singular_values.size == 0 or np.max(singular_values) <= 0:
            ax.set_title("Singular values unavailable")
            ax.set_xlabel("Singular value index")
            ax.set_ylabel("Singular value")
            return suggestion

        normalized = singular_values / singular_values[0]
        indices = np.arange(1, singular_values.size + 1)
        ax.semilogy(
            indices, normalized, marker="o", linewidth=1.5, label="Normalized singular values"
        )

        for cond in conditioning_lines:
            if cond is None or cond <= 0:
                continue
            ax.axhline(
                1.0 / cond, linestyle="--", linewidth=0.8, alpha=0.5, label=f"cond={cond:.0e}"
            )

        if fit is not None:
            ax.semilogy(
                indices,
                fit["fit_curve"],
                color="tab:green",
                linestyle="-.",
                linewidth=1.2,
                label="Leading log-fit",
            )
            ax.axvline(
                fit["suggested_index"] + 1,
                color="tab:red",
                linestyle=":",
                linewidth=1.2,
                label=f"turnoff idx={fit['suggested_index'] + 1}",
            )

        if suggestion is not None and suggestion > 0:
            ax.axhline(
                1.0 / suggestion,
                color="black",
                linestyle=":",
                linewidth=1.5,
                label=f"suggested={suggestion:.2e}",
            )

        ax.set_title("Normalized IM singular values")
        ax.set_xlabel("Singular value index")
        ax.set_ylabel("Singular value / max singular value")
        ax.legend(loc="best", fontsize="small")
        return suggestion

    def _compute_inverse_from_svd(self, matrix, method: str, conditioning, tikhonov_reg: float):
        matrix = np.asarray(matrix, dtype=np.float64)
        num_modes = matrix.shape[1]

        if matrix.size == 0:
            return (
                np.zeros((num_modes, matrix.shape[0]), dtype=self.cm.dtype),
                np.array([], dtype=np.float64),
                np.array([], dtype=bool),
            )

        singular_values = np.linalg.svd(matrix, compute_uv=False)
        if singular_values.size == 0 or singular_values[0] <= 0:
            return (
                np.zeros((num_modes, matrix.shape[0]), dtype=self.cm.dtype),
                singular_values,
                np.zeros_like(singular_values, dtype=bool),
            )

        U, singular_values, Vh = np.linalg.svd(matrix, full_matrices=False)
        retained = singular_values > 0
        if conditioning is not None:
            retained &= singular_values >= (singular_values[0] / conditioning)

        inverse_singular_values = np.zeros_like(singular_values)
        if method == "svd":
            inverse_singular_values[retained] = 1.0 / singular_values[retained]
        else:
            if tikhonov_reg < 0:
                raise ValueError("tikhonov_reg must be non-negative")
            inverse_singular_values[retained] = singular_values[retained] / (
                singular_values[retained] ** 2 + tikhonov_reg**2
            )

        inverse = (Vh.T * inverse_singular_values) @ U.T
        return inverse.astype(self.cm.dtype, copy=False), singular_values, retained

    def compute_cm(self, method=None, num_dropped_modes=None, conditioning=None, tikhonov_reg=None):
        """
        Compute the control matrix from the interaction matrix.

        Parameters
        ----------
        method : str, optional
            Inversion method to use. Supported values are ``svd`` and
            ``tikhonov``. Defaults to the configured ``cm_method``.
        num_dropped_modes : int, optional
            Number of modal commands to suppress before inversion. Defaults to
            the configured ``num_dropped_modes``.
        conditioning : float, optional
            Optional target conditioning number. Singular values below
            ``max(s) / conditioning`` are discarded.
        tikhonov_reg : float, optional
            Tikhonov regularization strength used when ``method`` is
            ``tikhonov``. Defaults to the configured ``tikhonov_reg``.
        """
        component_logger = getattr(self, "logger", logger)
        try:
            method = self._validate_cm_method(self.cm_method if method is None else method)
            requested_dropped_modes = (
                self.num_dropped_modes if num_dropped_modes is None else int(num_dropped_modes)
            )
            requested_conditioning = self.conditioning if conditioning is None else conditioning
            requested_tikhonov = self.tikhonov_reg if tikhonov_reg is None else float(tikhonov_reg)

            if requested_conditioning is not None:
                requested_conditioning = float(requested_conditioning)
                if requested_conditioning <= 1:
                    raise ValueError("conditioning must be greater than 1 when provided")

            self.num_dropped_modes = requested_dropped_modes
            self.cm_method = method
            self.conditioning = requested_conditioning
            self.tikhonov_reg = requested_tikhonov
            self.num_active_modes = self.num_modes - self.num_dropped_modes
            if self.num_active_modes < 0:
                raise ValueError("Invalid number of modes used in CM. Check num_dropped_modes")
            active_im = self.im[:, : self.num_active_modes]
            inverse, singular_values, retained = self._compute_inverse_from_svd(
                active_im,
                method=self.cm_method,
                conditioning=self.conditioning,
                tikhonov_reg=self.tikhonov_reg,
            )

            self.cm[:, :] = 0
            self.cm[: self.num_active_modes, :] = inverse
            self.cm[self.num_active_modes :, :] = 0
            self.g_cm = self.gain * self.cm
            self.f_im = np.copy(self.im)
            self.f_im[:, self.num_active_modes :] = 0
            self.last_singular_values = singular_values
            self.last_retained_singular_mask = retained
            suggestion, fit = self._suggest_conditioning_from_singular_values(singular_values)
            self.last_suggested_conditioning = suggestion
            self.last_singular_value_fit = fit
            component_logger.info(
                "Computed control matrix method=%s active_modes=%s dropped_modes=%s conditioning=%s retained_singular_values=%s tikhonov_reg=%s",
                self.cm_method,
                self.num_active_modes,
                self.num_dropped_modes,
                self.conditioning,
                int(np.count_nonzero(retained)),
                self.tikhonov_reg,
            )
        except Exception:
            component_logger.exception("Failed to compute control matrix")
            raise
        return

    # @jit(nopython=True)
    def update_correction_pol(
        self, correction=np.array([], dtype=np.float32), slopes=np.array([], dtype=np.float32)
    ):
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
        # Compute POL Slopes s_{POL} = s_{RES} + im*c_{n-1}
        # print(f'slopes: {slopes.shape}, im: {self.im.shape}, corr: {correction.shape}')
        s_pol = slopes - self.f_im @ correction

        # Update Command Vector c_n = g*CM*s_{POL} + (1 − g) c_{n-1}  https://arxiv.org/pdf/1903.12124.pdf Eq 3
        return (1 - self.gain) * correction - np.dot(self.g_cm, s_pol)

    def standard_integrator_pol(self):
        """
        Standard integrator using the pseudo open loop slopes.
        """
        residual_slopes = self.read_stream("signal", out=self._signal_buffer)
        current_correction = self.read_stream("wfc", block=False, out=self._wfc_buffer)
        # print(f'slopes: {residual_slopes.shape}, im: {self.im.shape}, corr: {current_correction.shape}')

        new_correction = self.update_correction_pol(
            correction=current_correction, slopes=residual_slopes
        )
        new_correction[self.num_active_modes :] = 0
        self.send_to_wfc(new_correction)

        return

    def standard_integrator(self):
        """
        Standard integrator.
        """
        slopes = self.read_stream("signal", out=self._signal_buffer)
        new_correction = leaky_integrator_numba(
            slopes,
            self.g_cm,
            self.read_stream("wfc", block=False, out=self._wfc_buffer).squeeze(),
            self.null_correction,
            np.float32(0),  # No leak
            self.num_active_modes,
        )
        self.send_to_wfc(new_correction, slopes=slopes)
        return

    def leaky_integrator(self):
        """
        Leaky integrator.
        """
        slopes = self.read_stream("signal", out=self._signal_buffer)
        new_correction = leaky_integrator_numba(
            slopes,
            self.g_cm,
            self.read_stream("wfc", block=False, out=self._wfc_buffer).squeeze(),
            self.null_correction,
            np.float32(self.leaky_gain),
            self.num_active_modes,
        )
        self.send_to_wfc(new_correction, slopes=slopes)
        return

    def pid_integrator_pol(self):
        """
        PID integrator using the pseudo-open loop slopes.
        """
        slopes = self.read_stream("signal", out=self._signal_buffer)
        correction = self.read_stream("wfc", block=False, out=self._wfc_buffer)
        pol_slopes = slopes - self.f_im @ correction
        return self.pid_integrator(slopes=pol_slopes, correction=correction)

    def pid_integrator(self, slopes=None, correction=None):
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
            slopes = self.read_stream("signal")
        if correction is None:
            correction = self.read_stream("wfc", block=False)

        # Compute raw error term (numba accelerated)
        wf_error = comp_correction(cm=self.cm, slopes=slopes)
        derivative = wf_error - self.previous_wf_error

        # Apply low-pass filter to the derivative to reduce noise
        derivative = (
            self.derivative_filter * derivative
            + (1 - self.derivative_filter) * self.previous_derivative
        )

        # Update integral (anti-windup: conditional integration)
        # not_output_limiting = self.control_limits[0] is None or self.control_limits[1] is None
        is_clipped = np.any(self.control_output == self.control_limits[0]) or np.any(
            self.control_output == self.control_limits[1]
        )
        # Check to make sure we aren't actively clipping the correction
        if not is_clipped:
            # Add to integral
            self.integral += wf_error
            # Clip integral term
            self.integral = np.clip(self.integral, *self.integral_limits)

        # Calculate PID output
        control_output = (
            self.p_gain * wf_error + self.i_gain * self.integral + self.d_gain * derivative
        )

        control_output = np.clip(control_output, *self.control_limits)

        # Get new correction vector from the control output
        new_correction = (
            1 - self.leaky_gain
        ) * correction - control_output  # Negative control direction is convention for pyrtc

        # Remove anything in non-corrected modes (might be redundant)
        new_correction[self.num_active_modes :] = 0

        # Clip correction (force the loop to not over correct a mode)
        new_correction = np.clip(new_correction, *self.absolute_limits)

        # Apply new correction to mirror
        self.send_to_wfc(new_correction, slopes=slopes)

        # Save state for next iteration
        self.previous_wf_error = wf_error
        self.previous_derivative = derivative
        self.control_output = control_output

        return

    def send_to_wfc(self, correction, slopes=None):
        # Get an initial slope reading to set shapes
        correction = correction.reshape(self.flat.shape)
        if self.cl_docrime and isinstance(slopes, np.ndarray):
            slopes = slopes.reshape(slopes.size, 1)
            # Compute new random shape
            rand_shape = (
                np.random.uniform(-self.poke_amp, self.poke_amp, correction.size)
                .astype(self.docrime_buffer[0].dtype)
                .reshape(self.docrime_buffer[0].shape)
            )

            # Adds to end of buffer (i.e. pos -1)
            add_to_buffer(self.docrime_buffer, rand_shape)

            rand_shape = rand_shape.astype(correction.dtype).reshape(correction.shape)

            # Only add randomness to active modes, otherwise it will build up
            if self.num_active_modes > 0:
                correction[: self.num_active_modes] += rand_shape[: self.num_active_modes]
                correction[self.num_active_modes :] = rand_shape[self.num_active_modes :]
            else:
                correction = rand_shape

            # Send our new pertubation to the WFC
            self.write_stream("wfc", correction)

            # Correlate Current response with old correction by delay time
            self.docrime_cross += slopes @ self.docrime_buffer[0].T
            self.docrime_auto += self.docrime_buffer[0] @ self.docrime_buffer[0].T

            self.num_iters_dc += 1

        else:
            self.write_stream("wfc", correction)
        return

    def solve_docrime(self):

        component_logger = getattr(self, "logger", logger)
        try:
            self.cl_dcim = (self.docrime_cross / self.num_iters_dc) @ np.linalg.inv(
                self.docrime_auto / self.num_iters_dc
            )
            tmp_file_path = get_tmp_filepath(self.im_file, unique_str="CL_docrime")
            component_logger.info("Saving DOCRIME matrix to %s", tmp_file_path)
            np.save(tmp_file_path, self.cl_dcim)
        except Exception:
            component_logger.exception("Failed to solve DOCRIME interaction matrix")
            raise

        return

    def plot_im(self, row=None):

        plt.imshow(self.im, cmap="inferno", aspect="auto")
        plt.show()


if __name__ == "__main__":
    launch_component(Loop, "loop", start=False)
