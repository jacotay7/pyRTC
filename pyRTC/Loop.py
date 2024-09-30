import os 
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ['NUMBA_NUM_THREADS'] = '1'

from pyRTC.Pipeline import *
from pyRTC.utils import *
from pyRTC.pyRTCComponent import *
import argparse

import numpy as np
import matplotlib.pyplot as plt
import time
from numba import jit

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
                                resconstructionMatrix:torch.tensor, 
                                oldCorrection:np.ndarray,
                                leak:float,
                                numActiveModes:int
                                ):
    slopes_GPU = torch.tensor(slopes, device='cuda')
    correctionGPU = torch.matmul(resconstructionMatrix, slopes_GPU) 
    correctionGPU[numActiveModes:] = 0
    return np.subtract((1-leak)*oldCorrection, correctionGPU.cpu().numpy())

@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def compCorrection(CM=np.array([[]], dtype=np.float32),  
                    slopes=np.array([], dtype=np.float32)):
    return np.dot(CM,slopes)

@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def updateCorrection(correction=np.array([], dtype=np.float32), 
                     gCM=np.array([[]], dtype=np.float32),  
                     slopes=np.array([], dtype=np.float32)):
    return correction - np.dot(gCM,slopes)

# @jit(nopython=True)
# def updateCorrectionPerturb(correction=np.array([], dtype=np.float32),
#                             pertub=np.array([], dtype=np.float32),  
#                      gCM=np.array([[]], dtype=np.float32),  
#                      slopes=np.array([], dtype=np.float32)):
#     return correction - np.dot(gCM,slopes) + pertub

class Loop(pyRTCComponent):
    """
    A pyRTCComponent which controls the AO loop.

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

        super().__init__(conf) 
        self.name = "Loop"
        self.conf = conf
        
        #Read wfs signal's metadata and open a stream to the shared memory
        self.signalShm, self.signalShape, self.signalDType = initExistingShm("signal", gpuDevice = self.gpuDevice)
        self.signalSize = int(np.prod(self.signalShape))
        self.nullSignal = np.zeros(self.signalShape, dtype=self.signalDType)

        #Read wfc metadata and open a stream to the shared memory
        self.wfcShm, self.wfcShape, self.wfcDType = initExistingShm("wfc", gpuDevice = self.gpuDevice)
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
        
        self.clDocrime = False     
        self.numItersDC = 0   
        tmp2 = self.flat.copy()
        tmp2 = tmp2.reshape(tmp2.size,1) 
        tmp = self.nullSignal.copy()
        tmp = tmp.reshape(tmp.size,1)
        self.docrimeCross = np.zeros_like(tmp@tmp2.T)
        self.docrimeAuto = np.zeros_like(tmp2@tmp2.T)
        self.docrimeBuffer = np.zeros((1+self.delay, *tmp2.shape), 
                                dtype=self.wfcDType)
        
        """
        Terms for PID integrator
        """
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

        return

    def setGain(self, gain):
        """
        Set the integrator gain. Only needed for certain integrators.

        Parameters
        ----------
        gain : float
            Gain to set.
        """
        self.gain = gain
        self.gCM = self.gain*self.CM
        return

    def setPeturbAmp(self, amp):
        """
        Set the perturbation amplitude.

        Parameters
        ----------
        amp : float
            Amplitude to set.
        """
        self.perturbAmp = amp
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
        corrShapeWFC = correction.shape
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
        if self.IMMethod == 'docrime':
            self.docrimeIM()
        else:
            self.pushPullIM()

        self.computeCM()
        return
    
    def saveIM(self,filename=''):
        """
        Save the interaction matrix to a file.

        Parameters
        ----------
        filename : str, optional
            File to save the interaction matrix to. If not specified, uses the configured IMFile.
        """
        if filename == '':
            filename = self.IMFile
        np.save(filename, self.IM)

    def loadIM(self,filename=''):
        """
        Load the interaction matrix from a file.

        Parameters
        ----------
        filename : str, optional
            File to load the interaction matrix from. If not specified, uses the configured IMFile.
        """
        if filename == '':
            filename = self.IMFile
        if filename == '':
            self.IM = np.zeros_like(self.IM)
        else:
            self.IM = np.load(filename)
        self.computeCM()

    def flatten(self):
        """
        Send the flat correction to the wavefront corrector.
        """
        self.sendToWfc(self.flat)
        return
    
    def computeCM(self):
        """
        Compute the control matrix from the interaction matrix.
        """
        self.numActiveModes = self.numModes-self.numDroppedModes
        if self.numActiveModes < 0:
            print("Invalid Number of Modes used in CM. Check numDroppedModes")
            return
        self.CM[:self.numActiveModes,:] = np.linalg.pinv(self.IM[:,:self.numActiveModes], rcond=0)
        self.CM[self.numActiveModes:,:] = 0
        self.gCM = self.gain*self.CM
        self.fIM = np.copy(self.IM)
        self.fIM[:,self.numActiveModes:] = 0
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
                         self.wfcShm.read(SAFE=False).squeeze(),
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

        self.clDCIM = (self.docrimeCross/self.numItersDC)@np.linalg.inv(self.docrimeAuto/self.numItersDC)
        tmpFilePath = get_tmp_filepath(self.IMFile,uniqueStr="CL_docrime")
        print(f"Saving DOCRIME matrix to: {tmpFilePath}")
        np.save(tmpFilePath, self.clDCIM)

        return


    def plotIM(self, row=None):

        plt.imshow(self.IM, cmap = 'inferno', aspect='auto')
        plt.show()

if __name__ == "__main__":

    launchComponent(Loop, "loop", start = False)
