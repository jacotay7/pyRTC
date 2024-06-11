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

@jit(nopython=True)
def compCorrection(CM=np.array([[]], dtype=np.float32),  
                    slopes=np.array([], dtype=np.float32)):
    return np.dot(CM,slopes)

@jit(nopython=True)
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
    confWFS : dict
        Wavefront sensor configuration.
    confWFC : dict
        Wavefront corrector configuration.
    confLoop : dict
        Loop configuration.
    name : str
        Name of the loop.
    signalMeta : numpy.ndarray
        Metadata of the wavefront sensor signal.
    signalDType : type
        Data type of the wavefront sensor signal.
    signalSize : int
        Size of the wavefront sensor signal.
    signalShm : ImageSHM
        Shared memory object for the wavefront sensor signal.
    nullSignal : numpy.ndarray
        Null signal.
    signal2DMeta : numpy.ndarray
        Metadata of the 2D wavefront sensor signal.
    signal2DDType : type
        Data type of the 2D wavefront sensor signal.
    signal2DSize : int
        Size of the 2D wavefront sensor signal.
    signal2D_width : int
        Width of the 2D wavefront sensor signal.
    signal2D_height : int
        Height of the 2D wavefront sensor signal.
    signal2DShm : ImageSHM
        Shared memory object for the 2D wavefront sensor signal.
    wfcMeta : numpy.ndarray
        Metadata of the wavefront corrector.
    wfcDType : type
        Data type of the wavefront corrector.
    numModes : int
        Number of modes in the wavefront corrector.
    wfcShm : ImageSHM
        Shared memory object for the wavefront corrector.
    wfc2DMeta : numpy.ndarray
        Metadata of the 2D wavefront corrector.
    wfc2DDType : type
        Data type of the 2D wavefront corrector.
    wfc2DSize : int
        Size of the 2D wavefront corrector signal.
    wfc2D_width : int
        Width of the 2D wavefront corrector signal.
    wfc2D_height : int
        Height of the 2D wavefront corrector signal.
    wfc2DShm : ImageSHM
        Shared memory object for the 2D wavefront corrector.
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
        self.confWFS = conf["wfs"]
        self.confWFC = conf["wfc"]
        self.confLoop = conf["loop"]
        self.name = "Loop"
        
        #Read wfs signal's metadata and open a stream to the shared memory
        self.signalMeta = ImageSHM("signal_meta", (ImageSHM.METADATA_SIZE,), np.float64).read_noblock_safe()
        self.signalDType = float_to_dtype(self.signalMeta[3])
        self.signalSize = int(self.signalMeta[2]//self.signalDType.itemsize)
        self.signalShm = ImageSHM("signal", (self.signalSize,), self.signalDType)
        self.nullSignal = np.zeros(self.signalSize, dtype=self.signalDType)

        #Read wfs SLOPES metadata and open a stream to the shared memory
        self.signal2DMeta = ImageSHM("signal2D_meta", (ImageSHM.METADATA_SIZE,), np.float64).read_noblock_safe()
        self.signal2DDType = float_to_dtype(self.signal2DMeta[3])
        self.signal2DSize = int(self.signal2DMeta[2]//self.signal2DDType.itemsize)
        self.signal2D_width, self.signal2D_height = int(self.signal2DMeta[4]),  int(self.signal2DMeta[5])
        print(self.signal2DMeta[3], (self.signal2D_width, self.signal2D_height), self.signal2DDType)
        self.signal2DShm = ImageSHM("signal2D", (self.signal2D_width, self.signal2D_height), self.signal2DDType)

        #Read wfc metadata and open a stream to the shared memory
        self.wfcMeta = ImageSHM("wfc_meta", (ImageSHM.METADATA_SIZE,), np.float64).read_noblock_safe()
        self.wfcDType = float_to_dtype(self.wfcMeta[3])
        self.numModes = int(self.wfcMeta[2]//self.wfcDType.itemsize)
        self.wfcShm = ImageSHM("wfc", (self.numModes,), self.wfcDType)

        #Read the wfc2D metadata and open a stream to the shared memory
        self.wfc2DMeta = ImageSHM("wfc2D_meta", (ImageSHM.METADATA_SIZE,), np.float64).read_noblock_safe()
        self.wfc2DDType = float_to_dtype(self.wfc2DMeta[3])
        self.wfc2DSize = int(self.wfc2DMeta[2]//self.wfc2DDType.itemsize)
        self.wfc2D_width, self.wfc2D_height = int(self.wfc2DMeta[4]),  int(self.wfc2DMeta[5])
        self.wfc2DShm = ImageSHM("wfc2D", (self.wfc2D_width, self.wfc2D_height), self.wfc2DDType)

        self.numDroppedModes = setFromConfig(self.confLoop, "numDroppedModes", 0)
        self.numActiveModes = self.numModes - self.numDroppedModes
        self.flat = np.zeros(self.numModes, dtype=self.wfcDType)

        self.IM = np.zeros((self.signalSize, self.numModes),dtype=self.signalDType)
        self.CM = np.zeros((self.numModes, self.signalSize),dtype=self.signalDType)
        self.gain = setFromConfig(self.confLoop, "gain", 0.1)
        self.leakyGain = setFromConfig(self.confLoop, "leakyGain", 0.0)
        self.perturbAmp = 0
        self.hardwareDelay = setFromConfig(self.confWFC, "hardwareDelay", 0.0)
        self.pokeAmp = setFromConfig(self.confLoop, "pokeAmp", 1e-2)
        self.numItersIM = setFromConfig(self.confLoop, "numItersIM", 100) 
        self.delay = setFromConfig(self.confLoop, "delay", 0)
        self.IMMethod = setFromConfig(self.confLoop, "IMMethod", "push-pull") 
        self.IMFile = setFromConfig(self.confLoop, "IMFile", "")
        

        """
        Terms for PID integrator
        """
        self.pGain = setFromConfig(self.confLoop, "pGain", 0.1)
        self.iGain = setFromConfig(self.confLoop, "iGain", 0.0)
        self.dGain = setFromConfig(self.confLoop, "dGain", 0.0)
        self.controlLimits = setFromConfig(self.confLoop, "controlLimits", [-np.inf, np.inf])
        self.integralLimits = setFromConfig(self.confLoop, "integralLimits", [-np.inf, np.inf])
        self.absoluteLimits = setFromConfig(self.confLoop, "absoluteLimits", [-np.inf, np.inf])
        self.derivativeFilter = setFromConfig(self.confLoop, "derivativeFilter", 0.1)
        self.integral = 0

        self.previousWfError = np.zeros_like(self.wfcShm.read_noblock())
        self.previousDerivative = np.zeros_like(self.previousWfError)
        self.controlOutput = np.zeros_like(self.previousWfError)

        self.loadIM()

        super().__init__(self.confLoop)        
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
            self.wfcShm.write(correction)
            #Add some delay to ensure one-to-one
            time.sleep(self.hardwareDelay)
            #Burn the first new image since we were moving the DM during the exposure
            self.signalShm.read()
            #Average out N new WFS frames
            tmp_plus = np.zeros_like(self.IM[:,i])
            for n in range(self.numItersIM):
                tmp_plus += self.signalShm.read()
            tmp_plus /= self.numItersIM

            #Minus amplitude
            correction[i] = -self.pokeAmp
            #Post a new shape to be made
            self.wfcShm.write(correction)
            #Add some delay to ensure one-to-one
            time.sleep(self.hardwareDelay)
            #Burn the first new image since we were moving the DM during the exposure
            self.signalShm.read()
            #Average out N new WFS frames
            tmp_minus = np.zeros_like(self.IM[:,i])
            for n in range(self.numItersIM):
                tmp_minus += self.signalShm.read()
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
        corrections = np.zeros((1+self.delay, *correction.shape), dtype=correction.dtype)

        #Get an initial slope reading to set shapes
        slopes = self.nullSignal.copy()
        slopes = slopes.reshape(slopes.size,1)
        cross = np.zeros_like(slopes@correction.T)
        auto = np.zeros_like(correction@correction.T)

        for i in range(self.numItersIM):
            #Compute new random shape
            correction = np.random.uniform(-self.pokeAmp,self.pokeAmp,correction.size).astype(correction.dtype).reshape(correction.shape)
            #If we are in Closed Loop
            if self.running:
                #Read the current shape of the WFC and add our perturbation ontop
                correction = self.wfcShm.read_noblock() + correction

            #Send our new pertubation to the WFC
            self.wfcShm.write(correction.reshape(corrShapeWFC))
            #Move old shapes back in history
            corrections[:-1] = corrections[1:]
            #Add new correction
            corrections[-1] = correction

            #Get current WFS response
            slopes = self.signalShm.read().reshape(slopes.shape)
        
            #Correlate Current response with old correction by delay time
            cross += slopes@corrections[0].T
            auto += corrections[0]@corrections[0].T

        cross /= self.numItersIM 
        auto /= self.numItersIM
        self.IM = cross@np.linalg.inv(auto) 
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
        self.wfcShm.write(self.flat)
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
        residual_slopes = self.signalShm.read()
        currentCorrection = self.wfcShm.read()
        # print(f'slopes: {residual_slopes.shape}, IM: {self.IM.shape}, corr: {currentCorrection.shape}')

        newCorrection = self.updateCorrectionPOL(correction=currentCorrection, 
                                                 slopes=residual_slopes)
        newCorrection[self.numActiveModes:] = 0
        self.wfcShm.write(newCorrection)

        return

    
    def standardIntegrator(self):
        """
        Standard integrator.
        """
        slopes = self.signalShm.read()
        currentCorrection = self.wfcShm.read()
        newCorrection = updateCorrection(correction=currentCorrection, 
                                        gCM=self.gCM, 
                                        slopes=slopes)
        newCorrection[self.numActiveModes:] = 0
        self.wfcShm.write(newCorrection)
        return
    
    
    def leakyIntegrator(self):
        """
        Leaky integrator.
        """
        slopes = self.signalShm.read()
        currentCorrection = (1-self.leakyGain)*self.wfcShm.read()
        newCorrection = updateCorrection(correction=currentCorrection, 
                                        gCM=self.gCM, 
                                        slopes=slopes)
        newCorrection[self.numActiveModes:] = 0
        self.wfcShm.write(newCorrection)
        return

    def pidIntegratorPOL(self):
        """
        PID integrator using the pseudo-open loop slopes.
        """
        slopes = self.signalShm.read()
        correction = self.wfcShm.read()
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
            slopes = self.signalShm.read()
        if correction is None:
            correction = self.wfcShm.read()

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
        self.wfcShm.write(newCorrection)

        # Save state for next iteration
        self.previousWfError = wfError
        self.previousDerivative = derivative
        self.controlOutput = controlOutput
        
        return

    def plotIM(self):
        """
        Plot the interaction matrix.
        """
        plt.imshow(self.IM, cmap = 'inferno', aspect='auto')
        plt.show()

if __name__ == "__main__":

    # Create argument parser
    parser = argparse.ArgumentParser(description="Read a config file from the command line.")

    # Add command-line argument for the config file
    parser.add_argument("-c", "--config", required=True, help="Path to the config file")
    parser.add_argument("-p", "--port", required=True, help="Port for communication")

    # Parse command-line arguments
    args = parser.parse_args()

    conf = read_yaml_file(args.config)

    pid = os.getpid()
    set_affinity((conf["loop"]["affinity"])%os.cpu_count()) 
    decrease_nice(pid)

    loop = Loop(conf=conf)
    
    l = Listener(loop, port= int(args.port))
    while l.running:
        l.listen()
        time.sleep(1e-3)