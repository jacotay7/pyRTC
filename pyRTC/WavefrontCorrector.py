"""
Wavefront Corrector Superclass
"""
import os 
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['TBB_NUM_THREADS'] = '1'

from pyRTC.Pipeline import ImageSHM, work
from pyRTC.utils import *
from pyRTC.pyRTCComponent import *
import numpy as np
import matplotlib.pyplot as plt
from numba import jit

@jit(nopython=True)
def ModaltoZonalWithFlat(correction=np.array([],dtype=np.float32), 
                       M2C=np.array([[]],dtype=np.float32),
                       flat=np.array([],dtype=np.float32)):
    return M2C@correction + flat

class WavefrontCorrector(pyRTCComponent):
    """
    A pyRTCComponent which represents a Wavefront Corrector (DM, SLM, other). This is a general class which is 
    reponsible for all components of wavefront correcting which are common to all wavefront correctors. This
    class should be used by defining a child class held in pyRTC.hardware, which overwrites
    the relevant functions which actual hardware connectivity code. The child class can call its parent
    implementations in order to make use of the code which sets the relevant parameters, write to shared
    memory, etc... or they can overwrite them completely. See hardware/ALPAODM.py for an example.

    Config
    ------
    name : str
        Name of the wavefront corrector.
    numActuators : int
        Number of actuators. Required.
    numModes : int
        Number of modes. Required.
    affinity : str
        Affinity setting.
    m2cFile : str
        Path to the mode-to-command file.
    floatingInfluenceRadius : int, optional
        Radius for floating influence. Default is 1.
    frameDelay : int, optional
        Frame delay. Default is 0.
    saveFile : str, optional
        File to save the shape. Default is "wfcShape.npy".

    Attributes
    ----------
    name : str
        Name of the wavefront corrector.
    numActuators : int
        Number of actuators.
    numModes : int
        Number of modes.
    affinity : str
        Affinity setting.
    m2cFile : str
        Path to the mode-to-command file.
    correctionVector : ImageSHM
        Correction vector.
    correctionVector2D : ImageSHM or None
        2D correction vector for display.
    flat : numpy.ndarray
        Initial flat shape.
    flatModal : numpy.ndarray
        Flat shape in modal basis.
    currentShape : numpy.ndarray
        Current shape.
    actuatorStatus : numpy.ndarray
        Status of each actuator.
    index_map : numpy.ndarray or None
        Index map for actuators.
    floatingInfluenceRadius : int
        Radius for floating influence.
    floatMatrix : numpy.ndarray
        Floating actuator matrix.
    frameDelay : int
        Frame delay.
    saveFile : str
        File to save the shape.
    layout : numpy.ndarray or None
        Layout of the actuators.
    M2C : numpy.ndarray
        Mode-to-command matrix.
    f_M2C : numpy.ndarray
        Floating mode-to-command matrix.
    C2M : numpy.ndarray
        Command-to-mode matrix.
    currentCorrection : numpy.ndarray
        Current correction vector.
    shapeBuffer : numpy.ndarray
        Buffer for shapes with frame delay.
    correctionVector2D_template : numpy.ndarray
        Template for the 2D correction vector.
    """
    def __init__(self, conf) -> None:

        self.name = conf["name"]
        self.numActuators = conf["numActuators"]
        self.numModes = conf["numModes"]
        self.affinity = conf["affinity"]
        self.m2cFile = conf["m2cFile"]
        
        self.correctionVector = ImageSHM("wfc", (self.numModes,), np.float32)
        self.correctionVector2D = None
        
        #If its an array it will initialize a 2D correction ImageSHM for display
        self.setLayout(None)

        #Set an initial Flat
        self.flat = np.zeros(self.numActuators, dtype=np.float32)
        self.flatModal = np.zeros(self.numModes,  dtype=self.flat.dtype)
        self.currentShape = np.zeros_like(self.flat)
        
        #Initialize Floating Actuator Matrix
        self.actuatorStatus = np.array([True]*self.numActuators)
        self.index_map = None
        self.floatingInfluenceRadius = setFromConfig(conf, "floatingInfluenceRadius", 1)
        self.floatMatrix = np.eye(self.numActuators, dtype=self.flat.dtype)

        self.setDelay(setFromConfig(conf, "frameDelay", 0))

        self.saveFile = setFromConfig(conf, "saveFile", "wfcShape.npy")

        #Initialize the basis for corrections
        self.readM2C()

        super().__init__(conf)
        return

    def setFlat(self, flat):
        """
        Set the flat shape.

        Parameters
        ----------
        flat : numpy.ndarray
            Flat shape to set.
        """
        self.flat = flat
        return

    def setLayout(self, layout):
        """
        Set the layout of the actuators.

        Parameters
        ----------
        layout : numpy.ndarray or None
            Layout of the actuators. Is converted to boolean if not already.
        """
        self.layout = layout
        if isinstance(self.layout, np.ndarray):
            self.layout = self.layout > 0
            self.correctionVector2D = ImageSHM("wfc2D", self.layout.shape, np.float32)
            self.correctionVector2D.write(np.zeros(self.layout.shape, dtype=np.float32))
            self.correctionVector2D_template = self.correctionVector2D.read_noblock()

            self.index_map = np.zeros(self.layout.shape, dtype = int)
            self.index_map[self.layout > 0] = np.arange(np.sum(self.layout)).astype(int) + 1

        return
    
    def deactivateActuators(self, actuators):
        """
        Deactivate specified actuators. Actuators are assumed to be floating

        Parameters
        ----------
        actuators : list of int
            List of actuator indices to deactivate.
        """
        if len(actuators) == 0:
            return
        #Make sure that the layout has been set already
        if isinstance(self.layout, np.ndarray):

            #Make a boolean mask of the actuators we are deactivating
            act_to_float_mask = np.zeros_like(self.index_map)
            for act in actuators:
                act_to_float_mask[np.where(self.index_map == act+1)] = 1
                #Record that we deactivated this actuator.
                self.actuatorStatus[act] = False

            #For all of the actuators
            for act in actuators:
                #Get spatial location of the actuator
                i,j = np.where(self.index_map == act+1)
                #Get a gaussian region of influence
                inlfluence_map = gaussian_2d_grid(i,j, self.floatingInfluenceRadius, self.layout.shape[0])
                #Apply the DM layout mask excluding other floating actuators
                inlfluence_map *= self.layout*(1-act_to_float_mask)
                #Renormalize to sum to 1
                inlfluence_map /= np.sum(inlfluence_map)
                #Set a bound on the lowest influence to a tenth of the maximum
                inlfluence_map[inlfluence_map < np.max(inlfluence_map)/10] = 0
                #Vectorize and add to matrix
                self.floatMatrix[act] = inlfluence_map[self.layout>0]

            self.setM2C(self.M2C)

        else:
            print("No Layout Set for DM")

        return
    
    def reactivateActuators(self, actuators):
        """
        Reactivate specified actuators.

        Parameters
        ----------
        actuators : list of int
            List of actuator indices to reactivate.
        """
        #Set the status of each actuator back to True
        for act in actuators:
            self.actuatorStatus[act] = True
        #Reset Floating Actuator Map
        self.floatMatrix = np.eye(self.numActuators, dtype=self.flat.dtype)
        #Deactivate all actuators that are still disabled
        self.deactivateActuators([i for i in range(self.numActuators) if self.actuatorStatus[i] == False])
        return

    def setM2C(self, M2C):
        """
        Set the mode-to-command matrix. This is the basis for correction.

        Parameters
        ----------
        M2C : numpy.ndarray or None
            Mode-to-command matrix to set. Axes are [numActuators, numModes]
        """
        if not isinstance(M2C, np.ndarray):
            self.M2C = np.eye(self.numActuators)[:,:self.numModes]
        else:
            self.M2C = M2C

        self.M2C = self.M2C.astype(self.flat.dtype)

        self.f_M2C = self.floatMatrix@self.M2C

        self.C2M = np.linalg.pinv(self.M2C)
        self.numModes = self.M2C.shape[1]
        self.currentCorrection = np.zeros(self.numModes, dtype=self.flat.dtype)
        del self.correctionVector
        self.correctionVector = ImageSHM("wfc", (self.numModes,), np.float32)
        self.flatModal = self.C2M@self.flat

    def setDelay(self,delay):
        """
        Sets an artificial frame delay. Used for testing, nominally the delay should always be zero.

        Parameters
        ----------
        delay : int
            Frame delay to set.
        """
        self.frameDelay = delay
        self.shapeBuffer = np.zeros((self.frameDelay+1, *self.currentShape.shape), dtype=self.currentShape.dtype)
        #Fill with current commands
        for i in range(self.shapeBuffer.shape[0]):
            self.shapeBuffer[i] = self.flat.copy()
        
        return
    
    def readM2C(self, filename=''):
        """
        Read the mode-to-command matrix from a file.

        Parameters
        ----------
        filename : str, optional
            File to read the mode-to-command matrix from. If not specified, uses the configured m2cFile.
        """
        if filename == '':
            filename = self.m2cFile

        if '.dat' in filename:
            M2C = np.fromfile(filename,dtype=np.float64).reshape(self.numActuators,self.numModes)
        elif '.npy' in filename:
            M2C = np.load(filename)
        else:
            self.setM2C(None)
            return
        
        #Normalize each mode
        for i in range(M2C.shape[1]):
            M2C[:,i] /= np.std(M2C[:,i])
        self.setM2C(M2C)
        return 
    
    def sendToHardware(self):
        """
        Send the current correction to the hardware. Nominally, this function is overwritten by the
        child hardware class and registered to the real-time loop from the config.
        """
        #Read a new modal correction in M2C basis
        self.currentCorrection = self.correctionVector.read()
        #If we added a frame delay
        if self.frameDelay > 0:
            #Roll back shape buffer by 1
            self.shapeBuffer[:-1] = self.shapeBuffer[1:]
            #Compute a new shape in zonal basis
            self.shapeBuffer[-1] = ModaltoZonalWithFlat(self.currentCorrection, 
                                                        self.f_M2C,
                                                        self.flat)
            #Set the current shape
            self.currentShape = self.shapeBuffer[0]
        else:
            self.currentShape = ModaltoZonalWithFlat(self.currentCorrection, 
                                                     self.f_M2C,
                                                     self.flat)
        
        #If we have a 2D SHM instance, update it 
        if isinstance(self.correctionVector2D, ImageSHM):
            self.correctionVector2D_template[self.layout] = self.currentShape - self.flat
            self.correctionVector2D.write(self.correctionVector2D_template)
        #Overwrite with hardware instructions after this to send to hardware
        return

    def read(self, block = False):
        """
        Read the current correction vector.

        Returns
        -------
        numpy.ndarray
            Current correction vector.
        """
        if block:
            return self.correctionVector.read()
        return self.correctionVector.read_noblock()

    def write(self, correction):
        """
        Write a new correction.

        Parameters
        ----------
        correction : numpy.ndarray
            Correction vector to write.
        """
        self.currentCorrection = correction
        #We assume that sendToHardware is registered to the real-time loop
        #And that the WFC is running (i.e. start has been called)
        self.correctionVector.write(self.currentCorrection)
        return 

    def flatten(self):
        """
        Flatten the wavefront corrector.
        """
        #Sending a zero correction will be the flat since the correction
        #is always assumed to be on top of the flat.
        self.write(np.zeros_like(self.currentCorrection))
        return

    def push(self, mode, amp):
        """
        Push a specific mode with a given amplitude.

        Parameters
        ----------
        mode : int
            Mode index to push.
        amp : float
            Amplitude to push the mode with.
        """
        corr = np.zeros_like(self.currentCorrection)
        corr[int(mode)] = float(amp)
        self.write(corr)
        return

    def saveShape(self, filename=''):
        """
        Save the current shape to a file.

        Parameters
        ----------
        filename : str, optional
            File to save the shape to. If not specified, uses the configured saveFile.
        """
        if filename == '':
            filename = self.saveFile
        np.save(filename, self.currentShape)
        return

    def plot(self, removeFlat=False):
        """
        Plot the current correction.

        Parameters
        ----------
        removeFlat : bool, optional
            If True, removes the flat shape from the current correction before plotting. Default is False.
        """
        curCorrection = self.read()
        if removeFlat:
            curCorrection -= self.flat

        if isinstance(self.layout, np.ndarray):
            newShape = np.zeros(self.layout.shape)
            newShape[self.layout] = curCorrection
        else:
            newShape = curCorrection
            
        if len(newShape.shape) == 1:
            # plt.figure(figsize=(12,5))
            plt.plot(newShape)
            plt.show()
        elif len(newShape.shape) == 2:
            # plt.figure(figsize=(10,8))
            plt.imshow(newShape, cmap = "inferno", aspect='auto', origin='lower')
            plt.colorbar()
            plt.show()

        return