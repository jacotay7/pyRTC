"""Wavefront-corrector abstractions and modal-to-zonal mapping helpers.

This module defines the base class used by pyRTC deformable mirrors and other
corrective devices. It manages command streams, flat handling, actuator masks,
and optional 2D layout views, while leaving hardware transport details to the
concrete adapter subclasses.
"""

import os 
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['TBB_NUM_THREADS'] = '1'

import numpy as np
import matplotlib.pyplot as plt
from numba import jit

from pyRTC.logging_utils import get_logger
from pyRTC.Pipeline import ImageSHM, launchComponent
from pyRTC.pyRTCComponent import pyRTCComponent
from pyRTC.utils import gaussian_2d_grid, setFromConfig

logger = get_logger(__name__)

@jit(nopython=True)
def ModaltoZonalWithFlat(correction=np.array([],dtype=np.float32), 
                       M2C=np.array([[]],dtype=np.float32),
                       flat=np.array([],dtype=np.float32)):
    """Project a modal correction into actuator space and add the flat shape."""

    return M2C@correction + flat

class WavefrontCorrector(pyRTCComponent):
    """
    Base class for deformable mirrors and other wavefront-correction devices.

    ``WavefrontCorrector`` is responsible for the control-plane machinery around
    command generation: SHM output, flat shapes, mode-to-command transforms,
    floating actuator handling, and delayed command buffers. Subclasses are left
    to implement the device-specific transport in ``sendToHardware``.

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
    commandCap : float or None
        Optional absolute limit applied to actuator-space commands before they
        are handed to the hardware adapter.
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
        try:
            super().__init__(conf)

            self.name = conf["name"]
            self.numActuators = conf["numActuators"]
            self.numModes = conf["numModes"]
            self.m2cFile = setFromConfig(conf, "m2cFile", "")

            self.correctionVector = ImageSHM("wfc", (self.numModes,), np.float32, gpuDevice=self.gpuDevice, consumer=False)
            self.register_output_stream("wfc", self.correctionVector)
            self.correctionVector2D = None

            self.setLayout(None)

            self.flat = np.zeros(self.numActuators, dtype=np.float32)
            self.flatModal = np.zeros(self.numModes, dtype=self.flat.dtype)
            self.currentShape = np.zeros_like(self.flat)
            self.flatFile = setFromConfig(conf, "flatFile", "")
            self.commandCap = setFromConfig(conf, "commandCap", None)
            self.loadFlat()

            self.actuatorStatus = np.array([True] * self.numActuators)
            self.index_map = None
            self.floatingInfluenceRadius = setFromConfig(conf, "floatingInfluenceRadius", 1)
            self.floatMatrix = np.eye(self.numActuators, dtype=self.flat.dtype)

            self.setDelay(setFromConfig(conf, "frameDelay", 0))

            self.saveFile = setFromConfig(conf, "saveFile", "wfcShape.npy")
            self.readM2C()
            self.logger.info(
                "Initialized wavefront corrector name=%s actuators=%s modes=%s commandCap=%s",
                self.name,
                self.numActuators,
                self.numModes,
                self.commandCap,
            )
        except Exception:
            logger.exception("Failed to initialize wavefront corrector")
            raise

        
        return

    def setFlat(self, flat):
        """
        Set the flat shape.

        Parameters
        ----------
        flat : numpy.ndarray
            Flat shape to set.
        """
        try:
            self.flat = flat.astype(self.flat.dtype)
            self.logger.info("Updated flat shape")
        except Exception:
            self.logger.exception("Failed to update flat shape")
            raise
        return

    def loadFlat(self,filename=''):
        """
        Loads the Flat from a file.

        Parameters
        ----------
        filename : str, optional
            Filename to load the dark frame from. If not specified, uses the dark file path from the configuration.
        """
        #If no file given, first try dark file
        try:
            if filename == '':
                filename = self.flatFile
            if filename == '':
                flat = np.zeros_like(self.flat)
                self.logger.info("No flat file configured; using zeros")
            else:
                if '.txt' in filename:
                    flat = np.genfromtxt(filename)
                elif '.npy' in filename:
                    flat = np.load(filename)
                else:
                    raise ValueError(f"Unsupported flat file format: {filename}")
                self.logger.info("Loaded flat from %s", filename)
            self.setFlat(flat)
        except Exception:
            self.logger.exception("Failed to load flat from %s", filename or self.flatFile)
            raise
        return
    def setLayout(self, layout):
        """
        Set the layout of the actuators.

        Parameters
        ----------
        layout : numpy.ndarray or None
            Layout of the actuators. Is converted to boolean if not already.
        """
        try:
            self.layout = layout
            if isinstance(self.layout, np.ndarray):
                self.layout = self.layout > 0
                self.correctionVector2D = ImageSHM("wfc2D", self.layout.shape, np.float32, gpuDevice=self.gpuDevice, consumer=False)
                self.register_output_stream("wfc2D", self.correctionVector2D, source_streams=["wfc"], lineage_source="wfc")
                self.write_stream("wfc2D", np.zeros(self.layout.shape, dtype=np.float32), source_streams=["wfc"], lineage_source="wfc")
                self.correctionVector2D_template = self.read_stream("wfc2D", block=False)

                self.index_map = np.zeros(self.layout.shape, dtype=int)
                self.index_map[self.layout > 0] = np.arange(np.sum(self.layout)).astype(int) + 1
                self.logger.info("Configured 2D correction layout shape=%s", self.layout.shape)
            else:
                self.logger.info("Cleared 2D correction layout")
        except Exception:
            self.logger.exception("Failed to set wavefront corrector layout")
            raise

        return
    
    def deactivateActuators(self, actuators):
        """
        Deactivate specified actuators. Actuators are assumed to be floating

        Parameters
        ----------
        actuators : list of int
            List of actuator indices to deactivate.
        """

        try:
            if hasattr(actuators, '__len__') and len(actuators) < 1:
                raise Exception("You have provided no actuators")
            if not hasattr(actuators, '__len__'):
                raise Exception("Actuators given as wrong type, please provide array or list")

            if isinstance(self.layout, np.ndarray):
                if len(self.layout.shape) != 2:
                    raise Exception("Layout must be 2 dimensions to float actuators. To remove dead actuators, remove them from the M2C. OR set the layout to be 2D and the floatingInfluenceRadius to a 0")
                act_to_float_mask = np.zeros_like(self.index_map)
                for act in actuators:
                    act_to_float_mask[np.where(self.index_map == act + 1)] = 1
                    self.actuatorStatus[act] = False

                for act in actuators:
                    i, j = np.where(self.index_map == act + 1)
                    inlfluence_map = gaussian_2d_grid(i, j, self.floatingInfluenceRadius, self.layout.shape[0])
                    inlfluence_map *= self.layout * (1 - act_to_float_mask)
                    inlfluence_map /= np.sum(inlfluence_map)
                    inlfluence_map[inlfluence_map < np.max(inlfluence_map) / 10] = 0
                    self.floatMatrix[act] = inlfluence_map[self.layout > 0]

                self.setM2C(self.M2C)
                self.logger.info("Deactivated actuators %s", actuators)
            else:
                logger.warning("No layout set for DM")
        except Exception:
            self.logger.exception("Failed to deactivate actuators %s", actuators)
            raise

        return
    
    def reactivateActuators(self, actuators):
        """
        Reactivate specified actuators.

        Parameters
        ----------
        actuators : list of int
            List of actuator indices to reactivate.
        """
        try:
            for act in actuators:
                self.actuatorStatus[act] = True
            self.floatMatrix = np.eye(self.numActuators, dtype=self.flat.dtype)
            actsToDeactivate = [i for i in range(self.numActuators) if not self.actuatorStatus[i]]
            if len(actsToDeactivate) > 0:
                self.deactivateActuators(actsToDeactivate)
            self.logger.info("Reactivated actuators %s", actuators)
        except Exception:
            self.logger.exception("Failed to reactivate actuators %s", actuators)
            raise
        return

    def setM2C(self, M2C):
        """
        Set the mode-to-command matrix. This is the basis for correction.

        Parameters
        ----------
        M2C : numpy.ndarray or None
            Mode-to-command matrix to set. Axes are [numActuators, numModes]
        """
        try:
            if not isinstance(M2C, np.ndarray):
                self.M2C = np.eye(self.numActuators)[:, :self.numModes]
            else:
                self.M2C = M2C

            self.M2C = self.M2C.astype(self.flat.dtype)

            self.f_M2C = self.floatMatrix @ self.M2C

            self.C2M = np.linalg.pinv(self.M2C)
            self.numModes = self.M2C.shape[1]
            self.currentCorrection = np.zeros(self.numModes, dtype=self.flat.dtype)
            self.flatModal = self.C2M @ self.flat
            self.logger.info("Configured M2C matrix shape=%s", self.M2C.shape)
        except Exception:
            self.logger.exception("Failed to configure M2C matrix")
            raise

    def setDelay(self,delay):
        """
        Sets an artificial frame delay. Used for testing, nominally the delay should always be zero.

        Parameters
        ----------
        delay : int
            Frame delay to set.
        """
        try:
            self.frameDelay = delay
            self.shapeBuffer = np.zeros((self.frameDelay + 1, *self.currentShape.shape), dtype=self.currentShape.dtype)
            for i in range(self.shapeBuffer.shape[0]):
                self.shapeBuffer[i] = self.flat.copy()
            self.logger.info("Set artificial frame delay to %s", delay)
        except Exception:
            self.logger.exception("Failed to set frame delay to %s", delay)
            raise
        
        return
    
    def readM2C(self, filename=''):
        """
        Read the mode-to-command matrix from a file.

        Parameters
        ----------
        filename : str, optional
            File to read the mode-to-command matrix from. If not specified, uses the configured m2cFile.
        """
        try:
            if filename == '':
                filename = self.m2cFile

            if '.dat' in filename:
                M2C = np.fromfile(filename, dtype=np.float64).reshape(self.numActuators, self.numModes)
            elif '.npy' in filename:
                M2C = np.load(filename)
            else:
                self.setM2C(None)
                self.logger.info("No M2C file configured; using identity basis")
                return
        
            self.setM2C(M2C)
            self.logger.info("Loaded M2C matrix from %s", filename)
        except Exception:
            self.logger.exception("Failed to read M2C matrix from %s", filename or self.m2cFile)
            raise
        return 
    
    def sendToHardware(self):
        """
        Send the current correction to the hardware. Nominally, this function is overwritten by the
        child hardware class and registered to the real-time loop from the config.
        """
        #Read a new modal correction in M2C basis
        self.currentCorrection = self.read_stream("wfc")
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

        if self.commandCap is not None:
            self.currentShape = np.clip(self.currentShape, -self.commandCap, self.commandCap)
        
        #If we have a 2D SHM instance, update it 
        if isinstance(self.correctionVector2D, ImageSHM):
            self.correctionVector2D_template[self.layout] = self.currentShape - self.flat
            self.write_stream("wfc2D", self.correctionVector2D_template, source_streams=["wfc"], lineage_source="wfc")
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
            return self.read_stream("wfc")
        return self.read_stream("wfc", block=False)

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
        self.write_stream("wfc", self.currentCorrection)
        return 

    def flatten(self):
        """
        Flatten the wavefront corrector.
        """
        #Sending a zero correction will be the flat since the correction
        #is always assumed to be on top of the flat.
        try:
            self.write(np.zeros_like(self.currentCorrection))
            self.logger.info("Flattened wavefront corrector")
        except Exception:
            self.logger.exception("Failed to flatten wavefront corrector")
            raise
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
        try:
            corr = np.zeros_like(self.currentCorrection)
            corr[int(mode)] = float(amp)
            self.write(corr)
            self.logger.info("Pushed mode %s with amplitude %s", mode, amp)
        except Exception:
            self.logger.exception("Failed to push mode %s with amplitude %s", mode, amp)
            raise
        return

    def saveShape(self, filename=''):
        """
        Save the current shape to a file.

        Parameters
        ----------
        filename : str, optional
            File to save the shape to. If not specified, uses the configured saveFile.
        """
        try:
            if filename == '':
                filename = self.saveFile
            if filename == '':
                raise ValueError("No output filename provided for shape save")
            np.save(filename, self.currentShape)
            self.logger.info("Saved current shape to %s", filename)
        except Exception:
            self.logger.exception("Failed to save current shape to %s", filename or self.saveFile)
            raise
        return

    def plot(self, addFlat=False):
        """
        Plot the current correction.

        Parameters
        ----------
        removeFlat : bool, optional
            If True, removes the flat shape from the current correction before plotting. Default is False.
        """
        curCorrection = self.read()
        if addFlat:
            curCorrection += self.flatModal

        if isinstance(self.layout, np.ndarray):
            newShape = np.zeros(self.layout.shape)
            newShape[self.layout] = self.M2C@curCorrection
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

if __name__ == "__main__":

    launchComponent(WavefrontCorrector, "wfc", start = True)