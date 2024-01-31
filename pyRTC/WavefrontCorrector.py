"""
Wavefront Corrector Superclass
"""
from pyRTC.Pipeline import ImageSHM, work
from pyRTC.utils import *
import numpy as np
import matplotlib.pyplot as plt
import threading
import os
from numba import jit
from sys import platform

# @jit(nopython=True)
def ModaltoZonalWithFlat(correction=np.array([],dtype=np.float32), 
                       M2C=np.array([[]],dtype=np.float32), 
                       flat=np.array([],dtype=np.float32)):
    return M2C@correction + flat

class WavefrontCorrector:

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
        
        self.setDelay(setFromConfig(conf, "frameDelay", 0))

        self.saveFile = setFromConfig(conf, "saveFile", "wfcShape.npy")

        #Initialize the basis for corrections
        self.readM2C()

        self.alive = True
        self.running = False

        functionsToRun = conf["functions"]
        self.workThreads = []
        
        for i, functionName in enumerate(functionsToRun):
            # Launch a separate thread
            workThread = threading.Thread(target=work, args = (self,functionName), daemon=True)
            
            # Start the thread
            workThread.start()
            
            # Set CPU affinity for the thread
            if platform != 'darwin':
                os.sched_setaffinity(workThread.native_id, {self.affinity+i,})
            self.workThreads.append(workThread)
        return
    
    def __del__(self):
        self.stop()
        self.alive=False
        return
    
    def start(self):
        self.running = True
        return

    def stop(self):
        self.running = False
        return     

    def setFlat(self, flat):
        self.flat = flat
        return

    def setLayout(self, layout):
        self.layout = layout
        if isinstance(self.layout, np.ndarray):
            self.layout = self.layout > 0
            self.correctionVector2D = ImageSHM("wfc2D", self.layout.shape, np.float32)
            self.correctionVector2D.write(np.zeros(self.layout.shape, dtype=np.float32))
            self.correctionVector2D_template = self.correctionVector2D.read_noblock_safe()
        return
    
    def setM2C(self, M2C):

        if not isinstance(M2C, np.ndarray):
            self.M2C = np.eye(self.numActuators)
        else:
            self.M2C = M2C.astype(self.flat.dtype)

        self.C2M = np.linalg.pinv(self.M2C)
        self.numModes = self.M2C.shape[1]
        self.currentCorrection = np.zeros(self.numModes, dtype=self.flat.dtype)
        del self.correctionVector
        self.correctionVector = ImageSHM("wfc", (self.numModes,), np.float32)
        self.flatModal = self.C2M@self.flat

    def setDelay(self,delay):
        self.frameDelay = delay
        self.shapeBuffer = np.zeros((self.frameDelay+1, *self.currentShape.shape), dtype=self.currentShape.dtype)
        #Fill with current commands
        for i in range(self.shapeBuffer.shape[0]):
            self.shapeBuffer[i] = self.flat.copy()
        
        return
    
    def readM2C(self, filename=''):
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
        #Read a new modal correction in M2C basis
        self.currentCorrection = self.correctionVector.read()
        #If we added a frame delay
        if self.frameDelay > 0:
            #Roll back shape buffer by 1
            self.shapeBuffer[:-1] = self.shapeBuffer[1:]
            #Compute a new shape in zonal basis
            self.shapeBuffer[-1] = ModaltoZonalWithFlat(self.currentCorrection, self.M2C, self.flat)
            #Set the current shape
            self.currentShape = self.shapeBuffer[0]
        else:
            self.currentShape = ModaltoZonalWithFlat(self.currentCorrection, self.M2C, self.flat)
        #If we have a 2D SHM instance, update it 
        if isinstance(self.correctionVector2D, ImageSHM):
            self.correctionVector2D_template[self.layout] = self.currentShape - self.flat
            self.correctionVector2D.write(self.correctionVector2D_template)
        #Overwrite with hardware instructions after this to send to hardware
        return

    def read(self):
        return self.currentCorrection

    def write(self, correction):
        self.currentCorrection = correction
        self.correctionVector.write(self.currentCorrection, )
        return 

    def flatten(self):
        self.write(np.zeros_like(self.currentCorrection))
        return

    def push(self, mode, amp):
        corr = np.zeros_like(self.currentCorrection)
        corr[int(mode)] = float(amp)
        self.write(corr)
        return

    def saveShape(self, filename=''):
        if filename == '':
            filename = self.saveFile
        np.save(filename, self.currentShape)
        return

    def plot(self, removeFlat=False):
        
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