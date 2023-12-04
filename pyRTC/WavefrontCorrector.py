"""
Wavefront Corrector Superclass
"""
from pyRTC.Pipeline import ImageSHM, work
import numpy as np
import matplotlib.pyplot as plt
import threading
import os
from numba import jit

@jit(nopython=True)
def ModaltoZonalWithFlat(correction=np.array([],dtype=np.float32), 
                       M2C=np.array([[]],dtype=np.float32), 
                       flat=np.array([],dtype=np.float32)):
    return M2C@correction + flat


class WavefrontCorrector:

    def __init__(self, dofs, layout=None, M2C=None) -> None:

        self.dofs = dofs
        self.correctionVector = ImageSHM("wfc", (self.dofs,), np.float32)
        self.correctionVector2D = None
        #If its an array it will initialize a 2D correction ImageSHM for display
        self.setLayout(layout)

        #Set an initial Flat
        self.flat = np.zeros(self.dofs, dtype=np.float32)
        self.currentShape = np.zeros_like(self.flat)
        
        self.affinity = 10
        self.alive = True
        self.running = False

        #Initialize the basis for corrections
        self.M2C = M2C #[dofs, numModes]
        #If not specified, create zonal basis
        if self.M2C is None:
            self.M2C = np.eye(self.dofs)
        self.M2C = self.M2C.astype(self.flat.dtype)
        self.C2M = np.linalg.pinv(self.M2C)
        self.currentCorrection = np.zeros(self.M2C.shape[1], dtype=self.flat.dtype)

        self.numModes = M2C.shape[1]
        functionsToRun = ["sendToHardware"]
        self.workThreads = []
        for i, functionName in enumerate(functionsToRun):
            # Launch a separate thread
            workThread = threading.Thread(target=work, args = (self,functionName), daemon=True)
            # Start the thread
            workThread.start()
            # Set CPU affinity for the thread
            os.sched_setaffinity(workThread.native_id, {(self.affinity+i)%os.cpu_count(),})  
            self.workThreads.append(workThread)
        return
    
    def __del__(self):
        print("Deleeting WFC Object")
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
    
    
    def sendToHardware(self,flagInd=0):
        self.currentShape = self.correctionVector.read(flagInd=flagInd)
        #Overwrite with hardware instructions
        return
    
    # def sendToHardware(self,correction,flagInd=0):
    #     self.writeZonal(correction)
    #     self.sendToHardware()
    #     return

    def read(self):
        return self.currentCorrection

    def readZonal(self):
        return self.currentShape
    
    def write(self, correction):
        self.currentCorrection = correction
        return self.writeZonal(ModaltoZonalWithFlat(correction, self.M2C, self.flat))
    
    def writeZonal(self, correction):
        self.correctionVector.write(correction)
        if isinstance(self.correctionVector2D, ImageSHM):
            self.correctionVector2D_template[self.layout] = correction - self.flat
            self.correctionVector2D.write(self.correctionVector2D_template)
        return

    def flatten(self):
        self.writeZonal(self.flat)
        return

    def plot(self, removeFlat=False):
        
        curCorrection = self.readZonal(flagInd=1)
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