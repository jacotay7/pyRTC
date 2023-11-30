"""
Wavefront Corrector Superclass
"""
from pyRTC.Pipeline import ImageSHM
import numpy as np
import matplotlib.pyplot as plt

class WavefrontCorrector:

    def __init__(self, dofs, layout=None, M2C=None) -> None:

        self.dofs = dofs
        self.layout = layout
        self.correctionVector = ImageSHM("wfc", (self.dofs,), np.float64)

        self.flat = np.zeros_like(self.correctionVector.read())

        self.M2C = M2C
        if self.M2C is None:
            self.M2C = np.eye(self.dofs)
        self.C2M = np.linalg.pinv(self.M2C)

        return
    
    def setFlat(self, flat):
        self.flat = flat
        return

    def setLayout(self, layout):
        self.layout = layout
        return

    def applyCorrectionRaw(self, correction):
        self.correctionVector.write(correction)
        return
    
    def applyCorrection(self, correction):
        self.applyCorrectionRaw(self.M2C@correction + self.flat)

    def read(self):
        return self.getCorrection()

    def readRaw(self):
        return self.getCorrectionRaw()
    
    def write(self, correction):
        return self.applyCorrection(correction)

    def writeRaw(self, correction):
        return self.applyCorrectionRaw(correction)
    
    def getCorrection(self):

        return self.C2M@(self.correctionVector.read() - self.flat)

    def getCorrectionRaw(self):

        return self.correctionVector.read()

    def flatten(self):
        self.applyCorrectionRaw(self.flat)
        return

    def plot(self, removeFlat=False):
        
        curCorrection = self.readRaw()
        if removeFlat:
            curCorrection -= self.flat

        if not (self.layout is None):
            newShape = np.zeros(self.layout.shape)
            newShape[self.layout > 0] = curCorrection
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