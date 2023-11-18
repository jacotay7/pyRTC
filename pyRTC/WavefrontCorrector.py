"""
Wavefront Corrector Superclass
"""
from Pipeline import ImageSHM
import numpy as np

class WavefrontCorrector:

    def __init__(self, dofs) -> None:

        self.dofs = dofs
        self.CommandVector = ImageSHM("wfc", (self.dofs,), np.float64)
        return
    
    def applyCorrection(self):
        return
    
    def plot(self):
        return