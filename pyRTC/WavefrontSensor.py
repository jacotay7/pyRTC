"""
Wavefront Sensor Superclass
"""
from Pipeline import ImageSHM
import numpy as np

class WavefrontSensor:

    def __init__(self, imageShape) -> None:

        self.imageShape = imageShape
        self.image = ImageSHM("wfs", imageShape, np.float64)
        self.signal = ImageSHM("signal", imageShape, np.float64)
        return
    
    def expose(self):
        return

    def plot(self):
        return