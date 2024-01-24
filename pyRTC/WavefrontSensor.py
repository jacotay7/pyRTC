"""
Wavefront Sensor Superclass
"""
from pyRTC.Pipeline import ImageSHM, work
from pyRTC.utils import *
import numpy as np
import matplotlib.pyplot as plt
import threading
import os
import time
from numba import jit
from sys import platform

class WavefrontSensor:

    def __init__(self, conf) -> None:

        self.name = conf["name"]
        self.imageShape = (conf["width"], conf["height"])
        self.imageRawDType = np.uint16
        self.imageDType = np.int32
        
        self.imageRaw = ImageSHM("wfsRaw", self.imageShape, self.imageRawDType)
        self.image = ImageSHM("wfs", self.imageShape, self.imageDType)

        self.data = np.zeros(self.imageShape, dtype=self.imageRawDType)
        self.dark = np.zeros(self.imageShape, dtype=self.imageDType)

        self.affinity = conf["affinity"]
        self.darkCount = setFromConfig(conf, "darkCount", 1000)
        self.darkFile = setFromConfig(conf, "darkFile", "")

        self.loadDark()

        self.alive = True
        self.running = False

        self.workThreads = []
        functionsToRun = conf["functions"]
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
        self.alive = False
        return

    def start(self):

        self.running = True
        return
    
    def stop(self):

        self.running = False
        return
    
    def setRoi(self, roi):

        self.roiWidth = roi[0]
        self.roiHeight = roi[1]
        self.roiLeft = roi[2]
        self.roiTop = roi[3]
        return

    def setExposure(self, exposure):
        self.exposure = exposure
        return
    
    def setBinning(self, binning):
        self.binning = binning
        return
    
    def setGain(self, gain):
        self.gain = gain
        return
    
    def setBitDepth(self, bitDepth):
        self.bitDepth = bitDepth
        return
    
    def expose(self):
        self.imageRaw.write(self.data)
        self.image.write(self.data.astype(self.imageDType) - self.dark)
        return

    def read(self):
        return self.image.read()
    
    def takeDark(self):
        self.setDark(np.zeros_like(self.dark))
        dark = np.zeros(self.imageShape, dtype=np.float64)
        for i in range(self.darkCount):
            dark += self.read().astype(np.float64)
        dark /= self.darkCount
        self.setDark(dark)        
        return 

    def setDark(self, dark):
        self.dark = dark.astype(self.imageDType)
        return
    
    def saveDark(self,filename=''):
        if filename == '':
            filename = self.darkFile
        np.save(filename, self.dark)
        return
    
    def loadDark(self,filename=''):
        #If no file given, first try dark file
        if filename == '':
            filename = self.darkFile
        #If we are still without a file, set zeros
        if filename == '':
            self.dark = np.zeros_like(self.dark)
        else: #If we have a filename
            self.dark = np.load(filename)
        return
    
    def plot(self):
        arr = self.read()
        plt.imshow(arr, cmap = 'inferno', origin='lower')
        plt.colorbar()
        plt.show()
        return