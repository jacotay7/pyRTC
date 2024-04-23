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


class ScienceCamera:

    def __init__(self, conf) -> None:

        self.name = conf["name"]
        self.imageShape = (conf["width"], conf["height"])
        self.imageRawDType = np.uint16
        self.imageDType = np.int32
        self.psfLongDtype = np.float64
        
        self.psfShort = ImageSHM("psfShort", self.imageShape, self.imageDType)
        self.psfLong = ImageSHM("psfLong", self.imageShape, self.psfLongDtype)
        self.data = np.zeros(self.imageShape, dtype=self.imageRawDType)
        self.dark = np.zeros(self.imageShape, dtype=self.imageDType)
        self.darkCount = conf["darkCount"]
        self.darkFile = setFromConfig(conf, "darkFile", "")
        self.model = np.zeros(self.imageShape, dtype=self.psfLongDtype)
        self.modelFile = setFromConfig(conf, "modelFile", "")
        self.strehl_ratio = 0

        self.loadDark()
        self.loadModelPSF()

        self.integrationLength = conf["integration"]
        self.affinity = conf["affinity"]

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
            set_affinity((self.affinity+i)%os.cpu_count())
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
    
    def setIntegrationLength(self, integrationLength):
        self.integrationLength = integrationLength
        return
    
    def expose(self):
        self.psfShort.write(self.data.astype(self.imageDType) - self.dark)
        return

    def integrate(self):
        x = np.zeros(self.data.shape)
        for i in range(self.integrationLength):
            x += self.read().astype(x.dtype)
        self.psfLong.write(x/self.integrationLength)
        return 

    def read(self):
        return self.psfShort.read()
    
    def readLong(self):
        return self.psfLong.read()
    
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
    
    def takeModelPSF(self):
        self.model = self.readLong()
        return

    def setModelPSF(self, model):
        self.model = model.astype(self.psfLongDtype)
        return
    
    def saveModelPSF(self,filename=''):
        if filename == '':
            filename = self.modelFile
        np.save(filename, self.model)
        return
    
    def loadModelPSF(self,filename=''):
        #If no file given, first try dark file
        if filename == '':
            filename = self.modelFile
        #If we are still without a file, set zeros
        if filename == '':
            self.model = np.zeros_like(self.model)
        else: #If we have a filename
            self.model = np.load(filename)
        return

    def computeStrehl(self, median_filter_size = 1, gaussian_sigma = 0):


        model = clean_image_for_strehl(self.model, 
                                       median_filter_size = median_filter_size, 
                                       gaussian_sigma = gaussian_sigma)

        current = clean_image_for_strehl(self.readLong(), 
                                         median_filter_size = median_filter_size, 
                                         gaussian_sigma = gaussian_sigma)

        self.strehl_ratio = np.max(current) / np.max(model)

        return self.strehl_ratio

    def plot(self):
        arr = self.read()
        plt.imshow(arr, cmap = 'inferno', origin='lower')
        plt.colorbar()
        plt.show()
        return