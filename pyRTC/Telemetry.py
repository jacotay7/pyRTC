"""
Wavefront Sensor Superclass
"""
from pyRTC.Pipeline import ImageSHM, work
from pyRTC.utils import *
from pyRTC.pyRTCComponent import *
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from sys import platform

class Telemetry(pyRTCComponent):

    def __init__(self, conf) -> None:

        super().__init__(conf)

        self.dataDir = setFromConfig(conf, "dataDir", "./data/")

        self.mostRecentFile = ''
        self.allFiles = []
        self.dTypes = []
        self.dims = []
        return

    def save(self, shmName, numFrames, uniqueStr = ''):

        shm, shmDims, shmDtype = initExistingShm(shmName)

        self.mostRecentFile = generate_filepath(base_dir=self.dataDir,
                                     prefix=f"{shmName}_{uniqueStr}")

        self.allFiles.append(self.mostRecentFile)
        self.dTypes.append(shmDtype)
        self.dims.append(shmDims)
        for i in range(numFrames):
            append_to_file(self.mostRecentFile, shm.read(), dtype=shmDtype)

        return
    
    def read(self, filename="", dtype = None):

        if filename == "":
            filename = self.mostRecentFile

        if filename in self.allFiles:
            arr = np.fromfile(filename, 
                            dtype=self.dTypes[self.allFiles.index(filename)])
            arr = arr.reshape(-1, *self.dims[self.allFiles.index(filename)])
            return arr
        elif dtype is not None:
            return np.fromfile(filename, dtype=dtype)
    
        else:
            print("File not part of current capture, please provide a dtype")

        return