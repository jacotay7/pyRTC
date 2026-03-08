"""
Wavefront Sensor Superclass
"""
import numpy as np

from pyRTC.logging_utils import get_logger
from pyRTC.Pipeline import initExistingShm
from pyRTC.pyRTCComponent import pyRTCComponent
from pyRTC.utils import append_to_file, generate_filepath, setFromConfig


logger = get_logger(__name__)

class Telemetry(pyRTCComponent):

    def __init__(self, conf) -> None:
        try:
            super().__init__(conf)

            self.dataDir = setFromConfig(conf, "dataDir", "./data/")

            self.mostRecentFile = ''
            self.allFiles = []
            self.dTypes = []
            self.dims = []
            self.logger.info("Initialized telemetry dataDir=%s", self.dataDir)
        except Exception:
            logger.exception("Failed to initialize telemetry")
            raise
        return

    def save(self, shmName, numFrames, uniqueStr = ''):

        component_logger = getattr(self, "logger", logger)
        try:
            shm, shmDims, shmDtype = initExistingShm(shmName)

            self.mostRecentFile = generate_filepath(base_dir=self.dataDir,
                                         prefix=f"{shmName}_{uniqueStr}")

            self.allFiles.append(self.mostRecentFile)
            self.dTypes.append(shmDtype)
            self.dims.append(shmDims)
            for _ in range(numFrames):
                append_to_file(self.mostRecentFile, shm.read(), dtype=shmDtype)
            component_logger.info("Saved %s frames from %s to %s", numFrames, shmName, self.mostRecentFile)
        except Exception:
            component_logger.exception("Failed to save telemetry stream %s", shmName)
            raise

        return
    
    def read(self, filename="", dtype = None):

        component_logger = getattr(self, "logger", logger)
        try:
            if filename == "":
                filename = self.mostRecentFile

            if filename in self.allFiles:
                arr = np.fromfile(filename, dtype=self.dTypes[self.allFiles.index(filename)])
                arr = arr.reshape(-1, *self.dims[self.allFiles.index(filename)])
                component_logger.info("Read telemetry capture from %s", filename)
                return arr
            if dtype is not None:
                arr = np.fromfile(filename, dtype=dtype)
                component_logger.info("Read raw telemetry file %s with dtype=%s", filename, dtype)
                return arr

            raise ValueError("File not part of current capture, please provide a dtype")
        except Exception:
            component_logger.exception("Failed to read telemetry file %s", filename or getattr(self, "mostRecentFile", ""))
            raise