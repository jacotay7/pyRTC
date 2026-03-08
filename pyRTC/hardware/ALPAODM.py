"""ALPAO deformable-mirror adapter.

This module exposes a pyRTC-compatible wavefront-corrector implementation for
ALPAO mirrors driven through the vendor SDK. The adapter translates pyRTC modal
or zonal correction vectors into the actuator command format expected by the
device and centralizes mirror-specific initialization such as layout discovery,
command clipping, and optional floating-actuator masking.
"""

import os 
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ['NUMBA_NUM_THREADS'] = '1'

import struct
import sys

import numpy as np

from pyRTC.logging_utils import get_logger
from pyRTC.Pipeline import launchComponent
from pyRTC.WavefrontCorrector import WavefrontCorrector


logger = get_logger(__name__)


#Prevents camera output from messing with communication
original_stdout = sys.stdout
_devnull_stdout = open(os.devnull, 'w')
try:
    sys.stdout = _devnull_stdout
    ''' Add '/Lib' or '/Lib64' to path '''
    if (8 * struct.calcsize("P")) == 32:
        #Use x86 libraries.
        from Lib.asdk import DM
    else:
        #Use x86_64 libraries.
        from Lib64.asdk import DM
finally:
    sys.stdout = original_stdout
    _devnull_stdout.close()

class ALPAODM(WavefrontCorrector):
    """Wavefront-corrector adapter for an ALPAO deformable mirror.

    The class wraps the ALPAO SDK object and presents it through the standard
    ``WavefrontCorrector`` interface used by the rest of pyRTC. It is
    responsible for discovering the mirror geometry, applying safety limits to
    outgoing commands, handling optional floating-actuator masks, and resetting
    the device on teardown.
    """

    def __init__(self, conf) -> None:
        try:
            super().__init__(conf)

            self.serial = conf["serial"]
            self.dm = DM(self.serial)
            self.CAP = conf["commandCap"]
            self.numActuators = int(self.dm.Get('NBOfActuator'))

            layout = self.generateLayout()
            self.setLayout(layout)

            floating_file = conf.get("floatingActuatorsFile", "")
            if floating_file.endswith('.npy'):
                floatActuatorInds = np.load(floating_file)
                self.deactivateActuators(floatActuatorInds)
                self.logger.info("Loaded floating actuators from %s", floating_file)

            self.flatten()
            self.logger.info("Initialized ALPAO DM serial=%s actuators=%s cap=%s", self.serial, self.numActuators, self.CAP)
        except Exception:
            logger.exception("Failed to initialize ALPAO DM")
            raise

        return

    def generateLayout(self):
        try:
            if self.numActuators == 97:
                xx, yy = np.meshgrid(np.arange(11), np.arange(11))
                layout = np.sqrt((xx - 5)**2 + (yy - 5)**2) < 5.5
                self.logger.info("Generated ALPAO 97-actuator layout")
                return layout
            raise ValueError(f"Unsupported ALPAO actuator count: {self.numActuators}")
        except Exception:
            self.logger.exception("Failed to generate ALPAO layout for actuators=%s", getattr(self, "numActuators", None))
            raise
    
    def sendToHardware(self):
        #Do all of the normal updating of the super class
        super().sendToHardware()
        #Cap the Commands to reduce likelihood of DM failiure
        self.currentShape = np.clip(self.currentShape, -self.CAP, self.CAP)
        #Send the correction to the actual mirror
        self.dm.Send(self.currentShape)
        return

    def __del__(self):
        component_logger = getattr(self, "logger", logger)
        try:
            super().__del__()
        finally:
            dm = getattr(self, "dm", None)
            if dm is not None:
                try:
                    dm.Reset()
                    component_logger.info("Reset ALPAO DM")
                except Exception:
                    component_logger.exception("Failed while resetting ALPAO DM")
        return
    

if __name__ == "__main__":

    launchComponent(ALPAODM, "wfc", start = True)