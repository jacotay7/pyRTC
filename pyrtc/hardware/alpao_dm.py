"""ALPAO deformable-mirror adapter.

This module exposes a pyrtc-compatible wavefront-corrector implementation for
ALPAO mirrors driven through the vendor SDK. The adapter translates pyrtc modal
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

from pyrtc.logging_utils import get_logger
from pyrtc.manager import launch_component
from pyrtc.wavefront_corrector import WavefrontCorrector


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
    ``WavefrontCorrector`` interface used by the rest of pyrtc. It is
    responsible for discovering the mirror geometry, applying safety limits to
    outgoing commands, handling optional floating-actuator masks, and resetting
    the device on teardown.
    """

    def __init__(self, conf) -> None:
        try:
            super().__init__(conf)

            self.serial = conf["serial"]
            self.dm = DM(self.serial)
            self.CAP = self.command_cap
            self.num_actuators = int(self.dm.Get('NBOfActuator'))

            layout = self.generate_layout()
            self.set_layout(layout)

            floating_file = conf.get("floating_actuators_file", "")
            if floating_file.endswith('.npy'):
                float_actuator_inds = np.load(floating_file)
                self.deactivate_actuators(float_actuator_inds)
                self.logger.info("Loaded floating actuators from %s", floating_file)

            self.flatten()
            self.logger.info("Initialized ALPAO DM serial=%s actuators=%s cap=%s", self.serial, self.num_actuators, self.command_cap)
        except Exception:
            logger.exception("Failed to initialize ALPAO DM")
            raise

        return

    def generate_layout(self):
        try:
            if self.num_actuators == 97:
                xx, yy = np.meshgrid(np.arange(11), np.arange(11))
                layout = np.sqrt((xx - 5)**2 + (yy - 5)**2) < 5.5
                self.logger.info("Generated ALPAO 97-actuator layout")
                return layout
            raise ValueError(f"Unsupported ALPAO actuator count: {self.num_actuators}")
        except Exception:
            self.logger.exception("Failed to generate ALPAO layout for actuators=%s", getattr(self, "num_actuators", None))
            raise
    
    def send_to_hardware(self):
        #Do all of the normal updating of the super class
        super().send_to_hardware()
        #Send the correction to the actual mirror
        self.dm.Send(self.current_shape)
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

    launch_component(ALPAODM, "wfc", start = True)