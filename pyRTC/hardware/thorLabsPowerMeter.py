from pyRTC.ScienceCamera import *
from pyRTC.Pipeline import *
from pyRTC.utils import *

import argparse
import os 

from ThorlabsPM100 import ThorlabsPM100, usbtmc
import errno 

class powerMeter(ScienceCamera):

    def __init__(self, conf):
        super().__init__(conf)
        
        self.device = setFromConfig(conf, "device", "/dev/usbtmc0")

        self.power_meter = None

        try:
            inst = usbtmc.USBTMC(device=self.device)
            self.power_meter = ThorlabsPM100(inst=inst)
        except OSError as e:
            if e.errno != errno.EACCES:
                print('Device not found.')
            else:
                print(f'permission denied. Run: sudo chown USERNAME {self.device}')
        
        assert(self.power_meter is not None)

        if "exposure" in conf:
            self.setExposure(conf["exposure"])

        return

    
    def setExposure(self, exposure):
        super().setExposure(exposure)

        self.power_meter.sense.average.count = int(exposure) # write property

        return
    
    def expose(self):
        
        # power = self.power_meter.read

        self.data[0,0] = self.power_meter.read

        super().expose()

        return

    def __del__(self):
        super().__del__()
        
        return

if __name__ == "__main__":

    launchComponent(powerMeter, "power", start = True)
