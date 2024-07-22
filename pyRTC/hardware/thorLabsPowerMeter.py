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

    # Create argument parser
    parser = argparse.ArgumentParser(description="Read a config file from the command line.")

    # Add command-line argument for the config file
    parser.add_argument("-c", "--config", required=True, help="Path to the config file")
    parser.add_argument("-p", "--port", required=True, help="Port for communication")

    # Parse command-line arguments
    args = parser.parse_args()

    conf = read_yaml_file(args.config)

    pid = os.getpid()
    set_affinity((conf["power"]["affinity"])%os.cpu_count()) 
    decrease_nice(pid)

    power = powerMeter(conf=conf["power"])
    power.start()

    l = Listener(power, port = int(args.port))
    while l.running:
        l.listen()
        time.sleep(1e-3)
        
