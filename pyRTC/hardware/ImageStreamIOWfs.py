from pyRTC.WavefrontSensor import *
from pyRTC.Pipeline import *
from pyRTC.utils import *
import argparse
import sys
import os

import ImageStreamIOWrap as ISIO

class ISIOWfs(WavefrontSensor):

    def __init__(self, conf):
        super().__init__(conf)

        self.isioShmName = conf["isioShmName"]

        self.isioShm = ISIO.Image()
        self.isioShm.open(self.isioShmName)
        self.isioShm.semflush(-1)

        return

    def expose(self):
        
        self.isioShm.semwait(1)
        np.copyto(self.data, self.isioShm.copy().reshape(self.data.shape))

        super().expose()

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
    set_affinity((conf["wfs"]["affinity"])%os.cpu_count()) 
    decrease_nice(pid)

    confWFS = conf["wfs"]
    wfs = ISIOWfs(conf=confWFS)

    wfs.start()
    
    l = Listener(wfs, port= int(args.port))
    while l.running:
        l.listen()
        time.sleep(1e-3)