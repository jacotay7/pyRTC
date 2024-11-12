from pyRTC.ScienceCamera import *
from pyRTC.Pipeline import *
from pyRTC.utils import *

import argparse
import os 

import serial
import errno 

class photoDetector(ScienceCamera):

    def __init__(self, conf):
        super().__init__(conf)
        
        self.device = setFromConfig(conf, "device", '/dev/ttyACM0')
        self.baud = setFromConfig(conf, "baud", 9600)

        self.photodetector = serial.Serial(self.device,self.baud)

        self.tempShm = ImageSHM("temp", self.imageShape, self.imageDType)
        self.tempData = np.zeros_like(self.data)

        if "exposure" in conf:
            self.setExposure(conf["exposure"])

        return
    
    def singleExposure(self):
        # power = self.power_meter.read
        value = 0
        # self.data[0,0] = self.power_meter.read
        line = self.photodetector.readline()
        read = False
        if len(line) > 2:
            try:
                # print(line, line.decode('utf-8').rstrip(), line.decode('utf-8').rstrip().split('-',1))
                letter, val = line.decode('utf-8').rstrip().split('-',1)
                if is_numeric(val):
                    if letter == 'V':
                        value = np.float32(val)
                        read = True
                        
                        
                    elif letter == 'T':
                        self.tempData[0,0] = np.float32(val)
                        self.tempShm.write(self.tempData)
                        read = True

            except:
                pass
        if not read:
            print(f"Unable to Parse Line: {line}")
        return value
    def expose(self):
        value = 0
        for i in range(self.exposure):
            value += self.singleExposure()
        self.data[0,0] = np.float32(value/self.exposure)
        super().expose()
        return

    def __del__(self):
        self.photodetector.close()

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
    set_affinity((conf["photodetect"]["affinity"])%os.cpu_count()) 
    decrease_nice(pid)

    power = photoDetector(conf=conf["photodetect"])
    power.start()

    l = Listener(power, port = int(args.port))
    while l.running:
        l.listen()
        time.sleep(1e-3)
        
