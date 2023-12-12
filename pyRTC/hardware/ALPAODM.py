from pyRTC.WavefrontCorrector import *
from pyRTC.Pipeline import *
from pyRTC.utils import *
import struct
import argparse
import sys
import os 

#Prevents camera output from messing with communication
original_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')
''' Add '/Lib' or '/Lib64' to path '''
if (8 * struct.calcsize("P")) == 32:
    #Use x86 libraries.
    from Lib.asdk import DM
else:
    #Use x86_64 libraries.
    from Lib64.asdk import DM
#Go back to communicating with the main program through stdout
sys.stdout = original_stdout

class ALPAODM(WavefrontCorrector):

    def __init__(self, conf) -> None:
        #Initialize the pyRTC super class
        super().__init__(conf)

        #Initialize connection to ALPAO DM
        self.serial = conf["serial"]
        self.dm = DM(self.serial)
        self.CAP = conf["commandCap"]
        #Ask for the number of actuators
        self.numActuators = int(self.dm.Get('NBOfActuator'))

        #Generate the ALPAO actuator layout for the number of actuators
        layout = self.generateLayout()
        self.setLayout(layout)

        #Read the flat from the specified flat file
        if "flatFile" in conf.keys():
            if '.txt' in conf["flatFile"]:
                flat = np.genfromtxt(conf["flatFile"])
            elif '.npy' in conf["flatFile"]:
                flat = np.load(conf["flatFile"])
            self.setFlat(flat.astype(self.flat.dtype))
        
        #flatten the mirror
        self.flatten()

        return
    
    def generateLayout(self):

        if self.numActuators == 97:
            xx, yy = np.meshgrid(np.arange(11), np.arange(11))
            layout = np.sqrt((xx - 5)**2 + (yy-5)**2) < 5.5
        return layout
    
    def sendToHardware(self):
        #Do all of the normal updating of the super class
        super().sendToHardware()
        #Cap the Commands to reduce likelihood of DM failiure
        self.currentShape[self.currentShape > self.CAP] = self.CAP
        self.currentShape[self.currentShape < -self.CAP] = -self.CAP
        #Send the correction to the actual mirror
        self.dm.Send(self.currentShape)
        return

    def __del__(self):
        super().__del__()
        self.dm.Reset()
        return
    

if __name__ == "__main__":

    #Prevents camera output from messing with communication
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')

    # Create argument parser
    parser = argparse.ArgumentParser(description="Read a config file from the command line.")

    # Add command-line argument for the config file
    parser.add_argument("-c", "--config", required=True, help="Path to the config file")

    # Parse command-line arguments
    args = parser.parse_args()

    conf = read_yaml_file(args.config)

    pid = os.getpid()
    os.sched_setaffinity(pid, {conf["wfc"]["affinity"],})
    decrease_nice(pid)

    confWFC = conf["wfc"]
    wfc = ALPAODM(conf=confWFC)
    wfc.start()
    
    #Go back to communicating with the main program through stdout
    sys.stdout = original_stdout

    l = Listener(wfc)
    while l.running:
        l.listen()
        time.sleep(1e-3)