import os 
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ['NUMBA_NUM_THREADS'] = '1'

from pyRTC.WavefrontCorrector import *
from pyRTC.Pipeline import *
from pyRTC.utils import *
import struct
import argparse
import sys

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

        self.floatingActuatorFile = setFromConfig(conf, "floatingActuatorFile", "")

        if len(self.floatingActuatorFile) > 4 and self.floatingActuatorFile[-4:] == '.npy':
            floatActuatorInds = np.load(conf["floatingActuatorFile"])
            self.deactivateActuators(floatActuatorInds)

        #Read the flat from the specified flat file
        if "flatFile" in conf.keys():
            filename = conf["flatFile"]
            extension = os.path.splitext(filename)[1]
            print(f"Loading Flat: {filename}")
            if extension == ".txt":
                flat = np.genfromtxt(filename)
            elif  extension == ".npy":
                flat = np.load(filename)
            elif extension == ".fits" or extension == ".fit":
                from astropy.io import fits
                tmp = fits.open(filename)[0].data
                if tmp.size == self.numActuators:
                    flat = tmp.reshape((self.numActuators,))
                else:
                    print(f"Error with fits file. Expected Size {self.numActuators}, Received {tmp.size}")
                
            self.setFlat(flat.astype(self.flat.dtype))
        
        #flatten the mirror
        self.flatten()

        return

    def generateLayout(self):

        if self.numActuators == 97:
            xx, yy = np.meshgrid(np.arange(11), np.arange(11))
            layout = np.sqrt((xx - 5)**2 + (yy-5)**2) < 5.5
        elif self.numActuators == 277:
            #Note this just defines the shape, the numbers don't matter
            layout = np.array([[0,0,0,0,0,0,271,272,273,274,275,276,277,0,0,0,0,0,0],
                                [0,0,0,0,0,262,263,264,265,266,267,268,269,270,0,0,0,0,0],
                                [0,0,0,0,251,252,253,254,255,256,257,258,259,260,261,0,0,0,0],
                                [0,0,0,238,239,240,241,242,243,244,245,246,247,248,249,250,0,0,0],
                                [0,0,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,0,0],
                                [0,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,0],
                                [187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205],
                                [168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186],
                                [149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167],
                                [130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148],
                                [111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129],
                                [92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110],
                                [73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91],
                                [0,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,0],
                                [0,0,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,0,0],
                                [0,0,0,28,29,30,31,32,33,34,35,36,37,38,39,40,0,0,0],
                                [0,0,0,0,17,18,19,20,21,22,23,24,25,26,27,0,0,0,0],
                                [0,0,0,0,0,8,9,10,11,12,13,14,15,16,0,0,0,0,0],
                                [0,0,0,0,0,0,1,2,3,4,5,6,7,0,0,0,0,0,0]]) > 0
        else:
            print("ALPAO DM Layout Unknown, Please provide")
            layout = None
        return layout
    
    def sendToHardware(self):
        #Do all of the normal updating of the super class
        super().sendToHardware()
        #Cap the Commands to reduce likelihood of DM failiure
        self.currentShape = np.clip(self.currentShape, -self.CAP, self.CAP)
        #Send the correction to the actual mirror
        self.dm.Send(self.currentShape)
        return

    def __del__(self):
        super().__del__()
        self.dm.Reset()
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
    set_affinity((conf["wfc"]["affinity"])%os.cpu_count()) 
    decrease_nice(pid)

    confWFC = conf["wfc"]
    wfc = ALPAODM(conf=confWFC)
    wfc.start()

    l = Listener(wfc, port = int(args.port))
    while l.running:
        l.listen()
        time.sleep(1e-3)