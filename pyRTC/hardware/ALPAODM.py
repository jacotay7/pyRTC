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

from pyRTC.Pipeline import launchComponent
from pyRTC.WavefrontCorrector import WavefrontCorrector


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

        if conf["floatingActuatorsFile"][-4:] == '.npy':
            floatActuatorInds = np.load(conf["floatingActuatorsFile"])
            self.deactivateActuators(floatActuatorInds)

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
        self.currentShape = np.clip(self.currentShape, -self.CAP, self.CAP)
        #Send the correction to the actual mirror
        self.dm.Send(self.currentShape)
        return

    def __del__(self):
        super().__del__()
        self.dm.Reset()
        return
    

if __name__ == "__main__":

    launchComponent(ALPAODM, "wfc", start = True)