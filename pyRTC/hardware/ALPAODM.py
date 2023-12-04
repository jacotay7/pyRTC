from pyRTC.WavefrontCorrector import *
import struct

''' Add '/Lib' or '/Lib64' to path '''
if (8 * struct.calcsize("P")) == 32:
    #Use x86 libraries.
    from Lib.asdk import DM
else:
    #Use x86_64 libraries.
    from Lib64.asdk import DM

class ALPAODM(WavefrontCorrector):

    def __init__(self, serialName, flatFile=None, M2C=None) -> None:

        #Initialize connection to ALPAO DM
        self.serialName = serialName
        self.dm = DM(self.serialName)

        #Ask for the number of actuators
        self.numActuators = int(self.dm.Get('NBOfActuator'))

        #Generate the ALPAO actuator layout for the number of actuators
        layout = self.generateLayout()

        #Initialize the pyRTC super class
        super().__init__(self.numActuators, layout=layout, M2C=M2C)

        #Read the flat from the specified flat file
        if isinstance(flatFile, str):
            flat = np.genfromtxt(flatFile).astype(self.flat.dtype)
            self.setFlat(flat)
        
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
        #Send the correction to the actual mirror
        self.dm.Send(self.currentShape)
        return
    

    def __del__(self):
        self.dm.Reset()
        return