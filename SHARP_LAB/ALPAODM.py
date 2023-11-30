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

    def __init__(self, dofs, serialName, flatFile=None, M2C=None) -> None:
        super().__init__(dofs, layout=None, M2C=M2C)

        self.serialName = serialName
        self.dm = DM(self.serialName)
        self.numActuators = int(self.dm.Get('NBOfActuator'))

        self.setLayout()

        if not (flatFile is None):
            flat = np.genfromtxt("/etc/chai/dm_flat.txt")
            flat.dtype = self.flat.dtype
            self.setFlat(flat)
        
        self.flatten()

        return
    
    def setLayout(self):

        if self.dofs == 97:
            xx, yy = np.meshgrid(np.arange(11), np.arange(11))
            layout = np.sqrt((xx - 5)**2 + (yy-5)**2) < 5.5

        super().setLayout(layout)
    
    def applyCorrectionRaw(self, correction):
        super().applyCorrectionRaw(correction)
        self.dm.Send(correction)
        return

    def __del__(self):
        self.dm.Reset()
        return