# Import specific functions or classes from submodules
__all__ = []

try:
    from .ALPAODM import *
    __all__.append('ALPAODM')
except:
    print("ALPAO python SDK installation not found")
try:
    from .SpinnakerScienceCam import *
    __all__.append('spinCam')
except:
    print("Spinnaker python SDK installation not found")
try:
    from .ximeaWFS import *
    __all__.append('XIMEA_WFS')
except:
    print("ximea python SDK installation not found")
try:
    from .PIModulator import *
    __all__.append('PIModulator')
except:
    print("PI python SDK installation not found")
try:
    from .SUPERPAOWER import *
    __all__.append('SUPERPAOWER')
except:
    print("SUPERPAOWER modules not found. Likely pyserial needed.")
try:
    from .thorLabsPowerMeter import *
    __all__.append('powerMeter')
except:
    print("Power meter modules not found. Likely pyvisa-py & pysub needed.")
try:
    from .serialPhotodetector import *
    __all__.append('photoDetector')
except:
    print("Serial modules not found. Likely pyserial is needed.")
try:
    from .QHYCCDSciCam import *
    __all__.append('QHYCCD')
except:
    print("All QHY modules not found. Likely needs qhyccd-python: https://github.com/JiangXL/qhyccd-python/tree/master")


from .NCPAOptimizer import *
from .PIDOptimizer import *
from .PSGDLoop import *
from .psgdOptim import *

__all__.extend([
           'NCPAOptimizer', 
           'PIDOptimizer',
           'PSGDLoop',
           'psgdOptimizer'
           ])

#Remove Any duplicates
__all__ = list(set(__all__))
