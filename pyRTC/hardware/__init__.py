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

from .NCPAOptimizer import *
from .PIDOptimizer import *

__all__.extend([
           'NCPAOptimizer', 
           'PIDOptimizer'
           ])

#Remove Any duplicates
__all__ = list(set(__all__))