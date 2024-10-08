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
# try:
#     from .OOPAOInterface import OOPAOInterface
#     __all__.append('OOPAOInterface')
# except:
#     print("OOPAO installation not found")

from .NCPAOptimizer import *
from .PIDOptimizer import *
from .loopHyperparamsOptimizer import *

__all__.extend([
            'NCPAOptimizer', 
            'PIDOptimizer',
            'loopOptimizer'
           ])