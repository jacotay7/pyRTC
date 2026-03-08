# Import specific functions or classes from submodules
__all__ = []

try:
    from .ALPAODM import ALPAODM as ALPAODM
    __all__.append('ALPAODM')
except Exception:
    print("ALPAO python SDK installation not found")
try:
    from .SpinnakerScienceCam import spinCam as spinCam
    __all__.append('spinCam')
except Exception:
    print("Spinnaker python SDK installation not found")
try:
    from .ximeaWFS import XIMEA_WFS as XIMEA_WFS
    __all__.append('XIMEA_WFS')
except Exception:
    print("ximea python SDK installation not found")
try:
    from .PIModulator import PIModulator as PIModulator
    __all__.append('PIModulator')
except Exception:
    print("PI python SDK installation not found")
# try:
#     from .OOPAOInterface import OOPAOInterface
#     __all__.append('OOPAOInterface')
# except:
#     print("OOPAO installation not found")

from .NCPAOptimizer import NCPAOptimizer as NCPAOptimizer
from .PIDOptimizer import PIDOptimizer as PIDOptimizer
from .SyntheticSystems import SyntheticSHWFS as SyntheticSHWFS
from .SyntheticSystems import SyntheticScienceCamera as SyntheticScienceCamera
from .loopHyperparamsOptimizer import loopOptimizer as loopOptimizer

__all__.extend([
            'NCPAOptimizer', 
            'PIDOptimizer',
            'SyntheticSHWFS',
            'SyntheticScienceCamera',
            'loopOptimizer'
           ])