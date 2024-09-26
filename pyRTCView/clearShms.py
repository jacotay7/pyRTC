from pyRTC.Pipeline import *
from pyRTC.utils import *
from pyRTC.hardware import *
shm_names = ["wfs", "wfsRaw", "wfc", "wfc2D", "signal", "signal2D", "psfShort", "psfLong"] #list of SHMs to reset
clear_shms(shm_names)
