# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from pyRTC import *
from pyRTC.hardware import *
from pyRTC.utils import *
from pyRTC.Pipeline import *
#%% CLEAR SHMs
# shms = ["wfc", "wfc2D"]
# clear_shms(shms)
# %% Load Config
conf = read_yaml_file("./config.yaml")
# %%
wfc = SUPERPAOWER(conf["wfc"])
wfc.start()

# %%
# np.save( "./calib/flat.npy", wfc.flat +2)
# %%
wfc.push(0,1)
# %%
wfc.flatten()
# %%
