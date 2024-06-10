# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from pyRTC import *
from pyRTC.hardware import *
from pyRTC.utils import *
from pyRTC.Pipeline import *
#%% CLEAR SHMs
shms = ["wfc", "wfc2D"]
clear_shms(shms)
# %% Load Config
conf = read_yaml_file("./config.yaml")
# %%
wfc = SUPERPAOWER(conf["wfc"])
wfc.start()
# %%
wfc.push(0,2)
# %%
wfc.flatten()
# %%
# for i in range(16): 
#     wfc.push(i,1)
#     time.sleep(1)
# %%
for volt in np.linspace(0,4):
    wfc.push(0,volt)
    time.sleep(1)
# %%
