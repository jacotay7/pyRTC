# %% IMPORTS
#Import pyRTC classes
from pyRTC.Pipeline import *
from pyRTC.utils import *
import os
os.chdir("/home/whetstone/SUPERPAOWER/pyRTC/SUPERPAOWER")

# %% Clear SHMs
# from pyRTC.Pipeline import clear_shms
# shm_names = ["signal"]
# shm_names = ["wfs", "wfsRaw", "wfc", "wfc2D", "signal", "signal2D", "psfShort", "psfLong"] #list of SHMs to reset
# clear_shms(shm_names)

# %% IMPORTS
# config = '/home/whetstone/SUPERPAOWER/pyRTC/SUPERPAOWER/config_SR.yaml'
config = '/home/whetstone/SUPERPAOWER/pyRTC/SUPERPAOWER/config.yaml'
N = np.random.randint(3000,6000)
# %% Launch DM
wfc = hardwareLauncher("../pyRTC/hardware/SUPERPAOWER.py", config, N)
wfc.launch()

# %% Launch PSF Cam
# power = hardwareLauncher("../pyRTC/hardware/thorLabsPowerMeter.py", config, N+10)
# power.launch()
# power = hardwareLauncher("../pyRTC/hardware/serialPhotodetector.py", config, N+10)
# power.launch()
power = hardwareLauncher("../pyRTC/hardware/QHYCCDSciCam.py", config, N+10)
power.launch()
# %% Launch Loop Class
loop = hardwareLauncher("../pyRTC/hardware/PSGDLoop.py", config, N+4)
loop.launch()

# %%
#%%Set-up Loop
loop.setProperty("rate", 3)
loop.setProperty("amp", 0.2)
loop.setProperty("useLong", True)
power.setProperty("integrationLength", 7)
loop.setProperty("gradientDamp", 3e-2)
#%%Start Loop
loop.run("start")

#%%Start Loop
loop.run("stop")
loop.run("flatten")
wfc.run("flatten")
wfc.run("flatten")
# %%

optim = psgdOptimizer(conf["optimizer"], loop, power)
optim.numSteps = 50

optim.gdTime = 3
optim.avgTime = 5
optim.relaxTime = 0
optim.numFlats = 1
optim.maxAmp = 1
optim.minAmp = 0.2
optim.maxRate = 1e-1
optim.minRate = 1e-2
optim.minIntegrate = 1
optim.maxIntegrate = 20

#%%

# FREQ = 400 # Hz
# delay = 1/(2*FREQ)

# # act = 2
# wfcShm, _, _ = initExistingShm("wfc")

# correction = np.zeros_like(wfcShm.read_noblock())

# BIN = True
# HEIGHT = 3
# LOW = 1
# HIGH = LOW + HEIGHT

# correction[:] = LOW
# while True:
#     if BIN:
#         correction[:] = HIGH
#     else:
#         correction[:] = LOW
#     BIN = not BIN
#     wfcShm.write(correction)
#     time.sleep(2e-3)
wfc.run("startClock", 1, 0.2)
# %%
wfc.run("stopClock")
# %%
