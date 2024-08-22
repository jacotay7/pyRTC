# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from pyRTC.hardware.ximeaWFS import *
from pyRTC.hardware.ALPAODM import *
from pyRTC.utils import *
from pyRTC.hardware.SpinnakerScienceCam import *
from pyRTC.SlopesProcess import *
from pyRTC.hardware.RLLoop import *
from pyRTC.hardware.NCPAOptimizer import *

# Load Config
conf = read_yaml_file("/home/whetstone/RLAO/pyRTC/SHARP_LAB/config.yaml")
RECALIBRATE = False

import os
os.chdir("/home/whetstone/RLAO/pyRTC/SHARP_LAB/")


#%% CLEAR SHMs
# shms = ["wfs", "wfsRaw", "signal", "signal2D", "wfc", "wfc2D", "psfShort", "psfLong"]
# clear_shms(shms)


# %% Launch WFS
confWFS = conf["wfs"]
wfs = XIMEA_WFS(conf=confWFS)
time.sleep(0.5)
wfs.start()
# %% Launch slopes
slopes = SlopesProcess(conf=conf)
slopes.start()
time.sleep(0.5)
# %% Launch WFC
confWFC = conf["wfc"]
wfc = ALPAODM(conf=confWFC)
time.sleep(0.5)
wfc.start()
# %% Launch PSF
confPSF = conf["psf"]
psf = spinCam(conf=confPSF)
time.sleep(0.5)
psf.start()
# %% Launch loop
loop = RLLoop(conf=conf)
# loop = Loop(conf=conf)
time.sleep(1)

# %% Recalibrate
if RECALIBRATE:
    # Darks
    input("Sources Off?")
    wfs.takeDark()
    wfs.darkFile = "/home/whetstone/RLAO/pyRTC/SHARP_LAB/dark.npy"
    wfs.saveDark()
    time.sleep(1)
    psf.takeDark()
    psf.darkFile = "/home/whetstone/RLAO/pyRTC/SHARP_LAB/psfDark.npy"
    psf.saveDark()
    input("Sources On?")
    input("Is Atmosphere Out?")

    slopes.computeImageNoise()
    slopes.refSlopesFile =  ""
    slopes.loadRefSlopes()
    slopes.takeRefSlopes()
    slopes.refSlopesFile = "/home/whetstone/RLAO/pyRTC/SHARP_LAB/ref.npy"
    slopes.saveRefSlopes()

    wfc.flatten()
    psf.takeModelPSF()
    psf.modelFile = "/home/whetstone/RLAO/pyRTC/SHARP_LAB/modelPSF.npy"
    psf.saveModelPSF()

    #  STANDARD IM
    loop.IMMethod = "push-pull"
    loop.pokeAmp = 0.03
    loop.numItersIM = 100
    loop.IMFile = "/home/whetstone/RLAO/pyRTC/SHARP_LAB/IM.npy"
    wfc.flatten()
    loop.computeIM()
    loop.saveIM()
    wfc.flatten()
    time.sleep(1)

# # %% Initial flat system
# wfc.flatten()
# time.sleep(2)
# psf.takeModelPSF()
# psf.modelFile = "/home/whetstone/RLAO/pyRTC/SHARP_LAB/modelPSF.npy"
# psf.saveModelPSF()
# slopes.takeRefSlopes()
# time.sleep(1)

# # %% NCPAOptimizer
# ncpa = NCAPOptimizer(conf["optimizer"])
# ncpa.numSteps = 1000
# ncpa.optimize(numSteps=1000)
# # wfc.saveShape("./new_best_flat")
# # %% Compute new CM
# loop.IMFile = "/home/whetstone/RLAO/pyRTC/SHARP_LAB/IM.npy"
# loop.loadIM()
# loop.numDroppedModes = 30
# loop.computeCM()

# # %% Check reward
# loop.getReward()
# %% Test env
for i in range(100):
    loop.sendRandomAction()
    time.sleep(0.1)
# loop.env.reset()

# %% Learn
loop.learningTimesteps = 2048
loop.learn()
# %% test model in loop
for i in range(1000):
    action, _ = loop.model.predict(loop.env._get_obs(), deterministic=True)
    loop.wfcShm.write(action)
loop.env.reset()
# %% Test Reward
loop.reset()
x = loop.env.action_space.sample()
loop.env.step(x)
time.sleep(0.1)
plt.plot(x)
for i in range(1000):
    step = 0.1*loop.CM@loop.signalShm.read()
    x -= step
    loop.env.step(step)
    loop.env.comp_reward(verbose=True)
    plt.plot(x)
plt.show()

# %%
start = time.time()
N = 1000
for i in range(N):
    wfc.push(0,0)
    loop.env._get_obs()
time_per_obs = (time.time() - start)/N * 1000
print(f"Time per obs {time_per_obs:.2f}ms")
# %%
# %% Reset SHWFS
slopes.setRefSlopes(np.zeros_like(slopes.refSlopes))
slopes.shwfsContrast = 20
slopes.offsetX = 8
slopes.offsetY = 4

# %% Find SHWFS Offsets
# slopes.subApSpacing = 15.54
vals = []
for offsetX in range(0,int(slopes.subApSpacing)):
    for offsetY in range(0,int(slopes.subApSpacing)):
        slopes.offsetX = offsetX
        slopes.offsetY = offsetY
        arr = []
        for i in range(20):
            arr.append(slopes.read())
        arr = np.array(arr)
        arr = np.mean(arr, axis = 0)
        arr = arr.flatten()
        vals.append((offsetX, offsetY, np.mean(np.abs(arr))))
vals = np.array(vals)
print(vals[vals[:,2] == np.nanmin(vals[:,2])])
# %%
import rl_zoo3
gym.make("pyRTCEnvPID-v0")
# %%
