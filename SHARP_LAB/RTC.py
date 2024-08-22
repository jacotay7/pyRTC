# %% IMPORTS
#Import pyRTC classes
from pyRTC.Pipeline import *
from pyRTC.utils import *
import os
os.chdir("/home/whetstone/RLAO/pyRTC/SHARP_LAB")
RECALIBRATE = False

# %% Clear SHMs
# shm_names = ["wfs", "wfsRaw", "wfc", "wfc2D", "signal", "signal2D", "psfShort", "psfLong"] #list of SHMs to reset
# clear_shms(shm_names)

# %% IMPORTS
config = '/home/whetstone/RLAO/pyRTC/SHARP_LAB/config.yaml'
N = np.random.randint(3000,6000)

# %% Launch DM
wfc = hardwareLauncher("../pyRTC/hardware/ALPAODM.py", config, N)
wfc.launch()

# %% Launch WFS
wfs = hardwareLauncher("../pyRTC/hardware/ximeaWFS.py", config, N+1)
wfs.launch()

# %% Launch slopes
slopes = hardwareLauncher("../pyRTC/SlopesProcess.py", config, N+2)
slopes.launch()

# %% Launch PSF Cam
psfCam = hardwareLauncher("../pyRTC/hardware/SpinnakerScienceCam.py", config, N+10)
psfCam.launch()
# %% Launch Loop Class
# loop = hardwareLauncher("../pyRTC/hardware/RLLoop.py", config, N+4, timeout = 6*60)
# loop = hardwareLauncher("../pyRTC/Loop.py", config, N+4, timeout = 6*60)
# loop.launch()
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.preprocessing import  is_image_space
from sb3_contrib import RecurrentPPO

gym.register(
    id='pyRTCEnvPID-v0',
    entry_point='pyRTC.hardware.AOEnv:pyRTCEnvPID'
)

env = gym.make("pyRTCEnvPID-v0")
model = RecurrentPPO("MlpLstmPolicy", env=env)

env.reset()
for i in range(100):
    obs, rw, a,b,c = env.step(env.action_space.sample())
print(obs)

# features_extractor_kwargs = dict(
#     normalized_image = True
# )
# policy_kwargs = dict(
#     normalize_images = False
#     # features_extractor_kwargs = features_extractor_kwargs,
# )

# env = gym.make("pyRTCEnvGain-v0")
# is_image_space(env.observation_space, check_channels=False, normalized_image=True)
# model = RecurrentPPO("CnnLstmPolicy",env, policy_kwargs=policy_kwargs)

# %% NCPA OPTIMIZER
optim  = hardwareLauncher("../pyRTC/hardware/NCPAOptimizer.py", config, N+4)
optim.launch()
# %% Calibrate

if RECALIBRATE:

    slopes.setProperty("refSlopesFile", "")
    slopes.run("loadRefSlopes")
    ##### slopes.setProperty("offsetY", 3)

    input("Sources Off?")
    wfs.run("takeDark")
    wfs.setProperty("darkFile", "/home/whetstone/RLAO/pyRTC/SHARP_LAB/dark.npy")
    wfs.run("saveDark")
    time.sleep(1)
    psfCam.run("takeDark")
    psfCam.setProperty("darkFile", "/home/whetstone/RLAO/pyRTC/SHARP_LAB/psfDark.npy")
    psfCam.run("saveDark")
    input("Sources On?")
    input("Is Atmosphere Out?")

    slopes.run("computeImageNoise")
    slopes.run("takeRefSlopes")
    slopes.setProperty("refSlopesFile", "/home/whetstone/RLAO/pyRTC/SHARP_LAB/ref.npy")
    slopes.run("saveRefSlopes")

    wfc.run("flatten")
    psfCam.run("takeModelPSF")
    psfCam.setProperty("modelFile", "/home/whetstone/RLAO/pyRTC/SHARP_LAB/modelPSF.npy")
    psfCam.run("saveModelPSF")

    loop.setProperty("IMFile","/home/whetstone/RLAO/pyRTC/SHARP_LAB/IM_zern.npy")
    wfc.run("flatten")
    loop.run("computeIM")
    loop.run("saveIM")

# %%
# loop.setProperty("IMFile", "/home/whetstone/pyRTC/SHARP_LAB/IM_zern.npy")
# loop.run("loadIM")
# loop.setProperty("numDroppedModes", 60)
# loop.run("computeCM")
# loop.run("setGain", 0.1)
# loop.setProperty("learningTimesteps", int(2**18))
# print(loop.getProperty("modelName"))
wfc.run("flatten")
time.sleep(1)
loop.run("start")
# mins_per_mode = 30
# time.sleep(60*mins_per_mode)
# for i in range(1, 60):
#     loop.setProperty("numDroppedModes", 93-i)
#     loop.run("computeCM")
#     loop.run("setGain", 0.3)
#     time.sleep(60*mins_per_mode)

# %%
loop.run("reset")
loop.setProperty("train", False)
time.sleep(0.5)
loop.run("start")
time.sleep(3)
loop.run("stop")
# %%
for i in range(100):
    loop.run("learn")


#%% NCPAs

#%% Optimize NCPA
optim.setProperty("endMode", 30)
for i in range(10):
    optim.run("optimize")
    slopes.run("takeRefSlopes")
    slopes.setProperty("refSlopesFile", "/home/whetstone/RLAO/pyRTC/SHARP_LAB/ref.npy")
    slopes.run("saveRefSlopes")
optim.run("applyOptimum")
slopes.run("takeRefSlopes")
slopes.setProperty("refSlopesFile", "/home/whetstone/RLAO/pyRTC/SHARP_LAB/ref.npy")
slopes.run("saveRefSlopes")
# %% CODE TO UPDATE FLAKY SUB APERTURES

# subApFile = slopes.getProperty("validSubApsFile")
# print(subApFile)
# vsubAp = np.load(subApFile)
# plt.imshow(vsubAp)
# plt.show()

# from pyRTC.Pipeline import initExistingShm
# shm, _, _ = initExistingShm("signal2D")

# x = shm.read_noblock()
# N = 1000
# data = np.zeros((N, *x.shape))
# for i in range(N):
#     data[i] = shm.read_noblock()
#     time.sleep(3e-3)

# arr = np.std(data,axis =0)
# aps_to_remove = arr > np.percentile(arr, 99)
# aps_to_remove[:aps_to_remove.shape[0]//2] |= aps_to_remove[aps_to_remove.shape[0]//2:]
# aps_to_remove[aps_to_remove.shape[0]//2:] |= aps_to_remove[:aps_to_remove.shape[0]//2]
# # aps_to_remove |= vsubAp
# plt.imshow(aps_to_remove)
# plt.show()
# new_valid = np.copy(vsubAp)
# new_valid[aps_to_remove] = False
### np.save(subApFile, new_valid)
# %%
