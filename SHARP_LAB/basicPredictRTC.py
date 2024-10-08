# %% Imports
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
#%%
import sys
tmp = sys.stdout
from pyRTC import *
RECALIBRATE = False
sys.stdout = tmp
import logging
import matplotlib
logging.getLogger('matplotlib').setLevel(logging.WARNING)
#%%
config = '/home/whetstone/pyRTC/SHARP_LAB/config_basic_predict_pywfs.yaml'
N = np.random.randint(3000,6000)
folder = "/home/whetstone/pyRTC/SHARP_LAB/calib/"
conf = utils.read_yaml_file(config)

# %% Launch WFS
wfs = hardwareLauncher("../pyRTC/hardware/ximeaWFS.py", config, N+1)
wfs.launch()

#%% Create Slope computation 
slopes = hardwareLauncher("../pyRTC/SlopesProcess.py", config, N+2)
slopes.launch()

# %% Launch DM
wfc = hardwareLauncher("../pyRTC/hardware/ALPAODM.py", config, N)
wfc.launch()

# %% Initialize our AO loop object
loop = hardwareLauncher("../pyRTC/hardware/basicPredictLoop.py", config, N+4)
loop.launch()
# conf = utils.read_yaml_file(config)
# confLoop = conf["loop"]
# from pyRTC.hardware.basicPredictLoop import basicPredictLoop
# loop = basicPredictLoop(conf=confLoop)

# %% Launch PSF Cam
psfCam = hardwareLauncher("../pyRTC/hardware/SpinnakerScienceCam.py", config, N+10)
psfCam.launch()

# %% Launch Modulator
modulator = hardwareLauncher("../pyRTC/hardware/PIModulator.py", config, N+100)
modulator.launch()

#%%Adjust frame delay
psfCam.setProperty("integrationLength", 50)
# wfc.run("setDelay",loop.T)#loop.getProperty("T"))
wfc.run("flatten")  
# loop.predict = False
# loop.numDroppedModes = 9
# loop.gain = 0.3
# loop.leakyGain =  0.01
# loop.loadIM()
# time.sleep(1)
# loop.start()

wfc.run("setDelay",loop.getProperty("T"))
loop.setProperty("predict", False)
loop.setProperty("numDroppedModes", 9)
loop.setProperty("gain",0.3)
loop.setProperty("leakyGain", 0.01)
loop.run("loadIM")
time.sleep(1)
loop.run("start")
# %%
loop.listen(int(2**14))
loop.stop()
# loop.run("listen", int(2**10))
# # loop.slopesBuffer = np.load("./calib/slopesBuffer.npy")
# loop.run("stop") 
# %%
loop.num_epochs = 100
loop.train()

# loop.run("loadModels")
# loop.setProperty("predict", True)
# %%
loop.run("start") 
loop.run("setGain", 0.4)
loop.predict=False
time.sleep(5)
loop.gamma = 0.6
# loop.predict=True
#%%
def recordStrehl(strehlShm, N=10):
    val = 0
    for j in range(N):
        val += np.mean(strehlShm.read())
    return val/N

def resetLoop(hardReal = False):
    if hardReal:
        loop.run("stop")
        loop.run("flatten")
        time.sleep(0.5)
        loop.run("start") 
        time.sleep(0.5)
    else:

        loop.stop() 
        loop.flatten()
        wfc.run("flatten")
        time.sleep(0.5)
        loop.start() 
        time.sleep(0.5)
    return
strehlShm = initExistingShm("strehl")[0]

#%% Find best gain
# loop.gamma = 0
gains = np.linspace(0.1,1,10)
strehls = np.zeros_like(gains)
N = 100

for i in tqdm(range(gains.size), desc="Optimal Gain"):
    g = gains[i]
    loop.run("setGain", g)
    resetLoop()
    strehls[i] = recordStrehl(strehlShm, N=N)

plt.plot(gains, strehls)
plt.show()
#%% Find best gamma
loop.gamma = 0
loop.predict=True
loop.run("setGain", 0.2)
gammas = np.linspace(0,1,10)
strehls = np.zeros_like(gammas)
N = 100

for i in tqdm(range(gammas.size), desc="Optimal Gamma"):
    g = gammas[i]
    loop.gamma = g
    resetLoop()
    strehls[i] = recordStrehl(strehlShm, N=N)
    
plt.plot(gammas, strehls)
plt.show()
# %% Find best gamma/gain
loop.setProperty("gamma", 0)
loop.setProperty("predict",True)
loop.run("setGain", 0.0)
loop.run("start") 
N = 300
gammas = np.linspace(0,1,10)
gains = np.linspace(0.1,1,10)
strehls = np.zeros((gammas.size,gains.size))
for i in tqdm(range(gammas.size), desc="Optimal Gamma"):
    gamma = gammas[i]
    for j in range(gains.size):
        gain = gains[j]
        loop.setProperty("gamma", gamma)
        loop.run("setGain", gain)
        resetLoop(hardReal=True)
        strehls[i,j] = recordStrehl(strehlShm, N=N)
np.save("./calib/performance.npy",strehls )
plt.imshow(strehls)
plt.show()
# %% Find best gamma/gain
loop.gamma = 0
loop.predict=True
loop.setGain(0.0)
loop.start() 
N = 300
gammas = np.linspace(0,1,10)
gains = np.linspace(0.1,1,10)
strehls = np.zeros((gammas.size,gains.size))
for i in tqdm(range(gammas.size), desc="Optimal Gamma"):
    gamma = gammas[i]
    for j in range(gains.size):
        gain = gains[j]
        loop.gamma = gamma
        loop.setGain(gain)
        resetLoop()
        strehls[i,j] = recordStrehl(strehlShm, N=N)
np.save("./calib/performance.npy",strehls )
plt.imshow(strehls)
plt.show()
#%%
pred = loop.runInference(loop.history)
x = np.arange(loop.history.shape[0]+2)
for i in range(loop.history.shape[1]):
    if i > 0:
        break
    
    plt.plot(x[:-2],loop.history[:,i])
    plt.plot(x[-1], pred[i], 'x')
plt.show()
plt.plot(pred)
plt.plot(loop.history[-1])
plt.show()
#%%
loop.run("stop") 
loop.flatten()
wfc.run("flatten")
# %%
