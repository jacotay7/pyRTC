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

#%
psfCam.setProperty("integrationLength", 50)
wfc.run("flatten")
loopOptim = loopOptimizer(read_yaml_file(config)["optimizer"]["loop"], loop)
#%%Adjust frame delay

# loop.predict = True
# loop.numDroppedModes = 10
# loop.gain = 0.55
# loop.gamma = 0.6
# loop.leakyGain =  0.02
# loop.loadIM()
# time.sleep(1)
# loop.start()

DELAY = 0
wfc.run("setDelay",DELAY)
loop.setProperty("T", 2 + DELAY) #Predict 1 ahead of the delay in the system for the ML inference
loop.setProperty("gamma", 0.0)
loop.setProperty("predict", True)
loop.setProperty("numDroppedModes", 5)
loop.setProperty("gain", 0.8)
loop.setProperty("leakyGain", 0.01)
loop.run("loadIM")
time.sleep(1)
loop.run("start")
# %%
# input("Put ATM to training Location")
# loop.listen(int(2**12))
# loop.stop()
loop.run("listen", int(2**14))
loop.run("stop") 
loop.run("flatten") 
# %%
# loop.num_epochs = 10
# loop.train()
# loop.saveModels()
# loop.run("loadModels")
loop.setProperty("num_epochs", 10)
for i in range(1):
    loop.run("train")
    loop.run("saveModels")
# loop.run("loadModels")
# loop.setProperty("predict", True)
#%%
input("Put ATM to verification Location")
psfCam.setProperty("integrationLength", 2000)
time.sleep(2)
loop.setProperty("predict", True)
loopOptim.adjustParam("gain", 0.3, 1.2)
loopOptim.adjustParam("gamma", 0.4, 0.9)
loopOptim.adjustParam("numDroppedModes", 0, 20, int)
loopOptim.adjustParam("leakyGain", 0, 0.05)
time.sleep(1)
loopOptim.resetStudy()
time.sleep(5)
loopOptim.numReads = 2
loopOptim.numSteps = 30
for i in range(1):
    loopOptim.optimize()
loopOptim.applyNext()
#%%
def recordStrehl(strehlShm, N=10):
    val = 0
    diverge = False
    for j in range(N):
        if diverge:
            return -1
        val += np.mean(strehlShm.read())
        diverge = np.max(wfc2DShm.read_noblock()) > 0.6
    return val/N

def resetLoop(hardReal = False):
    if hardReal:
        for i in range(10):
            loop.run("stop")
            loop.run("flatten")
            wfc.run("flatten")
            time.sleep(0.1)
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
wfc2DShm = initExistingShm("wfc2D")[0]
# %% Find best gamma/gain soft RT loop
psfCam.setProperty("integrationLength", 500)
time.sleep(1)
loop.gamma=  0
loop.predict - True
loop.setGain(0.3)
loop.start() 
N = 10
gammas = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
gains = np.array([0.4])
strehls = np.zeros((gammas.size,gains.size))
strehlShm.read()
for i in tqdm(range(gammas.size), desc="Network Fraction"):
    gamma = gammas[i]
    for j in range(gains.size):
        gain = gains[j]
        loop.gamma =  0
        loop.setGain(gain)
        resetLoop(hardReal=False)
        time.sleep(1)
        loop.gamma = gamma
        time.sleep(1)
        strehls[i,j] = recordStrehl(strehlShm, N=N)
        print(f"Gamma: {gamma} Gain {gain} SR {strehls[i,j]}")
resetLoop(hardReal=False)
loop.run("stop") 
np.save("./calib/performance.npy",strehls )

# %% Find best gamma/gain
psfCam.setProperty("integrationLength", 1000)
time.sleep(1)
loop.setProperty("gamma", 0)
loop.setProperty("predict",True)
loop.run("setGain", 0.3)
loop.run("start") 
N = 10
gammas = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
gains = np.array([0.4])
strehls = np.zeros((gammas.size,gains.size))
strehlShm.read()
for i in tqdm(range(gammas.size), desc="Network Fraction"):
    gamma = gammas[i]
    for j in range(gains.size):
        gain = gains[j]
        loop.setProperty("gamma", 0)
        loop.run("setGain", gain)
        resetLoop(hardReal=True)
        time.sleep(1)
        loop.setProperty("gamma", gamma)
        time.sleep(1)
        strehls[i,j] = recordStrehl(strehlShm, N=N)
        print(f"Gamma: {gamma} Gain {gain} SR {strehls[i,j]}")
resetLoop(hardReal=True)
loop.run("stop") 
np.save("./calib/performance.npy",strehls )
#%%
plt.figure(figsize=(12,5))
for i, gamma in enumerate(gammas):
    if i ==0:
        continue
    plt.plot(gains, strehls[i,:] - strehls[0,:], 
             alpha = 0.7,
             label= f"Network %: {gamma:.2f}")
plt.legend()
plt.ylabel("Strehl Change [%, absolute]", size = 16)
plt.xlabel("Loop Gain", size = 16)
plt.ylim(0,0.1)
plt.title(f"Empirical Predictive Controller, Delay = {loop.getProperty("T")} Frames")
plt.show()
#%%
plt.imshow(strehls)#, #extent=[np.min(gains),np.max(gains),
                    #        np.min(gammas),np.max(gammas)])
plt.colorbar(label='Strehl')
plt.ylabel("Network %", size = 12)
plt.xlabel("Loop Gain", size = 12)
plt.show()
#%%
plt.plot(gammas, strehls[:,0])
plt.xlabel("Network %")
plt.ylabel("Strehl Ratio")
plt.show()
# %%
psfShm, _, _ = initExistingShm("psfShort")
N = 10000
plateTag = 'D'
speedTag = '20'
trainTag = "NOATM"
onOffTag = 'OFF'
if loop.getProperty("predict"):
    onOffTag = 'ON'
delayTag = wfc.getProperty('frameDelay')
netUtePerc = str(int(100*loop.getProperty("gamma")))
dataName = f"plate_{plateTag}_speed_{speedTag}_{trainTag}_predict_{onOffTag}_delay_{delayTag}_network_{netUtePerc}"
print(dataName)
psfs = np.empty((N, *psfShm.read_noblock().shape))
for i in range(N):
    psfs[i] = psfShm.read()
np.save(f"/home/whetstone/predictControlData/{dataName}", psfs)
# %%
strehlShm = initExistingShm("strehl")[0]
wfc2DShm = initExistingShm("wfc2D")[0]
def recordStrehl(N=10):
    vals = np.zeros(N)
    diverge = False
    for j in range(N):
        if diverge:
            return -1
        vals[j] = np.mean(strehlShm.read())
        diverge = np.max(wfc2DShm.read_noblock()) > 0.6
    return np.mean(vals), np.std(vals)
volts = []
SRs = []
SRs_err = []
#%%
volts.append(8)
a, b = recordStrehl(N = 10)
SRs.append(a)
SRs_err.append(b)

# %%
T = wfc.getProperty("frameDelay")
np.save(f"/home/whetstone/servoLagTest/predict_volts_delay_{T}", np.array(volts))
np.save(f"/home/whetstone/servoLagTest/predict_sr_delay_{T}", np.array(SRs))
np.save(f"/home/whetstone/servoLagTest/predict_srErr_delay_{T}", np.array(SRs_err))

#%%
loop.setProperty("T", 2 + DELAY) 
loop.run("loadModels")
loop.setProperty("num_epochs", 5)
for i in range(1):
    loop.run("train")
    loop.run("saveModels")
# %%
