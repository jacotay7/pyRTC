# %% Imports
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
#%%
import sys
tmp = sys.stdout
from pyRTC import *
from pyRTC.hardware.OOPAOInterface import OOPAOInterface
from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis
RECALIBRATE = False
sys.stdout = tmp
import logging
import matplotlib

logging.getLogger('matplotlib').setLevel(logging.WARNING)
#%% Read Config
conf = utils.read_yaml_file("pywfs_OOPAO_config.yaml")

#%%
"""
Create the OOPAO simulation interface object 
Running this cell will initialize the dm, wfs, psf, and slopes objects, 
but will not start their real time computations. This inialization includes
the creation of the Shared Memory Objects, and the simulation inialization.
"""
sim = OOPAOInterface(conf=conf, param=None)
wfs, dm, psf = sim.get_hardware()

"""
Start the processes. Here the real-time computations selected in
the config will begin.
"""
dm.start()
dm.flatten()
wfs.start()

#Comment out if not made yet
psf.loadModelPSF("./calib/modelPSF.npy")
psf.start()
#Remove the atmosphere from the simulation
sim.removeAtmosphere()

psf.takeModelPSF() #Take a new model for the strehl calculation

psf.saveModelPSF("./calib/modelPSF.npy")
"""
It's important to set the full basis and number of possible modes before
initializing the loop object. Here I define a KL basis for the system
"""
NUM_MODES = conf["wfc"]["numModes"] #must be less than total KL modes

M2C_KL = compute_KL_basis(sim.tel, sim.atm, sim.dm)
dm.setM2C(M2C_KL[:,:NUM_MODES])

#%% Create Slope computation 
slopes = SlopesProcess(conf=conf["slopes"])
slopes.start()

# %% Initialize our AO loop object

#Adjust the config for predictive control test
confLoop = conf["loop"]
confLoop["T"] = 3
confLoop["K"] = 10
confLoop["hidden_size"] = 64
confLoop["num_layers"] = 1
confLoop["learning_rate"] = 1e-3
confLoop["num_epochs"] = 100
confLoop["batch_size"] = 32
confLoop["validSubApsFile"] = "./calib/validSubAps.npy"
confLoop["functions"] = ["predictiveIntegrator"]

from pyRTC.hardware.basicPredictLoop import basicPredictLoop
loop = basicPredictLoop(conf=confLoop)
loop.IMFile = "./calib/simIM.npy"
loop.loadIM()

#%%
if RECALIBRATE:

    #Remove the atmosphere from the simulation
    sim.removeAtmosphere()

    psf.takeModelPSF() #Take a new model for the strehl calculation

    psf.saveModelPSF("./calib/modelPSF.npy")

    loop.pokeAmp = 1e-7

    #Compute the IM, blocking
    loop.computeIM()

    loop.saveIM("./calib/simIM.npy")
    loop.loadIM()

#Add the atmosphere back to the simulation
sim.addAtmosphere()

#%%Adjust frame delay
dm.setDelay(confLoop["T"])
for i in range(5):
    dm.flatten()
loop.setGain(0.3)
time.sleep(1)
loop.start()
# %%
loop.listen(int(2**14))
# loop.slopesBuffer = np.load("./calib/slopesBuffer.npy")
loop.stop()
# %%
# loop.num_epochs = 1
# loop.train()
# loop.loadModels()

loop.num_epochs = 100
loop.train()
# %%
loop.start()
loop.setGain(0.2)
loop.predict=False
time.sleep(5)
loop.gamma = 0.6
loop.predict=True
#%%
def recordStrehl(strehlShm, N=10):
    val = 0
    for j in range(N):
        val += np.mean(strehlShm.read())
    return val/N

def resetLoop():
    loop.stop()
    loop.flatten()
    dm.flatten()
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
    loop.setGain(g)
    resetLoop()
    strehls[i] = recordStrehl(strehlShm, N=N)

plt.plot(gains, strehls)
plt.show()
#%% Find best gamma
loop.gamma = 0
loop.predict=True
loop.setGain(0.2)
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
loop.gamma = 0
loop.predict=True
loop.setGain(0.0)
loop.start()
N = 100
gammas = np.linspace(0,1,10)
gains = np.linspace(0.1,1,10)
strehls = np.zeros((gammas.size,gains.size))
for i in tqdm(range(gammas.size), desc="Optimal Gamma"):
    gamma = gammas[i]
    for j in range(gains.size):
        gain = gains[j]
        loop.gamma = 0
        loop.setGain(gain)
        resetLoop()
        time.sleep(1)
        loop.gamma = gamma
        loop.setGain(gain)
        time.sleep(1)
        strehls[i,j] = recordStrehl(strehlShm, N=N)
np.save("./calib/performance.npy",strehls )

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
loop.stop()
loop.flatten()
dm.flatten()
# %%
