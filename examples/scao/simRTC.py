# %% Imports
import matplotlib.pyplot as plt
import time
#%%
import sys
tmp = sys.stdout
from pyRTC import *
from pyRTC.hardware.OOPAOInterface import OOPAOInterface
from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis
RECALIBRATE = False
sys.stdout = tmp
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
confLoop["K"] = 20
confLoop["hidden_size"] = 512
confLoop["num_layers"] = 2
confLoop["learning_rate"] = 1e-3
confLoop["num_epochs"] = 100
confLoop["batch_size"] = 8
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
dm.flatten()
loop.setGain(0.3)
time.sleep(1)
loop.start()
# %%
loop.listen(int(2**13))
loop.stop()
# %%
# loop.num_epochs = 1
# loop.train()
loop.loadModels()

# loop.num_epochs = 5
# loop.train()
# %%
loop.start()
loop.setGain(0.3)
loop.predict=False
time.sleep(5)
loop.gamma = 0.0
loop.predict=True
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
