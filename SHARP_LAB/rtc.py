# %% Imports
import numpy as np
import yaml
import matplotlib.pyplot as plt
#Import pyRTC classes
from pyRTC.Loop import *
#Import hardware classes
from pyRTC.hardware.ALPAODM import *
from pyRTC.hardware.ximeaWFS import *

# %% clean up
# from pyRTC.Pipeline import clear_all_shms
# clear_all_shms()
# %% Read Config
def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        conf = yaml.safe_load(file)
    return conf
conf = read_yaml_file("config.yaml")
confWFS = conf["wfs"]
confWFC = conf["wfc"]
# %% Initialize DM
M2C = np.fromfile(confWFC["m2cFile"], dtype=np.float64).reshape(confWFC["numActuators"],confWFC["numModes"])
#Normalize M2C
for i in range(M2C.shape[1]):
    M2C[:,i] /= np.std(M2C[:,i])
dm = ALPAODM(confWFC["serial"], flatFile = confWFC["flatFile"], M2C=M2C)
dm.start()
# %% Initialize WFS
#Create camera object, take an image
wfs = XIMEA_WFS(exposure=confWFS["exposure"], 
                roi=[confWFS["width"],confWFS["height"],confWFS["left"],confWFS["top"]], 
                binning=confWFS["binning"], 
                gain=confWFS["gain"], 
                bitDepth=confWFS["bitDepth"])
pupilLocs = [(int(x.split(',')[1]), int(x.split(',')[0])) for x in confWFS["pupils"]]
wfs.setPupils(pupilLocs, confWFS["pupilsRadius"])
wfs.start()
import time
time.sleep(1)
# wfs.plot()
# %% Compute IM}
loop = Loop(wfs, dm)
# %% Comput IM
loop.computeIM(0.005,N=10,hardwareDelay=1e-2)
# for i in np.random.choice(np.arange(loop.numModes),5).astype(int):
#     loop.plotIM(row=i)
# plt.imshow(loop.IM, aspect='auto')
# %% Compute CM
loop.computeCM(numDropped=20)
plt.imshow(loop.CM, aspect='auto', vmin = np.percentile(loop.CM, 5), 
           vmax = np.percentile(loop.CM, 95))
# %% Run Loop 
dm.flatten()
time.sleep(1e-2)
loop.setGain(1e-3)
loop.start()
time.sleep(10)
loop.stop()
# %% Save Shape and Flatten 
newFlat = dm.currentShape
np.save("newFlat", newFlat)
time.sleep(1e-2)
dm.flatten()
# # %% Exit
print("Stopping WFS")
wfs.stop()
del wfs
del dm

# %%
