# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from pyRTC.hardware.ImageStreamIOWfs import *
from pyRTC.hardware.ImageStreamIOWfc import *
from pyRTC.utils import *
from pyRTC.SlopesProcess import *
from pyRTC.Loop import *
#%% CLEAR SHMs
# shms = ["wfs", "wfsRaw", "signal", "signal2D", "wfc", "wfc2D", "psfShort", "psfLong"]
# clear_shms(shms)
# %% Load Config
conf = read_yaml_file("/home/jtaylor/pyRTC/MAPS/config.yaml")
# %% Launch WFS
confWFS = conf["wfs"]
wfs = ISIOWfs(conf=confWFS)
time.sleep(0.5)
wfs.start()
# %% Launch slopes
slopes = SlopesProcess(conf=conf)
slopes.start()
time.sleep(0.5)
# %% Launch WFC
confWFC = conf["wfc"]
wfc = ISIOWfc(conf=confWFC)
time.sleep(0.5)
wfc.start()

# %% Launch loop
loop = Loop(conf=conf)
time.sleep(1)
# %% Recalibrate

# Darks
if False:
    input("Sources Off?")
    wfs.takeDark()
    wfs.darkFile = "/home/jtaylor/pyRTC/MAPS/calib/dark.npy"
    wfs.saveDark()

    input("ON SKY?")
    wfc.flatten()
    #  DOCRIME OL
    loop.IMMethod = "docrime"
    loop.delay = 2
    loop.pokeAmp = 2e-2
    loop.numItersIM = 10000
    loop.IMFile = "/home/jtaylor/pyRTC/MAPS/calib/docrime_IM.npy"
    wfc.flatten()
    loop.computeIM()
    loop.saveIM()
    wfc.flatten()
    time.sleep(1)


# %% Compute CM
loop.IMFile = "/home/jtaylor/pyRTC/MAPS/calib/docrime_IM.npy"
loop.loadIM()
loop.numDroppedModes = 40
loop.computeCM()
loop.setGain(0.01)
loop.leakyGain = 1e-2

# %% Start Loop
wfc.flatten()
time.sleep(0.3)
loop.start()

# %% Stop Loop
loop.stop()
wfc.flatten()
time.sleep(0.3)
wfc.flatten()

# %% Time A SHM
shmName = 'wfc2D'
metadataSHM = ImageSHM(shmName+"_meta", (ImageSHM.METADATA_SIZE,), np.float64)
N = 1000
times = np.empty(N)
counts = np.empty(N)
for i in range(N):
    metadata = metadataSHM.read()
    counts[i] = metadata[0]
    times[i] = metadata[1]
    time.sleep(1e-3)

#Plot the Timing Variance
dt = times[1:] - times[:-1]
dc = counts[1:] - counts[:-1]
speeds = 1000*(dt[dc > 0]/dc[dc > 0])
plt.hist(1/speeds, bins = 'sturges')
plt.show()

#%% Inspect IM
IM = np.load("/home/jtaylor/pyRTC/MAPS/calib/docrime_IM.npy")
vsa = np.load("/home/jtaylor/pyRTC/MAPS/calib/validSubAps.npy")

x_slopes = vsa[:,:vsa.shape[1]//2]
y_slopes = vsa[:,vsa.shape[1]//2:]


mode = np.zeros(vsa.shape)
modeNum = 2
mode[:,:vsa.shape[1]//2][x_slopes] = IM[:IM.shape[0]//2,modeNum]
mode[:,vsa.shape[1]//2:][y_slopes] = IM[IM.shape[0]//2:,modeNum]

plt.imshow(mode, aspect="auto")
plt.show()
# %%
