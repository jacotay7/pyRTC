# %% IMPORTS
import numpy as np
import matplotlib.pyplot as plt

#pyRTC
from pyRTC.hardware.ALPAODM import *
from pyRTC.hardware.fliCBlueOneWFS import *
from pyRTC.SlopesProcess import *
from pyRTC.Loop import *

# %% Load Config
conf = read_yaml_file("/home/revoltuser/pyRTC/REVOLT/config.yaml")

# %% Clear SHM
# shm_names = ["wfs", "wfsRaw", "wfc", "wfc2D", "signal", "signal2D"] #list of SHMs to reset
# clear_shms(shm_names)

# %% Launch WFC
confWFC = conf["wfc"]
BASIS = "MODAL"
if BASIS == "ZONAL":
    confWFC["numModes"] = confWFC["numActuators"]
    m2c = np.eye(confWFC["numActuators"])
    np.save(confWFC["m2cFile"], m2c)
elif BASIS == "MODAL":
    m2c = np.load("revolt_kl_basis.npy")[:, :confWFC["numModes"]]
    np.save(confWFC["m2cFile"], m2c)
wfc = ALPAODM(conf=confWFC)
time.sleep(0.5)
wfc.start()


# %% Launch WFS
confWFS = conf["wfs"]
wfs = FliCBlueOneWFS(conf=confWFS)
wfs.start()

# %% Launch Slopes
slopes = SlopesProcess(conf=conf)
slopes.start()
# %% Monitor WFS
slopes.shwfsContrast = 7 # set to 0 if not dark subtracting, slowly dial up to ideal
print(np.mean(wfs.read()))
slopes.offsetX = 0
slopes.offsetY = 0

for i in range(1):
    im = wfs.read()
    plt.imshow(im > np.mean(im) * slopes.shwfsContrast)
    plt.colorbar()
    plt.show()
    plt.imshow(slopes.read())
    plt.colorbar()
    plt.show()
    time.sleep(0.5)



# %% Calibration
#Take Dark
# input("Is Lamp OFF???")
# wfs.takeDark()
# wfs.darkFile = "/home/revoltuser/pyRTC/REVOLT/dark.npy"
# wfs.saveDark()

#Take Reference Slopes
# input("Is Lamp ON???")
# wfc.flatten()
# slopes.refSlopesFile = "/home/revoltuser/pyRTC/REVOLT/refSlopes.npy"
# slopes.takeRefSlopes()
# slopes.saveRefSlopes()

# %% Loop
loop = Loop(conf)


# %% Compute IM
if BASIS == "MODAL":
    loop.pokeAmp = 0.03
else:
    loop.pokeAmp = 0.02
loop.numItersIM = 100
loop.IMFile =  "./IM.npy"
# %%
wfc.flatten()
loop.computeIM()
loop.saveIM()

IM = np.load("IM.npy").reshape(*slopes.read().shape, -1)
IM = np.moveaxis(IM, 2,0)
for i in range(5):
    plt.imshow(IM[i])
    plt.show()
# %% Compute DOCRIME IM
loop.pokeAmp = 5e-3
loop.numItersIM = 10000
loop.IMMethod = "docrime"
loop.IMFile =  "./docrimeIM.npy"
loop.delay = 2 #3 when running 1kHZ


wfc.flatten()
loop.computeIM()
time.sleep(1e-1)
wfc.flatten()
loop.saveIM()
IM = np.load("docrimeIM.npy").reshape(*slopes.read().shape, -1)
IM = np.moveaxis(IM, 2,0)
for i in range(5):
    plt.imshow(IM[i+5])
    plt.show()
# %%
metadata_shm = ImageSHM("wfs_meta", (ImageSHM.METADATA_SIZE,), np.float64)
metadata = metadata_shm.read_noblock()
old_count = metadata[0]
old_time = metadata[1]

for i in range(10):
    time.sleep(0.5)
    metadata = metadata_shm.read_noblock()
    new_count = metadata[0]
    new_time = metadata[1]
    if new_time > old_time:
        speed_fps = np.round((new_count - old_count)/(new_time- old_time),2)
        speed_fps = str(speed_fps) + "FPS"
        print(speed_fps)

# %% Compute Valid Subap mask
IM = np.load("IM.npy").reshape(*slopes.read().shape, -1)
IM = np.moveaxis(IM, 2,0)

arr = np.sum(np.abs(IM), axis = 0)
plt.imshow(arr)
plt.colorbar()
plt.show()

for i in range(10):
    plt.imshow(IM[i])
    plt.show()


# mask = (arr > np.percentile(arr, 20)).astype(np.float32)
# plt.imshow(mask)
# plt.colorbar()
# plt.show()
# print(mask.dtype)
# np.save("validSubAps", mask)
# %%