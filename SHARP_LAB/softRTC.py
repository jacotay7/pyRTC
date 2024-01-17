# %% Imports
import numpy as np
import matplotlib.pyplot as plt
# from pyRTC.hardware.ximeaWFS import *
from pyRTC.WavefrontSensor import *
from pyRTC.hardware.ALPAODM import *
from pyRTC.utils import *
from pyRTC.hardware.SpinnakerScienceCam import *
# %% Load Config
conf = read_yaml_file("/home/whetstone/pyRTC/SHARP_LAB/config.yaml")
# %% Launch WFS
confWFS = conf["wfs"]
wfs = WavefrontSensor(conf=confWFS)
time.sleep(4)
wfs.start()
# %% Launch WFC
confWFC = conf["wfc"]
wfc = ALPAODM(conf=confWFC)
time.sleep(0.5)
wfc.start()
# %% Launch PSF
confPSF = conf["psf"]
psf = spinCam(conf=confPSF)
psf.start()
time.sleep(2)
#  %% Visualize IM
IM = np.load("/home/whetstone/pyRTC/SHARP_LAB/IM.npy")
plt.imshow(IM, aspect="auto")
plt.show()

# for row in np.linspace(0,10,5).astype(int):
#     x = signal2D(IM[:,row], wfs.layout)
#     plt.imshow(x)
#     plt.show()

# %% Flatten
FLAT = np.load(confWFC['flatFile'])
wfc.setFlat(FLAT)
wfc.flatten()

# %% Tip-Tilt-Focus-Sweep
from tqdm import tqdm

numModes = 10
startMode = 3
endMode = wfc.numModes - 1

N = 100
RANGE = 0.5
psfs = np.empty((numModes, N, *psf.imageShape))
wfc.flatten()
time.sleep(0.1)

modelist = np.linspace(startMode, endMode, numModes).astype(int)

for i, mode in enumerate(modelist): #range(numModes):
    correction = np.zeros_like(wfc.read())
    for j, amp in enumerate(tqdm(np.linspace(-RANGE,RANGE,N))):
        correction[mode] = amp
        wfc.write(correction)
        #Burn some images
        psf.readLong()
        psf.readLong()
        #Save the next PSF in the dataset
        psfs[i, j, :, :] = psf.readLong()
wfc.flatten()

# %%
filename = f"psfs_{startMode}_{endMode}_{numModes}_{N}"
np.save(filename, psfs)

# %%


shmName = 'wfc2D'
metadataSHM = ImageSHM(shmName+"_meta", (4,), np.float64)
N = 1000
times = np.empty(N)
counts = np.empty(N)
for i in range(N):
    metadata = metadataSHM.read()
    counts[i] = metadata[0]
    times[i] = metadata[1]
    time.sleep(1e-3)

# %%
dt = times[1:] - times[:-1]
dc = counts[1:] - counts[:-1]
speeds = 1000*(dt[dc > 0]/dc[dc > 0])
plt.hist(1/speeds, bins = 'sturges')
plt.show()
# %%
# plt.plot(times)
# plt.show()
# %% Generate Valid SubAps for SHWFS
wfsMeta = ImageSHM("signal_meta", (4,), np.float64).read_noblock_safe()
signalDType = float_to_dtype(wfsMeta[3])
signalSize = int(wfsMeta[2]//signalDType.itemsize)
wfsShm = ImageSHM("signal", (signalSize,), signalDType)
N = int(np.round(np.sqrt(signalSize/2),0))
test_slopes = wfsShm.read_noblock_safe().reshape(2*N,N)
plt.imshow(test_slopes)
plt.show()
x = test_slopes[:test_slopes.shape[0]//2]
xx, yy = np.meshgrid(np.arange(x.shape[0]), np.arange(x.shape[1]))
mid = (x.shape[0]-1)/2, (x.shape[1]-1)/2
zz = np.sqrt((yy-mid[0])**2 + (xx-mid[1])**2) < mid[0]+1
validSubAps = np.vstack([zz,zz]).astype(test_slopes.dtype)

plt.imshow(validSubAps)
plt.show()
np.save("validSubAps", validSubAps)
# %%

# %%
