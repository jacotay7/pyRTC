# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from pyRTC.hardware.ximeaWFS import *
# from pyRTC.WavefrontSensor import *
from pyRTC.hardware.ALPAODM import *
from pyRTC.utils import *
from pyRTC.hardware.SpinnakerScienceCam import *
from pyRTC.SlopesProcess import *
from pyRTC.Loop import *
#%% CLEAR SHMs
# shms = ["wfs", "wfsRaw", "signal", "signal2D", "wfc", "wfc2D", "psfShort", "psfLong"]
# clear_shms(shms)
# %% Load Config
conf = read_yaml_file("/home/whetstone/pyRTC/SHARP_LAB/config.yaml")
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
loop = Loop(conf=conf)
time.sleep(1)

# %% Recalibrate

# Darks

input("Sources Off?")
wfs.takeDark()
wfs.darkFile = "/home/whetstone/pyRTC/SHARP_LAB/dark.npy"
wfs.saveDark()
time.sleep(1)
psf.takeDark()
psf.darkFile = "/home/whetstone/pyRTC/SHARP_LAB/psfDark.npy"
psf.saveDark()
input("Sources On?")
input("Is Atmosphere Out?")

slopes.computeImageNoise()
slopes.refSlopesFile =  ""
slopes.loadRefSlopes()
slopes.takeRefSlopes()
slopes.refSlopesFile = "/home/whetstone/pyRTC/SHARP_LAB/ref.npy"
slopes.saveRefSlopes()

wfc.flatten()
psf.takeModelPSF()
psf.modelFile = "/home/whetstone/pyRTC/SHARP_LAB/modelPSF.npy"
psf.saveModelPSF()

#  STANDARD IM
loop.IMMethod = "push-pull"
loop.pokeAmp = 0.03
loop.numItersIM = 100
loop.IMFile = "/home/whetstone/pyRTC/SHARP_LAB/IM.npy"
wfc.flatten()
loop.computeIM()
loop.saveIM()
wfc.flatten()
time.sleep(1)

input("Is Atmosphere In?")
#  DOCRIME OL
loop.IMMethod = "docrime"
loop.delay = 3
loop.pokeAmp = 2e-2
loop.numItersIM = 10000
loop.IMFile = "/home/whetstone/pyRTC/SHARP_LAB/docrime_IM.npy"
wfc.flatten()
loop.computeIM()
loop.saveIM()
wfc.flatten()
time.sleep(1)



# %% Compute CM
loop.IMFile = "/home/whetstone/pyRTC/SHARP_LAB/IM.npy"
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

#%% Float Actuators
wfc.floatingInfluenceRadius = 1
wfc.reactivateActuators([i for i in range(97)])
wfc.deactivateActuators([0,1,2,3,4,5,11,12,20,21,31,32,42,43,53,54,64,65,75,76,84,85,91,92,93,94,95,96])

# %% Find Best Strehl Method
psf.takeModelPSF()
metric = []
gs = np.linspace(0,2, 10)
for g in gs:
    g = 1
    strehls = []
    for i in range(100):
        strehls.append(psf.computeStrehl(gaussian_sigma=g))
    metric.append(np.std(np.array(strehls)))
plt.plot(gs, metric)
plt.show()

# %% Tip-Tilt-Focus-Sweep
from tqdm import tqdm

numModes = 10
startMode = 0
endMode = wfc.numModes - 1

N = 101
RANGE = 0.05
psfs = np.empty((numModes, N, *psf.imageShape))
cmd = wfc.read()
cmds = np.empty((numModes, N, *cmd.shape), dtype=cmd.dtype)
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
        #Save the next PSF in the dataset
        psfs[i, j, :, :] = psf.readLong()
        cmds[i,j,:] = correction
wfc.flatten()

# %%
folder = "/media/whetstone/bc08ec4b-bb23-407e-92d4-5585e3dfe2d2/data/"
run_name = "mode_sweep_apr_9_final"
folder += run_name
filename = f"{folder}/psfs_{startMode}_{endMode}_{numModes}_{N}"
np.save(filename, psfs)
filename = f"{folder}/cmds_{startMode}_{endMode}_{numModes}_{N}"
np.save(filename, cmds)
#Save WFC info
np.save(f"{folder}/M2C", wfc.M2C)
np.save(f"{folder}/DM_LAYOUT", wfc.layout)

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
# %% Generate Valid SubAps for SHWFS
#First make an IM with all valid subAps
IM = np.load("../SHARP_LAB/IM.npy")
IM = np.moveaxis(IM, 0 ,1)
pdiam = int(np.sqrt(IM.shape[1]/2))
IM = IM.reshape(IM.shape[0], 2*pdiam, pdiam)
mean_IM = np.std(IM,axis = 0)#np.mean(np.abs(IM), axis = 0)
min_threshold = 6
max_threshold = 20
valid_sub_aps = (mean_IM > min_threshold) & (mean_IM < max_threshold)
combineXY = valid_sub_aps[:valid_sub_aps.shape[0]//2,:] & valid_sub_aps[valid_sub_aps.shape[0]//2:,:]
valid_sub_aps[:valid_sub_aps.shape[0]//2,:] = combineXY
valid_sub_aps[valid_sub_aps.shape[0]//2:,:] = combineXY
plt.imshow(mean_IM)
plt.colorbar()
plt.show()
plt.imshow(valid_sub_aps)
plt.show()

np.save("/home/whetstone/pyRTC/SHARP_LAB/validSubAps.npy", valid_sub_aps.astype(bool))


# %%
img = wfs.read()
spacing = 7.7
i,j = 5,4
offsetX, offsetY = -2, 4
plt.imshow(img[int(spacing*i)+ offsetY: int(spacing*(i+1))+ offsetY, 
               int(spacing*j)+ offsetX: int(spacing*(j+1))+ offsetX])

plt.show()
# %%
from scipy.signal import find_peaks
arr = np.sum(img,axis = 1)
peaks = find_peaks(arr)
plt.plot(arr)
spots = []
for p in peaks[0]:
    if arr[p] > 1000:
        plt.axvline(x=p, color = "r")
        spots.append(p)
plt.show()
spots = np.array(spots)
print(np.mean(spots[1:]- spots[:-1]))
print(peaks)
# %% Plot Flat
x = np.zeros(wfc.layout.shape)
x[wfc.layout] = wfc.flat
plt.imshow(x)
plt.colorbar()
plt.show()

#%% Plot slope err vs contrast

contrasts = np.linspace(0,100,20)
vars = []
for c in contrasts:
    slopes.shwfsContrast = c
    slopes.takeRefSlopes()
    arr = []
    for i in range(1000):
        arr.append(slopes.read())
    arr = np.array(arr)
    vars.append(np.std(arr))
plt.plot(contrasts, vars, 'x')
plt.xlabel("SHWFS Threshold Parameter")
plt.ylabel("Slope Deviation")
plt.show()

# %% Reset SHWFS
slopes.setRefSlopes(np.zeros_like(slopes.refSlopes))
slopes.shwfsContrast = 20
slopes.offsetX = 0
slopes.offsetY = 0

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

# %% Find SHWFS Spacing
cur_wfs = wfs.read()
plt.imshow(cur_wfs)
plt.show()

spacing = slopes.subApSpacing - 0.02
plt.plot(np.mean(cur_wfs, axis = 0))
x = slopes.offsetX
while x  < cur_wfs.shape[1]:
    plt.axvline(x=x, color = 'r', linestyle = '--')
    x += spacing
plt.show()

plt.plot(np.mean(cur_wfs, axis = 1))
y = slopes.offsetY
while y  < cur_wfs.shape[0]:
    plt.axvline(x=y, color = 'r', linestyle = '--')
    y += spacing
plt.show()

# %%
def compute_fwhm(image):
    # Filter to keep only negative values
    negative_pixels = image[image < 1]
    
    # Compute the histogram of negative values
    # Adjust bins and range as necessary for your specific image
    hist, bins = np.histogram(negative_pixels, bins=np.arange(np.min(negative_pixels), 1)+0.5)
    print(bins)
    # Since the distribution is symmetric, we can mirror the histogram to get the full distribution
    hist_full = np.concatenate((hist[::-1], hist))
    
    # Compute the bin centers from the bin edges
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_centers_full = np.concatenate((-bin_centers[::-1], bin_centers))
    
    print(bin_centers_full)
    plt.plot(bin_centers_full, hist_full, 'x')
    plt.show()

    # Find the maximum value (mode of the distribution)
    peak_value = np.max(hist_full)
    half_max = peak_value / 2
    
    # Find the points where the histogram crosses the half maximum
    cross_points = np.where(np.diff((hist_full > half_max).astype(int)))[0]
    
    # Assuming the distribution is sufficiently smooth and has a single peak,
    # the FWHM is the distance between the first and last crossing points
    fwhm_value = np.abs(bin_centers_full[cross_points[-1]] - bin_centers_full[cross_points[0]])
    
    return fwhm_value, bin_centers_full, hist_full, half_max

print(compute_fwhm(wfs.read())[0])

# %% Refine Valid Sup Aps
slopes2D = np.zeros_like(slopes.computeSignal2D(slopes.read()))
for i in range(1000):
    slopes2D += slopes.computeSignal2D(slopes.read())
slopes2D /= 1000
plt.imshow(slopes2D)
plt.show()
mask = np.abs(slopes2D > 1.5)

combineXY = mask[:mask.shape[0]//2,:] | mask[mask.shape[0]//2:,:]
mask[:mask.shape[0]//2,:] = combineXY
mask[mask.shape[0]//2:,:] = combineXY
plt.imshow(mask)
plt.show()

valid_sub_aps = slopes.validSubAps.copy()
valid_sub_aps[mask] = 0
plt.imshow(valid_sub_aps)
plt.show()
# np.save("/home/whetstone/pyRTC/SHARP_LAB/validSubAps.npy", valid_sub_aps.astype(bool))



# %%
