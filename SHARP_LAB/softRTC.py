# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import time
from pyRTC import *
from pyRTC.hardware import *
from pyRTC.utils import *
from pyRTC.Pipeline import *
# from pyRTC.hardware.ximeaWFS import *
# # from pyRTC.WavefrontSensor import *
# from pyRTC.hardware.ALPAODM import *

# from pyRTC.hardware.SpinnakerScienceCam import *
# from pyRTC.SlopesProcess import *
# from pyRTC.Loop import *
#%% CLEAR SHMs
# from pyRTC.Pipeline import clear_shms
# shms = ["wfs", "wfsRaw", "signal", "signal2D", "wfc", "wfc2D", "psfShort", "psfLong"]
# clear_shms(shms)
# %% Load Config
# cfile = "/home/whetstone/pyRTC/SHARP_LAB/config_SR.yaml"
cfile = "/home/whetstone/pyRTC/SHARP_LAB/config.yaml"
conf = read_yaml_file(cfile)

# %% Launch Modulator (PyWFS)
# confMod = conf["modulator"]
# mod = PIModulator(conf=confMod)
# time.sleep(0.5)
# mod.start()
# %% Launch WFS
confWFS = conf["wfs"]
wfs = XIMEA_WFS(conf=confWFS)
time.sleep(0.5)
wfs.start()
# %% Launch slopes
confSlopes = conf["slopes"]
slopes = SlopesProcess(conf=confSlopes)
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
confLoop = conf["loop"]
loop = Loop(conf=confLoop)
time.sleep(1)

# %% Recalibrate

# Darks
if False:
    input("Sources Off?")
    wfs.takeDark()
    wfs.darkFile = "/home/whetstone/pyRTC/SHARP_LAB/calib/dark.npy"
    wfs.saveDark()
    time.sleep(1)
    psf.takeDark()
    psf.darkFile = "/home/whetstone/pyRTC/SHARP_LAB/calib/psfDark_SR.npy"
    psf.saveDark()
    input("Sources On?")
    input("Is Atmosphere Out?")

    slopes.computeImageNoise()
    slopes.refSlopesFile =  ""
    slopes.loadRefSlopes()
    slopes.takeRefSlopes()
    slopes.refSlopesFile = "/home/whetstone/pyRTC/SHARP_LAB/calib/ref.npy"
    slopes.saveRefSlopes()

    wfc.flatten()
    psf.takeModelPSF()
    psf.modelFile = "/home/whetstone/pyRTC/SHARP_LAB/calib/modelPSF.npy"
    psf.saveModelPSF()

    #  STANDARD IM
    loop.IMMethod = "push-pull"
    loop.pokeAmp = 0.03
    loop.numItersIM = 100
    loop.IMFile = "/home/whetstone/pyRTC/SHARP_LAB/calib/IM.npy"
    wfc.flatten()
    loop.computeIM()
    loop.saveIM()
    wfc.flatten()
    time.sleep(1)

    # input("Is Atmosphere In?")
    # #  DOCRIME OL
    # loop.IMMethod = "docrime"
    # loop.delay = 3
    # loop.pokeAmp = 2e-2
    # loop.numItersIM = 10000
    # loop.IMFile = "/home/whetstone/pyRTC/SHARP_LAB/calib/docrime_IM.npy"
    # wfc.flatten()
    # loop.computeIM()
    # loop.saveIM()
    # wfc.flatten()
    # time.sleep(1)


# %% Compute CM
loop.IMFile = "/home/whetstone/pyRTC/SHARP_LAB/calib/IM.npy"
loop.numDroppedModes = 15
loop.gain = 0.1
loop.leakyGain = 0.02
loop.loadIM()

# %% Start Loop %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
wfc.flatten()
time.sleep(0.3)
loop.start()

# %% Stop Loop
loop.stop()
wfc.flatten()
time.sleep(0.3)
wfc.flatten()

#%% NCPA
ncpaOptim = NCPAOptimizer(read_yaml_file('/home/whetstone/pyRTC/SHARP_LAB/config.yaml'
)["optimizer"]["ncpa"], loop, slopes)
psf.integrationLength = 1
ncpaOptim.numReads = 10
ncpaOptim.startMode = 0
ncpaOptim.endMode = 60
ncpaOptim.numSteps = 1500
ncpaOptim.isCL = False
for i in range(1):
    ncpaOptim.optimize()
ncpaOptim.applyOptimum()
wfc.saveShape()
slopes.takeRefSlopes()
slopes.refSlopesFile = "/home/whetstone/pyRTC/SHARP_LAB/calib/ref.npy"
slopes.saveRefSlopes()
psf.integrationLength = 10000
psf.takeModelPSF()
psf.modelFile = "/home/whetstone/pyRTC/SHARP_LAB/calib/modelPSF.npy"
psf.saveModelPSF()


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

# %% Bench Conversion
# %% Compute CM
# loop.IMFile = "/home/whetstone/pyRTC/SHARP_LAB/cIM.npy"
loop.loadIM()
loop.numDroppedModes = 20
loop.computeCM()
# %%
SIM = np.load("/home/whetstone/pyRTC/SHARP_LAB/calib/sprint_IM_nomisreg_valid.npy").reshape(94, -1).T
bench_converter_SL = (np.linalg.pinv(SIM) @ loop.IM) #[:,:21]
bench_converter_LS = (loop.CM @ SIM) #[:,:21]
bench_converter_NP = np.eye(94) #[:,:21]
# %% Tip-Tilt-Focus-Sweep
from tqdm import tqdm
import glob

folder_out = "/media/whetstone/storage2/data/robin-aug"
folder_in = "/home/whetstone/Downloads/torch/torch/phases"
# filelist = ['cnnx2_phase', 'cnnx4_phase','linx2_phase', 'linx4_phase', 'cnnx8n3_phase','linx8n3_phase']
# filelist = ['_zern_cnn_x2_n4_phase', '_zern_cnn_x4_n4_phase', '_zern_cnn_x8_n4_phase', '_fixzern_lin_x2_n4_phase', '_fixzern_lin_x4_n4_phase', '_fixzern_lin_x8_n4_phase']
# N = 3
filelist = glob.glob(f"{folder_in}/*.npy")
# numModes = 11
# powerlist = [0, 0.25, 0.5, 1., 2., 4., 8., 16] #np.linspace(-RANGE, RANGE, numModes) #.astype(int)
powerlist = np.linspace(-3, 3, 13)
numPokePowers = len(powerlist)
slopecorrect = 0.0021
psf.integrationLength = 1000
psf.setGamma(4)

# for k, bench_converter in enumerate([bench_converter_NP]): #bench_converter_SL, bench_converter_LS, bench_converter_NP]):
#     bc = "NP" #["SL", "LS", "NP"][k]
modelist = [1, 2, 10, 20]
modelength = len(modelist)

original_flat = wfc.flat

# for ff, ffn in enumerate(tqdm(modelist)): #filelist:
for ff, ffn in enumerate(tqdm(filelist[39:]), start=39):
    cmd = wfc.read()
    wfc.flatten()
    time.sleep(0.01)
    
    d = np.load(ffn) #f'{folder_in}/{ff}.npy')
    # d = np.zeros((2, *cmd.shape))
    # d[:, ffn] = 1

    N = d.shape[0]

    psfs = np.empty((numPokePowers, N, *psf.imageShape))
    # cmd = wfc.read()
    cmds = np.empty((numPokePowers, N, *cmd.shape), dtype=cmd.dtype)
    shps = np.empty((numPokePowers, N, *wfc.layout.shape), dtype=cmd.dtype)

    for i, poke_power in enumerate(powerlist): #range(numModes):
        correction = np.zeros_like(wfc.read())
        for j in range(N):

            # if j == 0:
            #     psf.setGamma(1)
            # elif j == 1:
            #     psf.setGamma(4)

            #calib only
            # correction = (slopecorrect * poke_power * d[j, :].flatten()) #bench_converter @
            #actual testing only:
            correction = (slopecorrect * poke_power * d[j, wfc.layout].flatten()) #@wfc.M2C #bench_converter @
            
            wfc.flat = original_flat + correction
            # wfc.write(correction)

            #Burn some images
            psf.readLong()
            # psf.readLong()
            # psf.readLong()
            # time.sleep(0.5)

            #Save the next PSF in the dataset
            psfs[i, j, :, :] = psf.readLong()
            # cmds[i, j, :] = correction
            # shps[i, j, wfc.layout] = wfc.currentShape - wfc.flat
            wfc.flatten()
            psf.readLong()
    
    np.savez(f'{folder_out}/LandR_allfiles_take4_{ff}.npz', psf_out=psfs, file_name=ffn, power_list=powerlist)
    time.sleep(10)
        # np.save(f'{folder}/pinhole_cmds_{bc}', cmds)
        # np.save(f'{folder}/pinhole_shps_{bc}', shps)

# %%
# take flat image
wfc.flatten()
psf.readLong()
psf.readLong()
#Save the next PSF in the dataset
psfs = psf.readLong()
np.save(f'{folder}/flat', psfs)
wfc.flatten()

# %%
run_name = "psf_Sweep"
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
import matplotlib.pyplot as plt
IM = np.load("/home/whetstone/pyRTC/SHARP_LAB/calib/IM.npy")
IM = np.moveaxis(IM, 0 ,1)
pdiam = int(np.sqrt(IM.shape[1]/2))
IM = IM.reshape(IM.shape[0], 2*pdiam, pdiam)
mean_IM = np.std(IM,axis = 0)#np.mean(np.abs(IM), axis = 0)
min_threshold = 3
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
slopes.shwfsContrast = 10
slopes.offsetX = 13
slopes.offsetY = 3

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
# %%Scan Tip Tilts
wfcShm = initExistingShm("wfc")[0]
amp = 0.1
N = 50
tips = np.linspace(-amp,amp, N)
tilts = np.copy(tips)
res = np.empty((tips.size, tilts.size))
for i, tip in enumerate(tips):
    for j, tilt in enumerate(tilts):
        x = np.zeros_like(wfcShm.read_noblock())
        x[0] = tip
        x[1] = tilt
        wfcShm.write(x)
        time.sleep(0.01)
        res[i,j] = np.std(wfc.currentShape)
plt.imshow(res)
plt.show()

a,b = np.where(res==np.min(res))
x = np.zeros_like(wfcShm.read_noblock())
x[0] = tips[a]
x[1] = tilts[b]
wfcShm.write(x)