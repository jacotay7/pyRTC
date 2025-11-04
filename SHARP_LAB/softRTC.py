# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import time
from pyRTC import *
from pyRTC.hardware import *
from pyRTC.utils import *
from pyRTC.Pipeline import *
from tqdm import trange

# from pyRTC.hardware.ximeaWFS import *
# # from pyRTC.WavefrontSensor import *
# from pyRTC.hardware.ALPAODM import *

# from pyRTC.hardware.SpinnakerScienceCam import *
# from pyRTC.SlopesProcess import *
# from pyRTC.Loop import *

# %% CLEAR SHMs
# from pyRTC.Pipeline import clear_shms
# shms = ["pupil", "wfs", "wfsRaw", "signal", "signal2D", "wfc", "wfc2D", "psfShort", "psfLong"]
# clear_shms(shms)

# %% Load Config
# cfile = "/home/whetstone/pyRTC_backup/SHARP_LAB/config_predict.yaml"
cfile = "/home/whetstone/pyRTC_backup/SHARP_LAB/config.yaml"
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
wfc.flatten()

# %% rom prediction.predictLoop import *
# loop = predictLoop(conf=conf['loop'])
psfShortShm, _, _ = initExistingShm("psfShort")
pupilShm, _, _ = initExistingShm("pupil")
time.sleep(1)

# %% Recalibrate

# Darks
if False:
    input("Sources Off?")
    wfs.takeDark()
    wfs.darkFile = "/home/whetstone/pyRTC_backup/SHARP_LAB/calib/dark.npy"
    wfs.saveDark()
    time.sleep(1)
    psf.takeDark()
    psf.darkFile = "/home/whetstone/pyRTC_backup/SHARP_LAB/calib/psfDark.npy"
    psf.saveDark()
    input("Sources On?")
    input("Is Atmosphere Out?")

    slopes.computeImageNoise()
    slopes.refSlopesFile = ""
    slopes.loadRefSlopes()
    wfc.flatten()
    slopes.takeRefSlopes()
    slopes.refSlopesFile = "/home/whetstone/pyRTC_backup/SHARP_LAB/calib/ref_SH.npy"
    slopes.saveRefSlopes()

    wfc.flatten()
    psf.takeModelPSF()
    psf.modelFile = "/home/whetstone/pyRTC_backup/SHARP_LAB/calib/modelPSF.npy"
    psf.saveModelPSF()

    #  STANDARD IM
    # wfc.setM2C(None)
    loop.IMMethod = "push-pull"
    loop.pokeAmp = 0.02
    loop.numItersIM = 100
    loop.IMFile = "/home/whetstone/pyRTC_backup/SHARP_LAB/calib/IM_SH.npy"
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
    # loop.IMFile = "/home/whetstone/pyRTC_backup/SHARP_LAB/calib/docrime_IM.npy"
    # wfc.flatten()
    # loop.computeIM()
    # loop.saveIM()
    # wfc.flatten()
    # time.sleep(1)

# %%
# wfc.loadFlat("calib/new_optim_flat.npy")
# wfc.flatten()
# time.sleep(0.5)

# %% Compute CM
# loop.IMFile = "/home/whetstone/pyRTC_backup/SHARP_LAB/calib/IM.npy"
# loop.numDroppedModes = 15
# loop.gain = 0.2
# loop.leakyGain = 0.02
loop.loadIM()

# %% Start Loop %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
wfc.flatten()
time.sleep(0.1)
loop.start()

# %%
loop.stop()
wfc.flatten()

# %% Stop Loop
psf.setExposure(500)
for nol in [4,5,6]:
    # %%

    # Currently the strehl optimizer maximizes the single pixel maximum intensity (inf norm)
    # N.B. that the 16bit float max value is 65504. Make sure your PSF image range falls well below 
    # this so that you aren't oversaturated when trying to optimize. If a pixel is always the 
    # max value then our optimization won't ever do anything :) 
    # e.g., psf.setExposure(500)

    # %%
    ncpaOptim = NCPAOptimizer(read_yaml_file(cfile)["optimizer"]["ncpa"], loop, slopes)

    loop.stop()
    wfc.flatten()
    time.sleep(0.3)
    wfc.flatten()
    loop.stop()

    # %% NCPA
    # ncpaOptim = NCPAOptimizer(read_yaml_file('/home/whetstone/pyRTC_backup/SHARP_LAB/config.yaml'
    # )["optimizer"]["ncpa"], loop, slopes)
    psf.integrationLength = 4
    ncpaOptim.numReads = 5
    ncpaOptim.startMode = 2
    ncpaOptim.endMode = 10 + 5*nol
    ncpaOptim.numSteps = 2000
    ncpaOptim.isCL = False
    ncpaOptim.correctionMag = 0.01 / ((1 + nol/2.0))

    # %%
    for i in range(1):
        ncpaOptim.optimize()
    ncpaOptim.applyOptimum()

    # %%
    ncpaOptim.applyOptimum()

    # %%
    wfc.saveShape("/home/whetstone/pyRTC_backup/SHARP_LAB/calib/new_optim_flat.npy")
    time.sleep(0.5)
    # %%
    wfc.loadFlat("/home/whetstone/pyRTC_backup/SHARP_LAB/calib/new_optim_flat.npy")
    wfc.flatten()
    time.sleep(0.5)

    # %%
    slopes.takeRefSlopes()
    time.sleep(0.5)

    # %%
    slopes.refSlopesFile = "/home/whetstone/pyRTC_backup/SHARP_LAB/calib/ref_SHWFS.npy"
    slopes.saveRefSlopes()
    time.sleep(0.5)

    # %%    wfc.flatten()
    psf.integrationLength = 5000
    psf.takeModelPSF()
    psf.takeModelPSF()
    # time.sleep(0.5)

    # %%
    psf.modelFile = "/home/whetstone/pyRTC_backup/SHARP_LAB/calib/modelPSF_SHWFS.npy"
    psf.saveModelPSF()
    time.sleep(0.5)
    psf.loadModelPSF()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

valid_sub_aps = slopes.validSubAps.copy()

# %% Refine Valid Sup Aps
nreads = 1000
slope_shape = slopes.signal2DShape
slopesx = np.zeros((nreads, slope_shape[0], slope_shape[1]))
# slopes.setValidSubAps(np.ones_like(valid_sub_aps))
time.sleep(1)

for i in trange(nreads):
    slopesx[i] = slopes.computeSignal2D(slopes.signal.read())
# slopes2D /= nreads

# %%
slopes2D = np.abs(slopesx).sum(axis=0).squeeze()
slopestd = slopesx.std(axis=0).squeeze()

plt.subplot(221)
plt.imshow(slopes2D)
plt.title("Abs Avg Slopes")
plt.colorbar()
# plt.show()

plt.subplot(222)
plt.imshow(slopestd)
plt.title("STD Slopes")
plt.colorbar()

mask = (slopes2D < 20) | (slopes2D > 4500)
combineXY = mask[: mask.shape[0] // 2, :] | mask[mask.shape[0] // 2 :, :]
mask[: mask.shape[0] // 2, :] = combineXY
mask[mask.shape[0] // 2 :, :] = combineXY
plt.subplot(234)
plt.imshow(mask)
plt.title("Intensity Mask")

mask2 = np.abs(slopestd) > 0.15
combineXY = mask2[: mask.shape[0] // 2, :] | mask2[mask.shape[0] // 2 :, :]
mask[: mask.shape[0] // 2, :] += combineXY
mask[mask.shape[0] // 2 :, :] += combineXY
plt.subplot(235)
plt.imshow(mask2)
plt.title("Flakey Sub-Aps Mask")
# plt.show()

# valid_sub_aps[mask] = 0
plt.subplot(236)
plt.imshow(mask2 | mask)
plt.title("Combined Mask")
plt.show()

# %%
slopes.validSubApsFile = "/home/whetstone/pyRTC_backup/SHARP_LAB/calib/validSubAps.npy"
slopes.setValidSubAps(~(mask2 | mask).astype(bool))
slopes.saveValidSubAps()

# %%
# np.save(
#     "/home/whetstone/pyRTC_backup/SHARP_LAB/calib/validSubAps.npy",
#     mask.astype(bool),
# )



# # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# # %%
# else:
#     # %

#     ##
#     ## This is the one I use with RTC predict validation :)
#     ##

#     import optuna
#     from tqdm import trange

#     optuna.logging.set_verbosity(optuna.logging.DEBUG)
#     numOptim = 1
#     maxAMP = 0.01
#     amps = np.linspace(maxAMP, maxAMP / 5, numOptim)
#     ncpaOptim = extendedImageOptimizer(
#         read_yaml_file(config)["optimizer"]["ncpa"], loop, slopes
#     )

#     for i in range(numOptim):
#         # hardFlat()
#         psf.setProperty("integrationLength", 1)

#         ncpaOptim.resetStudy()
#         ncpaOptim.numReads = 1
#         ncpaOptim.startMode = 0
#         ncpaOptim.endMode = 40  # *(i+1)
#         ncpaOptim.numSteps = 1000
#         ncpaOptim.correctionMag = amps[i]
#         ncpaOptim.isCL = True

#         time.sleep(2)

#         # for _ in trange(100):
#         ncpaOptim.optimize()
#         ncpaOptim.applyOptimum()

#         wfc.run("saveShape")
#         slopes.run("takeRefSlopes")
#         slopes.setProperty(
#             "refSlopesFile", "/home/whetstone/pyRTC_backup/SHARP_LAB/calib/ref_SH.npy"
#         )
#         slopes.run("saveRefSlopes")
#         psf.setProperty("integrationLength", 2000)
#         # time.sleep(2)
#         # psf.run("takeModelPSF")
#         # psf.setProperty("modelFile", "/home/whetstone/pyRTC_backup/SHARP_LAB/calib/modelPSF_SH.npy")
#         # psf.run("saveModelPSF")
#         wfc.run("loadFlat")

#     # % Float Actuators
#     wfc.floatingInfluenceRadius = 1
#     wfc.reactivateActuators([i for i in range(97)])
#     wfc.deactivateActuators(
#         [
#             0,
#             1,
#             2,
#             3,
#             4,
#             5,
#             11,
#             12,
#             20,
#             21,
#             31,
#             32,
#             42,
#             43,
#             53,
#             54,
#             64,
#             65,
#             75,
#             76,
#             84,
#             85,
#             91,
#             92,
#             93,
#             94,
#             95,
#             96,
#         ]
#     )

#     # % Find Best Strehl Method
#     psf.takeModelPSF()
#     metric = []
#     gs = np.linspace(0, 2, 10)
#     for g in gs:
#         g = 1
#         strehls = []
#         for i in range(100):
#             strehls.append(psf.computeStrehl(gaussian_sigma=g))
#         metric.append(np.std(np.array(strehls)))
#     plt.plot(gs, metric)
#     plt.show()

#     # % Bench Conversion
#     # % Compute CM
#     # loop.IMFile = "/home/whetstone/pyRTC_backup/SHARP_LAB/cIM.npy"
#     loop.loadIM()
#     loop.numDroppedModes = 20
#     loop.computeCM()
#     # %
#     SIM = (
#         np.load(
#             "/home/whetstone/pyRTC_backup/SHARP_LAB/calib/sprint_IM_nomisreg_valid.npy"
#         )
#         .reshape(94, -1)
#         .T
#     )
#     bench_converter_SL = np.linalg.pinv(SIM) @ loop.IM  # [:,:21]
#     bench_converter_LS = loop.CM @ SIM  # [:,:21]
#     bench_converter_NP = np.eye(94)  # [:,:21]
# # %% Tip-Tilt-Focus-Sweep
# from tqdm import tqdm
# import glob

# folder_out = "/media/whetstone/storage2/data/robin-aug"
# folder_in = "/home/whetstone/Downloads/torch/torch/phases"
# # filelist = ['cnnx2_phase', 'cnnx4_phase','linx2_phase', 'linx4_phase', 'cnnx8n3_phase','linx8n3_phase']
# # filelist = ['_zern_cnn_x2_n4_phase', '_zern_cnn_x4_n4_phase', '_zern_cnn_x8_n4_phase', '_fixzern_lin_x2_n4_phase', '_fixzern_lin_x4_n4_phase', '_fixzern_lin_x8_n4_phase']
# # N = 3
# filelist = glob.glob(f"{folder_in}/LastAttempt8N_*.npy")

# ## based on my investigations....
# # ['z=2: 4.650000', 'z=4: 3.175000', 'z=6: 1.700000', 'z=8: 0.225000']
# # 2x := 4.60 (ind 23)
# # 4x := 3.25 (ind 13)
# # 8x := 0.20 (ind 1)
# powerlist = [0, 0.15, 0.20, 0.25, 3.15, 3.25, 3.35, 4.5, 4.6, 4.7]

# # filelist = [
# # 'cnn_8X_VARON_modal_0BNM_10INM.npy',
# # 'cnn_4X_VARON_modal_0BNM_10INM.npy',
# # 'cnn_2X_VARON_modal_0BNM_10INM.npy'
# # ]

# # numModes = 11
# # powerlist = [0, 0.25, 0.5, 1., 2., 4., 8., 16] #np.linspace(-RANGE, RANGE, numModes) #.astype(int)
# # powerlist = np.linspace(0, 10, 51)
# numPokePowers = len(powerlist)
# slopecorrect = 0.0021
# # assert(psf.run("setIntegrationLength", 100))
# # assert(psf.run("setExposure", 100000))
# # assert(psf.run("setGamma", 0.5))
# # assert(psf.run("setGain", 16))

# # USAF
# psf.setIntegrationLength(2)
# psf.setExposure(500000)
# psf.setGamma(1)
# psf.setGain(10)

# # CGLA Slides
# # psf.setExposure(900000)
# # psf.setGamma(3)
# # psf.setGain(16)


# # 8mm Film
# # psf.setExposure(900000)
# # psf.setGamma(0.5)
# # psf.setGain(17)

# # psf.setGamma(4)
# psfshm, _, psfDtype = initExistingShm("psfLong")
# psfShortShm, _, _ = initExistingShm("psfShort")

# # for k, bench_converter in enumerate([bench_converter_NP]): #bench_converter_SL, bench_converter_LS, bench_converter_NP]):
# #     bc = "NP" #["SL", "LS", "NP"][k]
# modelist = np.linspace(0, 90, 30, dtype=int)
# modelength = len(modelist)

# # flat_file = wfc.getProperty("flatFile")
# original_flat = slopes.refSlopes.copy()  # = wfc.flat.copy() #np.load(flat_file)

# # %%
# # for ff, ffn in enumerate(modelist): #filelist:
# for ff, ffn in enumerate(filelist):
#     # for ff, ffn in enumerate(tqdm(filelist[39:]), start=39):
#     refSlopesAdjust = np.zeros_like(original_flat)  # original_flat.shape
#     wfc.flatten()
#     time.sleep(0.01)

#     # d = np.load('/home/whetstone/Downloads/torch/torch/phases/' + ffn) #f'{folder_in}/{ff}.npy')
#     d = np.load(ffn)  # f'{folder_in}/{ff}.npy')
#     # d = np.zeros((1, loop.IM.shape[-1]))
#     # d[:, ffn] = 1

#     N = d.shape[0]
#     psfs = np.empty((numPokePowers, N, *psfShortShm.read_noblock().shape))

#     true_powers = powerlist.copy()
#     if "8X" in ffn:
#         true_powers = true_powers[0:4]
#         print(f"8X: {true_powers} {ffn}")
#     elif "4X" in ffn:
#         true_powers = [0] + true_powers[4:7]
#         print(f"4X: {true_powers} {ffn}")

#     elif "2X" in ffn:
#         true_powers = [0] + true_powers[7:]
#         print(f"2X: {true_powers} {ffn}")
#     else:
#         print(f"Unknown Mag X")

#     for i, poke_power in enumerate(tqdm(true_powers)):  # range(numModes):
#         correction = np.zeros_like(original_flat)

#         for j in range(N) if (i > 0) else range(1):

#             # calib only
#             # correction = (slopecorrect * poke_power * d[j, :].flatten()) #bench_converter @
#             # actual testing only:
#             correction = wfc.C2M @ (
#                 slopecorrect * poke_power * d[j, wfc.layout].flatten()
#             )  # @wfc.M2C #bench_converter @
#             refSlopesAdjust[slopes.validSubAps] = loop.IM @ correction

#             # Adjust reference slopes
#             slopes.setRefSlopes(original_flat + refSlopesAdjust)
#             wfc.flatten()
#             wfc.flatten()

#             # Burn some images
#             psfshm.read()
#             wfc.flatten()
#             wfc.flatten()

#             # Save the next PSF in the dataset
#             psfs[i, j, :, :] = psfshm.read()
#             slopes.setRefSlopes(original_flat)
#             wfc.flatten()

#     print("saving...")
#     np.savez(
#         f'{folder_out}/lasttry8N_usaf_{ffn.split("/")[-1][:-4]}.npz',
#         mode_list=modelist,
#         psf_out=psfs,
#         file_name=ffn,
#         power_list=powerlist,
#     )
#     # np.savez(f'{folder_out}/actual_last_0to10_usaf_{ffn}_{ff}.npz', mode_list=modelist, psf_out=psfs, file_name=ffn, power_list=powerlist)
#     # np.savez(f'{folder_out}/oneshot_flatfield_grape.npz', mode_list=modelist, psf_out=psfs, file_name=ffn, power_list=powerlist)

#     # time.sleep(1)
#     # np.save(f'{folder}/pinhole_cmds_{bc}', cmds)
#     # np.save(f'{folder}/pinhole_shps_{bc}', shps)


# # %%
# # take flat image
# wfc.flatten()
# psf.readLong()
# psf.readLong()
# # Save the next PSF in the dataset
# psfs = psf.readLong()
# np.save(f"{folder}/flat", psfs)
# wfc.flatten()

# # %%
# run_name = "psf_Sweep"
# folder += run_name
# filename = f"{folder}/psfs_{startMode}_{endMode}_{numModes}_{N}"
# np.save(filename, psfs)
# filename = f"{folder}/cmds_{startMode}_{endMode}_{numModes}_{N}"
# np.save(filename, cmds)
# # Save WFC info
# np.save(f"{folder}/M2C", wfc.M2C)
# np.save(f"{folder}/DM_LAYOUT", wfc.layout)

# # %% Time A SHM
# shmName = "wfc2D"
# metadataSHM = ImageSHM(shmName + "_meta", (ImageSHM.METADATA_SIZE,), np.float64)
# N = 1000
# times = np.empty(N)
# counts = np.empty(N)
# for i in range(N):
#     metadata = metadataSHM.read()
#     counts[i] = metadata[0]
#     times[i] = metadata[1]
#     time.sleep(1e-3)

# # Plot the Timing Variance
# dt = times[1:] - times[:-1]
# dc = counts[1:] - counts[:-1]
# speeds = 1000 * (dt[dc > 0] / dc[dc > 0])
# plt.hist(1 / speeds, bins="sturges")
# plt.show()
# # %% Generate Valid SubAps for SHWFS
# # First make an IM with all valid subAps
# import matplotlib.pyplot as plt

# IM = np.load("/home/whetstone/pyRTC_backup/SHARP_LAB/calib/IM_SH.npy")
# IM = np.moveaxis(IM, 0, 1)
# pdiam = int(np.sqrt(IM.shape[1] / 2))
# IM = IM.reshape(IM.shape[0], 2 * pdiam, pdiam)
# mean_IM = np.mean(np.abs(IM), axis=0)  # np.mean(np.abs(IM), axis = 0)
# min_threshold = 3.25
# max_threshold = 20
# valid_sub_aps = (mean_IM > min_threshold) & (mean_IM < max_threshold)
# combineXY = (
#     valid_sub_aps[: valid_sub_aps.shape[0] // 2, :]
#     & valid_sub_aps[valid_sub_aps.shape[0] // 2 :, :]
# )
# valid_sub_aps[: valid_sub_aps.shape[0] // 2, :] = combineXY
# valid_sub_aps[valid_sub_aps.shape[0] // 2 :, :] = combineXY
# plt.imshow(mean_IM)
# plt.colorbar()
# plt.show()
# plt.imshow(valid_sub_aps)
# plt.show()

# # %%
# np.save(
#     "/home/whetstone/pyRTC_backup/SHARP_LAB/validSubAps.npy", valid_sub_aps.astype(bool)
# )


# # %%
# img = wfs.read()
# spacing = 7.7
# i, j = 5, 4
# offsetX, offsetY = -2, 4
# plt.imshow(
#     img[
#         int(spacing * i) + offsetY : int(spacing * (i + 1)) + offsetY,
#         int(spacing * j) + offsetX : int(spacing * (j + 1)) + offsetX,
#     ]
# )

# plt.show()
# # %%
# from scipy.signal import find_peaks

# arr = np.sum(img, axis=1)
# peaks = find_peaks(arr)
# plt.plot(arr)
# spots = []
# for p in peaks[0]:
#     if arr[p] > 1000:
#         plt.axvline(x=p, color="r")
#         spots.append(p)
# plt.show()
# spots = np.array(spots)
# print(np.mean(spots[1:] - spots[:-1]))
# print(peaks)
# # %% Plot Flat
# x = np.zeros(wfc.layout.shape)
# x[wfc.layout] = wfc.flat
# plt.imshow(x)
# plt.colorbar()
# plt.show()

# # %% Plot slope err vs contrast

# contrasts = np.linspace(0, 100, 20)
# vars = []
# for c in contrasts:
#     slopes.shwfsContrast = c
#     slopes.takeRefSlopes()
#     arr = []
#     for i in range(1000):
#         arr.append(slopes.read())
#     arr = np.array(arr)
#     vars.append(np.std(arr))
# plt.plot(contrasts, vars, "x")
# plt.xlabel("SHWFS Threshold Parameter")
# plt.ylabel("Slope Deviation")
# plt.show()

# # %% Reset SHWFS
# slopes.setRefSlopes(np.zeros_like(slopes.refSlopes))
# slopes.shwfsContrast = 10
# slopes.offsetX = 9
# slopes.offsetY = 12

# # %% Find SHWFS Offsets
# # slopes.subApSpacing = 15.54
# vals = []
# for offsetX in range(0, int(slopes.subApSpacing)):
#     for offsetY in range(0, int(slopes.subApSpacing)):
#         slopes.offsetX = offsetX
#         slopes.offsetY = offsetY
#         arr = []
#         for i in range(20):
#             arr.append(slopes.read())
#         arr = np.array(arr)
#         arr = np.mean(arr, axis=0)
#         arr = arr.flatten()
#         vals.append((offsetX, offsetY, np.mean(np.abs(arr))))
# vals = np.array(vals)
# print(vals[vals[:, 2] == np.nanmin(vals[:, 2])])

# # %% Find SHWFS Spacing
# cur_wfs = wfs.read()
# plt.imshow(cur_wfs)
# plt.show()

# spacing = slopes.subApSpacing - 0.02
# plt.plot(np.mean(cur_wfs, axis=0))
# x = slopes.offsetX
# while x < cur_wfs.shape[1]:
#     plt.axvline(x=x, color="r", linestyle="--")
#     x += spacing
# plt.show()

# plt.plot(np.mean(cur_wfs, axis=1))
# y = slopes.offsetY
# while y < cur_wfs.shape[0]:
#     plt.axvline(x=y, color="r", linestyle="--")
#     y += spacing
# plt.show()


# # %%
# def compute_fwhm(image):
#     # Filter to keep only negative values
#     negative_pixels = image[image < 1]

#     # Compute the histogram of negative values
#     # Adjust bins and range as necessary for your specific image
#     hist, bins = np.histogram(
#         negative_pixels, bins=np.arange(np.min(negative_pixels), 1) + 0.5
#     )
#     print(bins)
#     # Since the distribution is symmetric, we can mirror the histogram to get the full distribution
#     hist_full = np.concatenate((hist[::-1], hist))

#     # Compute the bin centers from the bin edges
#     bin_centers = (bins[:-1] + bins[1:]) / 2
#     bin_centers_full = np.concatenate((-bin_centers[::-1], bin_centers))

#     print(bin_centers_full)
#     plt.plot(bin_centers_full, hist_full, "x")
#     plt.show()

#     # Find the maximum value (mode of the distribution)
#     peak_value = np.max(hist_full)
#     half_max = peak_value / 2

#     # Find the points where the histogram crosses the half maximum
#     cross_points = np.where(np.diff((hist_full > half_max).astype(int)))[0]

#     # Assuming the distribution is sufficiently smooth and has a single peak,
#     # the FWHM is the distance between the first and last crossing points
#     fwhm_value = np.abs(
#         bin_centers_full[cross_points[-1]] - bin_centers_full[cross_points[0]]
#     )

#     return fwhm_value, bin_centers_full, hist_full, half_max


# print(compute_fwhm(wfs.read())[0])


# # %%
# # %%Scan Tip Tilts
# wfcShm = initExistingShm("wfc")[0]
# amp = 0.1
# N = 50
# tips = np.linspace(-amp, amp, N)
# tilts = np.copy(tips)
# res = np.empty((tips.size, tilts.size))
# for i, tip in enumerate(tips):
#     for j, tilt in enumerate(tilts):
#         x = np.zeros_like(wfcShm.read_noblock())
#         x[0] = tip
#         x[1] = tilt
#         wfcShm.write(x)
#         time.sleep(0.01)
#         res[i, j] = np.std(wfc.currentShape)
# plt.imshow(res)
# plt.show()

# a, b = np.where(res == np.min(res))
# x = np.zeros_like(wfcShm.read_noblock())
# x[0] = tips[a]
# x[1] = tilts[b]
# wfcShm.write(x)
