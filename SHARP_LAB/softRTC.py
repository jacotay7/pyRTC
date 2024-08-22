# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import time
from pyRTC import *
from pyRTC.hardware import *
from pyRTC.utils import *
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
cfile = "/home/whetstone/pyRTC/SHARP_LAB/config_SR.yaml"
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
loop = RLLoop(conf=conf)
# loop = Loop(conf=conf)
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
loop.computeCM()
loop.setGain(0.1)
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
powerlist = np.linspace(-10, 10, 21)
numPokePowers = len(powerlist)
slopecorrect = 0.0021
psf.integrationLength = 50

# for k, bench_converter in enumerate([bench_converter_NP]): #bench_converter_SL, bench_converter_LS, bench_converter_NP]):
#     bc = "NP" #["SL", "LS", "NP"][k]
modelist = range(10)
modelength = len(modelist)
# for ff in tqdm(modelist): #filelist:
for ff, ffn in enumerate(tqdm(filelist)):
    cmd = wfc.read()
    wfc.flatten()
    time.sleep(0.01)
    
    d = np.load(ffn) #f'{folder_in}/{ff}.npy')
    # d = np.zeros((modelength, *cmd.shape))
    # d[:, ff] = 1

    N = d.shape[0]

    psfs = np.empty((numPokePowers, N, *psf.imageShape))
    # cmd = wfc.read()
    cmds = np.empty((numPokePowers, N, *cmd.shape), dtype=cmd.dtype)
    shps = np.empty((numPokePowers, N, *wfc.layout.shape), dtype=cmd.dtype)

    for i, mode in enumerate(powerlist): #range(numModes):
        correction = np.zeros_like(wfc.read())
        for j in range(N):
            # correction[:21] = slopecorrect * mode * d[j, :].flatten()
            correction = (slopecorrect * mode * d[j, wfc.layout].flatten())@wfc.M2C #bench_converter @
            wfc.write(correction)
            #Burn some images
            psf.readLong()
            psf.readLong()
            psf.readLong()
            #Save the next PSF in the dataset
            psfs[i, j, :, :] = psf.readLong()
            cmds[i, j, :] = correction
            shps[i, j, wfc.layout] = wfc.currentShape - wfc.flat
            wfc.flatten()
            psf.readLong()
    
    np.savez(f'{folder_out}/laserandled_and_reschart_{ff}.npz', psfs, cmds, shps)
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
for i in range(N):
    wfc.push(0,0)
    loop.env._get_obs()
time_per_obs = (time.time() - start)/N * 1000
print(f"Time per obs {time_per_obs:.2f}ms")
# %%
# %% Reset SHWFS
slopes.setRefSlopes(np.zeros_like(slopes.refSlopes))
slopes.shwfsContrast = 20
slopes.offsetX = 8
slopes.offsetY = 4

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
# %%
import rl_zoo3
gym.make("pyRTCEnvPID-v0")
# %%
