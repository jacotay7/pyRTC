# %% IMPORTS
#Import pyRTC classes
from pyRTC.Pipeline import *
from pyRTC.utils import *
import os
os.chdir("/home/whetstone/pyRTC/SHARP_LAB")
RECALIBRATE = False

# %% Clear SHMs
# from pyRTC.Pipeline import clear_shms
# shm_names = ["signal"]
# shm_names = ["wfs", "wfsRaw", "wfc", "wfc2D", "signal", "signal2D", "psfShort", "psfLong"] #list of SHMs to reset
# clear_shms(shm_names)

# %% IMPORTS
# config = '/home/whetstone/pyRTC/SHARP_LAB/config_SR.yaml'
config = '/home/whetstone/pyRTC/SHARP_LAB/config.yaml'
N = np.random.randint(3000,6000)
# %% Launch DM
wfc = hardwareLauncher("../pyRTC/hardware/ALPAODM.py", config, N)
wfc.launch()

# %% Launch WFS
wfs = hardwareLauncher("../pyRTC/hardware/ximeaWFS.py", config, N+1)
wfs.launch()

# %% Launch slopes
slopes = hardwareLauncher("../pyRTC/SlopesProcess.py", config, N+2)
slopes.launch()

# %% Launch PSF Cam
psfCam = hardwareLauncher("../pyRTC/hardware/SpinnakerScienceCam.py", config, N+10)
psfCam.launch()

# %% Launch Loop Class
loop = hardwareLauncher("../pyRTC/Loop.py", config, N+4)
# loop = hardwareLauncher("../pyRTC/hardware/predictLoop.py", config)
loop.launch()

# %% OPTIMIZERS
from pyRTC.hardware.PIDOptimizer import PIDOptimizer
pidOptim = PIDOptimizer(read_yaml_file(config)["optimizer"]["pid"], loop)

from pyRTC.hardware.NCPAOptimizer import NCPAOptimizer
ncpaOptim = NCPAOptimizer(read_yaml_file(config)["optimizer"]["ncpa"], loop, slopes)

# %% Calibrate

if RECALIBRATE == True:

    slopes.setProperty("refSlopesFile", "")
    slopes.run("loadRefSlopes")

    input("Sources Off?")
    wfs.run("takeDark")
    wfs.setProperty("darkFile", "/home/whetstone/pyRTC/SHARP_LAB/calib/dark.npy")
    wfs.run("saveDark")
    time.sleep(1)
    psfCam.run("takeDark")
    psfCam.setProperty("darkFile", "/home/whetstone/pyRTC/SHARP_LAB/calib/psfDark.npy")
    psfCam.run("saveDark")
    input("Sources On?")
    input("Is Atmosphere Out?")

    slopes.run("computeImageNoise")
    slopes.run("takeRefSlopes")
    slopes.setProperty("refSlopesFile", "/home/whetstone/pyRTC/SHARP_LAB/calib/ref.npy")
    slopes.run("saveRefSlopes")

    wfc.run("flatten")
    psfCam.run("takeModelPSF")
    psfCam.setProperty("modelFile", "/home/whetstone/pyRTC/SHARP_LAB/calib/modelPSF.npy")
    psfCam.run("saveModelPSF")

    #  STANDARD IM
    loop.setProperty("IMMethod", "push-pull")
    loop.setProperty("pokeAmp", 0.03)
    loop.setProperty("numItersIM", 100)
    loop.setProperty("IMFile", "/home/whetstone/pyRTC/SHARP_LAB/calib/IM.npy")
    wfc.run("flatten")
    loop.run("computeIM")
    loop.run("saveIM")
    wfc.run("flatten")
    time.sleep(1)

    # input("Is Atmosphere In?")
    # #  DOCRIME OL
    # loop.setProperty("IMMethod", "docrime")
    # loop.setProperty("delay", 1)
    # loop.setProperty("pokeAmp", 2e-2)
    # loop.setProperty("numItersIM", 30000)
    # loop.setProperty("IMFile", "/home/whetstone/pyRTC/SHARP_LAB/calib/docrime_IM.npy")
    # wfc.run("flatten")
    # loop.run("computeIM")
    # loop.run("saveIM")
    # wfc.run("flatten")
    # time.sleep(1)



# %% Compute CM
loop.setProperty("IMFile", "/home/whetstone/pyRTC/SHARP_LAB/calib/IM.npy")
loop.run("loadIM")
time.sleep(0.5)
loop.setProperty("numDroppedModes", 15)
loop.run("computeCM")
time.sleep(0.5)
# %% Adjust Gains
# loop.run("setGain",1e-2)
# loop.setProperty("leakyGain", 0.3)
loop.setProperty("leakyGain", 1e-2)
#PID GAINS (only for PID integrator)
loop.setProperty("pGain", 0.3)
loop.setProperty("iGain", 5e-2)
loop.setProperty("dGain", 5e-2)
loop.setProperty("controlLimits", [-0.05, 0.05])
loop.setProperty("absoluteLimits", [-1.0, 1.0])
loop.setProperty("integralLimits", [-1.0, 1.0])
# %%Launch Loop for 5 seconds
wfc.run("flatten")
loop.run("start")
time.sleep(1)
# %% Stop Loop
loop.run("stop")
wfc.run("flatten")
wfc.run("flatten")

#%% Optimize PID
pidOptim.numReads = 10
pidOptim.numSteps = 50
pidOptim.isPOL = False
pidOptim.mode = "tiptilt"
pidOptim.maxPGain = 0.4
pidOptim.maxDGain = 1e-1
pidOptim.maxIGain = 1e-1
for i in range(1):
    pidOptim.optimize()
pidOptim.applyOptimum()
#%% Optimize NCPA
ncpaOptim.numReads = 2
ncpaOptim.startMode = 2
ncpaOptim.endMode = 30
ncpaOptim.optimize()
# ncpaOptim.applyOptimum()
# wfc.run("saveShape")
# slopes.run("takeRefSlopes")
# slopes.setProperty("refSlopesFile", "/home/whetstone/pyRTC/SHARP_LAB/calib/ref.npy")
# slopes.run("saveRefSlopes")
# psfCam.run("takeModelPSF")
# psfCam.setProperty("modelFile", "/home/whetstone/pyRTC/SHARP_LAB/calib/modelPSF.npy")
# psfCam.run("saveModelPSF")




# %% Plots
im = np.load("../SHARP_LAB/calib/docrime_IM.npy")
plt.imshow(im, aspect="auto")
plt.show()

im = im.reshape(*slopes.getProperty("signalShape"), -1)
im = np.moveaxis(im, 2, 0)
plt.imshow(np.sum(np.abs(im), axis = 0))
plt.show()

for i in range(85,90):
    plt.imshow(im[i])
    plt.colorbar()
    plt.show()


im_dc = np.load("../SHARP_LAB/calib/docrime_IM.npy")
im = np.load("../SHARP_LAB/calib/IM.npy")

im_dc = im_dc.reshape(*slopes.getProperty("signalShape"), -1)
im_dc = np.moveaxis(im_dc, 2, 0)
plt.plot(np.std(im_dc, axis = (1,2)))


im = im.reshape(*slopes.getProperty("signalShape"), -1)
im = np.moveaxis(im, 2, 0)
plt.plot(np.std(im, axis = (1,2)))
plt.show()

plt.plot(np.mean(im_dc, axis = (1,2)))
plt.plot(np.mean(im, axis = (1,2)))
plt.show()

plt.imshow(im_dc[-1], aspect="auto")
plt.colorbar()
plt.show()
plt.imshow(im[-1], aspect="auto")
plt.colorbar()
plt.show()
# %% SVD

im_dc = np.load("../SHARP_LAB/calib/docrime_IM.npy")
im = np.load("../SHARP_LAB/calib/IM.npy")
im_sprint = np.load("../SHARP_LAB/calib/sprint_IM.npy")

#RESCALES SPRINT TO MATCH EMPIRICAL
# for i in range(im_sprint.shape[1]):
#     im_sprint[:,i] *= np.std(im[:,i])/np.std(im_sprint[:,i])
# np.save("../SHARP_LAB/calib/sprint_IM.npy", im_sprint)

u,s,v = np.linalg.svd(im)
plt.plot(s/np.max(s), label = 'EMPIRICAL')
u,s,v = np.linalg.svd(im_dc)
plt.plot(s/np.max(s), label = 'DOCRIME')
u,s,v = np.linalg.svd(im_sprint)
plt.plot(s/np.max(s), label = 'SPRINT')
plt.yscale("log")
plt.ylim(1e-3,1.5)
plt.xlabel("Eigen Mode #", size = 18)
plt.ylabel("Normalizaed Eigenvalue", size = 18)
plt.legend()
plt.show()

plt.plot(np.std(im, axis = 0), label = 'EMPIRICAL')
plt.plot(np.std(im_dc, axis = 0), label = 'DOCRIME')
plt.plot(np.std(im_sprint, axis = 0), label = 'SPRINT')
plt.xlabel("Mode #", size = 18)
plt.ylabel("Standard Deviation", size = 18)
plt.legend()
plt.show()


N, M = 0,5
im_sprint = im_sprint.reshape(*slopes.getProperty("signalShape"), -1)
im_sprint = np.moveaxis(im_sprint, 2, 0)
a = np.vstack(im_sprint[N:M])

im_dc = im_dc.reshape(*slopes.getProperty("signalShape"), -1)
im_dc = np.moveaxis(im_dc, 2, 0)
b = np.vstack(im_dc[N:M])

im = im.reshape(*slopes.getProperty("signalShape"), -1)
im = np.moveaxis(im, 2, 0)
c = np.vstack(im[N:M])

plt.imshow(np.hstack([a,b,c]))
plt.show()

plt.imshow(a-b)
plt.show()

# %% Kill everything
# hardware = [slopes, psfCam, wfs, wfc, loop]
# for h in hardware:
#     h.shutdown()
#     time.sleep(1)

# %%
wfc.run("deactivateActuators",[0,1,2,3,4,5,11,12,20,21,31,32,42,43,53,54,64,65,75,76,84,85,91,92,93,94,95,96])
# %%
wfc.run("reactivateActuators",[i for i in range(97)])
# %% Strehl Monitor
psfCam.run("computeStrehl")
print(psfCam.getProperty("strehl_ratio"))
# %%

# %%
from tqdm import tqdm

folder = "~/Downloads/robin-april-16/"

# numModes = 10
# startMode = 0
# endMode = wfc.numModes - 1
filelist = ['cnnx2_phase.npy', 'cnnx4_phase', 'linx2_phase.npy', 'linx4_phase.npy']
N = 4
numModes = 11
RANGE = 2
modelist = np.linspace(-RANGE, RANGE, numModes) #.astype(int)

for ff in filelist:
    psfs = np.empty((numModes, N, *psfCam.getProperty("imageShape")))
    cmd = wfc.read()
    cmds = np.empty((numModes, N, *cmd.getProperty("shape")), dtype=cmd.dtype)
    wfc.flatten()
    time.sleep(0.1)
    d = np.read(f'{folder}/{ff}')
    for i, mode in enumerate(modelist): #range(numModes):
        correction = np.zeros_like(wfc.read())
        for j in range(N):
            correction[mode] = mode * d[j, :].flatten()
            wfc.write(correction)
            #Burn some images
            psfCam.readLong()
            #Save the next PSF in the dataset
            psfs[i, j, :, :] = psfCam.readLong()
            cmds[i,j,:] = correction
            wfc.flatten()
            time.sleep(0.1)
    
    np.save(f'{folder}/psfs_{ff}', psfs)
    np.save(f'{folder}/cmds_{ff}', cmds)
# %%
# %% CODE TO UPDATE FLAKY SUB APERTURES

# subApFile = slopes.getProperty("validSubApsFile")
# print(subApFile)
# vsubAp = np.load(subApFile)
# plt.imshow(vsubAp)
# plt.show()

# from pyRTC.Pipeline import initExistingShm
# shm, _, _ = initExistingShm("signal2D")

# x = shm.read_noblock()
# N = 1000
# data = np.zeros((N, *x.shape))
# for i in range(N):
#     data[i] = shm.read_noblock()
#     time.sleep(3e-3)

# arr = np.std(data,axis =0)
# aps_to_remove = arr > np.percentile(arr, 99)
# aps_to_remove[:aps_to_remove.shape[0]//2] |= aps_to_remove[aps_to_remove.shape[0]//2:]
# aps_to_remove[aps_to_remove.shape[0]//2:] |= aps_to_remove[:aps_to_remove.shape[0]//2]
# # aps_to_remove |= vsubAp
# plt.imshow(aps_to_remove)
# plt.show()
# new_valid = np.copy(vsubAp)
# new_valid[aps_to_remove] = False
### np.save(subApFile, new_valid)


#%%
shm, a, b = initExistingShm("wfc")

data = np.empty((10000, *a))
for i in range(data.shape[0]):
    data[i] = shm.read()
plt.imshow(data, aspect='auto')
plt.colorbar()
# %%
for trial in pidOptim.study.trials:
    print(trial)
    break