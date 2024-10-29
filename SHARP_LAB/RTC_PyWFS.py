# %% IMPORTS
#Import pyRTC classes
import matplotlib.pyplot as plt
from pyRTC.Pipeline import *
from pyRTC.utils import *
from pyRTC.hardware import *
import os
os.chdir("/home/whetstone/pyRTC/SHARP_LAB")
RECALIBRATE = False
CLEAR_SHMS =  False
# %% Clear SHMs
if CLEAR_SHMS:
    shm_names = ["wfs", "wfsRaw", "wfc", "wfc2D", "signal", "signal2D", "psfShort", "psfLong"] #list of SHMs to reset
    clear_shms(shm_names)
# %% IMPORTS
config = '/home/whetstone/pyRTC/SHARP_LAB/config_pywfs.yaml'
N = np.random.randint(3000,6000)
folder = "/home/whetstone/pyRTC/SHARP_LAB/calib/"

# %% Launch WFS
wfs = hardwareLauncher("../pyRTC/hardware/ximeaWFS.py", config, N+1)
wfs.launch()

# %% Launch slopes
slopes = hardwareLauncher("../pyRTC/SlopesProcess.py", config, N+2)
slopes.launch()

# %% Launch PSF Cam
psfCam = hardwareLauncher("../pyRTC/hardware/SpinnakerScienceCam.py", config, N+10)
psfCam.launch()

# %% Launch DM
wfc = hardwareLauncher("../pyRTC/hardware/ALPAODM.py", config, N)
wfc.launch()

# %% Launch Modulator
modulator = hardwareLauncher("../pyRTC/hardware/PIModulator.py", config, N+100)
modulator.launch()

# %% Launch Loop Class
loop = hardwareLauncher("../pyRTC/Loop.py", config, N+4)
loop.launch()

# %% OPTIMIZERS
from pyRTC.hardware.PIDOptimizer import PIDOptimizer
pidOptim = PIDOptimizer(read_yaml_file(config)["optimizer"]["pid"], loop)
# %%
from pyRTC.hardware.NCPAOptimizer import NCPAOptimizer
ncpaOptim = NCPAOptimizer(read_yaml_file(config)["optimizer"]["ncpa"], loop, slopes)


loopOptim = loopOptimizer(read_yaml_file(config)["optimizer"]["loop"], loop)

# %% Calibrate
if RECALIBRATE == True:

    input("Sources Off?")
    wfs.run("takeDark")
    wfs.setProperty("darkFile", folder + "darkPyWFS.npy")
    wfs.run("saveDark")
    time.sleep(1)
    psfCam.run("takeDark")
    psfCam.setProperty("darkFile", folder + "psfDark_PyWFS.npy")
    psfCam.run("saveDark")

    input("Sources On?")
    input("Is Atmosphere Out?")
    wfc.run("flatten")
    psfCam.run("takeModelPSF")
    psfCam.setProperty("modelFile", folder + "modelPSF_PyWFS.npy")
    psfCam.run("saveModelPSF")

    slopes.run("takeRefSlopes")
    slopes.setProperty("refSlopesFile", folder + "refPyWFS.npy")
    slopes.run("saveRefSlopes")

    #  STANDARD IM
    loop.setProperty("IMMethod", "push-pull")
    loop.setProperty("pokeAmp", 0.03)
    loop.setProperty("numItersIM", 100)
    loop.setProperty("IMFile", folder + "IM_PYWFS.npy")
    wfc.run("flatten")
    time.sleep(1)
    loop.run("computeIM")
    loop.run("saveIM")
    wfc.run("flatten")
    time.sleep(1)

    input("Is Atmosphere In?")
    #  DOCRIME OL
    loop.setProperty("IMMethod", "docrime")
    loop.setProperty("delay", 2) #Needs to be set in the CONFIG
    loop.setProperty("pokeAmp", 2e-2)
    loop.setProperty("numItersIM", 50000)
    loop.setProperty("IMFile", folder + "IM_PYWFS_OL_docrime.npy")
    wfc.run("flatten")
    time.sleep(1)
    loop.run("computeIM")
    loop.run("saveIM")
    wfc.run("flatten")
    time.sleep(1)


# %% Compute CM
loop.setProperty("IMFile", folder + "IM_PYWFS.npy")
psfCam.setProperty("integrationLength", 2000)
loop.setProperty("numDroppedModes", 5)
loop.setProperty("gain",0.6)
loop.setProperty("leakyGain", 0.01)

"""
LE
"""
loop.setProperty("alpha", 1.0)
wfc.run("setDelay", 0)

loop.run("loadIM")
time.sleep(0.5)

# %%Launch Loop for 5 seconds
wfc.run("flatten")
loop.run("start")
time.sleep(1)
# %% Stop Loop
loop.run("stop")
wfc.run("flatten")
wfc.run("flatten")

# %% CL DOCRIME Start
loop.setProperty("pokeAmp", 0.002)
loop.setProperty("clDocrime", True)

# %% SOLVE CL DOCRIME
loop.run("solveDocrime")

# %% CL DOCRIME Stop
loop.setProperty("clDocrime", False)
#%%
psfCam.setProperty("integrationLength", 3000)
time.sleep(2)
loopOptim.adjustParam("gain", 0.1, 0.9)
loopOptim.adjustParam("numDroppedModes", 0, 20, int)
loopOptim.adjustParam("leakyGain", 0, 0.05)
loopOptim.adjustParam("alpha", 0, 1.0)
time.sleep(1)
loopOptim.resetStudy()
time.sleep(5)
loopOptim.numReads = 2
loopOptim.numSteps = 30
for i in range(1):
    loopOptim.optimize()
loopOptim.applyOptimum()
#%% Optimize PID
pidOptim.numReads = 20
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
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
numOptim = 3
maxAMP = 0.004
amps = np.linspace(maxAMP, maxAMP/5, numOptim)
for i in range(numOptim):
    ncpaOptim.resetStudy()
    psfCam.setProperty("integrationLength", 5)
    time.sleep(2)
    ncpaOptim.numReads = 3
    ncpaOptim.startMode = 0
    ncpaOptim.endMode = 50 #wfc.getProperty("numModes")
    ncpaOptim.numSteps = 2000
    ncpaOptim.correctionMag = amps[i]
    ncpaOptim.isCL = False
    for i in range(1):
        ncpaOptim.optimize()
    ncpaOptim.applyNext()

    wfc.run("saveShape")
    slopes.run("takeRefSlopes")
    slopes.setProperty("refSlopesFile", "/home/whetstone/pyRTC/SHARP_LAB/calib/refPyWFS.npy")
    slopes.run("saveRefSlopes")
    psfCam.setProperty("integrationLength", 500)
    time.sleep(2)
    psfCam.run("takeModelPSF")
    psfCam.setProperty("modelFile", "/home/whetstone/pyRTC/SHARP_LAB/calib/modelPSF_PyWFS.npy")
    psfCam.run("saveModelPSF")
    wfc.run("loadFlat")
    
# %%
"""
Update FLAT/REF/MODEL after NCPA
"""
# #New Flat
# wfc.run("saveShape")
# #New PSF Model
# psfCam.run("takeModelPSF")
# psfCam.setProperty("modelFile", folder + "modelPSF_PyWFS.npy")
# psfCam.run("saveModelPSF")
# #New Reference Slopes
# slopes.run("takeRefSlopes")
# slopes.setProperty("refSlopesFile", folder + "refPyWFS.npy")
# slopes.run("saveRefSlopes")
# %%
# loop.run("solveDocrime")
baseline = np.load(folder + "IM_PYWFS.npy")
OLDOCRIME = np.load(folder + "IM_PYWFS_OL_docrime.npy")
CLDOCRIME = np.load(folder + "IM_PYWFS_OL_docrime_CL_docrime.npy")
# CLDOCRIME = np.load(dataFolder + "ESCAPE_CL_docrime.npy")
ESCAPE = np.load(folder + "ESCAPE_PYWFS.npy")
im = OLDOCRIME
plt.imshow(im, aspect="auto")
plt.show()

a = np.std(baseline,axis = (0))
b = np.std(OLDOCRIME,axis = (0))
c = np.std(CLDOCRIME,axis = (0))
# a,b,c = np.std(baseline,axis = (0)), np.std(OLDOCRIME,axis = (0)), np.std(CLDOCRIME,axis = (0))
d = np.std(ESCAPE,axis = (0))

plt.plot(a, label = 'baseline')
plt.plot(b, label = 'OL DOCRIME')
plt.plot(c, label = 'CL DOCRIME')
plt.plot(d, label = 'ESCAPE')
plt.legend()
plt.show()

plt.title("Optical Gains (Relative to OL DOCRIME)")
plt.plot(c/b, label = 'CL DOCRIME')
plt.legend()
plt.show()

vsa = np.load(folder + "validSubApsPyWFS.npy")

imFull = np.zeros((im.shape[-1],*vsa.shape))
def im_col_to_mode(col, vsa):
    curSignal2D = np.zeros(vsa.shape)
    slopemask = vsa[:,:vsa.shape[1]//2]
    curSignal2D[:,:vsa.shape[1]//2][slopemask] = col[:col.size//2]
    curSignal2D[:,vsa.shape[1]//2:][slopemask] = col[col.size//2:]
    return curSignal2D

for i in range(imFull.shape[0]):
    imFull[i] = im_col_to_mode(im[:,i], vsa)

N = 0
for i in range(N,N + 5):
    plt.imshow(imFull[i])
    plt.colorbar()
    plt.show()
# %%

dataFolder =  "/home/whetstone/pyRTC/SHARP_LAB/calib/" # "/home/whetstone/pyRTC/SHARP_LAB/data/PYWFS_ESCAPE_DATA/"
loop.run("solveDocrime")
time.sleep(0.1)

# Save the BASELINE IM, NO OG
imBaseline = np.load(folder + "IM_PYWFS.npy")
np.save(dataFolder + "BASELINE_NO_OG", imBaseline)

# Save the DOCRIME IM, NO OG
imDOCRIME = np.load(folder + "IM_PYWFS_OL_docrime.npy")
np.save(dataFolder + "DOCRIME_NO_OG", imDOCRIME)

# Save the CL DOCRIME IM
imCLDOCRIME = np.load(folder + "IM_PYWFS_OL_docrime_CL_docrime.npy")
np.save(dataFolder + "CL_DOCRIME", imCLDOCRIME)

# Save the CL DOCRIME IM
imESCAPE = np.load(folder + "sprint_IM_original_pywfs.npy")

a = np.std(imBaseline,axis = (0))
b = np.std(imDOCRIME,axis = (0))
c = np.std(imCLDOCRIME,axis = (0))
d = np.std(imESCAPE,axis = (0))

plt.plot(a)
plt.plot(b)
plt.plot(c)
# plt.plot(d)
plt.show()

ogBaseline = c/a

imBaselineWithOG = np.copy(imBaseline)
for i in range(imBaseline.shape[1]):
    imBaselineWithOG[:,i] *= ogBaseline[i]

ogDOCRIME = c/b

imDOCRIMEWithOG = np.copy(imDOCRIME)
for i in range(imBaseline.shape[1]):
    imDOCRIMEWithOG[:,i] *= ogDOCRIME[i]

ogESCAPE = c/d

imESCAPEWithOG = np.copy(imESCAPE)
for i in range(imBaseline.shape[1]):
    imESCAPEWithOG[:,i] *= ogESCAPE[i]

fig, ax1 = plt.subplots()

# Plot the first set of data on the
color = 'tab:red'
ax1.set_xlabel('MODE')
ax1.set_ylabel('OG BASELINE', color=color)
ax1.plot(ogBaseline, color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Create a second y-axis that shares the same x-axis
ax2 = ax1.twinx()

# Plot the second set of data on the second axis
color = 'tab:blue'
ax2.set_ylabel('OG DOCRIME', color=color)
ax2.plot(ogDOCRIME, color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.show()

np.save(dataFolder + "BASELINE_WITH_OG", imBaselineWithOG)
np.save(dataFolder + "DOCRIME_WITH_OG", imDOCRIMEWithOG)
np.save(dataFolder + "ESCAPE", imESCAPEWithOG)

a = np.std(imBaselineWithOG,axis = (0))
b = np.std(imDOCRIMEWithOG,axis = (0))
c = np.std(imCLDOCRIME,axis = (0))

plt.plot(a)
plt.plot(b)
plt.plot(c)
plt.show()

# %% Sweep
import wandb

def stopLoop():
    loop.run("stop")
    time.sleep(0.1)
    for i in range(5):
        wfc.run("flatten")
        time.sleep(0.05)

def getNextStrehl(strehlShm, wfc2Dshm, maxCommand=0.5, strehlToAverage=10):

    start_strehl = strehlShm.read_noblock()
    next_strehl = start_strehl
    mean_strehl = 0
    count = 0

    while count < strehlToAverage:
        curr_wfc = wfc2Dshm.read_noblock()
        if np.any(curr_wfc > maxCommand):

            stopLoop()
            wandb.log({"strehl": np.nan})
            wandb.finish()
            return -1

        next_strehl = strehlShm.read_noblock()
        if next_strehl != start_strehl:
            start_strehl = next_strehl
            mean_strehl += next_strehl
            count += 1

    return mean_strehl/strehlToAverage

def wandfunction():

    strehlShm, _, _ = initExistingShm("strehl")
    wfc2Dshm, _, _ = initExistingShm("wfc2D")

    run = wandb.init()
    wconfig = wandb.config

    # Adjust Loop
    loop.setProperty("IMFile", 
                     f"/home/whetstone/pyRTC/SHARP_LAB/data/PYWFS_ESCAPE_DATA/{wconfig.im_file}.npy")
    loop.setProperty("numDroppedModes", wconfig.num_dropped_modes)
    loop.setProperty("gain", wconfig.loop_gain)
    loop.setProperty("leakyGain", wconfig.leaky_gain)
    loop.run("loadIM")
    time.sleep(1)

    # Launch Loop for 5 seconds
    wfc.run("flatten")
    time.sleep(0.1)
    loop.run("start")

    #Burn in time
    time.sleep(2)

    SR = getNextStrehl(strehlShm, wfc2Dshm, maxCommand=0.8, strehlToAverage=20)

    if SR == -1:
        return

    # Stop Loop
    stopLoop()

    wandb.log({"strehl": SR})

    wandb.finish()

    return

sweep_config = {
    "name": f'escape_sweep_pywfs',
    "method": 'random', 
    "metric": {"name": 'strehl', "goal": "maximize"},
    "parameters": {
        "leaky_gain": {"values": np.linspace(start=0, stop=3e-2, num=20).tolist()},
        "loop_gain": {"values": np.linspace(start=0.05, stop=0.8, num=20).tolist()},
        "num_dropped_modes": {"values": np.linspace(start=0, stop=40, num=30, dtype=int).tolist()},
        "im_file": {"values": ["BASELINE_NO_OG", "BASELINE_WITH_OG",
                               "DOCRIME_NO_OG", "DOCRIME_WITH_OG",
                               "ESCAPE"]}
    }
}

sweep_id = wandb.sweep(sweep_config, project="ESCAPE-PYWFS")
wandb.agent(sweep_id, wandfunction, count=2000)
wandb.teardown()
# %%
# %% Save long PSF
##CHANGE THIS LINE
filename = "DOCRIME_NO_OG_PSF"
numReads = 10
folder = "/home/whetstone/pyRTC/SHARP_LAB/data/PYWFS_ESCAPE_DATA/"
filename = folder + filename

psfShm, _, _ = initExistingShm("psfLong")
strehlShm, _, _ = initExistingShm("strehl")
mean_psf = np.zeros_like(psfShm.read_noblock())
mean_strehl = 0

for i in range(numReads):
    mean_psf += psfShm.read()
    mean_strehl += strehlShm.read()[0]

mean_psf /= numReads
mean_strehl /= numReads
mean_strehl = np.round(mean_strehl, 2)
# filename += str(mean_strehl)

np.save(filename,mean_psf)

plt.imshow(mean_psf)
plt.title(f"SR: {mean_strehl}")
plt.show()
# %% Measure Jitter
wfsShm, _, _ = initExistingShm("wfsRaw")
wfcShm, _, _ = initExistingShm("wfc")
# %% Measure Jitter
N = 1000000
imgLatency = np.empty(N)
sysLatency = np.empty(N)
lastImageTime = wfsShm.metadata[1]
for i in range(N):
    #Wait for new image
    newImageTime = wfsShm.metadata[1]
    while newImageTime == lastImageTime:
        newImageTime = wfsShm.metadata[1]
    currentCorrectionTime = wfcShm.metadata[1]
    imgLatency[i] = newImageTime-lastImageTime
    sysLatency[i] = currentCorrectionTime-lastImageTime
    lastImageTime = wfsShm.metadata[1]

# %%
# Create histogram plot
plt.figure(figsize=(10, 6))
plt.hist(sysLatency, 
         bins=np.logspace(-4,-2.5,200), 
         log=True, 
         color = 'k', 
         histtype='step', 
         density=False )
x = np.percentile(sysLatency, 99)
plt.axvline(x = x,
             color = 'green', 
             label = f'1 in 100 > {1e6*x:.0f}us')
x = np.percentile(sysLatency, 99.9)
plt.axvline(x = x,
             color = 'orange',
             label = f'1 in 1,000 > {1e6*x:.0f}us')
x = np.percentile(sysLatency, 99.99)
plt.axvline(x = x,
             color = 'red',
             label = f'1 in 10,000 > {1e6*x:.0f}us')
# Set log scale for both axes
plt.xscale('log')
plt.yscale('log')

# Set xticks with custom labels
xticks = [1e-4, 1e-3]#, 1e-2]
xtick_labels = ['100µs', '1ms']#, '10ms']
plt.xticks(xticks, xtick_labels)

plt.xlabel('System Latency', size = 16)
plt.ylabel('Counts', size = 16)
plt.title('pyRTC Latency (64x64 PyWFS, 11x11 DM)', size = 18)
plt.ylim(1,N/10)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("jitter_pywfs.png")
plt.show()
# %%
wfcShm = initExistingShm("wfc")[0]
slopesShm = initExistingShm("signal")[0]
IM = np.load(loop.getProperty("IMFile"))
CM = np.linalg.pinv(IM)
N = 10000
OL_shapes = np.empty((N,*wfcShm.read_noblock().shape))
CL_shapes = np.zeros_like(OL_shapes)
loop.run("stop")
loop.run("flatten")
time.sleep(1)
for i in range(N):
    # OL_shapes[i] = wfcShm.read()
    # CL_shapes[i] = CM@slopesShm.read()
    OL_shapes[i] = CM@slopesShm.read()

loop.run("start")
time.sleep(1)
for i in range(N):
    # OL_shapes[i] = wfcShm.read()
    CL_shapes[i] = CM@slopesShm.read()
    # OL_shapes[i] = CM@slopesShm.read()
# %%
plt.plot(np.mean(OL_shapes**2,axis = 0), label = 'CM@Slopes^2 (OL)')
plt.plot(np.mean(CL_shapes**2,axis = 0), label = 'CM@Slopes^2 (CL)')
plt.yscale('log')
plt.xscale('log')
# plt.ylim(1e-5,1e-2)
plt.plot(5e-4*np.arange(95)**(-11/8))
plt.plot(8e-6*np.arange(95)**(-11/8))
plt.legend()
plt.show()


plt.plot( 10*np.log10(np.mean(CL_shapes**2,axis = 0) / np.mean(OL_shapes**2,axis = 0)))
# plt.yscale('log')
plt.xscale('log')
plt.ylabel("Attenuation (dB)")
plt.legend()
plt.show()

# %%
strehlShm = initExistingShm("strehl")[0]
wfc2DShm = initExistingShm("wfc2D")[0]
def recordStrehl(N=10):
    vals = np.zeros(N)
    diverge = False
    for j in range(N):
        if diverge:
            return -1
        vals[j] = np.mean(strehlShm.read())
        diverge = np.max(wfc2DShm.read_noblock()) > 0.6
    return np.mean(vals), np.std(vals)
volts = []
SRs = []
SRs_err = []
#%%
volts.append(20)
a, b = recordStrehl(N = 5)
SRs.append(a)
SRs_err.append(b)

# %%
T = wfc.getProperty("frameDelay")
np.save(f"/home/whetstone/servoLagTest/volts_delay_{T}", np.array(volts))
np.save(f"/home/whetstone/servoLagTest/sr_delay_{T}", np.array(SRs))
np.save(f"/home/whetstone/servoLagTest/srErr_delay_{T}", np.array(SRs_err))
# %%
