# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import time
from pyRTC import *
from pyRTC.hardware import *
from pyRTC.utils import *
from pyRTC.Pipeline import *
#%% CLEAR SHMs
# shms = ["wfs= "wfsRaw= "signal= "signal2D= "wfc= "wfc2D= "psfShort= "psfLong"]
# clear_shms(shms)
# %% Load Config
conf = read_yaml_file("/home/whetstone/pyRTC/SHARP_LAB/config_pywfs.yaml")
# %% Launch WFS
confWFS = conf["wfs"]
wfs = XIMEA_WFS(conf=confWFS)
time.sleep(0.5)
wfs.start()
# %% Launch Modulator (PyWFS)
confMod = conf["modulator"]
mod = PIModulator(conf=confMod)
time.sleep(0.5)
mod.start()

# %% Launch slopes
slopes = SlopesProcess(conf=conf["slopes"])
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
loop = Loop(conf=conf["loop"])
time.sleep(1)
# %%
from pyRTC.hardware.NCPAOptimizer import NCPAOptimizer
ncpaOptim = NCPAOptimizer(conf["optimizer"]["ncpa"], loop, slopes)

# %% Recalibrate

# Darks
if False:
    input("Sources Off?")
    wfs.takeDark()
    wfs.darkFile = "/home/whetstone/pyRTC/SHARP_LAB/calib/darkPyWFS.npy"
    wfs.saveDark()
    time.sleep(1)
    psf.takeDark()
    psf.darkFile = "/home/whetstone/pyRTC/SHARP_LAB/calib/psfDark.npy"
    psf.saveDark()

    input("Sources On?")
    input("Is Atmosphere Out?")
    wfc.flatten()
    psf.takeModelPSF()
    psf.modelFile = "/home/whetstone/pyRTC/SHARP_LAB/calib/modelPSF.npy"
    psf.saveModelPSF()


    slopes.takeRefSlopes()
    slopes.refSlopesFile = "/home/whetstone/pyRTC/SHARP_LAB/calib/refPyWFS.npy"
    slopes.saveRefSlopes()

    #  STANDARD IM
    loop.IMMethod = "push-pull"
    loop.pokeAmp = 0.03
    loop.numItersIM = 100
    loop.IMFile = "/home/whetstone/pyRTC/SHARP_LAB/calib/IM_PyWFS.npy"
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
    loop.IMFile = "/home/whetstone/pyRTC/SHARP_LAB/calib/docrime_IM.npy"
    wfc.flatten()
    loop.computeIM()
    loop.saveIM()
    wfc.flatten()
    time.sleep(1)

# %% Compute CM
loop.IMFile = "/home/whetstone/pyRTC/SHARP_LAB/calib/IM_PyWFS.npy"
loop.numDroppedModes = 10
loop.gain = 0.1
loop.leakyGain = 0.01
loop.loadIM()
# %% Start Loop
wfc.flatten()
time.sleep(0.3)
loop.start()

# %% Stop Loop
loop.stop()
wfc.flatten()
time.sleep(0.3)

wfc.flatten()

#%% Optimize NCPA
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
numOptim = 3
maxAMP = 0.005
amps = np.linspace(maxAMP, maxAMP/5, numOptim)
for i in range(numOptim):
    ncpaOptim.resetStudy()
    psf.integrationLength= 5
    time.sleep(2)
    ncpaOptim.numReads = 3
    ncpaOptim.startMode = 0
    ncpaOptim.endMode = 15 #wfc.getProperty("numModes")
    ncpaOptim.numSteps = 1000
    ncpaOptim.correctionMag = amps[i]
    ncpaOptim.isCL = False
    for i in range(1):
        ncpaOptim.optimize()
    ncpaOptim.applyNext()

    wfc.saveShape()
    # slopes.takeRefSlopes()
    # slopes.refSlopesFile= "/home/whetstone/pyRTC/SHARP_LAB/calib/ref.npy")
    # slopes.saveRefSlopes()
    psf.integrationLength= 2000
    time.sleep(2)
    psf.takeModelPSF()
    psf.modelFile= "/home/whetstone/pyRTC/SHARP_LAB/calib/modelPSF_PyWFS.npy"
    psf.saveModelPSF()
    wfc.loadFlat()


#%%

def find_threshold_average(values, threshold):
    first_index = None
    last_index = None
    
    # Loop through the list to find the first index
    for i, value in enumerate(values):
        if value > threshold:
            first_index = i
            break
    
    # Loop through the list in reverse to find the last index
    for i in range(len(values) - 1, -1, -1):
        if values[i] > threshold:
            last_index = i
            break
    
    # Check if the indices were found
    if first_index is None or last_index is None:
        return None
    
    # Calculate the average of the two indices
    average_index = (first_index + last_index) / 2.0
    return average_index

im = wfs.read()

theshold = 6500

q1 = im[:im.shape[0]//2, im.shape[1]//2:]
plt.imshow(q1)
plt.show()

a = np.sum(q1, axis = 0)
b = np.sum(q1, axis = 1)

plt.plot(a)
plt.plot(b)
indexA = find_threshold_average(a,theshold)
plt.axvline(x=indexA, color = 'r')
indexB = find_threshold_average(b,theshold)
plt.axvline(x=indexB, color = 'r')
plt.show()
print(indexA, indexB)
print(indexA+ im.shape[0]//2, indexB+im.shape[0]//2)
# %%
