# %% IMPORTS
# Import pyRTC classes
from pyRTC.Pipeline import *
from pyRTC.utils import *
from pyRTC.hardware import *
import os

os.chdir("/home/whetstone/pyRTC_backup/SHARP_LAB")
RECALIBRATE = False
CLEAR_SHMS = False
# %% Clear SHMs
if CLEAR_SHMS:
    from pyRTC.Pipeline import clear_shms

    shm_names = [
        "wfs",
        "wfsRaw",
        "wfc",
        "wfc2D",
        "signal",
        "signal2D",
        "psfShort",
        "psfLong",
    ]  # list of SHMs to reset
    clear_shms(shm_names)

# %% IMPORTS
# config = '/home/whetstone/pyRTC/SHARP_LAB/config_SR.yaml'
config = "/home/whetstone/pyRTC_backup/SHARP_LAB/config_pywfs_2026.yaml"
N = np.random.randint(3000, 6000)
# %% Launch DM
wfc = hardwareLauncher("../pyRTC/hardware/ALPAODM.py", config, N)
wfc.launch()

# %% Launch WFS
wfs = hardwareLauncher("../pyRTC/hardware/ximeaWFS.py", config, N + 1)
wfs.launch()

# %% Launch slopes
slopes = hardwareLauncher("../pyRTC/SlopesProcess.py", config, N + 2)
slopes.launch()

# %% Launch PSF Cam
psf = hardwareLauncher("../pyRTC/hardware/SpinnakerScienceCam.py", config, N + 10)
psf.launch()

# %% Launch Modulator
modulator = hardwareLauncher("../pyRTC/hardware/PIModulator.py", config, N + 100)
modulator.launch()

# %% Launch Loop Class
loop = hardwareLauncher("../pyRTC/Loop.py", config, N + 4)
# loop = hardwareLauncher("../pyRTC/hardware/predictLoop.py", config)
loop.launch()

time.sleep(3)

# %%Launch Loop for 5 seconds ###########################################################
wfc.run("flatten")
wfc.run("flatten")
wfc.run("flatten")
wfc.run("flatten")
loop.run("start")
time.sleep(100)

wfc.run("flatten")
wfc.run("flatten")
wfc.run("flatten")
time.sleep(1)

while True:
    time.sleep(3)
