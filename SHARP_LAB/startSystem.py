# %% IMPORTS
# Import pyRTC classes
from pyRTC.Pipeline import *
from pyRTC.utils import *
from pyRTC.hardware import *
import os

os.chdir("/home/whetstone/pyRTC/SHARP_LAB")

# %% IMPORTS
# config = '/home/whetstone/pyRTC/SHARP_LAB/config_SR.yaml'
config = "/home/whetstone/pyRTC/SHARP_LAB/config_predict.yaml"
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

# %% Launch Loop Class
# loop = hardwareLauncher("../pyRTC/Loop.py", config, N+4)
loop = hardwareLauncher(
    "/home/whetstone/pyRTC/SHARP_LAB/prediction/predictLoop.py", config, N + 4
)
loop.launch()

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
wfc.run("flatten")

while True:
    time.sleep(3)

# %%
