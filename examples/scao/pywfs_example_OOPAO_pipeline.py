
# %% IMPORTS
#Import pyRTC classes
from pyRTC.Pipeline import *
from pyRTC.utils import *

RECALIBRATE = False
hardware_folder = "../../pyRTC/hardware/"
# %% IMPORTS
config = './pywfs_OOPAO_config.yaml'
N = np.random.randint(3000,6000)
# %% Launch sim, this is WFS, DM and PSF Camera
sim = hardwareLauncher(hardware_folder+"OOPAOInterface.py", config, N)
sim.launch()

# %% Launch slopes
slopes = hardwareLauncher(hardware_folder+"SlopesProcess.py", config, N+1)
slopes.launch()

# %% Launch Loop Class
loop = hardwareLauncher(hardware_folder+"./Loop.py", config, N+2)
loop.launch()

# %% Calibrate
if RECALIBRATE == True:

    loop.setProperty("IMFile", "./pwfs_example_IM.npy")
    sim.run("flatten")
    loop.run("computeIM")
    loop.run("saveIM")
    wfc.run("flatten")
    time.sleep(1)

# %%
loop.setProperty("numDroppedModes", 20)
loop.run("computeCM")

# %%Launch Loop for 5 seconds
loop.run("setGain",0.01)
wfc.run("flatten")
loop.run("start")
time.sleep(5)
# %% Stop Loop
loop.run("stop")
wfc.run("flatten")
wfc.run("flatten")
# %% Kill e89verything
hardware = [slopes, psfCam, wfs, wfc, loop]
for h in hardware:
    h.shutdown()
    time.sleep(1)

# %%
