# %% IMPORTS
#Import pyRTC classes
from pyRTC.Pipeline import *
from pyRTC.utils import *

RECALIBRATE = False

# %% IMPORTS
config = '/home/whetstone/pyRTC/REVOLT/config.yaml'
N = np.random.randint(3000,6000)
# %% Launch DM
wfc = hardwareLauncher("./hardware/ALPAODM.py", config, N)
wfc.launch()

# %% Launch WFS
wfs = hardwareLauncher("./hardware/ximeaWFS.py", config, N+1)
wfs.launch()

# %% Launch slopes
slopes = hardwareLauncher("./SlopesProcess.py", config, N+2)
slopes.launch()

# %% Launch PSF Cam
psfCam = hardwareLauncher("./hardware/SpinnakerScienceCam.py", config, N+3)
psfCam.launch()

# %% Launch Loop Class
loop = hardwareLauncher("./Loop.py", config, N+4)
# loop = hardwareLauncher("./hardware/predictLoop.py", config)
loop.launch()

# %% Calibrate

if RECALIBRATE == True:

    slopes.setProperty("refSlopesFile", "")
    slopes.run("loadRefSlopes")
    ##### slopes.setProperty("offsetY", 3)

    input("Sources Off?")
    psfCam.run("takeDark")
    psfCam.run("saveDark")
    wfs.run("takeDark")
    wfs.run("saveDark")
    input("Sources On?")

    slopes.run("takeRefSlopes")
    slopes.setProperty("refSlopesFile", "/home/whetstone/pyRTC/SHARP_LAB/ref.npy")
    slopes.run("saveRefSlopes")

    loop.setProperty("pokeAmp", 0.03)
    loop.setProperty("numItersIM", 100)
    loop.setProperty("IMFile", "/home/whetstone/pyRTC/SHARP_LAB/IM.npy")
    wfc.run("flatten")
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
