# %% IMPORTS
# Import pyRTC classes
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
config = "/home/whetstone/pyRTC/SHARP_LAB/config.yaml"
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
psfCam = hardwareLauncher("../pyRTC/hardware/SpinnakerScienceCam.py", config, N + 10)
psfCam.launch()

# %% Launch Loop Class
loop = hardwareLauncher("../pyRTC/Loop.py", config, N + 4)
# loop = hardwareLauncher("../pyRTC/hardware/predictLoop.py", config)
loop.launch()

# %% OPTIMIZERS
from pyRTC.hardware.PIDOptimizer import PIDOptimizer

pidOptim = PIDOptimizer(read_yaml_file(config)["optimizer"]["pid"], loop)
# %%
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
    psfCam.setProperty(
        "modelFile", "/home/whetstone/pyRTC/SHARP_LAB/calib/modelPSF.npy"
    )
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

    input("Is Atmosphere In?")
    #  DOCRIME OL
    loop.setProperty("IMMethod", "docrime")
    loop.setProperty("delay", 2)  # Needs to be set in the CONFIG
    loop.setProperty("pokeAmp", 1e-2)
    loop.setProperty("numItersIM", 1000)
    loop.setProperty(
        "IMFile", "/home/whetstone/pyRTC/SHARP_LAB/calib/IM_OL_docrime.npy"
    )
    wfc.run("flatten")
    loop.run("computeIM")
    loop.run("saveIM")
    wfc.run("flatten")
    time.sleep(1)


# %% Compute CM
# loop.setProperty("IMFile", "/home/whetstone/pyRTC/SHARP_LAB/calib/IM.npy")
loop.setProperty("IMFile", "/home/whetstone/pyRTC/SHARP_LAB/calib/IM_OL_docrime.npy")
loop.run("loadIM")
time.sleep(0.5)
loop.setProperty("numDroppedModes", 25)
loop.run("computeCM")
time.sleep(0.5)

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

    return mean_strehl / strehlToAverage


def wandfunction():

    strehlShm, _, _ = initExistingShm("strehl")
    wfc2Dshm, _, _ = initExistingShm("wfc2D")

    run = wandb.init()
    wconfig = wandb.config

    # Adjust Loop
    loop.setProperty(
        "IMFile", f"/home/whetstone/pyRTC/SHARP_LAB/calib/{wconfig.im_file}.npy"
    )
    loop.setProperty("numDroppedModes", wconfig.num_dropped_modes)
    loop.setProperty("gain", wconfig.loop_gain)
    loop.setProperty("leakyGain", wconfig.leaky_gain)
    loop.run("loadIM")
    time.sleep(1)

    # Launch Loop for 5 seconds
    wfc.run("flatten")
    time.sleep(0.1)
    loop.run("start")

    # Burn in time
    time.sleep(2)

    SR = getNextStrehl(strehlShm, wfc2Dshm, maxCommand=0.8, strehlToAverage=10)

    if SR == -1:
        return

    # Stop Loop
    stopLoop()

    wandb.log({"strehl": SR})

    wandb.finish()

    return


sweep_config = {
    "name": f"escape_sweep_shwfs",
    "method": "random",
    "metric": {"name": "strehl", "goal": "maximize"},
    "parameters": {
        "leaky_gain": {"values": np.linspace(start=0, stop=3e-2, num=20).tolist()},
        "loop_gain": {"values": np.linspace(start=0.05, stop=0.5, num=20).tolist()},
        "num_dropped_modes": {
            "values": np.linspace(start=0, stop=40, num=30, dtype=int).tolist()
        },
        "im_file": {"values": ["baseline", "OL_DOCRIME", "ESCAPE"]},
    },
}

sweep_id = wandb.sweep(sweep_config, project="ESCAPE-SHWFS")
wandb.agent(sweep_id, wandfunction, count=1000)
wandb.teardown()

# %%
