# %% IMPORTS
# Import pyRTC classes
from pyRTC.Pipeline import *
from pyRTC.utils import *
import wandb
import os

os.environ["WANDB_SILENT"] = "true"
RECALIBRATE = False

# %% Clear SHMs
# from pyRTC.Pipeline import clear_shms
# shm_names = ["wfs", "wfsRaw", "wfc", "wfc2D", "signal", "signal2D", "psfShort", "psfLong"] #list of SHMs to reset
# clear_shms(shm_names)

# %% IMPORTS
config = "/home/whetstone/pyRTC/SHARP_LAB/config.yaml"
N = np.random.randint(3000, 6000)
# %% Launch DM
wfc = hardwareLauncher("/home/whetstone/pyRTC/pyRTC/hardware/ALPAODM.py", config, N)
wfc.launch()

# %% Launch WFS
wfs = hardwareLauncher(
    "/home/whetstone/pyRTC/pyRTC/hardware/ximeaWFS.py", config, N + 1
)
wfs.launch()

# %% Launch slopes
slopes = hardwareLauncher("/home/whetstone/pyRTC/pyRTC/SlopesProcess.py", config, N + 2)
slopes.launch()

# %% Launch PSF Cam
psfCam = hardwareLauncher(
    "/home/whetstone/pyRTC/pyRTC/hardware/SpinnakerScienceCam.py", config, N + 10
)
psfCam.launch()

# %% Launch Loop Class
loop = hardwareLauncher("/home/whetstone/pyRTC/pyRTC/Loop.py", config, N + 4)
# loop = hardwareLauncher("./hardware/predictLoop.py", config)
loop.launch()

# %% Calibrate

if RECALIBRATE == True:

    slopes.setProperty("refSlopesFile", "")
    slopes.run("loadRefSlopes")
    ##### slopes.setProperty("offsetY", 3)

    input("Sources Off?")
    wfs.run("takeDark")
    wfs.setProperty("darkFile", "/home/whetstone/pyRTC/SHARP_LAB/dark.npy")
    wfs.run("saveDark")
    time.sleep(1)
    psfCam.run("takeDark")
    psfCam.setProperty("darkFile", "/home/whetstone/pyRTC/SHARP_LAB/psfDark.npy")
    psfCam.run("saveDark")
    input("Sources On?")
    input("Is Atmosphere Out?")

    slopes.run("computeImageNoise")
    slopes.run("takeRefSlopes")
    slopes.setProperty("refSlopesFile", "/home/whetstone/pyRTC/SHARP_LAB/ref.npy")
    slopes.run("saveRefSlopes")

    wfc.run("flatten")
    psfCam.run("takeModelPSF")
    psfCam.setProperty("modelFile", "/home/whetstone/pyRTC/SHARP_LAB/modelPSF.npy")
    psfCam.run("saveModelPSF")

    #  STANDARD IM
    loop.setProperty("IMMethod", "push-pull")
    loop.setProperty("pokeAmp", 0.03)
    loop.setProperty("numItersIM", 100)
    loop.setProperty("IMFile", "/home/whetstone/pyRTC/SHARP_LAB/IM.npy")
    wfc.run("flatten")
    loop.run("computeIM")
    loop.run("saveIM")
    wfc.run("flatten")
    time.sleep(1)

    input("Is Atmosphere In?")
    #  DOCRIME OL
    loop.setProperty("IMMethod", "docrime")
    loop.setProperty("delay", 1)
    loop.setProperty("pokeAmp", 2e-2)
    loop.setProperty("numItersIM", 100000)
    loop.setProperty("IMFile", "/home/whetstone/pyRTC/SHARP_LAB/docrime_IM.npy")
    wfc.run("flatten")
    loop.run("computeIM")
    loop.run("saveIM")
    wfc.run("flatten")
    time.sleep(1)

# %%
"""
RESCALES SPRINT TO MATCH EMPIRICAL
"""
# im = np.load("../SHARP_LAB/IM.npy")
# # im_sprint = np.load("../SHARP_LAB/sprint_IM_original.npy")
# # im_sprint *= np.std(im)/np.std(im_sprint)
# # np.save("../SHARP_LAB/sprint_IM_original_scaled", im_sprint)
# im_sprint = np.load("../SHARP_LAB/sprint_IM_original_scaled.npy")
# plt.plot(np.std(im_sprint, axis = 0), label = 'SPRINT_ORG')
# for i in range(im_sprint.shape[1]):
#     im_sprint[:,i] *= np.std(im[:,i])/np.std(im_sprint[:,i])
# np.save("../SHARP_LAB/sprint_IM.npy", im_sprint)

# plt.plot(np.std(im, axis = 0), label = 'EMPIRICAL')
# plt.plot(np.std(im_sprint, axis = 0), label = 'SPRINT')
# plt.xlabel("Mode #", size = 18)
# plt.ylabel("Standard Deviation", size = 18)
# plt.legend()
# plt.show()
signalSize = int(np.sum(np.load(slopes.getProperty("validSubApsFile"))))
wfcshm = ImageSHM("wfc2D", (11, 11), np.float32)
slopeshm = ImageSHM("signal", (signalSize, 1), np.float32)


# %%
def wandfunction():

    run = wandb.init()
    wconfig = wandb.config

    # Adjust Loop
    loop.setProperty("IMFile", f"/home/whetstone/pyRTC/SHARP_LAB/{wconfig.im_file}.npy")
    loop.run("loadIM")
    time.sleep(0.5)
    loop.setProperty("numDroppedModes", wconfig.num_dropped_modes)
    loop.run("computeCM")
    time.sleep(0.5)
    loop.run("setGain", wconfig.loop_gain)
    loop.setProperty("leakyGain", wconfig.leaky_gain)
    # wfc.run("reactivateActuators",[i for i in range(97)])
    # if wconfig.floating_edge:
    #     wfc.run("deactivateActuators",[0,1,2,3,4,5,11,12,20,21,31,32,42,43,53,54,64,65,75,76,84,85,91,92,93,94,95,96])
    # else:
    #     wfc.run("reactivateActuators",[i for i in range(97)])

    # Launch Loop for 5 seconds
    wfc.run("flatten")
    time.sleep(0.1)
    loop.run("start")

    start_strehl = psfCam.getProperty("strehl_ratio")
    next_strehl = start_strehl
    count = 0
    slopes_mean = []
    slopes_std = []

    while count < 2:
        curr_wfc = wfcshm.read_noblock()
        if np.any(curr_wfc > 1.0):
            loop.run("stop")
            time.sleep(0.5)
            wfc.run("flatten")
            time.sleep(0.1)
            wfc.run("flatten")

            wandb.log({"strehl": np.nan, "slope_mean": np.nan, "slope_std": np.nan})
            wandb.finish()
            return
        time.sleep(0.1)
        cur_slope = slopeshm.read_noblock()
        slopes_mean.append(np.mean(cur_slope))
        slopes_std.append(np.mean(cur_slope))
        next_strehl = psfCam.getProperty("strehl_ratio")
        if next_strehl != start_strehl:
            start_strehl = next_strehl
            count += 1

    # Stop Loop
    loop.run("stop")
    time.sleep(0.1)
    wfc.run("flatten")
    wfc.run("flatten")

    wandb.log(
        {
            "strehl": next_strehl,
            "slope_mean": np.mean(np.array(slopes_mean)),
            "slope_std": np.mean(np.array(slopes_std)),
        }
    )

    wandb.finish()


sweep_config = {
    "name": f"sharp_gain_sweep_plateD",
    "method": "random",
    "metric": {"name": "strehl", "goal": "maximize"},
    "parameters": {
        "leaky_gain": {"values": np.linspace(start=0, stop=3e-2, num=25).tolist()},
        "loop_gain": {"values": np.linspace(start=0.1, stop=0.45, num=25).tolist()},
        "num_dropped_modes": {
            "values": np.linspace(start=0, stop=80, num=25, dtype=int).tolist()
        },
        "im_file": {"values": ["sprint_IM", "IM"]},
    },
}

sweep_id = wandb.sweep(sweep_config, project="sharp-bench")
wandb.agent(sweep_id, wandfunction, count=2000)
wandb.teardown()

# %%
