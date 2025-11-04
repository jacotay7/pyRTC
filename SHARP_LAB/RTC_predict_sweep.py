# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from pyRTC import *
from pyRTC.hardware import *
from pyRTC.utils import *
import wandb
from dotmap import DotMap
from pyRTC.Pipeline import initExistingShm, clear_shms
import subprocess
import time
from tqdm import tqdm, trange

# logging.disable(logging.ERROR)
from prediction.prepidopt import PrePIDOptimizer

# from pyRTC.hardware import PIDOptimizer as PrePIDOptimizer
wandb.require("core")
import os

os.environ["WANDB_DIR"] = os.path.abspath("/media/whetstone/storage2/data/wandb")

dataDir = "/media/whetstone/storage/pyRTCTelem"
uniqueStr = "Nov20tests"

SAVE_TO_DIR = False
DRY_RUN = False
SOFT_MODE = True
N_LOOP = 51

# %%
N = np.random.randint(3000, 6000)
config = "/home/whetstone/pyRTC/SHARP_LAB/config_SR.yaml"

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


# %%
def resetDM():
    time.sleep(1)
    subprocess.run("kasa --host 192.168.2.100 off".split(" "))
    time.sleep(3)
    subprocess.run("kasa --host 192.168.2.100 on".split(" "))
    time.sleep(20)
    return


# %% Launch Loop Class
if SOFT_MODE:
    # Soft Loop -- Don't forget to change predictLoop.py import
    from prediction.predictLoop import *

    conf = read_yaml_file("/home/whetstone/pyRTC/SHARP_LAB/config_predict.yaml")
    loop = predictLoop(conf=conf["loop"])
else:
    loop = hardwareLauncher("prediction/predictLoop.py", config, N + 4)
    # loop = hardwareLauncher("../pyRTC/Loop.py", config, N+4)
    loop.launch()

# %%
# psfshm, _, psfDtype = initExistingShm("psfLong")
# strehlshm, _, _ = initExistingShm("strehl")
# slopeshm, _, _ = initExistingShm("signal")
# slopeshm2D, _, s2DDtype = initExistingShm("signal2D")
# wfc2D, _, _ = initExistingShm("wfc2D")
# pidOptim = PIDOptimizer(read_yaml_file(config)["optimizer"]["pid"], loop)
# loopOptim = PrePIDOptimizer(read_yaml_file(config)["optimizer"]["loop"], loop)
# ncpaOptim = NCPAOptimizer(read_yaml_file(config)["optimizer"]["ncpa"], loop, slopes)
# telem = Telemetry(read_yaml_file(config)["telemetry"])

# %%
BASELINETESTER = True
if BASELINETESTER:
    if SOFT_MODE:
        loop.baseline_mode = True
        loop.numDroppedModes = 10
        loop.gain = 0.7

        loop.loadIM()
        wfc.run("flatten")
        loop.resetBuffer()
        loop.computeCM()
        loop.toDevice()
        loop.start()
    else:
        loop.setProperty("baseline_mode", True)
        loop.setProperty("numDroppedModes", 10)
        loop.setProperty("gain", 0.25)

        loop.run("loadIM")
        wfc.run("flatten")
        loop.run("resetBuffer")
        loop.run("computeCM")
        loop.run("toDevice")
        loop.run("start")

    # from prediction.measure_jitter import measure
    # measure("Torch.Compile Inf. Mode Prediction 2k", "GPU SHM")

# %% Loop Opt
LOOP_OPTS = False
if LOOP_OPTS:

    # loop.setProperty('baseline_mode', True)
    psf.setProperty("integrationLength", 250)
    time.sleep(1)
    loopOptim.numReads = 100
    # loopOptim.maxGain = 0.6
    # loopOptim.maxLeak = 0.1
    # loopOptim.maxDroppedModes = 40
    loopOptim.numSteps = 1
    for i in trange(20):
        loopOptim.optimize()

        loop.run("stop")
        time.sleep(5)
        loopOptim.applyOptimum()

        loop.run("loadIM")
        wfc.run("flatten")
        loop.run("resetBuffer")
        loop.run("computeCM")
        loop.run("toDevice")
        loop.run("start")


opt_gains = [1.13, 0.63, 0.289, 0.1]
opt_leaks = [0.03, 0.17, 0.003, 0.001]


# %%
def runSweepIteration(dry_run=None, wandb_config=None):

    if dry_run is None:
        dry_run = False

    if not dry_run:
        wandb.init()
        wandb_config = wandb.config

    print(wandb_config)

    if SOFT_MODE:
        loop.numDroppedModes = wandb_config.num_dropped_modes
        loop.gain = wandb_config.gain
        loop.pGain = wandb_config.gain
        loop.leakyGain = wandb_config.gain
        loop.use_next_pol = wandb_config.use_next_slope
        loop.norm_slopes = wandb_config.norm_slopes
        loop.extra_noise = wandb_config.extra_noise
    else:
        loop.setProperty("numDroppedModes", wandb_config.num_dropped_modes)
        loop.setProperty("gain", opt_gains[int(wandb_config.frame_delay)])
        loop.setProperty("leakyGain", opt_leaks[int(wandb_config.frame_delay)])
        loop.setProperty("use_next_pol", wandb_config.use_next_slope)
        loop.setProperty("norm_slopes", wandb_config.norm_slopes)
        loop.setProperty("extra_noise", wandb_config.extra_noise)
        loop.setProperty("tented", wandb_config.tented)
        loop.setProperty("tent_steps", wandb_config.tent_steps)
        loop.setProperty(
            "tent_file",
            f"/home/whetstone/pyRTC/SHARP_LAB/prediction/tent_data_{wandb_config.frame_delay}.npy",
        )
        loop.setProperty("tent_LR", wandb_config.tent_LR)
    if wandb_config.model != "baseline":
        if SOFT_MODE:
            loop.halfPrecision = wandb_config.half_p
            loop.loadModel(wandb_config.model)
            loop.baseline_mode = False
        else:
            loop.setProperty("halfPrecision", wandb_config.half_p)
            loop.run("loadModel", wandb_config.model)
            loop.setProperty("baseline_mode", False)
    else:
        if SOFT_MODE:
            loop.halfPrecision = wandb_config.half_p
            loop.baseline_mode = True
        else:
            loop.setProperty("halfPrecision", wandb_config.half_p)
            loop.setProperty("baseline_mode", True)

    if SOFT_MODE:
        loop.resetBuffer()
        loop.loadIM()
        loop.toDevice()
    else:
        loop.run("resetBuffer")
        loop.run("loadIM")
        loop.run("toDevice")

    time.sleep(1)
    wfc.run("flatten")
    psf.run("start")

    wfc.run("setDelay", wandb_config.frame_delay)

    if SOFT_MODE:
        loop.start()
    else:
        loop.run("start")

    # loopOptim.resetStudy()
    # loopOptim.numReads = 6

    psfshm.read()
    nruns = 0
    psf.setProperty("integrationLength", 1000)

    # if SOFT_MODE:
    #     for _ in range(nruns):
    #         loopOptim.optimize()
    #         loop.stop()
    #         time.sleep(5)
    #         loopOptim.applyOptimum()
    #         loop.resetBuffer()
    #         loop.loadIM
    #         loop.toDevice()
    #         time.sleep(2.5)
    #         loop.start()
    #         time.sleep(2.5)
    # else:
    #     loop.setProperty("gain", 0.3)
    #     for _ in range(nruns):
    #         loopOptim.optimize()
    #         loop.run("stop")
    #         time.sleep(1)
    #         loopOptim.applyOptimum()
    #         loop.run("resetBuffer")
    #         loop.run("loadIM")
    #         loop.run("toDevice")
    #         time.sleep(1)

    #         loop.run("start")
    #         wfc.run("flatten")
    #         wfc.run("flatten")
    #         time.sleep(2.5)

    safety_valve = 0
    pbar = trange(N_LOOP - 1)

    slope2DFile = generate_filepath(
        base_dir=dataDir,
        prefix=f"slope2D_f{wandb_config.frame_delay}_{uniqueStr}_m{wandb_config.model}",
    )
    psfFile = generate_filepath(
        base_dir=dataDir,
        prefix=f"psf_f{wandb_config.frame_delay}_{uniqueStr}_m{wandb_config.model}",
    )

    print("Loop Starting:")
    PSF_raw = psfshm.read()
    strehl_raw = strehlshm.read_noblock().max()

    for step_num in pbar:

        cPSF = psfshm.read()
        cStrehl = strehlshm.read_noblock().max()
        cSlopes = slopeshm.read_noblock()
        cSlop2D = slopeshm2D.read_noblock()
        PSF_raw += cPSF
        strehl_raw += cStrehl

        if SAVE_TO_DIR:
            append_to_file(slope2DFile, cSlop2D, dtype=s2DDtype)
            append_to_file(psfFile, cPSF, dtype=psfDtype)

        if not dry_run:
            wandb.log(
                {
                    "strehl_": cStrehl,
                    "slope_mean": cSlopes.mean(),
                    "slope_std": cSlopes.std(),
                    "slope_min": cSlopes.min(),
                    "slope_max": cSlopes.max(),
                },
                step=step_num,
                commit=False,
            )
            if np.mod(step_num, N_LOOP // 100) == 0:
                wandb.log(
                    {
                        "all_slopes": wandb.Image(cSlop2D.squeeze()[:, :, None]),
                        "PSF_": wandb.Image(cPSF.squeeze()),
                    },
                    step=step_num,
                    commit=False,
                )
        if cStrehl < 0.005:
            safety_valve += 1
            if safety_valve > 250:
                break
        if np.mod(step_num, step_num // 10) == 0:
            pbar.set_description(f"{step_num}:{strehl_raw/(1+step_num):05f}")

    if SOFT_MODE:
        loop.stop()
    else:
        loop.run("stop")

    if not dry_run:

        PSFdata = PSF_raw - PSF_raw.min()
        PSFdata /= PSFdata.max()

        wandb.log(
            {  # "PSF_raw": wandb.Image(PSF_raw.squeeze()),
                "PSF_norm": wandb.Image(PSFdata.squeeze()),
                "strehl_avg": strehl_raw / N_LOOP,
                "safety_valve": safety_valve,
            },
            commit=True,
        )
        if nruns > 0:
            wandb.log({"opt_vals": loopOptim.study.best_params}, commit=True)

    resetDM()
    # time.sleep(1)
    # subprocess.run("kasa --host 192.168.2.100 off".split(" "))
    # time.sleep(30)
    # subprocess.run("kasa --host 192.168.2.100 on".split(" "))
    # time.sleep(1)

    return


sweep_config = {
    "method": "grid",
    "name": "pyRTC_TENTsweep",  # VARsweep',
    "metric": {"goal": "maximize", "name": "strehl_avg"},
    "parameters": {
        "model": {
            "distribution": "categorical",
            "values": [
                "shlong_10_default0m_SAM",  # legacy maybe re-try zone
                # 'shlong_10k_default0m_SAM',
                # 'shlong_10_1step0m',
                "shlong_10_1step0m_SAM",
                # 'shlong_10k_1step0m',
                # 'shlong_10k_1step0m_SAM',
                # 'shlong_10_3step0m',
                "shlong_10_3step0m_SAM",
                # 'shlong_10k_3step0m',
                # 'shlong_10k_3step0m_SAM',
                "baseline",
                "asam",
                "nogan-nosam",
                # 'NOGAN_NOSAM_0step_0rec',
                # 'NOGAN_NOSAM_0step_4rec',
                "NOGAN_NOSAM_0step_8rec",
                # 'NOGAN_NOSAM_1step_0rec',
                # 'NOGAN_NOSAM_1step_4rec',
                "NOGAN_NOSAM_1step_8rec",
                # 'NOGAN_NOSAM_2step_0rec',
                # 'NOGAN_NOSAM_2step_4rec',
                "NOGAN_NOSAM_2step_8rec",
                # 'NOGAN_SAM_0step_0rec',
                # 'GAN_NOSAM_0step_10rec',
                # 'GAN_NOSAM_0step_20rec',
                # 'GAN_NOSAM_1step_10rec',
                # 'GAN_NOSAM_2step_10rec',
                # 'GAN_NOSAM_1step_25rec_001n',           # doesn't work from here down \/
                # 'GAN_NOSAM_2step_25rec_001n',
                # 'GAN_NOSAM_3step_25rec_001n',
                "GAN_NOSAM_1step_25rec_01n",
                "GAN_NOSAM_2step_25rec_01n",
                "GAN_NOSAM_3step_25rec_01n",
                "GAN_NOSAM_1step_25rec_001n_longer",
                # 'GAN_NOSAM_2step_25rec_001n_longer',
                # 'GAN_NOSAM_1step_25rec_01n_ga1',      #always bad zone
                # 'GAN_NOSAM_1step_25rec_01n_ga100',
                # 'GAN_NOSAM_2step_25rec_01n_ga100',
                # 'GAN_NOSAM_3step_25rec_01n_ga100',
                # 'GAN_NOSAM_0step_noVAR',                #VAR ZONE
                #'GAN_NOSAM_1step_noVAR',
                #'GAN_NOSAM_2step_noVAR',
                # 'GAN_NOSAM_0step_VAR',
                #'GAN_NOSAM_1step_VAR',
                #'GAN_NOSAM_2step_VAR',
                # 'NOGAN_NOSAM_2step_8rec',
                # 'GAN_NOSAM_0step_10rec',
                # 'GAN_NOSAM_0step_20rec',
                # 'GAN_NOSAM_1step_10rec',
                # 'GAN_NOSAM_2step_10rec',
                "NOGAN_SAM_1step_1block_275f",
                "NOGAN_NOSAM_2step_6block_128f",
                "GAN_NOSAM_2step_6block_128f",
                # "NOGAN_NOSAM_RIV_2step_6block_128f",
                # "GAN_NOSAM_RIV_2step_6block_128f",
                "NOGAN_SAM_2step_6block_128f",
            ],
        },
        "half_p": {
            "value": False,  # maybe we should re-try this
        },
        "tent_steps": {
            "values": [10, 100, 500, 1000, 5000],
        },
        "tent_LR": {
            "values": [1e-2, 1e-3, 1e-4],
        },
        "frame_delay": {
            "values": [1, 2],
        },
        "gain": {
            "value": -1,  # 1.0 if SOFT_MODE else loop.getProperty("gain"),
        },
        "leaky_gain": {
            "value": -1,  # 1.0 if SOFT_MODE else loop.getProperty("leakyGain"),
        },
        "num_dropped_modes": {  # maybe this should be optimized
            "value": 20,
        },
        "use_next_slope": {  # maybe we should re-try this
            "value": True,
        },
        "norm_slopes": {
            "value": True,
        },
        "tented": {
            "value": True,
        },
        "wind_speed": {
            "value": 16,
        },
        "laser_dB": {
            "value": 0.03,
        },
        "extra_noise": {
            "value": False,  # This is always bad. Never turn it on!
        },
        "loop_iters": {
            "value": N_LOOP,
        },
        "soft_mode": {
            "value": SOFT_MODE,
        },
        "opt_mode": {
            "values": ["tiptilt"],  # slopes/tiptilt/strehl
        },
    },
}


# %%
if not DRY_RUN:
    sweep_id = wandb.sweep(sweep=sweep_config, project="pyRTC_LastTests")
    wandb.agent(sweep_id, function=runSweepIteration, count=10000)

else:
    wind = 15
    laser = 0.1
    delays = [0, 1, 2, 3]
    gammas = [0, 1, 2, 3]
    model_list = [
        "shlong_10_1step0m",
        "shlong_10_1step0m_SAM",
        "shlong_10_3step0m",
        "shlong_10_3step0m_SAM",
        "shlong_10_default0m",
        "shlong_10_default0m_SAM",
        "shlong_10k_1step0m",
        "shlong_10k_1step0m_SAM",
        "shlong_10k_3step0m",
        "shlong_10k_3step0m_SAM",
        "shlong_10k_default0m",
        "shlong_10k_default0m_SAM",
    ]
    half_p = [True, False]
    wgain = 0.3
    wdrop = 25

    wandb_config = DotMap(
        {
            "frame_delay": 0,  # delays[0],
            "model": "baseline",  # ,
            "half_p": False,
            "gain": wgain,
            "num_dropped_modes": wdrop,
            "output_name": "insert_fake_name_here",
            "use_next_slope": True,
            "norm_slopes": True,
            "extra_noise": False,
        }
    )
    print(wandb_config.output_name)
    PSF = runSweepIteration(dry_run=True, wandb_config=wandb_config)

# %%
