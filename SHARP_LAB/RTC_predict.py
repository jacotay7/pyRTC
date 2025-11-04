# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from pyRTC import *
from pyRTC.hardware import *
from pyRTC.utils import *
import wandb
from dotmap import DotMap
from pyRTC.Pipeline import initExistingShm, clear_shms
import logging
import time
from tqdm import tqdm

# logging.disable(logging.ERROR)
from prediction.prepidopt import PrePIDOptimizer

# from pyRTC.hardware import PIDOptimizer as PrePIDOptimizer
wandb.require("core")
import os
from tqdm import trange

os.environ["WANDB_DIR"] = os.path.abspath("/media/whetstone/storage2/data/wandb")

dataDir = "/media/whetstone/storage/pyRTCTelem"
uniqueStr = "oct9tests2"

SAVE_TO_DIR = False
DRY_RUN = False
SOFT_MODE = True
N_LOOP = 200

# %%
N = np.random.randint(3000, 6000)
config = "/home/whetstone/pyRTC/SHARP_LAB/config_predict.yaml"

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
psfshm, _, psfDtype = initExistingShm("psfLong")
strehlshm, _, _ = initExistingShm("strehl")
slopeshm, _, _ = initExistingShm("signal")
slopeshm2D, _, s2DDtype = initExistingShm("signal2D")
wfc2D, _, _ = initExistingShm("wfc2D")
# pidOptim = PIDOptimizer(read_yaml_file(config)["optimizer"]["pid"], loop)
# loopOptim = PrePIDOptimizer(read_yaml_file(config)["optimizer"]["loop"], loop)
# ncpaOptim = NCPAOptimizer(read_yaml_file(config)["optimizer"]["ncpa"], loop, slopes)
# telem = Telemetry(read_yaml_file(config)["telemetry"])

# %%

model_list = [  #'shlong_10_default0m_SAM',        # legacy maybe re-try zone
    #'shlong_10k_default0m_SAM',
    # 'shlong_10_1step0m',
    # 'shlong_10_1step0m_SAM',
    #'shlong_10k_1step0m',
    #'shlong_10k_1step0m_SAM',
    # 'shlong_10_3step0m',
    # 'shlong_10_3step0m_SAM',
    # 'shlong_10k_3step0m',
    # 'shlong_10k_3step0m_SAM',
    # 'asam',
    # 'nogan-nosam',
    # 'NOGAN_NOSAM_0step_0rec',
    # 'NOGAN_NOSAM_0step_4rec',
    # 'NOGAN_NOSAM_0step_8rec',
    # 'NOGAN_NOSAM_1step_0rec',
    # 'NOGAN_NOSAM_1step_4rec',
    # 'NOGAN_NOSAM_1step_8rec',
    # 'NOGAN_NOSAM_2step_0rec',
    # 'NOGAN_NOSAM_2step_4rec',
    # 'NOGAN_NOSAM_2step_8rec',
    # 'NOGAN_SAM_0step_0rec',
    # 'GAN_NOSAM_0step_10rec',
    # 'GAN_NOSAM_0step_20rec',
    # 'GAN_NOSAM_1step_10rec',
    # 'GAN_NOSAM_2step_10rec',
    # 'GAN_NOSAM_1step_25rec_001n',           # doesn't work from here down \/
    # 'GAN_NOSAM_2step_25rec_001n',
    # 'GAN_NOSAM_3step_25rec_001n',
    # 'GAN_NOSAM_1step_25rec_01n',
    # 'GAN_NOSAM_2step_25rec_01n',
    # 'GAN_NOSAM_3step_25rec_01n',
    # 'GAN_NOSAM_1step_25rec_001n_longer',
    "baseline",
    "GAN_NOSAM_2step_25rec_001n_longer",
    "GAN_NOSAM_1step_25rec_01n_ga1",  # always bad zone
    "GAN_NOSAM_1step_25rec_01n_ga100",
    "GAN_NOSAM_2step_25rec_01n_ga100",
    "GAN_NOSAM_3step_25rec_01n_ga100",
    "GAN_NOSAM_0step_noVAR",  # VAR ZONE
    "GAN_NOSAM_1step_noVAR",
    "GAN_NOSAM_2step_noVAR",
    "GAN_NOSAM_0step_VAR",
    "GAN_NOSAM_1step_VAR",
    "GAN_NOSAM_2step_VAR",
    "NOGAN_NOSAM_2step_8rec",
    "GAN_NOSAM_0step_10rec",
    "GAN_NOSAM_0step_20rec",
    "GAN_NOSAM_1step_10rec",
    "GAN_NOSAM_2step_10rec",
    "baseline",
]

# %%
half_p = False
frame_delay = 1
use_next_slope = True
norm_slopes = True
extra_noise = False
loop_iters = 1000
soft_mode = False
opt_mode = "tiptilt"
model_ind = -3

model = "asam"  # model_list[model_ind]

if SOFT_MODE:
    loop.stop()
    loop.use_next_pol = use_next_slope
    loop.norm_slopes = norm_slopes
    loop.extra_noise = extra_noise
    loop.halfPrecision = half_p

    if model != "baseline":
        loop.loadModel(model)
        loop.baseline_mode = False
    else:
        loop.baseline_mode = True
    loop.resetBuffer()
    loop.loadIM()
    loop.toDevice()
else:
    loop.run("stop")
    loop.setProperty("use_next_pol", use_next_slope)
    loop.setProperty("norm_slopes", norm_slopes)
    loop.setProperty("extra_noise", extra_noise)
    loop.setProperty("halfPrecision", half_p)

    if model != "baseline":
        loop.run("loadModel", model)
    loop.setProperty("baseline_mode", model == "baseline")

    loop.run("resetBuffer")
    loop.run("loadIM")
    loop.run("toDevice")

psf.setProperty("integrationLength", 250)

time.sleep(1)

wfc.run("flatten")
psf.run("start")
wfc.run("setDelay", frame_delay)

if SOFT_MODE:
    loop.start()
    loop.gain = 0.3
else:
    loop.run("start")
    loop.setProperty("gain", 0.3)

loopOptim = PrePIDOptimizer(
    read_yaml_file(config)["optimizer"]["loop"], loop, softMode=SOFT_MODE
)
loopOptim.resetStudy()
loopOptim.numReads = 8
nruns = 30

psfshm.read()

psf.setProperty("integrationLength", 1000)

# %%
print(f"Starting! {model=}")  # {loop.getProperty("baseline_mode")=}')

for _ in range(nruns):
    loopOptim.optimize()
    if SOFT_MODE:
        loop.stop()
    else:
        loop.run("stop")
    time.sleep(1)
    loopOptim.applyOptimum()

    if SOFT_MODE:
        loop.resetBuffer()
        loop.loadIM()
        loop.toDevice()
        time.sleep(1)
        loop.start()
    else:
        loop.run("resetBuffer")
        loop.run("loadIM")
        loop.run("toDevice")
        time.sleep(1)

        loop.run("start")

    wfc.run("flatten")
    wfc.run("flatten")
    time.sleep(2.5)

if SOFT_MODE:
    loop.stop()
else:
    loop.run("stop")
# %%
