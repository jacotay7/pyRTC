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
import logging

# logging.disable(logging.ERROR)
from prediction.extendedImageOptimizer import *

# from pyRTC.hardware import PIDOptimizer as PrePIDOptimizer
wandb.require("core")
import os

os.environ["WANDB_DIR"] = os.path.abspath("/media/whetstone/storage2/data/wandb")

# %%
N = np.random.randint(3000, 6000)
# config = "/home/whetstone/pyRTC/SHARP_LAB/config_predict.yaml"
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
# loop = hardwareLauncher("prediction/predictLoop.py", config, N+4)
loop = hardwareLauncher("../pyRTC/Loop.py", config, N + 4)
loop.launch()


# %%
psfshm, _, psfDtype = initExistingShm("psfLong")
psfShortShm, _, _ = initExistingShm("psfShort")
strehlshm, _, _ = initExistingShm("strehl")
slopeshm, _, _ = initExistingShm("signal")
slopeshm2D, _, s2DDtype = initExistingShm("signal2D")
wfc2D, _, _ = initExistingShm("wfc2D")
pidOptim = PIDOptimizer(read_yaml_file(config)["optimizer"]["pid"], loop)
loopOptim = loopOptimizer(read_yaml_file(config)["optimizer"]["loop"], loop)
ncpaOptim = NCPAOptimizer(read_yaml_file(config)["optimizer"]["ncpa"], loop, slopes)
extendedOptim = extendedImageOptimizer(
    read_yaml_file(config)["optimizer"]["ncpa"], loop, slopes
)


# %%
def getMeanSR(N):
    val = 0
    strehlshm.read()
    for i in range(N):
        val += np.mean(strehlshm.read())
        x = wfc2D.read_noblock()
        if np.any(np.isnan(x)):  # or np.any(np.abs(x) > 0.8):
            return -1
    return val / N


psf.setProperty("integrationLength", 1000)
DELAY = 1


def hardFlat():
    loop.run("stop")
    for i in range(DELAY + 1):
        loop.run("flatten")
        time.sleep(0.1)


def resetDM():
    logging.log(level=logging.DEBUG, msg=f"Resetting DM")
    time.sleep(1)
    subprocess.run("kasa --host 192.168.2.100 off".split(" "))
    time.sleep(3)
    subprocess.run("kasa --host 192.168.2.100 on".split(" "))
    time.sleep(10)
    return


# %% Set-up baseline
def resetBaseline(gain, delay=0, leaky=0.02):
    loop.run("stop")
    time.sleep(0.5)
    hardFlat()

    wfc.run("setDelay", delay)
    loop.setProperty("baseline_mode", True)
    loop.setProperty("tented", False)
    loop.setProperty("numDroppedModes", 15)
    loop.setProperty("gain", gain)
    loop.setProperty("leakyGain", leaky)
    loop.setProperty("pGain", 0.1)
    loop.run("loadIM")
    loop.run("start")


def resetPredictor(
    model,
    gain,
    norm=0.6,
    delay=0,
    leaky=0.02,
    tented=True,
    tent_steps=100,
    tent_LR=1e-4,
    tent_ahead=1,
):
    loop.run("stop")
    time.sleep(10)
    hardFlat()
    loop.setProperty("tented", tented)
    loop.setProperty(
        "tent_file", f"/home/whetstone/pyRTC/SHARP_LAB/prediction/tent_data_{delay}.npy"
    )
    loop.setProperty("tent_steps", tent_steps)
    loop.setProperty("tent_LR", tent_LR)
    loop.setProperty("tent_ahead", tent_ahead)
    wfc.run("setDelay", delay)
    loop.setProperty("halfPrecision", False)
    loop.setProperty("baseline_mode", False)
    loop.setProperty("numDroppedModes", 10)
    loop.setProperty("gain", gain)
    loop.setProperty("leakyGain", leaky)
    loop.setProperty("use_next_pol", True)
    loop.setProperty("norm_slopes", True)
    loop.setProperty("extra_noise", False)
    # loop.setProperty("opt_mean", 0.2)
    # loop.setProperty("opt_std", 2.2)
    loop.setProperty("opt_mean", 0.0)  # CHANGED TO MATCH DATA
    loop.setProperty("opt_std", norm)
    time.sleep(1)
    loop.run("loadModel", model)
    loop.run("resetBuffer")
    loop.run("loadIM")
    time.sleep(0.5)

    loop.run("start")
    time.sleep(3)


# %%
plateTag = "D"
speedTag = "5"
laserTag = "0.02"


def recordPSFs(N, model):

    trainTag = model
    onOffTag = "ON"
    delayTag = wfc.getProperty("frameDelay")
    # dataName = f"plate_{plateTag}_speed_{speedTag}_{trainTag}_predict_{onOffTag}_delay_{delayTag}"
    dataName = f"{model}M_{delayTag}D_{speedTag}W_{laserTag}L"
    psfs = np.empty((N, *psfShortShm.read_noblock().shape))
    slopes = np.empty((N, *slopeshm2D.read_noblock().shape))

    for i in trange(N):
        psfs[i] = psfShortShm.read()
        slopes[i] = slopeshm2D.read()

    np.save(f"/media/whetstone/storage/predictData/{dataName}_psf", psfs)
    np.save(f"/media/whetstone/storage/predictData/{dataName}_slopes", slopes)
    return


# %%
resetBaseline(gain=1.0, leaky=0.002)
# # %%
# resetBaseline(gain=0.75, delay=1)
# # %%
# resetBaseline(gain=0.313, leaky=0.004, delay=2)
# # %%
# resetPredictor('NOGAN_SAM_1step_1block_275f', gain=1.13, leaky=0.03)

# %%
# resetPredictor('NOGAN_SAM_1step_1block_275f', gain=0.63, leaky=0.017, delay=1)
# # %%
# resetPredictor('optuna_vecout', gain=1.0, delay=0)

# # %%
# resetPredictor('GAN_NOSAM_0step_20rec', gain=0.1, leaky=0.001, delay=3)
# %%
hardFlat()
# %%
# models = [ 'baseline',
#             'shlong_10_default0m_SAM',
#             # 'shlong_10k_3step0m',
#             'shlong_10k_3step0m_SAM',
#             'asam',
#             # 'nogan-nosam',
#             # 'NOGAN_NOSAM_0step_8rec',
#             # 'NOGAN_NOSAM_1step_8rec',
#             # 'NOGAN_NOSAM_2step_8rec',
#             # 'NOGAN_SAM_0step_0rec',
#             'GAN_NOSAM_0step_20rec',
#             # 'GAN_NOSAM_1step_25rec_01n',
#             # 'NOGAN_NOSAM_2step_8rec',
#             'GAN_NOSAM_1step_10rec',
#             'GAN_NOSAM_2step_10rec']

# %%
# VOLTS = 16
# TAG = "overnightOct24"
# psf.setProperty("integrationLength", 3000)
# gArr = []
# gArr.append(np.linspace(0.1, 2, 20))
# gArr.append(np.linspace(0.1, 2, 20))
# gArr.append(np.linspace(0.1, 2, 20))

# for norm in np.linspace(0.6,2.5,5):
#     for DELAY in range(3):
#         gains = gArr[DELAY]
#         # models = modelsMatrix[DELAY]
#         results = np.zeros((len(models), len(gains)))

#         np.save(f"/home/whetstone/predictControlData/slModelGains_delay_{DELAY}_volts_{VOLTS}_norm_{np.round(norm,2)}_tag_{TAG}", gains)
#         np.save(f"/home/whetstone/predictControlData/slModelModels_delay_{DELAY}_volts_{VOLTS}_norm_{np.round(norm,2)}_tag_{TAG}", np.array(models))
#         for i, model in enumerate(models):
#             gainTooHigh = False #Stop trying once we diverge
#             for j, g in enumerate(gains):
#                 if gainTooHigh:
#                     results[i,j] = -1
#                 else:
#                     print(f"Running Model: {model}, Gain: {g:.2f}")
#                     if model == 'baseline':
#                         resetBaseline(g, delay= DELAY)
#                     else:
#                         resetPredictor(model, g, norm=norm, delay= DELAY)
#                         # time.sleep(3)
#                     results[i,j] = getMeanSR(5)
#                 print(f"SR: {np.round(100*results[i,j])}%")
#                 np.save(f"/home/whetstone/predictControlData/slModelSRs_delay_{DELAY}_volts_{VOLTS}_norm_{np.round(norm,2)}_tag_{TAG}", results)

#                 if results[i,j] < 0:
#                     #If first time it diverged for this model
#                     if not gainTooHigh:
#                         resetDM()
#                         gainTooHigh = True
#                     hardFlat()
#                 elif model != 'baseline' and i > 0 and results[i,j] > np.nanmax(results[0]):
#                     recordPSFs(10000, model)


# %%
# def find_opt_mean_std():
#     polShm = initExistingShm("pol")[0]
#     N = 5000
#     data = np.empty((N, *polShm.read_noblock().shape))
#     for i in range(N):
#         data[i] = polShm.read()
#     filteredData = data[data!=0].flatten()
#     plt.hist(filteredData)
#     plt.show()
#     return filteredData.mean(), filteredData.std()
# %%
# SRs = []
# # norms = np.linspace(3,3.5,1)
# norm = 3
# gainsBL = [1.5, 0.8, 0.45]
# results = []
# model = "GAN_NOSAM_0step_20rec"
# for DELAY in range(1,3):
#     resetBaseline(gainsBL[DELAY], delay= DELAY)
#     baselineSR = getMeanSR(5)
#     recordPSFs(10000, 'baseline')
#     resetPredictor(model, gains[DELAY], norm=norm, delay= DELAY)
#     SR = getMeanSR(5)
#     SRs.append(SR)
#     print(f"SR: {np.round(100*SR)}%, Baseline: {np.round(100*baselineSR)}")
#     if SR < 0:
#         resetDM()
#         continue
#     recordPSFs(10000, model)
# # %%
# model = "NOGAN_NOSAM_1step_8rec"
# SRs = []
# for gamma in np.linspace(0, 0.1, 10):
#     loop.setProperty("gamma", gamma)
#     resetPredictor(model, 0.75, norm=0.6, delay= 1)
#     SR = getMeanSR(5)
#     SRs.append(SR)
#     print(f"SR: {np.round(100*SR)}%")
#     if SR < 0:
#         hardFlat()
#         resetDM()
#         continue
# # %%
# plt.plot(np.linspace(0, 1, 10), SRs)
# plt.show()
# %%
"""
asam, gamma = 0.4, norm = 0.6, delay 1
GAN_NOSAM_0step_20rec, gamma = 0.4, norm = 0.6, delay 1
shlong_10_default0m_SAM. gamma = 0.1, norm = 0.6, delay 1
"""

# %%
loopOptim = loopOptimizer(read_yaml_file(config)["optimizer"]["loop"], loop)


def checkValid():
    x = wfc2D.read_noblock()
    if np.any(np.isnan(x)):
        hardFlat()
        resetDM()
        return False
    # elif np.any(np.abs(x) > 0.8):
    #    hardFlat()
    #    return False
    return True


loopOptim.checkValidFunc = checkValid

time.sleep(1)
models = [
    "optuna_vecout",
    # "NOGAN_SAM_1step_1block_275f",
    # "NOGAN_NOSAM_2step_6block_128f",
    "GAN_NOSAM_2step_6block_128f",
    # "NOGAN_SAM_2step_6block_128f",
    "asam",
]
# baselines = []#[0.53, 0.23, 0.03]

# %%
if True:
    for DELAY in range(2):

        hardFlat()
        resetDM()

        fname = f"baseline_{DELAY}D_{speedTag}W_{laserTag}L"
        loopOptim.storage = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(
                f"/media/whetstone/storage/predictData/{fname}.log"
            )
        )
        loopOptim.resetStudy()

        if DELAY == 0:
            resetBaseline(1.5, delay=0)
        elif DELAY == 1:
            resetBaseline(0.75, delay=1)
        elif DELAY == 2:
            resetBaseline(0.45, delay=2)

        psf.setProperty("integrationLength", 2000)
        time.sleep(1)

        loopOptim.numReads = 10
        loopOptim.numSteps = 10
        loopOptim.adjustParam("gain", 0.15, 0.75)  # /(DELAY+1), 1.8/(DELAY+1))
        loopOptim.adjustParam("leaky_gain", -0.05, 0.05)

        # for _ in range(2):
        time.sleep(1)
        loopOptim.optimize()
        loopOptim.applyOptimum()

        psf.setProperty("integrationLength", 10000)

        time.sleep(3)
        recordPSFs(5000, "baseline")
        time.sleep(3)

# psf.setProperty("integrationLength", 2000)
# print(baselines)

# assert False
# %%
ID = 0
psf.setProperty("integrationLength", 2000)
# getMeanSR(1)

# loopOptim.adjustParam("gamma", 0, 1.1)
# loopOptim.adjustParam("opt_std", 0, 5.0)
loopOptim.adjustParam("tented_steps", 1, 4900, type=int, log=True)
loopOptim.adjustParam("tented_LR", 1e-5, 1e-2, log=True)
loopOptim.adjustParam("tented_ahead", 0, 50, type=int)
loopOptim.adjustParam("leaky_gain", 0.0, 0.02)

for DELAY in range(2):
    for model in models:

        hardFlat()
        resetDM()

        fname = f"{model}_{DELAY}D_{speedTag}W_{laserTag}L"
        loopOptim.storage = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(
                f"/media/whetstone/storage/predictData/{fname}.log"
            )
        )
        loopOptim.resetStudy()
        # loopOptim.numReads = 5
        # loopOptim.numSteps = 30

        psf.setProperty("integrationLength", 2000)
        if "vecout" in model.lower():
            loopOptim.adjustParam("cudaGraph.nAhead", 0, 5)
        else:
            loopOptim.adjustParam("cudaGraph.nAhead", 0, 0)

        # if DELAY == 0:
        resetPredictor(model, gain=0.8 / (DELAY + 1), norm=0.6, delay=DELAY)
        # loopOptim.adjustParam("gain", 1.5/2, 1.5*1.1)
        # elif DELAY == 1:
        #     resetPredictor(model, 0.4, norm=0.6, delay= 1)
        #     loopOptim.adjustParam("gain", 0.75/2, 0.75*1.1)
        # elif DELAY == 2:
        #     resetPredictor(model, 0.2, norm=0.6, delay= 2)
        #     loopOptim.adjustParam("gain", 0.45/2, 0.45*1.1)

        # loop.setProperty("cudaGraph.nAhead", NA)
        # Let the strehl SHM catch up

        # loopOptim.resetStudy()
        loopOptim.numReads = 10
        loopOptim.numSteps = 10
        loopOptim.adjustParam("gain", 0.25, 1.25)  # /(DELAY+1), 1.8/(DELAY+1))
        loopOptim.adjustParam("leaky_gain", -0.05, 0.05)

        # for _ in range(2):
        time.sleep(3)
        loopOptim.optimize()
        loopOptim.applyOptimum()

        # time.sleep(40)
        # modelSRLong = getMeanSR(4)

        # print(f'{runTag}_{model}_{DELAY}_{NA}: {modelSRLong}')

        # if modelSRLong - baselines[DELAY] > 0.03:

        psf.setProperty("integrationLength", 10000)

        time.sleep(3)
        recordPSFs(5000, model)
        time.sleep(3)

        hardFlat()
        time.sleep(5)
# %%
assert False

# %%

N = 100
# D = 3

# if D==0:
#     resetBaseline(gain=1.5, delay=D)
# elif D==1:
#     resetBaseline(gain=0.75, delay=D)
# elif D==2:
#     resetBaseline(gain=0.313, leaky=0.004, delay=D)
# else:
#     resetBaseline(gain=0.1, leaky=0.002, delay=D)


polShm = initExistingShm("pol")[0]
slopesShm = initExistingShm("signal")[0]
wfcShm = initExistingShm("wfc")[0]
networkOutData = np.zeros((N, *polShm.read_noblock().shape))
polBuffer = np.zeros((N, *polShm.read_noblock().shape))
IM = np.load(loop.getProperty("IMFile"))
validSubAps = np.load(loop.getProperty("validSubApsFile"))
for i in tqdm(range(N)):

    # Compute POL Slopes
    polBuffer[i][validSubAps] = slopesShm.read() - IM @ wfcShm.read_noblock()
    networkOutData[i] = polShm.read_noblock() * validSubAps

# np.save("/home/whestone/thesisPlots/networkOut", networkOutData)
# np.save("/home/whestone/thesisPlots/polBuffer", polBuffer)
# np.save(f'/home/whetstone/pyRTC/SHARP_LAB/prediction/tent_data_{D}.py', polBuffer)

# %%
n = 20
fig, axes = plt.subplots(1, 4)
axes[0].imshow(networkOutData[n], vmin=-1, vmax=1)
axes[0].set_title(f"Network Output")
print(0, np.std(networkOutData[n]))
for i in range(1, 4):
    delta = polBuffer[n + i] - polBuffer[n]
    im = axes[i].imshow(delta, vmin=-1, vmax=1)
    axes[i].set_title(f"Delay {i}")
    print(i, np.std(delta))
# fig.colorbar(im)
plt.tight_layout()
plt.show()
# fig, axes = plt.subplots(1,4)
# for i in range(4):
#     im = axes[i].imshow((polBuffer[n+i]), vmin = -3, vmax = 3)
#     axes[i].set_title(f"Frame #{i}")
# # fig.colorbar(im)
# plt.tight_layout()
# plt.show()

# %%
polShm = initExistingShm("pol")[0]
N = 10000
polBuffer = np.zeros((N, *polShm.read_noblock().shape))
for i in tqdm(range(N)):

    polBuffer[i] = polShm.read()
np.save("/media/whetstone/storage/predictData/buffer_baseline", polBuffer)
# %%


# %%
def restart_loop(d=0):
    loopOptim.resetStudy()
    loopOptim.numReads = 5
    loopOptim.numSteps = 10

    psf.setProperty("integrationLength", 2000)

    if d == 0:
        maxG = 2.0
        maxL = 0.05
    elif d == 1:
        maxG = 1.0
        maxL = 0.03
    else:
        maxG = 0.5
        maxL = 0.01

    loopOptim.adjustParam("gain", 0.1, maxG)
    loopOptim.adjustParam("leaky_gain", 0.00, maxL)
    loopOptim.adjustParam("numDroppedModes", 0, 25)
    time.sleep(1)
    loopOptim.optimize()
    loopOptim.applyOptimum()


models = [
    # 'baseline',
    "NOGAN_SAM_1step_1block_275f",
    "NOGAN_NOSAM_2step_6block_128f",
    "GAN_NOSAM_2step_6block_128f",
    # "NOGAN_NOSAM_RIV_2step_6block_128f",
    # "GAN_NOSAM_RIV_2step_6block_128f",
    "NOGAN_SAM_2step_6block_128f",
    "shlong_10_default0m_SAM",
    # 'shlong_10k_3step0m',
    "shlong_10k_3step0m_SAM",
    "asam",
    # 'nogan-nosam',
    # 'NOGAN_NOSAM_0step_8rec',
    # 'NOGAN_NOSAM_1step_8rec',
    # 'NOGAN_NOSAM_2step_8rec',
    # 'NOGAN_SAM_0step_0rec',
    "GAN_NOSAM_0step_20rec",
    # 'GAN_NOSAM_1step_25rec_01n',
    # 'NOGAN_NOSAM_2step_8rec',
    "GAN_NOSAM_1step_10rec",
    "GAN_NOSAM_2step_10rec" "NOGAN_SAM_1step_1block_275f",
]

delays = [0, 1, 2, 3]

optimals = []
opt_params = []

resetBaseline(gain=1.5)
restart_loop(d=0)
opt_params.append(loopOptim.study.best_params)
optimals.append(loopOptim.study.best_value)

# %%
resetBaseline(gain=0.75, delay=1)
restart_loop(d=1)
opt_params.append(loopOptim.study.best_params)
optimals.append(loopOptim.study.best_value)

# %%
resetBaseline(gain=0.313, leaky=0.004, delay=2)
restart_loop(d=2)
opt_params.append(loopOptim.study.best_params)
optimals.append(loopOptim.study.best_value)

# %%
resetBaseline(gain=0.313, leaky=0.004, delay=3)
restart_loop(d=2)
opt_params.append(loopOptim.study.best_params)
optimals.append(loopOptim.study.best_value)

for mx, mo in enumerate(models):

    resetPredictor(mo, gain=1.13, leaky=0.03)
    restart_loop(d=0)
    opt_params.append(loopOptim.study.best_params)
    optimals.append(loopOptim.study.best_value)

    # %%
    resetPredictor(mo, gain=0.63, leaky=0.017, delay=1)
    restart_loop(d=1)
    opt_params.append(loopOptim.study.best_params)
    optimals.append(loopOptim.study.best_value)

    # %%
    resetPredictor(mo, gain=0.289, leaky=0.003, delay=2)
    restart_loop(d=2)
    opt_params.append(loopOptim.study.best_params)
    optimals.append(loopOptim.study.best_value)

    # %%
    resetPredictor(mo, gain=0.289, leaky=0.003, delay=3)
    restart_loop(d=3)
    opt_params.append(loopOptim.study.best_params)
    optimals.append(loopOptim.study.best_value)
