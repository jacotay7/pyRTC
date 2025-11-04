# %%

import optuna
from prediction.extendedImageOptimizer import *
from tqdm import trange

optuna.logging.set_verbosity(optuna.logging.DEBUG)
numOptim = 1
maxAMP = 0.001
amps = np.linspace(maxAMP, maxAMP / 5, numOptim)
ncpaOptim = extendedImageOptimizer(
    read_yaml_file(config)["optimizer"]["ncpa"], loop, slopes
)


# %%
assert psf.run("setIntegrationLength", 100)
assert psf.run("setExposure", 300000)
assert psf.run("setGamma", 0.5)
assert psf.run("setGain", 16)

# %%

for i in range(numOptim):
    # hardFlat()
    psf.setProperty("integrationLength", 1)

    ncpaOptim.resetStudy()
    ncpaOptim.numReads = 1
    ncpaOptim.startMode = 0
    ncpaOptim.endMode = 75  # *(i+1)
    ncpaOptim.numSteps = 5
    ncpaOptim.correctionMag = amps[i]
    ncpaOptim.isCL = False

    time.sleep(2)

    # for _ in trange(100):
    ncpaOptim.optimize()
    ncpaOptim.applyOptimum()

    # wfc.run("saveShape")
    # slopes.run("takeRefSlopes")
    # slopes.setProperty("refSlopesFile", "/home/whetstone/pyRTC/SHARP_LAB/calib/ref_SH.npy")
    # slopes.run("saveRefSlopes")
    # psf.setProperty("integrationLength", 2000)
    # time.sleep(2)
    # psf.run("takeModelPSF")
    # psf.setProperty("modelFile", "/home/whetstone/pyRTC/SHARP_LAB/calib/modelPSF_SH.npy")
    # psf.run("saveModelPSF")
    # wfc.run("loadFlat")
# %%
