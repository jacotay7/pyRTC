# %%
import matplotlib.pyplot as plt
import time
import numpy as np

import pdb

# Import pyRTC Core classes and OOPAO interface
# from pyRTC import *
import sys

tmp = sys.stdout
# from pyRTC import *
from pyRTC.hardware.OOPAOInterface import OOPAOInterface
from pyRTC.utils import read_yaml_file

sys.stdout = tmp
# from pyRTC.hardware.OOPAOInterface import OOPAOInterface

# %%

conf = read_yaml_file("shwfs_OOPAO_config.yaml")

# Split into component sections
confLoop = conf["loop"]
confWFS = conf["wfs"]
confWFC = conf["wfc"]
confPSF = conf["psf"]
confSlopes = conf["slopes"]

print(confLoop)
print(confWFS)
print(confWFC)
print(confPSF)
print(confSlopes)

# %%
param = dict()

###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ATMOSPHERE PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

param["r0"] = 0.15  # value of r0 in the visible in [m]
param["L0"] = 30  # value of L0 in the visible in [m]
param["fractionnalR0"] = [0.45, 0.1, 0.1, 0.25, 0.1]  # Cn2 profile
param["windSpeed"] = [
    10,
    12,
    11,
    15,
    20,
]  # wind speed of the different layers in [m.s-1]
param["windDirection"] = [
    0,
    72,
    144,
    216,
    288,
]  # wind direction of the different layers in [degrees]
param["altitude"] = [
    0,
    1000,
    5000,
    10000,
    12000,
]  # altitude of the different layers in [m]

###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% WFS PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

param["nSubaperture"] = 98  # confSlopes["lensletsX"]
# number of subaperture along the telescope diameter
param["nPixelPerSubap"] = 14  # sampling of the PWFS subapertures

###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% M1 PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

param["diameter"] = 8  # diameter in [m]
param["resolution"] = (
    param["nSubaperture"] * param["nPixelPerSubap"]
)  # resolution of the telescope driven by the PWFS
param["sizeSubaperture"] = (
    param["diameter"] / param["nSubaperture"]
)  # size of a sub-aperture projected in the M1 space
param["samplingTime"] = 1 / 300  # loop sampling time in [s]
param["centralObstruction"] = 0.112  # central obstruction in percentage of the diameter
param["nMissingSegments"] = 0  # number of missing segments on the M1 pupil
param["m1_reflectivity"] = 1  # reflectivity of the 798 segments

###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% NGS PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

param["magnitude"] = 8  # magnitude of the guide star
param["opticalBand"] = "R"  # optical band of the guide star
param["sourceBand"] = "J"

###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DM PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
param["nActuator"] = param["nSubaperture"] + 1  # number of actuators
param["mechanicalCoupling"] = 0.45
param["isM4"] = False  # tag for the deformable mirror class
param["dm_coordinates"] = None  # tag for the eformable mirror class

# mis-registrations
param["shiftX"] = 0  # shift X of the DM in pixel size units ( tel.D/tel.resolution )
param["shiftY"] = 0  # shift Y of the DM in pixel size units ( tel.D/tel.resolution )
param["rotationAngle"] = 0  # rotation angle of the DM in [degrees]
param["anamorphosisAngle"] = 0  # anamorphosis angle of the DM in [degrees]
param["radialScaling"] = 0  # radial scaling in percentage of diameter
param["tangentialScaling"] = 0  # tangential scaling in percentage of diameter

###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% WFS PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

param["modulation"] = (
    5  # modulation radius in ratio of wavelength over telescope diameter
)
param["n_pix_separation"] = 4  # separation ratio between the PWFS pupils
param["psfCentering"] = (
    False  # centering of the FFT and of the PWFS mask on the 4 central pixels
)
param["lightThreshold"] = 0.1  # light threshold to select the valid pixels
param["postProcessing"] = "slopesMaps"  # post-processing of the PWFS signals

###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OUTPUT DATA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# name of the system
param["name"] = (
    "VLT_"
    + param["opticalBand"]
    + "_band_"
    + str(param["nSubaperture"])
    + "x"
    + str(param["nSubaperture"])
)

# location of the calibration data
param["pathInput"] = "data_calibration/"

# location of the output data
param["pathOutput"] = "data_cl/"

# %%
# shm_names = ["wfs", "wfsRaw", "wfc", "wfc2D", "signal", "signal2D", "psfShort", "psfLong"]
# clear_shms(shm_names)

# %%

# Create OOPAO simulation interface
pdb.set_trace()
sim = OOPAOInterface(conf=conf, param=param)

# Extract references to the simulated hardware
wfs, dm, psf = sim.get_hardware()

# %%
from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis

NUM_MODES = confWFC["numModes"]  # 97
M2C_KL = compute_KL_basis(sim.tel, sim.atm, sim.dm)
dm.setM2C(M2C_KL[:, :NUM_MODES])
