# %% IMPORTS
import argparse
#Import pyRTC classes
from pyRTC.Loop import *
from pyRTC.Pipeline import *
#Import hardware classes
# from pyRTC.hardware.ALPAODM import *
# from pyRTC.hardware.ximeaWFS import *
from pyRTC.hardware.SpinnakerScienceCam import *
#Import utils
from pyRTC.utils import *

# # Create argument parser
# parser = argparse.ArgumentParser(description="Read a config file from the command line.")

# # Add command-line argument for the config file
# parser.add_argument("-c", "--config", required=True, help="Path to the config file")

# # Parse command-line arguments
# args = parser.parse_args()
# config = args.config

# %% IMPORTS
config = '../SHARP_LAB/config.yaml'

# %% Launch WFS
wfs = hardwareLauncher("./hardware/ximeaWFS.py", config)
wfs.launch()
time.sleep(4)
# %% Launch PSF Cam
psfCam = hardwareLauncher("./hardware/SpinnakerScienceCam.py", config)
psfCam.launch()
time.sleep(2)

# %% Launch DM
wfc = hardwareLauncher("./hardware/ALPAODM.py", config)
wfc.launch()
time.sleep(0.5)
# %% Launch Loop Class
loop = hardwareLauncher("./Loop.py", config)
loop.launch()
time.sleep(0.5)


# %% Take Darks
# time.sleep(2)
psfCam.run("takeDark")
psfCam.run("saveDark")
wfs.run("takeDark")
wfs.run("saveDark")

# %%Compute an new IM
wfc.run("flatten")
loop.run("computeIM")
loop.run("saveIM")
wfc.run("flatten")
time.sleep(1)

# %%
loop.setProperty("numDroppedModes", 10)
loop.run("computeCM")

# %%Launch Loop for 5 seconds
loop.run("setGain",0.3)
wfc.run("flatten")
loop.run("start")
time.sleep(5)
loop.run("stop")
wfc.run("flatten")


# input()


# %% Kill everything
hardware = [psfCam, wfs, wfc, loop]
for h in hardware:
    h.shutdown()
    time.sleep(1)

# %%
