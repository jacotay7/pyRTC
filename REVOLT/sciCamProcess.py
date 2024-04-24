# %% Imports
from pyRTC.hardware.alliedVisionScienceCam import *
from pyRTC.utils import *
config = "c:/Users/mcaousr/pyRTC/REVOLT/config.yaml"
conf = read_yaml_file("c:/Users/mcaousr/pyRTC/REVOLT/config.yaml")
# %% Soft RTC
# psf = alliedVisionScienceCam(conf["psf"])
# psf.start()
# %% Create a local process for the remote RTC
from pyRTC.Pipeline import *
from subprocess import PIPE, Popen
port = 3142
hardwareFile = "c:/Users/mcaousr/pyRTC/pyRTC/hardware/alliedVisionScienceCam.py"
IP = "132.246.193.118"
command = ["python", hardwareFile, "-c", f"{config}", "-p", f"{port}", '-h', f"{IP}"]
print(" ".join(command))
# process = Popen(command,stdin=PIPE,stdout=PIPE, text=True, bufsize=1)

# %%
