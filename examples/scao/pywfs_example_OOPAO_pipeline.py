
# %% IMPORTS
#Import pyRTC classes
from pyRTC.Pipeline import *
from pyRTC.utils import *
from pyRTC.hardware.OOPAOInterface import OOPAOInterface

RECALIBRATE = False
hardware_folder = "../../pyRTC/"
# %% Load Config
config = './pywfs_OOPAO_config.yaml'

# Load the configuration file
def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        conf = yaml.safe_load(file)
    return conf

#Now we can read our YAML config file 
conf = read_yaml_file(config)

N = np.random.randint(3000,6000)
# %% Launch sim, this is WFS, DM and PSF Camera
# sim = hardwareLauncher(hardware_folder+"OOPAOInterface.py", config, N)
# sim.launch()

sim = OOPAOInterface(conf=conf, param=None)
wfs, dm, psf = sim.get_hardware()

# %% Launch slopes
slopes = hardwareLauncher(hardware_folder+"SlopesProcess.py", config, N+1)
slopes.launch()

# %% Launch Loop Class
loop = hardwareLauncher(hardware_folder+"./Loop.py", config, N+2)
loop.launch()

# %% Calibrate
if RECALIBRATE == True:

    loop.setProperty("IMFile", "./pwfs_example_IM.npy")
    sim.run("flatten")
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
