# %% IMPORTS
#Import pyRTC classes
from pyRTC.Pipeline import *
from pyRTC.utils import *
import os
os.chdir("/home/jtaylor/pyRTC/MAPS")
RECALIBRATE = False

# %% Clear SHMs
# from pyRTC.Pipeline import clear_shms
# # shm_names = ["signal"]
# shm_names = ["wfc", 'og']
# shm_names = ["wfs", "wfsRaw", "wfc", "wfc2D", "signal", "signal2D", "psfShort", "psfLong"] #list of SHMs to reset
# clear_shms(shm_names)

# %% IMPORTS
config = '/home/jtaylor/pyRTC/MAPS/config.yaml'
N = np.random.randint(3000,6000)

# %% Launch WFS
wfs = hardwareLauncher("../pyRTC/hardware/ImageStreamIOWfs.py", config, N)
wfs.launch()

# %% Launch slopes
# slopes = hardwareLauncher("../pyRTC/SlopesProcess.py", config, N+1)
slopes = hardwareLauncher("../pyRTC/hardware/mapsSlopes.py", config, N+1)
slopes.launch()

# %% Launch DM
wfc = hardwareLauncher("../pyRTC/hardware/ImageStreamIOWfc.py", config, N+2)
wfc.launch()

# %% Launch DM
psf = hardwareLauncher("../pyRTC/hardware/ImageStreamIOPsf.py", config, 3142, remoteProcess=True)
psf.host = "mirac.mmto.arizona.edu"
psf.launch()

psf.run("takeDark")
# %% Launch Loop Class
loop = hardwareLauncher("../pyRTC/Loop.py", config, N+4)
loop.launch()

# %% PID OSlopesProcessPTIMIZER
# from pyRTC.hardware.OGOptimizer import *
# optim  = OGOptimizer(read_yaml_file(config)["optimizer"]['og'], loop, psf)
from pyRTC.hardware.PIDOptimizer import *
optim  = PIDOptimizer(read_yaml_file(config)["optimizer"]['pid'], loop, psf)

# %% Telemetry Service
from pyRTC.Telemetry import *
telem  = Telemetry(read_yaml_file(config)["telemetry"])


# %% Calibrate
if False:

    wfc.run("flatten")

    #  DOCRIME OL
    loop.setProperty("IMMethod", "docrime")
    loop.setProperty("delay", 1)
    loop.setProperty("pokeAmp", 0.1)
    loop.setProperty("numItersIM", 100000)
    loop.setProperty("IMFile", "/home/jtaylor/pyRTC/MAPS/calib/docrime_IM.npy")
    wfc.run("flatten")
    loop.run("computeIM")
    loop.run("saveIM")
    wfc.run("flatten")
    time.sleep(1)

    loop.setProperty("IMMethod", "docrime")
    loop.setProperty("delay", 1)
    loop.setProperty("numActiveModes", 0)
    loop.setProperty("clDocrime", True)
    loop.run("start")
    
    loop.run("stop")
    wfc.run("flatten")
    wfc.run("flatten")


#%% deactivate edge
# edgeActuators = list(.astype(int))
# edgeActuators = [int(x) for x in np.arange(275,335)]
# wfc.run("deactivateActuators",
#         edgeActuators)

# %% Adjust Loop
loop.setProperty("IMFile", 
                 "/home/jtaylor/pyRTC/MAPS/calib/docrime_IM.npy",
                #  "/home/jtaylor/pyRTC/MAPS/calib/cl_dc_tmp_IM.npy",
                # "/home/jtaylor/pyRTC/MAPS/calib/50_mode_with_TT.npy",
                 )
loop.run("loadIM")
time.sleep(0.5)
loop.setProperty("numDroppedModes", 30)
# loop.setProperty("delay", 2)
loop.setProperty("controlClipModeStart", 0)
loop.run("computeCM")
time.sleep(0.5)
loop.setProperty("leakyGain",5e-2)
# loop.setProperty("gain",5e-4)
# loop.setProperty("stabilityLimit", 1e-2)
loop.setProperty("controlLimits",[-0.02, 0.02])
loop.setProperty("pGain",0.28)
loop.setProperty("iGain",0.04)
loop.setProperty("dGain",0.03)
# loop.setProperty("integralLimits",[-0.1, 0.1])

# %%Launch Loop for 5 seconds
wfc.run("flatten")
loop.run("start")
loop.setProperty("pokeAmp", 0.02)
# loop.setProperty("numActiveModes", )
# loop.setProperty("clDocrime", False)
# loop.run("start")
loop.setProperty("clDocrime", False)
time.sleep(1)
# %% Stop Loop
loop.run("stop")
for i in range(10):
    wfc.run("flatten")
    wfc.run("flatten")

# %% Optimize PID
optim.metric = "strehl"
optim.numReads =  3
# optim.maxPGain = 0.1
# optim.maxIGain = 0.01
# optim.maxDGain = 0.01
# optim.maxCLim = 0.3
optim.numStep = 10
optim.mode = 'relative'
optim.startMode = 2
optim.endMode = 10
for i in range(1):
    optim.optimize()
optim.applyOptimum()
#%% CL DOCRIME
loop.setProperty("IMMethod", "docrime")
loop.setProperty("delay", 1)
loop.setProperty("pokeAmp", 0.1)
# loop.setProperty("numActiveModes", )
loop.setProperty("clDocrime", True)
loop.run("start")

# %% Save Telemetry
telem.save("wfs", 2000, "slopemask_test_2")

#%% Extend IM
IM = np.load("/home/jtaylor/pyRTC/MAPS/calib/cl_dc_tmp_IM.npy")
newIM = np.zeros((IM.shape[0], 50), dtype = IM.dtype)
newIM[:,:20] = IM
# np.save("/home/jtaylor/pyRTC/MAPS/calib/50_mode_with_TT.npy", newIM)

# %% Plot IM
IM = np.load("/home/jtaylor/pyRTC/MAPS/calib/docrime_IM.npy")
# IM = np.load("/home/jtaylor/pyRTC/MAPS/calib/cl_dc_tmp_IM.npy")
vsa = np.load("/home/jtaylor/pyRTC/MAPS/calib/validSubAps.npy")

x_slopes = vsa[:,:vsa.shape[1]//2]
y_slopes = vsa[:,vsa.shape[1]//2:]


mode = np.zeros(vsa.shape)
modeNum = 2
mode[:,:vsa.shape[1]//2][x_slopes] = IM[:IM.shape[0]//2,modeNum]
mode[:,vsa.shape[1]//2:][y_slopes] = IM[IM.shape[0]//2:,modeNum]

plt.imshow(mode, aspect="auto")
plt.show()

cubeIM = np.zeros((IM.shape[1], *mode.shape))

for i in range(cubeIM.shape[0]):
    mode = np.zeros(vsa.shape)
    modeNum = i
    mode[:,:vsa.shape[1]//2][x_slopes] = IM[:IM.shape[0]//2,modeNum]
    mode[:,vsa.shape[1]//2:][y_slopes] = IM[IM.shape[0]//2:,modeNum]
    cubeIM[i] = mode
# %% Make Movie
from matplotlib.animation import FuncAnimation
def make_movie(data, title="anim.mp4", fps = 10):

    # Create a figure and an axes object
    fig, ax = plt.subplots()

    # Set up the initial frame; initially, we display the first image
    im = ax.imshow(data[0], cmap='inferno', interpolation='none', vmin = np.min(data), vmax = np.max(data))

    # Initialization function: plot the background of each frame
    def init():
        im.set_data(data[0])
        return (im,)

    # Animation function: this is called sequentially
    def animate(i):
        im.set_data(data[i])  # Update the data
        return (im,)

    # Call the animator
    # frames is the number of total frames (equal to the size of the first dimension of the array)
    anim = FuncAnimation(fig, animate, init_func=init, frames=data.shape[0], interval=200, blit=True)

    # To display the animation in a Jupyter notebook
    # from IPython.display import HTML
    # HTML(anim.to_jshtml())

    # To save the animation to a file
    anim.save(title, writer='ffmpeg', fps=fps)

    plt.show()

make_movie(cubeIM, title="docrime_IM.mp4", fps = 2)



# %%
optim.numSteps = 10
optim.maxPGain = 0.5
optim.maxIGain = 0.05
optim.maxDGain = 0.05
optim.numReads = 3

for i in range(5):
    optim.optimize()
optim.applyOptimum()
# %%
