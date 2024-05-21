# %% IMPORTS
#Import pyRTC classes
from pyRTC.Pipeline import *
from pyRTC.utils import *
import os
os.chdir("/home/jtaylor/pyRTC/MAPS")
RECALIBRATE = False

# %% Clear SHMs
# from pyRTC.Pipeline import clear_shms
# shm_names = ["signal"]
# shm_names = ["wfs", "wfsRaw", "wfc", "wfc2D", "signal", "signal2D", "psfShort", "psfLong"] #list of SHMs to reset
# clear_shms(shm_names)

# %% IMPORTS
config = '/home/jtaylor/pyRTC/MAPS/config.yaml'
N = np.random.randint(3000,6000)

# %% Launch WFS
wfs = hardwareLauncher("../pyRTC/hardware/ImageStreamIOWfs.py", config, N+1)
wfs.launch()

# %% Launch slopes
slopes = hardwareLauncher("../pyRTC/SlopesProcess.py", config, N+2)
slopes.launch()

# %% Launch DM
wfc = hardwareLauncher("../pyRTC/hardware/ImageStreamIOWfc.py", config, N)
wfc.launch()

# %% Launch Loop Class
loop = hardwareLauncher("../pyRTC/Loop.py", config, N+4)
loop.launch()

# %% NCAP OPTIMIZER
optim  = hardwareLauncher("../pyRTC/hardware/NCPAOptimizer.py", config, N+4)
optim.launch()

# %% Calibrate
if RECALIBRATE == True:

    input("Sources Off?")
    wfs.run("takeDark")
    wfs.setProperty("darkFile", "/home/jtaylor/pyRTC/MAPS/calib/dark.npy")
    wfs.run("saveDark")

    input("ON TARGET?")

    wfc.run("flatten")

    input("Is Atmosphere In?")
    #  DOCRIME OL
    loop.setProperty("IMMethod", "docrime")
    loop.setProperty("delay", 1)
    loop.setProperty("pokeAmp", 0.25)
    loop.setProperty("numItersIM", 30000)
    loop.setProperty("IMFile", "/home/jtaylor/pyRTC/MAPS/calib/docrime_IM.npy")
    wfc.run("flatten")
    loop.run("computeIM")
    loop.run("saveIM")
    wfc.run("flatten")
    time.sleep(1)


# %% Adjust Loop
loop.setProperty("IMFile", "/home/jtaylor/pyRTC/MAPS/calib/docrime_IM.npy")
loop.run("loadIM")
time.sleep(0.5)
loop.setProperty("numDroppedModes", 0)
loop.run("computeCM")
time.sleep(0.5)
loop.setProperty("leakyGain",1e-2)
loop.setProperty("pGain",1e-3)
loop.setProperty("iGain",0)
loop.setProperty("dGain",0)
loop.setProperty("controlLimits",[-0.1, 0.1])
loop.setProperty("integralLimits",[-0.5, 0.5])

# %%Launch Loop for 5 seconds
wfc.run("flatten")
loop.run("start")
time.sleep(1)
# %% Stop Loop
loop.run("stop")
wfc.run("flatten")
wfc.run("flatten")

# %% Plot IM
IM = np.load("/home/jtaylor/pyRTC/MAPS/calib/docrime_IM.npy")
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

# %% Plots
im = np.load("../MAPS/calib/docrime_IM.npy")
plt.imshow(im, aspect="auto")
plt.show()

im = im.reshape(*slopes.getProperty("signalShape"), -1)
im = np.moveaxis(im, 2, 0)
plt.imshow(np.sum(np.abs(im), axis = 0))
plt.show()

for i in range(85,90):
    plt.imshow(im[i])
    plt.colorbar()
    plt.show()


im_dc = np.load("../MAPS/calib/docrime_IM.npy")
im = np.load("../MAPS/calib/IM.npy")

im_dc = im_dc.reshape(*slopes.getProperty("signalShape"), -1)
im_dc = np.moveaxis(im_dc, 2, 0)
plt.plot(np.std(im_dc, axis = (1,2)))


im = im.reshape(*slopes.getProperty("signalShape"), -1)
im = np.moveaxis(im, 2, 0)
plt.plot(np.std(im, axis = (1,2)))
plt.show()

plt.plot(np.mean(im_dc, axis = (1,2)))
plt.plot(np.mean(im, axis = (1,2)))
plt.show()

plt.imshow(im_dc[-1], aspect="auto")
plt.colorbar()
plt.show()
plt.imshow(im[-1], aspect="auto")
plt.colorbar()
plt.show()
# %% SVD

im_dc = np.load("../MAPS/calib/docrime_IM.npy")
im = np.load("../MAPS/calib/IM.npy")
im_sprint = np.load("../MAPS/calib/sprint_IM.npy")

#RESCALES SPRINT TO MATCH EMPIRICAL
# for i in range(im_sprint.shape[1]):
#     im_sprint[:,i] *= np.std(im[:,i])/np.std(im_sprint[:,i])
# np.save("../MAPS/calib/sprint_IM.npy", im_sprint)

u,s,v = np.linalg.svd(im)
plt.plot(s/np.max(s), label = 'EMPIRICAL')
u,s,v = np.linalg.svd(im_dc)
plt.plot(s/np.max(s), label = 'DOCRIME')
u,s,v = np.linalg.svd(im_sprint)
plt.plot(s/np.max(s), label = 'SPRINT')
plt.yscale("log")
plt.ylim(1e-3,1.5)
plt.xlabel("Eigen Mode #", size = 18)
plt.ylabel("Normalizaed Eigenvalue", size = 18)
plt.legend()
plt.show()

plt.plot(np.std(im, axis = 0), label = 'EMPIRICAL')
plt.plot(np.std(im_dc, axis = 0), label = 'DOCRIME')
plt.plot(np.std(im_sprint, axis = 0), label = 'SPRINT')
plt.xlabel("Mode #", size = 18)
plt.ylabel("Standard Deviation", size = 18)
plt.legend()
plt.show()


N, M = 0,5
im_sprint = im_sprint.reshape(*slopes.getProperty("signalShape"), -1)
im_sprint = np.moveaxis(im_sprint, 2, 0)
a = np.vstack(im_sprint[N:M])

im_dc = im_dc.reshape(*slopes.getProperty("signalShape"), -1)
im_dc = np.moveaxis(im_dc, 2, 0)
b = np.vstack(im_dc[N:M])

im = im.reshape(*slopes.getProperty("signalShape"), -1)
im = np.moveaxis(im, 2, 0)
c = np.vstack(im[N:M])

plt.imshow(np.hstack([a,b,c]))
plt.show()

plt.imshow(a-b)
plt.show()

# %% Kill everything
# hardware = [slopes, psfCam, wfs, wfc, loop]
# for h in hardware:
#     h.shutdown()
#     time.sleep(1)

# %%
wfc.run("deactivateActuators",[0,1,2,3,4,5,11,12,20,21,31,32,42,43,53,54,64,65,75,76,84,85,91,92,93,94,95,96])
# %%
wfc.run("reactivateActuators",[i for i in range(97)])
# %% Strehl Monitor
psfCam.run("computeStrehl")
print(psfCam.getProperty("strehl_ratio"))
# %%

# %%
from tqdm import tqdm

folder = "~/Downloads/robin-april-16/"

# numModes = 10
# startMode = 0
# endMode = wfc.numModes - 1
filelist = ['cnnx2_phase.npy', 'cnnx4_phase', 'linx2_phase.npy', 'linx4_phase.npy']
N = 4
numModes = 11
RANGE = 2
modelist = np.linspace(-RANGE, RANGE, numModes) #.astype(int)

for ff in filelist:
    psfs = np.empty((numModes, N, *psfCam.getProperty("imageShape")))
    cmd = wfc.read()
    cmds = np.empty((numModes, N, *cmd.getProperty("shape")), dtype=cmd.dtype)
    wfc.flatten()
    time.sleep(0.1)
    d = np.read(f'{folder}/{ff}')
    for i, mode in enumerate(modelist): #range(numModes):
        correction = np.zeros_like(wfc.read())
        for j in range(N):
            correction[mode] = mode * d[j, :].flatten()
            wfc.write(correction)
            #Burn some images
            psfCam.readLong()
            #Save the next PSF in the dataset
            psfs[i, j, :, :] = psfCam.readLong()
            cmds[i,j,:] = correction
            wfc.flatten()
            time.sleep(0.1)
    
    np.save(f'{folder}/psfs_{ff}', psfs)
    np.save(f'{folder}/cmds_{ff}', cmds)
# %%
