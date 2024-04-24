# %% IMPORTS
#Import pyRTC classes
from pyRTC.Pipeline import *
from pyRTC.utils import *
import os

os.chdir("/home/revoltuser/pyRTC/pyRTC")
RECALIBRATE = False

# %% Clear SHMs
# shm_names = ["wfs", "wfsRaw", "wfc", "wfc2D", "signal", "signal2D", "psfShort", "psfLong"] #list of SHMs to reset
# clear_shms(shm_names)

# %% IMPORTS
config = '../REVOLT/config.yaml'
N = np.random.randint(3000,6000)
# %% Launch DM
wfc = hardwareLauncher("./hardware/ALPAODM.py", config, N)
wfc.launch()

# %% Launch WFS
wfs = hardwareLauncher("./hardware/fliCBlueOneWFS.py", config, N+1)
wfs.launch()

# %% Launch slopes
slopes = hardwareLauncher("./SlopesProcess.py", config, N+2)
slopes.launch()

# %% Launch Loop Class
loop = hardwareLauncher("./Loop.py", config, N+4)
loop.launch()

# %% Calibrate

if RECALIBRATE == True:

    slopes.setProperty("refSlopesFile", "")
    slopes.run("loadRefSlopes")
    ##### slopes.setProperty("offsetY", 3)

    input("Sources Off?")
    wfs.run("takeDark")
    wfs.setProperty("darkFile", "/home/revoltuser/pyRTC/REVOLT/dark.npy")
    wfs.run("saveDark")
    time.sleep(1)
    input("Sources On?")
    input("Is Atmosphere Out?")

    slopes.run("computeImageNoise")
    slopes.run("takeRefSlopes")
    slopes.setProperty("refSlopesFile", "/home/revoltuser/pyRTC/REVOLT/ref.npy")
    slopes.run("saveRefSlopes")
    wfc.run("flatten")

    #  STANDARD IM
    loop.setProperty("IMMethod", "push-pull")
    loop.setProperty("pokeAmp", 0.02)
    loop.setProperty("numItersIM", 100)
    loop.setProperty("IMFile", "/home/revoltuser/pyRTC/REVOLT/IM.npy")
    wfc.run("flatten")
    loop.run("computeIM")
    loop.run("saveIM")
    wfc.run("flatten")
    time.sleep(1)

    input("Is Atmosphere In?")
    #  DOCRIME OL
    loop.setProperty("IMMethod", "docrime")
    loop.setProperty("delay", 1)
    loop.setProperty("pokeAmp", 8e-3)
    loop.setProperty("numItersIM", 10000)
    loop.setProperty("IMFile", "/home/revoltuser/pyRTC/REVOLT/docrime_IM.npy")
    wfc.run("flatten")
    loop.run("computeIM")
    loop.run("saveIM")
    wfc.run("flatten")
    time.sleep(1)


# %% Adjust Loop
loop.setProperty("IMFile", "/home/revoltuser/pyRTC/REVOLT/IM.npy")
loop.run("loadIM")
time.sleep(0.5)
loop.setProperty("numDroppedModes", 90)
loop.run("computeCM")
time.sleep(0.5)
loop.run("setGain",1e-2)
loop.setProperty("leakyGain", 1e-2)
# %%Launch Loop for 5 seconds
wfc.run("flatten")
loop.run("start")
time.sleep(1)
# %% Stop Loop
loop.run("stop")
wfc.run("flatten")
wfc.run("flatten")

# %% Plots
im = np.load("../REVOLT/docrime_IM.npy")
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


im_dc = np.load("../REVOLT/docrime_IM.npy")
im = np.load("../REVOLT/IM.npy")

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

im_dc = np.load("../REVOLT/docrime_IM.npy")
im = np.load("../REVOLT/IM.npy")
im_sprint = np.load("../REVOLT/sprint_IM.npy")

#RESCALES SPRINT TO MATCH EMPIRICAL
# for i in range(im_sprint.shape[1]):
#     im_sprint[:,i] *= np.std(im[:,i])/np.std(im_sprint[:,i])
# np.save("../REVOLT/sprint_IM.npy", im_sprint)

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

