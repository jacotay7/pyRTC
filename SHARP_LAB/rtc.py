# %% Imports
import numpy as np
import yaml
import matplotlib.pyplot as plt
#Import pyRTC classes
from pyRTC.Loop import *
#Import hardware classes
from pyRTC.hardware.ALPAODM import *
from pyRTC.hardware.ximeaWFS import *
from pyRTC.hardware.SpinnakerScienceCam import *
# %% clean up
# from pyRTC.Pipeline import clear_shms
# shms = ["wfs","wfc","wfc2D","signal","psfShort","psfLong"]
# clear_shms(shms)
# %% Read Config
def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        conf = yaml.safe_load(file)
    return conf
conf = read_yaml_file("config.yaml")
confLoop = conf["loop"]
confWFS = conf["wfs"]
confWFC = conf["wfc"]
confPSF = conf["psf"]

M2C = np.fromfile("/etc/chai/m2c_kl.dat",dtype=np.float64).reshape(97,95)
for i in range(M2C.shape[1]):
    M2C[:,i] /= np.std(M2C[:,i])
# M2C = np.load("/home/whetstone/m2c_zern_manual.npy")
# %% Initialize Science Camera
psf = spinCam(conf=confPSF)
psf.start()
psf.takeDark()
# # psf.saveDark("./psfDark")
# psf.loadDark("./psfDark.npy")
# psf.plot()
# %% Initialize DM
dm = ALPAODM(confWFC["serial"], 
             functionsToRun = confWFC["functions"],
             flatFile = confWFC["flatFile"], 
             M2C=M2C)
dm.setFlat(np.load("newFlat.npy"))
dm.start()
time.sleep(1e-1)
dm.flatten()
# # %% probe focus
# correction = np.zeros_like(dm.read())
# correction[0] = 0.02
# correction[1] = -0.06
# # for amp in np.linspace(-0.02,0.00,5):
# #     correction[2] = amp
# dm.write(correction)

# %% Initialize WFS
#Create camera object, take an image
wfs = XIMEA_WFS(exposure=confWFS["exposure"],
                functionsToRun = confWFS["functions"], 
                roi=[confWFS["width"],confWFS["height"],confWFS["left"],confWFS["top"]], 
                binning=confWFS["binning"], 
                gain=confWFS["gain"], 
                bitDepth=confWFS["bitDepth"])
pupilLocs = [(int(x.split(',')[1]), int(x.split(',')[0])) for x in confWFS["pupils"]]
wfs.setPupils(pupilLocs, confWFS["pupilsRadius"])
wfs.start()
time.sleep(1e-1)
# wfs.takeDark()
# wfs.saveDark("./dark")
wfs.loadDark("./dark.npy")

# # %%Pupil Finding
# from matplotlib.colors import LogNorm
# pupilLocs = [(int(x.split(',')[1]), int(x.split(',')[0])) for x in confWFS["pupils"]]
# pupilLocs[0] = (pupilLocs[0][0]-1,pupilLocs[0][1])
# pupilLocs[1] = (pupilLocs[1][0]-1,pupilLocs[1][1])
# pupilLocs[2] = (pupilLocs[2][0]-1,pupilLocs[2][1])
# pupilLocs[3] = (pupilLocs[3][0]-1,pupilLocs[3][1])
# r = confWFS["pupilsRadius"] -1
# wfs.setPupils(pupilLocs, r)

# for i in range(4):
#     plt.imshow(wfs.readImage()*wfs.pupilMask, norm = LogNorm(vmin=20,vmax=1000))
#     r = wfs.pupilRadius
#     plt.xlim(wfs.pupilLocs[i][0]-r, wfs.pupilLocs[i][0]+r)
#     plt.ylim(wfs.pupilLocs[i][1]-r, wfs.pupilLocs[i][1]+r)
#     plt.show()
# %%  Inialize Loop
loop = Loop(wfs, dm, functionsToRun=confLoop["functions"])
# %% Compute OL IM
# loop.computeIM(pokeAmp=0.01,
#             #    delay = 1,
#                N=10,
#                hardwareDelay=1e-3,
#                method='push-pull')
# np.save("ppIM", loop.IM)
# loop.computeIM(pokeAmp=0.01,
#                delay = 1,
#                N=10000,
#                hardwareDelay=1e-3,
#                method='docrime')
# np.save("dcIM", loop.IM)
loop.IM = np.load("ppIM.npy")
# for i in range(5):
#     loop.plotIM(row=i)

# %%
# a = np.load("ppIM.npy")
# b = np.load("dcIM.npy")
# plt.plot(np.std(a,axis = 0))
# plt.plot(np.std(b,axis = 0))
# plt.show()

# plt.imshow(a-b, aspect="auto")
# plt.show()
# for i in np.random.choice(np.arange(loop.numModes),5).astype(int):
#     a_row2D = wfs.signal2D(a[:,i])
#     b_row2D = wfs.signal2D(b[:,i])
#     plt.imshow(np.vstack([a_row2D,b_row2D,a_row2D-b_row2D]), cmap = 'inferno', aspect='auto')
#     plt.colorbar()
#     plt.show()
# %% Compute CM
loop.computeCM(rcond=0.1)
plt.imshow(loop.CM, aspect='auto')
plt.show()
dm.flatten()
# %% Run Loop 
dm.flatten()
time.sleep(1e-2)
loop.setGain(0)
loop.start()
time.sleep(100)
loop.stop()
# %% Run Loop 
# dm.flatten()
# time.sleep(1e-2)
# loop.start()
# time.sleep(2)
# loop.computeIM(pokeAmp=0.0002,
#                delay = 2,
#                N=10000,
#                hardwareDelay=1e-3,
#                method='docrime')
# loop.stop()

# # %% Compute OG 
# for i in range(5):#np.random.choice(np.arange(loop.numActiveModes),5).astype(int):
#     loop.plotIM(row=i)
# %% Test some modes
# numModes = 3
# N = 10
# psfs = np.empty((numModes, N, *psf.imageShape))
# dm.flatten()
# time.sleep(0.1)
# #Burn one image to sync up
# psf.psfLong.read(flagInd=4)
# for mode in range(numModes):
#     correction = np.zeros_like(dm.read())
#     n = 0
#     for amp in np.linspace(-0.05,0.05,N):
#         correction[mode] = amp
#         dm.write(correction)
#         time.sleep(1e-3)
#         #Save the next PSF in the dataset
#         psfs[mode, n, :, :] = psf.psfLong.read(flagInd=4)
#         n += 1
# dm.flatten()
# np.save("firstThreeModes", psfs)
# %% Turn into a modulator
# amp = 0.2
# correction = np.zeros_like(dm.read())
# for i in range(50):
#     for t in np.linspace(0,2*np.pi):
#         correction[0] = amp*np.sin(t)
#         correction[1] = amp*np.cos(t)
#         dm.write(correction)
#         time.sleep(1e-3)
# dm.flatten()
# %% Save Shape and Flatten 
# newFlat = dm.currentShape
# np.save("newFlat", newFlat)
# # # %% Exit
# time.sleep(1e-2)
# dm.flatten()
# time.sleep(1e-2)
# %% Clean everything up
print("Cleaning Up")
# psf.__del__()
dm.__del__()
wfs.__del__()
loop.__del__()


# %%
