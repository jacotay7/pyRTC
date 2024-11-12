#%%imports
import matplotlib.pyplot as plt
from pyRTC import *
from pyRTC.Pipeline import *
from pyRTC.hardware import *
from pyRTC.utils import *
import tqdm


# %% Measure Jitter
TAG = 'predict'
WFS = "SHWFS"
Naps = 22
if WFS == "PYWFS":
    Naps = 64
shm1, _, _ = initExistingShm("wfsRaw")
shm2, _, _ = initExistingShm("wfc2D")
# %% Measure Jitter
N = 1000

shm1WriteTimes = np.empty(N)
shm2WriteTimes = np.empty(N)
shm1Counts = np.empty(N)
shm2Counts = np.empty(N)
for i in tqdm.trange(N):
    #Wait for new write to shm1
    shm1.hold()
    #Get write time
    shm1Counts[i] = shm1.metadata[0]
    shm1WriteTimes[i] = shm1.metadata[1]
    #Wait for new write to shm1
    shm2.hold()
    #Get write time
    shm2Counts[i] = shm2.metadata[0]
    shm2WriteTimes[i] = shm2.metadata[1]

# %%
print(np.min(shm1Counts-shm2Counts), np.max(shm1Counts-shm2Counts))
sysLatency = shm2WriteTimes-shm1WriteTimes
frameShift = 0
while np.mean(sysLatency) < 0:
    frameShift += 1
    sysLatency = shm2WriteTimes[frameShift:]-shm1WriteTimes[:-frameShift]


# Create histogram plot
plt.figure(figsize=(10, 6))
plt.hist(sysLatency, 
         bins=np.logspace(-4,-2.5,200), 
         log=True, 
         color = 'k', 
         histtype='step', 
         density=False )
x = np.percentile(sysLatency, 99)
plt.axvline(x = x,
             color = 'green', 
             label = f'1 in 100 > {1e6*x:.0f}us')
x = np.percentile(sysLatency, 99.9)
plt.axvline(x = x,
             color = 'orange',
             label = f'1 in 1,000 > {1e6*x:.0f}us')
x = np.percentile(sysLatency, 99.99)
plt.axvline(x = x,
             color = 'red',
             label = f'1 in 10,000 > {1e6*x:.0f}us')
# Set log scale for both axes
plt.xscale('log')
plt.yscale('log')

# Set xticks with custom labels
xticks = [1e-4, 5e-4, 1e-3, 5e-3]#, 1e-2]
xtick_labels = ['100µs','500us', '1ms', '5ms']#, '10ms']
plt.xticks(xticks, xtick_labels)

plt.xlabel('System Latency', size = 16)
plt.ylabel('Counts', size = 16)
plt.title(f'pyRTC Latency ({Naps}x{Naps} {WFS}, 11x11 DM)', size = 18)
plt.ylim(0.5,N/2)
plt.xlim(np.min(xticks)*0.9,np.max(xticks)*1.1)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(f"jitter_{WFS}_{TAG}.pdf")
plt.show()
# %%
# wfs.run("stop")
# wfs.run("setExposure", 1000)
# wfs.run("start")
