# %% Imports
import sys
orig_stdout = sys.stdout
import numpy as np
import matplotlib.pyplot as plt
from pyRTC import *
from pyRTC.hardware import *
from pyRTC.utils import *
from pyRTC.Pipeline import *
sys.stdout = orig_stdout

#%% CLEAR SHMs
# shms = ["wfc", "wfc2D","psfShort", "psfLong"]
# clear_shms(shms)
# %% Load Config
conf = read_yaml_file("./config.yaml")
# %%
power = powerMeter(conf["power"])
power.start()
# %%
wfc = SUPERPAOWER(conf["wfc"])
wfc.start()
time.sleep(1)

#%% RAMP SINGLE ACTUATOR

ACT = 10
RAMP_START = 0 # Volt
RAMP_STOP = 4 # Volt
RESOLUTION = 30e-3 #Volt

ramp = np.arange(RAMP_START, RAMP_STOP, RESOLUTION)
response = np.empty_like(ramp)

power.readLong()
for i, volt in enumerate(ramp):
    wfc.push(ACT, volt)
    response[i] = power.readLong()

plt.plot(ramp, response)
plt.show()

#%% Plot Power Meter Noise
power.setExposure(100)

response = np.empty(100)
for i in range(100):
    response[i] = power.read()[0][0]
plt.plot(response)
# plt.ylim(0, 1e-7)
plt.show()
print(f"NOISE: {np.std(response)}")

# %%
VOLTLIM = 10
resolution = 30e-3 #Volts

numAct = wfc.numActuators
ramp = np.arange(0, VOLTLIM, resolution)
response = np.zeros_like(ramp)

responses = np.zeros((numAct, *response.shape))

wfcShm, _, _ = initExistingShm("wfc")
powerShm, _, _ = initExistingShm("psfShort")

correction = np.zeros_like(wfcShm.read_noblock())


# for act in range(4):
#     correction *= 0
#     for i, volt in enumerate(ramp):
#         correction[4*act: 4*(act+1)] = volt
#         wfcShm.write(correction)
#         response[i] = powerShm.read()
#         responses[act] = response

for act in range(7,8):#numAct):
    correction *= 0
    for i, volt in enumerate(ramp):
        correction[act] = volt
        wfcShm.write(correction)
        time.sleep(1)
        response[i] = powerShm.read()
        responses[act] = response
        plt.plot(response)
# np.save("/home/whetstone/SUPERPAOWER/pyRTC/SUPERPAOWER/data/responseCurves", responses)

# %%
responses = np.load("/home/whetstone/SUPERPAOWER/pyRTC/SUPERPAOWER/data/responseCurves.npy")
plt.figure(figsize=(12,5))

for act in range(responses.shape[0]):
    if act != 7:
        continue
    plt.plot(ramp,responses[act], label = f"Actuator #{act+1}")

plt.legend()
plt.xlabel("Voltage [V]", size = 18)
plt.ylabel("Power [W]", size = 18)
plt.title("Phase Shifter Ramp Response (4x4)", size = 20)
plt.show()
# %% Make Clock
FREQ = 1 # Hz
delay = 1/FREQ

act = 0
wfcShm, _, _ = initExistingShm("wfc")

correction = np.zeros_like(wfcShm.read_noblock())

BIN = True
VOLT = 1
while True:
    if BIN:
        correction[act] = VOLT
    else:
        correction[act] = 0
    BIN = not BIN
    wfcShm.write(correction)
    time.sleep(delay)



# %%
