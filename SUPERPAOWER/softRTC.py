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
# shms = ["wfc", "wfc2D","psfShort", "psfLong", "temp"]
# clear_shms(shms)
# %% Load Config
conf = read_yaml_file("/home/whetstone/SUPERPAOWER/pyRTC/SUPERPAOWER/config.yaml")
# %% Power Meter PSF Cam
# power = powerMeter(conf["power"])
# power.start()
# %% Photodetector PSF Cam
# power = photoDetector(conf["photodetect"])
# power.start()
# %% QHYCCD PSF Cam
power = QHYCCD(conf["psf"])
power.start()
# %%
conf = read_yaml_file("/home/whetstone/SUPERPAOWER/pyRTC/SUPERPAOWER/config.yaml")
# for i in range (5,8):
#     try:
#         conf["wfc"]["serialPort"] = f'/dev/ttyUSB{i}'
#         print(conf["wfc"]["serialPort"])
#         wfc = SUPERPAOWER(conf["wfc"])
#         time.sleep(1)
#     except:
#         print(f"Wrong port? {i}")

conf["wfc"]["serialPort"] = f'/dev/ttyUSB2'
wfc = SUPERPAOWER(conf["wfc"])
time.sleep(1)
wfc.start()
# %%
loop = PSGDLoop(conf)
time.sleep(1)
loop.norm = np.mean(power.readLong())
# loop.norm = np.mean(power.powerShm.read())

#%%Set-up Loop
loop.rate = 120
loop.amp = 0.15
loop.useLong = True
power.integrationLength = 10
# loop.norm = np.mean(power.readLong())
loop.flatten()
#%%Start Loop
loop.start()

#%%Start Loop
loop.stop()
loop.flatten()
wfc.flatten()
#%%
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from pyRTC.hardware.optimizerController import controlOptim
optim = controlOptim(conf["optimizer"], loop, power)
optim.min=-1
optim.max=1
power.integrationLength = 10
optim.numSteps = 100000
optim.optimize()
#%%
optim = psgdOptimizer(conf["optimizer"], loop, power)

optim.numSteps = 100

optim.gdTime = 2
optim.avgTime = 3
optim.relaxTime = 5
optim.numFlats = 10
optim.maxAmp = 0.3
optim.minAmp = 0.03
optim.maxRate = 10
optim.minRate = 1
optim.minIntegrate = 1
optim.maxIntegrate = 1

optim.optimize()

#%% RAMP

def ramp(mode, start, stop, res = 10e-3):

    ramp = np.arange(start, stop, res)
    response = np.zeros_like(ramp)
    correction = np.zeros(wfc.numModes, dtype=np.float32)
    for i, volt in enumerate(ramp):
        correction[mode] = volt
        droppedFrames = wfc.numDroppedFrames
        wfc.write(correction)
        time.sleep(1e-3)
        if wfc.numDroppedFrames > droppedFrames:
            print("DROPPING FRAMES")
        response[i] = np.mean(power.readLong())
    response -= np.mean(response)
    # response /= np.max(np.abs(response))
    return response

def hardFlat(N=5, sleepTime=1e-2):
    for k in range(N):
        wfc.flatten()
        time.sleep(sleepTime/N)

# %% Computes Mode reponses
from tqdm import tqdm
VOLT_RANGE = (0,5) #Absolute voltage
res = 5e-3 #in volts
N = 1 #Number of trials
rampVolt = np.arange(VOLT_RANGE[0], VOLT_RANGE[1], res)
responses = np.zeros((wfc.numModes, rampVolt.size))
# power.setExposure(100)
power.integrationLength = 5

for j in range(N):
    for i in tqdm(range(responses.shape[0]), desc=f"Testing Actuators"):
        hardFlat(sleepTime=3)
        #Compute effective 0 volts relative to flat
        start = wfc.flat[i]*-1
        #Now set the start to the desired voltage
        start += VOLT_RANGE[0]
        #Stop is the start + the size of the range
        stop = start + (VOLT_RANGE[1] - VOLT_RANGE[0])

        responses[i] += ramp(i,start, stop, res=res)
responses /= N

# %% Plot Responses:
for i in range(responses.shape[0]):
    responses[i] -= np.mean(responses[i])
np.save("rampResponseWithAtm", responses)
plt.imshow(responses, 
           aspect="auto", 
           cmap='inferno', 
           interpolation="none", 
           extent=[VOLT_RANGE[0],VOLT_RANGE[1],wfc.numModes, 0])
plt.xlabel("Voltage", size = 18)
plt.ylabel("Actuator", size = 18)
plt.title("Modal", size = 18)
plt.colorbar()
plt.show()

zonal_resp = wfc.M2C@responses

plt.imshow(zonal_resp, 
           aspect="auto", 
           cmap='inferno', 
           interpolation="none", 
           extent=[VOLT_RANGE[0],VOLT_RANGE[1],wfc.numActuators, 0])
plt.xlabel("Voltage", size = 18)
plt.ylabel("Actuator", size = 18)
plt.title("Zonal", size = 18)
plt.colorbar()
plt.show()
#%%
import numpy as np
from scipy.signal import butter, filtfilt
from numpy.polynomial import Polynomial

# Function to apply a low-pass Butterworth filter
def low_pass_filter(data, cutoff_freq, sample_rate, order=5):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# Function to fit and subtract a 3D polynomial
def remove_3d_polynomial(y, degree=3):
    x = np.arange(y.size)
    # Fit a polynomial of the specified degree
    poly_fit = np.poly1d(np.polyfit(x,y, degree))
    # Evaluate the polynomial fit
    fitted_values = poly_fit(x)
    # Subtract the polynomial fit from the original data
    data_no_poly = y - fitted_values

    return data_no_poly
filt_resp = np.array([row for row in zonal_resp])
filt_resp = np.array([low_pass_filter(row, 0.07, 1) for row in zonal_resp])
filt_resp = np.array([remove_3d_polynomial(row,degree = 4) for row in filt_resp])

plt.imshow(filt_resp, 
           aspect="auto", 
           cmap='inferno', 
           interpolation="none", 
           extent=[VOLT_RANGE[0],VOLT_RANGE[1],wfc.numActuators, 0])
plt.xlabel("Voltage", size = 18)
plt.ylabel("Actuator", size = 18)
plt.title("Zonal", size = 18)
plt.colorbar()
plt.show()

colors = plt.get_cmap('tab20').colors
filt_resp /= np.std(filt_resp)
plt.figure(figsize=(12,6))
deadThreshold = np.max(np.abs(filt_resp))/7
diffs = []
deadActuators = []
for i in range(zonal_resp.shape[0]):
    diff = np.max(filt_resp[i]) - np.min(filt_resp[i])
    diffs.append(diff)
    if  diff > deadThreshold:
        plt.plot(rampVolt, filt_resp[i], label = f'Act #{i}', 
                color = colors[i], alpha = 0.7)
    else:
        deadActuators.append(i)
plt.xlabel("Voltage", size = 18)
plt.ylabel("Response", size = 18)
plt.legend()
plt.show()
print(f"Dead Actuators: {deadActuators}")
#%%
plt.hist(diffs,bins = np.linspace(0,max(diffs), 100))
plt.axvline(x=deadThreshold, color = 'red')
plt.show()
#%%Test Histeris
DV = 1.0
# LOW = wfc.currentCorrection #-wfc.flat + 2
LOW = np.zeros_like(wfc.currentShape)+1.5
HIGH = LOW + DV

FREQ = 200  #Hz

dt = 1/FREQ/2
TIME = 10
start = time.time()
temp = []
pwr = []
cmds = [LOW, HIGH]
i = 0
tempShm = initExistingShm("temp")[0]
pwfShm = initExistingShm("power")[0]
while i < TIME/dt:
    # temp.append(np.max(tempShm.read()))
    pwr.append(np.max(pwfShm.read()))
    elapsedTime = time.time()-start
    if elapsedTime > dt:
        wfc.write(cmds[i%2])
        start = time.time()
        # print(["LOW","HIGH"][i%2])
        i += 1
wfc.write(LOW)
temp = np.array(temp)
pwr = np.array(pwr)
#%%
normTemp = (temp-np.mean(temp))#/np.std(temp)
normPwr = (pwr-np.mean(pwr))#/np.std(pwr)
# normPwr = low_pass_filter(normPwr, 4e-4, TIME/len(normPwr), order=5)
# plt.plot(normTemp)
plt.plot(normPwr)
plt.show()
print(np.percentile(normPwr, 80) - np.percentile(normPwr, 20) )
#%% RAMP SINGLE ACTUATOR

acts = [1]
RAMP_START = 0 # Volt
RAMP_STOP = 6 # Volt
RESOLUTION = 30e-3 #Volt

ramp = np.arange(RAMP_START, RAMP_STOP, RESOLUTION)
response = np.empty_like(ramp)
nullresponse = np.empty_like(ramp)

correction = np.zeros_like(wfc.read())
power.readLong()
for i, volt in enumerate(ramp):
    for act in acts:
        correction[act] = volt
    wfc.write(correction)
    response[i] = np.mean(power.readLong())
    correction *= 0

for i in range(100):
    wfc.write(correction)
    time.sleep(0.01)
time.sleep(30)
power.readLong()
for i, volt in enumerate(ramp):
    nullresponse[i] = np.mean(power.readLong())

plt.plot(ramp, response)
plt.plot(ramp, nullresponse)
plt.show()

#%% PSD
plt.figure(figsize=(10, 6))
voltage = np.arange(0, 8, res)
for i in range(zonal_resp.shape[0]):
    # Perform FFT to calculate the frequency components
    fft_result = np.fft.fft(zonal_resp[i])
    frequencies = np.fft.fftfreq(len(voltage), d=(voltage[1] - voltage[0]))

    # Calculate the power spectrum (magnitude squared of the FFT result)
    power_spectrum = np.abs(fft_result) ** 2

    # Plot the power spectrum

    plt.plot(frequencies[:len(frequencies)//2], power_spectrum[:len(power_spectrum)//2])  # Only positive frequencies
    plt.title('Power Spectrum of the Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.grid(True)
plt.show()

#%% Plot Power Meter Noise
# power.setExposure(100)

N = int(1e3)
response = np.empty(N)
for i in range(N):
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
FREQ = 10 # Hz
delay = 1/(2*FREQ)

# act = 2
wfcShm, _, _ = initExistingShm("wfc")

correction = np.zeros_like(wfcShm.read_noblock())

BIN = True
HEIGHT = 3
LOW = 1
HIGH = LOW + HEIGHT

correction[:] = LOW
while True:
    if BIN:
        correction[:] = HIGH
    else:
        correction[:] = LOW
    BIN = not BIN
    wfcShm.write(correction)
    time.sleep(delay)



# %%
basis = np.load("./calib/m2c_4x4_paower.npy")
basis = basis.reshape(4,4,16)

#Zero out corners if desired
basis[0,0,:] = 0
basis[0,-1,:] = 0
basis[-1,0,:] = 0
basis[-1,-1,:] = 0
for i in range(basis.shape[-1]):
    mode = basis[:,:,i]
    mode /= np.max(np.abs(mode))
    plt.imshow(mode, cmap='inferno')
    plt.show()


u,s,v = np.linalg.svd(basis.reshape(16,16))
# s = [s[i,i] for i in range(16)]
plt.plot(s)
plt.show()
# %% RESET LOOP
loop.stop()
time.sleep(1)
loop.flatten()
loop.currentShape *= 0
# %% overcurrent tests stuff
m = 1000
for j in range (16):
    for i in range(2*m):
        wfc.flat[j] = i/m
        wfc.flatten()
        wfc.flatten()
        time.sleep(0.001)
        #print(wfc.currentShape)
        print(j, wfc.flat[j])
    wfc.flat[j] = 0
    wfc.flatten()
    wfc.flatten()    
    time.sleep(1)

m = 10
for j in range(50):
    for i in range(5*m):
        wfc.flat = np.zeros_like(wfc.currentShape)+i/m
        #for k in range(1):
        wfc.flatten()
            #time.sleep(0.0015)
        time.sleep(0.0015)
    #time.sleep(0.5)
    print(j)

m = 100
for j in range(50):
    for i in range(5*m):
        wfc.flat = np.mod(i,2) * (np.zeros_like(wfc.currentShape)+5)
        #for k in range(1):
        wfc.flatten()
            #time.sleep(0.0015)
        time.sleep(0.005)
    #time.sleep(0.5)
    print(j)
#%% Record data
def recordForNSeconds(T):
    start = time.time()
    pwr = []
    pwfShm = initExistingShm("psfShort")[0]
    while time.time()-start < T:
        pwr.append(np.max(pwfShm.read()))
    return np.array(pwr)
loop.start()
CL_data = recordForNSeconds(120)
loop.stop()
loop.flatten()
wfc.flatten()
OL_data = recordForNSeconds(120)
# %%
np.save("cl_data",CL_data)
np.save("ol_data",OL_data)
# %%
CL_data_filtered = low_pass_filter(CL_data, 3e-6, 1/300)
OL_data_filtered = low_pass_filter(OL_data,  3e-6, 1/300)
plt.plot(CL_data_filtered, color = 'red')
plt.axhline(y= np.mean(CL_data), color = 'red', linestyle = '--')
plt.axhline(y= np.mean(OL_data), color = 'black', linestyle = '--')
plt.plot(OL_data_filtered, color = 'black')
plt.show()
print(np.mean(CL_data)/np.mean(OL_data))
# %%
