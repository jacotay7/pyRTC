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
power.noiseThrehold = 3
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
wfc.maxVoltage = 3.5
wfc.minVoltage = 0.5
# wfc.flat = np.zeros_like(wfc.flat) + (wfc.maxVoltage + wfc.minVoltage)/2
# %%
loop = PSGDLoop(conf["loop"])
time.sleep(1)
loop.norm = 20 #np.mean(power.readLong())
# loop.norm = np.mean(power.powerShm.read())

#%%Set-up Loop
loop.rate = 3.5 # 3
loop.amp = 0.08 #0.2
loop.useLong = True
power.integrationLength = 3 #7
loop.gradientDamp = 5e-3
# loop.norm = np.mean(power.readLong())
loop.flatten()
#%%Set-up Loop for no screen
loop.rate = 2.2
loop.amp = 0.03
loop.useLong = True
power.integrationLength = 10
loop.gradientDamp = 2e-2
# loop.norm = np.mean(power.readLong())
loop.flatten()
#%%Start Loop
loop.start()

#%%Start Loop
loop.stop()
loop.flatten()
wfc.flatten()
# wfc.write(-wfc.flat)
#%%
# import optuna
# optuna.logging.set_verbosity(optuna.logging.WARNING)
# from pyRTC.hardware.optimizerController import controlOptim
# optim = controlOptim(conf["optimizer"], loop, power)
# optim.min=-1
# optim.max=1
# power.integrationLength = 10
# optim.numSteps = 100000
# optim.optimize()
#%%
optim = psgdOptimizer(conf["optimizer"], loop, power)

optim.numSteps = 300

optim.gdTime = 0.1
optim.avgTime = 5
optim.relaxTime = 3
optim.numFlats = 10
optim.maxAmp = 0.3
optim.minAmp = 0.001
optim.maxRate = 5
optim.minRate = 0.1
optim.minIntegrate = 3
optim.maxIntegrate = 7

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
VOLT_RANGE = (wfc.minVoltage,wfc.maxVoltage) #Absolute voltage
res = 5e-3 #in volts
N = 1 #Number of trials
rampVolt = np.arange(VOLT_RANGE[0], VOLT_RANGE[1], res)
responses = np.zeros((wfc.numModes, rampVolt.size))
# power.setExposure(100)
power.integrationLength = 10

for j in range(N):
    for i in tqdm(range(responses.shape[0]), desc=f"Testing Actuators"):
        # hardFlat(sleepTime=10)
        wfc.write(-wfc.flat)
        #Compute effective 0 volts relative to flat
        start = wfc.flat[i]*-1
        #Now set the start to the desired voltage
        start += VOLT_RANGE[0]
        #Stop is the start + the size of the range
        stop = start + (VOLT_RANGE[1] - VOLT_RANGE[0])

        responses[i] += ramp(i,start, stop, res=res)
responses /= N

# %% Compute Mode response
from tqdm import tqdm
wfc.minVoltage = 0.5
wfc.maxVoltage = 3
VOLT_RANGE = (-0.5,0.5) #Absolute voltage
res = 5e-3 #in volts
N = 1 #Number of trials
rampVolt = np.arange(VOLT_RANGE[0], VOLT_RANGE[1], res)
responses = np.zeros((wfc.numModes, rampVolt.size))
power.integrationLength = 10
N = 1
for j in range(N):
    for i in tqdm(range(responses.shape[0]), desc=f"Testing Actuators"):
        responses[i] += ramp(i, VOLT_RANGE[0], VOLT_RANGE[1], res=res)
responses /= N

# %% Plot Responses:
responses[np.isnan(responses)] = 0
for i in range(responses.shape[0]):
    responses[i] -= np.mean(responses[i])
np.save("rampResponse", responses)
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

zonal_resp = responses #wfc.M2C@responses

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
def remove_nd_polynomial(y, degree=3):
    x = np.arange(y.size)
    # Fit a polynomial of the specified degree
    poly_fit = np.poly1d(np.polyfit(x,y, degree))
    # Evaluate the polynomial fit
    fitted_values = poly_fit(x)
    # Subtract the polynomial fit from the original data
    data_no_poly = y - fitted_values

    return data_no_poly
filt_resp = np.array([row for row in zonal_resp])
filt_resp = np.array([low_pass_filter(row, res/5, res) for row in zonal_resp])
# filt_resp = np.array([remove_nd_polynomial(row,degree = 4) for row in filt_resp])
# filt_resp = np.array([row - np.mean(row[:100]) for row in filt_resp])
plt.imshow(filt_resp, 
           aspect="auto", 
           cmap='inferno', 
           interpolation="none", 
           extent=[VOLT_RANGE[0],VOLT_RANGE[1],wfc.numActuators, 0],
           vmin = -3,
           vmax = 3)
plt.xlabel("Voltage", size = 18)
plt.ylabel("Actuator", size = 18)
plt.title("Zonal", size = 18)
plt.colorbar()
plt.show()

colors = plt.get_cmap('tab20').colors
# filt_resp /= np.std(filt_resp)
plt.figure(figsize=(12,6))
deadThreshold = np.max(np.abs(filt_resp))/20
diffs = []
deadActuators = []
bounds = 50
for i in range(zonal_resp.shape[0]):
    diff = np.max(filt_resp[i][bounds:-bounds]) - np.min(filt_resp[i][bounds:-bounds])
    diffs.append(diff)
    if  diff > deadThreshold:
        plt.plot(rampVolt[bounds:-bounds], filt_resp[i][bounds:-bounds], label = f'Act #{i}', 
                color = colors[i], alpha = 0.7)
    else:
        deadActuators.append(i)
plt.xlabel("Voltage", size = 18)
plt.ylabel("Response", size = 18)
plt.legend()
plt.show()
print(f"Dead Actuators: {deadActuators}")
#Should be [8,9,12,15]
#%%
plt.hist(diffs,bins = np.linspace(0,max(diffs), 100))
plt.axvline(x=deadThreshold, color = 'red')
plt.show()
#%%Test Histeris
DV = 0.6

FREQ = 10  #Hz
wfc.startClock(FREQ, DV, checkerboard=True)

pwr = []
pwfShm = initExistingShm("power")[0]

LENGTH = 10 #seconds
start = time.time()
while time.time()-start < LENGTH:
    pwr.append(np.max(pwfShm.read()))

wfc.stopClock()
wfc.flatten()
pwr = np.array(pwr)
#%%
normPwr = (pwr-np.mean(pwr))#/np.std(pwr)
plt.plot(normPwr[-200:])
plt.show()

# Compute the FFT of the signal
fft_values = np.fft.fft(normPwr)
FRAME_RATE = 404 #FPS
frequencies = np.fft.fftfreq(len(fft_values), 1/FRAME_RATE)

# Take the magnitude of the FFT values to get the power spectrum
power_spectrum = np.abs(fft_values) ** 2

# Plot the positive half of the power spectrum (frequencies above 0)
positive_frequencies = frequencies[:len(frequencies)//2]
positive_power_spectrum = power_spectrum[:len(power_spectrum)//2]

plt.figure(figsize=(10, 6))
plt.plot(positive_frequencies, positive_power_spectrum)
plt.title("Power Spectrum of the Time Series")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.grid(True)
plt.xlim(0, 2*FREQ)
plt.show()
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
FREQ = 100 # Hz
delay = 1/(2*FREQ)

# act = 2
wfcShm, _, _ = initExistingShm("wfc")

correction = np.zeros_like(wfcShm.read_noblock())

BIN = True
HEIGHT = 0.1
LOW = wfc.flat*0
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
    break
#%%
def recordPower(length):
    pwfShm = initExistingShm("power")[0]
    ret = []
    start = time.time()
    while time.time()-start < length:
        ret.append(np.max(pwfShm.read()))
        time.sleep(1e-1)
    ret = np.array(ret)
    return ret
# %% RECORD BASELINE 
import numpy as np
import matplotlib.pyplot as plt

def smooth_signal(signal, window_size):
    """Smooth the signal using a moving average filter."""
    return np.convolve(signal, np.ones(window_size)/window_size, mode='same')

recordTime = 20*60 # [sec]
# Baseline measurement
input("Prepare the system for the baseline measurement. Press Enter when ready.")
loop.stop()
loop.flatten()
wfc.flatten()
baseline_power = recordPower(recordTime)

# Measurement without the phase screen
input("Add the phase screen to the system. Press Enter when ready.")
loop.stop()
loop.flatten()
off_power = recordPower(recordTime)

# Measurement with the phase screen
input("Reset the phase screen. Press Enter when ready.")
loop.start()
on_power = recordPower(recordTime)

loop.stop()
loop.flatten()
wfc.flatten()

# Save the data with string keys separating the arrays
data_dict = {
    'baseline': baseline_power,
    'on': on_power,
    'off': off_power
}
np.savez('power_measurements.npz', **data_dict)
#%%
# Normalize the 'on' and 'off' signals by the mean of the baseline
mean_baseline = np.mean(baseline_power)
on_power_norm = on_power / mean_baseline
off_power_norm = off_power / mean_baseline

# Smooth the power signals
window_size = 100  # Adjust the window size for smoothing as needed
on_power_smoothed = smooth_signal(on_power_norm, window_size)
off_power_smoothed = smooth_signal(off_power_norm, window_size)

# Compute the averages of the smoothed signals
mean_on = np.mean(on_power_smoothed)
mean_off = np.mean(off_power_smoothed)

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(on_power_smoothed, label='PSGD on')
plt.plot(off_power_smoothed, label='PSGD off')

# Add dotted horizontal lines for the averages
plt.axhline(mean_on, color='blue', linestyle='--', label='Mean With PSGD on')
plt.axhline(mean_off, color='orange', linestyle='--', label='Mean With PSGD off')

# Add labels, title, legend, and grid
plt.xlabel('Sample Index')
plt.ylabel('Strehl Ratio')
plt.title('Smoothed Power Signal Comparison')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()

#%% Plot SR without smoothing
mean_on = np.mean(on_power_norm)
mean_off = np.mean(off_power_norm)
# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(on_power_norm, label='PSGD on')
plt.plot(off_power_norm, label='PSGD off')

# Add dotted horizontal lines for the averages
plt.axhline(mean_on, color='blue', linestyle='--', label='Mean With PSGD on')
plt.axhline(mean_off, color='orange', linestyle='--', label='Mean With PSGD off')

# Add labels, title, legend, and grid
plt.xlabel('Sample Index')
plt.ylabel('Strehl Ratio')
plt.title('Power Signal Comparison')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()


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
import pandas as pd

def save_study_to_csv(study, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    trials = study.trials
    data = []
    for t in trials:
        entry = {'trial_id': t.number, 'objective_value': t.value}
        entry.update(t.params)
        data.append(entry)
    df = pd.DataFrame(data)
    df.to_csv(f"{directory}/study_data.csv", index=False)

def load_study_data_from_csv(directory):
    df = pd.read_csv(f"{directory}/study_data.csv")
    return df

import seaborn as sns
import matplotlib.pyplot as plt
import itertools

def plot_parameter_heatmaps(df, objective_column='objective_value', bins=10):
    """
    Plot heatmaps of parameter pairs showing the objective values.

    Parameters:
    - df: pandas DataFrame containing the study data.
    - objective_column: str, name of the objective value column.
    - bins: int, number of bins for continuous parameters.
    """
    # Identify parameter columns
    parameter_columns = [col for col in df.columns if col not in ['trial_id', objective_column]]

    # Generate all unique pairs of parameters
    parameter_pairs = list(itertools.combinations(parameter_columns, 2))

    for (param_x, param_y) in parameter_pairs:
        df_plot = df.copy()

        # Bin continuous parameters
        for param in [param_x, param_y]:
            if pd.api.types.is_numeric_dtype(df_plot[param]):
                df_plot[param] = pd.cut(df_plot[param], bins=bins)

        # Create pivot table
        pivot_table = df_plot.pivot_table(
            index=param_y,
            columns=param_x,
            values=objective_column,
            aggfunc='mean'
        )

        # Ensure the axes are sorted
        pivot_table = pivot_table.sort_index().sort_index(axis=1)

        # Plot the heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            pivot_table,
            annot=True,
            fmt=".2f",
            cmap='viridis',
            cbar_kws={'label': objective_column}
        )
        plt.title(f'Heatmap of {objective_column} for {param_x} vs {param_y}')
        plt.xlabel(param_x)
        plt.ylabel(param_y)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

# %%
