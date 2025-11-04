# %%
from pyRTC import *
from pyRTC.utils import *
from pyRTC.Pipeline import *
import numpy as np
from datetime import datetime


# %%
wfsShm, wfsShape, wfsDtype = initExistingShm("wfs")
wfcShm, wfcShape, wfcDtype = initExistingShm("wfc")
savedir = "/media/whetstone/storage/trainingSet"
NUM_FRAME = 20000


def makeTag(WFS, METHOD, info, num_frame):

    tag = info
    for s in [
        WFS,
        METHOD,
        "numframe-" + str(int(num_frame)),
        "date:" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
    ]:
        tag += s + "_"
    tag = tag[:-1]
    tag = tag.replace(".", "_")

    return tag


# %%
def hardWrite(wfcShm, command, N=3, hardwareDelay=1e-3):
    command = command.reshape(wfcShape)
    for n in range(N):
        wfcShm.write(command)
    time.sleep(hardwareDelay)
    return


def flatten(wfcShm):
    hardWrite(wfcShm, np.zeros_like(wfcShm.read_noblock()))
    return


# Function to generate random vectors with values between -1 and 1
def generate_random_vector(size):
    return np.random.uniform(-1, 1, size)


# Function to generate power-law pattern vectors
def generate_powerlaw_vector(size, powerLawCoeffs):
    if len(powerLawCoeffs) == 2:
        a, b = powerLawCoeffs
        b = -1.9046
        x = np.arange(1, size + 1)
        pwr_spectrum = a * np.power(x, b)
    elif len(powerLawCoeffs) > 2:
        pwr_spectrum = powerLawCoeffs
    else:
        raise Exception("Invalid Power law given")

    vector = np.random.normal(0, np.sqrt(pwr_spectrum))

    return vector


# Function to generate clustered random vectors
def generate_clustered_random_vector(size):
    cluster_size_ratio = np.random.uniform(0, 0.3)
    vector = np.zeros(size)

    # Pick a random start position for the perturbation
    perturb_size = int(cluster_size_ratio * size)  # Perturb a portion of the vector
    perturb_size = max(perturb_size, 1)
    start_idx = np.random.randint(0, size - perturb_size)

    # Perturb a random region within the vector while keeping the rest zero
    vector[start_idx : start_idx + perturb_size] = np.random.uniform(
        -1, 1, perturb_size
    )

    return vector


if not os.path.exists(savedir):
    os.makedirs(savedir)


def makeDataSet(powerLaw, tag, size):
    dm_commands = np.memmap(
        savedir + f"/dm_cmds_{tag}.npy",
        dtype="float32",
        mode="w+",
        shape=(size, *wfcShape),
    )
    wfs_frames = np.memmap(
        savedir + f"/wfs_frames_{tag}.npy",
        dtype="float32",
        mode="w+",
        shape=(size, *wfsShape),
    )
    frame = 0

    start = time.time()
    for j in range(size):

        # Flatten
        flatten(wfcShm)

        command = generate_powerlaw_vector(wfcShape[0], powerLaw)
        command = command.reshape(wfcShape)

        hardWrite(wfcShm, command)
        wfs_frames[frame] = wfsShm.read().astype(np.float32)
        dm_commands[frame] = command.astype(np.float32)

        frame += 1

        if frame % 1000 == 0:
            print(f"Generated {frame} samples in {time.time()-start} seconds")
            start = time.time()


powerLaw = [8.35e-6, -1.4049]  # Cl
# powerLaw = [2.24e-2, -1.9255] #OL
powerLaw = y2  # Cl
powerLaw = y  # OL
makeDataSet(y, makeTag("pywfs", "powerLaw", "OL", 200000), 200000)
makeDataSet(y, makeTag("pywfs", "powerLaw", "OL", 20000), 20000)
makeDataSet(y2, makeTag("pywfs", "powerLaw", "CL", 200000), 200000)
makeDataSet(y2, makeTag("pywfs", "powerLaw", "CL", 20000), 20000)
# %%
wfsShm, wfsShape, wfsDtype = initExistingShm("wfs")
wfcShm, wfcShape, wfcDtype = initExistingShm("wfc")
cmd = np.zeros_like(wfcShm.read_noblock())
slopesShm = initExistingShm("signal")[0]
N = 10000
cmds = np.zeros((N, *cmd.shape))
CL_cmds = np.zeros((N, *cmd.shape))
IM = np.load("/home/whetstone/pyRTC/SHARP_LAB/calib/IM_PYWFS.npy")
CM = np.linalg.pinv(IM)
for i in range(N):
    # command = generate_powerlaw_vector(wfcShape[0], scale)
    # hardWrite(wfcShm, command)
    cmds[i] = wfcShm.read()
    CL_cmds[i] = (CM @ slopesShm.read()).reshape(cmd.shape)


# %%
def power_law_fit(x, y):
    """
    Fits a power-law model y = a * x^b to the input data (x, y).

    Parameters:
    - x: Independent variable data (1D array-like)
    - y: Dependent variable data (1D array-like)

    Returns:
    - a: Scaling coefficient of the power law
    - b: Exponent of the power law
    - y_fit: Fitted y-values based on the model
    """
    # Remove zeros or negative values to avoid issues with log transformation
    mask = (x > 0) & (y > 0)
    x_filtered = x[mask]
    y_filtered = y[mask]

    # Take logarithms of the data
    log_x = np.log(x_filtered)
    log_y = np.log(y_filtered)

    # Perform linear regression on the log-transformed data
    coefficients = np.polyfit(log_x, log_y, deg=1)
    b = coefficients[0]
    log_a = coefficients[1]
    a = np.exp(log_a)

    # Generate fitted y-values
    y_fit = a * x**b

    return a, b, y_fit


y = np.squeeze(np.mean(cmds**2, axis=0))
y2 = np.squeeze(np.mean(CL_cmds**2, axis=0))
x = np.arange(y.size).astype(float)
a_fit, b_fit, y_fit = power_law_fit(x, y)
a_fit2, b_fit2, y_fit2 = power_law_fit(x, y2)
# Plot the original power spectrum
plt.scatter(x, y, label="Data", color="blue", s=15)
plt.plot(x, y_fit, label="Power-law Fit (OL)", color="red", linewidth=2)
plt.scatter(x, y2, label="Data", color="blue", s=15)
plt.plot(x, y_fit2, label="Power-law Fit (CL)", color="red", linewidth=2)
plt.yscale("log")
plt.xscale("log")
print(f"Fitted parameters:\n a = {a_fit:.7f}\n b = {b_fit:.4f}")
print(f"Fitted parameters:\n a = {a_fit2:.7f}\n b = {b_fit2:.4f}")
# %%
