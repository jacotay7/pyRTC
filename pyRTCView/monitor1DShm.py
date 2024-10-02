#%% 
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import time
from pyRTC.Pipeline import *
import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Get the SHM name")

# Add a required positional argument for the word
parser.add_argument("shm", type=str, help="name of SHM to plot", default="strehl")

# Parse the arguments
args = parser.parse_args()

#%% 
shmName = args.shm#'psfShort'
shm,_,_ = initExistingShm(shmName)

# Parameters
update_interval = 0.1  # seconds between updates, modify as needed
WINDOW_SIZE = 10
MAX_SIZE = 1000

def rolling_average(data, window_size):
    n = len(data)
    if n < window_size:
        return []  # Return empty list if window is larger than data length

    averages = []
    # Compute the initial window's sum
    window_sum = sum(data[:window_size])
    averages.append(window_sum / window_size)

    # Slide the window across the data
    for i in range(1, n - window_size + 1):
        window_sum += data[i + window_size - 1] - data[i - 1]
        averages.append(window_sum / window_size)

    return averages

# Function to compute the next value in the time series
def compute_next_value():
    return np.max(shm.read_noblock())


# Create a figure and axis
fig, ax = plt.subplots(figsize=(12, 5))
line, = ax.plot([], [], lw=2)

# Initialize the data
xdata, ydata = [], []
past_values = [compute_next_value()]*WINDOW_SIZE

# Set the plot limits
ax.set_xlim(0, MAX_SIZE)
ax.set_ylabel(shmName, size = 16)
ax.set_xlabel("Time [arb]", size = 16)
ax.grid()

# Enable interactive mode
plt.ion()
plt.show()

# Function to update the plot
def update_plot():

    global past_values
    
    if len(past_values) >= MAX_SIZE:
        past_values[:-1] = past_values[1:]
        past_values[-1] = compute_next_value()
    else:
        past_values.append(compute_next_value())

    ydata = rolling_average(past_values, WINDOW_SIZE)
    xdata = list(range(len(ydata)))
    line.set_data(xdata, ydata)
    ax.set_ylim(np.percentile(ydata, 5), np.percentile(ydata, 95))
    fig.canvas.draw()
    fig.canvas.flush_events()

while True:
    update_plot()
    time.sleep(update_interval)  # Adjust the sleep time as needed

# %%
