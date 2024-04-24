import yaml
import sys
import select
import os 
from subprocess import PIPE, Popen
import numpy as np
import psutil
from scipy.ndimage import median_filter, gaussian_filter

def compute_fwhm_dark_subtracted_image(image):
    # Filter to keep only negative values
    negative_pixels = image[image < 1]
    
    # Compute the histogram of negative values
    # Adjust bins and range as necessary for your specific image
    hist, bins = np.histogram(negative_pixels, bins=np.arange(np.min(negative_pixels), 1)+0.5)
    # Since the distribution is symmetric, we can mirror the histogram to get the full distribution
    hist_full = np.concatenate((hist[::-1], hist))
    
    # Compute the bin centers from the bin edges
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_centers_full = np.concatenate((-bin_centers[::-1], bin_centers))

    # Find the maximum value (mode of the distribution)
    peak_value = np.max(hist_full)
    half_max = peak_value / 2
    
    # Find the points where the histogram crosses the half maximum
    cross_points = np.where(np.diff((hist_full > half_max).astype(int)))[0]
    
    # Assuming the distribution is sufficiently smooth and has a single peak,
    # the FWHM is the distance between the first and last crossing points
    fwhm_value = np.abs(bin_centers_full[cross_points[-1]] - bin_centers_full[cross_points[0]])
    
    return fwhm_value

def clean_image_for_strehl(img, median_filter_size = 3, gaussian_sigma = 1):

    corrected_img = median_filter(img, size=median_filter_size)  # Hot pixel correction
    corrected_img = gaussian_filter(corrected_img, sigma=gaussian_sigma)  # Smoothing
    
    return corrected_img

def gaussian_2d_grid(i, j, sigma, grid_size):
    grid = np.zeros((grid_size, grid_size))
    for x in range(grid_size):
        for y in range(grid_size):
            if x == i and y == j:
                continue  # Skip the center point as its value should be 0
            else:
                # Compute the Gaussian value
                grid[x, y] = np.exp(-((x - i)**2 + (y - j)**2) / (2 * sigma**2))
    
    grid /= np.sum(grid)

    return grid

def set_affinity(affinity):
    # Unsupported by MacOS
    if sys.platform != 'darwin':
        psutil.Process(os.getpid()).cpu_affinity([affinity,])
    return

def setFromConfig(conf, name, default):
    if name in conf.keys():
        return conf[name]
    return default

def signal2D(signal, layout):
    curSignal2D = np.zeros(layout.shape)
    slopemask = layout[:,:layout.shape[1]//2]
    curSignal2D[:,:layout.shape[1]//2][slopemask] = signal[:signal.size//2]
    curSignal2D[:,layout.shape[1]//2:][slopemask] = signal[signal.size//2:]
    return curSignal2D

def dtype_to_float(dtype):
    """
    Convert a NumPy dtype to a unique float.

    Parameters:
    - dtype: NumPy dtype object

    Returns:
    - float: Unique float representing the dtype
    """
    dtypes = np.array(list(np.sctypeDict.values()))
    dtypesNames = np.array([str(d) for d in dtypes], dtype=str)
    dtypes = dtypes[np.argsort(dtypesNames)]
    for i, d in enumerate(dtypes):
        if dtype == d:
            return i
    return -1

def float_to_dtype(dtype_float):
    """
    Convert a unique float back to the original NumPy dtype.

    Parameters:
    - dtype_float: Unique float representing the dtype

    Returns:
    - np.dtype: NumPy dtype object
    """
    dtypes = np.array(list(np.sctypeDict.values()))
    dtypesNames = np.array([str(d) for d in dtypes], dtype=str)
    dtypes = dtypes[np.argsort(dtypesNames)]
    return np.dtype(dtypes[int(dtype_float)])

def decrease_nice(pid):
    # Unsupported by MacOS
    if sys.platform != 'darwin'  and sys.platform != 'win32':
        cmd = ["sudo","-S","/usr/bin/renice","-n","-19","-p",str(pid)]
        Popen(cmd,stdin=open(os.devnull, 'w'),stdout=open(os.devnull, 'w'))
    return

def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        conf = yaml.safe_load(file)
    return conf

def read_input_with_timeout(timeout):
    # Set the list of file descriptors to watch for input (stdin)
    inputs = [sys.stdin]
    
    # Use select to wait for input or timeout
    readable, _, _ = select.select(inputs, [], [], timeout)
    
    if readable:
        user_input = sys.stdin.readline().strip()
        return user_input
    else:
        return None
    
def is_numeric(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
