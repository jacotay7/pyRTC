import yaml
import sys
import select
import os 
from astropy.io import fits
import numpy as np
import psutil
from scipy.ndimage import median_filter, gaussian_filter
import socket
from datetime import datetime
import time 
import logging
from typing import Any, Iterable, Mapping, Optional

from pyRTC.logging_utils import get_logger


logger = get_logger(__name__)

NP_DATA_TYPES = [
    np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64,
    np.float16, np.float32, np.float64, #np.float128,  # np.float128 availability depends on the system
    np.complex64, np.complex128, #np.complex256,       # np.complex256 availability depends on the system
    np.bool_,
    np.object_,
    np.bytes_, np.str_,
    np.datetime64, np.timedelta64
]


class ConfigValidationError(ValueError):
    pass


def _require_mapping(conf: Any, component: str) -> Mapping[str, Any]:
    if not isinstance(conf, Mapping):
        raise ConfigValidationError(f"{component}: config must be a mapping/dict, got {type(conf).__name__}")
    return conf


def _validate_optional_numeric(conf: Mapping[str, Any], key: str, component: str, minimum: Optional[float] = None):
    if key not in conf:
        return
    value = conf[key]
    if not isinstance(value, (int, float)):
        raise ConfigValidationError(f"{component}: '{key}' must be numeric, got {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise ConfigValidationError(f"{component}: '{key}' must be >= {minimum}, got {value}")


def validate_wfs_config(conf: Any) -> None:
    component = "wfs"
    conf = _require_mapping(conf, component)

    _validate_optional_numeric(conf, "width", component, minimum=1)
    _validate_optional_numeric(conf, "height", component, minimum=1)
    _validate_optional_numeric(conf, "darkCount", component, minimum=0)
    _validate_optional_numeric(conf, "downsampleFactor", component, minimum=0)
    _validate_optional_numeric(conf, "rotationAngle", component)


def validate_wfc_config(conf: Any) -> None:
    component = "wfc"
    conf = _require_mapping(conf, component)

    required = ["name", "numActuators", "numModes"]
    missing = [key for key in required if key not in conf]
    if missing:
        missing_str = ", ".join(missing)
        raise ConfigValidationError(f"{component}: missing required config key(s): {missing_str}")

    if not isinstance(conf["name"], str) or not conf["name"].strip():
        raise ConfigValidationError(f"{component}: 'name' must be a non-empty string")

    if not isinstance(conf["numActuators"], int) or conf["numActuators"] <= 0:
        raise ConfigValidationError(f"{component}: 'numActuators' must be a positive int, got {conf['numActuators']}")

    if not isinstance(conf["numModes"], int) or conf["numModes"] <= 0:
        raise ConfigValidationError(f"{component}: 'numModes' must be a positive int, got {conf['numModes']}")

    _validate_optional_numeric(conf, "floatingInfluenceRadius", component, minimum=0)
    _validate_optional_numeric(conf, "frameDelay", component, minimum=0)


def validate_loop_config(conf: Any) -> None:
    component = "loop"
    conf = _require_mapping(conf, component)

    _validate_optional_numeric(conf, "numDroppedModes", component, minimum=0)
    _validate_optional_numeric(conf, "gain", component)
    _validate_optional_numeric(conf, "leakyGain", component)
    _validate_optional_numeric(conf, "hardwareDelay", component, minimum=0)
    _validate_optional_numeric(conf, "pokeAmp", component, minimum=0)
    _validate_optional_numeric(conf, "numItersIM", component, minimum=1)
    _validate_optional_numeric(conf, "delay", component, minimum=0)
    _validate_optional_numeric(conf, "pGain", component)
    _validate_optional_numeric(conf, "iGain", component)
    _validate_optional_numeric(conf, "dGain", component)
    _validate_optional_numeric(conf, "derivativeFilter", component)

    for key in ["controlLimits", "integralLimits", "absoluteLimits"]:
        if key not in conf:
            continue
        value = conf[key]
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ConfigValidationError(f"{component}: '{key}' must be a list/tuple of length 2")


def validate_component_config(conf: Any, mro_names: Iterable[str]) -> None:
    _require_mapping(conf, "component")

    mro_name_set = set(mro_names)
    if "Loop" in mro_name_set:
        validate_loop_config(conf)
    if "WavefrontSensor" in mro_name_set:
        validate_wfs_config(conf)
    if "WavefrontCorrector" in mro_name_set:
        validate_wfc_config(conf)


def precise_delay(microseconds):
    target_time = time.perf_counter() + microseconds / 1_000_000
    while np.float64(time.perf_counter()) < target_time:
        pass

# Function to measure execution time
def measure_execution_time(f, args, numIters=10):
   
    #init once
    f(*args)

    # Measure time
    exTimes = np.empty(numIters)
    for i in range(numIters):
        start_time = time.time()
        f(*args)
        end_time = time.time()
        exTimes[i] = (end_time - start_time)

    sorted_times = np.sort(exTimes)

    def _percentile_from_sorted(sorted_arr, pct):
        if sorted_arr.size == 0:
            return np.float64(0.0)
        if sorted_arr.size == 1:
            return np.float64(sorted_arr[0])
        rank = (pct / 100.0) * (sorted_arr.size - 1)
        low = int(np.floor(rank))
        high = int(np.ceil(rank))
        if low == high:
            return np.float64(sorted_arr[low])
        weight = rank - low
        return np.float64(sorted_arr[low] * (1.0 - weight) + sorted_arr[high] * weight)

    median = _percentile_from_sorted(sorted_times, 50.0)
    q1 = _percentile_from_sorted(sorted_times, 25.0)
    q3 = _percentile_from_sorted(sorted_times, 75.0)
    iqr = q3 - q1
    CI_1 = _percentile_from_sorted(sorted_times, 0.5)
    CI_99 = _percentile_from_sorted(sorted_times, 99.5)

    return median, iqr, CI_1, CI_99

def change_directory(directory):
    try:
        os.chdir(directory)
        logger.info("Successfully changed the current directory to %s", os.getcwd())
    except FileNotFoundError:
        logger.error("The directory '%s' does not exist", directory)
    except PermissionError:
        logger.error("Permission denied to access the directory '%s'", directory)
    except Exception as e:
        logger.exception("Unexpected error while changing directory: %s", e)
    return

def add_to_path(directory):
    # Check if the directory exists
    if not os.path.isdir(directory):
        logger.error("The directory '%s' does not exist", directory)
        return

    # Add the directory to the PATH environment variable
    current_path = os.environ.get('PATH', '')
    if directory not in current_path:
        new_path = f"{directory}:{current_path}"
        os.environ['PATH'] = new_path
        logger.info("Directory '%s' added to PATH", directory)
    else:
        logger.info("Directory '%s' is already in PATH", directory)

    return

def powerLawOG(numModes, k):
    return (1- (np.arange(numModes)/numModes)**k)


def append_to_file(filename, data, dtype=np.float32):
    """
    Append a numpy array to a binary file on disk.

    Parameters:
    filename : str
        The name of the file to which data will be appended.
    data : numpy array
        The numpy array to append to the file.
    dtype : data-type, optional
        The desired data-type for the array. Default is np.float32.
    """
    if os.path.exists(filename):
        # If the file exists, append to it
        with open(filename, 'ab') as f:
            data.tofile(f)
    else:
        # If the file does not exist, create it and write the initial data
        with open(filename, 'wb') as f:
            data.tofile(f)

def generate_circular_aperture_mask(N, R, ratio):
    """
    Generates a binary mask of size NxN with a circular aperture of radius R and a central obscuration of radius r.
    
    Parameters:
    N (int): The size of the mask (NxN).
    R (float): The radius of the outer circular aperture.
    ratio (float): The ratio of the inner obscuration radius to the outer radius (r/R).

    Returns:
    numpy.ndarray: Binary mask with the circular aperture.
    """
    r = R * ratio
    x = np.linspace(-N/2, N/2, N)
    xx, yy = np.meshgrid(x,x)
    mask = (xx**2 + yy**2 <= R**2) 
    if r > 0:
        mask &= (xx**2 + yy**2 >= r**2)
    return mask.astype(bool)

def load_data(filename, dtype=None):
    if filename.endswith('.npy'):
        data = np.load(filename)
    elif filename.endswith('.fits'):
        with fits.open(filename) as hdul:
            data = hdul[0].data
    else:
        raise ValueError("Unsupported file format. Please provide a .npy or .fits file.")
    
    if dtype is not None:
        return data.astype(dtype)
    return data

def generate_filepath(base_dir='.', prefix='file', extension='.dat'):
    """
    Generate a file path based on the current date and time.

    Parameters:
    base_dir : str
        The base directory where the file will be saved.
    prefix : str
        The prefix for the file name.
    extension : str
        The file extension.

    Returns:
    str
        The generated file path.
    """
    # Get the current date and time
    current_time = datetime.now()

    # Format the date and time
    timestamp = current_time.strftime('%Y%m%d_%H%M%S')

    # Construct the file name
    filename = f"{prefix}_{timestamp}{extension}"

    # Construct the full file path
    filepath = os.path.join(base_dir, filename)

    return filepath

def get_tmp_filepath(file_path, uniqueStr = 'tmp'):
    """
    Append '_tmp' to the filename part of the given file path, before the file extension.

    :param file_path: str, the original file path
    :return: str, modified file path with '_tmp' before the extension
    """
    # Split the file path into directory path and filename
    dir_path, filename = os.path.split(file_path)

    # Split the filename into name and extension
    file_name, file_ext = os.path.splitext(filename)

    # Add '_tmp' to the filename
    new_filename = f"{file_name}_{uniqueStr}{file_ext}"

    # Construct the new full path
    new_file_path = os.path.join(dir_path, new_filename)

    return new_file_path

def centroid(array):
    arr = np.asarray(array, dtype=np.float64)
    total = np.add.reduce(arr.ravel(), dtype=np.float64) + 1e-4
    y_indices, x_indices = np.indices(arr.shape, dtype=np.float64)
    x_weighted = np.add.reduce((x_indices * arr).ravel(), dtype=np.float64)
    y_weighted = np.add.reduce((y_indices * arr).ravel(), dtype=np.float64)
    return np.array([x_weighted / total, y_weighted / total], dtype=np.float64)

def add_to_buffer(buffer, vec):
    buffer[:-1] = buffer[1:]
    buffer[-1] = vec
    return

def next_power_of_two(n):
    # Handle case for non-positive input
    if n <= 0:
        return 1

    power = 1
    while power <= n:
        power *= 2
    return power


def adjusted_cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0
    cosine_similarity = dot_product / (norm_a * norm_b)
    magnitude_similarity = min(norm_a, norm_b) / max(norm_a, norm_b)
    return cosine_similarity * magnitude_similarity


def robust_variance(data):
    median = np.median(data)
    deviations = np.abs(data - median)
    mad = np.median(deviations)
    return (mad / 0.6745) ** 2

def cosine_similarity(v1, v2):
    # Calculate the magnitudes of the vectors
    mag_v1 = np.linalg.norm(v1)
    mag_v2 = np.linalg.norm(v2)

    # Calculate the dot product of vectors
    dot_product = np.dot(v1, v2)

    if mag_v1 == 0 or mag_v2 == 0:
        return 0

    return dot_product / (mag_v1 * mag_v2)

def angle_between_vectors(v1, v2):

    # Calculate the cosine of the angle
    return np.abs(np.arccos(cosine_similarity(v1, v2)))

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
    corrected_img = np.asarray(img)

    if median_filter_size is not None and median_filter_size > 1:
        corrected_img = median_filter(
            corrected_img,
            size=median_filter_size,
            output=None,
            mode='reflect',
            cval=0.0,
            origin=0,
        )

    if gaussian_sigma is not None and gaussian_sigma > 0:
        corrected_img = gaussian_filter(
            corrected_img,
            sigma=gaussian_sigma,
            order=0,
            output=None,
            mode='reflect',
            cval=0.0,
            truncate=4.0,
        )

    return corrected_img

def gaussian_2d_grid(i, j, sigma, grid_size):
    i = int(np.asarray(i).reshape(-1)[0])
    j = int(np.asarray(j).reshape(-1)[0])
    sigma = float(np.asarray(sigma).reshape(-1)[0])
    grid_size = int(np.asarray(grid_size).reshape(-1)[0])

    grid = np.zeros((grid_size, grid_size))
    if sigma == 0:
        return grid
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
    if isinstance(affinity, int) or isinstance(affinity, float):
        affinity = [int(affinity),]
    elif isinstance(affinity, np.ndarray):
        affinity = list(affinity)
    else:
        return -1
    if sys.platform != 'darwin':
        psutil.Process(os.getpid()).cpu_affinity(affinity)
    return



def setFromConfig(conf, name, default):
    if name in conf.keys():
        val = conf[name]
    else:
        val = default

    debugStr = f"There is a type mismatch between the default value for config variable {name} and the given value: {type(val).__name__} != {type(default).__name__}"

    if default is not None:
        assert isinstance(val, type(default)), debugStr

    return val

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
    for i, d in enumerate(NP_DATA_TYPES):
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
    return np.dtype(NP_DATA_TYPES[int(dtype_float)])

def bind_socket(host, start_port, max_attempts=5):
    """Attempts to bind a socket on a range of ports, handling OSError exceptions."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # Allow reuse of socket addresses

    for attempt in range(max_attempts):
        try:
            # Attempt to bind the socket
            sock.bind((host, start_port + attempt))
            logger.info("Bound socket to %s:%s", host, start_port + attempt)
            return sock
        except OSError as e:
            logger.warning("Failed to bind to %s:%s: %s", host, start_port + attempt, e)
            if e.errno == socket.errno.EADDRINUSE:
                logger.info("Address already in use. Trying next port.")
            else:
                logger.error("Unexpected socket bind failure. Stopping attempts.")
                break
    else:
        # After all attempts, if no binding was successful, raise an exception
        raise RuntimeError("Failed to bind socket after multiple attempts")

    return -1

def decrease_nice():
    # Unsupported by MacOS or Windows
    if sys.platform != 'darwin' and sys.platform != 'win32':
        try:
            p = psutil.Process(os.getpid())
            p.nice(-20)  # Unix uses a numeric value (lower means higher priority)
        except Exception:
            logging.log(level=logging.WARNING, msg="Unable to adjust nice level.\
                         Give your user sudo privledges without passowrd to use this feature.")
    return

# Set CPU affinity and priority for a thread
def set_affinity_and_priority(thread_id, cpu_cores):
    set_affinity(cpu_cores)
    decrease_nice()
    logger.info("Thread %s: priority set to REALTIME", thread_id)

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
