import yaml
import sys
import select
import os 
from subprocess import PIPE, Popen
import numpy as np
import psutil

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
    if sys.platform != 'darwin' and sys.platform != 'win32':
        cmd = ["sudo","renice","-n","-19","-p",str(pid)]
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
