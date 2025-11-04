#%%
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ['NUMBA_NUM_THREADS'] = '1'

from pyRTC.utils import *
from pyRTC.Pipeline import *
import numpy as np
import torch
import matplotlib.pyplot as plt
from numba import jit, njit

pid = os.getpid()
set_affinity(3) 
decrease_nice(pid)

@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def leakyIntegratorNumba(slopes: np.ndarray, 
                         resconstructionMatrix: np.ndarray, 
                         oldCorrection: np.ndarray,
                         correction: np.ndarray,
                         leak: float,
                         numActiveModes: int) -> np.ndarray:
    
    # Perform the matrix-vector multiplication using np.dot
    np.dot(resconstructionMatrix, slopes, out=correction)
    
    # Apply the leaky integrator formula with an unrolled loop
    for i in range(numActiveModes + 1):
        correction[i] = (1 - leak) * oldCorrection[i] - correction[i]
    
    # Zero out the rest of the correction vector
    for i in range(numActiveModes + 1, correction.size):
        correction[i] = 0.0
    
    return correction

def leakyIntegratorNumpy(slopes:np.ndarray, 
                                resconstructionMatrix:np.ndarray, 
                                oldCorrection:np.ndarray,
                                correction: np.ndarray,
                                leak:float,
                                numActiveModes:int
                                ):
   #Perform the matrix-vector multiplication
   correction = np.dot(resconstructionMatrix, slopes) 
   correction[numActiveModes:] = 0
   return np.subtract((1-leak)*oldCorrection, correction)

def leakIntegratorGPU(slopes:np.ndarray, 
                                resconstructionMatrix:torch.tensor, 
                                oldCorrection:np.ndarray,
                                correction: np.ndarray,
                                leak:float,
                                numActiveModes:int
                                ):
    slopes_GPU = torch.tensor(slopes, device='cuda')
    correctionGPU = torch.matmul(resconstructionMatrix, slopes_GPU) 
    correctionGPU[numActiveModes:] = 0
    return np.subtract((1-leak)*oldCorrection, correctionGPU.cpu().numpy())

# Test the function with increasing values of N
# Test the function with increasing values of N
numIters = 100
N_values = np.logspace(0.8, 2, 10).astype(int)
execution_times = np.empty(N_values.size)
execution_times_err = np.empty(N_values.size)
execution_times_min = np.empty(N_values.size)
execution_times_max = np.empty(N_values.size)

execution_times_numba = np.empty(N_values.size)
execution_times_numba_err = np.empty(N_values.size)
execution_times_numba_min = np.empty(N_values.size)
execution_times_numba_max = np.empty(N_values.size)

funcs = [leakyIntegratorNumpy, leakIntegratorGPU]

for i, N in enumerate(N_values):
    nAct = int(N*N)
    # Create random NxN arrays
    slopes = np.random.rand(2*nAct).astype(np.float32)
    resconstructionMatrix = np.random.rand(nAct,slopes.size).astype(np.float32)
    oldCorrection = np.zeros(nAct, dtype=np.float32)
    correction = np.zeros(nAct, dtype=np.float32)
    leak = 0.01
    args = [slopes, resconstructionMatrix, oldCorrection, correction, leak, int(correction.size-1)]
    args2 = []
    for a in args:
        if isinstance(a, np.ndarray):
            args2.append(a.copy())
        else:
            args2.append(a)

    args2[1] = torch.tensor(resconstructionMatrix, device='cuda')

    execution_times[i], execution_times_err[i], \
        execution_times_min[i], execution_times_max[i] = \
            measure_execution_time(funcs[0], args, numIters=numIters)
    print(f"N: {N}, Execution time: {execution_times[i]:.6f} seconds")

    execution_times_numba[i], execution_times_numba_err[i],\
          execution_times_numba_min[i], execution_times_numba_max[i] =\
              measure_execution_time(funcs[1], args2, numIters=numIters)
    print(f"N: {N}, Execution time: {execution_times_numba[i]:.6f} seconds")

#%%
a = funcs[0](*args) 
b = funcs[1](*args2)
print(np.any(np.abs(a - b) > 1e-5))

execution_times *= 1000
execution_times_err *= 1000
execution_times_min *= 1000
execution_times_max *= 1000

execution_times_numba *= 1000
execution_times_numba_err *= 1000
execution_times_numba_min *= 1000
execution_times_numba_max *= 1000

#%%

plt.figure(figsize=(12, 5))
plt.plot(N_values, execution_times, 
                  color = 'k',  label = "NumPy Only")
plt.fill_between(N_values, execution_times_min, execution_times_max, 
                  color = 'k', alpha=0.3, label = '99% CI')
plt.plot(N_values, execution_times_numba, 
                  color = 'r',  label = "GPU Accelerated")
plt.fill_between(N_values, execution_times_numba_min, execution_times_numba_max, 
                  color = 'r', alpha=0.3, label = '99% CI')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('# Subapertures Across Pupil', size = 16)
plt.ylabel('Execution Time (ms)', size = 16)
plt.title('Execution Time vs. # Subapertures (Integrator)', size = 18)
plt.grid(True)

normal_ao_shwfs_size_min = 8
normal_ao_shwfs_size_max = 32
normal_ao_comp_time_min = 0
normal_ao_comp_time_max = 2

x_ao_shwfs_size_min = 32
x_ao_shwfs_size_max = 70
x_ao_comp_time_min = 0.0
x_ao_comp_time_max = 0.5

x = np.linspace(np.min(N_values), np.max(N_values), 1000)

# plt.text(0.5*(x_ao_shwfs_size_max-x_ao_shwfs_size_min), 
#          0.5*(x_ao_comp_time_max-x_ao_comp_time_min), 
#          'XAO Compute Region', horizontalalignment='left', fontsize=12, color='black')
# plt.text(0.5*(normal_ao_shwfs_size_max-normal_ao_shwfs_size_min), 
#          0.5*(normal_ao_comp_time_max-normal_ao_comp_time_min), 
#          'Normal AO Compute Region', horizontalalignment='left', fontsize=12, color='black')

plt.fill_between(x, normal_ao_comp_time_min, normal_ao_comp_time_max, 
                 where=(x > normal_ao_shwfs_size_min) & (x < normal_ao_shwfs_size_max), 
                 color='blue', alpha=0.3, label = 'Normal AO Compute Region')
plt.fill_between(x, x_ao_comp_time_min, x_ao_comp_time_max, 
                 where=(x > x_ao_shwfs_size_min) & (x < x_ao_shwfs_size_max), 
                 color='orange', alpha=0.3, label = "XAO Compute Region")
plt.legend(loc=4)
plt.savefig("loop_mvm_time.pdf")
plt.show()
# %%
