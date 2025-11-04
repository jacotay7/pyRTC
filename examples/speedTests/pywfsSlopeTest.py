#%%
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ['NUMBA_NUM_THREADS'] = '1'
import numpy as np
import matplotlib.pyplot as plt
from numba import jit

from pyRTC.utils import *
from pyRTC.Pipeline import *

pid = os.getpid()
set_affinity(3) 
decrease_nice(pid)

def computeSlopesPYWFSOptimNumpy(image:np.ndarray,
                            p1Mask:np.ndarray, 
                            p2Mask:np.ndarray,
                            p3Mask:np.ndarray, 
                            p4Mask:np.ndarray,
                            p1:np.ndarray, 
                            p2:np.ndarray,
                            p3:np.ndarray, 
                            p4:np.ndarray,
                            tmp1:np.ndarray,
                            tmp2:np.ndarray,
                            numSlopes:int, 
                            slopes:np.ndarray,
                            refSlopes:np.ndarray,
                        ):
    
    #Mask Pupils out of image and convert to floats
    p1 = image[p1Mask].astype(np.float32)
    p2 = image[p2Mask].astype(np.float32)
    p3 = image[p3Mask].astype(np.float32)
    p4 = image[p4Mask].astype(np.float32)

    #Sub Pupils horizontally
    tmp1 = np.add(p1,p2)
    tmp2 = np.add(p3,p4)

    #Compute Y slopes
    slopes[:numSlopes] = np.subtract(tmp1,tmp2)

    #Compute X slopes
    slopes[numSlopes:] = np.subtract(np.add(p1,p3),np.add(p2,p4))

    #Normalize slopes by overall mean of pupils
    slopes = np.divide(slopes, np.mean(np.add(tmp1,tmp2)))
    
    #Return difference from reference slopes
    return slopes - refSlopes

@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def computeSlopesPYWFSOptimNumba(image:np.ndarray,
                            p1Mask:np.ndarray, 
                            p2Mask:np.ndarray,
                            p3Mask:np.ndarray, 
                            p4Mask:np.ndarray,
                            p1:np.ndarray, 
                            p2:np.ndarray,
                            p3:np.ndarray, 
                            p4:np.ndarray,
                            tmp1:np.ndarray,
                            tmp2:np.ndarray,
                            numSlopes:int, 
                            slopes:np.ndarray,
                            refSlopes:np.ndarray,
                        ):
    # Mask Pupils out of image and convert to floats
    p1_count, p2_count, p3_count, p4_count = 0, 0, 0, 0
    for i in range(len(image)):
        if p1Mask[i]:
            p1[p1_count] = np.float32(image[i])
            p1_count += 1
        if p2Mask[i]:
            p2[p2_count] = np.float32(image[i])
            p2_count += 1
        if p3Mask[i]:
            p3[p3_count] = np.float32(image[i])
            p3_count += 1
        if p4Mask[i]:
            p4[p4_count] = np.float32(image[i])
            p4_count += 1

    # Sum Pupils, Saving partial sums to avoid recomputing later
    total_sum = 0.0
    for i in range(numSlopes):  # Assuming all counts are equal
        tmp1[i] = p1[i] + p2[i]
        tmp2[i] = p3[i] + p4[i]
        total_sum += tmp1[i] + tmp2[i]
    mean_value = total_sum / p1_count

    for i in range(numSlopes):
        # Compute Y slopes
        slopes[i] = (tmp1[i] - tmp2[i])/mean_value - refSlopes[i]
        # Compute X slopes
        slopes[numSlopes + i] = ((p1[i] + p3[i]) - (p2[i] + p4[i]))/mean_value \
            - refSlopes[numSlopes + i]

    return slopes
# Test the function with increasing values of N
numIters = 10000
N_values = np.logspace(0.8, 2, 10).astype(int)
execution_times = np.empty(N_values.size)
execution_times_err = np.empty(N_values.size)
execution_times_min = np.empty(N_values.size)
execution_times_max = np.empty(N_values.size)

execution_times_numba = np.empty(N_values.size)
execution_times_numba_err = np.empty(N_values.size)
execution_times_numba_min = np.empty(N_values.size)
execution_times_numba_max = np.empty(N_values.size)

for i, N in enumerate(N_values):
    # Create random NxN arrays
    
    numSlopes = int(N*N)
    image = (1000*np.random.rand(4*numSlopes)).astype(np.int32)
    p1Mask = np.zeros(4*numSlopes).astype(bool)
    p1Mask[:numSlopes] = 1
    p2Mask = np.copy(p1Mask)
    p3Mask = np.copy(p1Mask)
    p4Mask = np.copy(p1Mask)

    p1 = np.random.rand(numSlopes).astype(np.float32)
    p2 = np.random.rand(numSlopes).astype(np.float32)
    p3 = np.random.rand(numSlopes).astype(np.float32)
    p4 = np.random.rand(numSlopes).astype(np.float32)
    slopes = np.random.rand(2 * numSlopes).astype(np.float32)
    refSlopes = np.random.rand(2 * numSlopes).astype(np.float32)
    tmp1, tmp2 = np.empty_like(p1), np.empty_like(p2)

    args = [image, p1Mask, p2Mask, p3Mask, p4Mask, p1,p2,p3,p4,tmp1,tmp2, numSlopes, slopes, refSlopes]
    args2 = []
    for a in args:
        if isinstance(a, np.ndarray):
            args2.append(a.copy())
        else:
            args2.append(a)

    execution_times[i], execution_times_err[i], execution_times_min[i], execution_times_max[i] = measure_execution_time(computeSlopesPYWFSOptimNumpy, args, numIters=numIters)
    print(f"N: {N}, Execution time: {execution_times[i]:.6f} seconds")

    execution_times_numba[i], execution_times_numba_err[i], execution_times_numba_min[i], execution_times_numba_max[i] = measure_execution_time(computeSlopesPYWFSOptimNumba, args2, numIters=numIters)
    # exec_time = measure_execution_time(computeSlopesOptim, args, numIters=numIters)
    print(f"N: {N}, Execution time: {execution_times_numba[i]:.6f} seconds")

# %%
a = computeSlopesPYWFSOptimNumba(*args2) 
b = computeSlopesPYWFSOptimNumpy(*args)
print(a - b)
execution_times *= 1000
execution_times_err *= 1000
execution_times_min *= 1000
execution_times_max *= 1000

execution_times_numba *= 1000
execution_times_numba_err *= 1000
execution_times_numba_min *= 1000
execution_times_numba_max *= 1000
#%% PLOT
# Plot the results
plt.figure(figsize=(12, 5))
plt.plot(N_values, execution_times, 
                  color = 'k',  label = "NumPy Only")
plt.fill_between(N_values, execution_times_min, execution_times_max, 
                  color = 'k', alpha=0.3, label = '99% CI')
plt.plot(N_values, execution_times_numba, 
                  color = 'r',  label = "Numba Accelerated")
plt.fill_between(N_values, execution_times_numba_min, execution_times_numba_max, 
                  color = 'r', alpha=0.3, label = '99% CI')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('# Subapertures Across Pupil', size = 16)
plt.ylabel('Execution Time (ms)', size = 16)
plt.title('Execution Time vs. # Subapertures (PyWFS Slopes)', size = 18)
plt.grid(True)

normal_ao_pywfs_size_min = 8
normal_ao_pywfs_size_max = 32
normal_ao_comp_time_min = 0
normal_ao_comp_time_max = 2

x_ao_pywfs_size_min = 32
x_ao_pywfs_size_max = 70
x_ao_comp_time_min = 0.0
x_ao_comp_time_max = 0.5

x = np.linspace(np.min(N_values), np.max(N_values), 1000)

# plt.text(0.5*(x_ao_pywfs_size_max-x_ao_pywfs_size_min), 
#          0.5*(x_ao_comp_time_max-x_ao_comp_time_min), 
#          'XAO Compute Region', horizontalalignment='left', fontsize=12, color='black')
# plt.text(0.5*(normal_ao_pywfs_size_max-normal_ao_pywfs_size_min), 
#          0.5*(normal_ao_comp_time_max-normal_ao_comp_time_min), 
#          'Normal AO Compute Region', horizontalalignment='left', fontsize=12, color='black')

plt.fill_between(x, normal_ao_comp_time_min, normal_ao_comp_time_max, 
                 where=(x > normal_ao_pywfs_size_min) & (x < normal_ao_pywfs_size_max), 
                 color='blue', alpha=0.3, label = 'Normal AO Compute Region')
plt.fill_between(x, x_ao_comp_time_min, x_ao_comp_time_max, 
                 where=(x > x_ao_pywfs_size_min) & (x < x_ao_pywfs_size_max), 
                 color='orange', alpha=0.3, label = "XAO Compute Region")
plt.legend(loc=4)
plt.savefig("pyWFS_slope_time.pdf")
plt.show()

#%%
