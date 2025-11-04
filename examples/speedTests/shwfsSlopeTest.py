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
import cupy as cp
import matplotlib.pyplot as plt
from numba import jit, njit

pid = os.getpid()
set_affinity(3) 
decrease_nice(pid)


@jit(nopython=True, nogil=True, cache=True)
def computeSlopesSHWFSOptimNumba(image:np.ndarray, 
                                 slopes:np.ndarray, 
                                 unaberratedSlopes:np.ndarray, 
                                 threshold:np.float32, 
                                 spacing:np.float32,
                                 xvals:np.ndarray,
                                 offsetX:int, 
                                 offsetY:int,
                                 intN:int,
                                 ):
    
    # Convert image to the same dtype as unaberratedSlopes
    image = image.astype(np.float32)
    
    # Compute the number of sub-apertures
    numRegions = unaberratedSlopes.shape[1]

    # Loop over all regions
    for i in range(numRegions):
        for j in range(numRegions):
            # Compute where to start
            start_i = int(round(spacing * i)) + offsetY
            start_j = int(round(spacing * j)) + offsetX
            
            # Ensure we stay within the bounds of the image
            if start_j + intN <= image.shape[1] and start_i + intN <= image.shape[0]:
                #Create a local subimage around the lenslet spot
                sub_im = image[start_i:start_i + intN, start_j:start_j + intN]

                #loop through the sub image
                norm = np.float32(0)
                weightX = np.float32(0)
                weightY = np.float32(0)
                for m in range(intN):
                    for n in range(intN):
                        #If we are counting the pixel
                        if sub_im[m,n] > threshold:
                            #Add it to the normalization
                            norm += sub_im[m,n]
                            #Compute the X and Y centroids (before normalization)
                            weightX += xvals[m,n] * sub_im[m,n]
                            weightY += xvals[n,m] * sub_im[m,n]

                #If we have flux in the sub aperture
                if norm > 0:
                    #Normalize the centroids and remove the reference slope
                    slopes[i, j] = weightX/norm - unaberratedSlopes[i, j]
                    slopes[i + numRegions, j] = weightY/norm - \
                        unaberratedSlopes[i + numRegions, j]
                #If we have no flux slopes should be zero
    
    return slopes

def computeSlopesSHWFSOptimNumpy(image:np.ndarray, 
                                 slopes:np.ndarray, 
                                 unaberratedSlopes:np.ndarray, 
                                 threshold:float, 
                                 spacing:int, 
                                 xvals:np.ndarray):

    #Only works for integer spacings
    spacing = int(spacing)

    # Convert the image to floats and threshold in one operation
    image = np.where(image > threshold, image.astype(np.float32), 0.0)

    # Reshape the image into blocks of size spacing X spacing
    reshaped_image = image.reshape(image.shape[0] // spacing, spacing,\
                                    image.shape[1] // spacing, spacing)

    # Compute the sum of pixel values in each MxM region
    region_sums = np.sum(reshaped_image, axis=(1, 3))

    # Precompute the dot products instead of tensordot 
    weighted_sum_x = np.einsum('ijkl,jl->ik', reshaped_image, xvals)
    weighted_sum_y = np.einsum('ijkl,jl->ik', reshaped_image, xvals.T)
    
    # Get mask for non-zero value sums
    mask = region_sums > 0.0

    # Compute the centroids directly on the valid regions
    valid_region_sums = region_sums[mask]
    slopes[:slopes.shape[1]][mask] = weighted_sum_x[mask] / valid_region_sums \
        - unaberratedSlopes[:slopes.shape[1]][mask]
    slopes[slopes.shape[1]:][mask] = weighted_sum_y[mask] / valid_region_sums \
        - unaberratedSlopes[slopes.shape[1]:][mask]

    # Return the difference with reference slopes
    return slopes

# Define the host function
def computeSlopesSHWFSOptimGPU(image: np.ndarray, 
                         slopes: np.ndarray, 
                         unaberratedSlopes: np.ndarray, 
                         threshold: float, 
                         spacing: int, 
                         xvals: np.ndarray):
    
    spacing = int(spacing)
    
    # Transfer data to GPU and apply thresholding
    image_gpu = cp.asarray(image, dtype=cp.float32)
    image_gpu = cp.where(image_gpu > threshold, image_gpu, 0.0)
    
    # Reshape and compute sums in one go
    reshaped_image = image_gpu.reshape(image_gpu.shape[0] // spacing, spacing,
                                       image_gpu.shape[1] // spacing, spacing)
    
    # Sum regions
    region_sums = cp.sum(reshaped_image, axis=(1, 3))
    
    # Compute weighted sums using einsum
    weighted_sum_x = cp.einsum('ijkl,jl->ik', reshaped_image, cp.asarray(xvals, dtype=cp.float32))
    weighted_sum_y = cp.einsum('ijkl,jl->ik', reshaped_image, cp.asarray(xvals.T, dtype=cp.float32))
    
    # Valid mask
    mask = region_sums > 0.0
    
    # Initialize slopes on GPU
    slopes_gpu = cp.zeros_like(cp.asarray(slopes, dtype=cp.float32))
    
    # Compute slopes, avoiding division by zero
    valid_region_sums = region_sums[mask]
    
    slopes_gpu[:slopes.shape[1]][mask] = (weighted_sum_x[mask] / valid_region_sums
                                          - cp.asarray(unaberratedSlopes[:slopes.shape[1]])[mask])
    slopes_gpu[slopes.shape[1]:][mask] = (weighted_sum_y[mask] / valid_region_sums
                                          - cp.asarray(unaberratedSlopes[slopes.shape[1]:])[mask])
    
    # Transfer result back to CPU
    return cp.asnumpy(slopes_gpu)


# Test the function with increasing values of N
# Test the function with increasing values of N
numIters = 10
N_values = np.logspace(0.8, 2, 10).astype(int)
execution_times = np.empty(N_values.size)
execution_times_err = np.empty(N_values.size)
execution_times_min = np.empty(N_values.size)
execution_times_max = np.empty(N_values.size)

execution_times_numba = np.empty(N_values.size)
execution_times_numba_err = np.empty(N_values.size)
execution_times_numba_min = np.empty(N_values.size)
execution_times_numba_max = np.empty(N_values.size)

funcs = [computeSlopesSHWFSOptimNumpy, computeSlopesSHWFSOptimNumba]

for i, N in enumerate(N_values):
    # Create random NxN arrays
    numRegions = N
    spacing = np.float32(10)
    threshold = np.float32(0)
    # Create random NxN arrays
    imageShape = (int(numRegions*spacing), int(numRegions*spacing))
    image = np.random.rand(*imageShape).astype(np.float32)
    unaberratedSlopes = 10*np.random.rand(2*numRegions, numRegions).astype(np.float32)
    slopes = np.zeros_like(unaberratedSlopes)
    # Compute the closest integer size of the sub-apertures
    intN = int(round(spacing))
    # Pre-compute the array to bias our centroid by
    xvals = np.arange(intN).astype(int) - intN // 2
    xvals, _ = np.meshgrid(xvals, xvals)
    xvals = xvals.astype(np.float32)
    # xvals = xvals

    args = [image, slopes, unaberratedSlopes, threshold, spacing, xvals]
    args2 = []
    for a in args:
        if isinstance(a, np.ndarray):
            args2.append(a.copy())
        else:
            args2.append(a)

    args2.extend([0, 0, intN])
    ### FOR GPU STUFF
    # args[2] = cp.asarray(args2[2])
    # args[-1] = cp.asarray(args2[-1])

    execution_times[i], execution_times_err[i], \
        execution_times_min[i], execution_times_max[i] = \
            measure_execution_time(funcs[0], args, numIters=numIters)
    print(f"N: {N}, Execution time: {execution_times[i]:.6f} seconds")

    execution_times_numba[i], execution_times_numba_err[i],\
          execution_times_numba_min[i], execution_times_numba_max[i] =\
              measure_execution_time(funcs[1], args2, numIters=numIters)
    # exec_time = measure_execution_time(computeSlopesOptim, args, numIters=numIters)
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
                  color = 'r',  label = "Numba Accelerated")
plt.fill_between(N_values, execution_times_numba_min, execution_times_numba_max, 
                  color = 'r', alpha=0.3, label = '99% CI')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('# Subapertures Across Pupil', size = 16)
plt.ylabel('Execution Time (ms)', size = 16)
plt.title('Execution Time vs. # Subapertures (SHWFS Slopes)', size = 18)
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
plt.savefig("SHWFS_slope_time.pdf")
plt.show()
# %%
