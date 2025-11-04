import numpy as np
import scipy.sparse as sp
from numba import jit
import cupy as cp  # Import CuPy for GPU-based operations
import torch  # Import PyTorch
import time
import matplotlib.pyplot as plt

def timeF(f,x, N = 10):
    start = time.time()
    for i in range(N):
        f(*x)
    
    return (time.time()-start)/N

# Basic Python Implementation
def mvm_basic(matrix, vector):
    result = [0] * len(vector)
    for i in range(len(matrix)):
        for j in range(len(vector)):
            result[i] += matrix[i][j] * vector[j]
    return result

# NumPy Implementation
def mvm_numpy(matrix, vector):
    return np.dot(matrix, vector)

# SciPy Sparse Implementation
def mvm_scipy(matrix, vector):
    return matrix.dot(vector)

# Numba JIT-Accelerated Implementation
@jit(nopython=True)
def mvm_numba(matrix=np.array([],dtype=np.float32), vector=np.array([],dtype=np.float32)):
    # result = np.zeros(len(vector))
    # for i in range(len(matrix)):
    #     for j in range(len(vector)):
    #         result[i] += matrix[i, j] * vector[j]
    # return result
    np.dot(matrix, vector)
    return
# CuPy GPU Implementation
def mvm_cupy(matrix, vector):
    return cp.dot(matrix, vector)  # Transfer result back to CPU

# PyTorch GPU Implementation
def mvm_torch(matrix, vector):
    return torch.matmul(matrix, vector)  # Move result back to CPU and convert to NumPy

# Function to time each method as the matrix size increases
def time_mvm_methods(sizes):
    times_basic = []
    times_numpy = []
    times_scipy = []
    times_numba = []
    times_cupy = []
    times_torch = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for size in sizes:
        print(f"Testing for size: {size}x{size}")
        matrix = np.random.rand(size, size)
        vector = np.random.rand(size)
        matrix = matrix.astype(np.float32)
        vector = vector.astype(np.float32)
        sparse_matrix = sp.csr_matrix(matrix)  # Convert to sparse matrix
        
        matrix_torch = torch.tensor(matrix).to(device)  # Move matrix to GPU
        vector_torch = torch.tensor(vector).to(device)  # Move vector to GPU
        matrix_gpu = cp.array(matrix)  # Convert matrix to a CuPy array
        vector_gpu = cp.array(vector)  # Convert vector to a CuPy array
        mvm_cupy(matrix_gpu, vector_gpu)
        mvm_torch(matrix_torch, vector_torch)
        mvm_numba(matrix, vector)

        # Time the basic Python implementation
        if size < 1000:
            time_basic = timeF(mvm_basic, (matrix.tolist(), vector.tolist()))
            times_basic.append(time_basic)
        else:
            times_basic.append(np.nan)

        if size < 5000:
            # Time the NumPy implementation
            time_numpy = timeF(mvm_numpy,(matrix, vector))
            times_numpy.append(time_numpy)

            # Time the SciPy sparse implementation
            time_scipy = timeF( mvm_scipy,(sparse_matrix, vector))
            times_scipy.append(time_scipy)

            # Time the Numba JIT-accelerated implementation
            time_numba = timeF( mvm_numba,(matrix, vector))
            times_numba.append(time_numba)
        else:
            times_numpy.append(np.nan)
            times_scipy.append(np.nan)
            times_numba.append(np.nan)
        # Time the CuPy GPU implementation
        time_cupy = timeF( mvm_cupy,(matrix_gpu, vector_gpu))
        times_cupy.append(time_cupy)

        # Time the PyTorch GPU implementation
        time_torch = timeF( mvm_torch,(matrix_torch, vector_torch))
        times_torch.append(time_torch)

    return times_basic, times_numpy, times_scipy, times_numba, times_cupy, times_torch

# Define matrix sizes to test
sizes = np.logspace(2,4, 20).astype(int)

# Run the timing tests
times_basic, times_numpy, times_scipy, times_numba, times_cupy, times_torch = time_mvm_methods(sizes)

# Plotting the results
plt.figure(figsize=(12, 5))
plt.plot(sizes, 1000*np.array(times_basic), label='Basic Python', marker='x')
plt.plot(sizes, 1000*np.array(times_numpy), label='NumPy', marker='x')
plt.plot(sizes, 1000*np.array(times_scipy), label='SciPy (Sparse)', marker='x')
plt.plot(sizes, 1000*np.array(times_numba), label='Numba', marker='x')
plt.plot(sizes, 1000*np.array(times_cupy), label='CuPy (GPU)', marker='x')
plt.plot(sizes, 1000*np.array(times_torch), label='PyTorch (GPU)', marker='x')
# Add a shaded region (using fill_between)
x = np.linspace(np.min(sizes), np.max(sizes), 1000)
plt.fill_between(x, 0, 1, where=(x > 100) & (x < 1000), color='black', alpha=0.3)
plt.fill_between(x, 0, 0.5, where=(x > 2000) & (x < 4000), color='red', alpha=0.3)

# Add a label to the shaded region
plt.text(2000, 0.25, 'XAO Compute Region', horizontalalignment='center', fontsize=12, color='black')
plt.text(3000, 0.5, 'Normal AO Compute Region', horizontalalignment='center', fontsize=12, color='black')

plt.xlabel('Matrix Size (n x n)', size =  18)
plt.ylabel('Time (ms)', size =  18)
plt.title('Performance Comparison of MVM Methods', size =  20)
plt.legend()
plt.yscale("log")
plt.xscale("log")
plt.grid(True)
plt.savefig("MVM_comp.pdf")
plt.show()
