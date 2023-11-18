"""
Pipeline Superclasss
"""
from multiprocessing import shared_memory
import numpy as np
import matplotlib.pyplot as plt

class ImageSHM:

    def __init__(self, name, shape, dtype ) -> None:

        self.name = name
        self.arr = np.empty(shape, dtype=dtype)

        try:
            self.shm = shared_memory.SharedMemory(name=name, create=True, size=self.arr.nbytes)
        except:
            self.shm = shared_memory.SharedMemory(name=name, size=self.arr.nbytes)


        self.arr = np.ndarray(shape, dtype=dtype, buffer=self.shm.buf)
        return

    def __del__(self):

        self.shm.close()
        self.shm.unlink()
        del self.arr

    def write(self, arr):
        if arr.shape != self.arr.shape:
            return -1
        np.copyto(self.arr, arr)
        return 1
    
    def read(self):
        return np.copy(self.arr)
    
    def plot(self):
        arr = self.read()
        plt.imshow(arr)
        plt.show()
