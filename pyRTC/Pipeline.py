"""
Pipeline Superclasss
"""
from multiprocessing import shared_memory, resource_tracker
import numpy as np
import matplotlib.pyplot as plt

class ImageSHM:

    def __init__(self, name, shape, dtype) -> None:

        self.name = name
        self.arr = np.empty(shape, dtype=dtype)

        try:
            self.shm = shared_memory.SharedMemory(name= name, create=True, size=self.arr.nbytes)
            print("Creating New Shared Memory Object {self.name}")
        except:
            self.shm = shared_memory.SharedMemory(name=name)
            print("Opening Existing Shared Memory Object {self.name}")
        resource_tracker.unregister(self.shm._name, 'shared_memory')
        self.arr = np.ndarray(shape, dtype=dtype, buffer=self.shm.buf)
        return

    # def __del__(self):

    #     self.close()
        # del self.arr

    def close(self):
        print(f"Closing {self.name}")
        self.shm.close()
        return

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
