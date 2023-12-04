"""
Pipeline Superclasss
"""
from multiprocessing import shared_memory, resource_tracker
import numpy as np
import matplotlib.pyplot as plt
import time
import threading

def work(obj, functionName):
    """
    The main working thread for the any Pipeline object
    """
    #Get what function we need to run
    workFunction = getattr(obj, functionName, None)
    count = 0
    times = np.zeros(100)
    #If the wfs object is still alive
    while obj.alive:
        #If we are meant to be running
        if obj.running:
            start = time.time()
            #Call it
            workFunction()
            diff = time.time()-start
            times[count % len(times)] = diff
            count += 1
            if count % 10000 == 0:
                print(f"Thread {functionName} -- Mean Execution Time {1000*np.mean(times):.3f}ms")
        else:
            time.sleep(1e-4)
    return

class ImageSHM:

    def __init__(self, name, shape, dtype) -> None:

        self.name = name
        self.arr = np.empty(shape, dtype=dtype)
        self.count = 0
        try:
            self.shm = shared_memory.SharedMemory(name= name, create=True, size=self.arr.nbytes)
            print(f"Creating New Shared Memory Object {self.name}")
        except:
            self.shm = shared_memory.SharedMemory(name=name)
            print(f"Opening Existing Shared Memory Object {self.name}")
        resource_tracker.unregister(self.shm._name, 'shared_memory')
        self.arr = np.ndarray(shape, dtype=dtype, buffer=self.shm.buf)

        self.flags = [Lock() for i in range(5)]

        return

    def __del__(self):

        self.close()

    def close(self):
        print(f"Closing {self.name}")
        self.shm.close()
        return

    def write(self, arr, flagInd=0):

        # print(f"Writing to SHM: {self.name}")
        if not isinstance(arr, np.ndarray) or arr.shape != self.arr.shape:
            return -1

        #attach to main lock
        if self.flags[flagInd].attach():
            #copy new data to the memory
            np.copyto(self.arr, arr)
            #Tell every lock there is new data
            for flag in self.flags:
                flag.update()
            #Release the main lock
            self.flags[flagInd].release()

            return 1
        return -1
    
    def read(self, flagInd=0):
        # print(f"Reading from SHM: {self.name}")

        #Wait until there is a fresh array to read
        while not self.flags[flagInd].fresh:
            time.sleep(5e-5)
        #attach to main lock
        if self.flags[flagInd].attach(blocking=True):
            #copy the data from the memory
            arr = np.copy(self.arr)
            #Tell this lock that we have read the data before
            self.flags[flagInd].fresh=False
            #Release the lock
            self.flags[flagInd].release()
            return arr
        return -1
    
    def read_noblock(self, flagInd=0):
        #Try to attach to main lock
        if self.flags[flagInd].attach(blocking=False):
            #If we are successful, copy the data from memory
            arr = np.copy(self.arr)
            #Tell this lock that we have read the data before
            self.flags[flagInd].reset()
            #Release the lock
            self.flags[flagInd].release()
            #Return the data to the user
            return arr
        #Tell the user we failed to read
        return False

    def read_noblock_safe(self, flagInd=0):
        tmp = False
        while isinstance(tmp, type(False)) and tmp == False:
            tmp = self.read_noblock(flagInd=flagInd)
        return tmp
    
class Lock:

    def __init__(self) -> None:
        
        self.semaphore = threading.Semaphore(1)
        self.lastWriteTime = None
        self.count = 0
        self.fresh = False
        return
    
    def attach(self, blocking = True):
        
        if blocking:
            return self.semaphore.acquire(blocking=True,timeout=1)
        else:
            return self.semaphore.acquire(blocking=False)
        
    def release(self):
        return self.semaphore.release()
    
    def update(self):
        self.count += 1
        self.lastWriteTime = time.time()
        self.fresh = True
    
    def reset(self):
        self.fresh = False
    

def clear_all_shms():
    shms = [ImageSHM("wfs",(1,),np.uint16), 
            ImageSHM("wfc",(1,),np.uint16), 
            ImageSHM("wfc2D",(1,),np.uint16),
            ImageSHM("signal",(1,),np.float64)]
    for shm in shms:
        shm.shm.unlink()