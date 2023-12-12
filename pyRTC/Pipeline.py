"""
Pipeline Superclasss
"""
from multiprocessing import shared_memory, resource_tracker
from subprocess import PIPE, Popen
import numpy as np
import matplotlib.pyplot as plt
import time
import threading
import os 
import sys
from pyRTC.utils import *

def work(obj, functionName):
    """
    The main working thread for the any Pipeline object
    """
    #Get what function we need to run
    workFunction = getattr(obj, functionName, None)
    count = 0
    N = 10000
    times = np.zeros(N)
    #If the wfs object is still alive
    while obj.alive:
        #If we are meant to be running
        if obj.running:
            start = time.time()
            #Call it
            workFunction()
            diff = time.time()-start
            times[count % N] = diff
            count += 1
            # if count % N == 0:
            #     print(f"Thread {functionName} -- Mean Execution Time {1000*np.mean(times):.3f}ms")
        else:
            time.sleep(1e-3)
    return

class ImageSHM:

    def __init__(self, name, shape, dtype) -> None:

        self.name = name
        self.arr = np.empty(shape, dtype=dtype)
        self.size = self.arr.nbytes
        self.metadata = np.zeros(4, dtype=np.float64)
        self.count = 0
        self.lastWriteTime = 0
        self.lastReadTime = 0
        self.areData = not ("meta" in name)

        try:
            self.shm = shared_memory.SharedMemory(name= name, create=True, size=self.arr.nbytes)
            print(f"Creating New Shared Memory Object {self.name}")
        except:
            self.shm = shared_memory.SharedMemory(name=name)
            print(f"Opening Existing Shared Memory Object {self.name}")

        resource_tracker.unregister(self.shm._name, 'shared_memory')
        self.arr = np.ndarray(shape, dtype=dtype, buffer=self.shm.buf)
        #If we are not opening a metadata shm
        if self.areData:
            #Create/Open an associated metadata SHM 
            try:
                self.metadataShm = shared_memory.SharedMemory(name= name+"_meta", create=True, size=self.metadata.nbytes)
                print(f"Creating New Shared Memory Object {self.name}"+"_meta")
            except:
                self.metadataShm = shared_memory.SharedMemory(name= name+"_meta")
                print(f"Opening Existing Shared Memory Object {self.name}"+"_meta")
            resource_tracker.unregister(self.metadataShm._name, 'shared_memory')
            self.metadata = np.ndarray(self.metadata.shape, dtype=self.metadata.dtype, buffer=self.metadataShm.buf)
            self.updateMetadata()
        # self.flags = [Lock() for i in range(5)]

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
        np.copyto(self.arr, arr)
        self.count += 1
        self.lastWriteTime = time.time()
        if self.areData:
            self.updateMetadata()
        return 1
        #attach to main lock
        # if self.flags[flagInd].attach():
        #     #copy new data to the memory
        #     np.copyto(self.arr, arr)
        #     #Tell every lock there is new data
        #     for flag in self.flags:
        #         flag.update()
        #     #Release the main lock
        #     self.flags[flagInd].release()

            # return 1
        # return -1
    
    def read(self, flagInd=0):
        # print(f"Reading from SHM: {self.name}")
        
        while not self.checkNew():
            time.sleep(1e-4)
        arr = np.copy(self.arr)
        return arr

        #Wait until there is a fresh array to read
        # while not self.flags[flagInd].fresh:
        #     time.sleep(1e-4)
        # #attach to main lock
        # if self.flags[flagInd].attach(blocking=True):
        #     #copy the data from the memory
        #     arr = np.copy(self.arr)
        #     #Tell this lock that we have read the data before
        #     self.flags[flagInd].fresh=False
        #     #Release the lock
        #     self.flags[flagInd].release()
        #     return arr
        # return -1
    
    def read_noblock(self, flagInd=0):
        arr = np.copy(self.arr)
        return arr
        #Try to attach to main lock
        # if self.flags[flagInd].attach(blocking=False):
        #     #If we are successful, copy the data from memory
        #     arr = np.copy(self.arr)
        #     #Tell this lock that we have read the data before
        #     self.flags[flagInd].reset()
        #     #Release the lock
        #     self.flags[flagInd].release()
        #     #Return the data to the user
        #     return arr
        # #Tell the user we failed to read
        # return False

    def read_noblock_safe(self, flagInd=0):
        return self.read_noblock()
        # tmp = False
        # while isinstance(tmp, type(False)) and tmp == False:
        #     tmp = self.read_noblock(flagInd=flagInd)
        # return tmp
    
    def checkNew(self):
        
        if self.areData:
            metadata = np.copy(self.metadata)
            if metadata[1] != self.lastReadTime:
                self.lastReadTime = metadata[1]
                return True
        else: #If we are just reading a meta data object directly
            return True
        return False
    
    def updateMetadata(self):
        
        metadata = np.array([self.count, self.lastWriteTime, self.size, dtype_to_float(self.arr.dtype)], 
                            dtype=self.metadata.dtype)
        # print(f"Writing metadata {float_to_dtype(metadata[3])}")
        np.copyto(self.metadata, metadata)
        return

# class Lock:

#     def __init__(self) -> None:
        
#         self.semaphore = threading.Semaphore(1)
#         self.lastWriteTime = None
#         self.count = 0
#         self.fresh = False
#         return
    
#     def attach(self, blocking = True):
        
#         if blocking:
#             return self.semaphore.acquire(blocking=True,timeout=1)
#         else:
#             return self.semaphore.acquire(blocking=False)
        
#     def release(self):
#         return self.semaphore.release()
    
#     def update(self):
#         self.count += 1
#         self.lastWriteTime = time.time()
#         self.fresh = True
    
#     def reset(self):
#         self.fresh = False
    

def clear_shms(names):
    
    for n in names:
        shm = ImageSHM(n,(1,),np.uint8)
        shm.shm.unlink()


class hardwareLauncher:

    def __init__(self, hardwareFile, configFile) -> None:
        self.hardwareFile = hardwareFile
        self.command = ["python", hardwareFile, "-c", f"{os.getcwd() + '/' + configFile}"]
        self.running = False

        return
    
    def launch(self):
        if not self.running:
            print(f"Launching: {self.hardwareFile}")
            self.process = Popen(self.command,stdin=PIPE,stdout=PIPE, text=True, bufsize=1)
            self.running = True
        return
    
    def shutdown(self):
        message = f'shutdown,1'
        return self.writeAndRead(message)

    def getProperty(self, property):
        message = f'Get,{property}'
        return self.writeAndRead(message)
    
    def setProperty(self, property, value):
        message = f'Set,{property},{str(value)}'
        return self.writeAndRead(message)

    def run(self, function, *args):
        message = f'Run,{function}'
        for arg in args:
            message += ','+str(arg)
        return self.writeAndRead(message)

    def writeAndRead(self,message):
        _property = "NOT_IN_WORD"
        keyword = message.split(',')[0].lower()
        if 'get' in keyword: 
            _property = message.split(',')[1]
        if self.running:
           print(f"Sending: {message}")
           self.write(message)
           resp = self.read()
           print(f"Received Reply: {resp}")
           if resp == 'OK':
               return 1
           elif resp == 'BAD':
               return -1
           elif _property in resp:
               return resp.split(',')[1]
           else:
               return -1
        return -1

    def write(self, message):
        # self.process.stdin.flush()
        self.process.stdin.write(message + '\n')
        self.process.stdin.flush()
        return
    
    def read(self):

        return self.process.stdout.readline().rstrip()
    

class Listener:

    def __init__(self, hardware) -> None:
        self.hardware = hardware
        self.running = True
        return
    
    def listen(self):
        line = read_input_with_timeout(1)
        # line = sys.stdin.readline().rstrip()#.lower()
        # print(f"Received: {line}", )
        if not isinstance(line, str) or line == '':
            return
        # line = line.lower()
        lineSplit = line.split(',')
        keyword = lineSplit[0].lower()
        if "shutdown" in keyword:
            try:
                self.hardware.__del__()
                self.running = False
                self.write("OK")
            except:
                self.write("BAD")
        elif "get" in keyword:
            try:
                propertyName = lineSplit[1]
                property = getattr(self.hardware, propertyName)
                self.write(propertyName+','+str(property))
            except:
                self.write("BAD")
        elif "set" in keyword:
            try:
                propertyName = lineSplit[1]
                propertyValue = lineSplit[2]
                property = getattr(self.hardware, propertyName)
                setattr(self.hardware, propertyName, type(property)(propertyValue))
                self.write("OK")
            except:
                self.write("BAD")
        elif "run" in keyword:
            # try:
            functionName = lineSplit[1]
            args = []
            if len(lineSplit) > 2:
                for arg in lineSplit[2:]:
                    if is_numeric(arg):
                        args.append(float(arg))
                    else:
                        args.append(arg)
            function = getattr(self.hardware, functionName)
            if len(args) > 0:
                function(*args)
            else:
                function()
            self.write("OK")
            # except:
            #     self.write("BAD")
        else:
            self.write("BAD")

    def write(self,message):
        # print(f"Replying: {message}")
        sys.stdout.write(message + '\n')
        sys.stdout.flush()