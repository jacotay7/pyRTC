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
import socket
import json 

def work(obj, functionName):
    """
    The main working thread for the any Pipeline object
    """
    #Get what function we need to run
    workFunction = getattr(obj, functionName, None)
    # count = 0
    # N = 10000
    # times = np.zeros(N)
    #If the wfs object is still alive
    while obj.alive:
        #If we are meant to be running
        if obj.running:
            # start = time.time()
            #Call it
            workFunction()
            # time.sleep(1e-5)
            # diff = time.time()-start
            # times[count % N] = diff
            # count += 1
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
            time.sleep(1e-5)
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

    def __init__(self, hardwareFile, configFile, port) -> None:
        self.hardwareFile = hardwareFile
        self.command = ["python", hardwareFile, "-c", f"{configFile}", "-p", f"{port}"]
        self.running = False
        # Client configuration
        self.host = '127.0.0.1'  # localhost
        self.port = port
        return
    
    def launch(self):
        if not self.running:
            print(f"Launching Process: {self.hardwareFile}")
            self.process = Popen(self.command,stdin=PIPE,stdout=PIPE, text=True, bufsize=1)
            self.running = True

            # Create a socket object
            self.processSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            print(f"Waiting for Process at {self.host}:{self.port}")
            connected = False
            restTime = 1
            while not connected:
                time.sleep(restTime)
                try:
                    # Connect to the server
                    self.processSocket.connect((self.host, self.port))
                    connected = True
                except Exception as e:
                    print(f"Connection failed: {e}")
                    print("Retrying in {} seconds...".format(restTime))

            print("Connected")

        return
    
    def shutdown(self):
        message = {"type": "shutdown"}
        return self.writeAndRead(message)

    def getProperty(self, property):
        message = {"type": "get", "property": property}
        return self.writeAndRead(message)
    
    def setProperty(self, property, value):
        message = {"type": "set", "property": property, "value": value}
        return self.writeAndRead(message)

    def run(self, function, *args):
        message = {"type": "run", "function": function}
        for i, arg in enumerate(args):
            message[f"arg_{i+1}"] = arg
        return self.writeAndRead(message)

    def writeAndRead(self,message):
        if self.running:
            self.write(message)
            reply = self.read()
            #If there are issues with the reply format
            if type(reply) != type(dict()) or "status" not in reply.keys():
                return -1
            #If there was an issue on the process end
            if reply["status"] == 'BAD':
                return -1
            #If our request went through
            if reply["status"] == 'OK':
                #If the reply came with a property to return
                if "property" in reply.keys():
                    return reply["property"]
                #Otherwise just return OK
                else:
                    return 1
        #default is a fail
        return -1

    def write(self, message):
        message = json.dumps(message)
        self.processSocket.send(message.encode())
        return
    
    def read(self):
        reply = self.processSocket.recv(4096).decode()
        return json.loads(reply)
    

class Listener:

    def __init__(self, hardware, port) -> None:
        self.hardware = hardware
        self.running = True
        self.keyCharacter = '$'
        self.host = '127.0.0.1'  # localhost
        self.port = port

        # Create a socket object
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        print(f"{hardware.name}: Binding to {self.host}:{self.port}")
        # Bind the socket to a specific address and port
        server_socket.bind((self.host, self.port))
        # Listen for incoming connections
        server_socket.listen()
        print(f"{hardware.name}: Awaiting RTC connection")
        #Connect to the RTC process that spawned you
        self.RTCsocket, self.RTCaddress = server_socket.accept()

        self.OKMessage = {"status": "OK"}
        self.BadMessage = {"status": "BAD"}

        return
    
    def listen(self):

        #Read request from the RTC
        request = self.read()
        if "type" not in request:
            self.write(self.BadMessage)

        #Sort behaviour by request type
        requestType = request["type"]
        if requestType == "shutdown":
            try:
                self.hardware.__del__()
                self.running = False
                self.write(self.OKMessage)
            except:
                self.write(self.BadMessage)
        elif requestType == "get":
            try:
                propertyName = request["property"]
                property = getattr(self.hardware, propertyName)
                message = self.OKMessage.copy()
                message["property"] = property
                self.write(message)
            except:
                self.write(self.BadMessage)
        elif requestType == "set":
            try:
                propertyName = request["property"]
                propertyValue = request["value"]
                property = getattr(self.hardware, propertyName)
                setattr(self.hardware, propertyName, type(property)(propertyValue))
                self.write(self.OKMessage)
            except:
                self.write(self.BadMessage)
        elif requestType == "run":
            try:
                functionName = request["function"]
                args = []
                for i in range(0, len(request.keys())-2):
                    arg = request[f"arg_{i+1}"]
                    args.append(arg)
                function = getattr(self.hardware, functionName)
                if len(args) > 0:
                    function(*args)
                else:
                    function()
                self.write(self.OKMessage)
            except:
                self.write(self.BadMessage)
        else:
            self.write(self.BadMessage)

    def write(self, message):
        message = json.dumps(message)
        self.RTCsocket.send(message.encode())
        return
    
    def read(self):
        reply = self.RTCsocket.recv(4096).decode()
        return json.loads(reply)