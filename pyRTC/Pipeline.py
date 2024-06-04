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

    METADATA_SIZE = 10
    def __init__(self, name, shape, dtype) -> None:

        self.name = name
        self.arr = np.empty(shape, dtype=dtype)
        self.size = self.arr.nbytes
        self.metadata = np.zeros(self.METADATA_SIZE, dtype=np.float64)
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

        #Doesn't work in windows
        if sys.platform != 'win32':
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
            if sys.platform != 'win32':
                resource_tracker.unregister(self.metadataShm._name, 'shared_memory')
            self.metadata = np.ndarray(self.metadata.shape, dtype=self.metadata.dtype, buffer=self.metadataShm.buf)
            self.updateMetadata()

        return

    def __del__(self):

        self.close()

    def close(self):
        print(f"Closing {self.name}")
        self.shm.close()
        return

    def write(self, arr):

        if not isinstance(arr, np.ndarray) or arr.shape != self.arr.shape:
            return -1
        np.copyto(self.arr, arr)
        self.count += 1
        self.lastWriteTime = time.time()
        if self.areData:
            self.updateMetadata()
        return 1
    
    def read(self):
        while not self.checkNew():
            time.sleep(1e-5)
        arr = np.copy(self.arr)
        return arr
    
    def read_timeout(self, timeout):
        start = time.time()
        while not self.checkNew() and (time.time() - start) < timeout:
            time.sleep(1e-5)
        arr = np.copy(self.arr)
        return arr
    
    def read_noblock(self):
        arr = np.copy(self.arr)
        return arr

    def read_noblock_safe(self):
        return self.read_noblock()
    
    def checkNew(self):
        
        if self.areData:
            # metadata = np.copy(self.metadata)
            if self.metadata[1] != self.lastReadTime:
                self.markSeen()
                # self.lastReadTime = self.metadata[1]
                return True
        else: #If we are just reading a meta data object directly
            return True
        return False
    
    def markSeen(self):
        self.lastReadTime = self.metadata[1]
        return

    def updateMetadata(self):
        
        metadata = np.zeros_like(self.metadata)
        metadata[0] = self.count
        metadata[1] = self.lastWriteTime
        metadata[2] = self.size
        metadata[3] = dtype_to_float(self.arr.dtype)
        for i in range(len(self.arr.shape)):
            if i + 4 < self.metadata.size:
                metadata[i+4] = self.arr.shape[i]
        np.copyto(self.metadata, metadata)
        return
    
def clear_shms(names):
    
    for n in names:
        shm = ImageSHM(n,(1,),np.uint8)
        shm.shm.unlink()
        shm = ImageSHM(n+"_meta",(1,),np.uint8)
        shm.shm.unlink()


class hardwareLauncher:

    def __init__(self, hardwareFile, configFile, port, remoteProcess=False, timeout=None) -> None:
        self.hardwareFile = hardwareFile
        self.command = ["python", hardwareFile, "-c", f"{configFile}", "-p", f"{port}"]
        self.running = False
        # Client configuration
        self.host = '127.0.0.1'  # localhost
        self.port = port
        self.remoteProcess = remoteProcess
        self.timeout = timeout

        return
    
    def launch(self):
        if not self.running:
            if not self.remoteProcess:
                print(f"Launching Process: {self.hardwareFile}")
                self.process = Popen(self.command,stdin=PIPE,stdout=PIPE, text=True, bufsize=1)
                
            # Create a socket object
            self.processSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            print(f"Waiting for Process at {self.host}:{self.port}")
            connected = False
            restTime = 2
            while not connected:
                time.sleep(restTime)
                try:
                    # Connect to the server
                    self.processSocket.connect((self.host, self.port))
                    connected = True
                except Exception as e:
                    print(f"Connection failed: {e}")
                    print("Retrying in {} seconds...".format(restTime))

            self.running = True
            if isinstance(self.timeout,float) or isinstance(self.timeout,int):
                self.processSocket.settimeout(self.timeout)

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

    def run(self, function, *args, timeout = None):
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
        try:
            reply = self.processSocket.recv(4096).decode()
            return json.loads(reply)
        except socket.timeout:
            return -1
        
    

class Listener:

    def __init__(self, hardware, port, host = '127.0.0.1') -> None:
        self.hardware = hardware
        self.running = True
        self.keyCharacter = '$'
        self.host = host  # default localhost
        self.port = port

        server_socket = bind_socket(self.host, self.port)

        # # Create a socket object
        # server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # try:
        #     print(f"{hardware.name}: Binding to {self.host}:{self.port}")
        #     # Bind the socket to a specific address and port
        #     server_socket.bind((self.host, self.port))
        # except OSError as
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
    
def initExistingShm(shmName):
    #Read wfc metadata and open a stream to the shared memory
    shmMeta = ImageSHM(shmName+"_meta", (ImageSHM.METADATA_SIZE,), np.float64).read_noblock_safe()
    shmDType = float_to_dtype(shmMeta[3])
    shmSize = int(shmMeta[2]//shmDType.itemsize)
    shmDims = []
    i = 0
    while int(shmMeta[4+i]) > 0:
        shmDims.append(int(shmMeta[4+i]))
        i += 1
    shm = ImageSHM(shmName, shmDims, shmDType)
    return shm, shmDims, shmDType