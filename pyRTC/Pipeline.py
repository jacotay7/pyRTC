"""
Pipeline Superclasss
"""
from multiprocessing import shared_memory, resource_tracker
from subprocess import PIPE, Popen
import numpy as np
import time
import sys
import argparse
import socket
import json 
import struct
import logging

from pyRTC.utils import *

try:
    import torch
    # Mapping dictionary
    dtype_mapping = {
        np.float32: torch.float32,
        np.float64: torch.float64,
        np.int32: torch.int32,
        np.int64: torch.int64,
        np.uint8: torch.uint8,
        np.uint16: torch.uint16,
        np.dtype("float32"): torch.float32,
        np.dtype("float64"): torch.float64,
        np.dtype("int32"): torch.int32,
        np.dtype("int64"): torch.int64,
        np.dtype("uint8"): torch.uint8,
        np.dtype("uint16"): torch.uint16,
    }
except:
    pass

def work(obj, functionName, affinity):
    """
    The main working thread for the any Pipeline object
    """
    set_affinity_and_priority(functionName, [affinity])
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
            # precise_delay(1)
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
    def __init__(self, name, shape, dtype, gpuDevice=None, consumer=True) -> None:

        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.arr = np.empty(shape, dtype=self.dtype)
        self.size = self.arr.nbytes
        self.metadata = np.zeros(self.METADATA_SIZE, dtype=np.float64)
        self.count = 0
        self.lastWriteTime = 0
        self.lastReadTime = 0
        self.areData = not ("meta" in name)
        self.gpuDevice = gpuDevice

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
            self.updateMetadata(FULL_UPDATE=True)

            if self.gpuDevice is not None:
                self.torchDtype = dtype_mapping.get(self.dtype, None)
                assert(self.torchDtype is not None)
                
                #If we expect the SHM to already exist
                if consumer:
                    self.initGPUMemFromSHM()
                else:
                    self.createGPUMemSHM()

        return

    def __del__(self):

        self.close()

    def createGPUMemSHM(self):

        # Create a GPU tensor
        self.shmGPU = torch.empty(self.shape, dtype=self.torchDtype, device=self.gpuDevice)
        storage = self.shmGPU.untyped_storage()

        # Get all outputs from storage._share_cuda_()
        (
            device_index,
            handle_bytes,
            storage_size_bytes,
            storage_offset_bytes,
            path_bytes,
            unknown,
            additional_bytes,
            is_host_device  # Assuming the 9th element is a boolean
        ) = storage._share_cuda_()

        # Prepare variable-length byte fields
        handle_length = len(handle_bytes)
        path_length = len(path_bytes)
        additional_length = len(additional_bytes)

        # Define header format
        # 'I' - uint32, 'Q' - uint64, 'B' - uint8
        header_format = 'I I I I I I I I I'  # device_index, handle_length, storage_size, storage_offset, size_bytes, view_size, view_offset, is_host_device
        header = struct.pack(
            header_format,
            device_index,
            handle_length,
            storage_size_bytes,
            storage_offset_bytes,
            unknown,
            path_length,
            additional_length,
            int(is_host_device),  # Convert bool to int
            os.getpid(),
        )

        # Total size: header + handle_bytes + path_bytes + additional_bytes
        total_size = struct.calcsize(header_format) + handle_length + path_length + additional_length

        # Create or open the shared memory segment
        # try:
        gpuHandleShm = shared_memory.SharedMemory(
            name=self.name + "_gpu_handle", create=True, size=total_size)
        print(f"Creating New Shared Memory Object {self.name}_gpu_handle")
        # except FileExistsError:
        #     gpuHandleShm = shared_memory.SharedMemory(name=self.name + "_gpu_handle")
        #     print(f"Opening Existing Shared Memory Object {self.name}_gpu_handle")
        # if sys.platform != 'win32':
        #     resource_tracker.unregister(gpuHandleShm._name, 'shared_memory')
        # Write data to shared memory
        buf = gpuHandleShm.buf
        offset = 0

        # Write header
        buf[offset:offset + struct.calcsize(header_format)] = header
        offset += struct.calcsize(header_format)

        print(offset, handle_length, type(offset), type(handle_length), handle_bytes, type(offset + handle_length))

        # Write handle_bytes
        buf[offset:offset + handle_length] = handle_bytes
        offset += handle_length

        # Write path_bytes
        buf[offset:offset + path_length] = path_bytes
        offset += path_length

        # Write additional_bytes
        buf[offset:offset + additional_length] = additional_bytes
        offset += additional_length

        return self.shmGPU

    def initGPUMemFromSHM(self):
        # Open the shared memory segment
        try:
            gpuHandleShm = shared_memory.SharedMemory(name=self.name + "_gpu_handle")
            print(f"Opened Shared Memory Object {self.name}_gpu_handle")
        except:
            self.gpuDevice=None
            logging.log(level=logging.WARNING, msg=f"{self.name}: Trying to initialize GPU memory which does not exist. Defaulting to CPU")
            return
            # raise Exception(f"{self.name}: Trying to initialize GPU memory which does not exist")
        # if sys.platform != 'win32':
        #     resource_tracker.unregister(gpuHandleShm._name, 'shared_memory')
        buf = gpuHandleShm.buf
        offset = 0

        # Define header format
        header_format = 'I I I I I I I I I'  # device_index, handle_length, storage_size, storage_offset, size_bytes, view_size, view_offset, is_host_device
        header_size = struct.calcsize(header_format)

        # Read and unpack header
        header_bytes = bytes(buf[offset:offset + header_size])
        (
            device_index,
            handle_length,
            storage_size_bytes,
            storage_offset_bytes,
            unknown,
            path_length,
            additional_length,
            is_host_device,  # Convert bool to int
            pid
        ) = struct.unpack(header_format, header_bytes)
        if pid == os.getpid():
            raise Exception(f"{self.name}:GPU SHMs only work in hard real-time mode, set gpuDevice to None or remove from config")
        offset += header_size

        # Read handle_bytes
        handle_bytes = bytes(buf[offset:offset + handle_length])
        offset += handle_length

        # Read path_bytes
        path_bytes = bytes(buf[offset:offset + path_length])
        offset += path_length

        # Read additional_bytes
        # Assuming additional_length is known or fixed; alternatively, store it in header
        additional_bytes = bytes(buf[offset:offset + additional_length])
        offset += additional_length

        # Reconstruct the storage
        storage = torch.UntypedStorage._new_shared_cuda(
            device_index,
            handle_bytes,
            storage_size_bytes,
            storage_offset_bytes,
            path_bytes,
            unknown,
            additional_bytes,
            bool(is_host_device)
        )

        # Create a tensor from the storage
        self.shmGPU = torch.tensor([], dtype=self.torchDtype, device=device_index).set_(storage).reshape(self.shape)

        return

    def close(self):
        print(f"Closing {self.name}")
        self.shm.close()
        return

    def write(self, arr):

        #Check if we are a GPU shm
        if self.gpuDevice is not None:
            #If you didn't write a numpy array or tensor
            if not isinstance(arr, np.ndarray) and not isinstance(arr, torch.Tensor): 
                return -1
            #If you wrote the wrong shape
            if arr.shape != self.arr.shape:
                logging.log(level=logging.ERROR, msg=f"{self.name}: Writing Wrong size array to SHM. Expecting {self.arr.shape}, Got {arr.shape}")
                return -1
            #If you passed a tensor
            if isinstance(arr, torch.Tensor):
                #copy the tensor to the GPU shm
                self.shmGPU.copy_(arr)
                #copy a CPU numpy version to the CPU shm
                np.copyto(self.arr, arr.detach().cpu().numpy())
            elif isinstance(arr, np.ndarray):
                #copy a CPU numpy version to the CPU shm
                np.copyto(self.arr, arr)
                #copy the tensor to the GPU shm
                tensor = torch.from_numpy(arr)
                self.shmGPU.copy_(tensor)
        else:
            #If you didn't write a numpy array
            if not isinstance(arr, np.ndarray): 
                return -1
            #If you wrote the wrong shape
            if arr.shape != self.arr.shape:
                logging.log(level=logging.ERROR, msg=f"{self.name}: Writing Wrong size array to SHM. Expecting {self.arr.shape}, Got {arr.shape}")
                return -1
            #Copy to SHM
            np.copyto(self.arr, arr)

        #Update metadata
        self.count += 1
        self.lastWriteTime = time.time()
        if self.areData:
            self.updateMetadata()

        #Return Success
        return 1
    
    def hold(self, timeout=None, RELEASE_GIL = True):
        if timeout is None:
            while not self.checkNew():
                if RELEASE_GIL:
                    time.sleep(1e-5)
                else:
                    precise_delay(5)
        elif isinstance(timeout, float) or isinstance(timeout,int):
            start = time.time()
            while not self.checkNew() and (time.time() - start) < timeout:
                if RELEASE_GIL:
                    time.sleep(1e-5)
                else:
                    precise_delay(5)
        return

    def read(self, SAFE=True, GPU = False, RELEASE_GIL = True):
        self.hold(RELEASE_GIL = RELEASE_GIL)
        return self.read_noblock(SAFE=SAFE, GPU=GPU)    
    
    def read_timeout(self, timeout, SAFE = True, GPU = False, RELEASE_GIL = True):
        self.hold(timeout=timeout, RELEASE_GIL = RELEASE_GIL)
        return self.read_noblock(SAFE=SAFE, GPU=GPU)
    
    def read_noblock(self, SAFE=True, GPU=False):

        #Mark that we have seen the shm before
        self.markSeen()
        #Return a copy of the CPU shm
        if SAFE:
            #If the user asks to read the GPU shm
            if GPU and self.gpuDevice is not None:
                return self.shmGPU.clone()
            else:
                arr = np.copy(self.arr)
                return arr
        else:##EXPERIMENTAL if not safe, return the raw shm memory
            if GPU and self.gpuDevice is not None:
                return self.shmGPU
            else:
                return self.arr

    
    def checkNew(self):
        
        if self.areData:
            # metadata = np.copy(self.metadata)
            if self.metadata[1] != self.lastReadTime:
                self.markSeen()
                return True
        else: #If we are just reading a meta data object directly
            return True
        return False
    
    def markSeen(self):
        self.lastReadTime = self.metadata[1]
        return

    def updateMetadata(self, FULL_UPDATE=False):
        
        # self.metadata = np.zeros_like(self.metadata)
        self.metadata[0] = self.count
        self.metadata[1] = self.lastWriteTime
        if FULL_UPDATE:
            self.metadata[2] = self.size
            self.metadata[3] = dtype_to_float(self.arr.dtype)
            for i in range(len(self.arr.shape)):
                if i + 4 < self.metadata.size:
                    self.metadata[i+4] = self.arr.shape[i]
        # np.copyto(self.metadata, metadata)
        return
    
def clear_shms(names):
    
    for n in names:
        shm = ImageSHM(n,(1,),np.uint8)
        shm.shm.unlink()
        shm = ImageSHM(n+"_meta",(1,),np.uint8)
        shm.shm.unlink()
        try:
            shm = ImageSHM(n+"_gpu_handle",(1,),np.uint8)
            shm.shm.unlink()
        except:
            pass


class hardwareLauncher:

    def __init__(self, hardwareFile, configFile, port, timeout=None) -> None:
        self.hardwareFile = hardwareFile
        self.command = ["python", hardwareFile, "-c", f"{configFile}", "-p", f"{port}"]
        self.running = False
        # Client configuration
        self.host = '127.0.0.1'  # localhost
        self.port = port
        self.timeout = timeout

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

    def __init__(self, hardware, port) -> None:
        self.hardware = hardware
        self.running = True
        self.keyCharacter = '$'
        self.host = '127.0.0.1'  # localhost
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
    
def initExistingShm(shmName, gpuDevice=None):
    #Read wfc metadata and open a stream to the shared memory
    shmMeta = ImageSHM(shmName+"_meta", (ImageSHM.METADATA_SIZE,), np.float64).read_noblock()
    shmDType = float_to_dtype(shmMeta[3])
    shmSize = int(shmMeta[2]//shmDType.itemsize)
    shmDims = []
    i = 0
    while int(shmMeta[4+i]) > 0:
        shmDims.append(int(shmMeta[4+i]))
        i += 1
    shm = ImageSHM(shmName, shmDims, shmDType, gpuDevice=gpuDevice, consumer=True)
    return shm, shmDims, shmDType


def launchComponent(component, confKey, start = True):

    # Create argument parser
    parser = argparse.ArgumentParser(description="Read a config file from the command line.")

    # Add command-line argument for the config file
    parser.add_argument("-c", "--config", required=True, help="Path to the config file")
    parser.add_argument("-p", "--port", required=True, help="Port for communication")

    # Parse command-line arguments
    args = parser.parse_args()

    conf = read_yaml_file(args.config)[confKey]

    set_affinity_and_priority("", setFromConfig(conf, "affinity", 0))

    obj = component(conf=conf)
    obj.RELEASE_GIL = False
    if start:
        obj.start()
    
    l = Listener(obj, port= int(args.port))
    while l.running:
        l.listen()
        time.sleep(1e-3)
