"""Shared-memory transport and hard-RTC process helpers for pyRTC.

This module contains the infrastructure that lets pyRTC components exchange
frames and command vectors through named shared-memory blocks, optionally mirror
those blocks onto CUDA tensors for compatible deployments, and launch hardware-
facing child processes that communicate with the main RTC over a small
localhost-based JSON protocol.
"""
import argparse
from dataclasses import dataclass
import importlib
import inspect
import json 
import logging
import os
import socket
import struct
import sys
import time
from pathlib import Path

from multiprocessing import shared_memory, resource_tracker
from subprocess import PIPE, Popen

import numpy as np

from pyRTC.logging_utils import add_logging_cli_args, configure_logging_from_args, ensure_logging_configured, get_logger
from pyRTC.utils import (
    bind_socket,
    dtype_to_float,
    float_to_dtype,
    precise_delay,
    setFromConfig,
    set_affinity_and_priority,
)


logger = get_logger(__name__)

TORCH_AVAILABLE = False
torch = None
dtype_mapping = {}

try:
    import torch
    TORCH_AVAILABLE = True
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
except Exception:
    TORCH_AVAILABLE = False


def gpu_torch_available() -> bool:
    return TORCH_AVAILABLE


def normalize_gpu_device(gpuDevice, context: str = ""):
    if gpuDevice is None:
        return None
    if not TORCH_AVAILABLE:
        prefix = f"{context}: " if context else ""
        logging.log(
            level=logging.WARNING,
            msg=f"{prefix}gpuDevice was requested but PyTorch is not installed; defaulting to CPU mode.",
        )
        return None
    return gpuDevice

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
    """Named shared-memory array with metadata and optional GPU mirror state.

    ``ImageSHM`` is the transport primitive used throughout pyRTC. Each stream
    has a CPU shared-memory block, a small metadata block containing shape,
    dtype, and timing information, and optionally a GPU-backed tensor mirror in
    hard-RTC deployments where CUDA sharing is supported.

    Producers create the stream and update it with NumPy arrays. Consumers
    reconstruct the stream by name and read either safe copies or direct views,
    depending on their performance and synchronization needs.
    """

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
        self.areData = "meta" not in name
        self.gpuDevice = normalize_gpu_device(gpuDevice, name)

        try:
            self.shm = shared_memory.SharedMemory(name= name, create=True, size=self.arr.nbytes)
            logger.debug("Creating shared memory object %s", self.name)
        except Exception:
            self.shm = shared_memory.SharedMemory(name=name)
            logger.debug("Opening existing shared memory object %s", self.name)

        #Doesn't work in windows
        if sys.platform != 'win32':
            resource_tracker.unregister(self.shm._name, 'shared_memory')
            
        self.arr = np.ndarray(shape, dtype=dtype, buffer=self.shm.buf)
        #If we are not opening a metadata shm
        if self.areData:
            #Create/Open an associated metadata SHM 
            try:
                self.metadataShm = shared_memory.SharedMemory(name= name+"_meta", create=True, size=self.metadata.nbytes)
                logger.debug("Creating shared memory object %s_meta", self.name)
            except Exception:
                self.metadataShm = shared_memory.SharedMemory(name= name+"_meta")
                logger.debug("Opening existing shared memory object %s_meta", self.name)
            if sys.platform != 'win32':
                resource_tracker.unregister(self.metadataShm._name, 'shared_memory')
            self.metadata = np.ndarray(self.metadata.shape, dtype=self.metadata.dtype, buffer=self.metadataShm.buf)
            self.updateMetadata(FULL_UPDATE=True)

            if self.gpuDevice is not None:
                self.torchDtype = dtype_mapping.get(self.dtype, None)
                if self.torchDtype is None:
                    self.gpuDevice = None
                    logging.log(level=logging.WARNING, msg=f"{self.name}: dtype {self.dtype} not supported for GPU SHM; defaulting to CPU mode.")
                    return
                
                #If we expect the SHM to already exist
                if consumer:
                    self.initGPUMemFromSHM()
                else:
                    self.createGPUMemSHM()

        return

    def __del__(self):

        self.close()

    def createGPUMemSHM(self):

        if self.gpuDevice is None or not TORCH_AVAILABLE:
            return None

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
        logger.debug("Creating shared memory object %s_gpu_handle", self.name)
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
        if self.gpuDevice is None or not TORCH_AVAILABLE:
            return

        # Open the shared memory segment
        try:
            gpuHandleShm = shared_memory.SharedMemory(name=self.name + "_gpu_handle")
            logger.debug("Opened shared memory object %s_gpu_handle", self.name)
        except Exception:
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
        logger.debug("Closing shared memory object %s", self.name)
        try:
            self.shm.close()
        except Exception:
            logger.debug("Failed to close data SHM %s", self.name, exc_info=True)

        metadata_shm = getattr(self, "metadataShm", None)
        if metadata_shm is not None:
            try:
                metadata_shm.close()
            except Exception:
                logger.debug("Failed to close metadata SHM %s_meta", self.name, exc_info=True)
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
                np.copyto(self.arr, arr.cpu().numpy())
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
    def _close_and_unlink(name):
        try:
            shm = shared_memory.SharedMemory(name=name)
        except FileNotFoundError:
            return
        except Exception:
            logger.debug("Failed opening shared memory object %s for cleanup", name, exc_info=True)
            return

        try:
            shm.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            logger.debug("Failed unlinking shared memory object %s", name, exc_info=True)
        finally:
            try:
                shm.close()
            except Exception:
                logger.debug("Failed closing shared memory object %s", name, exc_info=True)

    for n in names:
        _close_and_unlink(n)
        _close_and_unlink(f"{n}_meta")
        _close_and_unlink(f"{n}_gpu_handle")


class hardwareLauncher:
    """Launch and supervise a hardware-side child process.

    The launcher is the client-side helper for pyRTC's hard-RTC deployment
    model. It starts a Python subprocess, waits for the child to expose a socket
    listener, and then sends simple JSON messages to get or set properties,
    invoke helper methods, or request shutdown.

    Logging-related environment variables are propagated so parent and child
    processes share the same operator-facing logging policy.
    """

    def __init__(self, hardwareFile, configFile, port, timeout=None) -> None:
        self.hardwareFile = hardwareFile
        self.command = [sys.executable, hardwareFile, "-c", f"{configFile}", "-p", f"{port}"]
        self.running = False
        # Client configuration
        self.host = '127.0.0.1'  # localhost
        self.port = port
        self.timeout = timeout

        return

    @staticmethod
    def _discover_pythonpath_root(hardware_file: str) -> str | None:
        script_path = Path(hardware_file).resolve()
        for parent in (script_path.parent, *script_path.parents):
            if (parent / "pyproject.toml").exists() and (parent / "pyRTC").is_dir():
                return str(parent)
        return None
    
    def launch(self):
        ensure_logging_configured(app_name="pyrtc-hardware-launcher", component_name=self.hardwareFile)
        if not self.running:
            logger.info("Launching process %s", self.hardwareFile)
            child_env = os.environ.copy()
            pythonpath_root = self._discover_pythonpath_root(self.hardwareFile)
            if pythonpath_root is not None:
                existing_pythonpath = child_env.get("PYTHONPATH", "")
                if existing_pythonpath:
                    child_env["PYTHONPATH"] = f"{pythonpath_root}{os.pathsep}{existing_pythonpath}"
                else:
                    child_env["PYTHONPATH"] = pythonpath_root
            self.process = Popen(self.command,stdin=PIPE,stdout=PIPE, text=True, bufsize=1, env=child_env)
            self.running = True

            # Create a socket object
            self.processSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            logger.info("Waiting for process at %s:%s", self.host, self.port)
            connected = False
            restTime = 2
            while not connected:
                time.sleep(restTime)
                try:
                    # Connect to the server
                    self.processSocket.connect((self.host, self.port))
                    connected = True
                except Exception as e:
                    logger.warning("Connection failed: %s", e)
                    logger.info("Retrying in %s seconds", restTime)

            if isinstance(self.timeout,float) or isinstance(self.timeout,int):
                self.processSocket.settimeout(self.timeout)

            logger.info("Connected to child process socket")

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
            if not isinstance(reply, dict) or "status" not in reply.keys():
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
    """Server-side control socket for a launched hardware object.

    ``Listener`` is the child-process counterpart to :class:`hardwareLauncher`.
    It binds a localhost socket, accepts the RTC-side connection, and services a
    narrow JSON RPC surface for property access, method calls, and clean
    shutdown.
    """

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
        logger.info("%s: awaiting RTC connection", hardware.name)
        #Connect to the RTC process that spawned you
        self.RTCsocket, self.RTCaddress = server_socket.accept()

        self.OKMessage = {"status": "OK"}
        self.BadMessage = {"status": "BAD"}

        return
    
    def listen(self):
        try:
            request = self.read()
        except Exception:
            logger.exception("Failed to read listener request")
            self.write(self.BadMessage)
            return
        if "type" not in request:
            self.write(self.BadMessage)
            logger.error("Listener request missing type field: %s", request)
            return

        #Sort behaviour by request type
        requestType = request["type"]
        if requestType == "shutdown":
            try:
                self.hardware.__del__()
                self.running = False
                self.write(self.OKMessage)
            except Exception:
                logger.exception("Listener shutdown request failed")
                self.write(self.BadMessage)
        elif requestType == "get":
            try:
                propertyName = request["property"]
                property = getattr(self.hardware, propertyName)
                message = self.OKMessage.copy()
                message["property"] = property
                self.write(message)
            except Exception:
                logger.exception("Listener get request failed for %s", request.get("property"))
                self.write(self.BadMessage)
        elif requestType == "set":
            try:
                propertyName = request["property"]
                propertyValue = request["value"]
                property = getattr(self.hardware, propertyName)
                setattr(self.hardware, propertyName, type(property)(propertyValue))
                self.write(self.OKMessage)
            except Exception:
                logger.exception("Listener set request failed for %s", request.get("property"))
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
            except Exception:
                logger.exception("Listener run request failed for %s", request.get("function"))
                self.write(self.BadMessage)
        else:
            logger.error("Unknown listener request type: %s", requestType)
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
    shmDims = []
    i = 0
    while int(shmMeta[4+i]) > 0:
        shmDims.append(int(shmMeta[4+i]))
        i += 1
    shm = ImageSHM(shmName, shmDims, shmDType, gpuDevice=gpuDevice, consumer=True)
    return shm, shmDims, shmDType


def launchComponent(component, confKey, start = True):
    from pyRTC.config_schema import read_system_config

    # Create argument parser
    parser = argparse.ArgumentParser(description="Read a config file from the command line.")

    # Add command-line argument for the config file
    parser.add_argument("-c", "--config", required=True, help="Path to the config file")
    parser.add_argument("-p", "--port", required=True, help="Port for communication")
    add_logging_cli_args(parser)

    # Parse command-line arguments
    args = parser.parse_args()
    configure_logging_from_args(args, app_name=f"pyrtc-{confKey}", component_name=confKey)

    conf = read_system_config(args.config)[confKey]

    set_affinity_and_priority("", setFromConfig(conf, "affinity", 0))

    try:
        obj = component(conf=conf)
        obj.RELEASE_GIL = False
        if start:
            obj.start()

        listener = Listener(obj, port= int(args.port))
        while listener.running:
            listener.listen()
            time.sleep(1e-3)
    except Exception:
        logger.exception("Failed to launch component %s", confKey)
        raise


DEFAULT_COMPONENT_ORDER = ("modulator", "wfc", "wfs", "slopes", "loop", "psf", "telemetry")


@dataclass
class ComponentRuntimeStatus:
    section_name: str
    mode: str
    state: str
    component_class: str
    error: str | None = None
    port: int | None = None
    target: str | None = None

    def to_dict(self) -> dict:
        return {
            "section_name": self.section_name,
            "mode": self.mode,
            "state": self.state,
            "component_class": self.component_class,
            "error": self.error,
            "port": self.port,
            "target": self.target,
        }


class BaseComponentRuntime:
    def __init__(self, section_name, mode, component_class) -> None:
        self.section_name = section_name
        self.mode = mode
        self.component_class = component_class
        self.state = "created"
        self.error = None

    @property
    def component_class_path(self) -> str:
        return f"{self.component_class.__module__}.{self.component_class.__name__}"

    def status(self) -> dict:
        return ComponentRuntimeStatus(
            section_name=self.section_name,
            mode=self.mode,
            state=self.state,
            component_class=self.component_class_path,
            error=self.error,
        ).to_dict()


class SoftComponentRuntime(BaseComponentRuntime):
    def __init__(self, section_name, component_class, conf: dict) -> None:
        super().__init__(section_name=section_name, mode="soft-rtc", component_class=component_class)
        self.conf = conf
        self.component = None

    def start(self) -> None:
        if self.state == "running":
            return
        try:
            if self.component is None:
                self.component = self.component_class(self.conf)
            self.component.start()
            self.state = "running"
            self.error = None
        except Exception as exc:
            self.state = "failed"
            self.error = str(exc)
            raise

    def stop(self) -> None:
        if self.component is None:
            self.state = "stopped"
            return
        try:
            self.component.stop()
            self.state = "stopped"
            self.error = None
        except Exception as exc:
            self.state = "failed"
            self.error = str(exc)
            raise


class HardComponentRuntime(BaseComponentRuntime):
    def __init__(
        self,
        section_name,
        component_class,
        script_path: str,
        config_path: str,
        port: int,
        *,
        launcher_cls=hardwareLauncher,
    ) -> None:
        super().__init__(section_name=section_name, mode="hard-rtc", component_class=component_class)
        self.script_path = script_path
        self.config_path = config_path
        self.port = port
        self.launcher_cls = launcher_cls
        self.launcher = None

    def start(self) -> None:
        if self.state == "running":
            return
        try:
            if self.launcher is None:
                self.launcher = self.launcher_cls(self.script_path, self.config_path, self.port)
                self.launcher.launch()
            self.launcher.run("start")
            self.state = "running"
            self.error = None
        except Exception as exc:
            self.state = "failed"
            self.error = str(exc)
            raise

    def stop(self) -> None:
        if self.launcher is None:
            self.state = "stopped"
            return
        try:
            self.launcher.run("stop")
        except Exception:
            pass
        try:
            self.launcher.shutdown()
        except Exception as exc:
            self.state = "failed"
            self.error = str(exc)
            raise
        self.state = "stopped"
        self.error = None

    def status(self) -> dict:
        payload = super().status()
        payload.update({"port": self.port, "target": self.script_path})
        return payload


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _import_symbol(path_or_name: str):
    if "." in path_or_name:
        module_name, attr_name = path_or_name.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)

    for module_name in ("pyRTC.hardware", "pyRTC"):
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, path_or_name):
                return getattr(module, path_or_name)
        except Exception:
            continue
    raise ImportError(f"Unable to resolve component symbol '{path_or_name}'")


def _normalize_manager_mode(mode: str | None) -> str | None:
    if mode is None:
        return None
    normalized = str(mode).strip().lower()
    mode_aliases = {
        "soft": "soft-rtc",
        "soft-rtc": "soft-rtc",
        "hard": "hard-rtc",
        "hard-rtc": "hard-rtc",
    }
    if normalized not in mode_aliases:
        raise ValueError("mode must be one of: soft, soft-rtc, hard, hard-rtc")
    return mode_aliases[normalized]


class RTCManager:
    """Validate, launch, stop, and inspect a pyRTC system as one unit.

    The implementation intentionally lives in ``pyRTC.Pipeline`` because this
    orchestration layer is an extension of the existing shared-memory and
    launcher runtime rather than a separate subsystem.
    """

    def __init__(
        self,
        config: dict,
        *,
        config_path: str | None = None,
        mode: str | None = None,
        launcher_cls=hardwareLauncher,
    ) -> None:
        self.config = dict(config)
        self.config_path = config_path
        self.mode = _normalize_manager_mode(mode)
        if self.mode is not None:
            manager_conf = dict(self.config.get("manager", {}))
            manager_conf["mode"] = self.mode
            self.config["manager"] = manager_conf
        self.launcher_cls = launcher_cls
        self.validated = False
        self.state = "created"
        self.error = None
        self.runtimes = {}

    @classmethod
    def from_config_file(cls, config_path: str | Path, *, mode: str | None = None, launcher_cls=hardwareLauncher):
        from pyRTC.config_schema import read_system_config

        normalized = read_system_config(config_path)
        manager = cls(normalized, config_path=str(config_path), mode=mode, launcher_cls=launcher_cls)
        manager.validated = True
        manager.state = "validated"
        return manager

    @classmethod
    def from_config(
        cls,
        config: dict,
        *,
        config_path: str | None = None,
        mode: str | None = None,
        launcher_cls=hardwareLauncher,
    ):
        return cls(config, config_path=config_path, mode=mode, launcher_cls=launcher_cls)

    def validate(self) -> dict:
        from pyRTC.config_schema import validate_system_config

        self.config = validate_system_config(self.config, config_path=self.config_path)
        self.validated = True
        self.state = "validated"
        self.error = None
        return self.config

    def _component_sections(self) -> list:
        from pyRTC.component_descriptors import list_component_sections

        sections = [section for section in DEFAULT_COMPONENT_ORDER if section in self.config]
        for section in list_component_sections():
            if section in self.config and section not in sections:
                sections.append(section)
        manager_conf = self.config.get("manager", {})
        for mapping_name in ("componentModes", "ports", "componentClasses", "componentFiles"):
            mapping = manager_conf.get(mapping_name, {})
            if not isinstance(mapping, dict):
                continue
            for section in mapping:
                if section in self.config and section not in sections:
                    sections.append(section)
        return sections

    def _resolve_component_class(self, section_name: str):
        from pyRTC.component_descriptors import get_component_descriptor

        manager_conf = self.config.get("manager", {})
        component_classes = manager_conf.get("componentClasses", {})
        target = component_classes.get(section_name)
        if target is None:
            section_conf = self.config[section_name]
            target = section_conf.get("name")

        if target is not None:
            if inspect.isclass(target):
                return target
            return _import_symbol(target)

        descriptor = get_component_descriptor(section_name)
        if descriptor is None:
            raise ValueError(f"No component descriptor found for section '{section_name}'")
        return descriptor.component_class

    def _resolve_script_path(self, section_name: str, component_class) -> str:
        manager_conf = self.config.get("manager", {})
        component_files = manager_conf.get("componentFiles", {})
        target = component_files.get(section_name)
        if target is not None:
            return str(target)

        source_file = inspect.getsourcefile(component_class)
        if source_file is None:
            raise ValueError(f"Unable to determine script path for section '{section_name}'")
        return source_file

    def _resolve_component_mode(self, section_name: str) -> str:
        manager_conf = self.config.get("manager", {})
        component_modes = manager_conf.get("componentModes", {})
        return component_modes.get(section_name, manager_conf.get("mode", "soft-rtc"))

    def _resolve_port(self, section_name: str) -> int:
        manager_conf = self.config.get("manager", {})
        ports = manager_conf.get("ports", {})
        return int(ports.get(section_name, _find_free_port()))

    def _build_runtimes(self) -> None:
        if self.runtimes:
            return
        for section_name in self._component_sections():
            component_class = self._resolve_component_class(section_name)
            mode = self._resolve_component_mode(section_name)
            if mode == "soft-rtc":
                runtime = SoftComponentRuntime(section_name, component_class, self.config[section_name])
            else:
                if not self.config_path:
                    raise ValueError(
                        f"manager: hard-rtc component '{section_name}' requires a config_path so child processes can load the YAML"
                    )
                runtime = HardComponentRuntime(
                    section_name,
                    component_class,
                    self._resolve_script_path(section_name, component_class),
                    self.config_path,
                    self._resolve_port(section_name),
                    launcher_cls=self.launcher_cls,
                )
            self.runtimes[section_name] = runtime

    def start(self) -> None:
        if not self.validated:
            self.validate()
        self._build_runtimes()
        self.state = "starting"
        started = []
        try:
            for section_name in self._component_sections():
                runtime = self.runtimes[section_name]
                runtime.start()
                started.append(runtime)
        except Exception as exc:
            self.state = "failed"
            self.error = str(exc)
            for runtime in reversed(started):
                try:
                    runtime.stop()
                except Exception:
                    pass
            raise
        self.state = "running"
        self.error = None

    def stop(self) -> None:
        if self.state in {"stopped", "created", "validated"} and not self.runtimes:
            self.state = "stopped"
            return
        self.state = "stopping"
        failures = []
        for section_name in reversed(self._component_sections()):
            runtime = self.runtimes.get(section_name)
            if runtime is None:
                continue
            try:
                runtime.stop()
            except Exception as exc:
                failures.append(f"{section_name}: {exc}")
        if failures:
            self.state = "failed"
            self.error = "; ".join(failures)
            raise RuntimeError(self.error)
        self.state = "stopped"
        self.error = None

    def status(self) -> dict:
        return {
            "state": self.state,
            "mode": self.config.get("manager", {}).get("mode", "soft-rtc"),
            "validated": self.validated,
            "config_path": self.config_path,
            "error": self.error,
            "components": {
                section_name: runtime.status()
                for section_name, runtime in self.runtimes.items()
            },
        }

    def get_component(self, section_name: str):
        runtime = self.runtimes[section_name]
        return getattr(runtime, "component", getattr(runtime, "launcher", None))
