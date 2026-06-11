"""Hard-RTC control-plane RPC for pyRTC.

``hardwareLauncher`` (RTC side) and ``Listener`` (child side) exchange
newline-delimited JSON messages over a localhost socket to control
hardware-facing child processes.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import time
from pathlib import Path

from subprocess import PIPE, Popen

from pyRTC.logging_utils import ensure_logging_configured, get_logger
from pyRTC.utils import bind_socket

logger = get_logger(__name__)

def _socket_send_json(sock: socket.socket, message: dict) -> None:
    payload = json.dumps(message, separators=(",", ":")) + "\n"
    sock.sendall(payload.encode("utf-8"))


def _socket_read_json(sock: socket.socket, buffer: str) -> tuple[dict, str]:
    while "\n" not in buffer:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("socket closed")
        buffer += chunk.decode("utf-8")

    line, buffer = buffer.split("\n", 1)
    while line == "":
        if "\n" not in buffer:
            chunk = sock.recv(4096)
            if not chunk:
                raise ConnectionError("socket closed")
            buffer += chunk.decode("utf-8")
        line, buffer = buffer.split("\n", 1)
    return json.loads(line), buffer


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
        self.process = None
        self.processSocket = None
        # Client configuration
        self.host = '127.0.0.1'  # localhost
        self.port = port
        self.timeout = timeout
        self.lastLaunchTime = None
        self.lastContactTime = None
        self.lastError = None
        self._read_buffer = ""

        return

    @property
    def pid(self) -> int | None:
        return getattr(self.process, "pid", None)

    def is_process_alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

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
            self.lastLaunchTime = time.time()
            self.lastError = None

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
            self.lastContactTime = time.time()

        return
    
    def shutdown(self):
        message = {"type": "shutdown"}
        try:
            return self.writeAndRead(message)
        finally:
            self.close(force=False)

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
            try:
                self.write(message)
                reply = self.read()
            except Exception as exc:
                self.lastError = str(exc)
                return -1
            #If there are issues with the reply format
            if not isinstance(reply, dict) or "status" not in reply.keys():
                self.lastError = "invalid launcher reply"
                return -1
            #If there was an issue on the process end
            if reply["status"] == 'BAD':
                self.lastError = "child process returned BAD status"
                return -1
            #If our request went through
            if reply["status"] == 'OK':
                self.lastContactTime = time.time()
                self.lastError = None
                #If the reply came with a property to return
                if "property" in reply.keys():
                    return reply["property"]
                #Otherwise just return OK
                else:
                    return 1
        #default is a fail
        return -1

    def write(self, message):
        _socket_send_json(self.processSocket, message)
        return
    
    def read(self):
        try:
            reply, self._read_buffer = _socket_read_json(self.processSocket, self._read_buffer)
            return reply
        except socket.timeout:
            self.lastError = "socket timeout"
            return -1

    def close(self, *, force: bool = False) -> None:
        self.running = False
        if self.processSocket is not None:
            try:
                self.processSocket.close()
            except Exception:
                logger.debug("Failed to close process socket for %s", self.hardwareFile, exc_info=True)
            self.processSocket = None

        if self.process is None:
            return

        if force and self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=1.0)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    logger.debug("Failed to kill process for %s", self.hardwareFile, exc_info=True)
        self.process = None

    def health_check(self, *, timeout=None) -> dict:
        if not self.is_process_alive():
            exit_code = None if self.process is None else self.process.poll()
            return {
                "state": "failed",
                "pid": self.pid,
                "last_contact_time": self.lastContactTime,
                "error": f"child process exited with code {exit_code}",
            }

        original_timeout = None
        if self.processSocket is not None and timeout is not None and hasattr(self.processSocket, "gettimeout"):
            original_timeout = self.processSocket.gettimeout()
            self.processSocket.settimeout(timeout)

        try:
            running = self.getProperty("running")
        finally:
            if self.processSocket is not None and original_timeout is not None:
                self.processSocket.settimeout(original_timeout)

        if running == -1:
            return {
                "state": "degraded",
                "pid": self.pid,
                "last_contact_time": self.lastContactTime,
                "error": self.lastError or "health check RPC failed",
            }

        if not bool(running):
            return {
                "state": "degraded",
                "pid": self.pid,
                "last_contact_time": self.lastContactTime,
                "error": "component reported running=False",
            }

        return {
            "state": "running",
            "pid": self.pid,
            "last_contact_time": self.lastContactTime,
            "error": None,
        }
        
    

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
        self._read_buffer = ""

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
        _socket_send_json(self.RTCsocket, message)
        return
    
    def read(self):
        reply, self._read_buffer = _socket_read_json(self.RTCsocket, self._read_buffer)
        return reply
