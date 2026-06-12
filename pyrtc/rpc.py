"""Hard-RTC control-plane RPC for pyrtc.

``HardwareLauncher`` (RTC side) and ``Listener`` (child side) exchange
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

from pyrtc.logging_utils import ensure_logging_configured, get_logger
from pyrtc.utils import bind_socket

logger = get_logger(__name__)

# Version of the launcher/listener message envelope. Bump when the message
# format changes incompatibly; both sides reject mismatched versions.
PROTOCOL_VERSION = 1


def _coerce_property_value(current, value):
    """Coerce an RPC-provided value onto the type of the current property.

    JSON round-trips lose Python types, so the listener re-types incoming
    values against the property's current value. Unlike a bare
    ``type(current)(value)``, this handles bools (``bool("False")`` is
    ``True``) and leaves the raw value alone when the current value is
    ``None`` or no sensible coercion exists.
    """

    if current is None:
        return value
    if isinstance(current, bool):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("true", "1", "yes", "on"):
                return True
            if lowered in ("false", "0", "no", "off"):
                return False
            raise ValueError(f"cannot interpret {value!r} as a boolean")
        raise ValueError(f"cannot interpret {value!r} as a boolean")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if isinstance(current, str):
        return str(value)
    if isinstance(current, (list, tuple)):
        return type(current)(value)
    return value


def _json_safe(value):
    """Return a JSON-serializable version of ``value``, or ``None`` if none.

    NumPy scalars and arrays are converted via ``item()``/``tolist()``; other
    unserializable results are dropped (the RPC reply stays a plain OK).
    """

    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        pass
    for converter in ("item", "tolist"):
        method = getattr(value, converter, None)
        if callable(method):
            try:
                converted = method()
                json.dumps(converted)
                return converted
            except (TypeError, ValueError):
                continue
    return None


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


class HardwareLauncher:
    """Launch and supervise a hardware-side child process.

    The launcher is the client-side helper for pyrtc's hard-RTC deployment
    model. It starts a Python subprocess, waits for the child to expose a socket
    listener, and then sends simple JSON messages to get or set properties,
    invoke helper methods, or request shutdown.

    Logging-related environment variables are propagated so parent and child
    processes share the same operator-facing logging policy.
    """

    def __init__(self, hardware_file, config_file, port, timeout=None) -> None:
        self.hardware_file = hardware_file
        self.command = [sys.executable, hardware_file, "-c", f"{config_file}", "-p", f"{port}"]
        self.running = False
        self.process = None
        self.process_socket = None
        # Client configuration
        self.host = '127.0.0.1'  # localhost
        self.port = port
        self.timeout = timeout
        self.last_launch_time = None
        self.last_contact_time = None
        self.last_error = None
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
            if (parent / "pyproject.toml").exists() and (parent / "pyrtc").is_dir():
                return str(parent)
        return None

    def launch(self):
        ensure_logging_configured(app_name="pyrtc-hardware-launcher", component_name=self.hardware_file)
        if not self.running:
            logger.info("Launching process %s", self.hardware_file)
            child_env = os.environ.copy()
            pythonpath_root = self._discover_pythonpath_root(self.hardware_file)
            if pythonpath_root is not None:
                existing_pythonpath = child_env.get("PYTHONPATH", "")
                if existing_pythonpath:
                    child_env["PYTHONPATH"] = f"{pythonpath_root}{os.pathsep}{existing_pythonpath}"
                else:
                    child_env["PYTHONPATH"] = pythonpath_root
            self.process = Popen(self.command,stdin=PIPE,stdout=PIPE, text=True, bufsize=1, env=child_env)
            self.running = True
            self.last_launch_time = time.time()
            self.last_error = None

            # Create a socket object
            self.process_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            logger.info("Waiting for process at %s:%s", self.host, self.port)
            connected = False
            rest_time = 2
            while not connected:
                time.sleep(rest_time)
                try:
                    # Connect to the server
                    self.process_socket.connect((self.host, self.port))
                    connected = True
                except Exception as e:
                    logger.warning("Connection failed: %s", e)
                    logger.info("Retrying in %s seconds", rest_time)

            if isinstance(self.timeout,float) or isinstance(self.timeout,int):
                self.process_socket.settimeout(self.timeout)

            logger.info("Connected to child process socket")
            self.last_contact_time = time.time()

        return

    def shutdown(self):
        message = {"type": "shutdown", "protocol": PROTOCOL_VERSION}
        try:
            return self.write_and_read(message)
        finally:
            self.close(force=False)

    def get_property(self, property):
        message = {"type": "get", "property": property, "protocol": PROTOCOL_VERSION}
        return self.write_and_read(message)

    def set_property(self, property, value):
        message = {"type": "set", "property": property, "value": value, "protocol": PROTOCOL_VERSION}
        return self.write_and_read(message)

    def run(self, function, *args, timeout=None):
        """Invoke a method on the remote component.

        Returns the method's return value when the child reports one (it must
        be JSON-serializable), otherwise ``1`` for success and ``-1`` for any
        failure. ``timeout`` temporarily overrides the socket timeout for this
        call only — long-running calls such as ``compute_im`` need more than
        the default RPC timeout.
        """
        message = {
            "type": "run",
            "function": function,
            "args": list(args),
            "protocol": PROTOCOL_VERSION,
        }
        return self.write_and_read(message, timeout=timeout)

    def write_and_read(self, message, *, timeout=None):
        if not self.running:
            return -1

        original_timeout = None
        override_timeout = (
            timeout is not None
            and self.process_socket is not None
            and hasattr(self.process_socket, "gettimeout")
        )
        if override_timeout:
            original_timeout = self.process_socket.gettimeout()
            self.process_socket.settimeout(float(timeout))
        try:
            try:
                self.write(message)
                reply = self.read()
            except Exception as exc:
                self.last_error = str(exc)
                return -1
        finally:
            if override_timeout and self.process_socket is not None:
                self.process_socket.settimeout(original_timeout)

        #If there are issues with the reply format
        if not isinstance(reply, dict) or "status" not in reply.keys():
            self.last_error = "invalid launcher reply"
            return -1
        #If there was an issue on the process end
        if reply["status"] == 'BAD':
            self.last_error = str(reply.get("error", "child process returned BAD status"))
            return -1
        #If our request went through
        if reply["status"] == 'OK':
            self.last_contact_time = time.time()
            self.last_error = None
            #If the reply came with a property to return
            if "property" in reply.keys():
                return reply["property"]
            #Otherwise just return OK
            else:
                return 1
        #default is a fail
        return -1

    def write(self, message):
        _socket_send_json(self.process_socket, message)
        return

    def read(self):
        try:
            reply, self._read_buffer = _socket_read_json(self.process_socket, self._read_buffer)
            return reply
        except socket.timeout:
            self.last_error = "socket timeout"
            return -1

    def close(self, *, force: bool = False) -> None:
        self.running = False
        if self.process_socket is not None:
            try:
                self.process_socket.close()
            except Exception:
                logger.debug("Failed to close process socket for %s", self.hardware_file, exc_info=True)
            self.process_socket = None

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
                    logger.debug("Failed to kill process for %s", self.hardware_file, exc_info=True)
        self.process = None

    def health_check(self, *, timeout=None) -> dict:
        if not self.is_process_alive():
            exit_code = None if self.process is None else self.process.poll()
            return {
                "state": "failed",
                "pid": self.pid,
                "last_contact_time": self.last_contact_time,
                "error": f"child process exited with code {exit_code}",
            }

        original_timeout = None
        if self.process_socket is not None and timeout is not None and hasattr(self.process_socket, "gettimeout"):
            original_timeout = self.process_socket.gettimeout()
            self.process_socket.settimeout(timeout)

        try:
            running = self.get_property("running")
        finally:
            if self.process_socket is not None and original_timeout is not None:
                self.process_socket.settimeout(original_timeout)

        if running == -1:
            return {
                "state": "degraded",
                "pid": self.pid,
                "last_contact_time": self.last_contact_time,
                "error": self.last_error or "health check RPC failed",
            }

        if not bool(running):
            return {
                "state": "degraded",
                "pid": self.pid,
                "last_contact_time": self.last_contact_time,
                "error": "component reported running=False",
            }

        return {
            "state": "running",
            "pid": self.pid,
            "last_contact_time": self.last_contact_time,
            "error": None,
        }



class Listener:
    """Server-side control socket for a launched hardware object.

    ``Listener`` is the child-process counterpart to :class:`HardwareLauncher`.
    It binds a localhost socket, accepts the RTC-side connection, and services a
    narrow JSON RPC surface for property access, method calls, and clean
    shutdown.
    """

    def __init__(self, hardware, port) -> None:
        self.hardware = hardware
        self.running = True
        self.key_character = '$'
        self.host = '127.0.0.1'  # localhost
        self.port = port

        server_socket = bind_socket(self.host, self.port)
        server_socket.listen()
        logger.info("%s: awaiting RTC connection", hardware.name)
        #Connect to the RTC process that spawned you
        self.RTCsocket, self.RTCaddress = server_socket.accept()

        self.OKMessage = {"status": "OK", "protocol": PROTOCOL_VERSION}
        self.BadMessage = {"status": "BAD", "protocol": PROTOCOL_VERSION}
        self._read_buffer = ""

        return

    def _bad(self, error: str) -> dict:
        message = dict(self.BadMessage)
        message["error"] = str(error)
        return message

    def handle_request(self, request) -> dict:
        """Dispatch one RPC request dict and return the reply dict.

        Separated from socket I/O so the dispatch logic is directly
        testable. Never raises: failures produce a BAD reply carrying an
        ``error`` string.
        """

        if not isinstance(request, dict) or "type" not in request:
            logger.error("Listener request missing type field: %s", request)
            return self._bad("request missing 'type' field")

        protocol = request.get("protocol", PROTOCOL_VERSION)
        if protocol != PROTOCOL_VERSION:
            logger.error("Listener protocol mismatch: got %s, expected %s", protocol, PROTOCOL_VERSION)
            return self._bad(
                f"protocol version mismatch: got {protocol}, expected {PROTOCOL_VERSION}"
            )

        request_type = request["type"]
        if request_type == "shutdown":
            try:
                self.hardware.__del__()
                self.running = False
                return dict(self.OKMessage)
            except Exception as exc:
                logger.exception("Listener shutdown request failed")
                return self._bad(f"shutdown failed: {exc}")

        if request_type == "get":
            property_name = request.get("property")
            try:
                value = getattr(self.hardware, property_name)
                message = dict(self.OKMessage)
                message["property"] = value
                return message
            except Exception as exc:
                logger.exception("Listener get request failed for %s", property_name)
                return self._bad(f"get '{property_name}' failed: {exc}")

        if request_type == "set":
            property_name = request.get("property")
            try:
                current = getattr(self.hardware, property_name)
                coerced = _coerce_property_value(current, request["value"])
                setattr(self.hardware, property_name, coerced)
                return dict(self.OKMessage)
            except Exception as exc:
                logger.exception("Listener set request failed for %s", property_name)
                return self._bad(f"set '{property_name}' failed: {exc}")

        if request_type == "run":
            function_name = request.get("function")
            try:
                args = request.get("args", [])
                if not isinstance(args, list):
                    return self._bad("'args' must be a list")
                function = getattr(self.hardware, function_name)
                result = function(*args)
                message = dict(self.OKMessage)
                if result is not None:
                    serializable = _json_safe(result)
                    if serializable is not None:
                        message["property"] = serializable
                    else:
                        logger.debug(
                            "Dropping non-JSON-serializable result of %s: %r",
                            function_name,
                            type(result),
                        )
                return message
            except Exception as exc:
                logger.exception("Listener run request failed for %s", function_name)
                return self._bad(f"run '{function_name}' failed: {exc}")

        logger.error("Unknown listener request type: %s", request_type)
        return self._bad(f"unknown request type: {request_type}")

    def listen(self):
        try:
            request = self.read()
        except (ConnectionError, OSError):
            logger.info("RTC connection closed; stopping listener")
            self.running = False
            return
        except Exception as exc:
            logger.exception("Failed to read listener request")
            reply = self._bad(f"unreadable request: {exc}")
        else:
            reply = self.handle_request(request)
        try:
            self.write(reply)
        except (ConnectionError, OSError):
            logger.info("RTC connection closed before reply was sent; stopping listener")
            self.running = False

    def write(self, message):
        _socket_send_json(self.RTCsocket, message)
        return

    def read(self):
        reply, self._read_buffer = _socket_read_json(self.RTCsocket, self._read_buffer)
        return reply
