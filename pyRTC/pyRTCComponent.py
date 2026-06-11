"""Base class for threaded pyRTC runtime components.

Most pyRTC subsystems share the same lifecycle model: validate configuration,
normalize optional GPU settings, spawn one or more worker threads from YAML,
and expose lightweight start/stop controls. This module provides that shared
behavior.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

from pyRTC.logging_utils import ensure_logging_configured, get_logger
from pyRTC.Pipeline import launchComponent, normalize_gpu_device, work
from pyRTC.utils import setFromConfig, validate_component_config


logger = get_logger(__name__)


class pyRTCComponent:
    """
    Common threaded component base used throughout pyRTC.

    The base class standardizes the repeated mechanics shared by the wavefront
    sensor, slopes processor, loop controller, telemetry recorder, and many
    hardware-facing helpers. Components list runtime methods under the
    configuration key ``functions`` and the base class starts one worker thread
    per listed method.

    Those worker functions are assumed to matter for their side effects rather
    than their return values. They usually read, write, or transform shared-
    memory streams inside the running RTC.

    For examples:

    psf:
        functions:
        - expose
        - integrate

    Config Parameters
    -----------------
    affinity : int
        Base CPU affinity for the component. Additional worker functions are
        assigned subsequent cores when possible.
    functions : list
        Bound method names to run in worker threads.
    gpuDevice : str, optional
        Requested GPU device identifier. When PyTorch is unavailable this is
        normalized back to CPU mode.

    Attributes
    ----------
    alive : bool
        Indicates whether the component is alive.
    running : bool
        Indicates whether the component is currently running.

    The class intentionally does not define component-specific data flow. It is
    only responsible for the shared runtime lifecycle.
    """
    def __init__(self, conf) -> None:
        """
        Constructs all the necessary attributes for the real-time control component object.

        Parameters
        ----------
        conf : dict
            Configuration dictionary for the component. The following keys are used:
            - affinity (int, optional): The CPU affinity for the component. Default 0.
            - functions (list, optional): A list of functions to run in separate threads. Default is an empty list.
        """
        ensure_logging_configured(app_name="pyrtc", component_name=self.__class__.__name__)
        self.logger = get_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")

        try:
            validate_component_config(conf, [cls.__name__ for cls in self.__class__.mro()])

            self.alive = True
            self.running = False
            self.section_name = conf.get("_sectionName")
            self.className = conf.get("className")
            self.classFile = conf.get("classFile")
            self.system_streams = dict(conf.get("_systemStreams", {}))
            self.affinity = setFromConfig(conf, "affinity", 0)
            requested_gpu_device = setFromConfig(conf, "gpuDevice", None)
            self.gpuDevice = normalize_gpu_device(requested_gpu_device, self.__class__.__name__)
            self._stream_inputs = {}
            self._stream_outputs = {}
            self._last_stream_metadata = {}
            self._input_stream_names = self._normalize_stream_name_map(conf.get("inputStreams", {}), direction="input")
            self._output_stream_names = self._normalize_stream_name_map(conf.get("outputStreams", {}), direction="output")

            functions_to_run = setFromConfig(conf, "functions", [])
            self.workThreads = []
            self.RELEASE_GIL = True

            if isinstance(functions_to_run, list) and len(functions_to_run) > 0:
                for i, function_name in enumerate(functions_to_run):
                    threadAffinity = (self.affinity + i) % os.cpu_count()
                    workThread = threading.Thread(
                        target=work,
                        args=(self, function_name, threadAffinity),
                        daemon=True,
                    )
                    workThread.start()
                    self.workThreads.append(workThread)

            self.logger.info(
                "Initialized component affinity=%s gpuDevice=%s functions=%s",
                self.affinity,
                self.gpuDevice,
                functions_to_run,
            )
        except Exception:
            self.logger.exception("Failed to initialize component")
            raise

        return

    def _default_stream_name_map(self, direction: str) -> dict[str, str]:
        defaults: dict[str, str] = {}
        try:
            descriptor = self.describe()
        except Exception:
            descriptor = None
        if descriptor is None:
            return defaults
        streams = descriptor.input_streams if direction == "input" else descriptor.output_streams
        for stream in streams:
            if stream.name != "*":
                defaults[str(stream.name)] = str(stream.name)
        return defaults

    def _normalize_stream_name_map(self, raw_mapping: Any, *, direction: str) -> dict[str, str]:
        normalized = self._default_stream_name_map(direction)
        if not isinstance(raw_mapping, dict):
            return normalized
        for semantic_name, value in raw_mapping.items():
            if not isinstance(semantic_name, str) or not semantic_name.strip():
                continue
            if isinstance(value, str):
                shm_name = value.strip()
            elif isinstance(value, dict):
                shm_name = str(value.get("shm", value.get("name", semantic_name))).strip()
            else:
                continue
            if shm_name:
                normalized[str(semantic_name)] = shm_name
        return normalized

    def input_stream_name(self, stream_name: str) -> str:
        self._ensure_stream_state()
        return self._input_stream_names.get(str(stream_name), str(stream_name))

    def output_stream_name(self, stream_name: str) -> str:
        self._ensure_stream_state()
        return self._output_stream_names.get(str(stream_name), str(stream_name))

    def stream_aliases(self, direction: str) -> dict[str, str]:
        self._ensure_stream_state()
        if direction == "input":
            return dict(self._input_stream_names)
        if direction == "output":
            return dict(self._output_stream_names)
        raise ValueError("direction must be 'input' or 'output'")

    def _ensure_stream_state(self) -> None:
        """Initialize stream-tracking state for partially constructed objects."""

        if not hasattr(self, "_stream_inputs"):
            self._stream_inputs = {}
        if not hasattr(self, "_stream_outputs"):
            self._stream_outputs = {}
        if not hasattr(self, "_last_stream_metadata"):
            self._last_stream_metadata = {}
        if not hasattr(self, "_input_stream_names"):
            self._input_stream_names = {}
        if not hasattr(self, "_output_stream_names"):
            self._output_stream_names = {}
        if not hasattr(self, "system_streams"):
            self.system_streams = {}
        if not hasattr(self, "section_name"):
            self.section_name = None

    def _stream_object(self, stream_name: str):
        """Return the registered SHM object for an input or output stream."""

        self._ensure_stream_state()
        if stream_name in self._stream_inputs:
            return self._stream_inputs[stream_name]
        if stream_name in self._stream_outputs:
            return self._stream_outputs[stream_name]
        conventional_name = f"{stream_name}Shm"
        if hasattr(self, conventional_name):
            return getattr(self, conventional_name)
        raise KeyError(stream_name)

    def _read_stream_metadata(self, stream_name: str) -> dict[str, float | int]:
        """Capture the latest metadata snapshot for a registered stream."""

        self._ensure_stream_state()
        stream = self._stream_object(stream_name)
        return {
            "count": int(stream.count),
            "write_time": float(stream.write_time),
            "read_time": time.time(),
        }

    def register_input_stream(self, stream_name: str, shm) -> None:
        """Register a stream that this component reads from."""

        self._ensure_stream_state()
        self._stream_inputs[str(stream_name)] = shm

    def register_output_stream(self, stream_name: str, shm) -> None:
        """Register a stream that this component writes to."""

        self._ensure_stream_state()
        self._stream_outputs[str(stream_name)] = shm

    def read_stream(self, stream_name: str, *, block: bool = True, timeout: float | None = None):
        """Read one registered input or output stream.

        Parameters
        ----------
        stream_name : str
            Name of the registered stream.
        block : bool, optional
            When ``True``, wait for a write this component has not yet seen
            before returning. The first read of a stream returns the current
            payload immediately.
        timeout : float, optional
            Maximum seconds to wait for a new write when ``block`` is
            ``True``. ``None`` waits indefinitely.
        """

        self._ensure_stream_state()
        name = str(stream_name)
        stream = self._stream_object(name)
        last_seen = self._last_stream_metadata.get(name)
        if (
            block
            and last_seen is not None
            and int(stream.count) == int(last_seen["count"])
        ):
            payload = stream.read_new(timeout=timeout)
        else:
            payload = stream.read()
        self._last_stream_metadata[name] = self._read_stream_metadata(name)
        return payload

    def write_stream(self, stream_name: str, arr):
        """Write one registered output stream.

        A component's own writes intentionally do not update its last-seen
        state, so a following blocking :meth:`read_stream` returns the
        just-written payload immediately.
        """

        self._ensure_stream_state()
        name = str(stream_name)
        stream = self._stream_outputs.get(name)
        if stream is None:
            stream = self._stream_object(name)
        stream.write(arr)

    @classmethod
    def describe(cls):
        """Return the nearest built-in component descriptor for this class."""

        from pyRTC.component_descriptors import describe_component_class

        return describe_component_class(cls)

    def __del__(self):
        """
        Destructor to clean up the component.
        """
        component_logger = getattr(self, "logger", logger)
        try:
            self.stop()
        except Exception:
            component_logger.exception("Failed while stopping component during destruction")
        finally:
            self.alive = False
        return

    def start(self):
        """
        Start the registered real-time functions.
        """
        component_logger = getattr(self, "logger", logger)
        try:
            self.running = True
            component_logger.info("Started component")
        except Exception:
            component_logger.exception("Failed to start component")
            raise
        return

    def stop(self):
        """
        Stops the registered real-time functions.
        """
        component_logger = getattr(self, "logger", logger)
        try:
            self.running = False
            component_logger.info("Stopped component")
        except Exception:
            component_logger.exception("Failed to stop component")
            raise
        return

if __name__ == "__main__":

    launchComponent(pyRTCComponent, "component", start = True)