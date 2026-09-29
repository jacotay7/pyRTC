"""Base class for threaded pyrtc runtime components.

Most pyrtc subsystems share the same lifecycle model: validate configuration,
normalize optional GPU settings, spawn one or more worker threads from YAML,
and expose lightweight start/stop controls. This module provides that shared
behavior.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

from pyrtc.config_runtime import stream_alias_map
from pyrtc.logging_utils import ensure_logging_configured, get_logger
from pyrtc.manager import launch_component, work
from pyrtc.streams import normalize_gpu_device
from pyrtc.utils import set_from_config, validate_component_config


logger = get_logger(__name__)

# A blocking ``read_stream`` waits in slices of this length so it notices
# ``close()`` promptly. Writers still wake it at once (notify-enabled streams
# park on a futex), so the slice only bounds teardown, not latency.
STREAM_WAIT_SLICE = 0.1

# Default time ``close()`` waits for each worker thread to leave its function.
DEFAULT_CLOSE_TIMEOUT = 2.0


class ComponentClosedError(RuntimeError):
    """Raised by a blocking read when its component is closed while waiting."""


class Component:
    """
    Common threaded component base used throughout pyrtc.

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
    affinity : int, optional
        Base CPU core for the component. When set, each worker function thread
        is pinned to its own core (``affinity + i``) on Linux; when unset,
        threads are not pinned.
    realtime_priority : int, optional
        When > 0, worker threads run under ``SCHED_FIFO`` at this priority
        (Linux, needs ``CAP_SYS_NICE``). Default 0 lowers the nice value only.
    functions : list
        Bound method names to run in worker threads.
    gpu_device : str, optional
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
            self._warn_unknown_config_keys(conf)

            self._init_runtime_state(conf)
            self.class_name = conf.get("class_name")
            self.class_file = conf.get("class_file")
            self.affinity = conf.get("affinity")
            self.realtime_priority = set_from_config(conf, "realtime_priority", 0)
            requested_gpu_device = set_from_config(conf, "gpu_device", None)
            self.gpu_device = normalize_gpu_device(requested_gpu_device, self.__class__.__name__)

            functions_to_run = set_from_config(conf, "functions", [])

            if isinstance(functions_to_run, list) and len(functions_to_run) > 0:
                for i, function_name in enumerate(functions_to_run):
                    thread_affinity = (
                        None if self.affinity is None else (int(self.affinity) + i) % os.cpu_count()
                    )
                    work_thread = threading.Thread(
                        target=work,
                        args=(self, function_name, thread_affinity),
                        daemon=True,
                    )
                    work_thread.start()
                    self.work_threads.append(work_thread)

            self.logger.info(
                "Initialized component affinity=%s gpu_device=%s functions=%s",
                self.affinity,
                self.gpu_device,
                functions_to_run,
            )
        except Exception:
            self.logger.exception("Failed to initialize component")
            raise

        return

    def _init_runtime_state(self, conf) -> None:
        """Initialize lifecycle flags, stream registries and stream aliases.

        ``__init__`` calls this before starting worker threads. Tests that
        build a component without ``__init__`` call it through
        ``testsupport.bare_component`` so the stream helpers work unchanged.
        """

        self.alive = True
        self.running = False
        self._closed = False
        self.work_threads = []
        self.section_name = conf.get("_sectionName")
        self.system_streams = dict(conf.get("_systemStreams", {}))
        self._stream_inputs = {}
        self._stream_outputs = {}
        self._consumed_counts = {}
        self.frame_id = None
        self._input_stream_names = self._normalize_stream_name_map(
            conf.get("input_streams", {}), direction="input"
        )
        self._output_stream_names = self._normalize_stream_name_map(
            conf.get("output_streams", {}), direction="output"
        )

    def _warn_unknown_config_keys(self, conf) -> None:
        """Log a warning for each config key this component class does not read.

        Subclasses that read keys of their own list them in
        ``EXTRA_CONFIG_KEYS``; see
        :func:`pyrtc.component_descriptors.known_config_keys`.
        """

        from pyrtc.component_descriptors import unknown_config_key_warnings

        for message in unknown_config_key_warnings(conf.get("_sectionName"), conf, self.__class__):
            self.logger.warning(message)

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
        return stream_alias_map(raw_mapping, defaults=self._default_stream_name_map(direction))

    def input_stream_name(self, stream_name: str) -> str:
        return self._input_stream_names.get(str(stream_name), str(stream_name))

    def output_stream_name(self, stream_name: str) -> str:
        return self._output_stream_names.get(str(stream_name), str(stream_name))

    def stream_aliases(self, direction: str) -> dict[str, str]:
        if direction == "input":
            return dict(self._input_stream_names)
        if direction == "output":
            return dict(self._output_stream_names)
        raise ValueError("direction must be 'input' or 'output'")

    def _stream_object(self, stream_name: str):
        """Return the registered handle for an input or output stream."""

        if stream_name in self._stream_inputs:
            return self._stream_inputs[stream_name]
        if stream_name in self._stream_outputs:
            return self._stream_outputs[stream_name]
        raise KeyError(
            f"{self.__class__.__name__} has no registered stream {stream_name!r}; "
            "register it with register_input_stream or register_output_stream"
        )

    def _register_stream(self, registry: dict, stream_name: str, shm) -> None:
        name = str(stream_name)
        previous = registry.get(name)
        registry[name] = shm
        if previous is None or previous is shm:
            return
        # A new handle has its own write counter baseline.
        self._consumed_counts.pop(name, None)
        if not self._is_registered_handle(previous):
            _close_quietly(previous, name, getattr(self, "logger", logger))

    def _is_registered_handle(self, shm) -> bool:
        return any(
            candidate is shm
            for registry in (self._stream_inputs, self._stream_outputs)
            for candidate in registry.values()
        )

    def register_input_stream(self, stream_name: str, shm) -> None:
        """Register a stream that this component reads from.

        Reading a registered input with :meth:`read_stream` records its
        ``frame_id``, so every stream a component reads must be registered.
        The component owns the handle from then on: :meth:`close` closes it,
        as does registering a different handle under the same name.
        """

        self._register_stream(self._stream_inputs, stream_name, shm)

    def register_output_stream(self, stream_name: str, shm) -> None:
        """Register a stream that this component writes to.

        Ownership of the handle passes to the component, as for
        :meth:`register_input_stream`.
        """

        self._register_stream(self._stream_outputs, stream_name, shm)

    def read_stream(
        self, stream_name: str, *, block: bool = True, timeout: float | None = None, out=None
    ):
        """Read one registered input or output stream.

        Parameters
        ----------
        stream_name : str
            Name of the registered stream.
        block : bool, optional
            When ``True``, consume the stream: wait for a write newer than the
            one returned by this component's previous blocking read. The first
            blocking read returns the current payload immediately. When
            ``False``, peek at the current payload without consuming it.
        timeout : float, optional
            Maximum seconds to wait for a new write when ``block`` is
            ``True``. ``None`` waits until a write arrives or the component
            is closed.
        out : numpy.ndarray, optional
            Pre-allocated buffer receiving the payload (zero-alloc reads on
            the hot path). Ignored for GPU-attached streams.

        Reading a registered input stream also records its publication
        ``frame_id`` on :attr:`frame_id`, so the next :meth:`write_stream`
        carries the frame identity downstream.

        Raises
        ------
        ComponentClosedError
            When :meth:`close` is called while a blocking read waits.
        TimeoutError
            When ``timeout`` expires before a new write arrives.
        """

        name = str(stream_name)
        stream = self._stream_object(name)
        if stream.gpu_device is not None:
            out = None
        consumed = self._consumed_counts.get(name) if block else None
        if consumed is None:
            publication = stream.read_publication(out=out)
        else:
            publication = self._read_after(stream, name, consumed, timeout, out)
        if block:
            self._consumed_counts[name] = publication.count
        if name in self._stream_inputs:
            self.frame_id = publication.frame_id
        return publication.payload

    def _read_after(self, stream, name: str, consumed: int, timeout: float | None, out):
        # Level-triggered on the last consumed count, so a write published
        # between two calls is never folded into the wait baseline. The wait
        # is sliced so close() can end it.
        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
        while True:
            if not self.alive:
                raise ComponentClosedError(
                    f"{self.__class__.__name__} closed while waiting on stream {name!r}"
                )
            if deadline is None:
                wait = STREAM_WAIT_SLICE
            else:
                wait = min(STREAM_WAIT_SLICE, max(0.0, deadline - time.monotonic()))
            try:
                return stream.read_after_publication(consumed, timeout=wait, out=out)
            except TimeoutError:
                if deadline is not None and time.monotonic() >= deadline:
                    raise

    def write_stream(self, stream_name: str, arr):
        """Write one registered output stream.

        The write is stamped with :attr:`frame_id` (the frame identity of the
        most recently read input, or a source component's own frame counter)
        when one is set. A component's own writes do not count as reads, so a
        following blocking :meth:`read_stream` returns the just-written
        payload immediately.
        """

        name = str(stream_name)
        stream = self._stream_outputs.get(name)
        if stream is None:
            stream = self._stream_object(name)
        stream.write(arr, frame_id=self.frame_id)

    @classmethod
    def describe(cls):
        """Return the nearest built-in component descriptor for this class."""

        from pyrtc.component_descriptors import describe_component_class

        return describe_component_class(cls)

    def close(self, timeout: float | None = DEFAULT_CLOSE_TIMEOUT) -> None:
        """Tear the component down for good.

        :meth:`stop` only pauses the worker threads; ``close`` ends them and
        releases the component's shared-memory handles. It stops the
        component if it is running, clears :attr:`alive` so every worker
        thread leaves its loop (a blocking :meth:`read_stream` notices within
        ``STREAM_WAIT_SLICE`` seconds), joins the workers, and closes every
        registered stream handle. The streams themselves are not unlinked, so
        observers keep reading the last frames.

        ``close`` is idempotent and safe to call from ``__del__``. Subclasses
        that hold other resources (device SDK handles, extra threads) override
        it, call ``super().close(timeout)`` first, then release their own.

        Parameters
        ----------
        timeout : float, optional
            Seconds to wait, in total, for the worker threads to exit. A
            worker still busy after that (for example blocked inside a device
            SDK call) is logged and abandoned; it is a daemon thread and exits
            with the process. ``None`` waits indefinitely.
        """

        if getattr(self, "_closed", False):
            return
        self._closed = True
        component_logger = getattr(self, "logger", logger)
        if getattr(self, "running", False):
            try:
                self.stop()
            except Exception:
                component_logger.exception("Failed to stop component while closing")
        self.running = False
        self.alive = False

        current = threading.current_thread()
        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
        stuck = []
        for thread in list(getattr(self, "work_threads", [])):
            if thread is current:
                continue
            if not hasattr(thread, "join"):
                continue
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            thread.join(remaining)
            if thread.is_alive():
                stuck.append(thread.name)
        if stuck:
            component_logger.warning(
                "Worker threads %s did not exit within %ss; closing streams anyway",
                stuck,
                timeout,
            )

        self._close_streams()
        if getattr(self, "_closing_from_del", False):
            component_logger.debug("Closed component during garbage collection")
        else:
            component_logger.info("Closed component")

    def _close_streams(self) -> None:
        component_logger = getattr(self, "logger", logger)
        seen = set()
        for registry in (
            getattr(self, "_stream_inputs", {}),
            getattr(self, "_stream_outputs", {}),
        ):
            for name, shm in list(registry.items()):
                if id(shm) in seen:
                    continue
                seen.add(id(shm))
                _close_quietly(shm, name, component_logger)
            registry.clear()
        getattr(self, "_consumed_counts", {}).clear()

    def __del__(self):
        """Close the component when it is garbage collected.

        Worker threads keep a reference to their component, so this only runs
        once the workers are gone (or were never started). Call :meth:`close`
        explicitly instead of relying on it.
        """
        try:
            self._closing_from_del = True
            self.close(timeout=0.0)
        except Exception:
            try:
                getattr(self, "logger", logger).exception(
                    "Failed while closing component during destruction"
                )
            except Exception:
                pass
        return

    def start(self):
        """
        Start the registered real-time functions.
        """
        component_logger = getattr(self, "logger", logger)
        if getattr(self, "_closed", False):
            raise RuntimeError(f"{self.__class__.__name__} is closed and cannot be restarted")
        try:
            self.running = True
            component_logger.info("Started component")
        except Exception:
            component_logger.exception("Failed to start component")
            raise
        return

    def stop(self):
        """
        Pause the registered real-time functions.

        The worker threads stay alive and :meth:`start` resumes them. Use
        :meth:`close` to end them and release the component's streams.
        """
        component_logger = getattr(self, "logger", logger)
        try:
            self.running = False
            component_logger.info("Stopped component")
        except Exception:
            component_logger.exception("Failed to stop component")
            raise
        return


def _close_quietly(shm, name: str, component_logger) -> None:
    close = getattr(shm, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception:
        component_logger.warning("Failed to close stream %r", name, exc_info=True)


if __name__ == "__main__":
    launch_component(Component, "component", start=True)
