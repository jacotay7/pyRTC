"""Base class for threaded pyRTC runtime components.

Most pyRTC subsystems share the same lifecycle model: validate configuration,
normalize optional GPU settings, spawn one or more worker threads from YAML,
and expose lightweight start/stop controls. This module provides that shared
behavior.
"""
import os
import threading
import time

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
            self.system_streams = dict(conf.get("_systemStreams", {}))
            self.affinity = setFromConfig(conf, "affinity", 0)
            requested_gpu_device = setFromConfig(conf, "gpuDevice", None)
            self.gpuDevice = normalize_gpu_device(requested_gpu_device, self.__class__.__name__)
            self._stream_inputs = {}
            self._stream_outputs = {}
            self._stream_defaults = self._build_default_stream_flow()
            self._last_stream_metadata = {}

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

    def _build_default_stream_flow(self) -> dict[str, dict[str, object]]:
        """Build default lineage rules from the component descriptor."""

        defaults = {}
        try:
            descriptor = self.describe()
        except Exception:
            descriptor = None
        if descriptor is None:
            return defaults

        input_names = [stream.name for stream in descriptor.input_streams if stream.name != "*"]
        primary_input = input_names[0] if input_names else None
        for stream in descriptor.output_streams:
            if stream.name == "*":
                continue
            defaults[stream.name] = {
                "source_streams": [primary_input] if primary_input is not None else [],
                "lineage_source": primary_input,
            }
        return defaults

    def _stream_runtime_override(self, stream_name: str) -> dict:
        """Return any config override that applies to one output stream.

        The preferred terminology is ``outputComponent`` for the section that
        writes the stream. ``producer`` is still accepted as a backward-
        compatible alias.
        """

        stream_conf = self.system_streams.get(stream_name, {})
        if not isinstance(stream_conf, dict):
            return {}

        output_component = stream_conf.get("outputComponent", stream_conf.get("producer"))
        if output_component is not None and self.section_name is not None and str(output_component) != self.section_name:
            return {}
        return stream_conf

    def _stream_object(self, stream_name: str):
        """Return the registered SHM object for an input or output stream."""

        if stream_name in self._stream_inputs:
            return self._stream_inputs[stream_name]
        if stream_name in self._stream_outputs:
            return self._stream_outputs[stream_name]
        raise KeyError(stream_name)

    def _read_stream_metadata(self, stream_name: str) -> dict[str, float | int]:
        """Capture the latest metadata snapshot for a registered stream."""

        stream = self._stream_object(stream_name)
        frame_metadata = getattr(stream, "frame_metadata", None)
        if callable(frame_metadata):
            snapshot = dict(frame_metadata())
        else:
            metadata = getattr(stream, "metadata", None)
            if metadata is None:
                snapshot = {
                    "count": 0,
                    "write_time": 0.0,
                    "root_time": 0.0,
                    "upstream_write_time": 0.0,
                    "upstream_consume_time": 0.0,
                }
            else:
                snapshot = {
                    "count": int(metadata[0]),
                    "write_time": float(metadata[1]),
                    "root_time": float(metadata[2]) if len(metadata) > 2 else 0.0,
                    "upstream_write_time": float(metadata[3]) if len(metadata) > 3 else 0.0,
                    "upstream_consume_time": float(metadata[4]) if len(metadata) > 4 else 0.0,
                }
        snapshot["read_time"] = time.time()
        return snapshot

    def _resolve_output_lineage(
        self,
        stream_name: str,
        *,
        source_streams: list[str] | tuple[str, ...] | None,
        lineage_source: str | None,
    ) -> tuple[list[str], str | None]:
        """Resolve the lineage inputs for one output stream write."""

        defaults = self._stream_defaults.get(stream_name, {})
        resolved_sources = [
            str(item)
            for item in (
                source_streams
                if source_streams is not None
                else defaults.get("source_streams", [])
            )
        ]
        resolved_lineage = lineage_source if lineage_source is not None else defaults.get("lineage_source")
        return resolved_sources, resolved_lineage

    def _lineage_metadata_for_write(
        self,
        source_streams: list[str],
        lineage_source: str | None,
    ) -> tuple[float | None, float | None, float | None]:
        """Resolve root, upstream-write, and input-read times for a write."""

        source_metadata = [
            self._last_stream_metadata[source_name]
            for source_name in source_streams
            if source_name in self._last_stream_metadata
        ]
        lineage_metadata = self._last_stream_metadata.get(str(lineage_source)) if lineage_source is not None else None
        if lineage_metadata is None and source_metadata:
            lineage_metadata = source_metadata[0]

        root_time = None
        upstream_time = None
        input_read_time = None
        if lineage_metadata is not None:
            lineage_root = float(lineage_metadata.get("root_time", 0.0))
            lineage_write = float(lineage_metadata.get("write_time", 0.0))
            root_time = lineage_root if lineage_root > 0 else lineage_write
            upstream_time = lineage_write
        if source_metadata:
            input_read_time = max(float(metadata.get("read_time", 0.0)) for metadata in source_metadata)
        return root_time, upstream_time, input_read_time

    def register_input_stream(self, stream_name: str, shm) -> None:
        """Register a stream that this component reads from."""

        self._stream_inputs[str(stream_name)] = shm

    def register_output_stream(
        self,
        stream_name: str,
        shm,
        *,
        source_streams: list[str] | tuple[str, ...] | None = None,
        lineage_source: str | None = None,
    ) -> None:
        """Register a stream that this component writes to."""

        name = str(stream_name)
        self._stream_outputs[name] = shm
        defaults = self._stream_defaults.get(name, {})
        config_override = self._stream_runtime_override(name)
        resolved_sources = source_streams
        if resolved_sources is None:
            resolved_sources = config_override.get("sourceStreams", defaults.get("source_streams", []))
        resolved_lineage = lineage_source
        if resolved_lineage is None:
            resolved_lineage = config_override.get("lineageSource", defaults.get("lineage_source"))
        self._stream_defaults[name] = {
            "source_streams": [str(item) for item in resolved_sources or []],
            "lineage_source": None if resolved_lineage is None else str(resolved_lineage),
        }

    def read_stream(self, stream_name: str, *, block: bool = True, SAFE: bool = True, GPU: bool = False, RELEASE_GIL: bool = True, record_consumption: bool = True):
        """Read one registered input or output stream.

        Parameters
        ----------
        stream_name : str
            Name of the registered stream.
        block : bool, optional
            When ``True`` use the blocking SHM read path.
        SAFE : bool, optional
            Forwarded to the SHM read method.
        GPU : bool, optional
            Forwarded to the SHM read method.
        RELEASE_GIL : bool, optional
            Forwarded to the blocking SHM read method.
        record_consumption : bool, optional
            When ``True`` remember this read for downstream lineage metadata.
        """

        stream = self._stream_object(str(stream_name))
        if block:
            payload = stream.read(SAFE=SAFE, GPU=GPU, RELEASE_GIL=RELEASE_GIL)
        else:
            payload = stream.read_noblock(SAFE=SAFE, GPU=GPU)
        if record_consumption:
            self._last_stream_metadata[str(stream_name)] = self._read_stream_metadata(str(stream_name))
        return payload

    def write_stream(
        self,
        stream_name: str,
        arr,
        *,
        source_streams: list[str] | tuple[str, ...] | None = None,
        lineage_source: str | None = None,
    ):
        """Write one registered output stream with lineage metadata."""

        name = str(stream_name)
        stream = self._stream_outputs[name]
        resolved_sources, resolved_lineage = self._resolve_output_lineage(
            name,
            source_streams=source_streams,
            lineage_source=lineage_source,
        )
        root_time, upstream_time, input_read_time = self._lineage_metadata_for_write(
            resolved_sources,
            resolved_lineage,
        )

        result = stream.write(
            arr,
            root_time=root_time,
            upstream_time=upstream_time,
            consumer_time=input_read_time,
        )
        self._last_stream_metadata[name] = self._read_stream_metadata(name)
        return result

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