"""Public package exports for pyrtc.

pyrtc provides the building blocks needed to assemble an adaptive-optics real-
time controller in Python. The package root re-exports the main component base
classes, orchestration helpers, pyshmem-backed stream helpers, and logging
utilities so users can build systems from a compact public API surface.
"""

from .loop import Loop
from .component_descriptors import (
    ComponentDescriptor,
    ConfigFieldDescriptor,
    StreamDescriptor,
    build_descriptor_catalog,
    describe_component_class,
    get_component_descriptor,
    list_component_descriptors,
    list_component_sections,
    register_component_descriptor,
    unregister_component_descriptor,
    validate_config_with_descriptor,
)
from .config_schema import normalize_system_config, read_system_config, validate_system_config
from .logging_utils import (
    add_logging_cli_args,
    configure_logging,
    configure_logging_from_args,
    get_logger,
)
from .latency import LatencyReport, LatencySegment, LatencyStatistics, format_latency_report
from .modulator import Modulator
from .optimizer import Optimizer
from .manager import RTCManager, launch_component
from .rpc import Listener, HardwareLauncher
from .streams import (
    clear_shms,
    create_stream,
    gpu_torch_available,
    normalize_gpu_device,
    open_stream,
)
from .science_camera import ScienceCamera
from .slopes_process import SlopesProcess
from .telemetry import (
    Telemetry,
    list_telemetry_sessions,
    load_telemetry_manifest,
    load_telemetry_session,
)
from .wavefront_corrector import WavefrontCorrector
from .wavefront_sensor import WavefrontSensor
from .component import Component
from .utils import set_from_config
from . import pipeline, utils

__all__ = [
    "clear_shms",
    "create_stream",
    "Listener",
    "Loop",
    "LatencyReport",
    "LatencySegment",
    "LatencyStatistics",
    "Modulator",
    "Optimizer",
    "pipeline",
    "RTCManager",
    "ScienceCamera",
    "SlopesProcess",
    "Telemetry",
    "list_telemetry_sessions",
    "load_telemetry_manifest",
    "load_telemetry_session",
    "WavefrontCorrector",
    "WavefrontSensor",
    "normalize_system_config",
    "gpu_torch_available",
    "get_logger",
    "HardwareLauncher",
    "open_stream",
    "read_system_config",
    "configure_logging",
    "configure_logging_from_args",
    "format_latency_report",
    "add_logging_cli_args",
    "ComponentDescriptor",
    "ConfigFieldDescriptor",
    "build_descriptor_catalog",
    "launch_component",
    "describe_component_class",
    "normalize_gpu_device",
    "get_component_descriptor",
    "list_component_descriptors",
    "list_component_sections",
    "Component",
    "register_component_descriptor",
    "set_from_config",
    "StreamDescriptor",
    "unregister_component_descriptor",
    "validate_config_with_descriptor",
    "validate_system_config",
    "utils",
]
