"""Public package exports for pyRTC.

pyRTC provides the building blocks needed to assemble an adaptive-optics real-
time controller in Python. The package root re-exports the main component base
classes, orchestration helpers, shared-memory transport primitives, and logging
utilities so users can build systems from a compact public API surface.
"""

from .Loop import Loop
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
from .logging_utils import add_logging_cli_args, configure_logging, configure_logging_from_args, get_logger
from .Modulator import Modulator
from .Optimizer import Optimizer
from .Pipeline import (
	ImageSHM,
	Listener,
	RTCManager,
	gpu_torch_available,
	hardwareLauncher,
	initExistingShm,
	launchComponent,
	normalize_gpu_device,
)
from .ScienceCamera import ScienceCamera
from .SlopesProcess import SlopesProcess
from .Telemetry import Telemetry
from .WavefrontCorrector import WavefrontCorrector
from .WavefrontSensor import WavefrontSensor
from .pyRTCComponent import pyRTCComponent
from .utils import setFromConfig
from . import Pipeline, utils

__all__ = [
	"ImageSHM",
	"Listener",
	"Loop",
	"Modulator",
	"Optimizer",
	"Pipeline",
	"RTCManager",
	"ScienceCamera",
	"SlopesProcess",
	"Telemetry",
	"WavefrontCorrector",
	"WavefrontSensor",
	"normalize_system_config",
	"gpu_torch_available",
	"get_logger",
	"hardwareLauncher",
	"initExistingShm",
	"read_system_config",
	"configure_logging",
	"configure_logging_from_args",
	"add_logging_cli_args",
	"ComponentDescriptor",
	"ConfigFieldDescriptor",
	"build_descriptor_catalog",
	"launchComponent",
	"describe_component_class",
	"normalize_gpu_device",
	"get_component_descriptor",
	"list_component_descriptors",
	"list_component_sections",
	"pyRTCComponent",
	"register_component_descriptor",
	"setFromConfig",
	"StreamDescriptor",
	"unregister_component_descriptor",
	"validate_config_with_descriptor",
	"validate_system_config",
	"utils",
]
