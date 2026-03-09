"""Descriptor metadata for pyRTC components.

The descriptor model captures the stable, machine-readable information that
future manager, GUI, and plugin layers need: config fields, worker functions,
stream contracts, and extension metadata.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Type

from pyRTC.Loop import Loop
from pyRTC.ScienceCamera import ScienceCamera
from pyRTC.SlopesProcess import SlopesProcess
from pyRTC.Telemetry import Telemetry
from pyRTC.WavefrontCorrector import WavefrontCorrector
from pyRTC.WavefrontSensor import WavefrontSensor


@dataclass(frozen=True)
class ConfigFieldDescriptor:
    """Describe one configuration field exposed by a component."""

    name: str
    field_type: str
    description: str
    required: bool = False
    default: Any = None
    minimum: int | float | None = None
    choices: tuple[str, ...] = ()
    allow_none: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["choices"] = list(payload["choices"])
        return payload

    def __getitem__(self, key: str) -> Any:
        if not hasattr(self, key):
            raise KeyError(key)
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def summary(self) -> str:
        parts = [f"ConfigFieldDescriptor<{self.name}>", f"  type: {self.field_type}"]
        parts.append(f"  required: {self.required}")
        if self.default is not None:
            parts.append(f"  default: {self.default!r}")
        if self.minimum is not None:
            parts.append(f"  minimum: {self.minimum!r}")
        if self.choices:
            parts.append(f"  choices: {', '.join(self.choices)}")
        if self.allow_none:
            parts.append("  allow_none: True")
        parts.append(f"  description: {self.description}")
        return "\n".join(parts)

    def __repr__(self) -> str:
        return self.summary()

    __str__ = __repr__


@dataclass(frozen=True)
class StreamDescriptor:
    """Describe one named pyRTC stream used by a component."""

    name: str
    direction: str
    dtype: str | None = None
    shape: str | None = None
    optional: bool = False
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __repr__(self) -> str:
        parts = [f"name='{self.name}'", f"direction='{self.direction}'"]
        if self.dtype is not None:
            parts.append(f"dtype='{self.dtype}'")
        if self.shape is not None:
            parts.append(f"shape='{self.shape}'")
        if self.optional:
            parts.append("optional=True")
        return f"StreamDescriptor({', '.join(parts)})"


@dataclass(frozen=True)
class ComponentDescriptor:
    """Machine-readable metadata for one component type."""

    section_name: str
    category: str
    component_class: Type[Any]
    description: str
    required_fields: tuple[ConfigFieldDescriptor, ...] = field(default_factory=tuple)
    optional_fields: tuple[ConfigFieldDescriptor, ...] = field(default_factory=tuple)
    worker_functions: tuple[str, ...] = field(default_factory=tuple)
    input_streams: tuple[StreamDescriptor, ...] = field(default_factory=tuple)
    output_streams: tuple[StreamDescriptor, ...] = field(default_factory=tuple)
    supports_hard_rtc: bool = True
    external_dependencies: tuple[str, ...] = field(default_factory=tuple)
    calibration_artifacts: tuple[str, ...] = field(default_factory=tuple)

    @property
    def class_name(self) -> str:
        return self.component_class.__name__

    @property
    def class_path(self) -> str:
        return f"{self.component_class.__module__}.{self.component_class.__name__}"

    @property
    def all_fields(self) -> tuple[ConfigFieldDescriptor, ...]:
        return self.required_fields + self.optional_fields

    @property
    def field_map(self) -> dict[str, ConfigFieldDescriptor]:
        return {field_descriptor.name: field_descriptor for field_descriptor in self.all_fields}

    @property
    def required_field_names(self) -> tuple[str, ...]:
        return tuple(field_descriptor.name for field_descriptor in self.required_fields)

    @property
    def optional_field_names(self) -> tuple[str, ...]:
        return tuple(field_descriptor.name for field_descriptor in self.optional_fields)

    def __getitem__(self, key: str) -> Any:
        if key in self.field_map:
            return self.field_map[key]
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> tuple[str, ...]:
        return tuple(self.to_dict().keys())

    def items(self) -> tuple[tuple[str, Any], ...]:
        payload = self.to_dict()
        return tuple(payload.items())

    def values(self) -> tuple[Any, ...]:
        payload = self.to_dict()
        return tuple(payload.values())

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and (key in self.field_map or hasattr(self, key))

    def summary(self) -> str:
        required = ", ".join(self.required_field_names) or "none"
        workers = ", ".join(self.worker_functions) or "none"
        inputs = ", ".join(stream.name for stream in self.input_streams) or "none"
        outputs = ", ".join(stream.name for stream in self.output_streams) or "none"
        return (
            f"ComponentDescriptor<{self.section_name}>"
            f"\n  class: {self.class_path}"
            f"\n  category: {self.category}"
            f"\n  required_fields: {required}"
            f"\n  optional_fields: {len(self.optional_fields)} fields"
            f"\n  worker_functions: {workers}"
            f"\n  input_streams: {inputs}"
            f"\n  output_streams: {outputs}"
            f"\n  supports_hard_rtc: {self.supports_hard_rtc}"
        )

    def __repr__(self) -> str:
        return self.summary()

    __str__ = __repr__

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "required_fields",
            "optional_fields",
            "worker_functions",
            "input_streams",
            "output_streams",
            "external_dependencies",
            "calibration_artifacts",
        ):
            payload[key] = list(payload[key])
        payload["required_fields"] = [field_descriptor.to_dict() for field_descriptor in self.required_fields]
        payload["optional_fields"] = [field_descriptor.to_dict() for field_descriptor in self.optional_fields]
        payload["input_streams"] = [stream.to_dict() for stream in self.input_streams]
        payload["output_streams"] = [stream.to_dict() for stream in self.output_streams]
        payload["class_name"] = self.class_name
        payload["class_path"] = self.class_path
        payload["component_class"] = self.class_path
        payload["fields"] = {
            field_name: field_descriptor.to_dict()
            for field_name, field_descriptor in self.field_map.items()
        }
        return payload


BUILTIN_COMPONENT_DESCRIPTORS: tuple[ComponentDescriptor, ...] = (
    ComponentDescriptor(
        section_name="wfs",
        category="wavefront_sensor",
        component_class=WavefrontSensor,
        description="Base wavefront-sensor interface that publishes raw and processed images.",
        required_fields=(
            ConfigFieldDescriptor("width", "int", "Raw image width in pixels.", required=True, minimum=1),
            ConfigFieldDescriptor("height", "int", "Raw image height in pixels.", required=True, minimum=1),
        ),
        optional_fields=(
            ConfigFieldDescriptor("name", "str", "Component display name.", default="wavefrontSensor"),
            ConfigFieldDescriptor("darkCount", "int", "Number of exposures to average for dark acquisition.", default=1000, minimum=0),
            ConfigFieldDescriptor("darkFile", "str", "Path to a persisted dark frame.", default=""),
            ConfigFieldDescriptor("downsampleFactor", "int", "Integer factor applied to the processed image.", default=0, minimum=0),
            ConfigFieldDescriptor("rotationAngle", "float", "Rotation angle in degrees applied to the processed image.", default=0.0),
            ConfigFieldDescriptor("functions", "list[str]", "Worker methods started in component threads.", default=[]),
            ConfigFieldDescriptor("affinity", "int", "Base CPU affinity for the component.", default=0),
            ConfigFieldDescriptor("gpuDevice", "str | None", "Optional GPU device identifier.", default=None),
        ),
        worker_functions=("expose",),
        input_streams=(),
        output_streams=(
            StreamDescriptor("wfsRaw", "output", dtype="uint16", shape="(width, height)", description="Raw WFS image stream."),
            StreamDescriptor("wfs", "output", dtype="int32", shape="(processed_width, processed_height)", description="Dark-subtracted processed WFS image stream."),
        ),
        supports_hard_rtc=True,
        calibration_artifacts=("darkFile",),
    ),
    ComponentDescriptor(
        section_name="slopes",
        category="slopes_process",
        component_class=SlopesProcess,
        description="Signal reduction stage that converts WFS images into slopes or related wavefront signals.",
        required_fields=(
            ConfigFieldDescriptor("type", "str", "Wavefront-sensor reduction mode such as SHWFS or PYWFS.", required=True, choices=("SHWFS", "PYWFS")),
            ConfigFieldDescriptor("signalType", "str", "Signal representation produced by the reducer.", required=True),
        ),
        optional_fields=(
            ConfigFieldDescriptor("imageNoise", "float", "Configured image noise estimate.", default=0.0, minimum=0.0),
            ConfigFieldDescriptor("centralObscurationRatio", "float", "Central obscuration ratio used by PYWFS paths.", default=0.0, minimum=0.0),
            ConfigFieldDescriptor("flatNorm", "bool", "Whether to normalize the PYWFS flat.", default=True),
            ConfigFieldDescriptor("pupils", "list[str]", "Pupil centers for PYWFS in 'x,y' form.", default=[]),
            ConfigFieldDescriptor("pupilsRadius", "int", "Pupil radius for explicit PYWFS geometry.", default=None, minimum=1),
            ConfigFieldDescriptor("contrast", "float", "SHWFS contrast parameter.", default=0.0),
            ConfigFieldDescriptor("subApSpacing", "int", "Sub-aperture spacing for SHWFS layouts.", default=None, minimum=1),
            ConfigFieldDescriptor("subApOffsetX", "int", "SHWFS X offset in pixels.", default=0, minimum=0),
            ConfigFieldDescriptor("subApOffsetY", "int", "SHWFS Y offset in pixels.", default=0, minimum=0),
            ConfigFieldDescriptor("refSlopeCount", "int", "Number of frames used to average reference slopes.", default=1000, minimum=1),
            ConfigFieldDescriptor("validSubApsFile", "str", "Path to the valid sub-aperture mask file.", default=""),
            ConfigFieldDescriptor("refSlopesFile", "str", "Path to the reference slopes file.", default=""),
            ConfigFieldDescriptor("functions", "list[str]", "Worker methods started in component threads.", default=[]),
            ConfigFieldDescriptor("affinity", "int", "Base CPU affinity for the component.", default=0),
            ConfigFieldDescriptor("gpuDevice", "str | None", "Optional GPU device identifier.", default=None),
        ),
        worker_functions=("computeSignal",),
        input_streams=(
            StreamDescriptor("wfs", "input", dtype="int32", shape="(processed_width, processed_height)", description="Processed wavefront-sensor image stream."),
        ),
        output_streams=(
            StreamDescriptor("signal", "output", dtype="float32", shape="(signal_size,)", description="Flattened residual signal stream."),
            StreamDescriptor("signal2D", "output", dtype="float32", shape="(signal_rows, signal_cols)", description="2D visualization of the residual signal."),
        ),
        supports_hard_rtc=True,
        calibration_artifacts=("validSubApsFile", "refSlopesFile"),
    ),
    ComponentDescriptor(
        section_name="loop",
        category="control_loop",
        component_class=Loop,
        description="Adaptive-optics controller that converts residual signals into correction commands.",
        required_fields=(),
        optional_fields=(
            ConfigFieldDescriptor("numDroppedModes", "int", "Number of controlled modes to suppress.", default=0, minimum=0),
            ConfigFieldDescriptor("CMMethod", "str", "Control-matrix inversion method ('svd' or 'tikhonov').", default="svd"),
            ConfigFieldDescriptor("conditioning", "float | None", "Optional target conditioning number used to truncate small singular values.", default=None, minimum=1.0),
            ConfigFieldDescriptor("tikhonovReg", "float", "Tikhonov regularization strength used when CMMethod is 'tikhonov'.", default=0.0, minimum=0.0),
            ConfigFieldDescriptor("gain", "float", "Integrator gain.", default=0.1),
            ConfigFieldDescriptor("leakyGain", "float", "Leaky-integrator gain.", default=0.0),
            ConfigFieldDescriptor("hardwareDelay", "float", "Estimated hardware delay.", default=0.0, minimum=0.0),
            ConfigFieldDescriptor("pokeAmp", "float", "Calibration poke amplitude.", default=1e-2, minimum=0.0),
            ConfigFieldDescriptor("numItersIM", "int", "Interaction-matrix calibration iteration count.", default=100, minimum=1),
            ConfigFieldDescriptor("delay", "int", "Artificial delay in frames.", default=0, minimum=0),
            ConfigFieldDescriptor("IMMethod", "str", "Interaction-matrix calibration method.", default="push-pull"),
            ConfigFieldDescriptor("IMFile", "str", "Path to the interaction-matrix file.", default=""),
            ConfigFieldDescriptor("pGain", "float", "PID proportional gain.", default=0.1),
            ConfigFieldDescriptor("iGain", "float", "PID integral gain.", default=0.0),
            ConfigFieldDescriptor("dGain", "float", "PID derivative gain.", default=0.0),
            ConfigFieldDescriptor("controlLimits", "list[float]", "PID control output limits.", default=[float("-inf"), float("inf")]),
            ConfigFieldDescriptor("integralLimits", "list[float]", "PID integral limits.", default=[float("-inf"), float("inf")]),
            ConfigFieldDescriptor("absoluteLimits", "list[float]", "Absolute correction limits.", default=[float("-inf"), float("inf")]),
            ConfigFieldDescriptor("derivativeFilter", "float", "PID derivative filter coefficient.", default=0.1),
            ConfigFieldDescriptor("functions", "list[str]", "Worker methods started in component threads.", default=[]),
            ConfigFieldDescriptor("affinity", "int", "Base CPU affinity for the component.", default=0),
            ConfigFieldDescriptor("gpuDevice", "str | None", "Optional GPU device identifier.", default=None),
        ),
        worker_functions=("standardIntegrator", "standardIntegratorPOL", "leakyIntegrator", "pidIntegrator", "pidIntegratorPOL"),
        input_streams=(
            StreamDescriptor("signal", "input", dtype="float32", shape="(signal_size,)", description="Residual signal from slopes processing."),
        ),
        output_streams=(
            StreamDescriptor("wfc", "output", dtype="float32", shape="(numModes,)", description="Modal correction vector sent to the wavefront corrector."),
        ),
        supports_hard_rtc=True,
        calibration_artifacts=("IMFile",),
    ),
    ComponentDescriptor(
        section_name="wfc",
        category="wavefront_corrector",
        component_class=WavefrontCorrector,
        description="Wavefront-corrector interface that maps modal commands into actuator space and hardware updates.",
        required_fields=(
            ConfigFieldDescriptor("name", "str", "Component display name.", required=True),
            ConfigFieldDescriptor("numActuators", "int", "Number of actuators in zonal space.", required=True, minimum=1),
            ConfigFieldDescriptor("numModes", "int", "Number of controlled modes in modal space.", required=True, minimum=1),
        ),
        optional_fields=(
            ConfigFieldDescriptor("m2cFile", "str", "Path to the mode-to-command matrix.", default=""),
            ConfigFieldDescriptor("flatFile", "str", "Path to the flat shape file.", default=""),
            ConfigFieldDescriptor("floatingInfluenceRadius", "int", "Radius used when floating inactive actuators.", default=1, minimum=0),
            ConfigFieldDescriptor("frameDelay", "int", "Artificial frame delay in actuator space.", default=0, minimum=0),
            ConfigFieldDescriptor("saveFile", "str", "Path used when saving a zonal shape.", default="wfcShape.npy"),
            ConfigFieldDescriptor("functions", "list[str]", "Worker methods started in component threads.", default=[]),
            ConfigFieldDescriptor("affinity", "int", "Base CPU affinity for the component.", default=0),
            ConfigFieldDescriptor("gpuDevice", "str | None", "Optional GPU device identifier.", default=None),
        ),
        worker_functions=("sendToHardware",),
        input_streams=(
            StreamDescriptor("wfc", "input", dtype="float32", shape="(numModes,)", description="Modal correction vector from the loop controller."),
        ),
        output_streams=(
            StreamDescriptor("wfc", "output", dtype="float32", shape="(numModes,)", description="Published correction vector for readers and launchers."),
            StreamDescriptor("wfc2D", "output", dtype="float32", shape="layout.shape", optional=True, description="Optional 2D actuator-layout visualization stream."),
        ),
        supports_hard_rtc=True,
        calibration_artifacts=("m2cFile", "flatFile"),
    ),
    ComponentDescriptor(
        section_name="psf",
        category="science_camera",
        component_class=ScienceCamera,
        description="Science-camera interface that publishes short- and long-exposure PSFs plus image-quality telemetry.",
        required_fields=(
            ConfigFieldDescriptor("name", "str", "Component display name.", required=True),
            ConfigFieldDescriptor("width", "int", "Image width in pixels.", required=True, minimum=1),
            ConfigFieldDescriptor("height", "int", "Image height in pixels.", required=True, minimum=1),
            ConfigFieldDescriptor("darkCount", "int", "Number of exposures to average for dark acquisition.", required=True, minimum=1),
            ConfigFieldDescriptor("integration", "int", "Number of frames averaged for the long-exposure PSF.", required=True, minimum=1),
        ),
        optional_fields=(
            ConfigFieldDescriptor("darkFile", "str", "Path to a persisted dark frame.", default=""),
            ConfigFieldDescriptor("modelFile", "str", "Path to a model PSF file.", default=""),
            ConfigFieldDescriptor("functions", "list[str]", "Worker methods started in component threads.", default=[]),
            ConfigFieldDescriptor("affinity", "int", "Base CPU affinity for the component.", default=0),
            ConfigFieldDescriptor("gpuDevice", "str | None", "Optional GPU device identifier.", default=None),
        ),
        worker_functions=("expose", "integrate"),
        input_streams=(),
        output_streams=(
            StreamDescriptor("psfShort", "output", dtype="int32", shape="(width, height)", description="Short-exposure PSF image stream."),
            StreamDescriptor("psfLong", "output", dtype="float64", shape="(width, height)", description="Long-exposure PSF image stream."),
            StreamDescriptor("strehl", "output", dtype="float", shape="(1,)", description="Scalar Strehl estimate."),
            StreamDescriptor("tiptilt", "output", dtype="float", shape="(1,)", description="Scalar tip-tilt estimate."),
        ),
        supports_hard_rtc=True,
        calibration_artifacts=("darkFile", "modelFile"),
    ),
    ComponentDescriptor(
        section_name="telemetry",
        category="telemetry",
        component_class=Telemetry,
        description="Telemetry capture helper for persisting existing pyRTC streams to disk.",
        required_fields=(),
        optional_fields=(
            ConfigFieldDescriptor("dataDir", "str", "Base directory used for telemetry capture files.", default="./data/"),
            ConfigFieldDescriptor("functions", "list[str]", "Worker methods started in component threads.", default=[]),
            ConfigFieldDescriptor("affinity", "int", "Base CPU affinity for the component.", default=0),
            ConfigFieldDescriptor("gpuDevice", "str | None", "Optional GPU device identifier.", default=None),
        ),
        worker_functions=(),
        input_streams=(
            StreamDescriptor("*", "input", description="Attaches to existing streams on demand via save()."),
        ),
        output_streams=(),
        supports_hard_rtc=False,
        calibration_artifacts=(),
    ),
)


_DESCRIPTORS_BY_SECTION = {
    descriptor.section_name: descriptor for descriptor in BUILTIN_COMPONENT_DESCRIPTORS
}
_DESCRIPTORS_BY_CLASS = {
    descriptor.component_class: descriptor for descriptor in BUILTIN_COMPONENT_DESCRIPTORS
}


def _field_type_matches(field_type: str, value: Any) -> bool:
    if field_type == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if field_type == "float":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if field_type == "bool":
        return isinstance(value, bool)
    if field_type == "str":
        return isinstance(value, str)
    if field_type == "list[str]":
        return isinstance(value, list) and all(isinstance(item, str) for item in value)
    if field_type == "list[float]":
        return isinstance(value, list) and all(
            isinstance(item, (int, float)) and not isinstance(item, bool) for item in value
        )
    if field_type == "str | None":
        return value is None or isinstance(value, str)
    return True


def register_component_descriptor(descriptor: ComponentDescriptor) -> ComponentDescriptor:
    """Register a descriptor so future layers can discover non-built-in components."""

    _DESCRIPTORS_BY_SECTION[descriptor.section_name] = descriptor
    _DESCRIPTORS_BY_CLASS[descriptor.component_class] = descriptor
    return descriptor


def unregister_component_descriptor(section_name: str) -> None:
    """Remove a registered descriptor by section name.

    This is primarily intended for tests and future plugin lifecycle hooks.
    Built-in descriptors can be restored by re-registering them.
    """

    descriptor = _DESCRIPTORS_BY_SECTION.pop(section_name, None)
    if descriptor is not None:
        _DESCRIPTORS_BY_CLASS.pop(descriptor.component_class, None)


def get_component_descriptor(section_name: str) -> ComponentDescriptor | None:
    """Return the built-in descriptor for a top-level config section."""

    return _DESCRIPTORS_BY_SECTION.get(section_name)


def list_component_descriptors() -> tuple[ComponentDescriptor, ...]:
    """Return all built-in component descriptors."""

    return tuple(_DESCRIPTORS_BY_SECTION.values())


def list_component_sections() -> tuple[str, ...]:
    """Return the known top-level config sections for built-in components."""

    return tuple(_DESCRIPTORS_BY_SECTION)


def build_descriptor_catalog() -> dict[str, dict[str, Any]]:
    """Return a machine-readable descriptor catalog keyed by top-level section."""

    return {
        descriptor.section_name: descriptor.to_dict()
        for descriptor in list_component_descriptors()
    }


def validate_config_with_descriptor(section_name: str, conf: Mapping[str, Any]) -> None:
    """Validate generic field presence and types using descriptor metadata."""

    descriptor = get_component_descriptor(section_name)
    if descriptor is None:
        return

    known_fields = {field_descriptor.name: field_descriptor for field_descriptor in descriptor.all_fields}

    for field_descriptor in descriptor.required_fields:
        if field_descriptor.name not in conf:
            raise ValueError(
                f"{section_name}: missing required config key(s): {field_descriptor.name}"
            )

    for key, value in conf.items():
        field_descriptor = known_fields.get(key)
        if field_descriptor is None:
            continue
        if value is None and field_descriptor.allow_none:
            continue
        if value is None and field_descriptor.default is None and field_descriptor.allow_none:
            continue
        if value is None and field_descriptor.default is None and field_descriptor.field_type.endswith("| None"):
            continue
        if value is None and not field_descriptor.allow_none:
            raise ValueError(f"{section_name}: '{key}' may not be null")
        if not _field_type_matches(field_descriptor.field_type, value):
            raise TypeError(
                f"{section_name}: '{key}' must match descriptor type {field_descriptor.field_type}"
            )
        if field_descriptor.minimum is not None and isinstance(value, (int, float)) and value < field_descriptor.minimum:
            raise ValueError(
                f"{section_name}: '{key}' must be >= {field_descriptor.minimum}, got {value}"
            )
        if field_descriptor.choices and isinstance(value, str) and value not in field_descriptor.choices:
            raise ValueError(
                f"{section_name}: '{key}' must be one of {field_descriptor.choices}, got {value}"
            )


def describe_component_class(component_class: Type[Any]) -> ComponentDescriptor:
    """Return the nearest built-in descriptor for a component class.

    Subclasses inherit the descriptor of the nearest built-in base class so the
    core metadata remains available for synthetic and hardware adapters.
    """

    descriptor = getattr(component_class, "COMPONENT_DESCRIPTOR", None)
    if isinstance(descriptor, ComponentDescriptor):
        return descriptor

    for cls in component_class.mro():
        descriptor = _DESCRIPTORS_BY_CLASS.get(cls)
        if descriptor is not None:
            return descriptor

    return ComponentDescriptor(
        section_name=component_class.__name__.lower(),
        category="component",
        component_class=component_class,
        description=(component_class.__doc__ or "").strip(),
        supports_hard_rtc=False,
    )
