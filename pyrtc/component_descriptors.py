"""Descriptor metadata for pyrtc components.

The descriptor model captures the stable, machine-readable information that
future manager, GUI, and plugin layers need: config fields, worker functions,
stream contracts, and extension metadata.
"""

from __future__ import annotations

import difflib
import inspect
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Type

from pyrtc.corrector_splitter import CorrectorSplitter
from pyrtc.image_reconstructor import TorchImageReconstructor
from pyrtc.isio_bridge import IsioBridge
from pyrtc.loop import Loop
from pyrtc.science_camera import ScienceCamera
from pyrtc.slopes_process import SlopesProcess
from pyrtc.telemetry import Telemetry
from pyrtc.wavefront_corrector import WavefrontCorrector
from pyrtc.wavefront_sensor import WavefrontSensor


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
    case_sensitive: bool = True

    def matches_choice(self, value: Any) -> bool:
        """Return whether ``value`` is one of :attr:`choices` (always true without choices)."""

        if not self.choices:
            return True
        if not isinstance(value, str):
            return False
        if self.case_sensitive:
            return value in self.choices
        return value.lower() in {choice.lower() for choice in self.choices}

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
            if not self.case_sensitive:
                parts.append("  case_sensitive: False")
        if self.allow_none:
            parts.append("  allow_none: True")
        parts.append(f"  description: {self.description}")
        return "\n".join(parts)

    def __repr__(self) -> str:
        return self.summary()

    __str__ = __repr__


@dataclass(frozen=True)
class StreamDescriptor:
    """Describe one named pyrtc stream used by a component."""

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
        payload["required_fields"] = [
            field_descriptor.to_dict() for field_descriptor in self.required_fields
        ]
        payload["optional_fields"] = [
            field_descriptor.to_dict() for field_descriptor in self.optional_fields
        ]
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
            ConfigFieldDescriptor(
                "width", "int", "Raw image width in pixels.", required=True, minimum=1
            ),
            ConfigFieldDescriptor(
                "height", "int", "Raw image height in pixels.", required=True, minimum=1
            ),
        ),
        optional_fields=(
            ConfigFieldDescriptor(
                "name", "str", "Component display name.", default="wavefrontSensor"
            ),
            ConfigFieldDescriptor(
                "dark_count",
                "int",
                "Number of exposures to average for dark acquisition.",
                default=1000,
                minimum=0,
            ),
            ConfigFieldDescriptor(
                "dark_file", "str", "Path to a persisted dark frame.", default=""
            ),
            ConfigFieldDescriptor(
                "downsample_factor",
                "int",
                "Integer factor applied to the processed image.",
                default=0,
                minimum=0,
            ),
            ConfigFieldDescriptor(
                "rotation_angle",
                "float",
                "Rotation angle in degrees applied to the processed image.",
                default=0.0,
            ),
            ConfigFieldDescriptor(
                "functions", "list[str]", "Worker methods started in component threads.", default=[]
            ),
            ConfigFieldDescriptor(
                "affinity",
                "int | None",
                "Base CPU core; worker threads are pinned to consecutive cores when set.",
                default=None,
            ),
            ConfigFieldDescriptor(
                "realtime_priority",
                "int",
                "SCHED_FIFO priority for worker threads (Linux; 0 keeps normal scheduling).",
                default=0,
                minimum=0,
            ),
            ConfigFieldDescriptor(
                "gpu_device", "str | None", "Optional GPU device identifier.", default=None
            ),
        ),
        worker_functions=("expose",),
        input_streams=(),
        output_streams=(
            StreamDescriptor(
                "wfs_raw",
                "output",
                dtype="uint16",
                shape="(width, height)",
                description="Raw WFS image stream.",
            ),
            StreamDescriptor(
                "wfs",
                "output",
                dtype="int32",
                shape="(processed_width, processed_height)",
                description="Dark-subtracted processed WFS image stream.",
            ),
        ),
        supports_hard_rtc=True,
        calibration_artifacts=("dark_file",),
    ),
    ComponentDescriptor(
        section_name="slopes",
        category="slopes_process",
        component_class=SlopesProcess,
        description="Signal reduction stage that converts WFS images into slopes or related wavefront signals.",
        required_fields=(
            ConfigFieldDescriptor(
                "type",
                "str",
                "Wavefront-sensor reduction mode (SHWFS or PYWFS, case-insensitive).",
                required=True,
                choices=SlopesProcess.SUPPORTED_WFS_TYPES,
                case_sensitive=False,
            ),
            ConfigFieldDescriptor(
                "signal_type",
                "str",
                "Signal representation produced by the reducer (only 'slopes' is supported).",
                required=True,
                choices=SlopesProcess.SUPPORTED_SIGNAL_TYPES,
                case_sensitive=False,
            ),
        ),
        optional_fields=(
            ConfigFieldDescriptor(
                "image_noise", "float", "Configured image noise estimate.", default=0.0, minimum=0.0
            ),
            ConfigFieldDescriptor(
                "central_obscuration_ratio",
                "float",
                "Central obscuration ratio used by PYWFS paths.",
                default=0.0,
                minimum=0.0,
            ),
            ConfigFieldDescriptor(
                "flat_norm", "bool", "Whether to normalize the PYWFS flat.", default=True
            ),
            ConfigFieldDescriptor(
                "pupils", "list[str]", "Pupil centers for PYWFS in 'x,y' form.", default=[]
            ),
            ConfigFieldDescriptor(
                "pupils_radius",
                "int",
                "Pupil radius for explicit PYWFS geometry.",
                default=None,
                minimum=1,
            ),
            ConfigFieldDescriptor("contrast", "float", "SHWFS contrast parameter.", default=0.0),
            ConfigFieldDescriptor(
                "centroider",
                "str",
                "SHWFS centroiding algorithm: thresholded CoG, Gaussian-weighted CoG, or correlation.",
                default="cog",
                choices=("cog", "wcog", "correlation"),
            ),
            ConfigFieldDescriptor(
                "wcog_fwhm",
                "float",
                "FWHM in pixels of the WCoG Gaussian weight (default: half the sub-aperture).",
                default=None,
            ),
            ConfigFieldDescriptor(
                "wcog_spot_fwhm",
                "float",
                "Spot FWHM in pixels for WCoG gain correction; 0 leaves the gain uncorrected.",
                default=0.0,
                minimum=0.0,
            ),
            ConfigFieldDescriptor(
                "correlation_search_radius",
                "int",
                "Correlation search half-width in pixels (default: a quarter of the sub-aperture).",
                default=None,
                minimum=1,
            ),
            ConfigFieldDescriptor(
                "reference_image_file",
                "str",
                "Path to the SHWFS reference image used by the wcog and correlation centroiders.",
                default="",
            ),
            ConfigFieldDescriptor(
                "sub_ap_spacing",
                "float",
                "Sub-aperture spacing for SHWFS layouts.",
                default=None,
                minimum=1,
            ),
            ConfigFieldDescriptor(
                "sub_ap_offset_x", "int", "SHWFS X offset in pixels.", default=0, minimum=0
            ),
            ConfigFieldDescriptor(
                "sub_ap_offset_y", "int", "SHWFS Y offset in pixels.", default=0, minimum=0
            ),
            ConfigFieldDescriptor(
                "ref_slope_count",
                "int",
                "Number of frames used to average reference slopes.",
                default=1000,
                minimum=1,
            ),
            ConfigFieldDescriptor(
                "valid_sub_aps_file", "str", "Path to the valid sub-aperture mask file.", default=""
            ),
            ConfigFieldDescriptor(
                "ref_slopes_file", "str", "Path to the reference slopes file.", default=""
            ),
            ConfigFieldDescriptor(
                "functions", "list[str]", "Worker methods started in component threads.", default=[]
            ),
            ConfigFieldDescriptor(
                "affinity",
                "int | None",
                "Base CPU core; worker threads are pinned to consecutive cores when set.",
                default=None,
            ),
            ConfigFieldDescriptor(
                "realtime_priority",
                "int",
                "SCHED_FIFO priority for worker threads (Linux; 0 keeps normal scheduling).",
                default=0,
                minimum=0,
            ),
            ConfigFieldDescriptor(
                "gpu_device", "str | None", "Optional GPU device identifier.", default=None
            ),
        ),
        worker_functions=("compute_signal",),
        input_streams=(
            StreamDescriptor(
                "wfs",
                "input",
                dtype="int32",
                shape="(processed_width, processed_height)",
                description="Processed wavefront-sensor image stream.",
            ),
        ),
        output_streams=(
            StreamDescriptor(
                "signal",
                "output",
                dtype="float32",
                shape="(signal_size,)",
                description="Flattened residual signal stream.",
            ),
            StreamDescriptor(
                "signal_2d",
                "output",
                dtype="float32",
                shape="(signal_rows, signal_cols)",
                description="2D visualization of the residual signal.",
            ),
        ),
        supports_hard_rtc=True,
        calibration_artifacts=("valid_sub_aps_file", "ref_slopes_file", "reference_image_file"),
    ),
    ComponentDescriptor(
        section_name="image_reconstructor",
        category="slopes_process",
        component_class=TorchImageReconstructor,
        description=(
            "Publishes a PyTorch model's output on each WFS image as the loop's signal "
            "(neural or focal-plane reconstructor). Goes in the slopes section."
        ),
        required_fields=(
            ConfigFieldDescriptor(
                "signal_size",
                "int",
                "Number of model outputs (the signal length); checked against the model.",
                required=True,
                minimum=1,
            ),
        ),
        optional_fields=(
            ConfigFieldDescriptor(
                "model_file", "str", "TorchScript model file (torch.jit.save).", default=""
            ),
            ConfigFieldDescriptor(
                "model_factory",
                "str",
                "'module:function' returning an nn.Module (a function name with "
                "model_factory_file).",
                default="",
            ),
            ConfigFieldDescriptor(
                "model_factory_file",
                "str",
                "Python file defining model_factory.",
                default="",
            ),
            ConfigFieldDescriptor(
                "model_kwargs",
                "dict | None",
                "Keyword arguments for model_factory.",
                default=None,
            ),
            ConfigFieldDescriptor(
                "state_dict_file",
                "str",
                "State dict loaded strictly into the model.",
                default="",
            ),
            ConfigFieldDescriptor(
                "device", "str", "Model device: cpu, cuda or cuda:N.", default="cpu"
            ),
            ConfigFieldDescriptor(
                "dtype",
                "str",
                "Model precision (float16 needs CUDA); outputs are float32.",
                default="float32",
                choices=("float32", "float16"),
                case_sensitive=False,
            ),
            ConfigFieldDescriptor(
                "input_shape",
                "list[int] | None",
                "Shape the model takes; default [1, 1, *image_shape].",
                default=None,
            ),
            ConfigFieldDescriptor(
                "flux_normalization",
                "str",
                "Divide the image by its total ('sum') or mean pixel ('mean') flux.",
                default="none",
                choices=("none", "sum", "mean"),
                case_sensitive=False,
            ),
            ConfigFieldDescriptor(
                "sqrt_stretch",
                "bool",
                "Square-root stretch after normalisation (negative pixels clipped to 0).",
                default=False,
            ),
            ConfigFieldDescriptor(
                "output_scale_file",
                "str",
                ".npy file of signal_size factors multiplied into the output.",
                default="",
            ),
            ConfigFieldDescriptor(
                "signal_2d_shape",
                "list[int] | None",
                "Shape of an optional signal_2d display stream (signal_size elements).",
                default=None,
            ),
            ConfigFieldDescriptor(
                "cuda_graph",
                "bool",
                "Capture the model in a CUDA graph on CUDA devices (eager fallback).",
                default=True,
            ),
            ConfigFieldDescriptor(
                "warmup_iters",
                "int",
                "Forward passes run before CUDA-graph capture.",
                default=10,
                minimum=0,
            ),
            ConfigFieldDescriptor(
                "cpu_threads",
                "int | None",
                "torch.set_num_threads for CPU models (process-wide).",
                default=None,
                minimum=1,
            ),
            ConfigFieldDescriptor(
                "timing_window",
                "int",
                "Recent per-frame compute times kept for timing_stats().",
                default=1000,
                minimum=1,
            ),
            ConfigFieldDescriptor(
                "functions", "list[str]", "Worker methods started in component threads.", default=[]
            ),
            ConfigFieldDescriptor(
                "affinity",
                "int | None",
                "Base CPU core; worker threads are pinned to consecutive cores when set.",
                default=None,
            ),
            ConfigFieldDescriptor(
                "realtime_priority",
                "int",
                "SCHED_FIFO priority for worker threads (Linux; 0 keeps normal scheduling).",
                default=0,
                minimum=0,
            ),
            ConfigFieldDescriptor(
                "gpu_device",
                "str | None",
                "Attach the wfs input on the GPU and create GPU-backed outputs.",
                default=None,
            ),
        ),
        worker_functions=("compute_signal",),
        input_streams=(
            StreamDescriptor(
                "wfs",
                "input",
                shape="(processed_width, processed_height)",
                description="Processed wavefront-sensor image stream.",
            ),
        ),
        output_streams=(
            StreamDescriptor(
                "signal",
                "output",
                dtype="float32",
                shape="(signal_size,)",
                description="Model output, one value per signal element.",
            ),
            StreamDescriptor(
                "signal_2d",
                "output",
                dtype="float32",
                shape="signal_2d_shape",
                optional=True,
                description="The output reshaped for display (only with signal_2d_shape).",
            ),
        ),
        supports_hard_rtc=True,
        external_dependencies=("torch",),
        calibration_artifacts=("model_file", "state_dict_file", "output_scale_file"),
    ),
    ComponentDescriptor(
        section_name="loop",
        category="control_loop",
        component_class=Loop,
        description="Adaptive-optics controller that converts residual signals into correction commands.",
        required_fields=(),
        optional_fields=(
            ConfigFieldDescriptor(
                "num_dropped_modes",
                "int",
                "Number of controlled modes to suppress.",
                default=0,
                minimum=0,
            ),
            ConfigFieldDescriptor(
                "cm_method",
                "str",
                "Control-matrix inversion method ('svd' or 'tikhonov').",
                default="svd",
            ),
            ConfigFieldDescriptor(
                "conditioning",
                "float | None",
                "Optional target conditioning number used to truncate small singular values.",
                default=None,
                minimum=1.0,
            ),
            ConfigFieldDescriptor(
                "tikhonov_reg",
                "float",
                "Tikhonov regularization strength used when cm_method is 'tikhonov'.",
                default=0.0,
                minimum=0.0,
            ),
            ConfigFieldDescriptor("gain", "float", "Integrator gain.", default=0.1),
            ConfigFieldDescriptor(
                "modal_gains",
                "list[float] | str | None",
                "Per-mode gain factors (num_modes values or a .npy file); unset means all 1.",
                default=None,
            ),
            ConfigFieldDescriptor(
                "optical_gains",
                "list[float] | str | None",
                "Per-mode WFS optical gains that the loop compensates; unset means all 1.",
                default=None,
            ),
            ConfigFieldDescriptor("leaky_gain", "float", "Leaky-integrator gain.", default=0.0),
            ConfigFieldDescriptor(
                "hardware_delay", "float", "Estimated hardware delay.", default=0.0, minimum=0.0
            ),
            ConfigFieldDescriptor(
                "poke_amp", "float", "Calibration poke amplitude.", default=1e-2, minimum=0.0
            ),
            ConfigFieldDescriptor(
                "num_iters_im",
                "int",
                "Interaction-matrix calibration iteration count.",
                default=100,
                minimum=1,
            ),
            ConfigFieldDescriptor(
                "delay", "int", "Artificial delay in frames.", default=0, minimum=0
            ),
            ConfigFieldDescriptor(
                "im_method",
                "str",
                "Interaction-matrix calibration method ('push-pull', 'hadamard' or 'docrime').",
                default="push-pull",
                choices=Loop.SUPPORTED_IM_METHODS,
                case_sensitive=False,
            ),
            ConfigFieldDescriptor(
                "im_settle_frames",
                "int",
                "Signal frames discarded after each calibration poke before averaging.",
                default=1,
                minimum=0,
            ),
            ConfigFieldDescriptor(
                "im_round_trip_check",
                "bool",
                "Check that a DM poke reaches the signal before compute_im calibrates.",
                default=True,
            ),
            ConfigFieldDescriptor(
                "im_timeout",
                "float",
                "Seconds allowed for the round-trip check and for each calibration frame.",
                default=30.0,
                minimum=0.0,
            ),
            ConfigFieldDescriptor(
                "predictor",
                "dict | None",
                "Predictive control for predictive_integrator: type (persistence, ar_kalman, "
                "least_squares), delay_frames, gain, fit_frames and the predictor's options.",
                default=None,
            ),
            ConfigFieldDescriptor(
                "watchdog_timeout",
                "float | None",
                "Seconds without a new signal frame before the closed loop reports its input "
                "stale; unset or 0 disables the watchdog.",
                default=1.0,
                minimum=0.0,
            ),
            ConfigFieldDescriptor(
                "watchdog_action",
                "str",
                "What a stale loop input does: 'hold', 'open' (stop the loop) or 'flatten'.",
                default="hold",
                choices=("hold", "open", "flatten"),
                case_sensitive=False,
            ),
            ConfigFieldDescriptor(
                "im_file", "str", "Path to the interaction-matrix file.", default=""
            ),
            ConfigFieldDescriptor("p_gain", "float", "PID proportional gain.", default=0.1),
            ConfigFieldDescriptor("i_gain", "float", "PID integral gain.", default=0.0),
            ConfigFieldDescriptor("d_gain", "float", "PID derivative gain.", default=0.0),
            ConfigFieldDescriptor(
                "control_limits",
                "list[float]",
                "PID control output limits.",
                default=[float("-inf"), float("inf")],
            ),
            ConfigFieldDescriptor(
                "integral_limits",
                "list[float]",
                "PID integral limits.",
                default=[float("-inf"), float("inf")],
            ),
            ConfigFieldDescriptor(
                "absolute_limits",
                "list[float]",
                "Absolute correction limits.",
                default=[float("-inf"), float("inf")],
            ),
            ConfigFieldDescriptor(
                "derivative_filter", "float", "PID derivative filter coefficient.", default=0.1
            ),
            ConfigFieldDescriptor(
                "functions", "list[str]", "Worker methods started in component threads.", default=[]
            ),
            ConfigFieldDescriptor(
                "affinity",
                "int | None",
                "Base CPU core; worker threads are pinned to consecutive cores when set.",
                default=None,
            ),
            ConfigFieldDescriptor(
                "realtime_priority",
                "int",
                "SCHED_FIFO priority for worker threads (Linux; 0 keeps normal scheduling).",
                default=0,
                minimum=0,
            ),
            ConfigFieldDescriptor(
                "gpu_device", "str | None", "Optional GPU device identifier.", default=None
            ),
        ),
        worker_functions=(
            "standard_integrator",
            "standard_integrator_pol",
            "leaky_integrator",
            "pid_integrator",
            "pid_integrator_pol",
            "predictive_integrator",
        ),
        input_streams=(
            StreamDescriptor(
                "signal",
                "input",
                dtype="float32",
                shape="(signal_size,)",
                description="Residual signal from slopes processing.",
            ),
        ),
        output_streams=(
            StreamDescriptor(
                "wfc",
                "output",
                dtype="float32",
                shape="(num_modes,)",
                description="Modal correction vector sent to the wavefront corrector.",
            ),
        ),
        supports_hard_rtc=True,
        calibration_artifacts=("im_file",),
    ),
    ComponentDescriptor(
        section_name="wfc",
        category="wavefront_corrector",
        component_class=WavefrontCorrector,
        description="Wavefront-corrector interface that maps modal commands into actuator space and hardware updates.",
        required_fields=(
            ConfigFieldDescriptor("name", "str", "Component display name.", required=True),
            ConfigFieldDescriptor(
                "num_actuators",
                "int",
                "Number of actuators in zonal space.",
                required=True,
                minimum=1,
            ),
            ConfigFieldDescriptor(
                "num_modes",
                "int",
                "Number of controlled modes in modal space.",
                required=True,
                minimum=1,
            ),
        ),
        optional_fields=(
            ConfigFieldDescriptor(
                "m2c_file",
                "str",
                "Path to the mode-to-command matrix. Takes precedence over 'basis'.",
                default="",
            ),
            ConfigFieldDescriptor(
                "basis",
                "dict | None",
                "Modal basis built with aobasis from the actuator geometry: "
                "{type: kl|zernike|fourier|zonal|zonal_fast|hadamard, n_modes, pupil_diameter, "
                "normalize, orthonormalize, positions_file, r0, L0, ignore_piston, use_gpu, "
                "min_distance}. Ignored when 'm2c_file' is set.",
                default=None,
                allow_none=True,
            ),
            ConfigFieldDescriptor("flat_file", "str", "Path to the flat shape file.", default=""),
            ConfigFieldDescriptor(
                "floating_influence_radius",
                "int",
                "Radius used when floating inactive actuators.",
                default=1,
                minimum=0,
            ),
            ConfigFieldDescriptor(
                "frame_delay",
                "int",
                "Artificial frame delay in actuator space.",
                default=0,
                minimum=0,
            ),
            ConfigFieldDescriptor(
                "save_file", "str", "Path used when saving a zonal shape.", default="wfc_shape.npy"
            ),
            ConfigFieldDescriptor(
                "command_cap",
                "float | None",
                "Symmetric clip applied to zonal actuator commands; unset disables clipping.",
                default=None,
                minimum=0.0,
            ),
            ConfigFieldDescriptor(
                "saturation_warn_fraction",
                "float",
                "Fraction of actuators at command_cap that triggers a saturation warning.",
                default=0.05,
                minimum=0.0,
            ),
            ConfigFieldDescriptor(
                "display_grid_size",
                "int",
                "Side length of the square wfc_2d visualization stream.",
                default=33,
                minimum=0,
            ),
            ConfigFieldDescriptor(
                "functions", "list[str]", "Worker methods started in component threads.", default=[]
            ),
            ConfigFieldDescriptor(
                "affinity",
                "int | None",
                "Base CPU core; worker threads are pinned to consecutive cores when set.",
                default=None,
            ),
            ConfigFieldDescriptor(
                "realtime_priority",
                "int",
                "SCHED_FIFO priority for worker threads (Linux; 0 keeps normal scheduling).",
                default=0,
                minimum=0,
            ),
            ConfigFieldDescriptor(
                "gpu_device", "str | None", "Optional GPU device identifier.", default=None
            ),
        ),
        worker_functions=("send_to_hardware",),
        input_streams=(
            StreamDescriptor(
                "wfc",
                "input",
                dtype="float32",
                shape="(num_modes,)",
                description="Modal correction vector from the loop controller.",
            ),
        ),
        output_streams=(
            StreamDescriptor(
                "wfc",
                "output",
                dtype="float32",
                shape="(num_modes,)",
                description="Published correction vector for readers and launchers.",
            ),
            StreamDescriptor(
                "wfc_2d",
                "output",
                dtype="float32",
                shape="layout.shape",
                optional=True,
                description="Optional 2D actuator-layout visualization stream.",
            ),
        ),
        supports_hard_rtc=True,
        calibration_artifacts=("m2c_file", "flat_file"),
    ),
    ComponentDescriptor(
        section_name="psf",
        category="science_camera",
        component_class=ScienceCamera,
        description="Science-camera interface that publishes short- and long-exposure PSFs plus image-quality telemetry.",
        required_fields=(
            ConfigFieldDescriptor("name", "str", "Component display name.", required=True),
            ConfigFieldDescriptor(
                "width", "int", "Image width in pixels.", required=True, minimum=1
            ),
            ConfigFieldDescriptor(
                "height", "int", "Image height in pixels.", required=True, minimum=1
            ),
            ConfigFieldDescriptor(
                "dark_count",
                "int",
                "Number of exposures to average for dark acquisition.",
                required=True,
                minimum=1,
            ),
            ConfigFieldDescriptor(
                "integration",
                "int",
                "Number of frames averaged for the long-exposure PSF.",
                required=True,
                minimum=1,
            ),
        ),
        optional_fields=(
            ConfigFieldDescriptor(
                "dark_file", "str", "Path to a persisted dark frame.", default=""
            ),
            ConfigFieldDescriptor("model_file", "str", "Path to a model PSF file.", default=""),
            ConfigFieldDescriptor(
                "functions", "list[str]", "Worker methods started in component threads.", default=[]
            ),
            ConfigFieldDescriptor(
                "affinity",
                "int | None",
                "Base CPU core; worker threads are pinned to consecutive cores when set.",
                default=None,
            ),
            ConfigFieldDescriptor(
                "realtime_priority",
                "int",
                "SCHED_FIFO priority for worker threads (Linux; 0 keeps normal scheduling).",
                default=0,
                minimum=0,
            ),
            ConfigFieldDescriptor(
                "gpu_device", "str | None", "Optional GPU device identifier.", default=None
            ),
        ),
        worker_functions=("expose", "integrate"),
        input_streams=(),
        output_streams=(
            StreamDescriptor(
                "psf_short",
                "output",
                dtype="int32",
                shape="(width, height)",
                description="Short-exposure PSF image stream.",
            ),
            StreamDescriptor(
                "psf_long",
                "output",
                dtype="float64",
                shape="(width, height)",
                description="Long-exposure PSF image stream.",
            ),
            StreamDescriptor(
                "strehl",
                "output",
                dtype="float",
                shape="(1,)",
                description="Scalar Strehl estimate.",
            ),
            StreamDescriptor(
                "tiptilt",
                "output",
                dtype="float",
                shape="(1,)",
                description="Scalar tip-tilt estimate.",
            ),
        ),
        supports_hard_rtc=True,
        calibration_artifacts=("dark_file", "model_file"),
    ),
    ComponentDescriptor(
        section_name="telemetry",
        category="telemetry",
        component_class=Telemetry,
        description="Telemetry capture helper for persisting existing pyrtc streams to disk.",
        required_fields=(),
        optional_fields=(
            ConfigFieldDescriptor(
                "data_dir",
                "str",
                "Base directory used for telemetry capture files.",
                default="./data/",
            ),
            ConfigFieldDescriptor(
                "streams",
                "list[str]",
                "Default streams for save_configured_streams() and the ring buffer.",
                default=[],
            ),
            ConfigFieldDescriptor(
                "ring_buffer",
                "dict | None",
                "Continuous recording: {streams, seconds, frames, probe_seconds, autostart}; "
                "started with the component and dumped with dump_ring_buffer().",
                default=None,
                allow_none=True,
            ),
            ConfigFieldDescriptor(
                "functions", "list[str]", "Worker methods started in component threads.", default=[]
            ),
            ConfigFieldDescriptor(
                "affinity",
                "int | None",
                "Base CPU core; worker threads are pinned to consecutive cores when set.",
                default=None,
            ),
            ConfigFieldDescriptor(
                "realtime_priority",
                "int",
                "SCHED_FIFO priority for worker threads (Linux; 0 keeps normal scheduling).",
                default=0,
                minimum=0,
            ),
            ConfigFieldDescriptor(
                "gpu_device", "str | None", "Optional GPU device identifier.", default=None
            ),
        ),
        worker_functions=(),
        input_streams=(
            StreamDescriptor(
                "*", "input", description="Attaches to existing streams on demand via save()."
            ),
        ),
        output_streams=(),
        supports_hard_rtc=False,
        calibration_artifacts=(),
    ),
    ComponentDescriptor(
        section_name="corrector_splitter",
        category="wavefront_corrector",
        component_class=CorrectorSplitter,
        description=(
            "Splits the loop's modal command across several correctors (woofer/tweeter) "
            "and offloads content between them. Goes in the loop's wfc section."
        ),
        required_fields=(
            ConfigFieldDescriptor(
                "correctors",
                "list",
                "Correctors as {name, stream, modes}; the loop's modes are theirs, in order.",
                required=True,
            ),
        ),
        optional_fields=(
            ConfigFieldDescriptor(
                "offload",
                "dict | None",
                "Offloading: {source, target, gain, coupling (matrix or .npy)}.",
                default=None,
            ),
            ConfigFieldDescriptor(
                "num_modes",
                "int",
                "Total modes; must equal the correctors' total when given.",
                default=None,
                minimum=1,
            ),
        ),
        worker_functions=("split",),
        input_streams=(
            StreamDescriptor(
                "wfc",
                "input",
                dtype="float32",
                shape="(num_modes,)",
                description="Combined modal command from the loop.",
            ),
        ),
        output_streams=(
            StreamDescriptor(
                "wfc",
                "output",
                dtype="float32",
                shape="(num_modes,)",
                description="The loop's command stream, owned by the splitter.",
            ),
        ),
        supports_hard_rtc=False,
        calibration_artifacts=(),
    ),
    ComponentDescriptor(
        section_name="isio_bridge",
        category="bridge",
        component_class=IsioBridge,
        description="Mirrors one stream to or from ImageStreamIO (milk/CACAO) shared memory.",
        required_fields=(
            ConfigFieldDescriptor(
                "direction",
                "str",
                "'to_isio' (pyrtc -> ISIO) or 'from_isio' (ISIO -> pyrtc).",
                required=True,
                choices=("to_isio", "from_isio"),
                case_sensitive=False,
            ),
            ConfigFieldDescriptor("isio_name", "str", "ImageStreamIO stream name.", required=True),
        ),
        optional_fields=(
            ConfigFieldDescriptor(
                "poll_interval", "float", "Seconds between ISIO polls.", default=1e-4, minimum=0.0
            ),
            ConfigFieldDescriptor(
                "wait_slice",
                "float",
                "Longest single wait before the worker rechecks its state.",
                default=0.1,
                minimum=0.0,
            ),
            ConfigFieldDescriptor(
                "num_semaphores",
                "int",
                "Semaphores of a created ISIO stream.",
                default=10,
                minimum=1,
            ),
        ),
        worker_functions=("mirror",),
        input_streams=(
            StreamDescriptor(
                "input", "input", optional=True, description="pyrtc stream copied to ISIO."
            ),
        ),
        output_streams=(
            StreamDescriptor(
                "output", "output", optional=True, description="pyrtc stream fed from ISIO."
            ),
        ),
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
    if field_type == "dict | None":
        return value is None or isinstance(value, Mapping)
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
        descriptor.section_name: descriptor.to_dict() for descriptor in list_component_descriptors()
    }


def validate_config_with_descriptor(
    section_name: str, conf: Mapping[str, Any], descriptor: ComponentDescriptor | None = None
) -> None:
    """Validate generic field presence and types using descriptor metadata.

    ``descriptor`` defaults to the section's built-in descriptor.
    """

    if descriptor is None:
        descriptor = get_component_descriptor(section_name)
    if descriptor is None:
        return

    known_fields = {
        field_descriptor.name: field_descriptor for field_descriptor in descriptor.all_fields
    }

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
        if (
            value is None
            and field_descriptor.default is None
            and field_descriptor.field_type.endswith("| None")
        ):
            continue
        if value is None and not field_descriptor.allow_none:
            raise ValueError(f"{section_name}: '{key}' may not be null")
        if not _field_type_matches(field_descriptor.field_type, value):
            raise TypeError(
                f"{section_name}: '{key}' must match descriptor type {field_descriptor.field_type}"
            )
        if (
            field_descriptor.minimum is not None
            and isinstance(value, (int, float))
            and value < field_descriptor.minimum
        ):
            raise ValueError(
                f"{section_name}: '{key}' must be >= {field_descriptor.minimum}, got {value}"
            )
        if isinstance(value, str) and not field_descriptor.matches_choice(value):
            qualifier = "" if field_descriptor.case_sensitive else " (case-insensitive)"
            raise ValueError(
                f"{section_name}: '{key}' must be one of {field_descriptor.choices}"
                f"{qualifier}, got {value!r}"
            )


def _find_component_descriptor(component_class: Type[Any]) -> ComponentDescriptor | None:
    """Return the declared or nearest registered descriptor, or ``None``."""

    descriptor = getattr(component_class, "COMPONENT_DESCRIPTOR", None)
    if isinstance(descriptor, ComponentDescriptor):
        return descriptor

    for cls in component_class.mro():
        descriptor = _DESCRIPTORS_BY_CLASS.get(cls)
        if descriptor is not None:
            return descriptor
        # A file-loaded copy of a built-in class (see _is_same_builtin_class).
        for builtin_class, builtin_descriptor in _DESCRIPTORS_BY_CLASS.items():
            if _is_same_builtin_class(cls, builtin_class):
                return builtin_descriptor
    return None


def describe_component_class(component_class: Type[Any]) -> ComponentDescriptor:
    """Return the nearest built-in descriptor for a component class.

    Subclasses inherit the descriptor of the nearest built-in base class so the
    core metadata remains available for synthetic and hardware adapters.
    """

    descriptor = _find_component_descriptor(component_class)
    if descriptor is not None:
        return descriptor

    return ComponentDescriptor(
        section_name=component_class.__name__.lower(),
        category="component",
        component_class=component_class,
        description=(component_class.__doc__ or "").strip(),
        supports_hard_rtc=False,
    )


# Keys every component section may carry regardless of its descriptor: class
# loading, worker threads and scheduling, stream aliases, and resource binding
# (read by ``Component``, the manager, and stream planning). ``name`` is used
# by the manager as a class-target fallback and by most components for display.
COMMON_COMPONENT_CONFIG_KEYS = frozenset(
    {
        "class_name",
        "class_file",
        "name",
        "functions",
        "affinity",
        "realtime_priority",
        "gpu_device",
        "input_streams",
        "output_streams",
        "resource",
    }
)


def _class_source_file(cls: Type[Any]) -> Path | None:
    """Return the file a class body was defined in, if it can be found."""

    for value in vars(cls).values():
        function = getattr(value, "__func__", value)
        code = getattr(function, "__code__", None)
        if code is not None:
            return Path(code.co_filename).resolve()
    try:
        return Path(inspect.getfile(cls)).resolve()
    except (TypeError, OSError):
        return None


def _is_same_builtin_class(candidate: Type[Any], builtin: Type[Any]) -> bool:
    """Return whether ``candidate`` is ``builtin`` or a file-loaded copy of it.

    A config ``class_file`` pointing at a pyrtc source file outside the
    installed package (e.g. a checkout next to a wheel install) is imported as
    a separate module, producing a distinct class object with the same code.
    """

    if candidate is builtin:
        return True
    if candidate.__qualname__ != builtin.__qualname__:
        return False
    candidate_file = _class_source_file(candidate)
    if candidate_file is None:
        return False
    builtin_parts = builtin.__module__.split(".")
    builtin_parts[-1] += ".py"
    return candidate_file.parts[-len(builtin_parts) :] == tuple(builtin_parts)


def known_config_keys(component_class: Type[Any]) -> frozenset[str] | None:
    """Return the config keys a component class is known to read.

    The known keys are :data:`COMMON_COMPONENT_CONFIG_KEYS`, the fields of the
    class's descriptor, and every ``EXTRA_CONFIG_KEYS`` tuple declared along
    its MRO. Subclasses that read keys of their own declare them with
    ``EXTRA_CONFIG_KEYS = ("serial", ...)`` (or a full ``COMPONENT_DESCRIPTOR``).

    Returns ``None`` when the key set is not known, so unknown-key checks are
    skipped: the class has no descriptor, or it is a subclass of a built-in
    component that declares neither ``EXTRA_CONFIG_KEYS`` nor
    ``COMPONENT_DESCRIPTOR`` in its own body (its extra keys are unknown, and
    warning about them would only produce false positives).
    """

    descriptor = _find_component_descriptor(component_class)
    if descriptor is None:
        return None
    own_attributes = vars(component_class)
    declares_keys = (
        _is_same_builtin_class(component_class, descriptor.component_class)
        or "EXTRA_CONFIG_KEYS" in own_attributes
        or "COMPONENT_DESCRIPTOR" in own_attributes
    )
    if not declares_keys:
        return None

    keys = set(COMMON_COMPONENT_CONFIG_KEYS)
    keys.update(field_descriptor.name for field_descriptor in descriptor.all_fields)
    for cls in component_class.mro():
        keys.update(vars(cls).get("EXTRA_CONFIG_KEYS", ()))
    return frozenset(keys)


def unknown_config_key_warnings(
    section_name: str | None, conf: Mapping[str, Any], component_class: Type[Any]
) -> list[str]:
    """Return one warning per config key that ``component_class`` does not read.

    Keys starting with ``_`` are private runtime keys (``_sectionName``,
    ``_systemStreams``, ...) and are never reported. See
    :func:`known_config_keys` for how the known key set is built and when the
    check is skipped.
    """

    known = known_config_keys(component_class)
    if known is None or not isinstance(conf, Mapping):
        return []
    label = section_name or describe_component_class(component_class).section_name
    warnings = []
    for key in conf:
        key = str(key)
        if key.startswith("_") or key in known:
            continue
        message = f"{label}: unknown config key '{key}' is ignored by {component_class.__name__}"
        suggestion = difflib.get_close_matches(key, sorted(known), n=1)
        if suggestion:
            message += f" (did you mean '{suggestion[0]}'?)"
        warnings.append(message)
    return warnings
