"""Bridge between pyRTC components and a SPECULA optical simulation.

This module adapts a narrow, pyRTC-owned SPECULA object graph into the
wavefront-sensor and wavefront-corrector interfaces already used throughout the
repository. The first implementation is intentionally limited to a soft-RTC,
single-process Pyramid-WFS workflow where SPECULA owns the optical state and
pyRTC owns slopes extraction and control.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import os
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from pyRTC.logging_utils import get_logger
from pyRTC.Pipeline import ImageSHM, Listener
from pyRTC.ScienceCamera import ScienceCamera
from pyRTC.WavefrontCorrector import WavefrontCorrector
from pyRTC.WavefrontSensor import WavefrontSensor
from pyRTC.pyRTCComponent import pyRTCComponent
from pyRTC.utils import decrease_nice, generate_circular_aperture_mask, read_yaml_file, setFromConfig, set_affinity


logger = get_logger(__name__)


def _mapping_or_none(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        return None
    return dict(value)


def _looks_like_specula_provider(provider_conf: Mapping[str, Any] | None, system_conf: Mapping[str, Any] | None = None) -> bool:
    provider_conf = _mapping_or_none(provider_conf)
    if provider_conf is not None:
        class_name = str(provider_conf.get("className", ""))
        if "SPECULAInterface" in class_name:
            return True

    system_conf = _mapping_or_none(system_conf)
    if system_conf is None:
        return False

    wfs_conf = _mapping_or_none(system_conf.get("wfs"))
    if wfs_conf is None:
        return False
    return "SPECULAWFSensor" in str(wfs_conf.get("className", ""))


def _load_specula_param_mapping(
    *,
    param: Mapping[str, Any] | None = None,
    provider_conf: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    if isinstance(param, Mapping):
        return dict(param)

    provider_conf = _mapping_or_none(provider_conf)
    if provider_conf is None:
        return None

    inline_param = provider_conf.get("param")
    if isinstance(inline_param, Mapping):
        return dict(inline_param)

    param_file = provider_conf.get("paramFile")
    if isinstance(param_file, str) and param_file:
        loaded = read_yaml_file(param_file)
        if isinstance(loaded, Mapping):
            return dict(loaded)

    return None


def derive_specula_pywfs_geometry(param: Mapping[str, Any]) -> dict[str, Any] | None:
    """Derive pyRTC PyWFS frame and pupil geometry from a SPECULA config.

    SPECULA's Pyramid parameters define the pupil footprint inside the detector
    image. pyRTC historically duplicated that geometry in the ``wfs`` and
    ``slopes`` sections, which drifted easily when the SPECULA YAML changed.
    This helper makes the SPECULA pyramid the source of truth.
    """

    param = _mapping_or_none(param)
    if param is None:
        return None

    pyramid_conf = _mapping_or_none(param.get("pyramid"))
    if pyramid_conf is None:
        return None

    detector_conf = _mapping_or_none(param.get("detector")) or {}
    detector_size = detector_conf.get("size")
    if isinstance(detector_size, (list, tuple)) and len(detector_size) >= 2:
        width = int(detector_size[0])
        height = int(detector_size[1])
    else:
        output_resolution = int(pyramid_conf.get("output_resolution", 0) or 0)
        if output_resolution < 1:
            return None
        width = output_resolution
        height = output_resolution

    output_resolution = int(pyramid_conf.get("output_resolution", width) or width)
    if output_resolution < 1:
        output_resolution = width

    pup_diam = float(pyramid_conf.get("pup_diam", 0) or 0)
    if pup_diam <= 0.0:
        return {
            "width": width,
            "height": height,
        }

    pup_margin = float(pyramid_conf.get("pup_margin", 2) or 2)
    pup_dist = float(pyramid_conf.get("pup_dist", pup_diam + 2.0 * pup_margin) or (pup_diam + 2.0 * pup_margin))

    scale_x = float(width) / float(output_resolution)
    scale_y = float(height) / float(output_resolution)
    scaled_dist_x = float(pup_dist) * scale_x
    scaled_dist_y = float(pup_dist) * scale_y
    scaled_diam = min(float(pup_diam) * scale_x, float(pup_diam) * scale_y)

    pupil_radius = max(1, int(np.floor(scaled_diam / 2.0)))

    x0 = int(np.round((float(width) - scaled_dist_x) / 2.0))
    y0 = int(np.round((float(height) - scaled_dist_y) / 2.0))
    x1 = int(x0 + np.round(scaled_dist_x))
    y1 = int(y0 + np.round(scaled_dist_y))

    return {
        "width": width,
        "height": height,
        "pupils": [
            f"{x0},{y0}",
            f"{x0},{y1}",
            f"{x1},{y0}",
            f"{x1},{y1}",
        ],
        "pupilsRadius": pupil_radius,
    }


def derive_specula_wfc_display_geometry(param: Mapping[str, Any]) -> dict[str, Any] | None:
    param = _mapping_or_none(param)
    if param is None:
        return None

    dm_conf = _mapping_or_none(param.get("dm")) or {}
    n_act = int(dm_conf.get("n_act", 0) or 0)
    if n_act < 1:
        return None

    geom = str(dm_conf.get("geom", "square")).lower()
    circ_geom = bool(dm_conf.get("circ_geom", geom == "circular"))
    if geom == "square" and not circ_geom:
        return {"displayGridSize": n_act}

    layout, _, _ = _circular_zonal_display_mapping(n_act, dm_conf.get("angle_offset", 0.0))
    return {"displayGridSize": int(layout.shape[0])}


def derive_specula_psf_geometry(param: Mapping[str, Any]) -> dict[str, Any] | None:
    param = _mapping_or_none(param)
    if param is None:
        return None

    main_conf = _mapping_or_none(param.get("main")) or {}
    psf_conf = _mapping_or_none(param.get("psf")) or {}
    pixel_pupil = int(main_conf.get("pixel_pupil", 0) or 0)
    nd = psf_conf.get("nd")
    if pixel_pupil < 1 or nd is None:
        return None

    side = int(np.around(pixel_pupil * float(nd) / 2.0) * 2)
    side = max(side, 2)
    return {"width": side, "height": side}


def sync_specula_pywfs_config(
    system_conf: Mapping[str, Any],
    *,
    param: Mapping[str, Any] | None = None,
    provider_conf: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Mutate a pyRTC config so SPECULA drives PyWFS geometry consistently."""

    if not isinstance(system_conf, Mapping):
        return None

    mutable_system_conf = system_conf
    if provider_conf is None:
        wfs_conf = _mapping_or_none(mutable_system_conf.get("wfs"))
        provider_name = str(wfs_conf.get("resource", "")).strip() if wfs_conf is not None else ""
        if provider_name:
            provider_conf = _mapping_or_none(mutable_system_conf.get(provider_name))
        elif "specula" in mutable_system_conf:
            provider_conf = _mapping_or_none(mutable_system_conf.get("specula"))

    if param is None and not _looks_like_specula_provider(provider_conf, mutable_system_conf):
        return None

    param_mapping = _load_specula_param_mapping(param=param, provider_conf=provider_conf)
    if param_mapping is None:
        return None

    geometry = derive_specula_pywfs_geometry(param_mapping)
    applied: dict[str, Any] = {}

    wfs_conf = mutable_system_conf.get("wfs")
    if isinstance(wfs_conf, dict) and geometry is not None:
        if "width" in geometry:
            wfs_conf["width"] = int(geometry["width"])
            applied["width"] = int(geometry["width"])
        if "height" in geometry:
            wfs_conf["height"] = int(geometry["height"])
            applied["height"] = int(geometry["height"])

    slopes_conf = mutable_system_conf.get("slopes")
    if isinstance(slopes_conf, dict) and str(slopes_conf.get("type", "")).lower() == "pywfs" and geometry is not None:
        pupils = geometry.get("pupils")
        if isinstance(pupils, list) and pupils:
            slopes_conf["pupils"] = list(pupils)
            applied["pupils"] = list(pupils)
        pupil_radius = geometry.get("pupilsRadius")
        if isinstance(pupil_radius, (int, np.integer)):
            slopes_conf["pupilsRadius"] = int(pupil_radius)
            applied["pupilsRadius"] = int(pupil_radius)

    wfc_conf = mutable_system_conf.get("wfc")
    wfc_geometry = derive_specula_wfc_display_geometry(param_mapping)
    if isinstance(wfc_conf, dict) and wfc_geometry is not None:
        display_grid_size = int(wfc_geometry["displayGridSize"])
        wfc_conf["displayGridSize"] = display_grid_size
        applied["displayGridSize"] = display_grid_size

    psf_component_conf = mutable_system_conf.get("psf")
    psf_geometry = derive_specula_psf_geometry(param_mapping)
    if isinstance(psf_component_conf, dict) and psf_geometry is not None:
        psf_component_conf["width"] = int(psf_geometry["width"])
        psf_component_conf["height"] = int(psf_geometry["height"])
        applied["psfWidth"] = int(psf_geometry["width"])
        applied["psfHeight"] = int(psf_geometry["height"])

    return applied or None


def _initialize_specula_runtime(specula_module, *, device_idx: int, precision: int) -> None:
    """Initialize SPECULA before importing data-object or processing modules.

    SPECULA's BaseTimeObj module imports global runtime state from the package
    root at module import time. That means `specula.init(...)` must happen
    before importing most SPECULA submodules, or those cached globals remain
    `None` and later constructors fail. If SPECULA submodules were already
    imported by the current process, patch their cached globals as well.
    """

    specula_module.init(device_idx, precision=precision)

    base_time_obj = sys.modules.get("specula.base_time_obj")
    if base_time_obj is not None:
        base_time_obj.global_precision = specula_module.global_precision
        base_time_obj.default_target_device_idx = specula_module.default_target_device_idx
        base_time_obj.default_target_device = specula_module.default_target_device
        base_time_obj.cpu_float_dtype_list = specula_module.cpu_float_dtype_list
        base_time_obj.gpu_float_dtype_list = specula_module.gpu_float_dtype_list
        base_time_obj.cpu_complex_dtype_list = specula_module.cpu_complex_dtype_list
        base_time_obj.gpu_complex_dtype_list = specula_module.gpu_complex_dtype_list


def _load_specula_bindings(*, device_idx: int, precision: int) -> SimpleNamespace:
    try:
        import specula
    except Exception as exc:
        raise ImportError(
            "SPECULA support requires the optional 'specula' package. "
            "Install SPECULA or add its repository root to PYTHONPATH before "
            f"constructing SPECULA-backed pyRTC components. Original error: {exc}"
        ) from exc

    _initialize_specula_runtime(specula, device_idx=device_idx, precision=precision)

    try:
        cpuArray = importlib.import_module("specula").cpuArray
        BaseValue = importlib.import_module("specula.base_value").BaseValue
        IFunc = importlib.import_module("specula.data_objects.ifunc").IFunc
        Pupilstop = importlib.import_module("specula.data_objects.pupilstop").Pupilstop
        SimulParams = importlib.import_module("specula.data_objects.simul_params").SimulParams
        Source = importlib.import_module("specula.data_objects.source").Source
        AtmoEvolution = importlib.import_module("specula.processing_objects.atmo_evolution").AtmoEvolution
        AtmoPropagation = importlib.import_module("specula.processing_objects.atmo_propagation").AtmoPropagation
        CCD = importlib.import_module("specula.processing_objects.ccd").CCD
        DM = importlib.import_module("specula.processing_objects.dm").DM
        ModulatedPyramid = importlib.import_module("specula.processing_objects.modulated_pyramid").ModulatedPyramid
        PSF = importlib.import_module("specula.processing_objects.psf").PSF
    except Exception as exc:
        raise ImportError(
            "SPECULA initialized but pyRTC could not import the required SPECULA "
            f"submodules for the bridge. Original error: {exc}"
        ) from exc

    return SimpleNamespace(
        specula=specula,
        cpuArray=cpuArray,
        BaseValue=BaseValue,
        IFunc=IFunc,
        Pupilstop=Pupilstop,
        SimulParams=SimulParams,
        Source=Source,
        AtmoEvolution=AtmoEvolution,
        AtmoPropagation=AtmoPropagation,
        CCD=CCD,
        DM=DM,
        ModulatedPyramid=ModulatedPyramid,
        PSF=PSF,
    )


def _as_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"SPECULA {name} must be a mapping/dict, got {type(value).__name__}")
    return dict(value)


def _default_wfc_layout(num_actuators: int) -> np.ndarray:
    """Return a centered, approximately circular boolean layout for display-only DMs."""

    if num_actuators < 1:
        raise ValueError("num_actuators must be positive")

    side = int(np.ceil(np.sqrt(float(num_actuators))))
    if side % 2 == 0:
        side += 1

    yy, xx = np.indices((side, side), dtype=np.float32)
    center = 0.5 * (side - 1)
    distances = (xx - center) ** 2 + (yy - center) ** 2
    selected = np.argsort(distances, axis=None)[:num_actuators]

    layout = np.zeros((side, side), dtype=bool)
    layout.flat[selected] = True
    return layout


def _specula_dm_layout(dm: Any, num_actuators: int) -> np.ndarray:
    """Infer a pyRTC display layout from a SPECULA DM, with a safe fallback.

    If SPECULA exposes a binary mask whose active cell count matches the pyRTC
    actuator count, use it directly. Otherwise fall back to the centered
    synthetic layout so `wfc2D` is always available even for modal bases.
    """

    ifunc_obj = getattr(dm, "ifunc_obj", None)
    mask = getattr(ifunc_obj, "mask_inf_func", None)
    if mask is not None:
        mask = np.asarray(mask) > 0
        if int(np.count_nonzero(mask)) == int(num_actuators):
            return mask
    return _default_wfc_layout(int(num_actuators))


def _square_zonal_layout(n_act: int) -> np.ndarray:
    if n_act < 1:
        raise ValueError("SPECULA zonal square geometry requires n_act >= 1")
    return np.ones((n_act, n_act), dtype=bool)


def _circular_ring_counts(n_act: int) -> np.ndarray:
    if n_act < 1:
        raise ValueError("SPECULA circular geometry requires n_act >= 1")
    n_act_radius = int(np.ceil((n_act + 1) / 2.0)) if n_act % 2 == 0 else int(np.ceil(n_act / 2.0))
    counts = np.arange(n_act_radius, dtype=int) * 6
    counts[0] = 1
    return counts


def _circular_zonal_display_mapping(n_act: int, angle_offset: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ring_counts = _circular_ring_counts(n_act)
    display_size = int(n_act + len(ring_counts) - 1)
    center = 0.5 * (display_size - 1)
    radial_step = 1.0 if len(ring_counts) == 1 else center / float(len(ring_counts) - 1)

    rows = []
    cols = []
    layout = np.zeros((display_size, display_size), dtype=bool)

    for ring_index, count in enumerate(ring_counts):
        if ring_index == 0:
            row = int(round(center))
            col = int(round(center))
            rows.append(row)
            cols.append(col)
            layout[row, col] = True
            continue

        for angle_index in range(int(count)):
            angle_deg = 360.0 / float(count) * float(angle_index) + float(angle_offset)
            angle_rad = np.deg2rad(angle_deg)
            x = center + radial_step * ring_index * np.cos(angle_rad)
            y = center + radial_step * ring_index * np.sin(angle_rad)
            row = int(np.rint(y))
            col = int(np.rint(x))
            rows.append(row)
            cols.append(col)
            layout[row, col] = True

    if int(np.count_nonzero(layout)) != int(np.sum(ring_counts)):
        raise ValueError("SPECULA circular actuator display mapping produced duplicate grid positions")

    return layout, np.asarray(rows, dtype=np.intp), np.asarray(cols, dtype=np.intp)


def _square_zonal_display_mapping(n_act: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    layout = _square_zonal_layout(n_act)
    rows, cols = np.indices(layout.shape, dtype=np.intp)
    return layout, rows.reshape(-1), cols.reshape(-1)


def _square_actuator_support_mask(n_act: int, obsratio: float = 0.0) -> np.ndarray:
    if n_act < 1:
        raise ValueError("square actuator support requires n_act >= 1")
    if n_act <= 2:
        return np.ones((n_act, n_act), dtype=bool)
    center = 0.5 * (float(n_act) - 1.0)
    radius = max(0.5 * (float(n_act) - 1.0), 0.5)
    inner_radius = radius * float(obsratio)
    yy, xx = np.indices((n_act, n_act), dtype=np.float32)
    rr = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)
    return np.logical_and(rr <= radius, rr >= inner_radius)


def _zonal_actuator_count(dm_conf: Mapping[str, Any]) -> int:
    n_act = int(dm_conf.get("n_act", 0))
    if n_act < 1:
        raise ValueError("SPECULA zonal DM configuration requires n_act >= 1")

    geom = str(dm_conf.get("geom", "square")).lower()
    circ_geom = bool(dm_conf.get("circ_geom", geom == "circular"))
    if geom == "square" and not circ_geom:
        return n_act * n_act

    n_act_radius = int(np.ceil((n_act + 1) / 2.0)) if n_act % 2 == 0 else int(np.ceil(n_act / 2.0))
    ring_counts = np.arange(n_act_radius, dtype=int) * 6
    ring_counts[0] = 1
    return int(np.sum(ring_counts))


class SPECULASystemContext:
    """Build and own the shared SPECULA objects used by soft-RTC components."""

    def __init__(
        self,
        resource_conf: Mapping[str, Any],
        system_conf: Mapping[str, Any] | None = None,
        *,
        simul_params=None,
        source=None,
        pupilstop=None,
        atmo=None,
        prop=None,
        pyramid=None,
        detector=None,
        psf=None,
        dm=None,
        seeing=None,
        wind_speed=None,
        wind_direction=None,
        command=None,
    ) -> None:
        self.resource_conf = dict(resource_conf)
        self.system_conf = {} if system_conf is None else dict(system_conf)
        self.param_file_path = self.resource_conf.get("paramFile") if isinstance(self.resource_conf.get("paramFile"), str) else None
        self.param = self._load_param_mapping(self.resource_conf.get("param"), self.param_file_path)
        self.components: dict[str, Any] = {}
        self._lock = threading.RLock()

        init_conf = _as_mapping(self.param.get("speculaInit"), name="speculaInit")
        self.device_idx = int(init_conf.get("device_idx", -1))
        self.precision = int(init_conf.get("precision", 1))
        self._bindings = _load_specula_bindings(device_idx=self.device_idx, precision=self.precision)

        self.simul_params = simul_params or self._build_simul_params()
        self.source = source or self._build_source()
        self.pupilstop = pupilstop or self._build_pupilstop()
        self.seeing = seeing or self._build_signal_value("seeing", self._signal_conf().get("seeing", 0.8))
        self.wind_speed = wind_speed or self._build_signal_value("wind_speed", self._signal_conf().get("wind_speed", [0.0]))
        self.wind_direction = wind_direction or self._build_signal_value("wind_direction", self._signal_conf().get("wind_direction", [0.0]))
        self.command = command or self._build_initial_command()
        self.atmo = atmo or self._build_atmo()
        self.dm = dm or self._build_dm()
        (
            self.dm_num_actuators,
            self.dm_layout,
            self.dm_display_rows,
            self.dm_display_cols,
            self.modal_to_command,
        ) = self._build_dm_mapping()
        self.prop = prop or self._build_propagation()
        self.pyramid = pyramid or self._build_pyramid()
        self.detector = detector or self._build_detector()
        self.psf = psf or self._build_psf()

        self._cached_psf_frame: np.ndarray | None = None
        self._cached_psf_model: np.ndarray | None = None
        self._cached_psf_strehl = 0.0
        self._cached_psf_tiptilt = 0.0

        self._wire_objects()
        self._setup_objects()

        self.current_time_t = 0
        self.step_index = 0
        self.dt_seconds = float(getattr(self.simul_params, "time_step", 0.001))
        self.dt_t = int(round(self.dt_seconds * 1e9))
        self.atmosphere_enabled = False

        if bool(setFromConfig(self.resource_conf, "useAtmosphere", True)):
            self.addAtmosphere()
        else:
            self.removeAtmosphere()

        self._seed_generation_times(0)

    def register_component(self, section_name: str, component: Any) -> None:
        self.components[str(section_name)] = component

    def get_component(self, section_name: str) -> Any:
        return self.components.get(str(section_name))

    def addAtmosphere(self) -> None:
        with self._lock:
            self.prop.inputs["atmo_layer_list"].set(self.atmo.outputs["layer_list"])
            self._refresh_propagation_setup()
            self.atmosphere_enabled = True

    def removeAtmosphere(self) -> None:
        with self._lock:
            self.prop.inputs["atmo_layer_list"].set([])
            self._refresh_propagation_setup()
            self.atmosphere_enabled = False

    def set_signal_value(self, signal_name: str, value: Any) -> None:
        with self._lock:
            signal = getattr(self, signal_name)
            array_value = np.asarray(value, dtype=np.float32)
            if array_value.ndim == 0:
                array_value = array_value.reshape(1)
            signal.set_value(array_value)
            signal.generation_time = self.scheduled_time_t()

    def update_atmo_parameter(self, name: str, value: Any) -> None:
        with self._lock:
            atmo_conf = _as_mapping(self.param.get("atmo"), name="atmo")
            atmo_conf[name] = value
            self.param["atmo"] = atmo_conf
            self._rebuild_atmosphere_locked()

    def _rebuild_atmosphere_locked(self) -> None:
        was_enabled = self.atmosphere_enabled
        self.atmo = self._build_atmo()
        self.atmo.inputs["seeing"].set(self.seeing)
        self.atmo.inputs["wind_speed"].set(self.wind_speed)
        self.atmo.inputs["wind_direction"].set(self.wind_direction)
        self.atmo.setup()
        self.prop.inputs["atmo_layer_list"].set(self.atmo.outputs["layer_list"] if was_enabled else [])
        self._refresh_propagation_setup()
        self.atmosphere_enabled = was_enabled

    def scheduled_time_t(self) -> int:
        return int(self.step_index * self.dt_t)

    def set_dm_command(self, command_vector: np.ndarray) -> None:
        with self._lock:
            command_vector = np.asarray(command_vector, dtype=np.float32).reshape(-1)
            self.command.set_value(command_vector)
            self.command.generation_time = self.scheduled_time_t()

    def capture_wfs(self) -> np.ndarray:
        with self._lock:
            time_t = self.scheduled_time_t()
            self._seed_generation_times(time_t)

            if self.atmosphere_enabled:
                self._advance(self.atmo, time_t)
            self._advance(self.dm, time_t)
            self._advance(self.prop, time_t)
            self._advance(self.pyramid, time_t)
            self._advance(self.detector, time_t)
            if self.psf is not None:
                self._advance(self.psf, time_t)
                self._refresh_psf_cache()

            self.current_time_t = time_t
            self.step_index += 1

            return np.asarray(
                self._bindings.cpuArray(self.detector.outputs["out_pixels"].pixels),
                dtype=np.uint16,
            )

    def capture_psf(self) -> tuple[np.ndarray, np.ndarray, float, float]:
        with self._lock:
            if self.psf is None:
                raise RuntimeError("SPECULA PSF camera is not configured")
            if self._cached_psf_frame is None:
                time_t = self.scheduled_time_t()
                self._seed_generation_times(time_t)
                if self.atmosphere_enabled:
                    self._advance(self.atmo, time_t)
                self._advance(self.dm, time_t)
                self._advance(self.prop, time_t)
                self._advance(self.psf, time_t)
                self.current_time_t = time_t
                self.step_index += 1
                self._refresh_psf_cache()

            return (
                np.array(self._cached_psf_frame, copy=True),
                np.array(self._cached_psf_model if self._cached_psf_model is not None else self._cached_psf_frame, copy=True),
                float(self._cached_psf_strehl),
                float(self._cached_psf_tiptilt),
            )

    def _advance(self, obj, time_t: int) -> None:
        if obj.check_ready(time_t):
            obj.trigger()
            obj.post_trigger()

    def _refresh_propagation_setup(self) -> None:
        """Refresh AtmoPropagation caches after atmosphere topology changes.

        SPECULA's AtmoPropagation caches layer-dependent interpolators and other
        topology metadata during setup(). When pyRTC toggles the atmosphere on
        or off after build time, we need to rebuild those caches explicitly or
        the newly attached atmospheric layers will not participate correctly in
        propagation.
        """

        self.prop.get_all_inputs()
        self.prop.atmo_layer_list = self.prop.local_inputs.get("atmo_layer_list") or []
        self.prop.common_layer_list = self.prop.local_inputs.get("common_layer_list") or []
        self.prop.nAtmoLayers = len(self.prop.atmo_layer_list)

        if len(self.prop.atmo_layer_list) + len(self.prop.common_layer_list) < 1:
            raise ValueError("At least one propagation layer must be set")

        self.prop.shiftXY_cond = {
            layer: np.any(layer.shiftXYinPixel)
            for layer in self.prop.atmo_layer_list + self.prop.common_layer_list
        }
        self.prop.magnification_list = {
            layer: max(layer.magnification, 1.0)
            for layer in self.prop.atmo_layer_list + self.prop.common_layer_list
        }

        self.prop._block_size = {}
        for layer in self.prop.atmo_layer_list + self.prop.common_layer_list:
            for div in [5, 4, 3, 2]:
                if layer.size[0] % div == 0:
                    self.prop._block_size[layer] = div
                    break

        self.prop.setup_interpolators()
        if getattr(self.prop, "doFresnel", False):
            self.prop.doFresnel_setup()

    def _refresh_psf_cache(self) -> None:
        if self.psf is None:
            return

        current_psf = np.asarray(
            self._bindings.cpuArray(self.psf.outputs["out_psf"].value),
            dtype=np.float64,
        )
        reference = getattr(getattr(self.psf, "ref", None), "i", None)
        if reference is None:
            reference = current_psf
        reference = np.asarray(self._bindings.cpuArray(reference), dtype=np.float64)

        ref_peak = float(np.max(reference)) if reference.size else 1.0
        if ref_peak <= 0.0:
            ref_peak = 1.0

        scaled_current = np.clip(current_psf / ref_peak, 0.0, None)
        scaled_reference = np.clip(reference / ref_peak, 0.0, None)
        scaled_current = np.clip(scaled_current * np.iinfo(np.uint16).max, 0, np.iinfo(np.uint16).max).astype(np.uint16)
        scaled_reference = np.clip(scaled_reference * np.iinfo(np.uint16).max, 0, np.iinfo(np.uint16).max).astype(np.uint16)

        self._cached_psf_frame = scaled_current
        self._cached_psf_model = scaled_reference

        sr_value = getattr(self.psf.outputs["out_sr"], "value", 0.0)
        sr_array = np.asarray(self._bindings.cpuArray(sr_value), dtype=np.float64).reshape(-1)
        self._cached_psf_strehl = float(sr_array[0]) if sr_array.size else 0.0
        self._cached_psf_tiptilt = 0.0

    def _seed_generation_times(self, time_t: int) -> None:
        for value in (self.seeing, self.wind_speed, self.wind_direction):
            value.generation_time = time_t
        if getattr(self.command, "generation_time", -1) < 0:
            self.command.generation_time = time_t
        if getattr(self.pupilstop, "generation_time", -1) < 0:
            self.pupilstop.generation_time = time_t

    def _load_param_mapping(self, param: Any, param_file: Any = None) -> dict[str, Any]:
        if param is None and isinstance(param_file, str) and param_file:
            param = read_yaml_file(param_file)
        return _as_mapping(param, name="param")

    def _resolve_path(self, path_value: str) -> str:
        if os.path.isabs(path_value):
            return path_value
        if self.param_file_path:
            return os.path.abspath(os.path.join(os.path.dirname(self.param_file_path), path_value))
        return os.path.abspath(path_value)

    def _main_conf(self) -> dict[str, Any]:
        return _as_mapping(self.param.get("main"), name="main")

    def _signal_conf(self) -> dict[str, Any]:
        return _as_mapping(self.param.get("signals"), name="signals")

    def _simul_object_kwargs(self, section_name: str) -> dict[str, Any]:
        return _as_mapping(self.param.get(section_name), name=section_name)

    def _build_simul_params(self):
        main_conf = {
            "root_dir": "./data/",
            "total_time": 0.01,
            "time_step": 0.001,
            **self._main_conf(),
        }
        root_dir = str(main_conf.get("root_dir", "./data/"))
        main_conf["root_dir"] = self._resolve_path(root_dir)
        os.makedirs(main_conf["root_dir"], exist_ok=True)
        return self._bindings.SimulParams(**main_conf)

    def _build_source(self):
        source_conf = self._simul_object_kwargs("source")
        return self._bindings.Source(
            target_device_idx=self.device_idx,
            precision=self.precision,
            **source_conf,
        )

    def _build_pupilstop(self):
        pupilstop_conf = self._simul_object_kwargs("pupilstop")
        return self._bindings.Pupilstop(
            self.simul_params,
            target_device_idx=self.device_idx,
            precision=self.precision,
            **pupilstop_conf,
        )

    def _build_signal_value(self, _name: str, value: Any):
        value = np.asarray(value, dtype=np.float32)
        if value.ndim == 0:
            value = value.reshape(1)
        signal_value = self._bindings.BaseValue(
            value=value,
            target_device_idx=self.device_idx,
            precision=self.precision,
        )
        signal_value.generation_time = 0
        return signal_value

    def _build_initial_command(self):
        dm_conf = self._simul_object_kwargs("dm")
        dm_type = str(dm_conf.get("type_str", "")).lower()
        if dm_type == "zonal":
            command_size = _zonal_actuator_count(dm_conf)
        else:
            raise ValueError(
                "SPECULAInterface requires dm.type_str='zonal' so pyRTC can send actuator commands "
                "through the zonal DM while using a separate modal basis as M2C."
            )
        command = self._bindings.BaseValue(
            value=np.zeros(command_size, dtype=np.float32),
            target_device_idx=self.device_idx,
            precision=self.precision,
        )
        command.generation_time = 0
        return command

    def _build_atmo(self):
        atmo_conf = self._simul_object_kwargs("atmo")
        data_dir = atmo_conf.get("data_dir")
        if not data_dir:
            data_dir = os.path.join(str(self.simul_params.root_dir), "phasescreens")
        atmo_conf["data_dir"] = self._resolve_path(str(data_dir))
        os.makedirs(atmo_conf["data_dir"], exist_ok=True)
        return self._bindings.AtmoEvolution(
            self.simul_params,
            target_device_idx=self.device_idx,
            precision=self.precision,
            **atmo_conf,
        )

    def _build_dm(self):
        dm_conf = self._simul_object_kwargs("dm")
        dm_type = str(dm_conf.get("type_str", "")).lower()

        if dm_type == "zonal":
            ifunc_signature = inspect.signature(self._bindings.IFunc)
            ifunc_supported = set(ifunc_signature.parameters)
            requested_geom = str(dm_conf.get("geom", "square")).lower()
            if "geom" not in ifunc_supported and requested_geom != "square":
                raise ValueError(
                    "The installed SPECULA version does not support the IFunc 'geom' parameter. "
                    f"Only square zonal geometry is supported with this SPECULA build, got geom={requested_geom!r}."
                )

            ifunc_kwargs = {
                "type_str": "zonal",
                "mask": getattr(self.pupilstop, "A", None),
                "npixels": int(getattr(self.simul_params, "pixel_pupil")),
                "n_act": int(dm_conf.get("n_act", 0)),
                "geom": requested_geom,
                "obsratio": dm_conf.get("obsratio", 0.0),
                "diaratio": dm_conf.get("diaratio", 1.0),
                "circ_geom": bool(dm_conf.get("circ_geom", requested_geom == "circular")),
                "angle_offset": dm_conf.get("angle_offset", 0.0),
                "do_mech_coupling": bool(dm_conf.get("do_mech_coupling", False)),
                "coupling_coeffs": dm_conf.get("coupling_coeffs", [0.31, 0.05]),
                "do_slaving": bool(dm_conf.get("do_slaving", False)),
                "slaving_thr": dm_conf.get("slaving_thr", 0.1),
                "target_device_idx": self.device_idx,
                "precision": self.precision,
            }
            if ifunc_kwargs["n_act"] < 1:
                raise ValueError("SPECULA zonal DM configuration requires n_act >= 1")

            ifunc_kwargs = {
                key: value
                for key, value in ifunc_kwargs.items()
                if key in ifunc_supported
            }

            ifunc_obj = self._bindings.IFunc(**ifunc_kwargs)
            dm_kwargs = {
                "height": dm_conf.get("height", 0.0),
                "ifunc": ifunc_obj,
                "pupilstop": self.pupilstop,
                "sign": dm_conf.get("sign", -1),
                "target_device_idx": self.device_idx,
                "precision": self.precision,
            }
            return self._bindings.DM(self.simul_params, **dm_kwargs)

        return self._bindings.DM(
            self.simul_params,
            target_device_idx=self.device_idx,
            precision=self.precision,
            **dm_conf,
        )

    def _build_dm_mapping(self) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        dm_conf = self._simul_object_kwargs("dm")
        dm_type = str(dm_conf.get("type_str", "")).lower()
        if dm_type != "zonal":
            raise ValueError(
                "SPECULAInterface requires dm.type_str='zonal' so pyRTC can send actuator commands "
                "and apply a modal M2C basis on top of them."
            )

        zonal_ifunc = np.asarray(
            self._bindings.cpuArray(self.dm.ifunc_obj.influence_function),
            dtype=np.float32,
        )
        num_actuators = int(zonal_ifunc.shape[0])

        geom = str(dm_conf.get("geom", "square")).lower()
        circ_geom = bool(dm_conf.get("circ_geom", geom == "circular"))
        n_act = int(dm_conf.get("n_act", 0))
        if geom == "square" and not circ_geom:
            layout, display_rows, display_cols = _square_zonal_display_mapping(n_act)
            if int(np.count_nonzero(layout)) != num_actuators:
                raise ValueError(
                    f"SPECULA zonal square geometry n_act={n_act} implies {int(np.count_nonzero(layout))} actuators, "
                    f"but the SPECULA DM built {num_actuators}."
                )
        else:
            layout, display_rows, display_cols = _circular_zonal_display_mapping(n_act, dm_conf.get("angle_offset", 0.0))
            if int(np.count_nonzero(layout)) != num_actuators:
                raise ValueError(
                    f"SPECULA circular geometry n_act={n_act} implies {int(np.count_nonzero(layout))} display sites, "
                    f"but the SPECULA DM built {num_actuators}."
                )

        wfc_conf = dict(self.system_conf.get("wfc", {}))
        num_modes = int(wfc_conf.get("numModes", num_actuators))
        basis_conf = _as_mapping(self.param.get("basis"), name="basis")
        basis_type = str(basis_conf.get("type_str", "zernike"))
        basis_kwargs = {
            "type_str": basis_type,
            "mask": self.dm.ifunc_obj.mask_inf_func,
            "npixels": int(self.dm.ifunc_obj.mask_inf_func.shape[0]),
            "nmodes": num_modes,
            "obsratio": basis_conf.get("obsratio", dm_conf.get("obsratio", 0.0)),
            "diaratio": basis_conf.get("diaratio", 1.0),
            "target_device_idx": self.device_idx,
            "precision": self.precision,
        }
        modal_ifunc_obj = self._bindings.IFunc(**basis_kwargs)
        modal_ifunc = np.asarray(
            self._bindings.cpuArray(modal_ifunc_obj.influence_function),
            dtype=np.float32,
        )
        modal_to_command = np.linalg.pinv(zonal_ifunc.T) @ modal_ifunc.T

        if geom == "square" and not circ_geom:
            actuator_support = _square_actuator_support_mask(n_act, dm_conf.get("obsratio", 0.0)).reshape(-1)
            if actuator_support.size != modal_to_command.shape[0]:
                raise ValueError(
                    "SPECULA square actuator support mask size does not match the modal-to-command row count"
                )
            modal_to_command[~actuator_support, :] = 0.0

        return (
            num_actuators,
            layout.astype(bool),
            display_rows.astype(np.intp),
            display_cols.astype(np.intp),
            modal_to_command.astype(np.float32),
        )

    def _build_propagation(self):
        propagation_conf = self._simul_object_kwargs("propagation")
        return self._bindings.AtmoPropagation(
            self.simul_params,
            source_dict={"on_axis_source": self.source},
            target_device_idx=self.device_idx,
            precision=self.precision,
            **propagation_conf,
        )

    def _build_psf(self):
        psf_conf = self._simul_object_kwargs("psf")
        if not psf_conf:
            return None
        psf_conf = dict(psf_conf)
        psf_conf.setdefault("verbose", False)
        return self._bindings.PSF(
            self.simul_params,
            target_device_idx=self.device_idx,
            precision=self.precision,
            **psf_conf,
        )

    def _build_pyramid(self):
        pyramid_conf = self._simul_object_kwargs("pyramid")
        return self._bindings.ModulatedPyramid(
            self.simul_params,
            target_device_idx=self.device_idx,
            precision=self.precision,
            **pyramid_conf,
        )

    def _build_detector(self):
        detector_conf = self._simul_object_kwargs("detector")
        return self._bindings.CCD(
            self.simul_params,
            target_device_idx=self.device_idx,
            precision=self.precision,
            **detector_conf,
        )

    def _wire_objects(self) -> None:
        self.atmo.inputs["seeing"].set(self.seeing)
        self.atmo.inputs["wind_speed"].set(self.wind_speed)
        self.atmo.inputs["wind_direction"].set(self.wind_direction)

        self.dm.inputs["in_command"].set(self.command)

        self.prop.inputs["common_layer_list"].set([self.pupilstop, self.dm.outputs["out_layer"]])
        self.prop.inputs["atmo_layer_list"].set([])

        self.pyramid.inputs["in_ef"].set(self.prop.outputs["out_on_axis_source_ef"])
        self.detector.inputs["in_i"].set(self.pyramid.outputs["out_i"])
        if self.psf is not None:
            self.psf.inputs["in_ef"].set(self.prop.outputs["out_on_axis_source_ef"])

    def _setup_objects(self) -> None:
        objects = [self.atmo, self.dm, self.prop, self.pyramid, self.detector]
        if self.psf is not None:
            objects.append(self.psf)
        for obj in objects:
            obj.setup()


def _unwrap_specula_context(resource):
    return getattr(resource, "context", resource)


class SPECULAWFSensor(WavefrontSensor):
    """Wavefront-sensor wrapper around a SPECULA pyramid plus detector chain."""

    def __init__(self, wfs_conf, context) -> None:
        self.context = _unwrap_specula_context(context)
        super().__init__(wfs_conf)
        if self.section_name:
            self.context.register_component(self.section_name, self)

    def expose(self):
        self.data = self.context.capture_wfs()
        super().expose()

    def addAtmosphere(self):
        self.context.addAtmosphere()

    def removeAtmosphere(self):
        self.context.removeAtmosphere()


class SPECULAWFCorrector(WavefrontCorrector):
    """Wavefront-corrector wrapper that pushes pyRTC commands into SPECULA."""

    def __init__(self, corrector_conf, context) -> None:
        self.context = _unwrap_specula_context(context)
        normalized_conf = dict(corrector_conf)
        normalized_conf["numActuators"] = int(self.context.dm_num_actuators)
        normalized_conf["numModes"] = int(self.context.modal_to_command.shape[1])
        super().__init__(normalized_conf)
        self.layout = self.context.dm_layout.astype(bool)
        self.display_rows = np.asarray(self.context.dm_display_rows, dtype=np.intp)
        self.display_cols = np.asarray(self.context.dm_display_cols, dtype=np.intp)
        self.correctionVector2D = ImageSHM(self.output_stream_name("wfc2D"), self.layout.shape, np.float32, gpuDevice=self.gpuDevice, consumer=False)
        self.register_output_stream("wfc2D", self.correctionVector2D, source_streams=["wfc"], lineage_source="wfc")
        self.correctionVector2D_template = np.zeros(self.layout.shape, dtype=np.float32)
        self.write_stream("wfc2D", self.correctionVector2D_template, source_streams=["wfc"], lineage_source="wfc")
        self.setM2C(self.context.modal_to_command)
        if self.section_name:
            self.context.register_component(self.section_name, self)

    def sendToHardware(self):
        super().sendToHardware()
        if isinstance(self.correctionVector2D, ImageSHM):
            self.correctionVector2D_template.fill(0)
            self.correctionVector2D_template[self.display_rows, self.display_cols] = self.currentShape - self.flat
            self.write_stream("wfc2D", self.correctionVector2D_template, source_streams=["wfc"], lineage_source="wfc")
        self.context.set_dm_command(self.currentShape.astype(np.float32, copy=False))


class SPECULAScienceCamera(ScienceCamera):
    """Science-camera wrapper backed by SPECULA's PSF processing object."""

    def __init__(self, science_conf, context) -> None:
        self.context = _unwrap_specula_context(context)
        super().__init__(science_conf)
        if self.section_name:
            self.context.register_component(self.section_name, self)

    def expose(self):
        frame, model, strehl, tiptilt = self.context.capture_psf()
        self.data = frame.astype(self.imageRawDType, copy=False)
        if not np.any(self.model):
            self.setModelPSF(model.astype(self.psfLongDtype, copy=False))
        super().expose()
        self.write_stream("strehl", np.array([strehl], dtype=float), source_streams=["psfShort"], lineage_source="psfShort")
        self.write_stream("tiptilt", np.array([tiptilt], dtype=float), source_streams=["psfShort"], lineage_source="psfShort")

    def integrate(self):
        super().integrate()
        if np.any(self.model):
            self.computeStrehl(median_filter_size=1, gaussian_sigma=0)


class SPECULAInterface(pyRTCComponent):
    """Manager-visible provider for SPECULA-backed soft-RTC components."""

    @staticmethod
    def sync_system_config(system_conf: Mapping[str, Any]) -> dict[str, Any] | None:
        return sync_specula_pywfs_config(system_conf)

    @staticmethod
    def gui_runtime_parameters(conf: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        provider_conf = _mapping_or_none(conf) or {}
        param_mapping = _load_specula_param_mapping(provider_conf=provider_conf) or {}
        signal_conf = _mapping_or_none(param_mapping.get("signals")) or {}
        atmo_conf = _mapping_or_none(param_mapping.get("atmo")) or {}

        def _signal_default(name: str, default: Any) -> Any:
            value = signal_conf.get(name, default)
            if isinstance(value, (list, tuple)):
                return [float(item) for item in value]
            return float(value)

        return [
            {
                "name": "useAtmosphere",
                "type": "bool",
                "description": "Enable or disable atmospheric layers in the active SPECULA propagation chain.",
                "default": bool(setFromConfig(provider_conf, "useAtmosphere", False)),
                "persist": True,
            },
            {
                "name": "seeing",
                "type": "float",
                "description": "Live seeing value fed into the SPECULA atmosphere model.",
                "default": _signal_default("seeing", 0.8),
            },
            {
                "name": "wind_speed",
                "type": "list[float]",
                "description": "Live wind-speed vector fed into the SPECULA atmosphere model.",
                "default": _signal_default("wind_speed", [0.0]),
            },
            {
                "name": "wind_direction",
                "type": "list[float]",
                "description": "Live wind-direction vector fed into the SPECULA atmosphere model.",
                "default": _signal_default("wind_direction", [0.0]),
            },
            {
                "name": "atmo_L0",
                "type": "list[float]",
                "description": "Outer-scale values. Updating this rebuilds the SPECULA atmosphere object in place.",
                "default": [float(item) for item in atmo_conf.get("L0", [25.0])],
            },
            {
                "name": "atmo_heights",
                "type": "list[float]",
                "description": "Atmospheric layer heights. Updating this rebuilds the SPECULA atmosphere object in place.",
                "default": [float(item) for item in atmo_conf.get("heights", [0.0])],
            },
            {
                "name": "atmo_Cn2",
                "type": "list[float]",
                "description": "Turbulence-strength fractions per layer. Updating this rebuilds the SPECULA atmosphere object in place.",
                "default": [float(item) for item in atmo_conf.get("Cn2", [1.0])],
            },
            {
                "name": "atmo_pixel_phasescreens",
                "type": "int",
                "description": "Phase-screen pixel size. Updating this rebuilds the SPECULA atmosphere object in place.",
                "default": int(atmo_conf.get("pixel_phasescreens", 256)),
            },
        ]

    def __init__(
        self,
        conf,
        param=None,
        *,
        simul_params=None,
        source=None,
        pupilstop=None,
        atmo=None,
        prop=None,
        pyramid=None,
        detector=None,
        psf=None,
        dm=None,
        seeing=None,
        wind_speed=None,
        wind_direction=None,
        command=None,
    ) -> None:
        self.conf = conf
        self.logger = get_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self._standalone_mode = isinstance(conf, Mapping) and all(key in conf for key in ("wfs", "wfc"))

        if self._standalone_mode:
            self.system_conf = dict(conf)
            sync_specula_pywfs_config(self.system_conf, param=param)
            resource_conf = {"param": param} if param is not None else {}
            self.context = SPECULASystemContext(
                resource_conf,
                self.system_conf,
                simul_params=simul_params,
                source=source,
                pupilstop=pupilstop,
                atmo=atmo,
                prop=prop,
                pyramid=pyramid,
                detector=detector,
                psf=psf,
                dm=dm,
                seeing=seeing,
                wind_speed=wind_speed,
                wind_direction=wind_direction,
                command=command,
            )
            self._useAtmosphere = True
            self.wfcSection = "wfc"
            self.wfsInterface = SPECULAWFSensor(self.system_conf["wfs"], self.context)
            self.dmInterface = SPECULAWFCorrector(self.system_conf["wfc"], self.context)
            self.psfInterface = SPECULAScienceCamera(self.system_conf["psf"], self.context) if "psf" in self.system_conf else None
            return

        self.system_conf = conf.get("_systemConfig", conf)
        synced = sync_specula_pywfs_config(self.system_conf, provider_conf=conf)
        if synced:
            self.logger.info("Synchronized SPECULA PyWFS geometry into pyRTC config: %s", synced)
        self.context = SPECULASystemContext(
            conf,
            self.system_conf,
            simul_params=simul_params,
            source=source,
            pupilstop=pupilstop,
            atmo=atmo,
            prop=prop,
            pyramid=pyramid,
            detector=detector,
            psf=psf,
            dm=dm,
            seeing=seeing,
            wind_speed=wind_speed,
            wind_direction=wind_direction,
            command=command,
        )
        self._useAtmosphere = bool(setFromConfig(conf, "useAtmosphere", False))
        self.wfcSection = str(setFromConfig(conf, "wfcSection", "wfc"))
        super().__init__(conf)
        if self.section_name:
            self.context.register_component(self.section_name, self)
        if self.useAtmosphere:
            self.addAtmosphere()
        else:
            self.removeAtmosphere()

    def addAtmosphere(self):
        self.context.addAtmosphere()
        self._useAtmosphere = True
        self.logger.info("Enabled SPECULA atmosphere")

    def removeAtmosphere(self):
        self.context.removeAtmosphere()
        self._useAtmosphere = False
        self.logger.info("Disabled SPECULA atmosphere")

    @property
    def useAtmosphere(self) -> bool:
        return bool(getattr(self, "_useAtmosphere", False))

    @useAtmosphere.setter
    def useAtmosphere(self, value: Any) -> None:
        enabled = bool(value)
        if enabled:
            self.addAtmosphere()
        else:
            self.removeAtmosphere()

    @property
    def seeing(self) -> float:
        values = np.asarray(self.context.seeing.value, dtype=np.float32).reshape(-1)
        return float(values[0]) if values.size else 0.0

    @seeing.setter
    def seeing(self, value: Any) -> None:
        self.context.set_signal_value("seeing", float(value))

    @property
    def wind_speed(self) -> list[float]:
        return np.asarray(self.context.wind_speed.value, dtype=np.float32).reshape(-1).astype(float).tolist()

    @wind_speed.setter
    def wind_speed(self, value: Any) -> None:
        self.context.set_signal_value("wind_speed", value)

    @property
    def wind_direction(self) -> list[float]:
        return np.asarray(self.context.wind_direction.value, dtype=np.float32).reshape(-1).astype(float).tolist()

    @wind_direction.setter
    def wind_direction(self, value: Any) -> None:
        self.context.set_signal_value("wind_direction", value)

    @property
    def atmo_L0(self) -> list[float]:
        return [float(item) for item in _as_mapping(self.context.param.get("atmo"), name="atmo").get("L0", [])]

    @atmo_L0.setter
    def atmo_L0(self, value: Any) -> None:
        self.context.update_atmo_parameter("L0", [float(item) for item in value])

    @property
    def atmo_heights(self) -> list[float]:
        return [float(item) for item in _as_mapping(self.context.param.get("atmo"), name="atmo").get("heights", [])]

    @atmo_heights.setter
    def atmo_heights(self, value: Any) -> None:
        self.context.update_atmo_parameter("heights", [float(item) for item in value])

    @property
    def atmo_Cn2(self) -> list[float]:
        return [float(item) for item in _as_mapping(self.context.param.get("atmo"), name="atmo").get("Cn2", [])]

    @atmo_Cn2.setter
    def atmo_Cn2(self, value: Any) -> None:
        self.context.update_atmo_parameter("Cn2", [float(item) for item in value])

    @property
    def atmo_pixel_phasescreens(self) -> int:
        return int(_as_mapping(self.context.param.get("atmo"), name="atmo").get("pixel_phasescreens", 0))

    @atmo_pixel_phasescreens.setter
    def atmo_pixel_phasescreens(self, value: Any) -> None:
        self.context.update_atmo_parameter("pixel_phasescreens", int(value))

    def get_hardware(self):
        if self._standalone_mode:
            return self.wfsInterface, self.dmInterface, self.psfInterface
        return self.context.get_component("wfs"), self.context.get_component(self.wfcSection), self.context.get_component("psf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the SPECULA provider component.")
    parser.add_argument("-c", "--config", required=True, help="Path to the pyRTC config file")
    parser.add_argument("-p", "--port", required=True, help="Port for communication")
    args = parser.parse_args()

    conf = read_yaml_file(args.config)

    pid = os.getpid()
    set_affinity((conf["wfs"]["affinity"]) % os.cpu_count())
    decrease_nice(pid)

    sim = SPECULAInterface(conf=conf)

    listener = Listener(sim, port=int(args.port))
    while listener.running:
        listener.listen()
        time.sleep(1e-3)