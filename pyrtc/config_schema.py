"""System-level configuration validation for pyrtc.

This module builds on the component-local validators in ``pyrtc.utils`` and
adds a whole-system view of a pyrtc configuration file. It is intentionally
lightweight: validation is implemented in Python without introducing a schema
framework dependency, and the result is a normalized config mapping that future
manager and GUI layers can reuse.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from pyrtc.component_descriptors import (
    describe_component_class,
    get_component_descriptor,
    list_component_sections,
    validate_config_with_descriptor,
)
from pyrtc.utils import (
    ConfigValidationError,
    read_yaml_file,
    validate_loop_config,
    validate_wfc_config,
    validate_wfs_config,
)


REQUIRED_COMPONENT_SECTIONS = ("wfs", "slopes", "loop", "wfc")
OPTIONAL_COMPONENT_SECTIONS = tuple(
    section for section in list_component_sections() if section not in REQUIRED_COMPONENT_SECTIONS
)
OPTIONAL_TOP_LEVEL_SECTIONS = ("manager", "streams", "metadata", "resources")
ALLOWED_MANAGER_MODES = {"soft-rtc", "hard-rtc"}
ALLOWED_RESTART_POLICIES = {"never", "on-failure", "always"}


def _resolve_class_symbol(class_name: str, class_file: str | None = None):
    from pyrtc.component_loading import resolve_class_symbol

    return resolve_class_symbol(class_name, class_file)


def _default_stream_aliases_for_section(
    section_name: str, section_conf: Mapping[str, Any]
) -> tuple[dict[str, str], dict[str, str]]:
    descriptor = get_component_descriptor(section_name)
    if descriptor is None:
        class_name = section_conf.get("class_name")
        class_file = section_conf.get("class_file")
        if isinstance(class_name, str) and class_name.strip():
            try:
                descriptor = describe_component_class(
                    _resolve_class_symbol(
                        class_name, class_file if isinstance(class_file, str) else None
                    )
                )
            except Exception:
                descriptor = None
    input_aliases = {}
    output_aliases = {}
    if descriptor is not None:
        input_aliases = {
            stream.name: stream.name for stream in descriptor.input_streams if stream.name != "*"
        }
        output_aliases = {
            stream.name: stream.name for stream in descriptor.output_streams if stream.name != "*"
        }
    return input_aliases, output_aliases


def _normalize_stream_alias_mapping(
    raw_mapping: Any, *, defaults: Mapping[str, str]
) -> dict[str, str]:
    normalized = dict(defaults)
    if not isinstance(raw_mapping, Mapping):
        return normalized
    for semantic_name, value in raw_mapping.items():
        if not isinstance(semantic_name, str) or not semantic_name.strip():
            continue
        if isinstance(value, str):
            shm_name = value.strip()
        elif isinstance(value, Mapping):
            shm_name = str(value.get("shm", value.get("name", semantic_name))).strip()
        else:
            continue
        if shm_name:
            normalized[str(semantic_name)] = shm_name
    return normalized


def _is_relative_path_string(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    if "://" in value:
        return False
    return not Path(value).is_absolute()


def _resolve_relative_path_fields(
    value: Any, *, base_dir: Path, parent_key: str | None = None
) -> Any:
    if isinstance(value, Mapping):
        resolved = {}
        for key, item in value.items():
            is_path_field = isinstance(key, str) and key.endswith(("_file", "_dir", "_path"))
            is_component_file_entry = parent_key == "component_files"
            if (is_path_field or is_component_file_entry) and _is_relative_path_string(item):
                resolved[key] = str((base_dir / item).resolve())
            else:
                resolved[key] = _resolve_relative_path_fields(
                    item, base_dir=base_dir, parent_key=str(key)
                )
        return resolved
    if isinstance(value, list):
        return [
            _resolve_relative_path_fields(item, base_dir=base_dir, parent_key=parent_key)
            for item in value
        ]
    return value


def _is_valid_manager_section(section_name: str, system_conf: Mapping[str, Any]) -> bool:
    if get_component_descriptor(section_name) is not None:
        return True
    return section_name in system_conf and section_name not in OPTIONAL_TOP_LEVEL_SECTIONS


def _require_mapping(conf: Any, component: str) -> Mapping[str, Any]:
    if not isinstance(conf, Mapping):
        raise ConfigValidationError(
            f"{component}: config must be a mapping/dict, got {type(conf).__name__}"
        )
    return conf


def _validate_optional_numeric(
    conf: Mapping[str, Any],
    key: str,
    component: str,
    *,
    minimum: float | None = None,
) -> None:
    if key not in conf:
        return
    value = conf[key]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigValidationError(
            f"{component}: '{key}' must be numeric, got {type(value).__name__}"
        )
    if minimum is not None and value < minimum:
        raise ConfigValidationError(f"{component}: '{key}' must be >= {minimum}, got {value}")


def _coerce_int(value: Any, component: str, key: str, *, minimum: int | None = None) -> int:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigValidationError(f"{component}: '{key}' must be an int-like numeric value")
    coerced = int(value)
    if coerced != value:
        raise ConfigValidationError(f"{component}: '{key}' must be an integer value, got {value}")
    if minimum is not None and coerced < minimum:
        raise ConfigValidationError(f"{component}: '{key}' must be >= {minimum}, got {coerced}")
    return coerced


def _validate_required_section_mappings(conf: Mapping[str, Any]) -> None:
    missing = [section for section in REQUIRED_COMPONENT_SECTIONS if section not in conf]
    if missing:
        raise ConfigValidationError(
            f"system: missing required top-level section(s): {', '.join(missing)}"
        )

    for section_name in (
        REQUIRED_COMPONENT_SECTIONS + OPTIONAL_COMPONENT_SECTIONS + OPTIONAL_TOP_LEVEL_SECTIONS
    ):
        if section_name in conf:
            _require_mapping(conf[section_name], section_name)


def _validate_functions_section(section_name: str, section_conf: Mapping[str, Any]) -> None:
    if "functions" not in section_conf:
        return

    functions = section_conf["functions"]
    if not isinstance(functions, list):
        raise ConfigValidationError(f"{section_name}: 'functions' must be a list of method names")

    descriptor = get_component_descriptor(section_name)
    if descriptor is None:
        return

    allowed_functions = set(descriptor.worker_functions)
    for function_name in functions:
        if not isinstance(function_name, str) or not function_name.strip():
            raise ConfigValidationError(
                f"{section_name}: all 'functions' entries must be non-empty strings"
            )

        if function_name in allowed_functions:
            continue

        component_cls = descriptor.component_class
        if not hasattr(component_cls, function_name):
            raise ConfigValidationError(
                f"{section_name}: function '{function_name}' is not defined on {component_cls.__name__}"
            )
        if not callable(getattr(component_cls, function_name)):
            raise ConfigValidationError(
                f"{section_name}: function '{function_name}' exists on {component_cls.__name__} but is not callable"
            )


def _validate_component_class_and_streams(
    section_name: str, section_conf: Mapping[str, Any]
) -> None:
    class_name = section_conf.get("class_name")
    class_file = section_conf.get("class_file")
    if not isinstance(class_name, str) or not class_name.strip():
        raise ConfigValidationError(f"{section_name}: 'class_name' must be a non-empty string")
    if class_file is not None and (not isinstance(class_file, str) or not class_file.strip()):
        raise ConfigValidationError(
            f"{section_name}: 'class_file' must be a non-empty string when provided"
        )
    try:
        component_class = _resolve_class_symbol(
            class_name, class_file if isinstance(class_file, str) else None
        )
    except Exception as exc:
        raise ConfigValidationError(
            f"{section_name}: unable to resolve class_name '{class_name}'"
        ) from exc

    descriptor = describe_component_class(component_class)
    for mapping_name, streams in (
        ("input_streams", descriptor.input_streams),
        ("output_streams", descriptor.output_streams),
    ):
        raw_mapping = section_conf.get(mapping_name, {})
        if raw_mapping is None:
            continue
        if not isinstance(raw_mapping, Mapping):
            raise ConfigValidationError(f"{section_name}: '{mapping_name}' must be a mapping")
        allowed = {stream.name for stream in streams if stream.name != "*"}
        for semantic_name, value in raw_mapping.items():
            if not isinstance(semantic_name, str) or not semantic_name.strip():
                raise ConfigValidationError(
                    f"{section_name}: '{mapping_name}' keys must be non-empty strings"
                )
            if allowed and semantic_name not in allowed:
                raise ConfigValidationError(
                    f"{section_name}: '{mapping_name}' key '{semantic_name}' is not a known stream role"
                )
            if isinstance(value, Mapping):
                shm_name = value.get("shm", value.get("name"))
            else:
                shm_name = value
            if not isinstance(shm_name, str) or not shm_name.strip():
                raise ConfigValidationError(
                    f"{section_name}: '{mapping_name}.{semantic_name}' must resolve to a non-empty SHM name"
                )


def _validate_slopes_config(conf: Any) -> None:
    component = "slopes"
    conf = _require_mapping(conf, component)

    required = ["type", "signal_type"]
    missing = [key for key in required if key not in conf]
    if missing:
        raise ConfigValidationError(
            f"{component}: missing required config key(s): {', '.join(missing)}"
        )

    slopes_type = conf["type"]
    if not isinstance(slopes_type, str) or not slopes_type.strip():
        raise ConfigValidationError(f"{component}: 'type' must be a non-empty string")
    slopes_type = slopes_type.lower()
    if slopes_type not in {"shwfs", "pywfs"}:
        raise ConfigValidationError(f"{component}: unsupported type '{conf['type']}'")

    signal_type = conf["signal_type"]
    if not isinstance(signal_type, str) or not signal_type.strip():
        raise ConfigValidationError(f"{component}: 'signal_type' must be a non-empty string")

    _validate_optional_numeric(conf, "image_noise", component, minimum=0.0)
    _validate_optional_numeric(conf, "central_obscuration_ratio", component, minimum=0.0)
    _validate_optional_numeric(conf, "ref_slope_count", component, minimum=1)

    if slopes_type == "shwfs":
        for key in ("sub_ap_spacing", "sub_ap_offset_x", "sub_ap_offset_y"):
            if key not in conf:
                raise ConfigValidationError(f"{component}: '{key}' is required for SHWFS")
        _validate_optional_numeric(conf, "sub_ap_spacing", component, minimum=1.0)
        _coerce_int(conf["sub_ap_offset_x"], component, "sub_ap_offset_x", minimum=0)
        _coerce_int(conf["sub_ap_offset_y"], component, "sub_ap_offset_y", minimum=0)

    if slopes_type == "pywfs" and "pupils" in conf:
        pupils = conf["pupils"]
        if not isinstance(pupils, list) or len(pupils) == 0:
            raise ConfigValidationError("slopes: 'pupils' must be a non-empty list when provided")
        for pupil in pupils:
            if not isinstance(pupil, str) or "," not in pupil:
                raise ConfigValidationError(
                    "slopes: pupil entries must be strings in the form 'x,y'"
                )
        if "pupils_radius" not in conf:
            raise ConfigValidationError(
                "slopes: 'pupils_radius' is required when 'pupils' is provided"
            )
        _coerce_int(conf["pupils_radius"], component, "pupils_radius", minimum=1)


def _validate_psf_config(conf: Any) -> None:
    component = "psf"
    conf = _require_mapping(conf, component)

    required = ["name", "width", "height", "dark_count", "integration"]
    missing = [key for key in required if key not in conf]
    if missing:
        raise ConfigValidationError(
            f"{component}: missing required config key(s): {', '.join(missing)}"
        )

    if not isinstance(conf["name"], str) or not conf["name"].strip():
        raise ConfigValidationError(f"{component}: 'name' must be a non-empty string")

    for key in ("width", "height", "dark_count", "integration"):
        _coerce_int(conf[key], component, key, minimum=1)


def _validate_telemetry_config(conf: Any) -> None:
    component = "telemetry"
    conf = _require_mapping(conf, component)

    if "data_dir" in conf and (
        not isinstance(conf["data_dir"], str) or not conf["data_dir"].strip()
    ):
        raise ConfigValidationError(
            "telemetry: 'data_dir' must be a non-empty string when provided"
        )
    if "streams" in conf:
        streams = conf["streams"]
        if not isinstance(streams, list) or not all(
            isinstance(item, str) and item.strip() for item in streams
        ):
            raise ConfigValidationError(
                "telemetry: 'streams' must be a list of non-empty stream names"
            )


def _validate_manager_config(conf: Any, *, system_conf: Mapping[str, Any]) -> None:
    component = "manager"
    conf = _require_mapping(conf, component)

    mode = conf.get("mode", "soft-rtc")
    if not isinstance(mode, str) or mode not in ALLOWED_MANAGER_MODES:
        raise ConfigValidationError(
            f"manager: 'mode' must be one of {sorted(ALLOWED_MANAGER_MODES)}"
        )

    if "restart_policy" in conf:
        policy = conf["restart_policy"]
        if not isinstance(policy, str) or policy not in ALLOWED_RESTART_POLICIES:
            raise ConfigValidationError(
                f"manager: 'restart_policy' must be one of {sorted(ALLOWED_RESTART_POLICIES)}"
            )

    if "component_restart_policies" in conf:
        restart_policies = _require_mapping(
            conf["component_restart_policies"], "manager.component_restart_policies"
        )
        for section_name, policy in restart_policies.items():
            if not _is_valid_manager_section(section_name, system_conf):
                raise ConfigValidationError(
                    f"manager.component_restart_policies: unknown component section '{section_name}'"
                )
            if not isinstance(policy, str) or policy not in ALLOWED_RESTART_POLICIES:
                raise ConfigValidationError(
                    f"manager.component_restart_policies: policy for '{section_name}' must be one of {sorted(ALLOWED_RESTART_POLICIES)}"
                )

    for key in ("health_check_interval", "heartbeat_timeout", "rpc_timeout"):
        _validate_optional_numeric(conf, key, component, minimum=1e-6)

    if "component_modes" in conf:
        component_modes = _require_mapping(conf["component_modes"], "manager.component_modes")
        for section_name, launch_mode in component_modes.items():
            if not _is_valid_manager_section(section_name, system_conf):
                raise ConfigValidationError(
                    f"manager.component_modes: unknown component section '{section_name}'"
                )
            if not isinstance(launch_mode, str) or launch_mode not in ALLOWED_MANAGER_MODES:
                raise ConfigValidationError(
                    f"manager.component_modes: mode for '{section_name}' must be one of {sorted(ALLOWED_MANAGER_MODES)}"
                )

    if "ports" in conf:
        ports = _require_mapping(conf["ports"], "manager.ports")
        for section_name, port in ports.items():
            if not _is_valid_manager_section(section_name, system_conf):
                raise ConfigValidationError(
                    f"manager.ports: unknown component section '{section_name}'"
                )
            _coerce_int(port, "manager.ports", section_name, minimum=1)

    for mapping_name in ("component_classes", "component_files"):
        if mapping_name in conf:
            mapping_value = _require_mapping(conf[mapping_name], f"manager.{mapping_name}")
            for section_name, target in mapping_value.items():
                if not _is_valid_manager_section(section_name, system_conf):
                    raise ConfigValidationError(
                        f"manager.{mapping_name}: unknown component section '{section_name}'"
                    )
                if not isinstance(target, str) or not target.strip():
                    raise ConfigValidationError(
                        f"manager.{mapping_name}: target for '{section_name}' must be a non-empty string"
                    )

    if "graph_layout" in conf:
        graph_layout = _require_mapping(conf["graph_layout"], "manager.graph_layout")
        positions = graph_layout.get("positions", {})
        if positions is not None:
            positions = _require_mapping(positions, "manager.graph_layout.positions")
            for section_name, value in positions.items():
                if not _is_valid_manager_section(str(section_name), system_conf):
                    raise ConfigValidationError(
                        f"manager.graph_layout.positions: unknown component section '{section_name}'"
                    )
                position_conf = _require_mapping(
                    value, f"manager.graph_layout.positions.{section_name}"
                )
                for axis in ("x", "y"):
                    if axis not in position_conf:
                        raise ConfigValidationError(
                            f"manager.graph_layout.positions.{section_name}: missing '{axis}'"
                        )
                    _validate_optional_numeric(
                        position_conf, axis, f"manager.graph_layout.positions.{section_name}"
                    )

    if "log_dir" in conf and (not isinstance(conf["log_dir"], str) or not conf["log_dir"].strip()):
        raise ConfigValidationError("manager: 'log_dir' must be a non-empty string when provided")
    if "log_file" in conf and (
        not isinstance(conf["log_file"], str) or not conf["log_file"].strip()
    ):
        raise ConfigValidationError("manager: 'log_file' must be a non-empty string when provided")


def _validate_resources_config(conf: Any) -> None:
    component = "resources"
    conf = _require_mapping(conf, component)
    for resource_name, resource_conf in conf.items():
        if not isinstance(resource_name, str) or not resource_name.strip():
            raise ConfigValidationError("resources: resource names must be non-empty strings")
        resource_conf = _require_mapping(resource_conf, f"resources.{resource_name}")
        class_name = resource_conf.get("class_name")
        class_file = resource_conf.get("class_file")
        if not isinstance(class_name, str) or not class_name.strip():
            raise ConfigValidationError(
                f"resources.{resource_name}: 'class_name' must be a non-empty string"
            )
        if class_file is not None and (not isinstance(class_file, str) or not class_file.strip()):
            raise ConfigValidationError(
                f"resources.{resource_name}: 'class_file' must be a non-empty string when provided"
            )
        try:
            _resolve_class_symbol(class_name, class_file if isinstance(class_file, str) else None)
        except Exception as exc:
            raise ConfigValidationError(
                f"resources.{resource_name}: unable to resolve class_name '{class_name}'"
            ) from exc


def _validate_component_resource_bindings(conf: Mapping[str, Any]) -> None:
    resources_conf = conf.get("resources", {}) if isinstance(conf.get("resources"), Mapping) else {}
    manager_conf = conf.get("manager", {}) if isinstance(conf.get("manager"), Mapping) else {}
    component_modes = (
        manager_conf.get("component_modes", {})
        if isinstance(manager_conf.get("component_modes"), Mapping)
        else {}
    )
    default_mode = str(manager_conf.get("mode", "soft-rtc"))
    component_sections = {
        section_name
        for section_name, section_conf in conf.items()
        if section_name not in OPTIONAL_TOP_LEVEL_SECTIONS and isinstance(section_conf, Mapping)
    }

    for section_name, section_conf in conf.items():
        if section_name in OPTIONAL_TOP_LEVEL_SECTIONS or not isinstance(section_conf, Mapping):
            continue
        resource_name = section_conf.get("resource")
        if resource_name is None:
            continue
        if not isinstance(resource_name, str) or not resource_name.strip():
            raise ConfigValidationError(
                f"{section_name}: 'resource' must be a non-empty string when provided"
            )
        if resource_name not in resources_conf and resource_name not in component_sections:
            raise ConfigValidationError(
                f"{section_name}: resource '{resource_name}' is not defined under top-level resources or as a component section"
            )
        effective_mode = str(component_modes.get(section_name, default_mode))
        if effective_mode != "soft-rtc":
            raise ConfigValidationError(
                f"{section_name}: resource-backed components are supported only in soft-rtc mode"
            )
        if resource_name in component_sections:
            provider_mode = str(component_modes.get(resource_name, default_mode))
            if provider_mode != "soft-rtc":
                raise ConfigValidationError(
                    f"{section_name}: resource provider component '{resource_name}' must run in soft-rtc mode"
                )


def _validate_streams_config(conf: Any, *, system_conf: Mapping[str, Any]) -> None:
    """Validate optional stream metadata overrides.

    The preferred terminology is ``output_component`` and ``input_components``.
    ``producer`` and ``consumers`` remain supported as backward-compatible
    aliases.
    """

    component = "streams"
    conf = _require_mapping(conf, component)

    for stream_name, stream_conf in conf.items():
        if not isinstance(stream_name, str) or not stream_name.strip():
            raise ConfigValidationError("streams: stream names must be non-empty strings")
        stream_conf = _require_mapping(stream_conf, f"streams.{stream_name}")
        if "shape" in stream_conf:
            shape = stream_conf["shape"]
            if not isinstance(shape, (list, tuple)) or len(shape) == 0:
                raise ConfigValidationError(
                    f"streams.{stream_name}: 'shape' must be a non-empty list/tuple"
                )
            for axis in shape:
                _coerce_int(axis, f"streams.{stream_name}", "shape axis", minimum=1)
        if "dtype" in stream_conf and (
            not isinstance(stream_conf["dtype"], str) or not stream_conf["dtype"].strip()
        ):
            raise ConfigValidationError(
                f"streams.{stream_name}: 'dtype' must be a non-empty string"
            )
        output_component = stream_conf.get("output_component", stream_conf.get("producer"))
        if output_component is not None and not _is_valid_manager_section(
            str(output_component), system_conf
        ):
            raise ConfigValidationError(
                f"streams.{stream_name}: 'output_component' must reference a known component section"
            )
        input_components = stream_conf.get("input_components", stream_conf.get("consumers"))
        if input_components is not None:
            if not isinstance(input_components, list):
                raise ConfigValidationError(
                    f"streams.{stream_name}: 'input_components' must be a list"
                )
            for input_component in input_components:
                if not _is_valid_manager_section(str(input_component), system_conf):
                    raise ConfigValidationError(
                        f"streams.{stream_name}: input component '{input_component}' is not a known component section"
                    )
        if "component_stream" in stream_conf:
            component_stream = stream_conf["component_stream"]
            if not isinstance(component_stream, str) or not component_stream.strip():
                raise ConfigValidationError(
                    f"streams.{stream_name}: 'component_stream' must be a non-empty semantic stream name"
                )


def _validate_metadata_config(conf: Any) -> None:
    component = "metadata"
    conf = _require_mapping(conf, component)
    for key in ("name", "description"):
        if key in conf and not isinstance(conf[key], str):
            raise ConfigValidationError(f"metadata: '{key}' must be a string when provided")
    if "tags" in conf:
        tags = conf["tags"]
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            raise ConfigValidationError("metadata: 'tags' must be a list of strings")


def _processed_wfs_shape(wfs_conf: Mapping[str, Any]) -> tuple[int, int]:
    width = _coerce_int(wfs_conf["width"], "wfs", "width", minimum=1)
    height = _coerce_int(wfs_conf["height"], "wfs", "height", minimum=1)
    downsample = _coerce_int(
        wfs_conf.get("downsample_factor", 0), "wfs", "downsample_factor", minimum=0
    )
    if downsample > 0:
        width //= downsample
        height //= downsample
    if width < 1 or height < 1:
        raise ConfigValidationError("wfs: processed image dimensions must remain positive")
    return width, height


def _validate_cross_component_consistency(conf: Mapping[str, Any]) -> None:
    wfs_conf = conf["wfs"]
    slopes_conf = conf["slopes"]
    loop_conf = conf["loop"]
    wfc_conf = conf["wfc"]

    num_modes = _coerce_int(wfc_conf["num_modes"], "wfc", "num_modes", minimum=1)
    dropped_modes = _coerce_int(
        loop_conf.get("num_dropped_modes", 0), "loop", "num_dropped_modes", minimum=0
    )
    if dropped_modes >= num_modes:
        raise ConfigValidationError(
            f"loop: 'num_dropped_modes' ({dropped_modes}) must be less than wfc.num_modes ({num_modes})"
        )

    slopes_type = str(slopes_conf["type"]).lower()
    if slopes_type == "shwfs":
        image_width, image_height = _processed_wfs_shape(wfs_conf)
        subap_spacing = float(slopes_conf["sub_ap_spacing"])
        if subap_spacing < 1:
            raise ConfigValidationError(
                f"slopes: 'sub_ap_spacing' must be >= 1, got {subap_spacing}"
            )
        num_regions = min(image_width, image_height) // subap_spacing
        if num_regions < 1:
            raise ConfigValidationError(
                "slopes: SHWFS geometry yields zero sub-apertures; adjust wfs dimensions or sub_ap_spacing"
            )

        if "num_modes" in wfs_conf:
            wfs_num_modes = _coerce_int(wfs_conf["num_modes"], "wfs", "num_modes", minimum=1)
            if wfs_num_modes != num_modes:
                raise ConfigValidationError(
                    f"wfs: 'num_modes' ({wfs_num_modes}) must match wfc.num_modes ({num_modes}) for a consistent control vector size"
                )

        streams_conf = conf.get("streams", {})
        if "signal" in streams_conf and "shape" in streams_conf["signal"]:
            expected_signal_size = 2 * num_regions**2
            actual_shape = tuple(int(axis) for axis in streams_conf["signal"]["shape"])
            if actual_shape != (expected_signal_size,):
                raise ConfigValidationError(
                    f"streams.signal: shape {actual_shape} does not match expected SHWFS signal shape {(expected_signal_size,)}"
                )
        if "signal_2d" in streams_conf and "shape" in streams_conf["signal_2d"]:
            expected_signal2d_shape = (2 * num_regions, num_regions)
            actual_shape = tuple(int(axis) for axis in streams_conf["signal_2d"]["shape"])
            if actual_shape != expected_signal2d_shape:
                raise ConfigValidationError(
                    f"streams.signal_2d: shape {actual_shape} does not match expected SHWFS signal_2d shape {expected_signal2d_shape}"
                )


def normalize_system_config(conf: Any) -> dict[str, Any]:
    """Return a normalized copy of a pyrtc system configuration.

    The first version keeps normalization intentionally conservative. It only
    injects defaults and copies nested mappings so future layers can rely on a
    stable shape without mutating the caller's object.
    """

    conf = deepcopy(dict(_require_mapping(conf, "system")))
    manager_conf = dict(_require_mapping(conf.get("manager", {}), "manager"))
    manager_conf.setdefault("mode", "soft-rtc")
    conf["manager"] = manager_conf
    conf.setdefault("metadata", {})
    conf.setdefault("streams", {})
    conf.setdefault("resources", {})

    component_classes = (
        manager_conf.get("component_classes", {})
        if isinstance(manager_conf.get("component_classes"), Mapping)
        else {}
    )
    component_files = (
        manager_conf.get("component_files", {})
        if isinstance(manager_conf.get("component_files"), Mapping)
        else {}
    )
    for section_name, section_conf in list(conf.items()):
        if section_name in OPTIONAL_TOP_LEVEL_SECTIONS or not isinstance(section_conf, Mapping):
            continue
        section_conf = dict(section_conf)
        class_name = section_conf.get("class_name")
        if not isinstance(class_name, str) or not class_name.strip():
            fallback_class = component_classes.get(section_name)
            if isinstance(fallback_class, str) and fallback_class.strip():
                section_conf["class_name"] = fallback_class
            else:
                descriptor = get_component_descriptor(section_name)
                if descriptor is not None:
                    section_conf["class_name"] = descriptor.class_path
        if "class_file" not in section_conf and isinstance(component_files.get(section_name), str):
            section_conf["class_file"] = component_files.get(section_name)
        default_inputs, default_outputs = _default_stream_aliases_for_section(
            section_name, section_conf
        )
        section_conf["input_streams"] = _normalize_stream_alias_mapping(
            section_conf.get("input_streams", {}), defaults=default_inputs
        )
        section_conf["output_streams"] = _normalize_stream_alias_mapping(
            section_conf.get("output_streams", {}), defaults=default_outputs
        )
        conf[section_name] = section_conf
    return conf


def validate_system_config(conf: Any, *, config_path: str | Path | None = None) -> dict[str, Any]:
    """Validate a whole pyrtc system config and return its normalized form."""

    normalized = normalize_system_config(conf)

    # Resolve relative paths (class_file, *File/*Dir fields) against the config
    # file's directory *before* any validation, so component classes referenced
    # by relative class_file entries resolve regardless of the caller's cwd.
    if config_path is not None:
        config_path = Path(config_path).resolve()
        normalized = _resolve_relative_path_fields(normalized, base_dir=config_path.parent)

    _validate_required_section_mappings(normalized)

    for section_name in list_component_sections():
        if section_name in normalized:
            try:
                validate_config_with_descriptor(section_name, normalized[section_name])
            except (TypeError, ValueError) as exc:
                raise ConfigValidationError(str(exc)) from exc

    validate_wfs_config(normalized["wfs"])
    _validate_slopes_config(normalized["slopes"])
    validate_loop_config(normalized["loop"])
    validate_wfc_config(normalized["wfc"])

    if "psf" in normalized:
        _validate_psf_config(normalized["psf"])
    if "telemetry" in normalized:
        _validate_telemetry_config(normalized["telemetry"])
    if "manager" in normalized:
        _validate_manager_config(normalized["manager"], system_conf=normalized)
    if "resources" in normalized:
        _validate_resources_config(normalized["resources"])
    if "streams" in normalized:
        _validate_streams_config(normalized["streams"], system_conf=normalized)
    if "metadata" in normalized:
        _validate_metadata_config(normalized["metadata"])

    for section_name in list_component_sections():
        if section_name in normalized:
            _validate_functions_section(section_name, normalized[section_name])

    for section_name, section_conf in normalized.items():
        if section_name not in OPTIONAL_TOP_LEVEL_SECTIONS and isinstance(section_conf, Mapping):
            _validate_component_class_and_streams(section_name, section_conf)

    _validate_component_resource_bindings(normalized)

    _validate_cross_component_consistency(normalized)

    if config_path is not None:
        normalized.setdefault("metadata", {})
        normalized["metadata"].setdefault("config_path", str(config_path))

    return normalized


def read_system_config(file_path: str | Path, *, validate: bool = True) -> dict[str, Any]:
    """Read a YAML config file and optionally validate it as a pyrtc system."""

    config_path = Path(file_path)
    conf = read_yaml_file(config_path)
    if validate:
        return validate_system_config(conf, config_path=config_path)
    return normalize_system_config(conf)
