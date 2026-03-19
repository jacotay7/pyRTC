"""Bridge layer between the Qt UI and the current RTCManager API."""

from __future__ import annotations

from ast import literal_eval
import importlib
import importlib.util
import inspect
from pathlib import Path
import subprocess
from typing import Any
import yaml

from pyRTC.Pipeline import DEFAULT_COMPONENT_ORDER, RTCManager
from pyRTC.component_descriptors import describe_component_class, get_component_descriptor, list_component_descriptors, list_component_sections
from pyRTC.config_schema import read_system_config

from .models import GraphEdgeModel, GraphNodeModel, GraphSnapshot


_CONFIG_ONLY_FIELDS = {
    "functions",
    "affinity",
    "gpuDevice",
    "type",
    "signalType",
}

_NON_COMPONENT_TOP_LEVEL_SECTIONS = {"manager", "streams", "metadata", "resources"}

_NON_ACTION_METHODS = {
    "start",
    "stop",
    "status",
    "read",
    "write",
    "listen",
    "launch",
    "validate",
    "refresh_health",
    "restart",
    "run",
    "getProperty",
    "setProperty",
    "shutdown",
    "get_hardware",
}


def _normalize_runtime_parameter_rows(raw_rows: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not isinstance(raw_rows, list):
        return rows
    for row in raw_rows:
        if not isinstance(row, dict):
            continue
        name = row.get("name")
        field_type = row.get("type")
        if not isinstance(name, str) or not name.strip() or not isinstance(field_type, str) or not field_type.strip():
            continue
        rows.append(
            {
                "name": name,
                "type": field_type,
                "required": bool(row.get("required", False)),
                "description": str(row.get("description", "")),
                "default": row.get("default"),
                "persist": bool(row.get("persist", False)),
            }
        )
    return rows


def _is_live_runtime_field(name: str) -> bool:
    if name in _CONFIG_ONLY_FIELDS:
        return False
    if name.endswith(("File", "Dir", "Path")):
        return False
    return True


def _ordered_sections(config: dict[str, Any]) -> list[str]:
    sections = [section for section in DEFAULT_COMPONENT_ORDER if section in config]
    for section in list_component_sections():
        if section in config and section not in sections:
            sections.append(section)
    for section in config:
        if section not in sections and section not in _NON_COMPONENT_TOP_LEVEL_SECTIONS:
            sections.append(section)
    return sections


def _infer_layout(index: int) -> tuple[float, float]:
    columns = 3
    col = index % columns
    row = index // columns
    return 80.0 + col * 260.0, 60.0 + row * 190.0


def _graph_layout_positions(config: dict[str, Any]) -> dict[str, dict[str, float]]:
    manager_conf = config.get("manager", {}) if isinstance(config.get("manager"), dict) else {}
    graph_layout = manager_conf.get("graphLayout", {}) if isinstance(manager_conf.get("graphLayout"), dict) else {}
    positions = graph_layout.get("positions", {}) if isinstance(graph_layout.get("positions"), dict) else {}
    normalized: dict[str, dict[str, float]] = {}
    for section_name, value in positions.items():
        if not isinstance(value, dict):
            continue
        try:
            normalized[str(section_name)] = {
                "x": float(value["x"]),
                "y": float(value["y"]),
            }
        except Exception:
            continue
    return normalized


def _import_component_class(class_path: str, component_file: str | None = None):
    if component_file:
        module_path = Path(component_file).expanduser().resolve()
        module_name = f"pyrtc_builder_{module_path.stem}_{abs(hash(str(module_path))) & 0xFFFFFFFF:x}"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load component module from '{module_path}'")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        attr_name = class_path.rsplit(".", 1)[-1]
        return getattr(module, attr_name)

    if "." not in class_path:
        for module_name in ("pyRTC.hardware", "pyRTC"):
            try:
                module = importlib.import_module(module_name)
                if hasattr(module, class_path):
                    return getattr(module, class_path)
            except Exception:
                continue
        raise ImportError(f"Unable to resolve component class '{class_path}'")

    module_name, attr_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def _default_value_for_field(field_descriptor) -> Any:
    if field_descriptor.default is not None:
        if isinstance(field_descriptor.default, list):
            return list(field_descriptor.default)
        return field_descriptor.default
    if field_descriptor.field_type == "int":
        return max(1, int(field_descriptor.minimum or 1))
    if field_descriptor.field_type == "float":
        return float(field_descriptor.minimum or 0.0)
    if field_descriptor.field_type == "bool":
        return False
    if field_descriptor.field_type in {"list[str]", "list[float]"}:
        return []
    if field_descriptor.field_type == "str | None":
        return None
    return ""


def _normalize_stream_alias_map(raw_mapping: Any) -> dict[str, str]:
    normalized: dict[str, str] = {}
    if not isinstance(raw_mapping, dict):
        return normalized
    for semantic_name, value in raw_mapping.items():
        if not isinstance(semantic_name, str):
            continue
        if isinstance(value, str):
            shm_name = value.strip()
        elif isinstance(value, dict):
            shm_name = str(value.get("shm", value.get("name", semantic_name))).strip()
        else:
            continue
        if shm_name:
            normalized[semantic_name] = shm_name
    return normalized


def _component_stream_name(config: dict[str, Any], section_name: str, direction: str, stream_name: str) -> str:
    section_conf = config.get(section_name, {}) if isinstance(config.get(section_name), dict) else {}
    mapping_name = "inputStreams" if direction == "input" else "outputStreams"
    aliases = _normalize_stream_alias_map(section_conf.get(mapping_name, {}))
    return aliases.get(stream_name, stream_name)


def _parse_float_list(raw_value: Any) -> list[float]:
    if isinstance(raw_value, list):
        return [float(item) for item in raw_value]

    text = str(raw_value).strip()
    if not text:
        return []
    if text[0] == "[" and text[-1] == "]":
        text = text[1:-1]
    if not text.strip():
        return []

    values = []
    for token in text.split(","):
        normalized = token.strip().lower()
        if normalized in {"inf", "+inf", "infinity", "+infinity"}:
            values.append(float("inf"))
        elif normalized in {"-inf", "-infinity"}:
            values.append(float("-inf"))
        else:
            values.append(float(normalized))
    return values


def _coerce_runtime_value(raw_value: Any, field_type: str) -> Any:
    if raw_value is None:
        return None
    if field_type == "int":
        return int(raw_value)
    if field_type == "float":
        return float(raw_value)
    if field_type == "bool":
        if isinstance(raw_value, bool):
            return raw_value
        normalized = str(raw_value).strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"Unable to parse boolean value from {raw_value!r}")
    if field_type == "str | None":
        text = str(raw_value)
        return None if text == "" else text
    if field_type == "dict[str,str]":
        if isinstance(raw_value, dict):
            return {str(key): str(value) for key, value in raw_value.items()}
        parsed = yaml.safe_load(str(raw_value))
        if not isinstance(parsed, dict):
            raise ValueError("Mapping fields must parse to a dict")
        return {str(key): str(value) for key, value in parsed.items()}
    if field_type in {"list[str]", "list[float]"}:
        if isinstance(raw_value, list):
            value = raw_value
        else:
            if field_type == "list[float]":
                return _parse_float_list(raw_value)
            value = literal_eval(str(raw_value))
        if field_type == "list[float]":
            return _parse_float_list(value)
        if not isinstance(value, list):
            raise ValueError("List fields must parse to a list")
        return [str(item) for item in value]
    return raw_value if not isinstance(raw_value, str) else str(raw_value)


class ManagerAdapter:
    def __init__(self) -> None:
        self.manager: RTCManager | None = None
        self.config: dict[str, Any] | None = None
        self.config_path: str | None = None
        self._last_status: dict[str, Any] | None = None

    def is_loaded(self) -> bool:
        return self.config is not None

    def initialize_empty_config(self, *, mode: str = "soft-rtc") -> dict[str, Any]:
        self.config = {
            "manager": {
                "mode": str(mode),
            },
            "metadata": {},
            "streams": {},
        }
        self.config_path = None
        self.manager = RTCManager.from_config(self.config, config_path=self.config_path, mode=mode)
        self._last_status = self.manager.status()
        return self.config

    def load_config(self, config_path: str, *, mode: str | None = None) -> dict[str, Any]:
        normalized = read_system_config(config_path, validate=False)
        self.config = dict(normalized)
        self.config_path = str(Path(config_path).expanduser().resolve())
        self.manager = RTCManager.from_config(self.config, config_path=self.config_path, mode=mode)
        self._last_status = self.manager.status()
        return self.config

    def ensure_manager(self) -> RTCManager:
        if self.manager is None:
            raise RuntimeError("No config is loaded")
        return self.manager

    def selected_mode(self) -> str:
        manager = self.ensure_manager()
        return str(manager.config.get("manager", {}).get("mode", "soft-rtc"))

    def set_mode(self, mode: str) -> dict[str, Any]:
        if not self.config:
            raise RuntimeError("No config is loaded")
        manager = self.ensure_manager()
        if manager.state in {"running", "degraded", "failed", "starting", "stopping"}:
            raise RuntimeError("Stop the system before changing manager mode")
        self.manager = RTCManager.from_config(self.config, config_path=self.config_path, mode=mode)
        self._last_status = self.manager.status()
        return self._last_status

    def _rebuild_manager(self) -> None:
        if not self.config:
            return
        mode = None if self.manager is None else self.selected_mode()
        self.manager = RTCManager.from_config(self.config, config_path=self.config_path, mode=mode)
        self._last_status = self.manager.status()

    def save_config(self, path: str | None = None) -> str:
        if not self.config:
            raise RuntimeError("No config is loaded")
        destination = path or self.config_path
        if not destination:
            raise RuntimeError("No destination path is available")
        resolved = str(Path(destination).expanduser().resolve())
        Path(resolved).write_text(yaml.safe_dump(self.config, sort_keys=False), encoding="utf-8")
        self.config_path = resolved
        return resolved

    def get_component_position(self, section_name: str, *, default_index: int | None = None) -> tuple[float, float]:
        if not self.config:
            if default_index is None:
                return 80.0, 60.0
            return _infer_layout(default_index)
        positions = _graph_layout_positions(self.config)
        if section_name in positions:
            return positions[section_name]["x"], positions[section_name]["y"]
        if default_index is None:
            return 80.0, 60.0
        return _infer_layout(default_index)

    def set_component_position(self, section_name: str, x: float, y: float) -> None:
        if not self.config:
            raise RuntimeError("No config is loaded")
        manager_conf = self.config.setdefault("manager", {})
        graph_layout = manager_conf.setdefault("graphLayout", {})
        positions = graph_layout.setdefault("positions", {})
        positions[section_name] = {"x": float(x), "y": float(y)}
        if self.manager is not None:
            self.manager.config.setdefault("manager", {}).setdefault("graphLayout", {}).setdefault("positions", {})[section_name] = {
                "x": float(x),
                "y": float(y),
            }

    def clear_component_position(self, section_name: str) -> None:
        if not self.config:
            raise RuntimeError("No config is loaded")
        manager_conf = self.config.get("manager", {})
        if isinstance(manager_conf, dict):
            graph_layout = manager_conf.get("graphLayout", {})
            if isinstance(graph_layout, dict):
                positions = graph_layout.get("positions", {})
                if isinstance(positions, dict):
                    positions.pop(section_name, None)
        if self.manager is not None:
            manager_conf = self.manager.config.get("manager", {})
            if isinstance(manager_conf, dict):
                graph_layout = manager_conf.get("graphLayout", {})
                if isinstance(graph_layout, dict):
                    positions = graph_layout.get("positions", {})
                    if isinstance(positions, dict):
                        positions.pop(section_name, None)

    def available_component_templates(self) -> list[dict[str, str]]:
        return [
            {
                "section_name": descriptor.section_name,
                "label": f"{descriptor.section_name} ({descriptor.class_name})",
                "class_path": descriptor.class_path,
            }
            for descriptor in list_component_descriptors()
        ]

    def _descriptor_for_section(self, section_name: str):
        descriptor = get_component_descriptor(section_name)
        if descriptor is not None:
            return descriptor
        if not self.config:
            return None
        section_conf = self.config.get(section_name, {}) if isinstance(self.config.get(section_name), dict) else {}
        class_path = section_conf.get("className") or section_conf.get("name")
        component_file = section_conf.get("classFile")
        if not class_path:
            return None
        try:
            component_class = _import_component_class(str(class_path), str(component_file) if component_file else None)
        except Exception:
            return None
        return describe_component_class(component_class)

    def add_component(
        self,
        section_name: str,
        *,
        template_section: str | None = None,
        class_path: str | None = None,
        component_file: str | None = None,
        config_overrides: dict[str, Any] | None = None,
        input_streams: dict[str, str] | None = None,
        output_streams: dict[str, str] | None = None,
    ) -> None:
        if not self.config:
            raise RuntimeError("No config is loaded")
        normalized = str(section_name).strip()
        if not normalized:
            raise ValueError("Section name must be non-empty")
        if normalized in self.config:
            raise ValueError(f"Component '{normalized}' already exists")
        if normalized in _NON_COMPONENT_TOP_LEVEL_SECTIONS:
            raise ValueError(f"'{normalized}' is reserved")

        descriptor = get_component_descriptor(template_section or normalized)
        resolved_class_path = class_path
        if descriptor is None:
            if not class_path:
                raise ValueError("A template section or class path is required")
            component_class = _import_component_class(class_path, component_file)
            descriptor = describe_component_class(component_class)
        elif resolved_class_path is None:
            resolved_class_path = descriptor.class_path

        component_conf: dict[str, Any] = {}
        for field_descriptor in descriptor.all_fields:
            component_conf[field_descriptor.name] = _default_value_for_field(field_descriptor)
        if "name" in descriptor.field_map and not component_conf.get("name"):
            component_conf["name"] = descriptor.class_name
        component_conf["className"] = resolved_class_path or descriptor.class_path
        if component_file:
            component_conf["classFile"] = str(Path(component_file).expanduser().resolve())
        descriptor_for_aliases = describe_component_class(_import_component_class(component_conf["className"], component_conf.get("classFile")))
        component_conf["inputStreams"] = {stream.name: stream.name for stream in descriptor_for_aliases.input_streams if stream.name != "*"}
        component_conf["outputStreams"] = {stream.name: stream.name for stream in descriptor_for_aliases.output_streams if stream.name != "*"}
        if config_overrides:
            coerced_overrides = {}
            for name, value in config_overrides.items():
                field_descriptor = descriptor.field_map.get(name)
                if field_descriptor is None:
                    coerced_overrides[name] = value
                else:
                    coerced_overrides[name] = _coerce_runtime_value(value, field_descriptor.field_type)
            component_conf.update(coerced_overrides)
        if input_streams is not None:
            component_conf["inputStreams"] = dict(input_streams)
        if output_streams is not None:
            component_conf["outputStreams"] = dict(output_streams)
        self.config[normalized] = component_conf

        manager_conf = self.config.setdefault("manager", {})
        component_classes = manager_conf.setdefault("componentClasses", {})
        component_classes[normalized] = resolved_class_path or descriptor.class_path
        if component_file:
            manager_conf.setdefault("componentFiles", {})[normalized] = str(Path(component_file).expanduser().resolve())
        self._rebuild_manager()

    def remove_component(self, section_name: str) -> None:
        if not self.config or section_name not in self.config:
            raise KeyError(section_name)
        self.config.pop(section_name, None)
        manager_conf = self.config.get("manager", {}) if isinstance(self.config.get("manager"), dict) else {}
        for mapping_name in ("componentClasses", "componentFiles", "componentModes", "ports", "componentRestartPolicies"):
            mapping = manager_conf.get(mapping_name)
            if isinstance(mapping, dict):
                mapping.pop(section_name, None)
        streams_conf = self.config.get("streams")
        if isinstance(streams_conf, dict):
            stale_names = []
            for stream_name, stream_conf in streams_conf.items():
                if not isinstance(stream_conf, dict):
                    continue
                output_component = stream_conf.get("outputComponent", stream_conf.get("producer"))
                input_components = stream_conf.get("inputComponents", stream_conf.get("consumers", []))
                if output_component == section_name:
                    stale_names.append(stream_name)
                    continue
                if isinstance(input_components, list) and section_name in input_components:
                    remaining = [item for item in input_components if item != section_name]
                    if remaining:
                        stream_conf["inputComponents"] = remaining
                    else:
                        stale_names.append(stream_name)
            for stream_name in stale_names:
                streams_conf.pop(stream_name, None)
        self._rebuild_manager()

    def add_connection(
        self,
        stream_name: str,
        *,
        output_component: str,
        input_components: list[str],
        component_stream: str | None = None,
        output_role: str | None = None,
        input_role: str | None = None,
    ) -> None:
        if not self.config:
            raise RuntimeError("No config is loaded")
        normalized_name = str(stream_name).strip()
        if not normalized_name:
            raise ValueError("Stream name must be non-empty")
        if output_component not in self.config:
            raise KeyError(output_component)
        for input_component in input_components:
            if input_component not in self.config:
                raise KeyError(input_component)
        resolved_output_role = (output_role or component_stream or normalized_name).strip()
        resolved_input_role = (input_role or component_stream or normalized_name).strip()
        producer_conf = self.config.setdefault(output_component, {})
        producer_outputs = producer_conf.setdefault("outputStreams", {})
        producer_outputs[resolved_output_role] = normalized_name
        for input_component in input_components:
            consumer_conf = self.config.setdefault(input_component, {})
            consumer_inputs = consumer_conf.setdefault("inputStreams", {})
            consumer_inputs[resolved_input_role] = normalized_name
        streams_conf = self.config.setdefault("streams", {})
        existing = streams_conf.get(normalized_name, {}) if isinstance(streams_conf.get(normalized_name), dict) else {}
        current_inputs = list(existing.get("inputComponents", []))
        for input_component in input_components:
            if input_component not in current_inputs:
                current_inputs.append(input_component)
        payload = {
            "outputComponent": output_component,
            "inputComponents": current_inputs,
        }
        payload["componentStream"] = resolved_output_role
        payload["inputRole"] = resolved_input_role
        streams_conf[normalized_name] = payload
        self._rebuild_manager()

    def remove_connection(self, stream_name: str) -> None:
        if not self.config:
            raise RuntimeError("No config is loaded")
        streams_conf = self.config.get("streams")
        if not isinstance(streams_conf, dict) or stream_name not in streams_conf:
            raise KeyError(stream_name)
        stream_conf = streams_conf.get(stream_name, {})
        if isinstance(stream_conf, dict):
            output_component = stream_conf.get("outputComponent", stream_conf.get("producer"))
            component_stream = stream_conf.get("componentStream", stream_name)
            input_role = stream_conf.get("inputRole", component_stream)
            if isinstance(output_component, str) and output_component in self.config:
                producer_outputs = self.config[output_component].get("outputStreams", {})
                if isinstance(producer_outputs, dict):
                    producer_outputs.pop(component_stream, None)
            input_components = stream_conf.get("inputComponents", stream_conf.get("consumers", []))
            if isinstance(input_components, list):
                for input_component in input_components:
                    if input_component in self.config:
                        consumer_inputs = self.config[input_component].get("inputStreams", {})
                        if isinstance(consumer_inputs, dict):
                            consumer_inputs.pop(input_role, None)
        streams_conf.pop(stream_name, None)
        self._rebuild_manager()

    def connection_names(self) -> list[str]:
        if not self.config:
            return []
        streams_conf = self.config.get("streams")
        if not isinstance(streams_conf, dict):
            return []
        return sorted(streams_conf)

    def runtime_sections(self) -> set[str]:
        manager = self.ensure_manager()
        return set(manager.runtimes) if manager.runtimes else set(section for section in _ordered_sections(manager.config))

    def status(self) -> dict[str, Any]:
        manager = self.ensure_manager()
        self._last_status = manager.status()
        return self._last_status

    def validate(self) -> dict[str, Any]:
        manager = self.ensure_manager()
        self.config = manager.validate()
        self._last_status = manager.status()
        return self.config

    def build(self) -> dict[str, Any]:
        manager = self.ensure_manager()
        self._last_status = manager.build()
        return self._last_status

    def start(self) -> dict[str, Any]:
        manager = self.ensure_manager()
        manager.start()
        self._last_status = manager.status()
        return self._last_status

    def stop(self) -> dict[str, Any]:
        manager = self.ensure_manager()
        manager.stop()
        self._last_status = manager.status()
        return self._last_status

    def reset(self) -> dict[str, Any]:
        manager = self.ensure_manager()
        if manager.state in {"running", "degraded", "failed"}:
            manager.stop()
        manager.start()
        self._last_status = manager.status()
        return self._last_status

    def refresh(self) -> dict[str, Any]:
        manager = self.ensure_manager()
        self._last_status = manager.refresh_health()
        return self._last_status

    def restart_component(self, section_name: str) -> dict[str, Any]:
        manager = self.ensure_manager()
        if section_name not in manager.runtimes:
            raise KeyError(section_name)
        manager.restart_component(section_name)
        self._last_status = manager.status()
        return self._last_status

    def start_component(self, section_name: str) -> dict[str, Any]:
        manager = self.ensure_manager()
        manager.start_component(section_name)
        self._last_status = manager.status()
        return self._last_status

    def stop_component(self, section_name: str) -> dict[str, Any]:
        manager = self.ensure_manager()
        manager.stop_component(section_name)
        self._last_status = manager.status()
        return self._last_status

    def get_component_functions(self, section_name: str) -> list[dict[str, Any]]:
        if not self.config or section_name not in self.config:
            raise KeyError(section_name)

        manager = self.ensure_manager()
        descriptor = self._descriptor_for_section(section_name)
        runtime = manager.runtimes.get(section_name) if manager.runtimes else None
        component_class = getattr(runtime, "component_class", None)
        if component_class is None:
            component_class = descriptor.component_class if descriptor is not None else None
        if component_class is None:
            return []

        excluded = set(_NON_ACTION_METHODS)
        if descriptor is not None:
            excluded.update(descriptor.worker_functions)

        target = manager.get_component(section_name) if runtime is not None else None
        enabled = target is not None
        rows = []
        for name, member in inspect.getmembers(component_class):
            if name.startswith("_") or name in excluded:
                continue
            if not callable(member):
                continue
            try:
                signature = inspect.signature(member)
            except (TypeError, ValueError):
                continue
            parameters = list(signature.parameters.values())
            required = [
                parameter
                for index, parameter in enumerate(parameters)
                if not (index == 0 and parameter.name == "self")
                and parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
                and parameter.default is inspect._empty
            ]
            if required:
                continue
            rows.append(
                {
                    "name": name,
                    "description": inspect.getdoc(member) or "",
                    "enabled": enabled,
                }
            )
        rows.sort(key=lambda row: row["name"])
        return rows

    def _runtime_parameter_definitions(self, section_name: str) -> list[dict[str, Any]]:
        if not self.config or section_name not in self.config:
            return []

        manager = self.ensure_manager()
        descriptor = self._descriptor_for_section(section_name)
        runtime = manager.runtimes.get(section_name) if manager.runtimes else None
        component_class = getattr(runtime, "component_class", None)
        if component_class is None:
            component_class = descriptor.component_class if descriptor is not None else None
        if component_class is None:
            return []

        hook = getattr(component_class, "gui_runtime_parameters", None)
        if not callable(hook):
            return []

        try:
            return _normalize_runtime_parameter_rows(hook(self.config[section_name]))
        except Exception:
            return []

    def run_component_function(self, section_name: str, function_name: str) -> Any:
        manager = self.ensure_manager()
        if section_name not in manager.runtimes:
            raise KeyError(section_name)

        runtime = manager.runtimes[section_name]
        target = manager.get_component(section_name)
        if target is None:
            raise RuntimeError(f"Component '{section_name}' is not active")

        available = {row["name"] for row in self.get_component_functions(section_name)}
        if function_name not in available:
            raise ValueError(f"Function '{function_name}' is not available for component '{section_name}'")

        if runtime.mode == "hard-rtc" and hasattr(target, "run"):
            result = target.run(function_name)
            if result == -1 and getattr(target, "lastError", None):
                raise RuntimeError(target.lastError)
            self._last_status = manager.status()
            return result

        result = getattr(target, function_name)()
        self._last_status = manager.status()
        return result

    def get_component_parameters(self, section_name: str) -> list[dict[str, Any]]:
        if not self.config or section_name not in self.config:
            raise KeyError(section_name)
        descriptor = self._descriptor_for_section(section_name)
        conf = self.config[section_name]
        rows = []
        rows.extend(
            [
                {
                    "name": "className",
                    "type": "str",
                    "required": True,
                    "description": "Fully qualified or built-in component class name.",
                    "value": conf.get("className", ""),
                },
                {
                    "name": "classFile",
                    "type": "str | None",
                    "required": False,
                    "description": "Optional Python file containing the component class.",
                    "value": conf.get("classFile"),
                },
                {
                    "name": "inputStreams",
                    "type": "dict[str,str]",
                    "required": False,
                    "description": "Semantic input stream roles mapped to SHM names.",
                    "value": conf.get("inputStreams", {}),
                },
                {
                    "name": "outputStreams",
                    "type": "dict[str,str]",
                    "required": False,
                    "description": "Semantic output stream roles mapped to SHM names.",
                    "value": conf.get("outputStreams", {}),
                },
            ]
        )
        for field_descriptor in descriptor.all_fields if descriptor is not None else ():
            value = conf.get(field_descriptor.name, field_descriptor.default)
            live_value = self.get_parameter(section_name, field_descriptor.name, fallback=value)
            rows.append(
                {
                    "name": field_descriptor.name,
                    "type": field_descriptor.field_type,
                    "required": field_descriptor.required,
                    "description": field_descriptor.description,
                    "value": live_value,
                }
            )

        existing_names = {row["name"] for row in rows}
        for runtime_row in self._runtime_parameter_definitions(section_name):
            if runtime_row["name"] in existing_names:
                continue
            rows.append(
                {
                    "name": runtime_row["name"],
                    "type": runtime_row["type"],
                    "required": runtime_row["required"],
                    "description": runtime_row["description"],
                    "value": self.get_parameter(section_name, runtime_row["name"], fallback=runtime_row.get("default")),
                }
            )
        return rows

    def get_parameter(self, section_name: str, name: str, *, fallback: Any = None) -> Any:
        manager = self.ensure_manager()
        if manager.state in {"running", "degraded", "failed"} and section_name in manager.runtimes:
            runtime = manager.runtimes[section_name]
            target = manager.get_component(section_name)
            if not _is_live_runtime_field(name):
                return fallback
            if runtime.mode == "hard-rtc" and hasattr(target, "getProperty"):
                try:
                    value = target.getProperty(name)
                    if value == -1 and getattr(target, "lastError", None):
                        return fallback
                    return value
                except Exception:
                    return fallback
            try:
                if not hasattr(target, name):
                    return fallback
                return getattr(target, name)
            except Exception:
                return fallback
        if self.config and section_name in self.config:
            return self.config[section_name].get(name, fallback)
        return fallback

    def set_parameter(self, section_name: str, name: str, raw_value: Any) -> Any:
        if not self.config or section_name not in self.config:
            raise KeyError(section_name)
        descriptor = self._descriptor_for_section(section_name)
        runtime_defs = {row["name"]: row for row in self._runtime_parameter_definitions(section_name)}
        field_type = "str"
        if descriptor is not None and name in descriptor.field_map:
            field_type = descriptor.field_map[name].field_type
        elif name in runtime_defs:
            field_type = runtime_defs[name]["type"]
        elif name == "classFile":
            field_type = "str | None"
        elif name in {"inputStreams", "outputStreams"}:
            field_type = "dict[str,str]"
        value = _coerce_runtime_value(raw_value, field_type)

        persist_in_config = name in self.config[section_name] or bool(runtime_defs.get(name, {}).get("persist", False))
        if persist_in_config:
            self.config[section_name][name] = value

        manager = self.ensure_manager()
        if persist_in_config:
            manager.config[section_name][name] = value
        if (
            manager.state in {"running", "degraded", "failed"}
            and section_name in manager.runtimes
            and _is_live_runtime_field(name)
        ):
            runtime = manager.runtimes[section_name]
            target = manager.get_component(section_name)
            if runtime.mode == "hard-rtc" and hasattr(target, "setProperty"):
                result = target.setProperty(name, value)
                if result == -1 and getattr(target, "lastError", None):
                    raise RuntimeError(target.lastError)
            else:
                if not hasattr(target, name):
                    return value
                setattr(target, name, value)
        return value

    def build_graph_snapshot(self, status: dict[str, Any] | None = None, *, runtime_controls_enabled: bool = True) -> GraphSnapshot:
        if not self.config:
            return GraphSnapshot()
        status = status or self._last_status or self.status()
        component_status = status.get("components", {}) if isinstance(status, dict) else {}
        sections = _ordered_sections(self.config)
        nodes = []
        output_map: dict[str, list[tuple[str, str]]] = {}
        streams_conf = self.config.get("streams", {}) if isinstance(self.config.get("streams"), dict) else {}
        telemetry_streams = set(self.config.get("telemetry", {}).get("streams", [])) if isinstance(self.config.get("telemetry"), dict) else set()

        for index, section_name in enumerate(sections):
            descriptor = self._descriptor_for_section(section_name)
            component_info = component_status.get(section_name, {})
            x, y = self.get_component_position(section_name, default_index=index)
            title = section_name.upper()
            subtitle = descriptor.class_name if descriptor is not None else self.config[section_name].get("name", "component")
            input_names = {
                _component_stream_name(self.config, section_name, "input", stream.name)
                for stream in descriptor.input_streams
                if descriptor is not None and stream.name != "*"
            } if descriptor is not None else set()
            output_names = {
                _component_stream_name(self.config, section_name, "output", stream.name)
                for stream in descriptor.output_streams
                if descriptor is not None and stream.name != "*"
            } if descriptor is not None else set()
            for stream_name, stream_conf in streams_conf.items():
                if not isinstance(stream_conf, dict):
                    continue
                if stream_conf.get("outputComponent", stream_conf.get("producer")) == section_name:
                    output_names.add(stream_name)
                input_components = stream_conf.get("inputComponents", stream_conf.get("consumers", []))
                if isinstance(input_components, list) and section_name in input_components:
                    input_names.add(stream_name)
            input_streams = tuple(sorted(input_names))
            output_streams = tuple(sorted(output_names))
            for stream_name in output_streams:
                output_map.setdefault(stream_name, []).append((section_name, stream_name))
            nodes.append(
                GraphNodeModel(
                    section_name=section_name,
                    title=title,
                    subtitle=subtitle,
                    x=x,
                    y=y,
                    state=component_info.get("state", status.get("state", "stopped")),
                    mode=component_info.get("mode", status.get("mode", "soft-rtc")),
                    error=component_info.get("error") or component_info.get("last_error"),
                    restart_policy=component_info.get("restart_policy"),
                    input_streams=input_streams,
                    output_streams=output_streams,
                    can_start=runtime_controls_enabled and component_info.get("state", "stopped") != "running",
                    can_stop=runtime_controls_enabled and component_info.get("state", "stopped") == "running",
                )
            )

        edges: list[GraphEdgeModel] = []
        seen_edges: set[tuple[str, str, str, str]] = set()
        for node in nodes:
            descriptor = self._descriptor_for_section(node.section_name)
            if descriptor is None:
                continue
            for input_stream in descriptor.input_streams:
                actual_input_name = _component_stream_name(self.config, node.section_name, "input", input_stream.name)
                if input_stream.name == "*":
                    for upstreams in output_map.values():
                        for upstream_section, upstream_stream in upstreams:
                            if telemetry_streams and upstream_stream not in telemetry_streams:
                                continue
                            edge_key = (upstream_section, node.section_name, upstream_stream, actual_input_name)
                            if edge_key not in seen_edges:
                                seen_edges.add(edge_key)
                                edges.append(
                                    GraphEdgeModel(
                                        source_section=upstream_section,
                                        target_section=node.section_name,
                                        source_stream=upstream_stream,
                                        target_stream=actual_input_name,
                                    )
                                )
                    continue
                for upstream_section, upstream_stream in output_map.get(actual_input_name, []):
                    if upstream_section == node.section_name:
                        continue
                    edge_key = (upstream_section, node.section_name, upstream_stream, actual_input_name)
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        edges.append(
                            GraphEdgeModel(
                                source_section=upstream_section,
                                target_section=node.section_name,
                                source_stream=upstream_stream,
                                target_stream=actual_input_name,
                            )
                        )

        for stream_name, stream_conf in streams_conf.items():
            if not isinstance(stream_conf, dict):
                continue
            source_section = stream_conf.get("outputComponent", stream_conf.get("producer"))
            input_components = stream_conf.get("inputComponents", stream_conf.get("consumers", []))
            if not isinstance(source_section, str) or not isinstance(input_components, list):
                continue
            for target_section in input_components:
                edge_key = (source_section, str(target_section), stream_name, stream_name)
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                edges.append(
                    GraphEdgeModel(
                        source_section=source_section,
                        target_section=str(target_section),
                        source_stream=stream_name,
                        target_stream=stream_name,
                    )
                )

        return GraphSnapshot(
            nodes=tuple(nodes),
            edges=tuple(edges),
            state=status.get("state", "created"),
            error=status.get("error"),
            config_path=self.config_path,
            mode=status.get("mode", "soft-rtc"),
            metadata={"positions": _graph_layout_positions(self.config)},
        )

    def suggested_viewer_streams(self) -> list[str]:
        snapshot = self.build_graph_snapshot(self._last_status)
        available = {stream for node in snapshot.nodes for stream in node.output_streams}
        preferred = ["wfsRaw", "wfs", "signal2D", "wfc2D", "psfShort", "psfLong"]
        selected = [stream for stream in preferred if stream in available]
        if selected:
            return selected
        return sorted(available)

    def launch_viewer(self) -> subprocess.Popen:
        streams = self.suggested_viewer_streams()
        if not streams:
            raise RuntimeError("No viewer-compatible output streams are available")
        return subprocess.Popen(["pyrtc-view", *streams])

    def log_files(self) -> list[str]:
        status = self._last_status or self.status()
        files = []
        for component_status in status.get("components", {}).values():
            log_file = component_status.get("log_file")
            if isinstance(log_file, str) and log_file and log_file not in files:
                files.append(log_file)
        return files