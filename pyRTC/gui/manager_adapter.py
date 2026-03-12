"""Bridge layer between the Qt UI and the current RTCManager API."""

from __future__ import annotations

from ast import literal_eval
import inspect
from pathlib import Path
import subprocess
from typing import Any

from pyRTC.Pipeline import DEFAULT_COMPONENT_ORDER, RTCManager
from pyRTC.component_descriptors import get_component_descriptor, list_component_sections
from pyRTC.config_schema import read_system_config

from .models import GraphEdgeModel, GraphNodeModel, GraphSnapshot


_CONFIG_ONLY_FIELDS = {
    "functions",
    "affinity",
    "gpuDevice",
    "type",
    "signalType",
}

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
}


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
    return sections


def _infer_layout(index: int) -> tuple[float, float]:
    columns = 3
    col = index % columns
    row = index // columns
    return 80.0 + col * 260.0, 60.0 + row * 190.0


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
        descriptor = get_component_descriptor(section_name)
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
        descriptor = get_component_descriptor(section_name)
        conf = self.config[section_name]
        rows = []
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
        descriptor = get_component_descriptor(section_name)
        field_type = "str"
        if descriptor is not None and name in descriptor.field_map:
            field_type = descriptor.field_map[name].field_type
        value = _coerce_runtime_value(raw_value, field_type)
        self.config[section_name][name] = value

        manager = self.ensure_manager()
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

    def build_graph_snapshot(self, status: dict[str, Any] | None = None) -> GraphSnapshot:
        if not self.config:
            return GraphSnapshot()
        status = status or self._last_status or self.status()
        component_status = status.get("components", {}) if isinstance(status, dict) else {}
        sections = _ordered_sections(self.config)
        nodes = []
        output_map: dict[str, list[tuple[str, str]]] = {}
        telemetry_streams = set(self.config.get("telemetry", {}).get("streams", [])) if isinstance(self.config.get("telemetry"), dict) else set()

        for index, section_name in enumerate(sections):
            descriptor = get_component_descriptor(section_name)
            component_info = component_status.get(section_name, {})
            x, y = _infer_layout(index)
            title = section_name.upper()
            subtitle = descriptor.class_name if descriptor is not None else self.config[section_name].get("name", "component")
            input_streams = tuple(stream.name for stream in descriptor.input_streams) if descriptor is not None else ()
            output_streams = tuple(stream.name for stream in descriptor.output_streams) if descriptor is not None else ()
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
                    can_start=component_info.get("state", "stopped") != "running",
                    can_stop=component_info.get("state", "stopped") == "running",
                )
            )

        edges: list[GraphEdgeModel] = []
        for node in nodes:
            descriptor = get_component_descriptor(node.section_name)
            if descriptor is None:
                continue
            for input_stream in descriptor.input_streams:
                if input_stream.name == "*":
                    for upstreams in output_map.values():
                        for upstream_section, upstream_stream in upstreams:
                            if telemetry_streams and upstream_stream not in telemetry_streams:
                                continue
                            edges.append(
                                GraphEdgeModel(
                                    source_section=upstream_section,
                                    target_section=node.section_name,
                                    source_stream=upstream_stream,
                                    target_stream=input_stream.name,
                                )
                            )
                    continue
                for upstream_section, upstream_stream in output_map.get(input_stream.name, []):
                    if upstream_section == node.section_name:
                        continue
                    edges.append(
                        GraphEdgeModel(
                            source_section=upstream_section,
                            target_section=node.section_name,
                            source_stream=upstream_stream,
                            target_stream=input_stream.name,
                        )
                    )

        return GraphSnapshot(
            nodes=tuple(nodes),
            edges=tuple(edges),
            state=status.get("state", "created"),
            error=status.get("error"),
            config_path=self.config_path,
            mode=status.get("mode", "soft-rtc"),
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