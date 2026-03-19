"""Runtime config synchronization hooks for optional integrations.

This module lets configured component or resource classes normalize derived
runtime config before SHM planning or component construction, without coupling
the core pipeline to any specific backend.
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from typing import Any, Mapping

from pyRTC.logging_utils import get_logger


logger = get_logger(__name__)

_NON_COMPONENT_TOP_LEVEL_SECTIONS = {"manager", "streams", "metadata", "resources"}


def _import_symbol(path_or_name: str):
    if "." in path_or_name:
        module_name, attr_name = path_or_name.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)

    for module_name in ("pyRTC.hardware", "pyRTC"):
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, path_or_name):
                return getattr(module, path_or_name)
        except Exception:
            continue
    raise ImportError(f"Unable to resolve component symbol '{path_or_name}'")


def _import_symbol_from_file(file_path: str, attr_name: str):
    module_path = Path(file_path).expanduser().resolve()
    module_name = f"pyrtc_runtime_{module_path.stem}_{abs(hash(str(module_path))) & 0xFFFFFFFF:x}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load component module from '{module_path}'")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, attr_name)


def _resolve_class_symbol(class_name: str, class_file: str | None = None):
    if class_file:
        component_file = Path(class_file).expanduser()
        if component_file.exists():
            attr_name = class_name.rsplit(".", 1)[-1]
            return _import_symbol_from_file(str(component_file), attr_name)
    return _import_symbol(class_name)


def _iter_configured_classes(system_conf: Mapping[str, Any]):
    resources_conf = system_conf.get("resources", {})
    if isinstance(resources_conf, Mapping):
        for resource_conf in resources_conf.values():
            if not isinstance(resource_conf, Mapping):
                continue
            class_name = resource_conf.get("className")
            class_file = resource_conf.get("classFile")
            if isinstance(class_name, str) and class_name.strip():
                yield class_name, class_file if isinstance(class_file, str) else None

    for section_name, section_conf in system_conf.items():
        if section_name in _NON_COMPONENT_TOP_LEVEL_SECTIONS:
            continue
        if not isinstance(section_conf, Mapping):
            continue
        class_name = section_conf.get("className")
        class_file = section_conf.get("classFile")
        if isinstance(class_name, str) and class_name.strip():
            yield class_name, class_file if isinstance(class_file, str) else None


def sync_runtime_config(system_conf: Mapping[str, Any]) -> None:
    """Apply optional runtime config normalization hooks exposed by classes."""

    seen: set[tuple[str, str | None]] = set()
    for class_name, class_file in _iter_configured_classes(system_conf):
        key = (class_name, class_file)
        if key in seen:
            continue
        seen.add(key)
        try:
            symbol = _resolve_class_symbol(class_name, class_file)
        except Exception:
            logger.debug("Runtime config sync skipped for unresolved class %s", class_name, exc_info=True)
            continue

        sync_hook = getattr(symbol, "sync_system_config", None)
        if not callable(sync_hook):
            continue

        try:
            sync_hook(system_conf)
        except Exception:
            logger.debug("Runtime config sync hook failed for %s", class_name, exc_info=True)