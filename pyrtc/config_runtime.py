"""Runtime config synchronization hooks for optional integrations.

This module lets configured component or resource classes normalize derived
runtime config before SHM planning or component construction, without coupling
the core pipeline to any specific backend.
"""

from __future__ import annotations

from typing import Any, Mapping

from pyrtc.component_loading import resolve_class_symbol as _resolve_class_symbol
from pyrtc.logging_utils import get_logger


logger = get_logger(__name__)

_NON_COMPONENT_TOP_LEVEL_SECTIONS = {"manager", "streams", "metadata", "resources"}


def stream_alias_map(raw_mapping: Any, defaults: Mapping[str, str] | None = None) -> dict[str, str]:
    """Normalize an ``input_streams``/``output_streams`` config mapping.

    Maps each semantic stream name to its SHM name. Values may be a plain
    string or a mapping with ``shm`` (or ``name``). Entries from ``raw_mapping``
    override ``defaults``; malformed entries are ignored.
    """

    normalized = dict(defaults or {})
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
            normalized[semantic_name] = shm_name
    return normalized


def _iter_configured_classes(system_conf: Mapping[str, Any]):
    resources_conf = system_conf.get("resources", {})
    if isinstance(resources_conf, Mapping):
        for resource_conf in resources_conf.values():
            if not isinstance(resource_conf, Mapping):
                continue
            class_name = resource_conf.get("class_name")
            class_file = resource_conf.get("class_file")
            if isinstance(class_name, str) and class_name.strip():
                yield class_name, class_file if isinstance(class_file, str) else None

    for section_name, section_conf in system_conf.items():
        if section_name in _NON_COMPONENT_TOP_LEVEL_SECTIONS:
            continue
        if not isinstance(section_conf, Mapping):
            continue
        class_name = section_conf.get("class_name")
        class_file = section_conf.get("class_file")
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
            logger.debug(
                "Runtime config sync skipped for unresolved class %s", class_name, exc_info=True
            )
            continue

        sync_hook = getattr(symbol, "sync_system_config", None)
        if not callable(sync_hook):
            continue

        try:
            sync_hook(system_conf)
        except Exception:
            logger.debug("Runtime config sync hook failed for %s", class_name, exc_info=True)
