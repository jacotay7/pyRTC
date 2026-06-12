"""Resolution of component classes from config names and file paths.

Both the config validator and the runtime manager accept components either as
importable dotted names (``pyRTC.Loop.Loop``), bare registry names
(``SyntheticSHWFS``), or an explicit ``classFile`` path. This module is the
single implementation of that lookup so every caller resolves to the *same*
class object — exec'ing duplicate copies of a module produces distinct class
objects that break descriptor lookups and ``isinstance`` checks.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
from pathlib import Path

from pyRTC.logging_utils import get_logger

logger = get_logger(__name__)


def canonical_pyrtc_module_name(module_path: Path) -> str | None:
    """Map a file inside the pyRTC package back to its canonical module name."""

    package_root = Path(__file__).resolve().parent
    try:
        relative = module_path.relative_to(package_root)
    except ValueError:
        return None
    if module_path.suffix != ".py":
        return None
    return ".".join(("pyRTC", *relative.with_suffix("").parts))


def import_symbol_from_file(file_path: str, attr_name: str):
    """Load ``attr_name`` from a Python file, preferring canonical modules.

    When the file belongs to the pyRTC package itself (configs commonly point
    ``classFile`` at e.g. ``pyRTC/Loop.py``), the canonical module is imported
    instead of exec'ing a second copy.
    """

    module_path = Path(file_path).expanduser().resolve()

    canonical_name = canonical_pyrtc_module_name(module_path)
    if canonical_name is not None:
        try:
            module = importlib.import_module(canonical_name)
            return getattr(module, attr_name)
        except Exception:
            logger.debug(
                "Falling back to file-based import for %s", module_path, exc_info=True
            )

    module_name = f"pyrtc_custom_{module_path.stem}_{abs(hash(str(module_path))) & 0xFFFFFFFF:x}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load component module from '{module_path}'")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, attr_name)


def import_symbol(path_or_name: str):
    """Resolve a dotted import path or a bare pyRTC component name to an object."""

    if "." in path_or_name:
        module_name, attr_name = path_or_name.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)

    for module_name in ("pyRTC.hardware", "pyRTC"):
        try:
            module = importlib.import_module(module_name)
            attr = getattr(module, path_or_name, None)
            if attr is None:
                continue
            # Importing pyRTC.hardware.<X> as a module binds it on the
            # package, shadowing same-named lazy class re-exports. Never
            # return a module where a component class is expected; if the
            # shadowing module defines a class of the same name, use that.
            if inspect.ismodule(attr):
                inner = getattr(attr, path_or_name, None)
                if inner is not None and not inspect.ismodule(inner):
                    return inner
                continue
            return attr
        except Exception:
            continue
    raise ImportError(f"Unable to resolve component symbol '{path_or_name}'")


def resolve_class_symbol(class_name: str, class_file: str | None = None):
    """Resolve a component class from config ``className`` / ``classFile``."""

    if class_file:
        module_path = Path(class_file).expanduser()
        if module_path.exists():
            attr_name = class_name.rsplit(".", 1)[-1]
            return import_symbol_from_file(str(module_path), attr_name)
    return import_symbol(class_name)
