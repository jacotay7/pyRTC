"""Resolution of component classes from config names and file paths.

Both the config validator and the runtime manager accept components either as
importable dotted names (``pyrtc.loop.Loop``), bare registry names
(``SyntheticSHWFS``), or an explicit ``class_file`` path. This module is the
single implementation of that lookup so every caller resolves to the *same*
class object — exec'ing duplicate copies of a module produces distinct class
objects that break descriptor lookups and ``isinstance`` checks.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import os
import sys
from pathlib import Path

from pyrtc.logging_utils import get_logger

logger = get_logger(__name__)


def _pyrtc_package_paths() -> list[Path]:
    """Return the on-disk locations of the loaded ``pyrtc`` package.

    Uses the *loaded* package's ``__path__`` so resolution works no matter
    whether ``pyrtc`` was imported from an installed wheel (e.g. CI's
    ``pip install .``), an editable install, or a local checkout that
    happens to be on ``sys.path``. Falls back to the directory containing
    this file if the package can't be located (e.g. a stale build that
    failed to install the package).
    """

    try:
        pkg = importlib.import_module("pyrtc")
    except Exception:
        return [Path(__file__).resolve().parent]

    raw_paths = getattr(pkg, "__path__", None)
    if not raw_paths:
        return [Path(__file__).resolve().parent]

    return [Path(os.path.normpath(p)) for p in raw_paths if p]


def canonical_pyrtc_module_name(module_path: Path) -> str | None:
    """Map a file inside the loaded ``pyrtc`` package to its module name.

    Returns ``None`` for files outside the package (including an editable
    source checkout when ``pyrtc`` was installed as a regular wheel) or
    for non-``.py`` paths. Callers fall back to a file-based import in
    that case.
    """

    if module_path.suffix != ".py":
        return None

    for package_root in _pyrtc_package_paths():
        try:
            relative = module_path.relative_to(package_root)
        except ValueError:
            continue
        return ".".join(("pyrtc", *relative.with_suffix("").parts))
    return None


def _identical_installed_module_name(module_path: Path) -> str | None:
    """Return ``pyrtc.<x>`` when ``module_path`` is a byte-identical copy of it.

    Covers a source checkout next to an installed wheel (CI runs the repo's
    example configs against ``pip install .``): the checkout's
    ``pyrtc/loop.py`` is the installed ``pyrtc.loop``, so it is imported
    canonically instead of exec'd as a second copy. An edited copy is not
    identical and still loads from its own file.
    """

    parts = module_path.with_suffix("").parts
    if module_path.suffix != ".py" or "pyrtc" not in parts:
        return None
    index = len(parts) - 1 - parts[::-1].index("pyrtc")
    dotted = ".".join(parts[index:])
    try:
        spec = importlib.util.find_spec(dotted)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin or not os.path.isfile(spec.origin):
        return None
    try:
        same = Path(spec.origin).read_bytes() == module_path.read_bytes()
    except OSError:
        return None
    return dotted if same else None


def _file_module_name(module_path: Path) -> str:
    """Return a module name for ``module_path`` that is the same in every process.

    numba's on-disk cache (``cache=True``) records the defining module's name
    and imports it when a later process loads the cache, so it must be stable
    (``hash()`` of a string is randomized per process) and in ``sys.modules``.
    """

    digest = hashlib.sha1(str(module_path).encode("utf-8")).hexdigest()[:10]
    return f"pyrtc_custom_{module_path.stem}_{digest}"


def import_symbol_from_file(file_path: str, attr_name: str):
    """Load ``attr_name`` from a Python file, preferring canonical modules.

    When the file belongs to the loaded ``pyrtc`` package (configs may
    point ``class_file`` at e.g. the installed ``pyrtc/loop.py``), or is a
    byte-identical copy of one of its modules (a source checkout next to an
    installed wheel), the canonical module is imported instead of exec'ing a
    second copy. Other files are imported once under a stable module name
    registered in ``sys.modules``, so repeated loads return the same class
    and numba can reload ``cache=True`` kernels defined there.
    """

    module_path = Path(file_path).expanduser().resolve()

    canonical_name = canonical_pyrtc_module_name(module_path) or _identical_installed_module_name(
        module_path
    )
    if canonical_name is not None:
        try:
            module = importlib.import_module(canonical_name)
            return getattr(module, attr_name)
        except Exception:
            logger.debug("Falling back to file-based import for %s", module_path, exc_info=True)

    module_name = _file_module_name(module_path)
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load component module from '{module_path}'")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
    return getattr(module, attr_name)


def import_symbol(path_or_name: str):
    """Resolve a dotted import path or a bare pyrtc component name to an object."""

    if "." in path_or_name:
        module_name, attr_name = path_or_name.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)

    for module_name in ("pyrtc.hardware", "pyrtc"):
        try:
            module = importlib.import_module(module_name)
            attr = getattr(module, path_or_name, None)
            if attr is None:
                continue
            # Importing pyrtc.hardware.<X> as a module binds it on the
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
    """Resolve a component class from config ``class_name`` / ``class_file``."""

    if class_file:
        module_path = Path(class_file).expanduser()
        if module_path.exists():
            attr_name = class_name.rsplit(".", 1)[-1]
            return import_symbol_from_file(str(module_path), attr_name)
    return import_symbol(class_name)
