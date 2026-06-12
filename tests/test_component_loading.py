"""Tests for component class resolution (pyrtc.component_loading)."""

import importlib.util
import os
from pathlib import Path

import pytest

from pyrtc.component_loading import (
    canonical_pyrtc_module_name,
    import_symbol,
    import_symbol_from_file,
    resolve_class_symbol,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _installed_pyrtc_root() -> Path:
    """Return the on-disk location of the loaded ``pyrtc`` package.

    Tests should look up components relative to whichever copy of the
    package is actually importable in the current environment — the
    installed wheel on CI (``pip install .``) or the local source tree
    during an editable install / in-repo test run. Hard-coding the
    source-checkout path breaks both the installed case (where the
    package lives in site-packages, not the repo) and editable installs
    on systems where the checkout isn't the package being imported.
    """

    spec = importlib.util.find_spec("pyrtc")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("pyrtc package is not importable")
    # ``__path__`` entries are strings pointing at the directories that
    # contain the package's submodules. The first one is sufficient for
    # these tests because the pyrtc package is single-rooted.
    return Path(os.path.normpath(next(iter(spec.submodule_search_locations))))


def test_canonical_name_maps_package_files():
    package_root = _installed_pyrtc_root()
    assert canonical_pyrtc_module_name(package_root / "loop.py") == "pyrtc.loop"
    assert (
        canonical_pyrtc_module_name(package_root / "hardware" / "synthetic_systems.py")
        == "pyrtc.hardware.synthetic_systems"
    )
    assert canonical_pyrtc_module_name(Path("/tmp/elsewhere.py")) is None
    assert canonical_pyrtc_module_name(package_root / "notes.txt") is None


def test_canonical_name_rejects_source_checkout_when_package_is_installed():
    # When pyrtc is installed (the CI setup: ``pip install .``), the
    # source-checkout path is *not* the loaded package, so it must not
    # be reported as a canonical module. This guards against regressions
    # where ``__file__`` is mistakenly used as the package root.
    package_root = _installed_pyrtc_root().resolve()
    checkout_loop = (REPO_ROOT / "pyrtc" / "loop.py").resolve()
    if checkout_loop.is_relative_to(package_root):
        # Editable install or in-repo run: paths overlap and the lookup
        # legitimately succeeds. Nothing to assert.
        return
    assert canonical_pyrtc_module_name(checkout_loop) is None


def test_import_symbol_from_file_reuses_canonical_class():
    from pyrtc.loop import Loop

    package_root = _installed_pyrtc_root()
    resolved = import_symbol_from_file(str(package_root / "loop.py"), "Loop")
    assert resolved is Loop


def test_import_symbol_from_file_loads_custom_module(tmp_path):
    module_path = tmp_path / "my_component.py"
    module_path.write_text("class MyComponent:\n    marker = 'custom'\n", encoding="utf-8")

    resolved = import_symbol_from_file(str(module_path), "MyComponent")
    assert resolved.marker == "custom"


def test_import_symbol_dotted_path():
    from pyrtc.loop import Loop

    assert import_symbol("pyrtc.loop.Loop") is Loop


def test_import_symbol_bare_name_skips_shadowing_module():
    import importlib

    # Importing the same-named *module* binds it on the package and shadows
    # the lazy class re-export; resolution must still find the class.
    importlib.import_module("pyrtc.hardware.synthetic_shwfs")
    resolved = import_symbol("SyntheticSHWFS")
    assert isinstance(resolved, type)
    assert resolved.__name__ == "SyntheticSHWFS"


def test_import_symbol_unknown_name_raises():
    with pytest.raises(ImportError):
        import_symbol("DefinitelyNotAComponent")


def test_resolve_class_symbol_prefers_existing_class_file(tmp_path):
    module_path = tmp_path / "adapter.py"
    module_path.write_text("class Adapter:\n    pass\n", encoding="utf-8")

    resolved = resolve_class_symbol("Adapter", str(module_path))
    assert resolved.__name__ == "Adapter"


def test_resolve_class_symbol_falls_back_to_name_when_file_missing():
    from pyrtc.loop import Loop

    resolved = resolve_class_symbol("pyrtc.loop.Loop", "/nonexistent/path.py")
    assert resolved is Loop
