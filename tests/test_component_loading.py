"""Tests for component class resolution (pyRTC.component_loading)."""

from pathlib import Path

import pytest

from pyRTC.component_loading import (
    canonical_pyrtc_module_name,
    import_symbol,
    import_symbol_from_file,
    resolve_class_symbol,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_canonical_name_maps_package_files():
    assert canonical_pyrtc_module_name(REPO_ROOT / "pyRTC" / "Loop.py") == "pyRTC.Loop"
    assert (
        canonical_pyrtc_module_name(REPO_ROOT / "pyRTC" / "hardware" / "SyntheticSystems.py")
        == "pyRTC.hardware.SyntheticSystems"
    )
    assert canonical_pyrtc_module_name(Path("/tmp/elsewhere.py")) is None
    assert canonical_pyrtc_module_name(REPO_ROOT / "pyRTC" / "notes.txt") is None


def test_import_symbol_from_file_reuses_canonical_class():
    from pyRTC.Loop import Loop

    resolved = import_symbol_from_file(str(REPO_ROOT / "pyRTC" / "Loop.py"), "Loop")
    assert resolved is Loop


def test_import_symbol_from_file_loads_custom_module(tmp_path):
    module_path = tmp_path / "my_component.py"
    module_path.write_text("class MyComponent:\n    marker = 'custom'\n", encoding="utf-8")

    resolved = import_symbol_from_file(str(module_path), "MyComponent")
    assert resolved.marker == "custom"


def test_import_symbol_dotted_path():
    from pyRTC.Loop import Loop

    assert import_symbol("pyRTC.Loop.Loop") is Loop


def test_import_symbol_bare_name_skips_shadowing_module():
    import importlib

    # Importing the same-named *module* binds it on the package and shadows
    # the lazy class re-export; resolution must still find the class.
    importlib.import_module("pyRTC.hardware.SyntheticSHWFS")
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
    from pyRTC.Loop import Loop

    resolved = resolve_class_symbol("pyRTC.Loop.Loop", "/nonexistent/path.py")
    assert resolved is Loop
