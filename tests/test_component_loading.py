"""Tests for component class resolution (pyrtc.component_loading)."""

from pathlib import Path

import pytest

from pyrtc.component_loading import (
    canonical_pyrtc_module_name,
    import_symbol,
    import_symbol_from_file,
    resolve_class_symbol,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_canonical_name_maps_package_files():
    assert canonical_pyrtc_module_name(REPO_ROOT / "pyrtc" / "loop.py") == "pyrtc.loop"
    assert (
        canonical_pyrtc_module_name(REPO_ROOT / "pyrtc" / "hardware" / "synthetic_systems.py")
        == "pyrtc.hardware.synthetic_systems"
    )
    assert canonical_pyrtc_module_name(Path("/tmp/elsewhere.py")) is None
    assert canonical_pyrtc_module_name(REPO_ROOT / "pyrtc" / "notes.txt") is None


def test_import_symbol_from_file_reuses_canonical_class():
    from pyrtc.loop import Loop

    resolved = import_symbol_from_file(str(REPO_ROOT / "pyrtc" / "loop.py"), "Loop")
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
