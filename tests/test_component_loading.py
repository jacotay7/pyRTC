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


_CACHED_KERNEL = """
import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def double(x):
    return x * 2.0


class Thing:
    pass
"""

_CALL_KERNEL = (
    "import sys, numpy as np;"
    "from pyrtc.component_loading import import_symbol_from_file;"
    "Thing = import_symbol_from_file(sys.argv[1], 'Thing');"
    "print(sys.modules[Thing.__module__].double(np.ones(3)).sum())"
)


def test_file_loaded_numba_cache_is_reusable_across_processes(tmp_path):
    """A cache=True kernel in a class_file must load from cache in a new process.

    Exec'ing the file without registering it in sys.modules made numba record
    the module as '<dynamic>', so the second process crashed importing it.
    """
    import subprocess
    import sys

    module_file = tmp_path / "custom_kernels.py"
    module_file.write_text(_CACHED_KERNEL)
    for _ in range(2):  # the first run writes the cache, the second loads it
        result = subprocess.run(
            [sys.executable, "-c", _CALL_KERNEL, str(module_file)],
            capture_output=True,
            text=True,
            cwd=tmp_path,
        )
        assert result.returncode == 0, result.stderr[-2000:]
        assert result.stdout.strip() == "6.0"
    assert list((tmp_path / "__pycache__").glob("custom_kernels.double-*.nbi"))


def test_file_loaded_module_has_a_stable_registered_name(tmp_path):
    import subprocess
    import sys

    module_file = tmp_path / "stable_name.py"
    module_file.write_text("class Thing:\n    pass\n")
    Thing = import_symbol_from_file(str(module_file), "Thing")
    assert sys.modules[Thing.__module__].Thing is Thing
    # Loading the same file again reuses the module instead of a second copy.
    assert import_symbol_from_file(str(module_file), "Thing") is Thing
    code = (
        "import sys; from pyrtc.component_loading import import_symbol_from_file;"
        "print(import_symbol_from_file(sys.argv[1], 'Thing').__module__)"
    )
    other = subprocess.run(
        [sys.executable, "-c", code, str(module_file)], capture_output=True, text=True, check=True
    )
    assert other.stdout.strip() == Thing.__module__


def test_identical_copy_of_a_pyrtc_module_resolves_to_the_installed_module(tmp_path):
    """A source checkout next to an installed wheel reuses the installed module."""
    from pyrtc.loop import Loop

    copy = tmp_path / "checkout" / "pyrtc" / "loop.py"
    copy.parent.mkdir(parents=True)
    copy.write_bytes((_installed_pyrtc_root() / "loop.py").read_bytes())
    assert import_symbol_from_file(str(copy), "Loop") is Loop

    edited = tmp_path / "edited" / "pyrtc" / "loop.py"
    edited.parent.mkdir(parents=True)
    edited.write_bytes(copy.read_bytes() + b"\n# local edit\n")
    assert import_symbol_from_file(str(edited), "Loop") is not Loop
