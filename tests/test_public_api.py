import json
import subprocess
import sys

import pyrtc


def test_package_root_imports():
    assert pyrtc.loop is not None
    assert pyrtc.RTCManager is not None
    assert pyrtc.wavefront_sensor is not None
    assert pyrtc.wavefront_corrector is not None
    assert pyrtc.slopes_process is not None
    assert pyrtc.science_camera is not None
    assert pyrtc.optimizer is not None
    assert pyrtc.telemetry is not None
    assert pyrtc.ComponentDescriptor is not None
    assert pyrtc.ConfigFieldDescriptor is not None
    assert pyrtc.get_component_descriptor is not None


def test_package_exposes_module_helpers():
    assert pyrtc.streams is not None
    assert pyrtc.utils is not None
    assert callable(pyrtc.set_from_config)
    assert callable(pyrtc.launch_component)
    assert callable(pyrtc.open_stream)
    assert callable(pyrtc.create_stream)
    assert callable(pyrtc.build_descriptor_catalog)
    assert callable(pyrtc.describe_component_class)
    assert callable(pyrtc.list_component_descriptors)
    assert callable(pyrtc.register_component_descriptor)
    assert callable(pyrtc.validate_config_with_descriptor)


def test_importing_pyrtc_does_not_modify_the_environment():
    code = (
        "import os, json; before = dict(os.environ); "
        "import pyrtc, pyrtc.loop, pyrtc.slopes_process, pyrtc.wavefront_corrector; "
        "print(json.dumps(sorted(k for k in set(before) | set(os.environ) "
        "if before.get(k) != os.environ.get(k))))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert json.loads(result.stdout.strip().splitlines()[-1]) == []


def test_importing_pyrtc_does_not_import_pyplot():
    code = (
        "import sys, pyrtc, pyrtc.loop, pyrtc.latency, pyrtc.slopes_process; "
        "print('matplotlib.pyplot' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip().splitlines()[-1] == "False"


def test_import_pyrtc_does_not_import_optional_extras():
    """The core install must work without the optimize/fits/plot/gpu extras (#50).

    torch (the ``gpu`` extra) is imported only by GPU code paths: importing it
    takes most of a second, which every component process would pay.
    """
    import subprocess
    import sys

    code = (
        "import sys, pyrtc, pyrtc.utils, pyrtc.latency, pyrtc.optimizer;"
        "print(sorted(m for m in ('optuna', 'astropy', 'matplotlib.pyplot', 'numexpr', 'torch') "
        "if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_missing_optional_extra_names_the_extra(monkeypatch):
    import sys

    import pytest

    from pyrtc.utils import load_data, require_optional

    # A None entry in sys.modules makes the import raise ImportError.
    for name in ("astropy", "astropy.io", "astropy.io.fits"):
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(ImportError, match=r"pyrtcao\[fits\]"):
        load_data("frame.fits")
    with pytest.raises(ImportError, match=r"pyrtcao\[optimize\]"):
        require_optional("pyrtc_no_such_module", "optimize", "Test feature")
