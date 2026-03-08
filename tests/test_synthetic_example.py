import importlib
import importlib.util
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "examples" / "synthetic_shwfs" / "run_soft_rtc.py"


def _load_example_module():
    spec = importlib.util.spec_from_file_location("synthetic_example", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_expected_stream_specs_match_example_config():
    module = _load_example_module()
    config = module.read_yaml_file(str(REPO_ROOT / "examples" / "synthetic_shwfs" / "config.yaml"))

    specs = module.expected_stream_specs(config)

    assert specs["wfs"]["shape"] == (32, 32)
    assert specs["signal"]["shape"] == (32,)
    assert specs["signal2D"]["shape"] == (8, 4)
    assert specs["wfc2D"]["shape"] == (8, 4)
    assert specs["psfShort"]["shape"] == (64, 64)


def test_ensure_expected_shms_reuses_matching_streams(monkeypatch):
    module = _load_example_module()
    config = module.read_yaml_file(str(REPO_ROOT / "examples" / "synthetic_shwfs" / "config.yaml"))
    specs = module.expected_stream_specs(config)

    monkeypatch.setattr(
        module,
        "_existing_shm_spec",
        lambda name: (specs[name]["shape"], np.dtype(specs[name]["dtype"])),
    )
    cleared = []
    monkeypatch.setattr(module, "clear_named_shms", lambda names: cleared.extend(names))

    rebuilt, reused = module.ensure_expected_shms(config)

    assert rebuilt == []
    assert set(reused) == set(specs)
    assert cleared == []


def test_ensure_expected_shms_clears_only_incompatible_streams(monkeypatch):
    module = _load_example_module()
    config = module.read_yaml_file(str(REPO_ROOT / "examples" / "synthetic_shwfs" / "config.yaml"))
    specs = module.expected_stream_specs(config)

    def _existing(name):
        if name == "wfs":
            return ((16, 16), np.dtype(np.int32))
        return (specs[name]["shape"], np.dtype(specs[name]["dtype"]))

    monkeypatch.setattr(module, "_existing_shm_spec", _existing)
    cleared = []
    monkeypatch.setattr(module, "clear_named_shms", lambda names: cleared.extend(names))

    rebuilt, reused = module.ensure_expected_shms(config)

    assert rebuilt == ["wfs"]
    assert "signal" in reused
    assert cleared == ["wfs"]


def test_hardware_package_import_is_lazy(capsys):
    for module_name in [
        "pyRTC.hardware",
        "pyRTC.hardware.ALPAODM",
        "pyRTC.hardware.SyntheticSystems",
    ]:
        sys.modules.pop(module_name, None)

    hardware = importlib.import_module("pyRTC.hardware")
    captured = capsys.readouterr()

    assert captured.out == ""
    assert "pyRTC.hardware.ALPAODM" not in sys.modules
    assert hardware.SyntheticSHWFS.__name__ == "SyntheticSHWFS"