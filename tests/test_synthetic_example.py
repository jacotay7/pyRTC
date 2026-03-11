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


def test_synthetic_wfc_default_layout_matches_expected_shape():
    from pyRTC.Pipeline import clear_shms
    from pyRTC.hardware.SyntheticSystems import SyntheticWFC

    config = _load_example_module().read_yaml_file(str(REPO_ROOT / "examples" / "synthetic_shwfs" / "config.yaml"))
    clear_shms(["wfc", "wfc2D"])
    wfc = SyntheticWFC(config["wfc"])

    try:
        assert wfc.layout.shape == (8, 4)
        assert wfc.correctionVector2D is not None
    finally:
        wfc.stop()
        clear_shms(["wfc", "wfc2D"])


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


def test_synthetic_example_drives_wfc2d_nonzero():
    module = _load_example_module()
    config = module.read_yaml_file(str(REPO_ROOT / "examples" / "synthetic_shwfs" / "config.yaml"))
    module.ensure_expected_shms(config, force_rebuild=True)
    system = module.build_system(config)

    try:
        module.start_system(system)
        deadline = module.time.perf_counter() + 1.0
        wfc_abs_max = 0.0
        wfc2d_abs_max = 0.0
        while module.time.perf_counter() < deadline:
            wfc = system["loop"].wfcShm.read_noblock(SAFE=False)
            wfc2d = system["wfc"].correctionVector2D.read_noblock()
            wfc_abs_max = max(wfc_abs_max, float(np.max(np.abs(wfc))))
            wfc2d_abs_max = max(wfc2d_abs_max, float(np.max(np.abs(wfc2d))))
            if wfc_abs_max > 0.0 and wfc2d_abs_max > 0.0:
                break
            module.time.sleep(0.05)
    finally:
        module.stop_system(system)
        module.clear_named_shms(list(module.expected_stream_specs(config)))

    assert wfc_abs_max > 0.0
    assert wfc2d_abs_max > 0.0