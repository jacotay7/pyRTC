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

    assert specs["wfs"]["shape"] == (49, 49)
    assert specs["signal"]["shape"] == (98,)
    assert specs["signal_2d"]["shape"] == (14, 7)
    assert specs["wfc_2d"]["shape"] == (11, 11)
    assert specs["psf_short"]["shape"] == (64, 64)


def test_synthetic_wfc_default_layout_matches_expected_shape():
    from pyrtc.pipeline import clear_shms
    from pyrtc.hardware.synthetic_systems import SyntheticWFC

    config = _load_example_module().read_yaml_file(
        str(REPO_ROOT / "examples" / "synthetic_shwfs" / "config.yaml")
    )
    clear_shms(["wfc", "wfc_2d"])
    wfc = SyntheticWFC(config["wfc"])

    try:
        assert wfc.layout.shape == (11, 11)
        assert int(np.count_nonzero(wfc.layout)) == 97
        assert wfc.correction_vector_2d.read().shape == (11, 11)
        assert wfc.correction_vector_2d is not None
    finally:
        wfc.stop()
        clear_shms(["wfc", "wfc_2d"])


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
        "pyrtc.hardware",
        "pyrtc.hardware.alpao_dm",
        "pyrtc.hardware.synthetic_systems",
    ]:
        sys.modules.pop(module_name, None)

    hardware = importlib.import_module("pyrtc.hardware")
    captured = capsys.readouterr()

    assert captured.out == ""
    assert "pyrtc.hardware.alpao_dm" not in sys.modules
    assert hardware.SyntheticSHWFS.__name__ == "SyntheticSHWFS"


def test_synthetic_example_drives_wfc2d_nonzero():
    module = _load_example_module()
    config = module.read_yaml_file(str(REPO_ROOT / "examples" / "synthetic_shwfs" / "config.yaml"))
    module.ensure_expected_shms(config, force_rebuild=True)
    system = module.build_system(config)

    try:
        module.start_system(system)
        import time as _time

        deadline = _time.perf_counter() + 1.0
        wfc_abs_max = 0.0
        wfc2d_abs_max = 0.0
        while _time.perf_counter() < deadline:
            wfc = system["loop"].wfc_shm.read()
            wfc2d = system["wfc"].correction_vector_2d.read()
            wfc_abs_max = max(wfc_abs_max, float(np.max(np.abs(wfc))))
            wfc2d_abs_max = max(wfc2d_abs_max, float(np.max(np.abs(wfc2d))))
            if wfc_abs_max > 0.0 and wfc2d_abs_max > 0.0:
                break
            _time.sleep(0.05)
    finally:
        module.stop_system(system)
        module.clear_named_shms(list(module.expected_stream_specs(config)))

    assert wfc_abs_max > 0.0
    assert wfc2d_abs_max > 0.0
