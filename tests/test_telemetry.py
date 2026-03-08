import numpy as np
import importlib

import pytest

tele_mod = importlib.import_module("pyRTC.Telemetry")


def test_telemetry_save_and_read(monkeypatch, tmp_path):
    class _SHM:
        def __init__(self):
            self._x = np.array([1.0, 2.0], dtype=np.float32)

        def read(self):
            return self._x

    monkeypatch.setattr(tele_mod, "initExistingShm", lambda name: (_SHM(), [2], np.float32))

    t = tele_mod.Telemetry({"dataDir": str(tmp_path), "functions": []})
    t.save("signal", 3, uniqueStr="u")
    arr = t.read()
    assert arr.shape == (3, 2)

    other_file = tmp_path / "raw.bin"
    np.array([1, 2, 3], dtype=np.int16).tofile(other_file)
    out = t.read(filename=str(other_file), dtype=np.int16)
    assert out.dtype == np.int16


def test_telemetry_error_paths(monkeypatch, tmp_path):
    telemetry_module = importlib.import_module("pyRTC.Telemetry")

    def bad_component_init(self, conf):
        raise RuntimeError("telemetry init failed")

    with monkeypatch.context() as mp:
        mp.setattr(telemetry_module.pyRTCComponent, "__init__", bad_component_init)
        with pytest.raises(RuntimeError, match="telemetry init failed"):
            telemetry_module.Telemetry({"functions": []})

    t = telemetry_module.Telemetry({"dataDir": str(tmp_path), "functions": []})

    monkeypatch.setattr(telemetry_module, "initExistingShm", lambda name: (_ for _ in ()).throw(RuntimeError("missing shm")))
    with pytest.raises(RuntimeError, match="missing shm"):
        t.save("signal", 1)

    unmanaged_file = tmp_path / "unmanaged.bin"
    np.array([1, 2, 3], dtype=np.int16).tofile(unmanaged_file)
    with pytest.raises(ValueError, match="please provide a dtype"):
        t.read(filename=str(unmanaged_file))
