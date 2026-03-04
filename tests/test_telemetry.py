import numpy as np
import importlib

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
