import numpy as np
import importlib
from pathlib import Path

import pytest

tele_mod = importlib.import_module("pyrtc.telemetry")


def test_telemetry_save_and_read(monkeypatch, tmp_path):
    class _SHM:
        def __init__(self):
            self._x = np.array([1.0, 2.0], dtype=np.float32)
            self.shape = (2,)
            self.dtype = np.float32
            self.write_time = 123.0

        def read_new(self, timeout=None):
            return self._x

    monkeypatch.setattr(tele_mod, "open_stream", lambda name: _SHM())

    t = tele_mod.Telemetry({"data_dir": str(tmp_path), "functions": []})
    session_path = t.save("signal", 3, unique_str="u")
    arr = t.read()
    assert arr.shape == (3, 2)
    assert Path(session_path).is_dir()

    data = t.read_last_save()
    assert data["signal"]["frames"].shape == (3, 2)
    assert data["signal"]["timestamps"].shape == (3,)
    assert np.all(data["signal"]["timestamps"] == 123.0)
    assert data["signal"]["metadata"]["dtype"] == "float32"
    assert t.list_sessions() == [str(Path(session_path).resolve())]

    reopened = t.read(session_path)
    assert reopened["signal"]["frames"].shape == (3, 2)

    other_file = tmp_path / "raw.bin"
    np.array([1, 2, 3], dtype=np.int16).tofile(other_file)
    out = t.read(filename=str(other_file), dtype=np.int16)
    assert out.dtype == np.int16


def test_telemetry_save_session_supports_multi_stream_grouped_capture(monkeypatch, tmp_path):
    class _SHM:
        def __init__(self, data):
            self._data = np.asarray(data)
            self.shape = self._data.shape
            self.dtype = self._data.dtype
            self.write_time = 456.0

        def read_new(self, timeout=None):
            return self._data

    streams = {
        "signal": _SHM(np.array([1.0, 2.0], dtype=np.float32)),
        "wfc": _SHM(np.array([[3, 4], [5, 6]], dtype=np.int16)),
    }
    monkeypatch.setattr(tele_mod, "open_stream", lambda name: streams[name])

    telemetry = tele_mod.Telemetry({"data_dir": str(tmp_path), "functions": [], "streams": ["signal", "wfc"]})
    session_path = telemetry.save(
        ["signal", "wfc"],
        {"signal": 2, "wfc": 1},
        unique_str="group",
        semantic_tags={"signal": ["signal"], "wfc": ["wfc", "control"]},
        sampling={"signal": {"mode": "every_frame"}},
        config={"metadata": {"name": "synthetic"}},
        config_path=tmp_path / "config.yaml",
        metadata={"operator": "pytest"},
    )

    loaded = tele_mod.load_telemetry_session(session_path)
    manifest = loaded["_session"]

    assert manifest["schema_version"] == tele_mod.TELEMETRY_SESSION_SCHEMA_VERSION
    assert manifest["metadata"]["operator"] == "pytest"
    assert manifest["config_path"] == str((tmp_path / "config.yaml").resolve())
    assert len(manifest["streams"]) == 2
    assert loaded["signal"]["frames"].shape == (2, 2)
    assert loaded["wfc"]["frames"].shape == (1, 2, 2)
    assert manifest["streams"][1]["semantic_tags"] == ["wfc", "control"]
    assert np.all(loaded["signal"]["timestamps"] == 456.0)


def test_telemetry_save_configured_streams_uses_component_config(monkeypatch, tmp_path):
    class _SHM:
        def __init__(self):
            self._data = np.array([7, 8, 9], dtype=np.float32)
            self.shape = (3,)
            self.dtype = np.float32
            self.write_time = 789.0

        def read_new(self, timeout=None):
            return self._data

    monkeypatch.setattr(tele_mod, "open_stream", lambda name: _SHM())
    telemetry = tele_mod.Telemetry({"data_dir": str(tmp_path), "functions": [], "streams": ["signal"]})

    session_path = telemetry.save_configured_streams(2, unique_str="cfg")
    loaded = telemetry.read_last_save()

    assert session_path == telemetry.most_recent_save
    assert loaded["signal"]["frames"].shape == (2, 3)


def test_telemetry_error_paths(monkeypatch, tmp_path):
    telemetry_module = importlib.import_module("pyrtc.telemetry")

    def bad_component_init(self, conf):
        raise RuntimeError("telemetry init failed")

    with monkeypatch.context() as mp:
        mp.setattr(telemetry_module.Component, "__init__", bad_component_init)
        with pytest.raises(RuntimeError, match="telemetry init failed"):
            telemetry_module.Telemetry({"functions": []})

    t = telemetry_module.Telemetry({"data_dir": str(tmp_path), "functions": []})

    monkeypatch.setattr(telemetry_module, "open_stream", lambda name: (_ for _ in ()).throw(RuntimeError("missing shm")))
    with pytest.raises(RuntimeError, match="missing shm"):
        t.save("signal", 1)

    unmanaged_file = tmp_path / "unmanaged.bin"
    np.array([1, 2, 3], dtype=np.int16).tofile(unmanaged_file)
    with pytest.raises(ValueError, match="please provide a dtype"):
        t.read(filename=str(unmanaged_file))

    broken_manifest = tmp_path / "broken_manifest.json"
    broken_manifest.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ValueError, match="Failed to read telemetry manifest"):
        telemetry_module.load_telemetry_manifest(broken_manifest)

    class _SHM:
        def __init__(self):
            self.shape = (2,)
            self.dtype = np.float32
            self.write_time = 10.0

        def read_new(self, timeout=None):
            return np.array([1.0, 2.0], dtype=np.float32)

    monkeypatch.setattr(telemetry_module, "open_stream", lambda name: _SHM())
    session_path = t.save("signal", 1)
    manifest = telemetry_module.load_telemetry_manifest(session_path)
    capture_path = Path(session_path) / manifest["streams"][0]["frames_file"]
    capture_path.unlink()

    with pytest.raises(FileNotFoundError, match="Telemetry capture file not found"):
        telemetry_module.load_telemetry_session(session_path)

    empty_telemetry = telemetry_module.Telemetry({"data_dir": str(tmp_path / "empty"), "functions": []})
    with pytest.raises(ValueError, match="no configured streams"):
        empty_telemetry.save_configured_streams(1)

    with pytest.raises(ValueError, match="No telemetry save is available"):
        empty_telemetry.read_last_save()
