import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from pyRTC import Telemetry
from pyRTC.exporters import aotpy_export
from pyRTC.scripts import export_aotpy as export_aotpy_cli


pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason=(
        "AOTPy integration tests require Python 3.11+ in CI. "
        "Python 3.9 is unsupported by aotpy>=3.2 and Python 3.10 currently has upstream incompatibilities."
    ),
)


aotpy = pytest.importorskip("aotpy", reason="aotpy is not installed in this test environment")
RUN_AOTPY_ROUNDTRIP = os.environ.get("PYRTC_RUN_AOTPY_ROUNDTRIP") == "1"


def _make_test_session(monkeypatch, tmp_path, *, include_science=True):
    telemetry_module = importlib.import_module("pyRTC.Telemetry")

    class _SHM:
        def __init__(self, data, timestamp):
            self._data = np.asarray(data)
            self.metadata = np.array([0.0, timestamp], dtype=np.float64)

        def read(self):
            return self._data

    streams = {
        "wfs": (_SHM(np.arange(16, dtype=np.int32).reshape(4, 4), 100.0), [4, 4], np.int32),
        "signal": (_SHM(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), 101.0), [4], np.float32),
        "wfc": (_SHM(np.array([0.1, 0.2], dtype=np.float32), 102.0), [2], np.float32),
    }
    semantic_tags = {
        "wfs": ["wfs"],
        "signal": ["signal", "slopes"],
        "wfc": ["wfc", "control"],
    }
    stream_names = ["wfs", "signal", "wfc"]

    if include_science:
        streams["psfShort"] = (_SHM(np.ones((8, 8), dtype=np.int32), 103.0), [8, 8], np.int32)
        streams["psfLong"] = (_SHM(np.full((8, 8), 2.0, dtype=np.float64), 104.0), [8, 8], np.float64)
        semantic_tags["psfShort"] = ["psf", "science"]
        semantic_tags["psfLong"] = ["psf", "science"]
        stream_names.extend(["psfShort", "psfLong"])

    monkeypatch.setattr(aotpy_export, "_import_aotpy", lambda: aotpy)
    monkeypatch.setattr(telemetry_module, "initExistingShm", lambda name: streams[name])

    telemetry = Telemetry({"dataDir": str(tmp_path), "functions": []})
    return telemetry.save(
        stream_names,
        2 if include_science else 1,
        semanticTags=semantic_tags,
        config={
            "metadata": {"name": "Synthetic Export"},
            "slopes": {"type": "SHWFS", "signalType": "slopes"},
            "wfc": {"numModes": 2},
            "loop": {"gain": 0.35},
        },
        metadata={"operator": "pytest"},
    )


def test_telemetry_session_to_aotpy_maps_synthetic_streams(monkeypatch, tmp_path):
    session_path = _make_test_session(monkeypatch, tmp_path, include_science=True)

    system = aotpy_export.telemetry_session_to_aotpy(session_path)

    assert system.name == "Synthetic Export"
    assert system.ao_mode == "SCAO"
    assert len(system.wavefront_sensors) == 1
    assert len(system.wavefront_correctors) == 1
    assert len(system.loops) == 1
    assert len(system.scoring_cameras) == 2
    assert system.wavefront_sensors[0].measurements.data.shape == (2, 2, 2)
    assert system.loops[0].commands.data.shape == (2, 2)


@pytest.mark.skipif(
    not RUN_AOTPY_ROUNDTRIP,
    reason="Set PYRTC_RUN_AOTPY_ROUNDTRIP=1 to run the upstream AOTPy FITS round-trip integration test.",
)
def test_aotpy_export_roundtrip_integration(monkeypatch, tmp_path):
    session_path = _make_test_session(monkeypatch, tmp_path, include_science=True)

    output_path = tmp_path / "exported_session.fits"
    exported = aotpy_export.export_telemetry_session_to_aotpy(session_path, output_path)
    reopened = aotpy.AOSystem.read_from_file(exported)

    assert reopened.name == "Synthetic Export"
    assert len(reopened.wavefront_sensors) == 1
    assert len(reopened.wavefront_correctors) == 1
    assert len(reopened.loops) == 1


def test_aotpy_export_surfaces_missing_optional_dependency(monkeypatch, tmp_path):
    telemetry_module = importlib.import_module("pyRTC.Telemetry")

    class _SHM:
        def __init__(self):
            self.metadata = np.array([0.0, 10.0], dtype=np.float64)

        def read(self):
            return np.array([1.0, 2.0], dtype=np.float32)

    monkeypatch.setattr(telemetry_module, "initExistingShm", lambda name: (_SHM(), [2], np.float32))
    monkeypatch.setattr(aotpy_export.importlib, "import_module", lambda name: (_ for _ in ()).throw(ImportError("missing aotpy")))

    telemetry = Telemetry({"dataDir": str(tmp_path), "functions": []})
    session_path = telemetry.save("signal", 1)

    with pytest.raises(RuntimeError, match="optional 'aotpy' dependency"):
        aotpy_export.telemetry_session_to_aotpy(session_path)


def test_export_aotpy_cli_writes_default_output(monkeypatch, tmp_path, capsys):
    session_path = tmp_path / "session_example"
    session_path.mkdir()
    default_output = session_path.parent / f"{session_path.name}.fits"

    def _fake_export(session, output_path, **kwargs):
        Path(output_path).write_text("fake aotpy file", encoding="utf-8")
        return Path(output_path)

    monkeypatch.setattr(export_aotpy_cli, "export_telemetry_session_to_aotpy", _fake_export)

    code = export_aotpy_cli.main([str(session_path)])
    captured = capsys.readouterr()

    assert code == 0
    assert "AOTPy export written" in captured.out
    assert default_output.exists()
