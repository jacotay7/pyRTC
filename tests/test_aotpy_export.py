import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

from pyRTC import Telemetry
from pyRTC.exporters import aotpy_export
from pyRTC.scripts import export_aotpy as export_aotpy_cli


REPO_ROOT = Path(__file__).resolve().parents[1]
AOTPY_REPO_ROOT = REPO_ROOT.parent / "aotpy"


def _ensure_local_aotpy_importable():
    astropy_module = sys.modules.get("astropy")
    if astropy_module is None or getattr(astropy_module, "__file__", None) is None:
        for module_name in ("astropy.io.fits", "astropy.io", "astropy"):
            sys.modules.pop(module_name, None)
        importlib.import_module("astropy.io.fits")

    if str(AOTPY_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(AOTPY_REPO_ROOT))


def test_telemetry_session_to_aotpy_maps_synthetic_streams(monkeypatch, tmp_path):
    _ensure_local_aotpy_importable()
    import aotpy
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
        "psfShort": (_SHM(np.ones((8, 8), dtype=np.int32), 103.0), [8, 8], np.int32),
        "psfLong": (_SHM(np.full((8, 8), 2.0, dtype=np.float64), 104.0), [8, 8], np.float64),
    }
    monkeypatch.setattr(aotpy_export, "_import_aotpy", lambda: aotpy)
    monkeypatch.setattr(telemetry_module, "initExistingShm", lambda name: streams[name])

    telemetry = Telemetry({"dataDir": str(tmp_path), "functions": []})
    session_path = telemetry.save(
        ["wfs", "signal", "wfc", "psfShort", "psfLong"],
        2,
        semanticTags={
            "wfs": ["wfs"],
            "signal": ["signal", "slopes"],
            "wfc": ["wfc", "control"],
            "psfShort": ["psf", "science"],
            "psfLong": ["psf", "science"],
        },
        config={
            "metadata": {"name": "Synthetic Export"},
            "slopes": {"type": "SHWFS", "signalType": "slopes"},
            "wfc": {"numModes": 2},
            "loop": {"gain": 0.35},
        },
        metadata={"operator": "pytest"},
    )

    system = aotpy_export.telemetry_session_to_aotpy(session_path)

    assert system.name == "Synthetic Export"
    assert system.ao_mode == "SCAO"
    assert len(system.wavefront_sensors) == 1
    assert len(system.wavefront_correctors) == 1
    assert len(system.loops) == 1
    assert len(system.scoring_cameras) == 2
    assert system.wavefront_sensors[0].measurements.data.shape == (2, 2, 2)
    assert system.loops[0].commands.data.shape == (2, 2)

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
    _ensure_local_aotpy_importable()
    import aotpy
    telemetry_module = importlib.import_module("pyRTC.Telemetry")

    class _SHM:
        def __init__(self, data, timestamp):
            self._data = np.asarray(data)
            self.metadata = np.array([0.0, timestamp], dtype=np.float64)

        def read(self):
            return self._data

    streams = {
        "signal": (_SHM(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), 10.0), [4], np.float32),
        "wfc": (_SHM(np.array([0.1, 0.2], dtype=np.float32), 11.0), [2], np.float32),
    }
    monkeypatch.setattr(aotpy_export, "_import_aotpy", lambda: aotpy)
    monkeypatch.setattr(telemetry_module, "initExistingShm", lambda name: streams[name])

    telemetry = Telemetry({"dataDir": str(tmp_path), "functions": []})
    session_path = telemetry.save(
        ["signal", "wfc"],
        1,
        semanticTags={"signal": ["signal"], "wfc": ["wfc", "control"]},
        config={
            "slopes": {"type": "SHWFS", "signalType": "slopes"},
            "wfc": {"numModes": 2},
        },
    )

    code = export_aotpy_cli.main([session_path])
    captured = capsys.readouterr()
    default_output = Path(session_path).parent / f"{Path(session_path).name}.fits"

    assert code == 0
    assert "AOTPy export written" in captured.out
    assert default_output.exists()