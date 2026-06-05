"""Shared AOTPy export helpers for the synthetic SHWFS tutorials."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from pyRTC import Telemetry
from pyRTC.exporters.aotpy_export import export_telemetry_session_to_aotpy


AOTPY_CAPTURE_STREAMS = ["wfs", "signal", "wfc", "psfShort"]
AOTPY_SEMANTIC_TAGS = {
    "wfs": ["wfs"],
    "signal": ["signal", "slopes"],
    "wfc": ["wfc", "control"],
    "psfShort": ["psf", "science"],
}


def import_aotpy_or_raise(repo_root: Path):
    try:
        return importlib.import_module("aotpy")
    except ImportError:
        sibling_repo = repo_root.parent / "aotpy"
        if sibling_repo.exists() and str(sibling_repo) not in sys.path:
            sys.path.insert(0, str(sibling_repo))
        try:
            return importlib.import_module("aotpy")
        except ImportError as exc:
            raise RuntimeError(
                "AOTPy export requested, but aotpy is not installed. "
                "Install it with 'pip install pyrtcao[aotpy]' or make the sibling aotpy repo available."
            ) from exc


def export_synthetic_session_to_aotpy(*, repo_root: Path, config: dict, config_path: Path, mode_label: str) -> tuple[str, Path, object]:
    aotpy = import_aotpy_or_raise(repo_root)

    telemetry_dir = repo_root / "examples" / "synthetic_shwfs" / "telemetry"
    export_path = telemetry_dir / f"synthetic_shwfs_{mode_label}.fits"
    telem = Telemetry({"dataDir": str(telemetry_dir), "functions": []})
    session_path = telem.save(
        AOTPY_CAPTURE_STREAMS,
        10,
        semanticTags=AOTPY_SEMANTIC_TAGS,
        config=config,
        config_path=config_path,
        metadata={"name": f"Synthetic SHWFS {mode_label.title()} Example"},
    )
    exported_path = export_telemetry_session_to_aotpy(
        session_path,
        export_path,
        system_name=f"Synthetic SHWFS {mode_label.title()} Example",
        overwrite=True,
    )
    reopened_system = aotpy.AOSystem.read_from_file(exported_path)
    return session_path, exported_path, reopened_system