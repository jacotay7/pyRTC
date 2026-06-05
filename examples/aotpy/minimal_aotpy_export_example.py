"""Minimal pyRTC -> AOTPy export example.

This example is intentionally small and self-contained.
It does not require a running RTC system or shared-memory streams.
Instead it writes one tiny telemetry session on disk, exports that session
through the Issue 06 AOTPy exporter, then reopens the FITS file to verify the
round-trip.

Run it from the repository root with:

    python examples/aotpy/minimal_aotpy_export_example.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import aotpy
except ImportError:
    sibling_repo = REPO_ROOT.parent / "aotpy"
    if sibling_repo.exists() and str(sibling_repo) not in sys.path:
        sys.path.insert(0, str(sibling_repo))
    import aotpy


from pyRTC.exporters.aotpy_export import export_telemetry_session_to_aotpy, telemetry_session_to_aotpy


EXAMPLE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EXAMPLE_DIR / "output"
SESSION_DIR = OUTPUT_DIR / "minimal_session"
EXPORT_PATH = OUTPUT_DIR / "minimal_session.fits"


def write_stream(session_dir: Path, name: str, frames: np.ndarray, timestamps: np.ndarray, semantic_tags: list[str]) -> dict:
    stream_dir = session_dir / name
    stream_dir.mkdir(parents=True, exist_ok=False)

    frames_path = stream_dir / "frames.npy"
    timestamps_path = stream_dir / "timestamps.npy"
    metadata_path = stream_dir / "metadata.json"

    np.save(frames_path, np.asarray(frames))
    np.save(timestamps_path, np.asarray(timestamps, dtype=np.float64))

    metadata = {
        "name": name,
        "dtype": np.asarray(frames).dtype.name,
        "shape": list(np.asarray(frames).shape[1:]),
        "frame_count": int(np.asarray(frames).shape[0]),
        "timestamp_unit": "unix_seconds",
        "sampling": {"mode": "every_frame"},
        "semantic_tags": semantic_tags,
        "capture_label": "minimal-example",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "name": name,
        "dtype": metadata["dtype"],
        "shape": metadata["shape"],
        "frame_count": metadata["frame_count"],
        "frames_file": str(frames_path.relative_to(session_dir)),
        "timestamps_file": str(timestamps_path.relative_to(session_dir)),
        "metadata_file": str(metadata_path.relative_to(session_dir)),
        "sampling": metadata["sampling"],
        "semantic_tags": semantic_tags,
    }


def create_fake_telemetry_stream(session_dir: Path) -> Path:
    """Create one tiny self-contained telemetry session for the tutorial.

    The example itself is about the export API, so all of the setup noise lives
    here instead of in ``main()``.
    """

    timestamps = np.array([1_700_000_000.0, 1_700_000_000.01], dtype=np.float64)
    stream_records = [
        write_stream(
            session_dir,
            "wfs",
            frames=np.arange(2 * 4 * 4, dtype=np.int32).reshape(2, 4, 4),
            timestamps=timestamps,
            semantic_tags=["wfs"],
        ),
        write_stream(
            session_dir,
            "signal",
            frames=np.array([[0.1, -0.1, 0.2, -0.2], [0.0, -0.05, 0.1, -0.1]], dtype=np.float32),
            timestamps=timestamps,
            semantic_tags=["signal", "slopes"],
        ),
        write_stream(
            session_dir,
            "wfc",
            frames=np.array([[0.02, -0.02], [0.01, -0.01]], dtype=np.float32),
            timestamps=timestamps,
            semantic_tags=["wfc", "control"],
        ),
        write_stream(
            session_dir,
            "psfShort",
            frames=np.stack(
                [
                    np.eye(8, dtype=np.int32),
                    np.fliplr(np.eye(8, dtype=np.int32)),
                ]
            ),
            timestamps=timestamps,
            semantic_tags=["psf", "science"],
        ),
    ]

    session_manifest = {
        "schema_version": 1,
        "session_id": "minimal-example-session",
        "created_at": "2026-03-09T12:00:00Z",
        "pyRTC_version": "1.0.0",
        "host": {
            "hostname": "example-host",
            "platform": "example-platform",
            "python_version": sys.version.split()[0],
        },
        "config_path": None,
        "config": {
            "metadata": {"name": "Minimal AOTPy Export Example"},
            "slopes": {"type": "SHWFS", "signalType": "slopes"},
            "wfc": {"numModes": 2},
            "loop": {"gain": 0.35},
        },
        "metadata": {"operator": "example-script"},
        "streams": stream_records,
    }
    (session_dir / "session.json").write_text(json.dumps(session_manifest, indent=2, sort_keys=True), encoding="utf-8")
    return session_dir


def main() -> int:
    # Start from a clean output directory so every run is reproducible.
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    SESSION_DIR.mkdir(parents=True, exist_ok=False)

    # Create one fake telemetry session so the rest of the example can focus on
    # the two lines you actually care about in normal usage.
    create_fake_telemetry_stream(SESSION_DIR)

    # Build the in-memory AOTPy object directly from the session.
    system = telemetry_session_to_aotpy(SESSION_DIR)

    # Export the session to an AOTPy FITS file.
    exported_path = export_telemetry_session_to_aotpy(SESSION_DIR, EXPORT_PATH)

    # Reopen the written FITS file and confirm that AOTPy can read it back.
    reopened_system = aotpy.AOSystem.read_from_file(exported_path)

    print(f"Session directory: {SESSION_DIR}")
    print(f"Exported file:     {exported_path}")
    print(f"AO mode:           {system.ao_mode}")
    print(f"System name:       {system.name}")
    print(f"Wavefront sensors: {len(reopened_system.wavefront_sensors)}")
    print(f"Correctors:        {len(reopened_system.wavefront_correctors)}")
    print(f"Loops:             {len(reopened_system.loops)}")
    print(f"Science cameras:   {len(reopened_system.scoring_cameras)}")
    print("Round-trip read succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())