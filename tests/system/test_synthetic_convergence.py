"""Closed-loop convergence regression for the synthetic SHWFS tutorial.

The synthetic example is the first thing every new user runs, so the loop
must demonstrably reduce the measured residual once the IM is calibrated
through the live pipeline and the loop is closed.
"""

import time
from pathlib import Path

import numpy as np

from pyRTC.Pipeline import RTCManager, clear_shms, open_stream
from pyRTC.config_schema import read_system_config

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_CONFIG_PATH = REPO_ROOT / "examples" / "synthetic_shwfs" / "config.yaml"
STREAMS = [
    "wfsRaw",
    "wfs",
    "signal",
    "signal2D",
    "wfc",
    "wfc2D",
    "psfShort",
    "psfLong",
    "strehl",
    "tiptilt",
]


def _residual_rms(signal_stream, samples: int = 30) -> float:
    values = []
    for _ in range(samples):
        frame = np.asarray(signal_stream.read_new(timeout=5.0), dtype=np.float64).ravel()
        values.append(float(np.sqrt(np.mean(frame * frame))))
    return float(np.mean(values))


def test_synthetic_loop_converges_after_calibration(tmp_path):
    clear_shms(STREAMS)

    config = read_system_config(SYNTHETIC_CONFIG_PATH)
    # Shorten calibration for test runtime; the example uses more iterations.
    config["loop"]["numItersIM"] = 400
    config["loop"]["IMFile"] = str(tmp_path / "im.npy")
    np.save(config["loop"]["IMFile"], np.zeros((98, 97), dtype=np.float32))

    manager = RTCManager.from_config(config, config_path=str(SYNTHETIC_CONFIG_PATH), mode="soft")
    try:
        manager.start()
        time.sleep(0.5)

        loop = manager.get_component("loop")
        signal_stream = open_stream("signal")

        loop.stop()
        loop.flatten()
        time.sleep(0.2)
        open_loop_rms = _residual_rms(signal_stream)

        loop.computeIM()
        loop.start()
        time.sleep(1.5)
        closed_loop_rms = _residual_rms(signal_stream)

        assert open_loop_rms > 0.0
        # Empirically the calibrated loop reaches ~0.15x; require 0.5x so the
        # assertion stays robust to scheduling noise on slow CI machines.
        assert closed_loop_rms < 0.5 * open_loop_rms, (
            f"closed-loop residual {closed_loop_rms:.4f} did not improve on "
            f"open-loop residual {open_loop_rms:.4f}"
        )
    finally:
        manager.stop()
        clear_shms(STREAMS)
