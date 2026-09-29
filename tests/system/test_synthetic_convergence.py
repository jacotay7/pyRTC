"""Closed-loop convergence regression for the synthetic SHWFS tutorial.

The synthetic example is the first thing every new user runs, so the loop
must demonstrably reduce the measured residual once the IM is calibrated
through the live pipeline and the loop is closed.
"""

import os
import time
import uuid
from pathlib import Path

import numpy as np

from pyrtc import RTCManager, clear_shms, open_stream
from pyrtc.config_schema import read_system_config
from testsupport import prefix_system_streams

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_CONFIG_PATH = REPO_ROOT / "examples" / "synthetic_shwfs" / "config.yaml"


def _residual_rms(signal_stream, samples: int = 30) -> float:
    values = []
    for _ in range(samples):
        frame = np.asarray(signal_stream.read_new(timeout=5.0), dtype=np.float64).ravel()
        values.append(float(np.sqrt(np.mean(frame * frame))))
    return float(np.mean(values))


def test_synthetic_loop_converges_after_calibration(tmp_path):
    config = read_system_config(SYNTHETIC_CONFIG_PATH)
    # Private stream names, so this can run alongside other systems or tests.
    streams = prefix_system_streams(config, f"t{os.getpid()}_{uuid.uuid4().hex[:6]}_")
    # Shorten calibration for test runtime; the example uses more iterations.
    config["loop"]["num_iters_im"] = 400
    config["loop"]["im_file"] = str(tmp_path / "im.npy")
    np.save(config["loop"]["im_file"], np.zeros((98, 97), dtype=np.float32))

    manager = RTCManager.from_config(config, config_path=str(SYNTHETIC_CONFIG_PATH), mode="soft")
    try:
        manager.start()
        time.sleep(0.5)

        loop = manager.get_component("loop")
        # Every component must read the renamed streams; a stream read via an
        # undeclared default name would silently break the closed loop.
        assert manager.get_component("wfs").input_stream_name("wfc") in streams
        assert manager.get_component("psf").input_stream_name("signal") in streams
        signal_stream = open_stream(loop.input_stream_name("signal"))

        loop.stop()
        loop.flatten()
        time.sleep(0.2)
        open_loop_rms = _residual_rms(signal_stream)

        loop.compute_im()
        loop.start()
        # Poll instead of sampling once after a fixed delay: on a loaded
        # machine the loop iterates more slowly, so allow a generous deadline.
        deadline = time.monotonic() + 20.0
        time.sleep(1.0)
        while True:
            closed_loop_rms = _residual_rms(signal_stream)
            if closed_loop_rms < 0.5 * open_loop_rms or time.monotonic() > deadline:
                break
            time.sleep(0.5)

        assert open_loop_rms > 0.0
        # Empirically the calibrated loop reaches ~0.15x; require 0.5x so the
        # assertion stays robust to scheduling noise on slow CI machines.
        assert closed_loop_rms < 0.5 * open_loop_rms, (
            f"closed-loop residual {closed_loop_rms:.4f} did not improve on "
            f"open-loop residual {open_loop_rms:.4f}"
        )
    finally:
        manager.stop()
        clear_shms(streams)
