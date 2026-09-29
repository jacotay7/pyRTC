"""Closed-loop convergence and atmosphere correction on the HCIPy SHWFS example (#55).

Skipped when HCIPy is not installed (``pip install pyrtcao[hcipy]``). Streams
get a per-test prefix so this can run next to other system tests.
"""

import importlib.util
import os
import time
import uuid
from pathlib import Path

import numpy as np
import pytest

from pyrtc import clear_shms
from pyrtc.utils import read_yaml_file
from testsupport import prefix_system_streams

pytest.importorskip("hcipy")

EXAMPLE = (
    Path(__file__).resolve().parents[2] / "examples" / "hcipy" / "hcipy_shwfs_soft_rtc_example.py"
)


def _load_example():
    spec = importlib.util.spec_from_file_location("_hcipy_example", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mean_signal_rms(loop, samples=10):
    values = []
    for _ in range(samples):
        signal = np.asarray(loop.read_stream("signal", timeout=10.0), dtype=np.float64)
        values.append(float(np.sqrt(np.mean(signal * signal))))
    return float(np.mean(values))


def _mean_strehl(psf, samples=8):
    values = []
    for _ in range(samples):
        time.sleep(0.25)
        values.append(float(psf.strehl_ratio))
    return float(np.mean(values))


def _wait_until(predicate, deadline_seconds):
    deadline = time.monotonic() + deadline_seconds
    while True:
        value = predicate()
        if value[0] or time.monotonic() > deadline:
            return value[1]
        time.sleep(0.3)


def test_hcipy_loop_nulls_a_dm_aberration_and_corrects_the_atmosphere():
    module = _load_example()
    config = read_yaml_file(str(module.CONFIG_PATH))
    streams = prefix_system_streams(config, f"t{os.getpid()}_{uuid.uuid4().hex[:6]}_")
    clear_shms(streams)
    system = module.build_system(config)
    try:
        module.start_system(system)
        # Gain 0.15 keeps the integrator stable up to ~9 frames of DM-to-WFS
        # delay, which a loaded CI runner can reach (0.3 is unstable beyond ~5).
        module.prepare_loop(system, gain=0.15, use_atmosphere=False)
        loop, sim, psf = system["loop"], system["sim"], system["psf"]
        time.sleep(0.3)
        calibrated = _mean_signal_rms(loop)

        # 1. A static aberration put on the DM is nulled.
        rng = np.random.default_rng(55)
        aberration = np.zeros(loop.num_modes, dtype=loop.wfc_dtype)
        aberration[:20] = rng.uniform(-1.0, 1.0, 20) * loop.poke_amp
        loop.send_to_wfc(aberration)
        time.sleep(0.3)
        aberrated = _mean_signal_rms(loop)
        loop.start()
        closed = _wait_until(
            lambda: (lambda rms: (rms < 0.05 * aberrated, rms))(_mean_signal_rms(loop)), 30.0
        )

        # 2. On the atmosphere, closing the loop raises the (time-averaged) Strehl.
        loop.stop()
        loop.flatten()
        sim.add_atmosphere()
        time.sleep(1.0)
        open_strehl = _mean_strehl(psf)
        loop.start()
        time.sleep(2.0)
        closed_strehl = _mean_strehl(psf)
        loop.stop()
    finally:
        module.stop_system(system)
        clear_shms(streams)

    assert aberrated > 0.0
    assert calibrated < 0.05 * aberrated
    assert closed < 0.05 * aberrated, f"residual {closed:.4g} from {aberrated:.4g}"
    assert closed_strehl > open_strehl + 0.1, (open_strehl, closed_strehl)
