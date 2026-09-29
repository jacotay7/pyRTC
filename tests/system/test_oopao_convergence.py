"""Closed-loop convergence for the OOPAO SHWFS and PYWFS examples.

Each example is calibrated through its own ``prepare_loop`` (atmosphere off,
DM round trip confirmed, reference slopes taken, IM measured). A random
aberration is then put on the DM and the loop must null it.

OOPAO is not on PyPI; the test runs when it can be imported (e.g. a clone on
``PYTHONPATH``, see the example docs) and is skipped otherwise. Streams get a
per-test prefix so this can run next to other system tests.
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

try:
    import pyrtc.hardware.oopao_interface  # noqa: F401  (imports OOPAO)
except Exception as exc:  # OOPAO missing or incomplete install
    pytest.skip(f"OOPAO is not importable: {exc}", allow_module_level=True)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = {
    "shwfs": REPO_ROOT / "examples" / "shwfs" / "shwfs_oopao_soft_rtc_example.py",
    "pywfs": REPO_ROOT / "examples" / "pywfs" / "pywfs_oopao_soft_rtc_example.py",
}


def _load_example(name: str):
    spec = importlib.util.spec_from_file_location(f"_oopao_example_{name}", EXAMPLES[name])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mean_signal_rms(loop, samples: int = 10) -> float:
    values = []
    for _ in range(samples):
        signal = np.asarray(loop.read_stream("signal", timeout=10.0), dtype=np.float64)
        values.append(float(np.sqrt(np.mean(signal * signal))))
    return float(np.mean(values))


@pytest.mark.parametrize("example", sorted(EXAMPLES))
def test_oopao_loop_converges_after_calibration(example):
    module = _load_example(example)
    config = read_yaml_file(str(module.CONFIG_PATH))
    assert config["oopao"]["use_atmosphere"] is False
    streams = prefix_system_streams(config, f"t{os.getpid()}_{uuid.uuid4().hex[:6]}_")

    clear_shms(streams)
    build_kwargs = {"oopao_param_file": module.PARAM_PATH}
    if example == "pywfs":
        build_kwargs["use_kl_basis"] = False
    system = module.build_system(config, **build_kwargs)
    try:
        module.start_system(system)
        loop = system["loop"]
        module.prepare_loop(
            system, gain=config["loop"]["gain"], poke_amp=loop.poke_amp, compute_im=True
        )

        time.sleep(0.2)
        calibrated_rms = _mean_signal_rms(loop)

        rng = np.random.default_rng(88)
        aberration = np.zeros(loop.num_modes, dtype=loop.wfc_dtype)
        aberration[:20] = rng.uniform(-1.0, 1.0, 20) * loop.poke_amp
        loop.send_to_wfc(aberration)
        time.sleep(0.2)
        aberrated_rms = _mean_signal_rms(loop)

        loop.start()
        deadline = time.monotonic() + 20.0
        time.sleep(1.0)
        while True:
            closed_rms = _mean_signal_rms(loop)
            if closed_rms < 0.05 * aberrated_rms or time.monotonic() > deadline:
                break
            time.sleep(0.5)
        loop.stop()
    finally:
        module.stop_system(system)
        clear_shms(streams)

    assert aberrated_rms > 0.0
    # Reference slopes make the flat, unaberrated system read (near) zero.
    assert calibrated_rms < 0.05 * aberrated_rms
    assert closed_rms < 0.05 * aberrated_rms, (
        f"{example}: closed-loop residual {closed_rms:.4g} did not converge "
        f"from aberrated residual {aberrated_rms:.4g}"
    )
