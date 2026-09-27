"""Closed-loop convergence regression for the SPECULA soft-RTC examples (#39).

Each example is calibrated through its own ``prepare_loop`` (atmosphere
removed, reference slopes, push-pull IM), a static modal aberration is put on
the DM, and the loop must null it: the residual drops to a small fraction of
the aberrated residual and the DM command returns to (near) flat instead of
running away.

Streams get a per-test prefix so this can run next to other system tests.
"""

import importlib.util
import os
import time
import uuid
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("specula")

from pyrtc import clear_shms  # noqa: E402
from pyrtc.utils import read_yaml_file  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = {
    "shwfs": REPO_ROOT / "examples" / "shwfs" / "shwfs_specula_soft_rtc_example.py",
    "pywfs": REPO_ROOT / "examples" / "pywfs" / "pywfs_specula_soft_rtc_example.py",
}
SECTIONS = ("wfs", "slopes", "loop", "wfc", "psf")


def _load_example(name: str):
    spec = importlib.util.spec_from_file_location(f"_specula_example_{name}", EXAMPLES[name])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prefix_streams(config: dict, prefix: str) -> list[str]:
    names = set()
    for section in SECTIONS:
        conf = config.get(section)
        if not isinstance(conf, dict):
            continue
        for key in ("input_streams", "output_streams"):
            aliases = conf.get(key) or {}
            conf[key] = {logical: f"{prefix}{target}" for logical, target in aliases.items()}
            names.update(conf[key].values())
    return sorted(names)


def _mean_signal_rms(loop, samples: int = 10) -> float:
    values = []
    for _ in range(samples):
        signal = np.asarray(loop.read_stream("signal", timeout=10.0), dtype=np.float64)
        values.append(float(np.sqrt(np.mean(signal * signal))))
    return float(np.mean(values))


@pytest.mark.parametrize("example", sorted(EXAMPLES))
def test_specula_loop_converges_after_calibration(example, tmp_path, monkeypatch):
    module = _load_example(example)
    config = read_yaml_file(str(module.CONFIG_PATH))
    assert config["specula"]["use_atmosphere"] is False
    # Fewer averaged frames per poke keeps the runtime down; the simulation
    # is noise free, so the IM is unchanged.
    config["loop"]["num_iters_im"] = 2
    streams = _prefix_streams(config, f"t{os.getpid()}_{uuid.uuid4().hex[:6]}_")
    specula_param = read_yaml_file(str(module.PARAM_PATH))
    specula_param["main"]["root_dir"] = str(tmp_path / "specula")
    param_file = tmp_path / "params.yaml"
    import yaml

    param_file.write_text(yaml.safe_dump(specula_param))
    monkeypatch.chdir(tmp_path)

    clear_shms(streams)
    system = module.build_system(config, specula_param_file=param_file)
    try:
        module.start_system(system)
        module.prepare_loop(system)
        loop = system["loop"]
        dm = system["dm"]

        time.sleep(0.2)
        calibrated_rms = _mean_signal_rms(loop)

        rng = np.random.default_rng(39)
        aberration = np.zeros(loop.num_modes, dtype=loop.wfc_dtype)
        aberration[:20] = rng.uniform(-1.0, 1.0, 20) * loop.poke_amp
        loop.send_to_wfc(aberration)
        time.sleep(0.2)
        aberrated_rms = _mean_signal_rms(loop)

        aberration_dm_rms = float(np.sqrt(np.mean(np.square(dm.M2C @ aberration))))
        loop.start()
        # Poll instead of sampling once after a fixed time: on a loaded machine
        # the loop iterates more slowly, so give it a generous deadline.
        deadline = time.monotonic() + 20.0
        time.sleep(1.0)
        while True:
            closed_rms = _mean_signal_rms(loop)
            dm_rms = float(np.sqrt(np.mean(np.square(dm.current_shape))))
            converged = closed_rms < 0.05 * aberrated_rms and dm_rms < 0.1 * aberration_dm_rms
            if converged or time.monotonic() > deadline:
                break
            time.sleep(0.5)
        loop.stop()
    finally:
        module.stop_system(system)
        clear_shms(streams)

    # Reference slopes make the flat, unaberrated system read (near) zero.
    assert calibrated_rms < 0.05 * aberrated_rms
    assert aberrated_rms > 0.0
    # Empirically the loop reaches < 1e-3 of the aberrated residual within a
    # few seconds; the thresholds and the 20 s deadline leave room for slow or
    # loaded machines.
    assert closed_rms < 0.05 * aberrated_rms, (
        f"{example}: closed-loop residual {closed_rms:.4g} did not converge "
        f"from aberrated residual {aberrated_rms:.4g}"
    )
    assert dm_rms < 0.1 * aberration_dm_rms, (
        f"{example}: DM command rms {dm_rms:.4g} did not return towards flat "
        f"(aberration rms {aberration_dm_rms:.4g})"
    )
