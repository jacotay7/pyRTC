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
from testsupport import drain_in_flight_signal, prefix_system_streams

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = {
    "shwfs": REPO_ROOT / "examples" / "shwfs" / "shwfs_specula_soft_rtc_example.py",
    "pywfs": REPO_ROOT / "examples" / "pywfs" / "pywfs_specula_soft_rtc_example.py",
}


def _load_example(name: str):
    spec = importlib.util.spec_from_file_location(f"_specula_example_{name}", EXAMPLES[name])
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
def test_specula_loop_converges_after_calibration(example, tmp_path, monkeypatch):
    module = _load_example(example)
    config = read_yaml_file(str(module.CONFIG_PATH))
    assert config["specula"]["use_atmosphere"] is False
    # Fewer averaged frames per poke keeps the runtime down; the simulation
    # is noise free, so the IM is unchanged.
    config["loop"]["num_iters_im"] = 2
    streams = prefix_system_streams(config, f"t{os.getpid()}_{uuid.uuid4().hex[:6]}_")
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

        drain_in_flight_signal(loop)
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


def _mean_strehl(sim, samples: int = 20) -> float:
    """Mean of SPECULA's instantaneous, on-axis PSF Strehl over ``samples`` frames."""
    values = []
    for _ in range(samples):
        values.append(float(sim.context._cached_psf_strehl))
        time.sleep(0.05)
    return float(np.mean(values))


@pytest.mark.parametrize("example", sorted(EXAMPLES))
def test_specula_loop_corrects_turbulence(example, tmp_path, monkeypatch):
    # With the atmosphere on, closing the loop must raise the Strehl well
    # above the open-loop value (#175). Edge actuators missing from the M2C
    # and an IM conditioning that truncated tip/tilt once made it fall.
    module = _load_example(example)
    config = read_yaml_file(str(module.CONFIG_PATH))
    config["specula"]["use_atmosphere"] = True
    config["loop"]["num_iters_im"] = 2
    streams = prefix_system_streams(config, f"t{os.getpid()}_{uuid.uuid4().hex[:6]}_")
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
        time.sleep(0.5)
        open_strehl = _mean_strehl(system["sim"])

        loop.start()
        # The loop iterates once per WFS frame; wait for the first good window
        # against a generous deadline for slow or loaded machines (the PYWFS
        # runs ~10 frames/s), then require the correction to hold. A broken
        # loop wanders and can cross the bar for a moment.
        deadline = time.monotonic() + 40.0
        time.sleep(2.0)
        while _mean_strehl(system["sim"]) <= 0.3 and time.monotonic() < deadline:
            pass
        closed_strehl = [_mean_strehl(system["sim"]) for _ in range(3)]
        loop.stop()
    finally:
        module.stop_system(system)
        clear_shms(streams)

    # Locally the K-band Strehl goes from ~0.03-0.07 open to ~0.45-0.7 closed.
    assert min(closed_strehl) > 0.3, (
        f"{example}: closed-loop Strehl {closed_strehl} (open {open_strehl:.3f})"
    )
    assert min(closed_strehl) > 3.0 * open_strehl


def _cupy_device_available() -> bool:
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@pytest.mark.gpu
@pytest.mark.skipif(not _cupy_device_available(), reason="no CuPy CUDA device")
@pytest.mark.parametrize("example", sorted(EXAMPLES))
def test_specula_gpu_frames_follow_the_atmosphere(example, tmp_path):
    # On a GPU, AtmoPropagation replays a CUDA graph captured before the
    # bridge attaches the atmosphere; without a new capture every frame
    # stayed flat.
    import gc

    import cupy

    from pyrtc.hardware.specula_interface import SPECULASystemContext

    # Earlier tests in the process (OOPAO picks CuPy when it is installed)
    # keep GPU memory pooled; on a small GPU this test then fails with cuSOLVER
    # internal errors and leaves the device unusable for the torch GPU tests.
    cupy.get_default_memory_pool().free_all_blocks()
    module = _load_example(example)
    param = read_yaml_file(str(module.PARAM_PATH))
    param["speculaInit"]["device_idx"] = 0
    param["main"]["root_dir"] = str(tmp_path / "specula")
    config = read_yaml_file(str(module.CONFIG_PATH))
    context = SPECULASystemContext({"param": param, "use_atmosphere": True}, config)

    def propagated_phase(ctx):
        ctx.capture_wfs()
        ef = ctx.prop.outputs["out_on_axis_source_ef"]
        return np.asarray(ctx._bindings.cpu_array(ef.phaseInNm), dtype=np.float64)

    try:
        first = propagated_phase(context)
        for _ in range(5):
            later = propagated_phase(context)
    finally:
        del context
        gc.collect()
        cupy.get_default_memory_pool().free_all_blocks()
    assert np.std(later) > 10.0  # nm; turbulence is hundreds of nm rms
    assert not np.array_equal(first, later)
