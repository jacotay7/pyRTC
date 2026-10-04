"""Construction warms the per-frame numba kernels without publishing anything.

The first call of a numba kernel compiles it (or loads it from the cache),
which used to stall the first frame after ``start()`` by 0.1 to 1 s. Each
component's ``warmup()`` (run by ``__init__``) calls its kernels on scratch
arrays. These tests check that it dispatches with exactly the argument types
the real worker call uses (otherwise numba compiles a second specialisation on
the first frame) and that it never writes a stream. The timing side is in
``tests/perf/test_first_iteration_bench.py``.
"""

import importlib
import inspect
import uuid

import numba
import numpy as np
import pytest

from testsupport import private_stream

loop_mod = importlib.import_module("pyrtc.loop")
slopes_mod = importlib.import_module("pyrtc.slopes_process")
wfc_mod = importlib.import_module("pyrtc.wavefront_corrector")
from pyrtc.streams import clear_shms  # noqa: E402


def _record_kernels(monkeypatch, module, names):
    """Replace kernels in ``module`` with wrappers logging their numba argument types."""

    log = []
    for name in names:
        kernel = getattr(module, name)
        signature = inspect.signature(kernel.py_func)

        def recorder(*args, _kernel=kernel, _name=name, _signature=signature, **kwargs):
            bound = _signature.bind(*args, **kwargs)
            log.append((_name, tuple(numba.typeof(v) for v in bound.arguments.values())))
            return _kernel(*args, **kwargs)

        monkeypatch.setattr(module, name, recorder)
    return log


def _loop_streams(num_signals=12, num_modes=6):
    signal = private_stream("signal", (num_signals,), np.float32)
    wfc = private_stream("wfc", (num_modes,), np.float32)
    conf = {
        "input_streams": {"signal": signal.name},
        "output_streams": {"wfc": wfc.name},
        "num_dropped_modes": 1,
        "leaky_gain": 0.05,
    }
    return signal, wfc, conf


@pytest.mark.parametrize(
    "function", ["leaky_integrator", "standard_integrator", "pid_integrator", "pid_integrator_pol"]
)
def test_loop_warmup_compiles_what_the_integrator_calls(monkeypatch, function):
    log = _record_kernels(monkeypatch, loop_mod, ("leaky_integrator_numba", "comp_correction"))
    signal, _, conf = _loop_streams()
    loop = loop_mod.Loop(conf)
    try:
        warmed = set(log)
        assert {name for name, _ in warmed} == {"leaky_integrator_numba", "comp_correction"}
        log.clear()
        loop.im = np.random.default_rng(0).normal(size=(12, 6)).astype(np.float32)
        loop.compute_cm()
        signal.write(np.ones(12, dtype=np.float32))

        getattr(loop, function)()

        assert log, "the integrator should call a numba kernel"
        assert set(log) <= warmed
    finally:
        loop.close()


def test_loop_construction_and_warmup_publish_nothing():
    signal, wfc, conf = _loop_streams()
    command = np.arange(6, dtype=np.float32)
    wfc.write(command, frame_id=41)
    counts = (signal.count, wfc.count)

    loop = loop_mod.Loop(conf)
    try:
        frame_id = loop.frame_id
        loop.warmup()
        assert (signal.count, wfc.count) == counts
        np.testing.assert_array_equal(wfc.read(), command)
        assert wfc.read_publication().frame_id == 41
        assert loop.frame_id == frame_id
    finally:
        loop.close()


def test_loop_warmup_failure_is_logged_not_raised(monkeypatch, caplog):
    def broken(*args, **kwargs):
        raise RuntimeError("no compiler")

    _, _, conf = _loop_streams()
    monkeypatch.setattr(loop_mod, "comp_correction", broken)
    loop = loop_mod.Loop(conf)
    loop.close()
    assert "Could not warm up comp_correction" in caplog.text


SLOPES_KERNELS = (
    "compute_slopes_pywfs_optim_numba",
    "compute_slopes_shwfs_optim_numba",
    "compute_slopes_shwfs_wcog_numba",
    "build_shwfs_wcog_weights_numba",
)


def _slopes_conf(wfs_name, outputs, wfs_type, centroider):
    conf = {
        "type": wfs_type,
        "signal_type": "slopes",
        "input_streams": {"wfs": wfs_name},
        "output_streams": outputs,
    }
    if wfs_type == "PYWFS":
        conf.update({"pupils": ["16,16", "16,48", "48,16", "48,48"], "pupils_radius": 10})
    else:
        conf.update(
            {
                "sub_ap_spacing": 8,
                "sub_ap_offset_x": 0,
                "sub_ap_offset_y": 0,
                "centroider": centroider,
            }
        )
    return conf


@pytest.mark.parametrize(
    "wfs_type, centroider", [("PYWFS", None), ("SHWFS", "cog"), ("SHWFS", "wcog")]
)
def test_slopes_warmup_compiles_what_compute_signal_calls_and_publishes_nothing(
    monkeypatch, wfs_type, centroider
):
    log = _record_kernels(monkeypatch, slopes_mod, SLOPES_KERNELS)
    wfs = private_stream("wfs", (64, 64), np.float32)
    frame = np.random.default_rng(1).uniform(0, 100, size=(64, 64)).astype(np.float32)
    wfs.write(frame)
    suffix = uuid.uuid4().hex[:8]
    outputs = {"signal": f"wsig_{suffix}", "signal_2d": f"wsig2d_{suffix}"}
    proc = slopes_mod.SlopesProcess(_slopes_conf(wfs.name, outputs, wfs_type, centroider))
    try:
        warmed = set(log)
        assert warmed
        counts = (proc.signal.count, proc.signal_2d.count)
        assert counts == (0, 0), "construction must not publish"
        proc.warmup()
        assert (proc.signal.count, proc.signal_2d.count) == counts
        assert wfs.count == 1
        log.clear()

        proc.compute_signal()

        assert log, "compute_signal should call a numba kernel"
        assert set(log) <= warmed
        assert proc.signal.count == 1
        assert np.any(proc.signal.read() != 0)
    finally:
        proc.close()
        clear_shms(list(outputs.values()))


def test_wavefront_corrector_warmup_compiles_what_send_to_hardware_calls(monkeypatch, tmp_path):
    monkeypatch.setattr(wfc_mod, "create_stream", private_stream)
    log = _record_kernels(monkeypatch, wfc_mod, ("ModaltoZonalWithFlat",))
    conf = {
        "name": "wfc",
        "num_actuators": 9,
        "num_modes": 4,
        "functions": [],
        "save_file": str(tmp_path / "shape.npy"),
    }
    wfc = wfc_mod.WavefrontCorrector(conf)
    try:
        warmed = set(log)
        assert warmed
        stream = wfc.correction_vector
        count = stream.count
        wfc.warmup()
        assert stream.count == count
        np.testing.assert_array_equal(wfc.current_shape, np.zeros(9, dtype=np.float32))
        log.clear()

        stream.write(np.ones(4, dtype=np.float32))
        wfc.send_to_hardware()

        assert log
        assert set(log) <= warmed
        assert np.any(wfc.current_shape != 0)
    finally:
        wfc.close()
