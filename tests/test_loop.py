import importlib

import numpy as np
import pyshmem
import pytest

from testsupport import private_stream

loop_mod = importlib.import_module("pyrtc.loop")


def test_loop_helper_functions(monkeypatch):
    slopes = np.array([1.0, 2.0], dtype=np.float32)
    cm = np.eye(2, dtype=np.float32)
    old = np.array([0.5, 0.5], dtype=np.float32)
    correction = np.zeros(2, dtype=np.float32)

    out = loop_mod.leaky_integrator_numba(slopes, cm, old, correction, np.float32(0.1), 1)
    assert out.shape == (2,)
    assert out is correction

    assert np.array_equal(loop_mod.comp_correction(cm, slopes), slopes)
    upd = loop_mod.update_correction(np.array([1.0, 1.0], dtype=np.float32), cm, slopes)
    assert np.array_equal(upd, np.array([0.0, -1.0], dtype=np.float32))

    monkeypatch.setattr(loop_mod, "gpu_torch_available", lambda: False)
    try:
        loop_mod.leak_integrator_gpu(slopes, cm, old, 0.1, 1)
        assert False
    except ImportError:
        assert True


def _integrator_case(num_modes=4, num_slopes=6, seed=0):
    rng = np.random.default_rng(seed)
    slopes = rng.standard_normal(num_slopes).astype(np.float32)
    recon = rng.standard_normal((num_modes, num_slopes)).astype(np.float32)
    old = rng.standard_normal(num_modes).astype(np.float32)
    return slopes, recon, old


@pytest.mark.parametrize("dropped", [0, 1, 3])
def test_leaky_integrator_controls_only_active_modes(dropped):
    slopes, recon, old = _integrator_case()
    num_active = recon.shape[0] - dropped
    leak = np.float32(0.1)
    expected = (1 - leak) * old - recon @ slopes
    expected[num_active:] = 0

    out = loop_mod.leaky_integrator_numba(slopes, recon, old, np.empty_like(old), leak, num_active)

    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)


def test_leaky_integrator_never_writes_past_the_correction_buffer():
    slopes, recon, old = _integrator_case()
    backing = np.full(recon.shape[0] + 1, 123.0, dtype=np.float32)
    correction = backing[: recon.shape[0]]

    loop_mod.leaky_integrator_numba(slopes, recon, old, correction, np.float32(0), recon.shape[0])

    assert backing[-1] == 123.0


@pytest.mark.gpu
@pytest.mark.skipif(not pyshmem.gpu_available(), reason="CUDA is not available")
@pytest.mark.parametrize("dropped", [0, 2])
def test_gpu_integrator_matches_cpu(dropped):
    import torch

    slopes, recon, old = _integrator_case()
    num_active = recon.shape[0] - dropped
    cpu = loop_mod.leaky_integrator_numba(
        slopes, recon, old, np.empty_like(old), np.float32(0.05), num_active
    )
    gpu = loop_mod.leak_integrator_gpu(
        slopes, torch.as_tensor(recon, device="cuda"), old, 0.05, num_active
    )

    np.testing.assert_allclose(gpu, cpu, rtol=1e-4, atol=1e-5)


def test_loop_methods_without_full_init(tmp_path):
    loop = loop_mod.Loop.__new__(loop_mod.Loop)
    loop.num_modes = 4
    loop.num_dropped_modes = 1
    loop.num_active_modes = 3
    loop.cm_method = "svd"
    loop.conditioning = None
    loop.tikhonov_reg = 0.0
    loop.last_singular_values = np.array([], dtype=np.float64)
    loop.last_retained_singular_mask = np.array([], dtype=bool)
    loop.last_suggested_conditioning = None
    loop.last_singular_value_fit = None
    loop.im = np.random.RandomState(0).randn(6, 4).astype(np.float32)
    loop.cm = np.zeros((4, 6), dtype=np.float32)
    loop.gain = 0.2
    loop.compute_cm()
    assert loop.cm.shape == (4, 6)

    loop.compute_cm(conditioning=10.0)
    assert loop.conditioning == 10.0
    assert loop.last_singular_values.size == min(loop.im[:, : loop.num_active_modes].shape)

    loop.compute_cm(method="tikhonov", conditioning=10.0, tikhonov_reg=0.05)
    assert loop.cm_method == "tikhonov"
    assert np.isclose(loop.tikhonov_reg, 0.05)

    suggestion = loop.suggest_conditioning_number()
    assert suggestion is None or suggestion >= 1.0
    if suggestion is not None:
        assert loop.last_singular_value_fit is not None
        assert "fit_curve" in loop.last_singular_value_fit

    plotted = loop.plot_singular_values()
    assert plotted is None or plotted >= 1.0

    loop.set_gain(0.5)
    assert np.allclose(loop.g_cm, 0.5 * loop.cm)

    loop.gain = 0.3
    assert np.allclose(loop.g_cm, 0.3 * loop.cm)

    loop.set_peturb_amp(0.3)
    assert np.isclose(loop.perturb_amp, 0.3)

    loop.im_file = str(tmp_path / "im.npy")
    loop.save_im()
    loop.im = np.zeros_like(loop.im)
    loop.load_im()
    assert np.any(loop.im != 0)

    loop.f_im = np.copy(loop.im)
    correction = np.ones(4, dtype=np.float32)
    slopes = np.ones(6, dtype=np.float32)
    upd = loop.update_correction_pol(correction, slopes)
    assert upd.shape == (4,)

    # pid integrator path
    loop.cm = np.eye(4, 6, dtype=np.float32)
    loop.leaky_gain = 0.0
    loop.control_limits = [-1.0, 1.0]
    loop.integral_limits = [-5.0, 5.0]
    loop.absolute_limits = [-2.0, 2.0]
    loop.p_gain = 0.1
    loop.i_gain = 0.01
    loop.d_gain = 0.01
    loop.derivative_filter = 0.5
    loop.previous_wf_error = np.zeros(4, dtype=np.float32)
    loop.previous_derivative = np.zeros(4, dtype=np.float32)
    loop.control_output = np.zeros(4, dtype=np.float32)
    loop.integral = np.zeros(4, dtype=np.float32)
    loop.send_to_wfc = lambda correction, slopes=None: setattr(loop, "_sent", correction)
    loop.num_active_modes = 3
    loop.pid_integrator(
        slopes=np.ones(6, dtype=np.float32), correction=np.zeros(4, dtype=np.float32)
    )
    assert hasattr(loop, "_sent")

    # send_to_wfc branch with CL DOCRIME
    loop.wfc_shm = private_stream("wfc", (4,), np.float32)
    loop.flat = np.zeros(4, dtype=np.float32)
    loop.cl_docrime = True
    loop.poke_amp = 0.1
    loop.docrime_buffer = np.zeros((2, 4, 1), dtype=np.float32)
    loop.docrime_cross = np.zeros((6, 4), dtype=np.float32)
    loop.docrime_auto = np.zeros((4, 4), dtype=np.float32)
    loop.num_iters_dc = 0
    loop.num_active_modes = 3
    loop.send_to_wfc = loop_mod.Loop.send_to_wfc.__get__(loop, loop_mod.Loop)
    loop.send_to_wfc(np.zeros(4, dtype=np.float32), slopes=np.ones(6, dtype=np.float32))
    assert loop.num_iters_dc == 1

    loop.im_file = str(tmp_path / "im.npy")
    loop.docrime_auto = np.eye(4, dtype=np.float32)
    loop.docrime_cross = np.ones((6, 4), dtype=np.float32)
    loop.solve_docrime()


def test_standard_integrator_uses_nonblocking_wfc_read():
    loop = loop_mod.Loop.__new__(loop_mod.Loop)
    loop.g_cm = np.eye(4, dtype=np.float32) * 0.25
    loop._correction_buffer = np.zeros(4, dtype=np.float32)
    loop.num_active_modes = 3

    sent = {}
    loop.signal_shm = private_stream("signal", (4,), np.float32)
    loop.wfc_shm = private_stream("wfc", (4,), np.float32)
    loop.signal_shm.write(np.ones(4, dtype=np.float32))
    loop._signal_buffer = np.empty(4, dtype=np.float32)
    loop._wfc_buffer = np.empty(4, dtype=np.float32)
    loop.send_to_wfc = lambda correction, slopes=None: sent.setdefault(
        "correction", correction.copy()
    )

    loop.standard_integrator()

    assert "correction" in sent
    assert np.max(np.abs(sent["correction"])) > 0

    # A second iteration must only wait on the signal: wfc is never rewritten
    # here, so a blocking wfc read would hang.
    loop.signal_shm.write(np.ones(4, dtype=np.float32))
    sent.clear()
    loop.standard_integrator()
    assert "correction" in sent


def test_loop_compute_cm_zero_matrix_without_failure():
    loop = loop_mod.Loop.__new__(loop_mod.Loop)
    loop.num_modes = 3
    loop.num_dropped_modes = 0
    loop.num_active_modes = 3
    loop.cm_method = "svd"
    loop.conditioning = None
    loop.tikhonov_reg = 0.0
    loop.last_singular_value_fit = None
    loop.im = np.zeros((5, 3), dtype=np.float32)
    loop.cm = np.zeros((3, 5), dtype=np.float32)
    loop.gain = 0.1

    loop.compute_cm()

    assert np.allclose(loop.cm, 0.0)
    assert loop.last_singular_values.size == 3


def test_conditioning_suggestion_tracks_knee():
    singular_values = np.array([1.0, 0.5, 0.25, 0.125, 1e-3, 5e-4], dtype=np.float64)

    suggestion, fit = loop_mod.Loop._suggest_conditioning_from_singular_values(singular_values)

    assert suggestion is not None
    assert fit is not None
    assert fit["suggested_index"] == 4
    assert np.isclose(suggestion, 1.0 / singular_values[4])
