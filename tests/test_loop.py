import numpy as np
import importlib

loop_mod = importlib.import_module("pyrtc.loop")


def test_loop_helper_functions(monkeypatch):
    slopes = np.array([1.0, 2.0], dtype=np.float32)
    cm = np.eye(2, dtype=np.float32)
    old = np.array([0.5, 0.5], dtype=np.float32)
    correction = np.zeros(2, dtype=np.float32)

    out = loop_mod.leaky_integrator_numba(slopes, cm, old, correction, np.float32(0.1), 1)
    assert out.shape == (2,)

    assert np.array_equal(loop_mod.comp_correction(cm, slopes), slopes)
    upd = loop_mod.update_correction(np.array([1.0, 1.0], dtype=np.float32), cm, slopes)
    assert np.array_equal(upd, np.array([0.0, -1.0], dtype=np.float32))

    monkeypatch.setattr(loop_mod, "gpu_torch_available", lambda: False)
    try:
        loop_mod.leak_integrator_gpu(slopes, cm, old, 0.1, 1)
        assert False
    except ImportError:
        assert True


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
    loop.pid_integrator(slopes=np.ones(6, dtype=np.float32), correction=np.zeros(4, dtype=np.float32))
    assert hasattr(loop, "_sent")

    # send_to_wfc branch with CL DOCRIME
    class _W:
        def __init__(self):
            self.last = None

        def write(self, x):
            self.last = np.asarray(x)

    loop.wfc_shm = _W()
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
    loop.null_correction = np.zeros(4, dtype=np.float32)
    loop.num_active_modes = 3

    class _Signal:
        count = 1
        write_time = 1.0

        def read(self, out=None):
            return np.ones(4, dtype=np.float32)

    class _Wfc:
        count = 1
        write_time = 1.0

        def read_new(self, timeout=None, out=None):
            raise AssertionError("standard_integrator should not block on wfc")

        def read(self, out=None):
            return np.zeros(4, dtype=np.float32)

    sent = {}
    loop.signal_shm = _Signal()
    loop.wfc_shm = _Wfc()
    loop._signalBuffer = np.empty(4, dtype=np.float32)
    loop._wfcBuffer = np.empty(4, dtype=np.float32)
    loop.send_to_wfc = lambda correction, slopes=None: sent.setdefault("correction", correction.copy())

    loop.standard_integrator()

    assert "correction" in sent
    assert np.max(np.abs(sent["correction"])) > 0


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
