import importlib
import time

import numpy as np
import pyshmem
import pytest

from testsupport import bare_component, private_stream

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
    loop = bare_component(loop_mod.Loop)
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
    loop.register_output_stream("wfc", loop.wfc_shm)
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
    loop = bare_component(loop_mod.Loop)
    loop.g_cm = np.eye(4, dtype=np.float32) * 0.25
    loop._correction_buffer = np.zeros(4, dtype=np.float32)
    loop.num_active_modes = 3

    sent = {}
    loop.signal_shm = private_stream("signal", (4,), np.float32)
    loop.wfc_shm = private_stream("wfc", (4,), np.float32)
    loop.register_input_stream("signal", loop.signal_shm)
    loop.register_output_stream("wfc", loop.wfc_shm)
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
    loop = bare_component(loop_mod.Loop)
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


# --- Interaction-matrix calibration (#87 round-trip check, #102 Hadamard) ---


class _FakeAOSystem:
    """Corrector + sensor stand-in on real pyshmem streams.

    A background thread publishes ``signal = true_im @ applied + noise`` at a
    fixed frame period, stamped with frame ids. ``applied`` is the latest
    ``wfc`` command, except that commands are ignored until ``startup_delay``
    seconds after :meth:`start` (like a corrector whose worker is still
    JIT-compiling) and reach the signal ``lag_frames`` frames late (a deeper
    pipeline). With ``respond=False`` commands never land.
    """

    def __init__(
        self,
        true_im,
        *,
        frame_period=2e-3,
        startup_delay=0.0,
        lag_frames=0,
        noise=0.0,
        respond=True,
        seed=0,
    ):
        import collections
        import threading

        self.true_im = np.asarray(true_im, dtype=np.float32)
        num_signals, num_modes = self.true_im.shape
        self.wfc = private_stream("wfc", (num_modes,), np.float32)
        self.signal = private_stream("signal", (num_signals,), np.float32)
        self.frame_period = frame_period
        self.startup_delay = startup_delay
        self.noise = noise
        self.respond = respond
        self.rng = np.random.default_rng(seed)
        self.pending = collections.deque(
            [np.zeros(num_modes, dtype=np.float32)] * (lag_frames + 1), maxlen=lag_frames + 1
        )
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def loop_config(self, **overrides):
        conf = {
            "input_streams": {"signal": self.signal.name},
            "output_streams": {"wfc": self.wfc.name},
            "poke_amp": 0.1,
            "num_iters_im": 2,
            "im_timeout": 10.0,
        }
        conf.update(overrides)
        return conf

    def start(self):
        self._start_time = time.monotonic()
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5.0)

    def _run(self):
        frame_id = 0
        while not self._stop.is_set():
            time.sleep(self.frame_period)
            live = time.monotonic() - self._start_time >= self.startup_delay
            if self.respond and live:
                self.pending.append(np.array(self.wfc.read(), dtype=np.float32))
            applied = self.pending[0]
            frame = self.true_im @ applied
            if self.noise > 0:
                frame = frame + self.rng.normal(0.0, self.noise, frame.shape)
            frame_id += 1
            self.signal.write(frame.astype(np.float32), frame_id=frame_id)


def _true_im(num_signals=12, num_modes=6, seed=87):
    return np.random.default_rng(seed).normal(size=(num_signals, num_modes)).astype(np.float32)


@pytest.fixture
def fake_ao_system():
    systems = []

    def _make(*args, **kwargs):
        system = _FakeAOSystem(*args, **kwargs)
        systems.append(system)
        return system

    yield _make
    for system in systems:
        system.stop()


def test_cold_start_im_matches_warm_im(fake_ao_system):
    true_im = _true_im()
    system = fake_ao_system(true_im, startup_delay=0.5).start()
    loop = loop_mod.Loop(system.loop_config())

    loop.compute_im()  # cold: the corrector ignores commands for 0.5 s
    cold = loop.im.copy()
    loop.compute_im()  # warm
    warm = loop.im.copy()

    np.testing.assert_allclose(cold, true_im, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(cold, warm, rtol=1e-4, atol=1e-5)
    assert loop.last_round_trip_frames >= 1
    # compute_im leaves the corrector flat.
    assert np.array_equal(system.wfc.read(), np.zeros(true_im.shape[1], dtype=np.float32))


def test_cold_start_without_round_trip_check_loses_leading_columns(fake_ao_system):
    """The failure #87 fixes: calibrating before commands land zeroes columns."""
    true_im = _true_im()
    system = fake_ao_system(true_im, startup_delay=0.5, frame_period=5e-3).start()
    loop = loop_mod.Loop(system.loop_config(im_round_trip_check=False))

    loop.compute_im()

    assert np.allclose(loop.im[:, 0], 0.0)
    assert not np.allclose(loop.im, true_im, atol=1e-3)


def test_round_trip_check_times_out_with_a_clear_error(fake_ao_system):
    system = fake_ao_system(_true_im(), respond=False).start()
    loop = loop_mod.Loop(system.loop_config())

    with pytest.raises(TimeoutError, match="poke_amp"):
        loop.check_round_trip(timeout=0.5)
    loop.im_timeout = 0.5
    with pytest.raises(TimeoutError, match="DM round-trip check"):
        loop.compute_im()  # im_round_trip_check defaults to on


def test_round_trip_check_tolerates_sensor_noise(fake_ao_system):
    true_im = _true_im()
    system = fake_ao_system(true_im, noise=0.02).start()
    loop = loop_mod.Loop(system.loop_config())

    assert loop.check_round_trip(timeout=10.0) >= 1


def test_settle_frames_cover_pipeline_lag(fake_ao_system, caplog):
    true_im = _true_im()
    system = fake_ao_system(true_im, lag_frames=3).start()
    loop = loop_mod.Loop(system.loop_config())

    # The measured round trip sets how many frames to discard after each poke,
    # so a slow pipeline still calibrates correctly with the default settings.
    with caplog.at_level("INFO", logger="pyrtc"):
        loop.compute_im()
    assert "discarding" in caplog.text
    np.testing.assert_allclose(loop.im, true_im, rtol=1e-4, atol=1e-5)
    assert loop._active_settle_frames is None

    # Without the round-trip check, one settle frame is not enough for the lag.
    loop.compute_im(round_trip_check=False)
    assert not np.allclose(loop.im, true_im, atol=1e-3)

    loop.im_settle_frames = 5
    loop.compute_im(round_trip_check=False)
    np.testing.assert_allclose(loop.im, true_im, rtol=1e-4, atol=1e-5)


def _synchronous_loop(true_im, *, noise=0.0, num_iters_im=1, seed=0, method="push-pull"):
    """A Loop whose signal is computed on read from the last command (no threads)."""
    rng = np.random.default_rng(seed)
    num_signals, num_modes = true_im.shape
    loop = loop_mod.Loop.__new__(loop_mod.Loop)
    loop.num_modes = num_modes
    loop.signal_size = num_signals
    loop.signal_dtype = np.dtype(np.float32)
    loop.wfc_dtype = np.dtype(np.float32)
    loop.flat = np.zeros(num_modes, dtype=np.float32)
    loop.poke_amp = 0.1
    loop.num_iters_im = num_iters_im
    loop.im_settle_frames = 1
    loop.hardware_delay = 0.0
    loop.im_timeout = 1.0
    loop.im_method = method
    state = {"command": loop.flat.copy(), "frames": 0}

    def _send(correction, slopes=None):
        state["command"] = np.asarray(correction, dtype=np.float64).copy()

    def _read(name, **_kwargs):
        state["frames"] += 1
        return true_im @ state["command"] + rng.normal(0.0, noise, num_signals)

    loop.send_to_wfc = _send
    loop.read_stream = _read
    return loop, state


@pytest.mark.parametrize("num_modes", [1, 2, 5, 8, 13])
def test_hadamard_patterns_are_orthogonal(num_modes):
    patterns = loop_mod.Loop.hadamard_patterns(num_modes)
    order = patterns.shape[0]
    assert order >= num_modes and order & (order - 1) == 0
    assert order < 2 * num_modes or num_modes == 1
    assert set(np.unique(patterns)) <= {-1.0, 1.0}
    np.testing.assert_array_equal(patterns.T @ patterns, order * np.eye(num_modes))


@pytest.mark.parametrize("num_modes", [6, 8])
def test_hadamard_im_recovers_noise_free_im(num_modes):
    true_im = _true_im(num_modes=num_modes)
    loop, state = _synchronous_loop(true_im)

    loop.hadamard_im()

    np.testing.assert_allclose(loop.im, true_im, rtol=1e-5, atol=1e-5)
    order = loop.hadamard_patterns(num_modes).shape[0]
    assert state["frames"] == 2 * order * (loop.im_settle_frames + loop.num_iters_im)


def test_hadamard_im_is_less_noisy_than_push_pull_at_equal_frames():
    true_im = _true_im(num_signals=40, num_modes=16)
    push_pull, pp_state = _synchronous_loop(true_im, noise=0.1, num_iters_im=4, seed=1)
    push_pull.push_pull_im()
    multiplexed, h_state = _synchronous_loop(true_im, noise=0.1, num_iters_im=4, seed=2)
    multiplexed.hadamard_im()

    assert pp_state["frames"] == h_state["frames"]  # 16 modes -> 16 patterns
    pp_error = np.sqrt(np.mean((push_pull.im - true_im) ** 2))
    h_error = np.sqrt(np.mean((multiplexed.im - true_im) ** 2))
    # White noise drops by sqrt(num_modes) = 4; allow for sampling scatter.
    assert h_error < 0.5 * pp_error


def test_compute_im_dispatches_hadamard_and_rejects_unknown_methods(monkeypatch):
    true_im = _true_im()
    loop, _ = _synchronous_loop(true_im, method="Hadamard")
    loop.im_round_trip_check = True
    calls = []
    monkeypatch.setattr(loop, "hadamard_im", lambda: calls.append("hadamard"))
    monkeypatch.setattr(loop, "check_round_trip", lambda: calls.append("check") or 1)
    monkeypatch.setattr(loop, "compute_cm", lambda: calls.append("cm"))
    monkeypatch.setattr(loop, "flatten", lambda: calls.append("flat"))

    loop.compute_im()
    assert calls == ["check", "hadamard", "flat", "cm"]

    calls.clear()
    loop.compute_im(round_trip_check=False)
    assert calls == ["hadamard", "flat", "cm"]

    loop.im_method = "push_pull"
    with pytest.raises(ValueError, match="Unsupported interaction-matrix method"):
        loop.compute_im()


def _watchdog_loop(action="hold", timeout=0.05):
    signal = private_stream("wd_signal", (4,), "float32")
    signal.write(np.ones(4, dtype=np.float32))
    loop = bare_component(loop_mod.Loop, inputs={"signal": signal})
    loop.watchdog_timeout = timeout
    loop.watchdog_action = action
    loop._stale_since = None
    loop._stale_events = 0
    loop._last_watchdog_action = None
    loop._producer_alive = None
    loop.running = True
    loop.flattened = False
    loop.flatten = lambda: setattr(loop, "flattened", True)
    return loop, signal


@pytest.mark.parametrize(
    ("action", "running", "flattened"),
    [("hold", True, False), ("open", False, False), ("flatten", False, True)],
)
def test_watchdog_flags_a_stale_signal_and_applies_its_action(action, running, flattened):
    loop, signal = _watchdog_loop(action)
    try:
        assert loop._read_signal() is not None  # first read: the current frame
        assert loop.safety_status()["input_stale"] is False
        assert loop._read_signal() is None  # nothing new within watchdog_timeout
        status = loop.safety_status()
        assert status["input_stale"] is True
        assert status["producer_alive"] is True  # this process created the stream
        assert status["stale_events"] == 1
        assert "signal stale" in status["alerts"][0] and "producer stalled" in status["alerts"][0]
        assert loop.running is running
        assert loop.flattened is flattened

        # A second timeout in the same episode does not count again.
        loop._read_signal()
        assert loop.safety_status()["stale_events"] == 1

        signal.write(np.full(4, 2.0, dtype=np.float32))
        frame = loop._read_signal()
        assert frame is not None and np.all(frame == 2.0)
        status = loop.safety_status()
        assert status["input_stale"] is False and status["alerts"] == []
    finally:
        loop.close()
        signal.close()


def test_watchdog_disabled_blocks_as_before():
    loop, signal = _watchdog_loop(timeout=None)
    try:
        loop._read_signal()
        with pytest.raises(TimeoutError):
            # Without the watchdog the read is unbounded; bound it here to prove
            # _read_signal did not add a timeout of its own.
            loop.read_stream("signal", timeout=0.05)
        assert loop.safety_status()["watchdog_timeout"] is None
    finally:
        loop.close()
        signal.close()


def test_standard_integrator_skips_the_iteration_when_the_signal_is_stale(monkeypatch):
    loop, signal = _watchdog_loop()
    sent = []
    loop.send_to_wfc = lambda correction, slopes=None: sent.append(correction)
    loop._signal_buffer = np.empty(4, dtype=np.float32)
    try:
        loop._read_signal()  # consume the current frame
        loop.standard_integrator()
        assert sent == []
        assert loop.safety_status()["input_stale"] is True
    finally:
        loop.close()
        signal.close()


@pytest.mark.parametrize(
    ("conf", "match"),
    [({"watchdog_action": "explode"}, "watchdog_action"), ({"watchdog_timeout": -1}, ">= 0")],
)
def test_watchdog_config_is_validated(conf, match):
    with pytest.raises(ValueError, match=match):
        loop_mod.Loop._validate_watchdog(
            conf.get("watchdog_timeout", 1.0), conf.get("watchdog_action", "hold")
        )
    assert loop_mod.Loop._validate_watchdog(0, "OPEN") == (None, "open")
    assert loop_mod.Loop._validate_watchdog(None, "hold") == (None, "hold")
