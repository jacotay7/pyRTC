"""Model-based predictive control (#57)."""

import numpy as np
import pytest

from pyrtc import predictive
from pyrtc.loop import Loop
from testsupport import bare_component, private_stream


def _ar2(frames, frequency, damping, sigma, seed):
    rng = np.random.default_rng(seed)
    a1, a2 = 2 * damping * np.cos(2 * np.pi * frequency), -damping * damping
    x = np.zeros(frames)
    drive = rng.normal(0.0, sigma, frames)
    for t in range(2, frames):
        x[t] = a1 * x[t - 1] + a2 * x[t - 2] + drive[t]
    return x, (a1, a2)


def _disturbance(frames, seed=0):
    vibration, _ = _ar2(frames, 0.03, 0.995, 1.0, seed)
    rng = np.random.default_rng(seed + 1)
    slow = np.zeros(frames)
    drive = rng.normal(0.0, 0.3, frames)
    for t in range(1, frames):
        slow[t] = 0.998 * slow[t - 1] + drive[t]
    return np.column_stack((vibration, slow))


def _closed_loop(phi, step, *, delay=2, noise=0.5, seed=1, warmup=1500):
    """Run a modal loop: residual = phi + applied command + noise.

    ``step(y, applied_now, last_sent)`` returns the next command, which lands
    ``delay`` frames later. Returns the residual variance and the POL log.
    """
    frames, modes = phi.shape
    noise = np.random.default_rng(seed).normal(0.0, noise, (frames, modes))
    applied = np.zeros((frames + delay + 1, modes))
    last = np.zeros(modes)
    pol_log = np.zeros((frames, modes))
    for t in range(frames):
        y = phi[t] + applied[t] + noise[t]
        pol_log[t] = y - applied[t]
        last = step(y, applied[t], last)
        applied[t + delay] = last
    return np.var((phi + applied[:frames])[warmup:], axis=0), pol_log


def _predictive_step(predictor, gain=1.0):
    def step(y, applied_now, last):
        predictor.update(y - applied_now)
        return (1 - gain) * last - gain * predictor.predict()

    return step


def test_registry_lists_and_builds_predictors():
    assert {"persistence", "ar_kalman", "least_squares"} <= set(predictive.available_predictors())
    built = predictive.make_predictor({"type": "LEAST_SQUARES", "order": 4, "gain": 0.5}, 3)
    assert isinstance(built, predictive.LeastSquaresPredictor)
    assert built.order == 4 and built.num_modes == 3
    with pytest.raises(ValueError, match="unknown predictor"):
        predictive.make_predictor({"type": "crystal_ball"}, 2)


def test_custom_predictors_plug_in():
    @predictive.register_predictor("zero_test")
    class ZeroPredictor(predictive.ModalPredictor):
        def fit(self, pol_history):
            self.fitted = True
            return self

        def update(self, pol):
            pass

        def predict(self):
            return np.zeros(self.num_modes)

        def reset(self):
            pass

    try:
        built = predictive.make_predictor({"type": "zero_test", "horizon": 3}, 2)
        assert isinstance(built, ZeroPredictor) and built.horizon == 3
    finally:
        predictive._REGISTRY.pop("zero_test")


def test_ar_kalman_recovers_an_ar2_vibration():
    x, (a1, a2) = _ar2(20000, 0.05, 0.99, 1.0, seed=3)
    noise_sigma = 0.8
    y = x + np.random.default_rng(4).normal(0.0, noise_sigma, x.size)
    model = predictive.ARKalmanPredictor(1, horizon=2).fit(y[:, None])
    assert model.a1[0] == pytest.approx(a1, abs=0.03)
    assert model.a2[0] == pytest.approx(a2, abs=0.03)
    assert model.fitted_noise_variance[0] == pytest.approx(noise_sigma**2, rel=0.3)


@pytest.mark.parametrize(("kind", "options"), [("ar_kalman", {}), ("least_squares", {"order": 16})])
def test_predictive_control_beats_the_best_integrator(kind, options):
    phi = _disturbance(12000)
    best = np.full(2, np.inf)
    for gain in np.linspace(0.05, 0.6, 8):
        variance, _ = _closed_loop(phi, lambda y, a, last, g=gain: last - g * y)
        best = np.minimum(best, variance)

    # Record POL data with the unfitted (persistence) predictor at a safe gain, fit, run.
    recorder = predictive.make_predictor({"type": "persistence", "horizon": 2}, 2)
    _, pol = _closed_loop(phi[:6000], _predictive_step(recorder, gain=0.3))
    model = predictive.make_predictor({"type": kind, "horizon": 2, **options}, 2).fit(pol[500:])
    variance, _ = _closed_loop(phi, _predictive_step(model))

    assert variance[0] < 0.2 * best[0]  # vibration: many times better
    assert variance[1] < 1.1 * best[1]  # slow turbulence: no worse


def test_predictors_validate_inputs():
    with pytest.raises(ValueError, match="horizon"):
        predictive.PersistencePredictor(2, horizon=0)
    with pytest.raises(ValueError, match="order"):
        predictive.LeastSquaresPredictor(2, order=0)
    with pytest.raises(ValueError, match="shape"):
        predictive.ARKalmanPredictor(2).fit(np.zeros((500, 3)))
    with pytest.raises(ValueError, match="at least"):
        predictive.ARKalmanPredictor(1).fit(np.zeros((10, 1)))


def _predictive_loop(num_modes=2, delay=2, gain=0.25):
    signal = private_stream("pred_signal", (num_modes,), "float32")
    wfc = private_stream("pred_wfc", (num_modes,), "float32")
    loop = bare_component(Loop, inputs={"signal": signal, "wfc": wfc})
    loop.num_modes = loop.num_active_modes = num_modes
    loop.cm = np.eye(num_modes, dtype=np.float32)
    loop._signal_buffer = np.empty(num_modes, dtype=np.float32)
    loop._wfc_buffer = np.empty(num_modes, dtype=np.float32)
    loop.gain = gain
    loop.configure_predictor({"type": "persistence", "delay_frames": delay, "fit_frames": 64})
    sent = []

    def send_to_wfc(correction, slopes=None):
        sent.append(np.array(correction))
        wfc.write(np.asarray(correction, dtype=np.float32))

    loop.send_to_wfc = send_to_wfc
    return loop, signal, wfc, sent


def test_predictive_integrator_uses_the_delayed_command_for_pol():
    loop, signal, wfc, sent = _predictive_loop()
    try:
        wfc.write(np.zeros(2, dtype=np.float32))
        for value in (1.0, 2.0, 3.0):
            signal.write(np.full(2, value, dtype=np.float32))
            loop.predictive_integrator()
        pol = loop.recorded_pol()
        assert pol.shape == (3, 2)
        # Iterations 1-2: nothing sent delay_frames ago yet, so the applied
        # command is the current one. Iteration 3 uses the command from
        # iteration 1, not the latest.
        np.testing.assert_allclose(pol[0], 1.0)
        np.testing.assert_allclose(pol[1], 2.0 - sent[0])
        np.testing.assert_allclose(pol[2], 3.0 - sent[0])
        # Unfitted: blends with the loop gain, c = (1 - g) c - g * pol.
        np.testing.assert_allclose(sent[0], -0.25 * 1.0)
        np.testing.assert_allclose(sent[1], 0.75 * sent[0] - 0.25 * pol[1], rtol=1e-6)
    finally:
        loop.close()
        signal.close()
        wfc.close()


def test_fit_predictor_swaps_in_a_fitted_model():
    loop, signal, wfc, sent = _predictive_loop()
    try:
        loop.configure_predictor(
            {"type": "least_squares", "order": 4, "delay_frames": 2, "fit_frames": 64}
        )
        wfc.write(np.zeros(2, dtype=np.float32))
        rng = np.random.default_rng(0)
        for _ in range(80):  # wraps the 64-frame record
            signal.write(rng.normal(size=2).astype(np.float32))
            loop.predictive_integrator()
        record = loop.recorded_pol()
        assert record.shape == (64, 2)
        old = loop.predictor
        fitted = loop.fit_predictor()
        assert fitted is loop.predictor and fitted is not old
        assert loop.predictor_fitted and fitted.fitted
        assert not np.allclose(fitted.taps[:, :-1], 0.0)  # learned, not persistence
        signal.write(np.ones(2, dtype=np.float32))
        loop.predictive_integrator()  # now blends with predictor.gain (1.0)
        np.testing.assert_allclose(sent[-1], -fitted.predict(), rtol=1e-5)
    finally:
        loop.close()
        signal.close()
        wfc.close()


@pytest.mark.parametrize(
    ("conf", "match"),
    [
        ({"delay_frames": 0}, "delay_frames"),
        ({"gain": 1.5}, "gain"),
        ({"fit_frames": 10}, "fit_frames"),
        ({"type": "nope"}, "unknown predictor"),
    ],
)
def test_predictor_config_is_validated(conf, match):
    loop = bare_component(Loop)
    loop.num_active_modes = 2
    with pytest.raises(ValueError, match=match):
        loop.configure_predictor(conf)
