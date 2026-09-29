"""Per-mode gains, optical-gain compensation and gain optimization (#56)."""

import numpy as np
import pytest

from pyrtc import modal_gains
from pyrtc.loop import Loop
from testsupport import bare_component


def _simulate(gain, a, sigma_w, sigma_n, *, delay=2, frames=12000, seed=0):
    """Closed-loop integrator on an AR(1) disturbance with white WFS noise.

    Returns the measured residual (what the loop sees) and the true residual.
    The correction applied at frame t+1 integrates the measurement from frame
    t - (delay - 1), i.e. an open-loop transfer ``g z^-delay / (1 - z^-1)``.
    """
    rng = np.random.default_rng(seed)
    drive = rng.normal(0.0, sigma_w, frames)
    noise = rng.normal(0.0, sigma_n, frames)
    phi = np.zeros(frames)
    correction = np.zeros(frames + 1)
    measured = np.zeros(frames)
    for t in range(frames):
        phi[t] = (a * phi[t - 1] if t else 0.0) + drive[t]
        measured[t] = phi[t] - correction[t] + noise[t]
        source = t - (delay - 1)
        correction[t + 1] = correction[t] + gain * (measured[source] if source >= 0 else 0.0)
    return measured[1000:], (phi - correction[:-1])[1000:]


def test_max_stable_gain_matches_theory():
    # Integrator with d frames of delay: g_max = 2 sin(pi / (2 (2d + 1))).
    for delay in (1, 2, 3):
        expected = 2 * np.sin(np.pi / (2 * (2 * delay + 1)))
        assert modal_gains.max_stable_gain(delay) == pytest.approx(expected, abs=1e-3)


def test_rejection_transfer_is_zero_at_dc_and_near_one_at_nyquist():
    rejection = modal_gains.rejection_transfer(np.array([1e-6, 500.0]), 1000.0, 0.3, 2)
    assert abs(rejection[0]) < 1e-4
    assert abs(rejection[1]) == pytest.approx(1 / abs(1 + 0.3 * 1 / 2), rel=1e-6)


MODES = [
    dict(a=0.995, sigma_w=1.0, sigma_n=0.5),  # strong, slow disturbance: high gain
    dict(a=0.95, sigma_w=0.2, sigma_n=3.0),  # noise dominated: low gain
    dict(a=0.999, sigma_w=0.3, sigma_n=0.1),  # high SNR: high gain
]


def test_optimized_gains_are_near_the_simulated_optimum():
    applied = 0.3
    measured = np.column_stack([_simulate(applied, **mode)[0] for mode in MODES])
    result = modal_gains.optimize_modal_gains(measured, 1000.0, applied, delay_frames=2)

    assert result.gains[1] < 0.1 < 0.3 < result.gains[0]
    assert np.all(result.gains < result.max_gain)
    grid = np.linspace(0.02, 0.58, 29)
    for index, mode in enumerate(MODES):
        true = [np.var(_simulate(g, **mode, seed=1)[1]) for g in grid]
        chosen = np.var(_simulate(result.gains[index], **mode, seed=1)[1])
        assert chosen <= 1.05 * min(true), (index, result.gains[index], grid[np.argmin(true)])


def test_optimizer_validates_inputs():
    with pytest.raises(ValueError, match="positive"):
        modal_gains.optimize_modal_gains(np.zeros((512, 2)), 1000.0, 0.0)
    with pytest.raises(ValueError, match="gain_grid"):
        modal_gains.optimize_modal_gains(np.zeros((512, 1)), 1000.0, 0.3, gain_grid=[0.1, 0.9])
    with pytest.raises(ValueError, match="at least 8"):
        modal_gains.optimize_modal_gains(np.zeros((4, 1)), 1000.0, 0.3)


def _gain_loop(num_modes=3, signal_size=4):
    loop = bare_component(Loop)
    loop.num_modes = num_modes
    loop.num_active_modes = num_modes
    loop.signal_size = signal_size
    loop.cm = np.arange(num_modes * signal_size, dtype=np.float32).reshape(num_modes, signal_size)
    loop._modal_gains = None
    loop._optical_gains = None
    loop.gain = 0.5
    return loop


def test_modal_and_optical_gains_scale_the_control_matrix_rows():
    loop = _gain_loop()
    np.testing.assert_allclose(loop.g_cm, 0.5 * loop.cm)

    loop.set_modal_gains([1.0, 0.5, 0.0])
    loop.set_optical_gains([1.0, 1.0, 0.5])
    np.testing.assert_allclose(loop.effective_gains, [0.5, 0.25, 0.0])
    np.testing.assert_allclose(loop.g_cm, np.array([0.5, 0.25, 0.0])[:, None] * loop.cm)
    assert loop.g_cm.dtype == loop.cm.dtype

    loop.set_optical_gains([0.5, 1.0, 1.0])
    np.testing.assert_allclose(loop.effective_gains, [1.0, 0.25, 0.0])
    loop.gain = 0.2  # the loop gain still scales everything
    np.testing.assert_allclose(loop.effective_gains, [0.4, 0.1, 0.0])

    loop.set_modal_gains(None)
    loop.set_optical_gains(None)
    np.testing.assert_allclose(loop.g_cm, 0.2 * loop.cm)


@pytest.mark.parametrize(
    ("setter", "value", "match"),
    [
        ("set_modal_gains", [1.0, 1.0], "3 entries"),
        ("set_modal_gains", [1.0, -1.0, 1.0], ">= 0"),
        ("set_optical_gains", [1.0, 0.0, 1.0], "positive"),
        ("set_optical_gains", [1.0, np.nan, 1.0], "finite"),
    ],
)
def test_gain_vectors_are_validated(setter, value, match):
    with pytest.raises(ValueError, match=match):
        getattr(_gain_loop(), setter)(value)


def test_modal_gains_load_from_npy(tmp_path):
    path = tmp_path / "gains.npy"
    np.save(path, np.array([0.2, 0.4, 0.6]))
    loop = _gain_loop()
    loop.set_modal_gains(str(path))
    np.testing.assert_allclose(loop.effective_gains, [0.1, 0.2, 0.3])


def test_pol_update_uses_per_mode_gains():
    loop = _gain_loop(num_modes=2, signal_size=2)
    loop.cm = np.eye(2, dtype=np.float32)
    loop.f_im = 0.5 * np.eye(2, dtype=np.float32)
    loop.set_modal_gains([1.0, 0.0])
    correction = np.array([1.0, 1.0], dtype=np.float32)
    out = loop.update_correction_pol(correction=correction, slopes=np.zeros(2, dtype=np.float32))
    # POL slopes are -0.5 c. Mode 0 (g = 0.5): 0.5 * 1 + 0.5 * 0.5 = 0.75; mode 1 (g = 0) holds.
    np.testing.assert_allclose(out, [0.75, 1.0])
    assert out.dtype == np.float32


def test_loop_optimize_modal_gains_applies_the_optimum():
    loop = _gain_loop(num_modes=3, signal_size=3)
    loop.cm = np.eye(3, dtype=np.float32)
    # Effective gain 0.3 on every mode, as in the simulation, via optical gains of 2.
    loop.gain = 0.6
    loop.set_optical_gains([2.0, 2.0, 2.0])
    np.testing.assert_allclose(loop.effective_gains, 0.3)
    measured = np.column_stack([_simulate(0.3, **mode)[0] for mode in MODES])
    residuals = loop.modal_residuals(measured)
    np.testing.assert_allclose(residuals, measured)

    result = loop.optimize_modal_gains(residuals, 1000.0, delay_frames=2)
    np.testing.assert_allclose(loop.effective_gains, result.gains, rtol=1e-6)
    np.testing.assert_allclose(loop._optical_gains, 2.0)  # compensation is kept
