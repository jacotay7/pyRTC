"""Closed-loop stability of the matrix integrator with a mismatched plant."""

import numpy as np
import pytest

from pyrtc import loop_stability, modal_gains
from pyrtc.dynamics import LinearPlant


def _simulate(cm, im, gain, *, leak=0.0, delay=0, plant=None, frames=400, seed=0):
    """Time-domain run of the loop the module models; returns the final command norm."""
    num_modes = cm.shape[0]
    command = np.random.default_rng(seed).normal(size=num_modes)
    history = [np.zeros(im.shape[1]) for _ in range(delay)]
    if plant is not None:
        plant.reset()
    for _ in range(frames):
        output = command if plant is None else plant.output
        history.append(output.copy())
        signal = im @ history.pop(0)
        if plant is not None:
            plant.step(command)
        command = (1.0 - leak) * command - gain * (cm @ signal)
    return float(np.linalg.norm(command))


@pytest.mark.parametrize("delay", [0, 1, 2, 5])
def test_perfect_calibration_matches_single_mode_limit(delay):
    rng = np.random.default_rng(delay)
    im = rng.normal(size=(12, 5))
    cm = np.linalg.pinv(im)
    limit = loop_stability.max_stable_gain(cm, im, delay_frames=delay)
    assert limit == pytest.approx(modal_gains.max_stable_gain(delay), abs=1e-3)


def test_zero_delay_eigenvalues_are_one_minus_gain():
    im = np.random.default_rng(1).normal(size=(8, 4))
    eigenvalues = loop_stability.closed_loop_eigenvalues(np.linalg.pinv(im), im, 0.3, leak=0.05)
    assert np.allclose(eigenvalues, 1.0 - 0.05 - 0.3)


@pytest.mark.parametrize("gain_factor, expect_stable", [(0.9, True), (1.1, False)])
def test_prediction_agrees_with_simulation(gain_factor, expect_stable):
    rng = np.random.default_rng(2)
    im_design = rng.normal(size=(10, 4))
    im_true = im_design + 0.3 * rng.normal(size=im_design.shape)  # miscalibrated
    cm = np.linalg.pinv(im_design)
    limit = loop_stability.max_stable_gain(cm, im_true, delay_frames=2)
    gain = gain_factor * limit
    assert loop_stability.is_stable(cm, im_true, gain, delay_frames=2) is expect_stable
    final = _simulate(cm, im_true, gain, delay=2, frames=3000)
    assert (final < 1e-3) is expect_stable


def test_gain_error_scales_the_limit():
    im = np.random.default_rng(3).normal(size=(9, 3))
    cm = np.linalg.pinv(im)
    nominal = loop_stability.max_stable_gain(cm, im, delay_frames=1)
    doubled = loop_stability.max_stable_gain(cm, 2.0 * im, delay_frames=1)
    assert doubled == pytest.approx(nominal / 2.0, abs=1e-3)


def test_sign_error_in_one_mode_is_unstable_at_any_gain():
    im = np.eye(3)
    im[1, 1] = -1.0
    assert loop_stability.max_stable_gain(np.eye(3), im) == 0.0
    assert loop_stability.max_stable_gain(np.eye(3), im, leak=0.5) > 0.0  # a leak bounds it


def test_modal_gains_and_upper_bound():
    im = np.eye(2)
    limit = loop_stability.max_stable_gain(im, im, modal_gains=[1.0, 2.0])
    assert limit == pytest.approx(1.0, abs=1e-3)  # the doubled mode reaches 2 first
    assert loop_stability.max_stable_gain(im, im, upper=1.5) == 1.5


def test_slow_plant_lowers_the_stable_gain_and_matches_simulation():
    plant = LinearPlant.first_order([5.0, 5.0]).discretize(1.0)  # lag of 5 frames
    im = np.eye(2)
    static = loop_stability.max_stable_gain(im, im, delay_frames=1)
    lagged = loop_stability.max_stable_gain(im, im, delay_frames=1, plant=plant)
    assert lagged < static
    assert _simulate(im, im, 0.9 * lagged, delay=1, plant=plant, frames=4000) < 1e-3
    assert _simulate(im, im, 1.1 * lagged, delay=1, plant=plant, frames=4000) > 1.0


def test_plant_slower_than_the_loop_period_behaves_as_static():
    plant = LinearPlant.first_order([1e-3]).discretize(1.0)
    with_plant = loop_stability.spectral_radius([[1.0]], [[1.0]], 0.4, plant=plant)
    assert with_plant == pytest.approx(
        loop_stability.spectral_radius([[1.0]], [[1.0]], 0.4, delay_frames=1), abs=1e-6
    )


def test_spectral_radius_gives_convergence_rate():
    radius = loop_stability.spectral_radius(np.eye(2), np.eye(2), 0.25)
    assert radius == pytest.approx(0.75)


def test_shape_checks():
    with pytest.raises(ValueError):
        loop_stability.closed_loop_matrix(np.eye(2), np.eye(3))
    with pytest.raises(ValueError):
        loop_stability.closed_loop_matrix(np.eye(2), np.eye(2), delay_frames=-1)
    with pytest.raises(ValueError):
        loop_stability.closed_loop_matrix(
            np.eye(2), np.eye(2), plant=LinearPlant.first_order([1.0]).discretize(1.0)
        )
    with pytest.raises(ValueError):
        loop_stability.closed_loop_matrix(np.ones((2, 2)), np.ones((2, 3)))


def test_slow_one_sided_plant_closed_loop_end_to_end():
    """The three modules together: a lagging plant, push-only actuators, a
    miscalibrated control matrix, and an integrator with anti-windup."""
    from pyrtc.allocation import BoundedAllocator, anti_windup

    rng = np.random.default_rng(11)
    num_actuators, num_modes, num_signals = 6, 3, 9
    m2c = rng.normal(size=(num_actuators, num_modes))
    plant = LinearPlant.first_order(np.full(num_actuators, 3.0)).discretize(1.0)
    im_true = rng.normal(size=(num_signals, num_modes))  # signal per mode
    cm = np.linalg.pinv(im_true * 1.3 + 0.1 * rng.normal(size=im_true.shape))
    modes_per_actuator = np.linalg.pinv(m2c)

    # Stability of the modal loop through the actuator lag.
    modal_plant = LinearPlant(
        -np.eye(num_actuators) / 3.0, m2c / 3.0, C=modes_per_actuator
    ).discretize(1.0)
    gain = 0.5 * loop_stability.max_stable_gain(cm, im_true, plant=modal_plant)
    assert gain > 0

    bias = np.full(num_actuators, 1.0)  # operating point with range on both sides
    allocator = BoundedAllocator(modes_per_actuator, 0.0, 2.0, regularization=1e-6, preferred=bias)
    disturbance = im_true @ np.array([0.3, -0.2, 0.1])
    state = np.zeros(num_modes)
    plant.reset(bias)
    for _ in range(600):
        signal = im_true @ (modes_per_actuator @ (plant.output - bias)) + disturbance
        requested = state - gain * (cm @ signal)
        result = allocator.allocate(requested + modes_per_actuator @ bias)
        state = anti_windup(requested, requested, result.achieved - modes_per_actuator @ bias)
        plant.step(result.command)
        assert np.all(result.command >= 0.0) and np.all(result.command <= 2.0)
    residual = im_true @ (modes_per_actuator @ (plant.output - bias)) + disturbance
    assert np.linalg.norm(residual) < 1e-3 * np.linalg.norm(disturbance)
