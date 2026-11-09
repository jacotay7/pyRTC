"""Bounded command allocation and integrator anti-windup."""

import numpy as np
import pytest

from pyrtc.allocation import AllocationResult, BoundedAllocator, anti_windup, clip_allocation


def _problem(seed=0, outputs=6, actuators=10):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(outputs, actuators)), rng.normal(size=outputs)


def test_inactive_limits_give_the_least_squares_solution():
    effect, target = _problem(outputs=10, actuators=6)
    result = BoundedAllocator(effect, -100.0, 100.0).allocate(target)
    assert np.allclose(result.command, np.linalg.lstsq(effect, target, rcond=None)[0])
    assert result.num_saturated == 0
    assert np.allclose(result.residual, effect @ result.command - target)


def test_identity_effect_reduces_to_clipping():
    target = np.array([-3.0, -0.5, 0.2, 4.0])
    result = BoundedAllocator(np.eye(4), -1.0, 2.0).allocate(target)
    assert np.allclose(result.command, np.clip(target, -1.0, 2.0))
    assert list(result.saturated) == [True, False, False, True]


def test_redundant_actuators_make_up_for_a_saturated_one():
    # Two actuators drive one output; the first runs out of range.
    effect = np.array([[1.0, 1.0]])
    allocator = BoundedAllocator(effect, [-1.0, -10.0], [1.0, 10.0], regularization=1e-6)
    result = allocator.allocate([5.0])
    assert result.achieved[0] == pytest.approx(5.0, abs=1e-4)
    assert result.command[0] == pytest.approx(1.0)
    clipped = clip_allocation(effect, [5.0], [-1.0, -10.0], [1.0, 10.0])
    assert abs(clipped.residual[0]) > 1.0  # clipping leaves the request unmet


def test_beats_clipping_in_the_matched_space():
    effect, target = _problem(seed=3, outputs=8, actuators=12)
    target = 5.0 * target
    bounded = BoundedAllocator(effect, -0.5, 0.5).allocate(target)
    clipped = clip_allocation(effect, target, -0.5, 0.5)
    assert np.all(np.abs(bounded.command) <= 0.5 + 1e-12)
    assert np.linalg.norm(bounded.residual) < 0.9 * np.linalg.norm(clipped.residual)


def test_one_sided_limits():
    # Actuators that can only push: lower limit 0, no upper limit.
    effect = np.array([[1.0, -1.0], [1.0, 1.0]])
    result = BoundedAllocator(effect, 0.0, np.inf).allocate([-2.0, 2.0])
    assert np.all(result.command >= 0.0)
    assert np.allclose(result.command, [0.0, 2.0])
    assert list(result.saturated) == [True, False]


def test_fast_method_matches_exact():
    effect, target = _problem(seed=5, outputs=12, actuators=20)
    kwargs = {"lower": -0.3, "upper": 0.4, "regularization": 1e-3}
    exact = BoundedAllocator(effect, **kwargs).allocate(3.0 * target)
    fast = BoundedAllocator(effect, method="fast", max_iterations=5000, **kwargs).allocate(
        3.0 * target
    )
    assert np.allclose(fast.command, exact.command, atol=1e-6)
    assert exact.num_saturated > 0


def test_fast_method_warm_start_needs_few_iterations():
    effect, target = _problem(seed=7, outputs=12, actuators=20)
    allocator = BoundedAllocator(
        effect, -0.3, 0.4, regularization=1e-3, method="fast", max_iterations=5000
    )
    cold = allocator.allocate(3.0 * target)
    repeat = allocator.allocate(3.0 * target, warm_start=cold.command)
    assert repeat.iterations <= 3
    assert np.allclose(repeat.command, cold.command, atol=1e-7)
    nearby = allocator.allocate(3.0 * target * 1.001, warm_start=cold.command)
    assert nearby.iterations < cold.iterations


def test_fast_method_respects_iteration_budget():
    effect, target = _problem(seed=9)
    result = BoundedAllocator(effect, -0.1, 0.1, method="fast", max_iterations=3).allocate(
        10 * target
    )
    assert result.iterations <= 3
    assert np.all(np.abs(result.command) <= 0.1)


def test_weights_select_which_outputs_are_matched():
    effect = np.array([[1.0], [1.0]])
    result = BoundedAllocator(effect, -10, 10, weights=[1.0, 0.0]).allocate([1.0, 3.0])
    assert result.command[0] == pytest.approx(1.0)


def test_regularization_pulls_toward_preferred():
    effect = np.array([[1.0, 1.0]])
    allocator = BoundedAllocator(effect, -10, 10, regularization=1e-6, preferred=[2.0, 0.0])
    assert np.allclose(allocator.allocate([2.0]).command, [2.0, 0.0], atol=1e-4)
    assert np.allclose(
        allocator.allocate([2.0], preferred=[0.0, 2.0]).command, [0.0, 2.0], atol=1e-4
    )


def test_fixed_actuators_are_held():
    effect = np.eye(3)
    result = BoundedAllocator(effect, [0.5, -1, -1], [0.5, 1, 1]).allocate([0.0, 0.2, 5.0])
    assert np.allclose(result.command, [0.5, 0.2, 1.0])
    all_fixed = BoundedAllocator(effect, 0.25, 0.25).allocate([1.0, 1.0, 1.0])
    assert np.allclose(all_fixed.command, 0.25)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lower": 1.0, "upper": 0.0},
        {"regularization": -1.0},
        {"method": "magic"},
        {"max_iterations": 0},
        {"weights": [-1.0, 1.0]},
        {"lower": np.nan},
    ],
)
def test_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        BoundedAllocator(np.eye(2), **kwargs)


def test_invalid_shapes():
    with pytest.raises(ValueError):
        BoundedAllocator(np.ones(3))
    with pytest.raises(ValueError):
        BoundedAllocator(np.eye(2)).allocate([1.0, 2.0, 3.0])


def test_clip_allocation_reports_clipped_actuators():
    result = clip_allocation(np.eye(2), [0.5, 3.0], -1.0, 1.0)
    assert isinstance(result, AllocationResult)
    assert list(result.saturated) == [False, True]


def test_anti_windup_resets_state_to_achieved():
    requested = np.array([3.0, -1.0])
    achieved = np.array([1.0, -1.0])
    assert np.allclose(anti_windup(requested, requested, achieved), achieved)
    assert np.allclose(anti_windup(requested, requested, achieved, gain=0.5), [2.0, -1.0])
    with pytest.raises(ValueError):
        anti_windup(requested, requested, achieved, gain=1.5)


def test_anti_windup_shortens_recovery_from_saturation():
    """An integrator pushed past a limit recovers at once with anti-windup."""

    def recovery_frames(use_anti_windup):
        allocator = BoundedAllocator(np.eye(1), -1.0, 1.0)
        state = np.zeros(1)
        frames_after_flip = None
        for frame in range(3000):
            disturbance = 5.0 if frame < 200 else -0.5  # unreachable, then reachable
            delivered = allocator.allocate(state).command
            error = disturbance + delivered
            if frame >= 200 and abs(error[0]) < 0.05:
                frames_after_flip = frame - 200
                break
            requested = state - 0.3 * error
            state = (
                anti_windup(requested, requested, allocator.clip(requested))
                if use_anti_windup
                else requested
            )
        return frames_after_flip

    with_aw, without_aw = recovery_frames(True), recovery_frames(False)
    assert with_aw is not None and without_aw is not None
    assert with_aw < 20
    assert without_aw > 5 * with_aw
