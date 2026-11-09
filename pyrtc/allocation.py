"""Command allocation inside actuator limits, and integrator anti-windup.

A loop asks for a command; the actuators can only deliver part of it once
some of them reach a limit. Clipping each actuator on its own is the closest
*actuator-space* command, but it is rarely the closest in the space that
matters: a clipped actuator changes every mode it takes part in, and the
actuators that still have range do nothing to make up for it.

:class:`BoundedAllocator` solves the allocation as a bounded least-squares
problem instead::

    minimize    || W (E a - t) ||^2  +  rho || a - a_pref ||^2
    subject to  lower <= a <= upper

``a`` is the actuator command, ``E`` (the *effect* matrix) maps it to the
quantity to be matched and ``t`` is the requested value of that quantity.
Typical choices of ``E``:

- the pseudo-inverse of the mode-to-command matrix, to deliver the requested
  controlled modes and let the uncontrolled directions absorb a saturation;
- an interaction matrix, to match the requested sensor signal;
- the identity, which reduces to clipping.

Limits may differ per actuator and may be one-sided (``-inf`` / ``+inf``),
which covers actuators that only push, such as heaters or a mirror resting
on a stop.

When the delivered command differs from the request, an integrator that
keeps integrating the request winds up. :func:`anti_windup` is the usual
back-calculation remedy: move the integrator state toward what was achieved.

Everything here is plain NumPy/SciPy with no streams, so it can be used from
a :class:`~pyrtc.loop.Loop` subclass, a corrector, or offline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear

__all__ = ["AllocationResult", "BoundedAllocator", "anti_windup", "clip_allocation"]


@dataclass(frozen=True)
class AllocationResult:
    """Outcome of one allocation.

    Attributes
    ----------
    command : numpy.ndarray
        Actuator command inside the limits.
    achieved : numpy.ndarray
        ``effect @ command``, the delivered value of the matched quantity.
    residual : numpy.ndarray
        ``achieved - target``.
    saturated : numpy.ndarray
        Boolean mask of the actuators sitting on a limit.
    iterations : int
        Solver iterations used.
    """

    command: np.ndarray
    achieved: np.ndarray
    residual: np.ndarray
    saturated: np.ndarray
    iterations: int

    @property
    def num_saturated(self) -> int:
        return int(np.count_nonzero(self.saturated))


def _as_bound(value, size: int, name: str) -> np.ndarray:
    bound = np.broadcast_to(np.asarray(value, dtype=np.float64), (size,)).copy()
    if np.any(np.isnan(bound)):
        raise ValueError(f"{name} must not contain NaN")
    return bound


class BoundedAllocator:
    """Bounded least-squares allocation of a target onto limited actuators.

    Parameters
    ----------
    effect : array_like, shape (num_outputs, num_actuators)
        Maps an actuator command to the quantity to be matched.
    lower, upper : float or array_like, shape (num_actuators,)
        Actuator limits. ``-numpy.inf`` / ``numpy.inf`` leave a side open.
    weights : array_like, shape (num_outputs,), optional
        Weight of each output in the fit. Default: all ones.
    regularization : float, optional
        ``rho``, the weight of the pull toward ``preferred``. A positive
        value makes the solution unique when ``effect`` has a null space
        (more actuators than outputs) and keeps it near ``preferred``.
    preferred : array_like, shape (num_actuators,), optional
        Command to stay close to. Default: zeros. It can be changed per call.
    method : {"exact", "fast"}, optional
        ``"exact"`` uses SciPy's bounded-variable least squares, which
        converges to the optimum. ``"fast"`` runs a fixed number of
        accelerated projected-gradient steps on the precomputed normal
        equations: its cost per call is constant, and a warm start from the
        previous command makes a few steps enough in closed loop.
    max_iterations : int, optional
        Step count for ``"fast"``; iteration cap for ``"exact"``.
    tolerance : float, optional
        ``"fast"`` stops early once a step changes no actuator by more than
        this fraction of the largest command magnitude.
    """

    def __init__(
        self,
        effect,
        lower=-np.inf,
        upper=np.inf,
        *,
        weights=None,
        regularization: float = 0.0,
        preferred=None,
        method: str = "exact",
        max_iterations: int = 200,
        tolerance: float = 1e-9,
    ) -> None:
        effect = np.asarray(effect, dtype=np.float64)
        if effect.ndim != 2:
            raise ValueError("effect must be a 2D matrix (num_outputs, num_actuators)")
        self.effect = effect
        self.num_outputs, self.num_actuators = effect.shape
        self.lower = _as_bound(lower, self.num_actuators, "lower")
        self.upper = _as_bound(upper, self.num_actuators, "upper")
        if np.any(self.lower > self.upper):
            raise ValueError("lower must not exceed upper")
        if regularization < 0:
            raise ValueError("regularization must be >= 0")
        if method not in ("exact", "fast"):
            raise ValueError("method must be 'exact' or 'fast'")
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        self.regularization = float(regularization)
        self.method = method
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.weights = (
            np.ones(self.num_outputs)
            if weights is None
            else _as_bound(weights, self.num_outputs, "weights")
        )
        if np.any(self.weights < 0):
            raise ValueError("weights must be >= 0")
        self.preferred = (
            np.zeros(self.num_actuators)
            if preferred is None
            else _as_bound(preferred, self.num_actuators, "preferred")
        )

        self._weighted_effect = self.weights[:, None] * effect
        root = np.sqrt(self.regularization)
        # Stacked system [W E; sqrt(rho) I] a = [W t; sqrt(rho) a_pref].
        self._stacked = (
            np.vstack([self._weighted_effect, root * np.eye(self.num_actuators)])
            if self.regularization > 0
            else self._weighted_effect
        )
        self._hessian = self._stacked.T @ self._stacked
        eigenvalues = np.linalg.eigvalsh(self._hessian)
        self._lipschitz = float(max(eigenvalues[-1], np.finfo(np.float64).tiny))
        # Fixed actuators (lower == upper) are legal; SciPy's solver is not
        # given them.
        self._free = self.upper > self.lower

    def clip(self, command) -> np.ndarray:
        """Return ``command`` clipped to the limits."""

        return np.clip(np.asarray(command, dtype=np.float64), self.lower, self.upper)

    def allocate(self, target, *, preferred=None, warm_start=None) -> AllocationResult:
        """Return the in-limits command whose effect is closest to ``target``.

        Parameters
        ----------
        target : array_like, shape (num_outputs,)
            Requested value of ``effect @ command``.
        preferred : array_like, optional
            Overrides the constructor's ``preferred`` for this call.
        warm_start : array_like, optional
            Starting command for ``"fast"`` (typically the previous result).
            Default: ``preferred`` clipped to the limits.
        """

        target = np.asarray(target, dtype=np.float64).reshape(-1)
        if target.size != self.num_outputs:
            raise ValueError(f"target has {target.size} elements, expected {self.num_outputs}")
        preferred = (
            self.preferred
            if preferred is None
            else _as_bound(preferred, self.num_actuators, "preferred")
        )
        rhs = self.weights * target
        if self.regularization > 0:
            rhs = np.concatenate([rhs, np.sqrt(self.regularization) * preferred])

        if self.method == "exact":
            command, iterations = self._solve_exact(rhs, preferred)
        else:
            start = self.clip(preferred if warm_start is None else warm_start)
            command, iterations = self._solve_fast(rhs, start)

        achieved = self.effect @ command
        span = np.maximum(np.abs(self.lower), np.abs(self.upper))
        span = np.where(np.isfinite(span), span, 1.0)
        at_limit = (command <= self.lower + 1e-9 * span) | (command >= self.upper - 1e-9 * span)
        return AllocationResult(command, achieved, achieved - target, at_limit, iterations)

    def _solve_exact(self, rhs: np.ndarray, preferred: np.ndarray):
        command = self.lower.copy()
        free = self._free
        if not np.any(free):
            return command, 0
        reduced_rhs = rhs - self._stacked[:, ~free] @ command[~free]
        solution = lsq_linear(
            self._stacked[:, free],
            reduced_rhs,
            bounds=(self.lower[free], self.upper[free]),
            method="bvls",
            max_iter=self.max_iterations,
        )
        command[free] = np.clip(solution.x, self.lower[free], self.upper[free])
        return command, int(solution.nit)

    def _solve_fast(self, rhs: np.ndarray, start: np.ndarray):
        linear = self._stacked.T @ rhs
        step = 1.0 / self._lipschitz
        command = start
        momentum_point = start
        momentum = 1.0
        iterations = 0
        for iterations in range(1, self.max_iterations + 1):
            gradient = self._hessian @ momentum_point - linear
            updated = np.clip(momentum_point - step * gradient, self.lower, self.upper)
            next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum * momentum))
            change = updated - command
            # Restart the momentum when it points uphill (O'Donoghue & Candes 2015).
            if np.dot(momentum_point - updated, change) > 0:
                next_momentum = 1.0
                momentum_point = updated
            else:
                momentum_point = updated + ((momentum - 1.0) / next_momentum) * change
            scale = max(float(np.max(np.abs(updated))), 1.0)
            command, momentum = updated, next_momentum
            if float(np.max(np.abs(change))) <= self.tolerance * scale:
                break
        return command, iterations


def clip_allocation(effect, target, lower, upper) -> AllocationResult:
    """Allocate by least squares without limits, then clip.

    This is what a corrector with a command cap does. It is here as the
    baseline to compare :class:`BoundedAllocator` against.
    """

    effect = np.asarray(effect, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    unconstrained = np.linalg.lstsq(effect, target, rcond=None)[0]
    lower = _as_bound(lower, effect.shape[1], "lower")
    upper = _as_bound(upper, effect.shape[1], "upper")
    command = np.clip(unconstrained, lower, upper)
    achieved = effect @ command
    return AllocationResult(command, achieved, achieved - target, command != unconstrained, 0)


def anti_windup(state, requested, achieved, gain: float = 1.0) -> np.ndarray:
    """Back-calculation anti-windup for an integrator behind a saturation.

    Returns ``state - gain * (requested - achieved)``: the integrator state
    moved toward the command that was delivered. With ``gain = 1`` and
    ``state == requested`` the state becomes exactly ``achieved``, so the
    integrator resumes from the saturated command as soon as the error
    changes sign. Smaller gains bleed the excess off over ``1 / gain``
    frames, which filters a noisy ``achieved``.

    ``requested`` and ``achieved`` must be expressed in the integrator's own
    coordinates (for a modal integrator, project the delivered actuator
    command back onto the modes).
    """

    if not 0.0 <= gain <= 1.0:
        raise ValueError("gain must be in [0, 1]")
    state = np.asarray(state, dtype=np.float64)
    excess = np.asarray(requested, dtype=np.float64) - np.asarray(achieved, dtype=np.float64)
    return state - gain * excess
