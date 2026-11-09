"""Stability of a matrix integrator when the plant is not the calibrated one.

:func:`pyrtc.modal_gains.max_stable_gain` gives the gain limit of one mode
behind a pure delay, assuming the control matrix inverts the plant exactly.
A real loop departs from that in ways that couple the modes: the interaction
matrix was measured with noise or has drifted (misregistration), modes were
truncated, the corrector has dynamics of its own, per-mode gains differ.

This module models the loop pyrtc runs,

    command[k+1] = (1 - leak) * command[k] - gains * (control_matrix @ signal[k])
    signal[k]    = interaction_matrix @ y[k - delay_frames]

where ``y`` is the command itself for a static corrector, or the output of a
:class:`pyrtc.dynamics.DiscretePlant` driven by the command. The loop is
stable when every eigenvalue of its state-transition matrix lies inside the
unit circle. ``interaction_matrix`` is the *true* response, which need not
be the one the control matrix was computed from.

With ``interaction_matrix @ control_matrix`` equal to the identity, no leak
and no plant, :func:`max_stable_gain` reproduces
:func:`pyrtc.modal_gains.max_stable_gain`.

Plain NumPy, no streams.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "closed_loop_eigenvalues",
    "closed_loop_matrix",
    "is_stable",
    "max_stable_gain",
    "spectral_radius",
]


def closed_loop_matrix(
    control_matrix,
    interaction_matrix,
    gain=1.0,
    *,
    leak: float = 0.0,
    delay_frames: int = 0,
    plant=None,
) -> np.ndarray:
    """Return the state-transition matrix of the integrator loop.

    Parameters
    ----------
    control_matrix : array_like, shape (num_modes, num_signals)
        Reconstructor the loop applies.
    interaction_matrix : array_like, shape (num_signals, num_outputs)
        True signal per unit of plant output. Without ``plant`` the plant
        output is the command, so ``num_outputs == num_modes``.
    gain : float or array_like, shape (num_modes,)
        Integrator gain, scalar or per mode.
    leak : float
        Integrator leak.
    delay_frames : int
        Whole frames between the plant output and the signal the loop uses.
    plant : pyrtc.dynamics.DiscretePlant, optional
        Plant between the command and the measured quantity, discretized at
        the loop period. Its inputs are the commands.

    Returns
    -------
    numpy.ndarray
        Square matrix acting on the stacked state
        ``[command, plant state, delayed outputs]``.
    """

    cm = np.atleast_2d(np.asarray(control_matrix, dtype=np.float64))
    im = np.atleast_2d(np.asarray(interaction_matrix, dtype=np.float64))
    num_modes = cm.shape[0]
    if im.shape[0] != cm.shape[1]:
        raise ValueError("interaction_matrix rows must match control_matrix columns")
    delay = int(delay_frames)
    if delay < 0:
        raise ValueError("delay_frames must be >= 0")
    gains = np.broadcast_to(np.asarray(gain, dtype=np.float64), (num_modes,))
    update = gains[:, None] * (cm @ im)  # modes per unit of measured plant output

    if plant is None:
        num_states = 0
        output_from_command = np.eye(num_modes)
        output_from_state = np.zeros((num_modes, 0))
    else:
        if plant.num_inputs != num_modes:
            raise ValueError("plant inputs must match the number of modes")
        num_states = plant.num_states
        output_from_command = plant.D
        output_from_state = plant.C
    num_outputs = output_from_command.shape[0]
    if im.shape[1] != num_outputs:
        raise ValueError("interaction_matrix columns must match the plant outputs")

    size = num_modes + num_states + delay * num_outputs
    matrix = np.zeros((size, size))
    command = slice(0, num_modes)
    state = slice(num_modes, num_modes + num_states)
    matrix[command, command] = (1.0 - leak) * np.eye(num_modes)
    if plant is not None:
        matrix[state, state] = plant.Ad
        matrix[state, command] = plant.Bd

    if delay == 0:
        matrix[command, command] -= update @ output_from_command
        matrix[command, state] -= update @ output_from_state
        return matrix

    # Delay line: at frame k, slot j holds y[k - 1 - j], so the last of the
    # ``delay`` slots holds y[k - delay], which is what the loop reads.
    def slot(index: int) -> slice:
        start = num_modes + num_states + index * num_outputs
        return slice(start, start + num_outputs)

    matrix[slot(0), command] = output_from_command
    matrix[slot(0), state] = output_from_state
    for index in range(1, delay):
        matrix[slot(index), slot(index - 1)] = np.eye(num_outputs)
    matrix[command, slot(delay - 1)] -= update
    return matrix


def closed_loop_eigenvalues(control_matrix, interaction_matrix, gain=1.0, **kwargs) -> np.ndarray:
    """Eigenvalues of :func:`closed_loop_matrix` (same arguments)."""

    return np.linalg.eigvals(closed_loop_matrix(control_matrix, interaction_matrix, gain, **kwargs))


def spectral_radius(control_matrix, interaction_matrix, gain=1.0, **kwargs) -> float:
    """Largest eigenvalue magnitude; below 1 means stable.

    Perturbations decay by this factor per frame at worst, so
    ``-1 / log(spectral_radius)`` is the slowest convergence time in frames.
    """

    eigenvalues = closed_loop_eigenvalues(control_matrix, interaction_matrix, gain, **kwargs)
    return float(np.max(np.abs(eigenvalues)))


def is_stable(control_matrix, interaction_matrix, gain=1.0, **kwargs) -> bool:
    """True when every closed-loop eigenvalue lies inside the unit circle."""

    return spectral_radius(control_matrix, interaction_matrix, gain, **kwargs) < 1.0


def max_stable_gain(
    control_matrix,
    interaction_matrix,
    *,
    modal_gains=1.0,
    upper: float = 4.0,
    resolution: float = 1e-4,
    **kwargs,
) -> float:
    """Largest scalar loop gain for which the loop is stable.

    The gain multiplies ``modal_gains``. Returns 0.0 when no positive gain
    is stable (for example, a sign error in one mode), and ``upper`` when
    every gain up to ``upper`` is stable. The search assumes one stable
    interval starting at zero, which holds for an integrator loop.
    """

    modal_gains = np.asarray(modal_gains, dtype=np.float64)

    def stable(gain: float) -> bool:
        return is_stable(control_matrix, interaction_matrix, gain * modal_gains, **kwargs)

    if stable(upper):
        return float(upper)
    low, high = 0.0, float(upper)
    while high - low > resolution:
        mid = 0.5 * (low + high)
        if stable(mid):
            low = mid
        else:
            high = mid
    return low
