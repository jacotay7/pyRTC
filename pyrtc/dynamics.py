"""Linear plant dynamics, for correctors and plants that have memory.

pyrtc's loop treats the corrector as static: a command sent this frame is
the shape seen ``frame_delay`` frames later. That holds for a deformable
mirror at a few kHz. It does not hold when the plant responds on the scale
of the loop period or slower: a tip-tilt stage near its resonance, a
woofer, a thermally actuated optic, an offload to a slow mechanism.

:class:`LinearPlant` is a continuous-time linear state-space model::

    dx/dt = A x + B u        y = C x + D u

with the helpers needed to put such a plant in a loop:

- :meth:`LinearPlant.discretize` returns the exact zero-order-hold
  :class:`DiscretePlant` for a loop period, to step once per frame inside a
  simulated corrector;
- :meth:`LinearPlant.dc_gain`, :meth:`LinearPlant.time_constants` and
  :meth:`LinearPlant.settling_time` say how long to wait after a poke before
  measuring an interaction matrix, and how slowly a static-plant loop must
  run;
- :meth:`LinearPlant.step_response` gives the response to a held input.

:func:`pyrtc.loop_stability.closed_loop_matrix` accepts a
:class:`DiscretePlant`, to check an integrator's stability with the plant in
the loop.

Plain NumPy/SciPy, no streams.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

__all__ = ["DiscretePlant", "LinearPlant"]


def _matrix(value, name: str) -> np.ndarray:
    matrix = np.atleast_2d(np.asarray(value, dtype=np.float64))
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix")
    return matrix


class DiscretePlant:
    """Discrete-time plant ``x[k+1] = Ad x[k] + Bd u[k]``, ``y[k] = C x[k] + D u[k]``.

    ``u[k]`` is held over the period that ends at step ``k + 1``, so the
    output read after :meth:`step` is the plant at the end of that period.
    """

    def __init__(self, Ad, Bd, C, D, period: float) -> None:
        self.Ad = _matrix(Ad, "Ad")
        self.Bd = _matrix(Bd, "Bd")
        self.C = _matrix(C, "C")
        self.D = _matrix(D, "D")
        self.period = float(period)
        self.state = np.zeros(self.Ad.shape[0])
        self._last_input = np.zeros(self.Bd.shape[1])

    @property
    def num_states(self) -> int:
        return self.Ad.shape[0]

    @property
    def num_inputs(self) -> int:
        return self.Bd.shape[1]

    @property
    def num_outputs(self) -> int:
        return self.C.shape[0]

    def reset(self, state=None) -> None:
        """Set the state (zeros by default) and forget the last input."""

        self.state = (
            np.zeros(self.num_states)
            if state is None
            else np.asarray(state, dtype=np.float64).reshape(self.num_states).copy()
        )
        self._last_input = np.zeros(self.num_inputs)

    @property
    def output(self) -> np.ndarray:
        """Output for the current state and the most recent input."""

        return self.C @ self.state + self.D @ self._last_input

    def step(self, command) -> np.ndarray:
        """Hold ``command`` for one period and return the output at its end."""

        command = np.asarray(command, dtype=np.float64).reshape(self.num_inputs)
        self.state = self.Ad @ self.state + self.Bd @ command
        self._last_input = command
        return self.output

    def simulate(self, commands) -> np.ndarray:
        """Step through ``commands`` (frames x inputs); return frames x outputs."""

        commands = np.asarray(commands, dtype=np.float64).reshape(-1, self.num_inputs)
        return np.array([self.step(command) for command in commands])


class LinearPlant:
    """Continuous-time linear plant ``dx/dt = A x + B u``, ``y = C x + D u``.

    Parameters
    ----------
    A : array_like, shape (num_states, num_states)
    B : array_like, shape (num_states, num_inputs)
    C : array_like, shape (num_outputs, num_states), optional
        Default: identity (the outputs are the states).
    D : array_like, shape (num_outputs, num_inputs), optional
        Direct feedthrough. Default: zeros.
    """

    def __init__(self, A, B, C=None, D=None) -> None:
        self.A = _matrix(A, "A")
        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError("A must be square")
        self.B = _matrix(B, "B")
        if self.B.shape[0] != self.A.shape[0]:
            raise ValueError("B must have one row per state")
        self.C = np.eye(self.A.shape[0]) if C is None else _matrix(C, "C")
        if self.C.shape[1] != self.A.shape[0]:
            raise ValueError("C must have one column per state")
        self.D = np.zeros((self.C.shape[0], self.B.shape[1])) if D is None else _matrix(D, "D")
        if self.D.shape != (self.C.shape[0], self.B.shape[1]):
            raise ValueError("D must have shape (num_outputs, num_inputs)")

    @property
    def num_states(self) -> int:
        return self.A.shape[0]

    @property
    def num_inputs(self) -> int:
        return self.B.shape[1]

    @property
    def num_outputs(self) -> int:
        return self.C.shape[0]

    @classmethod
    def first_order(cls, time_constants, gains=1.0) -> "LinearPlant":
        """Independent first-order lags ``gain / (tau s + 1)``, one per channel."""

        tau = np.atleast_1d(np.asarray(time_constants, dtype=np.float64))
        if np.any(tau <= 0):
            raise ValueError("time_constants must be positive")
        gains = np.broadcast_to(np.asarray(gains, dtype=np.float64), tau.shape)
        return cls(np.diag(-1.0 / tau), np.diag(gains / tau))

    @classmethod
    def second_order(cls, natural_frequency_hz: float, damping: float, gain: float = 1.0):
        """One resonant channel ``gain w^2 / (s^2 + 2 zeta w s + w^2)``.

        The state is (position, velocity) and the output is the position.
        """

        if natural_frequency_hz <= 0 or damping < 0:
            raise ValueError("natural_frequency_hz must be > 0 and damping >= 0")
        omega = 2.0 * np.pi * natural_frequency_hz
        A = [[0.0, 1.0], [-omega * omega, -2.0 * damping * omega]]
        return cls(A, [[0.0], [gain * omega * omega]], C=[[1.0, 0.0]])

    @classmethod
    def from_network(cls, capacity, conductance, input_map, C=None) -> "LinearPlant":
        """Lumped diffusive network ``capacity * dx/dt = -conductance @ x + input_map @ u``.

        This is the form of a thermal network (heat capacities, a conductance
        matrix that includes the losses, heater inputs) and of an RC circuit.

        Parameters
        ----------
        capacity : array_like, shape (num_nodes,)
            Positive storage of each node.
        conductance : array_like, shape (num_nodes, num_nodes)
            Symmetric positive-definite matrix: off-diagonal ``-g_ij`` for a
            link, diagonal the sum of a node's links plus its loss.
        input_map : array_like, shape (num_nodes, num_inputs)
            Where each input enters.
        """

        capacity = np.asarray(capacity, dtype=np.float64).reshape(-1)
        if np.any(capacity <= 0):
            raise ValueError("capacity must be positive")
        conductance = _matrix(conductance, "conductance")
        input_map = _matrix(input_map, "input_map")
        return cls(-conductance / capacity[:, None], input_map / capacity[:, None], C=C)

    def poles(self) -> np.ndarray:
        """Eigenvalues of ``A``."""

        return np.linalg.eigvals(self.A)

    def is_stable(self) -> bool:
        """True when every pole has a negative real part."""

        return bool(np.all(self.poles().real < 0))

    def time_constants(self) -> np.ndarray:
        """Decay times ``-1 / Re(pole)`` of the modes, slowest first.

        Raises ``ValueError`` for a plant with a pole on or right of the
        imaginary axis (an integrator or an unstable mode never settles).
        """

        real = self.poles().real
        if np.any(real >= 0):
            raise ValueError("the plant has a non-decaying mode")
        return np.sort(-1.0 / real)[::-1]

    def dc_gain(self) -> np.ndarray:
        """Steady-state output per unit held input, ``D - C A^-1 B``."""

        return self.D - self.C @ np.linalg.solve(self.A, self.B)

    def settling_time(self, fraction: float = 0.99) -> float:
        """Upper bound on the time for a step response to cover ``fraction``.

        It is ``-tau_max * ln(1 - fraction)`` for the slowest mode, so it is
        exact when that mode dominates the response and conservative when the
        input barely excites it. Use :meth:`step_response` for a specific
        input.
        """

        if not 0.0 < fraction < 1.0:
            raise ValueError("fraction must be in (0, 1)")
        return float(-self.time_constants()[0] * np.log(1.0 - fraction))

    def discretize(self, period: float) -> DiscretePlant:
        """Exact zero-order-hold discretization for a loop period (seconds)."""

        if period <= 0:
            raise ValueError("period must be positive")
        n, m = self.num_states, self.num_inputs
        # expm([[A, B], [0, 0]] * T) = [[Ad, Bd], [0, I]] also when A is singular.
        block = np.zeros((n + m, n + m))
        block[:n, :n] = self.A
        block[:n, n:] = self.B
        transition = expm(block * period)
        return DiscretePlant(transition[:n, :n], transition[:n, n:], self.C, self.D, period)

    def step_response(self, command, times) -> np.ndarray:
        """Output at ``times`` (seconds, from rest) for ``command`` held from t = 0."""

        command = np.asarray(command, dtype=np.float64).reshape(self.num_inputs)
        n, m = self.num_states, self.num_inputs
        block = np.zeros((n + m, n + m))
        block[:n, :n] = self.A
        block[:n, n:] = self.B
        feedthrough = self.D @ command
        outputs = []
        for time in np.atleast_1d(np.asarray(times, dtype=np.float64)):
            state = expm(block * time)[:n, n:] @ command
            outputs.append(self.C @ state + (feedthrough if time >= 0 else 0.0))
        return np.array(outputs)
