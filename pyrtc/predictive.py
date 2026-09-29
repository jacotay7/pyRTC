"""Model-based predictive control for the modal loop.

A loop running :meth:`pyrtc.loop.Loop.predictive_integrator` estimates the
pseudo open-loop (POL) disturbance of each controlled mode every frame,

    pol_t = residual_t - command_applied_t,

and asks a :class:`ModalPredictor` for the disturbance ``horizon`` frames
ahead, when the next command will be on the corrector. The command cancels
that prediction.

Predictors are pluggable. Each one learns its model from recorded POL data
(:meth:`ModalPredictor.fit`) and then filters online
(:meth:`ModalPredictor.update` / :meth:`ModalPredictor.predict`). Register a
new method with :func:`register_predictor`; the loop builds one from its
``predictor`` config with :func:`make_predictor`. Built in:

``persistence``
    Predicts that the disturbance stays at its last measurement. Used before
    any fit, where it makes the loop a pseudo open-loop integrator.
``ar_kalman``
    Modal LQG: an AR(2) disturbance model per mode, fitted by least squares
    with the measurement noise estimated from the spectrum's floor, and a
    steady-state Kalman filter that predicts ``horizon`` frames ahead.
``least_squares``
    A per-mode linear prediction filter over the last ``order`` POL samples,
    fitted by ridge regression to predict ``horizon`` frames ahead.

All predictors work on ``(num_modes,)`` vectors, one per frame, and are
vectorized over modes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Mapping

import numpy as np

_REGISTRY: dict[str, type["ModalPredictor"]] = {}


def register_predictor(name: str) -> Callable[[type["ModalPredictor"]], type["ModalPredictor"]]:
    """Class decorator registering a :class:`ModalPredictor` under ``name``."""

    def decorator(cls):
        if not issubclass(cls, ModalPredictor):
            raise TypeError(f"{cls.__name__} is not a ModalPredictor")
        _REGISTRY[str(name).lower()] = cls
        cls.name = str(name).lower()
        return cls

    return decorator


def available_predictors() -> tuple[str, ...]:
    """Return the registered predictor names."""

    return tuple(sorted(_REGISTRY))


def make_predictor(conf: Mapping[str, Any] | None, num_modes: int) -> "ModalPredictor":
    """Build a predictor from a ``predictor`` config mapping.

    ``conf["type"]`` names a registered predictor (default ``persistence``);
    the other keys are passed to its constructor, apart from the loop-level
    ``gain``, ``delay_frames`` and ``fit_frames``.
    """

    conf = dict(conf or {})
    kind = str(conf.pop("type", "persistence")).lower()
    for loop_key in ("gain", "delay_frames", "fit_frames"):
        conf.pop(loop_key, None)
    if kind not in _REGISTRY:
        raise ValueError(f"unknown predictor type {kind!r}; available: {available_predictors()}")
    return _REGISTRY[kind](num_modes, **conf)


class ModalPredictor(ABC):
    """Predict each mode's disturbance ``horizon`` frames ahead.

    Subclasses implement :meth:`fit` (learn from ``(num_frames, num_modes)``
    POL data), :meth:`update` (ingest the newest POL vector) and
    :meth:`predict` (return the prediction for ``horizon`` frames after the
    last update). :meth:`reset` clears the online state but keeps the model.
    """

    name = "base"

    def __init__(self, num_modes: int, horizon: int = 2) -> None:
        self.num_modes = int(num_modes)
        self.horizon = int(horizon)
        if self.num_modes < 1:
            raise ValueError("num_modes must be >= 1")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        self.fitted = False

    @abstractmethod
    def fit(self, pol_history) -> "ModalPredictor":
        """Learn the model from recorded POL data; returns ``self``."""

    @abstractmethod
    def update(self, pol) -> None:
        """Ingest the newest ``(num_modes,)`` POL measurement."""

    @abstractmethod
    def predict(self) -> np.ndarray:
        """Return the ``(num_modes,)`` prediction ``horizon`` frames ahead."""

    @abstractmethod
    def reset(self) -> None:
        """Clear the online state, keeping the fitted model."""

    def _history(self, pol_history, minimum: int) -> np.ndarray:
        history = np.asarray(pol_history, dtype=np.float64)
        if history.ndim == 1:
            history = history[:, None]
        if history.ndim != 2 or history.shape[1] != self.num_modes:
            raise ValueError(f"pol_history must have shape (num_frames, {self.num_modes})")
        if history.shape[0] < minimum:
            raise ValueError(f"need at least {minimum} frames of POL history")
        return history


@register_predictor("persistence")
class PersistencePredictor(ModalPredictor):
    """Predict that each mode keeps its last measured value."""

    def __init__(self, num_modes: int, horizon: int = 2) -> None:
        super().__init__(num_modes, horizon)
        self._last = np.zeros(self.num_modes)
        self.fitted = True

    def fit(self, pol_history):
        return self

    def update(self, pol) -> None:
        self._last = np.asarray(pol, dtype=np.float64).reshape(self.num_modes)

    def predict(self) -> np.ndarray:
        return self._last.copy()

    def reset(self) -> None:
        self._last = np.zeros(self.num_modes)


def _noise_variance(series: np.ndarray, band: float, segment: int = 256) -> np.ndarray:
    """White-noise variance per column from the high-frequency PSD floor.

    Uses a Hann-windowed Welch estimate: without the window, leakage from a
    strong low-frequency disturbance (a vibration peak) raises the floor.
    """

    num_frames = series.shape[0]
    segment = int(min(segment, num_frames))
    window = np.hanning(segment)
    step = max(1, segment // 2)
    total = None
    count = 0
    for start in range(0, num_frames - segment + 1, step):
        chunk = series[start : start + segment]
        chunk = chunk - chunk.mean(axis=0)
        power = np.abs(np.fft.rfft(chunk * window[:, None], axis=0)) ** 2
        total = power if total is None else total + power
        count += 1
    # White noise of variance s^2 gives a periodogram level s^2 * sum(w^2).
    level = total / (count * np.sum(window**2))
    band_bins = max(1, int(round(band * (level.shape[0] - 1))))
    return np.median(level[-band_bins:], axis=0)


@register_predictor("ar_kalman")
class ARKalmanPredictor(ModalPredictor):
    """Modal LQG predictor: an AR(2) model per mode with a steady-state Kalman filter.

    Each mode follows ``x_{t+1} = a1 x_t + a2 x_{t-1} + w_t`` and is measured
    as ``y_t = x_t + n_t``. :meth:`fit` estimates ``a1``, ``a2``, the drive
    variance and the noise variance from POL data: ``a1``/``a2`` by
    conditional least squares, the noise variance from the high-frequency
    floor of the spectrum (``noise_band``), and the drive variance from the
    noise-corrected autocovariances. It then iterates the Riccati equation to
    a steady-state Kalman gain.

    Parameters
    ----------
    num_modes : int
        Number of modes.
    horizon : int
        Frames ahead to predict (the loop delay). Default 2.
    noise_band : float
        High-frequency fraction of the spectrum used for the noise floor.
        Default 0.1.
    noise_variance : float or array_like, optional
        Known measurement-noise variance per mode, instead of estimating it.
    """

    def __init__(
        self,
        num_modes: int,
        horizon: int = 2,
        noise_band: float = 0.1,
        noise_variance=None,
    ) -> None:
        super().__init__(num_modes, horizon)
        if not 0 < noise_band < 1:
            raise ValueError("noise_band must be between 0 and 1")
        self.noise_band = float(noise_band)
        self.noise_variance = (
            None
            if noise_variance is None
            else np.broadcast_to(
                np.asarray(noise_variance, dtype=np.float64), (self.num_modes,)
            ).copy()
        )
        self.a1 = np.ones(self.num_modes)
        self.a2 = np.zeros(self.num_modes)
        self.drive_variance = np.ones(self.num_modes)
        self.kalman_gain = np.zeros((self.num_modes, 2))
        self.reset()

    def fit(self, pol_history):
        history = self._history(pol_history, 64)
        centred = history - history.mean(axis=0)
        n = centred.shape[0]
        r0 = np.mean(centred * centred, axis=0)
        r1 = np.mean(centred[1:] * centred[:-1], axis=0) * n / (n - 1)
        r2 = np.mean(centred[2:] * centred[:-2], axis=0) * n / (n - 2)
        noise = (
            self.noise_variance
            if self.noise_variance is not None
            else _noise_variance(history, self.noise_band)
        )
        noise = np.minimum(noise, 0.95 * r0)
        s0 = r0 - noise  # disturbance variance
        # Least squares for x_t = a1 x_{t-1} + a2 x_{t-2}. Noise in the
        # regressors inflates the normal matrix's diagonal by (rows * noise)
        # and biases the model towards damping; remove that bias, but by at
        # most half the smallest eigenvalue. For slow, near-unit-root modes a
        # full correction would make the system singular (the failure of
        # noise-corrected Yule-Walker), so they keep a slightly damped, safe
        # model.
        a1 = np.empty(self.num_modes)
        a2 = np.empty(self.num_modes)
        for mode in range(self.num_modes):
            series = centred[:, mode]
            design = np.column_stack((series[1:-1], series[:-2]))
            target = series[2:]
            normal = design.T @ design
            smallest = float(np.linalg.eigvalsh(normal)[0])
            correction = min(design.shape[0] * float(noise[mode]), 0.5 * max(smallest, 0.0))
            a1[mode], a2[mode] = np.linalg.solve(normal - correction * np.eye(2), design.T @ target)
        a1, a2 = self._stabilize(a1, a2)
        drive = np.maximum(s0 - a1 * r1 - a2 * r2, 1e-12 * np.maximum(s0, 1e-300))
        self.a1, self.a2 = a1, a2
        self.drive_variance = drive
        self.fitted_noise_variance = noise
        self.kalman_gain = self._steady_state_gain(a1, a2, drive, np.maximum(noise, 1e-300))
        self.fitted = True
        self.reset()
        return self

    @staticmethod
    def _stabilize(a1, a2):
        """Shrink AR(2) coefficients whose roots leave the unit circle."""

        a1 = np.array(a1, dtype=np.float64)
        a2 = np.array(a2, dtype=np.float64)
        for index in range(a1.size):
            roots = np.roots([1.0, -a1[index], -a2[index]])
            radius = np.max(np.abs(roots)) if roots.size else 0.0
            if radius >= 0.999:
                roots = roots * (0.999 / radius)
                poly = np.real(np.poly(roots))
                a1[index], a2[index] = -poly[1], -poly[2]
        return a1, a2

    @staticmethod
    def _steady_state_gain(a1, a2, drive, noise, iterations=500):
        """Iterate the Riccati recursion per mode to the steady Kalman gain."""

        num = a1.size
        transition = np.zeros((num, 2, 2))
        transition[:, 0, 0] = a1
        transition[:, 0, 1] = a2
        transition[:, 1, 0] = 1.0
        process = np.zeros((num, 2, 2))
        process[:, 0, 0] = drive
        covariance = np.zeros((num, 2, 2))
        covariance[:, 0, 0] = drive
        gain = np.zeros((num, 2))
        for _ in range(iterations):
            # Update with y = x[0] + n.
            innovation = covariance[:, 0, 0] + noise
            gain = covariance[:, :, 0] / innovation[:, None]
            updated = covariance - gain[:, :, None] * covariance[:, None, 0, :]
            covariance = transition @ updated @ np.transpose(transition, (0, 2, 1)) + process
        return gain

    def update(self, pol) -> None:
        y = np.asarray(pol, dtype=np.float64).reshape(self.num_modes)
        # Predict one step, then correct with the new measurement.
        predicted0 = self.a1 * self._state[:, 0] + self.a2 * self._state[:, 1]
        predicted = np.column_stack((predicted0, self._state[:, 0]))
        innovation = y - predicted[:, 0]
        self._state = predicted + self.kalman_gain * innovation[:, None]

    def predict(self) -> np.ndarray:
        current, previous = self._state[:, 0].copy(), self._state[:, 1].copy()
        for _ in range(self.horizon):
            current, previous = self.a1 * current + self.a2 * previous, current
        return current

    def reset(self) -> None:
        self._state = np.zeros((self.num_modes, 2))


@register_predictor("least_squares")
class LeastSquaresPredictor(ModalPredictor):
    """Per-mode linear prediction filter fitted by ridge regression.

    Predicts ``pol[t + horizon]`` from ``pol[t - order + 1 .. t]`` of the same
    mode. The filter taps minimize the prediction error on the fitted data,
    plus ``regularization`` times the tap energy (relative to the data
    variance).

    Parameters
    ----------
    num_modes : int
        Number of modes.
    horizon : int
        Frames ahead to predict (the loop delay). Default 2.
    order : int
        Number of past samples per mode. Default 8.
    regularization : float
        Relative ridge regularization. Default 1e-3.
    """

    def __init__(
        self, num_modes: int, horizon: int = 2, order: int = 8, regularization: float = 1e-3
    ) -> None:
        super().__init__(num_modes, horizon)
        self.order = int(order)
        if self.order < 1:
            raise ValueError("order must be >= 1")
        self.regularization = float(regularization)
        if self.regularization < 0:
            raise ValueError("regularization must be >= 0")
        # Until fitted, predict persistence (the newest tap is 1).
        self.taps = np.zeros((self.num_modes, self.order))
        self.taps[:, -1] = 1.0
        self.reset()

    def fit(self, pol_history):
        history = self._history(pol_history, self.order + self.horizon + 16)
        num_frames = history.shape[0]
        rows = num_frames - self.order - self.horizon + 1
        taps = np.empty((self.num_modes, self.order))
        for mode in range(self.num_modes):
            series = history[:, mode]
            design = np.lib.stride_tricks.sliding_window_view(series, self.order)[:rows]
            target = series[self.order - 1 + self.horizon : self.order - 1 + self.horizon + rows]
            normal = design.T @ design
            ridge = self.regularization * np.trace(normal) / self.order
            taps[mode] = np.linalg.solve(normal + ridge * np.eye(self.order), design.T @ target)
        self.taps = taps
        self.fitted = True
        self.reset()
        return self

    def update(self, pol) -> None:
        self._window[:, :-1] = self._window[:, 1:]
        self._window[:, -1] = np.asarray(pol, dtype=np.float64).reshape(self.num_modes)

    def predict(self) -> np.ndarray:
        return np.einsum("mo,mo->m", self.taps, self._window)

    def reset(self) -> None:
        self._window = np.zeros((self.num_modes, self.order))
