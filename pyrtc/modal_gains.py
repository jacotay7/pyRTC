"""Per-mode integrator gain optimization from closed-loop telemetry.

The loop applies a gain per controlled mode (``Loop.set_modal_gains``). This
module picks those gains from closed-loop data, following the modal control
optimization of Gendron & Léna (1994, A&A 291, 337):

1. Record the modal residuals the loop measures, ``y = signal @ cm.T``
   (:meth:`pyrtc.loop.Loop.modal_residuals`), while it runs with known gains.
2. With an integrator ``g / (1 - z^-1)`` and a pure delay of ``d`` frames,
   the measured residual is ``y = E_g (phi + n)``, where ``E_g`` is the
   rejection transfer function, ``phi`` the disturbance and ``n`` the
   measurement noise. Dividing the residual PSD by ``|E_g0|^2`` gives the
   pseudo open-loop PSD ``P_phi + P_n``.
3. The noise is white, and the disturbance falls steeply with frequency, so
   the high-frequency end of the pseudo open-loop PSD estimates ``P_n``.
4. For each mode, the chosen gain minimizes the predicted residual variance
   ``sum(|E_g|^2 P_phi + |T_g|^2 P_n)`` (``T_g = 1 - E_g``) over stable gains.

Optical gains (the reduced sensitivity of a pyramid WFS on a residual
wavefront) are applied separately: ``Loop.set_optical_gains`` divides the
effective gains by them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _z(freqs: np.ndarray, frame_rate: float) -> np.ndarray:
    return np.exp(2j * np.pi * np.asarray(freqs, dtype=np.float64) / float(frame_rate))


def open_loop_transfer(freqs, frame_rate: float, gain, delay_frames: int) -> np.ndarray:
    """Return the open-loop transfer ``g z^-d / (1 - z^-1)`` at ``freqs`` (Hz).

    ``gain`` may be a scalar or an array broadcast against ``freqs``.
    """

    z = _z(freqs, frame_rate)
    return np.asarray(gain) * z ** (-int(delay_frames)) / (1.0 - 1.0 / z)


def rejection_transfer(freqs, frame_rate: float, gain, delay_frames: int) -> np.ndarray:
    """Return the rejection (error) transfer ``E = 1 / (1 + H_ol)``."""

    return 1.0 / (1.0 + open_loop_transfer(freqs, frame_rate, gain, delay_frames))


def max_stable_gain(delay_frames: int, *, resolution: float = 1e-4) -> float:
    """Return the largest integrator gain that keeps the loop stable.

    The closed-loop poles solve ``z^(d+1) - z^d + g = 0``; the loop is stable
    while every root lies inside the unit circle.
    """

    delay = int(delay_frames)
    if delay < 0:
        raise ValueError("delay_frames must be >= 0")

    def stable(gain: float) -> bool:
        coefficients = np.zeros(delay + 2)
        coefficients[0] = 1.0
        coefficients[1] = -1.0
        coefficients[-1] += gain
        return bool(np.all(np.abs(np.roots(coefficients)) < 1.0))

    low, high = 0.0, 2.0
    while high - low > resolution:
        middle = 0.5 * (low + high)
        low, high = (middle, high) if stable(middle) else (low, middle)
    return low


@dataclass(frozen=True)
class ModalGainResult:
    """Output of :func:`optimize_modal_gains`."""

    gains: np.ndarray
    predicted_residual: np.ndarray
    noise_psd: np.ndarray
    freqs: np.ndarray
    gain_grid: np.ndarray
    max_gain: float


def _one_sided_psd(series: np.ndarray, frame_rate: float, segment: int):
    """Welch PSD per column with a Hann window and 50% overlap."""

    num_samples = series.shape[0]
    segment = int(min(segment, num_samples))
    if segment < 8:
        raise ValueError("need at least 8 residual samples per mode")
    step = segment // 2
    window = np.hanning(segment)
    scale = 1.0 / (frame_rate * np.sum(window**2))
    starts = range(0, num_samples - segment + 1, step)
    psd = None
    count = 0
    for start in starts:
        chunk = series[start : start + segment]
        chunk = chunk - chunk.mean(axis=0)
        spectrum = np.fft.rfft(chunk * window[:, None], axis=0)
        power = scale * np.abs(spectrum) ** 2
        psd = power if psd is None else psd + power
        count += 1
    psd /= count
    psd[1:-1] *= 2.0  # one-sided
    freqs = np.fft.rfftfreq(segment, d=1.0 / frame_rate)
    return freqs, psd


def optimize_modal_gains(
    residuals,
    frame_rate: float,
    current_gains,
    *,
    delay_frames: int = 2,
    gain_grid=None,
    stability_margin: float = 0.9,
    noise_band: float = 0.2,
    segment: int = 256,
) -> ModalGainResult:
    """Choose per-mode integrator gains from closed-loop modal residuals.

    Parameters
    ----------
    residuals : array_like, shape (num_frames, num_modes)
        Modal residuals measured in closed loop (``signal @ cm.T``), one row
        per loop frame, recorded with ``current_gains`` applied.
    frame_rate : float
        Loop frame rate in Hz.
    current_gains : float or array_like, shape (num_modes,)
        Effective integrator gain of each mode while ``residuals`` were
        recorded (loop gain times modal gain, over optical gain).
    delay_frames : int, optional
        Total loop delay in frames between exposure and correction, besides
        the integrator's own sample (``2`` for a typical one-frame read-out
        plus one-frame compute). Default 2.
    gain_grid : array_like, optional
        Candidate gains. Default: 200 values up to ``stability_margin`` times
        the largest stable gain for ``delay_frames``.
    stability_margin : float, optional
        Fraction of the largest stable gain allowed. Default 0.9.
    noise_band : float, optional
        Fraction of the spectrum, at the high-frequency end, used to estimate
        the white noise level. Default 0.2.
    segment : int, optional
        Welch segment length in frames. Default 256.

    Returns
    -------
    ModalGainResult
        ``gains`` (per mode), the ``predicted_residual`` variance at those
        gains, the estimated ``noise_psd`` level per mode, and the frequency
        and gain grids used.
    """

    residuals = np.asarray(residuals, dtype=np.float64)
    if residuals.ndim == 1:
        residuals = residuals[:, None]
    if residuals.ndim != 2:
        raise ValueError("residuals must have shape (num_frames, num_modes)")
    num_modes = residuals.shape[1]
    current = np.broadcast_to(np.asarray(current_gains, dtype=np.float64), (num_modes,))
    if np.any(current <= 0):
        raise ValueError("current_gains must be positive (the loop must be closed)")
    if not 0 < noise_band < 1:
        raise ValueError("noise_band must be between 0 and 1")

    max_gain = max_stable_gain(delay_frames)
    if gain_grid is None:
        gain_grid = np.linspace(max_gain * stability_margin / 200, max_gain * stability_margin, 200)
    gain_grid = np.asarray(gain_grid, dtype=np.float64)
    if np.any(gain_grid <= 0) or np.any(gain_grid >= max_gain):
        raise ValueError(f"gain_grid must lie in (0, {max_gain:.4g}) for this delay")

    freqs, residual_psd = _one_sided_psd(residuals, frame_rate, segment)
    # Leave out DC: the integrator rejects it completely, so E(0) = 0.
    freqs, residual_psd = freqs[1:], residual_psd[1:]

    rejection_now = rejection_transfer(freqs[:, None], frame_rate, current[None, :], delay_frames)
    pseudo_open_loop = residual_psd / np.abs(rejection_now) ** 2

    band = max(1, int(round(noise_band * freqs.size)))
    noise_psd = np.median(pseudo_open_loop[-band:], axis=0)
    disturbance_psd = np.clip(pseudo_open_loop - noise_psd[None, :], 0.0, None)

    # Predicted residual variance for every candidate gain and mode.
    rejection = rejection_transfer(freqs[:, None], frame_rate, gain_grid[None, :], delay_frames)
    rejection_power = np.abs(rejection) ** 2  # (freqs, gains)
    noise_power = np.abs(1.0 - rejection) ** 2
    df = freqs[1] - freqs[0] if freqs.size > 1 else 1.0
    cost = df * (
        rejection_power.T @ disturbance_psd + noise_power.sum(axis=0)[:, None] * noise_psd[None, :]
    )  # (gains, modes)
    best = np.argmin(cost, axis=0)
    return ModalGainResult(
        gains=gain_grid[best],
        predicted_residual=cost[best, np.arange(num_modes)],
        noise_psd=noise_psd,
        freqs=freqs,
        gain_grid=gain_grid,
        max_gain=max_gain,
    )
