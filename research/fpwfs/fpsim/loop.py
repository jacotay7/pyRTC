"""Closed-loop harness in simulated time.

Timing model (``delay`` in frames): frame ``k`` integrates the residual during
``[kT, (k+1)T)``. Its measurement updates the integrator, and the new command
is on the DM for frames ``k + delay`` onward. ``delay = 1`` is an ideal RTC
(command applied at the end of the exposure); ``delay = 2`` adds one frame of
readout + compute, the usual model of a real system.

Everything is batched over independent atmospheres: arrays are (B, ...).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import torch

from .optics import FocalPlaneImager


@dataclass
class LoopResult:
    strehl_se: torch.Tensor  # (steps, B) short-exposure Strehl at the science band
    residual_nm: torch.Tensor  # (steps, B) pupil rms residual OPD
    le_psf: torch.Tensor  # (B, npix, npix) long exposure after `settle`
    strehl_le: torch.Tensor  # (B,)
    extras: dict = field(default_factory=dict)


class DM:
    """Modal DM: commands are modal coefficients (metres of OPD, unit-rms surfaces)."""

    def __init__(self, ifs: torch.Tensor, m2c: torch.Tensor, n: int):
        self.surfaces = (ifs @ m2c).T.contiguous()  # (n_modes, n*n), unit rms over pupil
        self.n = n

    def opd(self, modes: torch.Tensor) -> torch.Tensor:
        return (modes @ self.surfaces).reshape(*modes.shape[:-1], self.n, self.n)


class ModalProjector:
    """Least-squares projection of a pupil OPD onto the DM modes (ideal WFS)."""

    def __init__(self, dm: DM, pupil: torch.Tensor):
        mask = pupil.reshape(-1) > 0.5
        a = dm.surfaces[:, mask].T.double()  # (npts, n_modes)
        self.mask = mask
        self.pinv = torch.linalg.pinv(a).float()  # (n_modes, npts)

    def __call__(self, opd: torch.Tensor) -> torch.Tensor:
        flat = opd.reshape(*opd.shape[:-2], -1)[..., self.mask]
        flat = flat - flat.mean(-1, keepdim=True)
        return flat @ self.pinv.T


def pupil_rms(opd: torch.Tensor, pupil: torch.Tensor) -> torch.Tensor:
    mask = pupil.reshape(-1) > 0.5
    flat = opd.reshape(*opd.shape[:-2], -1)[..., mask]
    return flat.std(-1)


def run_loop(
    turbulence,
    dm: DM,
    reconstructor,
    science: FocalPlaneImager,
    pupil: torch.Tensor,
    rate_hz: float,
    steps: int,
    gain: float | torch.Tensor = 0.4,
    leak: float = 1.0,
    delay: int = 2,
    settle: int = 200,
    substeps: int = 1,
    closed: bool = True,
    callback=None,
) -> LoopResult:
    """Run the loop; ``reconstructor(residual_opd, frame_index, dm_history)``.

    The reconstructor sees only what an RTC would: it is handed the residual
    OPD *solely* so it can render its own sensor frame (the sensor model lives
    with the reconstructor), plus the history of applied modal commands.
    """
    dt = 1.0 / rate_hz
    b = len(turbulence.atms)
    n_modes = dm.surfaces.shape[0]
    dev = pupil.device
    cmd = torch.zeros(b, n_modes, device=dev)
    pending = deque([torch.zeros(b, n_modes, device=dev) for _ in range(delay)])
    applied_hist = deque(maxlen=8)
    ref_peak = science(torch.zeros_like(pupil)).amax()
    se, rms = [], []
    le = torch.zeros(b, science.npix, science.npix, device=dev)
    n_le = 0
    if hasattr(reconstructor, "reset"):
        reconstructor.reset(b)
    for k in range(steps):
        applied = pending.popleft()
        applied_hist.append(applied)
        dm_opd = dm.opd(applied)
        # integrate the exposure over `substeps` atmosphere samples
        res_sub = []
        for _ in range(substeps):
            res_sub.append(turbulence.step(dt / substeps) - dm_opd)
        residual = torch.stack(res_sub, 0)  # (S, B, n, n)
        mid = residual[substeps // 2]
        psf = science(mid)
        se.append(psf.amax(dim=(-2, -1)) / ref_peak)
        rms.append(pupil_rms(mid, pupil) * 1e9)
        if k >= settle:
            le += psf
            n_le += 1
        est = reconstructor(residual, k, list(applied_hist)) if closed else torch.zeros_like(cmd)
        cmd = leak * cmd + gain * est
        pending.append(cmd.clone())
        if callback is not None:
            callback(k, residual=mid, est=est, cmd=cmd, applied=applied)
    se_t, rms_t = torch.stack(se).cpu(), torch.stack(rms).cpu()
    le = le / max(n_le, 1)
    return LoopResult(se_t, rms_t, le, (le.amax(dim=(-2, -1)) / ref_peak).cpu())


def h_band_science(pupil: torch.Tensor, grid_m: float, npix: int = 128) -> FocalPlaneImager:
    """NIRC2-like H-band science imager (monochromatic 1.65 um, Nyquist at grid)."""
    return FocalPlaneImager(pupil, grid_m, npix, 2.0, 1.65e-6)


def k_band_science(pupil: torch.Tensor, grid_m: float, npix: int = 128) -> FocalPlaneImager:
    return FocalPlaneImager(pupil, grid_m, npix, 2.0, 2.2e-6)


def marechal(rms_nm: torch.Tensor, lam_nm: float) -> torch.Tensor:
    return torch.exp(-((2 * math.pi * rms_nm / lam_nm) ** 2))
