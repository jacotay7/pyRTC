"""Collect closed-loop trajectories for training sequence/temporal reconstructors.

A *behaviour* controller drives B independent atmospheres; every frame we store
the noise-free sensor frame (photon noise is added at training time, so one
collection serves every photon level), the modal command on the DM during that
frame, and the true modal residual (label). The behaviour can be the ideal
projector corrupted by noise (bootstrapping a dataset) or a trained network
(DAgger: label the states the network itself visits).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .loop import DM, ModalProjector


@dataclass
class Trajectories:
    frames: torch.Tensor  # (T, B, npix, npix) fp16, noise-free, electrons for `photons`
    applied: torch.Tensor  # (T, B, n_modes) fp32 metres, command on the DM during frame t
    truth: torch.Tensor  # (T, B, n_modes) fp32 nm, true modal residual during frame t
    photons: float

    def save(self, path):
        torch.save(self.__dict__, path)

    @classmethod
    def load(cls, path):
        return cls(**torch.load(path))


def collect(
    turbulence,
    dm: DM,
    pupil: torch.Tensor,
    sensor,
    steps: int,
    rate_hz: float = 1000.0,
    delay: int = 2,
    behaviour=None,
    gains=(0.2, 0.6),
    est_noise=(0.0, 0.5),
    dither_nm: float = 0.0,
    open_loop_frames: int = 0,
    seed: int = 0,
    keep_from: int = 0,
) -> Trajectories:
    """Run B loops for ``steps`` frames and record (frame, applied, truth).

    ``behaviour(frames_e, k, applied_history) -> modal estimate (B, n_modes) m``.
    If None, the ideal projector is used, with relative Gaussian noise drawn per
    atmosphere from ``est_noise`` (fraction of the true residual's rms per mode)
    so the dataset covers imperfect-sensor states. Gains are drawn per
    atmosphere from ``gains``. ``open_loop_frames`` keeps the loop open first
    (bootstrap data). ``dither_nm`` adds known random DM dithers (diversity).
    """
    g = torch.Generator(device=pupil.device).manual_seed(seed)
    b = len(turbulence.atms)
    n_modes = dm.surfaces.shape[0]
    dev = pupil.device
    proj = ModalProjector(dm, pupil)
    gain = gains[0] + (gains[1] - gains[0]) * torch.rand(b, 1, device=dev, generator=g)
    noise = est_noise[0] + (est_noise[1] - est_noise[0]) * torch.rand(b, 1, device=dev, generator=g)
    dt = 1.0 / rate_hz
    cmd = torch.zeros(b, n_modes, device=dev)
    pending = [torch.zeros(b, n_modes, device=dev) for _ in range(delay)]
    frames, applied_l, truth = [], [], []
    hist = []
    for k in range(steps):
        applied = pending.pop(0)
        if dither_nm:
            applied = applied + dither_nm * 1e-9 * torch.randn(b, n_modes, device=dev, generator=g)
        hist.append(applied)
        hist = hist[-8:]
        residual = turbulence.step(dt) - dm.opd(applied)
        true_modes = proj(residual)  # metres
        clean = sensor.frame(residual, noise=False)
        if k >= keep_from:
            frames.append(clean.half().cpu())
            applied_l.append(applied.cpu())
            truth.append((true_modes * 1e9).cpu())
        if k < open_loop_frames:
            est = torch.zeros_like(cmd)
        elif behaviour is None:
            rms = true_modes.pow(2).mean(-1, keepdim=True).sqrt()
            est = true_modes + noise * rms * torch.randn(b, n_modes, device=dev, generator=g)
        else:
            noisy = sensor.frame(residual, noise=True)
            est = behaviour(noisy, k, hist)
        cmd = cmd + gain * est
        pending.append(cmd.clone())
    return Trajectories(torch.stack(frames), torch.stack(applied_l), torch.stack(truth), sensor.cfg.photons)


def noisy_frames(clean: torch.Tensor, sensor_cfg, photons: float, scale: torch.Tensor | None = None, generator=None):
    """Add photon + read noise to stored noise-free frames at a (per-sample) photon level."""
    from .optics import add_noise

    img = clean.float() / sensor_cfg.photons  # back to flux fractions
    ph = photons if scale is None else photons * scale
    return add_noise(img, ph, sensor_cfg.read_noise, sensor_cfg.background, sensor_cfg.excess_noise, generator)


def log_uniform(n, lo, hi, device, generator=None):
    u = torch.rand(n, device=device, generator=generator)
    return torch.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
