"""On-GPU generation of residual wavefronts for training reconstructors.

A residual is ``fitting + DM-space part``:

* fitting error: what the 349-actuator DM cannot correct. Drawn from a bank
  of real pyturb screens with their DM projection removed (exact statistics,
  including the outer scale), rescaled for seeing.
* DM-space part: KL coefficients ``alpha * sigma_i^beta * N(0, 1)``, with a
  per-sample correction level ``alpha`` (log-uniform) and spectral tilt
  ``beta``. ``alpha = 1, beta = 1`` is open-loop turbulence; small ``alpha`` is
  a well-corrected loop.

The label is the DM-space part in modal coefficients (nm). The modal surfaces
are orthonormal over the pupil, so the MSE over modes is the wavefront error
variance of the estimate.
"""

from __future__ import annotations

import math

import torch

from .loop import DM, ModalProjector


class ResidualGenerator:
    def __init__(
        self,
        cfg,
        sysd: dict,
        dm: DM,
        pupil: torch.Tensor,
        bank_size: int = 4096,
        seeing: float = 0.6,
        seed: int = 0,
        device: str = "cuda",
    ):
        from .atmos import make_atmosphere, to_torch

        self.dm = dm
        self.pupil = pupil
        self.device = device
        proj = ModalProjector(dm, pupil)
        atm = make_atmosphere(cfg, seeing=seeing, seed=seed, engine="spectral")
        bank = []
        for _ in range(math.ceil(bank_size / 256)):
            opd = to_torch(atm.sample(256))
            bank.append((opd - dm.opd(proj(opd))).half())
        self.fit_bank = torch.cat(bank)[:bank_size]
        # KL variances are rad^2 at 500 nm for the configured r0 -> nm rms per mode
        var = torch.tensor(sysd["kl_variance"], device=device, dtype=torch.float32)
        self.sigma_nm = var.sqrt() * 500.0 / (2 * math.pi) * (cfg.r0_m / (0.98 * 500e-9 / math.radians(seeing / 3600))) ** (5 / 6)
        self.gen = torch.Generator(device=device).manual_seed(seed)

    def sample(
        self,
        batch: int,
        alpha=(0.02, 1.0),
        beta=(0.6, 1.0),
        seeing_scale=(0.8, 1.3),
    ):
        """Return (residual OPD (B, n, n) metres, label (B, n_modes) nm)."""
        g, dev = self.gen, self.device

        def logu(lo, hi):
            u = torch.rand(batch, 1, device=dev, generator=g)
            return torch.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))

        def uni(lo, hi):
            return lo + (hi - lo) * torch.rand(batch, 1, device=dev, generator=g)

        a, b, s = logu(*alpha), uni(*beta), uni(*seeing_scale)
        sig = self.sigma_nm[None, :]
        # spectral tilt pivoting on mode 2: beta < 1 boosts high orders relative
        # to low orders (closed-loop temporal error is flatter than turbulence)
        shaped = sig * (sig / sig[:, 2:3]) ** (b - 1)
        coeffs = a * s ** (5 / 6) * shaped * torch.randn(batch, sig.shape[1], device=dev, generator=g)
        idx = torch.randint(0, len(self.fit_bank), (batch,), device=dev, generator=g)
        fit = self.fit_bank[idx].float() * s[..., None] ** (5 / 6)
        # random 180-degree rotation of the fitting error doubles the bank
        flip = torch.rand(batch, device=dev, generator=g) < 0.5
        fit = torch.where(flip[:, None, None], fit.flip(-1, -2), fit)
        opd = fit + self.dm.opd(coeffs * 1e-9)
        return opd, coeffs
