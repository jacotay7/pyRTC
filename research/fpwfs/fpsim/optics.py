"""Batched, GPU focal-plane imaging in torch.

Conventions
-----------
* OPD in metres on the ``n_pupil`` grid; phase = 2 pi OPD / lambda.
* Focal-plane coordinates are in units of ``lambda_ref / grid_m`` (the grid
  spans the 20 x 20 subaperture square, slightly larger than the pupil).
  ``sampling`` is pixels per ``lambda_ref / grid_m``; 2 is Nyquist at
  ``lambda_ref`` for the grid diameter.
* Images are normalised so a pixel holds the fraction of the pupil's total
  flux landing in it. ``image.sum()`` < 1 is the light outside the field.
"""

from __future__ import annotations

import math

import torch


class FocalPlaneImager(torch.nn.Module):
    def __init__(
        self,
        pupil: torch.Tensor,
        grid_m: float,
        npix: int,
        sampling: float,
        wavelength_ref: float,
        wavelengths: tuple[float, ...] | None = None,
        weights: tuple[float, ...] | None = None,
        static_opd: torch.Tensor | None = None,
    ):
        super().__init__()
        n = pupil.shape[-1]
        dev = pupil.device
        self.npix, self.n, self.sampling = npix, n, sampling
        self.wavelengths = tuple(wavelengths or (wavelength_ref,))
        w = torch.tensor(weights or [1.0] * len(self.wavelengths), dtype=torch.float64)
        self.register_buffer("weights", (w / w.sum()).float().to(dev))
        self.register_buffer("pupil", pupil.float())
        self.register_buffer(
            "static_opd", (static_opd if static_opd is not None else torch.zeros_like(pupil)).float()
        )
        x = (torch.arange(n, device=dev, dtype=torch.float64) + 0.5) / n - 0.5  # in grid units
        u = (torch.arange(npix, device=dev, dtype=torch.float64) - npix / 2) / sampling
        mats = []
        for lam in self.wavelengths:
            # u is in lambda_ref/grid; at lam the same sky angle is u*lam_ref/lam
            arg = -2 * math.pi * torch.outer(u * wavelength_ref / lam, x)
            mats.append(torch.polar(torch.ones_like(arg), arg).to(torch.complex64))
        self.register_buffer("mft", torch.stack(mats))  # (L, npix, n)
        # Parseval: a zero-padded DFT at q samples per cycle carries (q n)^2 * sum|E|^2
        norm = []
        total = float((self.pupil**2).sum())
        for lam in self.wavelengths:
            q = sampling * lam / wavelength_ref
            norm.append(1.0 / ((q * n) ** 2 * total))
        self.register_buffer("norm", torch.tensor(norm, device=dev, dtype=torch.float32))

    def forward(self, opd: torch.Tensor) -> torch.Tensor:
        """``opd``: (..., n, n) metres -> image (..., npix, npix), flux fraction."""
        opd = opd + self.static_opd
        img = None
        for i, lam in enumerate(self.wavelengths):
            field = torch.polar(self.pupil.expand_as(opd), (2 * math.pi / lam) * opd)
            a = self.mft[i]
            f = a @ field.to(torch.complex64) @ a.transpose(0, 1)
            term = (f.real**2 + f.imag**2) * (self.norm[i] * self.weights[i])
            img = term if img is None else img + term
        return img


def defocus_opd(pupil_n: int, grid_m: float, diameter_m: float, rms_m: float, device) -> torch.Tensor:
    """Zernike defocus (Noll Z4, unit-RMS over a circle of ``diameter_m``) scaled to ``rms_m``."""
    x = ((torch.arange(pupil_n, device=device) + 0.5) / pupil_n - 0.5) * grid_m
    yy, xx = torch.meshgrid(x, x, indexing="ij")
    r = torch.hypot(xx, yy) / (diameter_m / 2)
    return rms_m * math.sqrt(3) * (2 * r**2 - 1)


def strehl(imager: FocalPlaneImager, opd: torch.Tensor) -> torch.Tensor:
    """Peak-ratio Strehl of ``opd`` against the same imager's aberration-free PSF."""
    ref = imager(torch.zeros_like(imager.pupil)).amax()
    return imager(opd).amax(dim=(-2, -1)) / ref


def add_noise(
    img: torch.Tensor,
    photons: float | torch.Tensor,
    read_noise: float = 0.0,
    background: float = 0.0,
    excess_noise: bool = False,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Photon (Poisson, optional EMCCD excess noise) + Gaussian read noise, in electrons."""
    mean = img * photons + background
    if excess_noise:
        # EMCCD high-gain limit: output ~ Gamma(k=n_in, theta=1) for n_in Poisson electrons
        n_in = torch.poisson(mean, generator=generator)
        e = torch.distributions.Gamma(n_in.clamp_min(1e-6), torch.ones_like(n_in)).sample()
        e = torch.where(n_in > 0, e, torch.zeros_like(e))
    else:
        e = torch.poisson(mean, generator=generator)
    if read_noise:
        e = e + read_noise * torch.randn(e.shape, device=e.device, generator=generator)
    return e
