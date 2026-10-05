"""Focal-plane wavefront sensor: imager + detector noise + preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .optics import FocalPlaneImager, add_noise, defocus_opd


@dataclass
class FPSensorConfig:
    wavelength: float = 1.65e-6  # band centre
    bandwidth: float = 0.0  # fractional; 0 = monochromatic
    n_wavelengths: int = 1
    npix: int = 64  # output pixels across (after binning)
    sampling: float = 2.0  # RENDER px per lambda_c / grid (>= 2); output px = binning / sampling
    binning: int = 1  # detector binning: render npix*binning, sum binning x binning blocks
    defocus_rad: float = 0.0  # rms, at band centre
    photons: float = 1e4  # per frame, entering the pupil after throughput
    read_noise: float = 0.6  # e- rms (C-RED One-class eAPD)
    background: float = 0.0  # e- / px / frame
    excess_noise: bool = False


class FocalPlaneSensor(torch.nn.Module):
    def __init__(self, cfg: FPSensorConfig, pupil: torch.Tensor, grid_m: float, diameter_m=10.95):
        """``pupil`` may be finer than the OPDs passed to ``frame`` (large defocus needs
        a finer grid to avoid phase aliasing); OPDs are then bilinearly upsampled."""
        super().__init__()
        if cfg.sampling < 2.0:
            # FocalPlaneImager point-samples the field; detector pixels integrate. Below
            # Nyquist the point samples misrepresent the flux (~10 % captured at 0.5 px per
            # lambda/D). Binned modes need rendering at >= Nyquist and summing pixels.
            raise ValueError("sampling < 2 px per lambda/grid is not modelled (render at Nyquist and bin)")
        self.cfg = cfg
        lam = cfg.wavelength
        if cfg.bandwidth > 0 and cfg.n_wavelengths > 1:
            lams = torch.linspace(
                lam * (1 - cfg.bandwidth / 2), lam * (1 + cfg.bandwidth / 2), cfg.n_wavelengths
            ).tolist()
        else:
            lams = [lam]
        static = defocus_opd(
            pupil.shape[-1], grid_m, diameter_m, cfg.defocus_rad * lam / (2 * torch.pi), pupil.device
        )
        self.imager = FocalPlaneImager(
            pupil, grid_m, cfg.npix * cfg.binning, cfg.sampling, lam, tuple(lams), static_opd=static
        )
        self.n_pupil = pupil.shape[-1]

    def _upsample(self, opd: torch.Tensor) -> torch.Tensor:
        n = opd.shape[-1]
        if n == self.n_pupil:
            return opd
        lead = opd.shape[:-2]
        up = torch.nn.functional.interpolate(
            opd.reshape(-1, 1, n, n), size=(self.n_pupil, self.n_pupil), mode="bilinear", align_corners=False
        )
        return up.reshape(*lead, self.n_pupil, self.n_pupil)

    def frame(self, opd: torch.Tensor, noise: bool = True, generator=None) -> torch.Tensor:
        """Electrons per pixel. ``opd`` may be (S, B, n, n): sub-exposures are averaged."""
        img = self.imager(self._upsample(opd))
        if img.dim() == 4:
            img = img.mean(0)
        b = self.cfg.binning
        if b > 1:  # detector binning: pixels integrate
            n = self.cfg.npix
            img = img.reshape(*img.shape[:-2], n, b, n, b).sum((-1, -3))
        if not noise:
            return img * self.cfg.photons
        c = self.cfg
        return add_noise(img, c.photons, c.read_noise, c.background, c.excess_noise, generator)

    def preprocess(self, electrons: torch.Tensor) -> torch.Tensor:
        """Flux-normalised, sqrt-stretched image (B, 1, npix, npix) for a network."""
        flux = electrons.sum((-2, -1), keepdim=True).clamp_min(1.0)
        x = electrons.clamp_min(0) / flux
        return (x * self.cfg.npix**2).sqrt().unsqueeze(1)
