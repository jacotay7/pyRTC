"""Shack-Hartmann baseline: makewfs optics + getframes detector, torch centroiding.

The SH model is deliberately *not* our own optics code: makewfs (validated
against Keck HAKA RTC data) renders the frames, so the baseline is independent
of the focal-plane imager it is compared with.
"""

from __future__ import annotations

import pathlib
import tempfile
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SHConfig:
    lenslets: int = 20
    pixels_per_subaperture: int = 4  # KECK: see KECK_BASELINE.md
    spot_sampling: float = 0.5  # px per lambda/d at the reference wavelength
    reference_wavelength_m: float = 0.7e-6
    wavelengths_m: tuple = (0.55e-6, 0.70e-6, 0.85e-6)
    photons_per_frame: float = 1e5  # detector-surface photons per frame
    frame_rate_hz: float = 1000.0
    em_gain: float = 600.0
    threshold_e: float = 0.0  # centroid threshold, electrons
    noise: bool = True
    # Pupil samples per lenslet inside makewfs. The subaperture field of view
    # (pixels / spot_sampling lambda/d) must stay below this, or makewfs's
    # sampled DFT sums periodic spot replicas (flux 5x too high at 0.32 px,
    # 4 px with 6 samples; makewfs issue filed).
    samples_per_lenslet: int = 16


def _toml(cfg: SHConfig, grid_m: float, n: int, mask_path: str) -> str:
    rate = cfg.photons_per_frame * cfg.frame_rate_hz
    w = ", ".join(f"{x:.6e}" for x in cfg.wavelengths_m)
    weights = ", ".join("1.0" for _ in cfg.wavelengths_m)
    return f"""
schema_version = 1
[input]
quantity = "opd"
unit = "m"
shape = [{n}, {n}]
grid_extent_m = {grid_m}
[telescope]
pupil_diameter_m = 10.95
custom_mask_path = "{mask_path}"
[source]
kind = "ngs"
normalization = "detector_photon_rate"
detector_photon_rate_per_s = {rate:.6e}
wavelengths_m = [{w}]
wavelength_weights = [{weights}]
[sensor]
kind = "shack_hartmann"
wavelength_m = {cfg.reference_wavelength_m:.6e}
[shack_hartmann]
lenslets_across_pupil = {cfg.lenslets}
pixels_per_subaperture = {cfg.pixels_per_subaperture}
spot_sampling_pixels_per_lambda_over_d = {cfg.spot_sampling}
minimum_illuminated_fraction = 0.5
[detector]
preset = "andor_ocam2k"
exposure_s = {1.0 / cfg.frame_rate_hz:.6e}
temperature_c = -45.0
precision = "float32"
[numerics]
device = "gpu"
dtype = "float32"
fft_oversampling = 4
pupil_samples_per_lenslet = {cfg.samples_per_lenslet}
pupil_supersampling = 1
"""


class SHSensor:
    def __init__(self, cfg: SHConfig, pupil: np.ndarray, grid_m: float, keck_cfg=None):
        import makewfs

        from .system import KeckConfig, keck_pupil

        self.cfg = cfg
        fov = cfg.pixels_per_subaperture / cfg.spot_sampling
        if fov >= cfg.samples_per_lenslet:
            raise ValueError(f"subaperture FOV {fov:.1f} lambda/d needs > {fov:.0f} samples per lenslet")
        n_in = pupil.shape[0]
        # makewfs wants the mask on its internal grid (samples_per_lenslet per lenslet)
        kc = keck_cfg or KeckConfig()
        from dataclasses import replace as _replace

        pupil = keck_pupil(_replace(kc, n_pupil=cfg.lenslets * cfg.samples_per_lenslet))
        self._tmp = tempfile.TemporaryDirectory()
        mask = pathlib.Path(self._tmp.name) / "pupil.npy"
        np.save(mask, pupil.astype(np.float64))
        toml = pathlib.Path(self._tmp.name) / "sh.toml"
        toml.write_text(_toml(cfg, grid_m, n_in, str(mask)))
        wfs = makewfs.WavefrontSensor.from_toml(toml)
        self.wfs = wfs
        self.valid = torch.as_tensor(np.asarray(_np(wfs.valid_subapertures())), dtype=torch.bool)
        self.n_valid = int(self.valid.sum())
        cam = getattr(wfs.detector, "camera", None)
        self.gain_adu_per_e = None
        self.bias_adu = None
        if cam is not None:
            c = cam.config
            self.bias_adu = float(getattr(c, "bias_offset_adu", 0.0))
            em = float(getattr(c, "em_gain", 1.0) or 1.0)
            self.gain_adu_per_e = em / float(getattr(c, "gain_e_per_adu", 1.0))
        self.ref_slopes = None

    # ---- frames ---------------------------------------------------------------
    def frame_e(self, opd: torch.Tensor, seed: int | None = None) -> torch.Tensor:
        """(n, n) or (S, n, n) OPD [m] -> electrons-like image (H, W) as torch."""
        import cupy as cp

        if opd.dim() == 3 and opd.shape[0] > 1:
            samples = [cp.from_dlpack(o.contiguous().double()) for o in opd]
            if self.cfg.noise:
                fr = self.wfs.expose_integrated(samples, seed=seed)
            else:
                fr = None
                rate = sum(self.wfs.photon_rate(s) for s in samples) / len(samples)
        else:
            o = cp.from_dlpack((opd[0] if opd.dim() == 3 else opd).contiguous().double())
            fr = self.wfs.expose(o, seed=seed) if self.cfg.noise else None
            rate = None if self.cfg.noise else self.wfs.photon_rate(o)
        if fr is None:
            img = torch.from_dlpack(cp.ascontiguousarray(rate)).float() / self.cfg.frame_rate_hz
            return img
        adu = torch.from_dlpack(cp.ascontiguousarray(fr.data)).float()
        return (adu - self.bias_adu) / self.gain_adu_per_e

    def slopes(self, img: torch.Tensor) -> torch.Tensor:
        """Centre-of-gravity slopes (2 * n_valid,) in pixels."""
        p, n = self.cfg.pixels_per_subaperture, self.cfg.lenslets
        h, w = img.shape
        oy, ox = (h - n * p) // 2, (w - n * p) // 2
        cells = img[oy : oy + n * p, ox : ox + n * p].reshape(n, p, n, p).permute(0, 2, 1, 3)
        cells = (cells - self.cfg.threshold_e).clamp_min(0)
        c = torch.arange(p, device=img.device, dtype=img.dtype) - (p - 1) / 2
        tot = cells.sum((-1, -2)).clamp_min(1e-6)
        sx = (cells.sum(-2) * c).sum(-1) / tot
        sy = (cells.sum(-1) * c).sum(-1) / tot
        v = self.valid.to(img.device)
        s = torch.cat([sx[v], sy[v]])
        return s if self.ref_slopes is None else s - self.ref_slopes

    # ---- calibration ----------------------------------------------------------
    def calibrate(self, dm, n_modes: int, amp_m: float = 20e-9, rcond: float = 1e-3):
        """Noise-free push-pull modal interaction matrix and its pseudo-inverse."""
        noise = self.cfg.noise
        self.cfg.noise = False
        dev = dm.surfaces.device
        zero = torch.zeros(dm.n, dm.n, device=dev)
        self.ref_slopes = None
        self.ref_slopes = self.slopes(self.frame_e(zero))
        cols = []
        for i in range(n_modes):
            e = torch.zeros(dm.surfaces.shape[0], device=dev)
            e[i] = amp_m
            sp = self.slopes(self.frame_e(dm.opd(e)))
            sm = self.slopes(self.frame_e(dm.opd(-e)))
            cols.append((sp - sm) / (2 * amp_m))
        self.cfg.noise = noise
        self.im = torch.stack(cols, 1)  # (2 n_valid, n_modes) px per metre
        u, s, vh = torch.linalg.svd(self.im.double(), full_matrices=False)
        keep = s > rcond * s[0]
        self.rec = (vh[keep].T @ torch.diag(1 / s[keep]) @ u[:, keep].T).float()
        return s

    def __call__(self, residual: torch.Tensor, seed: int | None = None) -> torch.Tensor:
        """Residual (S, B, n, n) -> modal estimate (B, n_modes) metres."""
        out = []
        for b in range(residual.shape[1]):
            img = self.frame_e(residual[:, b], seed=seed)
            out.append(self.rec @ self.slopes(img))
        return torch.stack(out)


def _np(x):
    try:
        import cupy as cp

        return cp.asnumpy(x)
    except Exception:
        return np.asarray(x)
