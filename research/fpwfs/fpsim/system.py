"""Keck II system definition: pupil, DM, modal basis.

Geometry numbers marked ``# KECK`` come from KECK_BASELINE.md (published
sources); everything else is a simulation choice.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import sys
from dataclasses import asdict, dataclass, field

import numpy as np

CACHE = pathlib.Path(__file__).resolve().parent.parent / "cache"
MAKEWFS_KECK = pathlib.Path(__file__).resolve().parents[4] / "makewfs/examples/keck_haka/simulate.py"


@dataclass(frozen=True)
class KeckConfig:
    # Pupil sampling. The grid spans the 20 x 20 subaperture square.
    n_pupil: int = 120  # samples across the grid (6 per subaperture)
    subapertures: int = 20  # KECK legacy SH, subapertures across
    subaperture_m: float = 0.5625  # KECK, projected on the primary
    # DM: 21 x 21 Fried grid, actuators kept inside this radius (pitches).
    actuator_radius_pitch: float = 10.5
    coupling: float = 0.15  # Gaussian influence function value at one pitch
    # Modes
    n_modes: int = 300
    r0_m: float = 0.168  # 0.6" at 500 nm, for the KL statistics only
    outer_scale_m: float = 20.0  # KAON 303
    extra: dict = field(default_factory=dict)

    @property
    def grid_m(self) -> float:
        return self.subapertures * self.subaperture_m

    @property
    def pixel_m(self) -> float:
        return self.grid_m / self.n_pupil

    def key(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:12]


def _makewfs_keck():
    spec = importlib.util.spec_from_file_location("makewfs_keck_haka", MAKEWFS_KECK)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses look themselves up here
    spec.loader.exec_module(module)
    return module


def keck_pupil(cfg: KeckConfig) -> np.ndarray:
    """36-segment Keck pupil (makewfs HAKA model, validated against RTC data)."""
    keck = _makewfs_keck()
    return keck.make_keck_pupil((cfg.n_pupil, cfg.n_pupil), grid_extent_m=cfg.grid_m)


def actuator_positions(cfg: KeckConfig) -> np.ndarray:
    """Actuators of the 21 x 21 Fried grid within ``actuator_radius_pitch``."""
    n = cfg.subapertures + 1
    idx = np.arange(n) - cfg.subapertures / 2
    xx, yy = np.meshgrid(idx, idx)
    keep = np.hypot(xx, yy) <= cfg.actuator_radius_pitch
    return np.stack([xx[keep], yy[keep]], axis=1) * cfg.subaperture_m


def pupil_points(cfg: KeckConfig) -> np.ndarray:
    """(x, y) of every pupil-grid sample, row-major, (n_pupil^2, 2) in metres."""
    x = (np.arange(cfg.n_pupil) + 0.5) * cfg.pixel_m - cfg.grid_m / 2
    xx, yy = np.meshgrid(x, x)
    return np.stack([xx.ravel(), yy.ravel()], axis=1)


def influence_functions(cfg: KeckConfig, positions: np.ndarray) -> np.ndarray:
    """Gaussian influence functions on the pupil grid, (n_pupil^2, n_act)."""
    from aobasis import gaussian_influence_functions

    return gaussian_influence_functions(
        positions, pupil_points(cfg), coupling=cfg.coupling, pitch=cfg.subaperture_m
    )


def build(cfg: KeckConfig = KeckConfig(), refresh: bool = False) -> dict:
    """Pupil, actuators, influence functions, and DM-KL modes (cached)."""
    from aobasis import DMKLBasisGenerator

    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"keck_{cfg.key()}.npz"
    if path.exists() and not refresh:
        return dict(np.load(path))
    pupil = keck_pupil(cfg)
    pos = actuator_positions(cfg)
    ifs = influence_functions(cfg, pos)
    inside = pupil.ravel() > 0.5
    pts = pupil_points(cfg)[inside]
    gen = DMKLBasisGenerator(
        pos, pts, ifs[inside], fried_parameter=cfg.r0_m, outer_scale=cfg.outer_scale_m
    )
    m2c = gen.generate(cfg.n_modes, ignore_piston=True)  # (n_act, n_modes) commands
    out = dict(
        pupil=pupil,
        actuators=pos,
        ifs=ifs.astype(np.float32),
        m2c=m2c.astype(np.float32),
        kl_variance=np.asarray(gen.eigenvalues, dtype=np.float64),
    )
    np.savez(path, **out)
    return out
