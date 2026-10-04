"""pyturb atmospheres as torch tensors (zero-copy via DLPack)."""

from __future__ import annotations

import warnings

import torch

from .system import KeckConfig


def make_atmosphere(
    cfg: KeckConfig,
    seeing: float = 0.6,
    profile: str = "keck",
    seed: int | None = None,
    engine: str = "extrude",
    wind_scale: float = 1.0,
):
    import pyturb

    kwargs = {}
    if wind_scale != 1.0:
        kwargs["wind"] = [layer.wind_speed * wind_scale for layer in pyturb.get_profile(profile)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pyturb.Atmosphere.from_profile(
            profile,
            seeing=seeing,
            diameter=cfg.grid_m,
            n=cfg.n_pupil,
            device="gpu",
            engine=engine,
            seed=seed,
            **kwargs,
        )


def to_torch(array) -> torch.Tensor:
    return torch.from_dlpack(array).float()


class Turbulence:
    """One or more independent frozen-flow atmospheres stepped together."""

    # Each ``seed`` owns a disjoint block of SEED_STRIDE atmospheres, so callers can use
    # consecutive seeds for separate collections without sharing turbulence. (Before
    # 2026-10-03 atmosphere i used seed + i: collections whose seeds differed by less
    # than the batch size silently shared most of their atmospheres.)
    SEED_STRIDE = 1000

    def __init__(self, cfg: KeckConfig, batch: int = 1, seed: int = 0, **kwargs):
        if batch > self.SEED_STRIDE:
            raise ValueError(f"batch must be <= {self.SEED_STRIDE}")
        self.seeds = [seed * self.SEED_STRIDE + i for i in range(batch)]
        self.atms = [make_atmosphere(cfg, seed=s, **kwargs) for s in self.seeds]

    def step(self, dt: float) -> torch.Tensor:
        """Advance by ``dt`` seconds; OPD (batch, n, n) in metres."""
        return torch.stack([to_torch(a.evolve(dt)) for a in self.atms])

    def now(self) -> torch.Tensor:
        return torch.stack([to_torch(a.opd(a.time)) for a in self.atms])
