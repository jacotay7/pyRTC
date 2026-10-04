"""Linear (ridge) reconstructor probe: how much does one frame tell a linear estimator?

R2-style baseline: regress modal residual on pixel intensities around the
reference, per correction level, in focus vs defocus. If a linear map already
beats the CNN at high Strehl, the CNN is under-trained, not information-starved.
"""

import json
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fpsim import system as S  # noqa: E402
from fpsim.data import ResidualGenerator  # noqa: E402
from fpsim.loop import DM  # noqa: E402
from fpsim.sensor import FocalPlaneSensor, FPSensorConfig  # noqa: E402

dev = "cuda"
cfg = S.KeckConfig()
sysd = S.build(cfg)
pupil = torch.tensor(sysd["pupil"], device=dev)
dm = DM(torch.tensor(sysd["ifs"], device=dev), torch.tensor(sysd["m2c"], device=dev), cfg.n_pupil)
gen = ResidualGenerator(cfg, sysd, dm, pupil, bank_size=2048)
OUT = ROOT / "results" / "exp05"
OUT.mkdir(parents=True, exist_ok=True)
N_TRAIN, N_TEST = 24000, 2048
rows = []
for photons in (1e4, 1e6):
    for dfc in (0.0, 0.5, 1.0, 2.0):
        sensor = FocalPlaneSensor(FPSensorConfig(defocus_rad=dfc, photons=photons), pupil, cfg.grid_m).to(dev)
        for lo, hi in ((0.02, 0.05), (0.05, 0.15), (0.15, 0.4)):
            def frames(n):
                opd, y = gen.sample(n, alpha=(lo, hi))
                e = sensor.frame(opd)
                return (e / e.sum((-2, -1), keepdim=True)).flatten(1), y

            # accumulate normal equations in chunks (never hold X); centre with a pilot mean
            mu = frames(2048)[0].mean(0)
            npx = mu.numel()
            xtx = torch.zeros(npx, npx, device=dev, dtype=torch.float64)
            xty = torch.zeros(npx, gen.sigma_nm.numel(), device=dev, dtype=torch.float64)
            ysum = torch.zeros(gen.sigma_nm.numel(), device=dev, dtype=torch.float64)
            for _ in range(N_TRAIN // 1024):
                x, y = frames(1024)
                x = (x - mu).double()
                xtx += x.T @ x
                xty += x.T @ y.double()
                ysum += y.double().sum(0)
            xt, yt = frames(N_TEST)
            xt = xt - mu
            best = None
            tr = torch.trace(xtx) / xtx.shape[0]
            for lam in (1e-4, 1e-3, 1e-2, 1e-1):
                w = torch.linalg.solve(xtx + lam * tr * torch.eye(xtx.shape[0], device=dev, dtype=torch.float64), xty).float()
                err = ((xt @ w - yt) ** 2).sum(-1).mean().sqrt().item()
                if best is None or err < best[0]:
                    best = (err, lam)
            true = (yt**2).sum(-1).mean().sqrt().item()
            rows.append(dict(photons=photons, defocus=dfc, alpha=[lo, hi], true_nm=true, err_nm=best[0], lam=best[1]))
            print(f"ph={photons:.0e} defocus={dfc:.1f} alpha {lo:.2f}-{hi:.2f}: true {true:6.1f} nm, linear err {best[0]:6.1f} nm, rel {best[0] / true:.3f}", flush=True)
(OUT / "linear_probe.json").write_text(json.dumps(rows, indent=1))
