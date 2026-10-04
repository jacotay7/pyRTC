"""Information content of one focal-plane frame: Fisher matrix and Bayesian CRB.

For an operating point (residual OPD = fitting error + small DM-space part),
J = d(image)/d(modal coefficients) by forward-mode autodiff through the imager.
Poisson + read noise: F = N^2 J^T diag(1 / (N I + r^2)) J  (I in flux fractions).
Bayesian CRB with the residual prior: C = (F + Sigma^-1)^-1; sqrt(tr C) is the
best achievable rms error (nm) of ANY unbiased-ish estimator at that point.

Also reports the linear-response spread across operating points: if J changes
a lot between fitting-error realisations, one global linear map cannot reach
the CRB even though the information is there.
"""

import json
import pathlib
import sys

import torch
from torch.func import jacfwd

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
gen = ResidualGenerator(cfg, sysd, dm, pupil, bank_size=64)
OUT = ROOT / "results" / "exp06"
OUT.mkdir(parents=True, exist_ok=True)
n_modes = dm.surfaces.shape[0]
ALPHA = 0.03  # well-corrected loop: DM-space residual ~ 0.03 x open loop (~37 nm)
prior_sd = ALPHA * gen.sigma_nm  # nm, per mode
rows = []
for dfc in (0.0, 0.25, 0.5, 1.0, 2.0):
    sensor = FocalPlaneSensor(FPSensorConfig(defocus_rad=dfc), pupil, cfg.grid_m).to(dev)
    for fit_on in (False, True):
        jacs, imgs = [], []
        for i in range(4):
            fit = gen.fit_bank[i].float() if fit_on else torch.zeros_like(pupil)
            base = fit.clone()

            def image(a):
                return sensor.imager(base + dm.opd(a * 1e-9)).reshape(-1)

            a0 = torch.zeros(n_modes, device=dev)
            jacs.append(jacfwd(image)(a0))  # (npix^2, n_modes), per nm
            imgs.append(image(a0))
        # response variability across operating points
        j0 = jacs[0]
        spread = torch.stack([(j - j0).norm() / j0.norm() for j in jacs[1:]]).mean().item() if fit_on else 0.0
        for photons in (1e3, 1e4, 1e5, 1e6):
            crbs, mode_sd = [], []
            for j, img in zip(jacs, imgs):
                w = photons**2 / (photons * img.clamp_min(0) + sensor.cfg.read_noise**2)
                fisher = (j.T * w) @ j
                post = torch.linalg.inv(fisher.double() + torch.diag(1 / prior_sd.double() ** 2))
                crbs.append(torch.trace(post).sqrt().item())
                mode_sd.append(torch.diagonal(post).sqrt())
            prior = prior_sd.norm().item()
            row = dict(defocus=dfc, fitting=fit_on, photons=photons, prior_nm=prior,
                       crb_nm=sum(crbs) / len(crbs), response_spread=spread,
                       mode_sd_nm=torch.stack(mode_sd).mean(0).tolist())
            rows.append(row)
            print(f"defocus {dfc:4.2f} fitting {fit_on!s:5} photons {photons:7.0e}: prior {prior:5.1f} nm -> "
                  f"CRB {row['crb_nm']:5.1f} nm (rel {row['crb_nm'] / prior:.3f})"
                  + (f"   J spread {spread:.2f}" if fit_on and photons == 1e3 else ""), flush=True)
(OUT / "fisher.json").write_text(json.dumps(rows))
