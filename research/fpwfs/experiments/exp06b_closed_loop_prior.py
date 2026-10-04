"""Per-mode information bound under the CLOSED-LOOP residual prior.

1. Fly the ideal-WFS loop and record the per-mode residual std (the real prior).
2. Fisher information at closed-loop operating points (actual residual OPDs),
   per-mode posterior std for several photon levels and defocus values.
"""

import json
import pathlib
import sys

import torch
from torch.func import jacfwd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fpsim import plotting as P  # noqa: E402
from fpsim import system as S  # noqa: E402
from fpsim.atmos import Turbulence  # noqa: E402
from fpsim.loop import DM, ModalProjector, h_band_science, run_loop  # noqa: E402
from fpsim.sensor import FocalPlaneSensor, FPSensorConfig  # noqa: E402

dev = "cuda"
cfg = S.KeckConfig()
sysd = S.build(cfg)
pupil = torch.tensor(sysd["pupil"], device=dev)
dm = DM(torch.tensor(sysd["ifs"], device=dev), torch.tensor(sysd["m2c"], device=dev), cfg.n_pupil)
proj = ModalProjector(dm, pupil)
n_modes = dm.surfaces.shape[0]
OUT = ROOT / "results" / "exp06"
modes, states = [], []


def cb(k, residual, **kw):
    if k >= 300:
        modes.append(proj(residual) * 1e9)
        if k % 100 == 0:
            states.append(residual[0].clone())


turb = Turbulence(cfg, batch=4, seed=7, seeing=0.6)
run_loop(turb, dm, lambda r, k, h: proj(r.mean(0)), h_band_science(pupil, cfg.grid_m), pupil, 1000, 1300,
         gain=0.4, delay=2, settle=300, callback=cb)
m = torch.cat(modes)
prior_sd = m.std(0)  # nm, per mode, real closed-loop residual
print(f"closed-loop DM-space residual {m.pow(2).sum(-1).mean().sqrt():.1f} nm; per-mode std first/last "
      f"{prior_sd[:3].tolist()} / {prior_sd[-3:].tolist()}")
res = dict(prior_sd=prior_sd.tolist(), post={})
for dfc in (0.0, 1.0):
    sensor = FocalPlaneSensor(FPSensorConfig(defocus_rad=dfc), pupil, cfg.grid_m).to(dev)
    jacs, imgs = [], []
    for st in states[:6]:
        def image(a, st=st):
            return sensor.imager(st + dm.opd(a * 1e-9)).reshape(-1)
        a0 = torch.zeros(n_modes, device=dev)
        jacs.append(jacfwd(image)(a0))
        imgs.append(image(a0))
    for photons in (1e4, 1e5, 1e6):
        posts = []
        for j, img in zip(jacs, imgs):
            w = photons**2 / (photons * img.clamp_min(0) + 0.36)
            f = (j.T * w) @ j
            c = torch.linalg.inv(f.double() + torch.diag(1 / prior_sd.double() ** 2))
            posts.append(torch.diagonal(c).sqrt())
        post = torch.stack(posts).mean(0)
        res["post"][f"{dfc}_{photons:g}"] = post.tolist()
        tot = post.pow(2).sum().sqrt().item()
        print(f"defocus {dfc} photons {photons:.0e}: bound {tot:.1f} nm of {prior_sd.norm():.1f} nm; "
              f"modes with posterior > 0.7 prior: {(post.float() > 0.7 * prior_sd).sum().item()}", flush=True)
(OUT / "closed_loop_prior.json").write_text(json.dumps(res))

fig, ax = P.plt.subplots(figsize=(7.2, 4.2))
x = range(n_modes)
ax.plot(x, prior_sd.cpu(), color=P.INK2, lw=2, label="closed-loop residual (ideal WFS, 1 kHz)")
for i, (key, lab) in enumerate((("1.0_10000", "bound, defocus 1 rad, 1e4 ph"), ("1.0_1e+06", "bound, defocus 1 rad, 1e6 ph"),
                                 ("0.0_10000", "bound, in focus, 1e4 ph"))):
    ax.plot(x, res["post"][key], color=P.SERIES[i], lw=1.4, label=lab)
ax.set_yscale("log")
ax.set_xlabel("DM-KL mode index")
ax.set_ylabel("per-mode rms [nm]")
ax.set_title("Which modes one frame can sense in a closed Keck loop")
ax.legend(fontsize=7.5)
P.save(fig, "exp06_information", "per_mode_bound_closed_loop",
       """Grey: per-mode rms of the real closed-loop residual (ideal sensor flying the loop at 1 kHz, gain 0.4,
0.6" seeing): this is what the WFS has to measure each frame, and it is far flatter than the open-loop
KL spectrum. Coloured: per-mode information bound (posterior std) from one H-band frame at that
closed-loop operating point, with the grey curve as prior. **What to look at:** where a coloured
curve sits well below grey, that mode is measurable from one frame; where it touches grey, the frame
carries no useful information about it at that photon level and the estimator should leave it alone
(or use more frames). In focus the even modes stay pinned to the prior (sign ambiguity).""")
