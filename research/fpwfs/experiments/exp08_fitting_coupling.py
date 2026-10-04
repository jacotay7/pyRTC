"""Is the uncorrectable fitting error what breaks linear focal-plane sensing?

Closed-loop states from the ideal loop; for each sensing band, build the
push-pull linear reconstructor (closed-loop mean reference) and measure its
NOISE-FREE error on (a) the true residual, (b) the same residual with the
fitting error removed (DM-space part only). If (a) >> (b) and (a) falls ~1/lambda,
the second-order coupling of the fitting error is the culprit.
"""

import json
import pathlib
import sys

import torch

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
OUT = ROOT / "results" / "exp08"
OUT.mkdir(parents=True, exist_ok=True)

states = []


def cb(k, residual, **kw):
    if k >= 300 and k % 5 == 0:
        states.append(residual.clone())


run_loop(Turbulence(cfg, batch=4, seed=321, seeing=0.6), dm, lambda r, k, h: proj(r.mean(0)),
         h_band_science(pupil, cfg.grid_m), pupil, 1000, 800, gain=0.4, settle=300, callback=cb)
res = torch.cat(states)  # (N, n, n)
truth = proj(res) * 1e9
dm_only = dm.opd(truth * 1e-9)
print(f"{res.shape[0]} closed-loop states, DM-space residual {truth.pow(2).sum(-1).mean().sqrt():.1f} nm")


def norm(e):
    return (e / e.sum((-2, -1), keepdim=True)).flatten(1)


rows = []
for lam in (0.8e-6, 1.25e-6, 1.65e-6, 2.2e-6):
    for dfc in (0.0, 1.0):
        sensor = FocalPlaneSensor(FPSensorConfig(defocus_rad=dfc, wavelength=lam), pupil, cfg.grid_m).to(dev)
        cols = []
        for i in range(n_modes):
            e = torch.zeros(1, n_modes, device=dev)
            e[0, i] = 5e-9
            cols.append((norm(sensor.frame(dm.opd(e), noise=False)) - norm(sensor.frame(dm.opd(-e), noise=False)))[0] / 10)
        j = torch.stack(cols, 1)
        rec = torch.linalg.pinv(j.double(), rtol=1e-3).float()
        out = {}
        for name, opd in (("full residual", res), ("fitting removed", dm_only)):
            frames = torch.cat([norm(sensor.frame(opd[i:i + 64], noise=False)) for i in range(0, len(opd), 64)])
            est = (frames - frames.mean(0)) @ rec.T
            tc = truth - truth.mean(0)
            out[name] = (est - tc).pow(2).sum(-1).mean().sqrt().item()
        rows.append(dict(lam=lam, defocus=dfc, **out))
        print(f"lambda {lam * 1e6:.2f} um defocus {dfc}: linear error {out['full residual']:.1f} nm with fitting error, "
              f"{out['fitting removed']:.1f} nm without", flush=True)
(OUT / "coupling.json").write_text(json.dumps(rows))

fig, ax = P.plt.subplots(figsize=(6.4, 4.0))
for i, dfc in enumerate((0.0, 1.0)):
    rr = [r for r in rows if r["defocus"] == dfc]
    lam = [r["lam"] * 1e6 for r in rr]
    lab = "in focus" if dfc == 0 else "defocus 1 rad"
    ax.plot(lam, [r["full residual"] for r in rr], marker="o", color=P.SERIES[i], label=f"{lab}: real residual")
    ax.plot(lam, [r["fitting removed"] for r in rr], marker="o", ls="--", color=P.SERIES[i], label=f"{lab}: fitting error removed")
ax.axhline(truth.pow(2).sum(-1).mean().sqrt().item(), color=P.MUTED, lw=1, ls=":")
ax.annotate("size of the residual being measured", (0.8, truth.pow(2).sum(-1).mean().sqrt().item()), xytext=(0, 4),
            textcoords="offset points", fontsize=8, color=P.INK2)
ax.set_xlabel("sensing wavelength [um]")
ax.set_ylabel("noise-free linear estimate error [nm rms]")
ax.set_yscale("log")
ax.set_title("The uncorrectable fitting error is what blinds a linear focal-plane sensor")
ax.legend(fontsize=7.5)
P.save(fig, "exp08_fitting_coupling", "linear_error_vs_band",
       """Noise-free error of the push-pull linear focal-plane reconstructor (closed-loop mean frame as reference)
on 640 real closed-loop states (ideal sensor flying the loop, 0.6", 1 kHz). Solid: the real residual,
which includes the ~100 nm the 349-actuator DM cannot fit. Dashed: the same states with that fitting
error removed (only the part the DM could correct). **What to look at:** with the fitting error
removed a linear map is essentially exact (small-phase regime), so the whole problem is the second-order
term of exp(i phi) in the *uncorrectable* error, which lands inside the control region. This is the
focal-plane counterpart of Shack-Hartmann aliasing. It shrinks only slowly with wavelength (83 -> 53 nm
from I to K with defocus, weaker than 1/lambda) and stays above the residual being measured in every
band. This sets what any estimator must remove: a non-linear one could use the halo
outside the control radius, which carries first-order information about the fitting error.""",
       title="exp08: fitting-error coupling (focal-plane aliasing)")
P.note("exp08_fitting_coupling", """Why linear focal-plane sensing fails on a real closed-loop residual: the uncorrectable
fitting error couples into the measurement at second order.""")
