"""Shadow-mode diagnostic: evaluate a network on REAL closed-loop residuals.

The ideal WFS flies the loop; every frame the network also estimates the
residual from its own (noisy) frame. Per mode we fit estimate = k * truth + e
and report k (sensitivity/bias) and the error, against the generator-based
evaluation. k << 1 or k < 0 on some modes, or errors much worse than on
generated data, explains a loop that cannot be held.
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
from fpsim.nets import FPNet  # noqa: E402
from fpsim.sensor import FocalPlaneSensor, FPSensorConfig  # noqa: E402

dev = "cuda"
cfg = S.KeckConfig()
sysd = S.build(cfg)
pupil = torch.tensor(sysd["pupil"], device=dev)
dm = DM(torch.tensor(sysd["ifs"], device=dev), torch.tensor(sysd["m2c"], device=dev), cfg.n_pupil)
proj = ModalProjector(dm, pupil)
var = torch.tensor(sysd["kl_variance"], device=dev, dtype=torch.float32)
scale = var.sqrt() * 500 / (2 * torch.pi)
sci = h_band_science(pupil, cfg.grid_m)
OUT = ROOT / "results" / "exp02"
report = {}
fig, axes = P.plt.subplots(1, 2, figsize=(10, 3.8))
for i, dfc in enumerate((0.0, 1.0)):
    sensor = FocalPlaneSensor(FPSensorConfig(defocus_rad=dfc), pupil, cfg.grid_m).to(dev)
    net = FPNet(1, dm.surfaces.shape[0]).to(dev)
    net.load_state_dict(torch.load(OUT / f"defocus_{dfc}.pt"))
    net.eval()
    est, tru = [], []

    def cb(k, residual, **kw):
        if k < 300:
            return
        with torch.no_grad():
            est.append(net(sensor.preprocess(sensor.frame(residual[None]))) * scale)
        tru.append(proj(residual) * 1e9)

    turb = Turbulence(cfg, batch=4, seed=100, seeing=0.6)
    run_loop(turb, dm, lambda r, k, h: proj(r.mean(0)), sci, pupil, 1000, 1300, gain=0.4, delay=2,
             settle=300, callback=cb)
    e = torch.cat(est)
    t = torch.cat(tru)
    k_mode = (e * t).sum(0) / (t * t).sum(0)
    err = (e - t).pow(2).sum(-1).mean().sqrt().item()
    true = t.pow(2).sum(-1).mean().sqrt().item()
    report[f"defocus {dfc}"] = dict(k=k_mode.tolist(), err_nm=err, true_nm=true)
    print(f"defocus {dfc}: closed-loop residual {true:.1f} nm, NN error {err:.1f} nm (rel {err / true:.2f}); "
          f"k median {k_mode.median():.2f}, modes with k<0.2: {(k_mode < 0.2).sum().item()}, k<0: {(k_mode < 0).sum().item()}; "
          f"k[0:6] {[round(float(v), 2) for v in k_mode[:6]]}", flush=True)
    ax = axes[i]
    ax.plot(k_mode.cpu(), lw=1, color=P.SERIES[i])
    ax.axhline(1, color=P.MUTED, lw=1, ls=":")
    ax.axhline(0, color=P.MUTED, lw=1)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("DM-KL mode index (low order -> high order)")
    ax.set_ylabel("sensitivity k (estimate / truth)")
    ax.set_title(f"{'in focus' if dfc == 0 else 'defocus 1 rad'}: rel. error {err / true:.2f}")
P.save(fig, "exp02_single_frame", "shadow_mode_sensitivity",
       """Shadow mode: the ideal sensor flies the loop (gain 0.4, 1 kHz) and the trained single-frame
network estimates every frame's residual on the side. Per DM-KL mode we fit estimate = k x truth.
k = 1 is a perfect sensor, k ~ 0 means the network does not see that mode (it outputs the prior
mean), k < 0 means it gets the sign wrong. **What to look at:** modes with k well below 1 or
negative are the ones an integrator cannot hold, which is why a hand-over from the ideal sensor
to the network collapses. Compare the relative error here (real closed-loop residuals) with the
generator-based numbers in relative_error_vs_level.""")
(OUT / "shadow.json").write_text(json.dumps(report))
