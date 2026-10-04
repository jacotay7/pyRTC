"""Acquisition: does a LARGE defocus (applied with the DM, no new hardware) let one
frame sense the low-order modes of a seeing-limited wavefront?

States from loops controlling N in {0, 2, 5, 10, 20} modes (open-loop transients
included). For each defocus, train a CNN to estimate the first 20 modes and report
the error/residual ratio per stage. Ratio well below 1 at N = 0 means the loop can
be closed from seeing-limited conditions.
"""

import argparse
import json
import pathlib
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fpsim import plotting as P  # noqa: E402
from fpsim import system as S  # noqa: E402
from fpsim.atmos import Turbulence  # noqa: E402
from fpsim.collect import collect, log_uniform, noisy_frames  # noqa: E402
from fpsim.loop import DM  # noqa: E402
from fpsim.nets import FPNet  # noqa: E402
from fpsim.sensor import FocalPlaneSensor, FPSensorConfig  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--defocus", type=float, nargs="+", default=[1.0, 2.0, 4.0, 6.0])
p.add_argument("--n-out", type=int, default=20)
p.add_argument("--steps", type=int, default=6000)
p.add_argument("--photons", type=float, default=1e5)
args = p.parse_args()
dev = "cuda"
OUT = ROOT / "results" / "exp12"
OUT.mkdir(parents=True, exist_ok=True)
cfg = S.KeckConfig()
sysd = S.build(cfg)
pupil = torch.tensor(sysd["pupil"], device=dev)
dm = DM(torch.tensor(sysd["ifs"], device=dev), torch.tensor(sysd["m2c"], device=dev), cfg.n_pupil)
n_modes = dm.surfaces.shape[0]
STAGES = (0, 2, 5, 10, 20)
NO = args.n_out
rows = []
for dfc in args.defocus:
    scfg = FPSensorConfig(defocus_rad=dfc, photons=args.photons)
    sensor = FocalPlaneSensor(scfg, pupil, cfg.grid_m).to(dev)
    t0 = time.perf_counter()
    train, test = [], []
    for i, n in enumerate(STAGES):
        mask = torch.zeros(n_modes, device=dev)
        mask[:n] = 1
        for j, dest in ((0, train), (1, train), (2, test)):  # same seeds for every defocus: same states
            tr = collect(Turbulence(cfg, batch=16, seed=12000 + 50 * i + j, seeing=0.6), dm, pupil, sensor, 600,
                         gains=(0.2, 0.5), est_noise=(0.0, 0.6), seed=100 * i + j, keep_from=0, mode_mask=mask)
            dest.append((n, tr.frames.flatten(0, 1), tr.truth.flatten(0, 1)[:, :NO]))
    X = torch.cat([x for _, x, _ in train])
    Y = torch.cat([y for _, _, y in train])
    scale = Y.std(0).to(dev)
    net = FPNet(1, NO).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=args.steps, pct_start=0.1)
    for step in range(args.steps):
        idx = torch.randint(0, len(X), (128,))
        ph = log_uniform(128, 0.5, 2.0, dev)[:, None, None]
        x = sensor.preprocess(noisy_frames(X[idx].to(dev), scfg, args.photons, ph))
        y = Y[idx].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = net(x).float() * scale
        loss = ((pred - y).pow(2).sum(-1) / (y.pow(2).sum(-1) + 100.0)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    net.eval()
    for n, xt, yt in test:
        with torch.no_grad():
            errs, trues = [], []
            for k in range(0, len(xt), 256):
                x = sensor.preprocess(noisy_frames(xt[k:k + 256].to(dev), scfg, args.photons))
                y = yt[k:k + 256].to(dev)
                pred = net(x) * scale
                errs.append((pred - y).pow(2).sum(-1))
                trues.append(y.pow(2).sum(-1))
        e, t = torch.cat(errs).mean().sqrt().item(), torch.cat(trues).mean().sqrt().item()
        rows.append(dict(defocus=dfc, stage=n, true_nm=t, err_nm=e))
        print(f"defocus {dfc:3.1f} rad, loop controlling {n:2d} modes: first {NO} modes residual {t:6.0f} nm, "
              f"error {e:6.0f} nm, ratio {e / t:.2f} [{time.perf_counter() - t0:.0f} s]", flush=True)
(OUT / "acquisition.json").write_text(json.dumps(rows))

fig, ax = P.plt.subplots(figsize=(6.4, 4.0))
for i, dfc in enumerate(args.defocus):
    rr = [r for r in rows if r["defocus"] == dfc]
    ax.plot([r["stage"] for r in rr], [r["err_nm"] / r["true_nm"] for r in rr], marker="o", color=P.SERIES[i],
            label=f"defocus {dfc:g} rad rms")
ax.axhline(1, color=P.MUTED, lw=1, ls=":")
ax.set_xlabel("modes already controlled (0 = seeing-limited, open loop)")
ax.set_ylabel(f"error / residual, first {NO} modes")
ax.set_ylim(0, 1.2)
ax.set_title("Acquisition: more defocus, more capture range?")
ax.legend(fontsize=8)
P.save(fig, "exp12_acquisition", "ratio_vs_defocus",
       f"""For each defocus (on a real system: the fixed 1 rad plus a temporary offset applied with the DM, so no new
hardware), a CNN is trained to estimate the first {NO} DM-KL modes from one H-band frame
({args.photons:.0e} photons) on states from loops that control 0-20 modes, from open loop on (0.6").
y = rms error / rms residual of those {NO} modes on held-out states (below 1 the loop can contract).
**What to look at:** the left end (open loop). A curve well below 1 there means the loop can be closed
from seeing-limited conditions with the focal-plane camera alone; then the defocus can be stepped
down to 1 rad as the loop converges.""",
       title="exp12: acquisition with a large DM-applied defocus")
