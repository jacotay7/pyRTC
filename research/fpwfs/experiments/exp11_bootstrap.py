"""Bootstrap the Keck loop from open loop with only the focal-plane camera (goal G3).

Staged closing, as operators do: control a few modes first, then more as the
residual shrinks. One network learns every stage: it is trained on states from
loops controlling N in STAGES modes (including the open-loop transient), with
each sample penalised only on its controlled modes plus headroom. At run time
the controlled-mode count follows a schedule.
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
from fpsim.loop import DM, ModalProjector, h_band_science, run_loop  # noqa: E402
from fpsim.nets import FPNet  # noqa: E402
from fpsim.sensor import FocalPlaneSensor, FPSensorConfig  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--photons", type=float, default=1e5)
p.add_argument("--n-control", type=int, default=120)
p.add_argument("--atm", type=int, default=16)
p.add_argument("--steps", type=int, default=16000)
p.add_argument("--batch", type=int, default=128)
p.add_argument("--gain", type=float, default=0.3)
p.add_argument("--schedule", default="0:5,40:20,120:60,250:120", help="frame:modes,... controlled-mode schedule")
p.add_argument("--init", default=None, help="warm-start weights (e.g. exp10 n-control network)")
p.add_argument("--tag", default="bootstrap")
args = p.parse_args()
torch.manual_seed(0)
dev = "cuda"
OUT = ROOT / "results" / "exp11"
OUT.mkdir(parents=True, exist_ok=True)
cfg = S.KeckConfig()
sysd = S.build(cfg)
pupil = torch.tensor(sysd["pupil"], device=dev)
dm = DM(torch.tensor(sysd["ifs"], device=dev), torch.tensor(sysd["m2c"], device=dev), cfg.n_pupil)
proj = ModalProjector(dm, pupil)
n_modes = dm.surfaces.shape[0]
NC = args.n_control
scfg = FPSensorConfig(defocus_rad=1.0, photons=args.photons)
sensor = FocalPlaneSensor(scfg, pupil, cfg.grid_m).to(dev)
STAGES = (2, 5, 10, 20, 40, 60, 90, 120)
schedule = [tuple(int(v) for v in item.split(":")) for item in args.schedule.split(",")]


def mask_for(n):
    m = torch.zeros(n_modes, device=dev)
    m[:n] = 1
    return m


# ---- data: every stage, from open loop (transient kept) --------------------------------
t0 = time.perf_counter()
xs, ys, ws = [], [], []
for i, n in enumerate(STAGES):
    for j in range(2):
        tr = collect(Turbulence(cfg, batch=args.atm, seed=9000 + 50 * i + j, seeing=0.6), dm, pupil, sensor,
                     900, gains=(0.2, 0.5), est_noise=(0.0, 0.6), seed=100 * i + j, keep_from=0,
                     mode_mask=mask_for(n))
        xs.append(tr.frames.flatten(0, 1))
        ys.append(tr.truth.flatten(0, 1)[:, :NC])
        # loss weight: controlled modes plus 50 % headroom (the next stage)
        w = torch.zeros(NC)
        w[: min(NC, max(int(1.5 * n), n + 5))] = 1
        ws.append(w.expand(len(xs[-1]), NC))
X, Y, W = torch.cat(xs), torch.cat(ys), torch.cat(ws)
scale = Y.std(0).to(dev)
print(f"{len(X)} staged states (open-loop transients included) [{time.perf_counter() - t0:.0f} s]", flush=True)

net = FPNet(1, NC).to(dev)
if args.init:
    st = torch.load(args.init)
    st.pop("_scale", None)
    net.load_state_dict(st)
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=args.steps, pct_start=0.1)
t0 = time.perf_counter()
for step in range(args.steps):
    idx = torch.randint(0, len(X), (args.batch,))
    clean = X[idx].to(dev)
    ph = log_uniform(len(idx), 0.5, 2.0, dev)[:, None, None]
    x = sensor.preprocess(noisy_frames(clean, scfg, args.photons, ph))
    y, w = Y[idx].to(dev), W[idx].to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = net(x).float() * scale
    # relative error per sample, on its weighted modes (scale-free across stages)
    loss = (((pred - y) ** 2 * w).sum(-1) / ((y**2 * w).sum(-1) + 100.0)).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    sched.step()
    if step % 2000 == 0 or step == args.steps - 1:
        print(f"  step {step:5d} loss {loss.item():.3f} [{time.perf_counter() - t0:.0f} s]", flush=True)
net.eval()
torch.save({**net.state_dict(), "_scale": scale.cpu()}, OUT / f"{args.tag}.pt")


def n_at(k):
    n = schedule[0][1]
    for f, m in schedule:
        if k >= f:
            n = m
    return n


def recon(residual, k, hist):
    with torch.no_grad():
        est = net(sensor.preprocess(sensor.frame(residual))) * scale
    est = torch.nn.functional.pad(est, (0, n_modes - NC))
    return est * mask_for(n_at(k)) * 1e-9


report = {"args": vars(args)}
fig, ax = P.plt.subplots(figsize=(7.0, 4.0))
for i, seed in enumerate((100, 200, 300)):
    r = run_loop(Turbulence(cfg, batch=4, seed=seed, seeing=0.6), dm, recon, h_band_science(pupil, cfg.grid_m),
                 pupil, 1000, 1500, gain=args.gain, delay=2, settle=600)
    traj = r.strehl_se.mean(1)
    report[f"seed{seed}"] = dict(se_traj=traj.tolist(), strehl_le=r.strehl_le.tolist(),
                                 residual_nm=float(r.residual_nm[600:].mean()))
    print(f"BOOTSTRAP seeds {seed}+: LE Strehl {r.strehl_le.mean():.3f} (+-{r.strehl_le.std():.3f}), per-atmosphere "
          f"{[round(float(v), 2) for v in r.strehl_le]}; SE every 50 {[round(float(v), 2) for v in traj[:500:50]]}",
          flush=True)
    for b in range(4):
        ax.plot(r.strehl_se[:, b], lw=0.9, color=P.SERIES[i], alpha=0.8, label=f"atmospheres {seed}+" if b == 0 else None)
for f, m in schedule:
    ax.axvline(f, color=P.MUTED, lw=0.8, ls=":")
    ax.annotate(f"{m} modes", (f, 0.95), xytext=(3, 0), textcoords="offset points", fontsize=7.5, color=P.INK2)
ax.set_xlabel("frame (1 kHz) from open loop")
ax.set_ylabel("short-exposure H Strehl")
ax.set_ylim(0, 1)
ax.set_title("Closing the Keck loop from seeing-limited with only the focal-plane camera")
ax.legend(fontsize=7.5, loc="lower right")
(OUT / f"{args.tag}.json").write_text(json.dumps(report))
P.save(fig, "exp11_bootstrap", args.tag,
       f"""Bootstrap from open loop (0.6" seeing, 1 kHz, 2-frame delay, gain {args.gain}) with one
single-frame focal-plane network (H band, 1 rad defocus, {args.photons:.0e} photons) as the only sensor.
The number of controlled DM-KL modes follows the schedule marked by dotted lines ({args.schedule}).
Each thin line is one of 12 independent atmospheres. **What to look at:** whether every atmosphere
converges (goal G3 asks for >= 95 % of seeds) and how quickly it reaches the steady state of the
hand-over test (~0.71 with 120 modes).""",
       title="exp11: bootstrap from open loop")
