"""Bootstrap from seeing-limited conditions with stage-specific focal-plane networks.

exp12 showed a network dedicated to the first 20 modes estimates them from one
1 rad-defocus frame even in open loop (error/residual ~0.2); exp11's single
all-stage network failed. So:

  A (20 modes): trained on loops controlling 0-20 modes, from open loop;
  B (120 modes): trained on loops controlling 20-120 modes;
  M: the maintenance network (exp10 v3), for the converged loop.

Schedule (frames): A on 20 modes -> B on 60 -> B on 120 -> M on 120.
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
p.add_argument("--maintenance", default="results/exp10/h64_nc120_v3_r3.pt")
p.add_argument("--schedule", default="0:A:20,200:B:60,400:B:120,700:M:120")
p.add_argument("--gain", type=float, default=0.4)
p.add_argument("--leak", type=float, default=0.99)
p.add_argument("--steps", type=int, default=8000)
p.add_argument("--atm", type=int, default=16)
p.add_argument("--eval-steps", type=int, default=2000)
p.add_argument("--tag", default="bootstrap2")
args = p.parse_args()
torch.manual_seed(0)
dev = "cuda"
OUT = ROOT / "results" / "exp14"
OUT.mkdir(parents=True, exist_ok=True)
cfg = S.KeckConfig()
sysd = S.build(cfg)
pupil = torch.tensor(sysd["pupil"], device=dev)
dm = DM(torch.tensor(sysd["ifs"], device=dev), torch.tensor(sysd["m2c"], device=dev), cfg.n_pupil)
proj = ModalProjector(dm, pupil)
n_modes = dm.surfaces.shape[0]
scfg = FPSensorConfig(defocus_rad=1.0, photons=1e5)
sensor = FocalPlaneSensor(scfg, pupil, cfg.grid_m).to(dev)
_rng = torch.Generator().manual_seed(77)


def mask_for(n):
    m = torch.zeros(n_modes, device=dev)
    m[:n] = 1
    return m


def atmosphere(batch, seed):
    u = torch.rand(2, generator=_rng).tolist()
    return Turbulence(cfg, batch=batch, seed=seed, seeing=0.45 + 0.4 * u[0], wind_scale=0.7 + 0.8 * u[1])


def train_stage(stages, n_out, seed0, label):
    t0 = time.perf_counter()
    xs, ys = [], []
    for i, n in enumerate(stages):
        for j in range(3):
            tr = collect(atmosphere(args.atm, seed0 + 50 * i + j), dm, pupil, sensor, 700, gains=(0.2, 0.5),
                         est_noise=(0.0, 0.6), seed=seed0 + 10 * i + j, keep_from=0, mode_mask=mask_for(n))
            xs.append(tr.frames.flatten(0, 1))
            ys.append(tr.truth.flatten(0, 1)[:, :n_out])
    X, Y = torch.cat(xs), torch.cat(ys)
    scale = Y.std(0).to(dev)
    print(f"[{label}] {len(X)} states from loops controlling {stages} modes [{time.perf_counter() - t0:.0f} s]",
          flush=True)
    net = FPNet(1, n_out).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=args.steps, pct_start=0.1)
    for step in range(args.steps):
        idx = torch.randint(0, len(X), (128,))
        ph = log_uniform(128, 0.5, 2.0, dev)[:, None, None]
        x = sensor.preprocess(noisy_frames(X[idx].to(dev), scfg, 1e5, ph))
        y = Y[idx].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = net(x).float() * scale
        loss = ((pred - y).pow(2).sum(-1) / (y.pow(2).sum(-1) + 100.0)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 2000 == 0 or step == args.steps - 1:
            print(f"  [{label}] step {step:5d} loss {loss.item():.3f} [{time.perf_counter() - t0:.0f} s]", flush=True)
    net.eval()
    torch.save({**net.state_dict(), "_scale": scale.cpu()}, OUT / f"{args.tag}_{label}.pt")
    return net, scale


net_a, scale_a = train_stage((0, 2, 5, 10, 20), 20, 20000, "A")
net_b, scale_b = train_stage((20, 40, 60, 90, 120), 120, 21000, "B")
st = torch.load(ROOT / args.maintenance)
scale_m = st.pop("_scale").to(dev)
net_m = FPNet(1, 120).to(dev)
net_m.load_state_dict(st)
net_m.eval()
nets = {"A": (net_a, scale_a), "B": (net_b, scale_b), "M": (net_m, scale_m)}
schedule = [(int(f), name, int(n)) for f, name, n in (item.split(":") for item in args.schedule.split(","))]


def stage_at(k):
    cur = schedule[0]
    for s in schedule:
        if k >= s[0]:
            cur = s
    return cur


def recon(residual, k, hist):
    _, name, n = stage_at(k)
    net, scale = nets[name]
    with torch.no_grad():
        est = net(sensor.preprocess(sensor.frame(residual))) * scale
    est = torch.nn.functional.pad(est, (0, n_modes - est.shape[-1]))
    return est * mask_for(n) * 1e-9


report = {"args": vars(args)}
fig, ax = P.plt.subplots(figsize=(7.2, 4.0))
held_total = 0
for i, seed in enumerate((6000, 6100)):
    r = run_loop(Turbulence(cfg, batch=12, seed=seed, seeing=0.6), dm, recon, h_band_science(pupil, cfg.grid_m),
                 pupil, 1000, args.eval_steps, gain=args.gain, leak=args.leak, delay=2, settle=args.eval_steps - 800)
    held = int((r.strehl_le > 0.5).sum())
    held_total += held
    report[f"seed{seed}"] = dict(strehl_le=r.strehl_le.tolist(), se=r.strehl_se.mean(1).tolist(), held=held)
    print(f"BOOTSTRAP seeds {seed}+: converged {held}/12, final LE Strehl median {r.strehl_le.median():.3f}; "
          f"SE (mean) every 100 {[round(float(v), 2) for v in r.strehl_se.mean(1)[::100]]}", flush=True)
    for b in range(12):
        ax.plot(r.strehl_se[:, b], lw=0.7, color=P.SERIES[0] if r.strehl_le[b] > 0.5 else P.SERIES[7], alpha=0.8)
for f, name, n in schedule:
    ax.axvline(f, color=P.MUTED, lw=0.8, ls=":")
    ax.annotate(f"{name}: {n}", (f, 0.95), xytext=(3, 0), textcoords="offset points", fontsize=7.5, color=P.INK2)
ax.set_xlabel("frame (1 kHz) from open loop")
ax.set_ylabel("short-exposure H Strehl")
ax.set_ylim(0, 1)
ax.set_title(f"Bootstrap with stage-specific focal-plane networks: {held_total}/24 converge")
(OUT / f"{args.tag}.json").write_text(json.dumps(report))
P.save(fig, "exp14_bootstrap", args.tag,
       f"""Closing the Keck loop from seeing-limited conditions (0.6", 1 kHz, 2-frame delay, gain {args.gain}, leak
{args.leak}) with only the focal-plane camera (H band, 1 rad defocus, 1e5 photons). Three networks, switched
on the schedule marked by dotted lines (network: controlled modes): A estimates the first 20 modes and is
trained on loops controlling 0-20 modes; B estimates 120 modes and is trained on loops controlling 20-120
modes; M is the exp10 v3 maintenance network. 24 independent atmospheres; blue = converged (final LE
Strehl > 0.5), red = not. **What to look at:** the fraction that converges (goal G3: >= 95 %) and the
time to reach the ~0.72 steady state.""",
       title="exp14: bootstrap with stage-specific networks")
