"""Bootstrap with a defocus ramp (no added hardware).

Acquire with a large DM-applied defocus (curvature regime, widest capture, exp12),
then step the defocus down while raising the number of controlled modes; finish
with the maintenance network at the fixed 1 rad defocus. One network per stage,
trained on the states that stage visits (noisy-ideal schedule trajectories, then
DAgger with the networks flying the whole schedule).

Stage table (defocus rad, controlled modes, frames): see STAGES.
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
from fpsim.collect import log_uniform  # noqa: E402
from fpsim.loop import DM, ModalProjector, h_band_science  # noqa: E402
from fpsim.nets import FPNet  # noqa: E402
from fpsim.optics import add_noise  # noqa: E402
from fpsim.sensor import FocalPlaneSensor, FPSensorConfig  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--maintenance", default="results/exp10/slim24_nc120_r3.pt")
p.add_argument("--collections", type=int, default=10)
p.add_argument("--atm", type=int, default=16)
p.add_argument("--steps", type=int, default=5000)
p.add_argument("--rounds", type=int, default=2)
p.add_argument("--est-noise", type=float, nargs=2, default=[0.3, 0.8])
p.add_argument("--gain", type=float, default=0.3)
p.add_argument("--tag", default="ramp")
args = p.parse_args()
torch.manual_seed(0)
dev = "cuda"
OUT = ROOT / "results" / "exp17"
OUT.mkdir(parents=True, exist_ok=True)
cfg = S.KeckConfig()
sysd = S.build(cfg)
pupil = torch.tensor(sysd["pupil"], device=dev)
dm = DM(torch.tensor(sysd["ifs"], device=dev), torch.tensor(sysd["m2c"], device=dev), cfg.n_pupil)
proj = ModalProjector(dm, pupil)
n_modes = dm.surfaces.shape[0]
PH = 1e5
# (defocus rad, controlled modes, frames); last stage = maintenance network at 1 rad
STAGES = [(6.0, 20, 150), (4.5, 40, 150), (3.0, 60, 150), (2.0, 90, 150), (1.5, 120, 150), (1.0, 120, 450)]
sensors = [FocalPlaneSensor(FPSensorConfig(defocus_rad=d, photons=PH), pupil, cfg.grid_m).to(dev) for d, _, _ in STAGES]
T = sum(f for _, _, f in STAGES)
starts = [sum(f for _, _, f in STAGES[:i]) for i in range(len(STAGES))]
_rng = torch.Generator().manual_seed(17)


def stage_of(k):
    for i in reversed(range(len(STAGES))):
        if k >= starts[i]:
            return i
    return 0


def mask_for(n):
    m = torch.zeros(n_modes, device=dev)
    m[:n] = 1
    return m


def atmosphere(batch, seed, diverse=True):
    if not diverse:
        return Turbulence(cfg, batch=batch, seed=seed, seeing=0.6)
    u = torch.rand(2, generator=_rng).tolist()
    return Turbulence(cfg, batch=batch, seed=seed, seeing=0.45 + 0.4 * u[0], wind_scale=0.7 + 0.8 * u[1])


def run_schedule(turb, policy=None, est_noise=(0.0, 0.0), gain=None, record=True, seed=0):
    """Fly the whole schedule; returns per-stage (clean frames, labels) and SE Strehl."""
    g = torch.Generator(device=dev).manual_seed(seed)
    b = len(turb.atms)
    gain = args.gain if gain is None else gain
    noise = est_noise[0] + (est_noise[1] - est_noise[0]) * torch.rand(b, 1, device=dev, generator=g)
    cmd = torch.zeros(b, n_modes, device=dev)
    pending = [torch.zeros(b, n_modes, device=dev) for _ in range(2)]
    frames = [[] for _ in STAGES]
    labels = [[] for _ in STAGES]
    sci = h_band_science(pupil, cfg.grid_m)
    ref = sci(torch.zeros_like(pupil)).amax()
    se = []
    for k in range(T):
        s = stage_of(k)
        _, n, _ = STAGES[s]
        applied = pending.pop(0)
        residual = turb.step(1e-3) - dm.opd(applied)
        true = proj(residual)
        clean = sensors[s].frame(residual, noise=False)
        if record:
            frames[s].append(clean.half().cpu())
            labels[s].append((true * 1e9).cpu())
        se.append((sci(residual).amax(dim=(-2, -1)) / ref).cpu())
        if policy is None:
            rms = true.pow(2).mean(-1, keepdim=True).sqrt()
            est = true + noise * rms * torch.randn(b, n_modes, device=dev, generator=g)
        else:
            est = policy(s, add_noise(clean / PH, PH, 0.6))
        cmd = 0.99 * cmd + gain * est * mask_for(n)
        pending.append(cmd.clone())
    out = [(torch.cat(f) if f else None, torch.cat(lab) if lab else None) for f, lab in zip(frames, labels)]
    return out, torch.stack(se)  # se: (T, B)


# ---- data: noisy-ideal schedule trajectories -------------------------------------------
t0 = time.perf_counter()
data = [[[], []] for _ in STAGES]
for i in range(args.collections):
    per_stage, _ = run_schedule(atmosphere(args.atm, 700 + i), est_noise=tuple(args.est_noise), seed=i)
    for s, (x, y) in enumerate(per_stage):
        data[s][0].append(x)
        data[s][1].append(y)
test, _ = run_schedule(atmosphere(args.atm, 799, diverse=False), est_noise=tuple(args.est_noise), seed=99)
print(f"collected {args.collections} x {args.atm} schedule trajectories [{time.perf_counter() - t0:.0f} s]", flush=True)

nets, scales = [], []
for s, (d, n, _) in enumerate(STAGES[:-1]):
    X, Y = torch.cat(data[s][0]), torch.cat(data[s][1])[:, :n]
    scale = Y.std(0).to(dev)
    net = FPNet(1, n, width=24, stem_stride=2).to(dev)
    nets.append(net)
    scales.append(scale)


def train_stage(s, steps, lr):
    d, n, _ = STAGES[s]
    X, Y = torch.cat(data[s][0]), torch.cat(data[s][1])[:, :n]
    net, scale = nets[s], scales[s]
    sensor = sensors[s]
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    net.train()
    for _ in range(steps):
        idx = torch.randint(0, len(X), (128,))
        ph = log_uniform(128, 0.5, 2.0, dev)[:, None, None]
        e = add_noise(X[idx].to(dev).float() / PH, PH * ph, 0.6)
        y = Y[idx].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = net(sensor.preprocess(e)).float() * scale
        loss = ((pred - y).pow(2).sum(-1) / (y.pow(2).sum(-1) + 100.0)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
    net.eval()
    # held-out ratio on this stage's test states
    xt, yt = test[s]
    with torch.no_grad():
        idx = torch.arange(0, len(xt), max(1, len(xt) // 1024))
        e = add_noise(xt[idx].to(dev).float() / PH, PH, 0.6)
        pred = net(sensor.preprocess(e)) * scale
        y = yt[idx, :n].to(dev)
        ratio = ((pred - y).pow(2).sum(-1).mean() / y.pow(2).sum(-1).mean()).sqrt().item()
    return ratio


st = torch.load(ROOT / args.maintenance)
scale_m = st.pop("_scale").to(dev)
net_m = FPNet(1, 120, width=24, stem_stride=2).to(dev)
net_m.load_state_dict(st)
net_m.eval()


def policy(s, e):
    with torch.no_grad():
        if s == len(STAGES) - 1:
            est = net_m(sensors[s].preprocess(e)) * scale_m
        else:
            est = nets[s](sensors[s].preprocess(e)) * scales[s]
    return torch.nn.functional.pad(est, (0, n_modes - est.shape[-1])) * 1e-9


def evaluate(label):
    held, ses = 0, []
    for seed in (950, 951):
        _, se = run_schedule(Turbulence(cfg, batch=12, seed=seed, seeing=0.6), policy=policy, record=False, seed=seed)
        final = se[-300:].mean(0)
        held += int((final > 0.5).sum())
        ses.append(se)
    se = torch.cat(ses, 1)
    at_stage_end = [round(float(se[starts[i] + STAGES[i][2] - 1].median()), 2) for i in range(len(STAGES))]
    print(f"[{label}] BOOTSTRAP converged {held}/24; median SE Strehl at end of each stage {at_stage_end}", flush=True)
    return held, se


report = {"args": vars(args), "stages": STAGES, "rounds": []}
for r in range(args.rounds + 1):
    ratios = [train_stage(s, args.steps if r == 0 else args.steps // 2, 1e-3 if r == 0 else 3e-4)
              for s in range(len(STAGES) - 1)]
    print(f"round {r}: held-out error/residual per stage {[round(x, 2) for x in ratios]}", flush=True)
    held, se = evaluate(f"round {r}")
    report["rounds"].append(dict(ratios=ratios, held=held))
    if r < args.rounds:  # DAgger: the networks fly the schedule; relabel what they visit
        for j in range(3):
            per_stage, _ = run_schedule(atmosphere(args.atm, 800 + 10 * r + j), policy=policy, seed=300 + j)
            for s, (x, y) in enumerate(per_stage):
                ok = y.pow(2).sum(-1).sqrt() < 2500.0  # drop runaway states
                data[s][0].append(x[ok])
                data[s][1].append(y[ok])
(OUT / f"{args.tag}.json").write_text(json.dumps(report))
torch.save({f"stage{s}": {**nets[s].state_dict(), "_scale": scales[s].cpu()} for s in range(len(nets))},
           OUT / f"{args.tag}.pt")

fig, ax = P.plt.subplots(figsize=(7.4, 4.0))
for b in range(se.shape[1]):
    final = se[-300:, b].mean()
    ax.plot(se[:, b], lw=0.7, color=P.SERIES[0] if final > 0.5 else P.SERIES[7], alpha=0.8)
for i, (d, n, _) in enumerate(STAGES):
    ax.axvline(starts[i], color=P.MUTED, lw=0.6, ls=":")
    ax.annotate(f"{d:g} rad\n{n} modes", (starts[i], 0.88), xytext=(2, 0), textcoords="offset points", fontsize=6.5,
                color=P.INK2)
ax.set_xlabel("frame (1 kHz) from open loop")
ax.set_ylabel("short-exposure H Strehl")
ax.set_ylim(0, 1)
ax.set_title(f"Defocus-ramp bootstrap: {held}/24 converge")
P.save(fig, "exp17_defocus_ramp", args.tag,
       f"""Bootstrap from seeing-limited (0.6", 1 kHz) with only the focal-plane camera and no added hardware: the DM adds a
large defocus during acquisition (curvature regime), stepped down stage by stage (dotted lines: defocus, controlled
modes) while more modes are controlled; the last stage is the slim maintenance network at the fixed 1 rad defocus.
One network per stage, trained on the states that stage visits ({args.rounds} DAgger rounds). Gain {args.gain}, leak 0.99,
24 unseen atmospheres; blue = converged (mean SE Strehl > 0.5 over the last 300 frames). **What to look at:** whether
each stage hands a smaller residual to the next, and the converged fraction (goal G3 >= 95 %).""",
       title="exp17: defocus-ramp bootstrap")
