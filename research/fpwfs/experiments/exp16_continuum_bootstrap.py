"""Bootstrap by closing ALL controlled modes together, slowly (no mode staging).

exp15: during acquisition the uncorrected high-order turbulence scrambles the frame,
so sensing low orders first and high orders later cannot work. Instead train one
network on the continuum of states between open loop and closed loop with all
`--n-control` modes controlled (controller noise and gains diversify the paths),
plus, in DAgger rounds, states the network itself visits from open loop (runaway
states filtered). The loop then needs only a positively-correlated estimate at
every residual level, and closes with a gain ramp, then hands over to the
maintenance network.
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
p.add_argument("--n-control", type=int, default=120)
p.add_argument("--maintenance", default="results/exp10/h64_nc120_v3_r3.pt")
p.add_argument("--collections", type=int, default=8)
p.add_argument("--atm", type=int, default=16)
p.add_argument("--steps", type=int, default=12000)
p.add_argument("--rounds", type=int, default=2)
p.add_argument("--round-steps", type=int, default=5000)
p.add_argument("--gain-ramp", default="0:0.1,200:0.2,500:0.3,800:0.4", help="frame:gain,...")
p.add_argument("--handover", type=int, default=1000, help="frame to switch to the maintenance net (-1: never)")
p.add_argument("--tag", default="continuum")
args = p.parse_args()
torch.manual_seed(0)
dev = "cuda"
OUT = ROOT / "results" / "exp16"
OUT.mkdir(parents=True, exist_ok=True)
cfg = S.KeckConfig()
sysd = S.build(cfg)
pupil = torch.tensor(sysd["pupil"], device=dev)
dm = DM(torch.tensor(sysd["ifs"], device=dev), torch.tensor(sysd["m2c"], device=dev), cfg.n_pupil)
proj = ModalProjector(dm, pupil)
n_modes = dm.surfaces.shape[0]
NC = args.n_control
mask = torch.zeros(n_modes, device=dev)
mask[:NC] = 1
scfg = FPSensorConfig(defocus_rad=1.0, photons=1e5)
sensor = FocalPlaneSensor(scfg, pupil, cfg.grid_m).to(dev)
_rng = torch.Generator().manual_seed(5)
ramp = [(int(a), float(b)) for a, b in (item.split(":") for item in args.gain_ramp.split(","))]


def atmosphere(batch, seed):
    u = torch.rand(2, generator=_rng).tolist()
    return Turbulence(cfg, batch=batch, seed=seed, seeing=0.45 + 0.4 * u[0], wind_scale=0.7 + 0.8 * u[1])


class Data:
    def __init__(self):
        self.x, self.y = [], []

    def add(self, tr, max_nm=None):
        x, y = tr.frames.flatten(0, 1), tr.truth.flatten(0, 1)[:, :NC]
        if max_nm is not None:
            keep = y.pow(2).sum(-1).sqrt() < max_nm
            print(f"  kept {int(keep.sum())}/{len(keep)} states below {max_nm:.0f} nm", flush=True)
            x, y = x[keep], y[keep]
        self.x.append(x)
        self.y.append(y)
        self.X, self.Y = torch.cat(self.x), torch.cat(self.y)


t0 = time.perf_counter()
data = Data()
for i in range(args.collections):
    # slow, noisy closures from open loop: covers every residual level on the way down
    data.add(collect(atmosphere(args.atm, 300 + i), dm, pupil, sensor, 800, gains=(0.03, 0.4),
                     est_noise=(0.0, 1.5), seed=300 + i, keep_from=0, mode_mask=mask))
open_loop_nm = float(data.Y[: args.atm].pow(2).sum(-1).mean().sqrt())
scale = data.Y.std(0).to(dev)
print(f"{len(data.X)} continuum states, open-loop {open_loop_nm:.0f} nm in {NC} modes "
      f"[{time.perf_counter() - t0:.0f} s]", flush=True)
net = FPNet(1, NC).to(dev)


def train(steps, lr):
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    net.train()
    for step in range(steps):
        idx = torch.randint(0, len(data.X), (128,))
        ph = log_uniform(128, 0.5, 2.0, dev)[:, None, None]
        x = sensor.preprocess(noisy_frames(data.X[idx].to(dev), scfg, 1e5, ph))
        y = data.Y[idx].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = net(x).float() * scale
        loss = ((pred - y).pow(2).sum(-1) / (y.pow(2).sum(-1) + 100.0)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 3000 == 0 or step == steps - 1:
            print(f"  step {step:5d} loss {loss.item():.3f} [{time.perf_counter() - t0:.0f} s]", flush=True)
    net.eval()


st = torch.load(ROOT / args.maintenance)
scale_m = st.pop("_scale").to(dev)
net_m = FPNet(1, 120).to(dev)
net_m.load_state_dict(st)
net_m.eval()


def gain_at(k):
    g = ramp[0][1]
    for f, v in ramp:
        if k >= f:
            g = v
    return g


def policy(noisy, k, hist):
    with torch.no_grad():
        return torch.nn.functional.pad(net(sensor.preprocess(noisy)) * scale, (0, n_modes - NC)) * 1e-9


def recon(residual, k, hist):
    use_m = args.handover >= 0 and k >= args.handover
    with torch.no_grad():
        if use_m:
            est = net_m(sensor.preprocess(sensor.frame(residual))) * scale_m
        else:
            est = net(sensor.preprocess(sensor.frame(residual))) * scale
    # the gain ramp is applied here (run_loop gain = 1)
    return torch.nn.functional.pad(est, (0, n_modes - est.shape[-1])) * mask * gain_at(k) * 1e-9


def evaluate(label):
    rr, fig_rows = {}, []
    held_total = 0
    for seed in (900, 901):
        r = run_loop(Turbulence(cfg, batch=12, seed=seed, seeing=0.6), dm, recon, h_band_science(pupil, cfg.grid_m),
                     pupil, 1000, 2000, gain=1.0, leak=0.99, delay=2, settle=1200)
        held = int((r.strehl_le > 0.5).sum())
        held_total += held
        rr[f"seed{seed}"] = dict(strehl_le=r.strehl_le.tolist(), se=r.strehl_se.tolist(), held=held)
        fig_rows.append(r)
    med = torch.cat([r.strehl_le for r in fig_rows]).median().item()
    print(f"[{label}] BOOTSTRAP converged {held_total}/24, final LE Strehl median {med:.3f}; mean SE every 100 "
          f"{[round(float(v), 2) for v in torch.cat([r.strehl_se for r in fig_rows], 1).mean(1)[::100]]}", flush=True)
    return dict(label=label, held=held_total, runs=rr), fig_rows


train(args.steps, 1e-3)
report = {"args": vars(args), "rounds": []}
res, runs = evaluate("round 0")
report["rounds"].append(res)
for r_i in range(args.rounds):
    for j in range(2):  # DAgger from open loop, network flying (beta 0.3 -> 0)
        beta = 0.3 * (1 - r_i / max(args.rounds - 1, 1))
        data.add(collect(atmosphere(args.atm, 400 + 10 * r_i + j), dm, pupil, sensor, 800, behaviour=policy,
                         beta=beta, gains=(0.1, 0.3), seed=400 + 10 * r_i + j, keep_from=0, mode_mask=mask),
                 max_nm=1.5 * open_loop_nm)
    train(args.round_steps, 3e-4)
    res, runs = evaluate(f"round {r_i + 1}")
    report["rounds"].append(res)
    torch.save({**net.state_dict(), "_scale": scale.cpu()}, OUT / f"{args.tag}_r{r_i + 1}.pt")
(OUT / f"{args.tag}.json").write_text(json.dumps(report))

fig, ax = P.plt.subplots(figsize=(7.2, 4.0))
for i, r in enumerate(runs):
    for b in range(r.strehl_se.shape[1]):
        ax.plot(r.strehl_se[:, b], lw=0.7, color=P.SERIES[0] if r.strehl_le[b] > 0.5 else P.SERIES[7], alpha=0.8)
for f, g in ramp:
    ax.axvline(f, color=P.MUTED, lw=0.6, ls=":")
    ax.annotate(f"g={g:g}", (f, 0.95), xytext=(2, 0), textcoords="offset points", fontsize=7, color=P.INK2)
if args.handover >= 0:
    ax.axvline(args.handover, color=P.INK2, lw=1, ls="--")
    ax.annotate("maintenance net", (args.handover, 0.85), xytext=(3, 0), textcoords="offset points", fontsize=7.5,
                color=P.INK2)
held = sum(int((r.strehl_le > 0.5).sum()) for r in runs)
ax.set_xlabel("frame (1 kHz) from open loop")
ax.set_ylabel("short-exposure H Strehl")
ax.set_ylim(0, 1)
ax.set_title(f"Closing all {NC} modes together from open loop: {held}/24 converge")
P.save(fig, "exp16_continuum_bootstrap", args.tag,
       f"""Bootstrap without mode staging: one single-frame network (H band, 1 rad defocus, 1e5 photons) estimates all
{NC} controlled modes at every residual level, trained on the continuum of states from open loop to closed
loop (slow, noisy closures) plus DAgger rounds from open loop; the gain ramps up as marked (leak 0.99), then
the exp10 v3 maintenance network takes over (dashed). 24 unseen atmospheres at 0.6". Blue: converged.
**What to look at:** the fraction converging (goal G3: >= 95 %) and the time to reach ~0.72.""",
       title="exp16: closing all modes together from open loop")
