"""Single-frame focal-plane NN reconstructor: in focus vs fixed defocus.

Trains FPNet on generated residuals, reports estimation error by correction
level, then closes the Keck loop with it from open loop (no other sensor).

    python exp02_single_frame.py --defocus 1.0 --steps 6000 --tag d1
"""

import argparse
import json
import pathlib
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fpsim import system as S  # noqa: E402
from fpsim.atmos import Turbulence  # noqa: E402
from fpsim.data import ResidualGenerator  # noqa: E402
from fpsim.loop import DM, ModalProjector, h_band_science, run_loop  # noqa: E402
from fpsim.nets import FPNet  # noqa: E402
from fpsim.sensor import FocalPlaneSensor, FPSensorConfig  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--defocus", type=float, default=0.0, help="rms rad at band centre")
p.add_argument("--steps", type=int, default=6000)
p.add_argument("--batch", type=int, default=128)
p.add_argument("--photons", type=float, default=1e4)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--tag", default="run")
p.add_argument("--loop-steps", type=int, default=1500)
p.add_argument("--gain", type=float, default=0.4)
args = p.parse_args()
torch.manual_seed(0)
dev = "cuda"
OUT = ROOT / "results" / "exp02"
OUT.mkdir(parents=True, exist_ok=True)

cfg = S.KeckConfig()
sysd = S.build(cfg)
pupil = torch.tensor(sysd["pupil"], device=dev)
dm = DM(torch.tensor(sysd["ifs"], device=dev), torch.tensor(sysd["m2c"], device=dev), cfg.n_pupil)
gen = ResidualGenerator(cfg, sysd, dm, pupil)
sensor = FocalPlaneSensor(
    FPSensorConfig(defocus_rad=args.defocus, photons=args.photons), pupil, cfg.grid_m
).to(dev)
n_modes = dm.surfaces.shape[0]
net = FPNet(1, n_modes).to(dev)
scale = gen.sigma_nm  # per-mode output scaling
opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.1)
FLOOR = 10.0  # nm: relative-error loss floor


def batch(n, **kw):
    with torch.no_grad():
        opd, label = gen.sample(n, **kw)
        x = sensor.preprocess(sensor.frame(opd))
    return x, label


t0 = time.perf_counter()
net.train()
for step in range(args.steps):
    x, y = batch(args.batch)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = net(x).float() * scale
    err = ((pred - y) ** 2).sum(-1)
    loss = (err / ((y**2).sum(-1) + FLOOR**2 * n_modes / 100)).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    sched.step()
    if step % 500 == 0 or step == args.steps - 1:
        print(f"step {step:5d} loss {loss.item():.4f}  [{time.perf_counter() - t0:.0f} s]", flush=True)

# --- estimation error by correction level -------------------------------
net.eval()
report = {"args": vars(args), "levels": []}
even = torch.zeros(n_modes, dtype=torch.bool)
print("\nalpha      true rms   est err   rel")
for lo, hi in ((0.02, 0.05), (0.05, 0.15), (0.15, 0.4), (0.4, 1.0)):
    errs, trues = [], []
    with torch.no_grad():
        for _ in range(8):
            x, y = batch(256, alpha=(lo, hi))
            pred = net(x) * scale
            errs.append(((pred - y) ** 2).sum(-1))
            trues.append((y**2).sum(-1))
    e, t = torch.cat(errs).mean().sqrt().item(), torch.cat(trues).mean().sqrt().item()
    print(f"{lo:.2f}-{hi:.2f}  {t:8.1f}  {e:8.1f}  {e / t:.3f}")
    report["levels"].append(dict(alpha=[lo, hi], true_nm=t, err_nm=e))

# --- closed loop from open loop ----------------------------------------------
proj = ModalProjector(dm, pupil)


def recon(residual, k, hist):
    with torch.no_grad():
        x = sensor.preprocess(sensor.frame(residual))
        return net(x) * scale * 1e-9


sci = h_band_science(pupil, cfg.grid_m)
turb = Turbulence(cfg, batch=4, seed=100, seeing=0.6)
r = run_loop(turb, dm, recon, sci, pupil, 1000, args.loop_steps, gain=args.gain, delay=2, settle=500)
traj = r.strehl_se.mean(1)
print(
    f"\nclosed loop (gain {args.gain}): LE Strehl {r.strehl_le.mean():.3f} (+-{r.strehl_le.std():.3f}), "
    f"SE after settle {r.strehl_se[500:].mean():.3f}, residual {r.residual_nm[500:].mean():.0f} nm"
)
print("SE Strehl trajectory (every 100 frames):", [round(float(v), 3) for v in traj[::100]])
report["loop"] = dict(
    strehl_le=r.strehl_le.tolist(), se_traj=traj[::10].tolist(),
    residual_nm=float(r.residual_nm[500:].mean()),
)
torch.save(net.state_dict(), OUT / f"{args.tag}.pt")
(OUT / f"{args.tag}.json").write_text(json.dumps(report, indent=1))
