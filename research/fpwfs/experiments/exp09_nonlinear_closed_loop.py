"""Can a non-linear estimator beat the fitting-error coupling on closed-loop states?

Collect closed-loop states (ideal sensor flying the loop with random gains and
small estimate noise, so states cover a band around convergence), train a CNN on
noisy frames of them, and report the error/residual RATIO on held-out states.
The loop can only contract if the ratio is clearly below 1 (linear map: 1.1-1.7).

    python exp09_nonlinear_closed_loop.py --npix 128 --wavelength 2.2e-6 --tag k_128
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
from fpsim.collect import collect, log_uniform, noisy_frames  # noqa: E402
from fpsim.loop import DM, ModalProjector, h_band_science, run_loop  # noqa: E402
from fpsim.nets import FPNet  # noqa: E402
from fpsim.sensor import FocalPlaneSensor, FPSensorConfig  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--defocus", type=float, default=1.0)
p.add_argument("--wavelength", type=float, default=1.65e-6)
p.add_argument("--npix", type=int, default=64)
p.add_argument("--photons", type=float, default=1e5)
p.add_argument("--atm", type=int, default=16)
p.add_argument("--collect-steps", type=int, default=2300)
p.add_argument("--collections", type=int, default=5)
p.add_argument("--steps", type=int, default=12000)
p.add_argument("--batch", type=int, default=128)
p.add_argument("--width", type=int, default=48)
p.add_argument("--tag", default="run")
p.add_argument("--handover", action="store_true", help="also run an ideal -> network hand-over loop")
args = p.parse_args()
torch.manual_seed(0)
dev = "cuda"
OUT = ROOT / "results" / "exp09"
OUT.mkdir(parents=True, exist_ok=True)
cfg = S.KeckConfig()
sysd = S.build(cfg)
pupil = torch.tensor(sysd["pupil"], device=dev)
dm = DM(torch.tensor(sysd["ifs"], device=dev), torch.tensor(sysd["m2c"], device=dev), cfg.n_pupil)
proj = ModalProjector(dm, pupil)
n_modes = dm.surfaces.shape[0]
scfg = FPSensorConfig(defocus_rad=args.defocus, photons=args.photons, wavelength=args.wavelength, npix=args.npix)
sensor = FocalPlaneSensor(scfg, pupil, cfg.grid_m).to(dev)

# ---- data: closed-loop states only (drop the first 300 frames of each run) ----------
t0 = time.perf_counter()
frames, truth = [], []
for i in range(args.collections + 1):  # last collection = held-out test set
    turb = Turbulence(cfg, batch=args.atm, seed=2000 + 100 * i, seeing=0.6)
    tr = collect(turb, dm, pupil, sensor, args.collect_steps, gains=(0.25, 0.6), est_noise=(0.0, 0.6),
                 seed=i, keep_from=300)
    frames.append(tr.frames.flatten(0, 1))
    truth.append(tr.truth.flatten(0, 1))
test_x, test_y = frames.pop(), truth.pop()
train_x, train_y = torch.cat(frames), torch.cat(truth)
scale = train_y.std(0).to(dev)  # per-mode closed-loop residual std: whitened outputs, no shrinkage bias toward 0
print(f"{len(train_x)} train / {len(test_x)} test closed-loop states, residual "
      f"{train_y.pow(2).sum(-1).mean().sqrt():.1f} nm [{time.perf_counter() - t0:.0f} s]", flush=True)

net = FPNet(1, n_modes, npix=args.npix, width=args.width).to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=args.steps, pct_start=0.1)


def batch(x, y, idx, photon_range=(0.5, 2.0)):
    clean = x[idx].to(dev)
    ph = log_uniform(len(idx), *photon_range, dev)[:, None, None]
    e = noisy_frames(clean, scfg, args.photons, ph)
    return sensor.preprocess(e), y[idx].to(dev)


t0 = time.perf_counter()
for step in range(args.steps):
    idx = torch.randint(0, len(train_x), (args.batch,))
    x, y = batch(train_x, train_y, idx)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = net(x).float() * scale
    loss = (pred - y).pow(2).sum(-1).mean() / y.pow(2).sum(-1).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    sched.step()
    if step % 2000 == 0 or step == args.steps - 1:
        print(f"  step {step:5d} loss {loss.item():.3f} [{time.perf_counter() - t0:.0f} s]", flush=True)

net.eval()
errs, trues, ests = [], [], []
with torch.no_grad():
    for i in range(0, len(test_x), 256):
        idx = torch.arange(i, min(i + 256, len(test_x)))
        x, y = batch(test_x, test_y, idx, photon_range=(1.0, 1.0001))
        pred = net(x) * scale
        errs.append((pred - y).pow(2).sum(-1))
        trues.append(y.pow(2).sum(-1))
        ests.append(torch.stack([pred, y]))
err, true = torch.cat(errs).mean().sqrt().item(), torch.cat(trues).mean().sqrt().item()
pe = torch.cat([e[0] for e in ests])
ty = torch.cat([e[1] for e in ests])
k_mode = (pe * ty).sum(0) / (ty * ty).sum(0)
print(f"TEST: residual {true:.1f} nm, CNN error {err:.1f} nm, ratio {err / true:.2f}; "
      f"slope k median {k_mode.median():.2f}, modes with k<0.3: {(k_mode < 0.3).sum().item()}", flush=True)
report = dict(args=vars(args), residual_nm=true, err_nm=err, ratio=err / true, k=k_mode.tolist())

if args.handover:
    def recon(residual, k, hist):
        if k < 300:
            return proj(residual.mean(0))
        with torch.no_grad():
            return net(sensor.preprocess(sensor.frame(residual))) * scale * 1e-9

    for g in (0.2, 0.4):
        r = run_loop(Turbulence(cfg, batch=4, seed=100, seeing=0.6), dm, recon, h_band_science(pupil, cfg.grid_m),
                     pupil, 1000, 1500, gain=g, delay=2, settle=600)
        traj = r.strehl_se.mean(1)
        report[f"handover_gain_{g}"] = dict(se_traj=traj.tolist(), strehl_le=r.strehl_le.tolist())
        print(f"HANDOVER gain {g}: LE Strehl {r.strehl_le.mean():.3f}; SE every 100 {[round(float(v), 2) for v in traj[::100]]}",
              flush=True)
torch.save(net.state_dict(), OUT / f"{args.tag}.pt")
(OUT / f"{args.tag}.json").write_text(json.dumps(report))
