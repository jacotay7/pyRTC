"""Operating the focal-plane-only AO loop: bootstrap, hold, detect loss, re-acquire.

Everything an RTC would do with only the focal-plane camera (+ its focus stage):

  ACQUIRE  focus stage at 25 rad: centroid tip/tilt -> 60-mode net -> 120-mode net
  HOLD     focus stage at 1 rad: maintenance net (exp20 v7 networks)
  LOSS     detected from the frame alone: the flux fraction in the core of the 1 rad
           frame (tracks Strehl) below a threshold for `--loss-frames` frames
           -> flatten the DM and re-acquire.

Each atmosphere runs its own state machine. Reports the fraction of time locked,
losses/recoveries, and the effective (time-averaged) Strehl.
"""

import argparse
import dataclasses
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
from fpsim.loop import DM, h_band_science  # noqa: E402
from fpsim.nets import FPNet  # noqa: E402
from fpsim.optics import add_noise  # noqa: E402
from fpsim.sensor import FocalPlaneSensor, FPSensorConfig  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--nets", default="results/exp20/fs25_v7.pt")
p.add_argument("--steps", type=int, default=10000)
p.add_argument("--seeds", type=int, nargs="+", default=[980, 981])
p.add_argument("--gain", type=float, default=0.3)
p.add_argument("--loss-threshold", type=float, default=None, help="core-flux metric; default: calibrated")
p.add_argument("--loss-frames", type=int, default=20)
p.add_argument("--no-supervisor", action="store_true")
p.add_argument("--acq", type=int, nargs=3, default=[100, 200, 100], metavar=("CENT", "S1", "S2"),
               help="frames of each acquisition step (centroid TT, 60-mode net, 120-mode net)")
p.add_argument("--photons", type=float, default=1e5, help="photons per frame on the focal-plane camera")
p.add_argument("--tag", default="supervised")
args = p.parse_args()
dev = "cuda"
cfg = S.KeckConfig()
sysd = S.build(cfg)
pupil = torch.tensor(sysd["pupil"], device=dev)
pupil_fine = torch.tensor(S.keck_pupil(dataclasses.replace(cfg, n_pupil=2 * cfg.n_pupil)), device=dev)
dm = DM(torch.tensor(sysd["ifs"], device=dev), torch.tensor(sysd["m2c"], device=dev), cfg.n_pupil)
n_modes = dm.surfaces.shape[0]
PH = args.photons
far = FocalPlaneSensor(FPSensorConfig(defocus_rad=25.0, photons=PH, npix=88, binning=4), pupil_fine, cfg.grid_m).to(dev)
near = FocalPlaneSensor(FPSensorConfig(defocus_rad=1.0, photons=PH), pupil, cfg.grid_m).to(dev)
# acquisition schedule within ACQUIRE: (estimator, controlled modes, frames)
ACQ = [("cent", 2, args.acq[0]), ("s1", 60, args.acq[1]), ("s2", 120, args.acq[2])]
ACQ_LEN = sum(f for _, _, f in ACQ)

saved = torch.load(ROOT / args.nets)
nets = {}
for key, name, n, npix in (("stage1", "s1", 60, 88), ("stage2", "s2", 120, 88), ("stage3", "m", 120, 64)):
    st = dict(saved[key])
    scale = st.pop("_scale").to(dev)
    net = FPNet(1, n, npix=npix, width=24, stem_stride=2).to(dev)
    net.load_state_dict(st)
    net.eval()
    nets[name] = (net, scale)


def centroid(e):
    h, w = e.shape[-2:]
    yy = torch.arange(h, device=e.device, dtype=e.dtype)
    xx = torch.arange(w, device=e.device, dtype=e.dtype)
    tot = e.sum((-2, -1)).clamp_min(1e-9)
    return torch.stack([(e.sum(-2) * xx).sum(-1) / tot, (e.sum(-1) * yy).sum(-1) / tot], -1)


cols = []
for i in range(2):
    e = torch.zeros(1, n_modes, device=dev)
    e[0, i] = 100e-9
    cols.append(((centroid(far.frame(dm.opd(e), noise=False)) - centroid(far.frame(dm.opd(-e), noise=False))) / 200.0)[0])
cent_inv = torch.linalg.inv(torch.stack(cols, 1))
cent_ref = centroid(far.frame(torch.zeros(1, 120, 120, device=dev), noise=False))[0]


def core_metric(e):
    """Fraction of flux in the brightest 3x3 pixels of the 1 rad frame (tracks Strehl)."""
    flux = e.sum((-2, -1)).clamp_min(1.0)
    pooled = torch.nn.functional.avg_pool2d(e[:, None].clamp_min(0), 3, 1, 1)[:, 0] * 9
    return pooled.amax((-2, -1)) / flux


def mask_for(n):
    m = torch.zeros(n_modes, device=dev)
    m[:n] = 1
    return m


def run(seed):
    turb = Turbulence(cfg, batch=12, seed=seed, seeing=0.6)
    b = 12
    sci = h_band_science(pupil, cfg.grid_m)
    ref = sci(torch.zeros_like(pupil)).amax()
    cmd = torch.zeros(b, n_modes, device=dev)
    pending = [torch.zeros(b, n_modes, device=dev) for _ in range(2)]
    phase_t = torch.zeros(b, dtype=torch.long, device=dev)  # frames since (re)acquisition began
    holding = torch.zeros(b, dtype=torch.bool, device=dev)
    bad = torch.zeros(b, dtype=torch.long, device=dev)
    se_all, metric_all, losses = [], [], [0] * b
    for k in range(args.steps):
        applied = pending.pop(0)
        residual = turb.step(1e-3) - dm.opd(applied)
        se_all.append((sci(residual).amax(dim=(-2, -1)) / ref).cpu())
        est = torch.zeros(b, n_modes, device=dev)
        metric = torch.full((b,), float("nan"), device=dev)
        acq = (~holding).nonzero()[:, 0]
        hold = holding.nonzero()[:, 0]
        with torch.no_grad():
            if len(acq):
                e = add_noise(far.frame(residual[acq], noise=False) / PH, PH, 0.6)
                t = phase_t[acq]
                # which acquisition step each atmosphere is in
                edges = torch.tensor([ACQ[0][2], ACQ[0][2] + ACQ[1][2]], device=dev)
                step = (t >= edges[0]).long() + (t >= edges[1]).long()
                cent_est = torch.nn.functional.pad((centroid(e) - cent_ref) @ cent_inv.T, (0, n_modes - 2))
                n1, s1 = nets["s1"]
                n2, s2 = nets["s2"]
                x = far.preprocess(e)
                e1 = torch.nn.functional.pad(n1(x) * s1, (0, n_modes - 60))
                e2 = torch.nn.functional.pad(n2(x) * s2, (0, n_modes - 120))
                choice = torch.stack([cent_est * mask_for(2), e1 * mask_for(60), e2 * mask_for(120)], 0)
                est[acq] = choice[step, torch.arange(len(acq), device=dev)]
            if len(hold):
                e = add_noise(near.frame(residual[hold], noise=False) / PH, PH, 0.6)
                nm, sm = nets["m"]
                est[hold] = torch.nn.functional.pad(nm(near.preprocess(e)) * sm, (0, n_modes - 120))
                metric[hold] = core_metric(e)
        metric_all.append(metric.cpu())
        cmd = 0.99 * cmd + args.gain * est * 1e-9
        # state machine
        phase_t += 1
        done = (~holding) & (phase_t >= ACQ_LEN)
        holding = holding | done
        if not args.no_supervisor and args.loss_threshold is not None:
            low = holding & (metric < args.loss_threshold)
            bad = torch.where(low, bad + 1, torch.zeros_like(bad))
            lost = bad >= args.loss_frames
            if lost.any():
                for i in lost.nonzero()[:, 0].tolist():
                    losses[i] += 1
                cmd[lost] = 0  # flatten the DM, move the focus stage, re-acquire
                holding = holding & ~lost
                phase_t[lost] = 0
                bad[lost] = 0
        pending.append(cmd.clone())
    return torch.stack(se_all), torch.stack(metric_all), losses


# calibrate the loss threshold from the metric when locked vs lost (unsupervised run, one seed block)
if args.loss_threshold is None and not args.no_supervisor:
    se, met, _ = run(990)
    locked = met[se > 0.5]
    lost = met[(se < 0.1) & ~torch.isnan(met)]
    lk = locked[~torch.isnan(locked)]
    thr = float((lk.quantile(0.001) + (lost.quantile(0.99) if len(lost) else lk.quantile(0.001) * 0.5)) / 2)
    args.loss_threshold = thr
    print(f"loss threshold {thr:.4f} (locked metric p0.1 {lk.quantile(0.001):.4f}, lost p99 "
          f"{lost.quantile(0.99) if len(lost) else float('nan'):.4f})", flush=True)

t0 = time.perf_counter()
ses, all_losses = [], []
for seed in args.seeds:
    se, met, losses = run(seed)
    ses.append(se)
    all_losses += losses
se = torch.cat(ses, 1)  # (T, B)
first_lock = ACQ_LEN + 200
locked_frac = (se[first_lock:] > 0.5).float().mean().item()
eff = se[first_lock:].mean().item()
per_atm_locked = (se[first_lock:] > 0.5).float().mean(0)
print(f"{'supervised' if not args.no_supervisor else 'no supervisor'}: {se.shape[1]} atmospheres x "
      f"{args.steps / 1000:.0f} s: time locked {locked_frac * 100:.1f} %, effective SE H Strehl {eff:.3f}, "
      f"losses {sum(all_losses)} (per atmosphere {all_losses}), atmospheres locked > 95 % of the time: "
      f"{int((per_atm_locked > 0.95).sum())}/{se.shape[1]} [{time.perf_counter() - t0:.0f} s]", flush=True)
out = ROOT / "results" / "exp21"
out.mkdir(parents=True, exist_ok=True)
(out / f"{args.tag}.json").write_text(json.dumps(dict(args=vars(args), locked_frac=locked_frac, effective=eff,
                                                      losses=all_losses, per_atm_locked=per_atm_locked.tolist())))
fig, ax = P.plt.subplots(figsize=(8.0, 4.0))
for bb in range(se.shape[1]):
    ax.plot(se[::5, bb], lw=0.5, color=P.SERIES[0] if per_atm_locked[bb] > 0.95 else P.SERIES[1], alpha=0.7)
ax.set_xlabel("frame / 5 (1 kHz), from seeing-limited")
ax.set_ylabel("short-exposure H Strehl")
ax.set_ylim(0, 1)
ax.set_title(f"Focal-plane-only operation: locked {locked_frac * 100:.0f} % of the time, effective Strehl {eff:.2f}")
P.save(fig, "exp21_operation", args.tag,
       f"""{se.shape[1]} unseen atmospheres (0.6", 1 kHz) operated for {args.steps / 1000:.0f} s each with only the focal-plane
camera and its focus stage: automatic acquisition from seeing-limited (25 rad focus: centroid TT, 60-mode and 120-mode
networks), then the 1 rad maintenance network; {'a supervisor watches the core-flux fraction of the 1 rad frame and, on loss, flattens the DM and re-acquires' if not args.no_supervisor else 'no supervisor (a lost loop stays lost)'}.
Blue: locked > 95 % of the time; orange: less. **What to look at:** time locked and the effective Strehl, which
count every acquisition, loss and recovery.""",
       title="exp21: operating the focal-plane-only loop")
