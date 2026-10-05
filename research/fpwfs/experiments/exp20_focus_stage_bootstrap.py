"""Bootstrap from seeing-limited with a WFS focus stage (curvature-regime acquisition).

exp12 (focus-stage runs): with a large defocus from the WFS focus stage (15-35 rad rms,
binned 4x4 pixels, finer pupil grid) one frame carries real information about 60 modes
once tip/tilt is controlled (error/residual ~0.5), where near-focus frames had none.
Schedule (``--schedule``, ``kind:defocus_rad:modes:frames`` per stage):

  cent  - tip/tilt from the image centroid (classical, calibrated by pokes);
  net   - a stage network trained on the states that stage visits (+ DAgger);
  maint - the slim maintenance network (1 rad, 64 px).

Large-defocus stages (>= 15 rad) use 88 px binned 4x4 on a 240-sample pupil;
intermediate (4-15 rad) 96 px at Nyquist; near focus 64 px at Nyquist.
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
p.add_argument("--tag", default="focus_stage")
p.add_argument("--width", type=int, default=24, help="stage-network width")
p.add_argument("--stem-stride", type=int, default=2, help="stage-network first-layer stride")
p.add_argument("--init", default=None, help="warm-start 'net' stages from saved stage networks, then train")
p.add_argument("--freeze", action="store_true", help="with --init: keep warm-started 'net' stages fixed")
p.add_argument("--mtrain-replay", type=int, default=0,
               help="dithered closed-loop collections (exp10 recipe) added to 'mtrain' data, against forgetting")
p.add_argument("--load", default=None, help="evaluate saved stage networks (skip data and training)")
p.add_argument("--eval-seeds", type=int, nargs="+", default=[950, 951])
p.add_argument("--schedule", default="cent:15:2:100,net:15:60:300,net:3:120:300,maint:1:120:600")
args = p.parse_args()
torch.manual_seed(0)
dev = "cuda"
OUT = ROOT / "results" / "exp20"
OUT.mkdir(parents=True, exist_ok=True)
cfg = S.KeckConfig()
sysd = S.build(cfg)
pupil = torch.tensor(sysd["pupil"], device=dev)
dm = DM(torch.tensor(sysd["ifs"], device=dev), torch.tensor(sysd["m2c"], device=dev), cfg.n_pupil)
proj = ModalProjector(dm, pupil)
n_modes = dm.surfaces.shape[0]
PH = 1e5
KINDS, STAGES = [], []
for item in args.schedule.split(","):
    kind, d, n, f = item.split(":")
    KINDS.append(kind)
    STAGES.append((float(d), int(n), int(f)))
import dataclasses  # noqa: E402

pupil_fine = torch.tensor(S.keck_pupil(dataclasses.replace(cfg, n_pupil=2 * cfg.n_pupil)), device=dev)


def make_sensor(d):
    if d >= 15.0:  # focus-stage acquisition: binned pixels, finer pupil (no phase aliasing)
        c = FPSensorConfig(defocus_rad=d, photons=PH, npix=88, binning=4)
        return FocalPlaneSensor(c, pupil_fine, cfg.grid_m).to(dev)
    if d >= 4.0:  # intermediate focus: full resolution, wider field for the pupil image (+-24 lambda/D)
        return FocalPlaneSensor(FPSensorConfig(defocus_rad=d, photons=PH, npix=96), pupil, cfg.grid_m).to(dev)
    return FocalPlaneSensor(FPSensorConfig(defocus_rad=d, photons=PH), pupil, cfg.grid_m).to(dev)


sensors = [make_sensor(d) for d, _, _ in STAGES]
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
for i in range(0 if args.load else args.collections):
    per_stage, _ = run_schedule(atmosphere(args.atm, 700 + i), est_noise=tuple(args.est_noise), seed=i)
    for s, (x, y) in enumerate(per_stage):
        data[s][0].append(x)
        data[s][1].append(y)
test = None if args.load else run_schedule(atmosphere(args.atm, 799, diverse=False),
                                           est_noise=tuple(args.est_noise), seed=99)[0]
print(f"collected {args.collections} x {args.atm} schedule trajectories [{time.perf_counter() - t0:.0f} s]", flush=True)

# replay buffer for the trainable maintenance stage: exp10-style dithered closed-loop states
if args.mtrain_replay and "mtrain" in KINDS:
    from fpsim.collect import collect  # noqa: E402

    s_m = KINDS.index("mtrain")
    n_m = STAGES[s_m][1]
    mmask = mask_for(n_m)
    pilot = collect(atmosphere(8, 1999), dm, pupil, sensors[s_m], 700, gains=(0.3, 0.5), est_noise=(0.0, 0.3),
                    seed=98, keep_from=300, mode_mask=mmask)
    cl_std = pilot.truth.flatten(0, 1).std(0).to(dev) * mmask
    for i in range(args.mtrain_replay):
        tr = collect(atmosphere(args.atm, 1500 + i), dm, pupil, sensors[s_m], 1500, gains=(0.25, 0.6),
                     est_noise=(0.0, 0.6), seed=1500 + i, keep_from=300, dither_modes_nm=cl_std, dither_max=3.0,
                     mode_mask=mmask)
        data[s_m][0].append(tr.frames.flatten(0, 1))
        data[s_m][1].append(tr.truth.flatten(0, 1))
    print(f"replay: {args.mtrain_replay} dithered closed-loop collections added to stage {s_m}", flush=True)

nets, scales = {}, {}
for s, (d, n, _) in enumerate(STAGES):
    if KINDS[s] == "mtrain":  # trainable maintenance network, warm-started from --maintenance
        nets[s] = FPNet(1, n, width=24, stem_stride=2).to(dev)
        if args.load:
            st_s = torch.load(ROOT / args.load)[f"stage{s}"]
        else:
            st_s = torch.load(ROOT / args.maintenance)
        scales[s] = st_s.pop("_scale").to(dev)
        nets[s].load_state_dict(st_s)
        nets[s].eval()
        continue
    if KINDS[s] != "net":
        continue
    nets[s] = FPNet(1, n, npix=sensors[s].cfg.npix, width=args.width, stem_stride=args.stem_stride).to(dev)
    src = args.load or args.init
    if src and f"stage{s}" in torch.load(ROOT / src):
        st_s = torch.load(ROOT / src)[f"stage{s}"]
        scales[s] = st_s.pop("_scale").to(dev)
        nets[s].load_state_dict(st_s)
        nets[s].eval()
    else:
        scales[s] = torch.cat(data[s][1])[:, :n].std(0).to(dev)


# ---- centroid tip/tilt: data-driven 2x2 calibration by pokes --------------------------
def centroid(e):
    h, w = e.shape[-2:]
    yy = torch.arange(h, device=e.device, dtype=e.dtype)
    xx = torch.arange(w, device=e.device, dtype=e.dtype)
    tot = e.sum((-2, -1)).clamp_min(1e-9)
    return torch.stack([(e.sum(-2) * xx).sum(-1) / tot, (e.sum(-1) * yy).sum(-1) / tot], -1)


cent_cal = {}
for s, kind in enumerate(KINDS):
    if kind != "cent":
        continue
    cols = []
    for i in range(2):
        e = torch.zeros(1, n_modes, device=dev)
        e[0, i] = 100e-9
        cp = centroid(sensors[s].frame(dm.opd(e), noise=False))
        cm = centroid(sensors[s].frame(dm.opd(-e), noise=False))
        cols.append(((cp - cm) / 200.0)[0])  # px per nm
    cent_cal[s] = (torch.linalg.inv(torch.stack(cols, 1)), centroid(sensors[s].frame(torch.zeros(1, 120, 120, device=dev), noise=False))[0])


def train_stage(s, steps, lr):
    if KINDS[s] not in ("net", "mtrain") or (args.freeze and KINDS[s] == "net"):
        return float("nan")
    if KINDS[s] == "mtrain" or (args.init and s in nets and KINDS[s] == "net"):
        lr = min(lr, 3e-4)  # fine-tuning a warm-started network
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
        if KINDS[s] == "maint":  # frozen maintenance network
            est = net_m(sensors[s].preprocess(e)) * scale_m
        elif KINDS[s] == "cent":
            inv, ref = cent_cal[s]
            est = (centroid(e) - ref) @ inv.T  # nm, tip/tilt
        else:
            est = nets[s](sensors[s].preprocess(e)) * scales[s]
    return torch.nn.functional.pad(est, (0, n_modes - est.shape[-1])) * 1e-9


def evaluate(label):
    held, ses = 0, []
    for seed in args.eval_seeds:
        _, se = run_schedule(Turbulence(cfg, batch=12, seed=seed, seeing=0.6), policy=policy, record=False, seed=seed)
        final = se[-300:].mean(0)
        held += int((final > 0.5).sum())
        ses.append(se)
    se = torch.cat(ses, 1)
    at_stage_end = [round(float(se[starts[i] + STAGES[i][2] - 1].median()), 2) for i in range(len(STAGES))]
    print(f"[{label}] BOOTSTRAP converged {held}/{12 * len(args.eval_seeds)}; median SE Strehl at end of each stage {at_stage_end}", flush=True)
    return held, se


report = {"args": vars(args), "stages": STAGES, "rounds": []}
if args.load:  # evaluation only
    held, se = evaluate("loaded networks")
    report["rounds"].append(dict(held=held))
for r in range(0 if args.load else args.rounds + 1):
    ratios = [train_stage(s, args.steps if r == 0 else args.steps // 2, 1e-3 if r == 0 else 3e-4)
              for s in range(len(STAGES))]
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
torch.save({f"stage{s}": {**nets[s].state_dict(), "_scale": scales[s].cpu()} for s in nets},
           OUT / f"{args.tag}.pt")

fig, ax = P.plt.subplots(figsize=(7.4, 4.0))
for b in range(se.shape[1]):
    final = se[-300:, b].mean()
    ax.plot(se[:, b], lw=0.7, color=P.SERIES[0] if final > 0.5 else P.SERIES[7], alpha=0.8)
for i, (d, n, _) in enumerate(STAGES):
    ax.axvline(starts[i], color=P.MUTED, lw=0.6, ls=":")
    ax.annotate(f"{KINDS[i]}\n{d:g} rad\n{n} modes", (starts[i], 0.84), xytext=(2, 0), textcoords="offset points", fontsize=6.5,
                color=P.INK2)
ax.set_xlabel("frame (1 kHz) from open loop")
ax.set_ylabel("short-exposure H Strehl")
ax.set_ylim(0, 1)
ax.set_title(f"Focus-stage bootstrap: {held}/{12 * len(args.eval_seeds)} converge")
P.save(fig, "exp20_focus_stage", args.tag,
       f"""Bootstrap from seeing-limited (0.6", 1 kHz) with only the focal-plane camera and its focus stage. Schedule
(dotted lines: estimator, defocus, controlled modes): {args.schedule}. 'cent' = tip/tilt from the image centroid
(calibrated by pokes); 'net' = stage network trained on that stage's states ({args.rounds} DAgger rounds); 'maint' = the
slim maintenance network at the fixed 1 rad defocus. Large-defocus stages use 4x4-binned 88 px frames. Gain
{args.gain}, leak 0.99, 24 unseen atmospheres; blue = converged (mean SE Strehl > 0.5 over the last 300 frames).
**What to look at:** whether each stage hands a smaller residual to the next, and the converged fraction
(goal G3 >= 95 %).""",
       title="exp20: focus-stage bootstrap")
