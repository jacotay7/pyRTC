"""Acquisition with temporal phase diversity: known alternating DM probes, two frames.

During acquisition the DM adds a known focus probe with alternating sign on
consecutive frames (+P, -P, ...), on top of the fixed 1 rad defocus. A network
sees the last K frames and the known DM commands applied during them, and
estimates the probe-free residual of the first `--n-out` modes.

Compared on held-out atmospheres (disjoint seeds), per loop stage (modes
already controlled, 0 = open loop): single frame, no probe vs K = 2 with probes.
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
from fpsim.loop import DM, ModalProjector  # noqa: E402
from fpsim.nets import FPNet  # noqa: E402
from fpsim.optics import defocus_opd  # noqa: E402
from fpsim.sensor import FocalPlaneSensor, FPSensorConfig  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--configs", default="1:0,2:1,2:2", help="K:probe_rad,... (probe = rms focus at 1.65 um)")
p.add_argument("--n-out", type=int, default=20)
p.add_argument("--steps", type=int, default=8000)
p.add_argument("--atm", type=int, default=16)
p.add_argument("--collect-steps", type=int, default=600)
p.add_argument("--tag", default="probes")
args = p.parse_args()
torch.manual_seed(0)
dev = "cuda"
OUT = ROOT / "results" / "exp15"
OUT.mkdir(parents=True, exist_ok=True)
cfg = S.KeckConfig()
sysd = S.build(cfg)
pupil = torch.tensor(sysd["pupil"], device=dev)
dm = DM(torch.tensor(sysd["ifs"], device=dev), torch.tensor(sysd["m2c"], device=dev), cfg.n_pupil)
proj = ModalProjector(dm, pupil)
n_modes = dm.surfaces.shape[0]
NO = args.n_out
scfg = FPSensorConfig(defocus_rad=1.0, photons=1e5)
sensor = FocalPlaneSensor(scfg, pupil, cfg.grid_m).to(dev)
STAGES = (0, 2, 5, 10, 20)
# focus probe in DM-KL coordinates: projection of a unit-rms (1 m) Zernike focus
focus_modes = proj(defocus_opd(cfg.n_pupil, cfg.grid_m, 10.95, 1.0, dev)[None])[0]
focus_modes = focus_modes / focus_modes.norm()  # unit rms DM focus
rad_to_m = 1.65e-6 / (2 * torch.pi)


def mask_for(n):
    m = torch.zeros(n_modes, device=dev)
    m[:n] = 1
    return m


def gather(K, probe_rad, seed0):
    """Return windows of K consecutive frames, the applied-command diffs, and labels."""
    probe = focus_modes * probe_rad * rad_to_m if probe_rad > 0 else None
    xs, cs, ys, stage = [], [], [], []
    for i, n in enumerate(STAGES):
        for j in range(2):
            tr = collect(Turbulence(cfg, batch=args.atm, seed=seed0 + 10 * i + j, seeing=0.6), dm, pupil, sensor,
                         args.collect_steps, gains=(0.2, 0.5), est_noise=(0.0, 0.6), seed=seed0 + 10 * i + j,
                         keep_from=0, mode_mask=mask_for(n), probe_m=probe)
            t_len = tr.frames.shape[0]
            idx = torch.arange(K - 1, t_len)
            win = idx[:, None] + torch.arange(-K + 1, 1)[None]  # (T', K)
            f = tr.frames[win]  # (T', K, B, H, W)
            a = tr.applied[win]  # (T', K, B, n_modes)
            xs.append(f.permute(0, 2, 1, 3, 4).flatten(0, 1))
            d = (a[:, -1:] - a[:, :-1]).permute(0, 2, 1, 3).flatten(0, 1).flatten(1) * 1e9  # (N, (K-1) n_modes)
            cs.append(d)
            ys.append(tr.truth[idx].flatten(0, 1)[:, :NO])
            stage.append(torch.full((len(xs[-1]),), n))
    return torch.cat(xs), torch.cat(cs), torch.cat(ys), torch.cat(stage)


rows = []
for item in args.configs.split(","):
    K, probe_rad = int(item.split(":")[0]), float(item.split(":")[1])
    t0 = time.perf_counter()
    X, C, Y, _ = gather(K, probe_rad, seed0=100)  # disjoint atmosphere blocks (Turbulence.SEED_STRIDE)
    Xt, Ct, Yt, St = gather(K, probe_rad, seed0=200)
    scale = Y.std(0).to(dev)
    cscale = C.std(0).clamp_min(1e-3).to(dev) if K > 1 else None
    print(f"K={K} probe {probe_rad} rad: {len(X)} train / {len(Xt)} test windows [{time.perf_counter() - t0:.0f} s]",
          flush=True)
    net = FPNet(K, NO, cond=(K - 1) * n_modes).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=args.steps, pct_start=0.1)

    def inputs(x, c, ph=None):
        e = noisy_frames(x.to(dev), scfg, 1e5, ph)
        xx = torch.stack([sensor.preprocess(e[:, k])[:, 0] for k in range(K)], 1)
        cc = c.to(dev) / cscale if K > 1 else None
        return xx, cc

    for step in range(args.steps):
        idx = torch.randint(0, len(X), (128,))
        ph = log_uniform(128, 0.5, 2.0, dev)[:, None, None, None]
        xx, cc = inputs(X[idx], C[idx], ph)
        y = Y[idx].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = net(xx, cc).float() * scale
        loss = ((pred - y).pow(2).sum(-1) / (y.pow(2).sum(-1) + 100.0)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
    net.eval()
    torch.save({**net.state_dict(), "_scale": scale.cpu(), "_cscale": None if cscale is None else cscale.cpu(),
                "_focus": focus_modes.cpu()}, OUT / f"{args.tag}_K{K}_p{probe_rad:g}.pt")
    for n in STAGES:
        sel = (St == n).nonzero()[:, 0]
        errs, trues = [], []
        with torch.no_grad():
            for k in range(0, len(sel), 256):
                b = sel[k:k + 256]
                xx, cc = inputs(Xt[b], Ct[b])
                y = Yt[b].to(dev)
                pred = net(xx, cc) * scale
                errs.append((pred - y).pow(2).sum(-1))
                trues.append(y.pow(2).sum(-1))
        e, t = torch.cat(errs).mean().sqrt().item(), torch.cat(trues).mean().sqrt().item()
        rows.append(dict(K=K, probe_rad=probe_rad, stage=n, true_nm=t, err_nm=e))
        print(f"K={K} probe {probe_rad} rad | loop controlling {n:2d} modes: first {NO} modes residual {t:6.0f} nm, "
              f"error {e:6.0f} nm, ratio {e / t:.2f}", flush=True)
(OUT / f"{args.tag}.json").write_text(json.dumps(rows))

fig, ax = P.plt.subplots(figsize=(6.4, 4.0))
for i, item in enumerate(args.configs.split(",")):
    K, pr = int(item.split(":")[0]), float(item.split(":")[1])
    rr = [r for r in rows if r["K"] == K and r["probe_rad"] == pr]
    lab = "1 frame, no probe" if pr == 0 else f"{K} frames, +-{pr:g} rad focus probes"
    ax.plot([r["stage"] for r in rr], [r["err_nm"] / r["true_nm"] for r in rr], marker="o", color=P.SERIES[i], label=lab)
ax.axhline(1, color=P.MUTED, lw=1, ls=":")
ax.set_xlabel("modes already controlled (0 = seeing-limited, open loop)")
ax.set_ylabel(f"error / residual, first {NO} modes")
ax.set_ylim(0, 1.2)
ax.set_title("Acquisition: does temporal phase diversity give capture range?")
ax.legend(fontsize=8)
P.save(fig, "exp15_probe_diversity", args.tag,
       f"""Acquisition test on **held-out atmospheres** (disjoint seeds; the earlier exp12 had train/test overlap).
States from loops controlling 0-20 modes, from open loop (0.6", 1 kHz). Single frame (fixed 1 rad defocus)
vs two consecutive frames with a known focus probe added by the DM with alternating sign (+P, -P), the
network also receiving the known DM commands. Output: the probe-free residual of the first {NO} DM-KL modes.
y = rms error / rms residual (below 1 the loop can contract). **What to look at:** the left end (open
loop): two-image phase diversity is the classical way to resolve the even-mode sign in large aberrations.""",
       title="exp15: temporal phase diversity for acquisition")
