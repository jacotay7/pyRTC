"""Multi-frame focal-plane reconstructor with DM-command diversity (+ DAgger).

Input: the last K noisy frames and the modal DM commands applied during them
(what an RTC knows). Output: the current modal residual. Data come from
closed-loop trajectories (pyturb frozen flow, real loop dynamics), first under
a noisy ideal controller, then (DAgger) under the network itself.

    python exp04_temporal.py --defocus 0.0 --K 3 --tag infocus_k3
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
from fpsim.loop import DM, h_band_science, run_loop  # noqa: E402
from fpsim.nets import FPNet  # noqa: E402
from fpsim.sensor import FocalPlaneSensor, FPSensorConfig  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--defocus", type=float, default=0.0)
p.add_argument("--K", type=int, default=3, help="frames per input")
p.add_argument("--photons", type=float, default=1e4)
p.add_argument("--atm", type=int, default=16, help="atmospheres per collection")
p.add_argument("--collect-steps", type=int, default=2500)
p.add_argument("--collections", type=int, default=4)
p.add_argument("--train-steps", type=int, default=8000)
p.add_argument("--dagger", type=int, default=2, help="DAgger rounds")
p.add_argument("--dagger-steps", type=int, default=3000)
p.add_argument("--batch", type=int, default=128)
p.add_argument("--gain", type=float, default=0.4)
p.add_argument("--dither", type=float, default=0.0, help="nm rms random DM dither per mode")
p.add_argument("--loop-steps", type=int, default=1500)
p.add_argument("--tag", default="run")
args = p.parse_args()
torch.manual_seed(0)
dev = "cuda"
OUT = ROOT / "results" / "exp04"
OUT.mkdir(parents=True, exist_ok=True)

cfg = S.KeckConfig()
sysd = S.build(cfg)
pupil = torch.tensor(sysd["pupil"], device=dev)
dm = DM(torch.tensor(sysd["ifs"], device=dev), torch.tensor(sysd["m2c"], device=dev), cfg.n_pupil)
n_modes = dm.surfaces.shape[0]
scfg = FPSensorConfig(defocus_rad=args.defocus, photons=args.photons)
sensor = FocalPlaneSensor(scfg, pupil, cfg.grid_m).to(dev)
K = args.K
var = torch.tensor(sysd["kl_variance"], device=dev, dtype=torch.float32)
scale = var.sqrt() * 500 / (2 * torch.pi)  # open-loop nm rms per mode at 0.6"
net = FPNet(K, n_modes, cond=(K - 1) * n_modes).to(dev)


class Store:
    """All trajectories on the GPU, concatenated along the atmosphere axis."""

    def __init__(self):
        self.frames = self.applied = self.truth = None

    def add(self, tr):
        # pinned CPU memory: the GPU also holds the atmospheres and the network
        f, a, y = tr.frames.cpu(), tr.applied.cpu(), tr.truth.cpu()
        if self.frames is None:
            self.frames, self.applied, self.truth = f, a, y
        else:
            self.frames = torch.cat([self.frames, f], 1)
            self.applied = torch.cat([self.applied, a], 1)
            self.truth = torch.cat([self.truth, y], 1)

    def __len__(self):
        return self.frames.shape[0] * self.frames.shape[1]


def make_batch(store, n, gen, photon_range=(0.3, 3.0)):
    t_len, b_len = store.frames.shape[:2]
    t = torch.randint(K - 1, t_len, (n,), generator=gen)
    b = torch.randint(0, b_len, (n,), generator=gen)
    tk = t[:, None] + torch.arange(-K + 1, 1)[None]  # (n, K)
    clean = store.frames[tk, b[:, None]].to(dev, non_blocking=True)  # (n, K, npix, npix)
    a = store.applied[tk, b[:, None]].to(dev)  # (n, K, n_modes) metres
    c = ((a[:, -1:] - a[:, :-1]).reshape(n, -1) * 1e9) / scale.repeat(K - 1)
    ph = log_uniform(n, *photon_range, dev)[:, None, None, None]
    e = noisy_frames(clean, scfg, args.photons, ph)
    x = torch.stack([sensor.preprocess(e[:, k])[:, 0] for k in range(K)], 1)
    return x, c, store.truth[t, b].to(dev)


def train(trajs, steps, lr=1e-3):
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    gen = torch.Generator().manual_seed(len(trajs))
    net.train()
    t0 = time.perf_counter()
    for step in range(steps):
        x, c, y = make_batch(trajs, args.batch, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = net(x, c).float() * scale
        loss = (((pred - y) ** 2).sum(-1) / ((y**2).sum(-1) + 300.0)).mean()  # 17 nm rms floor
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 1000 == 0 or step == steps - 1:
            print(f"  step {step:5d} loss {loss.item():.4f} [{time.perf_counter() - t0:.0f} s]", flush=True)
    net.eval()


class NetRecon:
    """Closed-loop reconstructor: keeps its own frame history (noisy, as an RTC would)."""

    def reset(self, b):
        self.frames = []

    def estimate(self, e, hist):
        self.frames.append(e)
        self.frames = self.frames[-K:]
        while len(self.frames) < K:
            self.frames.insert(0, e)
        h = list(hist)[-K:]
        while len(h) < K:
            h.insert(0, h[0])
        a = torch.stack(h, 1)  # (B, K, n_modes)
        c = ((a[:, -1:] - a[:, :-1]).reshape(a.shape[0], -1) * 1e9) / scale.repeat(K - 1)
        x = torch.stack([sensor.preprocess(f)[:, 0] for f in self.frames], 1)
        with torch.no_grad():
            return net(x, c) * scale * 1e-9

    def __call__(self, residual, k, hist):  # run_loop interface
        return self.estimate(sensor.frame(residual), hist)

    def behaviour(self, noisy, k, hist):  # collect interface
        return self.estimate(noisy, hist)


def evaluate(label):
    sci = h_band_science(pupil, cfg.grid_m)
    turb = Turbulence(cfg, batch=4, seed=100, seeing=0.6)
    r = run_loop(turb, dm, NetRecon(), sci, pupil, 1000, args.loop_steps, gain=args.gain, delay=2, settle=500)
    traj = r.strehl_se.mean(1)
    print(
        f"[{label}] closed loop: LE Strehl {r.strehl_le.mean():.3f} (+-{r.strehl_le.std():.3f}), "
        f"residual {r.residual_nm[500:].mean():.0f} nm; SE every 100: {[round(float(v), 2) for v in traj[::100]]}",
        flush=True,
    )
    return dict(label=label, strehl_le=r.strehl_le.tolist(), se_traj=traj[::10].tolist(),
                residual_nm=float(r.residual_nm[500:].mean()))


report = dict(args=vars(args), rounds=[])
trajs = Store()
t0 = time.perf_counter()
for i in range(args.collections):
    turb = Turbulence(cfg, batch=args.atm, seed=1000 + 100 * i, seeing=0.6)
    trajs.add(collect(turb, dm, pupil, sensor, args.collect_steps, gains=(0.2, 0.6),
                      est_noise=(0.0, 1.0), dither_nm=args.dither, seed=i))
print(f"collected {len(trajs)} samples [{time.perf_counter() - t0:.0f} s]", flush=True)
train(trajs, args.train_steps)
report["rounds"].append(evaluate("round 0 (expert data)"))
for r_i in range(args.dagger):
    rec = NetRecon()
    rec.reset(args.atm)
    for j in range(2):
        turb = Turbulence(cfg, batch=args.atm, seed=5000 + 100 * r_i + j, seeing=0.6)
        trajs.add(collect(turb, dm, pupil, sensor, args.collect_steps, behaviour=rec.behaviour,
                             gains=(0.3, 0.5), dither_nm=args.dither, seed=100 + r_i))
        rec.reset(args.atm)
    train(trajs, args.dagger_steps, lr=3e-4)
    report["rounds"].append(evaluate(f"round {r_i + 1} (DAgger)"))
torch.save(net.state_dict(), OUT / f"{args.tag}.pt")
(OUT / f"{args.tag}.json").write_text(json.dumps(report, indent=1))
