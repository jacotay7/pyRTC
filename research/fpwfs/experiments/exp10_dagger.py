"""Single-frame focal-plane CNN that can HOLD the Keck loop: dithered data + DAgger.

exp09 showed a CNN beats the fitting-error coupling on closed-loop states
(error/residual 0.62) but the loop still collapses after a hand-over: the
network shrinks ~half the modes toward zero, those drift, and the states leave
the training distribution. Two fixes:

1. widen the data: known random modal DM offsets (up to `--dither-max` x the
   closed-loop per-mode std) so every mode carries signal; realistic (DM dither);
2. DAgger: start from a converged loop, fly beta * ideal + (1 - beta) * network,
   relabel the visited states, retrain; beta decays to 0 over the rounds.
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
p.add_argument("--collections", type=int, default=4)
p.add_argument("--dither-max", type=float, default=2.0)
p.add_argument("--steps", type=int, default=12000)
p.add_argument("--rounds", type=int, default=3)
p.add_argument("--round-steps", type=int, default=5000)
p.add_argument("--batch", type=int, default=128)
p.add_argument("--gain", type=float, default=0.3)
p.add_argument("--tag", default="run")
p.add_argument("--n-control", type=int, default=300, help="controlled modes (others left uncorrected)")
p.add_argument("--seeing-range", type=float, nargs=2, default=[0.6, 0.6], help="per-collection seeing (arcsec)")
p.add_argument("--wind-range", type=float, nargs=2, default=[1.0, 1.0], help="per-collection wind scale")
p.add_argument("--init", default=None, help="warm-start weights (with _scale)")
args = p.parse_args()
torch.manual_seed(0)
dev = "cuda"
OUT = ROOT / "results" / "exp10"
OUT.mkdir(parents=True, exist_ok=True)
cfg = S.KeckConfig()
sysd = S.build(cfg)
pupil = torch.tensor(sysd["pupil"], device=dev)
dm = DM(torch.tensor(sysd["ifs"], device=dev), torch.tensor(sysd["m2c"], device=dev), cfg.n_pupil)
proj = ModalProjector(dm, pupil)
n_modes = dm.surfaces.shape[0]
scfg = FPSensorConfig(defocus_rad=args.defocus, photons=args.photons, wavelength=args.wavelength, npix=args.npix)
sensor = FocalPlaneSensor(scfg, pupil, cfg.grid_m).to(dev)
sci = h_band_science(pupil, cfg.grid_m)
NC = args.n_control
mask = torch.zeros(n_modes, device=dev)
mask[:NC] = 1


_rng = torch.Generator().manual_seed(1234)


def atmosphere(batch, seed):
    """Atmospheres for one collection: seeing and wind drawn from the configured ranges."""
    u = torch.rand(2, generator=_rng).tolist()
    seeing = args.seeing_range[0] + u[0] * (args.seeing_range[1] - args.seeing_range[0])
    wind = args.wind_range[0] + u[1] * (args.wind_range[1] - args.wind_range[0])
    return Turbulence(cfg, batch=batch, seed=seed, seeing=seeing, wind_scale=wind)


class Data:
    def __init__(self):
        self.x, self.y = [], []

    def add(self, tr, max_nm=None):
        x, y = tr.frames.flatten(0, 1), tr.truth.flatten(0, 1)[:, :NC]
        if max_nm is not None:  # keep states the loop should be in; runaway frames teach nothing useful
            keep = y.pow(2).sum(-1).sqrt() < max_nm
            print(f"  kept {int(keep.sum())}/{len(keep)} states below {max_nm:.0f} nm", flush=True)
            x, y = x[keep], y[keep]
        self.x.append(x)
        self.y.append(y)
        self.X, self.Y = torch.cat(self.x), torch.cat(self.y)


def batch(X, Y, idx, photon_range=(0.5, 2.0)):
    clean = X[idx].to(dev)
    ph = log_uniform(len(idx), *photon_range, dev)[:, None, None]
    return sensor.preprocess(noisy_frames(clean, scfg, args.photons, ph)), Y[idx].to(dev)


def train(data, steps, lr):
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    net.train()
    t0 = time.perf_counter()
    for step in range(steps):
        idx = torch.randint(0, len(data.X), (args.batch,))
        x, y = batch(data.X, data.Y, idx)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = net(x).float() * scale
        # per-sample relative error: a few runaway states cannot dominate the batch
        loss = ((pred - y).pow(2).sum(-1) / (y.pow(2).sum(-1) + 100.0)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 2000 == 0 or step == steps - 1:
            print(f"  step {step:5d} loss {loss.item():.3f} [{time.perf_counter() - t0:.0f} s]", flush=True)
    net.eval()


def policy(noisy, k, hist):
    with torch.no_grad():
        return torch.nn.functional.pad(net(sensor.preprocess(noisy)) * scale, (0, n_modes - NC)) * 1e-9


def evaluate(label):
    # (a) error/residual on undithered closed-loop test states
    with torch.no_grad():
        e2, t2, pe, ty = [], [], [], []
        for i in range(0, len(test.X), 256):
            idx = torch.arange(i, min(i + 256, len(test.X)))
            x, y = batch(test.X, test.Y, idx, photon_range=(1.0, 1.0001))
            pred = net(x) * scale
            e2.append((pred - y).pow(2).sum(-1))
            t2.append(y.pow(2).sum(-1))
            pe.append(pred)
            ty.append(y)
    pe, ty = torch.cat(pe), torch.cat(ty)
    k_mode = (pe * ty).sum(0) / (ty * ty).sum(0)
    ratio = (torch.cat(e2).mean() / torch.cat(t2).mean()).sqrt().item()

    # (b) hand-over: ideal for 300 frames, then the network alone
    def recon(residual, k, hist):
        if k < 300:
            return proj(residual.mean(0)) * mask
        return policy(sensor.frame(residual), k, hist)

    r = run_loop(Turbulence(cfg, batch=12, seed=100, seeing=0.6), dm, recon, sci, pupil, 1000, 1500,
                 gain=args.gain, delay=2, settle=600)
    held = int((r.strehl_le > 0.5).sum())
    traj = r.strehl_se.mean(1)
    print(f"[{label}] test ratio {ratio:.2f}, k median {k_mode.median():.2f} (k<0.3: {(k_mode < 0.3).sum().item()}); "
          f"HANDOVER held {held}/12, LE Strehl median {r.strehl_le.median():.3f} mean {r.strehl_le.mean():.3f}; "
          f"SE every 100 {[round(float(v), 2) for v in traj[::100]]}", flush=True)
    return dict(label=label, ratio=ratio, k=k_mode.tolist(), se_traj=traj.tolist(), held=held,
                strehl_le=r.strehl_le.tolist(), residual_nm=float(r.residual_nm[600:].mean()))


t0 = time.perf_counter()
# pilot: per-mode closed-loop residual std (sets the dither scale and output scaling)
pilot = collect(Turbulence(cfg, batch=8, seed=1999, seeing=0.6), dm, pupil, sensor, 700, gains=(0.3, 0.5),
                est_noise=(0.0, 0.3), seed=99, keep_from=300, mode_mask=mask)
cl_std = pilot.truth.flatten(0, 1).std(0).to(dev) * mask  # dither only controlled modes
test = Data()
test.add(pilot)
data = Data()
for i in range(args.collections):
    data.add(collect(atmosphere(args.atm, 3000 + 100 * i), dm, pupil, sensor,
                     args.collect_steps, gains=(0.25, 0.6), est_noise=(0.0, 0.6), seed=i, keep_from=300,
                     dither_modes_nm=cl_std, dither_max=args.dither_max, mode_mask=mask))
scale = data.Y.std(0).to(dev)
print(f"{len(data.X)} dithered closed-loop states, residual {data.Y.pow(2).sum(-1).mean().sqrt():.1f} nm "
      f"(undithered test {test.Y.pow(2).sum(-1).mean().sqrt():.1f} nm) [{time.perf_counter() - t0:.0f} s]", flush=True)
net = FPNet(1, NC, npix=args.npix).to(dev)
if args.init:
    st = torch.load(args.init)
    scale = st.pop("_scale").to(dev)  # keep the warm-started network's output scaling
    net.load_state_dict(st)
train(data, args.steps, 3e-4 if args.init else 1e-3)
report = dict(args=vars(args), rounds=[evaluate("round 0: dithered expert data")])
torch.save({**net.state_dict(), "_scale": scale.cpu()}, OUT / f"{args.tag}_r0.pt")
for r_i in range(args.rounds):
    beta = max(0.0, 0.5 * (1 - r_i / max(args.rounds - 1, 1)))
    for j in range(2):
        data.add(collect(atmosphere(args.atm, 7000 + 100 * r_i + j), dm, pupil, sensor,
                         1500, behaviour=policy, beta=beta, ideal_frames=300, gains=(args.gain, args.gain),
                         seed=500 + 10 * r_i + j, keep_from=300, dither_modes_nm=cl_std, dither_max=0.5,
                         mode_mask=mask), max_nm=6 * float(test.Y.pow(2).sum(-1).mean().sqrt()))
    print(f"DAgger round {r_i + 1}: beta {beta:.2f}, {len(data.X)} states", flush=True)
    train(data, args.round_steps, 3e-4)
    report["rounds"].append(evaluate(f"round {r_i + 1}: DAgger beta {beta:.2f}"))
    torch.save({**net.state_dict(), "_scale": scale.cpu()}, OUT / f"{args.tag}_r{r_i + 1}.pt")
    (OUT / f"{args.tag}.json").write_text(json.dumps(report))
