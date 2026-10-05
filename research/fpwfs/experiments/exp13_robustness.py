"""Robustness of a trained focal-plane network holding the loop: leak, gain, seeds.

Hand-over (ideal sensor for 300 frames, then the network alone) on 12 fresh
atmospheres per setting; counts how many hold (LE Strehl > 0.5) and plots every
trajectory.
"""

import argparse
import json
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fpsim import plotting as P  # noqa: E402
from fpsim import system as S  # noqa: E402
from fpsim.atmos import Turbulence  # noqa: E402
from fpsim.loop import DM, ModalProjector, h_band_science, run_loop  # noqa: E402
from fpsim.nets import FPNet  # noqa: E402
from fpsim.sensor import FocalPlaneSensor, FPSensorConfig  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("weights")
p.add_argument("--n-control", type=int, default=120)
p.add_argument("--settings", default="1.0:0.3,0.99:0.3,0.99:0.2,0.995:0.3", help="leak:gain,...")
p.add_argument("--steps", type=int, default=2300)
p.add_argument("--seed", type=int, default=4000)
p.add_argument("--tag", default="robustness")
p.add_argument("--width", type=int, default=48)
p.add_argument("--stem-stride", type=int, default=1)
p.add_argument("--head", type=int, default=1024)
p.add_argument("--photons", type=float, default=1e5)
p.add_argument("--rate", type=float, default=1000.0, help="loop frame rate [Hz]")
args = p.parse_args()
dev = "cuda"
cfg = S.KeckConfig()
sysd = S.build(cfg)
pupil = torch.tensor(sysd["pupil"], device=dev)
dm = DM(torch.tensor(sysd["ifs"], device=dev), torch.tensor(sysd["m2c"], device=dev), cfg.n_pupil)
proj = ModalProjector(dm, pupil)
NC = args.n_control
sensor = FocalPlaneSensor(FPSensorConfig(defocus_rad=1.0, photons=args.photons), pupil, cfg.grid_m).to(dev)
st = torch.load(args.weights)
scale = st.pop("_scale").to(dev)
net = FPNet(1, NC, width=args.width, stem_stride=args.stem_stride, head=args.head).to(dev)
net.load_state_dict(st)
net.eval()
mask = torch.zeros(300, device=dev)
mask[:NC] = 1


def recon(residual, k, hist):
    if k < 300:
        return proj(residual.mean(0)) * mask
    with torch.no_grad():
        return torch.nn.functional.pad(net(sensor.preprocess(sensor.frame(residual))) * scale, (0, 300 - NC)) * 1e-9


settings = [tuple(float(v) for v in s.split(":")) for s in args.settings.split(",")]
fig, axes = P.plt.subplots(1, len(settings), figsize=(3.4 * len(settings), 3.6), sharey=True, squeeze=False)
axes = axes[0]
report = {"weights": args.weights}
for ax, (leak, gain) in zip(axes, settings):
    r = run_loop(Turbulence(cfg, batch=12, seed=args.seed, seeing=0.6), dm, recon, h_band_science(pupil, cfg.grid_m),
                 pupil, args.rate, args.steps, gain=gain, leak=leak, delay=2, settle=600)
    held = int((r.strehl_le > 0.5).sum())
    report[f"leak{leak}_gain{gain}"] = dict(held=held, strehl_le=r.strehl_le.tolist())
    print(f"leak {leak} gain {gain}: held {held}/12, LE Strehl median {r.strehl_le.median():.3f} "
          f"(held mean {r.strehl_le[r.strehl_le > 0.5].mean():.3f})", flush=True)
    for b in range(12):
        ax.plot(r.strehl_se[:, b], lw=0.7, color=P.SERIES[0] if r.strehl_le[b] > 0.5 else P.SERIES[7], alpha=0.8)
    ax.axvline(300, color=P.MUTED, lw=1, ls=":")
    ax.set_title(f"leak {leak:g}, gain {gain:g}: {held}/12 hold", fontsize=9.5)
    ax.set_xlabel("frame (1 kHz)")
    ax.set_ylim(0, 1)
axes[0].set_ylabel("short-exposure H Strehl")
out = ROOT / "results" / "exp13"
out.mkdir(parents=True, exist_ok=True)
(out / f"{args.tag}.json").write_text(json.dumps(report))
P.save(fig, "exp13_robustness", args.tag,
       f"""Network `{pathlib.Path(args.weights).name}` ({NC} modes controlled, single 1 rad-defocus H-band frame,
1e5 photons) holding the loop after the ideal sensor hands over at frame 300 (dotted), for 12 fresh
atmospheres (0.6", 1 kHz, 2-frame delay) per controller setting. Blue: held (LE Strehl > 0.5),
red: lost. **What to look at:** how many of the 12 hold and whether Keck's leaky integrator
(leak 0.99) or a lower gain removes the late divergences.""",
       title="exp13: does the focal-plane loop stay closed?")
