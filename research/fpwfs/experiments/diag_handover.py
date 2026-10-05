"""Trace a network hand-over frame by frame: which mode band runs away first?"""

import argparse
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fpsim import system as S  # noqa: E402
from fpsim.atmos import Turbulence  # noqa: E402
from fpsim.loop import DM, ModalProjector, h_band_science, run_loop  # noqa: E402
from fpsim.nets import FPNet  # noqa: E402
from fpsim.sensor import FocalPlaneSensor, FPSensorConfig  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("weights")
p.add_argument("--npix", type=int, default=64)
p.add_argument("--wavelength", type=float, default=1.65e-6)
p.add_argument("--gain", type=float, default=0.3)
p.add_argument("--n-control", type=int, default=300)
p.add_argument("--width", type=int, default=48)
p.add_argument("--stem-stride", type=int, default=1)
p.add_argument("--tag", default="handover_trace")
p.add_argument("--steps", type=int, default=520)
p.add_argument("--seed", type=int, default=100)
p.add_argument("--rate", type=float, default=1000.0)
p.add_argument("--photons", type=float, default=1e5)
p.add_argument("--batch", type=int, default=4)
args = p.parse_args()
dev = "cuda"
cfg = S.KeckConfig()
sysd = S.build(cfg)
pupil = torch.tensor(sysd["pupil"], device=dev)
dm = DM(torch.tensor(sysd["ifs"], device=dev), torch.tensor(sysd["m2c"], device=dev), cfg.n_pupil)
proj = ModalProjector(dm, pupil)
sensor = FocalPlaneSensor(FPSensorConfig(defocus_rad=1.0, photons=args.photons, wavelength=args.wavelength, npix=args.npix),
                          pupil, cfg.grid_m).to(dev)
NC = args.n_control
mask = torch.zeros(300, device=dev)
mask[:NC] = 1
net = FPNet(1, NC, npix=args.npix, width=args.width, stem_stride=args.stem_stride).to(dev)
state = torch.load(args.weights)
if "_scale" in state:
    scale = state.pop("_scale").to(dev)
else:  # re-derive the per-mode output scale with exp10's data recipe (matches to a few %)
    from fpsim.collect import collect

    pilot = collect(Turbulence(cfg, batch=8, seed=1999, seeing=0.6), dm, pupil, sensor, 700, gains=(0.3, 0.5),
                    est_noise=(0.0, 0.3), seed=99, keep_from=300, mode_mask=mask)
    cl_std = pilot.truth.flatten(0, 1).std(0).to(dev) * mask
    tr = collect(Turbulence(cfg, batch=16, seed=3000, seeing=0.6), dm, pupil, sensor, 1500, gains=(0.25, 0.6),
                 est_noise=(0.0, 0.6), seed=0, keep_from=300, dither_modes_nm=cl_std, dither_max=2.0, mode_mask=mask)
    scale = tr.truth.flatten(0, 1).std(0).to(dev)[:NC]
net.load_state_dict(state)
net.eval()
bands = ((0, 2), (2, 20), (20, 60), (60, 120), (120, 200), (200, 300))
rows = []


def recon(residual, k, hist):
    t = proj(residual.mean(0)) * 1e9
    with torch.no_grad():
        est = torch.nn.functional.pad(net(sensor.preprocess(sensor.frame(residual))) * scale, (0, 300 - NC))
    if k >= 250 and k % 25 == 0:
        bt = [t[:, a:b].norm(dim=-1).mean().item() for a, b in bands]
        be = [(est - t)[:, a:b].norm(dim=-1).mean().item() for a, b in bands]
        cos = ((est * t).sum(-1) / (est.norm(dim=-1) * t.norm(dim=-1))).mean().item()
        rows.append((k, bt, be, cos))
    return (t * mask if k < 300 else est) * 1e-9


res = run_loop(Turbulence(cfg, batch=args.batch, seed=args.seed, seeing=0.6), dm, recon, h_band_science(pupil, cfg.grid_m), pupil,
               args.rate, args.steps, gain=args.gain, delay=2, settle=0)
print("frame | residual per band " + str(bands) + " | error per band | cos")
for k, bt, be, cos in rows:
    print(f"{k:4d} | {' '.join(f'{v:6.1f}' for v in bt)} | {' '.join(f'{v:6.1f}' for v in be)} | {cos:5.2f}")

import json  # noqa: E402

from fpsim import plotting as P  # noqa: E402

out = ROOT / "results" / "diag"
out.mkdir(parents=True, exist_ok=True)
(out / f"{args.tag}.json").write_text(json.dumps(dict(rows=rows, bands=bands, se=res.strehl_se.mean(1).tolist())))
fig, (ax1, ax2) = P.plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True, height_ratios=[2, 1])
ks = [r[0] for r in rows]
for i, (a, b) in enumerate(bands):
    ax1.plot(ks, [r[1][i] for r in rows], color=P.SERIES[i], label=f"modes {a}-{b - 1}")
ax1.axvline(300, color=P.MUTED, lw=1, ls=":")
ax1.annotate("network takes over", (300, 1), xytext=(4, 4), textcoords="offset points", fontsize=8, color=P.INK2)
ax1.set_yscale("log")
ax1.set_ylabel("true residual per mode band [nm rms]")
ax1.set_title(f"Hand-over trace: which modes run away first ({NC} modes controlled)")
ax1.legend(fontsize=7.5, ncol=2)
se = res.strehl_se.mean(1)
ax2.plot(range(250, len(se)), se[250:], color=P.INK2)
ax2.axvline(300, color=P.MUTED, lw=1, ls=":")
ax2.set_ylabel("H Strehl (SE)")
ax2.set_xlabel("frame (1 kHz)")
ax2.set_ylim(0, 1)
P.save(fig, "exp10_dagger", args.tag,
       f"""The ideal sensor flies the loop until frame 300, then the network (`{pathlib.Path(args.weights).name}`)
is the only sensor ({NC} modes controlled, gain {args.gain}, 1 kHz, 0.6"). Top: true residual in each DM-KL
mode band (mean of 4 atmospheres); bottom: short-exposure H Strehl. **What to look at:** which band leaves
its pre-hand-over level first. A band that climbs immediately is one the network cannot sense; once it
grows, the frames leave the training distribution and every band follows.""")
