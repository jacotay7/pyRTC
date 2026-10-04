"""Figures for exp02 (single-frame focal-plane NN: in focus vs fixed defocus)."""

import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fpsim import plotting as P  # noqa: E402
from fpsim import system as S  # noqa: E402
from fpsim.data import ResidualGenerator  # noqa: E402
from fpsim.loop import DM  # noqa: E402
from fpsim.sensor import FocalPlaneSensor, FPSensorConfig  # noqa: E402

EXP = "exp02_single_frame"
RES = ROOT / "results" / "exp02"
runs = {}
for tag, label in (("defocus_0.0", "in focus"), ("defocus_1.0", "defocus 1 rad rms")):
    f = RES / f"{tag}.json"
    if f.exists():
        runs[label] = json.loads(f.read_text())
ideal = json.loads((ROOT / "results" / "exp01" / "ideal.json").read_text())
ideal_h = np.mean([r["strehl"] for r in ideal["rows"] if r["band"] == "H" and r["rate"] == 1000])

# 1. estimation error vs correction level ------------------------------------
fig, ax = P.plt.subplots(figsize=(6.4, 4.0))
for i, (label, rep) in enumerate(runs.items()):
    t = [lv["true_nm"] for lv in rep["levels"]]
    e = [lv["err_nm"] for lv in rep["levels"]]
    ax.plot(t, np.array(e) / np.array(t), marker="o", color=P.SERIES[i], label=label)
ax.axhline(1.0, color=P.MUTED, lw=1, ls=":")
ax.annotate("error = signal (no information)", (40, 1.0), xytext=(0, 4), textcoords="offset points", fontsize=8, color=P.INK2)
ax.set_xscale("log")
ax.set_xlabel("true residual in the DM modes [nm rms]  (left: well corrected, right: open loop)")
ax.set_ylabel("estimate error / true residual")
ax.set_ylim(0, 1.15)
ax.set_title("How well one H-band frame determines the residual")
ax.legend()
P.save(
    fig, EXP, "relative_error_vs_level",
    """Each point: 2048 generated residuals (DM-fitting error from real pyturb screens + random
DM-space part at a given correction level), one noisy H-band frame (1e4 photons, 0.6 e-
read noise), estimate from a CNN trained on the same distribution. y = rms estimate error /
rms true residual (lower is better, 1 = useless). **What to look at:** in focus the error
floors near ~0.7 at every level: a single in-focus image cannot tell the sign of even modes
(focus, astigmatism, ...), so about half the variance is unrecoverable. A fixed defocus
breaks the ambiguity; how far it lowers the curve, and at which levels, is the result.""",
)

# 2. closed-loop trajectories ------------------------------------------------
fig, ax = P.plt.subplots(figsize=(6.4, 4.0))
for i, (label, rep) in enumerate(runs.items()):
    traj = np.array(rep["loop"]["se_traj"])
    ax.plot(np.arange(len(traj)) * 10, traj, color=P.SERIES[i], label=f"{label}: LE Strehl {np.mean(rep['loop']['strehl_le']):.2f}")
ax.axhline(ideal_h, color=P.INK2, lw=1, ls="--")
ax.annotate(f"ideal WFS {ideal_h:.2f}", (0, ideal_h), xytext=(4, 4), textcoords="offset points", fontsize=8, color=P.INK2)
ax.set_xlabel("frame (1 kHz, loop closed at frame 0 from open loop)")
ax.set_ylabel("short-exposure H Strehl (mean of 4 atmospheres)")
ax.set_ylim(0, 1)
ax.set_title("Closing the Keck loop with only the focal-plane camera")
ax.legend(loc="center right")
P.save(
    fig, EXP, "closed_loop_trajectory",
    """The trained single-frame network is the *only* wavefront sensor: loop closed at
frame 0 from seeing-limited conditions (0.6", pyturb keck profile), 1 kHz, 2-frame delay,
integrator gain 0.4, 4 independent atmospheres (curves are their mean). Dashed: the
ideal-sensor ceiling from exp01. **What to look at:** whether the loop converges at all
from open loop (bootstrap, goal G3), how fast, and the gap to the dashed line (goal G2).""",
)

# 3. what the sensor sees ----------------------------------------------------
cfg = S.KeckConfig()
sysd = S.build(cfg)
dev = "cuda"
pupil = torch.tensor(sysd["pupil"], device=dev)
dm = DM(torch.tensor(sysd["ifs"], device=dev), torch.tensor(sysd["m2c"], device=dev), cfg.n_pupil)
gen = ResidualGenerator(cfg, sysd, dm, pupil, bank_size=256)
levels = [0.03, 0.1, 0.3, 1.0]
fig, axes = P.plt.subplots(2, 4, figsize=(9.6, 5.0))
for row, dfc in enumerate((0.0, 1.0)):
    sensor = FocalPlaneSensor(FPSensorConfig(defocus_rad=dfc), pupil, cfg.grid_m).to(dev)
    for col, a in enumerate(levels):
        gen.gen.manual_seed(5)
        opd, y = gen.sample(1, alpha=(a, a * 1.0001), beta=(1.0, 1.0001), seeing_scale=(1.0, 1.0001))
        img = sensor.frame(opd)[0].cpu().numpy()
        ax = axes[row, col]
        ax.imshow(np.sqrt(np.clip(img, 0, None)), cmap="magma", origin="lower")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        if row == 0:
            ax.set_title(f"residual {float(y.norm()):.0f} nm rms", fontsize=9)
    axes[row, 0].set_ylabel("in focus" if dfc == 0 else "defocus 1 rad", fontsize=9)
fig.suptitle("The focal-plane WFS frame (sqrt stretch, 64x64 px, Nyquist at 1.65 um)", fontsize=11, fontweight="bold")
P.save(
    fig, EXP, "sensor_frames",
    """Same residual wavefront per column, imaged in focus (top) and with a fixed 1 rad rms
defocus (bottom), 1e4 photons with photon + read noise. Left to right: well corrected ->
open loop. The 64x64 frame spans +-16 lambda/D, enough for the 20x20 DM's +-10 lambda/D
control radius. **What to look at:** the defocused frame spreads light (lower peak SNR)
but its asymmetric structure is what tells the network the sign of even modes.""",
)
P.note(EXP, """Can one focal-plane frame drive the Keck loop? A CNN maps a single H-band frame to
the 300 DM-mode residual. In focus vs with a fixed 1 rad rms defocus (allowed: no new
hardware). Trained on generated residuals, then used as the only WFS in closed loop.""",
       title="exp02: single-frame focal-plane reconstructor")
