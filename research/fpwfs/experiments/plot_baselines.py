"""Figures for exp03 (SH baseline) and exp05/06 (information content of one frame)."""

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fpsim import plotting as P  # noqa: E402

R = ROOT / "results"
ideal = json.loads((R / "exp01" / "ideal.json").read_text())
ideal_h = np.mean([r["strehl"] for r in ideal["rows"] if r["band"] == "H" and r["rate"] == 1000])

# ---- exp03: SH baseline ---------------------------------------------------------
sh = json.loads((R / "exp03" / "mag_sweep.json").read_text())["results"]
mags = [r["mag"] for r in sh]
s = [np.mean(r["H"]["strehl_le"]) for r in sh]
sd = [np.std(r["H"]["strehl_le"]) for r in sh]
fig, ax = P.plt.subplots(figsize=(6.4, 4.0))
ax.errorbar(mags, s, yerr=sd, marker="o", color=P.SERIES[0], capsize=3, label="legacy Keck II SH (20x20, OCAM2K)")
for m, v, r in zip(mags, s, sh):
    ax.annotate(f"{r['photons_per_frame']:.0f} ph", (m, v), xytext=(6, -12), textcoords="offset points", fontsize=7.5, color=P.INK2)
ax.axhline(ideal_h, color=P.INK2, lw=1, ls="--")
ax.annotate(f"ideal WFS {ideal_h:.2f}", (mags[0], ideal_h), xytext=(0, 4), textcoords="offset points", fontsize=8, color=P.INK2)
ax.set_xlabel("guide star V magnitude")
ax.set_ylabel("long-exposure H Strehl")
ax.set_ylim(0, 1)
ax.set_title("The bar to beat: legacy Keck II Shack-Hartmann at 1 kHz")
ax.legend(loc="lower left")
P.save(fig, "exp03_sh_baseline", "sh_strehl_vs_magnitude",
       """Closed loop with the **Shack-Hartmann baseline**: makewfs 20x20 SH (4x4 OCAM2K pixels per lenslet,
0.8"/px, 0.55-0.85 um), getframes OCAM2K noise (EM gain 600, Keck-measured), centre-of-gravity slopes,
modal least-squares reconstructor on the same 300 DM-KL modes, leaky integrator (gain 0.5, leak 0.99),
1 kHz, 2-frame delay, 0.6" seeing, 4 atmospheres (error bars = their spread). Photons per frame
(labels) follow the makewfs HAKA budget validated against Keck RTC data (52 M ph/s at V = 10.16).
**What to look at:** the gap between the points and the dashed ideal-sensor line is everything the SH
loses (aliasing, centroiding of undersampled spots, noise). At V <= 10 the SH gives ~0.70 in H
(~0.76 in K by Marechal; Keck delivers ~0.56 in K on sky, the model has no NCPA/vibration).
Gain is not yet optimised per magnitude; faint-star points will improve with a lower gain and the
Keck camera-mode frame rate.""",
       title="exp03: Shack-Hartmann baseline (legacy Keck II)")
P.note("exp03_sh_baseline", """The SH baseline that the focal-plane system must match (goal G2). Same
atmosphere, DM, modes, delay and controller as every other experiment; only the sensor differs.""")

# ---- exp05/06: information --------------------------------------------------------
fisher = json.loads((R / "exp06" / "fisher.json").read_text())
probe = json.loads((R / "exp05" / "linear_probe.json").read_text())
fig, ax = P.plt.subplots(figsize=(6.8, 4.3))
for i, d in enumerate((0.0, 0.5, 1.0, 2.0)):
    rows = [r for r in fisher if r["defocus"] == d and r["fitting"]]
    ph = [r["photons"] for r in rows]
    rel = [r["crb_nm"] / r["prior_nm"] for r in rows]
    label = "in focus" if d == 0 else f"defocus {d:g} rad"
    ax.plot(ph, rel, color=P.SERIES[i], label=f"{label}: information bound")
    pts = [(r["photons"], r["err_nm"] / r["true_nm"]) for r in probe if r["defocus"] == d and r["alpha"][0] == 0.02]
    ax.plot(*zip(*pts), ls="none", marker="s", ms=7, mfc="none", mew=1.6, color=P.SERIES[i])
ax.plot([1e4], [0.50], ls="none", marker="*", ms=12, color=P.SERIES[2])
ax.annotate("CNN, defocus 1 rad\n(generated residuals)", (1e4, 0.50), xytext=(8, 4), textcoords="offset points", fontsize=7.5, color=P.INK2)
ax.set_xscale("log")
ax.set_xlabel("photons per frame")
ax.set_ylabel("rms estimate error / rms residual")
ax.set_ylim(0, 1)
ax.set_title("How much one frame can tell: bound vs linear map vs CNN")
h, lab = ax.get_legend_handles_labels()
h.append(P.matplotlib.lines.Line2D([], [], ls="none", marker="s", mfc="none", mew=1.6, color=P.INK2))
lab.append("linear map (ridge), same colour = same defocus")
ax.legend(h, lab, loc="lower left", fontsize=7.5)
P.save(fig, "exp06_information", "error_vs_photons",
       """Well-corrected loop (DM-space residual ~30-37 nm rms, plus the real ~100 nm DM fitting error).
**Lines:** Bayesian Cramer-Rao bound, the best any estimator can do from one H-band frame
(Fisher information by autodiff through the imager, Poisson + 0.6 e- read noise, prior = residual
statistics). **Open squares:** a linear (ridge) map trained on 24k frames. **Star:** the exp02 CNN.
**What to look at:** (1) in focus the bound stays high until very bright (even modes are only seen
through the random fitting error), any defocus 0.25-2 rad brings it down ~2x; (2) at 1e4 photons
both the linear map and the CNN are within ~1.5x of the bound, so that regime is photon-limited;
(3) at 1e6 photons the linear map stalls at ~0.28 while the bound is 0.10: the sensor response
changes by ~28% from frame to frame with the fitting error, which a single linear map cannot follow.
That is the case for a non-linear estimator. Note 11.5 nm (bound at 1e4 photons) is already small
next to the 105 nm fitting floor, so photons are not what limits the loop.""",
       title="exp05/06: information content of one focal-plane frame")
P.note("exp06_information", """Fundamental limits of single-frame focal-plane sensing for the Keck model,
from the Fisher information, compared with a linear map (exp05) and the exp02 CNN.""")
