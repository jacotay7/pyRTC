"""R0: ideal-WFS closed loop on legacy Keck II (upper bound, no sensor).

Writes results/exp01/ideal.json and figures to plots/exp01_ideal_loop/.
"""

import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fpsim import plotting as P  # noqa: E402
from fpsim import system as S  # noqa: E402
from fpsim.atmos import Turbulence  # noqa: E402
from fpsim.loop import DM, ModalProjector, h_band_science, k_band_science, marechal, pupil_rms, run_loop  # noqa: E402

dev = "cuda"
EXP = "exp01_ideal_loop"
OUT = ROOT / "results" / "exp01"
OUT.mkdir(parents=True, exist_ok=True)
cfg = S.KeckConfig()
sysd = S.build(cfg)
pupil = torch.tensor(sysd["pupil"], device=dev)
dm = DM(torch.tensor(sysd["ifs"], device=dev), torch.tensor(sysd["m2c"], device=dev), cfg.n_pupil)
proj = ModalProjector(dm, pupil)


def ideal(residual, k, hist):
    return proj(residual.mean(0))


rates = [250, 500, 1000, 2000]
rows = []
sci = {"H": h_band_science(pupil, cfg.grid_m), "K": k_band_science(pupil, cfg.grid_m)}
for band, imager in sci.items():
    for rate in rates:
        turb = Turbulence(cfg, batch=4, seed=10, seeing=0.6)
        r = run_loop(turb, dm, ideal, imager, pupil, rate, steps=1200, gain=0.5, delay=2, settle=400)
        rows.append(dict(band=band, rate=rate, strehl=r.strehl_le.tolist(), residual_nm=float(r.residual_nm[400:].mean())))
        print(band, rate, f"{r.strehl_le.mean():.3f}", f"{r.residual_nm[400:].mean():.0f} nm", flush=True)
turb = Turbulence(cfg, batch=4, seed=10, seeing=0.6)
opd = turb.step(1e-3)
fit_nm = float((pupil_rms(opd - dm.opd(proj(opd)), pupil) * 1e9).mean())
ol_nm = float((pupil_rms(opd, pupil) * 1e9).mean())
(OUT / "ideal.json").write_text(json.dumps(dict(rows=rows, fitting_nm=fit_nm, open_loop_nm=ol_nm), indent=1))

# ---- figures ------------------------------------------------------------------
fig, ax = P.plt.subplots(figsize=(6.4, 4.0))
for i, band in enumerate(("H", "K")):
    rr = [r for r in rows if r["band"] == band]
    s = np.array([np.mean(r["strehl"]) for r in rr])
    ax.plot(rates, s, marker="o", color=P.SERIES[i], label=f"{band} band, ideal WFS")
    fit_sr = float(marechal(torch.tensor(fit_nm), 1650 if band == "H" else 2200))
    ax.axhline(fit_sr, color=P.SERIES[i], lw=1, ls="--")
    ax.annotate(f"{band}: DM fitting limit {fit_sr:.2f}", (rates[0], fit_sr), xytext=(0, 4),
                textcoords="offset points", fontsize=8, color=P.INK2)
ax.set_xscale("log")
ax.xaxis.set_minor_formatter(P.matplotlib.ticker.NullFormatter())
ax.set_xticks(rates, [str(r) for r in rates])
ax.set_xlabel("loop frame rate [Hz] (2-frame delay)")
ax.set_ylabel("long-exposure Strehl")
ax.set_ylim(0.6, 1.0)
ax.set_title("Upper bound: a perfect sensor on the legacy Keck II DM")
ax.legend(loc="lower right")
P.save(
    fig, EXP, "strehl_vs_rate",
    f"""Closed loop with an **ideal wavefront sensor** (exact least-squares projection of the
residual onto the 300 DM-KL modes, no noise), 2-frame delay, integrator gain 0.5, pyturb
`keck` profile (KAON 303) at 0.6" seeing, 4 independent atmospheres. Dashed lines: Strehl
allowed by DM fitting error alone ({fit_nm:.0f} nm rms; open loop is {ol_nm:.0f} nm).
**What to look at:** above ~1 kHz the loop is fitting-limited (curves flatten onto the
dashed lines), so at 1 kHz any extra error from a real WFS shows up almost one-for-one.
This is the ceiling every reconstructor in later experiments is compared against.""",
    title="exp01: ideal-WFS upper bound (legacy Keck II)",
)
P.note(EXP, """Ideal-sensor closed loop for the legacy Keck II model (36-segment pupil,
349-actuator DM with 300 KL modes). Sets the performance ceiling: at 1 kHz,
H-band Strehl ~0.85 and K-band ~0.91, almost all of it DM fitting error.""")
