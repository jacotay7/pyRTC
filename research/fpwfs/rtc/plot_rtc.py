"""Figures for the pyRTC real-time demo (fpsim.plotting style -> plots/rtc_demo/)."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from fpsim import plotting as P  # noqa: E402

RES = HERE.parent / "results" / "rtc"

p = argparse.ArgumentParser()
p.add_argument("--main", default="hard_1khz", help="run tag shown in the Strehl figure")
p.add_argument("--offline", default="offline_ref", help="offline_ref.py output (stem in results/rtc)")
p.add_argument("--frozen", default="frozen", help="run tag with the loop never started")
p.add_argument("--latency-runs", default="soft_free,hard_free,hard_1khz",
               help="tags for the latency histogram")
args = p.parse_args()


def load(tag):
    path = RES / tag
    return dict(np.load(path / "frames.npz")), json.loads((path / "summary.json").read_text())


def rolling(x, n=20):
    """Running mean over the last n frames, ignoring frames without a Strehl (NaN)."""
    ok = np.isfinite(x)
    k = np.ones(n)
    num = np.convolve(np.where(ok, x, 0.0), k, mode="valid")
    den = np.convolve(ok.astype(float), k, mode="valid")
    with np.errstate(invalid="ignore"):
        return num / den, np.arange(len(x))[n - 1:]


# -- Strehl vs time ----------------------------------------------------------------
L, S = load(args.main)
off = dict(np.load(RES / f"{args.offline}.npz"))
off_s = json.loads((RES / f"{args.offline}.json").read_text())
fig, axes = P.plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.0, 2.2]}, sharey=True)
se = L["strehl"][1:]
frames = np.arange(1, len(se) + 1)
for ax, lim in zip(axes, [(0, 1000), (0, len(se))]):
    ok = np.isfinite(se)
    ax.plot(frames[ok], se[ok], color=P.SERIES[0], lw=0.4, alpha=0.35)
    y, x = rolling(se)
    ax.plot(x + 1, y, color=P.SERIES[0], lw=1.6,
            label=f"pyRTC real time (LE {S['strehl']['le_after_600']:.3f})")
    for d, c in ((2, P.SERIES[1]), (3, P.SERIES[3])):
        key = f"se_delay{d}"
        if key in off:
            y, x = rolling(off[key][: len(se)])
            ax.plot(x + 1, y, color=c, lw=1.1, ls="--",
                    label=f"offline harness, delay {d} (LE {off_s[f'delay{d}']['le_after_600']:.3f})")
    if (RES / args.frozen / "frames.npz").exists():
        Lf, _ = load(args.frozen)
        sf = Lf["strehl"][1:]
        y, x = rolling(sf)
        ax.plot(x + 1, y, color=P.SERIES[7], lw=1.1, label="loop never started (DM frozen)")
    ax.axvline(S["handover_frame"], color=P.MUTED, lw=1, ls=":")
    ax.set_xlim(*lim)
    ax.set_xlabel("frame (1 ms of simulated time each)")
axes[0].text(S["handover_frame"] + 15, 0.08, "hand-over:\nnetwork alone", fontsize=8, color=P.INK2)
axes[0].set_ylabel("H-band short-exposure Strehl (20-frame mean)")
axes[0].set_ylim(0, 0.9)
axes[0].set_title("hand-over", fontsize=10)
axes[1].set_title(f"whole run ({S['frames']} frames, {S['wall_seconds_camera']:.1f} s wall, "
                  f"{S['frame_rate_hz']['mean']:.0f} frames/s)", fontsize=10)
axes[1].legend(loc="lower right")
P.save(fig, "rtc_demo", "strehl_vs_time",
       f"""Focal-plane-only loop running in real time inside pyRTC (run `{args.main}`): simulated Keck focal-plane
camera (H band, 1 rad defocus, 1e5 photons) -> TorchImageReconstructor (slim 6.7M CNN, fp16, CUDA graph) ->
pyRTC leaky integrator (gain 0.4, leak 0.99, identity CM over 120 modes) -> DM, with the true H-band Strehl
logged by the camera. The ideal sensor closes the loop for 300 frames (dotted line), then the network holds it
alone. Dashed: the offline harness on the same atmosphere with a fixed 2- and 3-frame delay; red: the same
start with the pyRTC loop never started (DM frozen at the hand-over shape). **What to look at:** the real-time
loop holds at the offline Strehl after the hand-over, and the frozen DM shows it is the loop doing it.""",
       title="pyRTC real-time demo: focal-plane-only AO loop")

# -- latency histogram -----------------------------------------------------------------
tags = [t for t in args.latency_runs.split(",") if (RES / t / "frames.npz").exists()]
LABELS = {
    "soft_1khz": "soft RTC (one process), recon A400",
    "hard_1khz_nospin": "hard RTC, recon A400, cores may deep-idle",
    "hard_1khz": "hard RTC, recon A400, idle-spinners",
    "hard_1khz_recon4060": "hard RTC, recon on the shared RTX 4060",
    "hard_500hz": "hard RTC at 500 Hz, recon A400, idle-spinners",
}
fig, ax = P.plt.subplots(figsize=(8, 4.6))
bins = np.linspace(0, 6, 121)
for i, tag in enumerate(tags):
    Lr, Sr = load(tag)
    ho = Sr["handover_frame"]
    fid = np.arange(ho + 3, len(Lr["t_publish"]) - 1)
    lat = (Lr["t_dm"][fid] - Lr["t_publish"][fid]) * 1e3
    lat = lat[np.isfinite(lat)]
    lt = Sr["latency"]["total"]["statistics"] if Sr.get("latency") else None
    extra = f", wfs->wfc p50/p99 {lt['p50_seconds'] * 1e3:.2f}/{lt['p99_seconds'] * 1e3:.2f} ms" if lt else ""
    ax.hist(lat, bins=bins, histtype="step", lw=1.6, color=P.SERIES[i],
            label=f"{LABELS.get(tag, tag)}: {Sr['frame_rate_hz']['mean']:.0f} frames/s, median "
                  f"{np.median(lat):.2f}, p99 {np.percentile(lat, 99):.2f} ms{extra}")
ax.axvline(1.0, color=P.MUTED, lw=1, ls=":")
ax.set_yscale("log")
ax.text(1.04, 3e3, "1 ms frame\nperiod", fontsize=8, color=P.INK2, va="top")
ax.set_xlabel("frame published -> DM updated (ms)")
ax.set_ylabel("frames")
ax.set_xlim(0, 5.5)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=7.5, ncol=1)
P.save(fig, "rtc_demo", "latency_hist",
       """Per-frame latency from the camera publishing a frame on the `wfs` stream to the simulated DM taking the
command computed from it (FPDM.send_to_hardware), during the network phase. It includes the reconstructor
(read, H2D, preprocessing, CNN, D2H), the loop's integrator and the corrector's M2C. Legend: achieved frame
rate and pyRTC `manager.latency` WFS -> wfc stream latency (frame-id matched). soft: everything in one process
(threads share the GIL with the Python camera simulator); hard: reconstructor and loop in their own processes.
**What to look at:** whether the latency fits in one frame period (1 ms at 1 kHz), which makes the loop delay
2 frames as in the harness.""",
       title="pyRTC real-time demo: focal-plane-only AO loop")
P.note("rtc_demo", """Phase 6 of PLAN.md: the focal-plane-only loop run in real time inside pyRTC (code in
`research/fpwfs/rtc/`, numbers in `results/rtc/` and `rtc/README.md`).""",
       title="pyRTC real-time demo: focal-plane-only AO loop")
print("saved")
