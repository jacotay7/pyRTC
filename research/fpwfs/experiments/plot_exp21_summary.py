"""Summary of exp21 operation runs against the SH on the same atmospheres (exp03 --seeds 980 981)."""

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fpsim import plotting as P  # noqa: E402

RUNS = [  # (exp21 tag, label)
    ("unsupervised_10s", "v7, no supervisor"),
    ("supervised_10s", "v7"),
    ("supervised_kick_r2", "kick r2"),
    ("supervised_kick_r3", "kick r3"),
    ("supervised_kick_r2_fastacq", "kick r2, fast acq."),
    ("supervised_kick_r3_fastacq", "kick r3, fast acq."),
    ("supervised_kick6_fastacq", "kick r6, fast acq."),
]
rows = []
for tag, label in RUNS:
    f = ROOT / "results" / "exp21" / f"{tag}.json"
    if f.exists():
        d = json.loads(f.read_text())
        sup = not d["args"].get("no_supervisor", False)
        rows.append((label, d["effective"], d["locked_frac"], sum(d["losses"]) if sup else None))
sh = {}
for f in sorted((ROOT / "results" / "exp03").glob("sh_10s_g*.json")):
    r = json.loads(f.read_text())["results"][0]["H"]
    sh[json.loads(f.read_text())["args"]["gain"]] = r["se_mean"]

fig, (a1, a2) = P.plt.subplots(1, 2, figsize=(11.0, 3.8), gridspec_kw=dict(width_ratios=[1.6, 1], wspace=0.08))
y = range(len(rows))
a1.barh(y, [r[1] for r in rows], color=P.SERIES[0])
for i, r in enumerate(rows):
    a1.text(r[1] - 0.01, i, f"{r[1]:.3f}  ({r[2] * 100:.1f} % locked)", va="center", ha="right", fontsize=8,
            color="white")
if sh:
    g, best = max(sh.items(), key=lambda kv: kv[1])
    a1.axvline(best, color=P.SERIES[1], lw=1.5, ls="--")
    a1.text(best, len(rows) - 0.4, f" SH, gain {g:g}: {best:.3f}", color=P.SERIES[1], fontsize=8, va="bottom")
a1.set_yticks(list(y), [r[0] for r in rows])
a1.set_xlim(0, 0.8)
a1.set_xlabel("effective (time-averaged) SE H Strehl")
a2.barh(y, [r[3] or 0 for r in rows], color=P.SERIES[7])
for i, r in enumerate(rows):
    a2.text((r[3] or 0) + 0.4, i, "no supervisor: losses not detected" if r[3] is None else str(r[3]), va="center",
            fontsize=8)
a2.set_yticks(list(y), [""] * len(rows))
a2.set_xlabel("loop losses in 240 atmosphere-seconds")
fig.suptitle("Focal-plane-only operation vs SH on the same 24 atmospheres (0.6\", 1 kHz, 10 s each)", fontsize=10)
P.save(fig, "exp21_operation", "summary",
       """Each bar is one configuration of the focal-plane-only system (acquisition networks at 25 rad focus, maintenance
network at 1 rad, supervisor that re-acquires on loss) run for 10 s on 24 unseen atmospheres. 'kick rN': maintenance
network trained to recover from random DM kicks (round N); 'fast acq.': re-acquisition schedule 50/100/100 frames
instead of 100/200/100. Dashed line: the 20x20 SH (OCAM2K, V~9.5) on the same atmospheres with the best gain tried,
same metric (mean short-exposure H Strehl after frame 600). **What to look at:** whether the focal-plane system's
effective Strehl, which pays for every loss and re-acquisition, reaches the SH line, and how the loss count falls.""",
       title="exp21: operating the focal-plane-only loop")
