"""Print the key numbers of one or more runs: ``python show.py TAG [TAG ...]``."""

import json
import pathlib
import sys

RES = pathlib.Path(__file__).resolve().parents[1] / "results" / "rtc"
for tag in sys.argv[1:]:
    s = json.loads((RES / tag / "summary.json").read_text())
    a = s["args"]
    print(f"== {tag}: {a['mode']} sim {a['sim_device']} recon {a['recon_device']} wall_rate {a['wall_rate']}"
          f" spinners {a['spinners']}")
    for k in ["frame_rate_hz", "period_ms", "missed_ticks", "camera_sim_ms", "publish_to_dm_ms", "strehl",
              "timing_stats", "dm_state_age_frames"]:
        print(" ", k, s.get(k))
    if s.get("latency"):
        t = s["latency"]["total"]["statistics"]
        print(f"  wfs->wfc p50 {t['p50_seconds'] * 1e3:.3f} p99 {t['p99_seconds'] * 1e3:.3f} ms")
        for g in s["latency"]["segments"]:
            st = g["statistics"]
            print(f"  {g['source_shm']} -> {g['target_shm']}: p50 {st['p50_seconds'] * 1e3:.3f}"
                  f" p99 {st['p99_seconds'] * 1e3:.3f} ms")
