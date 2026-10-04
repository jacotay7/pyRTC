"""Gather the key numbers of every run into results/rtc/results.json and print a Markdown table."""

import json
import pathlib

RES = pathlib.Path(__file__).resolve().parents[1] / "results" / "rtc"
RUNS = ["hard_1khz", "hard_500hz", "hard_1khz_nospin", "soft_1khz", "hard_1khz_recon4060", "hard_live_free",
        "frozen", "atm0_hard_1khz", "atm0_a400_hard_1khz"]


def ms(x):
    return None if x is None else round(1e3 * x, 3)


out = {"runs": {}}
rows = []
for tag in RUNS:
    path = RES / tag / "summary.json"
    if not path.exists():
        continue
    s = json.loads(path.read_text())
    a = s["args"]
    lat = s.get("latency") or {}
    tot = lat.get("total", {}).get("statistics", {})
    segs = {f"{g['source_shm'].split('_')[-1]}->{g['target_shm'].split('_')[-1]}": g["statistics"]
            for g in lat.get("segments", [])}
    ts = s.get("timing_stats", {})
    age = s["dm_state_age_frames"]
    n_age = sum(v for v in age.values())
    r = dict(
        mode=a["mode"], sim_device=a["sim_device"], recon_device=a["recon_device"], dtype=a["dtype"],
        wall_rate=a["wall_rate"], atmosphere=a["atmosphere"], seed=a["seed"], atm_index=a.get("atm_index", 0),
        spinners=a["spinners"], frames=s["frames"], wall_seconds=round(s["wall_seconds_camera"], 2),
        frame_rate_median=round(s["frame_rate_hz"]["median"], 1), frame_rate_mean=round(s["frame_rate_hz"]["mean"], 1),
        camera_sim_ms=s["camera_sim_ms"],
        recon_compute_ms=dict(median=ms(ts.get("median")), p99=ms(ts.get("p99")), max=ms(ts.get("max"))),
        wfs_to_signal_ms=dict(median=ms(segs.get("wfs->signal", {}).get("p50_seconds")),
                              p99=ms(segs.get("wfs->signal", {}).get("p99_seconds"))),
        signal_to_wfc_ms=dict(median=ms(segs.get("signal->wfc", {}).get("p50_seconds")),
                              p99=ms(segs.get("signal->wfc", {}).get("p99_seconds"))),
        wfs_to_wfc_ms=dict(median=ms(tot.get("p50_seconds")), p99=ms(tot.get("p99_seconds"))),
        publish_to_dm_ms=s["publish_to_dm_ms"],
        delay2_fraction=round(age.get("2", 0) / max(n_age, 1), 3),
        strehl=s["strehl"],
    )
    out["runs"][tag] = r
    rows.append(
        f"| {tag} | {r['mode']} | {r['sim_device']} / {r['recon_device']} | {r['wall_rate']:.0f} | "
        f"{r['frame_rate_median']:.0f} / {r['frame_rate_mean']:.0f} | "
        f"{r['camera_sim_ms']['median']:.2f} / {r['camera_sim_ms']['p99']:.2f} | "
        f"{r['recon_compute_ms']['median']} / {r['recon_compute_ms']['p99']} | "
        f"{r['wfs_to_signal_ms']['median']} / {r['wfs_to_signal_ms']['p99']} | "
        f"{r['wfs_to_wfc_ms']['median']} / {r['wfs_to_wfc_ms']['p99']} | "
        f"{r['delay2_fraction']:.2f} | {r['strehl']['le_after_600']:.3f} |")
for name in ("offline_ref_atm3", "offline_ref_atm0_4060", "parity", "survival_seed4000_4060_d2",
             "survival_seed4000_4060_d3", "survival_seed4000_a400_d2"):
    p = RES / f"{name}.json"
    if p.exists():
        d = json.loads(p.read_text())
        if name.startswith("survival"):
            d = dict(held_full_run=d["held_full_run"], batch=d["args"]["batch"], steps=d["args"]["steps"],
                     delay=d["args"]["delay"], device=d["args"]["device"],
                     fail_frames=[a["fail_frame"] for a in d["atmospheres"]])
        out[name] = d
(RES / "results.json").write_text(json.dumps(out, indent=1))
print("| run | mode | sim / recon GPU | target Hz | achieved Hz (median / mean) | camera sim ms (med / p99) | "
      "recon compute ms (med / p99) | wfs->signal ms (med / p99) | wfs->wfc ms (med / p99) | delay-2 frac | LE H Strehl |")
print("|" + " --- |" * 11)
print("\n".join(rows))
