"""Run the focal-plane-only AO loop in pyRTC and measure latency and Strehl.

    PYTHONPATH=<repo> python research/fpwfs/rtc/run_demo.py --tag soft_a400 \\
        --sim-device cuda:0 --recon-device cuda:1 --wall-rate 0

Sequence: build the system from ``system.yaml`` (private stream names), start
every component with the camera holding and the loop paused, then let the camera
run: ``warm_frames`` exposures of ideal-sensor control (see ``fprtc.py``), the
hand-over (the context's hook starts the pyRTC loop), then the network alone
until ``frames`` exposures. During the network phase the script measures
``manager.latency`` (WFS -> signal -> wfc, frame-id matched) and afterwards
collects the reconstructor's ``timing_stats()`` and the context's per-frame log
(true Strehl, residual, DM-state age, camera-sim time, DM-applied time).
Results: ``research/fpwfs/results/rtc/<tag>/`` (summary.json, frames.npz,
config.yaml).
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
FPWFS = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(FPWFS))

p = argparse.ArgumentParser()
p.add_argument("--tag", default="soft")
p.add_argument("--mode", choices=["soft", "hard"], default="soft",
               help="hard: reconstructor and loop run as hard-RTC child processes")
p.add_argument("--sim-device", default="cuda:0")
p.add_argument("--recon-device", default="cuda:1")
p.add_argument("--dtype", default="float16")
p.add_argument("--wall-rate", type=float, default=0.0, help="camera wall-clock rate, Hz (0: free-run)")
p.add_argument("--atmosphere", default="precomputed", choices=["precomputed", "live"])
p.add_argument("--frames", type=int, default=6000)
p.add_argument("--warm-frames", type=int, default=300)
p.add_argument("--seed", type=int, default=4000)
p.add_argument("--atm-index", type=int, default=0)
p.add_argument("--gain", type=float, default=0.4)
p.add_argument("--leak", type=float, default=0.99)
p.add_argument("--no-loop", action="store_true", help="never start the loop: DM frozen at hand-over")
p.add_argument("--strehl-every", type=int, default=1, help="science Strehl on every N-th exposure")
p.add_argument("--latency-samples", type=int, default=2000)
p.add_argument("--cores", default="20-27", help="cores for the pipeline threads (wfs, wfc, slopes, loop, ...)")
p.add_argument("--spinners", action="store_true",
               help="SCHED_IDLE busy loops on the pipeline cores (keeps them out of deep idle states)")
p.add_argument("--switch-interval", type=float, default=0.0, help="sys.setswitchinterval (s); 0 keeps 5 ms")
args = p.parse_args()

if args.switch_interval > 0:
    sys.setswitchinterval(args.switch_interval)

lo, hi = (int(v) for v in args.cores.split("-"))
cores = list(range(lo, hi + 1))
OUT = FPWFS / "results" / "rtc" / args.tag
OUT.mkdir(parents=True, exist_ok=True)

# -- files -------------------------------------------------------------------------
from models import DEFAULT_WEIGHTS, export_scale  # noqa: E402

scale_path = OUT / "scale_m.npy"
export_scale(DEFAULT_WEIGHTS, scale_path)
im_path = OUT / "identity_im.npy"
np.save(im_path, np.eye(120, dtype=np.float32))

conf = yaml.safe_load((HERE / "system.yaml").read_text())
prefix = f"fprtc_{args.tag}_"
res = conf["resources"]["fpsim"]
res.update(class_file=str(HERE / "fprtc.py"), device=args.sim_device, seed=args.seed, atm_index=args.atm_index,
           wall_rate=args.wall_rate, atmosphere=args.atmosphere, max_frames=args.frames,
           warm_frames=args.warm_frames, strehl_every=args.strehl_every, warm_gain=args.gain, warm_leak=args.leak)
for sec in ("wfs", "wfc"):
    conf[sec]["class_file"] = str(HERE / "fprtc.py")
conf["wfs"]["output_streams"] = {"wfs_raw": prefix + "wfs_raw", "wfs": prefix + "wfs"}
conf["wfc"]["input_streams"] = {"wfc": prefix + "wfc"}
conf["wfc"]["output_streams"] = {"wfc": prefix + "wfc"}
conf["slopes"].update(model_factory_file=str(HERE / "models.py"), output_scale_file=str(scale_path),
                      device=args.recon_device, dtype=args.dtype,
                      input_streams={"wfs": prefix + "wfs"}, output_streams={"signal": prefix + "signal"})
conf["loop"].update(im_file=str(im_path), gain=args.gain, leaky_gain=round(1.0 - args.leak, 10),
                    input_streams={"signal": prefix + "signal"}, output_streams={"wfc": prefix + "wfc"})
conf["wfs"]["affinity"], conf["wfc"]["affinity"] = cores[0], cores[1]
conf["slopes"]["affinity"], conf["loop"]["affinity"] = cores[2], cores[3]
if args.mode == "hard":
    conf["manager"]["component_modes"] = {"slopes": "hard-rtc", "loop": "hard-rtc"}
config_path = OUT / "config.yaml"
config_path.write_text(yaml.safe_dump(conf, sort_keys=False))
streams = [prefix + s for s in ("wfs_raw", "wfs", "signal", "wfc")]

spinners = []
if args.spinners:
    for c in cores:
        spinners.append(subprocess.Popen(
            ["chrt", "--idle", "0", "taskset", "-c", str(c), sys.executable, "-c", "while True: pass"]))

from pyrtc import RTCManager, clear_shms  # noqa: E402

clear_shms(streams)
summary: dict = {"args": vars(args), "streams": streams}
t_start = time.perf_counter()
manager = RTCManager.from_config_file(config_path)
try:
    manager.start()  # camera holds (no frames) until ctx.begin(); loop paused right below
    loop = manager.get_component("loop")
    slopes = manager.get_component("slopes")
    hard = args.mode == "hard"

    def call(component, name, *a):
        return component.run(name, *a) if hard else getattr(component, name)(*a)

    call(loop, "stop")
    ctx = manager.resources["fpsim"]
    wfc = manager.get_component("wfc")
    # Prime the paused loop: run one integrator step from the main thread (soft) or
    # over RPC (hard) on a zero signal, so its numba kernel is compiled before the
    # hand-over. (Unprimed, the first iteration after start took ~360 ms: 50 frames
    # with a frozen DM, after which the network had lost the loop.) The first zero
    # write releases a worker that may still be blocked in a read; the second is the
    # one the direct call consumes.
    from pyrtc.streams import open_stream

    sig = open_stream(prefix + "signal")
    for _ in range(2):
        sig.write(np.zeros(120, dtype=np.float32))
        time.sleep(0.2)
    try:
        call(loop, "leaky_integrator") if not hard else loop.run("leaky_integrator", timeout=10.0)
    except Exception as exc:
        print("priming call:", exc, flush=True)
    sig.close()
    wfc.flatten()
    summary["build_seconds"] = time.perf_counter() - t_start
    summary["camera_graph_active"] = bool(ctx.graph_active)
    summary["precompute_seconds"] = getattr(ctx, "precompute_seconds", None)
    print(f"built in {summary['build_seconds']:.1f} s (camera graph {ctx.graph_active})", flush=True)
    time.sleep(1.0)

    handover_done = threading.Event()

    def start_loop():
        call(slopes, "reset_timing")
        if not args.no_loop:
            call(loop, "start")
        summary["loop_started_wall"] = time.perf_counter()
        handover_done.set()

    ctx.on_handover = start_loop
    t_begin = time.perf_counter()
    ctx.begin()

    def status():
        e = ctx.exposure
        s = ctx.log["strehl"][max(1, e - 50): e + 1]
        print(f"t={time.perf_counter() - t_begin:5.1f}s frame {e:5d} phase {ctx.phase:7s} "
              f"SE Strehl(last 50) {np.nanmean(s):.3f} missed {ctx.missed_ticks}", flush=True)

    while not handover_done.is_set():
        time.sleep(0.25)
        status()
    time.sleep(0.5)
    status()
    lat = None
    if not args.no_loop:
        try:
            lat = manager.latency(stream_path=[prefix + "wfs", prefix + "signal", prefix + "wfc"],
                                  samples=args.latency_samples, timeout_seconds=120.0)
        except Exception as exc:
            print("latency failed:", exc, flush=True)
    summary["latency"] = lat
    while not ctx.done.is_set():
        time.sleep(0.5)
        status()
    t_end = time.perf_counter()
    summary["timing_stats"] = call(slopes, "timing_stats")
    summary["graph_active_recon"] = bool(slopes.runner.graph_active) if not hard else None
    call(loop, "stop")
    summary["wall_seconds_camera"] = t_end - t_begin
finally:
    try:
        ctx = manager.resources.get("fpsim")
        if ctx is not None:
            ctx.save_log(OUT / "frames.npz")
    finally:
        manager.close()
        clear_shms(streams)
        for sp in spinners:
            sp.kill()

# -- summary from the frame log ------------------------------------------------------
L = dict(np.load(OUT / "frames.npz"))
n = len(L["strehl"]) - 1
ho = int(L["handover_frame"])
e_idx = np.arange(1, n + 1)
net_frames = e_idx[e_idx > ho + 2]
ticks = L["t_tick"][1:]
period = np.diff(ticks[~np.isnan(ticks)])
age = (e_idx - L["dm_fid"][1:])[e_idx > ho + 5]
fids = net_frames[(net_frames < n)]
dm_lat = (L["t_dm"][fids] - L["t_publish"][fids])
dm_lat = dm_lat[~np.isnan(dm_lat)]
render = L["t_render"][1:][~np.isnan(L["t_render"][1:])]
se = L["strehl"]


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


summary["frames"] = n
summary["handover_frame"] = ho
summary["frame_rate_hz"] = {"median": 1.0 / float(np.median(period)), "mean": len(period) / float(period.sum())}
summary["period_ms"] = {"median": 1e3 * float(np.median(period)), "p99": 1e3 * pct(period, 99),
                        "max": 1e3 * float(period.max())}
summary["missed_ticks"] = int(L["missed_ticks"])
summary["camera_sim_ms"] = {"median": 1e3 * float(np.median(render)), "p99": 1e3 * pct(render, 99)}
summary["publish_to_dm_ms"] = {"median": 1e3 * float(np.median(dm_lat)) if len(dm_lat) else None,
                               "p99": 1e3 * pct(dm_lat, 99), "count": int(len(dm_lat))}
vals, counts = np.unique(age, return_counts=True)
summary["dm_state_age_frames"] = {int(v): int(c) for v, c in zip(vals, counts) if v <= 10}
summary["dm_state_age_frames"][">10"] = int(counts[vals > 10].sum())
summary["strehl"] = {
    "le_after_600": float(L["le_strehl"]),
    "se_mean_after_600": float(np.nanmean(se[601:])),
    "se_mean_warm_200_300": float(np.nanmean(se[201:ho + 1])) if ho > 200 else None,
    "se_mean_last_1000": float(np.nanmean(se[-1000:])),
    "rms_nm_mean_after_600": float(np.nanmean(L["rms_nm"][601:])),
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
print(json.dumps({k: summary[k] for k in summary if k not in ("args",)}, indent=1, default=float))
