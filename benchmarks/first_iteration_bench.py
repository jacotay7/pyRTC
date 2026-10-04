"""First-iteration latency of the per-frame worker functions.

A component's worker functions run numba kernels (``cache=True``). Unless
they are warmed beforehand, the first call in a process compiles each kernel
(cold cache) or loads it from the on-disk cache (warm cache), so the first
frame after ``start()`` stalls. This script builds each component on private
streams, publishes inputs by hand and times every call of one worker
function: the first call against the steady state that follows. Components
get no worker threads (``functions: []``); the method is called directly.
Before the first call the process idles for ``--idle`` seconds, like a paused
loop waiting for ``start()``, so thread pools (OpenBLAS) have gone to sleep.

Each case runs in a fresh subprocess, twice: once with an empty
``NUMBA_CACHE_DIR`` (cold cache) and once more with the same directory (warm
cache). Stream names are private, so it is safe to run next to a live RTC.

Usage::

    python benchmarks/first_iteration_bench.py --output first_iteration.json
    python benchmarks/first_iteration_bench.py --cases loop.leaky_integrator --iterations 500
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

LOOP_FUNCTIONS = (
    "leaky_integrator",
    "standard_integrator",
    "standard_integrator_pol",
    "pid_integrator",
    "pid_integrator_pol",
    "predictive_integrator",
)
CASES = (
    *(f"loop.{name}" for name in LOOP_FUNCTIONS),
    "slopes.shwfs",
    "slopes.pywfs",
    "wfc.send_to_hardware",
)
# Need CUDA-capable torch; run them with ``--cases``.
GPU_CASES = ("slopes.pywfs_gpu",)


# Seconds to idle between building a component and its first timed call.
IDLE_S = 0.25


def _private(name: str) -> str:
    return f"fib_{name}_{uuid.uuid4().hex[:8]}"


def _loop_case(
    function: str, iterations: int, signal_size: int, num_modes: int, idle: float
) -> dict:
    import pyshmem

    from pyrtc.loop import Loop

    rng = np.random.default_rng(0)
    signal = pyshmem.create(_private("signal"), shape=(signal_size,), dtype=np.float32)
    wfc = pyshmem.create(_private("wfc"), shape=(num_modes,), dtype=np.float32)
    try:
        conf = {
            "input_streams": {"signal": signal.name},
            "output_streams": {"wfc": wfc.name},
            "functions": [],
            "gain": 0.1,
            "leaky_gain": 0.01,
            "predictor": {"fit_frames": 64},
        }
        start = time.perf_counter()
        loop = Loop(conf)
        construct_s = time.perf_counter() - start
        try:
            im = rng.normal(size=(signal_size, num_modes)).astype(np.float32)
            loop.im = im
            loop.compute_cm()
            loop.gain = 0.1
            time.sleep(idle)
            frames = rng.normal(scale=1e-3, size=(iterations, signal_size)).astype(np.float32)
            timings = []
            method = getattr(loop, function)
            for frame in frames:
                signal.write(frame)
                start = time.perf_counter()
                method()
                timings.append(time.perf_counter() - start)
        finally:
            loop.close()
    finally:
        for stream in (signal, wfc):
            stream.close()
            stream.unlink()
    return {"construct_s": construct_s, "timings_s": timings}


def _slopes_case(wfs_type: str, iterations: int, idle: float) -> dict:
    import pyshmem

    from pyrtc.slopes_process import SlopesProcess
    from pyrtc.streams import clear_shms

    rng = np.random.default_rng(0)
    gpu = wfs_type.endswith("_gpu")
    wfs_type = wfs_type.removesuffix("_gpu")
    size = 64 if wfs_type == "pywfs" else 56
    wfs = pyshmem.create(_private("wfs"), shape=(size, size), dtype=np.float32)
    outputs = {"signal": _private("sig"), "signal_2d": _private("sig2d")}
    conf = {
        "type": wfs_type.upper(),
        "signal_type": "slopes",
        "functions": [],
        "input_streams": {"wfs": wfs.name},
        "output_streams": outputs,
    }
    if wfs_type == "pywfs":
        conf["pupils"] = ["16,16", "16,48", "48,16", "48,48"]
        conf["pupils_radius"] = 10
    else:
        conf.update({"sub_ap_spacing": 7, "sub_ap_offset_x": 0, "sub_ap_offset_y": 0})
    if gpu:
        conf["gpu_device"] = "cuda:0"
    try:
        start = time.perf_counter()
        proc = SlopesProcess(conf)
        construct_s = time.perf_counter() - start
        try:
            time.sleep(idle)
            frames = rng.uniform(0, 100, size=(iterations, size, size)).astype(np.float32)
            timings = []
            for frame in frames:
                wfs.write(frame)
                start = time.perf_counter()
                proc.compute_signal()
                timings.append(time.perf_counter() - start)
        finally:
            proc.close()
    finally:
        wfs.close()
        wfs.unlink()
        clear_shms(list(outputs.values()))
    return {"construct_s": construct_s, "timings_s": timings}


def _wfc_case(iterations: int, num_actuators: int, num_modes: int, idle: float) -> dict:
    import pyshmem

    from pyrtc.streams import clear_shms
    from pyrtc.wavefront_corrector import WavefrontCorrector

    rng = np.random.default_rng(0)
    name = _private("wfc")
    conf = {
        "name": "bench_wfc",
        "num_actuators": num_actuators,
        "num_modes": num_modes,
        "functions": [],
        "input_streams": {"wfc": name},
        "output_streams": {"wfc": name},
    }
    try:
        start = time.perf_counter()
        wfc = WavefrontCorrector(conf)
        construct_s = time.perf_counter() - start
        writer = pyshmem.open(name)
        try:
            time.sleep(idle)
            commands = rng.normal(scale=1e-3, size=(iterations, num_modes)).astype(np.float32)
            timings = []
            for command in commands:
                writer.write(command)
                start = time.perf_counter()
                wfc.send_to_hardware()
                timings.append(time.perf_counter() - start)
        finally:
            writer.close()
            wfc.close()
    finally:
        clear_shms([name])
    return {"construct_s": construct_s, "timings_s": timings}


def run_case(
    case: str, iterations: int, signal_size: int, num_modes: int, idle: float = IDLE_S
) -> dict:
    """Run one case in this process and summarise its timings."""

    component, _, function = case.partition(".")
    if component == "loop":
        raw = _loop_case(function, iterations, signal_size, num_modes, idle)
    elif component == "slopes":
        raw = _slopes_case(function, iterations, idle)
    elif component == "wfc":
        raw = _wfc_case(iterations, num_modes, num_modes, idle)
    else:
        raise ValueError(f"unknown case {case!r}")
    timings = np.asarray(raw["timings_s"], dtype=np.float64)
    steady = timings[1:]
    median = float(np.median(steady))
    return {
        "case": case,
        "construct_s": raw["construct_s"],
        "first_s": float(timings[0]),
        "steady_median_s": median,
        "steady_p99_s": float(np.percentile(steady, 99)),
        "first_over_median": float(timings[0] / median) if median > 0 else float("inf"),
    }


def run_case_in_subprocess(
    case: str,
    *,
    cache_dir: str | None = None,
    iterations: int = 200,
    signal_size: int = 1600,
    num_modes: int = 400,
    idle: float = IDLE_S,
) -> dict:
    """Run one case in a fresh interpreter, so no kernel is compiled yet.

    ``cache_dir`` becomes ``NUMBA_CACHE_DIR`` (an empty directory gives a
    cold cache); ``None`` keeps the default cache.
    """

    env = dict(os.environ)
    if cache_dir is not None:
        env["NUMBA_CACHE_DIR"] = str(cache_dir)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_ROOT), env.get("PYTHONPATH")]))
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        case,
        "--iterations",
        str(iterations),
        "--signal-size",
        str(signal_size),
        "--num-modes",
        str(num_modes),
        "--idle",
        str(idle),
    ]
    completed = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"case {case} failed:\n{completed.stdout}\n{completed.stderr}")
    return json.loads(completed.stdout.strip().splitlines()[-1])


def run_first_iteration_benchmarks(
    cases=CASES, *, iterations=200, signal_size=1600, num_modes=400, idle=IDLE_S
) -> dict:
    """Run each case with a cold and then a warm numba cache."""

    sizes = {"iterations": iterations, "signal_size": signal_size, "num_modes": num_modes}
    results = {}
    for case in cases:
        with tempfile.TemporaryDirectory(prefix="pyrtc_numba_cache_") as cache_dir:
            cold = run_case_in_subprocess(case, cache_dir=cache_dir, idle=idle, **sizes)
            warm = run_case_in_subprocess(case, cache_dir=cache_dir, idle=idle, **sizes)
        results[case] = {"cold_cache": cold, "warm_cache": warm}
    return {
        "meta": {
            "benchmark_type": "first_iteration",
            "python": platform.python_version(),
            "machine": platform.machine(),
            **sizes,
            "idle_s": idle,
        },
        "results": results,
    }


def _format(report: dict) -> str:
    lines = [
        f"{'case':32s} {'cache':5s} {'construct':>10s} {'first':>10s} {'median':>10s} {'ratio':>8s}"
    ]
    for case, by_cache in report["results"].items():
        for cache, row in by_cache.items():
            lines.append(
                f"{case:32s} {cache.split('_')[0]:5s} "
                f"{row['construct_s'] * 1e3:8.1f}ms {row['first_s'] * 1e3:8.2f}ms "
                f"{row['steady_median_s'] * 1e6:8.1f}us {row['first_over_median']:8.1f}"
            )
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", nargs="+", default=list(CASES), choices=CASES + GPU_CASES)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--signal-size", type=int, default=1600)
    parser.add_argument("--num-modes", type=int, default=400)
    parser.add_argument(
        "--idle", type=float, default=IDLE_S, help="Seconds idle before the first call"
    )
    parser.add_argument("--output", default=None, help="Write the JSON report here")
    parser.add_argument("--child", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.iterations < 2:
        parser.error("--iterations must be >= 2")

    if args.child:
        sys.path.insert(0, str(REPO_ROOT))
        row = run_case(args.child, args.iterations, args.signal_size, args.num_modes, args.idle)
        print(json.dumps(row))
        return 0

    report = run_first_iteration_benchmarks(
        args.cases,
        iterations=args.iterations,
        signal_size=args.signal_size,
        num_modes=args.num_modes,
        idle=args.idle,
    )
    print(_format(report))
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
