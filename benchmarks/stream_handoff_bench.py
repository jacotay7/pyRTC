"""One-way stream handoff latency with and without pyshmem ``notify``.

A producer publishes a timestamp every ``--period`` seconds through a stream
made with :func:`pyrtc.streams.create_stream`; a consumer blocks in
``read_after_publication`` (as ``Component.read_stream`` does) and records how
long after the write it woke up. The consumer runs either in a thread of the
same process (soft-RTC) or in a separate process (hard-RTC). The producer's
``write()`` call is timed as well, to show the writer-side cost of the futex
wake that notify adds.

Usage::

    python benchmarks/stream_handoff_bench.py --samples 2000 --output handoff.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import platform
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyrtc.streams import clear_shms, create_stream, open_stream  # noqa: E402


def _stats_us(values) -> dict:
    arr = np.asarray(values, dtype=np.float64) * 1e6
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean_us": float(arr.mean()),
        "p50_us": float(np.percentile(arr, 50)),
        "p99_us": float(np.percentile(arr, 99)),
        "max_us": float(arr.max()),
        "jitter_us": float(arr.std()),
    }


def _consume(name: str, samples: int, ready, done, results):
    """Read ``samples`` publications and record wake-up delay per frame."""
    stream = open_stream(name, readonly=True)
    delays = []
    consumed = stream.count
    ready.set()
    cpu_start = time.thread_time()
    wall_start = time.perf_counter()
    try:
        while len(delays) < samples:
            publication = stream.read_after_publication(consumed, timeout=5.0)
            now = time.perf_counter()
            wall = time.time()
            consumed = publication.count
            delays.append(
                (
                    now - float(np.asarray(publication.payload).ravel()[0]),
                    wall - float(publication.write_time),
                )
            )
    finally:
        done.set()
        stream.close()
    # Fraction of one core the consumer burned while waiting and reading.
    cpu_fraction = (time.thread_time() - cpu_start) / max(time.perf_counter() - wall_start, 1e-9)
    delays = {"delays": delays, "cpu_fraction": cpu_fraction}
    if isinstance(results, list):
        results.append(delays)
    else:
        results.put(delays)


def _produce(stream, samples: int, period: float, stop) -> list[float]:
    write_costs = []
    payload = np.zeros(1, dtype=np.float64)
    # Keep publishing until the consumer has its samples: it may skip
    # publications when the period is shorter than its wake-up time.
    for _ in range(4 * samples + 1000):
        if stop():
            break
        # Sleep (not spin) so a same-process consumer thread can take the GIL.
        time.sleep(period)
        payload[0] = time.perf_counter()
        start = time.perf_counter()
        stream.write(payload)
        write_costs.append(time.perf_counter() - start)
    return write_costs


def run_handoff(mode: str, notify: bool, *, samples: int, period: float) -> dict:
    name = f"bench_{uuid.uuid4().hex[:10]}_handoff"
    stream = create_stream(name, (1,), np.float64, notify=notify)
    try:
        if mode == "thread":
            ready = threading.Event()
            done = threading.Event()
            results: list = []

            def _consume_recording_errors():
                try:
                    _consume(name, samples, ready, done, results)
                except BaseException as exc:  # surfaced to the caller above
                    results.append(exc)
                    done.set()

            worker = threading.Thread(target=_consume_recording_errors)
            worker.start()
            ready.wait(5.0)
            write_costs = _produce(stream, samples, period, done.is_set)
            worker.join(timeout=10.0)
            if not results:
                raise RuntimeError("handoff consumer thread produced no results")
            if isinstance(results[0], BaseException):
                raise RuntimeError("handoff consumer thread failed") from results[0]
            delays = results[0]
        else:
            ctx = mp.get_context("spawn")
            ready = ctx.Event()
            done = ctx.Event()
            queue = ctx.Queue()
            worker = ctx.Process(target=_consume, args=(name, samples, ready, done, queue))
            worker.start()
            if not ready.wait(30.0):
                raise RuntimeError("consumer process did not start")
            write_costs = _produce(stream, samples, period, done.is_set)
            delays = queue.get(timeout=30.0)
            worker.join(timeout=10.0)
        cpu_fraction = float(delays["cpu_fraction"])
        delays = delays["delays"]
        return {
            "mode": mode,
            "consumer_cpu_fraction": cpu_fraction,
            "notify": bool(notify),
            "notify_active": bool(stream.notify),
            # write() call start -> consumer awake with the payload
            "handoff": _stats_us([pair[0] for pair in delays]),
            # publication (pyshmem write_time) -> consumer awake
            "publish_to_wake": _stats_us([pair[1] for pair in delays]),
            "writer_write_call": _stats_us(write_costs),
        }
    finally:
        stream.close()
        clear_shms([name])


def run_all(*, samples: int, period: float) -> dict:
    results = []
    for mode in ("thread", "process"):
        for notify in (False, True):
            results.append(run_handoff(mode, notify, samples=samples, period=period))
    return {
        "meta": {
            "benchmark_type": "stream_handoff",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "samples": samples,
            "period_s": period,
        },
        "results": results,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--period", type=float, default=1e-3, help="Seconds between writes.")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args(argv)
    report = run_all(samples=args.samples, period=args.period)
    for row in report["results"]:
        handoff = row["handoff"]
        wake = row["publish_to_wake"]
        write = row["writer_write_call"]
        print(
            f"{row['mode']:>7} notify={'on ' if row['notify'] else 'off'} "
            f"handoff p50={handoff['p50_us']:6.1f} mean={handoff['mean_us']:6.1f} "
            f"p99={handoff['p99_us']:6.1f} | publish->wake p50={wake['p50_us']:6.1f} "
            f"mean={wake['mean_us']:6.1f} p99={wake['p99_us']:6.1f} | "
            f"write() p50={write['p50_us']:5.1f} mean={write['mean_us']:5.1f} (us) | "
            f"consumer cpu={100 * row['consumer_cpu_fraction']:4.1f}%"
        )
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
