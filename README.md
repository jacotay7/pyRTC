# pyrtc

`pyrtc` is an adaptive optics real-time control toolkit written in Python.

The user-facing name stays `pyrtc`.
For packaging only, the published PyPI package name is `pyrtcao`, while the Python import name remains `pyrtc`:

```bash
pip install pyrtcao
```

```python
import pyrtc
```

> **Name clash:** an unrelated WebRTC library is published on PyPI as `pyrtc` and
> also installs a top-level `pyrtc` module. Install this project as **`pyrtcao`**
> (`pip install pyrtc` gets the WebRTC library), and don't install both in the
> same environment: they overwrite each other's `pyrtc` package.

Documentation: [https://pyrtc-ao.readthedocs.io/en/latest/](https://pyrtc-ao.readthedocs.io/en/latest/)

Developer Guide: [https://pyrtc-ao.readthedocs.io/en/latest/guides/developers_guide.html](https://pyrtc-ao.readthedocs.io/en/latest/guides/developers_guide.html)

## Performance

The benchmark section is intentionally near the top because performance is a primary design constraint for `pyrtc`.
Two different things are reported here, and they should not be confused:

- **Kernel compute** — the time to run one AO iteration's compute kernels (image formation, slopes, control update) back to back in a single thread. This is a lower bound for what the math costs.
- **End-to-end pipeline latency** — the time from the wavefront sensor publishing a frame to the loop publishing the DM command computed from it, in a *running* pyrtc system. This adds shared-memory handoffs between components, thread/process scheduling and, in soft-RTC mode, GIL contention between worker threads. It is the number to use when estimating loop delay.

### Kernel Compute (single-threaded harness)

These measurements were captured on the current GPU-enabled host with the closed-loop synthetic benchmark harness:

```bash
python -m benchmarks.ao_loop_bench --output benchmarks/readme_benchmark_report.json --iterations 300 --warmup 30 --system-sizes 10 20 60
python benchmarks/readme_benchmark_table.py --report benchmarks/readme_benchmark_report.json --output benchmarks/readme_benchmark_table.md
```

The benchmark drives deterministic modal disturbances through synthetic `PYWFS` and `SHWFS` image formation, slope reduction, and a dense control update. That makes the reported numbers much closer to a real single-iteration AO control path than the earlier kernel-only table.

### Benchmark Host

| Component | Value |
| --- | --- |
| CPU | AMD Ryzen 9 9950X3D 16-Core Processor |
| CPU Threads | 32 |
| GPU | NVIDIA GeForce RTX 5090 |
| GPU Memory | 32607 MiB |
| NVIDIA Driver | 580.126.09 |
| Python | 3.12.0 |
| Torch | 2.10.0+cu128 |
| CUDA | 12.8 |

### Synthetic AO Loop Benchmarks

Values are reported as `p99 throughput / p99 latency` of the kernel compute for one iteration (no inter-component handoffs).

| Loop | 10x10 CPU | 10x10 GPU | 20x20 CPU | 20x20 GPU | 60x60 CPU | 60x60 GPU |
| --- | --- | --- | --- | --- | --- | --- |
| PYWFS full loop | 58.1 kHz / 17.2 us | 4.5 kHz / 219.9 us | 26.6 kHz / 37.6 us | 4.6 kHz / 218.5 us | 270 Hz / 3703.4 us | 3.0 kHz / 335.2 us |
| SHWFS full loop | 78.2 kHz / 12.8 us | 5.1 kHz / 195.0 us | 26.6 kHz / 37.6 us | 5.1 kHz / 196.7 us | 268 Hz / 3730.7 us | 3.5 kHz / 289.7 us |

For this host, the important pattern is the one we care about operationally: CPU wins the small `10x10` and `20x20` synthetic loops because launch overhead dominates, but the GPU is about an order of magnitude faster once the loop reaches the `60x60` regime. That crossover now shows up for both pyramid and Shack-Hartmann synthetic loops in the README numbers.

The benchmark artifacts committed for this host are:

- `benchmarks/readme_benchmark_report.json`
- `benchmarks/readme_benchmark_table.md`

### End-to-End Pipeline Latency

`benchmarks/pipeline_latency_bench.py` launches the synthetic SHWFS example (`examples/synthetic_shwfs/config.yaml`: 49x49 WFS, 7x7 subapertures, 97 modes, plus the synthetic DM and science camera) through `RTCManager`, lets it settle, and measures `manager.latency()` along `wfs -> signal -> wfc`, pairing writes by frame id. Each run uses private stream names, and every mode/notify combination is repeated three times, interleaved; the table shows the median across repeats.

```bash
python -m benchmarks.pipeline_latency_bench --samples 2048 --repeats 3 --output benchmarks/pipeline_latency_report.json
python -m benchmarks.pipeline_latency_bench --samples 4096 --repeats 3 --frame-rate-hz 1000 --output benchmarks/pipeline_latency_report_1khz.json
```

Host: Intel Core i7-10700 (16 threads), Linux 6.8, Python 3.13, CPU streams, default (unprivileged) scheduling. This is a different, smaller host than the kernel table above. WFS → DM command latency in microseconds (mean / p50 / p99):

| Mode | Stream notify | 200 Hz WFS (example) | 1 kHz WFS |
| --- | --- | --- | --- |
| soft-RTC (threads) | off | 321 / 308 / 567 | 287 / 277 / 560 |
| soft-RTC (threads) | on (default) | 299 / 278 / 627 | 298 / 288 / 555 |
| hard-RTC (processes) | off | 199 / 193 / 336 | 183 / 178 / 281 |
| hard-RTC (processes) | on (default) | 153 / 146 / 257 | 154 / 144 / 349 |

Compare that with the kernel table, where a comparable 10x10 SHWFS iteration computes in about 13 us at p99 (on a faster host): most of the end-to-end time is handoffs and scheduling, not math. In soft-RTC mode the component threads share one GIL, which is why hard-RTC is faster despite crossing process boundaries. "Stream notify" is pyshmem's futex wake-up, which pyrtc streams use by default (see the streams guide; `PYRTC_STREAM_NOTIFY=0` turns it off). It lowers hard-RTC mean and median latency by 15-25% (the 1 kHz p99 was worse in this run) and is within run-to-run noise in soft-RTC mode. The numbers move by 2x or more when the host is busy, so rerun the benchmark on the machine you care about.

Artifacts: `benchmarks/pipeline_latency_report.json`, `benchmarks/pipeline_latency_report_1khz.json`, and the single-handoff benchmark `benchmarks/stream_handoff_report.json` (`python -m benchmarks.stream_handoff_bench`).

The same commands on an 80-core aarch64 server (Neoverse-N1, Linux 6.17, Python 3.13, pinned to 8 cores, default scheduling), mean / p50 / p99 in microseconds:

| Mode | Stream notify | 200 Hz WFS (example) | 1 kHz WFS |
| --- | --- | --- | --- |
| soft-RTC (threads) | off | 2300 / 2155 / 6133 | 3623 / 2814 / 10054 |
| soft-RTC (threads) | on (default) | 1718 / 1481 / 3116 | 3225 / 2661 / 11226 |
| hard-RTC (processes) | off | 803 / 809 / 923 | 612 / 614 / 927 |
| hard-RTC (processes) | on (default) | 822 / 838 / 953 | 647 / 660 / 900 |

That is 4-10x the x86 numbers, and on this host the GIL-bound soft-RTC pipeline cannot keep up at 1 kHz. About half of the hard-RTC cost is the CPU waking from idle: the firmware advertises deep idle states with a ~3 ms exit latency, and cores enter them between frames. Keeping the cores busy with `SCHED_IDLE` spinners halved hard-RTC latency (notify on: ~320-380 µs p50). On a real system, limit idle states on the RTC cores instead (`cpupower idle-set`, or hold `/dev/cpu_dma_latency` at 0). Free-threaded Python fixes the soft-RTC case on this host: 843 µs p50 at 1 kHz instead of 6987 µs (see the developer guide). Artifacts: `benchmarks/pipeline_latency_report_aarch64.json` and `benchmarks/pipeline_latency_report_1khz_aarch64.json`.

## What It Is For

Adaptive optics (AO) systems measure optical aberrations and apply corrections quickly enough to recover image quality in dynamic environments. `pyrtc` is aimed at the software layer that connects those measurements, reconstructions, and corrections.

The project is designed for:

- laboratory AO systems and hardware integration work
- simulated AO development and algorithm prototyping
- moderate-performance real-time control in Python
- controller research: modal gain optimization and pluggable predictive control

## Release Posture

`pyrtc` `1.1.0` is the current release, published on PyPI as `pyrtcao`; see `CHANGELOG.md` for release notes. The release policy is conservative:

- User-facing project name: `pyrtc`
- PyPI distribution name: `pyrtcao`
- Python import name: `pyrtc`
- CLI prefix: `pyrtc-*`
- Primary supported release surface: Linux, Python 3.10-3.14 (free-threaded 3.14t is tested in CI)
- macOS and Windows: smoke-tested in GitHub Actions, but not part of the primary supported deployment story
- Windows: soft-RTC only — Windows named shared memory is freed when the last handle closes, so streams do not survive their producer process and hard-RTC restart/reattach flows are unsupported there
- GPU behavior: benchmark-validated on a Linux CUDA host for synthetic loop workloads, but still target-environment validation required for operational use
- Hardware integrations: examples and reference implementations, not universal plug-and-play support

## Core Capabilities

- Component-based AO pipeline built around wavefront sensing, slope processing, control, correction, telemetry, and science imaging
- Soft-RTC mode for single-process development and simulation workflows
- Hard-RTC mode for process-isolated hardware integration via shared memory and launcher utilities
- Control:
  - modal bases from [aobasis](https://github.com/jacotay7/aobasis) (KL, Zernike, Fourier, zonal, Hadamard);
  - push-pull, Hadamard and DOCRIME interaction matrices;
  - integrator, leaky, PID and POL controllers;
  - per-mode gains with an optimizer;
  - pluggable predictive control (modal LQG, least squares);
  - several correctors per loop (woofer/tweeter, tip-tilt offload).
- Safety: a loop input watchdog and DM saturation alerts, shown in manager status and the GUI
- Simulation backends: a built-in synthetic system, HCIPy, SPECULA and OOPAO
- Hardware adapters, as reference implementations:
  - cameras: GenICam (GigE/USB3 Vision) and Micro-Manager, plus XIMEA and Spinnaker;
  - DMs: ALPAO and Boston Micromachines;
  - a PI modulator.
- Interoperability: an ImageStreamIO (milk/CACAO) bridge and AOTPy telemetry export
- Optional viewer, manager GUI, latency measurement and benchmark tools

## Installation

### From PyPI

```bash
pip install pyrtcao
```

The base install is the soft-RTC core. Optional extras:

```bash
pip install pyrtcao[plot]      # matplotlib: plotting helpers, latency histograms, pyrtc-shm-monitor
pip install pyrtcao[fits]      # astropy: reading .fits files
pip install pyrtcao[optimize]  # optuna + cmaes: pyrtc.Optimizer and the hardware optimizers
pip install pyrtcao[viewer]    # Qt viewer (includes matplotlib)
pip install pyrtcao[gui]       # Qt manager GUI
pip install pyrtcao[aotpy]
pip install pyrtcao[docs]
pip install pyrtcao[gpu]
pip install pyrtcao[specula]
pip install pyrtcao[hcipy]         # HCIPy simulator backend
pip install pyrtcao[genicam]       # Harvesters: GenICam camera adapters
pip install pyrtcao[micromanager]  # pymmcore-plus: Micro-Manager camera adapters
pip install pyrtcao[hardware]      # PI, Spinnaker (rotpy) and XIMEA SDKs for those adapters
```

The ImageStreamIO bridge needs ImageStreamIO's Python module:
`pip install git+https://github.com/milk-org/ImageStreamIO`.

A feature whose extra is missing raises an `ImportError` naming the extra to install.

### From Source

```bash
git clone https://github.com/jacotay7/pyRTC.git
cd pyRTC
pip install .
```

The same extras work from a source checkout, e.g. `pip install .[viewer,hcipy]`.

The `specula` extra installs the [SPECULA](https://pypi.org/project/specula/) simulator used by the simulator-backed examples. The OOPAO simulator is not on PyPI and needs a manual install; see [Simulator-Backed Examples](#simulator-backed-examples).

If GPU mode is configured through `gpu_device` but PyTorch is unavailable, supported paths fall back to CPU mode with a warning instead of failing immediately.

## Quick Start

Verify the install:

```bash
python -c "import pyrtc; print(pyrtc.__all__)"
```

Validate a system config before launch:

```bash
pyrtc-validate-config examples/synthetic_shwfs/config.yaml
```

Export a telemetry session into AOTPy once the optional dependency is installed:

```bash
pyrtc-export-aotpy data/session_20260309_120000_abcd1234 session_export.fits
```

The best first end-to-end path today is the no-hardware synthetic Shack-Hartmann workflow under `examples/synthetic_shwfs/`.

Key files:

- `examples/synthetic_shwfs/config.yaml`
- `examples/synthetic_shwfs/synthetic_shwfs_soft_rtc_example.py`
- `examples/synthetic_shwfs/synthetic_shwfs_hard_rtc_example.py`

Run it with:

```bash
python examples/synthetic_shwfs/synthetic_shwfs_soft_rtc_example.py --duration 15
```

That tutorial now logs one `manager.latency(samples=256)` example after startup so you can inspect the full-loop latency breakdown directly from the manager API while the synthetic system is running.

Every primary CLI and example entry point now uses the shared `pyrtc` logger. By default you get timestamped `INFO` logs on the console. You can override that per run with `--log-level DEBUG`, write per-process logs with `--log-dir logs/`, or force one exact file with `--log-file session.log`.

The same settings can be exported for multi-process or repeated runs:

```bash
export PYRTC_LOG_LEVEL=INFO
export PYRTC_LOG_DIR=./logs
export PYRTC_LOG_COLOR=1
python examples/synthetic_shwfs/synthetic_shwfs_hard_rtc_example.py --duration 15
```

It publishes the normal `wfs`, `signal_2d`, `wfc_2d`, `psf_short`, and `psf_long` streams, so the standard viewer tools work unchanged while you evaluate the control flow and subclassing points.

Recommended composite viewer command while the demo is running:

```bash
pyrtc-view wfs signal_2d wfc_2d psf_short psf_long --geometry 2x3
```

Documentation guides on Read the Docs:

- [Getting Started](https://pyrtc-ao.readthedocs.io/en/latest/guides/getting_started.html)
- [Architecture Guide](https://pyrtc-ao.readthedocs.io/en/latest/guides/architecture.html)
- [Developer Guide](https://pyrtc-ao.readthedocs.io/en/latest/guides/developers_guide.html)
- [Synthetic SHWFS Example](https://pyrtc-ao.readthedocs.io/en/latest/examples/synthetic_shwfs.html)
- [PYWFS Example](https://pyrtc-ao.readthedocs.io/en/latest/examples/pywfs.html)
- [SHWFS Simulator Examples](https://pyrtc-ao.readthedocs.io/en/latest/examples/shwfs.html)
- [HCIPy Example](https://pyrtc-ao.readthedocs.io/en/latest/examples/hcipy.html)

## Architecture Overview

`pyrtc` is organized around a small set of component abstractions:

- `WavefrontSensor`
- `SlopesProcess`
- `Loop`
- `WavefrontCorrector`
- `ScienceCamera`
- `Telemetry`

These components exchange data through shared-memory streams and can be assembled in two main ways:

- `soft-RTC`: all relevant components run in one Python process
- `hard-RTC`: hardware-facing pieces run in separate Python processes and communicate through launchers/shared memory

Use `soft-RTC` first unless you have a clear need for process isolation or hardware-driver separation.

## Examples and Hardware

Real AO deployments are hardware-specific. The repo includes two kinds of support for that:

- abstract core classes for the AO pipeline
- example integrations in `pyrtc/hardware`

These hardware files should be treated as reference implementations and starting points, not as a guarantee that every SDK and device combination will work unchanged.

For no-hardware exploration, start with the synthetic SHWFS example. For a richer simulated optical path, use the simulator-backed examples below. They need an external simulator, so treat them as the second example, not the first one.

### Simulator-Backed Examples

`examples/pywfs/` (pyramid WFS) and `examples/shwfs/` (Shack-Hartmann WFS) each have an OOPAO and a SPECULA version, and `examples/hcipy/` runs a Shack-Hartmann system on HCIPy. All run in soft-RTC mode.

HCIPy is the quickest to install:

```bash
pip install pyrtcao[hcipy]
python examples/hcipy/hcipy_shwfs_soft_rtc_example.py --duration 10 --atmosphere
```

SPECULA is on PyPI:

```bash
pip install pyrtcao[specula]   # or: pip install specula
python examples/shwfs/shwfs_specula_soft_rtc_example.py --duration 10
```

OOPAO is not on PyPI. Installing it with `pip install git+https://github.com/cheritier/OOPAO.git` is not enough: `import OOPAO` then fails with `ValueError: attempt to get argmin of an empty sequence`, because OOPAO looks for its own checkout on `sys.path` at import time. Clone it and put the clone on `PYTHONPATH` instead:

```bash
git clone https://github.com/cheritier/OOPAO.git   # keep the directory name "OOPAO"
pip install ./OOPAO                                # installs OOPAO's dependencies
export PYTHONPATH="$PWD/OOPAO:$PYTHONPATH"
python -c "import OOPAO"                           # check the install
python examples/pywfs/pywfs_oopao_soft_rtc_example.py --duration 10
```

The clone directory's path must contain `OOPAO` (case-sensitive) and be writable, since OOPAO writes a small file there on import. See the [PYWFS Example](https://pyrtc-ao.readthedocs.io/en/latest/examples/pywfs.html) docs for details.

## Tools and Benchmarks

Viewer and CLI tools:

```bash
pyrtc-view wfs --log-level INFO
pyrtc-shm-monitor --log-dir logs
pyshmem list                # streams are pyshmem streams; unlink/purge them with the pyshmem CLI
pyrtc-measure-latency signal wfc --log-file latency.log
pyrtc-isio-bridge to-isio wfs pyrtc_wfs   # mirror a stream to ImageStreamIO (milk/CACAO)
```

Performance smoke report:

```bash
python benchmarks/perf_smoke.py --output perf_smoke_report.json --log-dir logs
```

Synthetic closed-loop AO benchmark:

```bash
pyrtc-ao-loop-bench --output ao_loop_bench_report.json --iterations 300 --warmup 30 --system-sizes 10 20 60
python benchmarks/readme_benchmark_table.py --report ao_loop_bench_report.json --output ao_loop_benchmark_table.md
python benchmarks/check_perf_baseline.py --current ao_loop_bench_report.json --baseline benchmarks/ao_loop_bench_baseline.json
```

Core compute benchmark:

```bash
pyrtc-core-bench --quick --cpu-only --output core_compute_bench_report.json --log-level INFO
```

Run without `--cpu-only` to include GPU kernels when CUDA and PyTorch are available.

Benchmark trends across CI runs (reads the uploaded perf artifacts; needs `GH_TOKEN`):

```bash
python -m benchmarks.perf_history --repo jacotay7/pyRTC --runs 20
```

The committed closed-loop baseline for the README host is [benchmarks/ao_loop_bench_baseline.json](benchmarks/ao_loop_bench_baseline.json).

The shared logging environment variables are:

- `PYRTC_LOG_LEVEL`: default log level, usually `INFO` or `DEBUG`
- `PYRTC_LOG_DIR`: write one log file per process into a directory
- `PYRTC_LOG_FILE`: write to one exact file path for single-process runs
- `PYRTC_LOG_COLOR`: set to `0` to disable ANSI colors
- `PYRTC_LOG_CONSOLE`: set to `0` to disable console logging when file logs are enough

Hard-RTC child processes inherit these settings automatically through the launcher, so one `PYRTC_LOG_DIR` is enough to collect parent and child logs together.

## Stability and Support Notes

- Not every platform or hardware stack is validated equally.
- Linux is the primary supported environment.
- macOS and Windows have smoke workflow coverage, but release validation and deployment guidance remain Linux-first.
- GPU support is validated in this repo through synthetic CPU/GPU benchmark coverage and should still be checked in the target environment before operational use.
- Example scripts and hardware adapters are intended to shorten development time, not replace system-specific commissioning.

## Contributing and Development

Maintainer and contributor workflow guidance (local setup, tests, docs builds, and releases) is in the [Developer Guide](https://pyrtc-ao.readthedocs.io/en/latest/guides/developers_guide.html). Coding agents should also read `AGENTS.md` in the repository root.

For release validation from a source checkout, the built-wheel smoke path is automated:

```bash
python -m build
python -m twine check dist/*
python pyrtc/scripts/validate_dist_install.py --dist-dir dist
```

The steps for publishing a release are in the Developer Guide's [Release Checklist](https://pyrtc-ao.readthedocs.io/en/latest/guides/developers_guide.html#release-checklist).

The GitHub Actions publish workflow lives in `.github/workflows/publish-package.yml`.

## Contact

For feedback, collaboration, and feature requests: `jtaylor@keck.hawaii.edu`
