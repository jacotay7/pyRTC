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

Documentation: [https://pyrtc-ao.readthedocs.io/en/latest/](https://pyrtc-ao.readthedocs.io/en/latest/)

Developer Guide: [https://pyrtc-ao.readthedocs.io/en/latest/guides/developers_guide.html](https://pyrtc-ao.readthedocs.io/en/latest/guides/developers_guide.html)

## Performance

The benchmark section is intentionally near the top because performance is a primary design constraint for `pyrtc`.
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

Values are reported as `p99 throughput / p99 latency`.

| Loop | 10x10 CPU | 10x10 GPU | 20x20 CPU | 20x20 GPU | 60x60 CPU | 60x60 GPU |
| --- | --- | --- | --- | --- | --- | --- |
| PYWFS full loop | 58.1 kHz / 17.2 us | 4.5 kHz / 219.9 us | 26.6 kHz / 37.6 us | 4.6 kHz / 218.5 us | 270 Hz / 3703.4 us | 3.0 kHz / 335.2 us |
| SHWFS full loop | 78.2 kHz / 12.8 us | 5.1 kHz / 195.0 us | 26.6 kHz / 37.6 us | 5.1 kHz / 196.7 us | 268 Hz / 3730.7 us | 3.5 kHz / 289.7 us |

For this host, the important pattern is the one we care about operationally: CPU wins the small `10x10` and `20x20` synthetic loops because launch overhead dominates, but the GPU is about an order of magnitude faster once the loop reaches the `60x60` regime. That crossover now shows up for both pyramid and Shack-Hartmann synthetic loops in the README numbers.

The benchmark artifacts committed for this host are:

- `benchmarks/readme_benchmark_report.json`
- `benchmarks/readme_benchmark_table.md`

## What It Is For

Adaptive optics (AO) systems measure optical aberrations and apply corrections quickly enough to recover image quality in dynamic environments. `pyrtc` is aimed at the software layer that connects those measurements, reconstructions, and corrections.

The project is designed for:

- laboratory AO systems and hardware integration work
- simulated AO development and algorithm prototyping
- moderate-performance real-time control in Python
- controller research, including machine-learning-assisted control paths

## Release Posture

`pyrtc` `1.0.0` is released on PyPI as `pyrtcao`. The next version, `1.1.0`, is in development on this repository; see `CHANGELOG.md` for what has changed since `1.0.0`. The release policy is conservative:

- User-facing project name: `pyrtc`
- PyPI distribution name: `pyrtcao`
- Python import name: `pyrtc`
- CLI prefix: `pyrtc-*`
- Primary supported release surface: Linux, Python 3.9-3.14
- macOS and Windows: smoke-tested in GitHub Actions, but not part of the primary supported deployment story
- Windows: soft-RTC only — Windows named shared memory is freed when the last handle closes, so streams do not survive their producer process and hard-RTC restart/reattach flows are unsupported there
- GPU behavior: benchmark-validated on a Linux CUDA host for synthetic loop workloads, but still target-environment validation required for operational use
- Hardware integrations: examples and reference implementations, not universal plug-and-play support

## Core Capabilities

- Component-based AO pipeline built around wavefront sensing, slope processing, control, correction, telemetry, and science imaging
- Soft-RTC mode for single-process development and simulation workflows
- Hard-RTC mode for process-isolated hardware integration via shared memory and launcher utilities
- Optional viewer and benchmarking tools for stream inspection and performance checks
- Example hardware adapters and simulation-oriented examples under `pyrtc/hardware` and `examples/`

## Installation

### From PyPI

```bash
pip install pyrtcao
```

Optional extras:

```bash
pip install pyrtcao[aotpy]
pip install pyrtcao[docs]
pip install pyrtcao[gpu]
pip install pyrtcao[specula]
pip install pyrtcao[viewer]
```

### From Source

```bash
git clone https://github.com/jacotay7/pyRTC.git
cd pyRTC
pip install .
```

Optional source extras:

```bash
pip install .[aotpy]
pip install .[docs]
pip install .[gpu]
pip install .[specula]
pip install .[viewer]
```

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

`examples/pywfs/` (pyramid WFS) and `examples/shwfs/` (Shack-Hartmann WFS) each have an OOPAO and a SPECULA version. Both run in soft-RTC mode only.

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
