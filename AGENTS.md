# pyrtc — Agent Guide

Guidance for coding agents (and humans) working in this repository. `CLAUDE.md`
only points here.

## Maintaining this file

- Keep this file accurate. When you change structure, commands, or conventions
  described here, update it in the same change.
- If something in the repo misled you or cost you time (a stale doc, a
  surprising default, a test that passes for the wrong reason), add it to
  **Gotchas** below so the next agent does not repeat it.
- Record instructions from maintainers about how the repository should work in
  **Maintainer guidance**, with enough context to apply them later.
- Keep it machine-independent: no local paths, environment names, hostnames, or
  hardware details of any particular computer.

## What this is

pyrtc is a Python adaptive-optics real-time controller. Components (wavefront
sensor, slopes processor, loop, wavefront corrector, science camera, telemetry,
optimizers) run as threads in one process ("soft RTC") or as separate processes
("hard RTC"), and exchange frames through named shared-memory streams.

Names: the repository is `pyRTC`, the PyPI distribution is `pyrtcao`, and the
import package is `pyrtc` (lowercase). The old `pyRTC` import name and
CamelCase module names (`pyRTC.Loop`, `pyRTC.Pipeline`, ...) are gone with no
aliases.

## Code layout

- `pyrtc/component.py` — `Component`, the base class for every runtime
  component: config parsing, worker threads (one per entry in `functions`), and
  the stream helpers `read_stream` / `write_stream`.
- Core components: `wavefront_sensor.py`, `slopes_process.py`, `loop.py`,
  `wavefront_corrector.py`, `science_camera.py`, `telemetry.py`,
  `modulator.py`, `optimizer.py`.
- `pyrtc/streams.py` — pyrtc's policy on top of pyshmem: `create_stream`,
  `open_stream`, `clear_shms`, and planning of the output streams a config
  implies (`expected_output_shm_specs_for_config`).
- `pyrtc/manager.py` — `RTCManager` and the soft/hard component runtimes.
  `pyrtc/rpc.py` — hard-RTC launcher/listener JSON socket protocol.
  `pyrtc/component_loading.py` — resolving `class_name` / `class_file`.
- `pyrtc/config_schema.py` (validation and normalization),
  `pyrtc/config_runtime.py` (runtime config hooks, `stream_alias_map`),
  `pyrtc/component_descriptors.py` (declared config fields and streams per
  component class).
- `pyrtc/latency.py` — stream latency measurement. `pyrtc/exporters/` — AOTPy
  export of telemetry sessions.
- `pyrtc/hardware/` — reference adapters (cameras, DMs, simulators, synthetic
  systems). Vendor SDKs are optional and may be missing.
- `pyrtc/gui/`, `pyrtc/scripts/` — manager GUI, viewer, and CLI entry points
  (declared in `pyproject.toml` under `[project.scripts]`).
- `examples/` — runnable systems (start with `examples/synthetic_shwfs/`, no
  hardware needed). `benchmarks/` — perf smoke and benchmark scripts.
- `docs/source/` — Sphinx docs; `guides/` holds the narrative guides
  (architecture, streams, developer guide) and `components/` the per-component
  pages.

## Shared memory (pyshmem)

All shared memory comes from the external
[pyshmem](https://github.com/jacotay7/pyshmem) package, which has its own tests
and docs. pyrtc must not reimplement transport features that pyshmem provides.

- A write is a *publication* with `count`, `write_time`, and a user `frame_id`.
  Use `read_publication()` / `read_new_publication()` when metadata must match
  the payload. Do not read the payload and then sample `write_time` separately.
- `Component.read_stream(name)` *consumes*: it returns the first write newer
  than the one it last returned, using pyshmem's level-triggered
  `read_after_publication`. `block=False` is a peek and does not consume.
- Frame identity: the WFS stamps each exposure with a new `frame_id`. Reading a
  registered input sets `Component.frame_id`, and `write_stream` stamps it on
  outputs. `pyrtc.latency` pairs writes across streams by `frame_id`. Keep
  new components on `read_stream` / `write_stream` so they propagate it.
- Observers (viewers, telemetry, latency, monitors) open streams with
  `open_stream(name, readonly=True)`.
- Do not use `read_new()` in request/response or lock-step code. It is
  edge-triggered from the moment it is called and can deadlock. Use
  `read_after` / `wait_for_count` with a known count instead.
- `out=` buffers only work for CPU handles. `read_stream` drops `out` for
  GPU-attached handles.
- Inspect or clean up streams with the `pyshmem` CLI (`pyshmem list`,
  `pyshmem unlink NAME`, `pyshmem purge`).
- If pyshmem is missing something or behaves wrongly, fix it in pyshmem,
  release it, and raise the `pyshmem>=` floor in `pyproject.toml` and
  `requirements.txt`. Do not work around it in pyrtc.

## Running tests

```bash
pip install -e . -r requirements-test.txt
pytest                                   # full suite with coverage (gate: 80%)
pytest tests/test_streams.py --no-cov    # a subset; --no-cov avoids the gate failing
pytest -m "not gpu"                      # skip CUDA tests (they auto-skip without CUDA)
pytest tests/system tests/notebooks -q --no-cov   # end-to-end runs, as in CI
ruff check . && ruff format --check .    # lint, as in CI
```

- `pytest.ini` adds coverage options to every run. Pass `--no-cov` for partial
  runs.
- `tests/testsupport.py` provides `private_stream` (a real pyshmem stream with a
  unique name, unlinked after each test), `publishing_chain` (a background
  producer stamping frame ids through several streams), and `StaticStream`
  (republishes one frame, for telemetry). Prefer real pyshmem streams over new
  hand-written fakes.
- The closed-loop regression `tests/system/test_synthetic_convergence.py` is
  the best end-to-end check that stream semantics still work.
- Perf gate, as in CI:
  `python benchmarks/perf_smoke.py --output perf.json` then
  `python benchmarks/check_perf_baseline.py --current perf.json --baseline benchmarks/perf_smoke_baseline.json --max-ratio 5.0`.

## Documentation

- Build: `pip install -e .[docs]` then
  `sphinx-build -b html docs/source docs/source/_build/html`.
- API pages are generated by autosummary into `docs/source/generated/`
  (git-ignored). Add modules to the list in `docs/source/api_reference.rst`.
- User-facing changes go in `CHANGELOG.md` under the unreleased version.
  Update the relevant guide (for stream behaviour, `guides/streams.rst`) when
  behaviour changes.

## Gotchas

- Hand-written stream fakes in tests drifted from pyshmem and hid real bugs,
  such as a blocking read that never blocked. Use `testsupport.private_stream`.
- The perf baseline (`benchmarks/perf_smoke_baseline.json`) is keyed by kernel
  function name. Renaming a benchmarked function without renaming its key makes
  the CI perf gate fail with "Missing baseline metrics". A single p99 over the
  limit is usually noise; rerun before treating it as a regression.
- If pytest segfaults while importing `torch` with coverage enabled, that is an
  interaction between pytest-cov and torch in the environment, not a pyrtc bug.
  Rerun with `--no-cov`.
- Components can be built with `__new__` in tests, so `Component` methods call
  `_ensure_stream_state()` before touching stream state.
- `IMPROVEMENT_PLAN.md` is a historical status tracker from the pyshmem
  migration. Check the code before trusting its status notes.
- Windows frees named shared memory when the last handle closes, so streams do
  not outlive their producer there. Treat Windows as soft-RTC only.

## Maintainer guidance

- The shared-memory layer lives in pyshmem. The migration to it is meant to
  make pyrtc simpler, so use pyshmem features rather than duplicating them.
- Fix pyshmem issues at the source (in the pyshmem repository), not with
  workarounds here.
- Keep this file current and machine-independent (see the top of this file).
