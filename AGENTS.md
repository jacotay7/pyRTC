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
pytest                                   # full suite with coverage (gate: 70%)
pytest tests/test_streams.py --no-cov    # a subset; --no-cov avoids the gate failing
pytest -m "not gpu"                      # skip CUDA tests (they auto-skip without CUDA)
pytest tests/system tests/notebooks -q --no-cov   # end-to-end runs, as in CI
ruff check . && ruff format --check .    # lint, as in CI
```

- `pytest.ini` adds coverage options to every run. Pass `--no-cov` for partial
  runs. Coverage measures the whole `pyrtc` package (`.coveragerc`); only the
  display-bound Qt windows are omitted.
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
- Never pass dotted submodules to coverage (`--cov=pyrtc.utils`). Coverage
  then imports the package early and numpy is imported a second time, which
  broke numpy sentinels (`_NoValueType` errors) and segfaulted torch imports
  mid-run. Use `--cov=pyrtc`; `pytest.ini` turns the "NumPy module was
  reloaded" warning into an error so this cannot silently return.
- Unknown-key warnings only cover classes whose keys are declared: built-in
  components, and subclasses that set `EXTRA_CONFIG_KEYS` (or their own
  `COMPONENT_DESCRIPTOR`) in their class body. When an adapter starts reading
  a new config key, add it to `EXTRA_CONFIG_KEYS` (or to the descriptor for a
  built-in), or configs using it will warn.
- Components can be built with `__new__` in tests, so `Component` methods call
  `_ensure_stream_state()` before touching stream state.
- A `KeyboardInterrupt` at a random point in a Windows test run was pyshmem
  (< 1.3.3) probing process liveness with `os.kill(pid, 0)`, which on Windows
  sends Ctrl+C to the console group. It is fixed in pyshmem 1.3.3. If the
  symptom returns, look for signal-0 probes before blaming the test.
- Tests force the non-GUI `Agg` matplotlib backend (`tests/conftest.py`)
  because some library code still calls `plt.show()` (#34).
- SPECULA processing objects only run when an input has a fresh
  `generation_time`. In `specula_interface.py`, anything that changes the
  optical setup without a new DM command (atmosphere on/off) must refresh an
  input, and every step must advance the WFS and PSF branches together, or
  the WFS silently repeats a stale (or blank) frame.
- The loop's IM method key is `im_method`; `method:` is ignored (it only
  produces an unknown-key warning).
  Calibrate only once the pipeline is live (worker kernels JIT-compile on
  first use, so the first DM command can take about a second to land).
- Windows frees named shared memory when the last handle closes, so streams do
  not outlive their producer there. Treat Windows as soft-RTC only.
- `import OOPAO` fails with `ValueError: attempt to get argmin of an empty
  sequence` when OOPAO was installed with plain `pip install` (it is not on
  PyPI). `OOPAO/__init__.py` picks the shortest `sys.path` entry containing
  `OOPAO` (case-sensitive) and writes `precision_oopao.npy` into it, so it only
  imports from a writable clone whose path contains `OOPAO` and that is on
  `PYTHONPATH`. OOPAO therefore cannot be a pyrtc extra; the recipe is in
  `docs/source/examples/pywfs.rst`. SPECULA is on PyPI (`specula` extra).

## Maintainer guidance

- The shared-memory layer lives in pyshmem. The migration to it is meant to
  make pyrtc simpler, so use pyshmem features rather than duplicating them.
- Fix pyshmem issues at the source (in the pyshmem repository), not with
  workarounds here.
- Keep this file current and machine-independent (see the top of this file).
- Do not commit planning, status-tracker, or scratch notes (such as an
  `IMPROVEMENT_PLAN.md`). Durable guidance belongs in this file, and
  user-facing changes belong in `CHANGELOG.md`.
- When you work around a minor issue instead of fixing it (out of scope, not
  worth blocking on), open a GitHub issue in the affected repository if you
  have credentials: what you hit, where, the workaround, and what the real fix
  would be. Link the issue from the workaround when it lives in code. If you
  cannot open an issue, tell the maintainer instead. The goal is to move on
  without forgetting it.
