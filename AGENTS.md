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
  component: config parsing, worker threads (one per entry in `functions`),
  the stream helpers `read_stream` / `write_stream`, and the lifecycle
  (`start`/`stop` pause and resume; `close` ends the workers and closes the
  registered streams for good).
- Core components: `wavefront_sensor.py`, `slopes_process.py`, `loop.py`,
  `wavefront_corrector.py`, `science_camera.py`, `telemetry.py`,
  `modulator.py`, `optimizer.py`. Hot loops are `@jit(..., cache=True)`
  Numba kernels; the first call after a source change recompiles (about 1 s
  each), so warm them before timing anything.
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
- `pyrtc/modal_basis.py` — modal bases (`M2C`) for wavefront correctors,
  built with the external [aobasis](https://github.com/jacotay7/aobasis)
  package from a `basis:` config section and the actuator geometry. Basis
  bugs get fixed in aobasis (maintainers allow PRs and releases there), not
  worked around here. Since aobasis 1.1, Zernikes carry the Noll factor, and
  pyrtc orthonormalizes Zernike and Fourier bases by default, so a test that
  expects raw values must set `orthonormalize: false`.
- Dependencies: keep `[project] dependencies` to what the soft-RTC core
  needs. Anything else goes in an extra and is imported lazily through
  `pyrtc.utils.require_optional(module, extra, feature)`, which names the
  extra in its error. `import pyrtc` must not import optional packages
  (checked in `tests/test_public_api.py`), and that includes torch: probe it
  with `pyrtc.streams.gpu_torch_available()` and import it inside GPU code
  paths. Keep `requirements.txt` identical to the core `dependencies`. Add
  test-only needs to `requirements-test.txt`.
- `pyrtc/modal_gains.py` (per-mode gain optimization) and `pyrtc/predictive.py`
  (pluggable predictors for `Loop.predictive_integrator`) hold control
  algorithms as plain numpy, so they are testable without streams.
- `pyrtc/corrector_splitter.py` — one loop driving several correctors
  (woofer/tweeter, offload); it sits in the `wfc` section.
  `pyrtc/isio_bridge.py` — mirrors a stream to or from ImageStreamIO
  (milk/CACAO).
- `pyrtc/latency.py` — stream latency measurement. `pyrtc/exporters/` — AOTPy
  export of telemetry sessions.
- `pyrtc/hardware/` — reference adapters:
  - cameras: GenICam, Micro-Manager, XIMEA, Spinnaker;
  - DMs: ALPAO, BMC;
  - simulators: synthetic, HCIPy, SPECULA, OOPAO;
  - optimizers.

  Vendor SDKs are optional and may be missing. New adapters import them
  inside `__init__` (`require_optional`), not at module load, so the module
  imports and documents without the SDK. Camera frames come as
  `(Height, Width)` and pyrtc streams are `(width, height)`, so transpose
  (#130).
- `pyrtc/gui/`, `pyrtc/scripts/` — manager GUI, viewer, and CLI entry points
  (declared in `pyproject.toml` under `[project.scripts]`). The GUI and viewer
  use Qt6 through `qtpy` (PySide6 by default, PyQt6 also works), selected by
  `pyrtc/qt_compat.py`. Import Qt from `qtpy`, never from a binding directly,
  and use fully scoped enums (`Qt.AlignmentFlag.AlignCenter`), which PyQt6
  requires. Without Qt the modules still import, with stand-in classes that
  raise `ImportError` when a window is built.
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
- Components must register every stream they touch
  (`register_input_stream` / `register_output_stream`); the helpers raise
  `KeyError` for anything else. A registered handle is owned by the
  component and closed by `Component.close()`.
- Close what you build: `RTCManager.close()` (or `with RTCManager... as m`)
  and `Component.close()`. `stop()` only pauses; worker threads hold their
  component, so garbage collection never ends them.
- Observers (viewers, telemetry, latency, monitors) open streams with
  `open_stream(name, readonly=True)`.
- Do not use `read_new()` in request/response or lock-step code. It is
  edge-triggered from the moment it is called and can deadlock. Use
  `read_after` / `wait_for_count` with a known count instead.
- `out=` buffers only work for CPU handles. `read_stream` drops `out` for
  GPU-attached handles.
- `create_stream` makes notify-enabled streams (`pyshmem.create(...,
  notify=True)`): writers wake parked `read_after_publication` consumers via a
  futex. `PYRTC_STREAM_NOTIFY=0` (inherited by hard-RTC children) or
  `create_stream(..., notify=False)` opts out. An existing stream whose notify
  flag differs is rebuilt, not reused.
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
  producer stamping frame ids through several streams), `StaticStream`
  (republishes one frame, for telemetry), `bare_component` (a component
  built without `__init__` but with real stream state, for single-method
  tests), `prefix_system_streams` (renames every stream in a loaded system
  config), and `private_synthetic_config` (the synthetic example with private
  stream names, for tests that run a whole system). Prefer real pyshmem
  streams over new hand-written fakes.
- Tests that start a system must use private stream names
  (`private_synthetic_config`) and close the manager. `RTCManager.build()`
  reconciles, and may clear, every output stream its config names, so a test
  on the canonical names can break another system running on the host.
- `tests/test_qt_smoke.py` builds the manager GUI and the viewer on Qt's
  `offscreen` platform (no display needed) and skips without a Qt6 binding.
  `requirements-test.txt` has no Qt, so it runs only in the `qt-offscreen`
  CI job (`gui-smoke.yml`). Run it locally after GUI or viewer changes, also
  with `QT_API=pyqt6` if PyQt6 is installed.
- Other CI jobs worth knowing:
  - `free-threaded`: Python 3.14t with `PYTHON_GIL=0`; it skips `tests/system`
    (#139).
  - `ISIO Bridge` (`isio-bridge.yml`): builds ImageStreamIO and runs
    `tests/test_isio_bridge.py`, which skips elsewhere.
  - Hardware adapter tests run against fake SDK modules
    (`tests/test_genicam_camera.py`, `test_bmc_dm.py`, ...). Follow that
    pattern for new adapters.
- The closed-loop regression `tests/system/test_synthetic_convergence.py` is
  the best end-to-end check that stream semantics still work.
- `benchmarks/pipeline_latency_bench.py` measures the running synthetic
  system's WFS -> DM latency (`manager.latency`) in soft and hard mode with
  notify on/off; `benchmarks/stream_handoff_bench.py` measures one stream
  handoff. Both use private stream-name prefixes, so they are safe to run
  next to other systems. They are not part of the CI perf gate.
- Perf gate, as in CI:
  `python benchmarks/perf_smoke.py --output perf.json` then
  `python benchmarks/check_perf_baseline.py --current perf.json --baseline benchmarks/perf_smoke_baseline.json --max-ratio 5.0`.
- Trends across CI runs: `python -m benchmarks.perf_history --repo <owner/repo>`
  (reads the uploaded perf artifacts; needs `GH_TOKEN`). CI runs only on pull
  requests into `dev`/`main` and pushes to `main`, so the history is PR runs.

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
- Build components for method-level tests with `testsupport.bare_component`,
  not `Cls.__new__(Cls)`: the stream helpers assume the state that
  `Component._init_runtime_state` sets up (there is no lazy-init guard).
- A `KeyboardInterrupt` at a random point in a Windows test run was pyshmem
  (< 1.3.3) probing process liveness with `os.kill(pid, 0)`, which on Windows
  sends Ctrl+C to the console group. It is fixed in pyshmem 1.3.3. If the
  symptom returns, look for signal-0 probes before blaming the test.
- Tests force the non-GUI `Agg` matplotlib backend (`tests/conftest.py`) so
  that plotting helpers and scripts (`pyrtc-shm-monitor` still calls
  `plt.show()`) never open windows.
- SPECULA processing objects only run when an input has a fresh
  `generation_time`. In `specula_interface.py`, anything that changes the
  optical setup without a new DM command (atmosphere on/off) must refresh an
  input, and every step must advance the WFS and PSF branches together, or
  the WFS silently repeats a stale (or blank) frame.
- Simulated systems run asynchronously: the loop iterates once per WFS frame,
  so its delay in iterations is the DM-to-WFS round trip in frames, which
  grows on a loaded CI runner. An integrator is stable only below
  `2 sin(pi / (2 (2d + 1)))` for a delay of `d` frames (0.62 at 2, 0.29 at
  5), so system tests use gains around 0.15. The HCIPy test diverged in CI at
  0.3 while passing locally.
- The synthetic example calibrates with DOCRIME (`im_method: docrime`,
  `num_iters_im: 800`). Do not shrink `num_iters_im` far in tests: at 50 the
  IM is noise and the loop diverges, which looks like a wiring bug (it cost a
  debugging session). The system tests use 400.
- A section's built-in checks (descriptor fields, `validate_wfc_config`,
  default stream roles, worker functions) apply only when its class belongs to
  that section's component family (`config_schema._section_descriptor`).
  Otherwise the class's own descriptor is used, which is how a
  `CorrectorSplitter` can sit in the `wfc` section.
- The loop's IM method key is `im_method`; `method:` is ignored (it only
  produces an unknown-key warning).
  Calibrate only once the pipeline is live (worker kernels JIT-compile on
  first use, so the first DM command can take about a second to land).
- Windows frees named shared memory when the last handle closes, so streams do
  not outlive their producer there. Treat Windows as soft-RTC only.
- OOPAO (not on PyPI) has two packaging bugs. Its `__init__` looks for a
  `sys.path` entry containing `OOPAO` and fails with `ValueError: attempt to
  get argmin of an empty sequence` otherwise, and a pip install from git omits
  its subpackages (`OOPAO.tools`, `OOPAO.calibration`). `oopao_interface`
  works around the first (it exposes the package directory on `sys.path` for
  the import) and raises a clear `ImportError` for the second. The supported
  install is a clone on `PYTHONPATH` (recipe in `docs/source/examples/pywfs.rst`).
  Per the maintainer, don't file OOPAO issues upstream; the write-up is kept
  outside the repo for the maintainer.
- `oopao_interface` targets current OOPAO propagation: `src ** tel * dm * wfs`,
  or `src ** atm * tel * dm * wfs` with atmosphere (`**` resets the source).
  DM commands are in metres. `tests/system/test_oopao_convergence.py` runs
  when OOPAO is importable (e.g. `PYTHONPATH=<clone>`) and skips otherwise.
- `hcipy_interface` builds its whole system from a flat parameter mapping
  (defaults in `DEFAULT_PARAMS`). HCIPy's Shack-Hartmann optics need the
  pupil magnified to the physical microlens-array size (the interface uses
  5 mm), or the "spots" are just the pupil image. Keep the sub-aperture size
  near r0 and the number of controlled modes modest, or the loop runs away on
  the atmosphere (see `docs/source/examples/hcipy.rst`). The atmosphere
  advances only on WFS exposures with it enabled.
- `pyrtc/isio_bridge.py` talks to ImageStreamIO through `ImageStreamIOWrap`
  (built from git; the ISIO CI workflow builds it). Its quirks (#138): write
  only Fortran-ordered arrays (`np.asfortranarray`), and never call its
  blocking `semwait`/`semtimedwait` from pyrtc threads, since they hold the
  GIL. Poll `semtrywait`. Only the creating handle's `destroy()` removes an
  ISIO file. A pip install can't import on its own (missing `$ORIGIN` RPATH);
  `isio_bridge._isio_module()` preloads `libImageStreamIO.so` first.
- pyshmem shares one lock state per stream name inside a process. Before
  pyshmem 1.3.5, `close()` on *any* handle failed while another thread held
  that lock (e.g. a latency observer closing while a soft-RTC producer was
  mid-write). It was fixed at the source, and pyrtc requires
  a pyshmem that includes the fix (now `>=1.3.7`). Don't add retry workarounds for it.
- Latency and handoff numbers on a shared host swing by 2x or more with load;
  compare notify on/off with interleaved `--repeats`, never single runs.
  With a load-following CPU governor, busy neighbours also raise the clock,
  so a system can run *faster* under load than idle. Compare configurations
  only at the same load.
- numpy and scipy each load their own OpenBLAS, with one worker thread per
  core in the process affinity. The numpy-heavy simulators (HCIPy above all)
  keep those pools spinning. In a 16-core cpuset the HCIPy example's pools used
  about 15 cores while the pipeline threads used half of one, and
  `OPENBLAS_NUM_THREADS=1` made the WFS faster (32 vs 23 frames/s) and the
  HCIPy system test 2.5x shorter. Cap BLAS threads before timing or
  load-testing a simulated system, or the BLAS pools are what you measure.
  The Loop's control multiply (`np.dot` inside numba goes to scipy's
  OpenBLAS) does use them for large matrices, so do not cap them blindly on a
  real RTC.
- numba's `workqueue` threading layer crashes the process when two threads
  call `parallel=True` kernels at once; `omp` and `tbb` are safe. Only the WFS
  thread runs one today (`rotate_image_jit`). A parallel kernel on a second
  component thread must require a thread-safe layer (#104).

## Maintainer guidance

- The shared-memory layer lives in pyshmem. The migration to it is meant to
  make pyrtc simpler, so use pyshmem features rather than duplicating them.
- Fix pyshmem issues at the source (in the pyshmem repository), not with
  workarounds here.
- All modal bases (KL, Zernike, Fourier, zonal, Hadamard) come from aobasis
  (same maintainer) through `pyrtc.modal_basis`; do not add per-backend basis
  code. Fix basis-generation bugs in aobasis and raise the `aobasis>=` floor.
- Keep this file current and machine-independent (see the top of this file).
- Do not commit planning, status-tracker, or scratch notes (such as an
  `IMPROVEMENT_PLAN.md`). Durable guidance belongs in this file, and
  user-facing changes belong in `CHANGELOG.md`.
- When you work around a minor issue instead of fixing it (out of scope, not
  worth blocking on), open a GitHub issue in this repository if you have
  credentials: what you hit, where, the workaround, and what the real fix
  would be. Link the issue from the workaround when it lives in code. If you
  cannot open an issue, tell the maintainer instead. The goal is to move on
  without forgetting it.
- Never report bugs to third-party repositories (OOPAO, ImageStreamIO/milk,
  HCIPy, SPECULA, vendor SDKs, ...). Work around them in pyrtc and record them
  in a pyrtc issue (e.g. #138). The maintainer's own packages (pyshmem,
  aobasis) are the exception: fix those at the source.
