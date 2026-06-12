# Changelog

All notable changes to `pyrtcao` will be documented in this file.

## 1.1.0 - 2026-06-11

### Fixed

- **`manager.latency()` no longer crashes in the synthetic tutorial.**
	Component classes referenced by `class_file` are now resolved to their
	canonical modules (`pyrtc.component_loading`, one shared implementation
	instead of three divergent copies), bare-name lookup is no longer broken
	by same-named submodule shadowing, and relative `class_file` paths resolve
	against the config file's directory instead of the caller's cwd.
- **The synthetic SHWFS tutorial now converges.** The examples calibrate the
	interaction matrix through the live pipeline (DOCRIME, `conditioning: 30`)
	instead of loading an identity placeholder: residual RMS drops 0.99 → 0.01
	and Strehl reaches ~0.97 in both soft and hard modes. A closed-loop
	convergence regression test runs in `tests/system/`.
- Hard-RTC child listeners now stop cleanly when the RTC closes the control
	socket instead of crashing with `BrokenPipeError`.

### Added

- Zero-allocation hot-path reads: `read_stream(..., out=buffer)` forwards a
	pre-allocated buffer (pyshmem >= 1.0.5), used by the SlopesProcess image
	read, all Loop integrators, and the WavefrontCorrector command read.
- Hard-RTC RPC protocol v1: versioned message envelope, type-safe property
	coercion (booleans round-trip correctly), `run()` returns JSON-serializable
	method results, error messages propagate to `hardwareLauncher.last_error`,
	and `run(..., timeout=)` applies a per-call socket timeout.
- `gpu` pytest marker with CUDA stream tests (auto-skip without CUDA; CI can
	deselect with `-m "not gpu"`).
- `benchmarks/check_perf_baseline.py --max-ratio` enforces a performance
	regression threshold; CI runs it at 5.0x against the committed baseline.
- Coverage gate extended to `pyrtc.streams`, `pyrtc.rpc`,
	`pyrtc.component_loading`, and `pyrtc.latency`.
- Streams guide in the documentation (`guides/streams`).

### Changed

- **`pyrtc.pipeline` split into focused modules**: `pyrtc.streams` (pyshmem
	stream policy + SHM planning), `pyrtc.rpc` (launcher/listener protocol),
	`pyrtc.manager` (component runtimes + `RTCManager`), and
	`pyrtc.component_loading`. `pyrtc.pipeline` remains as a re-export shim
	for one release.

- **Shared-memory transport replaced by `pyshmem`.** All shared memory in
	pyrtc is now provided by the external `pyshmem` package (new required
	dependency `pyshmem>=1.0.5`), using its native API directly. The legacy
	`ImageSHM` class, its `_meta` / `_gpu_handle` companion segments, and
	`initExistingShm` are gone. `pyrtc.pipeline` now exposes two thin policy
	helpers instead: `create_stream(name, shape, dtype, gpu_device=None)`
	(producer-side create-or-reuse) and `open_stream(name, gpu_device=None)`
	(consumer-side attach; CPU view by default, CUDA tensor attach with
	`gpu_device`). `clear_shms` now delegates to `pyshmem.unlink_quiet`.
- `pyrtcComponent.read_stream`/`write_stream` simplified: `read_stream`
	takes only `block` and `timeout`; the `SAFE`/`GPU`/`RELEASE_GIL`/
	`record_consumption` flags are removed (GPU vs CPU payloads are decided
	by how the stream was opened). Blocking reads wait for a write the
	component has not yet seen; a component's own writes do not mark the
	stream seen.
- Per-frame lineage metadata (root_time / upstream_write_time /
	upstream_consume_time) is no longer stored in shared memory. Latency
	reporting always uses cross-stream `count`/`write_time` event sampling
	(`pyrtc.latency.collect_stream_event_history`); the `sourceStreams` /
	`lineageSource` stream-config keys were removed.
- GPU streams are created with a CPU mirror, so CPU-only processes
	(viewers, telemetry) can always read them, and GPU stream sharing now
	also works in-process (soft-RTC), not just hard-RTC.
- **All remaining camelCase names converted to snake_case.** The two
	`Loop` matrix attributes were the last capitalized scalar names in
	pyrtc: `Loop.im`/`Loop.cm` (formerly `Loop.IM`/`Loop.CM`), the
	`comp_correction(cm=...)` jit kernel argument, and the `inputRole`
	streams-payload key in the GUI adapter. The `benchmarks/perf_smoke.py`
	`measure_execution_time` call site used the old `numIters=` keyword
	(it now passes `num_iters=`, matching the function signature; the
	mismatch was silently raising `TypeError` in `tests/perf`). The
	`pywfs_example_OOPAO.ipynb` tutorial, the four `examples/{pywfs,shwfs}/*_soft_rtc_example.py`
	files, and `tests/test_loop.py` / `tests/system/test_system_flow.py`
	were updated to the new attribute and method names. There is no
	backwards-compatibility alias — any out-of-tree code that referenced
	`loop.IM` / `loop.CM` / `loop.computeCM` / `loop.plotSingularValues`
	/ `loop.lastSingularValueFit` / `loop.CMMethod` / `loop.numDroppedModes`
	/ `loop.tikhonovReg` must move to the snake_case equivalents.

- **Last camelCase identifiers in pyrtc core code converted to snake_case.**
	`pyrtc/loop.py`: `leaky_integrator_numba`/`leak_integrator_gpu` parameter
	`resconstructionMatrix` → `reconstruction_matrix` (typo fixed at the same
	time), the `leak_integrator_gpu` local `slopes_GPU` → `slopes_gpu`, and
	the pre-allocated hot-path read buffers `self._signalBuffer` /
	`self._wfcBuffer` → `self._signal_buffer` / `self._wfc_buffer`.
	`pyrtc/slopes_process.py`: `self._imageBuffer` → `self._image_buffer`.
	`pyrtc/wavefront_corrector.py`: `self._wfcBuffer` → `self._wfc_buffer`.
	`pyrtc/hardware/pi_modulator.py`: local `originalDirectory` →
	`original_directory`. `pyrtc/hardware/specula_interface.py`: the local
	alias of the external `specula.cpuArray` is now `cpu_array` and the
	`SimpleNamespace` key it is stored under is `cpu_array`; the external
	`specula.cpuArray` import name is unchanged (it is specula's API).
	`tests/test_loop.py` and `tests/test_manager.py` were updated for the
	new attribute names. PEP 8 hygiene pass: 299 ruff auto-fixes applied
	(missing trailing newlines, blank-line whitespace, trailing
	whitespace), 14 manual whitespace fixes, and tab-indentation in
	multi-line imports converted to spaces in `pyrtc/__init__.py`,
	`pyrtc/optimizer.py`, `pyrtc/science_camera.py`, `pyrtc/slopes_process.py`,
	`pyrtc/utils.py`, and `pyrtc/wavefront_sensor.py`. `ruff check
	--select E,F,W pyrtc tests benchmarks examples` is now clean.

## 1.0.0 - 2026-03-07

First stable public release of `pyrtcao`.

This release establishes the initial supported package, CLI, documentation, and
CI/release surface for the `1.0.x` line. The published distribution name is
`pyrtcao`, the import name remains `pyrtc`, and the user-facing project name is
`pyrtc`.

### Added

- PyPI distribution packaging as `pyrtcao` while preserving `import pyrtc`.
- Stable console-script entry points with the `pyrtc-*` prefix:
	`pyrtc-view`, `pyrtc-view-launch-all`, `pyrtc-shm-monitor`,
	`pyrtc-clear-shms`, `pyrtc-measure-latency`, `pyrtc-core-bench`, and
	`pyrtc-ao-loop-bench`.
- Canonical no-hardware onboarding workflow in `examples/synthetic_shwfs/`.
- Shared logging system in `pyrtc.logging_utils` covering scripts, benchmarks,
	launchers, component base classes, and key hardware/control-plane paths.
- Maintainer-facing built-wheel validation helper at
	`python pyrtc/scripts/validate_dist_install.py --dist-dir dist`.
- Cross-platform smoke workflows for macOS and Windows plus Python-versioned
	Linux install/test coverage for Python 3.9 through 3.13.
- Docs-build validation in CI and repository-level Read the Docs
	configuration via `.readthedocs.yaml`.
- Closed-loop synthetic AO benchmark coverage and README-facing benchmark
	artifacts for CPU and GPU comparisons.
- Focused regression coverage for viewer behavior, package public API,
	synthetic onboarding, logging helpers, hardware adapter shims, benchmark
	entry points, and release/install validation.
- Dedicated tests for base-class lifecycle behavior, telemetry error paths,
	`ScienceCamera` branches, and package-install validation.

### Changed

- README and Sphinx docs were substantially rewritten around installation,
	architecture, examples, troubleshooting, support posture, and maintainer
	workflow.
- Documentation now has a clear getting-started path, architecture guide,
	developer guide, component pages, and updated example documentation.
- Benchmark tooling was upgraded from a narrow kernel-oriented view to include
	synthetic closed-loop AO reporting and README-ready markdown table
	generation.
- Public package metadata was consolidated in `pyproject.toml` with stable
	classifiers, extras, URLs, Python support declarations, and console scripts.
- Support posture was tightened and documented as Linux-first for `1.0.x`,
	with macOS and Windows treated as smoke-tested rather than primary deployment
	targets.
- Component, launcher, and hardware control-plane code now reports state
	changes and failures more consistently through the shared logger.
- Viewer and related SHM utilities were updated to use concrete submodule
	imports rather than fragile package-root re-export imports in order to remain
	robust when `pyrtc` is resolved as a namespace package.
- API-reference and component docs were reorganized to remove duplicate Sphinx
	object registrations and produce a clean docs build.

### Fixed

- Viewer/CLI import failures that occurred when running from outside the repo
	root or when `pyrtc` was resolved as a namespace package.
- Python 3.9 compatibility issues caused by bare PEP 604 union annotations at
	import time in logging and benchmark modules.
- Missing benchmark-table kernel mappings and multiple Ruff/lint regressions in
	scripts and tests.
- Headless/non-Qt test collection failures caused by eager Qt backend imports
	in the viewer module.
- Documentation import examples that incorrectly recommended
	`from pyrtc import ...` patterns for classes and launch helpers.
- Duplicate Sphinx autodoc warnings caused by repeated object indexing across
	component pages and the API reference.
- Test-suite warning noise from pytest helper imports and third-party startup
	warnings so the suite runs cleanly.

### Testing

- Full repository test coverage for the tracked coverage set now exceeds the
	release gate, reaching 87.53% at release time.
- `pyrtc.modulator`, `pyrtc.optimizer`, `pyrtc.telemetry`, and
	`pyrtc.component` now have 100% coverage in the tracked release suite.
- `pyrtc.science_camera` coverage was expanded materially as part of release
	stabilization.
- Built-wheel installation, CLI imports, docs builds, performance smoke tests,
	and synthetic system flows are all exercised in the release-facing workflow
	set.

### Notes

- Linux is the primary validated platform for the `1.0.x` line.
- Python 3.9 through 3.13 are covered by the release CI matrix.
- GPU and hardware-specific paths should still be validated in the target
	environment before operational use.
- Hardware adapters in `pyrtc.hardware` should be treated as reference
	integrations and starting points, not guarantees of site-specific SDK
	compatibility.