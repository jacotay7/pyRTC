# Changelog

All notable changes to `pyrtcao` will be documented in this file.

## 1.1.0 - Unreleased

### Fixed

- **Unsupported slopes types and unknown config keys are reported** (#45).
	Any `slopes.signal_type` other than `slopes` passed validation, and
	`SlopesProcess.compute_signal()` then never wrote the `signal` stream, so
	the loop blocked forever. `signal_type` (`slopes`) and `type` (`SHWFS`,
	`PYWFS`) are now checked case-insensitively by the config schema, the
	descriptor (`ConfigFieldDescriptor` gained `case_sensitive`), and
	`SlopesProcess.__init__`, which raises before starting worker threads.
	Config keys a component does not read (such as `method:` for the loop's
	`im_method`) now produce warnings in `pyrtc-validate-config` (text and
	JSON) and in the component log at construction. Private `_` keys and the
	common runtime keys are never reported. Subclasses of built-in components
	declare the keys they read with `EXTRA_CONFIG_KEYS`, which opts them into
	the check; undeclared subclasses are not checked. The in-repo hardware
	adapters declare theirs. The wfc descriptor now lists `command_cap` and
	`display_grid_size`, and the telemetry descriptor lists `streams`. The
	OOPAO, SPECULA and SHARP-lab example configs no longer set `wfc.hardware_delay`
	(a loop key the corrector ignores) or the OOPAO `psf.index`.
- **Importing pyrtc no longer changes the process environment** (#46).
	`loop`, `slopes_process`, `wavefront_corrector` and the ALPAO adapter set
	`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, etc. to 1 at import time,
	silently making the user's whole process single-threaded (and doing nothing
	if numpy was already imported). Hard-RTC children still default to 1 via
	the launcher environment unless the user set a value; soft-RTC users can set
	the variables themselves (see the architecture guide).
- **SPECULA examples now converge** (#39). Several faults combined:
	a PSF frame captured before the first WFS frame consumed SPECULA's
	propagation refresh, so the WFS emitted blank frames until the DM first
	moved; removing the atmosphere left the last turbulent frame on the WFS
	until the next DM command; the example calibrated the IM while the worker
	threads were still JIT-compiling (the first ~12 IM columns were zero, the
	rest smeared into their neighbours), took no reference slopes, and always
	re-enabled the atmosphere because the standalone bridge ignored
	`specula.use_atmosphere`. The PyWFS example also poked with 1e-3 nm, below
	the detector quantization. The bridge now steps both optical branches
	together, refreshes propagation on atmosphere changes and honours
	`use_atmosphere`; the examples wait for a DM round trip, take reference
	slopes, calibrate with the config's `poke_amp`, and leave the atmosphere
	off when configured. Example configs used `method:`, which `Loop` ignores;
	they now use `im_method:`.
- **CPU affinity and real-time priority now work** (#44). The configured
	`affinity` was never applied (worker threads passed a list, which
	`set_affinity` ignored), and it would have pinned the whole process.
	Worker threads are now pinned individually with `os.sched_setaffinity`
	(Linux) when `affinity` is set; unset means unpinned. A new
	`realtime_priority` option runs workers under `SCHED_FIFO`. The priority
	warning is logged once, points to `CAP_SYS_NICE`/limits instead of
	passwordless sudo, and the log reports what was actually applied. The
	hard-RTC entry points of the optimizers and simulator interfaces no longer
	crash calling `decrease_nice(pid)`.
- Requires `pyshmem>=1.3.4`, which fixes a deadlock when garbage collection
	finalized a stream handle while pyshmem was setting up another stream's
	lock (seen as a two-minute stall in the test suite).
- **Multi-stream telemetry now covers one time window** (#42).
	`Telemetry.save()` captured all frames of one stream before starting the
	next, so a `wfs` + `signal` + `wfc` session held three unrelated time
	windows that the AOTPy exporter treated as the same loop iterations.
	Streams are now captured concurrently (one reader thread each, started
	together); pair frames exactly with the recorded frame ids.
- **Coverage now measures the whole package** (#47). The gate listed
	dotted submodules, which left the control core (`loop`, `slopes_process`,
	`wavefront_sensor`, `manager`) unmeasured and made coverage re-import numpy
	mid-session (the cause of intermittent `_NoValueType` errors and torch
	segfaults under `pytest`). The gate is 70% of the whole package, excluding
	only the display-bound Qt windows.
- **Strehl estimate is now flux-normalized** (#43). `compute_strehl()`
	compared raw peaks (`max(current) / max(model)`), so the value scaled with
	source brightness and exposure time — and the PID/NCPA/hyperparameter
	optimizers use it as their objective. It now compares peak-to-total-flux
	ratios (`pyrtc.science_camera.estimate_strehl`).
- **`WavefrontSensor.take_dark()` built the dark at the processed shape**
	(#41). It averaged the downsampled/rotated `wfs` stream, while `expose()`
	subtracts the dark from the raw frame, so `downsample_factor` crashed the
	WFS worker and `rotation_angle` produced a misaligned dark. It now averages
	`wfs_raw`, rounds instead of truncating, and both WFS and science-camera
	darks discard the first frame, which may predate the call.
- **Integrator kernel wrote past the correction array** (#38).
	`leaky_integrator_numba` iterated `num_active_modes + 1` modes, so with no
	dropped modes (the default) every `standard_integrator`/`leaky_integrator`
	step wrote one element out of bounds, and with dropped modes the first
	dropped mode was still driven. It now controls exactly the active modes,
	zeroes the rest, fills the caller's buffer instead of allocating, and the
	GPU path has identical semantics.
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
- Windows runs no longer die with a random `KeyboardInterrupt`: pyshmem
	< 1.3.3 probed process liveness with `os.kill(pid, 0)`, which sends Ctrl+C
	on Windows. pyrtc now requires `pyshmem>=1.3.3`.
- Hard-RTC child listeners now stop cleanly when the RTC closes the control
	socket instead of crashing with `BrokenPipeError`.
- **Documentation links point at the live docs** (#68). The PyPI
	`Documentation` URL and README guide links pointed at a stale Read the
	Docs project (`pyrtc.readthedocs.io`) whose guide pages return 404; they
	now use `https://pyrtc-ao.readthedocs.io/en/latest/`. The README no
	longer references the missing `RELEASE_1_0_PLAN.md` or describes 1.0.0
	as unreleased, and the clone instructions `cd` into `pyRTC`, not `pyrtc`.

### Added

- **Modal bases from aobasis** (#53). A `basis:` section on the wavefront
	corrector builds `M2C` at start-up with
	[aobasis](https://github.com/jacotay7/aobasis) (now a core dependency;
	it needs only numpy, scipy and matplotlib) from the actuator geometry:
	`type: kl | zernike | fourier | zonal | zonal_fast | hadamard`, plus
	`pupil_diameter`, `r0`/`L0` (KL), `ignore_piston`, `normalize`
	(`peak` by default, so `poke_amp` bounds the actuator stroke),
	`orthonormalize`, `positions_file` and `min_distance` (zonal-fast).
	Positions come from the 2D `layout` mask (pupil-centred, outer rows and
	columns on the pupil edge) unless the adapter knows better: OOPAO uses
	`dm.coordinates` and the telescope diameter, SPECULA its zonal actuator
	positions and `pixel_pupil * pixel_pitch`. `m2c_file` still takes
	precedence; without either the identity is used as before, and the
	SPECULA examples keep their SPECULA-native Zernike basis unless
	`wfc.basis` is set. New `pyrtc.modal_basis` module,
	`WavefrontCorrector.build_basis_m2c()` and `m2c_source`. The OOPAO and
	SPECULA correctors now also honour `m2c_file` (they used to ignore it).
- Continuous telemetry recording (#61): `Telemetry.start_ring_buffer(streams,
	seconds=, frames=)` keeps the newest publications of each stream (payload,
	`write_time`, `frame_id`, `count`) in a preallocated in-memory ring,
	`dump_ring_buffer()` writes a consistent snapshot as an ordinary telemetry
	session (readable by `load_telemetry_session` and the AOTPy exporter), and
	`stop_ring_buffer()` stops it. Missed publications are counted, readers use
	read-only handles, and a `telemetry.ring_buffer` config section starts it
	with the component. Telemetry sessions now also store publication counts
	(`counts.npy`, loaded as `counts`).

- `specula` optional extra (`pip install pyrtcao[specula]`) for the
	SPECULA-backed examples (#40). The PYWFS and SHWFS example docs and the
	README now explain how to install each simulator, including the OOPAO
	recipe: OOPAO is not on PyPI, and a plain `pip install` of it fails on
	`import OOPAO`, so it must be cloned and put on `PYTHONPATH`.

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
- **SHWFS centroiding algorithms** (#59). The slopes `centroider` option
	selects `cog` (thresholded centre of gravity, the default and previous
	behaviour), `wcog` (Gaussian-weighted CoG, FWHM `wcog_fwhm`, optional gain
	correction from `wcog_spot_fwhm`) or `correlation` (square-difference
	correlation against a per-sub-aperture reference template within
	`correlation_search_radius`, with 2D quadratic sub-pixel refinement) for
	extended sources. The reference image comes from `take_reference_image()`,
	`set_reference_image()` or `reference_image_file`; WCoG centres its weights
	on the reference spots when one is set. All methods share the contrast
	threshold and publish 0 for sub-apertures without flux. The new Numba
	kernels preallocate their buffers and are part of the core compute
	benchmark and perf smoke baseline.

### Changed

- **NumPy is no longer capped below 2.3; Python 3.14 is supported** (#49).
	The `numpy>=1.26,<2.3` requirement is now `numpy>=1.26`; numba already
	limits NumPy to versions it supports. pyrtc is tested with NumPy 2.5,
	numba 0.67 and SciPy 1.18, and CI now also runs on Python 3.14. A
	Dependabot configuration opens weekly update PRs for the Python
	dependencies and the GitHub Actions used by CI.
- **CI actions moved off Node 20** (#36). Workflows now use
	`actions/checkout@v7`, `actions/setup-python@v7`,
	`actions/upload-artifact@v7` and `actions/download-artifact@v8`.
- **Plotting helpers return figures instead of showing them** (#34).
	`Loop.plot_im`, `WavefrontSensor.plot`, `WavefrontCorrector.plot`,
	`ScienceCamera.plot`, and `SlopesProcess.plot_pupils` build and return a
	matplotlib `Figure` (display it with `plt.show()` or in a notebook) instead
	of calling `plt.show()`, and `matplotlib.pyplot` is no longer imported when
	importing pyrtc, so no GUI backend is selected on import.
	`ScienceCamera.plot()` no longer blocks waiting for a new frame, and
	`Loop.plot_im()` drops its unused `row` argument.
- **GPU PYWFS slopes no longer re-upload masks every frame** (#64).
	`SlopesProcess.compute_signal()` copied the four pupil masks, the slopes
	buffer and the reference slopes to the GPU on every frame. They are now
	cached on the device (as pixel-index tensors) and rebuilt only when the
	pupils or reference slopes change, roughly halving the per-frame time of
	the GPU path. The GPU path also accepts a CPU-backed `wfs` stream (NumPy
	frames are copied to `gpu_device`), and writes the device tensor directly
	when the `signal` stream is GPU-backed.
- **The hard-RTC listener only exposes public names** (#48).
	`Listener` answered `get`/`set`/`run` for any attribute of the hardware
	object, including private (`_x`) and dunder (`__class__`, `__dict__`)
	ones, to any local process that reached its port. Names starting with `_`
	are now rejected with an error reply (surfaced as
	`HardwareLauncher.last_error`), `run` only calls callables, and `set` no
	longer overwrites methods. The socket stays localhost-only and
	unauthenticated; see the `Listener` docstring for the trust model.
- **The import package is now `pyrtc` (was `pyRTC`), with PEP 8 module
	names**: `pyRTC.Loop` → `pyrtc.loop`, `pyRTC.SlopesProcess` →
	`pyrtc.slopes_process`, `pyRTC.pyRTCComponent` → `pyrtc.component`
	(class `Component`), `pyRTC.hardware.SyntheticSystems` →
	`pyrtc.hardware.synthetic_systems`, and so on. There are no
	compatibility aliases; update imports and config `class_name` paths.
- **`pyRTC.Pipeline` split into focused modules**: `pyrtc.streams` (pyshmem
	stream policy + SHM planning), `pyrtc.rpc` (launcher/listener protocol),
	`pyrtc.manager` (component runtimes + `RTCManager`), and
	`pyrtc.component_loading`. There is no `pyrtc.pipeline` module; the
	public names are also exported from the `pyrtc` package root.
- The `pyrtc-clear-shms` CLI is removed. Streams are pyshmem streams, so
	use `pyshmem list` / `pyshmem unlink NAME` / `pyshmem purge`, or
	`pyrtc.streams.clear_shms(names)` from Python.

- **Shared-memory transport replaced by `pyshmem`.** All shared memory in
	pyrtc is now provided by the external `pyshmem` package (new required
	dependency `pyshmem>=1.3.3`), using its native API directly. The legacy
	`ImageSHM` class, its `_meta` / `_gpu_handle` companion segments, and
	`initExistingShm` are gone. `pyrtc.streams` now exposes two thin policy
	helpers instead: `create_stream(name, shape, dtype, gpu_device=None)`
	(producer-side create-or-reuse) and `open_stream(name, gpu_device=None)`
	(consumer-side attach; CPU view by default, CUDA tensor attach with
	`gpu_device`, `readonly=True` for observers). `clear_shms` now
	delegates to `pyshmem.unlink_quiet`.
- `Component.read_stream`/`write_stream` simplified: `read_stream` takes
	only `block`, `timeout`, and `out`; the `SAFE`/`GPU`/`RELEASE_GIL`/
	`record_consumption` flags are removed (GPU vs CPU payloads are decided
	by how the stream was opened). A blocking read consumes the stream: it
	returns the first write newer than the one it returned last time, using
	pyshmem's level-triggered `read_after_publication`. `block=False` is a
	peek and no longer consumes, so e.g. reading `wfc` over RPC cannot make
	the DM worker skip a correction. `out=` is ignored for GPU-attached
	streams instead of raising.
- Per-frame lineage metadata (root_time / upstream_write_time /
	upstream_consume_time) is replaced by pyshmem's publication `frame_id`.
	The wavefront sensor numbers each exposure and every component stamps
	its outputs with the `frame_id` of the input it consumed, so
	`pyrtc.latency` pairs source and target writes exactly (segments report
	`alignment: "frame_id"`) and falls back to count alignment for
	producers that do not stamp ids. The `sourceStreams` / `lineageSource`
	stream-config keys and the sequential `collect_timestamps` sampler were
	removed; `collect_stream_event_history` now also returns frame ids.
- Telemetry captures each frame with `read_new_publication`, so a frame's
	timestamp is its own write time rather than a later one. Sessions now
	also store `frame_ids.npy` (loaded as `frame_ids`) and a `missed_frames`
	count per stream.
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