# Changelog

All notable changes to `pyrtcao` will be documented in this file.

## Unreleased

### Added

- **PyTorch image reconstructor.** `TorchImageReconstructor`
	(`pyrtc.image_reconstructor`) publishes a PyTorch model's output on each
	WFS image as the loop's `signal`, for neural and focal-plane
	reconstructors. It sits in the `slopes` section in place of
	`SlopesProcess`, so the loop and the rest of the pipeline are unchanged
	(use an identity IM when the model outputs modes). Models come from a
	`.pt2` (`torch.export`) or TorchScript `model_file`, or from a
	`model_factory` plus an optional `state_dict_file`; `signal_size` is
	checked against the model output at startup. Options: `device`
	(CPU/CUDA), `dtype` (float32/float16), flux normalisation, square-root
	stretch and a per-element output scale. On CUDA it uses pinned host
	buffers, its own CUDA stream and a captured CUDA graph (with an eager
	fallback); `timing_stats()` reports the per-frame compute time.
	`benchmarks/image_reconstructor_bench.py` times it. Config validation
	applies the `SlopesProcess` checks only to `SlopesProcess`-family
	classes, and stream planning and the AOTPy export treat a typeless
	`slopes` section with `signal_size` as a generic signal.

### Fixed

- **A component whose constructor fails no longer leaks its worker threads**
	(#155). `Component.__init__` started one worker thread per entry in
	`functions` before the subclass finished its own setup, so a constructor
	that then raised (a missing input stream, a bad calibration file) left
	threads spinning for the life of the process, each holding the
	half-built component. The threads now start on the first `start()`.
	Construction starts none, and `stop()`/`start()` still pause and resume
	the same threads.
- **`Loop.pid_integrator_pol` is about 50x faster** (#158). Each frame ran
	the pseudo open-loop product `f_im @ correction` in NumPy and the control
	product in numba, which calls SciPy's OpenBLAS. The two libraries' thread
	pools (one spinning worker per core each) then fought over the cores, so
	a frame took 12 ms instead of 0.2 ms in a 16-core cpuset (signal 1600,
	400 modes). Both products now run in numba
	(`pyrtc.loop.pseudo_open_loop_slopes`).
	- `pid_integrator_pol` also no longer fails with a numba `TypingError`
	  on every frame when the interaction matrix is float64 (an `im_file`
	  saved as float64). `Loop.f_im` is now kept in the control matrix's
	  dtype.
- **The first frame after `start()` no longer stalls while numba compiles**
	(#157). `SlopesProcess`, `Loop` and `WavefrontCorrector` compiled their
	per-frame numba kernels during the first real frame. That took 0.15 s
	with a warm numba cache and up to 0.75 s cold, against 0.06 to 0.2 ms
	per frame in steady state. A loop started on a live system therefore
	held the DM still for hundreds of frames at kHz rates, enough to lose
	lock.
	- Each of these components now ends `__init__` with `warmup()`. It calls
	  the kernels once on zero scratch arrays typed like the real buffers, or
	  runs the torch PYWFS path when `gpu_device` is set. It writes no stream.
	- The first iteration now takes about 0.5 ms (GPU PYWFS: 88 ms down to
	  2 ms). Construction pays the compile time instead.
	- `Component.warmup()` is a no-op hook that other components can
	  override.
	- `benchmarks/first_iteration_bench.py` compares first-call and
	  steady-state latency with a cold and a warm numba cache.

## 1.1.0 - 2026-09-29

### Fixed

- **Telemetry sessions record the real pyrtc version.** The version lookup
	asked package metadata for `pyrtc`, but the distribution is `pyrtcao`, so
	sessions (and AOTPy exports, `PRTCVER`) always said "1.0.0", or the version
	of the unrelated WebRTC `pyrtc` package when it was installed. It now reads
	`pyrtcao`. `pyrtc.__version__` is new, and the docs take their version from
	`pyproject.toml`.
- **`manager.latency()` without `stream_path` follows renamed streams** (#119).
	Path inference used the descriptors' logical stream names (`wfs`,
	`signal`, `wfc`) instead of the shared-memory names configured in each
	section's `input_streams`/`output_streams`. It failed on any system with
	renamed streams.
- **Components loaded from `class_file` work with the numba disk cache**
	(follow-up to #92). Class files outside the loaded package were exec'd
	under a per-process random name, without a `sys.modules` entry. numba then
	recorded their kernels' module as `<dynamic>` (or that random name), and
	the next process to load the cache crashed importing it. This happened,
	for example, when a wheel install ran the repo's example configs. Now:
	- a `class_file` that is a byte-identical copy of an installed `pyrtc`
	  module imports that module;
	- other files load once under a stable name registered in `sys.modules`,
	  so loading the same file again returns the same class.

- **Latency reports align by frame id right after `start()`** (#112).
	`manager.latency()` and `pyrtc-measure-latency` used to sample each
	stream independently. While downstream workers were still starting, the
	WFS window could end before the first `signal`/`wfc` frame, so the report
	quietly fell back to heuristic count alignment. They now wait until a
	source frame has reached every stream on the path
	(`latency.wait_for_path_live`, within `timeout_seconds`). Each segment
	reports `matched_samples`, and a fallback despite stamped frame ids logs a
	warning and is labelled in the text report.
- **The synthetic SHWFS supports `downsample_factor`** (#76). It rendered
	the raw camera frame at the downsampled shape. It now renders in processed
	pixels (the geometry SlopesProcess uses) and expands each pixel to a D x D
	block, so downsampling reproduces the same image. The OOPAO WFS was not
	affected: its raw frame is OOPAO's camera frame.
- **The OOPAO examples run and converge again** (#88).
	- The interface uses current OOPAO propagation (`src ** tel * dm * wfs`),
	  fixing a crash in the DM relay on current OOPAO.
	- The standalone bridge honours `oopao.use_atmosphere`; it hard-coded the
	  atmosphere on, so calibration and the loop ran against turbulence.
	- The examples calibrate like the SPECULA ones: DM round-trip check,
	  reference slopes on the flat system, then the IM. The PYWFS example's KL
	  basis call no longer crashes.
	- `import OOPAO` works when OOPAO is installed without a clone on
	  `PYTHONPATH`, and an incomplete pip install gives a clear error.
	- New system test `tests/system/test_oopao_convergence.py` (skipped
	  without OOPAO).
- **IM calibration settles for the measured DM round trip** (follow-up to
	#87). `compute_im()` discarded a fixed `im_settle_frames` after each poke
	even when `check_round_trip()` had measured a longer lag, so on a slow or
	loaded pipeline pokes were averaged before they landed and the loop could
	diverge. Each calibration now discards at least the measured number of
	frames; `im_settle_frames` is the minimum.
- Requires `pyshmem>=1.3.8`: opening a stream while another process wrote it
	failed now and then (`lock owner and depth metadata are inconsistent`,
	about 0.2% of opens against a 1 kHz writer), which hit anything that
	attaches to a running system: viewers, `manager.latency()`, telemetry,
	hard-RTC children. Fixed at the source in jacotay7/pyshmem#20.
- Requires `pyshmem>=1.3.6`: closing a stream handle while another thread
	was blocked reading it crashed the process (fixed at the source in pyshmem),
	which `Component.close()` relies on when a worker does not exit in time.
- **Components can be torn down; managers no longer leak them** (#37).
	`stop()` only paused a component: its worker threads, and every
	shared-memory handle it opened, lived until the process exited, so each
	manager lifetime in a GUI, notebook, benchmark or test run leaked threads,
	mappings and file descriptors, and on Windows the open handles pinned
	stream names so a later rebuild with a new shape failed. `stop()` keeps its
	pause/resume meaning; the new `Component.close()` stops the component,
	ends its worker threads (a blocking `read_stream` notices within 0.1 s and
	raises `ComponentClosedError`), and closes its registered streams. It is
	idempotent, final (a closed component cannot be restarted), and called
	from `__del__`. `RTCManager.close()` stops the system, closes every
	soft-RTC component and shared resource, and shuts down hard-RTC children;
	`build()`/`start()` afterwards constructs fresh components. `RTCManager`
	is a context manager. Hard-RTC children now close their component on
	shutdown, the manager GUI closes a manager it replaces and closes the
	system when its window closes, and the examples and the pipeline
	benchmark close what they start. The ALPAO, XIMEA, Spinnaker and PI
	adapters release their devices in `close()` instead of `__del__` (which
	never ran while worker threads held the component), and `Telemetry.close()`
	also stops the ring buffer. A manager whose build fails now closes the
	components it had already built. The Windows skip on
	`test_manager_start_clears_stale_output_shms` is removed.
- **Every component stream is registered, so frame ids reach every output**
	(#35). `Component` looked up streams by registration and otherwise fell
	back to any `<name>_shm` attribute, which read and wrote the stream but
	silently dropped its `frame_id`. The fallback is gone: `read_stream` and
	`write_stream` raise `KeyError` for unregistered names. The synthetic WFS
	now registers the `wfc` stream it reads, the synthetic science camera the
	`signal` stream (so `strehl`, `tiptilt` and the PSFs carry frame ids), the
	wavefront corrector registers `wfc` as an input as well as an output (so
	`wfc_2d` carries the command's frame id), and the PID, NCPA and loop
	hyper-parameter optimizers register their streams. `WavefrontSensor`
	numbers exposures with its own counter, so reading an input cannot rewind
	it. Registering a new handle under an existing name closes the handle it
	replaces. `_ensure_stream_state()`, which existed only for tests that
	built components with `Cls.__new__`, is removed; tests use
	`testsupport.bare_component`, which runs the real stream-state setup.
- **Interaction-matrix calibration waits for a live pipeline** (#87).
	Worker kernels JIT-compile on first use, so right after start-up the first
	DM command reached the signal about a second late and a cold `compute_im()`
	recorded zero or smeared IM columns (41% off in the SPECULA example).
	`compute_im()` now first runs `Loop.check_round_trip()` (flatten, stable
	frames, poke, response, flatten, return; `im_round_trip_check`,
	`im_timeout`) and discards `im_settle_frames` frames after each poke. The
	SPECULA examples use it instead of their own helper.
- **`manager.latency()` no longer fails at random in soft-RTC mode.**
	Closing the observer handles raised `cannot close shared memory while
	another thread owns its lock` when a component thread was mid-write
	(pyshmem shares lock state per stream name within a process). Fixed at the
	source in pyshmem 1.3.5, now the minimum version. `RTCManager.latency()`
	also accepts `timeout_seconds`.
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
- **`take_ref_slopes()` no longer races the worker thread** (#89, #90).
	It averaged `compute_signal_2d()` results, which is the shared
	`cur_signal_2d` buffer the `compute_signal` thread rewrites every frame,
	and its first frame could still carry the old reference. It now averages
	private snapshots of the `signal` stream published after the reference
	was reset. GPU-backed signal streams (torch tensors) work too: that path
	failed with `AttributeError`, and a GPU PYWFS `SlopesProcess` could not
	even be constructed (the reference buffers were sized from a CUDA tensor).
	`compute_signal_2d()` gained an `out=` argument and honours an explicit
	`valid_sub_aps` (it used to return `-1`).
- **`set_pupils()` works on a running PYWFS slopes process** (#91). The
	pupil pixel count, the numba work buffers (`p1`..`p4`, `tmp1`, `tmp2`),
	`slopes_arr_1d` and `ref_slopes_1d` were only allocated in `__init__`, so
	changing the pupil radius broke both the CPU and the GPU path. They are now
	rebuilt by `set_pupils()`, which holds off the worker (a lock taken per
	frame, not while waiting for the WFS image) until the masks, buffers,
	reference slopes and signal streams are all swapped. Reference slopes of
	the old shape are reset to zero with a warning. The GPU device cache is
	rebuilt from the new buffers. Overlapping pupils, which would make the
	unchecked numba kernel write past its buffers, now raise `ValueError`.
	`set_ref_slopes()` also builds the new 1-D reference before swapping it in
	instead of zeroing and refilling the live array.

### Added

- **Benchmark results from an aarch64 host and newer GPUs.** The README's
	pipeline-latency section adds an 80-core Neoverse-N1 host
	(`benchmarks/pipeline_latency_report*_aarch64.json`) and explains its
	deep-idle wake-up cost. The developer guide adds free-threaded latency on
	that host (8x lower soft-RTC p50 at 1 kHz). #139, the free-threaded
	runaway on a loaded x86 host, was the HCIPy simulator's uncapped OpenBLAS
	pools: with them capped it passes there under load, and the free-threaded
	CI job runs `tests/system` again. The loop docs add fp16/bf16 control-matrix timings on Ampere
	and Ada GPUs and why the reduced formats lose on small systems.
- **ImageStreamIO (milk/CACAO) bridge** (#54). `pyrtc.isio_bridge.IsioBridge`
	(and the `pyrtc-isio-bridge` CLI) mirrors a stream from pyrtc to ISIO, or
	from ISIO into pyrtc, so pyrtc can use ISIO camera and DM drivers and milk
	viewers, or feed a CACAO RTC. Shapes carry over, and ISIO `cnt0` becomes
	the pyrtc `frame_id`. It polls ISIO's semaphores, because the module's
	blocking waits hold the GIL (#138). Tests run against a real
	ImageStreamIO build in a new CI workflow.
- **Micro-Manager camera adapters** (#133, from #74). `MicroManagerWFS`
	and `MicroManagerScienceCamera` drive any Micro-Manager camera (Andor,
	Hamamatsu, PCO, Photometrics, ...) through pymmcore-plus from a
	Micro-Manager `.cfg` file (`pip install pyrtcao[micromanager]`). They
	stream the newest frame from continuous acquisition, and are tested
	against a fake core.
- **An `aarch64` CI job** runs the whole suite, system tests included, on
	GitHub's ARM64 runner. pyrtc and pyshmem pass on an 80-core Neoverse-N1
	host, but ARM had no CI coverage; its weaker memory ordering sends
	pyshmem's cross-process publication through libatomic instead of plain
	stores.
- **Free-threaded Python support** (#66). pyrtc runs on CPython 3.13t/3.14t
	with the GIL off: every core import supports free threading, and the
	suite passes with `PYTHON_GIL=0`. A new `free-threaded` CI job checks
	both. On the synthetic soft-RTC pipeline, p50 latency drops 3-15% and p99
	5-30% (developer guide).

- **Performance history** (#65). CI now uploads an end-to-end
	`pipeline-latency` report next to the micro-benchmarks. The new
	`benchmarks/perf_history.py` reads the reports of recent CI runs (or a
	directory of saved reports), prints each metric's latest value against
	its history median, and flags regressions. The Python 3.12 job adds this
	table to its summary.
- **Boston Micromachines DM adapter** (#70). `pyrtc.hardware.bmc_dm.BMCDM`
	drives BMC MEMS mirrors through the BMC DM SDK. Bipolar commands map to
	the SDK's `[0, 1]` range about a `bias`. The actuator count comes from
	the SDK, and the layout from BMC's standard geometries or `layout_file`.
	The mirror is zeroed on close. Tested against a fake SDK.
- **GenICam camera adapters** (#72). `GenICamWFS` and
	`GenICamScienceCamera` drive GigE Vision / USB3 Vision cameras (Basler,
	Allied Vision, FLIR, IDS, ...) through Harvesters and the vendor's GenTL
	producer (`pip install pyrtcao[genicam]`). They apply exposure, gain, bit
	depth, binning, ROI and arbitrary `node_settings`, and are tested against
	a fake Harvesters API.

- **Multiple correctors per loop** (#58). A `CorrectorSplitter` in the `wfc`
	section splits the loop's modal command across several corrector
	sections (woofer/tweeter, DM plus tip-tilt stage), so one IM calibrates
	all of them. Its optional offload integrator moves the content the target
	can represent (coupling from `set_coupling_from_im`) onto the target
	without changing the wavefront. Config validation now uses a section's
	built-in rules only when its class belongs to that component family.
- **HCIPy simulator backend** (#55). `pyrtc.hardware.hcipy_interface`
	builds a telescope, DM, Shack-Hartmann or modulated pyramid WFS,
	frozen-flow atmosphere and science camera from a flat parameter file
	with HCIPy, which is pip-installable (`pip install pyrtcao[hcipy]`). It
	adapts them to the pyrtc WFS, corrector and camera components, standalone
	or as a manager `resource`. The corrector passes its real actuator
	positions to aobasis. New example `examples/hcipy/`, docs page, unit
	tests, and a system test that nulls a DM aberration and raises the
	Strehl on the atmosphere.

- **fp16/bf16 control matrix on GPU** (#67). `pyrtc.loop.ReducedPrecisionMatrix`
	stores the CM in half precision with per-row fp32 scales and multiplies
	with fp32 accumulation; `leak_integrator_gpu` accepts it. The error in the
	modal update is about 1e-3 (fp16) or 1e-2 (bf16), relative. On a Quadro
	P620, a 64x64 system's integrator step drops from 2.1 ms to 1.3 ms. The
	core benchmark times both variants.
- **Predictive control** (#57). The new `predictive_integrator` loop
	function forecasts each mode's pseudo open-loop disturbance for when the
	command lands, using the loop's `delay_frames`, and cancels it.
	- Predictors are pluggable (`pyrtc.predictive.register_predictor`). Built
	  in: `ar_kalman` (modal LQG with an AR(2) model per mode and a
	  steady-state Kalman filter) and `least_squares` (a per-mode linear
	  prediction filter).
	- The loop records POL data while it runs as a delay-aware POL
	  integrator; `loop.fit_predictor()` fits the predictor and switches the
	  loop to it.
	- In simulation, both cut a vibration mode's residual more than 15-fold
	  against the best integrator gain.

- **Per-mode gains, optical-gain compensation and gain optimization** (#56).
	- The loop gain of mode `i` is `gain * modal_gains[i] / optical_gains[i]`,
	  set in the config or at run time (`set_modal_gains`,
	  `set_optical_gains`). It is folded into the control matrix and used by
	  every integrator, including POL.
	- `loop.optimize_modal_gains(residuals, frame_rate, delay_frames=...)`
	  chooses per-mode gains from closed-loop modal residuals
	  (`loop.modal_residuals`), using the Gendron & Léna pseudo open-loop PSD
	  method in the new `pyrtc.modal_gains`. In simulation, the chosen gains
	  come within 5% of the brute-force optimum.
- **Safety watchdog and saturation reporting** (#60).
	- The closed loop waits at most `watchdog_timeout` (default 1 s) for a
	  new `signal` frame. On timeout it reports the input stale, including
	  whether the producer is alive, and applies `watchdog_action`: `hold`
	  (default), `open` or `flatten`.
	- The wavefront corrector counts actuators at `command_cap` every frame
	  and warns past `saturation_warn_fraction` (default 5%).
	- Both are reported by `safety_status()`, included as `safety` in
	  manager status for soft- and hard-RTC components, and shown as alerts
	  on the manager GUI's graph nodes.

- **Hadamard interaction-matrix calibration** (#102): `im_method: hadamard`
	pokes all modes at once with +/-`poke_amp` Hadamard patterns and
	demultiplexes, cutting white sensor noise in the IM by about
	`sqrt(num_modes)` for the same number of frames.
- **ALPAO adapter supports any actuator count** (#73). The layout is the
	smallest centred-disk grid holding the mirror's actuator count (identical
	to the previous DM97 layout), or an explicit `layout_file`. The SDK is
	imported when the mirror is created, from an optional `sdk_path`, so the
	module imports without the vendor SDK installed.
- **End-to-end pipeline latency benchmark** (#62).
	`benchmarks/pipeline_latency_bench.py` launches the synthetic SHWFS system
	through `RTCManager` in soft and hard mode, with stream notify on and off,
	and writes the frame-id aligned WFS -> DM latency (mean/p50/p99/jitter,
	total and per segment, median over interleaved `--repeats`) to JSON. The
	README Performance section now shows these pipeline numbers next to the
	kernel-compute table and labels which is which: the kernel harness reports
	~13 us per iteration, while the running pipeline takes ~150-300 us.
	`benchmarks/stream_handoff_bench.py` measures a single stream handoff
	between threads or processes. Both use private stream names; neither is
	in the CI perf gate.
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

- **The HCIPy example runs single-threaded BLAS by default.** It sets
	`OPENBLAS_NUM_THREADS` and related variables to `1` before importing
	numpy, unless they are already set, as hard-RTC children do. The HCIPy
	system test caps the pools with `threadpoolctl` (now in
	`requirements-test.txt`). HCIPy otherwise kept a full OpenBLAS pool busy.
	On 16 cores the pools used about 15 of them, the WFS ran slower (23
	against 32 frames/s), and the system test took 3x longer on a loaded host
	(69 s against 23 s).

- **`import pyrtc` no longer imports torch** (0.47 s instead of 1.3 s with
	torch installed). pyrtc probes torch lazily
	(`streams.gpu_torch_available()`), and pyshmem 1.3.7, now required, does
	the same. CPU-only component processes never load torch.
	`requirements.txt` again matches the core dependencies.

- **Heavy dependencies are optional extras** (#50). `pip install pyrtcao`
	installs only the soft-RTC core. `matplotlib` moved to the `plot` extra,
	`astropy` to `fits`, and `optuna`/`cmaes` to `optimize`. `numexpr`, which
	pyrtc never imported, is gone. Features that need a missing extra raise an
	`ImportError` naming it (`pyrtc.utils.require_optional`), and
	`import pyrtc` no longer imports optuna or astropy. Requires aobasis 1.2.0,
	which made its own matplotlib dependency optional.
- **Zernike and Fourier bases are orthonormalized by default** (#105).
	Sampled on a discrete actuator grid, they are not orthogonal, and the raw
	modes left the SPECULA SHWFS Zernike loop borderline (residual 4.9% of the
	aberration, against 1.0% orthonormalized). `basis.orthonormalize` now
	defaults to `true` for `zernike` and `fourier`; set it to `false` for the
	raw modes. Hadamard stays raw so its +/-1 patterns survive. The M2C rank
	check now runs on the raw modes, so orthonormalizing no longer hides a
	rank-deficient basis.
- **aobasis 1.1.0 or newer is required** (1.2.0 since #50). 1.1.0 fixes the Zernike Noll order (several
	modes swap index) and adds the Noll normalization (see its changelog), and
	a Fourier basis is now always full rank or an error: without piston it
	holds at most `num_actuators - 1` modes. Recalibrate IMs taken with
	Zernike or Fourier bases.
- **The Numba kernels are cached on disk** (`cache=True`, #92). Each new
	process used to recompile every hot-path kernel on first use; now only the
	first run after an install or source change compiles. The two hottest
	kernels' first-call cost drops from about 0.7 s to 0.1 s. Caches live in
	`__pycache__` next to the source, or numba's user cache directory when the
	install is read-only (set `NUMBA_CACHE_DIR` to override).
- **Python 3.9 is no longer supported** (end of life since October 2025);
	pyrtc requires Python 3.10 or newer.
- The README and getting-started guide warn about the unrelated WebRTC
	`pyrtc` package on PyPI, which installs the same import name (#52).
- **The manager GUI and viewer run on Qt6 instead of PyQt5** (#51).
	Qt5 reached end of life in 2025. `pyrtc-manager-gui` and `pyrtc-view`
	now import Qt through `qtpy` and work with PySide6 or PyQt6; the `gui`
	and `viewer` extras install `qtpy` and PySide6 instead of PyQt5. With
	both bindings installed PySide6 is used unless `QT_API=pyqt6` is set.
	Qt5 bindings are rejected with an install hint, as is a missing binding
	(`pyrtc-manager-gui` used to show a traceback there). The viewer draws
	with matplotlib's `backend_qtagg` (matplotlib 3.5 or later), and
	`launch_mosaic_viewer` reuses an existing `QApplication`. The install
	hints now name the `pyrtcao` distribution. A new offscreen smoke test
	(`tests/test_qt_smoke.py`) builds both windows and renders frames from a
	pyshmem stream; it skips when no Qt6 binding is installed.
- **Streams wake their consumers instead of being polled** (#63).
	`create_stream` now creates pyshmem streams with `notify=True`, so a write
	wakes consumers blocked in `read_stream` through a Linux futex instead of
	each consumer sleeping and re-checking. In the synthetic SHWFS pipeline
	this lowered hard-RTC WFS -> DM latency from 199 to 153 us mean at 200 Hz
	(183 to 154 us at 1 kHz); soft-RTC changed within noise. A single
	cross-process handoff at 5 kHz went from 40 to 15 us (write to wake-up,
	median). Each write costs about 5 us more for the wake system call. Set
	`PYRTC_STREAM_NOTIFY=0` (hard-RTC children inherit it) or pass
	`create_stream(..., notify=False)` to keep polling streams. An existing
	stream with the other notify setting, such as one left by an older pyrtc,
	is rebuilt instead of reused.
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
- **SHWFS CoG slopes no longer allocate per frame** (#103). The `cog` kernel
	copied the whole image to float32 every frame and `compute_signal()`
	allocated a fresh slopes array; the kernel now converts pixels as it reads
	them and writes every entry (no-flux and out-of-image sub-apertures still
	read 0), so the slopes buffer is reused. The valid-sub-aperture gather for
	all SHWFS centroiders also reuses a buffer. Results are bit-identical. On a
	480x480 frame (60x60 sub-apertures of 8 pixels) about 950 kB of per-frame
	temporaries (the image copy and two slopes arrays) are gone; kernel time is
	unchanged to ~10% faster.
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