# pyrtc Improvement Plan

Status tracker for the post-pyshmem-migration cleanup. Grounded in the README's
stated goals: performance as a primary design constraint, the synthetic SHWFS
workflow as the front door, a conservative Linux-first release posture, and
ML-assisted control research as a target audience.

**Status as of 2026-06-11 (fourth pass):** A1, A2, B3, B4, C6, C7, C8, D9,
D10, D11, E12, F1, F2 are done. Verified: 291 tests + 3 system + 19 perf + 1
notebook smoke all green, ruff clean (E/F/W with `E402,E501` ignored per the
existing config), `pyrtc-ao-loop-bench` within 1.5x of the committed baseline,
live soft and hard tutorials converge. Remaining: the second gate extension
round (Loop/SlopesProcess/manager) noted in D9.

**F2 — Final camelCase sweep + PEP 8 hygiene.** Closed the last camelCase
identifiers in pyrtc core code and tightened PEP 8 conformance in
non-GUI/non-viewer files. See the F2 bullet below for the full list.


## A. Front door (user-visible bugs in the advertised quick start)

- [x] **A1 — `manager.latency()` crashes in the flagship tutorial.** *Fixed.*
  Root causes (all pre-existing): (1) `class_file` entries pointing at files
  inside the pyrtc package were exec'd as duplicate modules, producing class
  objects that missed the descriptor registry — `pyrtc/component_loading.py`
  now resolves them to canonical modules and is the single implementation
  shared by `config_schema`, `config_runtime`, and the manager (was 3 copies);
  (2) bare-name lookup could return a shadowing submodule instead of the
  class (`pyrtc/hardware/synthetic_shwfs.py` file vs `SyntheticSHWFS` class) —
  lookup now skips modules and sees through the shadow; (3) relative
  `class_file` paths were resolved against the CWD — `validate_system_config`
  now resolves all relative path fields against the config file's directory
  *before* validation. Regression tests in `tests/test_manager.py`.

- [x] **A2 — The synthetic demo doesn't visibly converge.** *Fixed.* The
  example fed the loop an identity-placeholder IM; even the analytic response
  matrix diverges instantly (near-square, unregularized pinv). The demo now
  calibrates the IM through the live pipeline (DOCRIME, ~5 s, `conditioning:
  30`) exactly like a real AO system: residual RMS 0.99 → 0.010, Strehl
  0.25 → 0.97, in both soft and hard (RPC-driven) tutorials.
  `ensure_identity_interaction_matrix` → `ensure_synthetic_interaction_matrix`;
  `SyntheticWFC.sync_system_config` publishes `display_grid_size` so SHM
  planning matches the synthetic layout. Regression test:
  `tests/system/test_synthetic_convergence.py` (closed < 0.5 × open RMS).

## B. Performance (the README's #1 design constraint)

- [x] **B3 — Zero-alloc hot-path reads.** *Done.* pyshmem 1.0.5 forwards
  `out=` through `read_new()`/`read_new_async()` (tested); pyrtc
  `read_stream(..., out=)` forwards it, and preallocated buffers are used in
  `SlopesProcess.compute_signal` (full WFS image), all `Loop` integrators
  (signal + wfc), and `WavefrontCorrector.send_to_hardware` (wfc). `out=` is
  ignored for GPU-attached streams by design.

- [x] **B4 — Perf baseline gate in CI.** *Done.* The checker previously only
  verified metric *presence*; `check_perf_baseline.py --max-ratio` now fails
  on latency metrics above the threshold or throughput metrics below its
  inverse (unit-tested in `tests/perf/`). CI runs the perf-smoke comparison
  with `--max-ratio 5.0` (deliberately generous: the committed baseline is
  from the fast lab host and GitHub runners are slow/noisy); use
  `--max-ratio 1.5` on the lab host — the AO-loop bench passes that today.

- [ ] **B5 — Profile the per-write lock.** Benchmarks show no regression
  today (baseline ratio ~1.01). Revisit only if 60x60-GPU-scale profiles show
  the flock; the fix would be a designed pyshmem addition, not a pyrtc
  workaround.

## C. Architecture and code health

- [x] **C6 — Split `Pipeline.py`.** *Done.* Implementation now lives in:
  `pyrtc/streams.py` (pyshmem stream policy + SHM planning), `pyrtc/rpc.py`
  (`hardwareLauncher`/`Listener` JSON socket protocol), `pyrtc/manager.py`
  (component runtimes, `RTCManager`, `launch_component`, `work`), and
  `pyrtc/component_loading.py` (class resolution). `pyrtc.pipeline` is a
  re-export shim kept for one release; all internal imports use the new
  modules. Tests that monkeypatch by module path were retargeted.

- [x] **C7 — Harden the hard-RTC RPC.** *Done.* Protocol v1 envelope
  (mismatches rejected with an error reply), type-safe `set` coercion
  (booleans round-trip; `_coerce_property_value`), `run` passes an explicit
  `args` list and returns JSON-serializable results (NumPy converted via
  `item()`/`tolist()`), child error strings propagate to
  `hardwareLauncher.last_error`, `run(..., timeout=)` applies a per-call
  socket timeout (the hard tutorial now uses it for `compute_im`), and the
  listener stops cleanly on RTC disconnect instead of raising
  `BrokenPipeError`. `Listener.handle_request` is pure/testable;
  `tests/test_rpc.py` covers dispatch, coercion, lifecycle, and a
  socketpair round trip.

- [x] **C8 — Hygiene pass.** *Done.* Dead commented code removed from
  `work()` and `Loop.py`; vestigial `RELEASE_GIL` attribute removed;
  `run_soft_rtc.py` now delegates spec planning to
  `expected_output_shm_specs_for_config` instead of reimplementing it.

## D. Testing and platform posture

- [x] **D9 — Extend the coverage gate (round one).** *Done for
  `streams` (85%), `rpc` (82%), `component_loading` (90%), `latency` (79%)* —
  all added to the `pytest.ini` gate; combined gated total 85.5% under CI
  conditions (`-m "not gpu"`). New test files: `tests/test_rpc.py`,
  `tests/test_streams.py`, `tests/test_component_loading.py`.
  **Round two (still open):** `Loop.py` (71%), `SlopesProcess.py` (50%),
  `manager.py` (74%) need a dedicated test-writing session before they can
  join the gate without dropping the floor.

- [x] **D10 — GPU test lane.** *Done.* `gpu` marker registered in
  `pytest.ini`; `tests/test_gpu_streams.py` covers GPU producer with CPU
  mirror, NumPy reads from CPU consumers, CUDA-tensor reads from GPU
  consumers, tensor writes, the uint16 CPU fallback, and `read_new(out=)` on
  the mirror — 6 tests, all passing on the lab machine. Every test is
  `skipif(not pyshmem.gpu_available())` (GitHub runners have no CUDA) and the
  lane deselects cleanly with `-m "not gpu"`.

- [x] **D11 — Windows semantics.** *Done.* README release posture and the
  streams guide now state Windows is soft-RTC-only for the 1.x line (named
  shared memory dies with the last handle; no hard-RTC restart/reattach).

## E. Docs and release

- [x] **E12 — Docs + version.** *Done.* `guides/streams.rst` added to the
  docs toctree; `api_reference.rst` + generated rst stubs cover
  `streams`/`rpc`/`manager`/`component_loading`; version bumped to **1.1.0**
  with a full CHANGELOG entry; CLAUDE.md architecture/testing sections
  describe the new module layout, gate set, and `gpu` marker. (Still open:
  the committed built HTML under `docs/source/_build` is stale — consider
  removing build output from the repo when next regenerating docs.)

## Final verification checklist (run before calling this done)

- `python -m pytest` (coverage-gated suite) and `pytest tests/system
  tests/notebooks -q --no-cov`
- `ruff check pyrtc tests benchmarks examples`
- `pyrtc-ao-loop-bench` + `benchmarks/check_perf_baseline.py` against the
  committed baseline (confirm B3 didn't regress; last pre-B3 run was ratio
  ~1.01)
- Live run of `examples/synthetic_shwfs/synthetic_shwfs_soft_rtc_example.py`
  (expect calibration ~5 s, then residual_rms ≈ 0.01, strehl ≈ 0.97)

## F. Repo-wide conventions

- [x] **F1 — Standardize naming to snake_case across the entire repository
  (no backwards compatibility).** *Done.* The bulk of the rename landed in
  `1984a60` (config keys, descriptors, examples, docs, manager adapter,
  component bases, hardware adapters) and this pass closed the remaining
  holes:
  - **`benchmarks/perf_smoke.py`:** `numIters=` kwarg mismatch in the
    `measure_execution_time` smoke call (the function signature uses
    `num_iters`); the call was raising `TypeError` at runtime, which
    `tests/perf/test_perf_smoke.py` caught on the first run.
  - **`pyrtc/gui/manager_adapter.py`:** the streams-payload key `inputRole`
    was the last camelCase dict key in the GUI bridge; it now matches the
    surrounding `output_component` / `input_components` / `component_stream`
    keys as `input_role`.
  - **`pyrtc/loop.py` + `pyrtc/hardware/ncpa_optimizer.py`:** the two
    `Loop` matrix attributes were the last capitalized scalar names
    (`self.IM`, `self.CM`); renamed to `self.im` / `self.cm` across
    the class, the `@gain.setter` guard, the docstring, the debug
    comments, and the `comp_correction(cm=...)` jit kernel. All call
    sites in `tests/system/test_system_flow.py`, `tests/test_loop.py`,
    and the four `examples/{pywfs,shwfs}/*_soft_rtc_example.py` examples
    were updated in the same commit.
  - **`examples/pywfs/pywfs_example_OOPAO.ipynb`:** the OOPAO tutorial
    notebook was the only remaining place that referenced the old loop
    API directly (`loop.CM`, `loop.CMMethod`, `loop.numDroppedModes`,
    `loop.tikhonovReg`, `loop.lastSingularValueFit`, `loop.CM`,
    `loop.plotSingularValues()`, `loop.computeCM(...)`,
    `numDroppedModes=...`, `confWFC["numModes"]`); everything now uses
    the snake_case names.
  - **Synthetic tutorial docstrings:** the "loop IM" wording in
    `examples/synthetic_shwfs/synthetic_shwfs_{soft,hard}_rtc_example.py`
    is updated to match the attribute rename.
  - **Stale docs build output:** `docs/source/_build/` is stale (still
    renders the pre-snake_case API and `gpu_device` → `numItersIM` etc.
    on every page); not committed in this branch, but `docs/Makefile` will
    regenerate it. The release-cleanup note in E12 still stands for
    whoever next runs `make html` to commit the output.

  Verified post-rename: `python -m pytest tests/` → 291 passed, 1
  skipped; `pytest tests/system tests/notebooks` → 3 passed;
  `pytest tests/perf` → 19 passed; `ruff check pyrtc tests benchmarks
  examples` → clean. The GUI adapter `input_role` change is purely
  internal to `manager_adapter.py` and the generated `streams` config
  block; no on-disk configs depended on the old key.

- [x] **F2 — Final camelCase sweep + PEP 8 hygiene.** *Done.* F1 left a
  handful of camelCase identifiers in `pyrtc/loop.py` (the last
  numba-kernel parameters and the private hot-path read buffers) and a
  scattering of PEP 8 issues across the rest of the package. This pass
  closed all of them:
  - **`pyrtc/loop.py`:** the `leaky integrator_numba` / `leak integrator_gpu`
    parameter `resconstructionMatrix` → `reconstruction_matrix` (the
    `resconstruction` typo was also fixed), the `leak integrator_gpu` local
    `slopes_GPU` → `slopes_gpu`, and the pre-allocated hot-path read
    buffers `self._signalBuffer` / `self._wfcBuffer` →
    `self._signal_buffer` / `self._wfc_buffer`. All call sites in the
    integrators and the test fixture were updated together.
  - **`pyrtc/slopes_process.py`:** `self._imageBuffer` →
    `self._image_buffer`.
  - **`pyrtc/wavefront_corrector.py`:** `self._wfcBuffer` →
    `self._wfc_buffer`.
  - **`pyrtc/hardware/pi_modulator.py`:** local `originalDirectory` →
    `original_directory` (and the `os.chdir` call that restores it).
  - **`pyrtc/hardware/specula_interface.py`:** the local alias of the
    external `specula.cpuArray` is now `cpu_array`, the `SimpleNamespace`
    key it is stored under is `cpu_array`, and every
    `self._bindings.cpuArray(...)` call site is now
    `self._bindings.cpu_array(...)`. The external
    `specula.cpuArray` *import* name is unchanged — that is the
    specula library's API and we do not own it.
  - **`tests/test_loop.py` / `tests/test_manager.py`:** updated for the
    new attribute names (`loop._signal_buffer` / `loop._wfc_buffer`,
    fake-launcher `configFile` → `config_file`).
  - **PEP 8 hygiene (whole repo, excluding `pyrtc/gui/` and
    `pyrtc/scripts/viewer*` which are bound to Qt API):** 299 ruff
    auto-fixes applied (W292 missing trailing newlines, W293
    blank-line whitespace, W291 trailing whitespace); 14 manual
    whitespace fixes; tab indentation in multi-line imports converted
    to spaces in `pyrtc/__init__.py`, `pyrtc/optimizer.py`,
    `pyrtc/science_camera.py`, `pyrtc/slopes_process.py`,
    `pyrtc/utils.py`, and `pyrtc/wavefront_sensor.py`. The repo now
    passes `ruff check --select E,F,W pyrtc tests benchmarks examples`
    (W rules are not in the default config but are not violations
    either after this pass).

  Verified post-sweep: `python -m pytest tests/` → 291 passed, 1
  skipped; `pytest tests/system tests/perf` → 3 + 19 passed; `pytest
  tests/notebooks` → 1 passed; `ruff check --select E,F,W pyrtc tests
  benchmarks examples` → clean. The `resconstructionMatrix` rename also
  fixed a long-standing typo that the previous search-based audit had
  not caught.


## Deferred / future pyshmem additions

Only two things could not be expressed with pyshmem's native API during the
migration; neither is needed today. If profiling or features ever demand
them, design them properly in pyshmem rather than working around it in pyrtc:

1. **Per-frame provenance metadata** — only if frame-accurate cross-process
   lineage becomes a requirement again (the old root/upstream timestamps were
   dropped in favor of cross-stream `count`/`write_time` sampling).
2. **Single-writer lock-free write fast path** — only if the per-write flock
   ever shows up in profiles (it does not today).

(pyshmem addition already made and shipped as 1.0.5: `out=` passthrough on
`read_new`/`read_new_async` — uncommitted in /localhome/dev/pyshmem.)
