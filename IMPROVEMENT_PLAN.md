# pyrtc Improvement Plan

Status tracker for the post-pyshmem-migration cleanup. Grounded in the README's
stated goals: performance as a primary design constraint, the synthetic SHWFS
workflow as the front door, a conservative Linux-first release posture, and
ML-assisted control research as a target audience.

**Status as of 2026-06-11 (second pass):** A1, A2, B3, B4, C6, C7, C8, D9,
D10, D11, E12 are done. Verified: 285 tests passed with the extended coverage
gate at 85.5% (GPU lane deselected, as on CI), system + notebook smoke green,
ruff clean, `pyrtc-ao-loop-bench` within 1.5x of the committed baseline, live
soft and hard tutorials converge. Remaining: **F1 (naming standardization)**,
plus the second gate extension round (Loop/SlopesProcess/manager) noted in D9.

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

- [ ] **F1 — Standardize naming to snake_case across the entire repository
  (no backwards compatibility).** Eliminate camelCase everywhere:
  - **Python identifiers**: functions, methods, attributes, and locals
    (`launch_component` → `launch_component`, `get_property` → `get_property`,
    `compute_im` → `compute_im`, `num_modes` → `num_modes`,
    `hardwareLauncher` → rename class to `HardwareLauncher` or fold into the
    rpc rename, …). Class names stay PEP 8 `CapWords`.
  - **Config keys** in YAML system configs and `component_descriptors`
    (`gpu_device` → `gpu_device`, `im_file` → `im_file`,
    `num_dropped_modes` → `num_dropped_modes`, `input_streams` →
    `input_streams`, …) with all example configs and validation updated
    together. No alias layer — old keys become validation errors.
  - **Module filenames**: `WavefrontSensor.py` → `wavefront_sensor.py` etc.,
    updating imports, `component_files` entries, docs, and CI references.
  - **Open question to settle before starting**: whether the import package
    itself becomes lowercase `pyrtc` (full consistency, matches the
    user-facing name; CLAUDE.md currently says the import name stays
    `pyrtc`).
  - Suggested order: (1) config keys + descriptors, (2) Python identifiers
    module-by-module with the test suite green after each, (3) module/file
    renames, (4) docs/README/CLAUDE.md sweep. Each phase is a separate
    commit; RPC property names travel over the wire, so hard-RTC examples
    must be re-run live after phase 2.

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
