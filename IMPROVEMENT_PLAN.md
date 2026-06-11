# pyRTC Improvement Plan

Status tracker for the post-pyshmem-migration cleanup. Grounded in the README's
stated goals: performance as a primary design constraint, the synthetic SHWFS
workflow as the front door, a conservative Linux-first release posture, and
ML-assisted control research as a target audience.

**Status as of 2026-06-11:** A1, A2, B3, C6, C8 are done (full suite: 235
passed, lint clean). Remaining: C7, D9, D10, B4, D11, E12, plus a final
benchmark re-run against `benchmarks/ao_loop_bench_baseline.json`.

## A. Front door (user-visible bugs in the advertised quick start)

- [x] **A1 — `manager.latency()` crashes in the flagship tutorial.** *Fixed.*
  Root causes (all pre-existing): (1) `classFile` entries pointing at files
  inside the pyRTC package were exec'd as duplicate modules, producing class
  objects that missed the descriptor registry — `pyRTC/component_loading.py`
  now resolves them to canonical modules and is the single implementation
  shared by `config_schema`, `config_runtime`, and the manager (was 3 copies);
  (2) bare-name lookup could return a shadowing submodule instead of the
  class (`pyRTC/hardware/SyntheticSHWFS.py` file vs `SyntheticSHWFS` class) —
  lookup now skips modules and sees through the shadow; (3) relative
  `classFile` paths were resolved against the CWD — `validate_system_config`
  now resolves all relative path fields against the config file's directory
  *before* validation. Regression tests in `tests/test_manager.py`.

- [x] **A2 — The synthetic demo doesn't visibly converge.** *Fixed.* The
  example fed the loop an identity-placeholder IM; even the analytic response
  matrix diverges instantly (near-square, unregularized pinv). The demo now
  calibrates the IM through the live pipeline (DOCRIME, ~5 s, `conditioning:
  30`) exactly like a real AO system: residual RMS 0.99 → 0.010, Strehl
  0.25 → 0.97, in both soft and hard (RPC-driven) tutorials.
  `ensure_identity_interaction_matrix` → `ensure_synthetic_interaction_matrix`;
  `SyntheticWFC.sync_system_config` publishes `displayGridSize` so SHM
  planning matches the synthetic layout. Regression test:
  `tests/system/test_synthetic_convergence.py` (closed < 0.5 × open RMS).

## B. Performance (the README's #1 design constraint)

- [x] **B3 — Zero-alloc hot-path reads.** *Done.* pyshmem 1.0.5 forwards
  `out=` through `read_new()`/`read_new_async()` (tested); pyRTC
  `read_stream(..., out=)` forwards it, and preallocated buffers are used in
  `SlopesProcess.computeSignal` (full WFS image), all `Loop` integrators
  (signal + wfc), and `WavefrontCorrector.sendToHardware` (wfc). `out=` is
  ignored for GPU-attached streams by design.

- [ ] **B4 — Perf baseline gate in CI.** Wire the 10x10/20x20 CPU bench into
  CI with a generous ratio threshold (≥1.5× to absorb runner noise) using
  `benchmarks/check_perf_baseline.py`. GPU columns must be skipped on GitHub
  runners (no CUDA): use `--cpu-only` / system-size flags.

- [ ] **B5 — Profile the per-write lock.** Benchmarks show no regression
  today (baseline ratio ~1.01). Revisit only if 60x60-GPU-scale profiles show
  the flock; the fix would be a designed pyshmem addition, not a pyRTC
  workaround.

## C. Architecture and code health

- [x] **C6 — Split `Pipeline.py`.** *Done.* Implementation now lives in:
  `pyRTC/streams.py` (pyshmem stream policy + SHM planning), `pyRTC/rpc.py`
  (`hardwareLauncher`/`Listener` JSON socket protocol), `pyRTC/manager.py`
  (component runtimes, `RTCManager`, `launchComponent`, `work`), and
  `pyRTC/component_loading.py` (class resolution). `pyRTC.Pipeline` is a
  re-export shim kept for one release; all internal imports use the new
  modules. Tests that monkeypatch by module path were retargeted.

- [ ] **C7 — Harden the hard-RTC RPC.** The `get/set/run` protocol coerces
  with `type(property)(value)` (breaks for bools/None), silently drops `run()`
  return values, and has no protocol version. Add a versioned message
  envelope, safe type coercion, and return values for `run` (now isolated in
  `pyRTC/rpc.py`, so the change is contained). Note: `hardwareLauncher.run`
  accepts a `timeout` argument it ignores — honor it (the hard-RTC tutorial
  currently works around this by touching `processSocket.settimeout` for the
  long `computeIM` call).

- [x] **C8 — Hygiene pass.** *Done.* Dead commented code removed from
  `work()` and `Loop.py`; vestigial `RELEASE_GIL` attribute removed;
  `run_soft_rtc.py` now delegates spec planning to
  `expected_output_shm_specs_for_config` instead of reimplementing it.

## D. Testing and platform posture

- [ ] **D9 — Extend the coverage gate.** `pytest.ini` gates only 7 modules.
  Add `Loop.py`, `SlopesProcess.py`, and the new `streams.py` / `rpc.py` /
  `manager.py` module-by-module, with new tests where coverage falls short.
  (Measure first: `python -m pytest --cov=pyRTC.streams --cov=pyRTC.manager
  --cov=pyRTC.rpc --cov=pyRTC.Loop --cov=pyRTC.SlopesProcess ...`.)

- [ ] **D10 — GPU test lane.** Add a `gpu` pytest marker (register in
  `pytest.ini`) with real-pyshmem-stream tests for the
  `create_stream`/`open_stream` GPU paths (GPU producer + CPU-mirror
  consumer + CUDA-attached consumer + uint16 fallback — a working snippet
  exists in the session history as the "GPU smoke" script). **GitHub runners
  have no CUDA**, so every GPU test needs
  `@pytest.mark.skipif(not pyshmem.gpu_available(), reason="CUDA is not available")`
  and CI should also deselect with `-m "not gpu"`; the lane runs on the lab
  machine before releases.

- [ ] **D11 — Windows semantics.** pyshmem's Windows named shared memory dies
  with the last handle (no POSIX persistence). Hard-RTC restart/reattach
  flows silently assume persistence — document Windows as soft-RTC-only for
  the 1.x line (README release posture + CLAUDE.md).

## E. Docs and release

- [ ] **E12 — Docs + version.** Add a streams guide documenting
  `create_stream`/`open_stream` and the `pyshmem` CLI; refresh the generated
  API docs (source rst updated; built HTML in `docs/source/_build` is stale —
  consider not committing build output); add rst pages for the new
  `pyRTC.streams` / `pyRTC.rpc` / `pyRTC.manager` modules; ship as **1.1.0**
  (`pyproject.toml` still says the old version) with the CHANGELOG
  "Unreleased" section renamed. CHANGELOG should also gain entries for A1/A2
  fixes and the Pipeline split. Update `CLAUDE.md` architecture section for
  the new module layout (it still describes everything under `Pipeline.py`).

## Final verification checklist (run before calling this done)

- `python -m pytest` (coverage-gated suite) and `pytest tests/system
  tests/notebooks -q --no-cov`
- `ruff check pyRTC tests benchmarks examples`
- `pyrtc-ao-loop-bench` + `benchmarks/check_perf_baseline.py` against the
  committed baseline (confirm B3 didn't regress; last pre-B3 run was ratio
  ~1.01)
- Live run of `examples/synthetic_shwfs/synthetic_shwfs_soft_rtc_example.py`
  (expect calibration ~5 s, then residual_rms ≈ 0.01, strehl ≈ 0.97)

## Deferred / future pyshmem additions

Only two things could not be expressed with pyshmem's native API during the
migration; neither is needed today. If profiling or features ever demand
them, design them properly in pyshmem rather than working around it in pyRTC:

1. **Per-frame provenance metadata** — only if frame-accurate cross-process
   lineage becomes a requirement again (the old root/upstream timestamps were
   dropped in favor of cross-stream `count`/`write_time` sampling).
2. **Single-writer lock-free write fast path** — only if the per-write flock
   ever shows up in profiles (it does not today).

(pyshmem addition already made and shipped as 1.0.5: `out=` passthrough on
`read_new`/`read_new_async` — uncommitted in /localhome/dev/pyshmem.)
