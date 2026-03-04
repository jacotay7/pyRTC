# pyRTC Refactor + Productization Plan

## Guiding Product Goals
1. Keep `import pyRTC` clean, stable, and ergonomic for users.
2. Preserve real-time performance while increasing code transparency.
3. Support both script and notebook-first workflows.
4. Improve reliability with stronger CI quality gates and system-level testing.
5. Add fast-fail configuration validation to reduce runtime surprises.

## Current Snapshot (objective)
- Core quality baseline is stronger than before (pytest + coverage gate green; latest full run: 38 tests passing, coverage >80%).
- Core package API surface is now explicit:
  - `pyRTC/__init__.py` now uses explicit exports and `__all__`.
- Core import hygiene is substantially improved:
  - no wildcard imports remain in `pyRTC/*.py`.
  - staged `ruff` lint gate is active in CI for all migrated core files.
- Wildcard import cleanup has now been completed across core, hardware, viewer, and sharp_lab examples.
- Remaining debt is primarily dead/commented legacy blocks and broader lint/type normalization in examples.
- Boundary separation is still incomplete:
  - viewer scripts still import internal helpers broadly.
- Test strategy still leans unit-heavy:
  - missing explicit system-flow tests and notebook smoke tests in CI.
- Configs are still permissive YAML dictionaries with limited validation.

## Progress Since Plan Start
- ✅ WS0 (partial): Added staged `ruff` gate in CI; pytest+coverage gate retained.
- ✅ WS1 (major): Explicit top-level API exports in `pyRTC/__init__.py`; API smoke tests added.
- ✅ WS6 (core slice): Removed wildcard imports and cleaned lint issues across all core modules (`pyRTC/*.py`).
- ✅ WS6 (expanded): Removed active wildcard imports across `pyRTC/*.py`, `pyRTC/hardware/*.py`, legacy viewer scripts, and `examples/sharp_lab/*.py`.
- ⏳ WS2: Core/viewer separation work started indirectly via API stabilization; viewer cleanup now focused on CLI hardening/docs alignment.
- ✅ WS3: Config validation layer implemented in `utils`, centrally enforced in `pyRTCComponent`, and covered by tests.
- ⏳ WS4: System/notebook CI smoke tests now started (system + notebook surrogate tests added; dedicated smoke CI job added).
- ⏳ WS5: Performance regression instrumentation not yet implemented.

## Workstreams

### WS0 — Quality gates in CI (immediate)
Scope:
- `.github/workflows/python-install.yml`
- new lint/type config files (`pyproject.toml` or dedicated config files)

Actions:
- Add `ruff` as required CI gate for:
  - unused imports (`F401`), wildcard imports (`F403/F405`), bare `except` (`E722`), undefined names (`F821`).
- Add optional strict type check gate (`mypy` or `pyright`) in staged mode:
  - begin with `pyRTC` core only.
- Keep pytest + coverage as required gate.
- Add a second CI job for notebook/example smoke tests (non-hardware path).

Acceptance:
- CI blocks on lint failures and test failures.
- CI reports include lint, tests, and coverage in one run.

Status:
- In progress: staged `ruff` gate implemented for migrated core files and API smoke test.
- Remaining: widen lint/type coverage to hardware/viewer/examples and add notebook smoke job.

---

### WS1 — Public API contract for `import pyRTC`
Scope:
- `pyRTC/__init__.py`
- optional `docs/source` API reference pages

Actions:
- Replace wildcard exports with explicit imports and explicit `__all__`.
- Define API tiers:
  - **Public stable**: classes/functions users should import from `pyRTC`.
  - **Internal**: module-level symbols not guaranteed stable.
- Add API smoke tests:
  - `import pyRTC`
  - `from pyRTC import Loop, WavefrontSensor, WavefrontCorrector, SlopesProcess, ScienceCamera, Optimizer, Telemetry`

Acceptance:
- `import pyRTC` remains user-friendly and stable.
- Public API changes are intentional and reviewed.

Status:
- Largely complete for current public classes/utilities.
- Ongoing: keep API contract stable while cleaning downstream modules.

---

### WS2 — Core vs Viewer boundary
Scope:
- `pyRTC/*`
- `pyRTC/scripts/*`

Actions:
- Keep `pyRTC` dependency-light and runtime-core focused.
- Treat viewer tooling as separate optional application layer:
  - no viewer dependency required for `import pyRTC`.
  - install path via `.[viewer]` remains optional.
- Minimize direct imports of internals from viewer:
  - prefer narrow, explicit imports from stable core modules.
- Add packaging clarity in docs:
  - “Core install”, “Core + GPU”, “Core + Viewer”.

Acceptance:
- Core package imports and tests succeed without viewer dependencies.
- Viewer scripts operate when `.[viewer]` extras are installed.

Status:
- Core import path is clean.
- CLI migration completed: `pyRTC/scripts` now backs terminal tools (`pyrtc-view`, `pyrtc-shm-monitor`, `pyrtc-clear-shms`, `pyrtc-view-launch-all`) and `pyRTCView/` has been removed.

---

### WS3 — Config validation layer (fail fast)
Scope:
- component constructors and config loading path across `pyRTC/*`

Actions:
- Introduce explicit config schemas per component (`loop`, `wfs`, `slopes`, `wfc`, `psf`, etc.).
- Validate required keys, types, value ranges, and enum-like options.
- Error format requirements:
  - include component name, key path, expected type/range, actual value.
- Maintain backwards compatibility with existing YAML keys where possible.

Candidate implementation options:
1. `pydantic` models (strongest validation UX, additional dependency).
2. dataclass + manual validators (no extra dependency, more code).

Acceptance:
- Invalid config fails before threads/processes start.
- Common user mistakes produce actionable error messages.

---

### WS4 — Test pyramid upgrade (unit + system + notebook)
Scope:
- `tests/*`
- new `tests/system/*`
- new `tests/notebooks/*` or notebook smoke tooling

Actions:
- Keep existing unit tests.
- Add system-level pipeline tests that execute realistic component chains using simulated hardware paths.
- Add notebook smoke test:
  - execute a lightweight notebook cell sequence from `examples/scao/pywfs_example_OOPAO.ipynb` (or equivalent script surrogate) in CI-safe mode.
- Add contract tests for optional modes:
  - CPU-only install path.
  - GPU-enabled path (smoke only where environment supports it).

Acceptance:
- CI verifies at least one end-to-end system flow.
- Notebook usage path remains functional.

---

### WS5 — Performance transparency and regression checks
Scope:
- `tests/perf/*` (or `benchmarks/*`)
- docs/performance notes

Actions:
- Define measurable latency/throughput metrics for critical loops.
- Add repeatable microbenchmarks with clear environment notes.
- In CI, run lightweight performance smoke checks (not strict hard-RT pass/fail), and store trends/artifacts.
- Add runtime telemetry hooks/documentation so users can inspect timing without hidden behavior.

Acceptance:
- Performance-sensitive paths are measured and visible.
- Regressions are detectable early, even if CI hardware differs from lab hardware.

---

### WS6 — Import hygiene + dead code cleanup
Scope:
- `pyRTC/*.py`
- `pyRTC/hardware/*.py`
- `pyRTC/scripts/*.py`
- `examples/**/*.py`

Actions:
- Replace all wildcard imports with explicit imports.
- Remove unused imports and undefined implicit symbol usage.
- Remove large commented-out dead code while preserving useful explanatory comments/docstrings.

Acceptance:
- No wildcard imports in targeted Python files.
- `ruff` passes for unused-import and star-import rules.

Status:
- Wildcard import removal complete for core/hardware/viewer/sharp_lab examples.
- Remaining: dead/commented code reduction and optional expansion of strict lint gates to all examples.

## Priority Order (recommended)
1. WS0 (CI gates), WS1 (public API), WS2 (core/viewer boundary).
2. WS3 (config validation) and WS4 (system + notebook tests).
3. WS6 (hygiene cleanup) in parallel with WS3/WS4.
4. WS5 (performance transparency) once system tests are stable.

## First Two Implementation Sprints

### Sprint A (1–2 weeks)
- Add `ruff` CI gate and baseline config.
- Refactor `pyRTC/__init__.py` to explicit exports + `__all__`.
- Add API smoke tests for `import pyRTC` and key classes.
- Clean wildcard imports in lowest-risk modules first (`Telemetry`, `Modulator`, `pyRTCComponent`).

Progress:
- Completed and extended through all core modules (`Loop`, `SlopesProcess`, `Pipeline`, WFS/WFC, etc.).

### Sprint B (1–2 weeks)
- Create config validation MVP for 2–3 core components (`loop`, `wfs`, `wfc`).
- Add one end-to-end system test using simulated path.
- Add notebook smoke test strategy (direct `nbconvert` execution or equivalent Python script mirror).
- Document core/viewer install and runtime boundaries.

Current focus:
- Start WS3 config validation MVP (`loop`, `wfs`, `wfc`) and WS4 system/notebook smoke test lane.

## Risks and Mitigations
- **Risk:** Breaking existing user imports.
  - **Mitigation:** explicit API compatibility list + import smoke tests.
- **Risk:** Validation breaks legacy YAML configs.
  - **Mitigation:** compatibility mode + deprecation warnings.
- **Risk:** CI performance checks flaky across runners.
  - **Mitigation:** trend/report checks over strict hard thresholds.

## Definition of Done
- `import pyRTC` is stable, explicit, and documented.
- CI enforces lint + tests + coverage and includes system/notebook smoke coverage.
- Core and viewer responsibilities are clearly separated.
- Config validation catches common errors before runtime.
- Performance-critical paths are measured and transparent.
