# pyRTC Refactor + Productization Plan

## Guiding Product Goals
1. Keep `import pyRTC` clean, stable, and ergonomic for users.
2. Preserve real-time performance while increasing code transparency.
3. Support both script and notebook-first workflows.
4. Improve reliability with stronger CI quality gates and system-level testing.
5. Add fast-fail configuration validation to reduce runtime surprises.

## Current Snapshot (objective)
- Core quality baseline is stronger than before (pytest + coverage gate already green).
- API surface is still implicit and fragile:
  - `pyRTC/__init__.py` uses wildcard exports and duplicates imports.
- Code hygiene debt remains high:
  - ~100 wildcard import matches across core/hardware/examples/viewer.
  - dead/commented code concentrated in `pyRTC/Pipeline.py`, `pyRTC/Loop.py`, `pyRTC/SlopesProcess.py`, `pyRTC/WavefrontCorrector.py`.
- Boundary separation is incomplete:
  - viewer scripts in `pyRTCView/` import core internals directly and use wildcard imports.
- Test strategy still leans unit-heavy:
  - missing explicit system-flow tests and performance regression checks.
- Configs are permissive YAML dictionaries with limited validation.

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

---

### WS2 — Core vs Viewer boundary
Scope:
- `pyRTC/*`
- `pyRTCView/*`

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
- `pyRTCView/*.py`
- `examples/**/*.py`

Actions:
- Replace all wildcard imports with explicit imports.
- Remove unused imports and undefined implicit symbol usage.
- Remove large commented-out dead code while preserving useful explanatory comments/docstrings.

Acceptance:
- No wildcard imports in targeted Python files.
- `ruff` passes for unused-import and star-import rules.

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

### Sprint B (1–2 weeks)
- Create config validation MVP for 2–3 core components (`loop`, `wfs`, `wfc`).
- Add one end-to-end system test using simulated path.
- Add notebook smoke test strategy (direct `nbconvert` execution or equivalent Python script mirror).
- Document core/viewer install and runtime boundaries.

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
