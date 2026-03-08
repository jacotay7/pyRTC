# pyrtcao 1.0.0 Release Plan

## Goal

Publish a credible, maintainable `1.0.0` release to PyPI under the distribution name `pyrtcao` while keeping the import package as `pyRTC` for the initial stable release.

## Naming Decision

- PyPI distribution name: `pyrtcao`
- Import package name: `pyRTC`
- Repository name: `pyRTC`
- Short user-facing guidance: `pip install pyrtcao`, then `import pyRTC`

This is the lowest-risk path for `1.0.0` because it avoids a disruptive package-directory rename while still differentiating the published package from the existing `pyrtc` project on PyPI.

## Release Standard

`1.0.0` should not ship until all of the following are true:

1. A clean environment can install the built wheel from TestPyPI and run `import pyRTC` successfully.
2. The README provides a successful first-run path for a new user.
3. The docs contain no user-facing `TODO` placeholders.
4. Package metadata, classifiers, naming, and support claims are internally consistent.
5. CI validates build, install, tests, and docs.
6. The public API surface and support expectations are documented.
7. The release process is repeatable without manual guesswork.

## Priority Workstreams

### 1. Packaging and Distribution

- [x] Rename the PyPI distribution from `pyRTC` to `pyrtcao`
- [x] Consolidate package metadata in `pyproject.toml`
- [x] Align versioning and classifiers with a stable release
- [x] Add project URLs, keywords, and explicit support metadata
- [x] Validate wheel and sdist builds in CI
- [x] Test installation from built artifacts in a clean environment
- [x] Keep console scripts user-facing as `pyrtc-*`

### 2. Documentation Baseline

- [x] Replace the placeholder getting-started guide
- [x] Replace the placeholder example guide
- [x] Rewrite the README around install, quick start, architecture, and examples
- [x] Add a package naming note: `pip install pyrtcao`, `import pyRTC`
- [x] Add an architecture guide for soft-RTC vs hard-RTC and shared memory
- [x] Add troubleshooting guidance for optional dependencies and runtime setup
- [ ] Add citation guidance once the paper status is final

### 3. Examples and User Onboarding

- [x] Identify one canonical quick-start example
- [x] Ensure there is one installable, package-user-friendly example
- [x] Clean up placeholders under `examples/`
- [x] Document external example dependencies such as OOPAO clearly
- [x] Provide a no-hardware verification path

Current onboarding choice:

- Canonical first-run example: `examples/synthetic_shwfs/`
- Secondary simulator-backed example with external dependency: `examples/scao/` via OOPAO

### 4. Testing and Compatibility

- [x] Keep current Linux coverage green
- [ ] Add package-install tests against the built distribution
- [ ] Expand coverage targets beyond the current narrow list
- [ ] Add explicit tests for the stable public API surface
- [ ] Add negative-path tests for missing optional dependencies and bad configs
- [ ] Decide the support stance for GPU code paths and test it accordingly
- [ ] Either test Windows/macOS or narrow support claims before release

### 5. Runtime Logging and Error Handling

- [ ] Design and document one shared logging configuration model for library code, scripts, and multi-process hard-RTC components
- [ ] Provide colored, timestamped terminal logging with clear log levels by default for user-facing scripts
- [ ] Provide optional file logging with a straightforward default log directory and filename strategy
- [ ] Support configuring log level and log folder through environment variables and script-level/runtime overrides
- [ ] Default to `INFO` and avoid debug spam from functions that run continuously in the real-time pipeline
- [ ] Ensure logger configuration can be propagated cleanly to child processes in hard-RTC / multi-process mode
- [ ] Add a low-overhead pattern for error reporting around control-plane code without adding per-iteration exception/logging overhead to hot real-time loops
- [ ] Define explicit guidance for which errors should raise, which should warn, and which should be logged and suppressed in non-real-time paths
- [ ] Add targeted logging/error-handling tests for script entry points, environment-variable configuration, and multi-process startup behavior

Implementation constraints:

- Real-time loop functions should not pay recurring logging or exception-handling costs in the steady-state hot path.
- Debug logging in high-frequency pipeline functions should be opt-in and structured so it can be disabled completely in normal operation.
- Terminal and file logs should use the same shared logger configuration so soft-RTC and hard-RTC runs produce consistent records.
- Multi-process runs should preserve enough component/process identity in log output to debug startup, wiring, and failure modes.

### 6. Tooling and Release Operations

- [x] Add a release checklist to the repo
- [x] Add a changelog for `1.0.0`
- [x] Add TestPyPI publishing workflow
- [x] Add PyPI publishing workflow gated on tags or releases
- [x] Add docs build validation to CI
- [x] Expand lint coverage beyond a hard-coded file list

### 7. Community and Maintenance Readiness

- [x] Consolidate maintainer workflow into the docs Developer Guide
- [x] Define what is considered stable in `1.0.0`
- [x] Define support expectations and issue-reporting guidance
- [x] Clarify what hardware integrations are maintained vs community-provided

## Current Assessment

### Strengths

- The core package architecture is already organized around reusable AO components.
- The public package surface is intentionally exported in `pyRTC/__init__.py`.
- The test suite currently passes locally on Linux with Python 3.12.
- The repository already includes smoke and performance-oriented checks.

### Known Gaps

- Trusted publishing still needs to be configured in GitHub, TestPyPI, and PyPI before the publish workflow can be used.
- The current support claims are still broader than the verified CI surface.
- Logging and error-handling behavior are not yet standardized across scripts, library components, and multi-process hard-RTC runs.



## Recommended Implementation Order

1. Packaging rename and metadata cleanup
2. README rewrite and package naming clarification
3. Docs completion for getting started and examples
4. Release workflow and TestPyPI dry run
5. Example cleanup and installation verification
6. Shared logging and error-handling implementation
7. Support policy, changelog, and contributor docs

## Notes

- For `1.0.0`, prefer compatibility and clarity over broad refactors.
- Avoid renaming the import package unless there is a strong reason to accept a breaking change now.
- Treat GPU support, hardware integrations, and platform claims conservatively unless they are validated by tests.
- Treat logging and error handling as control-plane concerns first: improve debuggability and operator visibility without adding noise or measurable overhead to the steady-state real-time loop.