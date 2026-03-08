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
- [x] Add package-install tests against the built distribution
- [ ] Expand coverage targets beyond the current narrow list
- [x] Add explicit tests for the stable public API surface
- [ ] Add negative-path tests for missing optional dependencies and bad configs
- [ ] Decide the support stance for GPU code paths and test it accordingly
- [ ] Either test Windows/macOS or narrow support claims before release

Current testing state:

- Public API import coverage exists.
- Config validation coverage exists for bad configuration paths.
- Focused regression coverage exists for the synthetic example, viewer CLI, logging helpers, pipeline launcher behavior, hardware adapter shims, and benchmark/report entry points.
- Built-wheel validation now exists as both CI workflow coverage and a reusable maintainer script.
- A dedicated GitHub Actions smoke workflow now exists for `windows-latest` and `macos-latest` on Python 3.12, but support claims should stay conservative until those runs are green consistently.

### 5. Runtime Logging and Error Handling

- [x] Design and document one shared logging configuration model for library code, scripts, and multi-process hard-RTC components
- [x] Provide colored, timestamped terminal logging with clear log levels by default for user-facing scripts
- [x] Provide optional file logging with a straightforward default log directory and filename strategy
- [x] Support configuring log level and log folder through environment variables and script-level/runtime overrides
- [x] Default to `INFO` and avoid debug spam from functions that run continuously in the real-time pipeline
- [x] Ensure logger configuration can be propagated cleanly to child processes in hard-RTC / multi-process mode
- [x] Add a low-overhead pattern for error reporting around control-plane code without adding per-iteration exception/logging overhead to hot real-time loops
- [ ] Define explicit guidance for which errors should raise, which should warn, and which should be logged and suppressed in non-real-time paths
- [x] Add targeted logging/error-handling tests for script entry points, environment-variable configuration, and multi-process startup behavior

Current logging state:

- Shared logger configuration exists in `pyRTC/logging_utils.py`.
- User-facing scripts, benchmark entry points, primary component superclasses, major hardware adapters, and hardware optimizers now use the shared logger.
- Hard-RTC child processes inherit logging configuration through launcher environment propagation.
- The remaining work is policy and coverage cleanup, not logging-system bootstrapping.

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
- There is now a credible no-hardware onboarding path via `examples/synthetic_shwfs/`.
- The shared-memory viewer and related scripts are substantially more usable than the earlier baseline.
- Logging is now materially more consistent across scripts, components, and multi-process control-plane behavior.

### Known Gaps

- Trusted publishing still needs to be configured in GitHub, TestPyPI, and PyPI before the publish workflow can be used.
- The current support claims are still broader than the verified CI surface.
- Built-wheel installation testing is still not part of the validated release path.
- Support posture for GPU paths, non-Linux platforms, and hardware-specific integrations still needs a firmer release statement.
- Logging policy is implemented technically, but the repo still needs explicit maintainership guidance about when to raise, warn, or suppress in non-real-time control paths.



## Recently Completed

1. Packaging and naming cleanup for the `pyrtcao` distribution while preserving `import pyRTC`.
2. README and docs rewrite around install, architecture, examples, and developer workflow.
3. Canonical no-hardware onboarding example under `examples/synthetic_shwfs/`.
4. Major viewer overhaul with better layouting, controls, and stability.
5. Shared logging rollout across scripts, benchmark tools, core superclasses, launchers, and primary hardware-facing components.
6. Focused regression tests covering logging, onboarding flows, viewer behavior, package public API, and mocked hardware adapters.

## What Is Next

1. Decide and document the exact `1.0.x` support boundary for Linux, GPU paths, and hardware integrations, then align README/docs/metadata to it.
2. Finish the logging/error policy documentation so contributors know which non-real-time failures should raise, warn, or be logged-and-continue.
3. Expand negative-path coverage for optional dependency failures and hardware-adapter import/runtime failure modes.
4. Either add real validation for non-Linux platforms or narrow user-facing support claims before publishing `1.0.0`.
5. Finish the remaining citation/release-adjacent docs once the paper and release artifacts are finalized.

## Recommended Implementation Order From Here

1. Support policy tightening for platforms, GPU paths, and hardware integrations
2. Negative-path testing for optional dependencies and adapter failures
3. Logging/error-handling policy guidance for contributors and maintainers
4. TestPyPI dry run with trusted publishing configured
5. Final release-doc cleanup, citation guidance, and publish readiness review

## Notes

- For `1.0.0`, prefer compatibility and clarity over broad refactors.
- Avoid renaming the import package unless there is a strong reason to accept a breaking change now.
- Treat GPU support, hardware integrations, and platform claims conservatively unless they are validated by tests.
- Treat logging and error handling as control-plane concerns first: improve debuggability and operator visibility without adding noise or measurable overhead to the steady-state real-time loop.