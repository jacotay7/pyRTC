# Changelog

All notable changes to `pyrtcao` will be documented in this file.

## 1.0.0 - 2026-03-07

First stable public release of `pyrtcao`.

This release establishes the initial supported package, CLI, documentation, and
CI/release surface for the `1.0.x` line. The published distribution name is
`pyrtcao`, the import name remains `pyRTC`, and the user-facing project name is
`pyrtc`.

### Added

- PyPI distribution packaging as `pyrtcao` while preserving `import pyRTC`.
- Stable console-script entry points with the `pyrtc-*` prefix:
	`pyrtc-view`, `pyrtc-view-launch-all`, `pyrtc-shm-monitor`,
	`pyrtc-clear-shms`, `pyrtc-measure-latency`, `pyrtc-core-bench`, and
	`pyrtc-ao-loop-bench`.
- Canonical no-hardware onboarding workflow in `examples/synthetic_shwfs/`.
- Shared logging system in `pyRTC.logging_utils` covering scripts, benchmarks,
	launchers, component base classes, and key hardware/control-plane paths.
- Maintainer-facing built-wheel validation helper at
	`python -m pyRTC.scripts.validate_dist_install --dist-dir dist`.
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
	robust when `pyRTC` is resolved as a namespace package.
- API-reference and component docs were reorganized to remove duplicate Sphinx
	object registrations and produce a clean docs build.

### Fixed

- Viewer/CLI import failures that occurred when running from outside the repo
	root or when `pyRTC` was resolved as a namespace package.
- Python 3.9 compatibility issues caused by bare PEP 604 union annotations at
	import time in logging and benchmark modules.
- Missing benchmark-table kernel mappings and multiple Ruff/lint regressions in
	scripts and tests.
- Headless/non-Qt test collection failures caused by eager Qt backend imports
	in the viewer module.
- Documentation import examples that incorrectly recommended
	`from pyRTC import ...` patterns for classes and launch helpers.
- Duplicate Sphinx autodoc warnings caused by repeated object indexing across
	component pages and the API reference.
- Test-suite warning noise from pytest helper imports and third-party startup
	warnings so the suite runs cleanly.

### Testing

- Full repository test coverage for the tracked coverage set now exceeds the
	release gate, reaching 87.53% at release time.
- `pyRTC.Modulator`, `pyRTC.Optimizer`, `pyRTC.Telemetry`, and
	`pyRTC.pyRTCComponent` now have 100% coverage in the tracked release suite.
- `pyRTC.ScienceCamera` coverage was expanded materially as part of release
	stabilization.
- Built-wheel installation, CLI imports, docs builds, performance smoke tests,
	and synthetic system flows are all exercised in the release-facing workflow
	set.

### Notes

- Linux is the primary validated platform for the `1.0.x` line.
- Python 3.9 through 3.13 are covered by the release CI matrix.
- GPU and hardware-specific paths should still be validated in the target
	environment before operational use.
- Hardware adapters in `pyRTC.hardware` should be treated as reference
	integrations and starting points, not guarantees of site-specific SDK
	compatibility.