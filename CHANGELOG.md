# Changelog

All notable changes to `pyrtcao` will be documented in this file.

## 1.0.0 - 2026-03-07

Initial stable release line.

### Added

- PyPI distribution packaging as `pyrtcao` with `import pyRTC` preserved.
- Canonical no-hardware onboarding example in `examples/synthetic_shwfs/`.
- Reworked viewer with better layouting, controls, and stability.
- Shared logging system for scripts, multi-process launchers, component superclasses, and primary hardware adapters.
- Focused regression coverage for viewer behavior, synthetic onboarding, logging helpers, mocked hardware adapters, and hardware-side optimizers.
- Maintainer-facing built-wheel validation helper at `python -m pyRTC.scripts.validate_dist_install --dist-dir dist`.

### Changed

- README and docs rewritten around installation, architecture, examples, troubleshooting, and contributor workflow.
- User-facing console scripts standardized under the `pyrtc-*` prefix.
- Benchmark entry points now use the shared logging configuration model.
- Component and hardware control-plane methods now log state changes and exceptions more consistently.

### Notes

- Linux is the primary validated platform for the `1.0.x` line.
- GPU and hardware-specific paths should still be validated in the target environment before operational use.