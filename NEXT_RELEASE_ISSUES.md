# pyRTC Next Release Issue Set

This document turns the post-1.0 roadmap into implementation-ready GitHub issues.
It is written to be useful for direct issue creation and as high-context input for future AI coding sessions.

The central thesis for the next release is:

- keep pyRTC fast in the hot path
- make hard-RTC operable as a system, not just as a collection of primitives
- improve interoperability with the AO ecosystem
- reduce the amount of internal knowledge needed to configure, launch, observe, and debug a real system

The repository already has a solid base:

- core components and public API are stable
- soft-RTC onboarding is strong via `examples/synthetic_shwfs/`
- low-level hard-RTC orchestration exists in `pyRTC/Pipeline.py`
- partial config validation exists in `pyRTC/utils.py`
- telemetry capture exists in `pyRTC/Telemetry.py`
- logging, benchmarks, and release packaging are in good shape for a `1.0.x` baseline

The main gap is the layer above the primitives: system description, orchestration, observability, interoperability, and operator UX.

## Suggested Release Slicing

If this work is spread across several releases or milestones, the cleanest sequence is:

### Milestone A: Core platform foundation

- Issue 01: system-level config schema and validation CLI
- Issue 02: component capability descriptors
- Issue 03: pyRTC Manager backend
- Issue 04: supervision, restart policy, and health checks
- Issue 07: latency tracing and timing budgets

### Milestone B: Ecosystem and data interoperability

- Issue 05: structured telemetry sessions and capture manifests
- Issue 06: AOTPy export support
- Issue 10: SPECULA bridge
- Issue 11: plugin API

### Milestone C: Operator and integrator UX

- Issue 08: Manager GUI MVP
- Issue 09: schema-driven config editor GUI
- Issue 12: docs and examples overhaul

## Design Constraints For All Issues

These constraints should remain true throughout the roadmap:

- Do not add avoidable overhead inside steady-state real-time loops.
- Prefer explicit control-plane metadata and validation over implicit conventions.
- Keep Linux as the primary operational target unless there is a deliberate cross-platform effort.
- Keep optional integrations isolated so core install and import behavior remain lightweight.
- Preserve the current public import surface unless there is a strong compatibility reason to change it.
- Treat `pyRTC/hardware` as reference integrations unless a specific adapter is being productized.
- Design every new system-level feature to work first with the synthetic SHWFS example.

---

## Issue 01

Status: Completed on branch `issue-01-system-config-validation` and merged into `dev` on 2026-03-08.

### Title

Add system-level configuration schema and `pyrtc-validate-config` CLI

### Why this matters

Current validation is useful but incomplete. The repo validates pieces of `wfs`, `wfc`, and `loop` configuration in `pyRTC/utils.py`, but it does not yet validate a whole AO system as one coherent object.

Without a system-level schema:

- hard-RTC startup failures happen too late
- GUI tooling will become brittle and custom-coded per component
- stream mismatches are hard to catch before runtime
- examples and docs rely too much on implicit maintainer knowledge

This issue is the foundation for the manager backend, GUI, plugin discovery, and reliable integrations.

### Current repo anchors

- `pyRTC/utils.py`
- `tests/test_config_validation.py`
- `examples/synthetic_shwfs/config.yaml`
- `docs/source/guides/getting_started.rst`
- `docs/source/guides/architecture.rst`

### Goals

- Define a whole-system schema for pyRTC YAML configuration files.
- Validate individual component sections and cross-component consistency.
- Add a CLI to validate configs without launching the system.
- Produce human-readable validation errors.
- Make the validation logic reusable by future GUI and manager layers.

### Non-goals

- Do not build the GUI in this issue.
- Do not redesign every config key in the repo.
- Do not require all legacy examples to adopt a radically new format in one step.

### Proposed deliverables

- A system schema module, likely something like `pyRTC/config_schema.py` or `pyRTC/configuration.py`.
- A `validate_system_config` entry point.
- A `pyrtc-validate-config` CLI script registered in `pyproject.toml`.
- Focused tests covering valid and invalid whole-system configurations.
- Documentation describing the schema and the validation CLI.

### Schema expectations

The initial schema should support at least:

- top-level sections such as `wfs`, `slopes`, `loop`, `wfc`, and optional `psf`, `telemetry`, `manager`, `streams`
- per-section required keys and value types
- optional metadata such as `name`, `description`, `mode`, `tags`
- cross-component checks such as:
  - `loop` expecting a signal shape compatible with `slopes`
  - `wfc.numModes` matching loop control assumptions
  - stream names not colliding unintentionally
  - functions listed in `functions` actually existing on the component
- launch metadata for future manager use, such as:
  - launch mode
  - restart policy
  - ports or port allocation policy
  - log policy overrides

### Implementation notes

- Start simple with Python-native validation code rather than introducing a heavy dependency unless there is a strong reason.
- Keep validators composable: system-level validation should call component-level validators, not replace them.
- Separate parsing, normalization, and validation. That makes GUI and manager reuse easier.
- Make validation errors actionable. Error messages should say what is wrong, where it was found, and what shape or value was expected.
- Consider returning a normalized config object that later layers can consume directly.

### CLI behavior

Suggested interface:

```bash
pyrtc-validate-config examples/synthetic_shwfs/config.yaml
pyrtc-validate-config path/to/system.yaml --format json
```

Expected behavior:

- exit code `0` for valid configs
- nonzero exit code for invalid configs
- plain-text output by default
- optional JSON output for automation and GUI integration

### Acceptance criteria

- A valid synthetic SHWFS config passes whole-system validation.
- Representative malformed configs fail before startup with clear diagnostics.
- Validation catches at least one cross-component inconsistency, not only per-field type errors.
- The CLI is documented and covered by tests.

### Suggested test cases

- missing required top-level section
- invalid type for scalar values
- `functions` contains a nonexistent method
- incompatible `numModes` between `wfc` and `loop`
- invalid signal shape assumptions between `slopes` and `loop`
- valid minimal soft-RTC config
- valid hard-RTC config with manager-specific launch metadata

### Follow-on issues enabled by this work

- Issue 02
- Issue 03
- Issue 08
- Issue 09

---

## Issue 02

Status: Implemented on branch `issue-02-component-capability-descriptors` on 2026-03-08.

### Title

Add component capability descriptors for schema-driven tooling and extension metadata

### Why this matters

If config validation and GUI generation are hardcoded by component class name, pyRTC will become difficult to scale. The system needs machine-readable component metadata that describes config fields, default values, runtime behavior, stream IO, and optional capabilities.

This is the key abstraction that lets pyRTC become a platform rather than a fixed set of handwritten examples.

### Current repo anchors

- `pyRTC/pyRTCComponent.py`
- `pyRTC/WavefrontSensor.py`
- `pyRTC/SlopesProcess.py`
- `pyRTC/Loop.py`
- `pyRTC/WavefrontCorrector.py`
- `pyRTC/ScienceCamera.py`
- `pyRTC/Telemetry.py`

### Goals

- Define a standard descriptor model for pyRTC components.
- Expose descriptor data for built-in components.
- Use descriptors to power config validation and future GUI forms.
- Support plugin-provided components later without custom manager logic.

### Non-goals

- Do not implement plugin loading in this issue.
- Do not require every hardware adapter to become fully descriptor-driven immediately.

### Proposed descriptor content

Each component descriptor should be able to express at least:

- component type name
- Python class path
- base category such as `wfs`, `slopes`, `loop`, `wfc`, `science_camera`, `telemetry`
- required config fields
- optional config fields and defaults
- field types and validation rules
- available worker functions and their meaning
- input streams consumed
- output streams produced
- whether the component supports hard-RTC launch as a child process
- optional external dependencies
- optional calibration artifacts such as darks, flats, interaction matrices

### Suggested implementation shape

- Add a descriptor class or typed dictionary representation.
- Allow components to expose a classmethod such as `describe()`.
- Keep component-specific logic near the component class when possible.
- Add a registry for built-in descriptors.

### Design note

Do not overfit the first version to GUI form generation only. The descriptor model should also help:

- config validation
- stream introspection
- documentation generation
- plugin discovery
- manager status display

### Acceptance criteria

- Built-in core components expose descriptors.
- The synthetic SHWFS system can be described entirely from descriptor metadata plus config.
- The descriptor API is documented enough for future third-party component authors.

### Suggested test cases

- descriptor fields present for each core component
- descriptors expose stable required keys
- descriptors can be consumed by the system validator
- descriptors handle optional fields cleanly

### Follow-on issues enabled by this work

- Issue 03
- Issue 08
- Issue 09
- Issue 11

---

## Issue 03

Status: Completed on branch `issue-03-rtc-manager-backend` on 2026-03-08. This looks ready to merge into `dev`.

### Title

Introduce a `pyRTC Manager` backend for full-system orchestration

### Why this matters

Hard-RTC exists today, but it is still a low-level pattern rather than a productized system feature. The orchestration primitives in `pyRTC/Pipeline.py` are useful, but users still need to manually compose multi-process systems, manage launch ordering, and reason about process state themselves.

The next step is a manager layer that owns the lifecycle of a whole RTC system.

### Current repo anchors

- `pyRTC/Pipeline.py`
- `examples/synthetic_shwfs/config.yaml`
- `examples/synthetic_shwfs/synthetic_shwfs_soft_rtc_example.py`
- `examples/synthetic_shwfs/synthetic_shwfs_hard_rtc_example.py`
- `examples/sharp_lab/config.yaml`
- `examples/sharp_lab/config_pywfs.yaml`
- `examples/sharp_lab/sharp_lab_shwfs_soft_rtc_example.py`
- `examples/sharp_lab/sharp_lab_shwfs_hard_rtc_example.py`
- `examples/sharp_lab/sharp_lab_pywfs_soft_rtc_example.py`
- `examples/sharp_lab/sharp_lab_pywfs_hard_rtc_example.py`
- `tests/test_manager.py`

### Implemented outcome

The manager backend now exists as a first-class orchestration layer inside `pyRTC/Pipeline.py` rather than as a one-off example pattern.

Delivered pieces:

- `RTCManager` with `from_config_file(...)`, `from_config(...)`, `validate()`, `start()`, `stop()`, `status()`, and `get_component()`
- runtime wrappers for soft-RTC and hard-RTC component launch paths
- per-component status snapshots and manager-level lifecycle state
- structured startup failure handling and ordered shutdown
- explicit top-level mode override at the manager API, including friendly aliases like `mode="soft"` and `mode="hard"`
- shared config support where one YAML can back both soft and hard modes
- normalized relative path handling for config file artifacts and hard-RTC component launch files
- manager-driven synthetic SHWFS examples for both soft and hard modes
- manager-driven SHARP lab SHWFS and PyWFS examples for both soft and hard modes

The examples now also demonstrate the intended operator-facing distinction between modes:

- soft mode returns live Python objects, so code such as `loop.gain = 0.10` is valid and shown directly
- hard mode returns control proxies, so reads and writes use `getProperty(...)`, `setProperty(...)`, and `run(...)`

### What this issue does not include

These remain follow-on work, not blockers for closing Issue 03:

- supervision and restart policy behavior beyond basic launch/stop lifecycle handling
- health checks and degraded-state transitions
- GUI or config-editor layers
- broader docs polish beyond the example-path overhaul already landed

### Goals

- Build a manager abstraction that loads a validated system config and launches a whole RTC graph.
- Support soft-RTC and hard-RTC orchestration through one public control-plane API.
- Track per-component state and lifecycle transitions.
- Centralize shutdown, error handling, and startup sequencing.

### Non-goals

- Do not build the GUI in this issue.
- Do not rewrite the hot-path component logic.
- Do not force every example to switch immediately.

### Proposed public API

The manager should support a workflow such as:

```python
from pyRTC.Pipeline import RTCManager

manager = RTCManager.from_config_file("examples/synthetic_shwfs/config.yaml")
manager.validate()
manager.start()
status = manager.status()
manager.stop()
```

### Required capabilities

- load config from file or dict
- validate or require validated config
- instantiate soft-RTC components directly when appropriate
- launch hard-RTC child processes when configured
- expose status snapshots for all components
- support graceful stop ordering
- capture startup failures in a structured form

### Lifecycle state model

At minimum, define states such as:

- `created`
- `validated`
- `starting`
- `running`
- `degraded`
- `stopping`
- `stopped`
- `failed`

### Design notes

- Do not embed GUI assumptions in the manager API.
- The manager should be serializable enough to provide JSON status for future CLI or GUI use.
- The manager should treat synthetic soft-RTC examples as first-class supported systems, not only hardware deployments.
- Keep the existing `hardwareLauncher` usable as a lower-level primitive while introducing a higher-level orchestrator.

### Suggested implementation pieces

- orchestration types housed in `pyRTC/Pipeline.py`
- component runtime wrappers for soft and hard modes
- normalized config to runtime mapping
- startup dependency ordering
- stop-order dependency handling
- structured status model

### Acceptance criteria

- A manager can launch and stop the synthetic SHWFS system from one config. Completed.
- A manager can represent at least one hard-RTC example using the existing launcher infrastructure. Completed.
- Status reports include per-component state. Completed.
- Startup and shutdown errors are surfaced in structured form. Completed.

### Suggested test cases

- manager launches valid soft-RTC graph
- manager handles failed child launch cleanly
- manager stop is idempotent
- manager status returns machine-readable state for all components

### Follow-on issues enabled by this work

- Issue 04
- Issue 08
- Issue 09
- Issue 12

---

## Issue 04

Status: Completed on the current branch on 2026-03-09. Ready to merge into `dev`.

### Title

Add supervision, restart policy, and health checks to the Manager

### Why this matters

Launching a system is only the first step. Operators need to know when a process is unhealthy, whether it restarted, why it failed, and what they should do next. This is what turns a launcher into an operations layer.

### Current repo anchors

- `pyRTC/Pipeline.py`
- current logging system in `pyRTC/logging_utils.py`
- any future `pyRTC/manager.py` from Issue 03

### Implemented outcome

The manager now includes a first-pass supervision and health layer that stays in the control plane rather than the real-time hot path.

Delivered pieces:

- per-component health/status metadata in `RTCManager.status()` including state, PID, start time, uptime, last heartbeat, last success time, last failure time, restart count, last error, restart policy, and log file path when configured
- manager supervision polling with explicit `refresh_health()` and `restart_component(...)` entry points
- hard-RTC child health checks using process-alive detection plus launcher RPC responsiveness
- degraded-state handling for live-but-unresponsive children and failed-state handling for exited children
- restart policy support for `never`, `on-failure`, and `always`, with per-component overrides via manager config
- manager config validation for supervision-related fields including restart policies, health-check timing, RPC timeout, and log-path configuration
- regression coverage for degraded children, failed-child restart behavior, repeated restart counting, and structured health metadata

Verification completed on 2026-03-09:

- full test suite passed: `168 passed` via `pytest --no-cov`

### Goals

- Add health modeling and supervision to manager-owned processes and components.
- Support restart policies.
- Track key operational metadata.
- Prepare a status API that a GUI or CLI can consume directly.

### Non-goals

- Do not build a distributed system.
- Do not add heavyweight metrics infrastructure unless justified.

### Required health model

Track at least:

- current state
- PID for child processes where applicable
- start time
- uptime
- last heartbeat or last successful interaction time
- restart count
- last exception or failure message
- log file path if configured

### Restart policy support

Initial policy set could be:

- `never`
- `on-failure`
- `always`

Optional later enhancement:

- backoff timing
- max restart count
- restart cooldown

### Health signal options

The initial implementation does not need deep instrumentation. It can start with:

- process alive checks
- manager-side RPC responsiveness for hard-RTC children
- optional component heartbeat timestamps
- failure to respond within timeout moves state to degraded or failed

### Acceptance criteria

- A failed child process is detected and reflected in manager status.
- Restart policies are applied consistently.
- Operators can distinguish `running`, `degraded`, and `failed` states.
- Status data is available in a structured format suitable for GUI use.

### Suggested test cases

- child exits unexpectedly and manager records failure
- `on-failure` restart policy relaunches child
- repeated failures increment restart count and preserve last error
- status export contains health metadata

### Follow-on issues enabled by this work

- Issue 08
- Issue 12

---

## Issue 05

Status: Completed on the current branch on 2026-03-09. Ready to merge into `dev`.

### Title

Add structured telemetry sessions and capture manifests

### Why this matters

The current telemetry implementation is intentionally lightweight and useful for debugging, but it relies too much on in-memory knowledge of dtype and shape for reconstruction. That makes offline analysis, replay, and ecosystem export harder than it should be.

The next release should make telemetry captures self-describing.

### Current repo anchors

- `pyRTC/Telemetry.py`
- `tests/test_telemetry.py`
- stream usage across `WavefrontSensor`, `SlopesProcess`, `Loop`, `WavefrontCorrector`, and `ScienceCamera`

### Implemented outcome

Telemetry capture is now session-based, self-describing, and NumPy-native.

Delivered pieces:

- per-session telemetry directories under `dataDir` with one subdirectory per captured stream
- standard NumPy capture products per stream: `frames.npy` and `timestamps.npy`
- JSON metadata alongside each stream plus one session-level `session.json`
- simple user-facing capture flow centered on `Telemetry.save(...)` and `Telemetry.read_last_save()`
- grouped multi-stream capture via `Telemetry.save([...], ...)` and config-driven capture via `Telemetry.save_configured_streams(...)`
- offline helpers to list sessions, load session metadata, and reopen captures into a per-stream mapping with `frames`, `timestamps`, and `metadata`
- regression coverage for single-stream saves, grouped multi-stream saves, metadata reconstruction, corrupted metadata, and missing capture files

Verification completed on 2026-03-09:

- full test suite passed: `170 passed` via `pytest --no-cov`

### Goals

- Define a structured telemetry session concept.
- Persist capture metadata on disk alongside raw data.
- Support multi-stream grouped captures.
- Preserve lightweight writing behavior suitable for operational debugging.

### Non-goals

- Do not build a full archival database.
- Do not force every capture to use a complex container format immediately.

### Proposed session format

Each session should include:

- session identifier
- creation timestamp
- pyRTC version
- host metadata
- config file path or embedded normalized config subset
- stream records including:
  - stream name
  - dtype
  - shape
  - number of frames
  - sampling assumptions if known
  - file path
  - optional semantic tags such as `wfs`, `signal`, `wfc`, `psf`

This can be implemented as:

- raw binary data files remain for speed
- one JSON or YAML manifest per session captures metadata

### Design notes

- Keep session metadata versioned so future exporters can evolve safely.
- Consider adding utility helpers to enumerate and load sessions.
- Make room for time metadata even if timestamps are not available for every frame in the first version.

### Acceptance criteria

- A user can capture one or more streams into a self-describing session.
- A session can be reopened later without relying on in-memory object state.
- Session metadata includes enough information to support AOTPy export later.

### Suggested test cases

- single-stream session capture
- multi-stream grouped capture
- manifest reconstruction of dtype and shape
- corrupted manifest or missing file failure path

### Follow-on issues enabled by this work

- Issue 06
- Issue 12

---

## Issue 06

### Title

Add AOTPy export support for pyRTC telemetry sessions

### Why this matters

This is one of the most direct ways to make pyRTC part of the broader AO open-source ecosystem. If pyRTC telemetry can be exported into a community-facing format, users can analyze and exchange results without writing custom one-off converters.

### Current repo anchors

- telemetry work from Issue 05
- `pyRTC/Telemetry.py`
- examples under `examples/synthetic_shwfs/` and `examples/scao/`

### Goals

- Map pyRTC telemetry sessions into AOTPy structures.
- Provide a documented export workflow.
- Keep exporter logic isolated from the real-time loop.

### Non-goals

- Do not re-architect pyRTC around AOTPy internals.
- Do not block core installs on AOTPy unless installed as an optional extra.

### Suggested implementation shape

- add an optional exporter module such as `pyRTC/exporters/aotpy_export.py`
- add an extra dependency group if appropriate
- add a CLI such as `pyrtc-export-aotpy`

### Export mapping work

This issue needs careful design work on metadata mapping. At minimum, define how pyRTC concepts map to AOTPy concepts for:

- WFS image streams
- slope or signal streams
- control vectors
- science camera outputs
- session metadata and run metadata

The implementation should explicitly document:

- what pyRTC information maps cleanly
- what requires assumptions or approximation
- what is not exported yet

### Acceptance criteria

- A synthetic example telemetry session can be exported through a documented CLI or Python API.
- Exported output passes basic round-trip or load validation.
- Export limitations are clearly documented.

### Suggested test cases

- export from synthetic SHWFS telemetry session
- missing optional dependency gives clear guidance
- malformed session metadata fails clearly

### Docs required

- one guide showing pyRTC capture to AOTPy export
- one note describing current export coverage and limitations

### Follow-on issues enabled by this work

- Issue 12

---

## Issue 07

Status: Completed on the current branch on 2026-03-10. Ready to merge into `dev`.

### Title

Add component latency tracing and end-to-end timing budget tooling

### Why this matters

pyRTC already emphasizes performance and includes benchmark infrastructure, but operational users need more than static benchmark tables. They need to know where time is going in a live system and which component is the bottleneck.

This issue adds observability without compromising the existing performance-first philosophy.

### Current repo anchors

- `benchmarks/`
- `pyRTC/latency.py`
- `pyRTC/scripts/measure_latency.py`
- `pyRTC/pyRTCComponent.py`
- logging system and shared-memory metadata in `pyRTC/Pipeline.py`

### Implemented outcome

Latency tracing is now built into the shared-memory transport and exposed through
one consistent control-plane API.

Delivered pieces:

- `RTCManager.latency(...)` for end-to-end or explicit stream-path latency measurement
- a shared `pyRTC/latency.py` module with structured report models, stream-path inference, text formatting, JSON-ready payloads, and histogram support
- a rewritten `pyrtc-measure-latency` CLI that supports explicit pairs, explicit stream paths, or config-driven inferred paths
- lineage-aware SHM metadata in `ImageSHM` so latency can be derived from stream-boundary timestamps rather than inferred only from sampled write histories
- base-class stream helpers in `pyRTCComponent` so components propagate lineage metadata through registered input and output streams
- migrated core pipeline components (`WavefrontSensor`, `SlopesProcess`, `Loop`, `WavefrontCorrector`, and `ScienceCamera`) onto the shared stream read/write helpers for latency lineage propagation
- config support for stream-lineage overrides through `sourceStreams` and `lineageSource`, with backward-compatible validation for stream component metadata
- viewer compatibility updates for the expanded SHM metadata layout
- a manager-based latency example in the synthetic SHWFS soft-RTC tutorial so users can inspect `manager.latency()` in a runnable end-to-end example

The reporting layer now gives:

- full-loop latency summary statistics
- per-boundary breakdowns across inferred or explicit stream paths
- machine-readable JSON output for automation
- one operator-facing max-speed estimate derived from the full-loop P99 latency

Verification completed on 2026-03-10:

- focused regression coverage passed: `60 passed` via `pytest tests/test_pipeline.py tests/test_measure_latency_cli.py tests/test_manager.py tests/test_config_validation.py tests/test_wavefront_corrector.py tests/test_synthetic_example.py --no-cov -q`
- viewer regression coverage passed: `19 passed` via `pytest tests/test_viewer_cli.py tests/test_viewer_boundary.py --no-cov -q`

### Goals

- measure end-to-end latency across component boundaries
- report per-component timing where practical
- expose results through a CLI and machine-readable output
- keep runtime overhead controllable and off by default when necessary

### Non-goals

- do not add per-iteration heavy logging in hot loops
- do not require external tracing infrastructure for the first version

### Possible implementation strategies

- add optional timestamps or sequence markers at stream boundaries
- extend shared-memory metadata or accompanying side channels carefully
- aggregate measurements in batches rather than logging each frame
- reuse or extend the existing latency CLI

### Key design requirement

Tracing must be explicitly designed for low overhead. The default mode should remain lean. Tracing should be opt-in or limited to sampling modes.

### Acceptance criteria

- a user can run a command and receive an end-to-end latency breakdown for a running system
- at least one bottleneck signal is attributable to a specific component or boundary
- reports can be saved as JSON for later analysis

### Suggested test cases

- synthetic latency collection produces consistent structured output
- tracing can be disabled fully
- malformed or missing streams produce usable diagnostics

### Follow-on issues enabled by this work

- Issue 08
- Issue 12

---

## Issue 08

### Title

Build a Manager GUI MVP for launch, status, logs, and restart control

### Why this matters

The repo already has viewer tooling and operator-facing logging, but users still need terminals and internal knowledge to run complex systems. The first GUI should focus on operations, not config editing.

This issue should validate the manager backend as an operator product.

### Current repo anchors

- `pyRTC/scripts/view.py`
- `pyRTC/scripts/viewer_core.py`
- logging infrastructure
- manager work from Issue 03 and Issue 04

### Goals

- create a GUI that can load a system config, start and stop the system, show component states, show logs, and restart components
- keep the GUI modular and manager-driven
- support at least the synthetic SHWFS system and one hard-RTC path

### Non-goals

- do not build the config editor here
- do not build a giant monolithic application with deep custom logic in the view layer

### MVP feature set

- load config file
- validate config
- start system
- stop system
- show per-component state and health
- show recent logs or open log files
- restart a failed or selected component
- optionally launch viewer commands or expose common viewer suggestions

### UX design direction

Keep this simple and operational:

- left side: system tree and component state
- main area: status details and logs
- top-level controls: validate, start, stop, restart selected
- clear color and state conventions for healthy, degraded, failed, stopped

### Technical notes

- The GUI should consume a structured manager API, not inspect child process internals directly.
- The GUI should avoid assumptions specific to one example system.
- If PyQt remains the chosen stack, keep dependencies optional under an extra.

### Acceptance criteria

- a user can run the synthetic system from the GUI without using terminals
- per-component state and restart actions work
- logs are visible in a practical way
- the GUI remains responsive while the system runs

### Suggested test cases

- smoke test for window initialization
- manager integration mocked or real for state transitions
- component restart updates visible state

### Follow-on issues enabled by this work

- Issue 09
- Issue 12

---

## Issue 09

### Title

Build a schema-driven configuration editor GUI

### Why this matters

Once schema and descriptors exist, pyRTC can stop treating YAML as an expert-only interface. A good config editor will reduce onboarding friction, lower user error rates, and make the platform much easier to adopt.

### Current repo anchors

- outputs from Issue 01 and Issue 02
- any GUI base created in Issue 08

### Goals

- create a config editor generated from schema and component descriptors
- provide inline validation and documentation
- allow both form-based and raw YAML editing

### Non-goals

- do not replace text editing for expert users
- do not force all schema evolution through GUI assumptions

### MVP feature set

- create new config from template
- open existing config
- edit fields through forms
- validate in real time or on demand
- save valid config
- show raw YAML view for advanced users

### Design notes

- The UI should be schema-driven. Avoid hardcoding every field.
- The editor should clearly distinguish required fields, defaults, optional fields, and advanced fields.
- Field descriptions should come from descriptor metadata where possible.
- The GUI should surface cross-component validation errors, not only field-level errors.

### Acceptance criteria

- a user can create a valid synthetic SHWFS config entirely from the editor
- invalid configs show actionable validation messages
- expert users can still edit raw YAML directly

### Suggested test cases

- descriptor-driven form generation for core components
- save and reload without semantic drift
- invalid edits are surfaced before launch

### Follow-on issues enabled by this work

- Issue 12

---

## Issue 10

### Title

Build a SPECULA bridge and reference integration example

### Why this matters

SPECULA is a strategic interoperability target because it represents a modern GPU-oriented AO simulation workflow. A bridge would make pyRTC more relevant as a controller and experimentation environment inside the larger AO ecosystem.

### Current repo anchors

- `examples/scao/`
- `pyRTC/hardware/OOPAOInterface.py`
- `pyRTC/Pipeline.py`
- synthetic and simulation example infrastructure

### Goals

- establish one realistic integration path between pyRTC and SPECULA
- isolate bridge logic from pyRTC core runtime code
- document supported workflows clearly

### Non-goals

- do not attempt full bidirectional feature parity in the first issue
- do not tightly couple pyRTC internals to SPECULA internals

### Choose one first-class workflow

The first implementation should pick one narrow, useful path such as:

- SPECULA provides simulated WFS frames and pyRTC runs slopes plus control
- pyRTC provides commands back to a SPECULA-driven simulation
- offline import or replay of SPECULA-generated data into pyRTC telemetry or streams

Do not try to support all workflows at once.

### Proposed deliverables

- bridge module in a clearly isolated package area
- one documented example
- compatibility notes including dependency and platform assumptions

### Acceptance criteria

- one SPECULA integration example runs end to end
- unsupported assumptions are documented rather than left implicit
- failures due to missing optional dependencies are clear and non-destructive

### Suggested test cases

- bridge import smoke test when optional dependency is present
- graceful skip behavior when dependency is absent
- synthetic or mocked adapter data-path tests where feasible

### Follow-on issues enabled by this work

- Issue 12

---

## Issue 11

### Title

Add a plugin API for third-party components, exporters, and integrations

### Why this matters

If pyRTC is going to participate meaningfully in the AO open-source ecosystem, outside users need a supported way to extend it. A plugin API makes it possible to ship integrations and site-specific components without forcing everything into the core repository.

### Current repo anchors

- public package exports in `pyRTC/__init__.py`
- hardware adapters in `pyRTC/hardware/`
- future descriptor and manager work from Issue 02 and Issue 03

### Goals

- define an extension mechanism for components and exporters
- allow plugin-provided descriptors and optional manager discovery
- keep core install lightweight and stable

### Non-goals

- do not create a complicated plugin marketplace
- do not destabilize the core public API

### Potential extension targets

- custom components
- hardware adapters
- telemetry exporters
- manager or GUI integration helpers

### Suggested implementation shape

- use Python entry points or a similarly standard discovery mechanism
- require plugins to expose descriptor metadata and version compatibility info
- provide one internal plugin-style example or test fixture to prove the design

### Acceptance criteria

- a third-party package can register at least one component or exporter
- the manager and validator can discover plugin metadata without manual code edits
- missing or incompatible plugins fail clearly

### Suggested test cases

- plugin discovery
- plugin descriptor loading
- incompatible plugin version handling
- optional plugin absence does not break core import surface

### Follow-on issues enabled by this work

- future ecosystem integrations beyond AOTPy and SPECULA

---

## Issue 12

### Title

Overhaul documentation and examples around user journeys, manager workflows, and interoperability

### Why this matters

The current docs are substantially better than they were pre-1.0, but the next release needs docs that match the product pyRTC is becoming. Once manager, validation, telemetry sessions, and interoperability land, the examples and guides must be reorganized around how users actually work.

### Current repo anchors

- `README.md`
- `docs/source/guides/`
- `docs/source/examples/`
- `examples/synthetic_shwfs/`
- `examples/scao/`

### Goals

- reorganize docs by user journey rather than only by internal architecture
- add examples that demonstrate the new manager, validation, telemetry export, and ecosystem bridges
- make the repo easier for first-time users, integrators, and operators

### Non-goals

- do not rewrite docs before the underlying features exist
- do not create placeholder guides without runnable examples

### Recommended doc tracks

- first-time user track
- simulation and algorithm development track
- hardware integration track
- operator and deployment track
- telemetry and offline analysis track

### Specific documentation outputs

- update README quick start to use the most stable onboarding path
- add a manager quick start guide
- add a config validation guide
- add a telemetry sessions guide
- add an AOTPy export guide
- add a SPECULA integration guide if Issue 10 lands
- add a troubleshooting guide for hard-RTC startup and recovery

### Example priorities

- keep `examples/synthetic_shwfs/` as the canonical no-hardware path
- add one manager-based soft-RTC launch example
- add one manager-based hard-RTC launch example
- add one telemetry export example

### Acceptance criteria

- a new user can install pyRTC, validate a config, launch a synthetic system, inspect status, and export telemetry by following the docs
- at least one integrator-facing example demonstrates how to adapt or register a custom component
- docs contain clear support posture and optional dependency guidance

### Suggested validation

- docs build cleanly
- example commands in docs are exercised where practical
- no placeholder links or stale command names remain

---

## Additional Backlog Candidates

These are good follow-on issues if the core set above lands well.

### Add session replay as a first-class feature

Replay recorded WFS, signal, or command streams into a running or simulated system for debugging, regression testing, and offline controller development.

### Add stream registry and runtime introspection

Create a discoverable registry of active streams with dtype, shape, producer, consumer, and semantic meaning.

### Add manager-oriented CLI tools

Potential commands:

- `pyrtc-manager start`
- `pyrtc-manager stop`
- `pyrtc-manager status`
- `pyrtc-manager logs`
- `pyrtc-export-aotpy`
- `pyrtc-validate-config`

### Add optional ecosystem compatibility CI

Non-blocking or optional CI coverage for interoperability layers such as AOTPy export and SPECULA bridge would help keep those efforts real over time.

---

## Practical Advice On Sequencing Work

If the goal is maximum value per unit effort before the next release, do the work in this order:

1. Issue 01 because everything else becomes easier once the config is explicit.
2. Issue 02 because schema-driven metadata prevents future hardcoded special cases.
3. Issue 03 because users need a real orchestration layer before they need a GUI.
4. Issue 04 because reliable operation matters more than visual polish.
5. Issue 05 because telemetry should become self-describing before ecosystem export.
6. Issue 06 and Issue 10 because interoperability is most credible after data and control-plane structure exist.
7. Issue 08 and Issue 09 because the GUI will be much easier once the manager and schema are stable.
8. Issue 12 continuously as features stabilize, but do the major docs pass after the core interfaces settle.

## Recommended Labels

To keep the GitHub issue tracker usable, these labels would help:

- `architecture`
- `config`
- `manager`
- `gui`
- `telemetry`
- `interop`
- `performance`
- `docs`
- `examples`
- `high-priority`
- `good-first-design-task`
- `blocked-by-foundation`

## Definition Of Success For The Next Release

The next release should feel materially different from `1.0.0` in the following ways:

- a user can validate a full system config before launch
- a user can launch and supervise a system through a manager rather than manual script composition
- telemetry captures are self-describing and exportable
- pyRTC has at least one credible ecosystem bridge or exporter
- the path from install to operation requires less maintainer knowledge
- the repo still preserves its performance-first design in the hot path
