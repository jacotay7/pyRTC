.. architecture

Architecture Overview
=====================

This guide describes the current `pyRTC` execution model at a level useful for developers and system integrators preparing a real deployment.

System Model
------------

`pyRTC` is built around a small set of AO component abstractions that exchange data through shared-memory streams.
The main intent is to keep algorithm logic, device-facing logic, and runtime orchestration separable.

The primary components are:

- `WavefrontSensor`: captures or produces wavefront-sensor images
- `SlopesProcess`: transforms wavefront-sensor images into intermediate signal products
- `Loop`: reconstructs control outputs from signal streams
- `WavefrontCorrector`: applies correction vectors to the control hardware or simulator
- `ScienceCamera`: captures science frames and derived metrics
- `Telemetry`: persists selected outputs to disk

Each component can run one or more configured functions in worker threads, driven by the `functions` list in its configuration.

Data Flow
---------

The usual control path is:

1. `WavefrontSensor` produces image streams.
2. `SlopesProcess` reads those images and computes a wavefront signal.
3. `Loop` reads the signal and computes a new correction vector.
4. `WavefrontCorrector` consumes the correction and forwards it to hardware or simulation.
5. `ScienceCamera` and `Telemetry` optionally observe the loop state and persist outputs.

This is not the only legal layout, but it is the default conceptual model to keep in mind while reading the codebase.

Soft-RTC vs Hard-RTC
--------------------

`pyRTC` currently supports two broad operating styles.

Soft-RTC
~~~~~~~~

In `soft-RTC`, components are instantiated directly in the same Python process.
This is the simplest path for:

- simulation workflows
- algorithm development
- rapid prototyping
- low-complexity lab setups

This mode keeps debugging simple and reduces process orchestration overhead.

Hard-RTC
~~~~~~~~

In `hard-RTC`, hardware-facing components can run in separate Python processes and communicate through launcher and listener utilities.
This is useful when you need:

- process isolation from device SDKs
- clearer boundaries between subsystems
- reduced coupling between control logic and hardware code
- operational patterns that avoid keeping everything in one interpreter

The tradeoff is additional orchestration complexity and a higher burden on deployment discipline.

Shared Memory
-------------

Shared-memory streams are the main contract between components.
In practice, this means the shape, dtype, and semantic meaning of streams must be treated as part of the system design, not as incidental implementation details.

When extending the system:

- keep stream naming predictable
- keep dtypes explicit
- document expected shapes for custom components
- validate config and stream assumptions early

Configuration Model
-------------------

Configuration is typically expressed as nested dictionaries or YAML sections such as `wfs`, `slopes`, `loop`, and `wfc`.
Each component reads only the keys relevant to its role.

This design is simple and practical, but it means that release stability depends heavily on clear config documentation and conservative schema changes.

Extension Model
---------------

Most real deployments will subclass or adapt the core AO components for site-specific hardware.
The `pyRTC.hardware` package exists to show that pattern.

Treat those hardware files as reference integrations:

- they demonstrate expected extension points
- they are useful exemplars for new integrations
- they are not a guarantee of universal compatibility across SDK versions or operating systems

Operational Guidance
--------------------

For first deployments:

- start with simulation or `soft-RTC`
- validate stream shapes and config files before integrating real devices
- add hardware one component at a time
- keep GPU assumptions optional until validated on the target machine

Stability Guidance for 1.0
--------------------------

For the `1.0.0` release line, the most stable contract is the core component model and the public imports exposed at package level.
Hardware adapters, GPU-specific paths, and platform-specific deployment details should still be treated cautiously unless they are validated in the target environment.