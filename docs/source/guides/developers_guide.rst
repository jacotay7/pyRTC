.. developers guide

Developer Guide
===============

This guide collects the maintainer- and contributor-facing operational guidance for `pyrtc`.

Naming and Packaging
--------------------

For the stable release line:

- user-facing project name: `pyrtc`
- PyPI distribution name: `pyrtcao`
- Python import name: `pyrtc`
- command-line prefix: `pyrtc-*`

This keeps the public-facing name simple while avoiding a PyPI naming conflict.

Local Setup
-----------

.. code-block:: bash

   git clone https://github.com/jacotay7/pyRTC.git
   cd pyRTC
   pip install -r requirements-test.txt
   pip install -e .

Optional extras:

.. code-block:: bash

   pip install -e .[docs]
   pip install -e .[viewer]
   pip install -e .[gui]
   pip install -e .[gpu]
   pip install -e .[hcipy]   # runs the HCIPy tests instead of skipping them

``requirements-test.txt`` already installs the optional packages the suite
exercises (matplotlib, astropy, optuna, hcipy, aotpy).
The ``viewer`` and ``gui`` extras install ``qtpy`` and PySide6 (Qt6). Without
them, ``tests/test_qt_smoke.py`` skips; with them it builds the manager GUI and
the viewer on Qt's ``offscreen`` platform, so it needs no display. To check
PyQt6 as well, install ``PyQt6`` and run it with ``QT_API=pyqt6``.

Day-to-Day Checks
-----------------

Run the main validation commands before opening a pull request or preparing a release candidate:

.. code-block:: bash

   pytest -q
   ruff check pyrtc tests benchmarks
   python -m build
   python -m twine check dist/*

Validate the built wheel in a clean environment:

.. code-block:: bash

   python -m pyrtc.scripts.validate_dist_install --dist-dir dist

If you want to keep the validation environment for inspection instead of using a temporary venv:

.. code-block:: bash

   python -m pyrtc.scripts.validate_dist_install --dist-dir dist --venv-dir wheel-test-env

Continuous Integration
----------------------

GitHub Actions runs on pull requests into ``dev`` and ``main`` and on pushes to
``main``:

- ``python-install.yml``:

  - ``install`` runs the suite with coverage on Python 3.10-3.14, plus the wheel
    checks and the perf smoke gate;
  - ``smoke-system-notebook`` runs the end-to-end system tests and notebooks;
  - ``free-threaded`` runs the suite on 3.14t with ``PYTHON_GIL=0``, without
    ``tests/system``;
  - ``docs`` runs the Sphinx build.

- ``cross-platform-smoke.yml``: macOS and Windows smoke tests.
- ``gui-smoke.yml`` (``qt-offscreen``): the Qt manager GUI and viewer on the
  offscreen platform.
- ``isio-bridge.yml``: builds ImageStreamIO and runs the ISIO bridge tests.
- ``lint.yml``: ``ruff check`` and ``ruff format --check``.
- ``publish-package.yml``: builds and publishes to TestPyPI/PyPI, on a release
  or by hand.

The simulator system tests are timing-sensitive on busy runners; see the
notes on loop delay in ``AGENTS.md`` before tightening their thresholds.

Documentation Workflow
----------------------

Install docs dependencies:

.. code-block:: bash

   pip install -e .[docs]

Build the docs:

.. code-block:: bash

   cd docs/source
   make html

Live preview:

.. code-block:: bash

   cd docs/source
   sphinx-autobuild . _build/html

Benchmark Workflow
------------------

The README benchmark section is generated from a reproducible report captured on a target machine.

Generate a benchmark report:

.. code-block:: bash

    pyrtc-ao-loop-bench \
       --output benchmarks/readme_benchmark_report.json \
       --iterations 300 \
       --warmup 30 \
       --system-sizes 10 20 60 \
       --log-dir logs

Generate markdown tables for the README:

.. code-block:: bash

   python benchmarks/readme_benchmark_table.py \
     --report benchmarks/readme_benchmark_report.json \
     --output benchmarks/readme_benchmark_table.md

Compare the current host report against the committed baseline:

.. code-block:: bash

    python benchmarks/check_perf_baseline.py \
       --current benchmarks/readme_benchmark_report.json \
       --baseline benchmarks/ao_loop_bench_baseline.json

Performance History
-------------------

Every CI run uploads its micro-benchmark report (``perf-smoke-report-py<version>``)
and, from Python 3.12, an end-to-end ``pipeline-latency`` report of the running
synthetic system. ``benchmarks/perf_history.py`` collects those artifacts from
recent runs and prints each metric's latest value against the median of
earlier runs, flagging regressions. CI adds this table to the 3.12 job's
summary; it is informational, because shared runners are noisy.

.. code-block:: bash

    export GH_TOKEN=...   # any token that can read Actions artifacts
    python -m benchmarks.perf_history --repo jacotay7/pyRTC --runs 20
    python -m benchmarks.perf_history --repo jacotay7/pyRTC --artifact pipeline-latency
    # a lab host: keep nightly perf_smoke reports in a directory
    python -m benchmarks.perf_history --from-dir lab_reports/ --plot trends.png --max-ratio 1.3

It exits with status 1 when a metric's latest value exceeds ``--max-ratio``
times its history, so a scheduled lab job can gate on it.

Free-Threaded Python
--------------------

pyrtc runs on free-threaded CPython (3.13t/3.14t), where the worker threads of
a soft-RTC system no longer share one GIL. Everything the core imports
supports it, numba 0.67 included, and the whole test suite passes with the
GIL forced off. The ``free-threaded`` CI job checks both on 3.14t. Keep it
that way:

- don't import extensions that lack free-threading support from core
  modules;
- a single such import turns the GIL back on process-wide (astropy's
  ``erfa`` does, which is one reason ``fits`` is an optional extra);
- ``tests/test_public_api.py`` checks the core imports.

Measured end-to-end latency (``benchmarks/pipeline_latency_bench.py``, synthetic
SHWFS soft-RTC system, 1000 samples, median of two interleaved runs, 8-core
(16-thread) x86-64 host), in microseconds:

.. list-table::
   :header-rows: 1

   * - frame rate, consumers
     - p50 3.14
     - p50 3.14t
     - p99 3.14
     - p99 3.14t
   * - 200 Hz, notify
     - 306
     - 258
     - 444
     - 309
   * - 200 Hz, poll
     - 267
     - 232
     - 393
     - 333
   * - 1 kHz, notify
     - 252
     - 245
     - 413
     - 303
   * - 1 kHz, poll
     - 260
     - 229
     - 376
     - 358

Without the GIL, median latency drops 3-15% and the p99 tail 5-30%. The gain
is modest because the numba kernels already release the GIL (``nogil=True``),
so most of the remaining contention is in Python glue code. One caveat
(#139): on an oversubscribed host (more busy processes than hardware threads),
the free-threaded pipeline's latency grew to 6-19 frames, while the GIL build
stayed at 1-2. Give a free-threaded RTC dedicated cores. To try it:
``uv python install 3.14t``, make a venv with it, and install pyrtc as usual.

Logging Workflow
----------------

The main scripts, benchmark entry points, and hardware launcher paths use the shared `pyrtc` logging helpers.

Default behavior:

- console logging enabled
- level `INFO`
- color enabled when the terminal supports it

Useful controls:

.. code-block:: bash

   export PYRTC_LOG_LEVEL=INFO
   export PYRTC_LOG_DIR=./logs
   python examples/synthetic_shwfs/synthetic_shwfs_soft_rtc_example.py --duration 15

Per-command overrides:

.. code-block:: bash

   pyrtc-view wfs --log-level DEBUG
   python -m benchmarks.perf_smoke --log-file perf.log

Prefer `PYRTC_LOG_DIR` for multiprocess runs so parent and child processes write separate files.

Error-Handling Policy
---------------------

For `1.0.x`, prefer explicit, conservative behavior in non-real-time paths.

Raise exceptions when:

- required configuration is missing or invalid
- file loads or saves fail for requested user-visible artifacts
- startup or hardware-attachment steps fail and the component cannot provide its documented behavior
- a requested optional feature cannot be enabled safely

Warn and continue when:

- the code is falling back from GPU to CPU for a supported code path
- a convenience feature cannot be enabled but the main component behavior still works
- the software can continue safely with a documented default or degraded mode

Log and suppress only when:

- cleanup or teardown is best-effort
- a background diagnostic or optional observer path fails without affecting core control-plane behavior
- repeated operator-facing noise would be less useful than a single earlier warning

Avoid adding per-iteration exception handling or routine logging inside the steady-state real-time loop. Put detailed logging around setup, calibration, file I/O, control-plane state changes, and error boundaries instead.

Contribution Expectations
-------------------------

Contributions are most useful when they improve one or more of the following:

- core AO component reliability
- documentation and onboarding
- example quality
- performance observability
- broadly reusable hardware integration patterns

Before starting larger work:

- open an issue for major interface or architecture changes
- keep bug fixes focused and reproducible
- avoid mixing unrelated refactors with functional changes
- be explicit about platform and dependency assumptions

Component Descriptors
---------------------

`pyrtc` now exposes machine-readable component descriptors for the built-in core components.
These descriptors are intended to support:

- config validation
- future manager and GUI form generation
- stream introspection
- future plugin discovery

Useful entry points from Python are:

.. code-block:: python

   import pyrtc

   catalog = pyrtc.build_descriptor_catalog()
   loop_descriptor = pyrtc.get_component_descriptor("loop")
   wfs_descriptor = pyrtc.wavefront_sensor.describe()
   hardware_delay = loop_descriptor["hardware_delay"]
   default_gain = loop_descriptor["gain"]["default"]

In the REPL, descriptors now render as a compact summary rather than a full dataclass dump, and they support dict-like field lookup by config key.
This means calls such as `pyrtc.loop.describe()["hardware_delay"]` and `pyrtc.loop.describe()["gain"]["default"]` work naturally.

Each descriptor includes:

- top-level config section name
- component class path
- required and optional config fields
- worker functions intended for the `functions` list
- input and output stream metadata
- calibration artifact hints

When adding new built-in components, update `pyrtc/component_descriptors.py` and keep the descriptor aligned with the actual config and stream contract.
Future third-party integrations can also register descriptors programmatically without changing manager-specific code:

.. code-block:: python

   pyrtc.register_component_descriptor(custom_descriptor)

Descriptor-driven validation is intentionally generic and should be paired with component-specific validation for domain rules that cannot be captured as simple field metadata.

Unknown config keys
~~~~~~~~~~~~~~~~~~~

`pyrtc-validate-config` and component construction warn (without failing) about config keys that the component class does not read, such as ``method:`` where the loop reads ``im_method``.
The known keys are the common runtime keys (``class_name``, ``class_file``, ``name``, ``functions``, ``affinity``, ``realtime_priority``, ``gpu_device``, ``input_streams``, ``output_streams``, ``resource``), the descriptor fields, and any ``EXTRA_CONFIG_KEYS`` declared along the class hierarchy. Keys starting with ``_`` are private runtime keys and are never reported.

A subclass of a built-in component that reads keys of its own declares them, which also opts it into the check:

.. code-block:: python

   class MyCamera(pyrtc.WavefrontSensor):
       EXTRA_CONFIG_KEYS = ("serial", "exposure")

A subclass that declares neither ``EXTRA_CONFIG_KEYS`` nor its own ``COMPONENT_DESCRIPTOR`` is not checked, because its extra keys are unknown.

When opening a pull request:

- state the motivation clearly
- explain the user-visible behavior change
- list the validation commands you ran
- call out compatibility or deployment risks

Writing Components
------------------

Components talk to streams only through handles they register:

- Open each stream once and register it with ``register_input_stream(name, shm)`` (streams the component reads) or ``register_output_stream(name, shm)`` (streams it writes). A stream a component both reads and writes, such as the corrector's ``wfc``, is registered as both.
- Use ``read_stream`` / ``write_stream`` with the registered name. Reading a registered input records its ``frame_id`` and ``write_stream`` stamps it on the outputs, which is how frame identity reaches the DM. An unregistered name raises ``KeyError``; a ``<name>_shm`` attribute is not a registration.
- A registered handle belongs to the component: ``close()`` closes it, and registering a different handle under the same name closes the replaced one.
- Source components (wavefront sensors) number their own frames; ``WavefrontSensor.expose`` keeps a private exposure counter, so a simulator that also reads ``wfc`` does not disturb it.

Release anything else the component holds (device SDK handles, extra threads, ring buffers) by overriding ``close()``: call ``super().close(*args, **kwargs)`` first, which stops the component and joins its workers, then release your own resources. Do not put teardown in ``__del__``; the base ``__del__`` calls ``close()``, but it only runs once the worker threads are gone.

Tests that exercise a single method without running ``__init__`` build the component with ``testsupport.bare_component(Cls, inputs={...}, outputs={...})``, which runs the real stream-state initialization and registers the given streams. Tests that run a whole system use ``testsupport.private_synthetic_config`` for private stream names and close the manager (``with RTCManager... as manager`` or ``manager.close()`` in ``finally``).

Hardware Contributions
----------------------

Hardware-facing code is valuable but environment-specific.

For hardware integrations:

- isolate vendor SDK assumptions clearly
- document OS and dependency constraints
- avoid breaking generic component behavior
- provide a minimal usage example when possible
- prefer simulator-backed validation where practical

Support Posture
---------------

The most stable public surface for `1.0.x` is:

- installation as `pyrtcao`
- runtime import as `pyrtc`
- the core AO component model
- the documented shared-memory and configuration concepts
- Linux-based development and deployment workflows

Areas that still need target-environment validation before operational use:

- vendor SDK integrations
- GPU-specific execution paths beyond the documented synthetic benchmark coverage
- multi-process deployment details
- platform-specific driver and device behavior

Current platform stance for `1.0.0`:

- Linux is the primary supported operating system.
- macOS and Windows smoke jobs are useful compatibility signal, but they are not the primary release target.
- Hardware adapters remain environment-specific integrations, not universal support guarantees.

Issue Reporting
---------------

Useful bug reports should include:

- Python version
- operating system
- install method
- whether GPU support was enabled
- whether real hardware or simulation was used
- the smallest reproducible script or config

Release Checklist
-----------------

Before publishing a release candidate:

1. Update the changelog and confirm version metadata.
2. Verify the README matches the install, import, and support story.
3. Confirm docs contain no user-facing placeholders.
4. Run the full validation path:

   .. code-block:: bash

      pytest -q
      ruff check pyrtc tests benchmarks
      python -m build
      python -m twine check dist/*
      python -m pyrtc.scripts.validate_dist_install --dist-dir dist
      cd docs/source && make html

5. Upload to TestPyPI first.
6. Validate installation from TestPyPI in a clean environment.
7. Publish to PyPI only after the TestPyPI install passes.

Publishing Workflow
-------------------

The repository includes `.github/workflows/publish-package.yml`.

Expected usage:

- `workflow_dispatch` with `repository=testpypi` for pre-release uploads
- a published GitHub release, or manual dispatch with `repository=pypi`, for production uploads

This workflow assumes trusted publishing has been configured on both TestPyPI and PyPI.