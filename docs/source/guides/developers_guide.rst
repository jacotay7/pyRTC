.. developers guide

Developer Guide
===============

This guide collects the maintainer- and contributor-facing operational guidance for `pyrtc`.

Naming and Packaging
--------------------

For the stable release line:

- user-facing project name: `pyrtc`
- PyPI distribution name: `pyrtcao`
- Python import name: `pyRTC`
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
   pip install -e .[gpu]

Day-to-Day Checks
-----------------

Run the main validation commands before opening a pull request or preparing a release candidate:

.. code-block:: bash

   pytest -q
   ruff check pyRTC tests benchmarks
   python -m build
   python -m twine check dist/*

Validate the built wheel in a clean environment:

.. code-block:: bash

   python -m pyRTC.scripts.validate_dist_install --dist-dir dist

If you want to keep the validation environment for inspection instead of using a temporary venv:

.. code-block:: bash

   python -m pyRTC.scripts.validate_dist_install --dist-dir dist --venv-dir wheel-test-env

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

   python -m benchmarks.perf_smoke \
     --output benchmarks/readme_benchmark_report.json \
       --log-dir logs \
     --core-iterations 500 \
     --core-warmup 50 \
     --core-system-sizes 10 20 60

Generate markdown tables for the README:

.. code-block:: bash

   python benchmarks/readme_benchmark_table.py \
     --report benchmarks/readme_benchmark_report.json \
     --output benchmarks/readme_benchmark_table.md

Logging Workflow
----------------

The main scripts, benchmark entry points, and hardware launcher paths use the shared `pyRTC` logging helpers.

Default behavior:

- console logging enabled
- level `INFO`
- color enabled when the terminal supports it

Useful controls:

.. code-block:: bash

   export PYRTC_LOG_LEVEL=INFO
   export PYRTC_LOG_DIR=./logs
   python examples/synthetic_shwfs/run_soft_rtc.py --duration 15

Per-command overrides:

.. code-block:: bash

   pyrtc-view wfs --log-level DEBUG
   python -m benchmarks.perf_smoke --log-file perf.log

Prefer `PYRTC_LOG_DIR` for multiprocess runs so parent and child processes write separate files.

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

When opening a pull request:

- state the motivation clearly
- explain the user-visible behavior change
- list the validation commands you ran
- call out compatibility or deployment risks

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
- runtime import as `pyRTC`
- the core AO component model
- the documented shared-memory and configuration concepts

Areas that still need target-environment validation before operational use:

- vendor SDK integrations
- GPU-specific execution paths
- multi-process deployment details
- platform-specific driver and device behavior

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
      ruff check pyRTC tests benchmarks
      python -m build
      python -m twine check dist/*
      python -m pyRTC.scripts.validate_dist_install --dist-dir dist
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