.. getting started

Getting Started
===============

This guide is the shortest path from installation to a working `pyRTC` session.

Package Name
------------

Install the project from PyPI as `pyrtcao`:

.. code-block:: bash

	pip install pyrtcao

Import it in Python as `pyRTC`:

.. code-block:: python

	import pyRTC

If you are working from a source checkout instead of PyPI:

.. code-block:: bash

	git clone https://github.com/jacotay7/pyRTC.git
	cd pyRTC
	pip install .

Optional extras:

.. code-block:: bash

	pip install pyrtcao[docs]
	pip install pyrtcao[gpu]
	pip install pyrtcao[viewer]

Core Concepts
-------------

`pyRTC` is organized around adaptive optics components that exchange data through shared memory streams.
The core objects you will usually work with are:

- `WavefrontSensor`: produces images
- `SlopesProcess`: converts images into wavefront signal products
- `Loop`: reconstructs and applies control updates
- `WavefrontCorrector`: receives and outputs correction commands
- `ScienceCamera`: captures science images and derived metrics
- `Telemetry`: persists selected data products to disk

The package supports two common operating styles:

- `soft-RTC`: instantiate the objects directly in one Python process
- `hard-RTC`: run hardware-facing components in separate processes and interact through launchers/shared memory

For first use, start with `soft-RTC` or the simulation examples.

Minimal Import Check
--------------------

After installation, verify that the public package imports cleanly:

.. code-block:: bash

	python -c "import pyRTC; print(pyRTC.__all__)"

Validate a Config Before Launch
-------------------------------

Before starting a system, validate the full YAML file:

.. code-block:: bash

	pyrtc-validate-config examples/synthetic_shwfs/config.yaml

For automation or GUI-oriented tooling, JSON output is also available:

.. code-block:: bash

	pyrtc-validate-config examples/synthetic_shwfs/config.yaml --format json

Minimal Component Example
-------------------------

The base component class starts configured functions in worker threads. A minimal configuration looks like:

.. code-block:: python

	from pyRTC.pyRTCComponent import pyRTCComponent

	component = pyRTCComponent(
		 {
			  "affinity": 0,
			  "functions": [],
		 }
	)

This is useful for understanding the execution model, but practical systems usually instantiate concrete AO components such as `WavefrontSensor`, `Loop`, and `WavefrontCorrector`.

Configuration Basics
--------------------

Configuration is supplied as nested dictionaries or YAML files. A typical AO setup contains sections such as:

.. code-block:: yaml

	loop:
	  gain: 0.1
	  numDroppedModes: 0
	  functions:
		 - standardIntegrator

	wfs:
	  name: OOPAOWFS
	  width: 28
	  height: 28
	  darkCount: 1000
	  functions:
		 - expose

	slopes:
	  type: PYWFS
	  signalType: slopes
	  functions:
		 - computeSignal

	wfc:
	  name: OOPAOWFC
	  numActuators: 100
	  numModes: 80
	  functions:
		 - sendToHardware

Required keys depend on the component. For example, wavefront-corrector configs require `name`, `numActuators`, and `numModes`.

Suggested First Run
-------------------

For a practical first run, use the synthetic Shack-Hartmann example described in :doc:`../examples/synthetic_shwfs`.
That path needs no hardware and no external simulator, but it still exercises the standard `WavefrontSensor -> SlopesProcess -> Loop -> WavefrontCorrector` chain and publishes the same viewer-friendly streams you will use later with real devices.

After that, move to the OOPAO-based path in :doc:`../examples/pywfs` if you want a richer simulated optical model.
The script-driven entry point is:

.. code-block:: bash

	python examples/scao/run_soft_rtc.py --duration 10

Use the companion notebook only once you want to step through the same workflow interactively.

Viewer and Utility Commands
---------------------------

If you installed the `viewer` extra, the package exposes command-line tools for inspecting shared-memory streams:

.. code-block:: bash

	pyrtc-view wfs
	pyrtc-shm-monitor
	pyrtc-clear-shms

The performance benchmark entry point is also available after installation:

.. code-block:: bash

	pyrtc-core-bench --quick --cpu-only --output core_compute_bench_report.json

Logging
-------

The main CLI tools and example entry points use the shared `pyRTC` logger.
By default they log at `INFO` level to the console with timestamps.

Useful one-off overrides:

.. code-block:: bash

	pyrtc-view wfs --log-level DEBUG
	pyrtc-shm-monitor --log-dir logs
	pyrtc-measure-latency signal wfc --log-file latency.log

You can also set logging once in the shell for multi-process runs:

.. code-block:: bash

	export PYRTC_LOG_LEVEL=INFO
	export PYRTC_LOG_DIR=./logs
	export PYRTC_LOG_COLOR=1
	python examples/synthetic_shwfs/run_soft_rtc.py --duration 15

Supported environment variables are:

- `PYRTC_LOG_LEVEL`
- `PYRTC_LOG_DIR`
- `PYRTC_LOG_FILE`
- `PYRTC_LOG_COLOR`
- `PYRTC_LOG_CONSOLE`

When you use `hard-RTC`, child processes inherit the logging environment automatically.

Troubleshooting
---------------

- If GPU mode is configured but PyTorch is unavailable, `pyRTC` falls back to CPU mode for supported paths.
- If viewer commands fail, install the viewer extra: `pip install pyrtcao[viewer]`
- If a component fails at startup, check the YAML keys first; several components validate required config fields eagerly.
- If a multi-process run is hard to diagnose, set `PYRTC_LOG_DIR=./logs` before launching so each process writes a separate file.
- For first-time development, stay on Linux unless you have validated your target platform locally.

Next Steps
----------

- Read the component pages for the classes you plan to extend.
- Run the simulated example workflow.
- Decide early whether your deployment needs `soft-RTC` or `hard-RTC` mode.
