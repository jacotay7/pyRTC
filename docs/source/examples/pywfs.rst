.. PYWFS Examples

PYWFS Examples
==============

This page describes the simulated single-conjugate adaptive optics example based on a pyramid wavefront sensor.

Purpose
-------

The PYWFS example is the richer simulator-backed path for users who already want more optical realism than the synthetic SHWFS quick start described in :doc:`synthetic_shwfs`.

The example uses the OOPAO simulator to stand in for AO hardware and demonstrates the expected configuration shape for:

- a wavefront sensor
- a slopes processor
- a loop controller
- a wavefront corrector
- a science camera

Files
-----

The main example assets live under `examples/scao/`:

- `pywfs_oopao_soft_rtc_example.py`: notebook-style soft-RTC walkthrough with logging and status output
- `pywfs_example_OOPAO.ipynb`: notebook walkthrough of the same setup
- `pywfs_OOPAO_config.yaml`: example configuration

What the Config Shows
---------------------

The example configuration defines the standard sections used by a basic AO chain:

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
		 flatNorm: True
		 functions:
			 - computeSignal

	 wfc:
		 name: OOPAOWFC
		 numActuators: 100
		 numModes: 80
		 functions:
			 - sendToHardware

This is the configuration pattern to copy when building a simulator-backed system after the synthetic quick start is already familiar.

Running the Example
-------------------

The recommended first path is the script version because it keeps the setup reproducible and prints status updates while the loop is running.

.. code-block:: bash

	python examples/scao/pywfs_oopao_soft_rtc_example.py --duration 10

By default the script:

- clears the standard pyRTC streams
- builds the OOPAO wavefront sensor, deformable mirror, and science camera wrappers
- computes a quick interaction matrix with the atmosphere removed
- closes the loop for the requested duration

Useful variants:

.. code-block:: bash

	python examples/scao/pywfs_oopao_soft_rtc_example.py --skip-im --duration 5
	python examples/scao/pywfs_oopao_soft_rtc_example.py --no-kl-basis --duration 5

If you prefer interactive exploration, open `examples/scao/pywfs_example_OOPAO.ipynb` after the script workflow is familiar. The notebook walks through the same stages cell by cell.

This OOPAO path is intentionally soft-RTC only. The wavefront sensor, deformable mirror, and science camera adapters share one in-process optical simulation state, so it is not a good fit for the hard-RTC child-process launch model.

Recommended Validation Steps
----------------------------

Once the example is running, verify these behaviors:

- the wavefront sensor stream is updating
- the slopes product is non-empty and has the expected shape
- the loop can compute and write a correction vector
- the viewer tools can display `wfs`, `signal2D`, and `wfc2D`

Viewer commands:

.. code-block:: bash

	 pyrtc-view wfs signal2D wfc2D psfShort psfLong --geometry 2x3
	 pyrtc-view signal2D -1 1
	 pyrtc-view wfc2D -0.5 0.5

Notes and Limitations
---------------------

- This example depends on OOPAO and is not the zero-dependency first run.
- It is best suited to Linux-based development environments.
- The script path is better for repeatable setup; the notebook path is better for step-by-step debugging and inspection.
- Treat it as the reference simulation path, not as a drop-in hardware deployment recipe.

Next Steps
----------

After the simulated example works reliably, the next step is usually to replace one abstract component at a time with your hardware-specific implementation under `pyRTC.hardware`.
