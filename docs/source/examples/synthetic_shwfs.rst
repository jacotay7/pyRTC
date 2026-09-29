Synthetic SHWFS Example
=======================

This is the intended first end-to-end `pyrtc` example for a new user.

It uses synthetic hardware classes that subclass the normal `WavefrontSensor` and `ScienceCamera` base classes, publishes the standard shared-memory streams, and runs the real `SlopesProcess`, `Loop`, and `WavefrontCorrector` components.

What This Example Covers
------------------------

- how to subclass a wavefront sensor without touching the rest of the AO pipeline
- how to subclass a science camera for a simple synthetic PSF path
- what a minimal config layout looks like for a runnable system
- how to start a soft-RTC chain that still works with the standard viewer tools
- what update rates and residual metrics to expect from a small CPU-only quick-start setup

Files
-----

The example assets live under `examples/synthetic_shwfs/`:

- `config.yaml`: single runnable config used by both soft and hard modes
- `synthetic_shwfs_soft_rtc_example.py`: manager-driven soft-RTC tutorial script with direct object access
- `synthetic_shwfs_hard_rtc_example.py`: manager-driven hard-RTC tutorial script with proxy-style access

The synthetic hardware implementations live in `pyrtc/hardware/synthetic_systems.py`.

Why This Example Exists
-----------------------

The OOPAO notebook remains useful, but it is not the best first touchpoint for release-quality onboarding because it adds an extra simulator dependency and hides some of the component boundaries that matter when you are integrating real hardware.

This synthetic SHWFS path keeps the mental model simple:

1. `SyntheticSHWFS.expose()` generates a lenslet image from a deterministic disturbance and the current `wfc` correction.
2. `SlopesProcess.compute_signal()` turns that image into SHWFS slopes.
3. `Loop.standard_integrator()` reads `signal`, computes a correction, and writes `wfc`.
4. `WavefrontCorrector.send_to_hardware()` updates `wfc_2d` for display.
5. `SyntheticScienceCamera.expose()` reads residual slopes and generates a synthetic PSF plus Strehl estimate.

Running It
----------

From the repository root:

.. code-block:: bash

	python examples/synthetic_shwfs/synthetic_shwfs_soft_rtc_example.py --duration 15
	python examples/synthetic_shwfs/synthetic_shwfs_hard_rtc_example.py --duration 15

Both launchers read the same YAML file, generate the tiny interaction matrix referenced by that config, clear the standard pyrtc streams by default, and start the full chain through `RTCManager`.

The mode switch is intentionally obvious in the code:

.. code-block:: python

	soft_manager = RTCManager.from_config_file(CONFIG_PATH, mode="soft")
	hard_manager = RTCManager.from_config_file(CONFIG_PATH, mode="hard")

The soft example then demonstrates direct object syntax:

.. code-block:: python

	loop = soft_manager.get_component("loop")
	loop.gain = 0.10

The hard example demonstrates remote proxy syntax:

.. code-block:: python

	loop = hard_manager.get_component("loop")
	current_gain = loop.get_property("gain")
	loop.set_property("gain", 0.10)
	wfc = hard_manager.get_component("wfc")
	wfc.run("flatten")

Both scripts print one status line per second. A typical line looks like:

.. code-block:: text

	t=  5.0s wfs= 199.6 Hz psf=  49.8 Hz residual_rms=0.0312 correction_rms=0.1098 strehl=0.914

The exact numbers will vary by host, but the important pattern is:

- the synthetic WFS should stay close to its configured frame rate
- the residual RMS should settle below the open-loop disturbance amplitude
- the synthetic Strehl should rise when the loop is behaving sensibly

Telemetry Capture
-----------------

The intended telemetry workflow is intentionally small and NumPy-native:

.. code-block:: python

	from pyrtc import Telemetry

	telem = Telemetry()
	telem.save("wfs", 1000)
	telem.save(["wfs", "wfc"], 1000)
	data = telem.read_last_save()
	print(data["wfs"]["frames"].shape)
	print(data["wfs"]["timestamps"].shape)

Each save creates one session directory under ``data_dir``. For each captured
stream pyrtc writes:

- ``frames.npy`` for the frame stack
- ``timestamps.npy`` for producer timestamps
- ``metadata.json`` for stream metadata

That means users can reopen the files with plain NumPy if they want, while the
``Telemetry`` helper keeps the most common access pattern straightforward.

Viewer Commands
---------------

The preferred way to inspect the demo is a single composite viewer window:

.. code-block:: bash

	pyrtc-view wfs signal_2d wfc_2d psf_short psf_long --geometry 2x3

If you want a smaller science-camera-only view, open:

.. code-block:: bash

	pyrtc-view psf_short psf_long --geometry row

The composite viewer auto-sizes to the stream dimensions, so it is usually a better default than opening many separate windows.

Config Layout
-------------

The synthetic config deliberately uses the same top-level sections you will keep for a real system:

.. code-block:: yaml

	wfs:
	  name: SyntheticSHWFS
	  width: 32
	  height: 32
	  frame_rate_hz: 200
	  sub_ap_spacing: 8
	  functions:
	    - expose

	slopes:
	  type: SHWFS
	  signal_type: slopes
	  sub_ap_spacing: 8
	  sub_ap_offset_x: 0
	  sub_ap_offset_y: 0
	  functions:
	    - compute_signal

	wfc:
	  name: SyntheticWFC
	  num_actuators: 32
	  num_modes: 32
	  functions:
	    - send_to_hardware

	loop:
	  gain: 0.35
	  im_file: synthetic_identity_im.npy
	  functions:
	    - standard_integrator

	psf:
	  name: SyntheticScienceCamera
	  width: 64
	  height: 64
	  integration: 10
	  functions:
	    - expose
	    - integrate

	manager:
	  mode: soft-rtc
	  component_files:
	    wfs: ../../pyrtc/hardware/synthetic_shwfs.py
	    wfc: ../../pyrtc/hardware/synthetic_wfc.py
	    psf: ../../pyrtc/hardware/synthetic_science_camera.py

The transition from synthetic hardware to real hardware should mostly leave `slopes`, `loop`, and high-level wiring alone. In a typical integration, the first file you replace is the `wfs` section and the subclass behind it.

Subclassing Pattern
-------------------

The synthetic classes are intentionally short and can be used as templates.

For a wavefront sensor, the important rule is: populate `self.data` and then call the parent `expose()` so the normal dark subtraction and shared-memory publication still happen.

.. code-block:: python

	from pyrtc.wavefront_sensor import WavefrontSensor


	class MyWavefrontSensor(WavefrontSensor):
	    def __init__(self, conf):
	        super().__init__(conf)
	        self.serial_number = conf["serialNumber"]

	    def expose(self):
	        self.data = self.read_frame_from_camera_driver()
	        super().expose()

For a science camera, the pattern is the same: generate or acquire a frame in `self.data`, then call the parent implementation.

.. code-block:: python

	from pyrtc.science_camera import ScienceCamera


	class MyScienceCamera(ScienceCamera):
	    def __init__(self, conf):
	        super().__init__(conf)

	    def expose(self):
	        self.data = self.read_frame_from_camera_driver()
	        super().expose()

In both cases, that keeps the base-class shared-memory contract intact, which is what lets the rest of the pipeline remain unchanged.

Speed Expectations
------------------

This quick-start example is deliberately small. On a normal development workstation, a `32x32` synthetic SHWFS at `200 Hz` and a `64x64` synthetic science camera at `50 Hz` should be comfortably attainable in soft-RTC mode.

Use this example for:

- checking that your install works
- understanding the component boundaries
- validating viewer behavior
- experimenting with loop gain and SHWFS geometry

Do not use it to infer final hardware performance. For kernel-level performance expectations, use the benchmark tables in the project README and the benchmarking tools under `benchmarks/`.

Next Steps
----------

Once this example is working, the next sensible step is one of:

- replace `SyntheticSHWFS` with your real camera subclass while keeping the same `slopes`, `loop`, and `wfc` sections
- replace `SyntheticScienceCamera` with your real PSF camera subclass if you need science-image monitoring
- move to the OOPAO example when you want a richer simulated optical path