.. telemetry_aotpy_export

Telemetry To AOTPy Export
=========================

`pyrtc` telemetry sessions are now self-describing enough to support an offline
export path into `aotpy`.
The exporter is intentionally outside the real-time loop and treats the session
directory as the source of truth.

Install
-------

The export path is optional and does not affect the base `pyrtc` install:

.. code-block:: bash

	pip install pyrtcao[aotpy]

Capture Then Export
-------------------

One straightforward workflow is:

.. code-block:: python

	from pyrtc.telemetry import Telemetry
	from pyrtc.exporters.aotpy_export import export_telemetry_session_to_aotpy

	telemetry = Telemetry({"data_dir": "./data", "functions": []})
	session_path = telemetry.save(
		["wfs", "signal", "wfc", "psf_short"],
		200,
		semanticTags={
			"wfs": ["wfs"],
			"signal": ["signal", "slopes"],
			"wfc": ["wfc", "control"],
			"psf_short": ["psf", "science"],
		},
	)
	export_telemetry_session_to_aotpy(session_path, "synthetic_session.fits")

The equivalent CLI is:

.. code-block:: bash

	pyrtc-export-aotpy data/session_20260309_120000_abcd1234 synthetic_session.fits

If the output path is omitted, the CLI writes a sibling FITS file named after
the session directory.

Current Mapping
---------------

The exporter maps session data conservatively.
It prioritizes clean provenance over guessing hidden AO structure.

- `wfs`: exported as WFS detector pixel intensities when present
- `signal`: exported as WFS measurements when the shape is interpretable
- `wfc`: exported as the loop command history and associated deformable-mirror command stream
- `psf_short` and `psf_long`: exported as scoring-camera detector sequences
- session metadata, host metadata, config path, and unmapped stream names: preserved as AO-system metadata

Assumptions And Limitations
---------------------------

The current version is meant to make synthetic and early integration sessions
portable, not to claim complete AOT coverage for every `pyrtc` deployment.

- Export only includes streams that were actually captured in the telemetry session.
- `signal` is interpreted as Shack-Hartmann slopes when the config says `SHWFS` and the flattened signal length is even.
- `wfc` is treated as the command vector sent through the control path, which in many `pyrtc` systems is modal rather than zonal.
- Stream metadata that does not map directly into `aotpy` fields is preserved as metadata strings on the exported `AOSystem` or `Image` objects.
- Uncaptured calibration products such as interaction matrices, darks, flats, or explicit telescope geometry are not invented during export.

Python API
----------

Use the conversion helper when you want an in-memory `aotpy.AOSystem` without
writing a file immediately:

.. code-block:: python

	from pyrtc.exporters.aotpy_export import telemetry_session_to_aotpy

	system = telemetry_session_to_aotpy("data/session_20260309_120000_abcd1234")
	print(system)

Use `export_telemetry_session_to_aotpy(...)` when you want the file on disk in
one step.