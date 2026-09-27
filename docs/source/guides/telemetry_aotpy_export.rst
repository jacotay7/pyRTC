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

Continuous Recording (Ring Buffer)
----------------------------------

``save()`` captures the *next* N frames. To capture what *just* happened (a
loop divergence, a saturation event), keep recording into a bounded
in-memory ring buffer and dump it when the event occurs:

.. code-block:: python

	telemetry = Telemetry({"data_dir": "./data", "functions": []})
	telemetry.start_ring_buffer(["wfs", "signal", "wfc"], seconds=10)
	...
	session_path = telemetry.dump_ring_buffer("divergence", semantic_tags={...})
	telemetry.stop_ring_buffer()

A dump writes an ordinary telemetry session (same ``session.json``,
``frames.npy``, ``timestamps.npy`` and ``frame_ids.npy`` layout as
``save()``), so ``load_telemetry_session`` and the AOTPy exporter above work
on it unchanged. Behaviour:

- One reader thread per stream records every publication its read-only
  handle sees: payload, ``write_time``, ``frame_id`` and publication
  ``count`` (saved as ``counts.npy``; ``save()`` now records it too).
  Readers never block the RTC producers.
- Memory is bounded and allocated once at start: ``frames`` publications per
  stream (``frames * frame_nbytes``). With ``seconds`` alone, each stream's
  capacity is estimated by measuring its publication rate for
  ``probe_seconds`` (default 1 s) and adding 50% headroom; a stream that is
  idle during the probe raises and must be given ``frames=``. If a stream
  later publishes faster than measured, the ring holds less than ``seconds``.
  Pass ``frames`` (optionally with ``seconds``) for a fixed footprint.
- With ``seconds``, a dump keeps only publications written in the last
  ``seconds`` before the dump; ``dump_ring_buffer(seconds=...)`` overrides
  the window per dump.
- A dump locks every stream's ring together while copying it, so it is one
  consistent cut across streams with no torn frames, ordered oldest first.
  Recording continues afterwards; publications skipped during the copy are
  counted as missed.
- Publications a reader could not keep up with are detected from gaps in
  ``count``. Each stream's ``missed_frames`` metadata counts the gaps inside
  the dumped window, and ``metadata["ring_buffer"]`` records the capacity,
  window, totals recorded and missed since start, and any reader error.
- ``ring_buffer_status()`` reports capacity, fill, recorded and missed per
  stream. ``stop_ring_buffer()`` joins the readers, closes their handles and
  frees the memory; dump first if you need the contents.

The ring buffer can also start with the component, from the ``telemetry``
config section. ``start()`` starts it (unless ``autostart: false``) and
``stop()`` stops it:

.. code-block:: yaml

	telemetry:
	  data_dir: ./data
	  streams: [wfs, signal, wfc]
	  ring_buffer:
	    seconds: 10        # time window kept in dumps
	    frames: 20000      # capacity per stream; omit to estimate from rate
	    # streams: [wfc]   # defaults to telemetry.streams
	    # probe_seconds: 1.0
	    # autostart: true

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