Shared-Memory Streams
=====================

pyrtc components exchange frames and command vectors through named
shared-memory *streams* provided by the external
`pyshmem <https://github.com/jacotay7/pyshmem>`_ package. A stream is a
named, fixed-shape, typed array slot that any process on the machine can
attach to by name. pyrtc adds a thin policy layer in :mod:`pyrtc.streams`.

Creating and attaching
----------------------

Producers (components that own an output) call
:func:`pyrtc.streams.create_stream`; consumers call
:func:`pyrtc.streams.open_stream`:

.. code-block:: python

   import numpy as np
   from pyrtc.streams import create_stream, open_stream

   # producer side (e.g. inside a component)
   shm = create_stream("wfs", (49, 49), np.int32)
   shm.write(frame)

   # consumer side (any process)
   stream = open_stream("wfs")
   frame = stream.read()                  # consistent snapshot, returns a copy
   frame = stream.read_new(timeout=1.0)   # block until the next write
   frame = stream.read(out=buffer)        # zero-alloc read into a buffer

``create_stream`` reuses an existing stream when its shape, dtype and notify
setting already match (so viewers stay attached across component restarts)
and rebuilds it otherwise. Observers that must never write (viewers, telemetry, latency
probes) pass ``readonly=True`` to ``open_stream``.

Wake-up notifications
---------------------

Streams made by ``create_stream`` use pyshmem's ``notify=True``: a write
wakes consumers parked in ``read_stream`` (``read_after_publication``) through
a Linux futex, instead of each consumer sleeping and re-checking every few
tens of microseconds. Idle worker threads then stop competing for the GIL and
the CPU, which lowers end-to-end latency and jitter in the running pipeline.
Each write pays one extra wake system call (a few microseconds).

Set ``PYRTC_STREAM_NOTIFY=0`` in the environment to create polling streams
instead (hard-RTC component processes inherit the variable), or pass
``notify=False`` to ``create_stream``. On a quiet host with deep CPU idle
states a parked consumer can wake a little later than a polling one at low
frame rates, so measure your own system with
``python -m benchmarks.stream_handoff_bench`` and
``python -m benchmarks.pipeline_latency_bench``. The flag is fixed when a
stream is created, so an existing stream with the other setting (for example
one left over from an older pyrtc) is rebuilt rather than reused. Off Linux
the flag is recorded but consumers keep polling.

Publication metadata
--------------------

Every completed write is a *publication* carrying a write counter
(``stream.count``), its wall-clock time (``stream.write_time``) and a user
``frame_id``. ``stream.read_publication()`` returns the payload together with
the metadata of that same write, which is how telemetry records timestamps.

Inside components, :meth:`pyrtc.component.Component.read_stream` and
:meth:`~pyrtc.component.Component.write_stream` handle this for you:

- Both helpers work only on streams the component registered with
  ``register_input_stream`` / ``register_output_stream``; any other name
  raises ``KeyError``.
- ``read_stream(name)`` *consumes* the stream: it returns the first write newer
  than the one it returned last time (the first call returns immediately).
  ``read_stream(name, block=False)`` only peeks and does not consume. A
  blocking read raises ``ComponentClosedError`` if the component is closed
  while it waits.
- Reading a registered input records its ``frame_id``, and ``write_stream``
  stamps it on the outputs. The wavefront sensor numbers each exposure, so a
  ``wfc`` command carries the id of the WFS frame it was computed from.
  :mod:`pyrtc.latency` uses these ids to pair writes across streams exactly.
  It first waits until a frame from the source has reached every stream on
  the path, so the sample windows overlap even right after ``start()``. It
  falls back to aligning write counts, reported as ``alignment: count`` with a
  warning, only for producers that do not stamp frame ids or windows that
  share none.

GPU streams
-----------

Passing ``gpu_device="cuda:N"`` to ``create_stream`` backs the stream with a
CUDA tensor shared across processes, always paired with a CPU mirror:

- ``open_stream(name)`` (no device) reads the CPU mirror and returns NumPy
  arrays — this is what viewers and telemetry use.
- ``open_stream(name, gpu_device="cuda:N")`` attaches the producer's CUDA
  tensor and reads return ``torch.Tensor`` objects on that device.
- If CUDA or torch is unavailable, or the dtype is not in
  ``pyshmem.GPU_SUPPORTED_DTYPES``, stream creation falls back to a CPU stream
  with a warning rather than failing.

Inspecting and cleaning up
--------------------------

The ``pyshmem`` CLI works on all pyrtc streams:

.. code-block:: bash

   pyshmem list            # user-visible names of all live streams
   pyshmem unlink wfs      # destroy one stream
   pyshmem purge           # remove ALL pyshmem streams on this machine

From Python, :func:`pyrtc.streams.clear_shms` destroys a list of streams and
ignores names that do not exist.

Platform notes
--------------

Streams persist across process exits on Linux (POSIX shared memory), which is
what hard-RTC relies on for component restarts. **On Windows, named shared
memory is freed when the last handle closes**, so streams do not survive
their producer: treat Windows as soft-RTC-only for the 1.x line.

ImageStreamIO (milk / CACAO) bridge
-----------------------------------

:mod:`pyrtc.isio_bridge` mirrors a stream between pyrtc and ImageStreamIO
(ISIO), the shared-memory format of milk and CACAO. That lets pyrtc use ISIO
camera and DM drivers and milk viewers, or feed a CACAO RTC. Each
``IsioBridge`` component copies one stream in one direction:

.. code-block:: yaml

  isio_wfs:                       # pyrtc -> ISIO
    class_name: pyrtc.isio_bridge.IsioBridge
    direction: to_isio
    isio_name: pyrtc_wfs
    input_streams: {input: wfs}
    functions: [mirror]
  isio_dm:                        # ISIO -> pyrtc
    class_name: pyrtc.isio_bridge.IsioBridge
    direction: from_isio
    isio_name: dm00disp
    output_streams: {output: dm_from_cacao}
    functions: [mirror]

For one-off use there is also a CLI:
``pyrtc-isio-bridge to-isio wfs pyrtc_wfs`` or
``pyrtc-isio-bridge from-isio dm00disp dm_from_cacao``. It needs ImageStreamIO's
Python module: ``pip install git+https://github.com/milk-org/ImageStreamIO``.

- Axes are reversed, so images keep their orientation: pyrtc arrays are
  row-major ``[y, x]``, ISIO images column-major with ``size = [x, y]``. A
  pyrtc ``(height, width)`` stream is an ISIO image with
  ``size = [width, height]`` (the same bytes), and an ISIO ``[nx, ny]`` image
  a pyrtc ``(ny, nx)`` stream. (pyrtc 1.x kept the shape, which matched milk
  while its streams were ``(width, height)``.)
- ISIO-to-pyrtc frames carry the ISIO ``cnt0`` as their ``frame_id``.
- The bridge waits by polling ISIO's semaphore (about 0.1 ms latency),
  because the ISIO module's blocking waits hold the GIL (#138).
- An ISIO stream the bridge creates stays after it closes, unless
  ``remove_on_close`` is set. A different existing stream of the same name is
  refused.

