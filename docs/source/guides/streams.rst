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

``create_stream`` reuses an existing stream when its shape and dtype already
match (so viewers stay attached across component restarts) and rebuilds it
otherwise. Observers that must never write (viewers, telemetry, latency
probes) pass ``readonly=True`` to ``open_stream``.

Publication metadata
--------------------

Every completed write is a *publication* carrying a write counter
(``stream.count``), its wall-clock time (``stream.write_time``) and a user
``frame_id``. ``stream.read_publication()`` returns the payload together with
the metadata of that same write, which is how telemetry records timestamps.

Inside components, :meth:`pyrtc.component.Component.read_stream` and
:meth:`~pyrtc.component.Component.write_stream` handle this for you:

- ``read_stream(name)`` *consumes* the stream: it returns the first write newer
  than the one it returned last time (the first call returns immediately).
  ``read_stream(name, block=False)`` only peeks and does not consume.
- Reading a registered input records its ``frame_id``, and ``write_stream``
  stamps it on the outputs. The wavefront sensor numbers each exposure, so a
  ``wfc`` command carries the id of the WFS frame it was computed from.
  :mod:`pyrtc.latency` uses these ids to pair writes across streams exactly,
  and falls back to aligning write counts for producers that do not stamp
  frame ids.

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
