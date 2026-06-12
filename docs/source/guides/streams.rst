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
otherwise. Each stream natively carries a write counter (``stream.count``)
and the timestamp of the last completed write (``stream.write_time``); the
latency tooling in :mod:`pyrtc.latency` is built entirely on those two
fields.

GPU streams
-----------

Passing ``gpu_device="cuda:N"`` to ``create_stream`` backs the stream with a
CUDA tensor shared across processes, always paired with a CPU mirror:

- ``open_stream(name)`` (no device) reads the CPU mirror and returns NumPy
  arrays — this is what viewers and telemetry use.
- ``open_stream(name, gpu_device="cuda:N")`` attaches the producer's CUDA
  tensor and reads return ``torch.Tensor`` objects on that device.
- If CUDA, torch, or the dtype is unsupported (e.g. ``uint16``), stream
  creation falls back to a CPU stream with a warning rather than failing.

Inspecting and cleaning up
--------------------------

The ``pyshmem`` CLI works on all pyrtc streams:

.. code-block:: bash

   pyshmem list            # user-visible names of all live streams
   pyshmem unlink wfs      # destroy one stream
   pyshmem purge           # remove ALL pyshmem segments (incl. CUDA handles)

pyrtc also ships ``pyrtc-clear-shms`` for clearing the standard stream names
of a system.

Platform notes
--------------

Streams persist across process exits on Linux (POSIX shared memory), which is
what hard-RTC relies on for component restarts. **On Windows, named shared
memory is freed when the last handle closes**, so streams do not survive
their producer: treat Windows as soft-RTC-only for the 1.x line.
