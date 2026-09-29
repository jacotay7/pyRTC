.. loop:

.. currentmodule:: pyrtc.loop


Loop
====

The `Loop` component is the control stage of the AO pipeline.
It reads the processed wavefront signal, applies the configured control law, and writes the resulting correction vector to the wavefront-corrector stream.

In practical terms, `Loop` is where you:

- load or compute interaction and control matrices
- select an integration strategy
- tune gain or leak parameters
- apply dropped-mode and delay behavior
- manage correction updates sent to the wavefront corrector

Soft-RTC Example
----------------

The following example shows the general pattern for starting a loop in `soft-RTC` mode.
In this mode the control object lives in the same Python process as the rest of the AO chain.

.. code-block:: python

  import numpy as np
  from pyrtc.loop import Loop
  from pyrtc.utils import read_yaml_file

  conf = read_yaml_file("path/to/config.yaml")
  loop = Loop(conf["loop"])

  # A calibrated system usually loads or computes IM first, then derives CM.
  loop.IM = np.eye(loop.signal_size, loop.num_modes, dtype=np.float32)
  loop.compute_cm()
  loop.set_gain(0.1)
  loop.start()

Hard-RTC Example
----------------

The hard-RTC path is appropriate when the loop needs to interact with hardware-facing processes through shared memory while keeping process boundaries explicit.

.. code-block:: python
  
  from pyrtc import HardwareLauncher

  config = 'path/to/config.yaml'
  port = 3004

  loop = HardwareLauncher('path/to/pyrtc/loop.py', config, port)
  loop.launch()

  # Once launched, controller methods and properties can be accessed remotely.
  loop.run("compute_cm")
  loop.set_property("gain", 0.1)
  print(loop.get_property("gain"))

Control Notes
-------------

The loop class supports several control-related concepts that matter operationally:

- `gain` and `leaky_gain` for integrator behavior
- `num_dropped_modes` for excluding poorly behaved modes
- interaction-matrix and control-matrix workflows
- optional GPU-assisted paths when PyTorch is available
- delay and limit settings for controller tuning

In production use, the loop is usually one of the last components you tune after stream shapes, calibration files, and hardware-facing behavior are already stable.


Parameters
----------

.. autoclass:: Loop
  :members:
  :inherited-members:
  :undoc-members:
  :show-inheritance:
  :no-index:

Interaction-Matrix Calibration
------------------------------

``compute_im()`` calibrates with ``im_method``:

- ``push-pull`` (default): each mode is poked to ``+poke_amp`` and
  ``-poke_amp`` in turn.
- ``hadamard``: all modes are poked at once with ``+/-poke_amp`` Hadamard
  patterns and the IM is demultiplexed. For the same number of frames, white
  sensor noise in the IM drops by about ``sqrt(num_modes)``.
- ``docrime``: random-perturbation calibration.

After each poke, ``im_settle_frames`` frames (default 1) are discarded before
``num_iters_im`` frames are averaged, so frames exposed while the corrector was
still moving are not used. ``hardware_delay`` adds a fixed wait on top.

Before calibrating, ``compute_im()`` runs ``check_round_trip()`` (disable with
``im_round_trip_check: false``). It flattens the corrector, waits for stable
signal frames, pokes every mode, waits for the signal to move and settle, then
flattens and waits for it to return. Right after start-up the worker kernels
JIT-compile, and the first DM command can take about a second to reach the
signal; calibrating in that window produced zero or smeared IM columns. If the
round trip never completes within ``im_timeout`` seconds (default 30), a
``TimeoutError`` explains why, e.g. a ``poke_amp`` too small to measure. Call
``check_round_trip()`` yourself before other steps that need the live
pipeline, such as taking reference slopes.

Safety Watchdog
---------------

While the loop is closed, each integrator waits at most ``watchdog_timeout``
seconds (default 1.0) for a new ``signal`` frame. After that it declares the
input stale, skips the iteration, logs a warning saying whether the signal's
producer is still alive (stalled) or has exited, and applies
``watchdog_action``:

- ``hold`` (default): keep the last correction on the corrector and keep
  waiting. The loop resumes by itself with the next frame.
- ``open``: stop the loop. Start it again once the input is back.
- ``flatten``: stop the loop and flatten the corrector.

.. code-block:: yaml

  loop:
    watchdog_timeout: 0.5   # seconds; null or 0 disables the watchdog
    watchdog_action: open

``loop.safety_status()`` reports ``input_stale``, ``producer_alive``, the
number of stale episodes, and human-readable ``alerts``. The manager includes
it as ``safety`` in each component's status (soft and hard RTC), and the
manager GUI shows alerts on the component's graph node.

