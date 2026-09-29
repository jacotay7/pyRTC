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

Modal Gains
-----------

Each controlled mode ``i`` runs with the gain
``gain * modal_gains[i] / optical_gains[i]`` (``loop.effective_gains``). The
per-mode factors are folded into the control matrix, so they cost nothing per
frame, and every integrator uses them. ``modal_gains`` and ``optical_gains``
default to 1. Set them in the config (a list with ``num_modes`` values or a
``.npy`` file) or at run time with ``set_modal_gains`` and
``set_optical_gains``. ``optical_gains`` compensates a WFS's reduced
sensitivity on a residual wavefront, such as a pyramid's optical gains taken
from simulation or calibration.

``loop.optimize_modal_gains`` picks the gains from closed-loop telemetry
(Gendron & Léna 1994). It reconstructs each mode's pseudo open-loop PSD from
the residuals measured at the current gains, and estimates the white noise
from the PSD's high-frequency end. It then chooses the stable gain that
minimizes the predicted residual variance for the loop delay:

.. code-block:: python

  signal_frames = ...  # (num_frames, signal_size) closed-loop `signal` frames
  residuals = loop.modal_residuals(signal_frames)
  result = loop.optimize_modal_gains(residuals, frame_rate=1000.0, delay_frames=2)
  print(result.gains, result.max_gain)

``delay_frames`` is the loop delay in frames. The largest stable integrator
gain is 1.0 for a delay of 1 and 0.618 for 2. The functions in
:mod:`pyrtc.modal_gains` work on any residual array.

Predictive Control
------------------

The ``predictive_integrator`` worker function controls each mode from a
forecast instead of the current residual. Every frame it:

1. estimates the pseudo open-loop (POL) disturbance, ``cm @ signal`` minus the
   command that was on the corrector during the exposure (the one sent
   ``delay_frames`` iterations earlier);
2. asks a predictor for that disturbance ``horizon`` frames ahead
   (``horizon`` defaults to ``delay_frames``);
3. moves the command towards cancelling the forecast,
   ``c = (1 - g) c - g * prediction``.

.. code-block:: yaml

  loop:
    functions: [predictive_integrator]
    gain: 0.3                # used until the predictor is fitted
    predictor:
      type: ar_kalman        # or least_squares, persistence, or a registered name
      delay_frames: 2
      gain: 1.0              # blend towards the prediction once fitted
      fit_frames: 4096       # POL frames kept for fitting

Before a fit, the predictor is ``persistence`` and the loop blends with the
loop ``gain``, which makes it a delay-aware POL integrator. It records POL
data meanwhile. Once it has run long enough, fit the predictor:

.. code-block:: python

  loop.fit_predictor()   # fits on the recorded POL and switches to it

Built-in predictors (:mod:`pyrtc.predictive`):

- ``ar_kalman``: modal LQG. It fits an AR(2) model per mode (least squares,
  corrected for the measurement noise estimated from the spectrum's floor)
  and predicts with a steady-state Kalman filter. It suits vibrations and
  other resonant disturbances.
- ``least_squares``: a per-mode linear prediction filter over the last
  ``order`` POL samples (default 8), fitted by ridge regression.

In closed-loop simulations with a 2-frame delay, both cut the residual of a
lightly damped vibration mode more than 15-fold compared with the best
integrator gain, and match it on slow turbulence. To add a method, subclass
``ModalPredictor`` and register it with ``@register_predictor("name")``; the
loop then accepts ``predictor.type: name``.

Reduced-Precision Control Matrix (GPU)
--------------------------------------

On a GPU, the control-matrix multiply is memory-bound. ``ReducedPrecisionMatrix``
stores the matrix in ``float16`` or ``bfloat16`` with an fp32 scale per row, and
multiplies with fp32 accumulation. ``leak_integrator_gpu`` accepts it in place
of a tensor:

.. code-block:: python

  from pyrtc.loop import ReducedPrecisionMatrix, leak_integrator_gpu

  cm16 = ReducedPrecisionMatrix(loop.g_cm, "float16", device="cuda")
  correction = leak_integrator_gpu(slopes, cm16, correction, leak, loop.num_active_modes)

The row scale keeps large control-matrix entries from overflowing fp16, and
the input is scaled to unit peak before the cast. Measured on a regularized
control matrix with slopes from DM modes plus 10% noise, the error in the
modal update is:

- ``float16``: median 8e-4, p99 1.2e-3 relative;
- ``bfloat16``: median 6e-3, p99 1e-2.

Both are well below typical WFS noise. The gain comes from moving fewer bytes,
so it only pays off for large systems. On a Quadro P620, a 64x64 system
(8192 slopes x 4096 modes) went from 2.09 ms to 1.29 ms with fp16, while a
32x32 system was slightly slower because of the extra scaling steps.
``python -m benchmarks.core_compute_bench`` reports
``loop.leak_integrator_gpu_float16`` and ``_bfloat16`` next to fp32.

