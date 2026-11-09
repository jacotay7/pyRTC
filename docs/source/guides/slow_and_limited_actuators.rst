Slow and Limited Actuators
==========================

The loop assumes a corrector that is fast and has range to spare: a command
sent this frame is the shape seen a fixed number of frames later, and a
command cap only catches accidents. Three modules cover the cases where that
is not true. They are plain NumPy/SciPy with no streams, so they can be used
from a ``Loop`` or corrector subclass, in a simulator, or offline.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Module
     - Use it when
   * - :mod:`pyrtc.dynamics`
     - the plant responds on the scale of the loop period or slower (a
       tip-tilt stage near resonance, a woofer, a thermally actuated optic,
       an offload mechanism)
   * - :mod:`pyrtc.allocation`
     - actuators reach their limits in normal operation, the limits differ
       per actuator, or an actuator only works in one direction
   * - :mod:`pyrtc.loop_stability`
     - you need the stable gain range for the loop as it really is:
       miscalibrated, truncated, with per-mode gains or with plant dynamics

Plant dynamics
--------------

:class:`pyrtc.dynamics.LinearPlant` is a continuous-time state-space model.
Build it from matrices, or with ``first_order`` (independent lags),
``second_order`` (one resonance) or ``from_network`` (a lumped diffusive
network such as a thermal model). ``discretize(period)`` returns the exact
zero-order-hold :class:`~pyrtc.dynamics.DiscretePlant` for a loop period; a
simulated corrector calls its ``step`` once per frame.

.. code-block:: python

   from pyrtc.dynamics import LinearPlant

   plant = LinearPlant.first_order(time_constants=[120.0, 900.0])  # seconds
   plant.dc_gain()            # output per unit held command
   plant.settling_time(0.99)  # how long to wait after a poke
   discrete = plant.discretize(period=60.0)
   output = discrete.step([1.0, 0.0])

Two ways to run a loop on such a plant:

- **Wait it out.** Run one iteration per ``settling_time``. The plant then
  looks static, the usual interaction matrix and integrator apply, and the
  cost is convergence time.
- **Run faster than it settles.** The plant's lag adds phase, so the stable
  gain drops. Check it with :mod:`pyrtc.loop_stability` before closing the
  loop.

Either way, measure the interaction matrix on settled responses: wait
``settling_time`` after each poke.

Allocation inside limits
------------------------

Clipping each actuator gives the closest command in actuator space. It is
seldom the closest in the space the loop cares about, because the actuators
with range left do nothing to make up for the clipped ones.
:class:`pyrtc.allocation.BoundedAllocator` solves the bounded least-squares
problem instead: the in-limits command whose *effect* is closest to the
request. The effect matrix chooses the space, for example the pseudo-inverse
of the mode-to-command matrix to protect the controlled modes.

.. code-block:: python

   import numpy as np
   from pyrtc.allocation import BoundedAllocator, anti_windup

   modes_per_actuator = np.linalg.pinv(m2c)
   allocator = BoundedAllocator(
       modes_per_actuator,
       lower=0.0,              # one-sided: actuators that only push
       upper=stroke,
       regularization=1e-6,    # unique answer when actuators are redundant
       method="fast",          # fixed cost per frame; "exact" for offline use
       max_iterations=20,
   )
   result = allocator.allocate(requested_modes, warm_start=previous_command)
   result.command      # actuator command, inside the limits
   result.achieved     # modes actually delivered
   result.saturated    # which actuators sit on a limit

When the delivered command differs from the request, tell the integrator, or
it winds up and takes many frames to come back once the request is reachable
again:

.. code-block:: python

   state = anti_windup(state, requested_modes, result.achieved)

Stability with the real plant
-----------------------------

:func:`pyrtc.modal_gains.max_stable_gain` is the limit for one mode behind a
pure delay with a perfect control matrix. :mod:`pyrtc.loop_stability` builds
the state-transition matrix of the whole loop and takes the *true*
interaction matrix, which need not be the one the control matrix came from.

.. code-block:: python

   from pyrtc import loop_stability

   limit = loop_stability.max_stable_gain(cm, im_true, delay_frames=2)
   radius = loop_stability.spectral_radius(cm, im_true, gain=0.3, delay_frames=2)
   # with a plant that is slow compared with the loop period:
   limit = loop_stability.max_stable_gain(cm, im_true, plant=plant.discretize(period))

A few things this makes quantitative:

- A calibration that is wrong by a scale factor only scales the gain limit.
  An integrator converges as long as the gain is under the limit, so a loop
  can run on an interaction matrix that is right only to tens of percent, at
  a correspondingly lower gain.
- A sign error in one mode is unstable at every gain (the limit is 0).
- ``spectral_radius`` is the worst-case decay per frame, so
  ``-1 / log(radius)`` is the slowest convergence time in frames.

The analysis is linear. It does not cover the loop while actuators are
saturated; that is what the allocator and anti-windup are for.
