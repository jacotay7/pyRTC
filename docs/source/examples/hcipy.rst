.. HCIPy Simulator Example

HCIPy Simulator Example
=======================

`HCIPy <https://hcipy.org>`_ is a pip-installable optics simulator, which makes
it the quickest way to run pyrtc against a real optical model:

.. code-block:: bash

	pip install pyrtcao[hcipy]     # from a source checkout: pip install .[hcipy]
	python examples/hcipy/hcipy_shwfs_soft_rtc_example.py --duration 10 --atmosphere

Files
-----

The example lives under ``examples/hcipy/``:

- ``hcipy_shwfs_soft_rtc_example.py``: soft-RTC walkthrough. It calibrates on
  the unaberrated system (DM round-trip check, reference slopes, IM), then
  closes the loop.
- ``hcipy_shwfs_config.yaml``: the pyrtc config. An ``hcipy`` provider section
  owns the simulation, and ``wfs``, ``wfc`` and ``psf`` declare
  ``resource: hcipy``.
- ``hcipy_shwfs_params.yaml``: the simulated system. It has a 2 m telescope, a
  10x10 Shack-Hartmann (12 px per sub-aperture), an 11x11 DM with 89 actuators
  in the pupil, a frozen-flow layer with r0 = 0.15 m, and an H-band science
  camera.

The system
----------

:mod:`pyrtc.hardware.hcipy_interface` builds the whole system from the flat
parameter file. The module docstring lists every parameter and its default.

- The WFS is ``shwfs`` or a modulated ``pywfs``.
- The DM has Gaussian influence functions; the actuators inside the pupil are
  used.
- The atmosphere advances one frame (``1 / frame_rate``) per WFS exposure, and
  only while it is enabled.

The components take their detector sizes and actuator count from the
simulation. The corrector gives aobasis the real actuator positions, so a
``basis:`` section (KL here) matches the simulated DM.

Two design rules keep the loop well behaved:

- Keep the sub-aperture size near r0 at the WFS wavelength. With 0.8 m
  sub-apertures on an 8 m telescope, the spots are speckled and the loop
  cannot close on the atmosphere.
- Control only the modes the WFS senses well. With 50 KL modes on the 10x10
  SHWFS, poorly sensed high-order modes slowly ran away on the atmosphere; 30
  modes are stable.

The example script sets ``OPENBLAS_NUM_THREADS`` and the related variables to
``1`` before importing numpy, unless they are already set, just as hard-RTC
children do. HCIPy's propagation otherwise keeps a full OpenBLAS thread pool
busy (numpy and scipy each load one). On 16 cores those pools used about 15 of
them, and the WFS ran slower (23 against 32 frames/s). If you build the system
from your own script, set the variables before Python starts, or cap the pools
with ``threadpoolctl.threadpool_limits(1)``.

``tests/system/test_hcipy_convergence.py`` runs this example. It nulls a
static DM aberration and checks that closing the loop raises the Strehl on
the atmosphere. On the reference machine, the H-band Strehl was 0.2-0.6 in
open loop (it varies with the turbulence) and 0.85-0.89 closed.
