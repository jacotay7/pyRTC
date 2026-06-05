.. SHWFS Simulator Examples

SHWFS Simulator Examples
========================

This page describes the simulator-backed Shack-Hartmann examples that mirror the Pyramid-WFS walkthroughs under :doc:`pywfs`.

Purpose
-------

These examples are the next step after :doc:`synthetic_shwfs` when you want a real optical simulator in the loop while keeping the standard pyRTC control chain.

Files
-----

The example assets live under `examples/shwfs/`:

- `shwfs_oopao_soft_rtc_example.py`: OOPAO-backed Shack-Hartmann soft-RTC walkthrough
- `shwfs_OOPAO_config.yaml`: pyRTC config for the OOPAO Shack-Hartmann example
- `shwfs_OOPAO_params.yaml`: flat OOPAO parameter dictionary for the OOPAO Shack-Hartmann example
- `shwfs_specula_soft_rtc_example.py`: SPECULA-backed Shack-Hartmann soft-RTC walkthrough
- `shwfs_SPECULA_config.yaml`: pyRTC config for the SPECULA Shack-Hartmann example
- `shwfs_SPECULA_params.yaml`: SPECULA object-graph parameters for the SPECULA Shack-Hartmann example

What the Config Shows
---------------------

The SHWFS examples switch the slopes section from `PYWFS` to `SHWFS` and define the sub-aperture sampling directly:

.. code-block:: yaml

	 slopes:
		 type: SHWFS
		 signalType: slopes
		 subApSpacing: 4
		 subApOffsetX: 0
		 subApOffsetY: 0
		 functions:
			 - computeSignal

That means pyRTC still owns the final slope reduction and loop control, while OOPAO or SPECULA owns the optical image formation.

Running the Examples
--------------------

.. code-block:: bash

	python examples/shwfs/shwfs_oopao_soft_rtc_example.py --duration 10
	python examples/shwfs/shwfs_specula_soft_rtc_example.py --duration 10

Viewer commands:

.. code-block:: bash

	pyrtc-view wfs signal2D wfc2D psfShort psfLong --geometry 2x3

Notes
-----

- These are soft-RTC examples because the simulator-backed components share in-process optical state.
- The OOPAO path uses OOPAO's real `ShackHartmann` class.
- The SPECULA path uses SPECULA's real `SH` processing object.
- If you want the simplest zero-dependency onboarding path, stay with :doc:`synthetic_shwfs`.