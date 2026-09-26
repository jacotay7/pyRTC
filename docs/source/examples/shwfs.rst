.. SHWFS Simulator Examples

SHWFS Simulator Examples
========================

This page describes the simulator-backed Shack-Hartmann examples that mirror the Pyramid-WFS walkthroughs under :doc:`pywfs`.

Purpose
-------

These examples are the next step after :doc:`synthetic_shwfs` when you want a real optical simulator in the loop while keeping the standard pyrtc control chain.

Files
-----

The example assets live under `examples/shwfs/`:

- `shwfs_oopao_soft_rtc_example.py`: OOPAO-backed Shack-Hartmann soft-RTC walkthrough
- `shwfs_OOPAO_config.yaml`: pyrtc config for the OOPAO Shack-Hartmann example
- `shwfs_OOPAO_params.yaml`: flat OOPAO parameter dictionary for the OOPAO Shack-Hartmann example
- `shwfs_specula_soft_rtc_example.py`: SPECULA-backed Shack-Hartmann soft-RTC walkthrough
- `shwfs_SPECULA_config.yaml`: pyrtc config for the SPECULA Shack-Hartmann example
- `shwfs_SPECULA_params.yaml`: SPECULA object-graph parameters for the SPECULA Shack-Hartmann example

Installing the Simulator
------------------------

The simulator is not installed with pyrtc. Install the backend you want to run: OOPAO for the ``shwfs_oopao_*`` / ``shwfs_OOPAO_*`` files, SPECULA for the ``shwfs_specula_*`` / ``shwfs_SPECULA_*`` files.

OOPAO
~~~~~

OOPAO is not on PyPI, so it cannot be a pyrtc extra. Installing it straight from GitHub (``pip install git+https://github.com/cheritier/OOPAO.git``) is not enough: ``import OOPAO`` then fails with ``ValueError: attempt to get argmin of an empty sequence``. At import time, ``OOPAO/__init__.py`` looks for an entry containing ``OOPAO`` on ``sys.path`` and writes a small file into it. A plain ``pip install`` puts OOPAO in ``site-packages``, so no such entry exists.

Clone OOPAO and put the clone on ``PYTHONPATH``:

.. code-block:: bash

	git clone https://github.com/cheritier/OOPAO.git   # keep the directory name "OOPAO"
	pip install ./OOPAO                                # installs OOPAO's dependencies
	export PYTHONPATH="$PWD/OOPAO:$PYTHONPATH"
	python -c "import OOPAO"                           # check the install

The clone's path must contain ``OOPAO`` (the match is case-sensitive, so a directory renamed to ``oopao`` fails) and must be writable. ``PYTHONPATH`` takes precedence over ``site-packages``, so the clone is the copy that gets imported. Add the ``export`` line to your shell profile or environment activation script so it persists.

SPECULA
~~~~~~~

SPECULA is on PyPI and is available as a pyrtc extra:

.. code-block:: bash

	pip install pyrtcao[specula]   # from a source checkout: pip install .[specula]
	# or install SPECULA on its own: pip install specula

The SPECULA example scripts also add a sibling ``SPECULA`` checkout (next to the pyrtc repository) to ``sys.path`` when one exists, so a local SPECULA development checkout there is used instead of the installed package.

What the Config Shows
---------------------

The SHWFS examples switch the slopes section from `PYWFS` to `SHWFS` and define the sub-aperture sampling directly:

.. code-block:: yaml

	 slopes:
		 type: SHWFS
		 signal_type: slopes
		 sub_ap_spacing: 4
		 sub_ap_offset_x: 0
		 sub_ap_offset_y: 0
		 functions:
			 - compute_signal

That means pyrtc still owns the final slope reduction and loop control, while OOPAO or SPECULA owns the optical image formation.

Running the Examples
--------------------

.. code-block:: bash

	python examples/shwfs/shwfs_oopao_soft_rtc_example.py --duration 10
	python examples/shwfs/shwfs_specula_soft_rtc_example.py --duration 10

Viewer commands:

.. code-block:: bash

	pyrtc-view wfs signal_2d wfc_2d psf_short psf_long --geometry 2x3

Notes
-----

- These are soft-RTC examples because the simulator-backed components share in-process optical state.
- The OOPAO path uses OOPAO's real `ShackHartmann` class.
- The SPECULA path uses SPECULA's real `SH` processing object.
- The SPECULA example calibrates with the atmosphere removed: it waits until a DM poke
  reaches the slopes, takes reference slopes on the flat DM, then measures the push-pull IM.
  With ``specula.use_atmosphere: false`` (the default config) the loop then converges to zero
  residual; set it to ``true`` to close the loop on turbulence.
- If you want the simplest zero-dependency onboarding path, stay with :doc:`synthetic_shwfs`.