.. wfs:

.. currentmodule:: pyrtc.wavefront_corrector

Wavefront Corrector
====================

In pyrtc, one of the core components is the wavefront corrector object. It typically finishes the AO chain by continously 
waiting for new corrections and applying them. This object is a consumer of the `wfc` shared memory object and a producer of
the `wfc_2d` shared memory objects which contain the current correction vector and the 2D representation of the curent correction
respectively. This class required you to properly lay out the 2D actuator layout as well as define the correction basis.

Modal Basis
-----------

The corrector maps each modal command onto its actuators through a mode-to-command
matrix ``M2C`` of shape ``[num_actuators, num_modes]``. pyrtc picks it in this order:

1. ``m2c_file``: a pre-built matrix (``.npy``, or raw float64 ``.dat``).
2. ``basis``: a basis generated at start-up with `aobasis <https://github.com/jacotay7/aobasis>`_
   from the corrector's actuator geometry.
3. Otherwise the identity (the first ``num_modes`` actuators, i.e. zonal control).

A ``basis`` section looks like this:

.. code-block:: yaml

  wfc:
    num_actuators: 97
    num_modes: 50
    basis:
      type: kl             # kl | zernike | fourier | zonal | zonal_fast | hadamard
      pupil_diameter: 8.0  # metres spanned by the actuator grid (default 8.0)
      r0: 0.16             # kl only, metres (default 0.16)
      L0: 30.0             # kl only, metres (default 30.0)
      ignore_piston: true  # kl, zernike, fourier (default true)
      normalize: peak      # peak | rms | l2 | none (default peak)
      orthonormalize: true # default true for zernike and fourier, false otherwise

Keys:

``type``
  ``kl`` (Karhunen-Loeve modes of von Karman turbulence), ``zernike`` (Noll order and
  normalization),
  ``fourier``, ``zonal`` (single-actuator pokes), ``zonal_fast`` (groups of actuators at
  least ``min_distance`` apart, poked together) or ``hadamard``.
``n_modes``
  Optional. Must equal ``num_modes``, which sizes the ``wfc`` stream.
``pupil_diameter``
  Physical scale of the actuator grid, in metres. It sets the Zernike unit circle
  (radius ``pupil_diameter / 2``) and the Fourier fundamental (one cycle per
  ``pupil_diameter``). KL mode shapes depend on ``pupil_diameter / L0`` only; ``r0``
  scales the eigenvalues but not the modes.
``positions_file``
  Optional ``(num_actuators, 2)`` array of actuator ``(x, y)`` positions in metres
  (``.npy`` or text), with the pupil centre at the origin, in command order. Use it for
  geometries that a 2D layout cannot describe. Without ``pupil_diameter``, the pupil is
  taken as twice the largest actuator radius.
``normalize``
  Scale of each mode: ``peak`` (largest actuator command is 1, like a zonal poke, so
  ``loop.poke_amp`` bounds the stroke), ``rms`` (unit RMS over actuators), ``l2`` (unit
  norm) or ``none`` (aobasis output unchanged).
``orthonormalize``
  Gram-Schmidt the modes in order before normalizing (``aobasis.orthonormalize_modes``).
  KL, zonal and zonal-fast modes are already orthogonal on the actuators. Zernike, Fourier
  and Hadamard modes sampled on a discrete actuator grid are not, and orthonormalizing
  them improves the conditioning of the interaction matrix (with a Zernike basis on the
  SPECULA SHWFS example, the closed-loop residual dropped from 4.9% to 1.0% of the
  aberration). The default is ``true`` for ``zernike`` and ``fourier`` and ``false``
  otherwise. Hadamard stays raw by default because orthonormalizing it destroys its
  +/-1 patterns. Mode ``k`` of the result spans the same space as input modes
  ``0..k``, so Noll or frequency order is kept.
``min_distance``
  ``zonal_fast`` only: minimum distance (metres) between actuators poked together.
  Default: twice the nearest-neighbour actuator spacing.
``ignore_piston``
  ``kl``, ``zernike`` and ``fourier``: leave out piston (default ``true``). A Fourier
  basis without piston holds at most ``num_actuators - 1`` modes, and aobasis raises if
  the grid cannot support ``num_modes`` independent Fourier modes (frequencies step by
  one cycle per ``pupil_diameter``, so aliasing limits some grids).
``use_gpu``
  ``kl`` only: compute the covariance and eigen-decomposition with CuPy.

**Actuator coordinates.** Unless a ``positions_file`` or the adapter supplies physical
positions, pyrtc derives them from the 2D actuator ``layout`` mask: ``x`` follows columns
and ``y`` rows, the origin is the centre of the mask, and the outermost rows and columns
sit on the pupil edge, so the pitch is ``pupil_diameter / (max(layout.shape) - 1)``
(the Fried-geometry convention of OOPAO and ``aobasis.make_circular_actuator_grid``).
Actuators are ordered like ``layout[layout]``, the order of the command vector. A corrector
without a layout uses the default circular layout for ``num_actuators`` until
:meth:`WavefrontCorrector.set_layout` provides the real one, and the basis is rebuilt then.

Adapters:

- ``SyntheticWFC`` builds the basis on its synthetic layout. The synthetic sensor works
  in modal space, so the basis does not change the simulated optics.
- ``OOPAOWFCorrector`` uses the OOPAO DM's actuator coordinates (``dm.coordinates``,
  including mis-registration) and defaults ``pupil_diameter`` to the telescope
  diameter. Without ``basis`` or ``m2c_file`` it keeps the identity, and
  ``OOPAOInterface.compute_and_load_kl_basis`` still loads OOPAO's own KL basis.
- ``SPECULAWFCorrector`` evaluates the basis at the SPECULA zonal actuator positions and
  defaults ``pupil_diameter`` to ``pixel_pupil * pixel_pitch``; for square geometry,
  actuators outside the circular support get zero rows. Without ``wfc.basis`` it keeps
  the SPECULA-native basis from the parameter file (SPECULA modal surfaces fitted onto
  the zonal influence functions).

:meth:`WavefrontCorrector.build_basis_m2c` rebuilds the matrix at runtime, optionally
from a new basis mapping, and :attr:`WavefrontCorrector.m2c_source` reports where the
current matrix came from (``file``, ``basis``, ``identity`` or ``custom``).

Soft-RTC Example
----------------

The following is an example of how to initialize a WavefrontCorrector component in pyrtc. 

Here we are in the `soft-RTC` mode of pyrtc, which holds all components in the same python process. 
See below for how to launch a hard-RTC equivalent.

.. code-block:: python

  """
  First we import the relevant wavefront corrector class. Typically, this will be a
  specific hardware class which has been defined to work with the SDK of your corrector.

  As an example (see hardware/alpao_dm.py):

  from pyrtc.hardware import ALPAODM

  Here, I will just initialize the Wavefront Sensor Superclass as an example
  """

  #%% Run in interactive python or jupyter notebook to keep process alive
  from pyrtc.wavefront_corrector import WavefrontCorrector
  import matplotlib.pyplot as plt
  from pyrtc.utils import read_yaml_file

  confWFC = {
  "name": "example",
  "num_actuators": 97,
  "num_modes": 50,
  "m2c_file": "", #Here you put the path to your basis ([nAct,nMode]) ./EXAMPLE/calib/wfc_shape.npy"
  "basis": {"type": "kl"}, #Or build the basis with aobasis (ignored when m2c_file is set)
  "save_file": "", #Here you put where the WFC will save its corrections ./EXAMPLE/calib/wfc_shape.npy"
  "affinity": 2,
  "functions": ["send_to_hardware"]
  }

  """
  Alternatively, read the config from a file

  conf = read_yaml_file("./EXAMPLE/config.yaml")["wfs"]
  """

  #Initialize the WFS object
  wfc = WavefrontCorrector(confWFC)
  #Start the functions regiserted to the loop (i.e, expose)
  wfc.start()

  wfc.flatten()

Hard-RTC Example
----------------

The following is an example of how to initialize a WavefrontCorrector component in pyrtc. 

Here we are in the `hard-RTC` mode of pyrtc, which holds all components in the separate python processes. 
This circumvents the python Global Interpreter Lock.

See above for how to launch a soft-RTC equivalent.

.. code-block:: python
  
  from pyrtc import HardwareLauncher

  """
  For the Hard-RTC, you will need to set-up a config before hand and store it in a yaml file.

  It should look something like:

  wfc:
    name: "ALPAO"
    serial: "BAX118"
    num_actuators: 97
    num_modes: 94
    flat_file: "./examples/sharp_lab/calib/wfc_shape.npy"
    save_file: "./examples/sharp_lab/calib/wfc_shape.npy"
    m2c_file: "./examples/sharp_lab/calib/m2c_kl.npy" 
    affinity: 5
    command_cap: 0.8
    frame_delay: 0
    functions:
    - send_to_hardware
  """
  config = 'path/to/config.yaml'
  port = 3000

  #Initialize the hardware launcher for your WFS child hardware class
  wfc = HardwareLauncher('path/to/pyrtc/hardware/alpao_dm.py',config,port)
  
  """
  Launch the process.

  This will run the hardware file, which should establish a connection with the current process.
  This is accomplished with the Listener class (see hardware folder for examples).

  The functions registered in the config to the real-time loop will automatically be started.
  """
  wfc.launch()

  """
  Once the connection has been made successfully, you can run any function in the hardware class
  using the run function. You can also get and set properties of the hardware using get_property()
  and set_property() respectively.
  """
  wfc.run("flatten")

  wfc.set_property("command_cap", 0.6)

  print(wfc.get_property("command_cap"))

Actuator Saturation
-------------------

With ``command_cap`` set, the corrector counts the actuators at the cap in each
command before clipping. ``wfc.safety_status()`` reports the count, the
fraction, and how many commands saturated any actuator. When at least
``saturation_warn_fraction`` of the actuators (default 0.05) are at the cap,
it logs a warning (at most every 10 s) and reports an alert. The manager
status (``safety``) and the manager GUI's graph node show it.

Multiple Correctors (Woofer/Tweeter, Tip-Tilt Offload)
------------------------------------------------------

To drive several correctors from one loop, put a ``CorrectorSplitter`` in the
``wfc`` section and give each device its own section and ``wfc`` stream:

Boston Micromachines DMs
------------------------

``pyrtc.hardware.bmc_dm.BMCDM`` drives BMC MEMS mirrors (Multi-DM, Kilo-DM,
2K, 3K) through the BMC DM SDK's Python module ``bmc``:

.. code-block:: yaml

  wfc:
    class_name: pyrtc.corrector_splitter.CorrectorSplitter
    correctors:
      - {name: woofer, stream: woofer_wfc, modes: 20}
      - {name: tweeter, stream: tweeter_wfc, modes: 60}
    offload: {source: tweeter, target: woofer, gain: 0.02}   # optional
    functions: [split]
  woofer:
    class_name: ALPAODM          # any wavefront corrector
    num_modes: 20
    input_streams: {wfc: woofer_wfc}
    output_streams: {wfc: woofer_wfc, wfc_2d: woofer_wfc_2d}
    ...

The loop controls ``20 + 60`` modes: the woofer's first, then the tweeter's.
The splitter writes each corrector its slice every frame, so the interaction
matrix calibrates both devices with no change to the loop.

With ``offload``, the target corrector (the woofer, or a tip-tilt stage)
takes over, at rate ``gain``, the part of the source command it can
represent. The total wavefront does not change. It needs the coupling ``C``,
the source-mode equivalent of each target mode; calibrate the IM first, then
set it from the IM:

.. code-block:: python

  loop.compute_im()
  splitter = manager.get_component("wfc")
  splitter.set_coupling_from_im(loop.im)   # C = pinv(IM_source) @ IM_target
  # the configured gain now applies; set_offload_gain(g) changes it

``coupling`` can also be given in the config, as a matrix or a ``.npy`` file.
For tip-tilt offload to a two-mode stage, it has non-zero rows only for the
DM's tip and tilt modes. ``reset_offload()`` hands everything back to the
source.

    class_name: pyrtc.hardware.bmc_dm.BMCDM
    name: kilo_dm
    serial: "25CW012#023"
    sdk_path: /opt/Boston Micromachines/lib/python3   # if 'bmc' is not importable
    num_modes: 400
    bias: 0.5            # SDK value for a zero command
    command_scale: 0.5   # SDK value change per unit command
    command_cap: 0.8
    functions: [send_to_hardware]

The SDK takes one value per actuator in ``[0, 1]``. MEMS mirrors only pull,
so commands are applied about ``bias``:
``clip(bias + command_scale * command, 0, 1)``.

- The actuator count comes from the SDK.
- The layout is BMC's standard geometry: a square grid, or a square without
  its corners (Multi-DM 140 on 12x12, Kilo-DM 1020 on 32x32), otherwise a
  centred disk. Use ``layout_file`` for any other map.
- Closing the component zeroes the mirror before releasing it.

Parameters
----------

.. autoclass:: WavefrontCorrector
  :members:
  :inherited-members:
  :undoc-members:
  :show-inheritance:
  :no-index: