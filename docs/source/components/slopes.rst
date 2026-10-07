.. wfs:

.. currentmodule:: pyrtc.slopes_process


Slopes Process
==============

The slopes process is responsible for converting images from the wavefront sensor into a measurement consumable by the
AO loop. This object is the producer of the `signal` and `signal_2d` shared memory objects
which contain the vectorized and 2D mapped images of the slopes respectively. It is a consumer of the `wfs` shared memory object
which contains the image stream from the wavefront sensor. The images are then be processed to compute the intermediate data product 
used for wavefront reconstruction.

To compute the signal with a PyTorch model instead (a neural or focal-plane
reconstructor), put a :doc:`TorchImageReconstructor <image_reconstructor>` in
the ``slopes`` section.

Soft-RTC Example
----------------

The following is an example of how to initialize a SlopesProcess component in pyrtc. 

Here we are in the `soft-RTC` mode of pyrtc, which holds all components in the same python process. 
See below for how to launch a hard-RTC equivalent.

.. code-block:: python

  """
  First we import the relevant class.

  Here I will give an example for a Pyramid Wavefront Sensor
  """

  #%% Run in interactive python or jupyter notebook to keep process alive
  from pyrtc.slopes_process import SlopesProcess
  import matplotlib.pyplot as plt
  from pyrtc.utils import read_yaml_file

  confWFS = {
  "width": 256,
  "height": 256,
  }

  confSlopes = {
    "type": "SHWFS",
    "signal_type": "slopes",
    "ref_slopes_file": "", #"/home/whetstone/pyrtc/examples/sharp_lab/calib/ref.npy",
    "valid_sub_aps_file": "", #"/home/whetstone/pyrtc/examples/sharp_lab/calib/valid_sub_aps.npy",
    "sub_ap_spacing": 16,
    "sub_ap_offset_x": 0,
    "sub_ap_offset_y": 0,
    "image_noise": 0.5,
    "contrast": 20,
    "affinity": 4,
    "functions": ["compute_signal"],
  }

  conf = {"wfs": confWFS, "slopes": confSlopes}

  """
  Alternatively, read the config from a file

  conf = read_yaml_file("./EXAMPLE/config.yaml")
  """

  #Initialize the WFS object
  slopes = SlopesProcess(conf)
  #Start the functions regiserted to the loop (i.e, expose)
  slopes.start()

  signal = slopes.read(block=False)

  plt.plot(signal)
  plt.show()

  """
  Monitor the SHM in realtime by running the viewer command in a terminal
  pyrtc-view signal_2d &
  """

Hard-RTC Example
----------------

The following is an example of how to initialize a SlopesProcess component in pyrtc. 

Here we are in the `hard-RTC` mode of pyrtc, which holds all components in the separate python processes. 
This circumvents the python Global Interpreter Lock.

See above for how to launch a soft-RTC equivalent.

.. code-block:: python

  from pyrtc import HardwareLauncher

  """
  For the Hard-RTC, you will need to set-up a config before hand and store it in a yaml file.

  It should look something like:

  slopes:
    type: SHWFS
    signal_type: slopes
    ref_slopes_file: "/home/whetstone/pyrtc/examples/sharp_lab/calib/ref.npy"
    valid_sub_aps_file: "/home/whetstone/pyrtc/examples/sharp_lab/calib/valid_sub_aps.npy"
    sub_ap_spacing: 16
    sub_ap_offset_x: 8
    sub_ap_offset_y: 4
    image_noise: 0.5
    contrast: 20
    affinity: 4
    functions:
    - compute_signal
  """

  config = 'path/to/config.yaml'
  port = 3005

  #Initialize the hardware launcher for your WFS child hardware class
  slopes = HardwareLauncher('path/to/pyrtc/slopes_process.py', config, port)

  """
  Launch the process.

  This will run the hardware file, which should establish a connection with the current process.
  This is accomplished with the Listener class (see hardware folder for examples).

  The functions registered in the config to the real-time loop will automatically be started.
  """
  slopes.launch()

  """
  Once the connection has been made successfully, you can run any function in the hardware class
  using the run function. You can also get and set properties of the hardware using get_property()
  and set_property() respectively.
  """
  slopes.run("load_valid_sub_aps")

  slopes.set_property("ref_slopes_file", "test123")

  print(slopes.get_property("ref_slopes_file"))


Slope axes and layout
---------------------

The ``wfs`` image is ``(height, width)``, indexed ``[y, x]``. Slopes are in
pixels, with x along the image columns and y along the rows, and the
``signal`` vector is *blocked*: all x slopes, then all y slopes (aocore
CONVENTIONS 7.1). An OPD ramp along +x moves every spot towards +x and gives
positive x slopes.

- **SHWFS.** The sub-aperture grid has ``min(height, width) // round(sub_ap_spacing)``
  sub-apertures per side, ordered row-major over ``(subap_y, subap_x)``.
  ``signal_2d`` is ``(2N, N)``: the ``N x N`` x-slope map stacked on the
  y-slope map, and the valid sub-aperture mask has the same layout. Pixel
  ``k`` of a sub-aperture sits at ``k - (n - 1) / 2``, so a spot on the
  sub-aperture's optical axis reads 0 also for even ``n`` (CONVENTIONS 1.2).
- **PYWFS.** ``pupils`` entries are ``"x,y"``: the column and row of each
  pupil centre. List them as (low x, low y), (low x, high y), (high x, low
  y), (high x, high y); then ``sx = (p1 + p2) - (p3 + p4)`` compares the
  pupils across columns and ``sy = (p1 + p3) - (p2 + p4)`` across rows,
  each normalised by the mean pupil flux. Without ``pupils`` the four
  quadrant centres are used in that order. ``signal_2d`` holds the x and y
  slope images of the pupil side by side.

pyrtc 1.x used ``(width, height)`` streams and ``k - n // 2`` pixel
coordinates (#162, #163); see :doc:`../guides/migration_2_0` for 1.x
configs and calibration files.

SHWFS centroiding
-----------------

For a Shack-Hartmann sensor the ``centroider`` slopes option selects how each
sub-aperture is reduced to an x/y spot position (in pixels, relative to the
sub-aperture centre; reference slopes are subtracted afterwards, exactly as for
the default). Every method ignores pixels at or below ``image_noise * contrast``
and publishes 0 for a sub-aperture with no flux above that threshold.

``cog`` (default)
  Thresholded centre of gravity. Unbiased for a compact spot well inside the
  sub-aperture, but its noise grows with every pixel that passes the threshold.

``wcog``
  Gaussian-weighted centre of gravity. Each pixel is multiplied by a Gaussian of
  FWHM ``wcog_fwhm`` pixels (default: half the sub-aperture) centred on the
  reference position: the thresholded CoG of the reference image when one is
  set (see below), otherwise the sub-aperture centre. The weights are built
  once and rebuilt only when the centres or the FWHM change.

  The weighting shrinks the measured displacement. For a Gaussian spot of FWHM
  :math:`f_s` and a weight of FWHM :math:`f_w` the measured shift is
  :math:`g = f_w^2 / (f_w^2 + f_s^2)` of the true one. For Gaussian spots the
  response stays linear until the spot nears the sub-aperture edge; for other
  spot shapes (Airy rings, elongated or extended spots) :math:`g` differs and
  the response becomes non-linear as the spot leaves the weight, while the
  weighted flux, and so the SNR, falls. By default the gain is **not**
  corrected: in a closed loop the interaction matrix is measured with the same
  centroider and absorbs it. Set ``wcog_spot_fwhm`` to the spot FWHM in pixels
  (include the pixel, :math:`f_s \approx 2.355\sqrt{\sigma^2 + 1/12}`) to
  multiply by :math:`1/g`, e.g. when the slopes are used in physical units. The
  correction is exact only for Gaussian spots, so keep ``wcog_fwhm`` larger
  than the expected spot motion. The payoff is noise: with read noise the WCoG
  error is several times lower than the thresholded CoG's.

``correlation``
  Correlation centroiding for extended sources (solar or laser-guide-star
  wavefront sensing). The reference image supplies one template per
  sub-aperture: its central ``sub_ap_spacing - 2 * correlation_search_radius``
  square. For every integer shift within ``+/- correlation_search_radius``
  pixels (default: a quarter of the sub-aperture) the kernel evaluates the
  squared difference between the template and the flux-normalised live image;
  the template always overlaps the live sub-aperture fully, so there is no
  overlap bias. The minimum is refined with a 2D quadratic fit over its 3x3
  neighbourhood (Löfdahl 2010, A&A 524, A90), and the shift is added to the
  reference spot position. Shifts larger than the search window are clamped to
  its edge. The cost grows with ``(2R + 1)^2 * (n - 2R)^2`` per sub-aperture, so
  keep the search radius as small as the expected spot motion allows.

The reference image is a dark-subtracted WFS frame with the flat (or
calibration) wavefront applied. Take it once the pipeline is running, then take
reference slopes:

.. code-block:: python

  slopes.take_reference_image()   # averages ref_slope_count WFS frames
  slopes.save_reference_image()   # to reference_image_file
  slopes.take_ref_slopes()

It can also be loaded at start-up from ``reference_image_file`` or set with
``set_reference_image(image)``. Templates and reference positions are derived
lazily and rebuilt when the threshold or search radius changes. Until a
reference exists the ``correlation`` centroider publishes zero slopes and logs a
warning once.

.. code-block:: yaml

  slopes:
    type: SHWFS
    signal_type: slopes
    sub_ap_spacing: 16
    sub_ap_offset_x: 0
    sub_ap_offset_y: 0
    centroider: correlation        # cog | wcog | correlation
    correlation_search_radius: 4
    reference_image_file: calib/shwfs_reference.npy
    # wcog_fwhm: 8.0
    # wcog_spot_fwhm: 3.0

Cost relative to ``cog`` (single core, one frame of spots in read noise, on
an Arm Neoverse-N1; ``benchmarks/core_compute_bench.py`` reports absolute
timings as ``slopes.compute_slopes_shwfs_*_numba``):

.. list-table::
  :header-rows: 1

  * - Geometry
    - ``cog``
    - ``wcog``
    - ``correlation``
  * - 20x20 sub-apertures, 8 px, R = 2
    - ~27 us
    - ~1.7x
    - ~10x
  * - 20x20 sub-apertures, 16 px, R = 4
    - ~76 us
    - ~1.7x
    - ~18x
  * - 60x60 sub-apertures, 8 px, R = 2
    - ~0.24 ms
    - ~1.7x
    - ~10x

The CoG and WCoG sums are vectorized (numba ``fastmath`` reassociation):
slopes from integer frames, such as the ``int32`` ``wfs`` stream, are exact,
and slopes from float frames can differ from a strictly sequential sum in the
last bits.

PYWFS on the GPU
----------------

With ``gpu_device`` set to a CUDA device, PYWFS slopes are computed with
PyTorch on that device, from the ``wfs`` stream's CUDA tensor when the WFS
publishes one, or from a host frame copied up through a pinned buffer. The
dozen or so small kernels of one frame are captured once, at construction, as
a CUDA graph and replayed every frame, which removes most of their launch
cost. The capture is checked against the eager computation bit for bit; if it
fails, the component logs a warning and computes eagerly. New reference slopes
are copied into the graph; a new pupil geometry, image shape or dtype
captures a new graph on the next frame.

Parameters
----------

.. autoclass:: SlopesProcess
  :members:
  :inherited-members:
  :undoc-members:
  :show-inheritance:
  :no-index:
