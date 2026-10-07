.. scicam:

.. currentmodule:: pyrtc.science_camera


Science Camera
==============

The `ScienceCamera` component represents the imaging path used to evaluate science output from the AO system.
Unlike the wavefront sensor, this component is typically used to observe performance metrics such as PSF structure, long-exposure integration, tip-tilt behavior, and Strehl-related quantities.

The class manages the shared-memory outputs associated with science imaging, including short- and long-exposure PSF products.

Soft-RTC Example
----------------

The following example shows the typical `soft-RTC` pattern for science-camera setup.

.. code-block:: python

  from pyrtc.science_camera import ScienceCamera
  from pyrtc.utils import read_yaml_file

  conf = read_yaml_file("path/to/config.yaml")
  sci = ScienceCamera(conf["psf"])
  sci.start()

  # The short-exposure and long-exposure products are written to shared memory.
  sci.expose()
  sci.integrate()

Hard-RTC Example
----------------

If the science camera is tied to a specific vendor SDK or operational process boundary, it can also be launched in `hard-RTC` mode.

.. code-block:: python
  
  from pyrtc import HardwareLauncher

  config = 'path/to/config.yaml'
  port = 3003

  sci = HardwareLauncher('path/to/pyrtc/hardware/myScienceCamera.py', config, port)
  sci.launch()
  sci.run("integrate")

Operational Notes
-----------------

The science camera is usually responsible for image-quality observables rather than control observables.
Common configuration and workflow concerns include:

- short- vs long-exposure output products
- dark-frame handling
- model PSF loading
- ROI, gain, binning, and exposure settings
- downstream analysis such as Strehl and centroid-derived metrics

This class is often subclassed for site-specific cameras under `pyrtc.hardware`.


Image Shape Convention
----------------------

pyrtc image streams have shape ``(height, width)`` and are indexed
``[y, x]``: rows are the camera's y axis and columns its x axis (aocore
CONVENTIONS 1.1). That is the layout camera SDKs return, so adapters publish
frames as they come, without transposing them, and ``width``/``height`` in
the config mean the same as the camera's own settings.

pyrtc 1.x declared streams ``(width, height)`` and some adapters transposed
frames into them (#162, #130). Calibration files saved with 1.x need
``legacy_calibration`` or a re-measurement; see :doc:`../guides/migration_2_0`.

Parameters
----------

.. autoclass:: ScienceCamera
  :members:
  :inherited-members:
  :undoc-members:
  :show-inheritance:
  :no-index:
