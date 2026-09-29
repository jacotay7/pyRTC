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

pyrtc image streams have shape ``(width, height)``: ``width`` is the first
array axis. Camera SDKs return frames as ``(Height, Width)`` (rows,
columns), so an adapter must transpose each frame (``frame.T``) before
publishing it. Then ``width``/``height`` in the config mean the same as the
camera's own settings. The GenICam and Micro-Manager adapters do this. The
XIMEA and Spinnaker adapters don't yet, so use square ROIs with them (#130).

Parameters
----------

.. autoclass:: ScienceCamera
  :members:
  :inherited-members:
  :undoc-members:
  :show-inheritance:
  :no-index:
