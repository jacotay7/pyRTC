API Reference
=============

This section complements the narrative component pages with a compact reference
to the public package surface and the runtime/helper modules that users most
often inspect while extending pyrtc.

Public Package Surface
----------------------

.. automodule:: pyrtc
   :members:
   :undoc-members:

Module Index
------------

.. currentmodule:: pyrtc

.. autosummary::
   :toctree: generated
   :nosignatures:

   streams
   rpc
   manager
   component_loading
   modal_basis
   corrector_splitter
   image_reconstructor
   modal_gains
   predictive
   latency
   isio_bridge
   telemetry
   utils

.. currentmodule:: pyrtc.hardware

.. autosummary::
   :toctree: generated
   :nosignatures:

   synthetic_systems
   hcipy_interface
   genicam_camera
   micromanager_camera
   bmc_dm

Notes
-----

Hardware adapters whose vendor SDK is imported at module load (ALPAO, XIMEA,
Spinnaker, PI, OOPAO, SPECULA) are not listed here, because those SDKs may
not be installed on the docs host. The adapters listed above import their SDK
only when constructed.
Those adapters are still documented in source and in the hardware example
modules under ``pyrtc.hardware``.