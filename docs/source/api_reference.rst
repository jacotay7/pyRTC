API Reference
=============

This section complements the narrative component pages with a compact reference
to the public package surface and the runtime/helper modules that users most
often inspect while extending pyRTC.

Public Package Surface
----------------------

.. automodule:: pyRTC
   :members:
   :undoc-members:

Module Index
------------

.. currentmodule:: pyRTC

.. autosummary::
   :toctree: generated
   :nosignatures:

   Pipeline
   utils

.. currentmodule:: pyRTC.hardware

.. autosummary::
   :toctree: generated
   :nosignatures:

   SyntheticSystems

Notes
-----

Optional vendor-backed hardware adapters are not listed here because some of
them depend on site-specific SDKs that may not be installed on the docs host.
Those adapters are still documented in source and in the hardware example
modules under ``pyRTC.hardware``.