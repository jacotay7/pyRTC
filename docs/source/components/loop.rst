.. loop:

.. automodule:: pyRTC.Loop


Loop
====

The `Loop` component is the control stage of the AO pipeline.
It reads the processed wavefront signal, applies the configured control law, and writes the resulting correction vector to the wavefront-corrector stream.

In practical terms, `Loop` is where you:

- load or compute interaction and control matrices
- select an integration strategy
- tune gain or leak parameters
- apply dropped-mode and delay behavior
- manage correction updates sent to the wavefront corrector

Soft-RTC Example
----------------

The following example shows the general pattern for starting a loop in `soft-RTC` mode.
In this mode the control object lives in the same Python process as the rest of the AO chain.

.. code-block:: python

  import numpy as np
  from pyRTC.Loop import Loop
  from pyRTC.utils import read_yaml_file

  conf = read_yaml_file("path/to/config.yaml")
  loop = Loop(conf["loop"])

  # A calibrated system usually loads or computes IM first, then derives CM.
  loop.IM = np.eye(loop.signalSize, loop.numModes, dtype=np.float32)
  loop.computeCM()
  loop.setGain(0.1)
  loop.start()

Hard-RTC Example
----------------

The hard-RTC path is appropriate when the loop needs to interact with hardware-facing processes through shared memory while keeping process boundaries explicit.

.. code-block:: python
  
  from pyRTC.Pipeline import hardwareLauncher

  config = 'path/to/config.yaml'
  port = 3004

  loop = hardwareLauncher('path/to/pyRTC/Loop.py', config, port)
  loop.launch()

  # Once launched, controller methods and properties can be accessed remotely.
  loop.run("computeCM")
  loop.setProperty("gain", 0.1)
  print(loop.getProperty("gain"))

Control Notes
-------------

The loop class supports several control-related concepts that matter operationally:

- `gain` and `leakyGain` for integrator behavior
- `numDroppedModes` for excluding poorly behaved modes
- interaction-matrix and control-matrix workflows
- optional GPU-assisted paths when PyTorch is available
- delay and limit settings for controller tuning

In production use, the loop is usually one of the last components you tune after stream shapes, calibration files, and hardware-facing behavior are already stable.


Parameters
----------

.. autoclass:: Loop
  :members:
  :inherited-members:
  :undoc-members:
  :show-inheritance:
