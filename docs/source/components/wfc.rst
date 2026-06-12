.. wfs:

.. currentmodule:: pyrtc.wavefront_corrector

Wavefront Corrector
====================

In pyrtc, one of the core components is the wavefront corrector object. It typically finishes the AO chain by continously 
waiting for new corrections and applying them. This object is a consumer of the `wfc` shared memory object and a producer of
the `wfc_2d` shared memory objects which contain the current correction vector and the 2D representation of the curent correction
respectively. This class required you to properly lay out the 2D actuator layout as well as define the correction basis.

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
  
  from pyrtc.pipeline import HardwareLauncher

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
    hardware_delay: 0.001 #seconds
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

Parameters
----------

.. autoclass:: WavefrontCorrector
  :members:
  :inherited-members:
  :undoc-members:
  :show-inheritance:
  :no-index: