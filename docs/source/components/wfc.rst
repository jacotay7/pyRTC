.. wfs:

.. automodule:: pyRTC.WavefrontCorrector

Wavefront Corrector
====================

In pyRTC, one of the core components is the wavefront corrector object. It typically finishes the AO chain by continously 
waiting for new corrections and applying them. This object is a consumer of the `wfc` shared memory object and a producer of
the `wfc2D` shared memory objects which contain the current correction vector and the 2D representation of the curent correction
respectively. This class required you to properly lay out the 2D actuator layout as well as define the correction basis.

Soft-RTC Example
----------------

The following is an example of how to initialize a WavefrontCorrector component in pyRTC. 

Here we are in the `soft-RTC` mode of pyRTC, which holds all components in the same python process. 
See below for how to launch a hard-RTC equivalent.

.. code-block:: python

  """
  First we import the relevant wavefront corrector class. Typically, this will be a
  specific hardware class which has been defined to work with the SDK of your corrector.

  As an example (see hardware/ALPAODM.py):

  from pyRTC.hardware import ALPAODM

  Here, I will just initialize the Wavefront Sensor Superclass as an example
  """

  #%% Run in interactive python or jupyter notebook to keep process alive
  from pyRTC import WavefrontCorrector
  import matplotlib.pyplot as plt
  from pyRTC.utils import read_yaml_file

  confWFC = {
  "name": "example",
  "numActuators": 97,
  "numModes": 50,
  "m2cFile": "", #Here you put the path to your basis ([nAct,nMode]) ./EXAMPLE/calib/wfcShape.npy"
  "saveFile": "", #Here you put where the WFC will save its corrections ./EXAMPLE/calib/wfcShape.npy"
  "affinity": 2,
  "functions": ["sendToHardware"]
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

The following is an example of how to initialize a WavefrontCorrector component in pyRTC. 

Here we are in the `hard-RTC` mode of pyRTC, which holds all components in the separate python processes. 
This circumvents the python Global Interpreter Lock.

See above for how to launch a soft-RTC equivalent.

.. code-block:: python
  
  from pyRTC import hardwareLauncher

  """
  For the Hard-RTC, you will need to set-up a config before hand and store it in a yaml file.

  It should look something like:

  wfc:
    name: "ALPAO"
    serial: "BAX118"
    numActuators: 97
    numModes: 94
    flatFile: "./SHARP_LAB/calib/wfcShape.npy"
    saveFile: "./SHARP_LAB/calib/wfcShape.npy"
    m2cFile: "./SHARP_LAB/calib/m2c_kl.npy" 
    affinity: 5
    commandCap: 0.8
    hardwareDelay: 0.001 #seconds
    frameDelay: 0
    functions:
    - sendToHardware
  """
  config = 'path/to/config.yaml'
  port = 3000

  #Initialize the hardware launcher for your WFS child hardware class
  wfc = hardwareLauncher('path/to/pyRTC/hardware/ALPAODM.py',config,port)
  
  """
  Launch the process.

  This will run the hardware file, which should establish a connection with the current process.
  This is accomplished with the Listener class (see hardware folder for examples).

  The functions registered in the config to the real-time loop will automatically be started.
  """
  wfc.launch()

  """
  Once the connection has been made successfully, you can run any function in the hardware class
  using the run function. You can also get and set properties of the hardware using getProperty()
  and setProperty() respectively.
  """
  wfc.run("flatten")

  wfc.setProperty("commandCap", 0.6)

  print(wfc.getProperty("commandCap"))

Parameters
----------

.. autoclass:: WavefrontCorrector
  :members:
  :inherited-members:
  :undoc-members:
  :show-inheritance: