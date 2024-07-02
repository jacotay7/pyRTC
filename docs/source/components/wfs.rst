.. wfs:

.. automodule:: pyRTC.WavefrontSensor


Wavefront Sensor
================

In pyRTC, one of the core components is the wavefront sensor object. It typically starts the AO chain by running
a continues capture sequence of images. This object is the producer of the `wfs` and `wfsRaw` shared memory objects
which contain the dark subtracted and original image respectively. The images can then be processed by the slopesProcess
class to compute the intermediate data product used for wavefront reconstruction.

Soft-RTC Example
----------------

The following is an example of how to initialize a WavefrontSensor component in pyRTC. Here we are in the `soft-RTC` mode
of pyRTC, which holds all components in the same python process. See below for how to launch a hard-RTC equivalent.

.. code-block:: python

  """
  First we import the relevant wavefront sensor class. Typically, this will be a
  specific hardware class which has been defined to work with the SDK of your camera.

  As an example (see hardware/ximeaWFS.py):

  from pyRTC.hardware import XIMEA_WFS

  Here, I will just initialize the Wavefront Sensor Superclass as an example
  """

  #%% Run in interactive python or jupyter notebook to keep process alive
  from pyRTC import WavefrontSensor
  import matplotlib.pyplot as plt
  from pyRTC.utils import read_yaml_file

  confWFS = {
  "name": "example",
  "width": 256,
  "height": 256,
  "darkCount": 1000,
  "darkFile": "", #Here you can add a dark file, if existing "./EXAMPLE/calib/wfsDark.npy",
  "affinity": 2,
  "functions": ["expose"]
  }

  """
  Alternatively, read the config from a file

  conf = read_yaml_file("./EXAMPLE/config.yaml")["wfs"]
  """

  #Initialize the WFS object
  wfs = WavefrontSensor(confWFS)
  #Start the functions regiserted to the loop (i.e, expose)
  wfs.start()

  img = wfs.read()


  plt.imshow(img)
  plt.show()


  """
  Monitor the SHM in realtime by running the pyRTCView script in a terminal
  python pyRTCView.py wfs &
  """

Hard-RTC Example
----------------

The following is an example of how to initialize a WavefrontSensor component in pyRTC. Here we are in the `hard-RTC` mode
of pyRTC, which holds all components in the separate python processes. This circumvents the python Global Interpreter Lock
See above for how to launch a soft-RTC equivalent.

.. code-block:: python
  
  from pyRTC import hardwareLauncher

  """
  For the Hard-RTC, you will need to set-up a config before hand and store it in a yaml file.

  It should look something like:

  wfs:
    name: XIMEA
    serial: "46052550"
    binning: 1 
    exposure: 2000
    gain: 0
    bitDepth: 10
    left: 448 
    top: 280 
    width: 400 
    height: 400 
    darkCount: 2000
    darkFile: "/home/whetstone/pyRTC/SHARP_LAB/calib/dark.npy"
    affinity: 3
    functions:
    - expose
  """
  config = 'path/to/config.yaml'
  port = 3000

  #Initialize the hardware launcher for your WFS child hardware class
  wfs = hardwareLauncher('path/to/pyRTC/hardware/myHardwareWfs.py',config,port)
  """
  Launch the process.

  This will run the hardware file, which should establish a connection with the current process.
  This is accomplished with the Listener class (see hardware folder for examples).

  The functions registered in the config to the real-time loop will automatically be started.
  """
  wfs.launch()

  """
  Once the connection has been made successfully, you can run any function in the hardware class
  using the run function. You can also get and set properties of the hardware using getProperty()
  and setProperty() respectively.
  """
  wfs.run("expose")

  wfs.setProperty("exposure", 100)

  print(wfs.getProperty("exposure"))


Parameters
----------

.. autoclass:: WavefrontSensor
  :members:
  :inherited-members:
  :undoc-members:
  :show-inheritance:
