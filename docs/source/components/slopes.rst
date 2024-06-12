.. wfs:

.. automodule:: pyRTC.SlopesProcess


Slopes Process
==============

The slopes process is responsible for converting images from the wavefront sensor into a measurement consumable by the
AO loop. This object is the producer of the `signal` and `signal2D` shared memory objects
which contain the vectorized and 2D mapped images of the slopes respectively. It is a consumer of the `wfs` shared memory object
which contains the image stream from the wavefront sensor. The images are then be processed to compute the intermediate data product 
used for wavefront reconstruction.

Soft-RTC Example
----------------

The following is an example of how to initialize a SlopesProcess component in pyRTC. 

Here we are in the `soft-RTC` mode of pyRTC, which holds all components in the same python process. 
See below for how to launch a hard-RTC equivalent.

.. code-block:: python

  """
  First we import the relevant class.

  Here I will give an example for a Pyramid Wavefront Sensor
  """

  #%% Run in interactive python or jupyter notebook to keep process alive
  from pyRTC import SlopesProcess
  import matplotlib.pyplot as plt
  from pyRTC.utils import read_yaml_file

  confWFS = {
  "width": 256,
  "height": 256,
  }

  confSlopes = {
    "type": "SHWFS",
    "signalType": "slopes",
    "refSlopesFile": "", #"/home/whetstone/pyRTC/SHARP_LAB/calib/ref.npy",
    "validSubApsFile": "", #"/home/whetstone/pyRTC/SHARP_LAB/calib/validSubAps.npy",
    "subApSpacing": 16,
    "subApOffsetX": 0,
    "subApOffsetY": 0,
    "imageNoise": 0.5,
    "contrast": 20,
    "affinity": 4,
    "functions": ["computeSignal"],
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
  Monitor the SHM in realtime by running the pyRTCView script in a terminal
  python pyRTCView.py signal2D &
  """

Hard-RTC Example
----------------

The following is an example of how to initialize a SlopesProcess component in pyRTC. 

Here we are in the `hard-RTC` mode of pyRTC, which holds all components in the separate python processes. 
This circumvents the python Global Interpreter Lock.

See above for how to launch a soft-RTC equivalent.

.. code-block:: python

  from pyRTC import hardwareLauncher

  """
  For the Hard-RTC, you will need to set-up a config before hand and store it in a yaml file.

  It should look something like:

  slopes:
    type: SHWFS
    signalType: slopes
    refSlopesFile: "/home/whetstone/pyRTC/SHARP_LAB/calib/ref.npy"
    validSubApsFile: "/home/whetstone/pyRTC/SHARP_LAB/calib/validSubAps.npy"
    subApSpacing: 16
    subApOffsetX: 8
    subApOffsetY: 4
    imageNoise: 0.5
    contrast: 20
    affinity: 4
    functions:
    - computeSignal
  """

  config = 'path/to/config.yaml'
  port = 3005

  #Initialize the hardware launcher for your WFS child hardware class
  slopes = hardwareLauncher('path/to/pyRTC/SlopesProcess.py', config, port)

  """
  Launch the process.

  This will run the hardware file, which should establish a connection with the current process.
  This is accomplished with the Listener class (see hardware folder for examples).

  The functions registered in the config to the real-time loop will automatically be started.
  """
  slopes.launch()

  """
  Once the connection has been made successfully, you can run any function in the hardware class
  using the run function. You can also get and set properties of the hardware using getProperty()
  and setProperty() respectively.
  """
  slopes.run("loadValidSubAps")

  slopes.setProperty("refSlopesFile", "test123")

  print(slopes.getProperty("refSlopesFile"))


Parameters
----------

.. autoclass:: SlopesProcess
  :members:
  :inherited-members:
  :undoc-members:
  :show-inheritance:
