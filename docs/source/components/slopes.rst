.. wfs:

.. currentmodule:: pyrtc.slopes_process


Slopes Process
==============

The slopes process is responsible for converting images from the wavefront sensor into a measurement consumable by the
AO loop. This object is the producer of the `signal` and `signal_2d` shared memory objects
which contain the vectorized and 2D mapped images of the slopes respectively. It is a consumer of the `wfs` shared memory object
which contains the image stream from the wavefront sensor. The images are then be processed to compute the intermediate data product 
used for wavefront reconstruction.

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

  from pyrtc.pipeline import HardwareLauncher

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


Parameters
----------

.. autoclass:: SlopesProcess
  :members:
  :inherited-members:
  :undoc-members:
  :show-inheritance:
  :no-index:
