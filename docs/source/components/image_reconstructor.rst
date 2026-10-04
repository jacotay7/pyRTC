.. image_reconstructor:

.. currentmodule:: pyrtc.image_reconstructor


Torch Image Reconstructor
=========================

``TorchImageReconstructor`` turns wavefront-sensor images into the loop's
signal with a PyTorch model. Use it for image-based reconstructors, such as a
neural network that maps Shack-Hartmann or pyramid pixels to modes, or a
focal-plane wavefront sensor whose "WFS" is a science-camera image.

It goes in the ``slopes`` section in place of ``SlopesProcess``. It reads the
``wfs`` image stream and writes the model output to ``signal``, stamped with
each image's frame id, so the loop, its calibration methods, telemetry and
``manager.latency()`` work unchanged.

Configuration
-------------

.. code-block:: yaml

  slopes:
    class_name: TorchImageReconstructor
    signal_size: 120                  # number of model outputs
    model_file: calib/reconstructor.pt2
    device: cuda:0                    # cpu (default), cuda, cuda:N
    dtype: float32                    # or float16 (CUDA only)
    flux_normalization: sum           # none (default), sum or mean
    sqrt_stretch: false
    output_scale_file: ""             # optional .npy, signal_size factors
    cuda_graph: true
    input_streams: {wfs: wfs}
    output_streams: {signal: signal}
    functions: [compute_signal]

``signal_size`` is required. At startup the model runs once on a blank image
and the reconstructor raises ``ValueError`` if its output does not have
``signal_size`` elements, or if it fails on the input shape. Every key is
listed in the parameter reference below.

Model sources
-------------

Set exactly one of:

``model_file``
  An exported program (``.pt2``, from ``torch.export.save``) or a TorchScript
  file (any other suffix, from ``torch.jit.save``). Export in eval mode;
  exported modules keep the mode they were exported in. TorchScript is
  deprecated in recent PyTorch releases, so prefer ``.pt2`` for new models.

``model_factory``
  A Python callable returning an ``nn.Module``, as ``"package.module:function"``.
  With ``model_factory_file`` (a ``.py`` file, resolved relative to the config
  like ``class_file``), ``model_factory`` is just the function's name.
  ``model_kwargs`` are passed to it, and ``state_dict_file`` (a ``torch.save``
  of a state dict) is then loaded strictly.

.. code-block:: yaml

  slopes:
    class_name: TorchImageReconstructor
    signal_size: 120
    model_factory: build_cnn
    model_factory_file: models.py
    model_kwargs: {num_outputs: 120}
    state_dict_file: calib/cnn_weights.pt
    functions: [compute_signal]

In a soft-RTC session a model can also be passed directly, or swapped on a
running system (the new model is validated, warmed up and captured before it
replaces the old one between frames):

.. code-block:: python

  reconstructor = TorchImageReconstructor(conf, model=my_module)
  reconstructor.set_model(retrained_module)

The model is moved to ``device`` and ``dtype`` in place.

Preprocessing
-------------

The WFS component has already subtracted the dark (and background). On the
model's device, each frame then goes through, in order:

1. conversion to float32;
2. ``flux_normalization``: ``sum`` divides by the total flux (the image sums
   to 1), ``mean`` by the mean pixel. A frame whose flux is at most ``1e-12``
   is dark and gives an all-zero signal instead of the model's response to
   noise;
3. ``sqrt_stretch``: negative pixels are clipped to 0 and the square root is
   taken;
4. conversion to ``dtype`` and reshape to ``input_shape`` (default
   ``[1, 1, *image_shape]``: one single-channel image; set ``[1, N]`` for an
   MLP on the flattened image);
5. the model; its output is flattened and converted to float32;
6. ``output_scale_file``: element-wise multiplication by ``signal_size``
   factors (for example the inverse of per-mode training normalisation).

The image is the stream array as stored, ``(width, height)`` in pyrtc's
convention; train on frames read from the ``wfs`` stream to match.

Plugging into the loop
----------------------

The loop sizes its interaction and control matrices from the ``signal``
stream, so nothing changes in the ``loop`` section. Two common set-ups:

- **The model outputs modal coefficients** (``signal_size`` equals the number
  of controlled modes, in the corrector's modal basis and units). Load an
  identity interaction matrix, so the control matrix is the identity too:

  .. code-block:: python

    np.save("calib/identity_im.npy", np.eye(num_modes, dtype=np.float32))

  .. code-block:: yaml

    loop:
      im_file: calib/identity_im.npy
      gain: 0.3
      functions: [standard_integrator]

  The sign convention is the loop's: it subtracts ``gain * CM @ signal``, so
  for a corrector command ``c`` applied on a flat wavefront the model should
  output ``c`` (the response an identity interaction matrix describes).
- **The model outputs any other signal**, such as a feature vector. Calibrate
  the interaction matrix through the live pipeline (``loop.compute_im()``)
  exactly as with slopes.

No ``signal_2d`` stream is created unless ``signal_2d_shape`` is set (it must
hold ``signal_size`` elements).

Real-time path and GPU use
--------------------------

:class:`TorchModelRunner` holds the model and the per-frame path. On a CUDA
device:

- frames are read from the ``wfs`` stream straight into a pinned host buffer,
  copied to a static device buffer on the runner's own CUDA stream, and the
  output is copied back into a pinned buffer before one synchronisation;
- after ``warmup_iters`` forward passes, preprocessing, the model and the
  output scaling are captured in one CUDA graph (``cuda_graph: true``). The
  graph is checked against the eager result on a random frame. A model that
  cannot be captured (one that synchronises with the host, e.g. with
  ``.item()``, or has data-dependent control flow) falls back to eager
  execution with a warning; ``reconstructor.runner.graph_active`` says which
  path runs;
- with the common ``gpu_device`` key set as well, a GPU-backed ``wfs`` stream
  is attached on the device (no host round trip) and the ``signal`` stream is
  created GPU-backed with a CPU mirror, as ``SlopesProcess`` does.

On the CPU the model runs eagerly with ``torch.inference_mode``.
``cpu_threads`` sets ``torch.set_num_threads``, which is process-wide. Hard-RTC
children start with ``OMP_NUM_THREADS=1`` (unless you set it), so a CPU model
there runs on one thread unless ``cpu_threads`` asks for more.

Timing
------

``last_compute_time`` is the time in seconds the last frame took from the end
of the ``wfs`` read to the output being on the host (preprocessing, transfers,
model). ``timing_stats()`` returns the count, mean, median, p99 and max over the
last ``timing_window`` frames, and ``reset_timing()`` clears them. In hard-RTC
mode, call them through the launcher (``launcher.run("timing_stats")``).
Stream handoffs come on top; ``manager.latency()`` measures the whole
WFS-to-DM path.

``benchmarks/image_reconstructor_bench.py`` times the runner for a 64x64
image and 120 outputs with a ~15M-parameter CNN and a ~1.1M-parameter MLP.
Results vary by host and GPU load; on an RTX 4060 with a Neoverse-N1 host
(PyTorch 2.11, three runs, microseconds):

.. list-table::
  :header-rows: 1

  * - Model
    - Path
    - Median
    - p99
  * - CNN, 14.8M
    - CPU, float32, 16 threads
    - 4720-5080
    - 5250-7790
  * - CNN, 14.8M
    - eager, float32
    - 1210-1440
    - 1780-2720
  * - CNN, 14.8M
    - CUDA graph, float32
    - 580-750
    - 1150-2700
  * - CNN, 14.8M
    - CUDA graph, float16
    - 380-560
    - 1100-2550
  * - MLP, 1.1M
    - eager, float32
    - 610-640
    - 920-2260
  * - MLP, 1.1M
    - CUDA graph, float32
    - 90-180
    - 360-510

Eager execution is bound by kernel-launch overhead for models this size, so
the graph roughly halves the CNN's latency and cuts the MLP's by 3-6x.

Parameters
----------

.. autoclass:: TorchImageReconstructor
  :members:
  :show-inheritance:
  :no-index:

.. autoclass:: TorchModelRunner
  :members:
  :no-index:

.. autofunction:: load_torch_model
  :no-index:
