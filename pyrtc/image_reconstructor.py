"""Turn wavefront-sensor images into the loop's signal with a PyTorch model.

:class:`TorchImageReconstructor` is an image-based reconstructor: it reads the
``wfs`` image stream and publishes the output of a user-supplied PyTorch
model as the ``signal`` stream. It goes in the ``slopes`` section in place of
:class:`~pyrtc.slopes_process.SlopesProcess`, so the loop, the calibration
methods and the rest of the pipeline work unchanged. Typical uses are neural
reconstructors and focal-plane wavefront sensing, where a network maps pixels
straight to modal coefficients (run the loop with an identity interaction
matrix) or to any other signal the loop calibrates as usual.

Config (``slopes`` section)::

    slopes:
      class_name: TorchImageReconstructor
      signal_size: 120                 # elements of the model output
      model_file: calib/reconstructor.pt2  # torch.export (.pt2) or TorchScript
      # or a Python factory returning an nn.Module, plus optional weights:
      # model_factory: my_models:build_cnn   # "module:function"
      # model_factory_file: models.py        # then model_factory: build_cnn
      # model_kwargs: {num_outputs: 120}
      # state_dict_file: calib/weights.pt
      device: cuda:0                   # cpu (default), cuda, cuda:N
      dtype: float32                   # or float16 (CUDA only)
      cuda_graph: true                 # capture the model; eager fallback
      flux_normalization: sum          # none (default), sum or mean
      sqrt_stretch: false
      output_scale_file: ""            # .npy with signal_size factors
      functions: [compute_signal]

The work is split in two: :class:`TorchModelRunner` owns the model and the
real-time path (preprocessing, pinned host buffers, a dedicated CUDA stream,
CUDA-graph capture) and has no streams, so it can be benchmarked and tested
on its own; the component adds the stream wiring and timing.

torch is imported only when a reconstructor or runner is built, so
``import pyrtc`` stays torch-free.
"""

from __future__ import annotations

import importlib
import threading
import time
from collections import deque
from typing import Any, Callable, Mapping

import numpy as np

from pyrtc.component import Component
from pyrtc.logging_utils import get_logger
from pyrtc.manager import launch_component
from pyrtc.streams import create_stream, open_stream
from pyrtc.utils import require_optional

logger = get_logger(__name__)

#: Images whose normalisation flux is at or below this are treated as dark:
#: the reconstructor publishes zeros instead of the model's response to noise.
FLUX_EPS = 1e-12

FLUX_NORMALIZATIONS = ("none", "sum", "mean")
MODEL_DTYPES = ("float32", "float16")
DEFAULT_TIMING_WINDOW = 1000


def _torch(feature: str = "TorchImageReconstructor"):
    return require_optional("torch", "torch", feature)


def _parse_device(torch, device: Any):
    try:
        parsed = torch.device(str(device))
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"device must be 'cpu', 'cuda' or 'cuda:N', got {device!r}") from exc
    if parsed.type not in ("cpu", "cuda"):
        raise ValueError(f"device must be 'cpu', 'cuda' or 'cuda:N', got {device!r}")
    if parsed.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(f"device {device!r} requested but CUDA is not available")
        if parsed.index is None:
            parsed = torch.device("cuda", torch.cuda.current_device())
    return parsed


def _host_torch_dtype(torch, dtype: np.dtype):
    """Return the torch dtype matching ``dtype``, or ``None`` if torch lacks it."""

    try:
        return torch.from_numpy(np.empty(0, dtype=dtype)).dtype
    except TypeError:
        return None


class TorchModelRunner:
    """Run a PyTorch model on single WFS images, with optional CUDA graphs.

    The runner has no streams: :meth:`run` takes one image (NumPy array or
    torch tensor of ``image_shape``) and returns the model output as a flat
    float32 vector of ``signal_size`` elements. Every call goes through the
    same path:

    1. On CUDA, a NumPy image is copied into a pinned host buffer (or
       written there directly by the caller through :attr:`host_input`), then
       copied to a static device buffer on the runner's own CUDA stream.
    2. :meth:`forward` runs on the device buffer: conversion to float32,
       optional flux normalisation and square-root stretch, conversion to the
       model dtype, reshape to ``input_shape``, the model, conversion back to
       float32 and the optional per-element output scale. With ``cuda_graph``
       this whole step is one captured CUDA graph.
    3. The result is copied into a pinned host buffer and the stream is
       synchronised.

    Parameters
    ----------
    model : torch.nn.Module or torch.jit.ScriptModule
        The model. It is moved to ``device`` and ``dtype`` and put in eval
        mode. It must return one tensor with ``signal_size`` elements.
    image_shape : tuple of int
        Shape of the WFS image as stored in the stream.
    image_dtype : numpy dtype
        Dtype of the WFS image.
    signal_size : int
        Number of model outputs (the ``signal`` stream length).
    device : str
        ``"cpu"``, ``"cuda"`` or ``"cuda:N"``.
    dtype : {"float32", "float16"}
        Model precision. ``float16`` needs a CUDA device. Outputs are always
        float32.
    input_shape : tuple of int, optional
        Shape the model takes. Default ``(1, 1, *image_shape)``: one image,
        one channel. Must hold as many elements as the image.
    flux_normalization : {"none", "sum", "mean"}
        Divide the image by its total (``sum``) or mean pixel (``mean``)
        flux. A frame whose flux is at or below :data:`FLUX_EPS` gives an
        all-zero output.
    sqrt_stretch : bool
        Clip negative pixels to 0 and take the square root (after
        normalisation).
    output_scale : array_like, optional
        ``signal_size`` factors multiplied into the output.
    cuda_graph : bool
        Capture :meth:`forward` in a CUDA graph on CUDA devices. If capture
        fails, or the graph's output differs from the eager output, the
        runner logs a warning and stays eager (see :attr:`graph_active`).
    warmup_iters : int
        Forward passes run before capture (and before timing).
    """

    def __init__(
        self,
        model,
        *,
        image_shape: tuple[int, ...],
        image_dtype=np.float32,
        signal_size: int,
        device: str = "cpu",
        dtype: str = "float32",
        input_shape: tuple[int, ...] | None = None,
        flux_normalization: str = "none",
        sqrt_stretch: bool = False,
        output_scale=None,
        cuda_graph: bool = True,
        warmup_iters: int = 10,
    ) -> None:
        torch = _torch("TorchModelRunner")
        self._torch = torch

        self.image_shape = tuple(int(axis) for axis in image_shape)
        self.image_dtype = np.dtype(image_dtype)
        self.signal_size = int(signal_size)
        if self.signal_size < 1:
            raise ValueError(f"signal_size must be >= 1, got {signal_size}")
        self.device = _parse_device(torch, device)
        self.is_cuda = self.device.type == "cuda"

        dtype = str(dtype).lower()
        if dtype not in MODEL_DTYPES:
            raise ValueError(f"dtype must be one of {MODEL_DTYPES}, got {dtype!r}")
        if dtype == "float16" and not self.is_cuda:
            raise ValueError("dtype 'float16' needs a CUDA device; use float32 on the CPU")
        self.dtype = dtype
        self.model_dtype = getattr(torch, dtype)

        image_size = int(np.prod(self.image_shape))
        if input_shape is None:
            input_shape = (1, 1, *self.image_shape)
        self.input_shape = tuple(int(axis) for axis in input_shape)
        if int(np.prod(self.input_shape)) != image_size:
            raise ValueError(
                f"input_shape {self.input_shape} holds {int(np.prod(self.input_shape))} "
                f"elements but the WFS image {self.image_shape} has {image_size}"
            )

        normalization = str(flux_normalization).lower()
        if normalization not in FLUX_NORMALIZATIONS:
            raise ValueError(
                f"flux_normalization must be one of {FLUX_NORMALIZATIONS}, "
                f"got {flux_normalization!r}"
            )
        self.flux_normalization = normalization
        self.sqrt_stretch = bool(sqrt_stretch)
        self.warmup_iters = max(0, int(warmup_iters))
        self.cuda_graph_requested = bool(cuda_graph)

        self.output_scale = None
        if output_scale is not None:
            scale = np.asarray(output_scale, dtype=np.float32).ravel()
            if scale.size != self.signal_size:
                raise ValueError(
                    f"output scale has {scale.size} elements but signal_size is {self.signal_size}"
                )
            self.output_scale = torch.as_tensor(scale, device=self.device)

        self.model = self._prepare_model(model)

        # Host-side input: the image dtype when torch supports it (no cast on
        # the host), float32 otherwise (e.g. uint16 frames).
        host_dtype = _host_torch_dtype(torch, self.image_dtype)
        self._host_cast = host_dtype is None
        if host_dtype is None:
            host_dtype = torch.float32
        pin = self.is_cuda
        self._host_in = torch.empty(self.image_shape, dtype=host_dtype, pin_memory=pin)
        self._host_out = torch.empty(self.signal_size, dtype=torch.float32, pin_memory=pin)
        #: NumPy view of the (pinned) host input buffer. Reading a stream
        #: straight into it (``read_stream(..., out=runner.host_input)``)
        #: saves a copy; :meth:`run` notices and skips its own.
        self.host_input = self._host_in.numpy()
        self._host_out_np = self._host_out.numpy()

        if self.is_cuda:
            self._stream = torch.cuda.Stream(device=self.device)
            self._static_in = torch.zeros(self.image_shape, dtype=host_dtype, device=self.device)
        else:
            self._stream = None
            self._static_in = self._host_in
        self._static_out = None
        self._graph = None
        self.graph_active = False

        self._validate_output()
        self._warm_up()
        if self.is_cuda and self.cuda_graph_requested:
            self._capture_graph()

    # -- setup ---------------------------------------------------------------

    def _prepare_model(self, model):
        torch = self._torch
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                f"model must be a torch.nn.Module (or TorchScript module), got {type(model).__name__}"
            )
        model = model.to(device=self.device, dtype=self.model_dtype)
        try:
            model.eval()
        except NotImplementedError:
            # torch.export modules refuse eval(); they keep the mode they
            # were exported in.
            pass
        return model

    def forward(self, raw):
        """Preprocess ``raw`` (an image tensor on the device) and run the model.

        Returns a flat float32 tensor of ``signal_size`` elements. Everything
        here stays on the device without host synchronisation, so it can be
        captured in a CUDA graph.
        """

        torch = self._torch
        x = raw.to(torch.float32)
        dark = None
        if self.flux_normalization != "none":
            flux = x.sum() if self.flux_normalization == "sum" else x.mean()
            dark = flux.abs() <= FLUX_EPS
            x = x / torch.where(dark, torch.ones_like(flux), flux)
        if self.sqrt_stretch:
            x = x.clamp_min(0.0).sqrt()
        x = x.to(self.model_dtype).reshape(self.input_shape)
        y = self.model(x)
        if not isinstance(y, torch.Tensor):
            raise TypeError(f"the model must return one tensor, got {type(y).__name__}")
        y = y.reshape(-1).to(torch.float32)
        if self.output_scale is not None:
            y = y * self.output_scale
        if dark is not None:
            y = torch.where(dark, torch.zeros_like(y), y)
        return y

    def _forward_checked(self, raw):
        with self._torch.inference_mode():
            return self.forward(raw)

    def _validate_output(self) -> None:
        torch = self._torch
        try:
            with torch.inference_mode():
                x = self._static_in.to(torch.float32).reshape(self.input_shape)
                y = self.model(x.to(self.model_dtype))
        except Exception as exc:
            raise ValueError(
                f"the model failed on an input of shape {self.input_shape} "
                f"(dtype {self.dtype}): {exc}"
            ) from exc
        if not isinstance(y, torch.Tensor):
            raise TypeError(f"the model must return one tensor, got {type(y).__name__}")
        if y.numel() != self.signal_size:
            raise ValueError(
                f"model output has {y.numel()} elements (shape {tuple(y.shape)}) "
                f"but signal_size is {self.signal_size}"
            )

    def _warm_up(self) -> None:
        if self.warmup_iters == 0:
            return
        torch = self._torch
        if self.is_cuda:
            # Warm up on the side stream, as torch.cuda.graphs requires, so
            # lazy initialisation (cuDNN autotuning, allocator) is done first.
            self._stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(self._stream):
                for _ in range(self.warmup_iters):
                    self._forward_checked(self._static_in)
            self._stream.synchronize()
        else:
            for _ in range(self.warmup_iters):
                self._forward_checked(self._static_in)

    def _capture_graph(self) -> None:
        torch = self._torch
        try:
            graph = torch.cuda.CUDAGraph()
            with torch.inference_mode():
                # thread_local: other threads (soft-RTC components) may use
                # CUDA while this one captures.
                with torch.cuda.graph(
                    graph, stream=self._stream, capture_error_mode="thread_local"
                ):
                    static_out = self.forward(self._static_in)
            # A graph freezes host-side control flow; check it reproduces the
            # eager result on a non-trivial input before trusting it.
            probe = torch.rand(self.image_shape, device=self.device) * 100.0
            self._stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(self._stream):
                self._static_in.copy_(probe.to(self._static_in.dtype))
                graph.replay()
                eager = self._forward_checked(self._static_in)
            self._stream.synchronize()
            tolerance = 1e-2 if self.dtype == "float16" else 1e-4
            if not torch.allclose(static_out, eager, rtol=tolerance, atol=tolerance):
                raise RuntimeError("the captured graph's output differs from the eager output")
            self._static_in.zero_()
        except Exception as exc:
            logger.warning("CUDA graph capture failed (%s); running the model eagerly", exc)
            self._graph = None
            self._static_out = None
            self.graph_active = False
            return
        self._graph = graph
        self._static_out = static_out
        self.graph_active = True

    # -- real-time path --------------------------------------------------------

    def _stage_host(self, image) -> None:
        """Copy a NumPy image into the host input buffer, unless it is already there."""

        if isinstance(image, np.ndarray) and image is self.host_input:
            return
        array = np.asarray(image)
        if array.shape != self.image_shape:
            array = array.reshape(self.image_shape)
        np.copyto(self.host_input, array, casting="unsafe")

    def run_device(self, image):
        """Process one image and return the output tensor on the model device.

        The returned tensor is the runner's static output buffer when a CUDA
        graph is active: it is overwritten by the next call. On CUDA the
        work is queued on the runner's stream and *not* synchronised; use
        :meth:`run` for a host result.
        """

        torch = self._torch
        is_tensor = isinstance(image, torch.Tensor)
        if not self.is_cuda:
            if is_tensor:
                self._host_in.copy_(image.reshape(self.image_shape))
            else:
                self._stage_host(image)
            return self._forward_checked(self._host_in)

        with torch.cuda.stream(self._stream):
            if is_tensor:
                if image.is_cuda and image.device != self.device:
                    self._stream.wait_stream(torch.cuda.current_stream(image.device))
                self._static_in.copy_(image.reshape(self.image_shape), non_blocking=True)
            else:
                self._stage_host(image)
                self._static_in.copy_(self._host_in, non_blocking=True)
            if self._graph is not None:
                self._graph.replay()
                return self._static_out
            return self._forward_checked(self._static_in)

    def run(self, image) -> np.ndarray:
        """Process one image and return the output as a float32 NumPy vector.

        The returned array is the runner's (pinned) host output buffer and
        is overwritten by the next call; copy it to keep it.
        """

        output = self.run_device(image)
        if not self.is_cuda:
            np.copyto(self._host_out_np, output.numpy())
            return self._host_out_np
        torch = self._torch
        with torch.cuda.stream(self._stream):
            self._host_out.copy_(output, non_blocking=True)
        self._stream.synchronize()
        return self._host_out_np

    def synchronize(self) -> None:
        """Wait for the work queued on the runner's CUDA stream (no-op on CPU)."""

        if self._stream is not None:
            self._stream.synchronize()


def _import_factory(spec: str, factory_file: str | None) -> Callable[..., Any]:
    """Resolve ``model_factory`` (``module:function``, or a name in ``factory_file``)."""

    if factory_file:
        from pyrtc.component_loading import import_symbol_from_file

        attr = spec.rsplit(":", 1)[-1]
        return import_symbol_from_file(factory_file, attr)
    if ":" in spec:
        module_name, attr = spec.split(":", 1)
    elif "." in spec:
        module_name, attr = spec.rsplit(".", 1)
    else:
        raise ValueError(
            f"model_factory {spec!r} must be 'module:function' (or set model_factory_file)"
        )
    module = importlib.import_module(module_name)
    factory = module
    for part in attr.split("."):
        factory = getattr(factory, part)
    return factory


def load_torch_model(
    *,
    model_file: str = "",
    model_factory: str = "",
    model_factory_file: str = "",
    model_kwargs: Mapping[str, Any] | None = None,
    state_dict_file: str = "",
):
    """Build the model a reconstructor config describes (on the CPU).

    Exactly one source must be given: a ``model_file``, either an exported
    program (``.pt2``, ``torch.export.save``; export it in eval mode) or a
    TorchScript file (any other suffix, ``torch.jit.load``), or a
    ``model_factory`` called with ``model_kwargs`` that returns an
    ``nn.Module``. ``state_dict_file`` (a ``torch.save`` of
    a state dict, loaded with ``weights_only=True``) is then loaded into it
    strictly.
    """

    torch = _torch()
    model_file = str(model_file or "")
    model_factory = str(model_factory or "")
    if bool(model_file) == bool(model_factory):
        raise ValueError("set exactly one of model_file or model_factory")
    if model_file.endswith(".pt2"):
        model = torch.export.load(model_file).module()
    elif model_file:
        model = torch.jit.load(model_file, map_location="cpu")
    else:
        factory = _import_factory(model_factory, model_factory_file or None)
        model = factory(**dict(model_kwargs or {}))
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                f"model_factory {model_factory!r} returned {type(model).__name__}, "
                "not a torch.nn.Module"
            )
    if state_dict_file:
        state = torch.load(state_dict_file, map_location="cpu", weights_only=True)
        if isinstance(state, Mapping) and "state_dict" in state and len(state) == 1:
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)
    return model


class TorchImageReconstructor(Component):
    """Publish a PyTorch model's output on each WFS image as the loop's signal.

    Sits in the ``slopes`` section in place of ``SlopesProcess``: it reads
    the ``wfs`` image stream and writes ``signal`` (``signal_size`` float32
    values), stamped with each image's frame id. When ``signal_2d_shape`` is
    set it also writes the output reshaped to that shape to ``signal_2d``
    for viewers.

    Config
    ------
    signal_size : int
        Number of model outputs. The model is checked against it at startup.
    model_file : str
        Exported program (``.pt2``, ``torch.export.save``) or TorchScript
        file (``torch.jit.save``, any other suffix). Either this or
        ``model_factory``.
    model_factory : str
        ``"module:function"`` returning an ``nn.Module``; with
        ``model_factory_file``, the function's name in that file.
    model_factory_file : str
        Python file defining ``model_factory`` (resolved like ``class_file``).
    model_kwargs : dict
        Keyword arguments for ``model_factory``.
    state_dict_file : str
        State dict loaded (strictly) into the model.
    device : str
        Model device: ``"cpu"`` (default), ``"cuda"`` or ``"cuda:N"``.
    dtype : str
        ``"float32"`` (default) or ``"float16"`` (CUDA only).
    input_shape : list of int
        Shape the model takes; default ``[1, 1, *image_shape]``.
    flux_normalization : str
        ``"none"`` (default), ``"sum"`` or ``"mean"``.
    sqrt_stretch : bool
        Square-root stretch (negative pixels clipped to 0). Default False.
    output_scale_file : str
        ``.npy`` file of ``signal_size`` per-element output factors.
    signal_2d_shape : list of int
        Shape of an optional ``signal_2d`` display stream.
    cuda_graph : bool
        Capture the model in a CUDA graph (CUDA only). Default True.
    warmup_iters : int
        Forward passes before capture. Default 10.
    cpu_threads : int
        When set, ``torch.set_num_threads`` (process-wide) for CPU models.
    timing_window : int
        Number of recent per-frame compute times kept for
        :meth:`timing_stats`. Default 1000.

    The common ``gpu_device`` key keeps its usual meaning: it attaches the
    ``wfs`` input on the GPU (when that stream is GPU-backed) and creates
    GPU-backed output streams. ``device`` is where the model runs.

    Attributes
    ----------
    runner : TorchModelRunner
        The model and its real-time path.
    last_compute_time : float or None
        Seconds spent on the last frame, from the end of the ``wfs`` read to
        the output being on the host (preprocessing, transfers, model).
    frames_processed : int
        Frames published since construction.
    """

    def __init__(self, conf, model=None) -> None:
        """Build the reconstructor.

        ``model`` (an ``nn.Module``) overrides the config's model source,
        for soft-RTC sessions that build the network in Python.
        """

        settings = self._parse_settings(conf, model_given=model is not None)
        torch = _torch()
        super().__init__(conf)
        try:
            self.conf = conf
            self.name = "TorchImageReconstructor"
            self._torch = torch
            self._lock = threading.RLock()
            self.__dict__.update(settings)
            if self.cpu_threads is not None:
                torch.set_num_threads(self.cpu_threads)

            self.wfs_shm = open_stream(self.input_stream_name("wfs"), gpu_device=self.gpu_device)
            self.image_shape = tuple(int(axis) for axis in self.wfs_shm.shape)
            self.image_dtype = np.dtype(self.wfs_shm.dtype)
            self.register_input_stream("wfs", self.wfs_shm)

            self.output_scale = None
            if self.output_scale_file:
                self.output_scale = np.load(self.output_scale_file).astype(np.float32).ravel()

            self.last_compute_time = None
            self.frames_processed = 0
            self._compute_times = deque(maxlen=self.timing_window)
            self.runner = None
            self.set_model(model if model is not None else self.load_model())

            self.signal_shape = (self.signal_size,)
            self.signal = create_stream(
                self.output_stream_name("signal"),
                self.signal_shape,
                np.float32,
                gpu_device=self.gpu_device,
            )
            self.register_output_stream("signal", self.signal)
            self.signal_2d = None
            if self.signal_2d_shape is not None:
                self.signal_2d = create_stream(
                    self.output_stream_name("signal_2d"),
                    self.signal_2d_shape,
                    np.float32,
                    gpu_device=self.gpu_device,
                )
                self.register_output_stream("signal_2d", self.signal_2d)
            self.logger.info(
                "Initialized torch reconstructor image_shape=%s signal_size=%s device=%s "
                "dtype=%s cuda_graph=%s",
                self.image_shape,
                self.signal_size,
                self.runner.device,
                self.dtype,
                self.runner.graph_active,
            )
        except Exception:
            self.logger.exception("Failed to initialize torch image reconstructor")
            self.close()
            raise

    # -- configuration ---------------------------------------------------------

    @staticmethod
    def _parse_settings(conf, *, model_given: bool) -> dict[str, Any]:
        """Validate the config without touching torch or streams."""

        def get(key, default):
            value = conf.get(key, default)
            return default if value is None else value

        signal_size = conf.get("signal_size")
        if isinstance(signal_size, bool) or not isinstance(signal_size, (int, np.integer)):
            raise ValueError(f"slopes: 'signal_size' must be an integer, got {signal_size!r}")
        if signal_size < 1:
            raise ValueError(f"slopes: 'signal_size' must be >= 1, got {signal_size}")

        model_file = str(get("model_file", ""))
        model_factory = str(get("model_factory", ""))
        if not model_given and bool(model_file) == bool(model_factory):
            raise ValueError("slopes: set exactly one of 'model_file' or 'model_factory'")
        model_kwargs = get("model_kwargs", {})
        if not isinstance(model_kwargs, Mapping):
            raise ValueError("slopes: 'model_kwargs' must be a mapping")

        dtype = str(get("dtype", "float32")).lower()
        if dtype not in MODEL_DTYPES:
            raise ValueError(f"slopes: 'dtype' must be one of {MODEL_DTYPES}, got {dtype!r}")
        normalization = str(get("flux_normalization", "none")).lower()
        if normalization not in FLUX_NORMALIZATIONS:
            raise ValueError(
                f"slopes: 'flux_normalization' must be one of {FLUX_NORMALIZATIONS}, "
                f"got {normalization!r}"
            )

        def shape(key):
            value = conf.get(key)
            if value is None:
                return None
            if not isinstance(value, (list, tuple)) or not all(
                isinstance(axis, (int, np.integer)) and axis >= 1 for axis in value
            ):
                raise ValueError(f"slopes: '{key}' must be a list of positive integers")
            return tuple(int(axis) for axis in value)

        signal_2d_shape = shape("signal_2d_shape")
        if signal_2d_shape is not None and int(np.prod(signal_2d_shape)) != signal_size:
            raise ValueError(
                f"slopes: 'signal_2d_shape' {list(signal_2d_shape)} must hold "
                f"signal_size={signal_size} elements"
            )
        cpu_threads = conf.get("cpu_threads")
        if cpu_threads is not None and (not isinstance(cpu_threads, int) or cpu_threads < 1):
            raise ValueError(
                f"slopes: 'cpu_threads' must be a positive integer, got {cpu_threads!r}"
            )

        return {
            "signal_size": int(signal_size),
            "model_file": model_file,
            "model_factory": model_factory,
            "model_factory_file": str(get("model_factory_file", "")),
            "model_kwargs": dict(model_kwargs),
            "state_dict_file": str(get("state_dict_file", "")),
            "device": str(get("device", "cpu")),
            "dtype": dtype,
            "input_shape": shape("input_shape"),
            "flux_normalization": normalization,
            "sqrt_stretch": bool(get("sqrt_stretch", False)),
            "output_scale_file": str(get("output_scale_file", "")),
            "signal_2d_shape": signal_2d_shape,
            "cuda_graph": bool(get("cuda_graph", True)),
            "warmup_iters": int(get("warmup_iters", 10)),
            "cpu_threads": cpu_threads,
            "timing_window": max(1, int(get("timing_window", DEFAULT_TIMING_WINDOW))),
        }

    def load_model(self):
        """Build the model from the config's ``model_file`` / ``model_factory``."""

        return load_torch_model(
            model_file=self.model_file,
            model_factory=self.model_factory,
            model_factory_file=self.model_factory_file,
            model_kwargs=self.model_kwargs,
            state_dict_file=self.state_dict_file,
        )

    def set_model(self, model) -> None:
        """Install ``model``: validate its output size, warm it up and capture it.

        Safe on a running instance: the new runner is built first and
        swapped in between frames. Raises ``ValueError`` when the model's
        output does not have ``signal_size`` elements.
        """

        runner = TorchModelRunner(
            model,
            image_shape=self.image_shape,
            image_dtype=self.image_dtype,
            signal_size=self.signal_size,
            device=self.device,
            dtype=self.dtype,
            input_shape=self.input_shape,
            flux_normalization=self.flux_normalization,
            sqrt_stretch=self.sqrt_stretch,
            output_scale=self.output_scale,
            cuda_graph=self.cuda_graph,
            warmup_iters=self.warmup_iters,
        )
        with self._lock:
            self.runner = runner
            # The CPU path reads frames straight into the runner's buffer.
            self._image_buffer = runner.host_input if not runner._host_cast else None
        self.logger.info(
            "Installed model on %s (%s, cuda_graph=%s)",
            runner.device,
            runner.dtype,
            runner.graph_active,
        )

    # -- real-time path --------------------------------------------------------

    def compute_signal(self):
        """Read one WFS image, run the model and publish ``signal``."""

        image = self.read_stream("wfs", out=self._image_buffer)
        with self._lock:
            start = time.perf_counter()
            output = self._process(image)
            self.last_compute_time = time.perf_counter() - start
            self._compute_times.append(self.last_compute_time)
            self.write_stream("signal", output)
            if self.signal_2d is not None:
                self.write_stream("signal_2d", output.reshape(self.signal_2d_shape))
            self.frames_processed += 1

    def _process(self, image):
        """Return the model output as a NumPy vector, or a device tensor for GPU streams."""

        runner = self.runner
        if getattr(self.signal, "gpu_device", None) is not None and runner.is_cuda:
            output = runner.run_device(image)
            runner.synchronize()
            return output
        return runner.run(image)

    def read(self, block=True):
        """Read the current signal."""

        return self.read_stream("signal", block=block)

    def read_image(self, block=True):
        """Read the current WFS image."""

        return self.read_stream("wfs", block=block)

    def timing_stats(self) -> dict[str, float]:
        """Return per-frame compute-time statistics (seconds) over the recent window."""

        times = np.asarray(self._compute_times, dtype=np.float64)
        if times.size == 0:
            return {"count": 0}
        return {
            "count": int(times.size),
            "mean": float(times.mean()),
            "median": float(np.median(times)),
            "p99": float(np.percentile(times, 99)),
            "max": float(times.max()),
        }

    def reset_timing(self) -> None:
        """Forget the recorded compute times."""

        self._compute_times.clear()
        self.last_compute_time = None


if __name__ == "__main__":
    launch_component(TorchImageReconstructor, "slopes", start=True)
