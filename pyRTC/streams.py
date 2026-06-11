"""pyshmem-backed stream policy for pyRTC.

Shared-memory transport itself is provided by the ``pyshmem`` package; this
module holds the pyRTC-side policy for creating and attaching to streams,
plus the config-driven planning of which output streams a system implies.
"""

from __future__ import annotations

import logging

import numpy as np
import pyshmem

from pyRTC.config_runtime import sync_runtime_config
from pyRTC.logging_utils import get_logger

logger = get_logger(__name__)

TORCH_AVAILABLE = False
torch = None

try:
    import torch  # noqa: F401  (availability probe for normalize_gpu_device)
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


def gpu_torch_available() -> bool:
    return TORCH_AVAILABLE


def normalize_gpu_device(gpuDevice, context: str = ""):
    if gpuDevice is None:
        return None
    if not TORCH_AVAILABLE:
        prefix = f"{context}: " if context else ""
        logging.log(
            level=logging.WARNING,
            msg=f"{prefix}gpuDevice was requested but PyTorch is not installed; defaulting to CPU mode.",
        )
        return None
    return gpuDevice



def create_stream(name, shape, dtype, gpuDevice=None):
    """Create the pyshmem stream backing a component output.

    An existing CPU stream is reused when its shape and dtype already match,
    so attached readers (viewers, telemetry) keep working across component
    restarts; on any mismatch the stream is rebuilt. GPU-backed streams are
    always rebuilt because a previous producer's CUDA tensor cannot be
    re-exported. GPU streams are created with ``cpu_mirror=True`` so CPU-only
    processes can always read them.
    """
    shape = tuple(int(axis) for axis in shape)
    dtype = np.dtype(dtype)
    gpu_device = normalize_gpu_device(gpuDevice, name)
    if gpu_device is not None and not pyshmem.gpu_available():
        logger.warning(
            "%s: gpuDevice %s requested but CUDA is not available; using a CPU stream",
            name,
            gpu_device,
        )
        gpu_device = None
    if gpu_device is not None and dtype not in pyshmem.GPU_SUPPORTED_DTYPES:
        logger.warning(
            "%s: dtype %s is not supported for GPU SHM; using a CPU stream",
            name,
            dtype,
        )
        gpu_device = None

    create_kwargs = {}
    if gpu_device is not None:
        create_kwargs = {"gpu_device": gpu_device, "cpu_mirror": True}

    try:
        return pyshmem.create(name, shape=shape, dtype=dtype, **create_kwargs)
    except FileExistsError:
        pass

    if gpu_device is None:
        existing = None
        try:
            existing = pyshmem.open(name, gpu_device=False)
        except Exception:
            logger.debug("Failed to reopen existing stream %s", name, exc_info=True)
        if existing is not None:
            matches = (
                not existing.gpu_enabled
                and tuple(existing.shape) == shape
                and existing.dtype == dtype
            )
            if matches:
                logger.debug("Reusing existing stream %s", name)
                return existing
            existing.close()

    logger.debug("Rebuilding stream %s", name)
    pyshmem.unlink_quiet(name)
    return pyshmem.create(name, shape=shape, dtype=dtype, **create_kwargs)


def open_stream(name, gpuDevice=None):
    """Attach to an existing pyshmem stream.

    Without ``gpuDevice`` the stream is opened CPU-side: GPU-backed streams
    are read through their CPU mirror and reads return NumPy arrays. With
    ``gpuDevice`` the producer's CUDA tensor is attached and reads return
    torch tensors; if the attach fails (e.g. the producer exited), the CPU
    mirror is used instead.
    """
    gpu_device = normalize_gpu_device(gpuDevice, name)
    if gpu_device is None:
        return pyshmem.open(name, gpu_device=False)
    try:
        return pyshmem.open(name, gpu_device=gpu_device)
    except FileNotFoundError:
        raise
    except Exception:
        logger.warning(
            "%s: could not attach GPU device %s; falling back to the CPU mirror",
            name,
            gpu_device,
        )
        return pyshmem.open(name, gpu_device=False)


def clear_shms(names):
    """Destroy the named pyshmem streams, ignoring ones that do not exist."""
    for name in names:
        pyshmem.unlink_quiet(name)


def _existing_shm_spec(name: str):
    try:
        stream = open_stream(name)
    except Exception:
        return None
    try:
        return tuple(int(axis) for axis in stream.shape), np.dtype(stream.dtype)
    finally:
        try:
            stream.close()
        except Exception:
            logger.debug("Failed closing temporary SHM probe for %s", name, exc_info=True)


def _default_layout_shape(num_actuators: int) -> tuple[int, int]:
    if num_actuators < 1:
        raise ValueError("num_actuators must be positive")
    if num_actuators == 1:
        return 1, 1
    side = max(
        int(np.ceil(np.sqrt(float(num_actuators)))),
        int(np.ceil(np.sqrt(float(4 * num_actuators) / np.pi))),
    )
    return side, side


def expected_output_shm_specs_for_config(system_conf: dict) -> dict[str, dict[str, object]]:
    sync_runtime_config(system_conf)

    specs: dict[str, dict[str, object]] = {}

    wfs_conf = system_conf.get("wfs")
    if isinstance(wfs_conf, dict):
        output_aliases = _stream_aliases(wfs_conf, "outputStreams")
        width = int(wfs_conf.get("width", 1))
        height = int(wfs_conf.get("height", 1))
        downsample = int(wfs_conf.get("downsampleFactor", 0) or 0)
        image_shape = (width, height)
        if downsample > 0:
            image_shape = (max(1, width // downsample), max(1, height // downsample))
        specs[output_aliases.get("wfsRaw", "wfsRaw")] = {"shape": (width, height), "dtype": np.uint16}
        specs[output_aliases.get("wfs", "wfs")] = {"shape": image_shape, "dtype": np.int32}

    slopes_conf = system_conf.get("slopes")
    if isinstance(slopes_conf, dict) and isinstance(wfs_conf, dict):
        output_aliases = _stream_aliases(slopes_conf, "outputStreams")
        wfs_type = str(slopes_conf.get("type", "SHWFS")).lower()
        if wfs_type == "shwfs":
            downsample = int(wfs_conf.get("downsampleFactor", 0) or 0)
            width = int(wfs_conf.get("width", 1))
            if downsample > 0:
                width = max(1, width // downsample)
            spacing = int(round(float(slopes_conf.get("subApSpacing", 1))))
            num_regions = max(1, width // max(1, spacing))
            signal2d_shape = (2 * num_regions, num_regions)
            signal_size = int(np.prod(signal2d_shape))
            specs[output_aliases.get("signal", "signal")] = {"shape": (signal_size,), "dtype": np.float32}
            specs[output_aliases.get("signal2D", "signal2D")] = {"shape": signal2d_shape, "dtype": np.float32}
        elif wfs_type == "pywfs":
            from pyRTC.utils import generate_circular_aperture_mask

            width = int(wfs_conf.get("width", 1))
            height = int(wfs_conf.get("height", 1))
            default_radius = min(height - int(0.75 * height), width - int(0.75 * width))
            pupil_radius = int(slopes_conf.get("pupilsRadius", max(1, default_radius)))
            pupil_template = generate_circular_aperture_mask(
                int(np.ceil(2 * pupil_radius)),
                pupil_radius,
                float(slopes_conf.get("centralObscurationRatio", 0.0) or 0.0),
            )
            pupil_pixel_count = int(np.count_nonzero(pupil_template))
            signal_size = int(2 * pupil_pixel_count)
            signal2d_shape = (int(2 * pupil_radius), int(4 * pupil_radius))
            specs[output_aliases.get("signal", "signal")] = {"shape": (signal_size,), "dtype": np.float32}
            specs[output_aliases.get("signal2D", "signal2D")] = {"shape": signal2d_shape, "dtype": np.float32}

    wfc_conf = system_conf.get("wfc")
    if isinstance(wfc_conf, dict):
        output_aliases = _stream_aliases(wfc_conf, "outputStreams")
        num_modes = int(wfc_conf.get("numModes", 1))
        specs[output_aliases.get("wfc", "wfc")] = {"shape": (num_modes,), "dtype": np.float32}
        display_grid_size = int(wfc_conf.get("displayGridSize", 33))
        if display_grid_size > 0:
            specs[output_aliases.get("wfc2D", "wfc2D")] = {"shape": (display_grid_size, display_grid_size), "dtype": np.float32}

    psf_conf = system_conf.get("psf")
    if isinstance(psf_conf, dict):
        output_aliases = _stream_aliases(psf_conf, "outputStreams")
        psf_shape = (int(psf_conf.get("width", 1)), int(psf_conf.get("height", 1)))
        specs[output_aliases.get("psfShort", "psfShort")] = {"shape": psf_shape, "dtype": np.int32}
        specs[output_aliases.get("psfLong", "psfLong")] = {"shape": psf_shape, "dtype": np.float64}
        specs[output_aliases.get("strehl", "strehl")] = {"shape": (1,), "dtype": np.float64}
        specs[output_aliases.get("tiptilt", "tiptilt")] = {"shape": (1,), "dtype": np.float64}

    return specs


def expected_output_shms_for_config(system_conf: dict) -> list[str]:
    """Return the known output stream names implied by a validated config."""

    return list(expected_output_shm_specs_for_config(system_conf))


def reconcile_expected_output_shms(system_conf: dict, *, force_rebuild: bool = False) -> tuple[list[str], list[str]]:
    specs = expected_output_shm_specs_for_config(system_conf)
    rebuilt: list[str] = []
    reused: list[str] = []

    for name, spec in specs.items():
        current = None if force_rebuild else _existing_shm_spec(name)
        expected = (tuple(int(axis) for axis in spec["shape"]), np.dtype(spec["dtype"]))
        if current is None:
            continue
        if current != expected:
            clear_shms([name])
            rebuilt.append(name)
        else:
            reused.append(name)

    if rebuilt:
        logger.info("Rebuilt mismatched SHMs: %s", ", ".join(rebuilt))
    if reused:
        logger.debug("Reused matching SHMs: %s", ", ".join(reused))
    return rebuilt, reused



def _stream_aliases(section_conf: dict, mapping_name: str) -> dict[str, str]:
    raw_mapping = section_conf.get(mapping_name, {})
    aliases: dict[str, str] = {}
    if not isinstance(raw_mapping, dict):
        return aliases
    for semantic_name, value in raw_mapping.items():
        if not isinstance(semantic_name, str):
            continue
        if isinstance(value, str):
            shm_name = value.strip()
        elif isinstance(value, dict):
            shm_name = str(value.get("shm", value.get("name", semantic_name))).strip()
        else:
            continue
        if shm_name:
            aliases[semantic_name] = shm_name
    return aliases
