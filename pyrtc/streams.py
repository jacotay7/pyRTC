"""pyshmem-backed stream policy for pyrtc.

Shared-memory transport itself is provided by the ``pyshmem`` package; this
module holds the pyrtc-side policy for creating and attaching to streams,
plus the config-driven planning of which output streams a system implies.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pyshmem

from pyrtc.config_runtime import stream_alias_map, sync_runtime_config
from pyrtc.logging_utils import get_logger

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


def normalize_gpu_device(gpu_device, context: str = ""):
    if gpu_device is None:
        return None
    if not TORCH_AVAILABLE:
        prefix = f"{context}: " if context else ""
        logging.log(
            level=logging.WARNING,
            msg=f"{prefix}gpu_device was requested but PyTorch is not installed; defaulting to CPU mode.",
        )
        return None
    return gpu_device


#: Environment variable that overrides the default of :func:`stream_notify_default`.
STREAM_NOTIFY_ENV = "PYRTC_STREAM_NOTIFY"

_FALSE_STRINGS = {"0", "false", "no", "off"}


def stream_notify_default() -> bool:
    """Return whether new streams are created with pyshmem ``notify=True``.

    Notify-enabled streams let a writer wake parked consumers through a Linux
    futex instead of consumers sleeping between polls, which cuts the
    cross-process (hard-RTC) handoff latency. It is on by default; set the
    ``PYRTC_STREAM_NOTIFY`` environment variable to ``0`` to create polling
    streams instead. Because it is an environment variable, the choice also
    reaches hard-RTC component processes launched from the same shell.
    """
    value = os.environ.get(STREAM_NOTIFY_ENV)
    if value is None:
        return True
    return value.strip().lower() not in _FALSE_STRINGS


def _stored_notify_flag(stream) -> bool:
    """Return the notify flag recorded in a stream's metadata."""
    try:
        return bool(stream.to_config().get("notify", False))
    except Exception:
        return bool(getattr(stream, "notify", False))


def create_stream(name, shape, dtype, gpu_device=None, *, notify=None):
    """Create the pyshmem stream backing a component output.

    An existing CPU stream is reused when its shape, dtype, and notify flag
    already match, so attached readers (viewers, telemetry) keep working
    across component restarts; on any mismatch the stream is rebuilt. That
    includes a stream left over from a run with the other notify setting (or
    from a pyrtc version that did not enable notify): the flag is fixed at
    creation, so the stream is recreated to honour the requested setting.
    GPU-backed streams are always rebuilt because a previous producer's CUDA
    tensor cannot be re-exported. GPU streams are created with
    ``cpu_mirror=True`` so CPU-only processes can always read them.

    ``notify`` defaults to :func:`stream_notify_default` (on unless
    ``PYRTC_STREAM_NOTIFY=0``). On platforms without a futex pyshmem keeps
    the flag but consumers poll as before.
    """
    shape = tuple(int(axis) for axis in shape)
    dtype = np.dtype(dtype)
    notify = stream_notify_default() if notify is None else bool(notify)
    gpu_device = normalize_gpu_device(gpu_device, name)
    if gpu_device is not None and not pyshmem.gpu_available():
        logger.warning(
            "%s: gpu_device %s requested but CUDA is not available; using a CPU stream",
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

    create_kwargs = {"notify": notify}
    if gpu_device is not None:
        create_kwargs.update(gpu_device=gpu_device, cpu_mirror=True)

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
                and _stored_notify_flag(existing) == notify
            )
            if matches:
                logger.debug("Reusing existing stream %s", name)
                return existing
            existing.close()

    logger.debug("Rebuilding stream %s", name)
    pyshmem.unlink_quiet(name)
    return pyshmem.create(name, shape=shape, dtype=dtype, **create_kwargs)


def open_stream(name, gpu_device=None, *, readonly=False):
    """Attach to an existing pyshmem stream.

    Without ``gpu_device`` the stream is opened CPU-side: GPU-backed streams
    are read through their CPU mirror and reads return NumPy arrays. With
    ``gpu_device`` the producer's CUDA tensor is attached and reads return
    torch tensors; if the attach fails (e.g. the producer exited), the CPU
    mirror is used instead. ``readonly=True`` returns a handle that rejects
    writes, for observers such as viewers, telemetry, and latency probes.
    """
    gpu_device = normalize_gpu_device(gpu_device, name)
    if gpu_device is None:
        return pyshmem.open(name, gpu_device=False, readonly=readonly)
    try:
        return pyshmem.open(name, gpu_device=gpu_device, readonly=readonly)
    except FileNotFoundError:
        raise
    except Exception:
        logger.warning(
            "%s: could not attach GPU device %s; falling back to the CPU mirror",
            name,
            gpu_device,
        )
        return pyshmem.open(name, gpu_device=False, readonly=readonly)


def clear_shms(names):
    """Destroy the named pyshmem streams, ignoring ones that do not exist."""
    for name in names:
        pyshmem.unlink_quiet(name)


def _existing_shm_spec(name: str):
    try:
        stream = open_stream(name, readonly=True)
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
        output_aliases = stream_alias_map(wfs_conf.get("output_streams"))
        width = int(wfs_conf.get("width", 1))
        height = int(wfs_conf.get("height", 1))
        downsample = int(wfs_conf.get("downsample_factor", 0) or 0)
        image_shape = (width, height)
        if downsample > 0:
            image_shape = (max(1, width // downsample), max(1, height // downsample))
        specs[output_aliases.get("wfs_raw", "wfs_raw")] = {
            "shape": (width, height),
            "dtype": np.uint16,
        }
        specs[output_aliases.get("wfs", "wfs")] = {"shape": image_shape, "dtype": np.int32}

    slopes_conf = system_conf.get("slopes")
    if isinstance(slopes_conf, dict) and isinstance(wfs_conf, dict):
        output_aliases = stream_alias_map(slopes_conf.get("output_streams"))
        wfs_type = str(slopes_conf.get("type", "SHWFS")).lower()
        if wfs_type == "shwfs":
            downsample = int(wfs_conf.get("downsample_factor", 0) or 0)
            width = int(wfs_conf.get("width", 1))
            if downsample > 0:
                width = max(1, width // downsample)
            spacing = int(round(float(slopes_conf.get("sub_ap_spacing", 1))))
            num_regions = max(1, width // max(1, spacing))
            signal2d_shape = (2 * num_regions, num_regions)
            signal_size = int(np.prod(signal2d_shape))
            specs[output_aliases.get("signal", "signal")] = {
                "shape": (signal_size,),
                "dtype": np.float32,
            }
            specs[output_aliases.get("signal_2d", "signal_2d")] = {
                "shape": signal2d_shape,
                "dtype": np.float32,
            }
        elif wfs_type == "pywfs":
            from pyrtc.utils import generate_circular_aperture_mask

            width = int(wfs_conf.get("width", 1))
            height = int(wfs_conf.get("height", 1))
            default_radius = min(height - int(0.75 * height), width - int(0.75 * width))
            pupil_radius = int(slopes_conf.get("pupils_radius", max(1, default_radius)))
            pupil_template = generate_circular_aperture_mask(
                int(np.ceil(2 * pupil_radius)),
                pupil_radius,
                float(slopes_conf.get("central_obscuration_ratio", 0.0) or 0.0),
            )
            pupil_pixel_count = int(np.count_nonzero(pupil_template))
            signal_size = int(2 * pupil_pixel_count)
            signal2d_shape = (int(2 * pupil_radius), int(4 * pupil_radius))
            specs[output_aliases.get("signal", "signal")] = {
                "shape": (signal_size,),
                "dtype": np.float32,
            }
            specs[output_aliases.get("signal_2d", "signal_2d")] = {
                "shape": signal2d_shape,
                "dtype": np.float32,
            }

    wfc_conf = system_conf.get("wfc")
    if isinstance(wfc_conf, dict):
        output_aliases = stream_alias_map(wfc_conf.get("output_streams"))
        num_modes = int(wfc_conf.get("num_modes", 1))
        specs[output_aliases.get("wfc", "wfc")] = {"shape": (num_modes,), "dtype": np.float32}
        display_grid_size = int(wfc_conf.get("display_grid_size", 33))
        if display_grid_size > 0:
            specs[output_aliases.get("wfc_2d", "wfc_2d")] = {
                "shape": (display_grid_size, display_grid_size),
                "dtype": np.float32,
            }

    psf_conf = system_conf.get("psf")
    if isinstance(psf_conf, dict):
        output_aliases = stream_alias_map(psf_conf.get("output_streams"))
        psf_shape = (int(psf_conf.get("width", 1)), int(psf_conf.get("height", 1)))
        specs[output_aliases.get("psf_short", "psf_short")] = {
            "shape": psf_shape,
            "dtype": np.int32,
        }
        specs[output_aliases.get("psf_long", "psf_long")] = {
            "shape": psf_shape,
            "dtype": np.float64,
        }
        specs[output_aliases.get("strehl", "strehl")] = {"shape": (1,), "dtype": np.float64}
        specs[output_aliases.get("tiptilt", "tiptilt")] = {"shape": (1,), "dtype": np.float64}

    return specs


def expected_output_shms_for_config(system_conf: dict) -> list[str]:
    """Return the known output stream names implied by a validated config."""

    return list(expected_output_shm_specs_for_config(system_conf))


def reconcile_expected_output_shms(
    system_conf: dict, *, force_rebuild: bool = False
) -> tuple[list[str], list[str]]:
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
