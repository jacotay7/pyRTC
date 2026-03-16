"""Helper utilities for the live shared-memory viewer.

This module keeps the viewer entrypoint and Qt widgets focused on UI concerns by
collecting small helpers for SHM metadata access, frame reshaping, geometry
resolution, and stream polling.
"""

import math

import numpy as np

from pyRTC.Pipeline import ImageSHM
import pyRTC.utils as utils


def is_float_token(value: str) -> bool:
    """Return ``True`` when a CLI token can be parsed as a float."""

    try:
        float(value)
    except ValueError:
        return False
    return True


def split_targets_and_limits(items):
    """Split viewer positional items into stream names and optional limits."""

    shms = list(items)
    vmin = None
    vmax = None

    if len(shms) >= 3 and is_float_token(shms[-1]) and is_float_token(shms[-2]):
        vmax = float(shms.pop())
        vmin = float(shms.pop())
    elif len(shms) >= 2 and is_float_token(shms[-1]):
        vmax = float(shms.pop())

    if not shms:
        raise ValueError("At least one shared-memory stream name is required")

    return shms, vmin, vmax


def normalize_frame(frame):
    """Normalize frames to a 2D shape for consistent viewer rendering."""

    frame = np.asarray(frame)
    if frame.ndim == 1:
        return frame.reshape(1, frame.size)
    if frame.ndim == 2:
        return frame
    return frame.reshape(frame.shape[0], -1)


def read_shm_metadata(shm_name):
    """Read SHM structural metadata for a viewer stream.

    Parameters
    ----------
    shm_name : str
        Name of the data SHM whose metadata stream should be inspected.

    Returns
    -------
    tuple
        Metadata SHM handle, resolved shape tuple, and resolved numpy dtype.
    """

    metadata_shm = ImageSHM(shm_name + "_meta", (ImageSHM.METADATA_SIZE,), np.float64)
    metadata = metadata_shm.read_noblock()
    shm_dtype = utils.float_to_dtype(metadata[ImageSHM.METADATA_INDEX_DTYPE])
    shm_dims = []
    index = 0
    while (
        ImageSHM.METADATA_INDEX_SHAPE_START + index < metadata.size
        and int(metadata[ImageSHM.METADATA_INDEX_SHAPE_START + index]) > 0
    ):
        shm_dims.append(int(metadata[ImageSHM.METADATA_INDEX_SHAPE_START + index]))
        index += 1
    if not shm_dims:
        shm_dims = [1]
    return metadata_shm, tuple(shm_dims), shm_dtype


def resolve_grid(num_plots: int, geometry: str):
    """Resolve a viewer geometry string into a concrete grid size."""

    geometry = geometry.lower()
    if geometry == "row":
        return 1, num_plots
    if geometry == "column":
        return num_plots, 1
    if geometry == "square":
        cols = int(math.ceil(math.sqrt(num_plots)))
        rows = int(math.ceil(num_plots / cols))
        return rows, cols
    if "x" in geometry:
        parts = geometry.split("x", 1)
        try:
            rows = int(parts[0])
            cols = int(parts[1])
        except ValueError as exc:
            raise ValueError(f"Invalid geometry string: {geometry}") from exc
        if rows < 1 or cols < 1:
            raise ValueError(f"Invalid geometry string: {geometry}")
        if rows * cols < num_plots:
            raise ValueError(
                f"Geometry {geometry} does not have enough cells for {num_plots} SHMs"
            )
        return rows, cols
    raise ValueError(
        "Geometry must be one of: square, row, column, or an explicit grid like 2x3"
    )


def compute_window_size(frames, rows, cols, pixel_scale):
    """Estimate a practical viewer window size for the current frames."""

    max_height = max(frame.shape[0] for frame in frames)
    max_width = max(frame.shape[1] for frame in frames)
    panel_extent = max(max_height, max_width)
    plot_width = cols * panel_extent * pixel_scale
    plot_height = rows * panel_extent * pixel_scale
    width = int(max(360, min(1320, plot_width + cols * 110)))
    height = int(max(320, min(900, plot_height + rows * 120 + 70)))
    return width, height


def normalize_geometry_value(num_plots: int, geometry: str):
    """Return the normalized ``rows x cols`` geometry label for the viewer."""

    rows, cols = resolve_grid(num_plots, geometry)
    return f"{rows}x{cols}"


def format_shape(shape):
    """Format a shape tuple for compact display in stream labels."""

    return "x".join(str(int(dim)) for dim in shape)


class StreamConnection:
    """Tracks one SHM stream and its metadata for viewer refreshes.

    The viewer uses this wrapper to keep data access, cached frames, update
    counters, and derived display strings in one place. It avoids scattering SHM
    metadata handling logic throughout the UI layer.
    """

    def __init__(self, shm_name):
        self.name = shm_name
        self.metadata_shm, shm_shape, shm_dtype = read_shm_metadata(shm_name)
        self.shape = tuple(shm_shape)
        self.display_name = f"{shm_name} ({format_shape(self.shape)})"
        self.shm = ImageSHM(shm_name, shm_shape, shm_dtype)
        self.last_count = None
        self.last_time = None
        self.last_fps_text = None
        self._closed = False
        self.cached_frame = normalize_frame(np.array(self.shm.read_noblock(), copy=True))

    def prime(self):
        """Prime cached count and timestamp state before timer-driven refreshes."""

        metadata = self.metadata_shm.read_noblock()
        self.last_count = metadata[ImageSHM.METADATA_INDEX_COUNT]
        self.last_time = metadata[ImageSHM.METADATA_INDEX_WRITE_TIME]
        self.last_fps_text = None
        return self.cached_frame

    def poll(self):
        """Poll the latest metadata and return the current viewer snapshot."""

        metadata = self.metadata_shm.read_noblock()
        new_count = metadata[ImageSHM.METADATA_INDEX_COUNT]
        new_time = metadata[ImageSHM.METADATA_INDEX_WRITE_TIME]
        changed = self.last_count is None or self.last_time is None
        if not changed:
            changed = new_count != self.last_count or new_time != self.last_time

        if new_time > (self.last_time or 0):
            old_count = 0 if self.last_count is None else self.last_count
            old_time = 0 if self.last_time is None else self.last_time
            fps = np.round((new_count - old_count) / max(new_time - old_time, 1e-12), 2)
            fps_text = f"{fps} FPS"
        else:
            fps_text = "PAUSED"

        status_changed = fps_text != self.last_fps_text

        if changed:
            self.cached_frame = normalize_frame(np.array(self.shm.read_noblock(), copy=True))

        self.last_count = new_count
        self.last_time = new_time
        self.last_fps_text = fps_text
        return {
            "frame": self.cached_frame,
            "changed": changed,
            "status_changed": status_changed,
            "fps_text": fps_text,
            "count": new_count,
            "timestamp": new_time,
        }

    def close(self):
        """Close the data and metadata SHMs exactly once."""

        if self._closed:
            return
        self._closed = True
        try:
            self.shm.close()
        finally:
            self.metadata_shm.close()