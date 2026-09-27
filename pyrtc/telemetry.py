"""Telemetry capture helpers for pyrtc shared-memory streams.

The :class:`Telemetry` component provides a small, operator-friendly API for
capturing bounded stretches of existing pyrtc streams into standard NumPy data
products. Each save creates one session directory containing per-stream
``frames.npy`` and ``timestamps.npy`` files plus lightweight JSON metadata.

The intended user workflow is deliberately simple::

    telem = Telemetry()
    telem.save("wfs", 1000)
    telem.save(["wfs", "wfc"], 1000)
    data = telem.read_last_save()
    print(data["wfs"]["frames"].shape)

This keeps the hot path straightforward, stores frames in a standard NumPy
format, and makes the resulting capture easy to load for offline analysis and
future export layers such as AOTPy.

For "what just happened?" captures, telemetry can also record continuously
into a bounded in-memory ring buffer and dump it on demand, in the same
session format::

    telem.start_ring_buffer(["wfs", "wfc"], seconds=10)
    ...
    path = telem.dump_ring_buffer("event")
    telem.stop_ring_buffer()
"""

from __future__ import annotations

import json
import math
import os
import platform
import socket
import threading
import time
import uuid
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np

from pyrtc.logging_utils import get_logger
from pyrtc.streams import open_stream
from pyrtc.component import Component
from pyrtc.utils import set_from_config


logger = get_logger(__name__)
TELEMETRY_SESSION_SCHEMA_VERSION = 1
#: Extra capacity allocated over the measured publication rate when a ring
#: buffer is sized from ``seconds`` alone.
RING_BUFFER_RATE_HEADROOM = 1.5


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _resolve_pyrtc_version() -> str:
    try:
        return importlib_metadata.version("pyrtc")
    except importlib_metadata.PackageNotFoundError:
        return "1.0.0"


def _ensure_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def _sanitize_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


def _host_metadata() -> dict:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pid": os.getpid(),
    }


def _coerce_shape(shape) -> tuple[int, ...]:
    return tuple(int(axis) for axis in shape)


def _build_session_directory(base_dir: Path, session_id: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return base_dir / f"session_{timestamp}_{session_id[:8]}"


def _session_file(session_path: str | Path) -> Path:
    path = _ensure_path(session_path)
    if path.is_dir():
        return path / "session.json"
    return path


def _normalize_stream_specs(streams, num_frames, semantic_tags=None, sampling=None) -> list[dict]:
    if isinstance(streams, str):
        stream_names = [streams]
    elif isinstance(streams, (list, tuple)):
        stream_names = [str(name) for name in streams]
    else:
        raise TypeError("streams must be a stream name or a list of stream names")

    if isinstance(num_frames, dict):
        frame_counts = {str(name): int(value) for name, value in num_frames.items()}
    else:
        frame_counts = {name: int(num_frames) for name in stream_names}

    tag_mapping = semantic_tags if isinstance(semantic_tags, dict) else None
    sampling_mapping = sampling if isinstance(sampling, dict) else None

    specs = []
    for stream_name in stream_names:
        if stream_name not in frame_counts:
            raise ValueError(f"Missing frame count for telemetry stream '{stream_name}'")
        frame_count = int(frame_counts[stream_name])
        if frame_count <= 0:
            raise ValueError(f"Telemetry frame count for '{stream_name}' must be positive")

        if tag_mapping is not None:
            tags = tag_mapping.get(stream_name, [])
        elif semantic_tags is None:
            tags = []
        else:
            tags = semantic_tags

        if tags and not isinstance(tags, (list, tuple)):
            raise TypeError(
                "semantic_tags must be a list of strings or a mapping of stream name to string lists"
            )

        if sampling_mapping is not None:
            stream_sampling = sampling_mapping.get(stream_name)
        else:
            stream_sampling = sampling

        specs.append(
            {
                "name": stream_name,
                "frame_count": frame_count,
                "semantic_tags": [str(tag) for tag in tags],
                "sampling": stream_sampling,
            }
        )
    return specs


def load_telemetry_manifest(session_path: str | Path) -> dict:
    """Load the JSON metadata for one telemetry save.

    Parameters
    ----------
    session_path : str or Path
        Either a telemetry session directory or the ``session.json`` file
        within that directory.

    Returns
    -------
    dict
        Parsed session metadata.
    """

    session_file = _session_file(session_path)
    try:
        with session_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        raise ValueError(f"Failed to read telemetry manifest {session_file}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Telemetry manifest {session_file} must contain a JSON object")
    streams = payload.get("streams")
    if not isinstance(streams, list):
        raise ValueError(f"Telemetry manifest {session_file} is missing a valid 'streams' list")
    return payload


def load_telemetry_session(session_path: str | Path, *, mmap_mode=None) -> dict:
    """Load a telemetry save into an easy-to-use per-stream mapping.

    Parameters
    ----------
    session_path : str or Path
        Either a telemetry session directory or the ``session.json`` file.
    mmap_mode : str, optional
        NumPy memmap mode passed to :func:`numpy.load` for ``frames.npy``.

    Returns
    -------
    dict
        Mapping keyed by stream name. Each stream entry contains ``frames``,
        ``timestamps``, and ``metadata``. The special key ``_session`` contains
        session-level metadata.
    """

    session_file = _session_file(session_path).resolve()
    session_dir = session_file.parent
    manifest = load_telemetry_manifest(session_file)

    loaded = {"_session": manifest}
    for stream_record in manifest["streams"]:
        if not isinstance(stream_record, dict):
            raise ValueError(
                f"Telemetry manifest {session_file} contains a non-mapping stream record"
            )
        for required_key in ("name", "frames_file", "timestamps_file", "metadata_file"):
            if required_key not in stream_record:
                raise ValueError(
                    f"Telemetry manifest {session_file} stream record is missing '{required_key}'"
                )

        stream_name = stream_record["name"]
        frames_path = (session_dir / stream_record["frames_file"]).resolve()
        timestamps_path = (session_dir / stream_record["timestamps_file"]).resolve()
        metadata_path = (session_dir / stream_record["metadata_file"]).resolve()
        for required_path in (frames_path, timestamps_path, metadata_path):
            if not required_path.exists():
                raise FileNotFoundError(f"Telemetry capture file not found: {required_path}")

        with metadata_path.open("r", encoding="utf-8") as handle:
            stream_metadata = json.load(handle)

        loaded[stream_name] = {
            "frames": np.load(frames_path, mmap_mode=mmap_mode),
            "timestamps": np.load(timestamps_path),
            "metadata": stream_metadata,
        }
        # Sessions recorded before frame ids were captured have no file.
        if "frame_ids_file" in stream_record:
            frame_ids_path = (session_dir / stream_record["frame_ids_file"]).resolve()
            loaded[stream_name]["frame_ids"] = np.load(frame_ids_path)
        # Publication counts were added alongside the ring buffer (#61).
        if "counts_file" in stream_record:
            counts_path = (session_dir / stream_record["counts_file"]).resolve()
            loaded[stream_name]["counts"] = np.load(counts_path)

    return loaded


def list_telemetry_sessions(data_dir: str | Path) -> list[str]:
    """Return all telemetry session directories under one base directory."""

    base_dir = _ensure_path(data_dir)
    if not base_dir.exists():
        return []
    return [str(path.resolve().parent) for path in sorted(base_dir.glob("session_*/session.json"))]


class _StreamCapture:
    """Files and reader for one stream of a telemetry session."""

    def __init__(self, spec, stream_dir, shm, unique_str):
        self.spec = spec
        self.name = spec["name"]
        self.frame_count = int(spec["frame_count"])
        self.stream_dir = stream_dir
        self.shm = shm
        self.unique_str = unique_str
        self.frame_shape = _coerce_shape(tuple(shm.shape))
        self.dtype = np.dtype(shm.dtype)
        self.frames_path = stream_dir / "frames.npy"
        self.timestamps_path = stream_dir / "timestamps.npy"
        self.frame_ids_path = stream_dir / "frame_ids.npy"
        self.counts_path = stream_dir / "counts.npy"
        self.frames = np.lib.format.open_memmap(
            self.frames_path,
            mode="w+",
            dtype=self.dtype,
            shape=(self.frame_count, *self.frame_shape),
        )
        self.timestamps = np.lib.format.open_memmap(
            self.timestamps_path, mode="w+", dtype=np.float64, shape=(self.frame_count,)
        )
        self.frame_ids = np.empty(self.frame_count, dtype=np.uint64)
        self.counts = np.empty(self.frame_count, dtype=np.uint64)
        self.missed_frames = 0
        self._closed = False

    @classmethod
    def open(cls, spec, session_dir, unique_str):
        shm = open_stream(spec["name"], readonly=True)
        try:
            stream_dir = session_dir / _sanitize_name(spec["name"])
            stream_dir.mkdir(parents=True, exist_ok=False)
            return cls(spec, stream_dir, shm, unique_str)
        except Exception:
            shm.close()
            raise

    def capture(self, start: threading.Barrier) -> None:
        start.wait()
        for index in range(self.frame_count):
            # One publication gives the frame together with its own write
            # time and frame id, never a later write's.
            publication = self.shm.read_new_publication()
            self.frames[index] = np.asarray(publication.payload, dtype=self.dtype)
            self.timestamps[index] = publication.write_time
            self.frame_ids[index] = publication.frame_id
            self.counts[index] = publication.count
            if index > 0:
                self.missed_frames += publication.missed_publications

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.shm.close()

    def finalize(self, session_dir) -> dict:
        self.frames.flush()
        self.timestamps.flush()
        del self.frames
        del self.timestamps
        np.save(self.frame_ids_path, self.frame_ids)
        np.save(self.counts_path, self.counts)
        return _write_stream_metadata(
            session_dir,
            self.stream_dir,
            spec=self.spec,
            dtype=self.dtype,
            frame_shape=self.frame_shape,
            frame_count=self.frame_count,
            missed_frames=self.missed_frames,
            unique_str=self.unique_str,
        )


def _write_stream_metadata(
    session_dir,
    stream_dir,
    *,
    spec,
    dtype,
    frame_shape,
    frame_count,
    missed_frames,
    unique_str,
    extra_metadata=None,
) -> dict:
    """Write one stream's ``metadata.json`` and return its manifest record.

    ``stream_dir`` must already hold ``frames.npy``, ``timestamps.npy``,
    ``frame_ids.npy`` and ``counts.npy``. Shared by :meth:`Telemetry.save`
    and :meth:`Telemetry.dump_ring_buffer` so both produce one format.
    """
    stream_metadata = {
        "name": spec["name"],
        "dtype": dtype.name,
        "shape": list(frame_shape),
        "frame_count": int(frame_count),
        "missed_frames": int(missed_frames),
        "timestamp_unit": "unix_seconds",
        "sampling": spec["sampling"],
        "semantic_tags": spec["semantic_tags"],
        "capture_label": unique_str or None,
    }
    if extra_metadata:
        stream_metadata.update(extra_metadata)
    metadata_path = stream_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(stream_metadata, handle, indent=2, sort_keys=True)

    def _rel(filename):
        return str((stream_dir / filename).relative_to(session_dir))

    return {
        "name": spec["name"],
        "dtype": dtype.name,
        "shape": list(frame_shape),
        "frame_count": int(frame_count),
        "frames_file": _rel("frames.npy"),
        "timestamps_file": _rel("timestamps.npy"),
        "frame_ids_file": _rel("frame_ids.npy"),
        "counts_file": _rel("counts.npy"),
        "metadata_file": _rel("metadata.json"),
        "sampling": spec["sampling"],
        "semantic_tags": spec["semantic_tags"],
    }


def _capture_concurrently(captures) -> None:
    """Capture every stream over one shared time window.

    Each stream gets its own reader thread, released together, so a session
    holding ``wfs``, ``signal`` and ``wfc`` covers the same loop iterations
    (pair them exactly with the recorded frame ids).
    """
    start = threading.Barrier(len(captures))
    errors = []

    def _run(capture):
        try:
            capture.capture(start)
        except BaseException as exc:  # re-raised in the caller
            errors.append(exc)
            start.abort()

    threads = [
        threading.Thread(target=_run, args=(capture,), name=f"telemetry-{capture.name}")
        for capture in captures
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]


class _RingStream:
    """Preallocated in-memory ring of the most recent publications of one stream.

    A daemon reader thread records every publication its read-only handle
    sees (payload, ``write_time``, ``frame_id`` and ``count``) into
    fixed-size arrays allocated once at start, so memory stays bounded at
    ``capacity`` frames. Publications the reader could not keep up with are
    detected from gaps in ``count`` and counted in ``missed``.
    """

    def __init__(self, name, shm, capacity):
        self.name = str(name)
        self.shm = shm
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError(f"Ring buffer capacity for '{self.name}' must be positive")
        self.frame_shape = _coerce_shape(tuple(shm.shape))
        self.dtype = np.dtype(shm.dtype)
        self.frames = np.empty((self.capacity, *self.frame_shape), dtype=self.dtype)
        self.timestamps = np.empty(self.capacity, dtype=np.float64)
        self.frame_ids = np.empty(self.capacity, dtype=np.uint64)
        self.counts = np.empty(self.capacity, dtype=np.uint64)
        self.head = 0  # next slot to write
        self.size = 0  # valid slots
        self.recorded = 0  # publications recorded since start
        self.missed = 0  # publications the reader never saw
        self.error = None
        self.lock = threading.Lock()
        self.thread = None

    @property
    def nbytes(self) -> int:
        return int(
            self.frames.nbytes + self.timestamps.nbytes + self.frame_ids.nbytes + self.counts.nbytes
        )

    def run(self, stop: threading.Event, poll_interval: float) -> None:
        try:
            # Only record publications made after recording starts.
            last_count = int(self.shm.count)
            while not stop.is_set():
                try:
                    # Level-triggered: a write that lands between two
                    # iterations still satisfies ``count > last_count``.
                    publication = self.shm.read_after_publication(last_count, timeout=poll_interval)
                except TimeoutError:
                    continue
                payload = np.asarray(publication.payload, dtype=self.dtype)
                with self.lock:
                    slot = self.head
                    self.frames[slot] = payload
                    self.timestamps[slot] = publication.write_time
                    self.frame_ids[slot] = publication.frame_id
                    self.counts[slot] = publication.count
                    self.head = (slot + 1) % self.capacity
                    self.size = min(self.size + 1, self.capacity)
                    self.recorded += 1
                    self.missed += max(0, int(publication.count) - last_count - 1)
                last_count = int(publication.count)
        except BaseException as exc:  # reported through status and dumps
            self.error = exc
            logger.exception("Telemetry ring buffer reader for '%s' failed", self.name)

    def snapshot_locked(self) -> dict:
        """Copy the valid slots oldest-first. The caller holds ``self.lock``."""
        order = (self.head - self.size + np.arange(self.size)) % self.capacity
        return {
            "frames": self.frames[order],
            "timestamps": self.timestamps[order],
            "frame_ids": self.frame_ids[order],
            "counts": self.counts[order],
            "recorded": int(self.recorded),
            "missed": int(self.missed),
            "error": None if self.error is None else repr(self.error),
        }

    def status(self) -> dict:
        with self.lock:
            return {
                "capacity": self.capacity,
                "size": int(self.size),
                "recorded": int(self.recorded),
                "missed": int(self.missed),
                "nbytes": self.nbytes,
                "alive": bool(self.thread is not None and self.thread.is_alive()),
                "error": None if self.error is None else repr(self.error),
            }


def _estimate_ring_capacities(shms: dict, seconds: float, probe_seconds: float) -> dict:
    """Size each ring to hold ``seconds`` of data at its measured rate.

    Publication counts are sampled on every stream across one shared
    ``probe_seconds`` window; the capacity is the measured rate times
    ``seconds`` with :data:`RING_BUFFER_RATE_HEADROOM` headroom for rate
    jitter. A stream that does not publish during the probe cannot be sized
    and raises, asking for an explicit ``frames``.
    """
    start_counts = {name: int(shm.count) for name, shm in shms.items()}
    start = time.monotonic()
    time.sleep(float(probe_seconds))
    elapsed = time.monotonic() - start
    capacities = {}
    for name, shm in shms.items():
        published = int(shm.count) - start_counts[name]
        if published <= 0:
            raise ValueError(
                f"Telemetry ring buffer could not measure a publication rate for "
                f"'{name}' within {probe_seconds} s; pass frames= to size it explicitly"
            )
        rate = published / elapsed
        capacities[name] = max(1, int(math.ceil(seconds * rate * RING_BUFFER_RATE_HEADROOM)))
    return capacities


class Telemetry(Component):
    """Capture pyrtc streams into standard NumPy telemetry products.

    Parameters
    ----------
    conf : dict, optional
        Telemetry configuration. The most useful keys are:

        ``data_dir``
            Base directory used for capture output. Defaults to ``./data/``.

        ``streams``
            Optional default stream names for :meth:`save_configured_streams`
            and the ring buffer.

        ``ring_buffer``
            Optional continuous recording started by :meth:`start` and stopped
            by :meth:`stop`: a mapping with ``streams`` (defaults to
            ``streams``), ``seconds`` and/or ``frames``, ``probe_seconds`` and
            ``autostart`` (default ``True``). See :meth:`start_ring_buffer`.

        ``functions``
            Standard pyrtc worker-thread configuration inherited from
            :class:`pyrtc.component.Component`.

    Notes
    -----
    The public API is intentionally small and Sphinx-friendly:

    - :meth:`save` captures one or more streams
    - :meth:`read_last_save` reopens the most recent capture
    - :meth:`list_sessions` enumerates saved captures on disk
    - :meth:`start_ring_buffer`, :meth:`dump_ring_buffer` and
      :meth:`stop_ring_buffer` record continuously and dump on trigger

    Examples
    --------
    >>> telem = Telemetry()
    >>> telem.save('wfs', 1000)
    >>> telem.save(['wfs', 'wfc'], 1000)
    >>> data = telem.read_last_save()
    >>> data['wfs']['frames'].shape[0]
    1000
    """

    def __init__(self, conf=None) -> None:
        conf = {} if conf is None else conf
        try:
            super().__init__(conf)
            self.data_dir = Path(set_from_config(conf, "data_dir", "./data/")).expanduser()
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.configured_streams = list(set_from_config(conf, "streams", []))
            self.ring_buffer_conf = dict(set_from_config(conf, "ring_buffer", None) or {})
            self._ring_streams = []
            self.most_recent_save = ""
            self.most_recent_file = ""
            self.all_saves = []
            self.all_files = []
            self.dtypes = []
            self.dims = []
            self.logger.info(
                "Initialized telemetry data_dir=%s configured_streams=%s",
                self.data_dir,
                self.configured_streams,
            )
        except Exception:
            logger.exception("Failed to initialize telemetry")
            raise

    def _session_manifest(
        self,
        *,
        session_id: str,
        stream_records: list[dict],
        config=None,
        config_path: str | Path | None = None,
        metadata: dict | None = None,
    ) -> dict:
        return {
            "schema_version": TELEMETRY_SESSION_SCHEMA_VERSION,
            "session_id": session_id,
            "created_at": _utc_now_iso(),
            "pyrtc_version": _resolve_pyrtc_version(),
            "host": _host_metadata(),
            "config_path": str(_ensure_path(config_path).resolve())
            if config_path is not None
            else None,
            "config": config,
            "metadata": metadata or {},
            "streams": stream_records,
        }

    def _record_saved_stream(self, frames_path, dtype, frame_shape) -> None:
        self.most_recent_file = str(frames_path)
        self.all_files.append(self.most_recent_file)
        self.dtypes.append(dtype)
        self.dims.append(list(frame_shape))

    def _finish_session(
        self,
        session_dir: Path,
        *,
        session_id: str,
        stream_records: list[dict],
        config=None,
        config_path=None,
        metadata=None,
    ) -> str:
        session_manifest = self._session_manifest(
            session_id=session_id,
            stream_records=stream_records,
            config=config,
            config_path=config_path,
            metadata=metadata,
        )
        session_file = session_dir / "session.json"
        with session_file.open("w", encoding="utf-8") as handle:
            json.dump(session_manifest, handle, indent=2, sort_keys=True)
        self.most_recent_save = str(session_dir.resolve())
        self.all_saves.append(self.most_recent_save)
        return self.most_recent_save

    def save(
        self,
        streams,
        num_frames,
        *,
        unique_str="",
        session_id: str | None = None,
        semantic_tags=None,
        sampling=None,
        config=None,
        config_path: str | Path | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Save one or more streams into a NumPy-backed telemetry session.

        Parameters
        ----------
        streams : str or sequence of str
            Stream name or stream names to capture.
        num_frames : int or dict
            Number of frames to save. When ``streams`` is a list, this may be
            one shared integer or a mapping of stream name to frame count.
        unique_str : str, optional
            Optional suffix used in the session metadata for operator clarity.
        session_id : str, optional
            Explicit session identifier. A UUID is generated when omitted.
        semantic_tags : list or dict, optional
            Optional semantic labels for future export layers such as AOTPy.
        sampling : object or dict, optional
            Optional sampling metadata stored alongside each stream.
        config : dict, optional
            Optional config subset to embed in the session metadata.
        config_path : str or Path, optional
            Optional config path to store in the session metadata.
        metadata : dict, optional
            Arbitrary extra session metadata.

        Returns
        -------
        str
            Absolute path to the created telemetry session directory.
        """

        component_logger = getattr(self, "logger", logger)
        specs = _normalize_stream_specs(
            streams, num_frames, semantic_tags=semantic_tags, sampling=sampling
        )
        session_id = session_id or uuid.uuid4().hex
        session_dir = _build_session_directory(self.data_dir, session_id)
        session_dir.mkdir(parents=True, exist_ok=False)

        try:
            stream_records = []
            captures = []
            try:
                for spec in specs:
                    captures.append(_StreamCapture.open(spec, session_dir, unique_str))
                _capture_concurrently(captures)
            finally:
                for capture in captures:
                    capture.close()

            for capture in captures:
                stream_records.append(capture.finalize(session_dir))
                self._record_saved_stream(capture.frames_path, capture.dtype, capture.frame_shape)

            self._finish_session(
                session_dir,
                session_id=session_id,
                stream_records=stream_records,
                config=config,
                config_path=config_path,
                metadata=metadata,
            )
            component_logger.info(
                "Saved telemetry session %s streams=%s frames=%s path=%s",
                session_id,
                [spec["name"] for spec in specs],
                [spec["frame_count"] for spec in specs],
                self.most_recent_save,
            )
            return self.most_recent_save
        except Exception:
            component_logger.exception(
                "Failed to save telemetry streams %s", [spec["name"] for spec in specs]
            )
            raise

    def save_session(self, streams, num_frames, **kwargs) -> str:
        """Compatibility wrapper around :meth:`save`."""

        return self.save(streams, num_frames, **kwargs)

    def save_configured_streams(self, num_frames, **kwargs) -> str:
        """Save the streams configured on this telemetry component.

        This is a convenience wrapper for configs that already declare a fixed
        telemetry stream set.
        """

        if not self.configured_streams:
            raise ValueError("Telemetry has no configured streams to capture")
        return self.save(self.configured_streams, num_frames, **kwargs)

    # ------------------------------------------------------------------
    # Continuous recording (ring buffer)
    # ------------------------------------------------------------------

    @property
    def ring_buffer_running(self) -> bool:
        """``True`` while a ring buffer is recording."""

        return bool(getattr(self, "_ring_streams", None))

    def start_ring_buffer(
        self,
        streams=None,
        *,
        seconds: float | None = None,
        frames=None,
        probe_seconds: float = 1.0,
        poll_interval: float = 0.05,
    ) -> dict:
        """Continuously record the most recent publications of some streams.

        One reader thread per stream records every publication its read-only
        handle sees into a preallocated in-memory ring, keeping the newest
        ``frames`` publications (and, with ``seconds``, dumping only the last
        ``seconds`` of them). Recording never blocks the producers: readers
        use read-only handles and the lock they take is private to telemetry.
        Call :meth:`dump_ring_buffer` to write the buffer as an ordinary
        telemetry session, and :meth:`stop_ring_buffer` to stop recording.

        Parameters
        ----------
        streams : str or sequence of str, optional
            Streams to record. Defaults to ``ring_buffer.streams`` and then
            ``streams`` from the component config.
        seconds : float, optional
            Time window to keep. Dumps drop publications older than
            ``seconds`` before the dump. When ``frames`` is omitted, each
            ring's capacity is estimated by measuring the stream's
            publication rate for ``probe_seconds`` and adding
            :data:`RING_BUFFER_RATE_HEADROOM` headroom, so a stream that
            speeds up later is held for less than ``seconds``. Pass
            ``frames`` for a fixed, predictable memory footprint.
        frames : int or dict, optional
            Ring capacity in publications, shared or per stream. Memory is
            ``frames * frame_nbytes`` per stream, allocated once here.
        probe_seconds : float, optional
            Rate-measurement window used when only ``seconds`` is given.
        poll_interval : float, optional
            How often idle readers check for a stop request.

        Returns
        -------
        dict
            Capacity in publications per stream.
        """

        component_logger = getattr(self, "logger", logger)
        if self.ring_buffer_running:
            raise RuntimeError("Telemetry ring buffer is already running; stop it first")
        ring_conf = getattr(self, "ring_buffer_conf", {}) or {}
        if streams is None:
            streams = ring_conf.get("streams") or getattr(self, "configured_streams", [])
        if isinstance(streams, str):
            stream_names = [streams]
        elif isinstance(streams, (list, tuple)):
            stream_names = [str(name) for name in streams]
        else:
            raise TypeError("streams must be a stream name or a list of stream names")
        if not stream_names:
            raise ValueError("Telemetry ring buffer has no streams to record")
        if len(set(stream_names)) != len(stream_names):
            raise ValueError("Telemetry ring buffer streams must be unique")
        if seconds is None and frames is None:
            raise ValueError("Telemetry ring buffer needs seconds=, frames=, or both")
        if seconds is not None and float(seconds) <= 0:
            raise ValueError("Telemetry ring buffer seconds must be positive")

        shms = {}
        try:
            for name in stream_names:
                shms[name] = open_stream(name, readonly=True)
            if frames is None:
                capacities = _estimate_ring_capacities(shms, float(seconds), probe_seconds)
            elif isinstance(frames, dict):
                capacities = {name: int(frames[name]) for name in stream_names}
            else:
                capacities = {name: int(frames) for name in stream_names}
            rings = [_RingStream(name, shms[name], capacities[name]) for name in stream_names]
        except Exception:
            for shm in shms.values():
                shm.close()
            raise

        self._ring_stop = threading.Event()
        self._ring_seconds = None if seconds is None else float(seconds)
        self._ring_started_at = time.time()
        for ring in rings:
            ring.thread = threading.Thread(
                target=ring.run,
                args=(self._ring_stop, float(poll_interval)),
                name=f"telemetry-ring-{ring.name}",
                daemon=True,
            )
            ring.thread.start()
        self._ring_streams = rings
        component_logger.info(
            "Started telemetry ring buffer streams=%s capacities=%s seconds=%s bytes=%d",
            stream_names,
            capacities,
            self._ring_seconds,
            sum(ring.nbytes for ring in rings),
        )
        return dict(capacities)

    def ring_buffer_status(self) -> dict:
        """Return per-stream ring statistics (capacity, fill, recorded, missed)."""

        return {ring.name: ring.status() for ring in getattr(self, "_ring_streams", None) or []}

    def dump_ring_buffer(
        self,
        unique_str: str = "",
        *,
        seconds: float | None = None,
        session_id: str | None = None,
        semantic_tags=None,
        sampling=None,
        config=None,
        config_path: str | Path | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Write the ring buffer's current contents as a telemetry session.

        Every stream's ring is locked together while it is copied, so the
        dump is one consistent cut across streams: nothing recorded after the
        cut appears in any stream and no slot is torn. Recording carries on
        afterwards (readers wait only for the copy, and publications they
        skip meanwhile are counted as missed). The session has the same
        on-disk format as :meth:`save`, so :func:`load_telemetry_session` and
        the AOTPy exporter read it unchanged. Frames are ordered oldest first.

        Parameters
        ----------
        unique_str : str, optional
            Capture label stored in each stream's metadata.
        seconds : float, optional
            Keep only publications written within ``seconds`` of the dump.
            Defaults to the ``seconds`` given at start (all frames if none).
        session_id, semantic_tags, sampling, config, config_path, metadata
            As for :meth:`save`.

        Returns
        -------
        str
            Absolute path to the created telemetry session directory.
        """

        component_logger = getattr(self, "logger", logger)
        rings = getattr(self, "_ring_streams", None)
        if not rings:
            raise ValueError("Telemetry ring buffer is not running")
        window = self._ring_seconds if seconds is None else float(seconds)

        for ring in rings:
            ring.lock.acquire()
        try:
            cut_time = time.time()
            snapshots = [ring.snapshot_locked() for ring in rings]
        finally:
            for ring in reversed(rings):
                ring.lock.release()

        if window is not None:
            for snapshot in snapshots:
                keep = snapshot["timestamps"] >= cut_time - window
                for key in ("frames", "timestamps", "frame_ids", "counts"):
                    snapshot[key] = snapshot[key][keep]

        stream_names = [ring.name for ring in rings]
        sizes = {name: max(1, len(snap["counts"])) for name, snap in zip(stream_names, snapshots)}
        specs = _normalize_stream_specs(
            stream_names, sizes, semantic_tags=semantic_tags, sampling=sampling
        )
        session_id = session_id or uuid.uuid4().hex
        session_dir = _build_session_directory(self.data_dir, session_id)
        session_dir.mkdir(parents=True, exist_ok=False)
        try:
            stream_records = []
            for ring, spec, snapshot in zip(rings, specs, snapshots):
                stream_dir = session_dir / _sanitize_name(ring.name)
                stream_dir.mkdir(parents=True, exist_ok=False)
                frames_path = stream_dir / "frames.npy"
                np.save(frames_path, snapshot["frames"])
                np.save(stream_dir / "timestamps.npy", snapshot["timestamps"])
                np.save(stream_dir / "frame_ids.npy", snapshot["frame_ids"])
                np.save(stream_dir / "counts.npy", snapshot["counts"])
                counts = snapshot["counts"]
                frame_count = int(len(counts))
                # Gaps in the publication count inside the dumped window.
                missed = int(counts[-1]) - int(counts[0]) + 1 - frame_count if frame_count else 0
                if frame_count == 0:
                    component_logger.warning(
                        "Telemetry ring buffer dump has no frames for '%s'", ring.name
                    )
                if snapshot["error"] is not None:
                    component_logger.warning(
                        "Telemetry ring buffer reader for '%s' failed: %s",
                        ring.name,
                        snapshot["error"],
                    )
                stream_records.append(
                    _write_stream_metadata(
                        session_dir,
                        stream_dir,
                        spec=spec,
                        dtype=ring.dtype,
                        frame_shape=ring.frame_shape,
                        frame_count=frame_count,
                        missed_frames=missed,
                        unique_str=unique_str,
                        extra_metadata={
                            "ring_buffer": {
                                "capacity": ring.capacity,
                                "seconds": window,
                                "started_at": self._ring_started_at,
                                "dumped_at": cut_time,
                                "recorded_total": snapshot["recorded"],
                                "missed_total": snapshot["missed"],
                                "reader_error": snapshot["error"],
                            }
                        },
                    )
                )
                self._record_saved_stream(frames_path, ring.dtype, ring.frame_shape)

            self._finish_session(
                session_dir,
                session_id=session_id,
                stream_records=stream_records,
                config=config,
                config_path=config_path,
                metadata=metadata,
            )
            component_logger.info(
                "Dumped telemetry ring buffer session %s streams=%s frames=%s path=%s",
                session_id,
                stream_names,
                [record["frame_count"] for record in stream_records],
                self.most_recent_save,
            )
            return self.most_recent_save
        except Exception:
            component_logger.exception("Failed to dump telemetry ring buffer %s", stream_names)
            raise

    def stop_ring_buffer(self, timeout: float | None = 5.0) -> None:
        """Stop recording, join the readers, and release the ring memory.

        Dump first if the contents are needed. Safe to call when no ring
        buffer is running.
        """

        rings = getattr(self, "_ring_streams", None)
        if not rings:
            return
        self._ring_stop.set()
        for ring in rings:
            if ring.thread is not None:
                ring.thread.join(timeout)
                if ring.thread.is_alive():
                    getattr(self, "logger", logger).warning(
                        "Telemetry ring buffer reader for '%s' did not stop in time", ring.name
                    )
                    continue
            ring.shm.close()
        self._ring_streams = []
        getattr(self, "logger", logger).info(
            "Stopped telemetry ring buffer streams=%s", [ring.name for ring in rings]
        )

    def start(self):
        """Start the component and the configured ring buffer, if any."""

        super().start()
        ring_conf = getattr(self, "ring_buffer_conf", None)
        if ring_conf and ring_conf.get("autostart", True) and not self.ring_buffer_running:
            self.start_ring_buffer(
                ring_conf.get("streams"),
                seconds=ring_conf.get("seconds"),
                frames=ring_conf.get("frames"),
                probe_seconds=ring_conf.get("probe_seconds", 1.0),
            )

    def stop(self):
        """Stop the ring buffer (discarding its contents) and the component."""

        self.stop_ring_buffer()
        super().stop()

    def read(self, filename="", dtype=None, *, mmap_mode=None):
        """Read a telemetry save, one saved NumPy capture file, or a raw binary file.

        Parameters
        ----------
        filename : str, optional
            Path to a telemetry session directory, a ``session.json`` file, or
            a ``frames.npy`` file. When omitted, the most recent saved frames
            file is used.
        dtype : dtype, optional
            Raw dtype for non-NumPy binary files. This keeps backward
            compatibility with older ad-hoc telemetry files.
        mmap_mode : str, optional
            NumPy memmap mode passed through to :func:`numpy.load`.

        Returns
        -------
        numpy.ndarray or dict
            Returns a NumPy array for direct frame-file reads and a per-stream
            telemetry mapping for session-directory or ``session.json`` reads.
        """

        component_logger = getattr(self, "logger", logger)
        try:
            if filename == "":
                filename = self.most_recent_file
            if not filename:
                raise ValueError("No telemetry file available to read")

            path = Path(filename)
            if path.is_dir() or path.name == "session.json":
                payload = load_telemetry_session(path, mmap_mode=mmap_mode)
                component_logger.info("Read telemetry session from %s", path)
                return payload
            if path.suffix == ".npy":
                arr = np.load(path, mmap_mode=mmap_mode)
                component_logger.info("Read telemetry capture from %s", path)
                return arr
            if dtype is not None:
                arr = np.fromfile(path, dtype=dtype)
                component_logger.info("Read raw telemetry file %s with dtype=%s", path, dtype)
                return arr
            raise ValueError("File not part of current capture, please provide a dtype")
        except Exception:
            component_logger.exception(
                "Failed to read telemetry file %s",
                filename or getattr(self, "most_recent_file", ""),
            )
            raise

    def read_last_save(self, *, mmap_mode=None) -> dict:
        """Load the most recent telemetry save into a per-stream mapping.

        Returns
        -------
        dict
            Mapping such as ``data['wfs']['frames']`` and
            ``data['wfs']['timestamps']``. Session-level metadata is stored
            under ``data['_session']``.
        """

        if not self.most_recent_save:
            raise ValueError("No telemetry save is available to read")
        return load_telemetry_session(self.most_recent_save, mmap_mode=mmap_mode)

    def read_session(self, session_path: str | Path = "", *, mmap_mode=None) -> dict:
        """Load one telemetry save by path.

        When ``session_path`` is omitted, the most recent save is used.
        """

        target = session_path or self.most_recent_save
        if not target:
            raise ValueError("No telemetry session available to read")
        return load_telemetry_session(target, mmap_mode=mmap_mode)

    def list_sessions(self) -> list[str]:
        """Return all telemetry session directories under ``data_dir``."""

        return list_telemetry_sessions(self.data_dir)
