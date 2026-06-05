"""Telemetry capture helpers for pyRTC shared-memory streams.

The :class:`Telemetry` component provides a small, operator-friendly API for
capturing bounded stretches of existing pyRTC streams into standard NumPy data
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
"""

from __future__ import annotations

import json
import os
import platform
import socket
import time
import uuid
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np

from pyRTC.logging_utils import get_logger
from pyRTC.Pipeline import initExistingShm
from pyRTC.pyRTCComponent import pyRTCComponent
from pyRTC.utils import setFromConfig


logger = get_logger(__name__)
TELEMETRY_SESSION_SCHEMA_VERSION = 1


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _resolve_pyrtc_version() -> str:
    try:
        return importlib_metadata.version("pyRTC")
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


def _extract_timestamp(shm) -> float:
    metadata = getattr(shm, "metadata", None)
    if metadata is not None:
        try:
            timestamp = float(metadata[1])
            if timestamp > 0:
                return timestamp
        except Exception:
            pass
    return time.time()


def _normalize_stream_specs(streams, numFrames, semanticTags=None, sampling=None) -> list[dict]:
    if isinstance(streams, str):
        stream_names = [streams]
    elif isinstance(streams, (list, tuple)):
        stream_names = [str(name) for name in streams]
    else:
        raise TypeError("streams must be a stream name or a list of stream names")

    if isinstance(numFrames, dict):
        frame_counts = {str(name): int(value) for name, value in numFrames.items()}
    else:
        frame_counts = {name: int(numFrames) for name in stream_names}

    tag_mapping = semanticTags if isinstance(semanticTags, dict) else None
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
        elif semanticTags is None:
            tags = []
        else:
            tags = semanticTags

        if tags and not isinstance(tags, (list, tuple)):
            raise TypeError("semanticTags must be a list of strings or a mapping of stream name to string lists")

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
            raise ValueError(f"Telemetry manifest {session_file} contains a non-mapping stream record")
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

    return loaded


def list_telemetry_sessions(data_dir: str | Path) -> list[str]:
    """Return all telemetry session directories under one base directory."""

    base_dir = _ensure_path(data_dir)
    if not base_dir.exists():
        return []
    return [str(path.resolve().parent) for path in sorted(base_dir.glob("session_*/session.json"))]


class Telemetry(pyRTCComponent):
    """Capture pyRTC streams into standard NumPy telemetry products.

    Parameters
    ----------
    conf : dict, optional
        Telemetry configuration. The most useful keys are:

        ``dataDir``
            Base directory used for capture output. Defaults to ``./data/``.

        ``streams``
            Optional default stream names for :meth:`save_configured_streams`.

        ``functions``
            Standard pyRTC worker-thread configuration inherited from
            :class:`pyRTC.pyRTCComponent.pyRTCComponent`.

    Notes
    -----
    The public API is intentionally small and Sphinx-friendly:

    - :meth:`save` captures one or more streams
    - :meth:`read_last_save` reopens the most recent capture
    - :meth:`list_sessions` enumerates saved captures on disk

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
            self.dataDir = Path(setFromConfig(conf, "dataDir", "./data/")).expanduser()
            self.dataDir.mkdir(parents=True, exist_ok=True)
            self.configuredStreams = list(setFromConfig(conf, "streams", []))
            self.mostRecentSave = ""
            self.mostRecentFile = ""
            self.allSaves = []
            self.allFiles = []
            self.dTypes = []
            self.dims = []
            self.logger.info("Initialized telemetry dataDir=%s configuredStreams=%s", self.dataDir, self.configuredStreams)
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
            "pyRTC_version": _resolve_pyrtc_version(),
            "host": _host_metadata(),
            "config_path": str(_ensure_path(config_path).resolve()) if config_path is not None else None,
            "config": config,
            "metadata": metadata or {},
            "streams": stream_records,
        }

    def save(
        self,
        streams,
        numFrames,
        *,
        uniqueStr="",
        sessionId: str | None = None,
        semanticTags=None,
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
        numFrames : int or dict
            Number of frames to save. When ``streams`` is a list, this may be
            one shared integer or a mapping of stream name to frame count.
        uniqueStr : str, optional
            Optional suffix used in the session metadata for operator clarity.
        sessionId : str, optional
            Explicit session identifier. A UUID is generated when omitted.
        semanticTags : list or dict, optional
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
        specs = _normalize_stream_specs(streams, numFrames, semanticTags=semanticTags, sampling=sampling)
        session_id = sessionId or uuid.uuid4().hex
        session_dir = _build_session_directory(self.dataDir, session_id)
        session_dir.mkdir(parents=True, exist_ok=False)

        try:
            stream_records = []
            last_frames_path = ""
            for spec in specs:
                shm, shm_dims, shm_dtype = initExistingShm(spec["name"])
                stream_name = spec["name"]
                frame_count = int(spec["frame_count"])
                stream_dir = session_dir / _sanitize_name(stream_name)
                stream_dir.mkdir(parents=True, exist_ok=False)

                frames_path = stream_dir / "frames.npy"
                timestamps_path = stream_dir / "timestamps.npy"
                stream_metadata_path = stream_dir / "metadata.json"
                frame_shape = _coerce_shape(shm_dims)
                dtype = np.dtype(shm_dtype)

                frames = np.lib.format.open_memmap(
                    frames_path,
                    mode="w+",
                    dtype=dtype,
                    shape=(frame_count, *frame_shape),
                )
                timestamps = np.lib.format.open_memmap(
                    timestamps_path,
                    mode="w+",
                    dtype=np.float64,
                    shape=(frame_count,),
                )

                for index in range(frame_count):
                    frame = np.asarray(shm.read(), dtype=dtype)
                    frames[index] = frame
                    timestamps[index] = _extract_timestamp(shm)

                frames.flush()
                timestamps.flush()
                del frames
                del timestamps

                stream_metadata = {
                    "name": stream_name,
                    "dtype": dtype.name,
                    "shape": list(frame_shape),
                    "frame_count": frame_count,
                    "timestamp_unit": "unix_seconds",
                    "sampling": spec["sampling"],
                    "semantic_tags": spec["semantic_tags"],
                    "capture_label": uniqueStr or None,
                }
                with stream_metadata_path.open("w", encoding="utf-8") as handle:
                    json.dump(stream_metadata, handle, indent=2, sort_keys=True)

                stream_record = {
                    "name": stream_name,
                    "dtype": dtype.name,
                    "shape": list(frame_shape),
                    "frame_count": frame_count,
                    "frames_file": str(frames_path.relative_to(session_dir)),
                    "timestamps_file": str(timestamps_path.relative_to(session_dir)),
                    "metadata_file": str(stream_metadata_path.relative_to(session_dir)),
                    "sampling": spec["sampling"],
                    "semantic_tags": spec["semantic_tags"],
                }
                stream_records.append(stream_record)

                last_frames_path = str(frames_path)
                self.allFiles.append(last_frames_path)
                self.dTypes.append(dtype)
                self.dims.append(list(frame_shape))

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

            self.mostRecentSave = str(session_dir.resolve())
            self.mostRecentFile = last_frames_path
            self.allSaves.append(self.mostRecentSave)
            component_logger.info(
                "Saved telemetry session %s streams=%s frames=%s path=%s",
                session_id,
                [spec["name"] for spec in specs],
                [spec["frame_count"] for spec in specs],
                self.mostRecentSave,
            )
            return self.mostRecentSave
        except Exception:
            component_logger.exception("Failed to save telemetry streams %s", [spec["name"] for spec in specs])
            raise

    def save_session(self, streams, numFrames, **kwargs) -> str:
        """Compatibility wrapper around :meth:`save`."""

        return self.save(streams, numFrames, **kwargs)

    def save_configured_streams(self, numFrames, **kwargs) -> str:
        """Save the streams configured on this telemetry component.

        This is a convenience wrapper for configs that already declare a fixed
        telemetry stream set.
        """

        if not self.configuredStreams:
            raise ValueError("Telemetry has no configured streams to capture")
        return self.save(self.configuredStreams, numFrames, **kwargs)

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
                filename = self.mostRecentFile
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
            component_logger.exception("Failed to read telemetry file %s", filename or getattr(self, "mostRecentFile", ""))
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

        if not self.mostRecentSave:
            raise ValueError("No telemetry save is available to read")
        return load_telemetry_session(self.mostRecentSave, mmap_mode=mmap_mode)

    def read_session(self, session_path: str | Path = "", *, mmap_mode=None) -> dict:
        """Load one telemetry save by path.

        When ``session_path`` is omitted, the most recent save is used.
        """

        target = session_path or self.mostRecentSave
        if not target:
            raise ValueError("No telemetry session available to read")
        return load_telemetry_session(target, mmap_mode=mmap_mode)

    def list_sessions(self) -> list[str]:
        """Return all telemetry session directories under ``dataDir``."""

        return list_telemetry_sessions(self.dataDir)