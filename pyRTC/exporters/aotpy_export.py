"""AOTPy export helpers for pyRTC telemetry sessions.

The exporter is deliberately session-oriented and stays outside the real-time
path. It maps the self-describing telemetry sessions produced by
``pyRTC.Telemetry`` into partial ``aotpy.AOSystem`` objects while keeping any
assumptions explicit in metadata.

Current mapping priorities are:

- ``wfs`` streams become detector pixel-intensity sequences on a wavefront sensor
- ``signal`` streams become WFS measurements when they can be interpreted safely
- ``wfc`` streams become loop-command sequences on a deformable mirror
- ``psfShort`` and ``psfLong`` become scoring-camera detector sequences

Any session metadata or stream details that do not map cleanly into AOTPy are
preserved as AO-system metadata rather than being discarded silently.
"""

from __future__ import annotations

import importlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from pyRTC.Telemetry import load_telemetry_session
from pyRTC.config_schema import read_system_config


AOTPY_OPTIONAL_DEPENDENCY_MESSAGE = (
    "AOTPy export requires the optional 'aotpy' dependency. "
    "Install it with 'pip install pyrtcao[aotpy]' or 'pip install aotpy'."
)


def _import_aotpy():
    try:
        return importlib.import_module("aotpy")
    except ImportError as exc:
        raise RuntimeError(AOTPY_OPTIONAL_DEPENDENCY_MESSAGE) from exc


def _slug(value: str) -> str:
    sanitized = "".join(ch if ch.isalnum() else "_" for ch in str(value))
    sanitized = sanitized.strip("_")
    return sanitized or "PYRTC"


def _json_string(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _as_scalar_string(value: Any) -> str:
    if value is None or isinstance(value, (str, int, float, bool)):
        return str(value)
    return _json_string(value)


def _infer_mode(explicit_mode: str | None, resolved_config: dict | None) -> str:
    if explicit_mode:
        return str(explicit_mode).upper()

    metadata = (resolved_config or {}).get("metadata", {})
    for key in ("aoMode", "ao_mode"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().upper()
    return "SCAO"


def _resolve_config(manifest: dict, explicit_config: dict | None) -> dict | None:
    if explicit_config is not None:
        return explicit_config

    config = manifest.get("config")
    if isinstance(config, dict):
        return config

    config_path = manifest.get("config_path")
    if isinstance(config_path, str) and config_path.strip():
        path = Path(config_path).expanduser()
        if path.exists():
            try:
                return read_system_config(path, validate=False)
            except Exception:
                return None
    return None


def _session_time_bounds(loaded_session: dict) -> tuple[datetime | None, datetime | None]:
    timestamps = []
    for stream_name, payload in loaded_session.items():
        if stream_name == "_session":
            continue
        stream_timestamps = np.asarray(payload.get("timestamps", []), dtype=np.float64)
        if stream_timestamps.size == 0:
            continue
        finite = stream_timestamps[np.isfinite(stream_timestamps)]
        if finite.size:
            timestamps.append((float(np.min(finite)), float(np.max(finite))))

    if not timestamps:
        return None, None

    start = min(item[0] for item in timestamps)
    end = max(item[1] for item in timestamps)
    return (
        datetime.fromtimestamp(start, tz=timezone.utc),
        datetime.fromtimestamp(end, tz=timezone.utc),
    )


def _estimate_framerate(timestamps) -> float | None:
    values = np.asarray(timestamps, dtype=np.float64)
    if values.size < 2:
        return None

    diffs = np.diff(values)
    diffs = diffs[np.isfinite(diffs)]
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return None
    return float(1.0 / np.median(diffs))


def _build_time(aotpy, uid: str, timestamps) -> Any:
    values = np.asarray(timestamps, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return aotpy.Time(
        uid=uid,
        timestamps=finite.tolist(),
        frame_numbers=list(range(int(finite.size))),
    )


def _stream_record_by_name(manifest: dict, stream_name: str) -> dict:
    for record in manifest.get("streams", []):
        if record.get("name") == stream_name:
            return record
    raise KeyError(f"Unknown telemetry stream '{stream_name}'")


def _stream_tags(record: dict) -> set[str]:
    tags = {str(tag).strip().lower() for tag in record.get("semantic_tags", []) if str(tag).strip()}
    name = str(record.get("name", "")).strip().lower()
    if name:
        tags.add(name)
    return tags


def _classify_streams(manifest: dict) -> dict[str, Any]:
    roles = {
        "wfs": None,
        "signal": None,
        "wfc": None,
        "science": [],
        "unmapped": [],
    }

    exact_rank = {
        "wfs": {"wfs": 0, "wfsraw": 1},
        "signal": {"signal": 0, "signal2d": 1},
        "wfc": {"wfc": 0, "wfc2d": 1},
    }

    def _rank(role_name: str, record: dict) -> int:
        name = str(record.get("name", "")).strip().lower()
        return exact_rank.get(role_name, {}).get(name, 99)

    for record in manifest.get("streams", []):
        tags = _stream_tags(record)
        name = str(record.get("name", "")).strip().lower()

        if {"psf", "science", "camera", "scicam"} & tags or name.startswith("psf"):
            roles["science"].append(record["name"])
            continue

        if {"signal", "slopes", "measurement", "measurements"} & tags or name.startswith("signal"):
            current = roles["signal"]
            if current is None or _rank("signal", record) < _rank("signal", _stream_record_by_name(manifest, current)):
                roles["signal"] = record["name"]
            continue

        if {"wfc", "control", "command", "commands"} & tags or name.startswith("wfc"):
            current = roles["wfc"]
            if current is None or _rank("wfc", record) < _rank("wfc", _stream_record_by_name(manifest, current)):
                roles["wfc"] = record["name"]
            continue

        if {"wfs", "wavefront_sensor", "pixels", "image", "detector"} & tags or name in {"wfs", "wfsraw"}:
            current = roles["wfs"]
            if current is None or _rank("wfs", record) < _rank("wfs", _stream_record_by_name(manifest, current)):
                roles["wfs"] = record["name"]
            continue

        roles["unmapped"].append(record["name"])

    return roles


def _metadatum(aotpy, key: str, value: Any, comment: str | None = None):
    return aotpy.Metadatum(key.upper(), _as_scalar_string(value), comment)


def _image_metadata(aotpy, stream_name: str, entry: dict, record: dict) -> list:
    metadata = [
        _metadatum(aotpy, "PRTCSTRM", stream_name),
        _metadatum(aotpy, "PRTCDTYP", entry["metadata"].get("dtype")),
        _metadatum(aotpy, "PRTCSHAP", entry["metadata"].get("shape")),
        _metadatum(aotpy, "PRTCFCNT", entry["metadata"].get("frame_count")),
    ]
    if record.get("semantic_tags"):
        metadata.append(_metadatum(aotpy, "PRTCTAGS", record.get("semantic_tags")))
    if entry["metadata"].get("sampling") is not None:
        metadata.append(_metadatum(aotpy, "PRTCSAMP", entry["metadata"].get("sampling")))
    if entry["metadata"].get("capture_label") is not None:
        metadata.append(_metadatum(aotpy, "PRTCLABL", entry["metadata"].get("capture_label")))
    return metadata


def _build_image(aotpy, *, stream_name: str, entry: dict, record: dict, image_name: str, unit: str | None = None):
    timestamps = np.asarray(entry["timestamps"], dtype=np.float64)
    time = _build_time(aotpy, f"{_slug(stream_name).upper()}_TIME", timestamps)
    return aotpy.Image(
        image_name,
        np.asarray(entry["frames"]),
        unit=unit,
        time=time,
        metadata=_image_metadata(aotpy, stream_name, entry, record),
    )


def _control_gain_image(aotpy, loop_conf: dict | None):
    if not isinstance(loop_conf, dict):
        return None
    gain = loop_conf.get("gain")
    if gain is None:
        return None
    return aotpy.Image("PYRTC_LOOP_GAIN", np.asarray([[float(gain)]], dtype=np.float64))


def _build_wfs(aotpy, *, uid_prefix: str, telescope, resolved_config: dict | None, wfs_entry: dict | None, signal_entry: dict | None, manifest: dict):
    if wfs_entry is None and signal_entry is None:
        return None, None

    wfs_record = _stream_record_by_name(manifest, wfs_entry["metadata"]["name"]) if wfs_entry is not None else None
    signal_record = _stream_record_by_name(manifest, signal_entry["metadata"]["name"]) if signal_entry is not None else None
    slopes_conf = (resolved_config or {}).get("slopes", {})
    slopes_type = str(slopes_conf.get("type", "SHWFS")).strip().lower()

    source = aotpy.NaturalGuideStar(uid=f"{uid_prefix}_NGS")
    detector = None
    if wfs_entry is not None and wfs_record is not None:
        detector = aotpy.Detector(
            uid=f"{uid_prefix}_WFS_DETECTOR",
            pixel_intensities=_build_image(
                aotpy,
                stream_name=wfs_entry["metadata"]["name"],
                entry=wfs_entry,
                record=wfs_record,
                image_name="PYRTC_WFS_PIXELS",
                unit="adu",
            ),
            frame_rate=_estimate_framerate(wfs_entry["timestamps"]),
        )

    measurements = None
    ref_measurements = None
    n_valid_subapertures = 0
    dimensions = 1
    signal_frames = None
    if signal_entry is not None and signal_record is not None:
        signal_frames = np.asarray(signal_entry["frames"])
        if signal_frames.ndim >= 2:
            flattened = signal_frames.reshape(signal_frames.shape[0], -1)
        else:
            flattened = signal_frames.reshape(signal_frames.shape[0], 1)

        if slopes_type == "shwfs" and flattened.shape[1] % 2 == 0:
            dimensions = 2
            n_valid_subapertures = flattened.shape[1] // 2
            measurements_data = flattened.reshape(flattened.shape[0], 2, n_valid_subapertures)
            measurements = aotpy.Image(
                "PYRTC_SIGNAL_MEASUREMENTS",
                measurements_data,
                time=_build_time(aotpy, f"{uid_prefix}_MEASUREMENTS_TIME", signal_entry["timestamps"]),
                metadata=_image_metadata(aotpy, signal_entry["metadata"]["name"], signal_entry, signal_record),
            )
            ref_measurements = aotpy.Image(
                "PYRTC_REFERENCE_MEASUREMENTS",
                np.zeros((2, n_valid_subapertures), dtype=measurements_data.dtype),
                metadata=[_metadatum(aotpy, "PRTCGEN", True)],
            )
        else:
            dimensions = int(flattened.shape[1]) if flattened.ndim == 2 else 1
            n_valid_subapertures = 1
            measurements = aotpy.Image(
                "PYRTC_SIGNAL_MEASUREMENTS",
                flattened[:, np.newaxis, :],
                time=_build_time(aotpy, f"{uid_prefix}_MEASUREMENTS_TIME", signal_entry["timestamps"]),
                metadata=_image_metadata(aotpy, signal_entry["metadata"]["name"], signal_entry, signal_record),
            )

    if slopes_type == "pywfs":
        wfs = aotpy.Pyramid(
            uid=f"{uid_prefix}_PYRAMID_WFS",
            source=source,
            n_valid_subapertures=n_valid_subapertures,
            n_sides=int(slopes_conf.get("nSides", 4)),
            dimensions=dimensions,
            measurements=measurements,
            ref_measurements=ref_measurements,
            detector=detector,
        )
    else:
        wfs = aotpy.ShackHartmann(
            uid=f"{uid_prefix}_SHWFS",
            source=source,
            n_valid_subapertures=n_valid_subapertures,
            measurements=measurements,
            ref_measurements=ref_measurements,
            detector=detector,
        )
    return source, wfs


def _build_wfc(aotpy, *, uid_prefix: str, telescope, resolved_config: dict | None, wfc_entry: dict | None, manifest: dict):
    if wfc_entry is None:
        return None, None

    record = _stream_record_by_name(manifest, wfc_entry["metadata"]["name"])
    wfc_frames = np.asarray(wfc_entry["frames"])
    if wfc_frames.ndim >= 2:
        flattened = wfc_frames.reshape(wfc_frames.shape[0], -1)
    else:
        flattened = wfc_frames.reshape(wfc_frames.shape[0], 1)

    wfc_conf = (resolved_config or {}).get("wfc", {})
    actuator_count = int(flattened.shape[1])
    dm = aotpy.DeformableMirror(
        uid=f"{uid_prefix}_DM",
        telescope=telescope,
        n_valid_actuators=actuator_count,
        actuator_coordinates=[aotpy.Coordinates(float(index), 0.0) for index in range(actuator_count)],
    )
    commands = aotpy.Image(
        "PYRTC_WFC_COMMANDS",
        flattened,
        time=_build_time(aotpy, f"{uid_prefix}_COMMAND_TIME", wfc_entry["timestamps"]),
        metadata=_image_metadata(aotpy, wfc_entry["metadata"]["name"], wfc_entry, record)
        + [_metadatum(aotpy, "PRTCNMOD", wfc_conf.get("numModes"))],
    )
    return dm, commands


def _build_scoring_cameras(aotpy, *, uid_prefix: str, science_names: list[str], loaded_session: dict, manifest: dict):
    cameras = []
    for index, stream_name in enumerate(science_names, start=1):
        entry = loaded_session[stream_name]
        record = _stream_record_by_name(manifest, stream_name)
        detector = aotpy.Detector(
            uid=f"{uid_prefix}_SCIENCE_DETECTOR_{index}",
            pixel_intensities=_build_image(
                aotpy,
                stream_name=stream_name,
                entry=entry,
                record=record,
                image_name=f"PYRTC_{_slug(stream_name).upper()}",
                unit="adu",
            ),
            frame_rate=_estimate_framerate(entry["timestamps"]),
        )
        cameras.append(
            aotpy.ScoringCamera(
                uid=f"{uid_prefix}_SCIENCE_CAMERA_{index}",
                detector=detector,
            )
        )
    return cameras


def telemetry_session_to_aotpy(
    session_path: str | Path,
    *,
    system_name: str | None = None,
    ao_mode: str | None = None,
    config: dict | None = None,
):
    """Convert one pyRTC telemetry session into an ``aotpy.AOSystem``.

    Parameters
    ----------
    session_path : str or Path
        Telemetry session directory or ``session.json`` path.
    system_name : str, optional
        Override for the exported AO-system name.
    ao_mode : str, optional
        Explicit AOTPy AO mode such as ``SCAO``.
    config : dict, optional
        Explicit config payload to use instead of the embedded session config.

    Returns
    -------
    aotpy.AOSystem
        Export-ready AOTPy system object.
    """

    aotpy = _import_aotpy()
    loaded_session = load_telemetry_session(session_path)
    manifest = loaded_session["_session"]
    resolved_config = _resolve_config(manifest, config)
    roles = _classify_streams(manifest)

    exported_name = (
        system_name
        or manifest.get("metadata", {}).get("name")
        or (resolved_config or {}).get("metadata", {}).get("name")
        or f"pyRTC_{manifest['session_id'][:8]}"
    )
    uid_prefix = _slug(exported_name).upper()
    date_beginning, date_end = _session_time_bounds(loaded_session)

    system = aotpy.AOSystem(
        ao_mode=_infer_mode(ao_mode, resolved_config),
        name=exported_name,
        date_beginning=date_beginning,
        date_end=date_end,
        config=_json_string(resolved_config) if resolved_config is not None else None,
    )
    system.main_telescope = aotpy.MainTelescope(uid=f"{uid_prefix}_MAIN_TELESCOPE")

    system.metadata.extend(
        [
            _metadatum(aotpy, "PRTCSID", manifest.get("session_id")),
            _metadatum(aotpy, "PRTCSVER", manifest.get("schema_version")),
            _metadatum(aotpy, "PRTCREAT", manifest.get("created_at")),
            _metadatum(aotpy, "PRTCVER", manifest.get("pyRTC_version")),
            _metadatum(aotpy, "PRTCPATH", manifest.get("config_path")),
            _metadatum(aotpy, "PRTCSTRS", [record["name"] for record in manifest.get("streams", [])]),
            _metadatum(aotpy, "PRTCUNMP", roles["unmapped"]),
            _metadatum(aotpy, "PRTCXPRT", "pyRTC.exporters.aotpy_export"),
        ]
    )
    if manifest.get("metadata"):
        system.metadata.append(_metadatum(aotpy, "PRTCMETA", manifest.get("metadata")))
    if manifest.get("host"):
        system.metadata.append(_metadatum(aotpy, "PRTCHOST", manifest.get("host")))

    wfs_entry = loaded_session.get(roles["wfs"]) if roles["wfs"] else None
    signal_entry = loaded_session.get(roles["signal"]) if roles["signal"] else None
    source, wfs = _build_wfs(
        aotpy,
        uid_prefix=uid_prefix,
        telescope=system.main_telescope,
        resolved_config=resolved_config,
        wfs_entry=wfs_entry,
        signal_entry=signal_entry,
        manifest=manifest,
    )
    if source is not None:
        system.sources.append(source)
    if wfs is not None:
        system.wavefront_sensors.append(wfs)

    wfc_entry = loaded_session.get(roles["wfc"]) if roles["wfc"] else None
    corrector, commands = _build_wfc(
        aotpy,
        uid_prefix=uid_prefix,
        telescope=system.main_telescope,
        resolved_config=resolved_config,
        wfc_entry=wfc_entry,
        manifest=manifest,
    )
    if corrector is not None:
        system.wavefront_correctors.append(corrector)

    if wfs is not None and corrector is not None:
        timestamps = wfc_entry["timestamps"] if wfc_entry is not None else signal_entry["timestamps"]
        system.loops.append(
            aotpy.ControlLoop(
                uid=f"{uid_prefix}_CONTROL_LOOP",
                input_sensor=wfs,
                commanded_corrector=corrector,
                time=_build_time(aotpy, f"{uid_prefix}_LOOP_TIME", timestamps),
                commands=commands,
                framerate=_estimate_framerate(timestamps),
                time_filter_num=_control_gain_image(aotpy, (resolved_config or {}).get("loop")),
            )
        )

    system.scoring_cameras.extend(
        _build_scoring_cameras(
            aotpy,
            uid_prefix=uid_prefix,
            science_names=roles["science"],
            loaded_session=loaded_session,
            manifest=manifest,
        )
    )

    return system


def export_telemetry_session_to_aotpy(
    session_path: str | Path,
    output_path: str | Path,
    *,
    system_name: str | None = None,
    ao_mode: str | None = None,
    config: dict | None = None,
    file_type: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Export one pyRTC telemetry session to an on-disk AOTPy file."""

    output = Path(output_path).expanduser()
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {output}")

    system = telemetry_session_to_aotpy(
        session_path,
        system_name=system_name,
        ao_mode=ao_mode,
        config=config,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    system.write_to_file(output, file_type=file_type)
    return output.resolve()