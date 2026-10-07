"""Versioned calibration files and the migration of pyrtc 1.x calibrations.

pyrtc 2.0 changed two conventions that saved calibrations depend on:

- Image streams are ``(height, width)``, indexed ``[y, x]`` with x along the
  columns (aocore CONVENTIONS 1.1, #162). pyrtc 1.x declared them
  ``(width, height)``, and some camera adapters (GenICam, Micro-Manager)
  transposed each frame into that shape. Slope x/y now follow the camera's
  columns/rows.
- Shack-Hartmann sub-aperture pixel ``k`` sits at ``k - (n - 1) / 2``
  (CONVENTIONS 1.2, #163), not ``k - n // 2``. Raw slopes of even-sized
  sub-apertures moved by +0.5 px; residual slopes did not.

Calibration files written since 2.0 are ``.npz`` archives holding the array
(``data``) and a JSON record (``pyrtc_calibration``) with the format version
and the kind of calibration. They keep whatever file name the config gives,
and ``np.load`` opens them (it returns an ``NpzFile``; the array is
``["data"]``). :func:`load_calibration` reads both kinds of file.

A file without the record is a *legacy* (pyrtc 1.x) file. Components refuse
legacy files unless their config section sets ``legacy_calibration``, which
says how the file was made:

``"yx"``
    The camera adapter published frames unchanged (XIMEA, Spinnaker, OOPAO,
    HCIPy, SPECULA, the synthetic simulators). Arrays load as they are;
    SHWFS reference slopes of even sub-apertures move by +0.5 px (#163).
``"xy"``
    The adapter transposed frames into ``(width, height)`` streams
    (GenICam, Micro-Manager, or an adapter of your own that applied
    ``frame.T``). Images are transposed, slope maps and valid sub-aperture
    masks swap their x/y halves and transpose each, and SHWFS reference
    slopes also move by +0.5 px. An interaction matrix needs the 1.x valid
    sub-aperture mask to reorder its rows, so it is converted offline with
    ``pyrtc-migrate-calibration`` (:func:`migrate_calibration_file`).
``"as_is"``
    The file already follows the 2.0 conventions (made by hand or by another
    tool); it loads unchanged.

Every conversion here is exact. The one case that is not, a WCoG reference
taken without a reference image (its Gaussian weight moved by half a pixel
with #163), is refused. See the "Migrating to pyrtc 2.0" docs page.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

CALIBRATION_FORMAT = 2
"""Current calibration format: ``[y, x]`` image arrays, pixel-centred SHWFS coordinates."""

METADATA_KEY = "pyrtc_calibration"
DATA_KEY = "data"
LEGACY_CALIBRATION_CHOICES = ("yx", "xy", "as_is")
WFS_TYPES = ("shwfs", "pywfs")

#: What each kind of calibration holds, and how to measure it again.
CALIBRATION_KINDS = {
    "wfs_dark": ("WFS dark frame", "wfs", "take_dark(), then save_dark()"),
    "psf_dark": ("science-camera dark frame", "psf", "take_dark(), then save_dark()"),
    "psf_model": ("model PSF", "psf", "take_model_psf(), then save_model_psf()"),
    "reference_image": (
        "SHWFS reference image",
        "slopes",
        "take_reference_image(), then save_reference_image()",
    ),
    "valid_sub_aps": (
        "valid sub-aperture mask",
        "slopes",
        "rebuild the mask and call set_valid_sub_aps(), then save_valid_sub_aps()",
    ),
    "ref_slopes": ("reference slopes", "slopes", "take_ref_slopes(), then save_ref_slopes()"),
    "interaction_matrix": ("interaction matrix", "loop", "compute_im(), then save_im()"),
}
IMAGE_KINDS = ("wfs_dark", "psf_dark", "psf_model", "reference_image")


class CalibrationError(ValueError):
    """A calibration file that pyrtc cannot use as it is."""


class LegacyCalibrationError(CalibrationError):
    """A pyrtc 1.x calibration file that was not (or cannot be) converted."""


@dataclass(frozen=True)
class Calibration:
    """A loaded calibration array and its record (``None`` for a legacy file)."""

    data: np.ndarray
    metadata: dict | None
    path: str

    @property
    def legacy(self) -> bool:
        return self.metadata is None


def _pyrtc_version() -> str:
    try:
        from pyrtc import __version__

        return str(__version__)
    except Exception:  # pragma: no cover - only during a broken install
        return "unknown"


def _check_kind(kind: str) -> str:
    if kind not in CALIBRATION_KINDS:
        raise ValueError(
            f"unknown calibration kind {kind!r}; expected one of {', '.join(CALIBRATION_KINDS)}"
        )
    return kind


def save_calibration(filename, data, kind: str, **metadata: Any) -> str:
    """Write ``data`` as a versioned calibration file and return its path.

    The file is written exactly at ``filename`` (no extension is added), as an
    ``.npz`` archive with the array under ``data`` and the JSON record under
    ``pyrtc_calibration``. ``metadata`` adds JSON-serialisable entries to the
    record. The write goes through a temporary file in the same directory, so
    a reader never sees a half-written file.
    """

    _check_kind(kind)
    path = os.fspath(filename)
    if not path:
        raise ValueError("No calibration filename provided")
    record = {
        "format": CALIBRATION_FORMAT,
        "kind": kind,
        "image_axes": "yx",
        "pyrtc_version": _pyrtc_version(),
        **metadata,
    }
    directory = os.path.dirname(os.path.abspath(path))
    handle, tmp_path = tempfile.mkstemp(prefix=".pyrtc-cal-", suffix=".npz", dir=directory)
    try:
        with os.fdopen(handle, "wb") as stream:
            np.savez(
                stream,
                **{DATA_KEY: np.asarray(data), METADATA_KEY: np.array(json.dumps(record))},
            )
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    return path


def load_calibration(filename, kind: str | None = None) -> Calibration:
    """Read a calibration file, versioned (2.0+) or legacy (a plain ``.npy``).

    Raises :class:`CalibrationError` for a file written by a newer pyrtc, or
    whose record names a different ``kind``. Legacy files come back with
    ``metadata=None``; nothing is converted here.
    """

    if kind is not None:
        _check_kind(kind)
    path = os.fspath(filename)
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.ndarray):
        return Calibration(loaded, None, path)
    with loaded as archive:
        names = list(archive.files)
        if METADATA_KEY not in names:
            if len(names) != 1:
                raise CalibrationError(
                    f"{path} is an .npz archive without a pyrtc calibration record and "
                    f"with {len(names)} arrays; expected one"
                )
            return Calibration(archive[names[0]], None, path)
        record = json.loads(str(archive[METADATA_KEY][()]))
        if DATA_KEY not in names:
            raise CalibrationError(f"{path} has a calibration record but no '{DATA_KEY}' array")
        data = archive[DATA_KEY]
    file_format = int(record.get("format", 0))
    if file_format > CALIBRATION_FORMAT:
        raise CalibrationError(
            f"{path} uses calibration format {file_format}, written by a newer pyrtc "
            f"({record.get('pyrtc_version', 'unknown')}); this pyrtc reads format "
            f"{CALIBRATION_FORMAT}. Upgrade pyrtc."
        )
    if kind is not None and record.get("kind") not in (None, kind):
        raise CalibrationError(f"{path} holds a {record.get('kind')!r} calibration, not {kind!r}")
    return Calibration(data, record, path)


def normalize_legacy_calibration(value) -> str | None:
    """Validate a ``legacy_calibration`` config value (``None`` or empty: refuse)."""

    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    lowered = str(value).strip().lower()
    if lowered not in LEGACY_CALIBRATION_CHOICES:
        raise ValueError(
            f"legacy_calibration must be one of {', '.join(LEGACY_CALIBRATION_CHOICES)} "
            f"(or unset to refuse pyrtc 1.x files), got {value!r}"
        )
    return lowered


def legacy_calibration_message(path, kind: str, reason: str | None = None) -> str:
    """Explain why a legacy file was refused and how to convert or re-measure it."""

    description, section, remeasure = CALIBRATION_KINDS[_check_kind(kind)]
    head = (
        f"{path} is a pyrtc 1.x {description} (it has no pyrtc 2.0 calibration record). "
        "pyrtc 2.0 publishes images as (height, width) = [y, x] and centres SHWFS "
        "sub-aperture pixels at k - (n - 1) / 2 (#162, #163), so a 1.x file may not "
        "match the new frame."
    )
    if reason:
        head += " " + reason
        return f"{head} Re-measure it: {remeasure}."
    return (
        f"{head} Set 'legacy_calibration' in the '{section}' config section to say how "
        "it was made, and pyrtc converts it on load: 'yx' if the camera adapter "
        "published frames unchanged (XIMEA, Spinnaker, OOPAO, HCIPy, SPECULA, the "
        "synthetic simulators), 'xy' if it transposed them (GenICam, Micro-Manager, "
        "or your own adapter applying frame.T), 'as_is' if the file already follows "
        f"the 2.0 conventions. Or re-measure it: {remeasure}. See the 'Migrating to "
        "pyrtc 2.0' docs page (pyrtc-migrate-calibration converts files offline)."
    )


def _check_frame(legacy_frame: str) -> str:
    frame = normalize_legacy_calibration(legacy_frame)
    if frame is None:
        raise ValueError("legacy_frame must be one of 'yx', 'xy', 'as_is'")
    return frame


def _check_wfs_type(wfs_type: str) -> str:
    lowered = str(wfs_type).strip().lower()
    if lowered not in WFS_TYPES:
        raise ValueError(f"wfs_type must be 'shwfs' or 'pywfs', got {wfs_type!r}")
    return lowered


def image_from_legacy(image, legacy_frame: str) -> np.ndarray:
    """Convert a 1.x image-shaped array (dark, model PSF, reference image)."""

    image = np.asarray(image)
    if _check_frame(legacy_frame) != "xy":
        return image
    if image.ndim != 2:
        raise CalibrationError(f"expected a 2-D image, got shape {image.shape}")
    return np.ascontiguousarray(image.T)


def slope_map_from_legacy(
    slope_map, legacy_frame: str, wfs_type: str = "shwfs", *, swap_pywfs_halves: bool = False
) -> np.ndarray:
    """Convert a 1.x 2-D slope map or valid sub-aperture mask (the ``signal_2d`` layout).

    SHWFS maps are ``(2N, N)``: x slopes over the ``N x N`` sub-apertures
    stacked on the y slopes. From an ``"xy"`` frame the 1.x x half was the
    camera's y, so the halves swap and each is transposed. PYWFS maps are
    ``(2R, 4R)``: the x and y slope images of the pupil side by side. From an
    ``"xy"`` frame each image is transposed; the halves also swap when the
    pupils came from the default layout (no ``pupils`` in the config), whose
    second and third pupils swapped places with the frame. ``"yx"`` and
    ``"as_is"`` maps are returned unchanged.
    """

    slope_map = np.asarray(slope_map)
    wfs_type = _check_wfs_type(wfs_type)
    if _check_frame(legacy_frame) != "xy":
        return slope_map
    if wfs_type == "shwfs":
        n = slope_map.shape[1] if slope_map.ndim == 2 else 0
        if slope_map.ndim != 2 or slope_map.shape[0] != 2 * n:
            raise CalibrationError(f"expected a (2N, N) SHWFS slope map, got {slope_map.shape}")
        first, second = slope_map[n:].T, slope_map[:n].T
        return np.ascontiguousarray(np.concatenate([first, second], axis=0))
    m = slope_map.shape[0] if slope_map.ndim == 2 else 0
    if slope_map.ndim != 2 or slope_map.shape[1] != 2 * m:
        raise CalibrationError(f"expected a (2R, 4R) PYWFS slope map, got {slope_map.shape}")
    first, second = slope_map[:, :m].T, slope_map[:, m:].T
    if swap_pywfs_halves:
        first, second = second, first
    return np.ascontiguousarray(np.concatenate([first, second], axis=1))


def shwfs_ref_slopes_from_legacy(
    ref_slopes,
    legacy_frame: str,
    sub_aperture_size: int,
    *,
    centroider: str = "cog",
    has_reference_image: bool = False,
    path: str = "reference slopes",
) -> np.ndarray:
    """Convert 1.x SHWFS reference slopes (``(2N, N)`` in pixels).

    Besides the axes (:func:`slope_map_from_legacy`), raw slopes of even
    sub-apertures moved by +0.5 px with #163, so the reference moves with
    them and residual slopes stay as they were. That holds exactly for the
    CoG and correlation centroiders and for WCoG with a reference image (the
    weights follow the reference spots). A WCoG reference taken without a
    reference image is refused: its weight was centred half a pixel off.
    """

    frame = _check_frame(legacy_frame)
    ref_slopes = np.asarray(ref_slopes, dtype=np.float32)
    if frame == "as_is":
        return ref_slopes
    even = int(sub_aperture_size) % 2 == 0
    if even and str(centroider).lower() == "wcog" and not has_reference_image:
        raise LegacyCalibrationError(
            legacy_calibration_message(
                path,
                "ref_slopes",
                reason=(
                    "A WCoG reference taken without a reference image cannot be converted "
                    "exactly: the Gaussian weight was centred half a pixel off on even "
                    "sub-apertures."
                ),
            )
        )
    converted = slope_map_from_legacy(ref_slopes, frame, "shwfs").astype(np.float32)
    if even:
        converted = converted + np.float32(0.5)
    return converted


def _scatter_signal(signal, valid: np.ndarray, wfs_type: str, fill) -> np.ndarray:
    """Place a 1-D signal into its 2-D layout (as ``SlopesProcess.compute_signal_2d``)."""

    out = np.full(valid.shape, fill, dtype=np.asarray(signal).dtype)
    if wfs_type == "pywfs":
        half = valid.shape[1] // 2
        mask = valid[:, :half]
        count = int(np.count_nonzero(mask))
        out[:, :half][mask] = signal[:count]
        out[:, half:][mask] = signal[count:]
    else:
        out[valid] = signal
    return out


def _gather_signal(slope_map: np.ndarray, valid: np.ndarray, wfs_type: str) -> np.ndarray:
    """Read the 1-D signal out of its 2-D layout (inverse of :func:`_scatter_signal`)."""

    if wfs_type == "pywfs":
        half = valid.shape[1] // 2
        mask = valid[:, :half]
        return np.concatenate([slope_map[:, :half][mask], slope_map[:, half:][mask]])
    return slope_map[valid]


def signal_permutation_from_legacy(
    old_valid_sub_aps,
    legacy_frame: str,
    wfs_type: str = "shwfs",
    *,
    swap_pywfs_halves: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(order, new_valid_sub_aps)`` for 1.x slope vectors.

    A 1.x signal vector ``old`` (one entry per valid sub-aperture slope, as
    the loop sees it) becomes ``old[order]`` in the 2.0 order, which is
    what an interaction matrix's rows need. ``new_valid_sub_aps`` is the
    converted mask. For ``"yx"`` and ``"as_is"`` the order is the identity.
    """

    wfs_type = _check_wfs_type(wfs_type)
    valid = np.asarray(old_valid_sub_aps).astype(bool)
    if wfs_type == "pywfs":
        half = valid.shape[1] // 2
        size = 2 * int(np.count_nonzero(valid[:, :half]))
    else:
        size = int(np.count_nonzero(valid))
    frame = _check_frame(legacy_frame)
    if frame != "xy":
        return np.arange(size), valid
    index_map = _scatter_signal(np.arange(size), valid, wfs_type, fill=-1)
    new_index_map = slope_map_from_legacy(
        index_map, frame, wfs_type, swap_pywfs_halves=swap_pywfs_halves
    )
    new_valid = slope_map_from_legacy(valid, frame, wfs_type, swap_pywfs_halves=swap_pywfs_halves)
    order = _gather_signal(new_index_map, new_valid, wfs_type)
    if order.size != size or not np.array_equal(np.sort(order), np.arange(size)):
        raise CalibrationError("the valid sub-aperture mask does not describe a full signal")
    return order, new_valid


def interaction_matrix_from_legacy(
    interaction_matrix,
    legacy_frame: str,
    old_valid_sub_aps=None,
    wfs_type: str = "shwfs",
    *,
    swap_pywfs_halves: bool = False,
    path: str = "interaction matrix",
) -> np.ndarray:
    """Convert a 1.x interaction matrix ``(signal_size, num_modes)``.

    Residual slopes did not change with #163, so only an ``"xy"`` matrix
    changes: its rows are reordered with the 1.x valid sub-aperture mask
    (:func:`signal_permutation_from_legacy`), which is required then.
    """

    interaction_matrix = np.asarray(interaction_matrix)
    frame = _check_frame(legacy_frame)
    if frame != "xy":
        return interaction_matrix
    if old_valid_sub_aps is None:
        raise LegacyCalibrationError(
            legacy_calibration_message(
                path,
                "interaction_matrix",
                reason=(
                    "An interaction matrix measured through a transposing adapter ('xy') "
                    "needs the 1.x valid sub-aperture mask to reorder its rows; convert it "
                    "with 'pyrtc-migrate-calibration interaction_matrix IN OUT "
                    "--legacy-frame xy --wfs-type shwfs --valid-sub-aps OLD_MASK' instead."
                ),
            )
        )
    order, _ = signal_permutation_from_legacy(
        old_valid_sub_aps, frame, wfs_type, swap_pywfs_halves=swap_pywfs_halves
    )
    if interaction_matrix.shape[0] != order.size:
        raise CalibrationError(
            f"{path} has {interaction_matrix.shape[0]} rows but the valid sub-aperture mask "
            f"describes {order.size} slopes"
        )
    return np.ascontiguousarray(interaction_matrix[order])


def load_component_calibration(
    filename,
    kind: str,
    legacy_calibration: str | None,
    convert: Callable[[np.ndarray, str], np.ndarray],
    logger=None,
) -> np.ndarray:
    """Load a calibration for a component, converting or refusing a legacy file.

    ``convert(data, legacy_frame)`` turns a legacy array into the 2.0 frame;
    it may raise :class:`LegacyCalibrationError` when no exact conversion
    exists. Without ``legacy_calibration`` a legacy file is refused.
    """

    calibration = load_calibration(filename, kind)
    if not calibration.legacy:
        return calibration.data
    frame = normalize_legacy_calibration(legacy_calibration)
    if frame is None:
        raise LegacyCalibrationError(legacy_calibration_message(calibration.path, kind))
    data = convert(calibration.data, frame)
    if logger is not None:
        logger.warning(
            "Converted pyrtc 1.x %s %s (legacy_calibration=%s); save it again to store it "
            "in the 2.0 format",
            CALIBRATION_KINDS[kind][0],
            calibration.path,
            frame,
        )
    return data


def read_calibration_array(filename, kind: str) -> np.ndarray:
    """Return the array of a 2.0 calibration file; refuse a legacy one.

    For tools that read a component's calibration files directly (the NCPA
    optimizer) and cannot know how a legacy file should be converted.
    """

    calibration = load_calibration(filename, kind)
    if calibration.legacy:
        raise LegacyCalibrationError(legacy_calibration_message(calibration.path, kind))
    return calibration.data


def migrate_calibration_file(
    source,
    destination,
    kind: str,
    legacy_frame: str,
    *,
    wfs_type: str = "shwfs",
    sub_aperture_size: int | None = None,
    centroider: str = "cog",
    has_reference_image: bool = False,
    old_valid_sub_aps=None,
    default_pupils: bool = False,
) -> str:
    """Convert a 1.x calibration file and save it in the 2.0 format.

    ``old_valid_sub_aps`` (an array or a file path) is the 1.x valid
    sub-aperture mask; an ``"xy"`` interaction matrix needs it. A SHWFS
    ``ref_slopes`` file needs ``sub_aperture_size`` (the rounded
    ``sub_ap_spacing``). ``default_pupils`` says a PYWFS used the default
    pupil layout (no ``pupils`` in its config). Returns the written path.
    """

    _check_kind(kind)
    frame = _check_frame(legacy_frame)
    wfs_type = _check_wfs_type(wfs_type)
    calibration = load_calibration(source, kind)
    if not calibration.legacy:
        raise CalibrationError(f"{calibration.path} is already a pyrtc 2.0 calibration file")
    data = calibration.data
    if isinstance(old_valid_sub_aps, (str, os.PathLike)):
        old_valid_sub_aps = load_calibration(old_valid_sub_aps).data
    swap = wfs_type == "pywfs" and bool(default_pupils)
    if kind in IMAGE_KINDS:
        converted = image_from_legacy(data, frame)
    elif kind == "valid_sub_aps":
        converted = slope_map_from_legacy(data, frame, wfs_type, swap_pywfs_halves=swap)
    elif kind == "ref_slopes":
        if wfs_type == "shwfs":
            if sub_aperture_size is None:
                raise ValueError("SHWFS reference slopes need sub_aperture_size")
            converted = shwfs_ref_slopes_from_legacy(
                data,
                frame,
                sub_aperture_size,
                centroider=centroider,
                has_reference_image=has_reference_image,
                path=calibration.path,
            )
        else:
            converted = slope_map_from_legacy(data, frame, "pywfs", swap_pywfs_halves=swap)
    else:
        converted = interaction_matrix_from_legacy(
            data,
            frame,
            old_valid_sub_aps,
            wfs_type,
            swap_pywfs_halves=swap,
            path=calibration.path,
        )
    return save_calibration(
        destination,
        converted,
        kind,
        migrated_from=os.path.basename(calibration.path),
        legacy_calibration=frame,
        wfs_type=wfs_type,
    )
