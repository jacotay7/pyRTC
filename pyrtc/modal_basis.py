"""Modal bases for wavefront correctors, built with `aobasis`.

pyrtc builds every modal basis (mode-to-command matrix, ``M2C``) through the
`aobasis <https://github.com/jacotay7/aobasis>`_ package instead of
per-backend code. A wavefront corrector asks for a basis with a ``basis:``
section in its config::

    wfc:
      num_modes: 50
      basis:
        type: kl            # kl | zernike | fourier | zonal | zonal_fast | hadamard
        pupil_diameter: 8.0 # metres spanned by the actuator grid
        r0: 0.16            # KL only
        L0: 30.0            # KL only

This module holds the pure (stream-free) pieces: config parsing and
validation, actuator coordinates from a 2D layout mask, and the generation
and normalization of the ``(num_actuators, num_modes)`` matrix.

Coordinate convention
---------------------
aobasis takes actuator positions as an ``(N, 2)`` array of ``(x, y)`` in
metres with the pupil centre at the origin. When positions are derived from a
boolean layout mask, ``x`` is the column index and ``y`` the row index,
centred on the middle of the mask, and the first and last rows/columns of the
mask sit on the pupil edge: ``pitch = pupil_diameter / (max(mask.shape) - 1)``.
This is the Fried-geometry convention used by OOPAO and by
``aobasis.make_circular_actuator_grid``. Actuators are ordered like
``mask[mask]`` (row-major), which is the order pyrtc uses for command vectors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

BASIS_TYPES = ("kl", "zernike", "fourier", "zonal", "zonal_fast", "hadamard")
NORMALIZATIONS = ("peak", "rms", "l2", "none")
DEFAULT_PUPIL_DIAMETER = 8.0

_TYPE_ALIASES = {"zonalfast": "zonal_fast", "zonal-fast": "zonal_fast"}
_COMMON_KEYS = {
    "type",
    "n_modes",
    "normalize",
    "orthonormalize",
    "pupil_diameter",
    "positions_file",
}
_TYPE_KEYS = {
    "kl": {"r0", "L0", "ignore_piston", "use_gpu"},
    "zernike": {"ignore_piston"},
    "fourier": {"ignore_piston"},
    "zonal": set(),
    "zonal_fast": {"min_distance"},
    "hadamard": set(),
}
# Bases that are not orthogonal once sampled on a discrete actuator grid and
# are orthonormalized unless the config says otherwise (#105). Hadamard stays
# raw by default so its +/-1 patterns survive for calibration.
_ORTHONORMALIZE_BY_DEFAULT = {"zernike", "fourier"}


@dataclass(frozen=True)
class BasisConfig:
    """Normalized ``basis:`` section of a wavefront-corrector config."""

    type: str
    n_modes: int | None = None
    normalize: str = "peak"
    orthonormalize: bool = False
    pupil_diameter: float | None = None
    positions_file: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    def with_defaults(self, **defaults: Any) -> "BasisConfig":
        """Return a copy where unset fields take the given defaults.

        Backends use this to supply physical values they know, such as the
        telescope diameter, without overriding what the user configured.
        """

        values = {
            "type": self.type,
            "n_modes": self.n_modes,
            "normalize": self.normalize,
            "orthonormalize": self.orthonormalize,
            "pupil_diameter": self.pupil_diameter,
            "positions_file": self.positions_file,
            "params": dict(self.params),
        }
        for key, value in defaults.items():
            if key not in values:
                raise KeyError(f"unknown basis field '{key}'")
            if values[key] in (None, ""):
                values[key] = value
        return BasisConfig(**values)


def _positive_float(value: Any, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"basis.{key} must be a number, got {value!r}")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"basis.{key} must be positive and finite, got {value}")
    return value


def _bool(value: Any, key: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"basis.{key} must be true or false, got {value!r}")
    return value


def parse_basis_config(raw: Any, *, num_modes: int | None = None) -> BasisConfig | None:
    """Validate and normalize a ``basis:`` config section.

    Parameters
    ----------
    raw : mapping or None
        The ``basis`` value from a wavefront-corrector config. ``None`` or an
        empty mapping means no basis is configured.
    num_modes : int, optional
        The corrector's ``num_modes``. When given, ``basis.n_modes`` must match
        it (the ``wfc`` stream is sized from ``num_modes``).

    Returns
    -------
    BasisConfig or None

    Raises
    ------
    ValueError
        If the section is malformed.
    """

    if raw is None or (isinstance(raw, Mapping) and len(raw) == 0):
        return None
    if not isinstance(raw, Mapping):
        raise ValueError(f"basis must be a mapping, got {type(raw).__name__}")

    raw = dict(raw)
    basis_type = raw.get("type")
    if not isinstance(basis_type, str) or not basis_type.strip():
        raise ValueError(f"basis.type is required and must be one of {', '.join(BASIS_TYPES)}")
    basis_type = basis_type.strip().lower()
    basis_type = _TYPE_ALIASES.get(basis_type, basis_type)
    if basis_type not in BASIS_TYPES:
        raise ValueError(
            f"basis.type must be one of {', '.join(BASIS_TYPES)}, got {raw.get('type')!r}"
        )

    allowed = _COMMON_KEYS | _TYPE_KEYS[basis_type]
    unknown = sorted(str(key) for key in raw if key not in allowed)
    if unknown:
        raise ValueError(
            f"basis: unknown key(s) {', '.join(unknown)} for type '{basis_type}' "
            f"(allowed: {', '.join(sorted(allowed))})"
        )

    n_modes = raw.get("n_modes")
    if n_modes is not None:
        if isinstance(n_modes, bool) or not isinstance(n_modes, (int, np.integer)):
            raise ValueError(f"basis.n_modes must be an integer, got {n_modes!r}")
        n_modes = int(n_modes)
        if n_modes < 1:
            raise ValueError(f"basis.n_modes must be >= 1, got {n_modes}")
        if num_modes is not None and n_modes != int(num_modes):
            raise ValueError(
                f"basis.n_modes ({n_modes}) must equal num_modes ({num_modes}); "
                "the wfc stream is sized from num_modes"
            )

    normalize = raw.get("normalize", "peak")
    if normalize is None or normalize is False:
        normalize = "none"
    if not isinstance(normalize, str) or normalize.lower() not in NORMALIZATIONS:
        raise ValueError(
            f"basis.normalize must be one of {', '.join(NORMALIZATIONS)}, got {normalize!r}"
        )

    pupil_diameter = raw.get("pupil_diameter")
    if pupil_diameter is not None:
        pupil_diameter = _positive_float(pupil_diameter, "pupil_diameter")

    positions_file = raw.get("positions_file", "") or ""
    if not isinstance(positions_file, str):
        raise ValueError("basis.positions_file must be a path string")

    params: dict[str, Any] = {}
    for key in ("r0", "L0"):
        if key in raw:
            params[key] = _positive_float(raw[key], key)
    for key in ("ignore_piston", "use_gpu"):
        if key in raw:
            params[key] = _bool(raw[key], key)
    if "min_distance" in raw:
        value = raw["min_distance"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"basis.min_distance must be a non-negative number, got {value!r}")
        params["min_distance"] = float(value)

    return BasisConfig(
        type=basis_type,
        n_modes=n_modes,
        normalize=normalize.lower(),
        orthonormalize=_bool(
            raw.get("orthonormalize", basis_type in _ORTHONORMALIZE_BY_DEFAULT), "orthonormalize"
        ),
        pupil_diameter=pupil_diameter,
        positions_file=positions_file,
        params=params,
    )


def default_actuator_layout(num_actuators: int) -> np.ndarray:
    """Return a centred, approximately circular boolean layout for ``num_actuators``.

    The layout is the smallest odd square grid holding ``num_actuators`` cells,
    keeping the cells closest to the centre. pyrtc uses it when a corrector has
    no layout of its own (synthetic and display-only correctors).
    """

    if num_actuators < 1:
        raise ValueError("num_actuators must be positive")

    side = int(np.ceil(np.sqrt(float(num_actuators))))
    if side % 2 == 0:
        side += 1

    yy, xx = np.indices((side, side), dtype=np.float32)
    center = 0.5 * (side - 1)
    distances = (xx - center) ** 2 + (yy - center) ** 2
    selected = np.argsort(distances, axis=None)[:num_actuators]

    layout = np.zeros((side, side), dtype=bool)
    layout.flat[selected] = True
    return layout


def actuator_positions_from_layout(
    layout: np.ndarray, pupil_diameter: float = DEFAULT_PUPIL_DIAMETER
) -> np.ndarray:
    """Return ``(N, 2)`` actuator ``(x, y)`` positions in metres for a layout mask.

    ``x`` follows columns and ``y`` rows, the origin is the centre of the mask,
    and the pitch is ``pupil_diameter / (max(layout.shape) - 1)`` so the outer
    rows/columns of the mask lie on the pupil edge. Rows are in the order of
    ``layout[layout]``.
    """

    layout = np.asarray(layout) > 0
    if layout.ndim != 2:
        raise ValueError(f"layout must be 2D to derive actuator positions, got {layout.shape}")
    rows, cols = np.nonzero(layout)
    side = max(layout.shape)
    pitch = float(pupil_diameter) / (side - 1) if side > 1 else float(pupil_diameter)
    x = (cols - 0.5 * (layout.shape[1] - 1)) * pitch
    y = (rows - 0.5 * (layout.shape[0] - 1)) * pitch
    return np.column_stack((x, y)).astype(np.float64)


def load_actuator_positions(filename: str) -> np.ndarray:
    """Load ``(N, 2)`` actuator positions in metres from a ``.npy`` or text file."""

    if filename.endswith(".npy"):
        positions = np.load(filename)
    else:
        positions = np.loadtxt(filename, ndmin=2)
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(
            f"basis.positions_file must hold an (N, 2) array of x/y positions, got {positions.shape}"
        )
    return positions


def _nearest_neighbour_distance(positions: np.ndarray) -> float:
    if positions.shape[0] < 2:
        return 1.0
    from scipy.spatial import cKDTree

    distances, _ = cKDTree(positions).query(positions, k=2)
    spacing = distances[:, 1]
    spacing = spacing[spacing > 0]
    return float(spacing.min()) if spacing.size else 1.0


def normalize_modes(modes: np.ndarray, how: str) -> np.ndarray:
    """Scale each column of ``modes`` to unit peak, RMS, or L2 norm.

    All-zero columns are left untouched. ``how="none"`` returns the input.
    """

    how = how.lower()
    if how == "none":
        return modes
    if how == "peak":
        scale = np.max(np.abs(modes), axis=0)
    elif how == "rms":
        scale = np.sqrt(np.mean(modes**2, axis=0))
    elif how == "l2":
        scale = np.linalg.norm(modes, axis=0)
    else:
        raise ValueError(f"unknown normalization '{how}'")
    scale = np.where(scale > 0, scale, 1.0)
    return modes / scale


def generate_modes(
    basis: BasisConfig,
    positions: np.ndarray,
    n_modes: int,
    *,
    pupil_diameter: float,
) -> np.ndarray:
    """Call the matching aobasis generator and return its raw modes.

    ``positions`` are in metres. ``pupil_diameter`` sets the Zernike unit
    circle (radius ``pupil_diameter / 2``) and the Fourier fundamental
    frequency (one cycle per ``pupil_diameter``).
    """

    import aobasis

    params = basis.params
    ignore_piston = bool(params.get("ignore_piston", True))
    if basis.type == "kl":
        generator = aobasis.KLBasisGenerator(
            positions,
            fried_parameter=params.get("r0", 0.16),
            outer_scale=params.get("L0", 30.0),
            use_gpu=bool(params.get("use_gpu", False)),
        )
        return generator.generate(n_modes, ignore_piston=ignore_piston)
    if basis.type == "zernike":
        generator = aobasis.ZernikeBasisGenerator(positions, pupil_radius=0.5 * pupil_diameter)
        return generator.generate(n_modes, ignore_piston=ignore_piston)
    if basis.type == "fourier":
        generator = aobasis.FourierBasisGenerator(positions, pupil_diameter=pupil_diameter)
        return generator.generate(n_modes, ignore_piston=ignore_piston)
    if basis.type == "zonal":
        return aobasis.ZonalBasisGenerator(positions).generate(n_modes)
    if basis.type == "zonal_fast":
        min_distance = params.get("min_distance")
        if min_distance is None:
            min_distance = 2.0 * _nearest_neighbour_distance(positions)
        return aobasis.ZonalFastBasisGenerator(positions, min_distance=min_distance).generate(
            n_modes
        )
    if basis.type == "hadamard":
        return aobasis.HadamardBasisGenerator(positions).generate(n_modes)
    raise ValueError(f"unsupported basis type '{basis.type}'")


def build_m2c(
    basis: BasisConfig,
    *,
    num_modes: int,
    layout: np.ndarray | None = None,
    positions: np.ndarray | None = None,
    logger=None,
) -> np.ndarray:
    """Build a ``(num_actuators, num_modes)`` float32 M2C matrix from a basis config.

    Positions come from, in order: ``basis.positions_file``, the ``positions``
    argument (metres, supplied by a backend that knows its geometry), or the
    ``layout`` mask (see :func:`actuator_positions_from_layout`).

    The raw aobasis modes are optionally orthonormalized, then normalized per
    ``basis.normalize`` (default ``"peak"``: each mode's largest actuator
    command is 1, like a zonal poke).
    """

    if basis.n_modes is not None and basis.n_modes != int(num_modes):
        raise ValueError(f"basis.n_modes ({basis.n_modes}) must equal num_modes ({num_modes})")

    if basis.positions_file:
        positions = load_actuator_positions(basis.positions_file)
    if positions is not None:
        positions = np.asarray(positions, dtype=np.float64)
        pupil_diameter = basis.pupil_diameter
        if pupil_diameter is None:
            radius = float(np.max(np.linalg.norm(positions, axis=1))) if positions.size else 0.0
            pupil_diameter = 2.0 * radius if radius > 0 else 1.0
    elif layout is not None:
        pupil_diameter = basis.pupil_diameter or DEFAULT_PUPIL_DIAMETER
        positions = actuator_positions_from_layout(layout, pupil_diameter)
    else:
        raise ValueError("basis: no actuator positions (set a layout or basis.positions_file)")

    num_actuators = positions.shape[0]
    num_modes = int(num_modes)
    if num_modes > num_actuators:
        raise ValueError(
            f"basis: cannot build {num_modes} modes on {num_actuators} actuators "
            "(num_modes must be <= the number of actuators)"
        )

    modes = np.asarray(
        generate_modes(basis, positions, num_modes, pupil_diameter=pupil_diameter),
        dtype=np.float64,
    )
    if modes.shape != (num_actuators, num_modes):
        raise ValueError(
            f"aobasis {basis.type} returned shape {modes.shape}, "
            f"expected {(num_actuators, num_modes)}"
        )
    # Check the rank before orthonormalizing: QR of a rank-deficient matrix
    # still returns orthonormal columns, which would hide the problem.
    rank = int(np.linalg.matrix_rank(modes)) if num_modes else 0
    if basis.orthonormalize:
        from aobasis import orthonormalize_modes

        modes = orthonormalize_modes(modes)
    modes = normalize_modes(modes, basis.normalize)

    if logger is not None:
        if rank < num_modes:
            logger.warning(
                "basis %s: %s modes have rank %s on this actuator geometry; some modes "
                "are linearly dependent%s",
                basis.type,
                num_modes,
                rank,
                " (orthonormalize replaced them with arbitrary orthogonal directions)"
                if basis.orthonormalize
                else "",
            )
        logger.info(
            "Built %s basis with aobasis: actuators=%s modes=%s pupil_diameter=%.4g m",
            basis.type,
            num_actuators,
            num_modes,
            pupil_diameter,
        )
    return modes.astype(np.float32)
