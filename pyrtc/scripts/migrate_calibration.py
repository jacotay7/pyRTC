"""CLI converting a pyrtc 1.x calibration file to the pyrtc 2.0 format.

See :mod:`pyrtc.calibration` and the "Migrating to pyrtc 2.0" docs page for
what changed (#162, #163) and when a conversion is exact. Examples::

    pyrtc-migrate-calibration interaction_matrix im.npy im_v2.npz \\
        --legacy-frame xy --valid-sub-aps valid_sub_aps.npy
    pyrtc-migrate-calibration ref_slopes ref.npy ref_v2.npz \\
        --legacy-frame yx --sub-aperture-size 8
"""

from __future__ import annotations

import argparse
import sys

from pyrtc.calibration import (
    CALIBRATION_KINDS,
    LEGACY_CALIBRATION_CHOICES,
    CalibrationError,
    migrate_calibration_file,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a pyrtc 1.x calibration file to the pyrtc 2.0 format."
    )
    parser.add_argument("kind", choices=tuple(CALIBRATION_KINDS), help="What the file holds")
    parser.add_argument("source", help="The 1.x file (a plain .npy)")
    parser.add_argument("destination", help="Where to write the 2.0 file")
    parser.add_argument(
        "--legacy-frame",
        required=True,
        choices=LEGACY_CALIBRATION_CHOICES,
        help=(
            "yx: the 1.x camera adapter published frames unchanged (XIMEA, Spinnaker, "
            "simulators); xy: it transposed them (GenICam, Micro-Manager); as_is: the "
            "file already follows the 2.0 conventions"
        ),
    )
    parser.add_argument("--wfs-type", choices=("shwfs", "pywfs"), default="shwfs")
    parser.add_argument(
        "--sub-aperture-size",
        type=int,
        help="SHWFS sub-aperture size in pixels (rounded sub_ap_spacing); for ref_slopes",
    )
    parser.add_argument(
        "--centroider",
        choices=("cog", "wcog", "correlation"),
        default="cog",
        help="SHWFS centroider the reference slopes were taken with",
    )
    parser.add_argument(
        "--reference-image",
        action="store_true",
        help="The WCoG reference slopes were taken with a reference image",
    )
    parser.add_argument(
        "--valid-sub-aps",
        help="The 1.x valid sub-aperture mask; an xy interaction matrix needs it",
    )
    parser.add_argument(
        "--default-pupils",
        action="store_true",
        help="The PYWFS used the default pupil layout (no 'pupils' in its config)",
    )
    return parser


def main(argv=None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        path = migrate_calibration_file(
            args.source,
            args.destination,
            args.kind,
            args.legacy_frame,
            wfs_type=args.wfs_type,
            sub_aperture_size=args.sub_aperture_size,
            centroider=args.centroider,
            has_reference_image=args.reference_image,
            old_valid_sub_aps=args.valid_sub_aps,
            default_pupils=args.default_pupils,
        )
    except (CalibrationError, OSError, ValueError) as exc:
        print(f"Cannot convert {args.source}: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
