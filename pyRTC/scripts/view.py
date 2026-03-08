import argparse
import sys

from pyRTC import utils

from .viewer_helpers import (
    compute_window_size as _compute_window_size,
    normalize_frame as _normalize_frame,
    read_shm_metadata as _read_shm_metadata,
    resolve_grid as _resolve_grid,
    split_targets_and_limits as _split_targets_and_limits,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="View a pyRTC shared-memory stream in real time.")
    parser.add_argument(
        "items",
        nargs="+",
        help=(
            "One or more SHM names to display. For backward compatibility you can still append "
            "optional vmin and vmax values at the end, for example: pyrtc-view signal2D -1 1"
        ),
    )
    parser.add_argument("--fps", type=int, default=30, help="Refresh rate in frames per second")
    parser.add_argument("--affinity", type=int, default=0, help="CPU affinity index")
    parser.add_argument(
        "--geometry",
        default="square",
        help="Layout for multiple SHMs: square, row, column, or an explicit grid like 2x3",
    )
    parser.add_argument(
        "--pixel-scale",
        type=float,
        default=12.0,
        help="Approximate window pixels per SHM pixel when auto-sizing the viewer window",
    )
    parser.add_argument(
        "--theme",
        choices=["dark", "light"],
        default="dark",
        help="Viewer theme",
    )
    return parser


def main(argv=None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    try:
        from .viewer_core import launch_mosaic_viewer
    except ImportError as exc:
        raise SystemExit(
            "pyrtc-view requires viewer dependencies. Install with: pip install pyRTC[viewer]"
        ) from exc

    try:
        shm_names, static_vmin, static_vmax = _split_targets_and_limits(args.items)
        _resolve_grid(len(shm_names), args.geometry)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    utils.set_affinity(args.affinity)

    return launch_mosaic_viewer(
        sys.argv,
        shm_names,
        args.fps,
        args.geometry,
        args.pixel_scale,
        static_vmin,
        static_vmax,
        args.theme,
    )


if __name__ == "__main__":
    raise SystemExit(main())
