"""CLI utility for removing standard pyrtc shared-memory segments."""

import argparse

from pyrtc.logging_utils import add_logging_cli_args, configure_logging_from_args
import pyrtc.streams as streams


DEFAULT_SHM_NAMES = [
    "wfs",
    "wfs_raw",
    "wfc",
    "wfc_2d",
    "signal",
    "signal_2d",
    "psf_short",
    "psf_long",
]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clear pyrtc shared-memory objects.")
    parser.add_argument(
        "shms",
        nargs="*",
        default=DEFAULT_SHM_NAMES,
        help="Optional SHM names to clear; defaults to standard pyrtc streams",
    )
    add_logging_cli_args(parser)
    return parser


def main(argv=None) -> int:
    """Parse the requested SHM names and clear them through ``pyrtc.pipeline``."""

    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    logger = configure_logging_from_args(
        args, app_name="pyrtc-clear-shms", component_name="clear_shms"
    )
    logger.info("Clearing SHMs: %s", args.shms)
    streams.clear_shms(args.shms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
