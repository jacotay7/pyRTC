import argparse

from pyRTC import Pipeline
from pyRTC.logging_utils import add_logging_cli_args, configure_logging_from_args


DEFAULT_SHM_NAMES = [
    "wfs",
    "wfsRaw",
    "wfc",
    "wfc2D",
    "signal",
    "signal2D",
    "psfShort",
    "psfLong",
]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clear pyRTC shared-memory objects.")
    parser.add_argument(
        "shms",
        nargs="*",
        default=DEFAULT_SHM_NAMES,
        help="Optional SHM names to clear; defaults to standard pyRTC streams",
    )
    add_logging_cli_args(parser)
    return parser


def main(argv=None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    logger = configure_logging_from_args(args, app_name="pyrtc-clear-shms", component_name="clear_shms")
    logger.info("Clearing SHMs: %s", args.shms)
    Pipeline.clear_shms(args.shms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
