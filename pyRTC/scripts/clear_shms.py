import argparse

from pyRTC import Pipeline


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
    return parser


def main(argv=None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    Pipeline.clear_shms(args.shms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
