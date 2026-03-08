"""Convenience launcher for a default set of viewer windows."""

import argparse
import subprocess

from pyRTC.logging_utils import add_logging_cli_args, configure_logging_from_args


DEFAULT_VIEW_COMMANDS = [
    ["pyrtc-view", "wfs", "signal2D", "wfc2D", "psfShort", "psfLong", "--geometry", "2x3"],
]


def launch_all(commands=None, popen_fn=subprocess.Popen):
    """Spawn the configured viewer commands and return the child processes."""

    if commands is None:
        commands = DEFAULT_VIEW_COMMANDS
    processes = []
    for cmd in commands:
        processes.append(popen_fn(cmd))
    return processes


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch default pyRTC viewer windows.")
    add_logging_cli_args(parser)
    return parser


def main(argv=None) -> int:
    """Run the default viewer-launch workflow from the command line."""

    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    logger = configure_logging_from_args(args, app_name="pyrtc-view-launch-all", component_name="viewer")
    logger.info("Launching default viewer commands")
    launch_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
