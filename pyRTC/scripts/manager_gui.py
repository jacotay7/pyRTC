"""Command-line entrypoint for the pyRTC manager GUI."""

from __future__ import annotations

import argparse
from pathlib import Path

from pyRTC.logging_utils import add_logging_cli_args, configure_logging_from_args


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the pyRTC manager GUI.")
    parser.add_argument("config", nargs="?", help="Optional path to a pyRTC YAML config file")
    parser.add_argument("--mode", choices=["soft", "hard", "soft-rtc", "hard-rtc"], default=None, help="Optional manager mode override")
    parser.add_argument("--theme", choices=["dark", "light"], default="dark", help="Initial GUI theme")
    parser.add_argument("--refresh-ms", type=int, default=1000, help="Health and status refresh interval in milliseconds")
    add_logging_cli_args(parser)
    return parser


def main(argv=None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if getattr(args, "log_dir", None) is None and getattr(args, "log_file", None) is None:
        args.log_dir = str((Path.cwd() / "data" / "logs").resolve())
    logger = configure_logging_from_args(args, app_name="pyrtc-manager-gui", component_name="manager-gui")

    try:
        from pyRTC.gui.main_window import launch_manager_gui
    except ImportError as exc:
        logger.exception("GUI dependencies are unavailable")
        raise SystemExit(
            "pyrtc-manager-gui requires GUI dependencies. Install with: pip install pyRTC[gui]"
        ) from exc

    mode = args.mode
    if mode == "soft":
        mode = "soft-rtc"
    elif mode == "hard":
        mode = "hard-rtc"

    return launch_manager_gui(
        config_path=args.config,
        mode=mode,
        theme_name=args.theme,
        refresh_ms=max(int(args.refresh_ms), 200),
    )


if __name__ == "__main__":
    raise SystemExit(main())