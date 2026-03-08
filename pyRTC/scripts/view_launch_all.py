import argparse
import subprocess


DEFAULT_VIEW_COMMANDS = [
    ["pyrtc-view", "wfs", "signal2D", "wfc2D", "psfShort", "psfLong", "--geometry", "2x3"],
]


def launch_all(commands=None, popen_fn=subprocess.Popen):
    if commands is None:
        commands = DEFAULT_VIEW_COMMANDS
    processes = []
    for cmd in commands:
        processes.append(popen_fn(cmd))
    return processes


def _build_arg_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Launch default pyRTC viewer windows.")


def main(argv=None) -> int:
    parser = _build_arg_parser()
    parser.parse_args(argv)
    launch_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
