"""CLI for exporting pyRTC telemetry sessions into AOTPy files."""

from __future__ import annotations

import argparse

from pyRTC.exporters.aotpy_export import export_telemetry_session_to_aotpy


def _default_output_path(session_path: str) -> str:
    from pathlib import Path

    path = Path(session_path).expanduser()
    session_dir = path.parent if path.name == "session.json" else path
    return str((session_dir.parent / f"{session_dir.name}.fits").resolve())


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a pyRTC telemetry session into an AOTPy file.")
    parser.add_argument("session", type=str, help="Path to a telemetry session directory or session.json file")
    parser.add_argument("output", nargs="?", type=str, help="Output AOTPy file path. Defaults to <session>.fits")
    parser.add_argument("--name", dest="system_name", type=str, default=None, help="Override the exported AO-system name")
    parser.add_argument("--ao-mode", dest="ao_mode", type=str, default=None, help="Override the exported AOTPy AO mode")
    parser.add_argument("--file-type", dest="file_type", type=str, default=None, help="Explicit output writer type, for example fits")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output file when it already exists")
    return parser


def main(argv=None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    output_path = args.output or _default_output_path(args.session)
    try:
        exported_path = export_telemetry_session_to_aotpy(
            args.session,
            output_path,
            system_name=args.system_name,
            ao_mode=args.ao_mode,
            file_type=args.file_type,
            overwrite=args.overwrite,
        )
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(f"AOTPy export failed: {exc}")
        return 1

    print(f"AOTPy export written: {exported_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())