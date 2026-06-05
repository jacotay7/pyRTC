"""CLI for validating a pyRTC system configuration file."""

from __future__ import annotations

import argparse
import json

from pyRTC.config_schema import read_system_config
from pyRTC.utils import ConfigValidationError


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a pyRTC system configuration file.")
    parser.add_argument("config", type=str, help="Path to the YAML config file")
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=("text", "json"),
        default="text",
        help="Output format for validation results",
    )
    return parser


def _success_payload(config_path: str, normalized_conf: dict) -> dict:
    components = [section for section in ("wfs", "slopes", "loop", "wfc", "psf", "telemetry") if section in normalized_conf]
    return {
        "valid": True,
        "config": config_path,
        "mode": normalized_conf["manager"].get("mode", "soft-rtc"),
        "components": components,
    }


def _error_payload(config_path: str, error: Exception) -> dict:
    return {
        "valid": False,
        "config": config_path,
        "error": str(error),
    }


def main(argv=None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    try:
        normalized_conf = read_system_config(args.config, validate=True)
    except (ConfigValidationError, OSError, ValueError) as exc:
        payload = _error_payload(args.config, exc)
        if args.output_format == "json":
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"Config invalid: {args.config}")
            print(payload["error"])
        return 1

    payload = _success_payload(args.config, normalized_conf)
    if args.output_format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        components = ", ".join(payload["components"])
        print(f"Config valid: {args.config}")
        print(f"Mode: {payload['mode']}")
        print(f"Components: {components}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
