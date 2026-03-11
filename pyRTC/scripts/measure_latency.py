"""Latency-measurement CLI for pyRTC shared-memory streams."""

import argparse
import json
from pathlib import Path

from pyRTC.logging_utils import add_logging_cli_args, configure_logging_from_args
from pyRTC.Pipeline import RTCManager
from pyRTC.latency import (
    format_latency_report,
    measure_stream_path_latency,
    plot_latency_histogram,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure latency between pyRTC shared-memory streams."
    )
    parser.add_argument("source_shm", nargs="?", type=str, help="Name of source SHM (earlier in pipeline)")
    parser.add_argument("target_shm", nargs="?", type=str, help="Name of target SHM (later in pipeline)")
    parser.add_argument("--config", type=str, default=None, help="System config used to infer a default latency path")
    parser.add_argument("--path", nargs="+", default=None, help="Explicit stream path to measure, e.g. --path wfs signal wfc")
    parser.add_argument("--samples", type=int, default=4096, help="Number of timestamp samples to collect")
    parser.add_argument("--tag", type=str, default="latency", help="Output tag used in plot title")
    parser.add_argument("--format", choices=("text", "json"), default="text", help="Report output format")
    parser.add_argument("--bins", type=int, default=200, help="Number of histogram bins")
    parser.add_argument(
        "--xrange",
        type=float,
        nargs=2,
        default=(1e-4, 10 ** -2.5),
        metavar=("LOW", "HIGH"),
        help="Latency histogram range in seconds",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar")
    parser.add_argument("--show-plot", action="store_true", help="Show a latency histogram window")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output figure path for a latency histogram",
    )
    add_logging_cli_args(parser)
    return parser


def _default_output_path(args) -> Path:
    source_name = args.source_shm or "path"
    target_name = args.target_shm or "path"
    safe_source = source_name.replace("/", "_")
    safe_target = target_name.replace("/", "_")
    return Path(f"jitter_{safe_source}_to_{safe_target}_{args.tag}.pdf")


def _resolve_stream_path(args):
    if args.path:
        return [str(stream_name) for stream_name in args.path], False
    if args.config:
        manager = RTCManager.from_config_file(args.config)
        report = manager.latency(
            source_shm=args.source_shm,
            target_shm=args.target_shm,
            samples=args.samples,
            show_progress=not args.no_progress,
        )
        return report, True
    if args.source_shm and args.target_shm:
        return [args.source_shm, args.target_shm], False
    raise SystemExit("Provide source_shm and target_shm, --path, or --config")


def main(argv=None) -> int:
    """Run the latency measurement workflow from the command line."""

    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    logger = configure_logging_from_args(args, app_name="pyrtc-measure-latency", component_name=args.tag)

    if args.samples < 2:
        logger.error("--samples must be at least 2")
        raise SystemExit("--samples must be at least 2")

    resolved, from_manager = _resolve_stream_path(args)
    if from_manager:
        report_payload = resolved
        total_latency = None
    else:
        report, total_latency = measure_stream_path_latency(
            resolved,
            samples=args.samples,
            show_progress=not args.no_progress,
            include_total_samples=bool(args.output or args.show_plot),
        )
        report_payload = report.to_dict()

    if args.format == "json":
        print(json.dumps(report_payload, indent=2, sort_keys=True))
    else:
        print(format_latency_report(report_payload))

    if args.output or args.show_plot:
        if total_latency is None:
            path = report_payload["stream_path"]
            _, total_latency = measure_stream_path_latency(
                path,
                samples=args.samples,
                show_progress=not args.no_progress,
                include_total_samples=True,
            )
        title = f"pyRTC Latency ({report_payload['source_shm']} -> {report_payload['target_shm']}, tag={args.tag})"
        plot_latency_histogram(total_latency, title=title, bins=args.bins, xrange=args.xrange)
        if args.output:
            output_path = Path(args.output)
        elif args.show_plot:
            output_path = None
        else:
            output_path = _default_output_path(args)
        if output_path is not None:
            from matplotlib import pyplot as plt

            plt.savefig(output_path)
            logger.info("Saved latency plot to %s", output_path)
        if args.show_plot:
            from matplotlib import pyplot as plt

            plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
