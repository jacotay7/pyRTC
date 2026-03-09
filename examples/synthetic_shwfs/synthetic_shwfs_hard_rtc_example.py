"""Notebook-style synthetic SHWFS hard-RTC example.

This is the hard-RTC companion to ``synthetic_shwfs_soft_rtc_example.py``.
It uses the same YAML config and the same identity interaction matrix, but the
manager launches each component in its own child process.
"""

# %% Imports
import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from pyRTC.Pipeline import RTCManager, clear_shms, initExistingShm
from pyRTC.logging_utils import add_logging_cli_args, configure_logging_from_args, get_logger
from pyRTC.utils import read_yaml_file


logger = get_logger("examples.synthetic_shwfs.hard")
CONFIG_PATH = REPO_ROOT / "examples" / "synthetic_shwfs" / "config.yaml"
IDENTITY_IM_PATH = Path(tempfile.gettempdir()) / "pyrtc_synthetic_identity_im.npy"
RUNTIME_CONFIG_PATH = Path(tempfile.gettempdir()) / "pyrtc_synthetic_hard_rtc_runtime.yaml"
DEFAULT_STREAMS = [
    "wfs",
    "wfsRaw",
    "wfc",
    "wfc2D",
    "signal",
    "signal2D",
    "psfShort",
    "psfLong",
    "strehl",
    "tiptilt",
]


# %% Command-line interface
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the synthetic SHWFS tutorial in hard-RTC mode.")
    parser.add_argument("--duration", type=float, default=15.0, help="Seconds to run before stopping.")
    parser.add_argument(
        "--status-interval",
        type=float,
        default=1.0,
        help="Seconds between operator-friendly status lines.",
    )
    parser.add_argument(
        "--base-port",
        type=int,
        default=5701,
        help="Starting TCP port for the launched child processes.",
    )
    parser.add_argument(
        "--no-clear-shms",
        action="store_true",
        help="Leave existing pyRTC shared-memory streams in place.",
    )
    add_logging_cli_args(parser)
    return parser


# %% Tutorial helpers
def write_identity_interaction_matrix(config: dict, output_path: Path) -> Path:
    wfs_conf = config["wfs"]
    slopes_conf = config["slopes"]
    num_modes = int(config["wfc"]["numModes"])

    image_width = int(wfs_conf["width"])
    image_height = int(wfs_conf["height"])
    downsample = int(wfs_conf.get("downsampleFactor", 0))
    if downsample > 0:
        image_width //= downsample
        image_height //= downsample

    subap_spacing = int(slopes_conf["subApSpacing"])
    num_regions = min(image_width, image_height) // subap_spacing
    signal_size = 2 * num_regions * num_regions

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, np.eye(signal_size, num_modes, dtype=np.float32))
    return output_path


def build_manager_config(config: dict, base_port: int) -> dict:
    configured = dict(config)
    configured.setdefault("manager", {})
    configured["manager"].update(
        {
            "mode": "hard-rtc",
            "componentClasses": {
                "wfs": "pyRTC.hardware.SyntheticSHWFS",
                "slopes": "pyRTC.SlopesProcess.SlopesProcess",
                "loop": "pyRTC.Loop.Loop",
                "wfc": "pyRTC.hardware.SyntheticWFC",
                "psf": "pyRTC.hardware.SyntheticScienceCamera",
            },
            "componentFiles": {
                "wfs": str(REPO_ROOT / "pyRTC" / "hardware" / "SyntheticSHWFS.py"),
                "slopes": str(REPO_ROOT / "pyRTC" / "SlopesProcess.py"),
                "loop": str(REPO_ROOT / "pyRTC" / "Loop.py"),
                "wfc": str(REPO_ROOT / "pyRTC" / "hardware" / "SyntheticWFC.py"),
                "psf": str(REPO_ROOT / "pyRTC" / "hardware" / "SyntheticScienceCamera.py"),
            },
            "ports": {
                "wfs": base_port,
                "slopes": base_port + 1,
                "loop": base_port + 2,
                "wfc": base_port + 3,
                "psf": base_port + 4,
            },
        }
    )
    return configured


def write_runtime_config(config: dict, output_path: Path) -> Path:
    runtime_config = {key: value for key, value in config.items() if key != "manager"}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(runtime_config, handle, sort_keys=False)
    return output_path


def read_scalar_stream(name: str) -> float:
    stream, _, _ = initExistingShm(name)
    return float(np.asarray(stream.read_noblock(SAFE=False)).ravel()[0])


def format_status_line(start_time: float) -> str:
    signal_stream, _, _ = initExistingShm("signal")
    correction_stream, _, _ = initExistingShm("wfc")
    residual = np.asarray(signal_stream.read_noblock(SAFE=False), dtype=np.float32).ravel()
    correction = np.asarray(correction_stream.read_noblock(SAFE=False), dtype=np.float32).ravel()
    residual_rms = float(np.sqrt(np.mean(residual**2))) if residual.size else 0.0
    correction_rms = float(np.sqrt(np.mean(correction**2))) if correction.size else 0.0
    strehl = read_scalar_stream("strehl")
    tiptilt = read_scalar_stream("tiptilt")
    elapsed = time.perf_counter() - start_time
    return (
        f"t={elapsed:5.1f}s "
        f"residual_rms={residual_rms:0.4f} "
        f"correction_rms={correction_rms:0.4f} "
        f"strehl={strehl:0.3f} "
        f"tiptilt={tiptilt:0.3f}"
    )


# %% Main walkthrough
def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging_from_args(args, app_name="pyrtc-synthetic-shwfs", component_name="synthetic_hard_example")

    # Step 1: load the base config and generate a tiny IM file that the loop can load remotely.
    config = read_yaml_file(str(CONFIG_PATH))
    config["loop"]["IMFile"] = str(write_identity_interaction_matrix(config, IDENTITY_IM_PATH))

    # Step 2: inject the hard-RTC manager metadata.
    config = build_manager_config(config, base_port=args.base_port)
    runtime_config_path = write_runtime_config(config, RUNTIME_CONFIG_PATH)
    manager = RTCManager.from_config(config, config_path=str(runtime_config_path))

    # Step 3: clear old SHMs so the child processes start from a clean state.
    if not args.no_clear_shms:
        clear_shms(DEFAULT_STREAMS)

    logger.info("Synthetic SHWFS hard-RTC tutorial")
    logger.info("Config: %s", runtime_config_path)
    logger.info("Interaction matrix: %s", config["loop"]["IMFile"])
    logger.info("Base port: %s", args.base_port)
    logger.info("Viewer: pyrtc-view wfs signal2D psfShort psfLong --geometry 2x2")

    # Step 4: start the full stack in child processes.
    manager.start()

    try:
        # Step 5: flatten the mirror once the child processes are live.
        manager.get_component("wfc").run("flatten")

        start_time = time.perf_counter()
        next_status = start_time
        while True:
            now = time.perf_counter()
            elapsed = now - start_time
            if now >= next_status:
                logger.info(format_status_line(start_time))
                next_status = now + max(args.status_interval, 0.25)
            if args.duration > 0 and elapsed >= args.duration:
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("Stopping synthetic SHWFS tutorial")
    finally:
        manager.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())