"""Notebook-style SHARP lab PyWFS hard-RTC example."""

# %% Imports
import argparse
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from pyRTC.Pipeline import RTCManager, clear_shms, initExistingShm
from pyRTC.logging_utils import add_logging_cli_args, configure_logging_from_args, get_logger
from pyRTC.utils import read_yaml_file


logger = get_logger("examples.sharp_lab.pywfs.hard")
CONFIG_PATH = REPO_ROOT / "examples" / "sharp_lab" / "config_pywfs.yaml"
DEFAULT_STREAMS = ["wfs", "wfsRaw", "wfc", "wfc2D", "signal", "signal2D", "psfShort", "psfLong", "strehl", "tiptilt"]


# %% CLI
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the SHARP lab PyWFS stack in hard-RTC mode.")
    parser.add_argument("--duration", type=float, default=10.0, help="Seconds to run before stopping.")
    parser.add_argument("--status-interval", type=float, default=1.0, help="Seconds between status lines.")
    parser.add_argument("--base-port", type=int, default=5901, help="Starting TCP port for child processes.")
    parser.add_argument("--no-clear-shms", action="store_true", help="Reuse any existing SHM blocks.")
    add_logging_cli_args(parser)
    return parser


# %% Helpers
def build_manager_config(config: dict, base_port: int) -> dict:
    configured = dict(config)
    configured.setdefault("manager", {})
    configured["manager"].update(
        {
            "mode": "hard-rtc",
            "componentClasses": {
                "modulator": "pyRTC.hardware.PIModulator",
                "wfs": "pyRTC.hardware.XIMEA_WFS",
                "slopes": "pyRTC.SlopesProcess.SlopesProcess",
                "loop": "pyRTC.Loop.Loop",
                "wfc": "pyRTC.hardware.ALPAODM",
                "psf": "pyRTC.hardware.spinCam",
            },
            "componentFiles": {
                "modulator": str(REPO_ROOT / "pyRTC" / "hardware" / "PIModulator.py"),
                "wfs": str(REPO_ROOT / "pyRTC" / "hardware" / "ximeaWFS.py"),
                "slopes": str(REPO_ROOT / "pyRTC" / "SlopesProcess.py"),
                "loop": str(REPO_ROOT / "pyRTC" / "Loop.py"),
                "wfc": str(REPO_ROOT / "pyRTC" / "hardware" / "ALPAODM.py"),
                "psf": str(REPO_ROOT / "pyRTC" / "hardware" / "SpinnakerScienceCam.py"),
            },
            "ports": {
                "modulator": base_port,
                "wfs": base_port + 1,
                "slopes": base_port + 2,
                "loop": base_port + 3,
                "wfc": base_port + 4,
                "psf": base_port + 5,
            },
        }
    )
    return configured


def format_status_line(start_time: float) -> str:
    signal_stream, _, _ = initExistingShm("signal")
    correction_stream, _, _ = initExistingShm("wfc")
    strehl_stream, _, _ = initExistingShm("strehl")
    residual = np.asarray(signal_stream.read_noblock(SAFE=False), dtype=np.float32).ravel()
    correction = np.asarray(correction_stream.read_noblock(SAFE=False), dtype=np.float32).ravel()
    residual_rms = float(np.sqrt(np.mean(residual**2))) if residual.size else 0.0
    correction_rms = float(np.sqrt(np.mean(correction**2))) if correction.size else 0.0
    strehl = float(np.asarray(strehl_stream.read_noblock(SAFE=False)).ravel()[0])
    elapsed = time.perf_counter() - start_time
    return (
        f"t={elapsed:5.1f}s "
        f"residual_rms={residual_rms:0.4f} "
        f"correction_rms={correction_rms:0.4f} "
        f"strehl={strehl:0.3f}"
    )


# %% Main walkthrough
def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging_from_args(args, app_name="pyrtc-sharp-lab", component_name="sharp_lab_pywfs_hard")

    config = build_manager_config(read_yaml_file(str(CONFIG_PATH)), base_port=args.base_port)
    manager = RTCManager.from_config(config, config_path=str(CONFIG_PATH))

    if not args.no_clear_shms:
        clear_shms(DEFAULT_STREAMS)

    logger.info("SHARP lab PyWFS hard-RTC tutorial")
    logger.info("Config: %s", CONFIG_PATH)
    logger.info("Base port: %s", args.base_port)
    logger.info("Viewer: pyrtc-view wfs signal psfShort psfLong --geometry 2x2")

    manager.start()
    try:
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
        logger.info("Stopping SHARP lab PyWFS hard example")
    finally:
        manager.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())