"""Notebook-style synthetic SHWFS hard-RTC example.

This is the hard-RTC companion to ``synthetic_shwfs_soft_rtc_example.py``.
It uses the same YAML file, but switches behavior at the manager call with
``RTCManager.from_config_file(..., mode="hard")``. In hard mode,
``manager.get_component(...)`` returns a control proxy, so parameters are read
and written with ``getProperty`` and ``setProperty`` and methods are called via
``run``.
"""

# %% Imports
import argparse
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from pyRTC import Telemetry
from pyRTC.Pipeline import RTCManager, clear_shms, initExistingShm
from pyRTC.logging_utils import add_logging_cli_args, configure_logging_from_args, get_logger
from examples.synthetic_shwfs.aotpy_helpers import export_synthetic_session_to_aotpy


logger = get_logger("examples.synthetic_shwfs.hard")
CONFIG_PATH = REPO_ROOT / "examples" / "synthetic_shwfs" / "config.yaml"
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
        "--no-clear-shms",
        action="store_true",
        help="Leave existing pyRTC shared-memory streams in place.",
    )
    parser.add_argument(
        "--aotpy",
        action="store_true",
        help="Capture a short telemetry session, export it to AOTPy, and verify readback.",
    )
    add_logging_cli_args(parser)
    return parser


# %% Tutorial helpers
def ensure_identity_interaction_matrix(config: dict) -> Path:
    wfs_conf = config["wfs"]
    slopes_conf = config["slopes"]
    output_path = Path(config["loop"]["IMFile"])
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

    # Step 1: use the same config file as soft mode and switch behavior here.
    manager = RTCManager.from_config_file(CONFIG_PATH, mode="hard")
    config = manager.config

    # Step 2: generate the tiny IM file referenced by that same config.
    im_path = ensure_identity_interaction_matrix(config)

    # Step 3: clear old SHMs so the child processes start from a clean state.
    if not args.no_clear_shms:
        clear_shms(DEFAULT_STREAMS)

    logger.info("Synthetic SHWFS hard-RTC tutorial")
    logger.info("Config: %s", CONFIG_PATH)
    logger.info("Manager call: RTCManager.from_config_file(CONFIG_PATH, mode='hard')")
    logger.info("Interaction matrix: %s", im_path)
    logger.info("Viewer: pyrtc-view wfs signal2D psfShort psfLong --geometry 2x2")

    # Step 4: start the full stack in child processes.
    manager.start()

    try:
        # Step 5: in hard mode these are proxy objects that control child processes.
        loop = manager.get_component("loop")
        wfc = manager.get_component("wfc")

        starting_gain = loop.getProperty("gain")
        logger.info("Hard mode proxy read: loop.getProperty('gain') -> %s", starting_gain)
        logger.info("Hard mode proxy write: loop.setProperty('gain', 0.10)")
        loop.setProperty("gain", 0.10)
        logger.info("Remote loop gain is now %s", loop.getProperty("gain"))

        # Methods are invoked remotely with run(...).
        wfc.run("flatten")

        # Telemetry works the same way in hard mode because it only reads the
        # published streams and saves them for offline use.
        telem = Telemetry({"dataDir": str(REPO_ROOT / "examples" / "synthetic_shwfs" / "telemetry")})
        telem.save(["wfs", "wfc"], 10)
        telemetry_data = telem.read_last_save()
        logger.info(
            "Telemetry example: wfs=%s wfc=%s",
            telemetry_data["wfs"]["frames"].shape,
            telemetry_data["wfc"]["frames"].shape,
        )

        if args.aotpy:
            session_path, exported_path, reopened_system = export_synthetic_session_to_aotpy(
                repo_root=REPO_ROOT,
                config=config,
                config_path=CONFIG_PATH,
                mode_label="hard",
            )
            logger.info(
                "AOTPy export verified: session=%s file=%s wfs=%s loops=%s science_cameras=%s",
                session_path,
                exported_path,
                len(reopened_system.wavefront_sensors),
                len(reopened_system.loops),
                len(reopened_system.scoring_cameras),
            )

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