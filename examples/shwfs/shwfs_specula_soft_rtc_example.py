"""Notebook-style SPECULA SHWFS soft-RTC example."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_SPECULA_ROOT = REPO_ROOT.parent / "SPECULA"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if WORKSPACE_SPECULA_ROOT.exists() and str(WORKSPACE_SPECULA_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_SPECULA_ROOT))


from pyrtc.loop import Loop
from pyrtc import clear_shms
from pyrtc.slopes_process import SlopesProcess
from pyrtc.logging_utils import add_logging_cli_args, configure_logging_from_args, get_logger
from pyrtc.utils import read_yaml_file


logger = get_logger("examples.shwfs.shwfs_specula_soft")
CONFIG_PATH = REPO_ROOT / "examples" / "shwfs" / "shwfs_SPECULA_config.yaml"
PARAM_PATH = REPO_ROOT / "examples" / "shwfs" / "shwfs_SPECULA_params.yaml"
DEFAULT_STREAMS = [
    "wfs",
    "wfs_raw",
    "wfc",
    "wfc_2d",
    "signal",
    "signal_2d",
    "psf_short",
    "psf_long",
    "strehl",
    "tiptilt",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the SPECULA-backed SHWFS soft-RTC tutorial.")
    parser.add_argument(
        "--duration", type=float, default=10.0, help="Seconds to run before stopping."
    )
    parser.add_argument(
        "--status-interval",
        type=float,
        default=1.0,
        help="Seconds between operator-friendly status lines.",
    )
    parser.add_argument(
        "--poke-amp",
        type=float,
        default=None,
        help="Poke amplitude used when computing the interaction matrix (default: loop.poke_amp from the config).",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=None,
        help="Loop gain used after IM calibration (default: loop.gain from the config).",
    )
    parser.add_argument(
        "--skip-im",
        action="store_true",
        help="Skip calibration and use an identity-style fallback control matrix.",
    )
    parser.add_argument(
        "--no-clear-shms",
        action="store_true",
        help="Leave existing pyrtc shared-memory streams untouched.",
    )
    parser.add_argument(
        "--specula-param-file",
        type=Path,
        default=PARAM_PATH,
        help="YAML file describing the SPECULA object graph used by the bridge.",
    )
    add_logging_cli_args(parser)
    return parser


def build_system(config: dict, *, specula_param_file: Path) -> dict:
    from pyrtc.hardware.specula_interface import SPECULAInterface

    specula_param = read_yaml_file(str(specula_param_file))
    sim = SPECULAInterface(conf=config, param=specula_param)
    wfs, dm, psf = sim.get_hardware()
    return {
        "sim": sim,
        "wfs": wfs,
        "dm": dm,
        "psf": psf,
        "slopes": SlopesProcess(config["slopes"]),
        "loop": Loop(config["loop"]),
    }


def start_system(system: dict) -> None:
    system["dm"].start()
    system["dm"].flatten()
    system["wfs"].start()
    if system["psf"] is not None:
        system["psf"].start()
    system["slopes"].start()


def stop_system(system: dict) -> None:
    try:
        system["loop"].stop()
    except Exception:
        logger.exception("Failed to stop the loop cleanly")
    try:
        system["dm"].flatten()
    except Exception:
        logger.exception("Failed to flatten the DM during shutdown")
    for name in ("loop", "slopes", "wfs", "psf", "dm", "sim"):
        if system.get(name) is None or not hasattr(system[name], "close"):
            continue
        try:
            # close() stops the component, ends its worker threads and
            # releases its stream handles; stop() would only pause it.
            system[name].close()
        except Exception:
            logger.exception("Failed while stopping %s", name)


def prepare_loop(
    system: dict,
    *,
    gain: float | None = None,
    poke_amp: float | None = None,
    compute_im: bool = True,
) -> None:
    """Calibrate the loop on the unaberrated system.

    Calibration always runs with the atmosphere removed and the DM flat. The
    loop first confirms a DM round trip (``Loop.check_round_trip``), then the
    reference slopes are taken so the loop regulates to the diffraction-
    limited spots rather than to the static SH/PyWFS offset, then the IM is
    measured with the configured ``im_method`` (push-pull). The atmosphere is
    restored afterwards only if it was enabled before
    (``specula.use_atmosphere`` in the config).
    """

    loop = system["loop"]
    sim = system["sim"]
    slopes = system["slopes"]
    if poke_amp is not None:
        loop.poke_amp = poke_amp
    if gain is not None:
        loop.set_gain(gain)
    if compute_im:
        use_atmosphere = sim.use_atmosphere
        logger.info("Calibrating with the atmosphere removed")
        sim.remove_atmosphere()
        # Reference slopes need the live pipeline (the simulator may still be
        # starting), so confirm a DM round trip before taking them (compute_im
        # checks again on its own before poking).
        loop.check_round_trip()
        logger.info("Taking reference slopes on the flat DM")
        slopes.take_ref_slopes()
        loop.compute_im()
        loop.flatten()
        if use_atmosphere:
            sim.add_atmosphere()
    else:
        logger.info("Skipping IM calibration and using an identity-style fallback")
        loop.im = np.eye(loop.signal_size, loop.num_modes, dtype=loop.signal_dtype)
        loop.compute_cm()
    loop.flatten()


def format_status_line(system: dict, elapsed: float) -> str:
    slopes = system["slopes"].read(block=False)
    correction = np.asarray(
        getattr(system["dm"], "current_shape", system["dm"].read()), dtype=np.float64
    )
    residual_rms = float(np.sqrt(np.mean(slopes**2))) if slopes.size else 0.0
    correction_rms = float(np.sqrt(np.mean(correction**2))) if correction.size else 0.0
    return f"t={elapsed:5.1f}s residual_rms={residual_rms:0.4f} dm_rms={correction_rms:0.4f}"


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging_from_args(
        args, app_name="pyrtc-specula-shwfs", component_name="shwfs_specula_soft_example"
    )
    config = read_yaml_file(str(CONFIG_PATH))
    if not args.no_clear_shms:
        clear_shms(DEFAULT_STREAMS)
    logger.info("SPECULA SHWFS soft-RTC tutorial")
    logger.info("Config: %s", CONFIG_PATH)
    logger.info("SPECULA object params: %s", args.specula_param_file)
    logger.info("Viewer: pyrtc-view wfs signal_2d wfc_2d psf_short psf_long --geometry 2x3")
    system = build_system(config, specula_param_file=args.specula_param_file)
    try:
        start_system(system)
        prepare_loop(system, gain=args.gain, poke_amp=args.poke_amp, compute_im=not args.skip_im)
        start_time = time.perf_counter()
        next_status = start_time
        system["loop"].start()
        try:
            while True:
                now = time.perf_counter()
                elapsed = now - start_time
                if now >= next_status:
                    logger.info(format_status_line(system, elapsed))
                    next_status = now + max(args.status_interval, 0.25)
                if args.duration > 0 and elapsed >= args.duration:
                    break
                time.sleep(0.1)
        finally:
            system["loop"].stop()
            system["dm"].flatten()
    finally:
        stop_system(system)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
