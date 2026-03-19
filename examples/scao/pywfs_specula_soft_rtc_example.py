"""Notebook-style SPECULA PyWFS soft-RTC example.

SPECULA owns the optical backend for a small Pyramid-WFS simulation while
pyRTC owns the slopes extraction and control loop. This keeps the first bridge
close to the existing OOPAO operator workflow and avoids embedding pyRTC inside
SPECULA's generic YAML simulation engine.
"""

# %% Imports
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


from pyRTC.Loop import Loop
from pyRTC.Pipeline import clear_shms
from pyRTC.SlopesProcess import SlopesProcess
from pyRTC.logging_utils import add_logging_cli_args, configure_logging_from_args, get_logger
from pyRTC.utils import read_yaml_file


logger = get_logger("examples.scao.pywfs_specula_soft")
CONFIG_PATH = REPO_ROOT / "examples" / "scao" / "pywfs_SPECULA_config.yaml"
PARAM_PATH = REPO_ROOT / "examples" / "scao" / "pywfs_SPECULA_params.yaml"
DEFAULT_STREAMS = [
    "wfs",
    "wfsRaw",
    "wfc",
    "signal",
    "signal2D",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the SPECULA-backed PyWFS soft-RTC tutorial.")
    parser.add_argument("--duration", type=float, default=10.0, help="Seconds to run before stopping.")
    parser.add_argument(
        "--status-interval",
        type=float,
        default=1.0,
        help="Seconds between operator-friendly status lines.",
    )
    parser.add_argument(
        "--poke-amp",
        type=float,
        default=1e-3,
        help="Poke amplitude used when computing the interaction matrix.",
    )
    parser.add_argument("--gain", type=float, default=0.1, help="Loop gain used after IM calibration.")
    parser.add_argument(
        "--skip-im",
        action="store_true",
        help="Skip calibration and use an identity-style fallback control matrix.",
    )
    parser.add_argument(
        "--no-clear-shms",
        action="store_true",
        help="Leave existing pyRTC shared-memory streams untouched.",
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
    from pyRTC.hardware.SPECULAInterface import SPECULAInterface

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

    for name in ("slopes", "wfs", "psf", "dm"):
        if system.get(name) is None:
            continue
        try:
            system[name].stop()
        except Exception:
            logger.exception("Failed while stopping %s", name)


def prepare_loop(system: dict, *, gain: float, poke_amp: float, compute_im: bool) -> None:
    loop = system["loop"]
    sim = system["sim"]
    dm = system["dm"]

    if compute_im:
        logger.info("Computing interaction matrix with the atmosphere removed")
        sim.removeAtmosphere()
        dm.flatten()
        loop.pokeAmp = poke_amp
        loop.computeIM()
        sim.addAtmosphere()
    else:
        logger.info("Skipping IM calibration and using an identity-style fallback")
        loop.IM = np.eye(loop.signalSize, loop.numModes, dtype=loop.signalDType)
        loop.computeCM()

    loop.setGain(gain)
    dm.flatten()


def format_status_line(system: dict, elapsed: float) -> str:
    slopes = system["slopes"].read(block=False)
    correction = np.asarray(getattr(system["dm"], "currentShape", system["dm"].read()), dtype=np.float64)
    residual_rms = float(np.sqrt(np.mean(slopes**2))) if slopes.size else 0.0
    correction_rms = float(np.sqrt(np.mean(correction**2))) if correction.size else 0.0
    return (
        f"t={elapsed:5.1f}s "
        f"residual_rms={residual_rms:0.4f} "
        f"dm_rms={correction_rms:0.4f}"
    )


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging_from_args(args, app_name="pyrtc-specula-pywfs", component_name="pywfs_specula_soft_example")

    config = read_yaml_file(str(CONFIG_PATH))

    if not args.no_clear_shms:
        clear_shms(DEFAULT_STREAMS)

    logger.info("SPECULA PyWFS soft-RTC tutorial")
    logger.info("Config: %s", CONFIG_PATH)
    logger.info("SPECULA object params: %s", args.specula_param_file)
    logger.info("SPECULA is the optical backend; pyRTC still owns slopes extraction and control.")
    logger.info("Viewer: pyrtc-view wfs signal2D --geometry 1x2")

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