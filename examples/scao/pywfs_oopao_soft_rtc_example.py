"""Notebook-style OOPAO PyWFS soft-RTC example.

This tutorial keeps the same control chain as the existing notebook-backed SCAO
example, but the script is written as a sequence of clear steps with `# %%`
sections so users can read or execute it like a notebook.

The OOPAO adapters intentionally stay in soft-RTC mode because the wavefront
sensor, deformable mirror, and science camera all share one in-process optical
simulation state.
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


from pyRTC.Loop import Loop
from pyRTC.Pipeline import clear_shms
from pyRTC.SlopesProcess import SlopesProcess
from pyRTC.logging_utils import add_logging_cli_args, configure_logging_from_args, get_logger
from pyRTC.utils import read_yaml_file


logger = get_logger("examples.scao.pywfs_oopao_soft")
CONFIG_PATH = REPO_ROOT / "examples" / "scao" / "pywfs_OOPAO_config.yaml"
PARAM_PATH = REPO_ROOT / "examples" / "scao" / "pywfs_OOPAO_params.yaml"
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
    parser = argparse.ArgumentParser(description="Run the OOPAO-backed PyWFS soft-RTC tutorial.")
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
        default=1e-7,
        help="Poke amplitude used when computing the interaction matrix.",
    )
    parser.add_argument("--gain", type=float, default=0.1, help="Loop gain used after IM calibration.")
    parser.add_argument(
        "--skip-im",
        action="store_true",
        help="Skip calibration and use an identity-style fallback control matrix.",
    )
    parser.add_argument(
        "--no-kl-basis",
        action="store_true",
        help="Leave the DM on its default basis instead of computing a KL basis.",
    )
    parser.add_argument(
        "--no-clear-shms",
        action="store_true",
        help="Leave existing pyRTC shared-memory streams untouched.",
    )
    parser.add_argument(
        "--oopao-param-file",
        type=Path,
        default=PARAM_PATH,
        help="YAML file describing how to build the OOPAO tel/ngs/src/atm/dm/wfs objects.",
    )
    add_logging_cli_args(parser)
    return parser


# %% Tutorial helpers
def configure_kl_basis(sim, dm, num_modes: int) -> None:
    from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis

    basis = compute_KL_basis(sim.tel, sim.atm, sim.dm)
    dm.setM2C(basis[:, :num_modes])


def build_system(config: dict, *, use_kl_basis: bool, oopao_param_file: Path) -> dict:
    from pyRTC.hardware.OOPAOInterface import OOPAOInterface

    oopao_param = read_yaml_file(str(oopao_param_file))
    logger.info(
        "OOPAO param conventions: reuse the flat OOPAO constructor-style keys for tel/atm/dm/wfs values, "
        "and use ngs_band/ngs_magnitude plus science_band/science_magnitude for the two source objects."
    )
    sim = OOPAOInterface(conf=config, param=oopao_param)
    wfs, dm, psf = sim.get_hardware()
    if use_kl_basis:
        configure_kl_basis(sim, dm, int(config["wfc"]["numModes"]))

    return {
        "sim": sim,
        "wfs": wfs,
        "dm": dm,
        "slopes": SlopesProcess(config["slopes"]),
        "loop": Loop(config["loop"]),
        "psf": psf,
    }


def start_system(system: dict) -> None:
    system["dm"].start()
    system["dm"].flatten()
    system["wfs"].start()
    system["slopes"].start()


def start_science_camera(system: dict) -> None:
    system["psf"].start()


def stop_system(system: dict) -> None:
    try:
        system["loop"].stop()
    except Exception:
        logger.exception("Failed to stop the loop cleanly")

    try:
        system["dm"].flatten()
    except Exception:
        logger.exception("Failed to flatten the DM during shutdown")

    for name in ("psf", "slopes", "wfs", "dm"):
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
    strehl = float(system["psf"].strehl_ratio)
    tiptilt = float(system["psf"].peak_dist)
    return (
        f"t={elapsed:5.1f}s "
        f"residual_rms={residual_rms:0.4f} "
        f"dm_rms={correction_rms:0.4f} "
        f"strehl={strehl:0.3f} "
        f"tiptilt={tiptilt:0.3f}"
    )


# %% Main walkthrough
def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging_from_args(args, app_name="pyrtc-oopao-pywfs", component_name="pywfs_oopao_soft_example")

    # Step 1: load the same YAML config used by the notebook walkthrough.
    config = read_yaml_file(str(CONFIG_PATH))

    # Step 2: clear old SHMs so the example always starts from a predictable state.
    if not args.no_clear_shms:
        clear_shms(DEFAULT_STREAMS)

    logger.info("OOPAO PyWFS soft-RTC tutorial")
    logger.info("Config: %s", CONFIG_PATH)
    logger.info("OOPAO object params: %s", args.oopao_param_file)
    logger.info("Viewer: pyrtc-view wfs signal2D wfc2D psfShort psfLong --geometry 2x3")
    logger.info(
        "To reuse your own OOPAO objects instead of this YAML file, construct OOPAOInterface with explicit tel=..., atm=..., wfs=... arguments."
    )

    # Step 3: build the simulation-backed components.
    system = build_system(
        config,
        use_kl_basis=not args.no_kl_basis,
        oopao_param_file=args.oopao_param_file,
    )

    try:
        # Step 4: start the WFS, DM, and slopes stages needed for calibration.
        start_system(system)

        # Step 5: prepare the loop exactly once before closing it.
        prepare_loop(system, gain=args.gain, poke_amp=args.poke_amp, compute_im=not args.skip_im)

        # Step 6: start the science camera after calibration.
        # OOPAO's current source/telescope model is more stable when the PSF path
        # is started after the interaction-matrix phase rather than during it.
        start_science_camera(system)

        # Step 7: close the loop and print live status lines.
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