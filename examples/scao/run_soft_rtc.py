"""Run the OOPAO-backed PYWFS SCAO example in soft-RTC mode.

This script is the script-driven companion to the
``pywfs_example_OOPAO.ipynb`` notebook. It builds the same simulated
wavefront-sensor, deformable-mirror, slopes, loop, and science-camera chain,
then optionally computes a quick interaction matrix before closing the loop for
an operator-selected duration.

The example is intentionally Linux-oriented and is meant for development,
demonstration, and documentation rather than benchmarking or deployment.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from pyRTC.logging_utils import add_logging_cli_args, configure_logging_from_args, get_logger

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyRTC.Loop import Loop
from pyRTC.Pipeline import clear_shms
from pyRTC.SlopesProcess import SlopesProcess
from pyRTC.utils import read_yaml_file


logger = get_logger("examples.scao.run_soft_rtc")


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


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the OOPAO-backed PYWFS soft-RTC example.")
    parser.add_argument(
        "-c",
        "--config",
        default="examples/scao/pywfs_OOPAO_config.yaml",
        help="Path to the example YAML configuration.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="How long to run the closed loop after setup. Use 0 to skip loop execution.",
    )
    parser.add_argument(
        "--status-interval",
        type=float,
        default=1.0,
        help="Seconds between status updates while the loop is running.",
    )
    parser.add_argument(
        "--poke-amp",
        type=float,
        default=1e-7,
        help="Poke amplitude used when computing a quick interaction matrix.",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=0.3,
        help="Loop gain applied before closing the loop.",
    )
    parser.add_argument(
        "--skip-im",
        action="store_true",
        help="Skip interaction-matrix calibration and use a dense identity-style fallback.",
    )
    parser.add_argument(
        "--no-kl-basis",
        action="store_true",
        help="Leave the wavefront corrector on its default basis instead of computing a KL basis.",
    )
    parser.add_argument(
        "--no-clear-shms",
        action="store_true",
        help="Leave existing pyRTC shared-memory streams untouched.",
    )
    add_logging_cli_args(parser)
    return parser


def _configure_kl_basis(sim, dm, num_modes: int) -> None:
    from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis

    basis = compute_KL_basis(sim.tel, sim.atm, sim.dm)
    dm.setM2C(basis[:, :num_modes])
    logger.info("Configured KL modal basis with %s modes", num_modes)


def build_system(config: dict, use_kl_basis: bool = True) -> dict:
    from pyRTC.hardware.OOPAOInterface import OOPAOInterface

    sim = OOPAOInterface(conf=config, param=None)
    wfs, dm, psf = sim.get_hardware()

    if use_kl_basis:
        _configure_kl_basis(sim, dm, int(config["wfc"]["numModes"]))

    slopes = SlopesProcess(config["slopes"])
    loop = Loop(config["loop"])

    return {
        "sim": sim,
        "wfs": wfs,
        "dm": dm,
        "psf": psf,
        "slopes": slopes,
        "loop": loop,
    }


def start_system(system: dict) -> None:
    system["dm"].start()
    system["dm"].flatten()
    system["wfs"].start()
    system["slopes"].start()
    system["psf"].start()


def stop_system(system: dict) -> None:
    try:
        system["loop"].stop()
    except Exception:
        logger.exception("Failed while stopping loop")

    try:
        system["dm"].flatten()
    except Exception:
        logger.exception("Failed while flattening the deformable mirror during shutdown")

    for name in ("psf", "slopes", "wfs", "dm"):
        try:
            system[name].stop()
        except Exception:
            logger.exception("Failed while stopping %s", name)


def prepare_loop(system: dict, gain: float, poke_amp: float, compute_im: bool) -> None:
    loop = system["loop"]
    sim = system["sim"]
    dm = system["dm"]

    if compute_im:
        logger.info("Computing interaction matrix with atmosphere removed")
        sim.removeAtmosphere()
        dm.flatten()
        loop.pokeAmp = poke_amp
        loop.computeIM()
        sim.addAtmosphere()
    else:
        logger.info("Skipping interaction-matrix calibration; using a dense identity-style fallback")
        loop.IM = np.eye(loop.signalSize, loop.numModes, dtype=loop.signalDType)
        loop.computeCM()

    loop.setGain(gain)
    dm.flatten()


def _status_line(system: dict, elapsed_seconds: float) -> str:
    slopes = system["slopes"].read(block=False)
    correction = system["dm"].read()
    residual_rms = float(np.sqrt(np.mean(slopes**2))) if slopes.size > 0 else 0.0
    correction_rms = float(np.sqrt(np.mean(correction**2))) if correction.size > 0 else 0.0
    strehl = float(system["psf"].strehl_ratio)
    tiptilt = float(system["psf"].peak_dist)
    return (
        f"t={elapsed_seconds:5.1f}s "
        f"residual_rms={residual_rms:0.4f} "
        f"correction_rms={correction_rms:0.4f} "
        f"strehl={strehl:0.3f} "
        f"tiptilt={tiptilt:0.3f}"
    )


def run_loop(system: dict, duration: float, status_interval: float) -> None:
    if duration <= 0:
        logger.info("Duration is 0; skipping closed-loop run")
        return

    loop = system["loop"]
    dm = system["dm"]
    start_time = time.perf_counter()
    next_status = start_time
    logger.info("Starting loop for %.1f seconds", duration)
    loop.start()
    try:
        while True:
            now = time.perf_counter()
            elapsed = now - start_time
            if elapsed >= duration:
                break
            if now >= next_status:
                logger.info(_status_line(system, elapsed))
                next_status = now + max(status_interval, 0.1)
            time.sleep(0.1)
    finally:
        loop.stop()
        dm.flatten()
        logger.info("Stopped loop and flattened deformable mirror")


def main(argv=None) -> int:
    args = _build_arg_parser().parse_args(argv)
    configure_logging_from_args(args, app_name="pyrtc-oopao-pywfs", component_name="run_soft_rtc")
    config = read_yaml_file(args.config)

    if not args.no_clear_shms:
        clear_shms(DEFAULT_STREAMS)

    logger.info("Building OOPAO PYWFS soft-RTC example from %s", args.config)
    logger.info("Viewer command: pyrtc-view wfs signal2D wfc2D psfShort psfLong --geometry 2x3")

    system = build_system(config, use_kl_basis=not args.no_kl_basis)
    try:
        start_system(system)
        prepare_loop(system, gain=args.gain, poke_amp=args.poke_amp, compute_im=not args.skip_im)
        run_loop(system, duration=args.duration, status_interval=args.status_interval)
    finally:
        stop_system(system)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())