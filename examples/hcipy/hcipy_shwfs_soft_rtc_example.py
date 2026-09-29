"""HCIPy-backed Shack-Hartmann soft-RTC example.

Builds the HCIPy system from ``hcipy_shwfs_params.yaml``, calibrates the loop
on the unaberrated system, then closes it (against the atmosphere when
``hcipy.use_atmosphere`` is set in the config or ``--atmosphere`` is given).
Needs ``pip install pyrtcao[hcipy]``.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# One BLAS thread, as hard-RTC children get (see the architecture guide), unless
# the environment already says otherwise. It must be set before numpy loads.
# HCIPy's propagation otherwise keeps a full OpenBLAS pool per library busy: on
# 16 cores the pools took about 15 of them and the WFS ran slower (#139).
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyrtc import clear_shms  # noqa: E402
from pyrtc.logging_utils import add_logging_cli_args, configure_logging_from_args, get_logger  # noqa: E402
from pyrtc.loop import Loop  # noqa: E402
from pyrtc.slopes_process import SlopesProcess  # noqa: E402
from pyrtc.utils import read_yaml_file  # noqa: E402

logger = get_logger("examples.hcipy.hcipy_shwfs_soft")
CONFIG_PATH = REPO_ROOT / "examples" / "hcipy" / "hcipy_shwfs_config.yaml"
PARAM_PATH = REPO_ROOT / "examples" / "hcipy" / "hcipy_shwfs_params.yaml"
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
    parser = argparse.ArgumentParser(description="Run the HCIPy-backed SHWFS soft-RTC example.")
    parser.add_argument("--duration", type=float, default=10.0, help="Seconds to run.")
    parser.add_argument(
        "--status-interval", type=float, default=1.0, help="Seconds per status line."
    )
    parser.add_argument("--gain", type=float, default=0.3, help="Loop gain after calibration.")
    parser.add_argument(
        "--atmosphere", action="store_true", help="Close the loop on the atmosphere."
    )
    parser.add_argument(
        "--no-clear-shms", action="store_true", help="Leave existing pyrtc streams untouched."
    )
    parser.add_argument(
        "--param-file", type=Path, default=PARAM_PATH, help="HCIPy system parameters (YAML)."
    )
    add_logging_cli_args(parser)
    return parser


def build_system(config: dict, *, param_file: Path = PARAM_PATH) -> dict:
    from pyrtc.hardware.hcipy_interface import HCIPyInterface

    sim = HCIPyInterface(conf=config, param=read_yaml_file(str(param_file)))
    wfs, dm, psf = sim.get_hardware()
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
    system["psf"].start()


def stop_system(system: dict) -> None:
    try:
        system["loop"].stop()
        system["dm"].flatten()
    except Exception:
        logger.exception("Failed to stop the loop cleanly")
    for name in ("loop", "psf", "slopes", "sim"):
        try:
            system[name].close()
        except Exception:
            logger.exception("Failed while closing %s", name)


def prepare_loop(system: dict, *, gain: float, use_atmosphere: bool) -> None:
    """Calibrate on the unaberrated system, then set the gain.

    With the atmosphere off and the DM flat: confirm the DM round trip, take
    reference slopes (so the loop regulates to the unaberrated spots) and
    measure the IM. The atmosphere is switched on afterwards if requested.
    """
    loop, sim, slopes = system["loop"], system["sim"], system["slopes"]
    sim.remove_atmosphere()
    loop.check_round_trip()
    slopes.take_ref_slopes()
    loop.compute_im()
    if use_atmosphere:
        sim.add_atmosphere()
    loop.set_gain(gain)
    loop.flatten()


def format_status_line(system: dict, elapsed: float) -> str:
    signal = np.asarray(system["slopes"].read(block=False), dtype=np.float64)
    residual = float(np.sqrt(np.mean(signal**2))) if signal.size else 0.0
    strehl = float(system["psf"].strehl_ratio)
    return f"t={elapsed:5.1f}s residual_rms={residual:0.4f} strehl={strehl:0.3f}"


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging_from_args(args, app_name="pyrtc-hcipy-shwfs", component_name="hcipy_example")
    config = read_yaml_file(str(CONFIG_PATH))
    if not args.no_clear_shms:
        clear_shms(DEFAULT_STREAMS)
    logger.info("Viewer: pyrtc-view wfs signal_2d wfc_2d psf_short psf_long --geometry 2x3")
    system = build_system(config, param_file=args.param_file)
    use_atmosphere = args.atmosphere or bool(config.get("hcipy", {}).get("use_atmosphere"))
    try:
        start_system(system)
        prepare_loop(system, gain=args.gain, use_atmosphere=use_atmosphere)
        start = time.perf_counter()
        next_status = start
        system["loop"].start()
        while (elapsed := time.perf_counter() - start) < args.duration:
            if time.perf_counter() >= next_status:
                logger.info(format_status_line(system, elapsed))
                next_status = time.perf_counter() + max(args.status_interval, 0.25)
            time.sleep(0.1)
    finally:
        stop_system(system)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
