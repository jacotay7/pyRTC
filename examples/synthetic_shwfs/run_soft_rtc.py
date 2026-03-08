import argparse
import time
from multiprocessing import shared_memory
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyRTC.SlopesProcess import SlopesProcess
from pyRTC.WavefrontCorrector import WavefrontCorrector
from pyRTC.Loop import Loop
from pyRTC.hardware.SyntheticSystems import SyntheticSHWFS, SyntheticScienceCamera
from pyRTC.utils import read_yaml_file


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


def clear_named_shms(names):
    for name in names:
        for suffix in ("", "_meta", "_gpu_handle"):
            try:
                shm = shared_memory.SharedMemory(name=name + suffix)
            except FileNotFoundError:
                continue
            except Exception:
                continue
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
            finally:
                shm.close()


def build_system(config):
    wfc = WavefrontCorrector(config["wfc"])
    wfs = SyntheticSHWFS(config["wfs"])
    slopes = SlopesProcess(config["slopes"])
    loop = Loop(config["loop"])
    psf = SyntheticScienceCamera(config["psf"]) if "psf" in config else None

    control_matrix = np.eye(loop.signalSize, loop.numModes, dtype=loop.signalDType)
    loop.IM = control_matrix
    loop.computeCM()
    loop.setGain(loop.gain)

    if hasattr(slopes, "signal2DShape"):
        wfc.setLayout(np.ones(slopes.signal2DShape, dtype=bool))

    return {
        "wfc": wfc,
        "wfs": wfs,
        "slopes": slopes,
        "loop": loop,
        "psf": psf,
    }


def start_system(system):
    for component_name in ("wfc", "wfs", "slopes", "loop", "psf"):
        component = system.get(component_name)
        if component is not None:
            component.start()


def stop_system(system):
    for component in system.values():
        if component is not None:
            component.stop()


def status_line(system, elapsed_seconds, delta_seconds, previous_wfs_frames, previous_psf_frames):
    wfs = system["wfs"]
    psf = system.get("psf")
    slopes = system["slopes"]
    loop = system["loop"]

    wfs_rate = 0.0 if delta_seconds <= 0 else (wfs.frameCounter - previous_wfs_frames) / delta_seconds
    psf_rate = 0.0
    if psf is not None and delta_seconds > 0:
        psf_rate = (psf.frameCounter - previous_psf_frames) / delta_seconds

    residual = slopes.read(block=False)
    correction = loop.wfcShm.read_noblock(SAFE=False)
    residual_rms = float(np.sqrt(np.mean(residual**2))) if residual.size > 0 else 0.0
    correction_rms = float(np.sqrt(np.mean(correction**2))) if correction.size > 0 else 0.0
    strehl = psf.strehl_ratio if psf is not None else float("nan")

    return (
        f"t={elapsed_seconds:5.1f}s "
        f"wfs={wfs_rate:6.1f} Hz "
        f"psf={psf_rate:6.1f} Hz "
        f"residual_rms={residual_rms:0.4f} "
        f"correction_rms={correction_rms:0.4f} "
        f"strehl={strehl:0.3f}"
    )


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Run the synthetic SHWFS soft-RTC onboarding demo.")
    parser.add_argument(
        "-c",
        "--config",
        default="examples/synthetic_shwfs/config.yaml",
        help="Path to the synthetic example YAML config.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=15.0,
        help="How long to run the demo in seconds. Use 0 to run until Ctrl-C.",
    )
    parser.add_argument(
        "--status-interval",
        type=float,
        default=1.0,
        help="Seconds between status updates.",
    )
    parser.add_argument(
        "--no-clear-shms",
        action="store_true",
        help="Skip clearing standard pyRTC shared-memory streams before launch.",
    )
    return parser


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    if not args.no_clear_shms:
        clear_named_shms(DEFAULT_STREAMS)

    config = read_yaml_file(args.config)
    system = build_system(config)

    print("Synthetic SHWFS soft-RTC demo")
    print(f"Config: {args.config}")
    print("Viewer commands:")
    print("  pyrtc-view wfs")
    print("  pyrtc-view signal2D -0.8 0.8")
    print("  pyrtc-view wfc2D -0.8 0.8")
    if system.get("psf") is not None:
        print("  pyrtc-view psfShort")
        print("  pyrtc-view psfLong")

    start_system(system)
    start_time = time.perf_counter()
    last_status_time = start_time
    last_wfs_frames = 0
    last_psf_frames = 0

    try:
        while True:
            time.sleep(min(max(args.status_interval, 0.1), 1.0))
            now = time.perf_counter()
            elapsed_seconds = now - start_time
            if now - last_status_time >= args.status_interval:
                print(
                    status_line(
                        system,
                        elapsed_seconds,
                        now - last_status_time,
                        last_wfs_frames,
                        last_psf_frames,
                    )
                )
                last_status_time = now
                last_wfs_frames = system["wfs"].frameCounter
                if system.get("psf") is not None:
                    last_psf_frames = system["psf"].frameCounter
            if args.duration > 0 and elapsed_seconds >= args.duration:
                break
    except KeyboardInterrupt:
        print("Stopping synthetic demo")
    finally:
        stop_system(system)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())