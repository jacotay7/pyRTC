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
from pyRTC.Pipeline import ImageSHM
from pyRTC.hardware.SyntheticSystems import SyntheticSHWFS, SyntheticScienceCamera
from pyRTC.utils import float_to_dtype, read_yaml_file


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


def _existing_shm_spec(name):
    try:
        meta_shm = shared_memory.SharedMemory(name=name + "_meta")
    except FileNotFoundError:
        return None
    except Exception:
        return None

    try:
        metadata = np.ndarray((ImageSHM.METADATA_SIZE,), dtype=np.float64, buffer=meta_shm.buf).copy()
    finally:
        meta_shm.close()

    shape = []
    index = 0
    while 4 + index < metadata.size and int(metadata[4 + index]) > 0:
        shape.append(int(metadata[4 + index]))
        index += 1
    return tuple(shape), np.dtype(float_to_dtype(metadata[3]))


def expected_stream_specs(config):
    wfs_width = int(config["wfs"]["width"])
    wfs_height = int(config["wfs"]["height"])
    downsample = int(config["wfs"].get("downsampleFactor", 0))
    image_shape = [wfs_width, wfs_height]
    if downsample > 0:
        image_shape[0] //= downsample
        image_shape[1] //= downsample

    subap_spacing = int(config["slopes"]["subApSpacing"])
    num_regions = image_shape[0] // int(round(subap_spacing, 0))
    signal2d_shape = (2 * num_regions, num_regions)
    signal_shape = (int(np.prod(signal2d_shape)),)

    specs = {
        "wfsRaw": {"shape": (wfs_width, wfs_height), "dtype": np.dtype(np.uint16)},
        "wfs": {"shape": tuple(image_shape), "dtype": np.dtype(np.int32)},
        "signal": {"shape": signal_shape, "dtype": np.dtype(np.float32)},
        "signal2D": {"shape": signal2d_shape, "dtype": np.dtype(np.float32)},
        "wfc": {"shape": (int(config["wfc"]["numModes"]),), "dtype": np.dtype(np.float32)},
        "wfc2D": {"shape": signal2d_shape, "dtype": np.dtype(np.float32)},
    }

    if "psf" in config:
        psf_shape = (int(config["psf"]["width"]), int(config["psf"]["height"]))
        specs.update(
            {
                "psfShort": {"shape": psf_shape, "dtype": np.dtype(np.int32)},
                "psfLong": {"shape": psf_shape, "dtype": np.dtype(np.float64)},
                "strehl": {"shape": (1,), "dtype": np.dtype(float)},
                "tiptilt": {"shape": (1,), "dtype": np.dtype(float)},
            }
        )

    return specs


def ensure_expected_shms(config, force_rebuild=False):
    specs = expected_stream_specs(config)
    if force_rebuild:
        clear_named_shms(list(specs))
        return list(specs), []

    rebuilt = []
    reused = []
    for name, spec in specs.items():
        existing = _existing_shm_spec(name)
        if existing is None:
            rebuilt.append(name)
            continue
        if existing[0] != spec["shape"] or np.dtype(existing[1]) != np.dtype(spec["dtype"]):
            rebuilt.append(name)
        else:
            reused.append(name)

    if rebuilt:
        clear_named_shms(rebuilt)
    return rebuilt, reused


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
        help="Skip compatibility checks and leave existing pyRTC shared-memory streams untouched.",
    )
    return parser


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    config = read_yaml_file(args.config)
    if args.no_clear_shms:
        rebuilt = []
        reused = []
    else:
        rebuilt, reused = ensure_expected_shms(config)
    system = build_system(config)

    print("Synthetic SHWFS soft-RTC demo")
    print(f"Config: {args.config}")
    if args.no_clear_shms:
        print("Skipping SHM compatibility checks")
    elif rebuilt:
        print("Rebuilt SHMs:", ", ".join(rebuilt))
    else:
        print("Reusing existing compatible SHMs")
    if reused:
        print("Reused SHMs:", ", ".join(reused))
    print("Viewer commands:")
    print("  pyrtc-view wfs signal2D wfc2D psfShort psfLong --geometry 2x3")
    if system.get("psf") is not None:
        print("  pyrtc-view psfShort psfLong --geometry row")

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