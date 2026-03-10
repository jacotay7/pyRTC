"""Compatibility wrapper for the legacy synthetic SHWFS soft-RTC example.

The tutorial-facing examples were renamed to make the soft/hard manager flows
clearer, but the test suite still imports the original helper module name.
This file keeps that import surface stable while delegating to the current
synthetic component stack.
"""

import sys
import time
import importlib.util
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from pyRTC.Loop import Loop
from pyRTC.Pipeline import clear_shms, initExistingShm
from pyRTC.SlopesProcess import SlopesProcess
from pyRTC.config_schema import read_system_config
from pyRTC.hardware.SyntheticSystems import SyntheticSHWFS, SyntheticScienceCamera, SyntheticWFC


DEFAULT_STREAMS = (
    "wfsRaw",
    "wfs",
    "signal",
    "signal2D",
    "wfc",
    "wfc2D",
    "psfShort",
    "psfLong",
    "strehl",
    "tiptilt",
)

_SOFT_EXAMPLE_PATH = Path(__file__).with_name("synthetic_shwfs_soft_rtc_example.py")


def read_yaml_file(file_path):
    return read_system_config(file_path)


def expected_stream_specs(config: dict) -> dict:
    wfs_shape = (int(config["wfs"]["width"]), int(config["wfs"]["height"]))
    psf_shape = (int(config["psf"]["width"]), int(config["psf"]["height"]))
    num_modes = int(config["wfc"]["numModes"])
    signal2d_shape = _signal_2d_shape(config)
    wfc2d_shape = _wfc_2d_shape(config)

    return {
        "wfsRaw": {"shape": wfs_shape, "dtype": np.uint16},
        "wfs": {"shape": wfs_shape, "dtype": np.int32},
        "signal": {"shape": (num_modes,), "dtype": np.float32},
        "signal2D": {"shape": signal2d_shape, "dtype": np.float32},
        "wfc": {"shape": (num_modes,), "dtype": np.float32},
        "wfc2D": {"shape": wfc2d_shape, "dtype": np.float32},
        "psfShort": {"shape": psf_shape, "dtype": np.int32},
        "psfLong": {"shape": psf_shape, "dtype": np.float64},
        "strehl": {"shape": (1,), "dtype": np.float64},
        "tiptilt": {"shape": (1,), "dtype": np.float64},
    }


def _existing_shm_spec(name: str):
    try:
        _, shape, dtype = initExistingShm(name)
    except Exception:
        return None
    return tuple(shape), np.dtype(dtype)


def clear_named_shms(names):
    clear_shms(list(names))


def ensure_expected_shms(config: dict, force_rebuild: bool = False):
    specs = expected_stream_specs(config)
    rebuilt = []
    reused = []

    for name, spec in specs.items():
        current = None if force_rebuild else _existing_shm_spec(name)
        expected = (tuple(spec["shape"]), np.dtype(spec["dtype"]))
        if current is None:
            if force_rebuild:
                rebuilt.append(name)
            continue
        if current != expected:
            clear_named_shms([name])
            rebuilt.append(name)
        else:
            reused.append(name)

    if force_rebuild:
        clear_named_shms(specs.keys())
        rebuilt = list(specs)
        reused = []

    return rebuilt, reused


def build_system(config: dict) -> dict:
    ensure_identity_interaction_matrix(config)

    wfs = SyntheticSHWFS(config["wfs"])
    slopes = SlopesProcess(config["slopes"])
    wfc = SyntheticWFC(config["wfc"])
    wfc.setLayout(np.ones(_wfc_2d_shape(config), dtype=bool))
    loop = Loop(config["loop"])
    psf = SyntheticScienceCamera(config["psf"])

    return {
        "wfs": wfs,
        "slopes": slopes,
        "wfc": wfc,
        "loop": loop,
        "psf": psf,
    }


def start_system(system: dict) -> None:
    system["wfc"].start()
    system["wfc"].flatten()
    system["wfs"].start()
    system["slopes"].start()
    system["psf"].start()
    system["loop"].start()


def stop_system(system: dict) -> None:
    try:
        system["loop"].stop()
    except Exception:
        pass

    try:
        system["wfc"].flatten()
    except Exception:
        pass

    for name in ("psf", "slopes", "wfs", "wfc"):
        try:
            system[name].stop()
        except Exception:
            pass


def ensure_identity_interaction_matrix(config: dict) -> Path:
    output_path = Path(config["loop"]["IMFile"])
    signal_size = int(np.prod(_signal_2d_shape(config)))
    num_modes = int(config["wfc"]["numModes"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, np.eye(signal_size, num_modes, dtype=np.float32))
    return output_path


def _signal_2d_shape(config: dict) -> tuple[int, int]:
    spacing = int(round(float(config["slopes"]["subApSpacing"])))
    width = int(config["wfs"]["width"])
    height = int(config["wfs"]["height"])
    num_regions = min(width, height) // spacing
    return (2 * num_regions, num_regions)


def _wfc_2d_shape(config: dict) -> tuple[int, int]:
    num_modes = int(config["wfc"]["numModes"])
    rows = int(round(np.sqrt(float(num_modes * 2))))
    while rows > 1 and num_modes % rows != 0:
        rows -= 1
    cols = num_modes // rows
    return (rows, cols)


def _load_soft_example_module():
    spec = importlib.util.spec_from_file_location("synthetic_shwfs_soft_rtc_example", _SOFT_EXAMPLE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load soft example from {_SOFT_EXAMPLE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None) -> int:
    module = _load_soft_example_module()
    return int(module.main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
