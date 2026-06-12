from importlib import import_module


_EXPORTS = {
    "ALPAODM": (".alpao_dm", "ALPAODM"),
    "SPECULAInterface": (".specula_interface", "SPECULAInterface"),
    "SpinCam": (".spinnaker_science_cam", "SpinCam"),
    "XIMEA_WFS": (".ximea_wfs", "XIMEA_WFS"),
    "PIModulator": (".pi_modulator", "PIModulator"),
    "NCPAOptimizer": (".ncpa_optimizer", "NCPAOptimizer"),
    "PIDOptimizer": (".pid_optimizer", "PIDOptimizer"),
    "SyntheticSHWFS": (".synthetic_systems", "SyntheticSHWFS"),
    "SyntheticScienceCamera": (".synthetic_systems", "SyntheticScienceCamera"),
    "SyntheticWFC": (".synthetic_systems", "SyntheticWFC"),
    "LoopOptimizer": (".loop_hyperparams_optimizer", "LoopOptimizer"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    try:
        module = import_module(module_name, __name__)
    except Exception as exc:
        raise ImportError(
            f"Unable to import pyrtc.hardware.{name}. This usually means the required "
            f"vendor SDK or optional dependency is not installed. Original error: {exc}"
        ) from exc

    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))