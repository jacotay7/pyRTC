from importlib import import_module


_EXPORTS = {
    "ALPAODM": (".ALPAODM", "ALPAODM"),
    "spinCam": (".SpinnakerScienceCam", "spinCam"),
    "XIMEA_WFS": (".ximeaWFS", "XIMEA_WFS"),
    "PIModulator": (".PIModulator", "PIModulator"),
    "NCPAOptimizer": (".NCPAOptimizer", "NCPAOptimizer"),
    "PIDOptimizer": (".PIDOptimizer", "PIDOptimizer"),
    "SyntheticSHWFS": (".SyntheticSystems", "SyntheticSHWFS"),
    "SyntheticScienceCamera": (".SyntheticSystems", "SyntheticScienceCamera"),
    "loopOptimizer": (".loopHyperparamsOptimizer", "loopOptimizer"),
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
            f"Unable to import pyRTC.hardware.{name}. This usually means the required "
            f"vendor SDK or optional dependency is not installed. Original error: {exc}"
        ) from exc

    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))