"""Launchable hard-RTC entry point for the synthetic SHWFS adapter."""

from pyRTC.Pipeline import launchComponent
from pyRTC.hardware.SyntheticSystems import SyntheticSHWFS


if __name__ == "__main__":
    launchComponent(SyntheticSHWFS, "wfs", start=True)
