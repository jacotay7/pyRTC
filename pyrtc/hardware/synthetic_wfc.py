"""Launchable hard-RTC entry point for the synthetic wavefront corrector."""

from pyRTC.manager import launchComponent
from pyRTC.hardware.SyntheticSystems import SyntheticWFC


if __name__ == "__main__":
    launchComponent(SyntheticWFC, "wfc", start=True)
