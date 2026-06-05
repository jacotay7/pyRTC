"""Launchable hard-RTC entry point for the synthetic science-camera adapter."""

from pyRTC.Pipeline import launchComponent
from pyRTC.hardware.SyntheticSystems import SyntheticScienceCamera


if __name__ == "__main__":
    launchComponent(SyntheticScienceCamera, "psf", start=True)
