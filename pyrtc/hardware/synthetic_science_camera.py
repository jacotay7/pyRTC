"""Launchable hard-RTC entry point for the synthetic science-camera adapter."""

from pyrtc.manager import launch_component
from pyrtc.hardware.synthetic_systems import SyntheticScienceCamera


if __name__ == "__main__":
    launch_component(SyntheticScienceCamera, "psf", start=True)
