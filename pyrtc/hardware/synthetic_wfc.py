"""Launchable hard-RTC entry point for the synthetic wavefront corrector."""

from pyrtc.manager import launch_component
from pyrtc.hardware.synthetic_systems import SyntheticWFC


if __name__ == "__main__":
    launch_component(SyntheticWFC, "wfc", start=True)
