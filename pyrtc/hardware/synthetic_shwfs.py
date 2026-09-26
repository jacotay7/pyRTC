"""Launchable hard-RTC entry point for the synthetic SHWFS adapter."""

from pyrtc.manager import launch_component
from pyrtc.hardware.synthetic_systems import SyntheticSHWFS


if __name__ == "__main__":
    launch_component(SyntheticSHWFS, "wfs", start=True)
