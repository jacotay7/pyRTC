"""Model factory for ``TorchImageReconstructor`` (``model_factory_file``).

The focal-plane networks were trained on ``FocalPlaneSensor.preprocess``:
``sqrt(clamp(e, 0) / flux * npix**2)`` = ``npix * sqrt(clamp(e, 0) / flux)``,
on frames in (H, W) layout. The reconstructor's own preprocessing
(``flux_normalization: sum``, ``sqrt_stretch: true``) gives
``sqrt(clamp(e / flux, 0))`` on the stream array, which pyRTC stores as
(width, height). :class:`FPInputAdapter` closes the gap: transpose back to
(H, W) and multiply by ``npix``. The network output is in units of the
training ``_scale`` (per mode, nm); ``output_scale_file`` (written by
:func:`export_scale`) turns it into metres of DM-KL modal coefficient, the
corrector's units.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

FPWFS = pathlib.Path(__file__).resolve().parents[1]
if str(FPWFS) not in sys.path:
    sys.path.insert(0, str(FPWFS))

from fpsim.nets import FPNet  # noqa: E402

DEFAULT_WEIGHTS = FPWFS / "results" / "exp10" / "slim24_nc120_r3.pt"


class FPInputAdapter(torch.nn.Module):
    """(1, 1, W, H) sqrt(e / flux) from the reconstructor -> the network's training input."""

    def __init__(self, net: torch.nn.Module, npix: int = 64, transpose: bool = True):
        super().__init__()
        self.net = net
        self.gain = float(npix)
        self.transpose = bool(transpose)

    def forward(self, x):
        if self.transpose:
            x = x.transpose(-1, -2)
        return self.net(x * self.gain)


def load_state(weights=DEFAULT_WEIGHTS):
    """State dict without ``_scale``, and ``_scale`` (per-mode output scale, nm)."""
    state = torch.load(str(weights), map_location="cpu", weights_only=True)
    scale = state.pop("_scale")
    return state, scale


def build_fpnet(weights: str = str(DEFAULT_WEIGHTS), n_modes: int = 120, width: int = 24,
                stem_stride: int = 2, head: int = 1024, npix: int = 64, transpose: bool = True):
    """The slim exp10 maintenance network (6.7M parameters) behind the input adapter."""
    net = FPNet(1, n_modes, npix=npix, width=width, stem_stride=stem_stride, head=head)
    state, _ = load_state(weights)
    net.load_state_dict(state, strict=True)
    net.eval()
    return FPInputAdapter(net, npix=npix, transpose=transpose).eval()


def export_scale(weights, out_path, units: float = 1e-9) -> np.ndarray:
    """Write ``_scale * units`` (default nm -> m) as the reconstructor's ``output_scale_file``."""
    _, scale = load_state(weights)
    arr = (scale.double().numpy() * units).astype(np.float32)
    np.save(out_path, arr)
    return arr
