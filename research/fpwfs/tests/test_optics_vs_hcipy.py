"""Cross-check the torch focal-plane imager against HCIPy on identical grids."""

import sys
import pathlib

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")
hc = pytest.importorskip("hcipy")

from fpsim import system as S  # noqa: E402
from fpsim.optics import FocalPlaneImager, defocus_opd, strehl  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CFG = S.KeckConfig()
SYSTEM = S.build(CFG)
LAM = 1.65e-6
NPIX, Q = 64, 2.0


def hcipy_image(pupil, opd, lam, lam_ref=LAM):
    """HCIPy MFT on the same pixel grid, as a flux fraction per pixel.

    HCIPy's MFT works in angular spatial frequency k (rad/m, exp(-i k x)): a sky
    angle theta is k = 2 pi theta / lam, so the grid spacing depends on the
    wavelength and Parseval carries a 1 / (2 pi)^2.
    """
    n = pupil.shape[0]
    pgrid = hc.make_pupil_grid(n, CFG.grid_m)
    delta = 2 * np.pi * lam_ref / CFG.grid_m / Q / lam
    zero = -NPIX / 2 * delta
    fgrid = hc.CartesianGrid(hc.RegularCoords([delta, delta], [NPIX, NPIX], [zero, zero]))
    prop = hc.MatrixFourierTransform(pgrid, fgrid)
    field = hc.Field((pupil * np.exp(2j * np.pi * opd / lam)).ravel(), pgrid)
    img = np.abs(np.asarray(prop.forward(field))) ** 2 * fgrid.weights / (2 * np.pi) ** 2
    total = (pupil**2).sum() * pgrid.weights
    return img.reshape(NPIX, NPIX) / total


def _screen(seed, scale_m):
    rng = np.random.default_rng(seed)
    m2c, ifs = SYSTEM["m2c"], SYSTEM["ifs"]
    coeffs = rng.standard_normal(m2c.shape[1]) * np.sqrt(SYSTEM["kl_variance"]) * scale_m
    return (ifs @ (m2c @ coeffs)).reshape(CFG.n_pupil, CFG.n_pupil)


@pytest.mark.parametrize("scale_m", [0.0, 2e-9, 8e-9])
def test_monochromatic_matches_hcipy(scale_m):
    pupil = SYSTEM["pupil"]
    opd = _screen(0, scale_m)
    imager = FocalPlaneImager(torch.tensor(pupil, device=DEV), CFG.grid_m, NPIX, Q, LAM)
    ours = imager(torch.tensor(opd, device=DEV, dtype=torch.float32)).cpu().numpy()
    ref = hcipy_image(pupil, opd, LAM)
    assert np.abs(ours - ref).max() < 2e-3 * ref.max()
    assert abs(ours.sum() - ref.sum()) < 2e-3


def test_polychromatic_defocused_matches_hcipy():
    pupil = SYSTEM["pupil"]
    lams = (1.60e-6, 1.65e-6, 1.70e-6)
    defocus = defocus_opd(CFG.n_pupil, CFG.grid_m, 10.95, 0.1e-6, "cpu").numpy()
    opd = _screen(1, 4e-9)
    imager = FocalPlaneImager(
        torch.tensor(pupil, device=DEV), CFG.grid_m, NPIX, Q, LAM, wavelengths=lams,
        static_opd=torch.tensor(defocus, device=DEV),
    )
    ours = imager(torch.tensor(opd, device=DEV, dtype=torch.float32)).cpu().numpy()
    ref = sum(hcipy_image(pupil, opd + defocus, lam) for lam in lams) / 3
    assert np.abs(ours - ref).max() < 2e-3 * ref.max()


def test_strehl_matches_marechal_small_aberration():
    pupil = SYSTEM["pupil"]
    imager = FocalPlaneImager(torch.tensor(pupil, device=DEV), CFG.grid_m, NPIX, Q, LAM)
    opd = _screen(2, 1e-9)
    inside = pupil > 0.5
    sigma = 2 * np.pi * opd[inside].std() / LAM
    sr = float(strehl(imager, torch.tensor(opd, device=DEV, dtype=torch.float32)))
    assert abs(sr - np.exp(-(sigma**2))) < 0.02
