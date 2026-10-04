"""First-order numbers for choosing the focal-plane WFS band and DM size.

Prints, per band: r0, D/r0, open-loop and tip/tilt-removed phase rms, the
Marechal Strehl allowed by DM fitting error alone, the focal-plane sampling
needed to see the whole control radius, and photons per frame.
Textbook scalings only (Noll 1976, Hudgin fitting coefficient 0.28); the
simulations replace these numbers once they exist.
"""

from __future__ import annotations

import numpy as np

D = 8.0  # m
OBSCURATION = 0.14
SEEING_ARCSEC = 0.8  # at 500 nm
THROUGHPUT = 0.3  # telescope + AO + filter + QE, to the WFS detector
FRAME_RATE = 1000.0  # Hz
# Vega zero points: centre (um), photons s^-1 m^-2 um^-1 at mag 0, fractional bandwidth
BANDS = {
    "I": (0.79, 4.5e10, 0.15),
    "J": (1.22, 1.9e10, 0.10),
    "H": (1.63, 9.3e9, 0.10),
    "K": (2.19, 4.7e9, 0.10),
}
ACTUATORS_ACROSS = (20, 32, 40)


def main() -> None:
    r0_500 = 0.98 * 0.5e-6 / np.deg2rad(SEEING_ARCSEC / 3600)
    area = np.pi * (D / 2) ** 2 * (1 - OBSCURATION**2)
    print(
        f'D={D} m, seeing {SEEING_ARCSEC}" -> r0(500nm)={r0_500 * 100:.1f} cm, area {area:.1f} m^2\n'
    )
    print(
        "band  r0[m]  D/r0  rms[rad]  rms-TT[rad]  "
        + "  ".join(f"SR_fit({n}x{n})" for n in ACTUATORS_ACROSS)
    )
    for name, (lam, _, _) in BANDS.items():
        r0 = r0_500 * (lam / 0.5) ** 1.2
        x = (D / r0) ** (5 / 3)
        fits = [np.exp(-0.28 * ((D / n) / r0) ** (5 / 3)) for n in ACTUATORS_ACROSS]
        print(
            f"{name:4s} {r0:6.2f} {D / r0:5.1f} {np.sqrt(1.03 * x):8.1f} {np.sqrt(0.134 * x):11.1f}  "
            + "  ".join(f"{f:13.2f}" for f in fits)
        )

    print("\nfocal-plane frame to cover the control radius (N/2 lambda/D) at Nyquist, +25% margin:")
    for n in ACTUATORS_ACROSS:
        npx = int(np.ceil(2 * n * 1.25 / 8) * 8)
        print(f"  {n}x{n} DM: >= {2 * n} px, use {npx}x{npx} px")

    print(f"\nphotons per frame at {FRAME_RATE:.0f} Hz, throughput {THROUGHPUT}:")
    mags = (6, 8, 10, 12, 14)
    print("band  " + "  ".join(f"m={m:<7d}" for m in mags))
    for name, (lam, zp, frac) in BANDS.items():
        n = [zp * 10 ** (-0.4 * m) * area * THROUGHPUT * lam * frac / FRAME_RATE for m in mags]
        print(f"{name:4s}  " + "  ".join(f"{v:9.3g}" for v in n))


if __name__ == "__main__":
    main()
