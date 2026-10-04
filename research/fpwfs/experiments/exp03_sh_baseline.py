"""Legacy Keck II SH baseline: 20 x 20 SH (makewfs) on OCAM2K (getframes), 349-act DM.

Sweeps guide-star magnitude at fixed rate, plus the Keck camera-mode
rate for each magnitude, and saves results/exp03/*.json for plotting.
"""

import argparse
import json
import pathlib
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fpsim import system as S  # noqa: E402
from fpsim.atmos import Turbulence  # noqa: E402
from fpsim.loop import DM, h_band_science, run_loop  # noqa: E402
from fpsim.shwfs import SHConfig, SHSensor  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--mags", type=float, nargs="+", default=[8.0, 10.0, 12.0, 14.0])
p.add_argument("--rate", type=float, default=1000.0)
p.add_argument("--steps", type=int, default=1500)
p.add_argument("--batch", type=int, default=4)
p.add_argument("--gain", type=float, default=0.5)
p.add_argument("--seeing", type=float, default=0.6)
p.add_argument("--tag", default="mag_sweep")
args = p.parse_args()
dev = "cuda"
OUT = ROOT / "results" / "exp03"
OUT.mkdir(parents=True, exist_ok=True)

# makewfs keck_haka: eng519 (V = 10.16) gives 52.18 M photons/s in the SH windows
# (validated to +0.5 % against the RTC cube). Legacy WFS throughput is assumed equal.
PH_PER_S_V1016 = 52.18e6

cfg = S.KeckConfig()
sysd = S.build(cfg)
pupil = torch.tensor(sysd["pupil"], device=dev)
dm = DM(torch.tensor(sysd["ifs"], device=dev), torch.tensor(sysd["m2c"], device=dev), cfg.n_pupil)
sci = {"H": h_band_science(pupil, cfg.grid_m)}  # K follows from the residual (Marechal)

results = []
for mag in args.mags:
    photons = PH_PER_S_V1016 * 10 ** (-0.4 * (mag - 10.16)) / args.rate
    sh = SHSensor(
        SHConfig(spot_sampling=0.32, pixels_per_subaperture=4, photons_per_frame=photons, frame_rate_hz=args.rate),
        sysd["pupil"], cfg.grid_m,
    )
    sh.calibrate(dm, dm.surfaces.shape[0])

    def recon(residual, k, hist):
        return sh(residual)

    row = dict(mag=mag, photons_per_frame=photons, rate=args.rate, gain=args.gain)
    for band, imager in sci.items():
        turb = Turbulence(cfg, batch=args.batch, seed=100, seeing=args.seeing)
        t0 = time.perf_counter()
        r = run_loop(turb, dm, recon, imager, pupil, args.rate, args.steps, gain=args.gain, leak=0.99, delay=2, settle=500)
        row[band] = dict(
            strehl_le=r.strehl_le.tolist(), se_traj=r.strehl_se.mean(1)[::10].tolist(),
            residual_nm=float(r.residual_nm[500:].mean()),
        )
        print(
            f"V={mag:4.1f} ({photons:9.0f} ph/frame) {band}: LE Strehl {r.strehl_le.mean():.3f} "
            f"+- {r.strehl_le.std():.3f}, residual {r.residual_nm[500:].mean():.0f} nm "
            f"[{time.perf_counter() - t0:.0f} s]",
            flush=True,
        )
    results.append(row)
(OUT / f"{args.tag}.json").write_text(json.dumps(dict(args=vars(args), results=results), indent=1))
