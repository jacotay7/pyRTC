"""Can a single-frame focal-plane network HOLD a closed loop it did not close?

Ideal WFS closes the loop for `--switch` frames, then the trained exp02 network
takes over as the only sensor. Separates "maintenance" (G2) from bootstrap (G3).
"""

import argparse
import json
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fpsim import system as S  # noqa: E402
from fpsim.atmos import Turbulence  # noqa: E402
from fpsim.loop import DM, ModalProjector, h_band_science, run_loop  # noqa: E402
from fpsim.nets import FPNet  # noqa: E402
from fpsim.sensor import FocalPlaneSensor, FPSensorConfig  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--switch", type=int, default=300)
p.add_argument("--steps", type=int, default=1500)
p.add_argument("--gains", type=float, nargs="+", default=[0.2, 0.4])
p.add_argument("--photons", type=float, default=1e4)
args = p.parse_args()
dev = "cuda"
cfg = S.KeckConfig()
sysd = S.build(cfg)
pupil = torch.tensor(sysd["pupil"], device=dev)
dm = DM(torch.tensor(sysd["ifs"], device=dev), torch.tensor(sysd["m2c"], device=dev), cfg.n_pupil)
proj = ModalProjector(dm, pupil)
var = torch.tensor(sysd["kl_variance"], device=dev, dtype=torch.float32)
scale = var.sqrt() * 500 / (2 * torch.pi)
sci = h_band_science(pupil, cfg.grid_m)
OUT = ROOT / "results" / "exp02"
report = {}
for dfc in (0.0, 1.0):
    sensor = FocalPlaneSensor(FPSensorConfig(defocus_rad=dfc, photons=args.photons), pupil, cfg.grid_m).to(dev)
    net = FPNet(1, dm.surfaces.shape[0]).to(dev)
    net.load_state_dict(torch.load(OUT / f"defocus_{dfc}.pt"))
    net.eval()

    def recon(residual, k, hist):
        if k < args.switch:
            return proj(residual.mean(0))
        with torch.no_grad():
            return net(sensor.preprocess(sensor.frame(residual))) * scale * 1e-9

    for g in args.gains:
        turb = Turbulence(cfg, batch=4, seed=100, seeing=0.6)
        r = run_loop(turb, dm, recon, sci, pupil, 1000, args.steps, gain=g, delay=2, settle=args.switch + 300)
        traj = r.strehl_se.mean(1)
        key = f"defocus {dfc} gain {g}"
        report[key] = dict(se_traj=traj.tolist(), strehl_le=r.strehl_le.tolist(),
                           residual_nm=float(r.residual_nm[args.switch + 300:].mean()))
        print(f"{key}: LE Strehl after hand-over {r.strehl_le.mean():.3f} (+-{r.strehl_le.std():.3f}), "
              f"residual {report[key]['residual_nm']:.0f} nm; SE every 100: {[round(float(v), 2) for v in traj[::100]]}",
              flush=True)
(OUT / "handover.json").write_text(json.dumps(report))
