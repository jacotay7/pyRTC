"""Offline-harness reference for the pyRTC run: same atmosphere, same protocol.

``fpsim.loop.run_loop`` in simulated time on ``Turbulence(seed)`` (the screens
the pyRTC camera renders, when run on the same GPU): ideal sensor for 300 frames,
then the slim network alone, gain 0.4, leak 0.99, fixed delay 2 (and 3, the
delay the free-running pyRTC system realises when its latency exceeds one frame
period). Long-exposure H Strehl from frame 600 on, as exp13 and the pyRTC run.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from models import DEFAULT_WEIGHTS, load_state  # noqa: E402

from fpsim import system as S  # noqa: E402
from fpsim.atmos import Turbulence, make_atmosphere  # noqa: E402
from fpsim.loop import DM, ModalProjector, h_band_science, run_loop  # noqa: E402
from fpsim.nets import FPNet  # noqa: E402
from fpsim.sensor import FocalPlaneSensor, FPSensorConfig  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--device", default="cuda:0")
p.add_argument("--seed", type=int, default=4000)
p.add_argument("--atm-index", type=int, default=0)
p.add_argument("--steps", type=int, default=10000)
p.add_argument("--delays", default="2,3")
p.add_argument("--out", default=str(HERE.parent / "results" / "rtc" / "offline_ref.npz"))
args = p.parse_args()
dev = torch.device(args.device)
torch.cuda.set_device(dev)

cfg = S.KeckConfig()
sysd = S.build(cfg)
pupil = torch.tensor(sysd["pupil"], device=dev)
dm = DM(torch.tensor(sysd["ifs"], device=dev), torch.tensor(sysd["m2c"], device=dev), cfg.n_pupil)
proj = ModalProjector(dm, pupil)
NC = 120
sensor = FocalPlaneSensor(FPSensorConfig(defocus_rad=1.0, photons=1e5), pupil, cfg.grid_m).to(dev)
state, scale = load_state(DEFAULT_WEIGHTS)
scale = scale.to(dev)
net = FPNet(1, NC, width=24, stem_stride=2).to(dev)
net.load_state_dict(state)
net.eval()
mask = torch.zeros(300, device=dev)
mask[:NC] = 1


def recon(residual, k, hist):
    if k < 300:
        return proj(residual.mean(0)) * mask
    with torch.no_grad():
        out = net(sensor.preprocess(sensor.frame(residual))) * scale
        return torch.nn.functional.pad(out, (0, 300 - NC)) * 1e-9


out = {}
summary = {}
with torch.cuda.device(dev):
    import cupy

    with cupy.cuda.Device(dev.index or 0):
        for d in (int(v) for v in args.delays.split(",")):
            turb = Turbulence(cfg, batch=1, seed=args.seed, seeing=0.6)
            if args.atm_index:
                turb.atms = [make_atmosphere(cfg, seeing=0.6, seed=args.seed * Turbulence.SEED_STRIDE + args.atm_index)]
            r = run_loop(turb, dm, recon,
                         h_band_science(pupil, cfg.grid_m), pupil, 1000, args.steps,
                         gain=0.4, leak=0.99, delay=d, settle=600)
            se = r.strehl_se[:, 0].numpy()
            out[f"se_delay{d}"] = se
            out[f"rms_delay{d}"] = r.residual_nm[:, 0].numpy()
            summary[f"delay{d}"] = dict(le_after_600=float(r.strehl_le[0]),
                                        se_mean_after_600=float(se[600:].mean()),
                                        se_mean_warm_200_300=float(se[200:300].mean()))
            print(d, summary[f"delay{d}"], flush=True)
np.savez(args.out, **out)
pathlib.Path(args.out).with_suffix(".json").write_text(json.dumps(dict(args=vars(args), **summary), indent=1))
