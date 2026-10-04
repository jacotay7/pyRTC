"""How long does the slim network hold the loop? Offline harness, 10 s per atmosphere.

exp13's robustness test ran 2.3 s (2300 frames). The pyRTC demo runs 10 s, so this
runs the same protocol (300 ideal frames, then the network, gain 0.4, leak 0.99,
delay 2) for ``--steps`` frames on ``--batch`` atmospheres and reports, per
atmosphere, the first frame whose 100-frame mean short-exposure Strehl drops
below 0.3 (time to failure) and the long-exposure Strehl before it.
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
from fpsim.atmos import Turbulence  # noqa: E402
from fpsim.loop import DM, ModalProjector, h_band_science, run_loop  # noqa: E402
from fpsim.nets import FPNet  # noqa: E402
from fpsim.sensor import FocalPlaneSensor, FPSensorConfig  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--device", default="cuda:0")
p.add_argument("--seed", type=int, default=4000)
p.add_argument("--batch", type=int, default=12)
p.add_argument("--steps", type=int, default=10000)
p.add_argument("--delay", type=int, default=2)
p.add_argument("--tag", default="survival_seed4000")
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


import cupy  # noqa: E402

with cupy.cuda.Device(dev.index or 0):
    r = run_loop(Turbulence(cfg, batch=args.batch, seed=args.seed, seeing=0.6), dm, recon,
                 h_band_science(pupil, cfg.grid_m), pupil, 1000, args.steps,
                 gain=0.4, leak=0.99, delay=args.delay, settle=600)
se = r.strehl_se.numpy()  # (steps, B)
k = np.ones(100) / 100
report = dict(args=vars(args), atmospheres=[])
for b in range(args.batch):
    m = np.convolve(se[:, b], k, mode="valid")
    bad = np.nonzero(m[300:] < 0.3)[0]
    fail = int(bad[0] + 300 + 99) if len(bad) else None
    end = fail if fail is not None else args.steps
    report["atmospheres"].append(dict(atmosphere=b, fail_frame=fail,
                                      se_mean_600_to_fail=float(se[600:end, b].mean()) if end > 600 else None))
    print(b, fail, report["atmospheres"][-1]["se_mean_600_to_fail"], flush=True)
held = sum(a["fail_frame"] is None for a in report["atmospheres"])
report["held_full_run"] = held
print(f"held {held}/{args.batch} for {args.steps} frames", flush=True)
out = HERE.parent / "results" / "rtc"
out.mkdir(parents=True, exist_ok=True)
(out / f"{args.tag}.json").write_text(json.dumps(report, indent=1))
np.save(out / f"{args.tag}_se.npy", se.astype(np.float32))
