"""Preprocessing / units / sign parity between pyRTC's reconstructor path and the harness.

1. Real closed-loop frames: the offline harness (``fpsim.loop.run_loop``, ideal
   sensor, gain 0.4, leak 0.99, 2-frame delay) on the run's atmosphere; frames
   rendered with ``FocalPlaneSensor.frame``.
2. Each frame goes through exactly what pyRTC does: rounded to integer
   electrons, stored transposed (width, height) as the ``wfs`` stream holds it,
   then ``TorchModelRunner`` (the reconstructor's real-time path) with
   ``flux_normalization: sum``, ``sqrt_stretch: true``, the ``FPInputAdapter``
   factory and ``output_scale_file``.
3. Checks: the network input inside the runner equals
   ``FocalPlaneSensor.preprocess`` of the same (rounded) frame; the runner
   output (metres) equals the harness's ``net(preprocess(e)) * _scale * 1e-9``;
   float16 + CUDA graph vs float32; and the sign/units against the truth
   (projection of the residual on the 120 modes): regression slope > 0, ~0.7.
4. Static poke: no turbulence, DM command +c on a flat wavefront (pyRTC's DM
   adds its OPD), the reconstructor should read back ~+c.
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
from models import DEFAULT_WEIGHTS, build_fpnet, export_scale, load_state  # noqa: E402

from fpsim import system as S  # noqa: E402
from fpsim.atmos import Turbulence  # noqa: E402
from fpsim.loop import DM, ModalProjector, h_band_science, run_loop  # noqa: E402
from fpsim.nets import FPNet  # noqa: E402
from fpsim.sensor import FocalPlaneSensor, FPSensorConfig  # noqa: E402
from pyrtc.image_reconstructor import TorchModelRunner  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--device", default="cuda:0")
p.add_argument("--seed", type=int, default=4000)
p.add_argument("--out", default=str(HERE.parent / "results" / "rtc" / "parity.json"))
args = p.parse_args()
dev = torch.device(args.device)
torch.cuda.set_device(dev)
torch.manual_seed(0)

cfg = S.KeckConfig()
sysd = S.build(cfg)
pupil = torch.tensor(sysd["pupil"], device=dev)
m2c = torch.tensor(sysd["m2c"], device=dev)
ifs = torch.tensor(sysd["ifs"], device=dev)
dm = DM(ifs, m2c, cfg.n_pupil)
proj = ModalProjector(dm, pupil)
sensor = FocalPlaneSensor(FPSensorConfig(defocus_rad=1.0, photons=1e5), pupil, cfg.grid_m).to(dev)
NC = 120
mask = torch.zeros(300, device=dev)
mask[:NC] = 1

# 1. closed-loop residuals from the harness (ideal sensor), frames 200..299
res_store = []


def keep(k, residual, **_):
    if k >= 200:
        res_store.append(residual[0].clone())


run_loop(Turbulence(cfg, batch=1, seed=args.seed, seeing=0.6), dm,
         lambda r, k, h: proj(r.mean(0)) * mask, h_band_science(pupil, cfg.grid_m), pupil,
         1000, 300, gain=0.4, leak=0.99, delay=2, settle=0, callback=keep)
residuals = torch.stack(res_store)  # (100, n, n)
truth_m = proj(residuals)[:, :NC]  # (100, 120) metres
electrons = sensor.frame(residuals)  # (100, 64, 64) float, (H, W)
rounded = electrons.round()
stream_frames = rounded.transpose(-1, -2).to(torch.int32).cpu().numpy()  # as the wfs stream holds them

# harness network path
state, scale_nm = load_state(DEFAULT_WEIGHTS)
net = FPNet(1, NC, width=24, stem_stride=2).to(dev)
net.load_state_dict(state)
net.eval()
with torch.no_grad():
    harness_m = net(sensor.preprocess(electrons)) * scale_nm.to(dev) * 1e-9
    harness_rounded_m = net(sensor.preprocess(rounded)) * scale_nm.to(dev) * 1e-9

scale_path = HERE.parent / "results" / "rtc" / "scale_m.npy"
scale_path.parent.mkdir(parents=True, exist_ok=True)
scale_m = export_scale(DEFAULT_WEIGHTS, scale_path)


def runner(dtype, graph, capture=None):
    model = build_fpnet()
    if capture is not None:
        model.net.register_forward_pre_hook(lambda m, inp: capture.append(inp[0].detach().float().clone()))
    return TorchModelRunner(model, image_shape=(64, 64), image_dtype=np.int32, signal_size=NC,
                            device=str(dev), dtype=dtype, flux_normalization="sum", sqrt_stretch=True,
                            output_scale=scale_m, cuda_graph=graph)


captured = []
r32 = runner("float32", False, captured)
captured.clear()  # drop warm-up passes
out32 = np.stack([r32.run(f).copy() for f in stream_frames])
net_in = torch.cat(captured)[-len(stream_frames):]  # (100, 1, 64, 64) as the network saw it
ref_in = sensor.preprocess(rounded)
ref_in_float = sensor.preprocess(electrons)
r16 = runner("float16", True)
out16 = np.stack([r16.run(f).copy() for f in stream_frames])


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


t = truth_m.cpu().numpy()
h = harness_m.cpu().numpy()
report = {
    "frames": len(stream_frames),
    "graph_active_fp16": bool(r16.graph_active),
    "input_max_abs_diff_vs_preprocess_rounded": float((net_in - ref_in).abs().max()),
    "input_max_abs_vs_preprocess": float(ref_in.abs().max()),
    "input_rel_diff_vs_preprocess_unrounded": rel(net_in.cpu().numpy(), ref_in_float.cpu().numpy()),
    "out_fp32_rel_diff_vs_harness_rounded": rel(out32, harness_rounded_m.cpu().numpy()),
    "out_fp32_rel_diff_vs_harness_float_frames": rel(out32, h),
    "out_fp16_graph_rel_diff_vs_fp32": rel(out16, out32),
    "truth_rms_nm": float(np.sqrt((t ** 2).sum(-1).mean()) * 1e9),
    "error_over_truth_fp32": rel(out32, t),
    "error_over_truth_fp16": rel(out16, t),
    "error_over_truth_harness": rel(h, t),
    # per-mode regression slope of the estimate on the truth (sign and units)
    "slope_median_fp16": float(np.median((out16 * t).sum(0) / (t * t).sum(0))),
    "slope_global_fp16": float((out16 * t).sum() / (t * t).sum()),
}

# 4. static poke on a flat wavefront: the DM adds +OPD(M2C c)
poke = np.zeros(NC, dtype=np.float32)
pokes = {}
for mode, amp in [(0, 30e-9), (3, 30e-9), (10, 20e-9), (50, 10e-9)]:
    poke[:] = 0
    poke[mode] = amp
    act = m2c[:, :NC] @ torch.tensor(poke, device=dev)
    opd = (ifs @ act).reshape(cfg.n_pupil, cfg.n_pupil)
    outs = []
    for _ in range(20):
        f = sensor.frame(opd[None])[0].round().T.to(torch.int32).cpu().numpy()
        outs.append(r16.run(f).copy())
    o = np.mean(outs, 0)
    pokes[f"mode{mode}_{amp * 1e9:.0f}nm"] = dict(readback_nm=float(o[mode] * 1e9),
                                                 other_modes_rms_nm=float(np.sqrt(np.mean(np.delete(o, mode) ** 2)) * 1e9))
report["static_poke"] = pokes
pathlib.Path(args.out).write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1))
