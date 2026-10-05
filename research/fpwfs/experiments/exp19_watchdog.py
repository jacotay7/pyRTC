"""Long-horizon robustness (10 s) of the focal-plane loop with simple RTC safeguards.

Failures of the maintenance network are abrupt: one excursion leaves the narrow basin
it was trained on and the loop runs away. Cheap, realistic safeguards on the
controller side (no change to the network):

  none  : plain leaky integrator;
  clip  : the modal update norm is capped at `--clip` x the typical closed-loop
          estimate norm (one outlier estimate cannot kick the DM out of the basin);
  hold  : estimates whose norm exceeds `--hold` x typical are ignored (DM held).

Typical estimate norm is measured from the network itself during the first second
after hand-over (as an RTC would, from telemetry).
"""

import argparse
import json
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fpsim import plotting as P  # noqa: E402
from fpsim import system as S  # noqa: E402
from fpsim.atmos import Turbulence  # noqa: E402
from fpsim.loop import DM, ModalProjector, h_band_science, run_loop  # noqa: E402
from fpsim.nets import FPNet  # noqa: E402
from fpsim.sensor import FocalPlaneSensor, FPSensorConfig  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--weights", default="results/exp10/slim24_nc120_r3.pt")
p.add_argument("--width", type=int, default=24)
p.add_argument("--stem-stride", type=int, default=2)
p.add_argument("--steps", type=int, default=10300)
p.add_argument("--gain", type=float, default=0.4)
p.add_argument("--clip", type=float, default=2.5)
p.add_argument("--hold", type=float, default=4.0)
p.add_argument("--modes", default="none,clip,hold")
p.add_argument("--seed", type=int, default=600)
p.add_argument("--tag", default="watchdog")
args = p.parse_args()
dev = "cuda"
cfg = S.KeckConfig()
sysd = S.build(cfg)
pupil = torch.tensor(sysd["pupil"], device=dev)
dm = DM(torch.tensor(sysd["ifs"], device=dev), torch.tensor(sysd["m2c"], device=dev), cfg.n_pupil)
proj = ModalProjector(dm, pupil)
NC = 120
sensor = FocalPlaneSensor(FPSensorConfig(defocus_rad=1.0, photons=1e5), pupil, cfg.grid_m).to(dev)
st = torch.load(ROOT / args.weights)
scale = st.pop("_scale").to(dev)
net = FPNet(1, NC, width=args.width, stem_stride=args.stem_stride).to(dev)
net.load_state_dict(st)
net.eval()
mask = torch.zeros(300, device=dev)
mask[:NC] = 1
HANDOVER, CAL = 300, 1000  # hand-over frame; frames of telemetry to learn the typical estimate norm


def make_recon(mode):
    state = {"norms": [], "typ": None, "held": 0}

    def recon(residual, k, hist):
        if k < HANDOVER:
            return proj(residual.mean(0)) * mask
        with torch.no_grad():
            est = torch.nn.functional.pad(net(sensor.preprocess(sensor.frame(residual))) * scale, (0, 300 - NC)) * 1e-9
        n = est.norm(dim=-1, keepdim=True)
        if k < HANDOVER + CAL:
            state["norms"].append(n.squeeze(-1))
            return est
        if state["typ"] is None:
            state["typ"] = torch.stack(state["norms"]).median(0).values[:, None]  # per atmosphere (per RTC)
        typ = state["typ"]
        if mode == "clip":
            est = est * torch.clamp(args.clip * typ / n.clamp_min(1e-15), max=1.0)
        elif mode == "hold":
            bad = n > args.hold * typ
            state["held"] += int(bad.sum())
            est = torch.where(bad, torch.zeros_like(est), est)
        return est

    return recon, state


report = {"args": vars(args)}
modes = args.modes.split(",")
fig, axes = P.plt.subplots(1, len(modes), figsize=(3.6 * len(modes), 3.6), sharey=True, squeeze=False)
for ax, mode in zip(axes[0], modes):
    recon, state = make_recon(mode)
    r = run_loop(Turbulence(cfg, batch=12, seed=args.seed, seeing=0.6), dm, recon, h_band_science(pupil, cfg.grid_m),
                 pupil, 1000, args.steps, gain=args.gain, leak=0.99, delay=2, settle=HANDOVER + CAL)
    se = r.strehl_se  # (T, B)
    lost = [(int((se[HANDOVER:, b] < 0.2).nonzero()[0, 0]) + HANDOVER) if (se[HANDOVER:, b] < 0.2).any() else None
            for b in range(se.shape[1])]
    held = sum(x is None for x in lost)
    alive = r.strehl_le[[i for i, x in enumerate(lost) if x is None]]
    report[mode] = dict(held=held, lost_frame=lost, strehl_le=r.strehl_le.tolist(), held_frames=state["held"])
    print(f"{mode:5s}: held {held}/12 for {(args.steps - HANDOVER) / 1000:.0f} s; lost at frames {[x for x in lost if x]}; "
          f"LE Strehl of survivors {alive.mean() if len(alive) else float('nan'):.3f}; estimates held {state['held']}",
          flush=True)
    for b in range(12):
        ax.plot(se[::10, b], lw=0.6, color=P.SERIES[0] if lost[b] is None else P.SERIES[7], alpha=0.8)
    ax.set_title(f"{mode}: {held}/12 survive 10 s", fontsize=9.5)
    ax.set_xlabel("frame / 10 (1 kHz)")
    ax.set_ylim(0, 1)
axes[0][0].set_ylabel("short-exposure H Strehl")
out = ROOT / "results" / "exp19"
out.mkdir(parents=True, exist_ok=True)
(out / f"{args.tag}.json").write_text(json.dumps(report))
P.save(fig, "exp19_watchdog", args.tag,
       f"""10 s of focal-plane-only closed loop (slim maintenance network, 120 modes, gain {args.gain}, leak 0.99, 1 kHz,
0.6") on 12 unseen atmospheres, after a 300-frame ideal warm start, with three controller safeguards: none;
clip (the modal update norm capped at {args.clip:g}x its typical value, learned from the first second of telemetry);
hold (estimates above {args.hold:g}x typical ignored). Blue: never lost (SE Strehl stays > 0.2), red: lost.
**What to look at:** whether a trivial RTC-side safeguard turns the abrupt failures (mean time to failure
~30-40 s per atmosphere without it) into survivals.""",
       title="exp19: long-horizon robustness and RTC safeguards")
