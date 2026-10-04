"""R2: linear focal-plane reconstructor calibrated like an SH (push-pull interaction matrix).

Calibration is what a real system can do on its internal source: no turbulence,
no fitting error, push-pull every DM mode around the (defocused) reference image,
record flux-normalised frames. Reconstructor = noise-weighted pseudo-inverse.
Then: (a) hand-over from the ideal sensor (maintenance), (b) bootstrap from open
loop, for several gains / photon levels.
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
from fpsim.sensor import FocalPlaneSensor, FPSensorConfig  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--defocus", type=float, nargs="+", default=[0.0, 1.0])
p.add_argument("--photons", type=float, nargs="+", default=[1e4, 1e5])
p.add_argument("--gains", type=float, nargs="+", default=[0.2, 0.4])
p.add_argument("--steps", type=int, default=1500)
p.add_argument("--switch", type=int, default=300)
p.add_argument("--rcond", type=float, default=1e-2)
p.add_argument("--wavelength", type=float, default=1.65e-6)
p.add_argument("--ref", choices=["lab", "closedloop"], default="closedloop",
               help="lab: diffraction-limited reference; closedloop: mean frame of a closed loop")
p.add_argument("--tag", default="linear_im")
p.add_argument("--n-control", type=int, nargs="+", default=[300], help="modes controlled (rest left alone)")
args = p.parse_args()
dev = "cuda"
cfg = S.KeckConfig()
sysd = S.build(cfg)
pupil = torch.tensor(sysd["pupil"], device=dev)
dm = DM(torch.tensor(sysd["ifs"], device=dev), torch.tensor(sysd["m2c"], device=dev), cfg.n_pupil)
proj = ModalProjector(dm, pupil)
n_modes = dm.surfaces.shape[0]
sci = h_band_science(pupil, cfg.grid_m)
OUT = ROOT / "results" / "exp07"
OUT.mkdir(parents=True, exist_ok=True)
EXP = "exp07_linear_im"


class LinearFP:
    def __init__(self, sensor, amp_nm=10.0, rcond=1e-2, n_control=n_modes):
        self.sensor = sensor
        self.n_control = n_control
        zero = torch.zeros(1, cfg.n_pupil, cfg.n_pupil, device=dev)
        self.ref = self.norm(sensor.frame(zero, noise=False))[0]
        cols = []
        for i in range(n_control):
            e = torch.zeros(1, n_modes, device=dev)
            e[0, i] = amp_nm * 1e-9
            ip = self.norm(sensor.frame(dm.opd(e), noise=False))[0]
            im = self.norm(sensor.frame(dm.opd(-e), noise=False))[0]
            cols.append((ip - im) / (2 * amp_nm))
        j = torch.stack(cols, 1)  # (npix^2, n_modes) per nm
        # Poisson weighting at the reference (+ read-noise floor), in flux-fraction units
        ph = sensor.cfg.photons
        w = (ph / (ph * self.ref + sensor.cfg.read_noise**2 / ph * ph)).clamp_max(1e12).sqrt()
        jw = j * w[:, None]
        u, s, vh = torch.linalg.svd(jw.double(), full_matrices=False)
        keep = s > rcond * s[0]
        self.rec = ((vh[keep].T / s[keep]) @ u[:, keep].T).float() * w[None, :]
        self.n_kept = int(keep.sum())

    @staticmethod
    def norm(e):
        return (e / e.sum((-2, -1), keepdim=True).clamp_min(1e-9)).flatten(1)

    def __call__(self, frame_e):
        est = (self.norm(frame_e) - self.ref) @ self.rec.T  # nm, first n_control modes
        return torch.nn.functional.pad(est, (0, n_modes - self.n_control))


def closed_loop_reference(sensor, lin, mask):
    """Mean flux-normalised frame of a converged loop (ideal sensor, separate atmosphere seeds).

    On sky this is the long-term average frame with the loop closed; it absorbs the
    mean halo of the uncorrectable fitting error.
    """
    acc = []

    def cb(k, residual, **kw):
        if k >= 300:
            acc.append(lin.norm(sensor.frame(residual[None], noise=False)).mean(0))

    run_loop(Turbulence(cfg, batch=4, seed=900, seeing=0.6), dm, lambda r, k, h: proj(r.mean(0)) * mask, sci, pupil,
             1000, 800, gain=0.4, settle=300, callback=cb)
    return torch.stack(acc).mean(0)


report = {}
fig, axes = P.plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
ci = 0
for dfc in args.defocus:
    for ph in args.photons:
        sensor = FocalPlaneSensor(FPSensorConfig(defocus_rad=dfc, photons=ph, wavelength=args.wavelength),
                                  pupil, cfg.grid_m).to(dev)
        for nc in args.n_control:
            mask = torch.zeros(n_modes, device=dev)
            mask[:nc] = 1
            lin = LinearFP(sensor, rcond=args.rcond, n_control=nc)
            if args.ref == "closedloop":
                lin.ref = closed_loop_reference(sensor, lin, mask)
            for g in args.gains:
                for mode, ax in (("handover", axes[0]), ("bootstrap", axes[1])):
                    def recon(residual, k, hist, mode=mode, lin=lin, mask=mask, sensor=sensor):
                        if mode == "handover" and k < args.switch:
                            return proj(residual.mean(0)) * mask
                        return lin(sensor.frame(residual)) * 1e-9

                    turb = Turbulence(cfg, batch=4, seed=100, seeing=0.6)
                    settle = args.switch + 300
                    r = run_loop(turb, dm, recon, sci, pupil, 1000, args.steps, gain=g, leak=0.99, delay=2,
                                 settle=settle)
                    traj = r.strehl_se.mean(1)
                    key = f"{mode} defocus {dfc} photons {ph:.0e} modes {nc} gain {g}"
                    report[key] = dict(se_traj=traj.tolist(), strehl_le=r.strehl_le.tolist(),
                                       residual_nm=float(r.residual_nm[settle:].mean()), n_kept=lin.n_kept)
                    print(f"{key}: LE Strehl {r.strehl_le.mean():.3f} (+-{r.strehl_le.std():.3f}), residual "
                          f"{report[key]['residual_nm']:.0f} nm [{lin.n_kept} modes kept]", flush=True)
                    if g == args.gains[0]:
                        ax.plot(traj, lw=1.3, color=P.SERIES[ci % 8],
                                label=f"{'in focus' if dfc == 0 else f'defocus {dfc:g}'}, {ph:.0e} ph, {nc} modes")
            ci += 1
for ax, t in zip(axes, ("hand-over from the ideal sensor at frame 300", "bootstrap from open loop")):
    ax.set_title(t)
    ax.set_xlabel("frame (1 kHz)")
    ax.set_ylim(0, 1)
axes[0].set_ylabel(f"short-exposure H Strehl (gain {args.gains[0]})")
axes[0].legend(fontsize=7.5)
(OUT / f"{args.tag}.json").write_text(json.dumps(report))
P.save(fig, EXP, args.tag,
       f"""**Linear focal-plane reconstructor calibrated like a Shack-Hartmann**: on the internal source (no
turbulence) each of the 300 DM-KL modes is pushed and pulled by 10 nm and the flux-normalised frame
change recorded (sensing at {args.wavelength * 1e6:.2f} um, reference = {args.ref} mean frame); the reconstructor is the Poisson-weighted pseudo-inverse (singular values below
{args.rcond:g} x max dropped). No model, no training: exactly what a real RTC can calibrate. Leaky
integrator (leak 0.99), 1 kHz, 2-frame delay, mean of 4 atmospheres at 0.6".
Left: the ideal sensor closes the loop, then the linear focal-plane reconstructor takes over at
frame 300. Right: the same reconstructor alone from open loop. **What to look at:** whether the
linear map holds the loop once closed (maintenance, goal G2) even though it cannot close it
(a linear map only works in the small-phase regime; bootstrap needs something else).""",
       title="exp07: linear (interaction-matrix) focal-plane reconstructor")
