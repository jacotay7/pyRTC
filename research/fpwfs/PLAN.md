# Focal-plane-only AO at > 500 Hz — plan and goals

**Question.** Can a single focal-plane camera, with no Shack-Hartmann, pyramid, or
other pupil-plane sensor, serve as the *only* wavefront sensor of an 8-m-class
single-conjugate AO system, close the loop at >= 500 Hz (target 1 kHz),
bootstrap from seeing-limited conditions, and match a Shack-Hartmann's
closed-loop performance?

**Value proposition.** Same performance as an SH system without the lenslet
array, the WFS relay optics, or (in the limit) a separate WFS camera. It also
brings three things an SH system cannot do: it senses at the science focal
plane, so non-common-path aberrations vanish; it sees island/petal piston
(low-wind effect) that slope sensors are blind to; and it has the highest
photon efficiency of any WFS once the loop runs at high Strehl.

**Novelty.** In the published record (see [LITERATURE.md](LITERATURE.md), §8),
focal-plane-only loops are slow, high-Strehl, low-order *second* stages:

- Fast & Furious on SCExAO: 4-25 Hz, <= 50 modes [R16].
- Fast & Furious at Keck: 30 modes [R17].
- vAPP: 30 modes [R32].
- Photonic lantern: 1 kHz, but only 5 modes [R35].

No one has shown a *first-stage* focal-plane-only loop correcting full
turbulence with hundreds of modes at >= 500 Hz, nor bootstrapping one from
seeing-limited conditions. Every slow demonstration was limited by software,
not by the maths: a neural reconstructor already runs at 125 us (fp16) for
1563 modes at 2 kHz on sky, on a pyramid [R72].

See [LITERATURE.md](LITERATURE.md) for the review this plan rests on ([Rnn] tags
below refer to it), and [ENVIRONMENT.md](ENVIRONMENT.md) for tools and measured
GPU headroom.

## Guiding constraints (from the brief)

1. **Implementable on a real system.** Every input the reconstructor uses must exist in
   a real RTC at runtime: focal-plane frames, past DM commands, loop
   telemetry. No truth phase, no auxiliary WFS, not even for bootstrapping.
2. **Single focal-plane camera.** A static optic in front of it (a pupil
   mask or a fixed defocus in a dedicated WFS arm) is allowed, but it is a
   design choice to be justified, not assumed.
3. **Data-driven first.** Calibration should be something one can do on the
   bench or on-sky, the way you take an interaction matrix. Model-based pieces
   are acceptable when their parameters are easily calibrated.
4. **GPU + ML are in scope.** Latency is measured on the RTX 4060 here and
   scaled to a modern RTC GPU.
5. **Free choice of camera and band at first.** It is a trade-study
   variable, not fixed.

## Reference system ("8-m class, representative")

| Item | Baseline (Tier 1) | Stretch (Tier 2) |
| --- | --- | --- |
| Telescope | 8.0 m, 14 % central obscuration, 4 spiders (VLT-like) | same |
| DM | 20 x 20 actuators (~ 320 valid, Keck-like), ~300 KL modes | 40 x 40 (SPHERE-like), ~1000-1200 modes |
| Atmosphere | pyturb `paranal-median`, 0.8" at 500 nm, L0 = 25 m; sweeps 0.5-1.2" | + strong-wind / fast tau0 cases, LWE |
| Loop | 1 kHz frame rate, 2-frame total delay, integrator/leaky + modal gains | 2 kHz |
| Source | NGS on axis; magnitude sweep | |
| Focal-plane WFS | NIR, narrowband (5-10 %), Nyquist at lambda_min, eAPD (C-RED One-class) via getframes | visible (OCAM2K-class EMCCD) |
| SH baseline | 20 x 20 SH in R+I (OCAM2K-class), same DM/basis/delay/rate | 40 x 40 |

First-order numbers ([tools/budget.py](tools/budget.py)): r0(H) = 0.52 m,
D/r0 = 15; open-loop phase 3.6 rad rms in H (2.7 in K) with tip/tilt removed,
so acquisition is deep in the non-linear regime. Fitting error alone caps
Strehl at 0.84 (H) / 0.47 (I) for 20 x 20. The focal-plane frame only needs
~56 x 56 px (20 x 20) to ~104 x 104 px (40 x 40) to cover the control radius
at Nyquist. Photons at 1 kHz with 0.3 throughput: H = 8 gives ~14 k ph/frame,
H = 12 gives ~360.

## Goals and success criteria

| ID | Goal | Pass criterion |
| --- | --- | --- |
| G1 | **Real time** | Reconstructor + control on GPU at 1 kHz: p99 compute <= 0.5 ms per frame (RTX 4060) for the Tier-1 system, measured inside pyRTC from WFS stream to DM stream; 2-frame total loop delay. |
| G2 | **Parity with SH** | Tier 1, 0.8", bright NGS: long-exposure H-band Strehl of the focal-plane loop >= 90 % of the SH loop's (same DM, basis, rate, delay, photon budget per their own bands). |
| G3 | **Bootstrap** | From open loop with no other sensor, reach G2's steady state within 1 s (1000 frames) for >= 95 % of 100 atmosphere seeds; never diverge. |
| G4 | **Robustness** | Strehl vs. guide-star magnitude and seeing within the same envelope as SH (report the magnitude where each loses 50 % of its bright-star Strehl); tolerant to +-10 % DM/pupil misregistration and 20 % bandwidth. |
| G5 | **Data-driven calibration** | A reconstructor trained only on data a real system can collect (DM-commanded aberrations on a calibration source + closed-loop telemetry) reaches >= 95 % of the simulation-trained one; it transfers across simulators (train in our GPU sim, test in HCIPy / OOPAO / SPECULA) with < 5 % Strehl loss. |
| G6 | **Beyond SH** | Show what SH cannot do: correct petal/LWE modes and NCPA in the same loop, with a measurable Strehl gain. |

G1 + G2 + G3 together are the headline claim: *closes the loop at >500 Hz on an
8-m system with only a focal-plane camera and matches SH*.

## Technical approach

### The two hard problems

1. **Even-mode sign ambiguity.** A single in-focus image only fixes the
   odd part of the phase; the even part has a sign ambiguity. Diversity breaks
   it. The candidates are ranked by how little hardware they need:
   - **D1 temporal / DM diversity** (the Fast & Furious principle [R11, R12],
     learned in PO4NCPA [R61]): the RTC knows every DM increment, so
     consecutive frames plus the commands between them resolve the sign. No
     hardware at all. Known weakness: once the loop converges, the DM steps
     become tiny and the sign is poorly conditioned. Variant **D1+** adds
     small known random dithers (DO-CRIME-like [R100]). It trades a little
     Strehl for conditioning, and the same dithers also refresh calibration.
   - **D2 asymmetric pupil** [R19-R21]: a small opaque patch (e.g. thickening
     one spider) makes the pupil non-centro-symmetric, so the PSF encodes the
     sign linearly. A few % of throughput. The patch can be Fisher-optimised
     with differentiable optics [R57, R58].
   - **D3 static aberration in a dedicated WFS arm** (a fixed defocus or
     astigmatism). It costs nothing in science Strehl if the arm is separate,
     but it rules out using the science camera itself.
   - D4 holographic/vAPP (needs a custom optic): reference only.
2. **Dynamic range / bootstrapping.**
   - Linearised focal-plane methods hold only to about 1-1.5 rad rms
     [R11, R19]. Open loop is about 3.6 rad rms in H.
   - Single-shot CNNs from seeing-limited images stall at about 1.4-2 rad
     residual [R46, R48].
   - The closed loop is itself an iteration, though, and needs only a
     *contraction*: each step must, on average, reduce the residual in the
     controlled modes, so every frame sees a smaller residual than the last.
     That is a weaker requirement than an accurate single-shot estimate, and
     a good fit for a learned non-linear estimator with uncertainty-scheduled
     gains.
   - Longer wavelength helps directly: tip/tilt-removed phase is 2.7 rad in K
     versus 7.3 in I. Band is a trade variable.

### Reconstructor families to compare (same harness, same data)

| Family | Idea | Why include it |
| --- | --- | --- |
| R0 ideal WFS | true residual, same delay/noise-free | upper bound, separates WFS from control error |
| R1 Fast & Furious (linear, model-based) | odd part from image, even part from previous frame + known DM step | the established focal-plane-only closed-loop method; small-phase baseline |
| R2 linear data-driven | interaction matrix in pixel space around a diversity reference (push-pull on DM) | simplest "calibrate like an SH" method; shows where linearity breaks |
| R3 learned non-linear (main bet) | CNN/ResNet on the last K frames + last K DM increments -> modal residual (+ uncertainty) | handles large phase; CUDA-graphed at ~0.2 ms |
| R4 hybrid | R3 for acquisition; R1/R2 or a residual-trained R3 at high Strehl | best of both if R3 saturates at high Strehl |
| R5 iterative model-based (offline) | differentiable optics phase retrieval, warm-started | accuracy oracle for diagnosing R3, not real-time |

R3 details:
- Input: K = 2-4 normalised frames (sqrt or log stretch, flux-normalised)
  plus the K - 1 DM increments between them, injected as modal vectors
  (FiLM / concatenated after the conv trunk) or rendered as predicted
  image-plane changes.
- Output: modal residual in the DM's KL basis, with a per-mode variance for
  gain scheduling and outlier rejection.
- Training: supervised on simulated *closed-loop* states, not just
  open-loop screens, with DAgger-style iterations (run the loop with the
  current net, relabel the states it visits, retrain) to kill covariate
  shift; amplitude curriculum from small residuals to open loop for G3.
- Inference: fp16, CUDA graph (TensorRT later), single stream; budget
  measured in ENVIRONMENT.md.
- Prior art it builds on: PO4NCPA (current + previous image + previous DM
  action; 55 modes, NCPA, simulation only [R61]), and the MagAO-X pyramid
  network (trained on internal-source DM-injected data, on sky at 2 kHz
  [R71, R72]). The new parts are multi-frame operation, full turbulence,
  hundreds of modes, and bootstrapping.
- Safety: a watchdog (flux, predicted variance, residual growth) that drops
  gains or falls back to a linear reconstructor. Non-linear estimators carry
  no closed-loop stability guarantee [R71].

### The data-driven calibration path (G5)

What a real system can label: DM commands are known exactly. On the internal
calibration source (or on-sky at high Strehl), command random DM shapes drawn
from the expected residual statistics, record frames, and train image ->
DM-shape. This is the non-linear generalisation of measuring an interaction
matrix. It automatically absorbs the true pupil, DM influence functions,
misregistration, NCPA, and detector, which are the things a model gets wrong.
Turbulence beyond the DM's reach acts as noise here, just as it does for an SH.
This is the recipe already proven on sky for a neural pyramid reconstructor
(random DM shapes on the internal source at the end of the night [R72]); iEFC
does the linear version for focal-plane intensities [R92].

Open questions:

- How many calibration frames are needed? A few minutes at 1 kHz is about
  1e5 frames.
- Does online fine-tuning from closed-loop telemetry close the remaining gap?
  The self-supervised signal is the predicted vs. observed frame change after
  a known DM step; differentiable optics would serve as the decoder [R51, R56].

Cross-simulator transfer (train in our sim, test in HCIPy/OOPAO/SPECULA, each
with its own pupil sampling, DM model, and propagation) is our stand-in for the
sim-to-real gap until there is a bench.

## Work plan

Phases are ordered by dependency; each ends with a go/no-go check.

**Phase 0 — Environment and baselines (this step).** Done: tools installed and
checked, GPU headroom measured, literature reviewed, plan written.

**Phase 1 — GPU simulation core** (`research/fpwfs/fpsim/`, torch).
Pupil (VLT-like, spiders, optional asymmetric patch), DM (influence functions
+ aobasis KL), atmosphere (pyturb on GPU via DLPack), polychromatic
focal-plane imager (MFT, Nyquist-sampled), detector (getframes eAPD/EMCCD), SH
baseline (makewfs on GPU), and a closed-loop harness in *simulated* time with
exact frame delays. Batched mode for data generation.
*Check:* PSF/Strehl agree with HCIPy to < 1 %; the SH loop reproduces
OOPAO/SPECULA residuals for the same system within ~5 %.

**Phase 2 — Baselines.** R0 and SH loops over the seeing/magnitude grid (this
is the bar for G2/G4); R1 (F&F) and R2 started from an SH-closed state.
*Check:* F&F holds the loop at high Strehl (reproduces the literature
regime); we know where linear methods fail.

**Phase 3 — Learned reconstructor in the high-Strehl regime.** R3 with each
diversity option D1/D2/D3, single-frame vs. multi-frame, model size vs.
latency. *Check:* R3 >= R1 at high Strehl and matches SH Strehl within 10 %
when started closed (G2 without bootstrap). If D1 alone fails, decide on
D2/D3.

**Phase 4 — Bootstrap.** Curriculum + DAgger training to open-loop
amplitudes, gain scheduling from predicted uncertainty, optional staged
closure (TT -> low order -> full). *Check:* G3.

**Phase 5 — Data-driven calibration and robustness.** DM-only training
protocol, sample-efficiency curve, online fine-tuning, cross-simulator
transfer, misregistration/bandwidth/magnitude/seeing sweeps, LWE and NCPA.
*Check:* G4, G5, G6.

**Phase 6 — Real time in pyRTC.** A focal-plane WFS source (simulator
adapter writing frames to the `wfs` stream) and a torch reconstructor
component in the `slopes` section that writes the modal `signal` (identity
CM in the Loop). GPU streams, CUDA graph, 1 kHz run with `pyrtc.latency`.
*Check:* G1, and the closed loop in pyRTC reproduces the harness Strehl.

**Phase 7 — Stretch.**

- Tier-2 40 x 40, compared with SPHERE/SAXO: 40 x 40 SH at 1.38 kHz,
  median H Strehl 80-90 % in good seeing [R104].
- Visible-band WFS.
- Science camera as the WFS.
- Data-driven predictive control on the pseudo-open-loop modes, which
  attacks the temporal error that dominates SAXO: EOF / SPC [R88, R91] via
  pyRTC `predictive.py`, or PO4AO-style RL [R64, R65].
- Lab demonstration.

## Branch and merge policy

- All research code, notes and results live under `research/fpwfs/` on branch
  `fpwfs` (off `dev`) and are never merged.
- Anything generally useful (a focal-plane camera mode in the simulator
  adapters, a GPU/torch reconstructor hook in the slopes/loop path, latency
  tooling, bug fixes) gets its own branch off `dev` and a PR following the
  repo conventions (CHANGELOG, tests, AGENTS.md). `fpwfs` then rebases onto
  `dev`.
- Changes that belong in the maintainer's other packages (makewfs focal-plane
  imager, pyturb, getframes, aobasis) go to those repositories.

## Main risks

| Risk | Mitigation |
| --- | --- |
| Sign ambiguity not resolved by D1 under noise/latency | D2 (asymmetric pupil) costs a few % throughput and is linear-friendly |
| Learned reconstructor fails far from training distribution (bootstrap, bad seeing) | curriculum + DAgger, uncertainty-gated gains, staged closure fallback |
| Chromatic smearing | speckles smear radially by r * dlambda/lambda, so the control edge needs dlambda/lambda <~ 2/N (10 % for 20 x 20, 5 % for 40 x 40); start narrowband, sweep bandwidth in G4, train polychromatic |
| Few photons per speckle near the control edge [R76] | NN sensitivity vs. spatial frequency is a reported metric; bin pixels outside the control radius |
| Non-linear loop instability / out-of-distribution frames | watchdog + linear fallback, uncertainty-gated gains |
| Sim-to-real gap | DM-only calibration data, cross-simulator tests, online fine-tuning |
| Latency jitter on a shared, display-driving GPU | CUDA graphs, report p99, reproduce on the A400 for a steadier card |
| Photon starvation for faint NGS in narrow NIR band | magnitude sweep vs. SH is part of G4; wider band + polychromatic model |
