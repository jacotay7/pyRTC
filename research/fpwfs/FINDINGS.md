# Findings log

Running record of what the experiments showed and why the plan changed. The
figures live in `plots/` (untracked); start at `plots/README.md`.

## 2026-10-03

**Simulation core validated against independent tools.**
- The torch focal-plane imager matches HCIPy within 0.2 % of peak: monochromatic,
  turbulent, and polychromatic + defocus cases (`tests/test_optics_vs_hcipy.py`).
- DM-KL mode variances agree to a few % across three sources: aobasis (analytic),
  pyturb screens projected on the modes, and a von Karman PSD integral. The 20 m
  outer scale cuts tilt variance to 0.08x Kolmogorov. That was larger than I expected,
  and the theory confirmed it.
- The Keck model: makewfs's 36-segment pupil (72.0 m^2), and 349 actuators on the
  21 x 21 Fried grid, which matches Keck's count exactly.

**exp01: ideal-sensor ceiling.**
- At 1 kHz with a 2-frame delay: H Strehl 0.85, K Strehl 0.91.
- The loop is fitting-limited (fitting error 100 nm, open loop 984 nm), and
  servo-lag adds only ~35 nm at 1 kHz.

**exp03: SH baseline (legacy Keck II).**
- Setup: makewfs 20 x 20 SH on OCAM2K, modal least squares, gain 0.5.
- H Strehl 0.70 / 0.68 / 0.60 / 0.25 at V = 8 / 10 / 12 / 14.
- So the SH adds ~150 nm in quadrature over the ideal sensor (aliasing, undersampled
  centroiding, noise).
- Found a makewfs bug on the way: SH flux was not conserved for wide/undersampled
  subapertures. The fix is in issue #4 / PR #6 (not merged); the HAKA example is affected too.

**exp02: single-frame CNN.**
- In focus, the relative error floors at 0.70 = 1/sqrt(2): the even-mode sign
  ambiguity, exactly as theory predicts. A fixed 1 rad defocus lowers it to 0.4-0.5.
- Neither network can hold a loop, even when handed a closed one.
- Shadow-mode diagnosis: the networks were blind beyond mode ~60. They were trained on
  residuals with the steep open-loop spectrum, so they learned to ignore high-order
  modes, which carry most of the variance in a real closed-loop residual.
- Lesson: train on real closed-loop trajectories (exp04, DAgger).

**exp05/06: information content (Fisher / Cramer-Rao).**
- At 1e4 photons with defocus 0.5-1 rad:
  - The single-frame Bayesian bound is 11.5 nm for a generic 37 nm residual.
  - The unbiased per-frame noise is ~62 nm under the real closed-loop prior. Loop
    filtering brings it to ~21 nm at gain 0.2, so photons are **not** the limiting factor.
- A global linear map stalls at 28 % error even at 1e6 photons, because the
  fitting error changes the sensor response by ~28 % from frame to frame.
- Per-frame optimal (shrinking) estimators ignore the faint high-order modes. The
  loop needs unbiased estimates instead; the integrator does the averaging.

**exp07: linear interaction-matrix reconstructor, calibrated like an SH.**
- Sensitivity is fine: median slope 0.92 on closed-loop residuals.
- It still diverges, for two reasons:
  1. A constant bias of ~67 nm: the on-sky halo from the fitting error, measured
     against a diffraction-limited lab reference.
  2. A ~64 nm noise-free, state-dependent error that remains with the closed-loop
     mean as reference.
- Explanation: the second-order term of e^{i phi} in the uncorrectable fitting error
  (phi_fit^2, mostly low spatial frequency) lands inside the control region. This is
  the focal-plane counterpart of SH aliasing. In nm it should scale as ~1/lambda.
- Testing: a closed-loop reference, H vs K sensing.

**exp07/08: linear focal-plane sensing fails, and why (quantified).**
- The push-pull linear reconstructor diverges in every case: H and K, gains 0.1-0.3,
  hand-over or bootstrap.
- On 400 real closed-loop states (36 nm DM-space residual, 1 rad defocus) its
  noise-free error is:
  - 83 / 69 / 60 / 53 nm in I / J / H / K;
  - 23 / 15 / 11.5 / 8.7 nm once the uncorrectable fitting error is removed.
- So **the fitting-error coupling is the dominant term (5-6x)**, and it is larger than
  the residual in every band. The loop cannot contract, the residual grows past the
  ~1 rad linear range, and the estimate turns anti-correlated with the truth.
- Unlike SH aliasing, which is additive inside a wide linear range, this error feeds
  back on itself. It scales more weakly than 1/lambda.
- Implication: beating it needs a non-linear estimator that sees the fitting halo,
  i.e. a wider field of view.

- Controlling fewer modes does not rescue it. Under each loop's own states, the
  error/residual ratio is:

  | Modes controlled | 30 | 60 | 120 | 200 |
  | --- | --- | --- | --- | --- |
  | H | 1.51 | 1.37 | 1.09 | 1.13 |
  | K | 1.65 | 1.31 | 1.00 | 1.01 |

  The uncontrolled modes add their own coupling. A ratio >= 1 means no contraction.
- **Conclusion: linear focal-plane sensing cannot run the legacy Keck loop.** Any
  focal-plane method must beat the fitting-error coupling non-linearly (or with
  temporal information), not just linearise around a reference.

**exp09/10: a non-linear estimator beats the coupling, but not on every mode.**
- A CNN trained on real closed-loop states (single defocused frame, 64 px, 1e5 photons)
  reaches error/residual 0.62 in H and 0.64 in K on held-out closed-loop states. The
  linear map gets 1.1-1.7. So the fitting-error coupling *can* be largely removed
  non-linearly; the sensing band barely matters.
- Adding known random modal DM offsets (dither) to the training states gives 0.53,
  with the median per-mode slope rising from 0.30 to 0.57.
- Hand-over still collapses ~100 frames after the switch, even after one DAgger round.
- Trace (plots/exp10_dagger/handover_trace_300modes): modes 120-299 double within
  10 frames, because the network has no usable information there (error = residual in
  modes 200-300). The integrator then accumulates temporally correlated errors and
  drags every mode out of distribution.
- Fix being tested: control only the modes the sensor can see (N = 80 / 120), as an
  SH system does. The ideal ceiling with 120 modes is H ~0.74, which is SH parity.

**exp10 N = 120: holds, but not yet robustly.**
- Control limited to the first 120 modes; network trained on dithered closed-loop states.
- In the hand-over test (4 atmospheres, seed 100) it held H Strehl 0.713 for 900 frames
  (rounds 0 and 1). The ideal ceiling for 120 modes is 0.74; the SH is 0.70 on 300 modes.
  The median per-mode slope was 0.75, with none below 0.3.
- **But** a third, fully on-policy DAgger round (beta = 0) made it worse: 2-3 of 4
  atmospheres diverged (mean 0.28).
- The good weights were overwritten (exp10 saved one file per run). The likely cause is
  runaway states in the DAgger data dominating a batch-normalised loss.
- Re-running with: per-round saves, runaway states filtered (> 6x the closed-loop rms),
  per-sample relative loss, and 12 atmospheres in the stability test.
- Not a robust result until that passes.

**exp10 N = 80:** holds after hand-over at H Strehl 0.63 (ideal 80-mode ceiling ~0.66),
flat for 1200 frames, rounds 0 and 1 (old code, 4 atmospheres).

**exp11: staged bootstrap from open loop fails (12/12 atmospheres).**
- Trace: at the very first stage (5 modes), the 1 rad-defocus frame carries no
  information about open-loop focus/astigmatism (error 517 nm vs residual 529 nm), and
  even tip/tilt is poor (348 vs 430 nm). They run away within ~10 frames.
- Physics: in open loop, focus/astigmatism alone is ~1.9 rad at H, larger than the
  1 rad diversity, so the sign ambiguity is effectively unbroken. This matches the
  literature (single-shot capture range ~1.5-2 rad).
- Next (exp12): a large *DM-applied* defocus during acquisition (no new hardware;
  curvature-sensor regime), stepped down to 1 rad as the loop converges.

**SH gain sweep (V = 8 / 10, 300 modes).**

| Gain | 0.3 | 0.4 | 0.5 | 0.6 |
| --- | --- | --- | --- | --- |
| V = 8 | 0.65 | 0.68 | 0.70 | 0.70 |
| V = 10 | 0.65 | 0.67 | 0.68 | 0.69 |

The baseline is tuned; its best is H ~0.70.

**exp10 v2 (N = 120, fixes applied), round 0, 12 atmospheres:** 10/12 held, median
H Strehl 0.705 = SH parity. The other 2 diverged late (~800 frames after hand-over).
That evaluation used leak 1.0; Keck runs leak 0.99. The N = 80 old-code run confirmed
that unfiltered on-policy DAgger data poisons the network (ratio 0.68 -> 0.84).

**exp13: robustness of the N = 120 network (v2 round 0), 12 fresh atmospheres, 2 s after
hand-over.**

| Leak | Gain | Held (of 12) | Median H Strehl |
| --- | --- | --- | --- |
| 1.0 | 0.3 | 7 | 0.61 |
| 0.99 | 0.2 | 0 | — |
| 0.99 | 0.3 | 9 | 0.70 |
| 0.99 | 0.4 | 11 | 0.72 |
| 0.99 | 0.5 | 10 | 0.73 |
| 0.99 | 0.6 | 9 | 0.73 |
| 0.995 | 0.3 | 9 | 0.71 |

- With Keck's leaky integrator and gain 0.4-0.5 the focal-plane loop beats the tuned
  SH (0.70) on the atmospheres where it holds.
- Failures are abrupt cliffs at random times. Lower gain is worse (the residual sits
  higher, outside the trained basin).
- The open problem for both maintenance and bootstrap is the network's narrow capture
  range (~2x the closed-loop residual). Mean time to failure is ~1-2 s, not acceptable.
- Next: train for a wider basin (larger dither, a range of loop gains), and use a larger
  fixed defocus if exp12 shows it widens capture.

**exp10 v2 round 3 (filtered on-policy DAgger) improves robustness.**
- Same 12 fresh atmospheres as the round-0 test: **11/12 held** at every setting
  (leak 1.0 / 0.99, gain 0.3-0.5), median H Strehl 0.70-0.73. Round 0 held 7-11.
- The one failure is the same atmosphere every time (seed 4008), so it is a
  turbulence-dependent weakness, not random.
- Trace: the mid bands (modes 20-120) drift up within ~50 frames of the hand-over
  and run away.
- The training used only ~30 atmospheres, all at 0.6" and nominal wind. Next (v3):
  seeing 0.45-0.85", wind x0.7-1.5, dither up to 3x, warm start from round 3.

**v3 (diverse turbulence, wider dither, warm start + 3 DAgger rounds): 23/24 hold.**
- Leak 0.99, gain 0.4, 24 unseen atmospheres over 2 s: **23/24 hold (96 %), median H
  Strehl 0.72**. The tuned SH gives 0.70.
- The single failure is still seed 4008 (old seed scheme). Its bulk statistics are
  ordinary (mid-ranked in amplitude and rate of change), so a specific transient event
  drives it.
- **The maintenance goal is met in simulation:** the loop is held by the focal-plane
  camera alone, at SH-level or slightly better Strehl, in >= 95 % of atmospheres.

**CORRECTION — exp12 was invalid (test-set leakage).**
- `Turbulence(batch, seed)` used atmosphere seeds seed..seed+batch-1, and exp11/12/14
  offset collections by 1, so train and "held-out" sets shared 14-15 of 16
  atmospheres. exp12's open-loop ratio of 0.17 was memorisation.
- On a genuinely new atmosphere, the stage-A network's open-loop ratio is 0.75. It has
  ~no information on focus/astigmatism (282 nm error on 314 nm), consistent with exp11
  and the literature. **Bootstrap from open loop remains unsolved.**
- exp14 (stage-specific networks) failed in 24/24 atmospheres for the same reason.
- The overlap also reduced the atmosphere diversity of the DAgger collections in exp10.
  It did NOT leak into exp09/10/13 test sets, which used disjoint seeds.
- Fixed: each `seed` now owns a disjoint block of 1000 atmospheres
  (`Turbulence.SEED_STRIDE`). Atmosphere numbers quoted above (e.g. "seed 4008") use
  the old scheme.

**exp15: acquisition on held-out atmospheres (clean seeds).**
- Error/residual for the first 20 modes, by number of modes already controlled
  (stage 0 = open loop):

  | Modes already controlled | 0 | 2 | 5 | 10 | 20 |
  | --- | --- | --- | --- | --- | --- |
  | Single frame (1 rad fixed defocus) | 0.69 | 0.84 | 0.76 | 0.86 | 0.84 |
  | Two frames with known +-1 rad focus probes | 0.66 | 0.83 | 0.76 | 0.89 | - |

  The probe pair adds nothing.
- Even with 20 modes already controlled, the single frame only reaches 0.84 on them,
  because everything above mode 20 is uncorrected (hundreds of nm) and its coupling
  swamps the low-order signal.
- So during acquisition the limit is the **uncorrected high-order turbulence**, not
  the even-mode sign ambiguity: low orders can't be sensed until high orders are partly
  corrected, and vice versa.
- +-2 rad probes are no better (0.76 at open loop, 0.93 with 2 modes controlled), and
  the 4 rad run was stopped. **Temporal phase diversity with focus probes does not
  solve acquisition.**
- exp16 is testing the alternative: close all 120 modes together, slowly, with a network
  trained on the open-to-closed continuum.

**Real-time path (G1): pyRTC PR #156 (`TorchImageReconstructor`, into dev, not merged, CI green).**
- A generic slopes-section component: WFS image -> torch model -> modal signal. CUDA
  graph, pinned buffers, per-frame timing.
- Full-path latency, 64x64 -> 120 outputs, RTX 4060 (shared host, also drives the display):

  | Model | Median | p99 |
  | --- | --- | --- |
  | 14.8M CNN, fp16, CUDA graph | 0.38-0.56 ms | 1.1-2.5 ms |
  | 1.1M MLP, CUDA graph | 0.09-0.18 ms | 0.36-0.51 ms |

- The current maintenance network meets the median budget but **misses the G1 p99 target**.
- Slimmer networks (stride-2 first layer, narrower), fp16, CUDA graph, idle A400
  (the steadier card), network compute only:

  | Network | Median | p99 |
  | --- | --- | --- |
  | 14.8M (current) | 1.06 ms | 1.09 ms |
  | 6.7M | 0.36 ms | 0.38 ms |
  | 2.3M | 0.22 ms | 0.23 ms |

  The slim ones meet G1 even on the A400. A 6.7M network is being trained with the v3
  recipe to check that accuracy and robustness hold.
- Side effects: two config/stream-planning bugs fixed in the PR; issue #155 filed
  (component worker threads leak when __init__ fails).

**exp16: closing all 120 modes together from open loop also fails (0/24, rounds 0-1).**
- Network trained on the open-to-closed continuum, gain ramp 0.1 -> 0.4, leak 0.99.
- Neither mode staging (exp11/14) nor all-at-once closure bootstraps with a near-focus
  (1 rad) frame. Next: clean exp12 (large defocus = curvature regime).

**Slim maintenance network (6.7M parameters, stride-2 first layer, width 24).**
- **24/24 atmospheres held**, median H Strehl 0.715 (gain 0.4, leak 0.99, new-seed
  atmosphere sets, 2 s).
- Network compute 0.36 ms median / 0.38 ms p99 on the idle A400 (fp16, CUDA graph),
  so it meets the G1 latency target on the weaker card.
- Like-for-like, the 14.8M network on the same atmospheres: 24/24, median 0.72.
  Across all four 12-atmosphere test sets it holds 47/48.
- **The slim network matches the full one** (24/24 vs 24/24, 0.715 vs 0.72) at a
  third of the latency, so it is the candidate for the real-time demonstration.

**exp12 (clean seeds): a large defocus widens capture but doesn't solve acquisition.**
- Error/residual for the first 20 modes, by number of modes already controlled:

  | Defocus | 0 | 2 | 5 | 10 | 20 |
  | --- | --- | --- | --- | --- | --- |
  | 1 rad | 0.70 | 0.87 | 0.81 | 0.86 | 0.87 |
  | 3 rad | 0.64 | 0.69 | 0.76 | 0.84 | 0.98 |
  | 6 rad | 0.57 | 0.57 | 0.63 | 0.70 | 0.93 |

- The curvature regime helps capture but costs precision as the loop converges.
- Every stage stays >= ~0.55, the regime where exp11/14/16 closures failed. A
  defocus-stepping acquisition might work but is marginal.

**Real-time pyRTC demonstration (phase 6) and a robustness correction.**
- Setup: hard-RTC pyRTC system, simulated focal-plane camera + Keck DM as research
  components, `TorchImageReconstructor` (PR #156) with the slim 6.7M network, pyRTC
  leaky integrator (gain 0.4, leak 0.99). The loop is warm-started by the ideal sensor
  for 300 frames, then held by the network alone. Code: `rtc/`, see `rtc/README.md`.
- **The loop runs at ~1 kHz (982 Hz median), LE H Strehl 0.723**, identical to the
  offline harness. Where the harness loses an atmosphere, pyRTC loses it on the same
  frame (cross-validation of the harness timing model).
- Latency (median / p99), reconstructor on the idle A400:

  | Path | Median | p99 |
  | --- | --- | --- |
  | Reconstructor compute | 0.50 ms | 0.53 ms |
  | WFS -> signal | 0.64 ms | 0.69 ms |
  | WFS -> wfc | 0.79 ms | 0.93 ms |

  - That fits a 1 ms frame but **misses G1's p99 <= 0.5 ms**. On the shared 4060 the
    median is lower (0.33 ms) but p99 2.2 ms.
  - SCHED_IDLE spinners on the pipeline cores were needed to avoid deep-idle wake-up
    latency.
- **Correction: robustness was measured over 2.3 s windows.** Over 10 s, 8-10 of 12
  atmospheres hold. Failures come at random times 2.3-8.1 s in, a mean time to failure
  of roughly 30-40 s per atmosphere. The "23/24" and "24/24" figures above are
  2-second survival rates, not steady-state robustness.
- Found along the way: pyturb gives different screens for the same seed on the 4060 and
  the A400 (to verify and file). pyRTC's Loop JIT-compiles its numba kernel on the
  first iteration after start (a 0.36 s stall), which the demo works around by priming.

**Open threads.**
- exp04: multi-frame networks with DM-command diversity and DAgger.
- Whether a wider field of view (seeing the fitting halo) lets a nonlinear
  estimator cancel the fitting-error coupling.
- Sensing band (K vs H vs visible).
- Optimising the SH gain per magnitude for a fair comparison.
