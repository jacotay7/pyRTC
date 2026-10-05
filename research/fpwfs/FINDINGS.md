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

**exp17: defocus-ramp bootstrap (the user's chosen no-hardware direction) fails, 0/24.**
- Schedule: 6 -> 4.5 -> 3 -> 2 -> 1.5 rad DM-applied defocus with 20 -> 120 controlled
  modes, then the slim maintenance network. Stage networks, 2 DAgger rounds.
- Held-out stage ratios after DAgger: 0.77, 0.97, 0.85, 0.73, 0.69.
- The acquisition stage lifts median SE Strehl only to 0.09 (open loop 0.04-0.05).
  The 40-mode stage has no information (0.97) and collapses every time.
- **Why 6 rad was never the curvature regime:** at D/r0 ~ 15 (H, 0.6") the seeing blur
  is ~15 lambda/D, and a 6 rad rms defocus makes a pupil image only ~26 lambda/D across,
  i.e. < 2 resolution elements.
  - A real curvature-sensing regime resolving ~10 elements needs a pupil image of
    ~150 lambda/D, i.e. ~30 rad rms defocus (~8 um rms at H).
  - That exceeds the Keck DM stroke (4 um PV), so it would need a WFS-camera focus
    stage, or a longer sensing wavelength (smaller D/r0).
  - Modelling it needs binned-pixel rendering (render at Nyquist, sum pixels) and a
    finer pupil grid (30 rad of focus aliases at 120 samples). The imager now refuses
    sub-Nyquist sampling, which it silently mis-modelled (~10 % flux).

**exp19: 10 s robustness and RTC-side safeguards (slim network, gain 0.4, leak 0.99).**

| Safeguard | Survive 10 s (of 12) |
| --- | --- |
| None | 9 |
| Clip the update at 2.5x the typical norm | 8 |
| Hold estimates above 4x typical | 9 (never triggered) |

- Survivors run at H Strehl 0.716. The failures are not outlier estimates.
- Re-running the earliest-failing atmosphere with different photon-noise draws does not
  fail through frame 800 (exp19 lost it at ~771).
- So failures are **stochastic excursions**: a noise-driven random walk occasionally
  crosses the edge of the network's narrow basin (~2x the closed-loop residual).
  Typical estimate statistics give no warning, so controller-side guards cannot catch
  them.
- **Robustness and bootstrap are the same problem: capture range.** A working
  focus-stage acquisition would double as automatic re-acquisition after a loss.

**Focus-stage acquisition (user-approved; a WFS focus stage counts as standard equipment).**
- New sensor modes:
  - binned detector: render at Nyquist, sum 4x4 pixels into 88 px;
  - 240-sample pupil, since 30 rad of focus aliases at 120 samples.
  The imager now refuses sub-Nyquist point sampling, which had silently lost ~90 % of
  the flux.
- exp12, 60-mode error/residual on held-out atmospheres, by modes already controlled:

  | Defocus | 0 | 2 | 10 | 30 | 60 |
  | --- | --- | --- | --- | --- | --- |
  | 15 rad | 0.70 | 0.49 | 0.55 | 0.54 | 0.74 |
  | 25 rad | 0.64 | 0.43 | 0.43 | 0.44 | 0.73 |
  | 1 rad (20 modes only, for comparison) | 0.70 | 0.87 | 0.86 | - | 0.87 (20 ctrl) |

- **Once tip/tilt is controlled, a large defocus gives one frame real information about
  60 modes**, the first estimator well below 1 through every acquisition stage. It loses
  precision near convergence (0.73), so the defocus is stepped down there.
- Tip/tilt itself comes from the image centroid: classical, poke-calibrated, exact for
  any defocus.
- exp20 (running): centroid TT -> 25 rad network on 60 modes -> 3 rad network on 120
  modes -> 1 rad maintenance network.

**exp20 v1: first real acquisition (focus stage).**
- Schedule: centroid TT at 25 rad -> 60-mode network at 25 rad -> 120-mode network at
  3 rad -> maintenance network at 1 rad. Slim networks, 2 DAgger rounds, 24 unseen
  atmospheres.
- After DAgger, **the 25 rad stage alone takes the loop from seeing-limited (median SE
  H Strehl 0.05) to 0.36 in 0.3 s**; round 0 reached 0.11, round 1 0.26. The science path
  is unaffected by the WFS focus offset.
- The hand-over to the 3 rad / 120-mode stage collapses (0/24 converge).
- Stage-2 information test (states with 60 modes controlled, 120-mode estimate):

  | Defocus | 1 rad | 3 rad | 25 rad |
  | --- | --- | --- | --- |
  | Error/residual | 0.85 | 0.73 | 0.67 |

  Large defocus remains the most informative there, so the step straight to near
  focus is what fails.
- exp20 v2 (running): gentler descent, 25 rad (60) -> 25 rad (120) -> 8 rad (120) ->
  3 rad (120) -> 1 rad maintenance, 3 DAgger rounds. A full-width-network variant of v1
  is also running.
- 35 rad gives no gain over 25 rad (0.69 / 0.48 / 0.42 / 0.42 / 0.74).

**exp20 v4: first end-to-end bootstrap from seeing-limited conditions (focus stage).**
- Schedule:
  1. centroid tip/tilt at 25 rad (100 frames);
  2. 25 rad network, 60 modes (300);
  3. 25 rad network, 120 modes (400);
  4. focus stage back to 1 rad, slim maintenance network (600).
  Slim stage networks, 3 DAgger rounds, 24 unseen atmospheres, gain 0.3, leak 0.99.
- Converged atmospheres by round: 0/24 -> **14/24 -> 18/24 -> 18/24 (75 %)**.
- Stage medians: 0.05 -> 0.39 -> 0.56 -> **0.70 H Strehl**, i.e. the maintenance level.
- Failure anatomy (plots/exp20_focus_stage/fs25_v4.png):
  - acquisition works in all 24;
  - 5 of the 6 failures happen *during* the 25 rad / 120-mode stage at random times
    (500-700 frames), the same stochastic-excursion failure as in maintenance;
  - 1 happens at the hand-over;
  - every atmosphere that reaches the maintenance network holds.
- Both 25 rad stages plateau within ~30 frames, so shortening them (less exposure to
  excursions) is the next test, using the saved networks on 48 fresh atmospheres.
- The coarse-binned 8 rad intermediate stage of v2 was the wrong sensor: 4x4-binned
  pixels on a small pupil image. Stepping 25 rad -> 1 rad directly works better.

**Faint stars (G4, first look): slim maintenance network, not retrained.**
- Trained at 1e5 photons with 0.5-2x augmentation. Gain 0.4, leak 0.99, 12 atmospheres,
  2.3 s:

  | Photons / frame | 1e5 | 3e4 | 1e4 | 3e3 |
  | --- | --- | --- | --- | --- |
  | Held (of 12) | 12 | 11 | 11 | 0 |
  | Median H Strehl | 0.718 | 0.714 | 0.699 | - |

- It degrades gracefully down to 1e4 (~V 10.5 for a G star in H, same throughput
  assumptions as before; the SH gives 0.68 at V = 10). It is lost at 3e3 (~V 12;
  the SH gives 0.60 at V = 12).
- 3e3 is 33x below the training range; per-photon-level training is needed before
  drawing a limiting-magnitude conclusion.

**exp20 short-schedule evaluation (v4 networks, 48 atmospheres):** 37/48 (77 %).
Shortening the 25 rad stages moved the failures to just after the hand-over (frames
250-500): the hand-over state (Strehl ~0.55) sits at the edge of the maintenance
network's basin. v5 (running) fine-tunes the maintenance network on post-hand-over
states with DAgger.

**Faint stars, retrained (3e3 photons/frame, about V 12 in H).**
- A slim network retrained at 3e3 photons is information-starved:
  - 120 modes: error/residual 0.88, median per-mode slope 0.17;
  - 60 modes: 0.97 / 0.27.
  Hand-over holds 0/12 in both cases.
- At 1 kHz the single-frame focal-plane sensor runs out of photons near V ~12 in H. The
  SH still gives 0.60 there, with its broadband visible photons.
- Fair faint-star comparison (todo): lower the frame rate as Keck's camera table does
  (400 Hz at R = 12). ~300 Hz gives ~1e4 photons/frame, where the network works.

**exp20 v5-v7: fixing the hand-over (48 unseen atmospheres each).**
- Baseline: v4 networks with shortened 25 rad stages, untouched slim maintenance
  network: 37/48.
- v5: fine-tune everything on schedule data (warm start from v4): 26-28/48. Fine-tuning
  on noisy-ideal data undid the acquisition networks' DAgger.
- v6: freeze acquisition, fine-tune the maintenance network on schedule data only:
  5/48 at round 0. **The maintenance network forgets steady-state closed loop.**
- v7: freeze acquisition, fine-tune maintenance on schedule data **plus a replay buffer**
  of 6 exp10-style dithered closed-loop collections:

  | Round | 0 | 1 | 2 | 3 |
  | --- | --- | --- | --- | --- |
  | Converged (of 48) | 24 | 38 | **41 (85 %)** | 40 |

  Median final H Strehl 0.70.
- Remaining v7 failures are mostly inside the frozen 25 rad / 120-mode stage (the
  weakest estimator, ratio 0.84), plus ~1-2 at hand-over and 1 late excursion.

**333 Hz faint-star test:** a slim network trained at 333 Hz with 1e4 photons/frame
(the same V ~12 star) has normal estimation statistics (ratio 0.54, median slope 0.61)
but loses the hand-over within ~100 frames, even after DAgger. Cause unknown; to diagnose.

**Open threads.**
- exp04: multi-frame networks with DM-command diversity and DAgger.
- Whether a wider field of view (seeing the fitting halo) lets a nonlinear
  estimator cancel the fitting-error coupling.
- Sensing band (K vs H vs visible).
- Optimising the SH gain per magnitude for a fair comparison.
