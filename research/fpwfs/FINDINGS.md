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

**Open threads.**
- exp04: multi-frame networks with DM-command diversity and DAgger.
- Whether a wider field of view (seeing the fitting halo) lets a nonlinear
  estimator cancel the fitting-error coupling.
- Sensing band (K vs H vs visible).
- Optimising the SH gain per magnitude for a fair comparison.
