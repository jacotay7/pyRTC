# Focal-plane-only AO loop in real time inside pyRTC (PLAN phase 6, goal G1)

The slim maintenance network (exp10 `slim24_nc120_r3`, 6.7M parameters) holds a
Keck focal-plane-only loop inside pyRTC, at a 1 kHz camera rate, with the stock
`Loop` and the `TorchImageReconstructor` of pyRTC PR #156. The camera and DM are
simulated, everything else is pyRTC.

The real-time loop reproduces the offline harness: H-band long-exposure Strehl
0.723 vs 0.723 on the same atmosphere. On two atmospheres where the harness loses
the loop, pyRTC loses it on the same frame, within 5 frames of the harness's.

## What runs

| Section | Class | What it does |
| --- | --- | --- |
| `resources.fpsim` | `fprtc.FPSimContext` | The shared simulation. pyturb `keck` at 0.6", seed block 4000; the atmosphere advances 1 ms of simulated time per exposure. The Keck 349-actuator DM. `FocalPlaneSensor(FPSensorConfig(defocus_rad=1.0, photons=1e5))`: H band, 64 x 64 px, Poisson noise plus 0.6 e- read noise. Per-frame truth from the residual: H-band Strehl (`h_band_science`), rms, and the ideal modal projection. Everything after the atmosphere is one CUDA graph. |
| `wfs` | `fprtc.FPCamera` (`WavefrontSensor`) | Publishes the frame rounded to integer e- on a 100 ADU bias (`wfs_raw`); `wfs` is the frame minus the bias (the dark). Frames are transposed to pyRTC's (width, height) layout. The camera runs free or at a fixed wall-clock rate (`wall_rate`). |
| `slopes` | `pyrtc.image_reconstructor.TorchImageReconstructor` | `model_factory: build_fpnet` (`models.py`). Settings: `flux_normalization: sum`, `sqrt_stretch: true`, `dtype: float16`, `cuda_graph: true`, `output_scale_file` = `_scale` x 1e-9. The signal is the modal residual in metres. |
| `loop` | `pyrtc.loop.Loop`, `leaky_integrator` | Identity IM and CM over 120 modes; gain 0.4, `leaky_gain` 0.01 (leak 0.99). |
| `wfc` | `fprtc.FPDM` (`WavefrontCorrector`) | M2C = the first 120 columns of the 300-mode Keck DM-KL basis. Actuator commands go to the simulated DM, which *adds* its OPD (residual = turbulence + DM). A command `c` on a flat wavefront therefore reads back as `+c`, pyRTC's identity-IM sign convention. |

**Units and sign.** Both are checked end to end (`check_parity.py`, `results/rtc/parity.json`):

- The network outputs units of `_scale`. `output_scale_file` holds `_scale * 1e-9`, so the signal is in metres of unit-rms modal coefficient: the corrector's units, and those of the harness's `DM`.
- The loop does `c <- 0.99 c - 0.4 s`, and the DM adds `+OPD(c)`. That is negative feedback, and the same law as the harness's `cmd <- leak cmd + gain est` with `residual = turb - dm(cmd)` (`c = -cmd`).
- Sign check on 100 real closed-loop frames: the estimate's regression slope against the true projection is +0.91 globally and +0.78 per-mode median.
- Static pokes on a flat wavefront read back positive, at 0.3-0.35 of the poke. A flat wavefront with no fitting halo is far outside the training distribution, so this checks only the sign.
- Open vs closed loop on the same start (figure): with the loop never started, the DM stays frozen at the hand-over shape and the Strehl falls from 0.74 to 0.03 within ~50 frames (LE 0.008). With the loop running it holds at 0.72.

**Preprocessing parity.** The networks were trained on `FocalPlaneSensor.preprocess`
= `sqrt(clamp(e, 0) / flux * 64^2)` on (H, W) frames. The reconstructor computes
`sqrt(clamp(e / sum(e), 0))` on the stream array, which is (W, H).
`models.FPInputAdapter` transposes back and multiplies by 64.

Measured on 100 closed-loop frames through `TorchModelRunner`, the reconstructor's
real-time path:

- The network input equals `FocalPlaneSensor.preprocess` of the same frame exactly: max abs difference 0.0.
- fp32 output vs the harness's `net(preprocess(e)) * _scale`: 0.08 % relative.
- fp16 with CUDA graph vs fp32: 0.3 %.
- The only real difference is the camera's integer ADUs. Rounding read-noise-level pixels changes the input by 3.4 % relative and the output by 2.7 %. The estimation error does not move: error/truth is 0.405 for pyRTC fp16 and 0.405 for the harness.

## Hand-over (warm start)

The network only holds a loop that is already closed, as in exp10/13. The
context's warm start reproduces exp13's hand-over:

1. **Before the run.** `run_demo.py` starts every component. The camera holds (publishes nothing) and the `Loop` is paused right after start. It is then primed: one integrator step on a zero signal, so its numba kernel is compiled before the hand-over.
2. **Exposures 1-300.** The *context* is the controller. From each exposure's true residual it takes the least-squares projection on the 120 controlled modes (`ModalProjector`). It integrates `c <- 0.99 c - 0.4 est` (same gain, leak and units as the loop) and writes `c` to the `wfc` stream, stamped with that frame's id, at the moment the frame is published. The warm loop therefore has the same 2-frame delay and goes through the real `wfc -> FPDM -> M2C -> DM` path. The reconstructor already runs on every frame (shadow mode).
3. **Exposure 300.** The context stops writing `wfc` and calls the hand-over hook, which starts the pyRTC `Loop`. The loop's first update comes from the network's signal and is added to the last ideal command already in `wfc`. From then on, only the network and the pyRTC loop control the DM.

The hand-over needed one fix. Unprimed, the loop's first iteration after `start()`
took 360 ms (numba loading its cached kernel). That left 55 frames with a frozen
DM, after which the network had lost the loop. Priming fixed it.

## Timing model

At each tick the camera publishes frame e-1, then samples the DM state for
exposure e and renders it. That is a readout at the end of the exposure. A command
computed from frame k shapes exposure k+2 if it lands within one frame period of
frame k's publication; this is the harness's `delay = 2`. Otherwise it shapes
exposure k+3.

The context logs, per exposure, which frame's command was on the DM. The
realised delay is therefore measured ("delay-2 frac" in the table), not assumed.

The atmosphere is either:

- `precomputed`: the same pyturb atmosphere, stepped ahead of time into a GPU buffer, so the run sees exactly the harness's screens;
- `live`: pyturb stepped inside every exposure.

## Results

Atmosphere: seed block 4000, atmosphere 3, on the RTX 4060's screens. The offline
harness holds this atmosphere for 10 s at both delays.

Run settings:

- 10 000 exposures = 10 s of simulated time (`hard_live_free`: 6000).
- Reconstructor: fp16 with CUDA graph.
- Pipeline threads pinned to cores 20-23.

GPUs: torch `cuda:0` is the RTX 4060, which also drives the display and was
shared with another user's job at 25-99 % utilisation throughout. `cuda:1` is the
RTX A400, otherwise idle.

Run variants:

- **hard**: `slopes` and `loop` run as hard-RTC child processes (`manager.component_modes`). `wfs` and `wfc` stay in the manager process with the simulation.
- **soft**: everything in one process.
- **idle-spinners**: a `SCHED_IDLE` busy loop on each pipeline core (`--spinners`). The cores then never enter this host's ~3 ms-exit deep idle states, and the spinners yield instantly to real work. No root needed.

Latency columns:

- `wfs->signal` and `wfs->wfc` are `manager.latency` measurements: stream write to stream write, frame-id matched, 2000 samples taken during the network phase.
- "recon compute" is `TorchImageReconstructor.timing_stats()`: from the end of the `wfs` read to the output on the host, last 1000 frames.
- "camera sim" is the simulator's own time per exposure: atmosphere, CUDA graph and D2H.

All times are in ms.

| Run | Mode | Sim / recon GPU | Target Hz | Achieved Hz (median / mean) | Camera sim (med / p99) | Recon compute (med / p99) | wfs->signal (med / p99) | wfs->wfc (med / p99) | Delay-2 frac | LE H Strehl |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **hard_1khz** | hard, spinners | 4060 / A400 | 1000 | 982 / 841 | 0.54 / 2.56 | **0.497 / 0.533** | 0.644 / 0.689 | **0.794 / 0.928** | 0.90 | **0.723** |
| hard_500hz | hard, spinners | 4060 / A400 | 500 | 500 / 499 | 0.55 / 2.61 | 0.495 / 0.518 | 0.650 / 0.698 | 0.802 / 0.876 | 1.00 | 0.723 |
| hard_1khz_nospin | hard | 4060 / A400 | 1000 | 976 / 830 | 0.63 / 2.61 | 0.540 / 0.748 | 0.880 / 1.251 | 1.280 / 1.718 | 0.14 | 0.716 |
| soft_1khz | soft, spinners | 4060 / A400 | 1000 | 645 / 598 | 0.87 / 2.80 | 0.544 / 0.856 | 1.028 / 1.566 | 1.521 / 2.310 | 0.64 | 0.722 |
| hard_1khz_recon4060 | hard, spinners | A400 / 4060 | 1000 | 988 / 948 | 0.71 / 0.86 | 0.327 / 2.213 | 0.479 / 2.329 | 0.640 / 2.479 | 0.78 | 0.719 |
| hard_live_free | hard, spinners, live pyturb | 4060 / A400 | free | 518 / 477 | 1.61 / 3.92 | 0.492 / 0.526 | 0.644 / 0.687 | 0.799 / 0.945 | 1.00 | 0.725 |
| frozen (loop never started) | hard, spinners | 4060 / A400 | 1000 | 996 / 854 | 0.54 / 2.59 | 0.494 / 0.521 | - | - | - | 0.008 |

**Offline harness, same atmosphere** (`offline_ref.py`, `fpsim.loop.run_loop`, same protocol): LE 0.723 at delay 2 and 0.721 at delay 3.

**Two atmospheres the harness loses.** Both pyRTC runs used an earlier timing in which the DM was sampled at the tick rather than after publication, so delay 3 dominated.

| Atmosphere | pyRTC run | GPUs | pyRTC loses the loop at frame | Harness loses it at frame |
| --- | --- | --- | --- | --- |
| atm 0, RTX 4060 screens | `atm0_hard_1khz` | 4060 / A400 | 2309 | 2304 (delay 2), 2308 (delay 3) |
| atm 0, A400 screens | `atm0_a400_hard_1khz` | A400 / A400 | 7764 | 7766 |

Losing the loop means the 100-frame mean SE Strehl falls below 0.3. The
`atm0_a400_hard_1khz` run had everything on the A400: the camera reached only
860 frames/s, and the reconstructor slowed to 0.84 / 1.04 ms because it shared the
GPU with the simulator.

### Latency budget at 1 kHz (hard_1khz, idle A400, spinners)

- **Camera publish -> DM updated:** median 0.86 ms, p99 1.02 ms, over all 9613 network-phase frames.
- **The reconstructor's step (`wfs -> signal`, 0.64 ms):**
  - ~0.50 ms is reconstructor compute: H2D, preprocessing, the 6.7M CNN in fp16 and D2H. The network alone is ~0.36 ms on the A400 (FINDINGS).
  - ~0.15 ms is the wake-up and read of the `wfs` frame.
- **Loop and handoff (`signal -> wfc`):** 0.15 ms.
- **Delay:** 90 % of exposures saw a 2-frame delay; the rest 3. At 500 Hz it was 100 % delay 2.

**G1 verdict.** The pyRTC loop runs at 1 kHz with a 2-frame delay in 90 % of
frames, and the median WFS -> DM latency (0.79 ms) fits in a frame. The G1 compute
budget, p99 <= 0.5 ms from WFS stream to DM stream, is **not met on the A400**:
the stream-to-stream p99 is 0.93 ms, and the reconstructor alone has p99 0.53 ms.

The idle RTX 4060 would bring compute to ~0.33 ms median, but on this shared,
display-driving card its p99 is 2.2 ms, worse than the A400's. A dedicated modern
RTC GPU, or TensorRT, is needed for the 0.5 ms p99 target.

**What matters on this host:**

1. **Process isolation.** In soft-RTC mode, the reconstructor, loop and Python camera simulator share one GIL. Latency doubles (1.52 / 2.31 ms), and the camera cannot reach 1 kHz (645 Hz).
2. **CPU idle states.** Without idle-spinners, the pipeline threads' wake-ups add ~0.5 ms (1.28 / 1.72 ms) and push most frames to a 3-frame delay.
3. **GPU contention.** A reconstructor sharing a GPU with another job loses its tail (p99 0.53 -> 2.2 ms).

**Camera simulator.** The camera runs on the RTX 4060 and is accounted separately
from the RTC:

- Precomputed atmosphere: 0.3-0.55 ms median per exposure. The p99 of 2.6 ms comes from the other user's job on the card. 1 kHz is reached in median (982 Hz); the mean is 841 Hz because of those stalls.
- Live pyturb stepping inside each exposure: 1.6 ms median, so the camera free-runs at ~500 Hz.
- The context times every exposure, so the RTC numbers are unaffected by the simulator's rate. Simulated time always advances 1 ms per exposure.

### Strehl and robustness

- **pyRTC vs harness on atm 3.** H-band long-exposure Strehl after frame 600 is 0.723 in pyRTC at 1 kHz, against 0.723 (delay 2) and 0.721 (delay 3) in the harness. The exp13 value for this network is ~0.715.
- **Holding.** The loop holds for 10 s (9700 frames after the hand-over) in every atm 3 run: the lowest 100-frame mean SE Strehl is 0.67, except 0.61 in the no-spinner run. With the extra delay and latency (soft, no spinners, recon on the 4060), the LE Strehl drops by at most 0.007. The no-spinner run, mostly at a 3-frame delay, sagged to 0.64 over its last 500 frames, against 0.70 in the others.
- **New: 10 s survival (offline, `offline_survival.py`, 12 atmospheres, 10 000 frames).**
  - The exp13 robustness test ran 2.3 s. Over 10 s the network holds 8/12 (delay 2) and 9/12 (delay 3) on the 4060's screens, and 10/12 (delay 2) on the A400's.
  - Failures come at random times: 2.3-8.1 s after the start.
  - The mean time to failure is therefore of order 30-40 s per atmosphere. The 24/24 at 2 s in FINDINGS does not mean a robust loop.
- **pyturb screens depend on the GPU.** The same pyturb seed gives different screens on the RTX 4060 and the A400 (first-screen rms 991 vs 898 nm). Offline references must run on the GPU the camera simulator used.

## How to run

```bash
# from the repository root (the worktree's pyrtc must be the one imported)
export PYTHONPATH=$PWD
PY=~/miniforge3/envs/pyrtc313/bin/python
$PY research/fpwfs/rtc/check_parity.py --device cuda:1            # preprocessing / units / sign
taskset -c 20-31 $PY research/fpwfs/rtc/run_demo.py --tag hard_1khz --mode hard --atm-index 3 \
    --sim-device cuda:0 --recon-device cuda:1 --wall-rate 1000 --frames 10000 --spinners
$PY research/fpwfs/rtc/offline_ref.py --device cuda:0 --atm-index 3 --out research/fpwfs/results/rtc/offline_ref_atm3.npz
$PY research/fpwfs/rtc/offline_survival.py --device cuda:0 --tag survival_seed4000_4060_d2
$PY research/fpwfs/rtc/collect_results.py                          # results/rtc/results.json + table
$PY research/fpwfs/rtc/plot_rtc.py --main hard_1khz --offline offline_ref_atm3 --frozen frozen \
    --latency-runs soft_1khz,hard_1khz_nospin,hard_1khz,hard_1khz_recon4060
```

`run_demo.py` writes the effective config, with absolute paths and private stream
names `fprtc_<tag>_*`, to `results/rtc/<tag>/config.yaml`. It also writes:

- `summary.json`: the latency report, timing stats, rates, delay histogram and Strehl;
- `frames.npz`: per-exposure tick, publish, DM-update and render times, true Strehl and rms, the DM-state frame id, and the true 120-mode residual.

Options:

- `--mode soft|hard`;
- `--wall-rate` (0 = free-running);
- `--atmosphere live|precomputed`;
- `--no-loop` (open-loop control);
- `--spinners`;
- `--strehl-every N`: decimates the science Strehl, which costs only ~0.05 ms of the 0.6 ms camera graph on the A400.

Files:

- `fprtc.py`: the simulation context and the camera and DM components.
- `models.py`: the network factory and input adapter, and `export_scale`.
- `system.yaml`: the config template.
- `run_demo.py`: the run, latency and logging.
- `check_parity.py`: preprocessing, units and sign checks.
- `offline_ref.py`, `offline_survival.py`: the harness references.
- `collect_results.py`, `show.py`: tables.
- `plot_rtc.py`: the figures in `plots/rtc_demo/`: `strehl_vs_time`, `latency_hist`.

Weights: `results/exp10/slim24_nc120_r3.pt` (git-ignored; copy it in from the
main checkout).
