# Environment

Research branch `fpwfs` (off `dev`). Everything under `research/` stays on this
branch; only general fixes/features are split out to PRs against `dev`.

## Python environment

All work runs in the conda env `pyrtc313` (Python 3.13). Verify with:

```bash
python research/fpwfs/tools/env_check.py --device cuda:0   # imports + GPU latency/throughput
python research/fpwfs/tools/budget.py                       # first-order band / DM numbers
```

| Tool | Version | Install | Role in this project |
| --- | --- | --- | --- |
| torch | 2.11 cu128 | pip | reconstructor networks, batched differentiable optics, CUDA graphs |
| cupy | 14.2 cuda12x | pip | GPU backend of pyturb / getframes / makewfs / SPECULA |
| pyturb | 1.2.0 | editable `~/aosim/pyturb` | GPU atmosphere, site profiles (paranal-*, mauna-kea, keck, ...) |
| makewfs | 1.1.0 | editable `~/aosim/makewfs` | GPU Shack-Hartmann baseline (and pupil generation) |
| getframes | 2.2.0 | editable `~/aosim/getframes` | detector noise: C-RED One / SAPHIRA, OCAM2K, sCMOS presets |
| aobasis | 2.0.0 | editable `~/aosim/aobasis` | KL / DM-fitted KL / Zernike modal bases |
| pyRTC (pyrtcao) | 1.1.0+dev | editable (this repo) | real-time loop, streams, latency measurement |
| HCIPy | 0.7.1 | pip | reference optics (cross-check PSFs, SH, vAPP) |
| OOPAO | git f40b21e | clone `~/aosim/OOPAO` via `oopao_clone.pth` | independent SH/pyramid closed-loop reference |
| SPECULA | 1.0.4 | pip | independent GPU end-to-end reference |

Notes

- OOPAO must be a clone on `sys.path` (its `__init__` searches `sys.path` for
  "OOPAO"; a pip-from-git install also drops subpackages). The env has a
  `oopao_clone.pth` in site-packages pointing at the clone.
- GPUs: torch/CuPy `cuda:0` = RTX 4060 (8 GB, also drives the display),
  `cuda:1` = RTX A400 (4 GB). Use the 4060.
- The host is shared: pin benchmarks with `taskset` to idle cores, cap BLAS
  threads for CPU simulators (see pyRTC AGENTS.md gotchas).

## Measured headroom (2026-10-03, `env_check.py`, cores 60-67)

Batch-1 reconstructor latency, median / p99 in microseconds:

| Model | RTX 4060 | RTX A400 |
| --- | --- | --- |
| CNN 6 M params, 4x128^2 -> 800 modes, fp16, CUDA graph | 201 / 558 | 517 / 530 |
| same, eager PyTorch | 1627 / 1894 | 1671 / 1971 |
| CNN 4x96^2 -> 400 modes, fp16, CUDA graph | 192 / 549 | 454 / 479 |
| dense linear pixels -> 800 modes (fp16 matvec) | 444 | 1160 |

- Eager PyTorch is launch-bound at ~1.5 ms: CUDA graphs (or TensorRT) are
  mandatory for a 1 kHz loop.
- A dense pixel-space linear reconstructor is memory-bandwidth-bound and is
  *slower* than a small CNN on these cards.
- 4060 p99 (~0.55 ms) is display/host jitter; the A400 is steadier but 2.5x
  slower. Both leave a 1 kHz budget; a data-centre GPU would cut this ~5-10x.

Training-data generation (128 px pupil, Nyquist, 4 frames per sample, FFT):
~8.5k samples/s on the 4060, i.e. 1 M samples in ~2 minutes.

## State of sibling repos (2026-10-03)

- `~/aosim/makewfs` is checked out on `fix/sh-lenslet-field-aliasing` (PR #6,
  not merged), which fixes SH flux non-conservation for wide or undersampled
  subapertures (issue #4). The editable install uses that branch. The research SH
  model also sets 16 pupil samples per lenslet, so its results don't depend on the fix.
- The OOPAO clone is newer than what makewfs's OOPAO validation test expects
  (`wfs_measure` signature changed); that one makewfs test fails on `main` too.
- Two GPUs: run training on the 4060 and independent sweeps on the A400 with
  `CUDA_VISIBLE_DEVICES=1` (4 GB: keep data stores on CPU).
