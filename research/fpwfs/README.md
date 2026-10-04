# Focal-plane wavefront sensing research (branch `fpwfs`)

Can an 8-m AO system close its loop at >= 1 kHz using only a focal-plane
camera as the wavefront sensor, and match a Shack-Hartmann?

- [PLAN.md](PLAN.md): goals, success criteria, approach, phases
- [LITERATURE.md](LITERATURE.md): literature review
- [ENVIRONMENT.md](ENVIRONMENT.md): tools, install notes, measured GPU headroom
- `tools/`: `env_check.py` (environment + GPU latency check), `budget.py`
  (first-order band/DM numbers)

This directory is never merged into `dev`/`main`. General-purpose pyRTC
changes made along the way are split into their own PRs against `dev`.
