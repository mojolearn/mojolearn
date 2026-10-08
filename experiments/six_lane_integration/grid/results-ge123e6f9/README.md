# IDENTICAL switch grid run ge123e6f9 (2026-10-07/08)

Frozen source: branch `grid/freeze-20261007` (076a1345d, advanced to a050dd85f: tooling only, no kernel difference).
Boxes: nv = RunPod L40S (sm_89), amd = DigitalOcean MI325X (gfx942). One run per arm; noise floor from the incumbent
repeated across the workload's cells (min 5%); identity = equal output hash on NVIDIA and AMD; quality = board metric
within 0.1% relative. Verdicts, identity and quality per cell feed `tools/six_lane_grid_decide.py`.

Files here are snapshots of the decision tables as the run progressed (the measurement record travels with the code;
the raw lq results and race logs live in the orchestrator's evidence directory). `GRID_DECISIONS-*.md` lists every
switch with its recommendation, best arm, per-algorithm verdicts and interactions; losers that were deleted carry
their numbers in `docs/apple-fast/EXPERIMENTS.md` and a tombstone comment at the deletion site.
