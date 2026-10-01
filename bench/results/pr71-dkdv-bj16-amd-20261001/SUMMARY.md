# PR #71: dK/dV BJ16 as the AMD column default (lane/neural-pass65 23137af9b)

Box: DigitalOcean MI325X (gfx942), wheel 0.8.33 base, `ab_job amd pr71 lane/neural-pass65`, main vs lane, plain x2 then ATTN+STEP timers.

| arm | lm-forward ms | lm-train-step ms |
|---|---|---|
| main run 1 | 23.400 | 58.589 |
| main run 2 | 23.330 | 58.335 |
| pass65 run 1 | 23.366 | 57.217 |
| pass65 run 2 | 23.343 | 57.414 |

- train step -2.0%; forward neutral. dK/dV tick (timers build) 0.601 -> 0.438 ms (-27%).
- Bits: lm-forward digest 4a8e781b0739a038, losses 9.018733024597168, 8.418445587158203 on every arm.
- Gated on `TARGET_COLUMN == COLUMN_AMD`; NVIDIA and Apple compile the unchanged BJ32 path.

Bulk: R2 `measurements/2026-10-01/ab-pr71-amd.tar.gz` (see r2-index.tsv).
