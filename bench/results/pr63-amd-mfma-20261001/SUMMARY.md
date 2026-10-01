# PR #63 AMD MFMA forward default (lane/neural-pass56) + backward MFMA trials, MI325X (DO), 2026-10-01

ab_job p63, main merged into each arm, stage x2 + timers build. All arms: lm-forward digest 4a8e781b0739a038, losses 9.018733024597168 8.418445587158203.

| arm | lm-forward (ms) | lm-train-step (ms) |
|---|---|---|
| main | 42.0 / 40.9 | 75.0 / 74.9 |
| pass56 (#63) | 42.4 / 40.4 | 73.3 / 71.8 |
| pass56 + ATTN_DQ_MFMA | 40.0 / 39.6 | 65.0 / 65.3 |
| pass56 + ATTN_DKDV_MFMA | 39.8 / 40.1 | 70.6 / 70.4 |
| pass53-on-56 | 39.0 / 39.9 | 64.2 / 66.3 |

R2: measurements/2026-10-01/ab-p63-amd.tar.gz.
