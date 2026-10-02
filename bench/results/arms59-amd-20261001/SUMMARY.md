# AMD MI325X (DO) arms59: #66 page fix (pass59) +/- ATTN_FWD_TQ16, #64 (pass57@ATTN_DKDV_BJ16), 2026-10-01

ab_job, main merged into each arm. All arms: lm-forward digest 4a8e781b0739a038, losses 9.018733024597168 8.418445587158203.

| arm | lm-forward (ms) | lm-train-step (ms) |
|---|---|---|
| main | 41.6 / 41.0 | 75.0 / 75.1 |
| pass59 (#66) | 45.3 / 42.2 | 74.9 / 74.0 |
| pass59 + FWD_TQ16 | 41.5 / 37.7 | 71.5 / 71.5 |
| pass57 + DKDV_BJ16 (#64) | 44.2 / 43.4 | 71.9 / 71.9 |

#64: train step -4% on AMD but slower on NVIDIA (earlier), forward reads +6% here; not mergeable as an all-vendor default.
#66/TQ16 are inside #65 (AMD train 58.5 ms). R2: measurements/2026-10-01/ab-arms59-amd.tar.gz.
