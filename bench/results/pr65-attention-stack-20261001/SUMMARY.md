
## AMD MI325X (DO), ab_job stack65 (main merged into each arm)

| row | main | pass58 (#65) | pass50 |
|---|---|---|---|
| lm-forward stage (ms) | 41.1 / 40.2 | 41.5 / 39.6 | 40.9 / 40.1 |
| lm-train-step stage (ms) | 75.0 / 76.0 | 58.4 / 58.6 | 63.9 / 64.9 |
| digest / losses | 4a8e781b0739a038, 9.018733024597168 8.418445587158203 | same | same |

R2: measurements/2026-10-01/ab-stack65-amd.tar.gz.

## M3 Ultra, 16:00Z run: NOT USED

Same digest and losses on both arms. Main read ~3x slower than this morning's probe on the same box
(lm-forward stage 271-297 ms vs 91, train step 799-912 vs 275), no swap or thermal cause found. Race lm-forward
main 331 vs pass58 64 ms, train step 188 vs 192 ms. Rerun queued (~/m3_queue3.sh, after pr67); the old run is ~/pr65-metal-run1.
