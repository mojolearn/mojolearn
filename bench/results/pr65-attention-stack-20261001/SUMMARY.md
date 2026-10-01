# PR #65 attention stack (#58+#59+#60+#61 on main, lane/neural-pass58), 2026-10-01

## NVIDIA L40S (nvc1)

lm-train-step 41.3 -> 37.0 ms, same digest 4a8e781b0739a038 and losses. Raw: nvidia-raw.txt; R2 ab-stack65-nvidia.

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

## M3 Ultra rerun (~/pr65-metal, box back to normal speed): SLOWER, NOT MERGED

Same digest and losses. lm-forward stage 70.9/70.4 main vs 70.5/71.3; race 63.1 vs 63.7.
lm-train-step stage 198.9/201.5 vs 211.4/207.6 (+4.6%); race 184.5 vs 191.4 (+3.7%).
Timers (per layer): attn.bwd_kvgrid_dkdv_pf 2.75 -> 3.73 ms (+0.98), bwd.attention 8.43 -> 9.32; everything else flat
(m3-tick-diff.txt). The stack's dK/dV change regresses Metal; #65 waits on a Metal gate for it.
