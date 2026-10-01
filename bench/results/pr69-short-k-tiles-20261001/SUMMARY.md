# PR #69 NVIDIA short-k tile rule (lane/neural-pass64) and the pair with #68 (lane/neural-pass62-on-64), L40S, 2026-10-01

ab_job2 pr69 (main 3befccd28 incl. #65, pre-#67, merged into each arm) + ATTN+STEP timer runs. Same bits on every arm: 4a8e781b0739a038, same losses.

| arm | lm-forward stage (ms) | lm-train-step stage (ms) |
|---|---|---|
| main | 61.5 / 59.8 (timers 63.2 / 62.0) | 38.2 / 38.0 |
| #69 | 58.1 / 59.9 (timers 51.6 / 50.7) | 36.5 / 36.5 |
| #69 + #68 | 55.7 / 57.6 (timers 48.5 / 51.3) | 35.8 / 35.9 |

Forward is noisy on this base (the fresh 64 MiB logits output, removed by #67). Train step: #69 -4%, pair -6%.
Ticks ms/layer main / #69 / pair: q_proj 0.090/0.053/0.052, k 0.061/0.049/0.049, v 0.061/0.049/0.049, o 0.068/0.051/0.051,
gate 0.118/0.100/0.100, up 0.119/0.103/0.102, down 0.120/0.149/0.141 (slower), bwd.mlp_through_oproj 1.304/1.194/1.160, bwd.after_attention 0.705/0.596/0.580.

## AMD MI325X (DO pass11, main incl. #65 + #67)

Same bits. lm-forward 23.1 / 23.1 vs 23.4 / 23.3; train 58.3 / 58.5 vs 58.5 / 58.4. Neutral (NVIDIA-only rule).
MERGED after #68. R2 ab-pr69-amd.
