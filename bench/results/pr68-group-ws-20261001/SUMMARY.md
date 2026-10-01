# PR #68 grouped GEMM reuses the caller's workspace (lane/neural-pass62), NVIDIA L40S, 2026-10-01

ab_job2 pr68 (main = pre-#65 main at run time, merged into the arm), plus ATTN+STEP timer runs on both trees. Same bits: digest 4a8e781b0739a038, same losses.

| row | main | pass62 |
|---|---|---|
| lm-forward stage (ms) | 49.5 / 50.6 | 48.3 / 50.8 |
| lm-train-step stage (ms) | 41.8 / 41.8 | 40.5 / 40.5 |

Ticks (ms/layer, main / pass62): fwd.q_proj 0.084/0.080, k 0.061/0.056, v 0.060/0.055, o 0.070/0.064, gate 0.117/0.112, up 0.117/0.110, down 0.118/0.113;
bwd.mlp_through_oproj 1.196/1.142, bwd.after_attention 0.623/0.584. Train step -3%. AMD same-bits run: pass10.
