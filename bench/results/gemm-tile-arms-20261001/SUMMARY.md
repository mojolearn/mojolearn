# GEMM tile env arms on the #67 head (lane/neural-pass61 0f46b2a63, main merged), NVIDIA L40S, 2026-10-01

Runtime knobs only (no rebuild): default (NVIDIA MIN_K 4096) vs MOJOLEARN_GEMM_TILE_MIN_K=0 with
MOJOLEARN_GEMM_TILE_MIN_BLOCKS=128 or 192. Plain builds x2 interleaved, then byte_lm with ATTN+STEP phase timers.
All arms: lm-forward digest 4a8e781b0739a038, losses 9.018733024597168 8.418445587158203. R2: measurements/2026-10-01/gemm61-nvidia.tar.gz.

| arm | lm-forward (ms) | lm-train-step (ms) |
|---|---|---|
| default | 17.15 / 19.35 | 41.55 / 42.08 |
| K=0, BLOCKS=128 | 16.58 / 16.66 | 43.58 / 43.57 |
| K=0, BLOCKS=192 | 16.21 / 16.25 | 43.01 / 43.03 |

Timer means (ms per tick): fwd.q_proj 0.104 -> 0.051, o_proj 0.084 -> 0.049, k/v 0.060 -> 0.048, gate/up 0.147/0.116 -> 0.098/0.100 at 192 only;
logits.layers 14.82 -> 13.18 (128) / 12.57 (192); but logits.head_gemm 0.58 -> 1.10 at 192 and bwd.mlp_through_oproj 1.21 -> 1.41,
bwd.after_attention 0.66 -> 0.77 at both. Full table: nvidia-raw.txt.

## AMD MI325X (DO pass9, #67 tree, main merged)

Same bits. lm-forward default 23.3 / 23.1, K0B128 28.6 / 28.8, K0B192 25.7 / 25.6; train 58.4 / 58.2, 67.2 / 67.2, 62.4 / 61.5.
Smaller tiles lose on AMD at both floors; #69's rule is NVIDIA-only. R2 gemm61-amd.
