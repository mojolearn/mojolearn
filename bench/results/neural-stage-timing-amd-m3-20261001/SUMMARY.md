# Neural stage timing on the MI325X and the M3 Ultra (peer request), 2026-10-01

`tools/neural_stage_timing.py --lane lm-forward --lane lm-train-step --lane transformer-forward --calls 6`, main
(lane/stage-diag-20261001). AMD: released 0.8.32 + main's Python and the transformer/byte_lm/training/mamba bindings
built for gfx942. M3: main source tree built for Metal. Steady state = median after the first call.

| cell | MI325X | M3 Ultra | digest (equal on both) |
|---|---:|---:|---|
| lm-forward | 55.6 ms (board 0.8.25: 89.7) | 115.0 ms (board 0.8.25: 250.1) | 4a8e781b0739a038 |
| lm-train-step | 121.2 ms | 274.1 ms | (losses equal) |
| transformer-forward | 5.24 ms | 11.96 ms | d5a2b289afdb5709 |

lm-forward per layer (8 layers, average over 6 calls x 8):

| stage | MI325X | M3 Ultra |
|---|---:|---:|
| attn.core | 1.40 ms | **11.61 ms** |
| attn.qkv_proj | 1.20 ms | 0.88 ms |
| attn.o_proj | 0.40 ms | 0.59 ms |
| block.mlp_and_residuals | 1.39 ms | 2.20 ms |
| block.norm1 | 0.25 ms | 1.24 ms |
| block.attention_total | 3.07 ms | 14.29 ms |

lm-train-step backward per layer: bwd.attention 3.80 ms AMD / 7.58 ms M3; bwd.after_attention 2.63 / 3.49 ms;
envelope.blocks_backward 80.2 / 120.7 ms of a 121.5 / 249.3 ms native call.
Read: on Apple the forward attention core is ~80% of lm-forward; on AMD the time is spread (attention core, the qkv
projection at 1.2 ms, and the MLP), and the backward attention is the largest training stage on both.
