# Byte LM CPU inference token split (PR #10, lane/host-token-split) on the AMD box's host, 2026-10-01

AMD EPYC 9575F (64 cores). The PR as pushed did not compile; three fixes on lane/host-token-split-measure:
the one-element holder lists indexed as `xp[0]` (the held List) instead of `xp[][0]` (its first float),
the missing `ftz` import, and a trailing chunk that starts past the last token now returns (with 64 workers
the last chunks were empty and raised "token chunk 17 raised in the projections").

| check | result |
|---|---|
| `byte_lm_host_gate.py`, split on and `MOJOLEARN_BYTE_LM_HOST_TOKEN_SPLIT=0` | PASS 144/144 loss bytes equal, both |
| `byte_lm_host_path_sweep.py` | PASS 4752/4752 (logits, next bytes, loss bits) over 3 states, batch 1..8, length 1..32, threads 1..3 |
| lm-infer board cell (CPU column), split on vs off | 175.98 vs 642.70 ms (3.65x), mean_nll 9.017856651220221 both |

The gate and sweep need the base `_mojolearn` binding built in the tree (argmax_rows_f32 has no CPU
implementation); built for gfx942 here. NVIDIA host and Apple: not run.
