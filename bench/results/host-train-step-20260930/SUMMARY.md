# CPU training step over host tasks (PR #11, lane/neural-pass6) on the AMD box's host, 2026-10-01

CPU column (MOJOLEARN_TARGET_COLUMN=cpu, MOJOLEARN_VENDOR=cpu), AMD EPYC 9575F (64 cores), `tools/hostpr_job.sh pr11`.

| check | result |
|---|---|
| `byte_lm_cpu_train_gate.py cpu --steps all` | PASS, 640/640 array comparisons equal over 128 steps against the recorded bytes |
| `byte_lm_host_step_check.py --steps 4` (board shape, 8 layers) | PASS, both settings byte for byte; rows 885.8 ms/step vs serial 1495.7 (1.69x) |
| `byte_lm_host_step_check.py --small --steps 16` | PASS; 5.6 vs 7.8 ms/step (1.41x) |
| lm-host-train-step board cell, `MOJOLEARN_BYTE_LM_HOST_STEP_ROWS=1` vs `0` | 903.4 vs 1463.6 ms (1.62x), same losses |
| profile (ms): optimizer_rows 69.4 (vs optimizer_step serial 988.1), blocks_backward 319.8, optimizer_copies 53.5, grad_pack 27.2, blocks_pack 25.6, ce_rows 16.6 | total 809.4 |

NVIDIA host and Apple: not run.

## PR #12 (lane/neural-pass7: attention backward chains four vectors in flight), same host

| check | result |
|---|---|
| CPU train gate | PASS 640/640 |
| host step check, board / small | PASS both; rows 800.0 vs serial 1391.4 ms/step (1.74x) / 5.4 vs 7.7 |
| lm-host-train-step cell, rows on / off | 783.4 / 1363.2 ms (PR #11 alone: 903.4 / 1463.6) |
| profile total / blocks_backward | 684.0 / 301.4 ms (PR #11 alone: 809.4 / 319.8) |

## PR #14 (lane/neural-pass8: Adam and CE on lanes, the step owns its state, no packs, block timing)

| check | Zen 5 (EPYC 9575F, AMD box host) | Zen 4 (EPYC 9374F, NVIDIA pod host) |
|---|---|---|
| `pixi run check-host-lanes-sqrt` (sqrt on all 2^32 patterns, div 2^26 pairs, Adam element 2^24 quadruples) | PASS, 0 mismatches | PASS, 0 mismatches |
| CPU train gate | 640/640 | 640/640 |
| host step check, board shape | PASS; rows 684.8 vs serial 1295.9 ms/step (1.89x) | PASS; 1443.5 vs 2552.3 (1.77x) |
| lm-host-train-step cell, rows on / off | 648.4 / 1287.5 ms (PR #12: 783.4) | — |
| profile total / blocks_backward / optimizer_rows | 613.7 / 330.3 / 9.7 ms (PR #12: 684.0 / 301.4 / 71.1) | — |

Block timing (MOJOLEARN_HOST_BLOCK_TIMING=1, Zen 5): bwd.gate_up_proj 58.7 ms, bwd.qkv_proj 50.4, fwd.scores 40.4,
bwd.down_proj_gate_elements 32.4, fwd.gate_up_proj 27.1, fwd.qkv_proj 23.3, bwd.attention_chains 22.8, bwd.kv_gather_rope 21.7.
