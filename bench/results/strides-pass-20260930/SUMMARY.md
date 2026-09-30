# Strides pass (PR #5, lane/neural-net-experiment 5ecfc4117) on the NVIDIA L40S, 2026-09-30

Job `tools/strides_job.sh` on the shared pod nvc1 (job nvc1-0006). Measurement
overlay: mojolearn 0.8.31, Python sources plus the mamba and linalg IDENTICAL
bindings replaced from the head. Every change against its restore; bits compared.

## Bits

| change | check | result |
|---|---|---|
| Mamba-3 S16/S17 backward kernels | y and all 10 gradients, two calls, B2 L512 d384 and B8 L512 d768, new vs `MOJOLEARN_MAMBA3_S16_QK_NAIVE=1 MOJOLEARN_MAMBA3_S17_OPERANDS_NAIVE=1` | identical (digest-*.json) |
| gemm-int8 zero-copy codes | board cell digest | `9b16c7064e10cecd`, the same as every earlier L40S run |
| fixed15 transposing quantizer | 422 DIGEST lines, `MOJOLEARN_INT15_TRANSPOSED_TILE=1` vs `0` | 0 differ |
| fixed15 plan table | 422 DIGEST lines, table vs `MOJOLEARN_INT15_BOX=high` | 0 differ |

## Time

| cell | before | after |
|---|---|---|
| gemm-int8 board cell | 2,460 ms (priority pass) | 40.7 ms |
| Mamba-3 S17 operands, board shape / default shape | 9.0 / 46.9 ms | 6.8 / 19.0 ms |
| Mamba-3 S16+S15, board shape / default shape | 23.8 / 191.0 ms | 21.9 / 174.4 ms (the "a few ms" goal was not met) |
| samba-train-step board cell | ~136 ms (direction pass) | 131.1 ms |
| fixed15 training op, mlp_up bwd_dx (transposed tile) | 4.09 ms | 2.48 ms |
| fixed15 training op, mlp_down bwd_dx (transposed tile) | 3.69 ms | 2.84 ms |
| fixed15 product, mlp_up t512 (plan table vs H100 choice) | 3.31 ms | 1.72 ms |
| fixed15 product, lm_head t512 (plan table) | 4.45 ms | 2.47 ms |
| fixed15 product, mlp_down bwd_dx (plan table) | 3.33 ms | 1.73 ms |

One row is slower under the table: mlp_down t512 bwd_dw product 1.318 ms (H100
choice) vs 1.380 ms (table), 5%. Every other row is the same or faster.
AMD and Apple: not run.
