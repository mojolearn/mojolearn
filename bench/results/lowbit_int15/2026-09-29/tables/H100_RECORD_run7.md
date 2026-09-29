# H100 table of record: run 7, the fused tuned plan on lane/lowbit-mma-speed 8b2768883

Box: H100 NVL, pod nvc3. Job nvc3-0037, commit b9db1a41b (lane/lowbit-mma-speed 8b2768883
merged: its dispatcher now picks the two-page plans), 2026-09-29, exit 0. The lever between
run 6 and run 7 is Lane D's (the sums kernel's plans); nothing of this lane moved. Full log
`runs/h100_run7_nvc3-0037_b9db1a41b.txt`, full tables `TIMING_h100_run7.md`.

Gate before the clock, same job: tuned gate GREEN (every plan fused and two-launch equal to
the oracle, the row-scales case included; value, pieces, epilogue, host and exponent arms seen
failing). Warm and timed runs printed the same 422 digests; the sabotaged build changed every
fifteen-bit digest. MI325X at the same commit (1790659664486): gate and tuned gate GREEN.

## Complete inference call, fused tuned plan, over fp32.v1 in the same run

| row | fp32.v1 ms | fused ms | over |
|---|---:|---:|---:|
| qkv.t1 | 0.0694 | 0.0970 | 1.398 |
| qkv.t8 | 0.1024 | 0.1043 | 1.019 |
| qkv.t512 | 1.0313 | 0.3152 | 0.306 |
| mlp_up.t1 | 0.1808 | 0.1140 | 0.631 |
| mlp_up.t8 | 0.2714 | 0.1243 | 0.458 |
| mlp_up.t512 | 3.5103 | 1.0182 | 0.290 |
| mlp_down.t1 | 0.1698 | 0.2387 | 1.406 |
| mlp_down.t8 | 0.2882 | 0.2541 | 0.882 |
| mlp_down.t512 | 3.4331 | 0.8880 | 0.259 |
| lm_head.t1 | 1.2945 | 0.6073 | 0.469 |
| lm_head.t8 | 2.5144 | 0.6553 | 0.261 |
| lm_head.t512 (n capped at 16032) | 4.2485 | 1.2124 | 0.285 |

## Weight gradient, both operands converted, fused

| row | fp32.v1 ms | fused ms | over | two-launch over |
|---|---:|---:|---:|---:|
| qkv.t512.bwd_dw | 0.8824 | 0.4476 | 0.507 | 0.857 |
| mlp_up.t512.bwd_dw | 2.9755 | 1.3334 | 0.448 | 0.750 |
| mlp_down.t512.bwd_dw | 3.1123 | 1.4136 | 0.454 | 0.747 |
| lm_head.t512.bwd_dw | 3.3415 | 1.4888 | 0.446 | 0.744 |

## Three products of one layer, each timed alone and added (NOT a training step), fused

| layer | fp32.v1 ms | fused ms | over |
|---|---:|---:|---:|
| qkv.t512 | 2.9313 | 1.4450 | 0.493 |
| mlp_up.t512 | 9.8163 | 4.9062 | 0.500 |
| mlp_down.t512 | 10.1269 | 4.5579 | 0.450 |
| lm_head.t512 | 11.4094 | refused (input gradient k = 128256 > 65536) | refused |

READ WITH CARE: at qkv.t1 and qkv.t8 the complete fused call again reads above the complete
two-launch call (1.255, 1.242) while the fused product alone reads below it (0.917, 0.786), as
in run 6. Cause not established; it is not the product's time. fp32.v1 moved between runs 6
and 7 by up to 12% at the same row (lm_head.t512 3.79 / 4.25 ms), which is why every ratio
divides by the same run's fp32.v1.
