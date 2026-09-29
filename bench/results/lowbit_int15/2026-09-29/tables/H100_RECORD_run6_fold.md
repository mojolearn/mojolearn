# H100 table of record: the epilogue fold (run 6)

Box: H100 NVL, pod nvc3. Job nvc3-0034, commit 71db5bb26 (lane/lowbit-mma-speed 15e9ecf47
merged: its fused four-product form calls this lane's `int15_store_cell`), 2026-09-29 05:20Z
to 05:26Z, exit 0. One lever: fused (the epilogue at the sums kernel's store, one launch, no
sums in device memory) against two-launch (the sums stored, 12 bytes a cell, then the
epilogue launch). Full log `runs/h100_run6_nvc3-0034_71db5bb26.txt`, full tables
`TIMING_h100_run6.md`.

GATE BEFORE THE CLOCK, same job: tuned gate GREEN. Every plan fused and two-launch equals the
oracle on every shape, planted case and the row-scales case; the value, pieces, epilogue,
host and exponent arms each seen failing; the refusals hold. MI325X alongside (1790659063533)
GREEN with the same arms; tuned-gate digests of the two boxes: 149 cases, 0 disagreeing.
Timing: warm and timed runs printed the same 422 digests; the sabotaged build changed every
fifteen-bit digest. The first run 6 (nvc3-0032) was RED at its gate (the exponent arm could
not fail) and timed nothing.

## The weight-gradient product (task 4's target), both operands converted per call

| row | fp32.v1 ms | two-launch ms | over | fused ms | over | fused / two-launch |
|---|---:|---:|---:|---:|---:|---:|
| qkv.t512.bwd_dw | 0.8809 | 0.8973 | 1.019 | 0.5079 | 0.577 | 0.566 |
| mlp_up.t512.bwd_dw | 2.9816 | 2.7042 | 0.907 | 1.5100 | 0.506 | 0.558 |
| mlp_down.t512.bwd_dw | 3.1062 | 2.9039 | 0.935 | 1.6126 | 0.519 | 0.555 |
| lm_head.t512.bwd_dw (n capped) | 3.3422 | 2.9710 | 0.889 | 1.6675 | 0.499 | 0.561 |

## The complete inference call on the tuned (now fused) plan, over fp32.v1 in the same run

| row | fp32.v1 ms | fused ms | over | two-launch ms | over |
|---|---:|---:|---:|---:|---:|
| qkv.t1 | 0.0676 | 0.1568 | 2.320 | 0.1116 | 1.651 |
| qkv.t8 | 0.1010 | 0.1673 | 1.656 | 0.1273 | 1.260 |
| qkv.t512 | 1.0314 | 0.4500 | 0.436 | 0.4995 | 0.484 |
| mlp_up.t1 | 0.1791 | 0.1583 | 0.884 | 0.1542 | 0.861 |
| mlp_up.t8 | 0.2707 | 0.1803 | 0.666 | 0.1795 | 0.663 |
| mlp_up.t512 | 3.4303 | 1.5533 | 0.453 | 1.7186 | 0.501 |
| mlp_down.t1 | 0.1678 | 0.4489 | 2.675 | 0.4434 | 2.642 |
| mlp_down.t8 | 0.2885 | 0.4929 | 1.708 | 0.4929 | 1.708 |
| mlp_down.t512 | 3.3921 | 1.4140 | 0.417 | 1.4637 | 0.432 |
| lm_head.t1 | 1.2439 | 1.1158 | 0.897 | 1.1142 | 0.896 |
| lm_head.t8 | 2.3611 | 1.2392 | 0.525 | 1.2337 | 0.523 |
| lm_head.t512 (n capped at 16032) | 3.7860 | 1.7656 | 0.466 | 1.8941 | 0.500 |

READ WITH CARE, qkv.t1 and qkv.t8: the PRODUCT alone reads fused/two 0.982 and 1.052, but the
complete call reads fused 0.1568 against two-launch 0.1116 (1.405) and 0.1673 against 0.1273
(1.314). The complete two-launch there is below run 5's complete tuned call (0.1602, 0.1722),
and the fused one is at it. The two arms differ only in the product, which takes the same time
alone, so the difference is not the fold's; CAUSE NOT ESTABLISHED (the arm's place in the
alternation is one candidate). No other row shows it.

## Three products of one layer, each timed alone and added (NOT a training step), fused

| layer | fp32.v1 ms | fused ms | over | run 5 (two-launch) over |
|---|---:|---:|---:|---:|
| qkv.t512 | 2.9262 | 1.7894 | 0.612 | 0.765 |
| mlp_up.t512 | 9.7549 | 6.1903 | 0.635 | 0.765 |
| mlp_down.t512 | 9.8890 | 5.7950 | 0.586 | 0.730 |
| lm_head.t512 | 10.9529 | refused (input gradient k = 128256 > 65536) | refused | refused |
