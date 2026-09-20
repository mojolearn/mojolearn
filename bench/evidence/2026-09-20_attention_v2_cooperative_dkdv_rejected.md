# Attention-v2 cooperative dK/dV: rejected

Apple IDENTICAL-mode screening at `B=1, H=12, L=2048, HD=64` tested two
cooperative dK/dV ownership schemes. Both kept one block per `(group,key)`, one
feature lane per dK/dV cell, and each cell's required ascending-query fold.
Lane zero computed the pinned `identical_div` probability and score gradient;
no atomics, fast math, whole-kernel serialization, or LxL storage were used.

| arm | v2 backward times (ms) | median | versus stock v2 |
|---|---:|---:|---:|
| stock one-thread `(group,key)` owner | 764.295, 757.893, 773.415 | 764.295 | 1.00x |
| shared `p,ds`, two barriers/query | 2847.281, 2764.325, 3094.037 | 2847.281 | 3.72x slower |
| 32-query shared scalar tiles, two barriers/tile | 2789.124, 2837.340, 2961.939 | 2837.340 | 3.71x slower |

Both candidates passed the complete small/tail/window backward fixture with the
unchanged dQ/dK/dV digest
`6306341d1a3683938e52234ad77e6041bbac5f39dcc8963c902a3ec523577df1`.
Resident allocation stayed 38,240,256 bytes and quadratic saved bytes stayed
402,653,184. The same run's production-v1 median was 136.262 ms.

The extra feature parallelism did not pay for launching 24,576 64-lane blocks
that each traverse the causal query span, even after barriers were amortized
over 32 queries. Both source candidates were reverted. Do not retry this block
ownership on Apple without a materially different work decomposition.
