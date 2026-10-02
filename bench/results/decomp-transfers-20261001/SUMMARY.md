# Decomp kit: host finiteness check, one conversion, staged Apple downloads (PR #32, lane/neural-pass27), 2026-10-01

Istella cells, ours, 3 rounds. AMD/NVIDIA: released 0.8.32 (before) vs 0.8.32 + the branch's x_decomp/linalg (after).
M3 Ultra (Metal): main's source tree vs the branch's, both built there (+ the portable math library).

| cell | AMD MI325X | NVIDIA L40S | Apple M3 Ultra | digest (all) |
|---|---|---|---|---|
| gaussian-rp | 211.6 -> 98.0 ms (2.2x) | 460.2 -> 95.7 ms (4.8x) | 135.6 -> 55.8 ms (2.4x) | 66275aa7ff67e576 |
| sparse-rp | 205.9 -> 97.2 ms (2.1x) | 529.4 -> 104.2 ms (5.1x) | 143.5 -> 56.0 ms (2.6x) | d01cdcaa5dd4ef1c |
| svd | 92,292.8 -> 38,170.7 ms | 42,309.1 -> 30,505.2 ms | 45,434.2 -> 43,806.7 ms | 92c189cb2337dc73 |

The svd before/after on AMD and NVIDIA is mostly PR #24 (already on main; "before" is the released wheel); on the M3
(main vs branch) it is even.
