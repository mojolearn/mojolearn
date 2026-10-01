# lu_solve as a block-interleaved host walk (PR #42, lane/neural-pass39), 2026-10-01

Both board LU cells are lu_solve(lu_factor(A), B) with B 8192 x 64; the solve ran as one GPU thread per column of B
(64 threads, a 67M-step chain each). Restore = MOJOLEARN_XD_LU_SOLVE_HOST=0 (the device threads).
| | lu-factor before / after / restore | lu-solve before / after / restore |
|---|---|---|
| AMD MI325X (released 0.8.32 vs branch) | 10,404.7 / 805.1 / 9,884.4 ms (12x) | 10,533.3 / 807.3 / 10,143.7 ms (13x) |
| NVIDIA L40S (released 0.8.32 vs branch) | 6,270.2 / 1,217.4 / 4,499.7 ms (3.7x vs main) | 6,253.9 / 1,231.5 / 4,519.1 ms |
| Apple M3 Ultra (main vs branch) | 14,560.3 / 2,291.0 / 14,470.3 ms (6.3x) | 14,482.0 / 2,349.9 / 14,529.2 ms (6.2x) |
| digest, every run on all three vendors | 43e06346de01739b | 43e06346de01739b |
