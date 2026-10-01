# Bit-inert geqrf/orgqr staged kernels (PR #24, lane/neural-pass18), 2026-10-01

Board qr and svd cells, released 0.8.32 (before) vs 0.8.32 + this branch's x_decomp and linalg (after), 3 rounds.

| cell | AMD MI325X before / after | NVIDIA L40S before / after | digest (all runs, both vendors) |
|---|---|---|---|
| qr taxi | 4,062.7 / 894.2 ms (4.5x) | 966.6 / 269.4 ms (3.6x) | e54f8db0f5ca6525 |
| qr istella | 76,816.7 / 16,723.5 ms (4.6x) | 21,007.6 / 8,393.2 ms (2.5x) | 76e60d3379a2735a |
| svd taxi | 4,063.3 / 922.1 ms (4.4x) | 985.9 / 345.5 ms (2.9x) | f07891b9aaf42a98 |
| svd istella | 92,192.3 / 38,142.4 ms (2.4x) | 42,177.2 / 32,230.0 ms (1.3x) | 92c189cb2337dc73 |

The digests equal the priority-pass evidence (NVIDIA and AMD) for these cells.
