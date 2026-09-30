# Classical pass A/B, NVIDIA L40S, 2026-09-30

Old path (restore env on, or the `-D MOJOLEARN_IVF_IDENTICAL_SCAN_OFF` build for IVF) vs new path,
same seeded input, identical mode, `tools/classical_pass_ab.py`. mojolearn 0.8.31 wheel with the
x_decomp, x_linear and ivf bindings built from lane/classical-pass-measure-20260930 (sources for those
families equal to this merge). Every new path's output SHA-256 equals the old path's.

| fix | digest size | old ms | new ms | same bits | speedup | new path, board shape |
|---|---|---:|---:|---|---:|---|
| LU pivot block + solve per RHS | 1024x1024, 64 RHS | 3014 | 398 | yes | 7.6x | 8192x8192: 8.4 s (0.8.25 AMD board: 597 + 616 s) |
| SGD regression on host | 20k x 220, 100 epochs | 87897 | 3473 | yes | 25.3x | 1M x 220: 176 s (0.8.25 board: 630 s) |
| SGD classification on host | 20k x 220 | 12488 | 3293 | yes | 3.8x | 1M x 220: 175 s (0.8.25 board: 615 s) |
| LARS grid Gram | 200k x 220 | 9162 | 6488 | yes | 1.4x | 1M x 220: 6.7 s (0.8.25 board: 15.4 s) |
| IVF-Flat batched scan off Apple | 40k index, 400 queries | 3994 | 1517 | yes | 2.6x | 400k index, 4k queries: 4.7 s (0.8.25 board: 526 s) |

Inputs are synthetic (N(0,1), 220 columns), not Istella. The A/B driver process segfaulted at interpreter
exit after writing summary.json (rc -11); every arm ran in its own subprocess and completed.
AMD (WARP_SIZE 64 launch and merge in the IVF scan) NOT yet run: no MI300X stock on RunPod or Hot Aisle.
