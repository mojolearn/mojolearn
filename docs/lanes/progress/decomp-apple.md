# decomp-apple: progress (Apple Metal speed, FAST and IDENTICAL)

Brief: ~/mojolearn-evidence/apple_speed_brief.md. Branch lane/decomp-apple
(off origin/main, with origin/lane/algos-decomp merged in: its unmerged GPU
speed work and bench/decomp_speed.py are this lane's base). Home Mac for
speed: m4pro-a. Files: ~/mojolearn-evidence/decomp-apple/.

## Profile (existing Apple records, before this lane)

m4pro-a / m4pro-b, 1M x 28 synth (`~/mojolearn-evidence/algos-decomp/speed/`):
the x_decomp kit's cost on Apple is per-call data movement, not arithmetic.
`ew` over 1M x 5: ~2.9 ms a call on the M4 Pro against ~0.33 ms on MI300X
(FastICA 338 ew = 0.99 s vs 0.11 s); gemm 123 calls 1.13 s vs 0.24 s; orth
9 calls 1.5 s vs 0.65 s. Every kit call uploaded its inputs into fresh
buffers, synced twice and downloaded its output.

## Found and fixed

1. **Every x_decomp fit raised TypeError on CPython 3.13** (the pixi env on
   the Macs is 3.13.15): `_M.from_input` passed a typed memoryview to
   `array.frombytes` (since 2f2c2891b). Fix: `mv.cast("B")`, the same one
   copy (ec9a05770).
2. **Isomap hang at 10k rows** (AMD box, Andrew 2026-09-28): the shortest
   paths were a dense O(n^2) scan per source (O(n^3) total, with column
   reads), and the component walk an O(n^2) Python loop. Now: compressed
   arcs built once on the host, a (distance, index) binary heap per row
   (the same pick as the scan: least distance, ties to the lower index, and
   each distance the exact minimum of the same float32 sums, so the same
   bits), the component walk over adjacency lists. Host column bit-equal to
   the old dense routine on 12 random graphs with integer ties
   (dj_check.py). Arm 5314 regenerated on the new cell. Isomap past a few
   thousand rows is still bounded by its n x n Jacobi eigh (arpack is
   refused by name); bench/decomp_speed.py caps Isomap at 3000 rows.
3. **Device-resident matrices** (x_decomp/resident.mojo): ew, gemm, colsum,
   rowsum and sqdist/pdist launch on pooled device buffers with no sync; a
   result stays on the device until Python reads it. Same kernels, same
   launch sequence (DevExec now calls the same launch_* helpers). Python
   plumbing proven on the host with a fake resident binding over the host
   kit: 22 algorithms bit-equal to the plain host path, no buffer leaked
   (mock_resident.py). orth and absmax_sign followed (0456bf309): DevExec
   and the resident entries call the same orth_on_device / launch_absmax.

4. **geqrf / orgqr staged folds** (0936290ea): the one-thread strided
   column folds (geqrf_head + reflector_norm, geqrf_dot, orgqr_dot) now read
   their column through threadgroup memory: the block loads 2048-row chunks,
   thread 0 folds them in the cells' order with the cells' arithmetic.
5. **Wide Jacobi eigh** (fb281eb88): x_decomp's eigh launches
   jacobi_eigh_kernel 1024 wide from n = 256 (launch-width invariant under
   IDENTICAL, DEVIATION 2680); FAST keeps device_eigh.

## Verification done before Andrew's "no verification in your lane" (2026-09-28 ~08:30Z)

Steward 1790581701486-decomp-06752fa5ff (16 x-decomp lanes, e2e_host_all):
PASS on do-amd, m4-a, m3ultra-b, m2pro (Metal == CPU on M2/M3/M4 and
gfx942 == CPU; the sabotage bit). It covers items 1-3. Items 4-5 are proven
by digests inside their speed request (bench/decomp_out_digest.py GPU and
CPU columns; decomp_eigh_width.py hashes per width).

## Metal profile, BEFORE (m4pro-a, IDENTICAL, 1790581691614 at 112ef9d1c)

Micro (median of 10, ms): a kit call's fixed cost ~0.2 ms; ew add 1M x 28
64 ms (5 GB/s: copies), colsum 1M x 28 8.8 ms; gemm 1M x 28 @ 28 x 5
18.9 ms; orth 1M x 10 110 ms; eigh 256: 177 ms.

decomp_speed, synth N=200k (N2 20k, N3 1500, N4 20k), seconds:

| algorithm | before | top entries |
|---|---|---|
| PCA(randomized,5) | 0.264 | orth 0.129, gemm 0.070 |
| TruncatedSVD(randomized,5) | 0.320 | orth 0.160, gemm 0.093 |
| IncrementalPCA | 0.077 | ew 0.040 |
| GaussianRandomProjection / Sparse | 0.036 / 0.035 | ew 0.022 |
| NMF mu / cd (20 it) | 0.467 / 0.429 | gemm 0.225 / 0.186 |
| FastICA (20 it) | 0.323 | ew 338x 0.174 |
| FactorAnalysis (20 it) | 0.312 | ew 0.133, svd 0.100 |
| lstsq | 0.131 | svd 0.051 |
| randomized_svd | 0.260 | orth 0.132 |
| PLSRegression / CCA | 0.481 / 0.572 | gemm 0.310 / 0.245 |
| MinCovDet (n2=20k x 8) | 46.98 | ew 91k calls 17.1, eigh 8.4k 6.8, colsum 6.0, gemm 5.9, lu 4.9 |
| linalg.qr / linalg.svd | 8.86 / 8.91 | geqrf 6.94, orgqr 1.91 |
| solve(512) | 0.146 | |
| SparsePCA / DictionaryLearning | 0.363 / 0.375 | |
| MiniBatchDictionaryLearning | 1.668 | ew 5776x 1.04 |
| LatentDirichletAllocation | 0.503 | lda_rows 0.265 |
| ALS / ALS cg | 3.97 / 1.42 | als_rows 3.68 |
| Isomap (1500) | 107.5 | eigh 103.4, dijkstra 3.70 |
| LocallyLinearEmbedding (1500) | 429.3 | svd 428.9 (one-sided Jacobi of a 1500 x 1500) |
| ClassicalMDS (1500) | 69.1 | eigh 69.1 |
| MDS (5 it) | 0.207 | |
| SpectralEmbedding(knn, 20k) | FAILED | select_radix: k > 1024 refused (not this lane's code) |
| SpectralEmbedding(rbf, 5k) | 3.19 | Python 3.04 |
| UMAP default / c5 manhattan (20k) | 0.236 / 0.611 | |

## Requests in flight

| request | what |
|---|---|
| 1790582024178-speed-decomp-fb281eb88d (m4pro-a) | AFTER all five: eigh widths, micro, decomp_speed (same sizes), out digests GPU + CPU |
| 1790583099651-speed-decomp-06752fa5ff (m4pro-a) | after items 1-3 only (the before of items 4-5): micro, decomp_speed |

## Before -> after (per algorithm, IDENTICAL, m4pro-a)

(pending)

## FAST

x_decomp's FAST build runs the IDENTICAL cells (`_sets/identical` is
loaded in fast mode); no FAST-only path exists yet.
