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

## FINAL (2026-09-28 ~12:05Z): branch head 1a6b43a1b, nothing in flight

Resident threshold swept on m4pro-a (1790588268075, MOJOLEARN_XD_RES_MIN
1 / 1024 / 16384; the code of 1a6b43a1b is the RES_MIN=1 column):

| algorithm | before s | FINAL s (RES_MIN 1) | 1024 | 16384 | before/FINAL |
|---|---|---|---|---|---|
| TruncatedSVD(randomized,5) | 0.320 | 0.197 | 0.201 | 0.203 | 1.62x |
| IncrementalPCA | 0.077 | 0.044 | 0.050 | 0.048 | 1.75x |
| NMF mu | 0.467 | 0.195 | 0.222 | 0.220 | 2.39x |
| NMF cd | 0.429 | 0.331 | 0.365 | 0.359 | 1.30x |
| FastICA | 0.323 | 0.089 | 0.129 | 0.124 | 3.63x |
| FactorAnalysis | 0.312 | 0.193 | 0.206 | 0.214 | 1.62x |
| lstsq | 0.131 | 0.076 | 0.075 | 0.085 | 1.72x |
| randomized_svd | 0.260 | 0.145 | 0.144 | 0.148 | 1.79x |
| PLSRegression | 0.481 | 0.145 | 0.147 | 0.145 | 3.32x |
| CCA | 0.572 | 0.210 | 0.222 | 0.221 | 2.72x |
| MinCovDet | 46.98 | 48.43 | 55.01 | 51.06 | 0.97x |
| solve(512) | 0.146 | 0.151 | 0.163 | 0.166 | 0.97x |
| SparsePCA | 0.363 | 0.129 | 0.136 | 0.139 | 2.81x |
| DictionaryLearning | 0.375 | 0.262 | 0.339 | 0.346 | 1.43x |
| MiniBatchDictionaryLearning | 1.668 | 0.657 | 1.560 | 1.761 | 2.54x |
| LatentDirichletAllocation | 0.503 | 0.299 | 0.301 | 0.302 | 1.68x |

Not in the sweep (from E, whose code for them is the same): PCA randomized
0.264 -> 0.154 (1.71x), GaussianRandomProjection 0.036 -> 0.011 (3.3x),
SparseRandomProjection 0.035 -> 0.012 (2.9x), MDS 0.207 -> 0.052 (4.0x),
linalg.qr 8.855 -> 1.664 (5.3x), linalg.svd 8.911 -> 1.824 (4.9x), Isomap's
shortest paths at 1500 rows 3.70 -> 0.115 s (32x; the fit is its eigh).
Unchanged: ALS / ALS cg (row cells), LocallyLinearEmbedding, ClassicalMDS,
Isomap fit (n x n Jacobi). FAST: same binding, same numbers, same bits.

Bits: digests (bench/decomp_out_digest.py, 27 algorithms) equal on Metal
and CPU at C (always resident) and at E (threshold 2^14), and Metal C == E;
the threshold only picks between two launch paths of the same kernels
(proven on the host at thresholds 1 and 2^14 with the fake resident binding).

## NEXT (a later session, if any)

- The n x n cyclic Jacobi (eigh) and one-sided Jacobi (x_decomp svd) are the
  walls for ClassicalMDS / Isomap / LocallyLinearEmbedding past ~1000 rows;
  IDENTICAL pins the rotation order, so only per-rotation latency can move
  (the 1024-wide launch did not help). A FAST route (a different algorithm)
  would need the paired quality check.
- ALS als_rows: one thread per user folding all items.
- SpectralEmbedding(knn) at 20k rows refuses in select_radix (k > 1024), not
  x_decomp code: the neighbors lane's.

## Before -> after, CURRENT (m4pro-a, IDENTICAL, synth 200k x 28; E = 1790586709302 at 712cdc7d4)

A (1790583099651 at 06752fa5f, items 1-3 only) isolates item 4: linalg.qr
8.64 s at A -> 1.66 s at E (geqrf 6.75 -> 1.24, orgqr 1.88 -> 0.42).
Digests at E: all 27 algorithms of bench/decomp_out_digest.py print the same
hash on Metal and on the CPU, and the same Metal hashes as at C.

| algorithm | before s | C s | E s | before/E |
|---|---|---|---|---|
| PCA(randomized,5) | 0.264 | 0.164 | 0.154 | 1.71x |
| TruncatedSVD(randomized,5) | 0.320 | 0.207 | 0.204 | 1.57x |
| IncrementalPCA | 0.077 | 0.045 | 0.049 | 1.57x |
| GaussianRandomProjection | 0.036 | 0.010 | 0.011 | 3.27x |
| SparseRandomProjection | 0.035 | 0.011 | 0.012 | 2.92x |
| NMF mu | 0.467 | 0.225 | 0.221 | 2.11x |
| NMF cd | 0.429 | 0.338 | 0.348 | 1.23x |
| FastICA | 0.323 | 0.088 | 0.134 | 2.41x |
| FactorAnalysis | 0.312 | 0.196 | 0.301 | 1.04x |
| lstsq | 0.131 | 0.084 | 0.104 | 1.26x |
| randomized_svd | 0.260 | 0.148 | 0.175 | 1.49x |
| PLSRegression | 0.481 | 0.145 | 0.152 | 3.16x |
| CCA | 0.572 | 0.218 | 0.212 | 2.70x |
| MinCovDet | 46.98 | 67.68 | 51.36 | 0.91x |
| linalg.qr | 8.855 | 1.674 | 1.664 | 5.32x |
| linalg.svd | 8.911 | 1.840 | 1.824 | 4.89x |
| solve(512) | 0.146 | 0.306 | 0.150 | 0.97x |
| SparsePCA | 0.363 | 0.237 | 0.137 | 2.65x |
| DictionaryLearning | 0.375 | 0.451 | 0.346 | 1.08x |
| MiniBatchDictionaryLearning | 1.668 | 2.582 | 1.775 | 0.94x |
| LatentDirichletAllocation | 0.503 | 0.425 | 0.303 | 1.66x |
| MDS (5 it) | 0.207 | 0.054 | 0.052 | 3.98x |
| Isomap shortest paths (1500) | 3.703 | 0.115 | | 32x |
| ALS / ALS cg | 3.969 / 1.420 | | 3.910 / 1.324 | 1.02x / 1.07x |
| TruncatedSVD full / UMAP / SpectralEmbedding rbf (no x_decomp kit calls) | 0.014 / 0.236 / 3.19 | | 0.020 / 0.301 / 3.52 | slower; not this lane's code paths (A shows the same) |

The 2^14 threshold (E) gave back part of C's gain on FastICA and
FactorAnalysis (their small-matrix calls went synchronous again) while
fixing MinCovDet; request F sweeps it.

## Before -> after, round C (m4pro-a, IDENTICAL, synth 200k x 28; C = 1790582024178 at fb281eb88)

C carried items 1-5 without the pool fixes (wide eigh, since reverted, ran
in its Isomap/ClassicalMDS). Digests at C: all 27 algorithms of
bench/decomp_out_digest.py print the SAME hash on the Metal column and the
CPU column (and the CPU hashes equal the laptop host run).

| algorithm | before s | C s | speedup |
|---|---|---|---|
| PCA(randomized,5) | 0.264 | 0.164 | 1.61x |
| TruncatedSVD(randomized,5) | 0.320 | 0.207 | 1.55x |
| IncrementalPCA | 0.077 | 0.045 | 1.71x |
| GaussianRandomProjection | 0.036 | 0.010 | 3.60x |
| SparseRandomProjection | 0.035 | 0.011 | 3.18x |
| NMF mu | 0.467 | 0.225 | 2.08x |
| NMF cd | 0.429 | 0.338 | 1.27x |
| FastICA | 0.323 | 0.088 | 3.67x |
| FactorAnalysis | 0.312 | 0.196 | 1.59x |
| lstsq | 0.131 | 0.084 | 1.56x |
| randomized_svd | 0.260 | 0.148 | 1.76x |
| PLSRegression | 0.481 | 0.145 | 3.32x |
| CCA | 0.572 | 0.218 | 2.62x |
| linalg.qr | 8.855 | 1.674 | 5.29x |
| linalg.svd | 8.911 | 1.840 | 4.84x |
| SparsePCA | 0.363 | 0.237 | 1.53x |
| LatentDirichletAllocation | 0.503 | 0.425 | 1.18x |
| MDS (5 it) | 0.207 | 0.054 | 3.83x |
| Isomap shortest paths (1500) | 3.703 | 0.115 | 32x (the fit is its 1500 x 1500 eigh) |
| MinCovDet | 46.98 | 67.68 | 0.69x REGRESSED -> fixed in 712cdc7d4 (pending) |
| MiniBatchDictionaryLearning | 1.668 | 2.582 | 0.65x REGRESSED -> same fix |
| DictionaryLearning | 0.375 | 0.451 | 0.83x REGRESSED -> same fix |
| solve(512), TruncatedSVD full, UMAP | 0.146 / 0.014 / 0.236 | 0.306 / 0.023 / 0.297 | slowed after MinCovDet filled the pool (their code did not change) -> same fix |
| ALS, ALS cg, LLE, ClassicalMDS, Isomap fit | unchanged within noise | | row cells / n x n Jacobi |

Tried and reverted: a 1024-wide launch of jacobi_eigh_kernel (launch-width
invariant, same hash at 256/512/1024): n=400 1.59 -> 1.21 s, but n=800 10.9
-> 12.4 s and n=1500 slower too (c948ca1bb).

Walls left (IDENTICAL cannot reorder them): the n x n cyclic Jacobi eigh
(ClassicalMDS / Isomap at 1500 rows: ~70-100 s) and the one-sided Jacobi
SVD of a 1500 x 1500 (LocallyLinearEmbedding: 429 s); each rotation is
serial in the pinned order. ALS's als_rows (one thread per user, a dense
32 x 32 solve each).

## FAST

x_decomp's FAST mode loads the IDENTICAL binding (`_sets/identical`), so
every change above reaches FAST as-is with the same bits; this lane made no
FAST-only change, so there is no quality delta to pair.
