# lane/apple-fast-decomp-sparse: the decomp lane's sparse, manifold, covariance and projection rows (FAST on Apple)

Written without a Mojo toolchain (cloud peer); the first M3 build is the compile check
(bindings/build_x_decomp.sh FAST, each define through MOJOLEARN_MOJO_BUILD_FLAGS as tools/afc_ab_def.sh does).
Every switch is a `-D` build define, default off, compiled only under `XD_FAST_APPLE` (x_decomp/device.mojo:
`GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()`); the Python one learns the build's defines
from the binding (`x_decomp_fast_defines` in x_decomp/api.mojo, `_kit_fast_define` in
python/mojolearn/_expansion_decomp.py, asked once per kit), so nothing reads an env variable.
IDENTICAL compiles and runs main's code unchanged. Merged with origin/main (98f0ff53e).

M3 Ultra board 0831 (ours FAST ms / scikit-learn ms): sparse-coder Istella 6,669 / 29 (228x), taxi 4,995 / 42;
gaussian-rp Istella 136 / 24, sparse-rp 135 / 25; mb-sparse-pca Istella 9,817 / 2,006; mb-dict-learning
Istella 16,175 / 5,030; min-cov-det / elliptic-envelope taxi 3,985 / 1,413 (Istella: no result);
dict-learning Istella 19,308 / 16,442, taxi 10,976 / 11,561; fastica Istella 1,576 / 16,796; mds 608 / 2,666;
isomap and classical-mds: no M3 row.

| switch | kind | site | lanes | what it changes under FAST on Apple |
|---|---|---|---|---|
| `MOJOLEARN_DECOMP_FAST_OMP_BLOCK` | define (Mojo) | `x_decomp/device.mojo` DevExec.omp_rows -> `x_decomp/apple_fast.mojo` omp_block_kernel | sparse-coder | one threadgroup (128 threads) per row: a thread per atom for the residual correlations, a fold on (value, index) for the pick (ties to the LOWER atom, the scan's pick), the Cholesky row and the two solves on thread 0 in threadgroup memory. Replaces `omp_rows_kernel`: one thread per row with a k*k + 3k float scratch strip in a device buffer of n times that (1.7 GB at 100,000 rows x 64 atoms). Bound: k <= 128, nnz <= 32, else the old kernel. |
| `MOJOLEARN_DECOMP_FAST_LASSO_BLOCK` | define (Mojo) | `x_decomp/device.mojo` DevExec.lasso_rows -> `lasso_block_kernel` | dict-learning, mb-dict-learning (fit and lasso_cd transform), sparse-pca, mb-sparse-pca | one simdgroup per row: G, w, q, h in threadgroup memory; every thread computes the coordinate update, thread l applies it to h[l]. Replaces `lasso_rows_kernel` / `lasso_row` (x_decomp/cells.mojo:539): one thread per row, each coordinate a k-long read-modify-write of the row's h strip in global memory, rows 4k bytes apart. Bound: k <= 32. |
| `MOJOLEARN_DECOMP_FAST_SMALL_EIGH_J2` | define (Mojo) | `x_decomp/device.mojo` DevExec.eigh | fastica (and every kit eigh of order <= 64: MinCovDet's pinvh at taxi's p = 11 when it runs on the device executor) | an eigh of order <= 64 takes `_eigh2` (one launch of the cyclic jacobi2 kernel, one readback) instead of `_eigh_par`: n - 1 rounds of two launches per sweep and a host readback of the off-diagonal norm before each sweep (FastICA's 8 x 8 symmetric decorrelation, every one of up to 200 iterations: ~130 launches and ~16 syncs for 28 rotations). |
| `MOJOLEARN_DECOMP_FAST_DICT_UPDATE` | define (Python, read from the binding) | `python/mojolearn/_expansion_decomp.py` `_update_dict` -> `_update_dict_resident` | dict-learning, mb-dict-learning, sparse-pca, mb-sparse-pca | the atom loop on resident matrices: row j's update as a masked row of B^T - A D (one GEMM, a one-hot column), divided by A[j, j], added, row j alone renormalized through a select on the one-hot; 11 launches an atom, no sync. Replaces the loop that downloads every row of D (`_vstack(*rows)`), a column of B and a row of A per atom (~20 syncs an atom; 16 atoms x 100 iterations for dict-learning, 16 x 3,910 steps for mb-dict-learning). Falls back to the loop for an unused atom (A[j,j] <= 1e-6, a host Philox resample) and for positive_dict. |

Dropped at the merge, main's replacement does what they did: MOJOLEARN_DECOMP_FAST_RP_DIRECT (main's
`_RandomProjection._project`, MOJOLEARN_XD_RP_TILED default on: X uploaded from its own buffer, a tiled
`x_decomp_dev_project`, one download), MOJOLEARN_DECOMP_FAST_MDS_DIAG (main's MDS adds the Guttman diagonal on the
device by default, `k.diag_mask` + fma, lane hr2-mds-agglo) and MOJOLEARN_DECOMP_FAST_ISOMAP_KNN (main's Isomap builds
the kNN graph on the device, `_graph` / `graph_knn`, and `_knn_lists` selects beside D on the device).

Cause summary, per lane (the whole FAST fit path read; `_RES_MIN` = 1, so every kit `ew`/`mm`/`rowsum` is a
resident launch and every `.s`, `.copy()`, `.T`, `.rows/.cols`, `_vstack` is a download and a sync):

- sparse-coder: `omp_row` one thread per row, scratch n x (k*k + 3k) floats in device memory (fixed above).
- dict-learning / mb-dict-learning: `_update_dict` ~20 syncs an atom (fixed); `lasso_row` one thread per row
  (fixed); `_sparse_encode` uploads G, Q and W through the host address path (3 uploads, 2 downloads a call:
  stays); `_cost` 2 scalar readbacks an iteration (stays, the stop rule); mb-dict-learning's per-step
  convergence readback `k.total(sqdiff(D, old)).s[0]` (stays).
- sparse-pca / mb-sparse-pca: dictionary learning on X^T (220 "samples" of 20,000 / 100,000 values, nc = 8):
  the same `_update_dict` syncs on 100,000-wide rows (fixed); `_thin_svd` of the 220 x n input through
  `k.svd(X.T)` (QR + one-sided Jacobi of the n x 220, stays); the final `est._encode` is one lasso_rows call.
- fastica: per iteration ~14 resident launches, one `lim` readback (stays: the stop rule), and the 8 x 8
  `_sym_decorrelation` eigh at ~130 launches + ~16 syncs through `_eigh_par` (fixed by the small-eigh switch).
- mds: `B.copy()` + Python diagonal per iteration (fixed on main: `diag_mask` + fma); `stress` and `ssd` scalar readbacks (stay).
- isomap: kNN in Python heapq over the downloaded n x n (fixed on main: the graph built on the device); `Wg` built on the host and `dijkstra_arcs`
  (x_decomp/cells.mojo:765) a serial O(n^2) host pass (~0.2 s at 10,000; stays); `dijkstra_row` one thread
  per source with a binary heap in global memory (stays, see below); `_fix_components` host walk over the
  adjacency lists (cheap); `_lanczos_top` (see classical-mds).
- classical-mds: `sqdist` + `_center_kernel` resident; `_lanczos_top`: every Lanczos step rebuilds the basis
  `_M(QT[:], j + 1, n)` on the host and uploads it twice (j x n floats), a `.s` download of q, and each
  restart runs `k.eigh` on the j x j tridiagonal T (j up to 600) through `_eigh_par`; see below.
- gaussian-rp / sparse-rp: fit draws kc x d on the device (tiny); transform = `from_input` copy (fixed on main: `_project`) +
  upload + one GEMM (MAX matmul under FAST: the tier family's `MOJOLEARN_APPLE_FAST_GEMM_NT_TILED` covers
  it) + download of n x 10.
- min-cov-det / elliptic-envelope: `x_decomp/mcd.mojo` fast_mcd drives 333 subsets x 10 trials x 2 C-steps
  (taxi, n = 100,000) and then 3,330 merge C-steps on 1,500 rows, each C-step 12-40 Kit calls that run on
  the HOST executor below `MOJOLEARN_XD_RES_DEV_MIN` (65,536 elements: all of taxi's p = 11 subset work) and
  as synchronous device round trips above it (Istella p = 220: 300 x 220 > 65,536, every one of the ~100k
  calls a sync; the lane has no Istella result). Nothing cheap here is GPU-only and parallel: the fix is the
  grid of C-steps below.

Not done (shape and cost estimate), for the local session or a next lane:

1. MCD C-steps as a grid of blocks (`x_decomp/mcd.mojo` select_random / select_init): one block per
   (subset, trial): the Philox perm of 300 keys as an in-block bitonic sort, the h smallest by (dist, index)
   as an in-block select, mean and p x p covariance in threadgroup memory (p <= 64; p = 220 needs 193 KB,
   so cov to a per-block global strip), the log det by an in-block LU, pinvh by an in-block Jacobi eigh,
   Mahalanobis a thread per row; 2 iterations. 3,330 blocks for the subset phase, 3,330 for the merge
   (1,500 rows, <= 30 iterations, det-decrease stop per block). ~600 lines of new Mojo; expected taxi 4.0 s ->
   well under 1 s (the serial host executor is the whole cost), Istella from no result to seconds
   (bounded by 3,330 one-block 220 x 220 Jacobi eighs per phase, ~0.1 s each spread over the GPU).
2. Isomap all-pairs shortest paths: `dijkstra_row` is 10,000 threads each a serial heap over 10,000 nodes
   and ~20 arcs each (divergent, latency bound; dist and pos n x n = 2 x 400 MB). Floyd-Warshall as a grid
   per k is n^3 = 1e12 cell updates at 10,000 rows: slower. A frontier (delta-stepping) SSSP, one block per
   source with the frontier in threadgroup memory, is the GPU-shaped form: ~300 lines; cost unmeasured
   (the kNN heapq was the larger host step and is fixed).
3. mb-dict-learning / mb-sparse-pca native step: after the switch above a step is still ~200 resident
   launches from Python (3,910 steps on Istella). A Mojo step (sparse code block, A/B EWA update, the atom
   loop, the convergence norm) as 3-4 launches per step: ~250 lines in x_decomp; expected most of the
   remaining gap to scikit-learn's 5.0 s.
4. classical-mds / isomap Lanczos (`_lanczos_top`): keep the basis resident (append through a resident copy
   instead of `QT.extend(q.s)` + `_M(QT[:])`), 2 uploads of j x n floats a step (sum over j <= 600 at
   n = 5,000: ~7 GB moved when the restart goes to 600); the T eigh of order j through `_eigh_par`
   (j - 1 rounds x 2 launches x sweeps). Python-only change, ~60 lines; cost depends on how many restarts
   the kernel needs (converges at 20-40 vectors on a clean centred kernel).
5. The host-address entries `lasso_rows` / `omp_rows` (x_decomp/api.mojo) upload G, Q, W and download W
   each call: a resident form (`dev_lasso_rows`, as `dev_lda_rows` is) saves 3 uploads + 2 downloads per
   `_sparse_encode` (3,910 calls for mb-dict-learning). ~40 lines of binding + Python.
6. gaussian-rp / sparse-rp: on main the transform is `x_decomp_dev_project` (X uploaded from its own buffer,
   C = X W^T by 64 x 16 tiles through shared memory, one download): upload (880 MB at 1M x 220) + the tiled
   product + download, all on the device; nothing of this lane is left in it. Whether that closes the M3 gap
   (gaussian-rp taxi 3.7x, Istella 2.4x on the 0834 board, measured before main's kernel landed) is a board
   re-run of main, not an A/B of this branch.

Keep rule: a switch becomes the FAST default when its arm is faster on the M3 and held-out quality (the
lane's board quality column) stays within FAST's run-to-run spread; then the env read goes and the arm
is the code.

Risky compile sites (no toolchain here): x_decomp/api.mojo `fast_defines_py` (nested `comptime if` on
`is_defined[...]()`, `String +=`), x_decomp/device.mojo `comptime XD_FAST_APPLE` (`has_apple_gpu_accelerator`, the tier
branch's idiom), x_decomp/apple_fast.mojo `omp_block_kernel` / `lasso_block_kernel` (SHARED `stack_allocation`
strips, the `ns`-bounded in-block Cholesky on thread 0) and `_eigh2_on`, and the `_eigh2` route's single readback.
Request lines: docs/apple-fast/ab/decomp-sparse.txt (light form: one dataset per change, afc_ab_def.sh, binding x_decomp).
