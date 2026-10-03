# lane/apple-fast-neighbors2: the neighbors lanes not taken by other families (lof, radius-neighbors, label-propagation, label-spreading, ocsvm, svgp, louvain, pagerank, additive-chi2, skewed-chi2, poly-count-sketch)

Written without a Mojo toolchain (cloud peer); the first M3 build is the compile check. Every switch
defaults OFF and is compiled under FAST + Apple only; IDENTICAL compiles main's code unchanged. Bindings:
`build_x_neighbors.sh` for every switch but the radius one, which is in the `_mojolearn` binding
(`bindings/build.sh`, `base`). `x_neighbors/gen.py` was re-run (two custom ops: `lp_iterate_knn`,
`kernel_tiled`; the FAST_ALT table; the generated files are committed).

Merged with origin/main (hr2 / hr-graph lanes) on 2026-10-02, main winning:
- pagerank: main's `pr_iterate_gpu` (x_neighbors/graph_par.mojo) already folds the dangling mass and the
  stopping sum in blocks on the device, which is what `MOJOLEARN_PAGERANK_FAST_REDUCE` did: switch and lines dropped.
- the fused k-NN through fast_mma_knn: this branch's `_knn_sq_fast_mma` / `MOJOLEARN_XN_FAST_MMA_KNN` dropped;
  the route lives on lane/apple-fast-isotonic-knn as `-D MOJOLEARN_XN_FAST_MMA_ROUTE=1` (its `ik-lof-mma-taxi`
  line measures it on lof; the two branches merge cleanly, `git merge-tree` checked).
- svgp: the env switch and the `svgp_gpu` op became the define below; main's `op_svgp` is the staged device
  items (no longer HOST_RUN), so the A/B is FAST vs FAST.
- new device code uses `enqueue_create_buffer` / `enqueue_copy` directly (the pre-push hook refuses `_buf` /
  `_down`, which stage through host threads) and no one-block launch over a runtime size (the svgp folds are
  SVF_FOLD_BLOCKS partials then one block over that fixed count).

| switch | kind | site | lanes | what it changes under FAST |
|---|---|---|---|---|
| `MOJOLEARN_LP_FAST_RESIDENT=1` | env, Python (FAST tier) | `python/mojolearn/_expansion_neighbors.py` `_LabelPropagationBase.fit`; `iter_device.mojo` `op_lp_iterate_knn`; CPU column `iter_host.mojo` | label-propagation, label-spreading | the kernel='knn' fit loop as one resident op: cols/vals uploaded once, device stopping sum and flag, LPK_BATCH = 16 iterations per drain, the finite-x product item |
| `MOJOLEARN_XN_FAST_TILED_RBF=1` | env, Python (FAST tier) | `_expansion_neighbors.py` `_XNeighbors._kernel`; `iter_device.mojo` `op_kernel_tiled` / `kernel_rbf_tiled_kernel` | ocsvm (any rbf `_kernel` caller when set) | the rbf kernel matrix from 16 x 16 tiles with the x and y rows staged in threadgroup memory (d <= 224); other kinds and wider rows are `kernel` |
| `-D MOJOLEARN_SVGP_FAST_GPU=1` | build define (x_neighbors binding, tools/afc_ab_def.sh) | `x_neighbors/gen.py` FAST_ALT -> the generated `op_svgp` (`device_ops.mojo`) runs `x_neighbors/svgp_fast.mojo` `svgp_solve_device` under FAST + Apple + the define | svgp | the m x m solve through `potrf_lower` / `cho_solve` / `chol_logdet` of cholesky/checks/ for the three factorizations and the column solves, this lane's elementwise and matmul kernels, five scalars read back for the bound |
| `MOJOLEARN_RADIUS_FAST_REUSE_COUNT=1` | env | `neighbors/estimator.mojo` `radius_neighbors_fill` (`_rbc_index_only`) | radius-neighbors | the fill pass takes the count pass's row offsets (the caller's `indptr`) and total instead of re-running the eps query in counting mode; the index is still rebuilt |
| `-D MOJOLEARN_XN_PCS_SPARSE=1` | build define (existing opt-in, lane neighbors-apple3) | `iter_device.mojo` `op_pcs_resident` -> `x_neighbors/pcs_sparse.mojo` | poly-count-sketch | the convolution over the running product's nonzero components only |

## Causes (file:line at lane/apple-fast's head f5f61bde; pagerank and the kNN route: see the merge note above)

- lof / label-propagation / label-spreading kNN: `x_neighbors/iter_device.mojo:684` `knn_sq_tiled_kernel` is one
  thread per query row over scalar fmas against a 64-row shared tile, and d > 64 (Istella 220) falls to
  `device_ops.mojo:409` `knn_sq_kernel` (one thread per row, y streamed from device memory). x_neighbors never
  imported `fast_mma_knn` (`fast_mma_knn_applies(220, 21)` is True under FAST with the default BIGD arm).
- pagerank: `iter_device.mojo:284` `op_pr_iterate_sparse`: per iteration `pr_dangling_sum_kernel` (grid 1,
  block 1, a serial chain over n = 20,000) and `absdiff_sum_kernel` (same), then `enqueue_copy` + `synchronize`.
- label-propagation / label-spreading loop: with kernel='knn' `_expansion_neighbors.py:1128-1150` runs the
  loop in Python (the resident `lp_iterate` serves the dense graph only): `absdiff_sum` is a HOST_RUN one-item
  fold (`gen.py` HOST_RUN), `lp_knn_product` (`iter_device.mojo:1284`) uploads cols, vals and the
  distributions and downloads the product, `lp_clamp` / `ls_clamp` upload and download again, every iteration
  (LabelPropagation max_iter 1000 at 200,000 rows).
- ocsvm: `x_neighbors/items.mojo:109` `kernel_item` one thread per cell, both rows read from device memory per
  cell: the 10,000 x 10,000 Gram at d = 220 re-reads every y row 10,000 times. The SMO itself is already one
  threadgroup (`block_ops.mojo:77`, below).
- svgp: `gen.py` HOST_RUN `svgp`: `items.mojo:1498` `svgp_item` on the host, `_chol_inplace` three times at
  m = 512 (serial), the column solves over the host pool; a host step between two device ops.
- radius-neighbors: `neighbors/estimator.mojo:1819` `radius_neighbors_fill` calls `_rbc_index_and_count`
  (index build + counting eps pass) before its own fill pass: per `radius_neighbors` call two index builds and
  three eps passes, the second count identical to the first.

## Not done (shape and estimate)

- louvain: `gen.py` HOST_RUN `louvain` -> `louvain_sparse.mojo:103` `louvain_item_sparse`, the whole method on
  one host thread with a pinned vertex order (DEVIATION 5204), plus `_csr_of_dense` over the 20,000 x 20,000
  dense matrix on the host. A parallel sweep (all nodes' best community from the current partition, then
  moves applied) changes the pinned order's answer, so it is a new FAST method, not a switch: estimate 300+ lines
  (gather per node over CSR, segmented best-gain, conflict rule, aggregation by sort/scan), 2-5x on the sweep.
- pagerank host CSR: `pr_sparse.mojo:85` `pr_graph_from_dense` scans the 1.6 GB dense adjacency on the host
  pool. A device scan needs the 1.6 GB upload first (about the same time as the host scan on unified memory), so
  nothing to win until the adjacency arrives as CSR (`connected_components` already takes one; PageRank's
  Python contract is the dense matrix).
- svgp `svgp_stats` (`iter_device.mojo:1184`): B = Kuf Kfu as `matmul_tn_acc_kernel`, one thread per m x m cell
  folding over 32,768-row tiles (a 512 x 512 x 100,000 GEMM written as a per-cell fold). Routing it through
  core/gemm.mojo's FAST gemm_tn would be the next item (est. 2-5x on that stage).
- ocsvm SMO: `block_ops.mojo:77` `ocsvm_smo_block`, one threadgroup of 256 for the whole dual (n = 10,000:
  40 samples per thread per scan, two scans and a gradient update per iteration, tens of thousands of
  iterations). A multi-block SMO needs a grid-wide barrier per iteration; estimate 200+ lines, 2-4x when
  n is large. `variance` (gamma='scale') is a HOST_RUN one-item fold over n x d (small at 10,000 x 220).
- label-propagation / label-spreading `lp_knn_graph` (`lp_knn.mojo:13`): the compact graph is built on the
  host in one serial pass over n x k (200,000 x 7, with a sort of 7 per row): a few ms; the spreading
  variant's column degrees would need a device scatter.
- lof `lof_lrd` / `lof_score` (`items.mojo:510, 525`): one thread per point over k = 20 neighbors; already
  parallel and small next to the kNN.
- radius-neighbors Python surface (`neighbors.py:1600-1640`): the ragged result is built row by row in Python
  (two Array slices per query, 20,000 queries); the ball-cover exclusive scan is one block over n_queries
  (`scan.mojo:41`, 20,000 ints: microseconds). The second index build (`rbc_build_index` in the fill) remains:
  caching the index across the count/fill calls is a binding-contract change.
- additive-chi2 / skewed-chi2: `achi2_item`, `skew_weights_item`, `skew_transform_item` are one thread per
  output cell (parallel); the transform runs on 1,000 rows; the host-side draws are 220 x 256 doubles. The
  fit's `X.min()` is the native `reduce_stat` helper. Nothing to move onto the device at this shape.
- poly-count-sketch: `op_pcs_resident` is already resident and block-per-row; only the existing
  `MOJOLEARN_XN_PCS_SPARSE` opt-in is queued.

## Compile risks to watch (no toolchain here)

- `x_neighbors/iter_device.mojo` now imports `neighbors.impl.detail.fast_mma_knn` (its kernels are
  compiled only inside `comptime if FAST_MMA_KNN_ENABLED`), `core.pinned_reduce`, and at function scope
  `x_neighbors.svgp_fast` (which imports `cholesky.checks.potrf` / `trsm` and `core.identity_trace`, as
  bindings/_mojolearn_gp.mojo's path does). If the svgp module fails to compile, `op_svgp_gpu` in
  iter_device.mojo and the `svgp_gpu` CUSTOM_OPS row in gen.py (re-run gen.py) are the two things to drop; the
  other switches do not depend on it.
- `knn_drop_self_kernel` takes a `MutPointer[UInt32, MutAnyOrigin]` (`UP`) for fast_mma_knn's index lists.
- The three new custom ops raise under IDENTICAL builds (their Python callers only run on the FAST tier).

## Keep rule

A switch becomes the FAST default when its arm is faster on the M3 and held-out quality stays within FAST's
run-to-run spread; then the switch goes and the arm is the code. Request lines: `docs/apple-fast/ab/neighbors2.txt`
(light form, `1 2`, one dataset per change first, no -ident lines): n2-lp-res-taxi, n2-ls-res-taxi,
n2-ocsvm-tiled-istella (d = 220 is the tiled rbf's case), n2-svgp-gpu-taxi, n2-radius-reuse-taxi, n2-pcs-sparse-taxi.
