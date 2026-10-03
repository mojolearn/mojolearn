# HDBSCAN on the Apple GPU (FAST): profile of one fit (lane af-hdbscan2, 2026-10-03)

Board shape (tools/classical_two_datasets.py): m = 100,000 rows, taxi d = 11, istella d = 220,
min_samples = 10 (k-NN at k = 11 including the point), min_cluster_size = 100, EOM, epsilon 0, alpha 1.
Read from main (8897404da) after lanes cgr2-hdbscan and cgr3-hdbscan-mst: every stage is on the device;
what follows is the launch, wait, readback, allocation and host-loop count per stage.

## 0. Input (hdbscan/estimator.mojo:137-144, bindings/_mojolearn_hdbscan.mojo:75-146)
- `hdbscan_fit_host_output` copies X into a host `List` (m x d host loop), 1 upload, 2 synchronize.
- The binding copies labels, core distances and probabilities out with host loops over n.

## 1. Core distances (hdbscan/impl/detail/reachability.mojo, neighbors/estimator.mojo:655 `knn_self_search_resident`)
- Allocations: knn_dists, knn_inds (m x k, single_linkage.mojo:197-198), idx_u32 (m x k), index and queries
  (two device COPIES of X, m x d each), index_norm, query_norm, dist_tile (1 cell on the MMA arm), buf_val,
  buf_idx, out_i32: 11 buffers.
- Launches: 2 copies, 2 norm kernels, synchronize; FAST Apple k-NN = `fast_mma_knn` (taxi: d <= 32 arm;
  istella: the chunked `bigd` arm, which reads `MOJOLEARN_KNN_FAST_MMA_BIGD` from the environment on the hot
  path), synchronize; `_knn_order_rows_device` (scan + block sort); `knn_inds_to_i32_kernel` + synchronize;
  `core_distances_kernel` + synchronize; `refuse_nonfinite_device` (count kernel, 1-word readback, 2 synchronize).
- Waits: ~6 synchronize. Readbacks: 1 word.
- The m x k index set and the sorted rows are never read by HDBSCAN: only the k-th distance of each row is.

## 2. Mutual reachability MST (hdbscan/impl/cluster/detail/single_linkage.mojo:177-260)
No mutual reachability matrix is formed on either arm; weights are computed inside the search.

### taxi (d <= 64, m > 4096): hierarchy/impl/cluster/detail/fast_boruvka.mojo:213 `fast_euclidean_mst`
- Allocations: comp_d, bd_d, bj_d, todo_d (m each), host buffers comp_h, bd_h, bj_h, todo_h, listb_h;
  `MmaBoruvka` (fast_mma_boruvka.mojo): xc (m x d), nc (m), part_d, part_j (>= 122,880 each); prepare = 2
  launches + 1-word readback + synchronize.
- Per round (rounds ~10 to 17): upload comp (m words); phase A search = 1 matrix-unit launch + 1 merge launch
  (or 1 scalar launch); when points were deferred: readback bd, bj (2 x m words), synchronize, host loops over
  the listed and deferred points, phase B search; readback bd, bj, synchronize; then SEVEN host passes over
  m (edge pick with a host union-find, component loop, bound reset, relabel with find, drop/arg, todo/defer
  lists, reset). End: host sort of the m - 1 edges, 3 host buffers, 3 uploads, synchronize.
- Per round: 2 synchronize, 4 x m-word readbacks, 1 m-word upload, ~8 host passes over m. The MST's host
  union-find and planning are the only host loops left in the fit.
- Output order: (weight, discovery order); build_mr_linkage then orients each edge (lo, hi). The dense and
  sparse arms emit (weight, lo, hi) instead, so on exactly equal weights (common in mutual reachability: a
  point's edges all read its core distance) the FAST taxi edge LIST order differs from IDENTICAL's.

### istella (d > 64, m > 46,340): hdbscan/impl/cluster/detail/sparse_mr_mst.mojo:1033 `sparse_mr_mst_device`
- Allocations: norms, xt (m x d transpose), key_d, j_d (m + 65,536 each), todo, fk, fj, 18 m-word state
  buffers, bcount, boff, st_d, st_h: ~28 buffers.
- Per round: reset, classify, drop_arg, arg_idx, assign (5), compact (3), status readback + synchronize;
  phase A search; ub_from_a, drop_b, compact (5), readback + synchronize; phase B search; cmin x3, winner,
  hook, 18 jumps, relabel (24); readback + synchronize. 3 synchronize per round plus the search's.
- THE SEARCH ON APPLE (sparse_mr_mst.mojo:165 `SMR_TILED` is NVIDIA/AMD only): `sparse_mr_search_kernel`,
  one thread per (point, j slice), every pair reads d words of xt (coalesced) and d words of x (broadcast)
  with NO reuse, under `SPARSE_MR_LAUNCH_MACS = 2^30` multiply-adds per launch and a synchronize after
  EVERY launch. Round 1 (n_todo = m): span = 2^30 / (100,000 x 220) = 48 columns per launch, so
  ~2,084 launches x (memset + search + fold + merge + synchronize) for the first phase alone. The tiled
  kernel (`sparse_mr_search_tiled_kernel`: 64 x 64 tiles staged 16 features at a time in threadgroup
  memory, 4 x 4 cells per thread) is compiled out on Apple.
- Output: (weight key, lo, hi) sorted and oriented, by rank + scatter (2 launches).

### Both: orient (1 launch), dendrogram (hierarchy/impl/cluster/detail/dendrogram_device.mojo:288)
- 8 buffers + a host flag; 17 levels (2^17 > 99,999), each: claim, own, then (memset, hook, jump, 1-word
  readback, synchronize) until no hook fires (typically 2 to 3 iterations), top, size, relabel; out,
  synchronize. ~17 x 6 launches, ~40 synchronize, each over a one-word flag.

## 3. Condense (hdbscan/impl/detail/tree_device.mojo:643 `build_condensed_device`)
- ~35 allocations (par, wdep, wpre, live, side, lam, check, par0, pb, db, qb, lb, off, rank, skeys, svals,
  lab, ptrc, ptrl, ekey, evals, echild, elam, esize, indptr, cpar, kids, csize, clam, t_*, cdep, cp_a,
  cp_b, cd_b, mx, plus the sort and scan scratch).
- Launches: refuse_nonfinite (count + readback), init, merge, 17 jump3 rounds, split flag, exclusive scan,
  compact, 2 radix sorts, gather depth, rank, label, 2 pointer-jumping finds (17 each), edges, radix sort,
  gather, exclusive scan, cluster init, path reduce (~log n_clusters), max, refuse_nonfinite: ~100 launches.
- Waits: ~10 synchronize; 5 scalar readbacks (n_vertices, n_bad, n_split, max_cdepth, the two counts).

## 4. Extract (hdbscan/impl/detail/extract.mojo:416, select.mojo, stabilities.mojo)
- compute_stabilities: 2 memsets, 2 launches, synchronize. excess_of_mass: init, (max_cdepth + 1) level
  launches, negate (init, ~log launches, apply, synchronize). count_selected: flag, scan, 1-word readback
  (synchronize). label_map, synchronize. do_labelling: 5 buffers, init, edges, find (~17), points,
  synchronize. remap. stability scores: 3 memsets, 3 launches, 1 download (synchronize). deaths, probs.
- Then SEVEN downloads (raw m, final m, is_cluster, stabilities, label_map, inverse, probs m) and one
  scalar read, EACH with its own host buffer, synchronize and host copy loop: 8 synchronize.

## 5. Outputs (hdbscan/impl/runner.mojo:331-369)
- `download_condensed`: 4 downloads (4 synchronize); core distances: 2 synchronize + host loop; seven
  `List.copy()`s into HDBSCANOutput; the binding's host loops over n.

## Totals (one fit)
- taxi: ~115 synchronize (6 k-NN, ~35 MST, ~40 dendrogram, ~12 condense, ~15 extract, ~6 outputs), ~2 x 17
  m-word readbacks, ~8 x 17 host passes over m, ~90 allocations, ~350 launches.
- istella: the MST's round 1 alone is ~2,100 synchronize (one per 48-column search launch) and ~8,400
  launches; later rounds add hundreds more. Everything else as taxi.
- Live buffers when the condense and extract launches run: ~50 (x, core, the 7 runner buffers, the tree's
  ~14, the stage's scratch), so each launch there carries ~12 us of per-buffer cost.

## Candidates (docs/apple-fast/ab/hdbscan2.md has the mechanism, effect and risk of each)
- MOJOLEARN_HDB_SMR_TILED: the tiled search kernel on Apple with a 2^34-MAC launch bound and a drain every
  8 launches (istella: ~2,084 drained launches per phase -> ~128 launches, 16 drains).
- MOJOLEARN_HDB_CORE_TILE: core distances from one tiled kernel with a register top-k (d <= 64, k <= 16),
  no k-NN index set, no X copies, no norms, no row sort (taxi).
- MOJOLEARN_HDB_DEV_BORUVKA: the d <= 64 arm's Boruvka rounds on the device (the sparse arm's planning,
  min-edge, hook and jump kernels) with hierarchy's search kernels; no host union-find, no m-word readbacks,
  no host sort; 3 status words per round (taxi).
- MOJOLEARN_HDB_ONE_SYNC: the extract's 8 and the runner's 6 output waits become 2.
- MOJOLEARN_HDB_LINKAGE_DEVICE: main already builds the linkage and the condense on the device (no host
  linkage loop; the MST edges arrive sorted, so no sort). The define removes what is left: the dendrogram's
  per-level (hook, jump, flag readback, synchronize) loop becomes ONE lock-free CAS union launch per level
  (~40 waits -> 0, same roots = component minimum, same integers), and the condense's eight waits become two
  status readbacks (delta refusal + MST check + n_split; max_cdepth + lambda refusal). Bits unchanged.
- MOJOLEARN_HDB_SELECT_DEVICE: main already runs stabilities, EOM / leaf, labels, scores and probabilities on
  the device. The define removes the waits between them and the selected-count and scores readbacks: one
  readback for the whole extract (labels, probabilities, the rest, and the count). EOM stays one launch per
  cluster-tree level: its ordered float sums forbid a scan or pointer-jumping form, and a single-launch
  last-arriver form needs a cross-threadgroup fence Metal does not give. Epsilon != 0 keeps main's route.
  Bits unchanged.
- MOJOLEARN_HDBSCAN2_ALL: all six.
- LINKAGE_DEVICE and SELECT_DEVICE: compile owed: peer (not compiled in this lane, Andrew 2026-10-03).
