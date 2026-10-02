# lane/apple-fast-cluster: clustering lanes under FAST on Apple (PLAN-classical.md worst rows, next-experiments 8)

Written without a Mojo toolchain (cloud peer); the first M3 build of `x_cluster` (`bindings/build_x_cluster.sh`,
FAST) is the compile check. Every switch is a build define (`-D MOJOLEARN_<NAME>=1`, `is_defined` at module scope)
compiled under FAST + Apple only (`GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()`) and
defaults OFF; no build reads the environment for it; IDENTICAL, the host binding and a FAST build without the
define compile main's code (the new `ClusterOps.minibatch_fast` method answers False on the host).

Risky compile sites (M3 build-errors first): `x_cluster/minibatch_fast.mojo` (`minibatch_fast_steps`: device
buffers for the centers, counts and the SplitMix64 state, `UPtr = MutPointer[UInt64, MutAnyOrigin]`, the
`enqueue_copy` host pointers), `x_cluster/meanshift_fast.mojo` (`meanshift_fast_grid`: the done flags read
between groups of MSG_GROUP shifts), and `minibatch_fit`'s `comptime if MINIBATCH_FAST_DEV` block that sets
`n_steps = 0` and re-uploads `c` to `cslot` for main's MB_DEVICE_STEP readback.

| switch | kind | site | what it changes under FAST on Apple |
|---|---|---|---|
| `-D MOJOLEARN_X_CLUSTER_FAST_MEANSHIFT=1` | define (`MEANSHIFT_FAST_GRID`), taken at dispatch in `x_cluster/device_ops.mojo` `DeviceOps.meanshift` | `x_cluster/meanshift_fast.mojo` | every shift of every seed on a grid of (seed, 256-row chunk) blocks (`_msg_part_kernel`: bandwidth test, per-feature fold of the rows within, the chunk's count) plus one block per seed folding the chunks and running the quotients, the shift and the stop test (`_msg_finish_kernel`); 16 shifts enqueued per read of the done flags |
| `-D MOJOLEARN_X_CLUSTER_FAST_MINIBATCH=1` | define (`MINIBATCH_FAST_DEV`), taken in `x_cluster/minibatch.mojo` `minibatch_fit` | `x_cluster/minibatch_fast.mojo` via `ClusterOps.minibatch_fast` (`x_cluster/ops.mojo`, `device_ops.mojo`, `host/host_ops.mojo`) | the step loop resident on the device: 32 steps' batch indices drawn on the host (the fit's stream, in step order) and uploaded at once; per step four launches with no synchronize (assign a thread per batch row, per-(center, chunk) feature sums, the `update_center_dense` cell update into the next history slot, sklearn's `_random_reassign` on one block); the group's batch inertias read once, `_mini_batch_convergence` on the host as the loop runs it, the stop step's centers and counts taken from the history. Unit weights and tol <= 0 only (the board's shape) |

## Causes

- meanshift Istella 13x: `x_cluster/device_ops.mojo:169` `_meanshift_team_kernel` (the Apple default, `MEANSHIFT_TEAM`)
  takes ONE block per seed and walks all n rows of every shift inside it; Istella 10k x 220 with bin seeding is 18 seeds,
  so 18 blocks run the fit. The `-D MOJOLEARN_MEANSHIFT_BLOCK=1` WIP opt-in (`device_ops.mojo:265`, `_meanshift_block_kernel`)
  is the same one-block-per-seed shape and takes d <= 16 only, so it never runs on Istella; it is left as it was, the
  define above is the FAST Apple A/B arm.
- minibatch-kmeans Istella 5.5x: `x_cluster/minibatch.mojo` `minibatch_fit`, `for step in range(n_steps)`: per step a host
  gather of the batch (`gather_rows`), an upload (`ops.set`), one `nearest` launch, a read-back of labels and distances
  (`ops.get_if`: a synchronize per step) and the k center updates folded on the host over the batch (`minibatch_step`);
  up to `max_iter * n / batch` = 2,441 such round trips on Istella.
- connected-components 12-13x: NOT CHANGED (x_neighbors/ is another family's). The board hands ours the CSR, so
  `x_neighbors/iter_device.mojo:370` `op_cc_iterate_csr` -> `_cc_csr_device` already hooks and jumps on the device, but
  pays per call a pinned host buffer (`enqueue_create_host_buffer` + synchronize), three host memcpys of the CSR and the
  labels, and a synchronize per hooking round for the one changed flag; then `_cc_relabel` in Python
  (`python/mojolearn/_expansion_neighbors.py:1519`) walks the n labels through a dict. The baselines below measure the gap
  at head (IDENTICAL vs FAST): if they are equal the cost is these host steps, not arithmetic. A fix would keep the flag on
  the device for a fixed number of rounds between reads (as the meanshift grid does) and relabel on the device.
- agglomerative (classical2, single linkage, 10,000 rows): NOT CHANGED. `hierarchy/impl/cluster/detail/single_linkage.mojo:88`
  takes the FAST Boruvka route (`fast_boruvka.mojo`, no dense matrix) only for `n <= 64` features; Istella (220 features)
  falls to the dense m x m `get_distance_graph` + `build_sorted_mst[DENSE]` (`mst.mojo`, a host `get_n_components` readback
  per round) + `build_dendrogram_host` (a host union-find over the m - 1 edges, `agglomerative.mojo:119`). Not among the
  PLAN's worst rows; the baselines put a number on it. A fix would lift the 64-feature cap by chunking the features in
  `fb_nearest_other_kernel`'s shared tile (hierarchy/ is editable; left for a run that shows the row matters).
- spectral (classical2, 10,000 rows, 10-NN): NOT CHANGED. `spectral/impl/preprocessing/detail/spectral_embedding.mojo:162`
  `create_connectivity_graph` builds the kNN graph on the device but symmetrizes, sorts and drops zeros on the host
  (`coo_symmetrize`, `coo_sort`, `coo_remove_scalar`); FAST on Apple already takes `create_laplacian_fast`
  (`fast_graph.mojo`) and the Lanczos `FAST_NCV` arms, so the remaining host step is small. Baselines only.

## Keep rule

A switch becomes the FAST default when its arm is faster on the M3 and the paired quality (meanshift: cluster count and
labels agree with the team kernel's up to the summation order; minibatch-kmeans: inertia / ARI within FAST's run-to-run
spread, the reassignment stream differs by design) holds; then the define goes and the arm is the code. Rows that are
"FAST slower than IDENTICAL" in the baselines are the next thing to look at, not these switches.

## Queue (docs/apple-fast/ab/cluster.txt, 10 lines, the light form: 1 2, no -ident lines)

meanshift and minibatch-kmeans: `tools/afc_ab_def.sh` (two FAST builds of `x_cluster`, "" vs the define) on Istella and
taxi, one tag per lane x dataset. connected-components, agglomerative, spectral: a FAST baseline per dataset, no switch.
