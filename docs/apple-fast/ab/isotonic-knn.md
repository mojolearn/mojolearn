# lane/apple-fast-isotonic-knn: isotonic parallel PAVA; knn-imputer / LLE through fast_mma_knn (PLAN.md item 8)

Written without a Mojo toolchain (cloud peer); the first M3 build is the compile check.
Every switch defaults OFF and is compiled under FAST + Apple only; IDENTICAL compiles the old code.

| switch | kind | site | what it changes under FAST on Apple |
|---|---|---|---|
| `MOJOLEARN_ISOTONIC_FAST_PAR=1` | env, read at dispatch | `bindings/_mojolearn_x_linear.mojo` fit_binding -> `x_linear/isotonic_fast.mojo` isotonic_fast | the isotonic fit on the grid: rank-merge sort of (x, y, row) keys, pools of equal x by scan, part sums, PAVA on 256-point blocks then log2 boundary-merge rounds, trim by scan; falls back to the team fit when two unequal x sit under the serial 1e-6 pooling rule |
| `MOJOLEARN_X_NEIGHBORS_FAST_MMA_ROUTE=1` | env | `x_neighbors/iter_device.mojo` op_knn_sq_tiled | the fused k-NN (`knn_sq_tiled`, used by LOF, LabelPropagation/Spreading and LLE under the switch below) through `neighbors/impl/detail/fast_mma_knn.mojo` when `fast_mma_knn_applies(d, k + exclude_self)`; same k, same (distance, index) order, the query's own row dropped by `knn_mma_finish_kernel`, squared distances by the expanded form clamped at 0 |
| `MOJOLEARN_LLE_FAST_KNN=1` | env, Python, FAST tier only | `python/mojolearn/_expansion_decomp.py` `_knn_lists` (LLE's call passes `device_ok=True`) | LLE's neighbour lists from the x_neighbors binding's `xn_knn_sq_tiled` instead of the n x n squared-distance download plus `heapq.nsmallest` over every row in Python |

Causes:
- isotonic (12-16x): `x_linear/isotonic.mojo:129` `if not t.lead(): return`: the sort of a million rows (`_iso_sort`, :93, one device thread), `_make_unique` (:157), PAVA (:190) and the trim (:236) all on one thread of one block.
- lle (taxi 14x, Istella 6.3x): `_expansion_decomp.py` `_knn_lists`: `k.sqdist` (n x n, device) then a Python `heapq.nsmallest` per row (1e8 key calls at 10,000 rows). The kernels (barycenter, the deflated LU and subspace iteration) are already grid work; the Python selection is the serial step.
- knn-imputer (taxi 44x): `x_neighbors/iter_device.mojo` `knn_impute_split_kernel` (:897), one missing cell per 8 threads, each scanning every fit row with the per-pair nan_euclidean mask (donors are per column: the k nearest among fit rows with that column present). fast_mma_knn cannot take it: the distance depends on the pair's presence mask and the candidate set on the column, so no switch is written; the two baselines measure the FAST/IDENTICAL gap at head. A per-column MMA formulation (masked augmented vectors, four dot products per pair, a top-k per column) would need its own kernel in neighbors/ (not this lane's file).

Keep rule: a switch becomes the FAST default when its arm is faster on the M3 and held-out quality stays within FAST's run-to-run spread; then the env read goes and the arm is the code.
`ik-lle-mma-*` needs both LLE switches (the MMA route only matters once the lists come from the device). `ik-lof-mma-*` measures the route on a lane that already takes `knn_sq_tiled`.
Compile risks to watch: `x_linear/isotonic_fast.mojo` (new: UInt64 key buffers, comptime-parameter scan kernels, `Tuple[Int, Int]` helpers) and the function-scope import of `neighbors.impl.detail.fast_mma_knn` into the x_neighbors binding (first time that package is compiled into it).
