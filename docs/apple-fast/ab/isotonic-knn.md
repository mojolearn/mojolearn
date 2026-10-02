# lane/apple-fast-isotonic-knn: isotonic parallel PAVA; LLE / LOF k-NN through fast_mma_knn (PLAN.md item 8)

Written without a Mojo toolchain (cloud peer); the first M3 build is the compile check.
Every switch defaults OFF and is compiled under FAST + Apple only; IDENTICAL compiles main's code unchanged.
Merged with origin/main (hr2 lanes) on 2026-10-02: main's LLE standard path (`_knn_mats` -> `graph_knn` cell ->
`barycenter` -> `graph_lle_iw`, I - W resident) and its `knn_sq_tiled2_kernel` win; the FAST changes sit on top.

| switch | kind | site | what it changes under FAST on Apple |
|---|---|---|---|
| `MOJOLEARN_ISOTONIC_FAST_PAR=1` | env, read at dispatch | `bindings/_mojolearn_x_linear.mojo` fit_binding -> `x_linear/isotonic_fast.mojo` isotonic_fast | the isotonic fit on the grid: rank-merge sort of (x, y, row) keys, pools of equal x by scan, part sums, PAVA on 256-point blocks then log2 boundary-merge rounds, trim by scan; falls back to the team fit when two unequal x sit under the serial 1e-6 pooling rule |
| `-D MOJOLEARN_XN_FAST_MMA_ROUTE=1` | define (x_neighbors binding build; was an env read until main dropped `getenv` from iter_device.mojo) | `x_neighbors/iter_device.mojo` op_knn_sq_tiled | the fused k-NN (`knn_sq_tiled`, used by LOF, LabelPropagation/Spreading and LLE under the switch below) through `neighbors/impl/detail/fast_mma_knn.mojo` when `fast_mma_knn_applies(d, k + exclude_self)` (d <= 32 without the bigd arm: taxi first); same k, same (distance, index) order, the query's own row dropped by `knn_mma_finish_kernel`, squared distances by the expanded form clamped at 0 |
| `MOJOLEARN_LLE_FAST_KNN=1` | env, Python, FAST tier only | `python/mojolearn/_expansion_decomp.py` `_knn_mats_device` (standard LLE) / `_knn_lists` with `device_ok=True` (hessian, modified, ltsa) | LLE's neighbour index matrix from the x_neighbors binding's `xn_knn_sq_tiled` (no n x n squared-distance matrix) instead of `k.sqdist` + the `graph_knn` cell; the int32 indices become the exact-float index matrix `barycenter` / `graph_lle_iw` take by one C-level array cast (no Python loop) |

Causes:
- isotonic (12-16x): `x_linear/isotonic.mojo:129` `if not t.lead(): return`: the sort of a million rows (`_iso_sort`, :93, one device thread), `_make_unique` (:157), PAVA (:190) and the trim (:236) all on one thread of one block.
- lle (taxi 14x, Istella 6.3x on the 0.8.34 board): main now selects on the device (`graph_knn` over the n x n `sqdist`); the FAST arm skips the n x n matrix (n^2 floats built, read and selected over) for the fused per-row k-NN.
- knn-imputer (taxi 44x): `knn_impute_split_kernel`, one missing cell per 8 threads scanning every fit row with the per-pair nan_euclidean mask. fast_mma_knn cannot take it (the distance depends on the pair's presence mask and the candidate set on the column), so no switch and no A/B line; a per-column MMA formulation would need its own kernel in neighbors/.

A/B lines (`isotonic-knn.txt`, light form: one alternation, 2 rounds, taxi first, no -ident lines):
`ik-iso-par-taxi`, `ik-lle-knn-taxi` (env switch, one build), `ik-lle-mma-taxi` (the define A/B with `MOJOLEARN_LLE_FAST_KNN=1` on both arms: the route only matters once the lists come from the device), `ik-lof-mma-taxi` (the route on a lane that already takes `knn_sq_tiled`). Istella lines follow a taxi win.
Keep rule: a switch becomes the FAST default when its arm is faster on the M3 and held-out quality stays within FAST's run-to-run spread; then the switch goes and the arm is the code.
Compile risks to watch: `x_linear/isotonic_fast.mojo` (new: UInt64 key buffers, comptime-parameter scan kernels, `Tuple[Int, Int]` helpers) and the function-scope import of `neighbors.impl.detail.fast_mma_knn` into the x_neighbors binding (first time that package is compiled into it).
