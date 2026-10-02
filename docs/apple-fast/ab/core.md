# lane/apple-fast-core: the core switches under FAST on Apple (ols, kmeans, knn, kde, dbscan)

Written without a Mojo toolchain (cloud peer); the first M3 build is the compile check
(bindings base and estimators). Every switch is a `-D MOJOLEARN_<NAME>` define read with
`is_defined` inside the file's FAST + Apple guard, default off; IDENTICAL compiles main's code
unchanged. Datasets: taxi (4,000,000 x 11 train rows for ols/kmeans), istella (400,000 x 220).

THE MANAGER'S EARLIER ENV-FORM A/Bs FOR THIS BRANCH ARE NOW NO-OPS: the dbscan scan/cc-batch,
kde slices, knn k64 and kmeans switches were `MOJOLEARN_*=1` env reads on the fit path (a host
step, per the GPU-only audit of 67258a0df); from this head they are defines, so an env-form
arm runs old FAST twice. The `afc_ab_def.sh` lines in `core.txt` replace those requests, one
dataset per switch first. The kmeans device-scale switch (`MOJOLEARN_KMEANS_FAST_DEVICE_SCALE`)
is DROPPED: main 2d7eade5b's `plan_sum_scale(ctx, x, ...)` already forms the scale on the device.

| switch | lanes | site | what it changes under FAST on Apple |
|---|---|---|---|
| `-D MOJOLEARN_OLS_FAST_DEVICE_CENTER=1` | ols | `glm/estimator.mojo` (`OLS_FAST_DEVICE_CENTER`, `ols_center_tsqr_r_host`, `ols_fit_centered_host`, `ols_fast_column_means`, `ols_center_pack_kernel`); `bindings/_mojolearn_estimators.mojo` (`ols_center_tsqr_r`, `ols_fit_centered`, registered under the flag only); `python/mojolearn/linear_model.py` (`LinearRegression.fit` FAST branch, `_fast_device_center`, `_ols_fast_tsqr_r`) | `LinearRegression.fit` with an intercept made three device trips before the solve (`lm_col_sums`: X up, sums down; `lm_center`: X and means up, centered copy down; the solve's upload of that copy), with a host `[v / rows ...]` and `.tolist()` between each. Now the raw X and y go up once; the column means are a chunked float32 fold on the grid (512-row chunks, groups of 64), the centering is one thread per word straight into the TSQR's packed `[X - mu | y - mu_y]`, and `ts_factor_device` (main's default solver) returns the small R for the same SVD solve `_ols_tsqr` does. Off the TSQR (`MOJOLEARN_LINALG_TSQR=0` or cols + 1 > 512) the data is centered in place and solved by `ols_fit_traced` as `ols_fit_host`. FAST bits move: float32 means instead of `lm_col_sums`'s exact sums rounded once (last-place differences in the centered words) |
| `-D MOJOLEARN_KMEANS_FAST_ROWNORM=1` | kmeans | `cluster/impl/detail/kmeans_fast.mojo` (`kmeans_fast_rownorm_on`, `launch_fast_row_sqnorm`); call sites `cluster/estimator.mojo` (two), `cluster/impl/detail/kmeans.mojo` | row squared norms one thread per row instead of one block per row (d <= `KMEANS_FAST_ROWNORM_MAX_D`). FAST bits may move (fold order of the row sum) |
| `-D MOJOLEARN_KMEANS_FAST_SKIP_PREDICT=1` | kmeans | `kmeans_fast.mojo` (`kmeans_fast_skip_predict_on`); `cluster/estimator.mojo` fit_predict | the fit's last assignment already wrote the labels against the final centroids, so `fit_predict`'s second assignment and the `x_norm` pass feeding it are skipped (n_init == 1 or an array init). Same kernel, same inputs: no bit |
| `-D MOJOLEARN_KNN_FAST_MMA_K64=1` | knn | `neighbors/impl/detail/fast_mma_knn.mojo` (`_k64_on`) | admits 32 < k <= 64 (the board's k = 64) to the matrix-unit arm with K = 64 instantiations instead of the generic tiled arm. Same sorted (distance, index) insertions: no bit expected |
| `-D MOJOLEARN_KDE_FAST_SLICES=1` | kde | `kde/impl/neighbors/kernel_density.mojo` (`kde_fast_slices_on`) | the fused score pass sliced over the train rows, more threadgroups per query block. FAST bits may move (slice fold order) |
| `-D MOJOLEARN_DBSCAN_FAST_SCAN=1` | dbscan | `neighbors/impl/ball_cover/scan.mojo` (`rbc_fast_scan_on`) | the CSR offset scan device-wide (block partials and a fold) instead of the one-block scan. Same offsets: no bit |
| `-D MOJOLEARN_DBSCAN_FAST_CC_BATCH=1` | dbscan | `dbscan/impl/sparse/detail/csr.mojo` (`weak_cc_fast_batch_on`, `WEAK_CC_FAST_BATCH`) | `WEAK_CC_FAST_BATCH` label passes per changed-flag readback instead of one. Same fixed point: no bit |

Switch detection: the binding registers `ols_center_tsqr_r` only when built FAST on Apple with
the define, so the Python layer's `_fast_device_center` is a `getattr` on the loaded binding,
no env read (the WIP's `MOJOLEARN_OLS_FAST_DEVICE_CENTER=1` env switch is gone).

Risky compile sites (no toolchain here):
- The define branches are `comptime if not (<FAST + Apple guard> and is_defined["..."]()): return
  False else: return <Bool>` in `kmeans_fast.mojo`, `csr.mojo`, `scan.mojo`, `kernel_density.mojo`
  and `fast_mma_knn.mojo`; the `std.os.getenv` imports those files added are gone (the
  `getenv` left in `fast_mma_knn.mojo` and `kmeans.mojo` is main's own stage timing).
- `glm/estimator.mojo`: `from x_decomp.tsqr_device import ts_factor_device` into the glm package
  (x_decomp imports nothing from glm; the estimators build compiles from the repo root);
  `ts_factor_device(ctx, packed, n_rows, n, _OFP(unsafe_from_address=r_addr), False)` takes a
  `mut DeviceBuffer` and a `MutPointer[Float32, MutAnyOrigin]` host pointer, built from the
  address as `x_decomp/api.mojo::_f` does, so no origin cast is needed on the direct call.
  Kernel launches pass bare `buf.unsafe_ptr()` through `enqueue_function`, the idiom
  `glm/impl/ols.mojo` and `glm/impl/center_device.mojo` compile with.
- `comptime if OLS_FAST_DEVICE_CENTER:` around two `m.def_function` lines inside the module
  init of `bindings/_mojolearn_estimators.mojo` (the file had no `comptime if` in that body
  before).
- `comptime OLS_FAST_DEVICE_CENTER = (GLOBAL_NUMERIC_MODE == NUMERIC_FAST and
  has_apple_gpu_accelerator() and is_defined[...]())`: a parenthesised multi-line comptime
  expression at module scope.
