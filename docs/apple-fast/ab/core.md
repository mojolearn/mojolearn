# lane/apple-fast-core: the glm switch (ols under FAST on Apple)

Written without a Mojo toolchain (cloud peer); the first M3 build is the compile check
(binding estimators). The four finished switches of this branch (dbscan scan/cc-batch, kde
slices, knn k64, kmeans x3) already have their A/Bs queued by the manager: `core.txt` carries
only the new glm switch, taxi first (ols: 4,000,000 x 11 train rows; istella 400,000 x 220
only after taxi wins).

| switch | lanes | site | what it changes under FAST on Apple |
|---|---|---|---|
| `-D MOJOLEARN_OLS_FAST_DEVICE_CENTER=1` | ols | `glm/estimator.mojo` (`OLS_FAST_DEVICE_CENTER`, `ols_center_tsqr_r_host`, `ols_fit_centered_host`, `ols_fast_column_means`, `ols_center_pack_kernel`); `bindings/_mojolearn_estimators.mojo` (`ols_center_tsqr_r`, `ols_fit_centered`, registered under the flag only); `python/mojolearn/linear_model.py` (`LinearRegression.fit` FAST branch, `_fast_device_center`, `_ols_fast_tsqr_r`) | `LinearRegression.fit` with an intercept made three device trips before the solve (`lm_col_sums`: X up, sums down; `lm_center`: X and means up, centered copy down; the solve's upload of that copy), with a host `[v / rows ...]` and `.tolist()` between each. Now the raw X and y go up once; the column means are a chunked float32 fold on the grid (512-row chunks, groups of 64), the centering is one thread per word straight into the TSQR's packed `[X - mu | y - mu_y]`, and `ts_factor_device` (main's default solver) returns the small R for the same SVD solve `_ols_tsqr` does. Off the TSQR (`MOJOLEARN_LINALG_TSQR=0` or cols + 1 > 512) the data is centered in place and solved by `ols_fit_traced` as `ols_fit_host`. FAST bits move: float32 means instead of `lm_col_sums`'s exact sums rounded once (last-place differences in the centered words) |

Switch detection: the binding registers `ols_center_tsqr_r` only when built FAST on Apple with
the define, so the Python layer's `_fast_device_center` is a `getattr` on the loaded binding,
no env read (the WIP's `MOJOLEARN_OLS_FAST_DEVICE_CENTER=1` env switch is gone).

Risky compile sites (no toolchain here):
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
