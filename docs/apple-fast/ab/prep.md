# lane/apple-fast-prep: minmax-scaler, onehot, ordinal (merged with origin/main 2026-10-02)

Written without a Mojo toolchain; the first M3 build is the compile check. Every switch is
default OFF; IDENTICAL compiles main's code unchanged.

| switch | kind | files | what it changes under FAST on Apple |
|---|---|---|---|
| `-D MOJOLEARN_PREP_FAST_MINMAX` (`PREP_FAST_MINMAX`, FAST + Apple + define) | define, binding `preprocessing` | `preprocessing/minmax.mojo` extrema_rows_fast_kernel / minmax_fit_fast; `preprocessing/estimator.mojo` minmax_fit_direct, minmax_transform_direct; `bindings/_mojolearn_preprocessing.mojo` transform_direct_binding (registered only under the define); `python/mojolearn/preprocessing.py` MinMaxScaler._transform | fit: main's direct fit (X up from its own buffer, device finite scan) with the extrema by a coalesced row-tiled kernel (block = chunk x column group, a simdgroup reads consecutive words of one row) instead of one column per block; same (chunk, column) partials, same finalize, same words. transform: `minmax_transform_direct` (X, scale, min up from their addresses, the output scanned on the device, 0 = nonfinite output) instead of the List route's three host passes over n*d plus Python's `all_finite` walks of input and output |
| `MOJOLEARN_X_PREP_FAST_UNIQUE=1` | env read once at import, Python, FAST only | `_expansion_prep.py` _fit_categories (OneHot / Ordinal fit) | per column the labels' chunked run scan (uniq_count / uniq_scan / uniq_write, x_prep/labels.mojo) instead of `unique_cols` (one thread per column over every sorted row); same words |
| `MOJOLEARN_X_PREP_FAST_NONEG=1` | env read once at import, Python, FAST only | `_expansion_prep.py` _codes, OrdinalEncoder.transform, OneHotEncoder.transform | the `count_neg` stage (one thread per column over every row) left out when no caller reads its count: OneHot handle_unknown 'ignore' / 'infrequent_if_exist' with drop=None, Ordinal 'use_encoded_value' (the board's settings) |

Dropped at the merge: `MOJOLEARN_X_PREP_FAST_MULTILABEL` (main's lane gap-prep2 `_mlb_flat` / `row_ones`
already takes MultiLabelBinarizer's int labels through the device) and the old env MINMAX arms
(main's `minmax_fit_direct` already uploads from the caller's buffer and scans on the device; the
remaining FAST delta is the define above).

Causes (board 0834): minmax-scaler Istella 5.7x: `preprocessing/minmax.mojo` extrema_chunks_kernel
reads one column per block (256 words d apart, a cache line each) and the transform's List route
(`bindings/_mojolearn_preprocessing.mojo` transform_binding `load`, `preprocessing/estimator.mojo`
minmax_transform_host_into finite_values + upload_f32). onehot / ordinal taxi 3.6-3.8x:
`x_prep/prims.mojo` unique_cols_unit and count_neg_unit, one thread per column over 1M rows.

Risky compile sites: `preprocessing/minmax.mojo` `comptime if not PREP_FAST_MINMAX: return ...`
inside minmax_fit_fast (two comptime branches each returning), the 2-D `grid_dim=(chunks, cgroups)`
with runtime `block_dim=tpb`; `bindings/_mojolearn_preprocessing.mojo` `comptime if` inside PyInit's
try block; `preprocessing/estimator.mojo` minmax_transform_direct's comptime branches.
