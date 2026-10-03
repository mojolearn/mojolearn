# lane/apple-fast-trees-ensembles: the tools/bench_board_algos.py xlane="trees" ensemble lanes under FAST

Merged onto origin/main 2026-10-02 (eabeff395). Written without a Mojo toolchain (cloud peer); the first
M3 build is the compile check (binding x_trees only: two new exports `x_trees_fast_switches` and
`x_trees_device_folds`; no existing export changed, no other binding touched). Every switch is a BUILD-TIME define of the x_trees binding (`-D
MOJOLEARN_TE_<NAME>`), FAST + Apple build only, default OFF: `xtrees/api.mojo` folds the defines into
one comptime bit set (`XTREES_FAST_SWITCHES`, 0 unless `GLOBAL_NUMERIC_MODE == NUMERIC_FAST and
has_apple_gpu_accelerator()`), and `python/mojolearn/_expansion_trees.py` `_trees_switch` reads it
through the binding once per fit (no env read). IDENTICAL compiles main's code unchanged. Light A/Bs:
`afc_ab_def.sh` builds x_trees twice (arm A "", arm B the define), one dataset per change first.

| define | site | lanes | what it changes under FAST |
|---|---|---|---|
| `MOJOLEARN_TE_NATIVE_SPLITS` | `_trees_native_folds`, `_trees_splits(native=)`, `OneVsRestClassifier._indicator`, `MultiOutputClassifier.fit`; `xtrees/folds_device.mojo` (`x_trees_device_folds`) | stacking-clf, stacking-reg, calibrated, ovr, multioutput-clf | the cv=5 fold assignment and the ten fold row lists built ON THE DEVICE (class histogram and first rows by atomics, per-class flag-and-scan ranks, a row-per-thread fold assignment in sklearn's unshuffled StratifiedKFold / KFold law, per-fold flag-and-scan compaction, one download of the sizes and the lists) instead of by Python loops over every row; ovr's k one-vs-rest 0/1 targets through `x_trees_indicator_codes`; each label column of a float Y through `x_trees_column_f64` instead of `Y.tolist()` plus a per-column comprehension. Index bookkeeping only: same folds, same bits (digest must match) |
| `MOJOLEARN_TE_ADA_SESSION` | `_trees_ada_session_default`; `AdaBoostClassifier.fit`, `AdaBoostRegressor.fit` pass `default="1"` to `_trees_member_session` | adaboost-clf, adaboost-reg | the members' X staged on the device once (main's exact `ForestDataSession`, the default DART already uses) instead of per member (transposition, NaN scan, upload per member otherwise). Each member still draws its own quantile sample: same bits |
| `MOJOLEARN_TE_ADA_SESSION_SHARE` | with the above: `default="share"` | adaboost-clf | later members reuse the first member's quantile table: may move bits, keep only with held-out quality within FAST run-to-run spread |

## Dropped at the merge (main covers it)

- `MOJOLEARN_TREES_ENSEMBLES_BAG_SESSION` (bagging-clf, bagging-reg: one device X, per-member device
  gathers, `rf_classifier_fit_session_rows_export`): main's lane/hr2-gbdt-host (cc2743da9) fits a bag of
  best-splitter DecisionTree members on every feature as ONE batched forest fit on the device
  (`_BaggingBase._fit_batched`), on every tier, which is strictly less host work than the session loop.
  The new rf export and the Python session loop are gone; `bindings/_mojolearn_rf.mojo` is main's.
- The dart / dart-reg `MOJOLEARN_FOREST_SESSION=1` arms: main's lane/gap-nv-classical2 (e3372c826) opens
  the exact session by default for DART.
- The `-ident` baseline lines and the random-trees-embedding stage-time lines (no change on this branch
  for that lane; see below).

## Causes (the whole FAST fit path of each lane was read; `xtrees/ops.mojo` is a HOST binding: every glue op below runs on the CPU between the device fits)

- **adaboost-clf** (`_fit_members`, 50 rounds): per round `_w32` (host, n), the member's `_fit_weighted` on the SAME Xcm (staged per member unless the session is on), `est.predict(Xa)` which uploads the full row-major X AGAIN for the resident forest predict (`core/forest_inference.mojo` reads X row-major; the session holds it column-major, so a session-X predict needs a column-major predict kernel: owed, not done here), `x_trees_samme_step` (host, two serial passes over n), a 4-value host readback and `x_trees_scale` (host, n). **adaboost-reg**: the same plus `x_trees_weighted_sample` (host pool), the per-member row gather (device when the session is on, host otherwise) and `x_trees_r2_step` (host, serial). A device-resident weight plane updated by one launch per round is the next step (the glue binding has no device context today).
- **dart / dart-reg** (`_boost_loop`, 200 rounds): per round `x_trees_uniform` (host), the drop set on the host, `x_trees_gradients` (host, serial n), the member fit (now in main's default session), `x_trees_apply` of the new tree over all n rows (host pool), `x_trees_leaf_newton` (host, serial n), `x_trees_tree_score_add` (host, serial n) and one more `tree_score_add` per dropped tree twice. Score, g, h and the per-tree leaf ids live on the host for the whole fit: moving them resident with a (tree, row) apply launch and a per-leaf segmented reduction is a new device surface for `xtrees` (owed). Nothing queued here.
- **random-trees-embedding** (board median_ms = `fit(X)` only): `fit` is `x_trees_uniform` (host, serial n draws for the random target), `as_f32_c` of it, then `ExtraTreesRegressor(max_features=1, max_depth=5, n_estimators=10).fit` (`extratrees.py` `_fit_arrays`: ONE-thread `all_finite` over n x d, the row-major stage, the upload, then the ET builder's `compute_quantiles` and `et_bin_rows_kernel` over EVERY column, 220 on Istella, where the fit splits on at most 10 x 31 randomly drawn columns at depth 5). The likely fix (quantiles only for the drawn columns, or a sampled quantile pass) lives in the ET builder and `extratrees.py` (other families). Nothing changed, nothing queued.
- **stacking-clf / stacking-reg / calibrated** (`_fit_stack`, `CalibratedClassifierCV.fit`): `_trees_splits` built the folds with Python loops over every row (`_trees_stratified_folds`: a dict walk, a sort of n codes, k passes of n; `_trees_fold_rows`: two comprehensions of n per fold, ten at cv=5) and `Array.from_list` of each list: seconds at 1M rows; fixed behind NATIVE_SPLITS. Still host: `_gather` of each fold's X (host pool copy of 0.8 n x d per member per fold) and every sub-estimator uploading its own fold X: a shared device X across sub-estimators needs those families' fit entries to take a resident X (owed to them).
- **ovr**: the k 0/1 targets were list comprehensions over n (k = 4 taxi, 5 Istella): fixed behind NATIVE_SPLITS; each LogisticRegression clone then uploads X itself (owed to that family). **multioutput-clf**: `Y.tolist()` (n lists) and a per-column comprehension, then `encode_labels` of a Python list: fixed behind NATIVE_SPLITS for a float Y (the board's). **multioutput-reg / voting-clf / voting-reg**: the wrapper's own work is one host accumulate per member at predict time; nothing cheap at the wrapper level, no line queued.

## Keep rule

A define becomes the FAST default (define deleted) when its arm is faster on the M3 and the AFC-DEF digest matches arm A (NATIVE_SPLITS and ADA_SESSION only reorder where X lives or who builds the index lists; the model bits must not move). `MOJOLEARN_TE_ADA_SESSION_SHARE` may move bits: keep only with held-out quality within FAST run-to-run spread. Second datasets (taxi for the five NATIVE_SPLITS lanes, istella for the two AdaBoost lanes) only after the first wins.

## Compile risks for the local session to watch

- `xtrees/api.mojo` `XTREES_FAST_SWITCHES`: a comptime expression over `GLOBAL_NUMERIC_MODE`, `has_apple_gpu_accelerator()` and three `is_defined[...]()` (the form of `RF_FAST_BATCH` in `bindings/_mojolearn_rf.mojo`); `fast_switches_binding` is a zero-argument `def ... raises -> PythonObject` (the form of `mojolearn_numeric_mode_binding`). Both the GPU and the host x_trees binding call `register`, so both carry the export (0 on the host build).
- `xtrees/folds_device.mojo`: `prefix_sum[block_size=FOLD_SCAN_BLOCK, exclusive=True]` on Int32 (the form of gbdt/gpu_util/kernel/scan.mojo), `Atomic.fetch_add` / `Atomic.min` / `Atomic.max` on `MutPointer[Int32, MutAnyOrigin].unsafe_offset(i)` (the forms of hierarchy/impl), a `struct FoldScanWorkspace(Movable)` of DeviceBuffers built from `ctx`, `process_ctx["MojoXTreesPermContext"]()` (perm_device.mojo's slot), `comptime if TE_DEVICE_FOLDS: ... return` / `else: raise` in `device_folds`. Python reads the lists as `memory_at(...).cast("i")` views (`_buffer.memory_at`).
