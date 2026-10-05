# sym-feat profile: a symmetric (oblivious) fit outside the per-level histogram and estimation work

Lane `lane/apple-fast-sym-feat` (binding `gbdt`, board lanes `gbdt-symmetric` and `gbdt-symmetric-1000`, datasets taxi
1,000,000 x 11 and istella 1,000,000 x 220). Read from the code at origin/main 8897404da; nothing here was timed. The
board arm (`bench/speed/forest_speed_arm.py::our_gbdt_arm`) fits with `bootstrap_type='No'`, no `eval_set`, the
cfg's `random_strength`, and scores `predict` on the held-out rows through the resident model
(`gbdt_resident_prepare` once per model, `gbdt_resident_predict` per call). `_ours_X` is C-order float32 unless
`MOJOLEARN_SPEED_FORTRAN` is set, so `GradientBoosting.fit` first materializes a column-major copy.

## Inside the fit clock, before the first tree

| step | where | launches | waits | host work and copies |
|---|---|---|---|---|
| column-major copy of X | `python/mojolearn/ensemble.py:1922` `as_f32_colmajor` | 0 | 0 | one transposing copy of the whole matrix (istella 880 MB, taxi 44 MB), pure host time |
| y, weights, flags | `gbdt/estimator.mojo:gbdt_fit` | 0 | 0 | memcpy of y; X is borrowed (DEVIATION 2550) |
| float borders | `gbdt/grid_creator/gls_borders_device.mojo::device_float_borders` | per chunk of `min(n_float, 2^25 / n_rows)` columns (33 at 1M rows, so 7 chunks on istella, 1 on taxi): `width` column uploads straight from the caller's pointer, 2 fills, keys kernel, the segmented radix sort (32 bits, several launches per pass), budget kernel, columns kernel, 3 readbacks | 1 per chunk | the border lists are rebuilt on the host from the readback (small) |
| one-hot / CTR grids | `_quantize_training_columns` | 0 | 0 | none on the board (no categorical columns) |
| compressed index | `gbdt/train.mojo::_build_cindex_from_columns` | per bordered feature: column upload, border-slab upload, `binarize_float_feature_kernel` (220 launches + 440 uploads on istella) | 1 initial, then 1 per 8 features (27 on istella) and 1 final | per feature a `memcpy` of the whole column into a pinned slot (istella: 220 x 4 MB) plus the slab fill; the matrix goes host to device a SECOND time here |
| targets and weights | `train.mojo:1866` | 2 uploads | 1 | fill of two pinned buffers |
| held-out set | `train.mojo:2114` `_build_cindex_from_floats` over `t_rows = eval_rows if eval_rows > 0 else 1` | with NO eval set it still runs per bordered feature: 2 uploads + 1 launch over ONE row (220 launches, 440 uploads on istella), plus `make_test_arm`'s ~20 allocations and a fill | 1 initial, 1 per 8 features (27 on istella), 1 final, 1 after the target upload | per-feature host NaN scan of one value |
| bootstrap seeds | `gbdt/gpu_util/kernel/bootstrap.mojo::create_bootstrap_seeds` | 1 upload (only when a bootstrap is on; the board's is 'No') | 1 | 65,536-step splitmix64 loop on the host |

## Per iteration, outside histograms and leaf estimation

- bootstrap weights (`doc_parallel_boosting.mojo:2473`, when on): ONE `bootstrap_kernel` launch per tree, the per-thread
  seeds advanced in place on the device; plus one `deterministic_sum_lanes_kernel[2]` when fixed-point magnitudes are
  needed. No host step. The board runs with it off.
- score noise (`random_score_helper.mojo::compute_std_dev`): `std_dev_partials_kernel` + sum kernel + readback + ONE
  synchronize per tree, but only for `use_pointwise_searcher=True` (not the board's default); the doc-parallel symmetric
  searcher's std dev is af-sym-hist's.
- eval set (`_apply_last_tree_to_test` + `_test_loss`, only with an eval set): per tree SIX small uploads (off, shift,
  mask, bin, eq, leaf values) + 1 apply launch + the loss launch (`launch_approximate`) + 1 sum launch + 1 readback + ONE
  synchronize, then `detector.add_error` on the host. With `od_type='None'` the detector never stops, yet the loss is
  still read back and waited for every tree.

## After the last tree

- `model_text` (host string build) in `gbdt_fit`; Python parses the `bias` line.
- predict on the board's held-out rows (`gbdt/resident_model.mojo`): `gbdt_resident_prepare` parses the text, packs the
  ensemble, uploads the border slabs and the split records once (6 uploads, 1 drain). Each `gbdt_resident_predict`:
  1 upload of the raw rows (row-major, no host transpose), `resident_stage_kernel`, then ONE `binarize_float_feature_kernel`
  PER BORDERED FEATURE (220 launches on istella), then under FAST ONE `compute_bins_and_add_kernel` PER TREE (500 or 1000
  launches, each re-reading the cursor and the cindex words), `resident_link_kernel`, 2 readbacks, 1 drain.

## Where the candidates aim

1. The matrix crosses the host-device boundary twice (border build, cindex build) with 220 pinned memcpys and 28 drains in
   between, and on C-order input it is transposed on the host first. QUANT_DEVICE uploads it once and binarizes from the
   device copy; the row-major door skips the host transpose and transposes on the device.
2. 220 binarize launches become one pack launch (INDEX_PACK_DEVICE).
3. The empty held-out arm costs 220 launches, 440 uploads and 29 drains for nothing (EVAL_SKIP_EMPTY).
4. With an eval set: 6 uploads become 1, and the per-tree readback + drain goes away when no detector can stop the fit
   (EVAL_FUSED).
5. Predict: 220 + 1000 launches become 2 (PREDICT_PACKED).
6. Bootstrap seed fill: host loop + upload + drain become one launch, same seed values (BOOT_DEVICE).
