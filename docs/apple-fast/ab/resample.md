# lane/apple-fast-resample: bootstrap, permutation-test, resample, cross-val-score under FAST on Apple

Written without a Mojo toolchain (cloud peer); the first M3 build is the compile check
(binding `resample`: `bindings/build_resample.sh`; the cross-val switches are Python only).
Every switch is compiled under FAST + Apple only (`RESAMPLE_FAST_APPLE` in
`resample/fast_apple.mojo`; Python: `default_mode() == 'fast'`) and defaults OFF; IDENTICAL
compiles the old code and its bits never move. The parallel module's FAST refusal
(`parallel_model_selection.cross_val_score`) is untouched; the board's cross-val-score row runs
the serial `model_selection.cross_val_score`, which is where the two CV switches sit.

| switch | kind | site | what it changes under FAST on Apple |
|---|---|---|---|
| `MOJOLEARN_RESAMPLE_FAST_RANK_SORT=1` | env, host | `resample/estimator.mojo` bootstrap_host, bootstrap_unpaired_host | the sorted distribution by ONE rank launch (`fast_apple.mojo` rank_sort_f32_kernel, same total order `(twiddle_in(theta), r)` and the same bits at every rank) instead of `_sort_segments`: 32 one-bit radix passes x 4 launches (one of them `seg_scan_block_sums_kernel` on ONE thread) over a single segment of 9,999 keys |
| `MOJOLEARN_RESAMPLE_FAST_ONE_FOLD=1` | env, host | `resample/estimator.mojo` _bootstrap_theta | mean / diff_means replicates folded once per block (registers, then block.sum; `bootstrap_mean_fast_kernel`) instead of `_chunked_sum`'s virtual_block_sum per 256-draw chunk (79 block folds per replicate at n = 20,000); same draws |
| `MOJOLEARN_RESAMPLE_FAST_PERM_SELECT=1` | env, host | `resample/estimator.mojo` permutation_test_host | the null by a radix SELECT over the 64-bit key (`perm_select_fast_kernel`, 4-bit digits, 16 counters per thread, no atomics, keys recomputed from Philox, same permutation as the counting rank) with no `PERM_MAX_POOLED`; mean and diff_means |
| `MOJOLEARN_RESAMPLE_FAST_IDX_BULK=1` | env, host | `bindings/_mojolearn_resample.mojo` resample_indices_binding -> `resample_indices_fast_into` | the drawn indices copied out in one device-to-host copy and one memcpy instead of `_download_i32`'s per-element append plus the binding's per-element store (replace=True) |
| `MOJOLEARN_RESAMPLE_FAST_GATHER=1` | env, Python + host | `python/mojolearn/resample.py` resample -> new binding `resample_gather` -> `resample_gather_fast_host` | the draw and both row gathers on the device (`gather_rows_f32_kernel`, one thread per cell), one copy in and one out per array, instead of numpy's fancy-index gather of 1,000,000 rows on the host; float32 C-contiguous arrays, replace=True |
| `MOJOLEARN_CV_FAST_SLICE=1` | env, Python | `python/mojolearn/model_selection.py` cross_val_score -> `_kfold_slicer` | each unshuffled fold's test rows a zero-copy view and its training rows two memcpys (`_row_range_view`, `_rows_outside`) instead of four `_take_rows` per-row byte gathers per fold; any fold not KFold(shuffle=False)-shaped takes the gather |
| `MOJOLEARN_CV_FAST_TRUST_FOLDS=1` | env, Python | `python/mojolearn/model_selection.py` _prepare_folds | the native default folds (`fold_ids` + `select_fold_i64`, a partition by construction) skip `_indices` (range + duplicate pass) and `_overlap` on every fold's two int64 arrays; a splitter, groups or the sabotage control: as before |

## Causes (the whole FAST path of each lane, read)

bootstrap (`bootstrap_host`): the draws and the per-replicate folds are already one launch, one block
per replicate (`bootstrap_stat_kernel`, no host index draw, no per-replicate upload). What remains is
`_sort_segments` on the 9,999-float distribution: 130 launches, 32 of them a one-thread serial scan
(`core/segmented_sort.mojo:182` seg_scan_block_sums_kernel), which is launch latency on Metal (M3 16 ms
vs 2 ms on the L40S / MI325X at the same row); and 79 block folds per replicate in `_chunked_sum`
(`resample/checks/statistics.mojo:300`). Host steps left (not changed): the point estimate, the
interval, the standard error over `host_tree_sum` (O(n) and O(R) scalar work, both downloaded once).

permutation-test (`permutation_test_host`): one launch, one block per replicate, but `perm_stat_kernel`
(`statistics.mojo:733`) ranks by counting (O(N^2) per replicate, `PERM_MAX_POOLED = 1024` of
threadgroup keys), so `validate_pooled` REFUSES the board's 20,000 + 20,000 before any launch: the
row is REFUSED on every box. The select keeps the draws (`draw_permutation_key(key, r, j)` at the
same positions) and the same membership rule (the n_x smallest of the total order `(key, j)`).
Quality is `|p - scipy's p|` (Monte Carlo error, as the lane states). Host step left: the observed
statistic over `host_tree_sum` on the pooled sample (O(N)).

resample (`resample_indices_host` + `resample.py` `_take`): the draw is one launch; then
`_download_i32` (`estimator.mojo:213`) appends 1,000,000 indices one by one, the binding
(`_mojolearn_resample.mojo` resample_indices_binding) stores them one by one, and `_take` gathers
X (1,000,000 x d) and y with numpy on the host, which is the timed part.

cross-val-score (`model_selection.cross_val_score`): fold construction and validation are native
host passes (`_native_default_folds`, `_indices`, `_overlap`); each fold then gathers X and y twice
(`_take_rows`, per-row byte gather) and `LinearRegression.fit` (python/mojolearn/linear_model.py,
glm family, NOT this lane's) centers X on the host, uploads the fold's X, solves, and `predict`
uploads the test rows again. The upload-once design (X on the device, fold ids built on the device,
the fit run per fold from a device fold mask, scores reduced on the device and read once) needs a
glm entry (`ols_fit` with a device fold mask / sample mask, as `x_linear/enetcv_fast.mojo` does for
ENetCV) and is listed, not done: `LinearRegression` has no device fold-id or device-resident input.

## Not done (listed)

- cross-val-score: the per-fold re-upload of X, the host centering inside `LinearRegression.fit`
  (`_column_means`, `_center`), and `predict`'s upload of the test rows: all glm family.
- bootstrap: BCa's jackknife is already one launch; `point_estimate_host` and the interval stay on the
  host (O(n) / O(R) scalars). STAT_STD / PEARSON keep the chunked fold.
- permutation-test: STAT_STD keeps `perm_stat_kernel` and its bound; `permutation_type='samples'`
  is already one block per replicate with a per-pair coin.
- resample: replace=False (the keyed total order ranked on the host, `utils_first_by_key`) keeps
  the host rank; non-float32 or non-contiguous arrays keep numpy's gather.

Keep rule: a switch becomes the FAST default when its arm is faster on the M3 and held-out quality
stays within FAST's run-to-run spread (bootstrap: interval endpoints vs scipy; permutation-test:
|p - scipy's p|; resample: resampled column means vs the population; cross-val-score: fold R2 vs
scikit-learn's); then the env read goes and the arm is the code. For permutation-test the judge is a
number against REFUSED (no FAST or IDENTICAL baseline exists at the board size, so none is queued).
Compile watch: `fast_apple.mojo` uses `block_sum[block_size=256]` (as metrics/checks/pinned_sum.mojo),
a `SIMD[DType.int32, 16]` counter indexed at runtime, and 16 KB + 64 B of threadgroup memory per
block in the select; `estimator.mojo` imports `memcpy` and `f32_ptr`/`i32_ptr` from bindings/hostptr.
