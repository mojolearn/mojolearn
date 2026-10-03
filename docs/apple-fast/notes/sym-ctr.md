# sym-ctr: the categorical (CTR) pipeline on symmetric trees, Apple FAST. Read-profile, 2026-10-03

Lane af-sym-ctr, branch lane/apple-fast-sym-ctr, cut from main 8897404da (0.8.35). Board lane gbdt-categorical on
taxicat: taxi rows with columns 0, 3, 4, 5, 6 declared categorical (`tools/speed_gbdt_arm.py` TAXI_CAT_COLUMNS); the
two zone columns (~260 categories each) are above `one_hot_max_size` and take CTRs, the other three take one-hot.
Logloss, SymmetricTree, `permutation_count` 4 (the CatBoost GPU default once a CTR-bearing feature exists), 500 trees.
GPU default `simple_ctr` is 3 Borders priors + 1 FeatureFreq, so each CTR feature becomes 4 columns; the three
Borders columns are PERMUTATION DEPENDENT and are computed once per permutation.

Everything below is counted from the code, not measured (this lane never measures). N = rows, P = 4 permutations,
C = 2 CTR features, H = 3 one-hot features.

## What is already on main (from lane/apple-fast-trees-depthwise, 2026-10-02) and NOT redone here

| define | default | what |
|---|---|---|
| `MOJOLEARN_GBDT_CTR_FAST_SCAN` | OFF (opt-in) | phase 2 of the u32 scan and the segmented scan on one 256-thread block instead of one thread |
| `MOJOLEARN_GBDT_CTR_FAST_FREQ` | OFF (opt-in) | the permutation-independent FeatureFreq column through the device calcer instead of the host builder + calcer |
| `MOJOLEARN_GBDT_CTR_PERM_PTRS` | OFF (opt-in) | each permutation's cindex from column pointers instead of a host flat pack of n_columns x N floats |
| `MOJOLEARN_GBDT_SEG_SUMS_BLOCK` | OFF (opt-in) | block scan for the segmented sort sums |
| `CTR_PERM_BATCH` | ON (`_OFF` to disable) | the NON-symmetric per-tree loop over permutations batched behind one drain per walker round |

Note on the brief's item 2 ("reuse binarized CTR borders across iterations"): there is nothing to reuse. Simple
CTRs are computed ONCE before boosting, their grids are built once per fit (Borders columns take permutation 0's
Uniform-15 grid, FeatureFreq its own MinEntropy-15 grid), and tree CTRs (`max_ctr_complexity > 1`) are refused, so
no CTR value or border is recomputed per iteration. Item 2 is replaced below by moving the border inputs to the device.

## Per fit, the CTR prep (`gbdt/train.mojo` `train`, the `cat_features` walk)

Per declared categorical column (C + H = 5):
- host: memcpy of the raw column (N), `dense_category_code` loop (N, branchy float checks), `seen` denseness check.

Per CTR feature (C = 2), permutation-INDEPENDENT half, default (`CTR_FAST_FREQ` off):
- `compute_simple_ctrs` host: builder init (3 lists of N), current bins (N), gather (N), counting sort (3 passes of N),
  borders mask (N), calcer: segment ids (N), offsets (N), bin weights (N), column write (N). About 11 host passes over N.
With `CTR_FAST_FREQ`: `compute_simple_ctrs_device`: builder ctor (order upload: host loop N + copy + drain; 8 N-buffers
allocated; 2 fills), `add_cat_feature_bins` (compute_current_bins 5 launches; codes upload: host loop N + copy + drain;
gather 1; radix sort 9 bits x 4 launches + 2 copies; update mask 1), calcer ctor (5 N-buffers; weights upload: host
loop N + drain), visit (extract 1, scan 3, FULL readback of `bins` to read ONE element + drain, 2 small allocs,
partition offsets 3, gather 1, segmented reduce 1, freq kernel 1, column readback N + drain, host copy N).
About 60 launches, 4 drains, 3 N-uploads, 2 N-readbacks, 15 N-sized allocations per feature.

Per CTR feature x permutation (C x P = 8), permutation-DEPENDENT half (`compute_simple_ctrs_gpu`, device):
- `TCtrBinBuilderGpu(order)`: host loop N validating and staging the order, upload, 2 fills, drain; 8 N-buffers.
- `add_cat_feature_bins`: compute_current_bins (extract 1 + scan 3 + scatter 1) on a fresh order whose flags are all
  clear (the result is the zero fill it overwrites); codes upload (host loop N + copy + drain); gather 1; radix sort
  over `int_log2(unique_values)` = 9 bits: 36 launches + 2 copies; update mask 1.
- `THistoryBasedCtrCalcerGpu(builder)`: 10 N-buffers; reset = gather_trivial 1 + segmented scan (fill 1, 3 phases,
  scatter 1).
- `set_binarized_sample`: host loop N staging the SAME target bytes every call, upload, drain.
- `visit_cat_feature_ctr`: gather u8 1, fill stats 1, segmented scan 5, then per prior (3): make_means 1, readback N
  f32 + drain, host copy N.
About 62 launches, 6 drains, 3 N-uploads (order, codes, target: all re-staged per call although the order is shared
by both features and the codes and the target by all four permutations), 3 N-readbacks, 18 N-sized allocations.
For the board lane: ~500 launches, ~48 drains, 24 N-uploads, 24 N-readbacks, ~144 N-sized device allocations
(each ~16 MB at 4.1M rows), plus ~50 host passes over N for staging and column copies.

Per CTR feature: `build_ctr_tables` (apply tables): 2 host passes over N (counts, and the code x target class
histogram).

`_quantize_training_columns`:
- per one-hot column (H = 3): a host max loop over N (`maxc`);
- per Borders column (3 x C = 6): `compute_ctr_borders` Uniform over `dep_by_perm[0][..]`: a host min/max pass over N
  (then 15 Float64 borders);
- per FeatureFreq column (C = 2): MinEntropy-15: host copy N, host radix sort N, dedup N, DP over the unique values.

Cindex build, once per permutation (P = 4), n_columns = 14 raw + 8 CTR columns = 22 (taxi has 16 features, 2
replaced by 4 columns each):
- default (`CTR_PERM_PTRS` off): host flat pack of 22 x N floats per permutation (`flat.append`), then
  `_build_cindex_from_floats` (per column: NaN scan N, memcpy N, upload, binarize launch; one drain per 8 columns).
  Four permutations: 88 x N host appends, 88 N-uploads, 88 launches, ~12 drains, 4 cindex allocations of 22 x N u32 /
  packed.
- `CTR_PERM_PTRS`: pointers instead of the pack; same uploads and launches.

## Per tree, the boosting loop (`gbdt/methods/doc_parallel_boosting.mojo`), SYMMETRIC path

- structure search on the learn permutation's cindex (same work as the plain symmetric lane);
- estimation, `if need_estimation and not non_symmetric:` at :3026: a SERIAL host loop over the P = 4 permutations.
  For p == learn_p: `_estimate_and_apply` on the searcher's partition. For each other p: `enqueue_create_buffer`
  of N u32 (`d_bins`), `compute_bins_for_model` 1 launch, `partition_from_bins` (a device sort + a bounds readback
  + drain + its own allocations), `_estimate_and_apply` (gathers, the Newton walk with one drain per evaluation,
  the apply). Per tree: 4 estimation tasks each with >= 2 drains, 3 N-sized allocations, 3 partition readbacks.
  Over 500 trees: >= 4000 drains and ~1500 N-sized allocations that a plain symmetric fit (P = 1) does not pay.
- `CTR_PERM_BATCH` (main, default on) batches exactly this loop for the NON-symmetric policies only (:2689 is inside
  the `non_symmetric` branch). The symmetric loop is not batched.

The plain symmetric fit on the same rows is 11.25 s and the categorical fit 37.1 s (M3 Ultra FAST board). The CTR
prep above is a few seconds of launches, drains, allocations and host passes at most; the three extra permutations'
serial estimation tasks per tree are the only term that scales with the 500 trees, so they are the likely bulk of the
26 s. Both are addressed below, each behind its own define.

## Candidates (each FAST + Apple only, default OFF; `MOJOLEARN_SYM_CTR_ALL` turns on every one)

1. `MOJOLEARN_SYM_CTR_PERM_BATCH` (doc_parallel_boosting.mojo, the symmetric estimation block): the four
   permutations' bins + partitions enqueued back to back behind ONE drain, the four Newton walks in lock step
   through `_estimate_prepare` / `estimate_advance` / `_estimate_complete` (the batched path `CTR_PERM_BATCH` already
   uses for non-symmetric trees), partitioners and estimation workspaces held across trees. Per tree: (2 + walk
   rounds) drains instead of 4 x (2 + rounds); no per-tree N-sized allocation. Same kernels, same partitions, same
   leaf values (each permutation's walk is unchanged; only the host interleaving changes).
2. `MOJOLEARN_CTR_PREP_SHARED` (ctrs/fast_prep.mojo, train.mojo): one device prep context per fit: the binarized
   target uploaded once, each permutation's order uploaded once, the builder and calcer scratch allocated once and
   reused by every (feature, permutation); the fresh-order `compute_current_bins` (whose result is the zero fill)
   skipped; the `bins` readback for `ReadLast` replaced by `unique_values` (dense codes: every code is present, so
   the segment count is known). 24 N-uploads -> 1 + P + C, ~48 drains -> ~P + 2C, ~144 N-allocations -> ~20.
3. `MOJOLEARN_CTR_SORT_ONCE`: the FeatureFreq column derived from permutation 0's Borders builder (the rows are
   already sorted by category; segment lengths do not depend on the within-category order): segment ids, partition
   offsets, segmented reduce of the trivial weights, freq kernel. No identity-order builder, no second radix sort,
   no host sort + host calcer. Integer counts, so the column is bit-identical to both existing arms.
4. `MOJOLEARN_CTR_INDEX_FUSED`: CTR columns stay on the device. The calcer's prior divide writes each column into its
   own device buffer (no readback, no host copy, no re-upload per permutation); the per-permutation cindex binarizes
   those buffers in place (`_build_cindex_fast`); the Borders grid takes its min/max from a device stripe reduction
   (the 15 Float64 Uniform borders are computed from the same two floats, so the grid is bit-identical). The
   FeatureFreq column is read back once for its host MinEntropy grid (that sort stays host side; see "owed").
   Removes 24 N-readbacks and 32 N-uploads (8 CTR columns x 4 permutations), and the host flat pack / pointer pass for
   every column of every permutation (the non-CTR columns use the pointer path's uploads as `CTR_PERM_PTRS` does).
5. `MOJOLEARN_CTR_ONEHOT_DEVICE`: the dense-code pass on the device for every declared categorical column: one
   upload of the raw column, a strided stripe kernel that validates (finite, non-negative, integer, below 2^32),
   writes the u32 codes and reduces the max and the error flags per thread (4096 slots read back, not N); a
   histogram kernel (u32 atomics, exact) over the codes and the target class that gives the denseness check, the
   one-hot `maxc` and the apply-time CTR tables. Removes the per-column host code loop, the per-one-hot-column host
   max loop and the two host table passes per CTR feature; the codes buffer feeds the builder directly.
6. `MOJOLEARN_SYM_CTR_ALL`: all of the above.

Owed, not done here: the FeatureFreq grid (MinEntropy-15) still sorts the N-row column on the host because
`_exact_best_split` dedups inside; a weighted-uniques entry (the FeatureFreq column has at most `unique_values`
distinct values, with known multiplicities) would remove that sort. It is a refactor of `grid_creator/binarization.mojo`,
outside a FAST guard, so it is left for a lane that owns that file.
