# CTR / categorical recon (2026-08-21), the last feature block

Read from CatBoost `54a8143a`: `catboost/cuda/ctrs/*`,
`ctrs/kernel/ctr_calcers.cu` (429 lines, the kernels are SMALL; the
system around them is the work), `gpu_data/*` builders, and the options
defaults. This file is the warm-start for the port session.

STATUS, re-audited against `gbdt/` on 2026-08-20. **Read this before
planning anything below: this file has twice described work that already
existed, and once described a primitive as missing that was half built.**

BUILT:
* One-hot, end to end: the `build_layout` flag, the scan skip, the `==`
  predicate in `split_and_make_sequence`, the evaluator's `takeEqual`, and
  the `train.mojo` surface. Gated analytically in
  `mojo_only/one_hot_check.mojo`.
* Host-side FeatureFreq (`tools/ctr_prep.py`, 21c70ce) and a categorical
  CatBoost arm (`tools/catboost_arm.py`, `bench/interleaved/ctr_quality.mojo`).
* **MinEntropy border selection** (`gbdt/grid_creator/binarization.
  best_split_min_entropy`, 74ec1b0), their `TExactBinarizer<MinEntropy>` in
  mode `E_RLM2`, bit-exact to CatBoost over ten budgets
  (`pixi run check-minentropy`).
* A DEVICE-WIDE UNSEGMENTED exclusive scan, three-phase decoupled, in
  `gbdt/gpu_util/kernel/reorder_one_bit.mojo`
  (`block_scan_flags_kernel` / `scan_block_sums_kernel` /
  `add_block_carry_kernel`), built on `block.prefix_sum`.
* A ONE-BIT STABLE REORDER in the same file (`launch_reorder_one_bit`),
  ported from their `SortWithoutCub`.
* A leaf-segmented stable partition (count-chunks / scan-chunks / place) in
  `methods/greedy_subsets_searcher/kernel/split_points.mojo`, and a
  segmented REDUCE in `gpu_util/partitions_reduce.mojo`.

NOT BUILT, verified by search and not by memory:
* **Segmented SCAN.** The device-wide scan above is unsegmented; the
  split-points one is a purpose-built partition, not a reusable primitive.
  What is needed is the flag-in-bit-31 variant, where the carry resets at a
  segment start. That is a modification of the existing three-phase pattern.
* **Radix sort.** `launch_reorder_one_bit` is the one-bit building block and
  `FAST_SORT_SIZE = 500000` is already there; the multi-bit LSD driver that
  loops it over key bits does not exist.
* **Any CTR calcer.** There is no `gbdt/ctrs/`. `options/catboost_options.mojo`
  carries only REFUSALS ("CTRs are not ported", `:509`), plus a note at `:260`
  that `feature_weights` will diverge "the day CTRs land".
* **Target binarization.** Nothing. No `binarized_target`, no `target_border`
  anywhere in `gbdt/`.
* **CTR TABLES IN A MODEL FILE.** The numeric half of this landed
  2026-08-20: `gbdt/models/model_text.mojo` saves and loads a `TrainedModel`
  (ensemble, leaf values in leaf order, borders, fold counts, one-hot flags,
  losses) as plain text with every float carrying its IEEE-754 bits, gated
  bit-exact end to end by `pixi run check-model-io`. What is still missing is
  the CTR half: the per-categorical-feature category-to-value mapping an
  applied model needs. The format reserves `ctr_table`, `ctr_entry` and
  `ctr_borders` records and a `type` token on the `feature` record for
  exactly that, and the seam is written down in that file's THE CTR SEAM
  block. Nothing there is built.
* **A categorical path through `train()`.** It takes `one_hot: List[Bool]`
  and nothing else; the FeatureFreq calcer lives in the Python prep script,
  so categoricals work in the BENCHMARK and not in the library.

## What their GPU learner actually does with a cat feature

1. HASH -> PERFECT HASH -> BINS (host, `libs/data`): raw category ->
   ui32 hash -> dense bin ids per feature. We receive datasets pre-binned
   through prep, so v1 can take dense cat bins from the prep script the
   same way numeric bins arrive today.
2. ONE-HOT for cardinality <= `one_hot_max_size` (GPU default 2): the
   histogram/split kernels ALREADY carry `one_hot_feature` (predicate
   `bin == value`; threaded through layout, scan skip, split kernels,
   and the evaluator's XorMask slot). This is mostly dataset plumbing.
3. SIMPLE CTRs for the rest. GPU defaults: `Borders` (binarized-target,
   permutation-DEPENDENT) + `FeatureFreq` (permutation-INDEPENDENT).
   The CTR VALUES are then float features, binarized and fed to the
   numeric histogram pipeline unchanged. The CTR binarizer is NOT the
   GreedyLogSum we ported for numeric features: it is UNIFORM with 15
   borders (`private/libs/options/cat_feature_options.cpp:169`), at
   `min + (max - min) * k / 16` for k = 1..15, duplicates dropped. The
   FeatureFreq value itself is `(bin_count + prior) / (n_rows +
   prior_observations)`, default prior {0.0, 1}, so `count / (n + 1)`
   (`cuda/ctrs/kernel/ctr_calcers.cu:100`, defaults at
   `cat_feature_options.cpp:127-129`). Both are implemented in
   `tools/ctr_prep.py`.
4. TREE CTRs (feature combinations), DEFERRED; their own machinery
   (`tree_ctr_datasets_visitor`), not v1.
5. MODEL/INFERENCE: applied models need the final CTR tables. DEFERRED
   with tree CTRs; v1 gates on train-side quality only.

## The two calcers, by difficulty

* `FeatureFreq` (`TWeightedBinFreqCalcer`,
  `WeightedBinFreqCtrsImpl`): ctr = binWeight/total from bins SORTED by
  value, a segmented count, no history. EASY once the sort exists.
* `Borders`/history CTRs (`THistoryBasedCtrCalcer`): rows ordered by
  (bin, permutation position); running sums of preceding same-bin
  binarized-target stats via `SegmentedScanAndScatterNonNegativeVector`
  (segment flag packed in the float mask bit), then
  `(sum + prior) / (count + priorDenom)` (`MakeMeansImpl`), scattered
  back. Needs the segmented scan and the sorted order.

## The two vendor-lib decision points, CHECKED 2026-08-20

The VENDOR RULE says call MAX where they call CUB. **The check was run and
the answer is that neither exists**, so both are hand-written, and that is
recorded rather than assumed:

* MAX has **no device sort** and **no device-wide scan**. `algorithm` is a
  row-wise reduction library, `graph.ops.cumsum` is a graph-level Python op
  not callable from a Mojo kernel, and `nn.cumsum` ships CPU-only. None of
  the 14 packages expose either.
* MAX **does** expose `max.gpu.primitives.block.prefix_sum[exclusive=True]`,
  the counterpart of `cub::BlockScan::ExclusiveSum`, and it is portable.
  `reorder_one_bit.mojo` already builds the device-wide three-phase
  decoupled scan on top of it.

So **the sort is no longer the long pole**. That sentence predated
`reorder_one_bit` being ported. An LSD radix sort is a driver looping their
one-bit stable reorder over key bits, and the segmented scan is the existing
three-phase scan with a carry that resets on a flag. Both are composition.

**Both must be GPU-agnostic from the first line** (standing rule): no
wavefront-width assumption, 32 on NVIDIA and Apple against 64 on AMD, and
shared-memory sizing from a queried budget rather than from this laptop.
RAFT's `select_radix` has zero warp intrinsics, which is the precedent that
this is achievable in one source.

## Gates that exist to be reused

* Oracle: `tools/catboost_oracle.py` can emit a fixture WITH cat
  features and simple CTRs pinned (CPU CatBoost). CAUTION, the
  mojotrees trap from memory: CPU and GPU CatBoost may binarize
  targets/priors differently; verify their GPU defaults against the CPU
  oracle on a TINY fixture before trusting split-level gating. If they
  diverge (like bootstrap), fall back to the stochastic-style gate set:
  exact per-step device checks against host tallies + holdout quality.
* CONSTRUCTED fixtures, not real datasets. Correctness gates on (a) an
  analytic answer written down in advance (`mojo_only/one_hot_check.mojo`
  is the model: the right answer is arithmetic, no fixture needed) and
  (b) CatBoost's OWN output as oracle, via `save_model(format='json')`,
  which exposes the CTR tables, config, and CTR borders. A fixture we
  build beats a real dataset here, because we plant the adversarial
  cardinalities on purpose (1, 2, 3, 255, 256, thousands) and scattered
  values, where a real dataset has whatever it happens to have and we
  are hoping it covers the breaking case. See PORTING_RULES.md:229.
* THERE IS NO CATBOOST FeatureFreq ORACLE ON THIS BOX. `simple_ctr=
  'FeatureFreq'` "is not implemented on CPU yet"
  (`catboost_options.cpp:509`, verified in the shipped binary), and
  their GPU arm does not run on Apple. The CPU spelling of the same
  information is `Counter`, a different normalization of the same
  counts, so a CPU CatBoost arm is the same INFORMATION CLASS, not the
  same bits: quality bands compare, values do not. Any claim that our
  FeatureFreq is bit-exact to theirs rests on reading their source, not
  on a local oracle, and must say so.
* Real datasets are for QUALITY and SPEED reporting, a SEPARATE purpose
  from correctness. Rules when that reporting happens: fix the dataset
  list BEFORE seeing the numbers; every dataset run goes in the table
  with its attribution; never pick, drop, or defer a dataset by whether
  it flatters us, because "run it but report it only once we win" is
  the same defect as tuning on the test set. Match the arms on
  INFORMATION before comparing them, and say what the match excludes.
* Amazon (33k rows) and Adult (49k) are fine for QUALITY, which is what
  `ctr_quality.mojo` uses them for. They are NOT informative for SPEED:
  at that size we sit far inside the launch-floor regime, so a speed row
  from them measures Metal's launch price and not the CTR work. Report
  it that way or do not run it.

## Suggested port order (each step gated before the next)

1. Prep + layout: cat bins into the compressed index; one-hot for
   card <= 2 end to end; oracle fixture with one-hot only. (Their
   dispatch: one-hot features never get CTRs.)
2. DONE (21c70ce), host-side FeatureFreq in `tools/ctr_prep.py`,
   permutation-independent so it is computed once with no device work.
   OPEN QUESTION it leaves behind: the calcer lives in the Python prep
   script, so categoricals work in the BENCHMARK but `train()` still has
   no categorical path, and the arithmetic is a second hand-maintained
   numpy/Mojo equivalence of the kind `tools/interleaved_prep.py`
   already documents for `searchsorted` vs `binarization.binarize`.
   Moving it to `gbdt/ctrs/`, mirroring their path and called from
   `train.mojo` as one more host pass over `x_colmajor` beside the
   existing `best_split` and one-hot scans, would make it ship and would
   let the step-4 device port replace the body while call sites and
   gates stay put. Not urgent, but decide it before step 4 rather than
   after.
3. The segmented scan and the radix sort. Vendor check is DONE (above):
   both are ours to write, both are composition of pieces already in
   `gpu_util/kernel/reorder_one_bit.mojo`, and neither may assume a
   wavefront width.
4. History/Borders CTRs on device, learn-order permutation. **This is the
   real parity item**: their GPU `simple_ctr` default is Borders plus
   FeatureFreq, and Borders is the ordered-target-statistic half that makes
   CatBoost a distinct algorithm rather than a GBDT with frequency
   encoding. It needs, and none of these exist yet: target binarization
   (their default is MinEntropy with ONE border, set on GPU via
   `ctr_target_border_count`, with per-CTR target binarization REFUSED,
   `catboost_options.cpp:505`); the THREE-prior fan-out
   ({0,1},{0.5,1},{1,1}, so Borders emits three columns per cat feature,
   `cat_feature_options.cpp:117-129`); `THistoryBasedCtrCalcer`; and the
   elementwise kernels of `ctr_calcers.cu` (`UpdateBordersMask`,
   `MergeBins`, `MakeMeans`, `MakeMeansAndScatter`,
   `FillBinarizedTargetsStats`, `ExtractBorderMasks`,
   `GatherTrivialWeights`, `WriteMask`), every one of which is elementwise
   or gather/scatter with no warp intrinsics and no shared-memory
   accumulation, so this block should need NO new kernel-matrix rows.
5. MODEL SERIALIZATION, which was a prerequisite and not a CTR task. The
   numeric file format LANDED 2026-08-20 (`gbdt/models/model_text.mojo`,
   `pixi run check-model-io`), so what remains under this heading is the CTR
   tables themselves, added at the seam that file documents. Two things
   still block a categorical model from scoring raw data after that:
   `TBinarySplit` carries no split TYPE, so the one-hot predicate is
   recovered from the layout rather than from the model, and the device
   evaluator (`models/cuda/evaluator.mojo`) has no one-hot arm at all -- its
   `XorMask` slot is documented as staying zero.
6. Tree CTRs (feature combinations, `MaxTensorComplexity`, default 4).
   Their own `tree_ctr_datasets_visitor` machinery: combinations generated
   during tree growth, per level, with caching and memory limits
   (`ctr_leaf_count_limit`, `store_all_simple_ctr`). **This is where our
   category codes stop being harmless.** We use sorted-unique dense codes
   and they hash; the CTR VALUE depends only on counts so the difference
   cannot be seen today, but a COMBINATION bin is formed from their hash, so
   step 6 forces the hash port that FeatureFreq let us skip. Plausibly
   larger than 3 through 5 combined.
7. Move the calcer out of `tools/ctr_prep.py` into `gbdt/ctrs/`, called from
   `train.mojo` as a host pass, so `train()` gains a categorical path and
   the library ships what the benchmark already has.
