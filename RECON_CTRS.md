# CTR / categorical recon (2026-08-21), the last feature block

Read from CatBoost `54a8143a`: `catboost/cuda/ctrs/*`,
`ctrs/kernel/ctr_calcers.cu` (429 lines, the kernels are SMALL; the
system around them is the work), `gpu_data/*` builders, and the options
defaults. This file is the warm-start for the port session.

STATUS: steps 1 and 2 are BUILT. One-hot is end to end (`gbdt/gpu_data/
compressed_index_builder.mojo` flag, scan skip, `==` predicate,
evaluator `takeEqual`, `train.mojo` surface, gated analytically in
`mojo_only/one_hot_check.mojo`). Host-side FeatureFreq shipped in
21c70ce as `tools/ctr_prep.py`, with a categorical arm in
`tools/catboost_arm.py`. Steps 3 to 5 are open.

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

## The two vendor-lib decision points (VENDOR RULE)

* RADIX SORT: their CTR bins are sorted with cub radix sort
  (`cuda_util/sort.h`). Metal has no cub. CHECK MAX's algorithm/sort
  surface FIRST; if absent, a portable LSD radix sort over (bin ||
  permPosition) keys is the fallback, no warp intrinsics required
  (RAFT's select_radix precedent: zero warp intrinsics). This is the
  long pole of the whole block.
* SEGMENTED SCAN: cub segmented scan on their side. We already build
  two-phase scans (stable partition chunk scans, scan_histograms); a
  flagged Blelloch over 800k elements is the same pattern. Check
  MAX first anyway.

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
3. The radix sort + segmented scan (vendor-check first).
4. History/Borders CTRs on-device, learn-order permutation.
5. Tree CTRs + model export + inference CTR tables.
