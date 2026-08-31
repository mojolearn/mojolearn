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
  `checks/one_hot_check.mojo`.
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
* **A DEVICE-WIDE SEGMENTED SCAN**, `gbdt/gpu_util/kernel/segmented_scan.
  mojo`: both of their entry points, `SegmentedScanVector`
  (`cuda_util/segmented_scan.h:8`, flag in bit 31 of a separate word) and
  `SegmentedScanAndScatterNonNegativeVector` (`cuda_util/kernel/scan.cu:47`,
  flag in the value's sign bit, answer scattered through the permutation),
  inclusive and exclusive. `pixi run check-segscan` gates all four arms
  against a host tally cell by cell. PORTING.md 49.
* **AN LSD RADIX SORT**, `gbdt/gpu_util/kernel/radix_sort.mojo`: their
  `ReorderBins` (`cuda_util/sort.cpp:544`), built by looping their own
  `ReorderOneBit<ui32, ui32>` over the bit range. `pixi run
  check-radixsort` gates STABILITY separately from sortedness, on an even
  and an odd pass count. PORTING.md 50.

BUILT 2026-08-20, the CTR round:
* **`gbdt/ctrs/`**, mirroring `catboost/cuda/ctrs/`: `ctr.mojo`
  (`ctr.h` plus the ctr-type predicate tables), `ctr_bins_builder.mojo`,
  `ctr_calcers.mojo` (`TWeightedBinFreqCalcer` and
  `THistoryBasedCtrCalcer`), `index_wrapper.mojo`, `ctr_binarization.mojo`,
  and `kernel/ctr_calcers.mojo` -- all TEN elementwise kernels of
  `ctrs/kernel/ctr_calcers.cu`, enqueued and gated cell by cell by
  `checks/ctr_kernels_check.mojo`. The BIN ORDERING and the SEGMENTED
  SCAN moved to the DEVICE on 2026-08-21 (`TCtrBinBuilderGpu`,
  `THistoryBasedCtrCalcerGpu`); the FeatureFreq calcer is still host side
  and is all that is left of `PORTING.md` deviation 52.
* **Target binarization**, MinEntropy with one border, gated against
  CatBoost's own borders in `bench/ctr_target_oracle.txt`.
* **The CTR option surface**: `TCtrDescription`, `TCatFeatureParams`,
  `GetDefaultPriors`, `CreateDefaultCounter`, `SetCtrDefaults`, and
  `CreateCtrConfigsFromDescription`'s prior x target-bin fan-out, in
  `options/catboost_options.mojo`.
* **A categorical path through `train()`**: `cat_features` plus their
  one-hot / CTR dispatch off `one_hot_max_size`.

BUILT 2026-08-21, the permutation round:
* **THE CTR ESTIMATION PERMUTATION**, `gbdt/data/permutation.mojo`: their
  `TDataPermutation` over `Shuffle` (`cuda/data/data_utils.h:21`), `TRandom`
  and MT19937-64, bit for bit against their own generator
  (`pixi run check-permutation`, oracle built by
  `tools/permutation_oracle/`). `train()` now runs their TWO writers -- the
  identity order for permutation-independent CTRs
  (`doc_parallel_dataset_builder.cpp:206`, `:229`) and the estimation
  permutation for the dependent ones (`:255`, `:257`) -- and `Borders`
  trains rather than raising. ONE column set where their loop builds
  `permutation_count`, and never their identity id 0: `PORTING.md` 55.
* **THE BIN ORDERING AND THE HISTORY CALCER ON THE DEVICE**,
  `TCtrBinBuilderGpu` and `THistoryBasedCtrCalcerGpu`, wiring
  `check-segscan`'s scatter form and `check-radixsort`'s `ReorderBins` into
  real callers. Gated by `pixi run check-ctr-device`, which compares the
  device answer against the host reference AND an independent O(n^2) tally
  over the permutation, then measures that ROW ORDER LEAKS THE LABEL (0.303
  against the permutation's -0.0027) and that a `train()` fit in row order
  reaches half the training loss on a feature carrying no information.

NOT BUILT, verified by search and not by memory:
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

## THE Borders APPLY-TIME TABLE, read from their source 2026-08-21

Captured while porting the estimation permutation, because that is the
cheapest moment it will ever be, and because the single most surprising
thing about it is worth stating first:

**THE APPLY-TIME TABLE IS NOT ORDERED AND CONTAINS NO PERMUTATION.** The
ordered statistic is a TRAINING-TIME device only. What an applied model
carries for a `Borders` CTR is the FULL-LEARN-SET per-category histogram
over target classes, and the permutation never enters it.

`TCtrValueTable` (`catboost/libs/model/ctr_value_table.h:17-105`) is three
things:

* `IndexBuckets`, a dense index hash from the row's CATEGORY HASH to a
  bucket index (`NCatboost::TDenseIndexHashView`, `GetIndexHashViewer()`);
* `CTRBlob`, an untyped byte blob read as `int[uniqueHashes *
  TargetClassesCount]` for Borders and Buckets -- one COUNT PER TARGET CLASS
  per category (`static_ctr_provider.cpp:255-283` builds it);
* two scalars, `CounterDenominator` (Counter/FeatureFreq only) and
  `TargetClassesCount`.

Apply, for one row and one `TModelCtr` (`static_ctr_provider.cpp:92-122`):

    bucket = hashView.GetIndex(hashedCatFeature[doc])
    if bucket == NotFoundIndex: value = ctr->Calc(0, 0)
    else if TargetClassesCount > 2:
        hist = &blob[bucket * TargetClassesCount]
        total = sum(hist[0 .. TargetBorderIdx])
        good  = sum(hist[TargetBorderIdx+1 .. end]);  total += good
        value = ctr->Calc(good, total)
    else:                              // the GPU default lands here
        hist = &blob[bucket * 2]
        value = ctr->Calc(hist[1], hist[0] + hist[1])

and `Calc` is `((countInClass + PriorNum) / (totalCount + PriorDenom) +
Shift) * Scale` (`online_ctr.h:289-292`).

Four consequences for whoever builds this:

1. At the GPU default `ctr_target_border_count = 1`, `TargetClassesCount`
   is 2 and the table is just `{count of binarized target 0, count of 1}`
   per category over the whole learn set. **No scan, no sort, no
   permutation, no history** -- it is the FeatureFreq table with one extra
   axis, and `build_ctr_tables` builds both from one pass over the codes.
2. A category the model never saw is not an error: it scores
   `Calc(0, 0)`, the prior alone.
3. The three priors are three `TModelCtr`s over ONE table, exactly as the
   three training columns are three divides over one scan.
4. `Shift` and `Scale` come from the CTR VALUE binarization, and CPU and
   GPU disagree there (the CPU's fixed `scale 15, shift 0` over [0,1]
   against the GPU's Uniform-15 borders from the observed column, see the
   oracle note below). A table written here carries this port's own
   borders, so those two fields are ours to fill from
   `compute_ctr_borders`, not theirs to copy.
5. The KEY is their category hash and this port uses dense sorted-unique
   codes. That is the same open question step 6 already names, and it
   arrives one step earlier than expected: the table's key mapping has to
   be decided before a categorical model can score raw data.

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
   numeric histogram pipeline unchanged. **The CTR binarizer is not one
   grid, it is TWO, and neither is the GreedyLogSum we ported for numeric
   features.** `Borders` columns take `Uniform` with 15 borders, at
   `min + (max - min) * k / 16` for k = 1..15 with duplicates dropped --
   the two-argument `TCtrDescription` constructor's default
   (`cat_feature_options.cpp:167-170`), which is the constructor
   `SetCtrDefaults` builds the Borders description with. `FeatureFreq`
   columns take `MinEntropy` with 15, passed explicitly by
   `CreateDefaultCounter` (`catboost_options.cpp:392-415`) and re-applied
   to FeatureFreq descriptions ONLY by `SetDefaultBinarizationsIfNeeded`
   (`:418-427`). Both are real defaults on real code paths; this file
   previously gave the Borders grid as the FeatureFreq one.
   The FeatureFreq value itself is `(bin_count + prior) / (n_rows +
   prior_observations)`, default prior {0.0, 1}, so `count / (n + 1)`
   (`cuda/ctrs/kernel/ctr_calcers.cu:100`, defaults at
   `cat_feature_options.cpp:127-129`) -- though the DEFAULT dispatch is
   `WeightedBinFreqCtrsImpl` (`:56-67`) rather than that kernel, because
   `counter_calc_method` defaults to `SkipTest` and not `Full`; with no
   test pool the two are the same number. All of it is implemented in
   `gbdt/ctrs/`.
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
`reorder_one_bit` being ported. Both landed on 2026-08-20 as composition:
the radix sort loops their one-bit stable reorder over key bits, and the
segmented scan is the three-phase decoupled scan with a carry that resets on
a flag. Only the segmented scan's BLOCK-level half had to be written out,
because `prefix_sum` takes no custom operator.

**Both must be GPU-agnostic from the first line** (standing rule): no
wavefront-width assumption, 32 on NVIDIA and Apple against 64 on AMD, and
shared-memory sizing from a queried budget rather than from this laptop.
RAFT's `select_radix` has zero warp intrinsics, which is the precedent that
this is achievable in one source.

## Gates that exist to be reused

* Oracle: `tools/catboost_oracle.py` can emit a fixture WITH cat
  features and simple CTRs pinned (CPU CatBoost). THE CAUTION BELOW WAS
  ACTED ON, on a tiny fixture, and it was warranted: CPU and GPU DO
  binarize CTR values differently. See the Borders entry two bullets down.
* CONSTRUCTED fixtures, not real datasets. Correctness gates on (a) an
  analytic answer written down in advance (`checks/one_hot_check.mojo`
  is the model: the right answer is arithmetic, no fixture needed) and
  (b) CatBoost's OWN output as oracle, via `save_model(format='json')`,
  which exposes the CTR tables, config, and CTR borders. A fixture we
  build beats a real dataset here, because we plant the adversarial
  cardinalities on purpose (1, 2, 3, 255, 256, thousands) and scattered
  values, where a real dataset has whatever it happens to have and we
  are hoping it covers the breaking case. See PORTING_RULES.md:229.
* THERE IS NO CATBOOST FeatureFreq ORACLE ON THIS BOX. `simple_ctr=
  'FeatureFreq'` "is not implemented on CPU yet"
  (`catboost_options.cpp:509`, re-verified against the shipped 1.2.10
  binary on 2026-08-20), and their GPU arm does not run on Apple
  ("Environment for task type [GPU] not found", same run). The CPU
  spelling of the same information is `Counter`, a different
  normalization of the same counts, so a CPU CatBoost arm is the same
  INFORMATION CLASS, not the same bits: quality bands compare, values do
  not. Any claim that our FeatureFreq is bit-exact to theirs rests on
  reading their source, not on a local oracle, and must say so.
* THE Borders ORACLE QUESTION IS ANSWERED, on a tiny fixture as this file
  demanded. `simple_ctr='Borders'` IS accepted on their CPU and does
  train. What `save_model(format='json')` then exposes is the FINAL
  apply-time table -- `ctr_data.hash_map` keyed by category hash, plus
  per-config `prior_numerator` / `prior_denomerator` / `scale` / `shift`
  under `features_info.ctrs` -- and NOT the per-row ordered statistic the
  GPU calcer builds during training. It is still a real oracle for two
  things and both were used: the resolved `cat_feature_params` printed the
  three priors and both ctr binarization grids verbatim, and
  `pool.quantize(feature_border_type='MinEntropy')` gates the target grid.
  CPU and GPU DO diverge where this file warned they might -- the CPU
  quantizes online CTR values on a fixed `scale 15, shift 0` grid over
  [0,1] where the GPU builds Uniform-15 borders from the observed column.
  So the CPU arm is a quality comparison, not a value oracle, and the
  Borders VALUES are gated analytically instead, in
  `checks/ctr_check.mojo` against an independent O(n^2) tally.
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
2. DONE. Host-side FeatureFreq, first in `tools/ctr_prep.py` (21c70ce) and
   now in `gbdt/ctrs/` where its mirror address is, called from
   `train.mojo` as one more host pass over `x_colmajor` beside the existing
   `best_split` and one-hot scans. `tools/ctr_prep.py` keeps a numpy copy
   ONLY to generate the pre-binned fixture
   `bench/interleaved/ctr_quality.mojo` still reads; that harness rewire
   (hand `_catcodes.u8` and a `cat_features` flag list to `train()`)
   deletes the duplicate, and the manifest now carries the cardinalities it
   needs. OPEN, and owned by the bench lane.
3. The segmented scan and the radix sort. Vendor check is DONE (above):
   both are ours to write, both are composition of pieces already in
   `gpu_util/kernel/reorder_one_bit.mojo`, and neither may assume a
   wavefront width.
3. DONE 2026-08-20. The segmented scan and the radix sort, both in
   `gpu_util/kernel/`, both gated (`check-segscan`, `check-radixsort`),
   neither assuming a wavefront width and neither needing a kernel-matrix
   row -- the segmented scan's block comes from `column_shared_limit` at 8
   bytes a thread, which CatBoost's own `GetScanBlockSize()` of 768 caps on
   every vendor, so the geometry is identical across the three columns.
   WIRED 2026-08-21: `ReorderBins` by `TCtrBinBuilderGpu` and the scatter
   form of the segmented scan by `THistoryBasedCtrCalcerGpu`, both reached
   from `train(cat_features=...)`. `launch_segmented_scan_vector` is still
   unreached and stays in `UNWIRED.md`: its only call site in all of
   `catboost/cuda` is `VisitFloatFeatureMeanCtrs` (`ctr_calcers.h:182`),
   and `FloatTargetMeanValue` has no calcer here.
   One correction to the vendor note above: `block.prefix_sum` carries the
   radix sort's bit scan but CANNOT carry the segmented scan, because it
   takes no operator and a segmented combine is not addition. See
   PORTING.md 49 for why the two-unsegmented-scans trick was refused.
4. History/Borders CTRs on device, learn-order permutation. **This is the
   real parity item**: their GPU `simple_ctr` default is Borders plus
   FeatureFreq, and Borders is the ordered-target-statistic half that makes
   CatBoost a distinct algorithm rather than a GBDT with frequency
   encoding. FOUR of the five pieces now exist: target binarization
   (MinEntropy with ONE border, set on GPU via `ctr_target_border_count`,
   per-CTR REFUSED at `catboost_options.cpp:505`); the THREE-prior fan-out
   ({0,1},{0.5,1},{1,1}, so Borders emits three columns per cat feature,
   `cat_feature_options.cpp:117-129`); `THistoryBasedCtrCalcer` itself,
   over a host segmented scan; and every elementwise kernel of
   `ctr_calcers.cu`. **The prediction that they need NO new kernel-matrix
   rows HELD**: not one warp intrinsic, not one byte of shared memory and
   not one atomic in any of the ten, `blockSize = 256` throughout.
   WHAT REMAINS is the device segmented scan and radix sort (step 3) and,
   newly identified, the CTR ESTIMATION PERMUTATION -- see the NOT BUILT
   list above. `train()` raises on a Borders config rather than running
   the ordered statistic in row order.
5. MODEL SERIALIZATION, **DONE 2026-08-20**. The numeric file format
   landed first (`gbdt/models/model_text.mojo`, `pixi run check-model-io`),
   then the CTR half at the seam it documented, and a trained categorical
   model now saves, loads and scores RAW rows through both apply paths
   (`pixi run check-ctr-apply`). What that took, all three pieces named in
   the seam block:

   * `gbdt/models/ctr_value_table.mojo`, their `TCtrValueTable` plus the
     `TModelCtr` fields that turn its counts into a value, written to the
     file as `ctr_table` / `ctr_entry` records -- their `ctr_data.hash_map`,
     which for FeatureFreq stores ONE INTEGER per category and forms the
     value at apply time with `TModelCtr::Calc`. `ctr_borders` was in the
     plan and is NOT written: in this format a CTR column is a column of the
     feature table, so its `feature` record already carries exactly the
     borders their `TCtrFeature::Borders` does.
   * `TBinarySplit` carries their `EBinSplitType`, set by their `ToSplit`
     rule (`cuda/methods/helpers.cpp:164-170`) and written as a trailing
     `split_type take_bin` token so a float-only model's file stays
     byte-identical. The apply cross-checks it against the layout, which is
     their `CB_ENSURE(dataSet.IsOneHot(...))`.
   * The device evaluator has the one-hot arm its `XorMask` slot reserved.
     Their GPU evaluator refuses categorical models outright, so the
     predicate is transcribed from their CPU one
     (`libs/model/model.cpp:566-572` for the repack,
     `cpu/evaluator_impl.cpp:38` for the compare), under their own
     `NeedXorMask` dispatch. PORTING.md 55.

   CLOSED 2026-08-20 for `Borders` as well. `build_ctr_tables` grew the
   target-class axis their `CalcFinalCtrsImpl` allocates
   (`online_ctr.cpp:909-910`), `TCtrValueTable.value_for` and
   `empty_value` dispatch on `ctr_type` the way
   `TStaticCtrProvider::CalcCtrs` does, and `train()`'s implicit fallback
   moved to `TCatFeatureParams.default()` -- CatBoost's own GPU
   `simple_ctr` -- because the reason it was not there is gone. The
   apply-time table is the FULL-LEARN-SET histogram, so a `Borders`
   model's learn-row predictions do NOT reproduce the fit's loss the way a
   `FeatureFreq` model's do; the gate is an independent per-row tally
   instead.
6. Tree CTRs (feature combinations, `MaxTensorComplexity`, default 4).
   Their own `tree_ctr_datasets_visitor` machinery: combinations generated
   during tree growth, per level, with caching and memory limits
   (`ctr_leaf_count_limit`, `store_all_simple_ctr`).

   RE-SCOPED 2026-08-21 from their source; this entry used to say a
   combination bin is formed from their hash, and that is FALSE for
   training. `TFeatureTensorTracker::AddFeatureTensor`
   (`gpu_data/oblivious_tree_bin_builder.cpp:123-186`) composes a tensor by
   feeding each cat feature's DENSE COMPRESSED BINS and each binary
   split's bits into `TCtrBinBuilder::AddCompressedBins` -- device
   reindexing over exactly the bin-builder primitive `TCtrBinBuilderGpu`
   already ports. No hash is computed anywhere on the training path, so
   our dense codes stay harmless through tree-CTR TRAINING too.

   Where the hash IS the key is the MODEL/APPLY half: a tree-CTR table is
   looked up by folding `CalcHash(acc, element)` over the combination's
   elements -- each cat element a `CalcCatFeatureHash` (= their own
   CityHash64 1.0 variant, low 32 bits) sign-extended through
   `(ui64)(int)`, each binary-split element a bare 0/1
   (`libs/model/ctr_provider.h:94-122`, `libs/model/hash.h:11-14`,
   `libs/cat_feature/cat_feature.cpp:6-8`). THAT STACK IS PORTED AND GATED
   (2026-08-21): `gbdt/digest/city.mojo`, `gbdt/cat_feature/
   cat_feature.mojo`, `gbdt/models/hash.mojo`, `pixi run check-cityhash`
   against their own `city.cpp` compiled by `tools/cityhash_oracle/`.
   Public CityHash vectors would have been the WRONG oracle -- their
   `city.h` states its results differ from mainline CityHash -- and the
   chain rows caught a real defect on first contact (a Mojo cast chain
   zero-extending where `(ui64)(int)` sign-extends).

   What remains of step 6 is the machinery itself: the visitor, the
   memory estimator, per-level combination generation and caching, CTR
   borders per tree ctr, and the apply-side `CalcHashes` + tree-CTR
   tables in the model file.
7. DONE, and folded into step 2: the calcer lives in `gbdt/ctrs/` and
   `train(cat_features=...)` calls it, so the library ships the categorical
   path rather than the benchmark alone.
