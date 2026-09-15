# Lane status: lane/gbdt-rest (2026-09-15)

Andrew, Sep 15 2026: "gbdt as a lane". Work through `gbdt/NOT_IMPLEMENTED.tsv`
(28 rows on origin/main at a76c02d27) for the CatBoost-reference GBDT:
implement the user-facing items, each bitwise identical on the GPU and the
CPU verifier (with public CPU inference where the model changes), and give
every remaining row a precise reason. Reference tree:
`/Users/andrewhendel/CascadeProjects/upstream/catboost` (7055d33d).

Worktree `scratchpad/wt-gbdt-rest`, branch `lane/gbdt-rest`, main only (0.8.7).
Evidence per item is small fixtures only (memory scope-lane-evidence-small).

## Triage

Classes: (a) a user-facing feature a CatBoost user would set (a loss, a
metric, an option or an input type); (b) internal reference plumbing this
implementation does not need; (c) intentionally excluded (dead or unreached in
the reference, nondeterministic by nature, or a reference bug).

Counts: **(a) 10, (b) 13, (c) 5.**

| # | Row (their file, symbol) | Class | Decision and reason |
|---|---|---|---|
| 1 | multilogit.cu RMSEWithUncertainty | a | Loss. Trained in the reference by `TMultiClassTrainer` (train_lib/multiclass.cpp:8), a two-plane approx on a one-column target. PLANNED, item 4. |
| 2 | multilogit.cu MultiCrossEntropy | a | Loss (multilabel). `TMultiClassTrainer` (multiclass.cpp:11); needs a 2D label input. PLANNED, item 5. |
| 3 | multilogit.cu MultiRMSE | a | Loss (multi-target regression). `TMultiClassTrainer` (multiclass.cpp:13); needs a 2D label input. PLANNED, item 3. |
| 4 | multilogit.cu BuildConfusionMatrixBins | a | A GPU METRIC kernel (Accuracy, Precision, Recall, F1 at gpu_metrics.cpp:304-326). GradientBoosting has no `eval_metric` or `custom_metric` surface (`git grep eval_metric` in python/mojolearn and gbdt is empty), so there is nothing for it to serve; it waits on that surface. |
| 5 | pointwise_targets.cu TNumErrorsMetric | c | In the kernel switch, refused by their own Init: unreachable in training. |
| 6 | pointwise_targets.cu MseImpl | c | `ApproximateMse` has no caller in catboost/. |
| 7 | bootstrap.cu GammaBootstrapImpl | c | Its only call is commented out (bootstrap.cu:82). |
| 8 | segmented_sort.cu SortPairsDescending arm | c | No caller passes compareGreater=true. |
| 9 | leaves_estimation_helper.h MakeSupportPairsMatrix, ReorderPairs, FilterZeroLeafBins | b | Plumbing of the pairwise oracle (pairwise_oracle.h), which only the pairwise trainer family uses (row 11). Follows row 11. |
| 10 | targets/kernel/ query_softmax.cu, pfound_f.cu, query_cross_entropy.cu, dcg.cu | a | Split by what trains each. **QuerySoftMax**: the querywise trainer QueryRMSE already runs on (train_lib/querywise.cpp:7, querywise_targets_impl.h:242-257, pointwise oracle). PLANNED, item 1. **QueryCrossEntropy**: `TPairwiseGpuTrainer<TQueryCrossEntropy>` (train_lib/query_cross_entropy.cpp:5-6). **pfound_f.cu** is `TPFoundF`, the target of the loss **YetiRankPairwise** (train_lib/pfound_f.cpp:5-6), not a PFound loss; PFound itself is only a metric. Both need the pairwise GPU learner (pairwise structure searcher, pair matrices, pairwise oracle), which is not implemented. **dcg.cu**: NDCG/DCG GPU metric kernels, which wait on the eval metric surface of row 4. |
| 11 | pair_logit.cu PairLogitPairwiseImpl, RemoveOffsetsBiasImpl | a | Loss PairLogitPairwise, `TPairwiseGpuTrainer<TPairLogitPairwise>` (train_lib/pair_logit_pairwise.cpp:5-6): the pairwise learner, not implemented. |
| 12 | pairs/util.cpp max_pairs | a | Option of PairLogit (loss_description.cpp:230-232). PLANNED, item 7. |
| 13 | query_helper.cu FillQueryEndMask, CreateSortKeys, FillTakenDocsMask, ComputeGroupMax, offsets-only ComputeGroupMeans | b | Reached only through `TQuerywiseSampler` (querywise_helper.cpp:28, :90), held only by QueryCrossEntropy (query_cross_entropy.h:169-191) and TPFoundF (pfound_f.h:133-141, pfound_f.cpp:96 RemoveQueryMax). It is those pairwise targets' query sampler, not a user bootstrap. Follows rows 10 and 11. |
| 14 | overfitting_detector.cpp TOverfittingDetectorWilcoxon | a | Reachable only by spelling the enum (enums.h:10); their Python docs list IncToDec and Iter only (python-package core.py:4974-4977). The statistic is `NStatistics::Wilcoxon` in library/cpp/statistics/statistics.h, which IS in the reference checkout (the old reason implied it was unavailable). Lowest value; not planned. |
| 15 | pointwise_hist1.cu ComputeHist1 | c | Registered, never called. |
| 16 | compute_by_blocks_helper.h TComputeSplitPropertiesByBlocksHelper | b | Histogram block regrouping; a performance layer. Note kept: can move score ties on fold-heterogeneous one-byte policies. |
| 17 | compute_by_blocks_helper.cpp Rebuild, StreamCount, ForceOneBlockPerPolicy | b | Stream and block configuration; one Metal queue. |
| 18 | gather_bins.cu GatherCompressedIndex | b | Multiclass load policy; same histogram values. |
| 19 | hist_single_leaf.cuh TComputeSingleHistKernel | b | Root-level kernel family; same arithmetic. |
| 20 | feature_layout_doc_parallel.h CreateFeaturesMapping shuffle | b | Feature grouping for kernel specialization. |
| 21 | feature_layout_feature_parallel.h TFeatureParallelLayout | b | Collapses to doc-parallel at one device. |
| 22 | feature_layout_single.h TSingleDevLayout | b | One device. |
| 23 | feature_layout_common.h SkipInSplitSearch, SkipFirstBinInScoreCount | b | Driven by exclusive feature bundles (binarizations_manager.h:90, helpers.cpp:158), set by the data layer; the only surface is `dev_efb_max_buckets` (oblivious_tree_options.cpp:26), a dev option. |
| 24 | compressed_index.h TCompressedDataSet container | b | One flat buffer at one device. |
| 25 | doc_parallel_dataset.h TDocParallelDataSet | b | Bundling class; arguments carried separately. |
| 26 | doc_parallel_dataset_builder.cpp | b | Its one decision (blockSize 1) is pinned. |
| 27 | catboost_options.cpp:357-359 l2 0 to 1e-20 | a | Option semantics of an explicit `l2_leaf_reg=0`. PLANNED, item 6; it may move recorded l2=0 cells, which are named in the row. |
| 28 | loss_description.cpp YetiRank permutations, decay | a | Loss parameters. PLANNED, item 2, with CatBoost's `Loss:key=value` spelling. |

## Plan (order of user value)

1. QuerySoftMax (params lambda 0.01, beta 1.0; Gradient leaves at 100
   iterations by default, catboost_options.cpp:100-105).
2. Loss parameters in CatBoost's spelling for the ranking losses
   (YetiRank permutations and decay, QuerySoftMax lambda and beta).
3. MultiRMSE. 4. RMSEWithUncertainty. 5. MultiCrossEntropy.
6. l2_leaf_reg=0 substitution. 7. max_pairs.

Not planned in this lane, with reasons above: the pairwise learner family
(QueryCrossEntropy, PairLogitPairwise, YetiRankPairwise, rows 9, 10, 11, 13),
the GPU metric kernels (rows 4, 10) behind a missing eval metric surface, and
Wilcoxon (row 14).

## Progress

- Triage committed (this file).
