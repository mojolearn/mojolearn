# Apple FAST classical tree candidates — source only

All twelve candidate implementations are written on `lane/apple-fast-classical-ideas-20261006` from `fd6cf8045`. **NEVER RUN — PENDING MEASUREMENT.** Nothing in this lane has been compiled, verified, tested, linted or measured. No speed, correctness or quality acceptance is claimed. All new controls default OFF and require Apple FAST; IDENTICAL and non-Apple specializations retain their existing schedules.

This file records the exact implemented scope, including adaptations from the original idea cards. The machine-readable A/B selector input is `trees.json`. Define arrays contain bare compiler names; enabling a define with `=0` still counts as defined. FAST requires neither IDENTICAL nor DETERMINISTIC mode define. Existing prerequisites must match in both arms.

Only production source and these handoff artifacts were written. No generated binary, execution result or fabricated measurement is included. The coordinator granted narrowly expanded ownership of `core/forest_inference.mojo` for T08; its RF_INPUT guard leaves ExtraTrees launch geometry unchanged. `core/forest_inference_model.mojo` needed no edit.

## AFCL-T01 — Quantized boosting histogram row chunks

QH_MIN_ITEMS_PER_BLOCK is 4096 with T01 instead of 8192. The same constant feeds host replica selection and per-partition active-block limits; all rows keep their existing quantization and bin destinations.

- A defines: (none beyond matching workload settings).
- B defines: `MOJOLEARN_AFCL_T01`.
- Paths: `gbdt/apple_fast_classical.mojo`, `gbdt/methods/greedy_subsets_searcher/kernel/hist_quantized_shared.mojo`, `gbdt/methods/greedy_subsets_searcher/quantized_hist_launcher.mojo`.
- Callers: GradientBoosting/CatBoost-style Depthwise fit; GradientBoosting/CatBoost-style Lossguide fit.
- Actual scope: Existing one-byte, two-statistic quantized shared-histogram family; both mode-skip and ordinary/gather launchers consume the new row chunk.
- Prerequisites: Apple FAST; QUANTIZED_HIST_LIVE; do not define MOJOLEARN_2045_FAST_NO_QUANT_HIST; Every policy block is one-byte and stat_count is two.
- Limitations: Original broad symmetric idea adapted to the supported nonsymmetric shared family; symmetric histogram kernels unchanged. Occupancy/grid floors can make the replica count equal across arms for some shapes; this is not a promise of a change for every input. Binary/half-byte and multistat fallbacks unchanged.
- Future quality coverage: Same quantized addends and overflow bounds; later compare complete histogram cells and resulting splits. Full fit quality: heldout RMSE/logloss/AUC/ranking metrics as applicable to each actually admitted objective; full settings and all rows retained.
- Status: `source_written_uncompiled_unverified_unmeasured`; default OFF.

## AFCL-T02 — Quantized boosting histogram feature groups

QH_GROUP_FEATURES becomes eight; shared allocation, whole compressed-index-word count, feature offsets, skip-bin vector width and launcher group count derive from it. This reduces the per-group histogram to 16 KiB.

- A defines: (none beyond matching workload settings).
- B defines: `MOJOLEARN_AFCL_T02`.
- Paths: `gbdt/apple_fast_classical.mojo`, `gbdt/methods/greedy_subsets_searcher/kernel/hist_quantized_shared.mojo`, `gbdt/methods/greedy_subsets_searcher/quantized_hist_launcher.mojo`.
- Callers: GradientBoosting/CatBoost-style Depthwise fit; GradientBoosting/CatBoost-style Lossguide fit.
- Actual scope: Same one-byte, two-statistic shared-histogram family as T01; samples, features, bins and quantized statistics are unchanged.
- Prerequisites: Apple FAST; QUANTIZED_HIST_LIVE; MOJOLEARN_2045_FAST_NO_QUANT_HIST absent; One-byte policy blocks and two statistic planes.
- Limitations: Original symmetric feature-group idea adapted to the supported nonsymmetric family; symmetric packed-feature kernels unchanged. Sub-byte/multistat fallbacks unchanged. Extra groups may cost more launches/flushes; no performance result exists.
- Future quality coverage: Histogram totals, best-split tie behavior, tail groups, categorical one-byte encodings and missing bins. Heldout full-workload quality for both growth policies; T01+T02 must later be evaluated together.
- Status: `source_written_uncompiled_unverified_unmeasured`; default OFF.

## AFCL-T03 — Per-feature binarization block width

BINARIZE_BLOCK_SIZE becomes 512 instead of 1024, keeping eight documents per thread. Imported launcher constants and in-kernel row strides use the same geometry; all 256 border slots remain loadable.

- A defines: (none beyond matching workload settings).
- B defines: `MOJOLEARN_AFCL_T03`.
- Paths: `gbdt/apple_fast_classical.mojo`, `gbdt/gpu_data/kernel/binarize.mojo`.
- Callers: Boosted-tree fit quantization; Boosted-tree resident prediction quantization.
- Actual scope: binarize_float_feature_kernel and its callers using BINARIZE_BLOCK_SIZE/BINARIZE_DOCS_PER_THREAD; bin comparisons still use exact_f32_gt.
- Prerequisites: Apple FAST; A workload reaching the per-feature binarizer in both arms.
- Limitations: The existing packed-word binarizer is unchanged. A workload routed entirely through GBDT_INDEX_PACK_DEVICE/GBDT_PREDICT_PACKED does not establish T03 coverage. Borders, border count, NaN substitution and feature packing policy unchanged.
- Future quality coverage: Exact bins at border ties, signed zero, subnormals and NaNs; same feature word packing. Full fitted-model and prediction quality with numerical and categorical inputs.
- Status: `source_written_uncompiled_unverified_unmeasured`; default OFF.

## AFCL-T04 — Symmetric partition metadata geometry

SPLIT_BLOCK_SIZE becomes 128 instead of 256. Bin-update, partition-offset/size and associated fused winner/bin launches inherit the four-SIMD-group width and matching grids/shared-array sizing.

- A defines: (none beyond matching workload settings).
- B defines: `MOJOLEARN_AFCL_T04`.
- Paths: `gbdt/apple_fast_classical.mojo`, `gbdt/methods/pointwise_optimization_subsets.mojo`.
- Callers: Symmetric boosted-tree structure search; Ordered symmetric structure-search callers of pointwise optimization subsets.
- Actual scope: Pointwise optimization-subset row metadata and fused bin-update schedule. Existing radix/reorder kernels keep their own block widths and algorithm.
- Prerequisites: Apple FAST; A caller reaching pointwise_optimization_subsets.
- Limitations: This is partition metadata/bin scheduling, not a replacement of the physical radix partition. Depthwise/lossguide partition kernels outside this module unchanged. T05 shares the fused winner/bin launch and needs a later joint A/B.
- Future quality coverage: Every row assigned exactly once; child counts, missing routing and ordered membership. Full symmetric/ordered heldout quality, including categorical and weighted inputs.
- Status: `source_written_uncompiled_unverified_unmeasured`; default OFF.

## AFCL-T05 — Cooperative split-record batches

Inside existing cooperative _fold_block, a lane folds four adjacent candidate records before stepping by block_width*4, instead of one strided record. The same _record_less comparator, all candidates, shared winner reduction and descriptor writes remain.

- A defines: `MOJOLEARN_SYM_RESOLVE_BLOCK`.
- B defines: `MOJOLEARN_SYM_RESOLVE_BLOCK`, `MOJOLEARN_AFCL_T05`.
- Paths: `gbdt/apple_fast_classical.mojo`, `gbdt/methods/kernel/pointwise_split_resolve.mojo`.
- Callers: Symmetric/ordered boosted-tree fused split search.
- Actual scope: Cooperative winner resolution in pw_resolve_pack_bins_kernel. This is an additional scheduling variant of the existing opt-in cooperative arm.
- Prerequisites: Apple FAST; MOJOLEARN_SYM_RESOLVE_BLOCK enabled in BOTH arms; MOJOLEARN_GBDT_FUSED_LEVEL_OFF and MOJOLEARN_GBDT_FUSED_PW_OFF absent.
- Limitations: The new switch does not activate SYM_RESOLVE_BLOCK by itself. The unfused/serial resolver is unchanged. No claim that the prerequisite cooperative implementation is qualified; its status is inherited.
- Future quality coverage: Best score, feature/bin tie ordering, sentinel/empty helpers and tails. Constrained/categorical split outcomes and full fitted heldout quality; T04 interaction.
- Status: `source_written_uncompiled_unverified_unmeasured`; default OFF.

## AFCL-T06 — Apple leaf-statistic partial budget

_afcl_leaf_stats_sm maps the hardware-derived statistic scheduling budget to ceil(sm/2). Both scratch sizing and all one-/two-stat gathered/non-gathered leaf-statistic launches use the same budget; remaining solver/work kernels use the original sm.

- A defines: (none beyond matching workload settings).
- B defines: `MOJOLEARN_AFCL_T06`.
- Paths: `gbdt/apple_fast_classical.mojo`, `gbdt/methods/leaves_estimation/apple_fast_est.mojo`.
- Callers: Apple fast pointwise boosted-tree leaf estimation; Weighted/unweighted regression and binary-classification objectives admitted by apple_est_handles.
- Actual scope: Current AppleEstScratch and apple_fast_estimate_and_apply path. Float reduction association can change; solver iterations, acceptance/backtracking, regularization and row membership are preserved.
- Prerequisites: Apple FAST; Existing EST_STATS_FUSED/EST_ITERS_DEVICE path admitted by apple_est_handles; For standard current defaults, MOJOLEARN_SYM_EST_ALL_OFF absent; do not enable rejected EST_REUSE_PART merely for this card.
- Limitations: Multiclass, multi-RMSE, pair/group/ranking objectives and has_group cases are refused by the existing path and unchanged. No standalone change to generic partition reducers; unrelated callers unaffected. Small hardware budgets can collapse to the same integer count.
- Future quality coverage: Leaf statistics with sample weights, cancellation, skewed leaves, small Hessians and tail rows. Leaf values, solver acceptance/stopping behavior and independent heldout objective quality, allowing changed bits.
- Status: `source_written_uncompiled_unverified_unmeasured`; default OFF.

## AFCL-T07 — Resident boosted-tree prediction blocks

Resident oblivious model application uses AFCL_PREDICT_BLOCK=128 instead of 256 for both ceil-divided row grids and launches, including ordinary per-tree/four-tree and packed all-tree applications. Existing bounded grid stride covers every row.

- A defines: (none beyond matching workload settings).
- B defines: `MOJOLEARN_AFCL_T07`.
- Paths: `gbdt/apple_fast_classical.mojo`, `gbdt/resident_model.mojo`.
- Callers: Resident GradientBoosting/CatBoost-style predict; Resident boosted-tree predict_proba/link callers.
- Actual scope: Oblivious ResidentGbdtModel application. Tree order, approximation dimensions, categorical bins and model buffers unchanged.
- Prerequisites: Apple FAST; Resident oblivious model prediction in both arms.
- Limitations: Nonresident prediction and nonsymmetric resident delegation unchanged. Existing packed prediction remains its own prerequisite when that route is selected; T07 does not activate it. Quantization and output-link launch widths are unchanged by T07.
- Future quality coverage: Final raw scores/probabilities across all approximation dimensions, zero-tree/depth cases and categorical routes. Heldout accuracy/logloss/RMSE, repeated outputs and concurrent-live-model lifetime.
- Status: `source_written_uncompiled_unverified_unmeasured`; default OFF.

## AFCL-T08 — RF traversal workgroup width

RF_INPUT specializations use 64 instead of 128 threads per traversal group. Ordered and row-owned groves use matching ceil-divided grids; lane-owned groves use two rather than four rows, with matching shared row staging and partial-buffer layouts. Logical 32-tree grove reductions are retained.

- A defines: (none beyond matching workload settings).
- B defines: `MOJOLEARN_AFCL_T08`.
- Paths: `core/forest_inference.mojo`.
- Callers: RandomForestClassifier GPU inference; RandomForestRegressor GPU inference.
- Actual scope: Shared launch_forest_inference transient/resident ordered, scalar and vector grove paths, guarded additionally on RF_INPUT=True. ExtraTrees specializations retain 128 threads.
- Prerequisites: Apple FAST; RF caller reaching core forest inference, including the explicit parallel_groves engine; same engine and layout flags in both arms.
- Limitations: Legacy/host inference engines and OOB training kernels are unchanged. No source change was needed in core/forest_inference_model.mojo; its existing dispatch consumes launch_forest_inference. Shared-row/packed-node/vector flags must match across arms; their existing eligibility rules are unchanged.
- Future quality coverage: Scalar/vector probability or regression outputs, label argmax, uneven depths, output capacity tails and leaf comparisons. Heldout quality, same complete forest and every output; repeated calls and model lifetime.
- Status: `source_written_uncompiled_unverified_unmeasured`; default OFF.

## AFCL-T09 — ExtraTrees sampled-feature tiles

ET_FEATURE_TILE is eight instead of sixteen for tiled range searches and tiled regression-score kernels, including their ceil-divided feature grids. Selected feature slots and RNG/threshold assignment do not change.

- A defines: (none beyond matching workload settings).
- B defines: `MOJOLEARN_AFCL_T09`.
- Paths: `extratrees/impl/decisiontree/batched_levelalgo/builder.mojo`.
- Callers: ExtraTreesClassifier fit; ExtraTreesRegressor fit; RandomTreesEmbedding or other ExtraTrees-builder callers reaching tiled range search.
- Actual scope: Existing ET_RANGE_TILED and ET_SCORE_TILED routes only; the classifier receives range scheduling, while regression may receive range and score scheduling.
- Prerequisites: Apple FAST; Actual tiled range/score dispatch, with existing row-major eligibility and switches identical in both arms.
- Limitations: Does not force row-major layout or enable a previously disabled tiled route. Classification score kernels outside the tiled regression path unchanged. Existing row-major/cost routing and all thresholds/features/tree counts unchanged.
- Future quality coverage: Ranges, sampled-feature RNG mapping, threshold assignment, impurity/split ties and completed tree count. Full classifier/regressor heldout quality; embedding output quality for affected callers.
- Status: `source_written_uncompiled_unverified_unmeasured`; default OFF.

## AFCL-T10 — IsolationForest query blocks

IF_PATH_TPB defaults to 128 instead of 256; IFLaunchKnobs.default and existing path/score launchers consume it, preserving per-row traversal and correction arithmetic.

- A defines: (none beyond matching workload settings).
- B defines: `MOJOLEARN_AFCL_T10`.
- Paths: `isolation_forest/impl/isolation_tree_builder.mojo`.
- Callers: IsolationForest score_samples; IsolationForest decision_function/predict; IsolationForest path_lengths and fit-time contamination scoring using default launch knobs.
- Actual scope: Existing path/score and score-epilogue launch geometry selected by the default IFLaunchKnobs; training tree construction geometry is unchanged.
- Prerequisites: Apple FAST; Default path launch knobs or explicit IF_PATH_TPB use in both arms.
- Limitations: A caller supplying a fixed explicit path_tpb can override the candidate and must not be counted as reached. Tree count, max_samples, max_depth, contamination and seed unchanged.
- Future quality coverage: Path lengths and correction for degenerate leaves, anomaly score/ranking, contamination threshold and decisions. Full scoring and fit/contamination boundary, repeated output lifetime.
- Status: `source_written_uncompiled_unverified_unmeasured`; default OFF.

## AFCL-T11 — TreeSHAP query/fold block width

QUERY_TPB=64 instead of 128 is used by query grids for direct tree units, table row units, decision-word table row units and final contribution fold. Preparation and table construction retain TPB=128.

- A defines: (none beyond matching workload settings).
- B defines: `MOJOLEARN_AFCL_T11`.
- Paths: `xtrees/shap_device.mojo`.
- Callers: GPU TreeExplainer/TreeSHAP values for classical tree models.
- Actual scope: All existing row-query variants and the contribution fold in shap_device; table/direct eligibility and attribution arithmetic unchanged.
- Prerequisites: Apple FAST; Existing GPU TreeSHAP route; preparation/table flags identical in both arms.
- Limitations: Does not enable MOJOLEARN_SHAP_FAST_ROW_PAIR, a recorded loser. Background preparation, table construction, chunk memory budget and unsupported interaction APIs unchanged.
- Future quality coverage: Every feature/output contribution and additivity versus independent references, shallow/deep/zero-cover cases. No dropped rows/features; old-output and concurrent-model lifetime.
- Status: `source_written_uncompiled_unverified_unmeasured`; default OFF.

## AFCL-T12 — CTR elementwise/count preprocessing blocks

CTR_BLOCK_SIZE=128 instead of 256 changes the existing elementwise/calculation launchers and their matching per-thread row strides. CTR_DOCS_PER_THREAD stays four.

- A defines: (none beyond matching workload settings).
- B defines: `MOJOLEARN_AFCL_T12`.
- Paths: `gbdt/apple_fast_classical.mojo`, `gbdt/ctrs/kernel/ctr_calcers.mojo`.
- Callers: Categorical boosted-tree CTR preparation; Categorical boosted-tree inference/preparation callers actually using ctr_calcers launchers.
- Actual scope: Trivial weights, border masks, weighted frequency, means/scatter, target stats and related admitted CTR calculation kernels. Sort and segmented prefix/reduction algorithms unchanged.
- Prerequisites: Apple FAST; A categorical task reaching gbdt.ctrs.kernel.ctr_calcers launchers.
- Limitations: Inference paths using saved CTR table lookups directly are unchanged. Unimplemented groupwise-CTR kernels remain unsupported; no new ranking-group claim. Category order, smoothing priors and leakage prevention unchanged.
- Future quality coverage: Ordered statistics, unseen categories, category counts, fold isolation, tied values and scatter tails. Full categorical heldout quality; T03+T07+T12 complete-configuration A/B remains pending.
- Status: `source_written_uncompiled_unverified_unmeasured`; default OFF.

## Pending work

No known source blocker is being deferred. All build acceptance, route reachability, numerical quality, correctness, performance and full-workload evidence remain pending by the owner’s instruction. The broader symmetric/multistat/ranking/nonsymmetric/host paths excluded above are unchanged; these twelve source candidates are not claims that every original broadly named estimator path changed.

Later evaluation must map every affected estimator to its saved full workload before any timed operation, audit intrinsic row caps, include preparation/fit/synchronization/consumed outputs, and report inference/cold/repeated calls separately. Preserve settings, convergence effort and randomness. Quantify task quality independently while allowing changed FAST bits. Keep failures and losers visible. No board updates or promotion is appropriate without that evidence.

Joint candidates needing later evaluation include T01+T02, T04+T05, histogram scheduling+T06, and T03+T07+T12. Existing opt-ins such as the cooperative resolver and parallel-groves engine must be held constant across an A/B, with their own qualification status explicit.

Keep logs out of context: save complete output to files, use targeted rg/grep with bounded surrounding lines and short tails, and summarize exit status, coverage, failures, and evidence paths. Expand only relevant diagnostic blocks; never hide failures or infer full success from filtered output.
