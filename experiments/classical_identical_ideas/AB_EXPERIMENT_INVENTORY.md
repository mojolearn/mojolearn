# Classical ML IDENTICAL A/B experiment inventory

NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.

Branch: `ideas/classical-identical-20261006`. This inventory travels with the lane commit shown in the final handoff and Git history. It records **91 named new A/B source arms across 53 classical IDs**, plus **130 within-card and cross-card configurations**. C45–C51 are retained references owned by TREES IDENTICAL.

Every new switch defaults OFF and requires IDENTICAL mode. **A** enables the listed new define(s). **B** omits them and preserves the incumbent configuration, including its already-enabled optimizations. A named experiment is selected explicitly; a manifest’s `default_configuration` is only the selector’s initial choice, not a promoted product default.

Source integration means code is connected to a production caller. It does **not** establish that a compiler accepts it, a particular saved full workload takes the arm, its outputs match across vendors, quality is preserved, or performance improves. Numerical profiles are intended to be selected together across NVIDIA, AMD, Apple and host within a version. No source, compile, binary, identity, quality or timing evidence was fabricated.

Some cards retain unfinished estimator extensions. Those are recorded as source gaps, separately from unsupported public APIs/representations and missing full-workload facts. An arm cannot be qualified by substituting a reduced dataset or a different estimator route. The C42 x_cluster linkage variant is separate from the old hierarchy single-linkage recipe.

Machine inventory: [AB_EXPERIMENT_INVENTORY.json](AB_EXPERIMENT_INVENTORY.json). Per-ID coverage and gaps: [implementation ledger](implementation_ledger.json) and [source status](IMPLEMENTATION_STATUS.md). Full recipe mapping: [workload_map.json](workload_map.json).

## New source arms

| ID | New A/B arms | Existing-work references |
|---|---|---|
| [C01](C01/manifest.json) Versioned fixed-leaf reduction profiles | `leaf64`, `leaf128` | [I04](../performance_ideas/I04/manifest.json) |
| [C02](C02/manifest.json) Fuse independent column statistics with shared input loads | `paired_stats`, `linear_pair` | [I24](../performance_ideas/I24/manifest.json) |
| [C03](C03/manifest.json) Fuse finite checks and extrema production | `finite_extrema` | [I24](../performance_ideas/I24/manifest.json) |
| [C04](C04/manifest.json) Apply centering and scaling at the consumer load | `load_center` | New extension; incumbent source linked below |
| [C05](C05/manifest.json) Retain bounded classical scratch across phases | `phase_scratch`, `ols_phase` | [I02](../performance_ideas/I02/manifest.json) |
| [C06](C06/manifest.json) Batch row norms without changing feature folds | `rows2`, `rows4` | New extension; incumbent source linked below |
| [C07](C07/manifest.json) Stable radix layout for classical preparation | `digit4`, `digit6`, `keys1024`, `keys4096` | [I19](../performance_ideas/I19/manifest.json) |
| [C08](C08/manifest.json) Reuse category dictionaries and inverse maps | `dictionary_inverse`, `grouped_output` | [I19](../performance_ideas/I19/manifest.json) |
| [C09](C09/manifest.json) Fused regression and weighted metric reports | `regression_bundle` | [I24](../performance_ideas/I24/manifest.json) |
| [C10](C10/manifest.json) Reuse stable score sorting for ranking metrics | `ranking_bundle` | New extension; incumbent source linked below |
| [C11](C11/manifest.json) Generate resampling indices at their consumer | `draw_gather` | New extension; incumbent source linked below |
| [C12](C12/manifest.json) Sparse exact classification counts before dense output | `sparse_counts` | [I24](../performance_ideas/I24/manifest.json) |
| [C13](C13/manifest.json) Shared sufficient statistics across CV folds | `fold_stats`, `logcv_weights` | [I12](../performance_ideas/I12/manifest.json) |
| [C14](C14/manifest.json) Reuse one factorization for multiple right-hand sides | `group_rhs` | [I22](../performance_ideas/I22/manifest.json) |
| [C15](C15/manifest.json) Solve against factors instead of forming inverses | `factor_solve` | New extension; incumbent source linked below |
| [C16](C16/manifest.json) Fuse GLM response, residual and objective production | `glm_fused` | [I12](../performance_ideas/I12/manifest.json) |
| [C17](C17/manifest.json) Batch independent OVR and line-search tasks | `ovr_waves`, `line_search_pairs` | [I12](../performance_ideas/I12/manifest.json) |
| [C18](C18/manifest.json) Coordinate-descent residual tile reuse | `next_residual`, `tile64`, `gram_prefetch` | [I12](../performance_ideas/I12/manifest.json) |
| [C19](C19/manifest.json) Persistent ordered online linear updates | `ordered_128`, `ordered_32` | [I12](../performance_ideas/I12/manifest.json) |
| [C20](C20/manifest.json) Bounded SVM kernel-row cache and shared pair loads | `row_cache`, `pair_load` | New extension; incumbent source linked below |
| [C21](C21/manifest.json) Fuse canonical SMO extrema selection | `extrema_tree4` | New extension; incumbent source linked below |
| [C22](C22/manifest.json) Produce symmetric Gram/kernel triangles once | `symmetric_triangle` | New extension; incumbent source linked below |
| [C23](C23/manifest.json) Stream centered covariance for PCA and discriminants | `centered_panels` | [I22](../performance_ideas/I22/manifest.json) |
| [C24](C24/manifest.json) Versioned TSQR merge tree and panel sizes | `panel8`, `rows2048`, `tree4` | [I22](../performance_ideas/I22/manifest.json) |
| [C25](C25/manifest.json) Reuse randomized projection and decomposition panels | `projection_reuse` | New extension; incumbent source linked below |
| [C26](C26/manifest.json) Reuse NMF sufficient products within each accepted step | `fixed_products`, `update_fused` | New extension; incumbent source linked below |
| [C27](C27/manifest.json) Batch independent ICA and factor-analysis component work | `components`, `factor_components`, `normalize_components` | New extension; incumbent source linked below |
| [C28](C28/manifest.json) Bucket independent sparse-coding and ALS solves | `bucket_solves` | New extension; incumbent source linked below |
| [C29](C29/manifest.json) Streaming exact distance-to-top-k | `stream_topk`, `stream_topk_tile_128` | [A06](../performance_ideas/A06/manifest.json), [I15](../performance_ideas/I15/manifest.json) |
| [C30](C30/manifest.json) Versioned direct-difference distance arithmetic | `direct_distance`, `direct_distance_rows_4` | New extension; incumbent source linked below |
| [C31](C31/manifest.json) Bucket ragged CSR canonicalization by work | `device_buckets` | [I13](../performance_ideas/I13/manifest.json) |
| [C32](C32/manifest.json) Fuse exact radius threshold and compact emission | `count_fusion`, `emit_fusion`, `count_fusion_emit_fusion` | [N08](../performance_ideas/N08/manifest.json) |
| [C33](C33/manifest.json) Freeze first graph convergence state on device | `frozen_chunks`, `frozen_chunks_chunk_4` | [I14](../performance_ideas/I14/manifest.json) |
| [C34](C34/manifest.json) Canonical parallel MST edge selection | `parallel_edges` | New extension; incumbent source linked below |
| [C35](C35/manifest.json) IVF long-list chunking and short-list packing | `packed_lists`, `packed_lists_rows_128` | [I16](../performance_ideas/I16/manifest.json) |
| [C36](C36/manifest.json) Reuse KMeans centroid tiles through assignment | `centroid_tiles`, `centroid_tiles_rows_4` | New extension; incumbent source linked below |
| [C37](C37/manifest.json) Versioned fixed-row centroid accumulation | `row_panels`, `row_panels_panel_128` | [A08](../performance_ideas/A08/manifest.json) |
| [C38](C38/manifest.json) KMeans++ nearest-distance reuse | `reuse_nearest`, `device_potential`, `reuse_nearest_device_potential` | New extension; incumbent source linked below |
| [C39](C39/manifest.json) Retain mini-batch and bisecting clustering state | `retain_state` | New extension; incumbent source linked below |
| [C40](C40/manifest.json) Share MeanShift distance tiles across seed updates | `seed_tiles`, `active_seeds`, `seed_tiles_active_seeds` | New extension; incumbent source linked below |
| [C41](C41/manifest.json) OPTICS fused canonical reachability minima | `fused_minima` | New extension; incumbent source linked below |
| [C42](C42/manifest.json) Compact active agglomerative distance work | `active_triangle` | New extension; incumbent source linked below |
| [C43](C43/manifest.json) Resident normalized graph operators | `resident_normalization` | New extension; incumbent source linked below |
| [C44](C44/manifest.json) Seeded UMAP graph and sampling-state reuse | `sampling_descriptors` | New extension; incumbent source linked below |
| [C45](C45/manifest.json) Fuse tree split statistics and candidate scoring | External TREES IDENTICAL; no implementation here | [I17](../performance_ideas/I17/manifest.json) |
| [C46](C46/manifest.json) Exact histogram sibling subtraction | External TREES IDENTICAL; no implementation here | [I18](../performance_ideas/I18/manifest.json) |
| [C47](C47/manifest.json) Tree frontier tasks by live work and histogram bytes | External TREES IDENTICAL; no implementation here | [A07](../performance_ideas/A07/manifest.json), [I17](../performance_ideas/I17/manifest.json), [N07](../performance_ideas/N07/manifest.json) |
| [C48](C48/manifest.json) Stable tree partition fused with child bookkeeping | External TREES IDENTICAL; no implementation here | New extension; incumbent source linked below |
| [C49](C49/manifest.json) Batch independent forest construction with logical RNG IDs | External TREES IDENTICAL; no implementation here | [I18](../performance_ideas/I18/manifest.json) |
| [C50](C50/manifest.json) Packed forest inference and fixed tree-output reduction | External TREES IDENTICAL; no implementation here | New extension; incumbent source linked below |
| [C51](C51/manifest.json) TreeSHAP shared path metadata and bounded batching | External TREES IDENTICAL; no implementation here | New extension; incumbent source linked below |
| [C52](C52/manifest.json) Versioned stable pair-combine KDE reduction | `pair128`, `pair512` | [I20](../performance_ideas/I20/manifest.json) |
| [C53](C53/manifest.json) GMM reusable centered tiles and fused sufficient statistics | `center4`, `bgmm_center4` | [I21](../performance_ideas/I21/manifest.json) |
| [C54](C54/manifest.json) GP factor and batched prediction reuse | `kernel_tiles4` | New extension; incumbent source linked below |
| [C55](C55/manifest.json) Naive Bayes class-statistic pass fusion | `class_group` | New extension; incumbent source linked below |
| [C56](C56/manifest.json) Discriminant class solves and prediction fusion | `lda_input`, `qda_project4` | New extension; incumbent source linked below |
| [C57](C57/manifest.json) Robust covariance candidate state reuse | `candidate_mean` | New extension; incumbent source linked below |
| [C58](C58/manifest.json) Batch independent classical forecast candidates | `hw_scale_once`, `hw_series4`, `forecast_series4`, `team64` | [I23](../performance_ideas/I23/manifest.json) |
| [C59](C59/manifest.json) ARIMA device-resident order and likelihood trials | `immutable_observations` | [I23](../performance_ideas/I23/manifest.json) |
| [C60](C60/manifest.json) Reuse classical stationarity preparation and lag work | `lag4`, `diff_reuse` | New extension; incumbent source linked below |

### C01 — Versioned fixed-leaf reduction profiles

| Selectable configuration | A compiler defines | B |
|---|---|---|
| `leaf64` | `MOJOLEARN_CLASSICAL_C01_LEAF64` | Incumbent; no new C defines |
| `leaf128` | `MOJOLEARN_CLASSICAL_C01_LEAF128` | Incumbent; no new C defines |

Production source: [x_metrics/common.mojo](../../x_metrics/common.mojo), [core/classical_stats.mojo](../../core/classical_stats.mojo), [core/xtdz_coalesced.mojo](../../core/xtdz_coalesced.mojo), [decomposition/host/pca_oracle.mojo](../../decomposition/host/pca_oracle.mojo).

Caller chain: x_metrics.common.PairSum; x_metrics.plan.plan_program; core.xtdz_coalesced.column_mean_launch; decomposition.host.pca_oracle.host_column_mean_launch.

Harness: [C01/manifest.json](C01/manifest.json) → [full_workload.py](full_workload.py), selected through [tools/performance_ideas.py](../../tools/performance_ideas.py). Saved workload keys: `classical:pca`, `more:tsvd`, `expanded:sgd-reg@regression_report`. Dataset/version/hash, uncapped shape, settings, route admission and consumed-output facts remain pending.

Prior work: I04 proposes a GEMM fold revision; this is a classical scalar-statistic profile, separately gated. [I04](../performance_ideas/I04/manifest.json).

Remaining source/applicability scope:
- GLM objective reducers and non-decomposition column-statistic consumers not migrated to C01.

Status: NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. Runtime reach: **not demonstrated**.

### C02 — Fuse independent column statistics with shared input loads

| Selectable configuration | A compiler defines | B |
|---|---|---|
| `paired_stats` | `MOJOLEARN_CLASSICAL_C02_STATS_PAIR` | Incumbent; no new C defines |
| `linear_pair` | `MOJOLEARN_CLASSICAL_C02_LINEAR_PAIR` | Incumbent; no new C defines |

Production source: [preprocessing/standard.mojo](../../preprocessing/standard.mojo), [glm/impl/center_device.mojo](../../glm/impl/center_device.mojo), [glm/estimator.mojo](../../glm/estimator.mojo), [bindings/_mojolearn_estimators.mojo](../../bindings/_mojolearn_estimators.mojo), [bindings/_mojolearn_estimators_host.mojo](../../bindings/_mojolearn_estimators_host.mojo), [python/mojolearn/linear_model.py](../../python/mojolearn/linear_model.py), [x_decomp/device.mojo](../../x_decomp/device.mojo).

Caller chain: preprocessing.standard.standard_fit_dev -> standard_chunks_kernel[False]; python.mojolearn.linear_model._classical_xy_means -> lm_col_sums_pair bindings; glm.estimator.ols_fit_resident_host / ridge_fit_resident_host -> col_sums_pair_buf; x_decomp.device centered OLS entry -> col_sums_pair_buf.

Harness: [C02/manifest.json](C02/manifest.json) → [full_workload.py](full_workload.py), selected through [tools/performance_ideas.py](../../tools/performance_ideas.py). Saved workload keys: `expanded:standard-scaler`, `classical:ols`, `more:ridge`, `expanded:gaussian-nb`, `classical:pca`. Dataset/version/hash, uncapped shape, settings, route admission and consumed-output facts remain pending.

Prior work: I24 leaves broader preprocessing pass fusion open. [I24](../performance_ideas/I24/manifest.json).

Remaining source/applicability scope:
- GaussianNB/class-stat streams compose with C55. PCA column means use C01, while covariance centered loads use C04/C23; no separate fused PCA mean/variance profile was added.
- Weighted linear means keep incumbent scaling and sums. GPU schedules both exact streams together; host uses its incumbent exact sums under the same new binding.

Status: NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. Runtime reach: **not demonstrated**.

### C03 — Fuse finite checks and extrema production

| Selectable configuration | A compiler defines | B |
|---|---|---|
| `finite_extrema` | `MOJOLEARN_CLASSICAL_C03_FINITE_EXTREMA` | Incumbent; no new C defines |

Production source: [preprocessing/minmax.mojo](../../preprocessing/minmax.mojo), [preprocessing/estimator.mojo](../../preprocessing/estimator.mojo).

Caller chain: preprocessing.estimator.minmax_fit_direct -> _minmax_fit_direct_cls2 -> minmax_fit_flagged_dev.

Harness: [C03/manifest.json](C03/manifest.json) → [full_workload.py](full_workload.py), selected through [tools/performance_ideas.py](../../tools/performance_ideas.py). Saved workload keys: `expanded:minmax-scaler`, `expanded:maxabs-scaler`. Dataset/version/hash, uncapped shape, settings, route admission and consumed-output facts remain pending.

Prior work: Extension of I24. [I24](../performance_ideas/I24/manifest.json).

Remaining source/applicability scope:
- Public MinMax refusal is boolean, with no first-error-index API. Internal canonical first-error-index output is not added.
- MaxAbsScaler has no separate finite-validation pass: its existing NaN-skipping/Inf-accepting extrema semantics are preserved. _finite_2d is an _x2d shell, not an additional scan. No new refusal policy invented.

Status: NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. Runtime reach: **not demonstrated**.

### C04 — Apply centering and scaling at the consumer load

| Selectable configuration | A compiler defines | B |
|---|---|---|
| `load_center` | `MOJOLEARN_CLASSICAL_C04_LOAD_CENTER` | Incumbent; no new C defines |

Production source: [core/classical_centered.mojo](../../core/classical_centered.mojo), [decomposition/impl/linalg/detail/pca.mojo](../../decomposition/impl/linalg/detail/pca.mojo), [decomposition/host/pca_oracle.mojo](../../decomposition/host/pca_oracle.mojo), [x_prep/prims.mojo](../../x_prep/prims.mojo), [x_prep/units.mojo](../../x_prep/units.mojo), [bindings/_mojolearn_x_prep.mojo](../../bindings/_mojolearn_x_prep.mojo), [bindings/_mojolearn_x_prep_host.mojo](../../bindings/_mojolearn_x_prep_host.mojo), [python/mojolearn/_expansion_prep.py](../../python/mojolearn/_expansion_prep.py), [experiments/classical_identical_ideas/metric_workloads.py](../../experiments/classical_identical_ideas/metric_workloads.py).

Caller chain: decomposition.impl.linalg.detail.pca.compute_covariance; decomposition.host.pca_oracle.host_pca_fit; python.mojolearn._expansion_prep.LinearDiscriminantAnalysis.transform -> centered_matmul op178.

Harness: [C04/manifest.json](C04/manifest.json) → [full_workload.py](full_workload.py), selected through [tools/performance_ideas.py](../../tools/performance_ideas.py). Saved workload keys: `classical:pca`, `classical:ols`, `more:ridge`, `expanded:gaussian-nb`, `expanded:lda-clf@lda_outputs`. Dataset/version/hash, uncapped shape, settings, route admission and consumed-output facts remain pending.

Prior work: New extension New extension; incumbent source linked below.

Remaining source/applicability scope:
- PCA incumbent split-K already centers at load, so only its non-split route adds new behavior. LDA(svd) now uses the exact center_rows->matmul seam at load.
- OLS/Ridge/GaussianNB and normalized-distance consumer fusion remain unimplemented; no unsupported-toolchain prerequisite is claimed.

Status: NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. Runtime reach: **not demonstrated**.

### C05 — Retain bounded classical scratch across phases

| Selectable configuration | A compiler defines | B |
|---|---|---|
| `phase_scratch` | `MOJOLEARN_CLASSICAL_C05_PHASE_SCRATCH` | Incumbent; no new C defines |
| `ols_phase` | `MOJOLEARN_CLASSICAL_C05_OLS_PHASE` | Incumbent; no new C defines |

Production source: [x_metrics/plan.mojo](../../x_metrics/plan.mojo), [x_metrics/device.mojo](../../x_metrics/device.mojo), [x_metrics/host/program.mojo](../../x_metrics/host/program.mojo), [glm/estimator.mojo](../../glm/estimator.mojo), [glm/impl/linalg/detail/lstsq.mojo](../../glm/impl/linalg/detail/lstsq.mojo).

Caller chain: x_metrics.plan.plan_program -> run_program_device_ptr/run_program_host_ptr; glm.estimator.ols_fit_host; glm.estimator.ols_fit_resident_host; glm.estimator.ols_fit_weighted_host.

Harness: [C05/manifest.json](C05/manifest.json) → [full_workload.py](full_workload.py), selected through [tools/performance_ideas.py](../../tools/performance_ideas.py). Saved workload keys: `expanded:sgd-reg@regression_report`, `expanded:sgd-clf@ranking_report`, `classical:ols`. Dataset/version/hash, uncapped shape, settings, route admission and consumed-output facts remain pending.

Prior work: I02 covers GEMM scratch; extend only to classical caller-owned buffers. [I02](../performance_ideas/I02/manifest.json).

Remaining source/applicability scope:
- Within-call phase reuse is integrated in the metrics planner. Optional repeated-call caches remain unimplemented; no implicit global ownership or pointer-only cache was introduced.
- Default TSQR does not allocate these Gram/inverse buffers; do not claim its reach. No cross-call cache.

Status: NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. Runtime reach: **not demonstrated**.

### C06 — Batch row norms without changing feature folds

| Selectable configuration | A compiler defines | B |
|---|---|---|
| `rows2` | `MOJOLEARN_CLASSICAL_C06_ROWS2` | Incumbent; no new C defines |
| `rows4` | `MOJOLEARN_CLASSICAL_C06_ROWS4` | Incumbent; no new C defines |

Production source: [core/row_norms.mojo](../../core/row_norms.mojo), [neighbors/impl/detail/knn_brute_force.mojo](../../neighbors/impl/detail/knn_brute_force.mojo), [kde/impl/distance/distance.mojo](../../kde/impl/distance/distance.mojo).

Caller chain: core.row_norms.enqueue_row_norms; neighbors.impl.detail.knn_brute_force.compute_norms/compute_norms_for_metric; kde.impl.distance.distance; cluster.impl.detail.min_cluster_distance_compute; cluster.impl.detail.kmeans row norms (graph agent); neighbors.impl.detail.knn_brute_force ordinary/cosine norms (graph agent); cluster.impl.detail.min_cluster_distance_compute centroid norms (graph agent); hdbscan.impl.cluster.detail.sparse_mr_mst norms (graph agent); kde.impl.distance.distance ordinary/cosine norms (root).

Harness: [C06/manifest.json](C06/manifest.json) → [full_workload.py](full_workload.py), selected through [tools/performance_ideas.py](../../tools/performance_ideas.py). Saved workload keys: `classical:kmeans`, `classical:knn`, `classical:kde`, `classical:hdbscan`, `more:knn-clf`, `more:knn-reg`, `more:ivf`, `expanded:normalizer`. Dataset/version/hash, uncapped shape, settings, route admission and consumed-output facts remain pending.

Prior work: New extension New extension; incumbent source linked below.

Remaining source/applicability scope:
- Normalizer uses a distinct scalar fold and is not migrated. Additional kernel-matrix consumers remain unclaimed. Graph/root handoffs list their exact C06 production sites.

Status: NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. Runtime reach: **not demonstrated**.

### C07 — Stable radix layout for classical preparation

| Selectable configuration | A compiler defines | B |
|---|---|---|
| `digit4` | `MOJOLEARN_CLASSICAL_C07_DIGIT4` | Incumbent; no new C defines |
| `digit6` | `MOJOLEARN_CLASSICAL_C07_DIGIT6` | Incumbent; no new C defines |
| `keys1024` | `MOJOLEARN_CLASSICAL_C07_KEYS1024` | Incumbent; no new C defines |
| `keys4096` | `MOJOLEARN_CLASSICAL_C07_KEYS4096` | Incumbent; no new C defines |

Production source: [x_prep/dradix.mojo](../../x_prep/dradix.mojo), [x_prep/device.mojo](../../x_prep/device.mojo).

Caller chain: x_prep.device.run_program_device_ptr -> radix_sort_cols_device.

Harness: [C07/manifest.json](C07/manifest.json) → [full_workload.py](full_workload.py), selected through [tools/performance_ideas.py](../../tools/performance_ideas.py). Saved workload keys: `expanded:quantile-transformer`, `expanded:kbins`, `expanded:onehot`, `expanded:ordinal`, `expanded:target-encoder`, `expanded:label-encoder`, `expanded:isotonic`. Dataset/version/hash, uncapped shape, settings, route admission and consumed-output facts remain pending.

Prior work: Reuse I19 infrastructure and distinguish new geometries from existing digit-width arms. [I19](../performance_ideas/I19/manifest.json).

Remaining source/applicability scope:
- Ragged-segment bucket scheduling and isotonic route not integrated. Incumbent radix admission threshold retained, no new exact-shape route.

Status: NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. Runtime reach: **not demonstrated**.

### C08 — Reuse category dictionaries and inverse maps

| Selectable configuration | A compiler defines | B |
|---|---|---|
| `dictionary_inverse` | `MOJOLEARN_CLASSICAL_C08_DICTIONARY` | Incumbent; no new C defines |
| `grouped_output` | `MOJOLEARN_CLASSICAL_C08_GROUPED_OUTPUT` | Incumbent; no new C defines |

Production source: [python/mojolearn/_expansion_prep.py](../../python/mojolearn/_expansion_prep.py), [x_prep/prims.mojo](../../x_prep/prims.mojo), [x_prep/units.mojo](../../x_prep/units.mojo), [bindings/_mojolearn_x_prep.mojo](../../bindings/_mojolearn_x_prep.mojo), [bindings/_mojolearn_x_prep_host.mojo](../../bindings/_mojolearn_x_prep_host.mojo), [experiments/classical_identical_ideas/metric_workloads.py](../../experiments/classical_identical_ideas/metric_workloads.py), [x_prep/device.mojo](../../x_prep/device.mojo), [x_prep/host/program.mojo](../../x_prep/host/program.mojo).

Caller chain: OrdinalEncoder.fit_transform -> fit -> _fit_categories_with_codes -> x_prep unique_inverse_unit -> _transform_encoded; OneHotEncoder.fit_transform -> _fit_categories_with_codes -> native count_neg/grouping/onehot; _fit_infrequent consumes fit-owned inverse for Ordinal/OneHot; TargetEncoder._run consumes one canonical unsupervised dictionary/inverse before fold statistics; x_prep.units.run_unit[9] -> onehot_unit; device and host scheduled totals reduced by four.

Harness: [C08/manifest.json](C08/manifest.json) → [full_workload.py](full_workload.py), selected through [tools/performance_ideas.py](../../tools/performance_ideas.py). Saved workload keys: `expanded:onehot@onehot_fit_transform`, `expanded:ordinal@ordinal_fit_transform`, `expanded:target-encoder`, `expanded:label-encoder`, `expanded:categorical-nb`. Dataset/version/hash, uncapped shape, settings, route admission and consumed-output facts remain pending.

Prior work: I19 supplies dictionary primitives; this card concerns caller lifetime and reuse. [I19](../performance_ideas/I19/manifest.json).

Remaining source/applicability scope:
- LabelEncoder numeric fit_transform already builds its dictionary and inverse in one incumbent program; not relabeled new.
- NaiveBayes shared label consumers use existing canonical encoding; class-stat pass reuse tracked under C55. Explicit user-supplied dictionaries stay incumbent.
- Dictionary reuse is independent of grouped emission; only one-hot emission integrated.

Status: NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. Runtime reach: **not demonstrated**.

### C09 — Fused regression and weighted metric reports

| Selectable configuration | A compiler defines | B |
|---|---|---|
| `regression_bundle` | `MOJOLEARN_CLASSICAL_C09_REG_BUNDLE` | Incumbent; no new C defines |

Production source: [python/mojolearn/_expansion_metrics.py](../../python/mojolearn/_expansion_metrics.py), [python/mojolearn/_metrics_impl.py](../../python/mojolearn/_metrics_impl.py), [x_metrics/regression.mojo](../../x_metrics/regression.mojo), [x_metrics/plan.mojo](../../x_metrics/plan.mojo), [x_metrics/units.mojo](../../x_metrics/units.mojo), [experiments/classical_identical_ideas/metric_workloads.py](../../experiments/classical_identical_ideas/metric_workloads.py).

Caller chain: mojolearn.metrics.regression_report -> native adjacent reg_term stages -> x_metrics.plan op82 -> reg_pair_unit.

Harness: [C09/manifest.json](C09/manifest.json) → [full_workload.py](full_workload.py), selected through [tools/performance_ideas.py](../../tools/performance_ideas.py). Saved workload keys: `expanded:sgd-reg@regression_report`. Dataset/version/hash, uncapped shape, settings, route admission and consumed-output facts remain pending.

Prior work: I24 combines classification reports; extend to regression. [I24](../performance_ideas/I24/manifest.json).

Remaining source/applicability scope:
- Separate public scalar calls do not fuse. Full saved estimator selection/report weight settings/attrs unresolved; no standalone metric board recipe exists.

Status: NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. Runtime reach: **not demonstrated**.

### C10 — Reuse stable score sorting for ranking metrics

| Selectable configuration | A compiler defines | B |
|---|---|---|
| `ranking_bundle` | `MOJOLEARN_CLASSICAL_C10_RANK_REUSE` | Incumbent; no new C defines |

Production source: [python/mojolearn/_expansion_metrics.py](../../python/mojolearn/_expansion_metrics.py), [python/mojolearn/_metrics_impl.py](../../python/mojolearn/_metrics_impl.py), [x_metrics/ranking.mojo](../../x_metrics/ranking.mojo), [x_metrics/plan.mojo](../../x_metrics/plan.mojo), [x_metrics/units.mojo](../../x_metrics/units.mojo), [experiments/classical_identical_ideas/metric_workloads.py](../../experiments/classical_identical_ideas/metric_workloads.py).

Caller chain: mojolearn.metrics.ranking_report -> adjacent bin_curve -> planner curve_copy_unit and rank_epi/curve_out tails.

Harness: [C10/manifest.json](C10/manifest.json) → [full_workload.py](full_workload.py), selected through [tools/performance_ideas.py](../../tools/performance_ideas.py). Saved workload keys: `expanded:sgd-clf@ranking_report`. Dataset/version/hash, uncapped shape, settings, route admission and consumed-output facts remain pending.

Prior work: New extension New extension; incumbent source linked below.

Remaining source/applicability scope:
- Multiclass report and nonadjacent public-call reuse not integrated. Tiny/arena-limited planner fallback retains incumbent. Saved classifier binary-class/score-column facts remain pending.

Status: NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. Runtime reach: **not demonstrated**.

### C11 — Generate resampling indices at their consumer

| Selectable configuration | A compiler defines | B |
|---|---|---|
| `draw_gather` | `MOJOLEARN_CLASSICAL_C11_DRAW_GATHER` | Incumbent; no new C defines |

Production source: [resample/gather_fast.mojo](../../resample/gather_fast.mojo), [resample/estimator.mojo](../../resample/estimator.mojo), [python/mojolearn/resample.py](../../python/mojolearn/resample.py).

Caller chain: mojolearn.resample.resample -> _gpu_gather -> binding resample_gather_gpu -> classical_draw_gather_kernel.

Harness: [C11/manifest.json](C11/manifest.json) → [full_workload.py](full_workload.py), selected through [tools/performance_ideas.py](../../tools/performance_ideas.py). Saved workload keys: `expanded:resample`, `expanded:bootstrap`, `expanded:permutation-test`, `expanded:cross-val-score`. Dataset/version/hash, uncapped shape, settings, route admission and consumed-output facts remain pending.

Prior work: New extension New extension; incumbent source linked below.

Remaining source/applicability scope:
- Permutation/test/CV consumers not integrated. Bootstrap statistics already regenerate draws in incumbent (not relabeled new). Forest bootstrap belongs C45-C51 external TREES lane. Host keeps incumbent deterministic index consumer; no new host scheduling optimization.

Status: NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. Runtime reach: **not demonstrated**.

### C12 — Sparse exact classification counts before dense output

| Selectable configuration | A compiler defines | B |
|---|---|---|
| `sparse_counts` | `MOJOLEARN_CLASSICAL_C12_SPARSE_COUNTS` | Incumbent; no new C defines |

Production source: [x_metrics/plan.mojo](../../x_metrics/plan.mojo), [x_metrics/par.mojo](../../x_metrics/par.mojo), [x_metrics/group.mojo](../../x_metrics/group.mojo), [x_metrics/units.mojo](../../x_metrics/units.mojo), [experiments/classical_identical_ideas/metric_workloads.py](../../experiments/classical_identical_ideas/metric_workloads.py).

Caller chain: x_metrics.plan OP_GROUP_SORT -> stable KEY_GROUP sort -> sparse_group_offsets_unit -> existing cm/cls tails; metric_workloads.classical_classification_report -> public cohen_kappa_score -> _confusion -> group_sort.

Harness: [C12/manifest.json](C12/manifest.json) → [full_workload.py](full_workload.py), selected through [tools/performance_ideas.py](../../tools/performance_ideas.py). Saved workload keys: `expanded:sgd-clf@classification_report`, `expanded:gaussian-nb`, `expanded:categorical-nb`. Dataset/version/hash, uncapped shape, settings, route admission and consumed-output facts remain pending.

Prior work: Build on I24 exact report APIs; this is occupied-key accumulation. [I24](../performance_ideas/I24/manifest.json).

Remaining source/applicability scope:
- NaiveBayes class-count caller and explicit sparse public output are not integrated. Dense OFF/public table allocation remains included.
- Source guard is 6*n < m*C+m with scratch capacity; full saved low-cardinality data may retain B. No invented labels or reduced workloads to force admission. Candidate activation remains pending.

Status: NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. Runtime reach: **not demonstrated**.

### C13 — Shared sufficient statistics across CV folds

| Selectable configuration | A compiler defines | B |
|---|---|---|
| `fold_stats` | `MOJOLEARN_CLASSICAL_C13_FOLD_STATS` | Incumbent; no new C defines |
| `logcv_weights` | `MOJOLEARN_CLASSICAL_C13_LOGCV_WEIGHTS` | Incumbent; no new C defines |

Production source: [x_linear/classical_fold_stats.mojo](../../x_linear/classical_fold_stats.mojo), [x_linear/cd.mojo](../../x_linear/cd.mojo), [x_linear/cd_grid.mojo](../../x_linear/cd_grid.mojo), [bindings/_mojolearn_x_linear_host.mojo](../../bindings/_mojolearn_x_linear_host.mojo), [x_linear/ridgecv.mojo](../../x_linear/ridgecv.mojo), [x_linear/device.mojo](../../x_linear/device.mojo), [x_linear/logcv.mojo](../../x_linear/logcv.mojo), [x_linear/logcv_grid.mojo](../../x_linear/logcv_grid.mojo).

Caller chain: enetcv_fit; enetcv_fit_grid; LassoCV.fit; ElasticNetCV.fit; ridge_kfold_fit; _ridge_kfold_grid; logcv_fit; logcv_fit_grid; logistic_objective host/team/grid.

Harness: [C13/manifest.json](C13/manifest.json) → [full_workload.py](full_workload.py), selected through [tools/performance_ideas.py](../../tools/performance_ideas.py). Saved workload keys: `expanded:ridge-cv`, `expanded:lasso-cv`, `expanded:enet-cv`, `expanded:logreg-cv`, `expanded:logreg-cv@weighted_cv`. Dataset/version/hash, uncapped shape, settings, route admission and consumed-output facts remain pending.

Prior work: Explicit remaining I12 Gram/shared-fold proposal. [I12](../performance_ideas/I12/manifest.json).

Remaining source/applicability scope:
- Weighted LogCV saved full-workload recipe variant is pending; unweighted recipe does not exercise cached weight-normalization arithmetic.
- RidgeCV higher-precision fallback retains independent retained-row rescans when a factor is untrusted.
- C13 is a numerical profile and its quality/selected model results remain unverified.

Status: NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. Runtime reach: **not demonstrated**.

### C14 — Reuse one factorization for multiple right-hand sides

| Selectable configuration | A compiler defines | B |
|---|---|---|
| `group_rhs` | `MOJOLEARN_CLASSICAL_C14_GROUP_RHS` | Incumbent; no new C defines |

Production source: [x_decomp/cells.mojo](../../x_decomp/cells.mojo), [x_decomp/device.mojo](../../x_decomp/device.mojo), [x_linear/ops.mojo](../../x_linear/ops.mojo), [x_linear/ridge.mojo](../../x_linear/ridge.mojo), [x_linear/ridge_grid.mojo](../../x_linear/ridge_grid.mojo).

Caller chain: trs_tri_cols; launch_trs_tri; x_decomp repeated LU/triangular solve callers; _ridge_solve_best; t_ridge_solve_best; python/mojolearn/_expansion_linear.py::_ridge_run; RidgeClassifier.fit; RidgeCV refit.

Harness: [C14/manifest.json](C14/manifest.json) → [full_workload.py](full_workload.py), selected through [tools/performance_ideas.py](../../tools/performance_ideas.py). Saved workload keys: `classical:ols`, `more:ridge`, `more:gpr`, `expanded:pls`, `expanded:lstsq`, `expanded:lu-solve`, `expanded:ridge-clf`, `expanded:ridge-cv`. Dataset/version/hash, uncapped shape, settings, route admission and consumed-output facts remain pending.

Prior work: Reuse I22 factor-state work; extend actual multioutput callers without claiming new initial implementation. [I22](../performance_ideas/I22/manifest.json).

Remaining source/applicability scope:
- Primary linear_model.Ridge uses glm ridge_fit; grouped factor application in that route and OLS multioutput remain unfinished source work.
- Gaussian-process caller grouped RHS extension remains unfinished source work.
- Existing single-factor state is incumbent; only new bounded four-RHS applications are claimed.

Status: NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. Runtime reach: **not demonstrated**.

### C15 — Solve against factors instead of forming inverses

| Selectable configuration | A compiler defines | B |
|---|---|---|
| `factor_solve` | `MOJOLEARN_CLASSICAL_C15_FACTOR_SOLVE` | Incumbent; no new C defines |

Production source: [x_linear/bayes.mojo](../../x_linear/bayes.mojo), [x_linear/bayes_grid.mojo](../../x_linear/bayes_grid.mojo).

Caller chain: _ard_coef; _t_ard_coef; ARDRegression.fit.

Harness: [C15/manifest.json](C15/manifest.json) → [full_workload.py](full_workload.py), selected through [tools/performance_ideas.py](../../tools/performance_ideas.py). Saved workload keys: `expanded:bayesian-ridge`, `expanded:ard`, `more:ridge`, `expanded:min-cov-det`, `expanded:elliptic-envelope`. Dataset/version/hash, uncapped shape, settings, route admission and consumed-output facts remain pending.

Prior work: New extension New extension; incumbent source linked below.

Remaining source/applicability scope:
- No new BayesianRidge inverse-elision arm where production already solves factors.
- Gaussian-process inverse-materialization callers need independent integration by owning family.

Status: NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. Runtime reach: **not demonstrated**.

### C16 — Fuse GLM response, residual and objective production

| Selectable configuration | A compiler defines | B |
|---|---|---|
| `glm_fused` | `MOJOLEARN_CLASSICAL_C16_GLM_FUSED` | Incumbent; no new C defines |

Production source: [x_linear/glm.mojo](../../x_linear/glm.mojo), [x_linear/device.mojo](../../x_linear/device.mojo).

Caller chain: _unit_all; _objective_host; glm_obj_map_kernel; _glm_grid_objective; glm_fit; _glm_fit_grid.

Harness: [C16/manifest.json](C16/manifest.json) → [full_workload.py](full_workload.py), selected through [tools/performance_ideas.py](../../tools/performance_ideas.py). Saved workload keys: `more:logreg`, `more:linearsvc`, `more:linearsvr`, `expanded:poisson`, `expanded:gamma`, `expanded:tweedie`. Dataset/version/hash, uncapped shape, settings, route admission and consumed-output facts remain pending.

Prior work: Extends I12 beyond accepted-iterate scheduling. [I12](../performance_ideas/I12/manifest.json).

Remaining source/applicability scope:
- GLM trial derivative state is replaced on each objective and reused only at accepted/current point.
- Other logistic/linear-margin objectives have existing fusion but no distinct new C16 arm.

Status: NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. Runtime reach: **not demonstrated**.

### C17 — Batch independent OVR and line-search tasks

| Selectable configuration | A compiler defines | B |
|---|---|---|
| `ovr_waves` | `MOJOLEARN_CLASSICAL_C17_OVR` | Incumbent; no new C defines |
| `line_search_pairs` | `MOJOLEARN_CLASSICAL_C17_LS_TRIALS` | Incumbent; no new C defines |

Production source: [x_linear/device.mojo](../../x_linear/device.mojo), [glm/impl/qn/qn_linesearch.mojo](../../glm/impl/qn/qn_linesearch.mojo).

Caller chain: sgd_mb_chunk_ovr_kernel; _sgd_mb_grid OVR driver; ls_backtrack_exact_trials.

Harness: [C17/manifest.json](C17/manifest.json) → [full_workload.py](full_workload.py), selected through [tools/performance_ideas.py](../../tools/performance_ideas.py). Saved workload keys: `more:logreg`, `more:linearsvc`, `more:linearsvr`, `expanded:logreg-cv`. Dataset/version/hash, uncapped shape, settings, route admission and consumed-output facts remain pending.

Prior work: Reuse I12 existing OVR and bounded4trial code; new work is full-caller selection/combination mapping. [I12](../performance_ideas/I12/manifest.json).

Remaining source/applicability scope:
- OVR arm changes scheduling waves, not full model-state allocation.
- C17 line-search A permits two exact trials; B retains existing legacy flags and ordinary incumbent line search.
- Do not claim unchanged old I12 OVR/exact-trial experiments as new.

Status: NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. Runtime reach: **not demonstrated**.

### C18 — Coordinate-descent residual tile reuse

| Selectable configuration | A compiler defines | B |
|---|---|---|
| `next_residual` | `MOJOLEARN_CLASSICAL_C18_RESIDUAL_NEXT` | Incumbent; no new C defines |
| `tile64` | `MOJOLEARN_CLASSICAL_C18_TILE64` | Incumbent; no new C defines |
| `gram_prefetch` | `MOJOLEARN_CLASSICAL_C18_GRAM_PREFETCH` | Incumbent; no new C defines |

Production source: [x_linear/cd.mojo](../../x_linear/cd.mojo), [x_linear/cd_grid.mojo](../../x_linear/cd_grid.mojo), [solver/impl/cd.mojo](../../solver/impl/cd.mojo).

Caller chain: enet_gram_cd; t_enet_gram_cd; enetcv_fit; enetcv_fit_grid; cd_fused_step_kernel; cd_idn_gram_sweep_kernel.

Harness: [C18/manifest.json](C18/manifest.json) → [full_workload.py](full_workload.py), selected through [tools/performance_ideas.py](../../tools/performance_ideas.py). Saved workload keys: `more:lasso`, `more:elasticnet`, `expanded:lasso-cv`, `expanded:enet-cv`, `expanded:sparse-coder`. Dataset/version/hash, uncapped shape, settings, route admission and consumed-output facts remain pending.

Prior work: I12 independent-coordinate infrastructure is reusable; this card targets memory passes. [I12](../performance_ideas/I12/manifest.json).

Remaining source/applicability scope:
- Residual window arm requires existing fused row route; resident-Gram arm covers incumbent Gram route.
- Saved full workloads need actual route/provenance receipt; no runtime reach demonstrated.

Status: NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. Runtime reach: **not demonstrated**.

### C19 — Persistent ordered online linear updates

| Selectable configuration | A compiler defines | B |
|---|---|---|
| `ordered_128` | `MOJOLEARN_CLASSICAL_C19_SGD_CHUNK=128` | Incumbent; no new C defines |
| `ordered_32` | `MOJOLEARN_CLASSICAL_C19_SGD_CHUNK=32` | Incumbent; no new C defines |

Production source: [x_linear/device.mojo](../../x_linear/device.mojo).

Caller chain: _sgd_ps_grid; SGD_PS_CHUNK ordered chunk launch.

Harness: [C19/manifest.json](C19/manifest.json) → [full_workload.py](full_workload.py), selected through [tools/performance_ideas.py](../../tools/performance_ideas.py). Saved workload keys: `expanded:sgd-clf`, `expanded:sgd-reg`, `expanded:perceptron`, `expanded:pa-clf`, `expanded:pa-reg`, `expanded:sgd-ocsvm`. Dataset/version/hash, uncapped shape, settings, route admission and consumed-output facts remain pending.

Prior work: I12 supplies SGD fusion; do not duplicate its decided controls. [I12](../performance_ideas/I12/manifest.json).

Remaining source/applicability scope:
- SGDClassifier/Regressor saved recipes use minibatch 4096 and do not reach per-sample candidate; a genuine saved full per-sample recipe is pending.
- Incumbent ordered chunk2048 remains B; no minibatch substitution or changed update counter.

Status: NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. Runtime reach: **not demonstrated**.

### C20 — Bounded SVM kernel-row cache and shared pair loads

| Selectable configuration | A compiler defines | B |
|---|---|---|
| `row_cache` | `MOJOLEARN_CLASSICAL_C20_ROW_CACHE` | Incumbent; no new C defines |
| `pair_load` | `MOJOLEARN_CLASSICAL_C20_PAIR_LOAD` | Incumbent; no new C defines |

Production source: [svm/impl/classical_kernel_cells.mojo](../../svm/impl/classical_kernel_cells.mojo), [svm/impl/classical_kernel_device.mojo](../../svm/impl/classical_kernel_device.mojo), [svm/impl/kernelcache.mojo](../../svm/impl/kernelcache.mojo).

Caller chain: KernelCache.get_square_tile_without_caching; KernelCache.init_full_tile_batching.

Harness: [C20/manifest.json](C20/manifest.json) → [full_workload.py](full_workload.py), selected through [tools/performance_ideas.py](../../tools/performance_ideas.py). Saved workload keys: `classical:svc`, `more:svr`, `expanded:ocsvm`. Dataset/version/hash, uncapped shape, settings, route admission and consumed-output facts remain pending.

Prior work: New extension New extension; incumbent source linked below.

Remaining source/applicability scope:
- Kernel row cache currently serves working-set square tile production, not full update_f batches.
- Unsupported precomputed kernel routes preserve incumbent handling.
- Cache state allocated per immutable fit; no reuse across changed fit parameters.
- OneClassSVM public API uses x_neighbors, so svm SMO/kernel-cache changes do not establish its source reach.

Status: NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. Runtime reach: **not demonstrated**.

### C21 — Fuse canonical SMO extrema selection

| Selectable configuration | A compiler defines | B |
|---|---|---|
| `extrema_tree4` | `MOJOLEARN_CLASSICAL_C21_EXTREMA` | Incumbent; no new C defines |

Production source: [svm/impl/smoblocksolve.mojo](../../svm/impl/smoblocksolve.mojo).

Caller chain: _tree_dual; SMO block solve.

Harness: [C21/manifest.json](C21/manifest.json) → [full_workload.py](full_workload.py), selected through [tools/performance_ideas.py](../../tools/performance_ideas.py). Saved workload keys: `classical:svc`, `more:svr`, `expanded:ocsvm`. Dataset/version/hash, uncapped shape, settings, route admission and consumed-output facts remain pending.

Prior work: New extension New extension; incumbent source linked below.

Remaining source/applicability scope:
- Incumbent already fuses min/max; the new arm is only the lower-barrier canonical four-way tree schedule.
- Host canonical extrema result is unchanged; cross-vendor tie/NaN outcomes remain unverified.
- OneClassSVM public API uses x_neighbors, so svm SMO/kernel-cache changes do not establish its source reach.

Status: NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. Runtime reach: **not demonstrated**.

### C22 — Produce symmetric Gram/kernel triangles once

| Selectable configuration | A compiler defines | B |
|---|---|---|
| `symmetric_triangle` | `MOJOLEARN_CLASSICAL_C22_TRIANGLE` | Incumbent; no new C defines |

Production source: [svm/impl/classical_kernel_cells.mojo](../../svm/impl/classical_kernel_cells.mojo), [svm/impl/classical_kernel_device.mojo](../../svm/impl/classical_kernel_device.mojo), [svm/impl/distance/kernel_matrices.mojo](../../svm/impl/distance/kernel_matrices.mojo).

Caller chain: kernel_op same-buffer square dispatch; KernelRidge km_kernel_matrix; SVC/SVR working-set square tile.

Harness: [C22/manifest.json](C22/manifest.json) → [full_workload.py](full_workload.py), selected through [tools/performance_ideas.py](../../tools/performance_ideas.py). Saved workload keys: `more:kernel-ridge`, `expanded:kernel-pca`, `classical:svc`, `more:svr`, `expanded:ocsvm`. Dataset/version/hash, uncapped shape, settings, route admission and consumed-output facts remain pending.

Prior work: New extension New extension; incumbent source linked below.

Remaining source/applicability scope:
- Other x_decomp Gram consumers are not routed through svm kernel_op.
- Poly/sigmoid direct kernel_methods triangles are incumbent and not claimed new.
- Affected full kernel matrix recipes have caps; full intended workload coverage stays pending.
- KernelPCA uses separate x_neighbors kernel routines and is not reached by svm kernel_op.

Status: NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. Runtime reach: **not demonstrated**.

### C23 — Stream centered covariance for PCA and discriminants

| Selectable configuration | A compiler defines | B |
|---|---|---|
| `centered_panels` | `MOJOLEARN_CLASSICAL_C23_CENTERED_PANELS` | Incumbent; no new C defines |

Production source: [x_decomp/classical_cells.mojo](../../x_decomp/classical_cells.mojo), [x_decomp/classical_device.mojo](../../x_decomp/classical_device.mojo), [x_decomp/exec_trait.mojo](../../x_decomp/exec_trait.mojo), [x_decomp/host.mojo](../../x_decomp/host.mojo), [x_decomp/device.mojo](../../x_decomp/device.mojo), [x_decomp/kit.mojo](../../x_decomp/kit.mojo), [x_decomp/kit_device.mojo](../../x_decomp/kit_device.mojo), [decomposition/impl/linalg/detail/pca.mojo](../../decomposition/impl/linalg/detail/pca.mojo), [decomposition/host/pca_oracle.mojo](../../decomposition/host/pca_oracle.mojo), [x_decomp/mcd.mojo](../../x_decomp/mcd.mojo).

Caller chain: centered_gram_cell; Exec.classical_centered_gram; Kit.classical_centered_gram; DKit.classical_centered_gram; PCA.compute_covariance/host_pca_fit; MCD.emp_cov/emp_cov_at (root integration).

Harness: [C23/manifest.json](C23/manifest.json) → [full_workload.py](full_workload.py), selected through [tools/performance_ideas.py](../../tools/performance_ideas.py). Saved workload keys: `classical:pca`, `expanded:incremental-pca`, `expanded:min-cov-det`, `expanded:elliptic-envelope`, `expanded:lda-clf`, `expanded:qda`. Dataset/version/hash, uncapped shape, settings, route admission and consumed-output facts remain pending.

Prior work: Related to I22 blocked linalg, but a separate covariance numerical profile. [I22](../performance_ideas/I22/manifest.json).

Remaining source/applicability scope:
- QDA/LDA arena covariance callers need shared-cell migration; C55 class-stat profile is separate.
- PCA and MCD wiring performed by shared/root owners; ledger records source coordination only.
- IncrementalPCA centered incremental covariance caller remains unfinished source work.

Status: NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. Runtime reach: **not demonstrated**.

### C24 — Versioned TSQR merge tree and panel sizes

| Selectable configuration | A compiler defines | B |
|---|---|---|
| `panel8` | `MOJOLEARN_CLASSICAL_C24_PANEL8` | Incumbent; no new C defines |
| `rows2048` | `MOJOLEARN_CLASSICAL_C24_ROWS2048` | Incumbent; no new C defines |
| `tree4` | `MOJOLEARN_CLASSICAL_C24_TREE4` | Incumbent; no new C defines |

Production source: [x_decomp/tsqr_core.mojo](../../x_decomp/tsqr_core.mojo), [x_decomp/tsqr_host.mojo](../../x_decomp/tsqr_host.mojo), [x_decomp/tsqr_device.mojo](../../x_decomp/tsqr_device.mojo), [x_decomp/qr_sliced.mojo](../../x_decomp/qr_sliced.mojo).

Caller chain: ts_factor_host/ts_apply_host; ts_factor_device/ts_apply_device; qs_tree/qs_pair_tree; linalg QR/SVD/lstsq and OLS factor callers.

Harness: [C24/manifest.json](C24/manifest.json) → [full_workload.py](full_workload.py), selected through [tools/performance_ideas.py](../../tools/performance_ideas.py). Saved workload keys: `classical:ols`, `classical:pca`, `more:tsvd`, `expanded:lstsq`, `expanded:qr`, `expanded:svd`, `expanded:randomized-svd`. Dataset/version/hash, uncapped shape, settings, route admission and consumed-output facts remain pending.

Prior work: Explicit unimplemented profile extension of I22. [I22](../performance_ideas/I22/manifest.json).

Remaining source/applicability scope:
- Global TSQR uses must remain classical callers; no neural application is introduced.
- Panel-size numeric version quality/rank/sign behavior remains wholly unverified.

Status: NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. Runtime reach: **not demonstrated**.

### C25 — Reuse randomized projection and decomposition panels

| Selectable configuration | A compiler defines | B |
|---|---|---|
| `projection_reuse` | `MOJOLEARN_CLASSICAL_C25_PROJECTION_REUSE` | Incumbent; no new C defines |

Production source: [kernel_methods/rbf_fused.mojo](../../kernel_methods/rbf_fused.mojo), [kernel_methods/estimator.mojo](../../kernel_methods/estimator.mojo).

Caller chain: classical_projection_kernel; _rbf_idn_fused_launch; RBFSampler.transform public binding.

Harness: [C25/manifest.json](C25/manifest.json) → [full_workload.py](full_workload.py), selected through [tools/performance_ideas.py](../../tools/performance_ideas.py). Saved workload keys: `more:tsvd`, `expanded:randomized-svd`, `expanded:gaussian-rp`, `expanded:sparse-rp`, `more:nystroem`, `more:rbf-sampler`. Dataset/version/hash, uncapped shape, settings, route admission and consumed-output facts remain pending.

Prior work: New extension New extension; incumbent source linked below.

Remaining source/applicability scope:
- Random projection, Nyström and randomized SVD production callers still need distinct panel-reuse arms.
- Existing host canonical GEMM supplies intended same-version projection words; numerical identity not established.
- These broader production extensions are unfinished source work, not an asserted Mojo/Modular prerequisite.

Status: NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. Runtime reach: **not demonstrated**.

### C26 — Reuse NMF sufficient products within each accepted step

| Selectable configuration | A compiler defines | B |
|---|---|---|
| `fixed_products` | `MOJOLEARN_CLASSICAL_C26_PRODUCTS` | Incumbent; no new C defines |
| `update_fused` | `MOJOLEARN_CLASSICAL_C26_UPDATE_FUSED` | Incumbent; no new C defines |

Production source: [x_decomp/cells.mojo](../../x_decomp/cells.mojo), [x_decomp/nmf.mojo](../../x_decomp/nmf.mojo), [x_decomp/nmf_dev.mojo](../../x_decomp/nmf_dev.mojo).

Caller chain: mu_fit/mu_fit_dev; cd_fit/cd_fit_dev; NMF.fit and transform bindings.

Harness: [C26/manifest.json](C26/manifest.json) → [full_workload.py](full_workload.py), selected through [tools/performance_ideas.py](../../tools/performance_ideas.py). Saved workload keys: `expanded:nmf`, `expanded:dict-learning`, `expanded:mb-dict-learning`. Dataset/version/hash, uncapped shape, settings, route admission and consumed-output facts remain pending.

Prior work: New extension New extension; incumbent source linked below.

Remaining source/applicability scope:
- Product arm is reached for update_h=False/transform; fit with mutable H must recompute dependency products.
- Saved NMF default beta2 does not reach beta0/1 pointwise arm; full saved parameter variants pending.
- Nonnegative dictionary-learning callers not changed.

Status: NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. Runtime reach: **not demonstrated**.

### C27 — Batch independent ICA and factor-analysis component work

| Selectable configuration | A compiler defines | B |
|---|---|---|
| `components` | `MOJOLEARN_CLASSICAL_C27_COMPONENTS` | Incumbent; no new C defines |
| `factor_components` | `MOJOLEARN_CLASSICAL_C27_FA_COMPONENTS` | Incumbent; no new C defines |
| `normalize_components` | `MOJOLEARN_CLASSICAL_C27_NORM_VECTOR` | Incumbent; no new C defines |

Production source: [x_decomp/classical_cells.mojo](../../x_decomp/classical_cells.mojo), [x_decomp/classical_device.mojo](../../x_decomp/classical_device.mojo), [x_decomp/exec_trait.mojo](../../x_decomp/exec_trait.mojo), [x_decomp/host.mojo](../../x_decomp/host.mojo), [x_decomp/device.mojo](../../x_decomp/device.mojo), [x_decomp/kit.mojo](../../x_decomp/kit.mojo), [x_decomp/kit_device.mojo](../../x_decomp/kit_device.mojo), [x_decomp/ica.mojo](../../x_decomp/ica.mojo), [x_decomp/ica_dev.mojo](../../x_decomp/ica_dev.mojo), [x_decomp/cells.mojo](../../x_decomp/cells.mojo), [x_decomp/fa_em.mojo](../../x_decomp/fa_em.mojo), [x_decomp/fa_em_dev.mojo](../../x_decomp/fa_em_dev.mojo), [x_decomp/pls.mojo](../../x_decomp/pls.mojo), [x_decomp/pls_dev.mojo](../../x_decomp/pls_dev.mojo).

Caller chain: contrast_pair; Exec.classical_contrast; ica_g/ica_g_dev; FastICA.fit parallel and deflation paths; fa_em; fa_em_dev; power; power_dev.

Harness: [C27/manifest.json](C27/manifest.json) → [full_workload.py](full_workload.py), selected through [tools/performance_ideas.py](../../tools/performance_ideas.py). Saved workload keys: `expanded:fastica`, `expanded:factor-analysis`, `expanded:cca`, `expanded:pls`, `expanded:pls-canonical`. Dataset/version/hash, uncapped shape, settings, route admission and consumed-output facts remain pending.

Prior work: New extension New extension; incumbent source linked below.

Remaining source/applicability scope:
- New component maps retain existing dependent deflation/spectral order; they do not parallelize dependent ICA/PLS components.
- Additional centered projection tile reuse remains a broader design extension, with no toolchain prerequisite asserted.

Status: NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. Runtime reach: **not demonstrated**.

### C28 — Bucket independent sparse-coding and ALS solves

| Selectable configuration | A compiler defines | B |
|---|---|---|
| `bucket_solves` | `MOJOLEARN_CLASSICAL_C28_BUCKET_SOLVES` | Incumbent; no new C defines |

Production source: [x_decomp/device.mojo](../../x_decomp/device.mojo), [x_decomp/als_dev.mojo](../../x_decomp/als_dev.mojo), [x_decomp/kit_device.mojo](../../x_decomp/kit_device.mojo), [x_decomp/dictl.mojo](../../x_decomp/dictl.mojo), [x_decomp/dictl_dev.mojo](../../x_decomp/dictl_dev.mojo).

Caller chain: als_bucket_keys_kernel; launch_als_rows; als_block_kernel; ImplicitALS.fit exact row solver use_cg=False; launch_classical_code_rows; DevExec.lasso_rows/lars_rows/omp_rows; DKit._code_rows; SparseCoder transform; DictionaryLearning sparse_encode.

Harness: [C28/manifest.json](C28/manifest.json) → [full_workload.py](full_workload.py), selected through [tools/performance_ideas.py](../../tools/performance_ideas.py). Saved workload keys: `expanded:sparse-coder`, `expanded:dict-learning`, `expanded:mb-dict-learning`, `expanded:als`. Dataset/version/hash, uncapped shape, settings, route admission and consumed-output facts remain pending.

Prior work: New extension New extension; incumbent source linked below.

Remaining source/applicability scope:
- ALS CG route remains incumbent; saved exact-solve recipe specifies use_cg=False.
- Descriptor scans and radix scratch belong to whole-operation cost.
- Sparse-code rows share immutable Gram allocation; no claim of explicit shared-memory Gram tiling.

Status: NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. Runtime reach: **not demonstrated**.

### C29 — Streaming exact distance-to-top-k

| Selectable configuration | A compiler defines | B |
|---|---|---|
| `stream_topk` | `MOJOLEARN_C29_STREAM_TOPK` | Incumbent; no new C defines |
| `stream_topk_tile_128` | `MOJOLEARN_C29_STREAM_TOPK`, `MOJOLEARN_C29_TILE=128` | Incumbent; no new C defines |

Production source: [neighbors/impl/detail/knn_brute_force.mojo](../../neighbors/impl/detail/knn_brute_force.mojo), [neighbors/impl/detail/classical_stream_topk.mojo](../../neighbors/impl/detail/classical_stream_topk.mojo), [neighbors/estimator.mojo](../../neighbors/estimator.mojo), [core/knn_host_predict.mojo](../../core/knn_host_predict.mojo).

Harness: [C29/manifest.json](C29/manifest.json) → [full_workload.py](full_workload.py), selected through [tools/performance_ideas.py](../../tools/performance_ideas.py). Saved workload keys: `classical:knn`, `more:knn-clf`, `more:knn-reg`, `expanded:lof`, `expanded:label-propagation`, `expanded:label-spreading`, `expanded:isomap`, `expanded:lle`. Dataset/version/hash, uncapped shape, settings, route admission and consumed-output facts remain pending.

Prior work: Reuse I15/A06 selection machinery; split new full-caller tile/merge arms from prior experiments. [A06](../performance_ideas/A06/manifest.json), [I15](../performance_ideas/I15/manifest.json).

Remaining source/applicability scope:
- Expanded Euclidean row-major brute force entry is supported. Other metrics and layouts keep incumbent. Caller scratch allocation is retained even though candidate selection uses output-sized merge state. Full-recipe classifier/regressor variants and exact graph routes require declared settings and runtime reach evidence.

Status: NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. Runtime reach: **not demonstrated**.

### C30 — Versioned direct-difference distance arithmetic

| Selectable configuration | A compiler defines | B |
|---|---|---|
| `direct_distance` | `MOJOLEARN_C30_DIRECT_DISTANCE` | Incumbent; no new C defines |
| `direct_distance_rows_4` | `MOJOLEARN_C30_DIRECT_DISTANCE`, `MOJOLEARN_C30_ROWS_4` | Incumbent; no new C defines |

Production source: [core/classical_distance.mojo](../../core/classical_distance.mojo), [core/knn_host_predict.mojo](../../core/knn_host_predict.mojo), [neighbors/impl/detail/knn_brute_force.mojo](../../neighbors/impl/detail/knn_brute_force.mojo), [cluster/impl/detail/classical_assignment.mojo](../../cluster/impl/detail/classical_assignment.mojo), [cluster/host/kmeans_oracle.mojo](../../cluster/host/kmeans_oracle.mojo), [neighbors/impl/ball_cover/common.mojo](../../neighbors/impl/ball_cover/common.mojo), [dbscan/impl/neighbors/epsilon_neighborhood.mojo](../../dbscan/impl/neighbors/epsilon_neighborhood.mojo), [dbscan/host/dbscan_oracle.mojo](../../dbscan/host/dbscan_oracle.mojo), [cluster/impl/detail/kmeans_transform.mojo](../../cluster/impl/detail/kmeans_transform.mojo), [cluster/checks/plus_plus.mojo](../../cluster/checks/plus_plus.mojo), [cluster/impl/detail/kmeans.mojo](../../cluster/impl/detail/kmeans.mojo), [ivf/impl/neighbors/ivf_flat/identical_ivf_scan.mojo](../../ivf/impl/neighbors/ivf_flat/identical_ivf_scan.mojo), [ivf/impl/neighbors/ivf_flat/ivf_balanced_tasks.mojo](../../ivf/impl/neighbors/ivf_flat/ivf_balanced_tasks.mojo), [ivf/host/ivf_host.mojo](../../ivf/host/ivf_host.mojo), [hierarchy/checks/linkage_oracle.mojo](../../hierarchy/checks/linkage_oracle.mojo), [hdbscan/host/hdbscan_host_oracle.mojo](../../hdbscan/host/hdbscan_host_oracle.mojo), [hdbscan/impl/cluster/detail/sparse_mr_mst.mojo](../../hdbscan/impl/cluster/detail/sparse_mr_mst.mojo).

Harness: [C30/manifest.json](C30/manifest.json) → [full_workload.py](full_workload.py), selected through [tools/performance_ideas.py](../../tools/performance_ideas.py). Saved workload keys: `classical:knn`, `classical:kmeans`, `classical:kde`, `classical:dbscan`, `classical:hdbscan`, `more:agglomerative`, `more:ivf`, `more:knn-clf`, `more:knn-reg`, `more:kernel-ridge`, `expanded:kernel-pca`, `expanded:radius-neighbors`. Dataset/version/hash, uncapped shape, settings, route admission and consumed-output facts remain pending.

Prior work: New extension New extension; incumbent source linked below.

Remaining source/applicability scope:
- Direct separate-product ascending FTZ fold applies to supported Euclidean graph/clustering calls, KMeans init/transform and IVF scan/coarse calls. KDE integration owned root; GP/RBF and SVM owned root/linear. Non-Euclidean routes retain incumbent contracts. Runtime coverage across transitive callers remains undemonstrated.

Status: NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. Runtime reach: **not demonstrated**.

### C31 — Bucket ragged CSR canonicalization by work

| Selectable configuration | A compiler defines | B |
|---|---|---|
| `device_buckets` | `MOJOLEARN_C31_DEVICE_BUCKETS` | Incumbent; no new C defines |

Production source: [neighbors/checks/ball_cover_canonical_order.mojo](../../neighbors/checks/ball_cover_canonical_order.mojo), [neighbors/impl/ball_cover/registers.mojo](../../neighbors/impl/ball_cover/registers.mojo).

Harness: [C31/manifest.json](C31/manifest.json) → [full_workload.py](full_workload.py), selected through [tools/performance_ideas.py](../../tools/performance_ideas.py). Saved workload keys: `expanded:radius-neighbors`, `classical:dbscan`, `classical:hdbscan`. Dataset/version/hash, uncapped shape, settings, route admission and consumed-output facts remain pending.

Prior work: Follow-up to I13 stable canonical merge, not its initial implementation. [I13](../performance_ideas/I13/manifest.json).

Remaining source/applicability scope:
- RBC canonicalization is source-integrated. HDBSCAN sparse mutual-reachability route does not use this CSR canonicalizer; only a recipe explicitly using RBC graph preparation can reach this candidate.

Status: NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. Runtime reach: **not demonstrated**.

### C32 — Fuse exact radius threshold and compact emission

| Selectable configuration | A compiler defines | B |
|---|---|---|
| `count_fusion` | `MOJOLEARN_C32_COUNT_FUSION` | Incumbent; no new C defines |
| `emit_fusion` | `MOJOLEARN_C32_EMIT_FUSION` | Incumbent; no new C defines |
| `count_fusion_emit_fusion` | `MOJOLEARN_C32_COUNT_FUSION`, `MOJOLEARN_C32_EMIT_FUSION` | Incumbent; no new C defines |

Production source: [neighbors/impl/ball_cover/registers.mojo](../../neighbors/impl/ball_cover/registers.mojo), [neighbors/estimator.mojo](../../neighbors/estimator.mojo), [dbscan/impl/runner.mojo](../../dbscan/impl/runner.mojo).

Harness: [C32/manifest.json](C32/manifest.json) → [full_workload.py](full_workload.py), selected through [tools/performance_ideas.py](../../tools/performance_ideas.py). Saved workload keys: `expanded:radius-neighbors`, `classical:dbscan`. Dataset/version/hash, uncapped shape, settings, route admission and consumed-output facts remain pending.

Prior work: Reuse N08; extend vendor-neutral caller integration. [N08](../performance_ideas/N08/manifest.json).

Remaining source/applicability scope:
- Candidate retains canonical sorting; fused distances to final returned radius distances are still a separate pass. Brute-force DBSCAN already fuses threshold; new route is RBC exhaustive exact count/emission.

Status: NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. Runtime reach: **not demonstrated**.

### C33 — Freeze first graph convergence state on device

| Selectable configuration | A compiler defines | B |
|---|---|---|
| `frozen_chunks` | `MOJOLEARN_C33_FROZEN_CHUNKS` | Incumbent; no new C defines |
| `frozen_chunks_chunk_4` | `MOJOLEARN_C33_FROZEN_CHUNKS`, `MOJOLEARN_C33_CHUNK_4` | Incumbent; no new C defines |

Production source: [x_neighbors/iter_device.mojo](../../x_neighbors/iter_device.mojo), [dbscan/impl/sparse/detail/csr.mojo](../../dbscan/impl/sparse/detail/csr.mojo), [dbscan/impl/label/merge_labels.mojo](../../dbscan/impl/label/merge_labels.mojo).

Harness: [C33/manifest.json](C33/manifest.json) → [full_workload.py](full_workload.py), selected through [tools/performance_ideas.py](../../tools/performance_ideas.py). Saved workload keys: `classical:dbscan`, `expanded:connected-components`, `expanded:label-propagation`, `expanded:label-spreading`. Dataset/version/hash, uncapped shape, settings, route admission and consumed-output facts remain pending.

Prior work: Reuse I14 convergence machinery; include full graph callers and interactions. [I14](../performance_ideas/I14/manifest.json).

Remaining source/applicability scope:
- Dense/CSR ConnectedComponents and DBSCAN first-stop state source-integrated. Dense LP uses its canonical dense product under the new chunk arm; existing compact kNN LP already freezes first convergence and exposes changed chunk cadence. Default chunk16 equals incumbent compact-kNN cadence; chunk4 is the distinct cadence subarm there.

Status: NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. Runtime reach: **not demonstrated**.

### C34 — Canonical parallel MST edge selection

| Selectable configuration | A compiler defines | B |
|---|---|---|
| `parallel_edges` | `MOJOLEARN_C34_PARALLEL_EDGES` | Incumbent; no new C defines |

Production source: [hierarchy/impl/sparse/solver/detail/mst_kernels.mojo](../../hierarchy/impl/sparse/solver/detail/mst_kernels.mojo), [hierarchy/impl/sparse/solver/mst_solver.mojo](../../hierarchy/impl/sparse/solver/mst_solver.mojo), [hdbscan/impl/cluster/detail/sparse_mr_mst.mojo](../../hdbscan/impl/cluster/detail/sparse_mr_mst.mojo).

Harness: [C34/manifest.json](C34/manifest.json) → [full_workload.py](full_workload.py), selected through [tools/performance_ideas.py](../../tools/performance_ideas.py). Saved workload keys: `classical:hdbscan`, `more:agglomerative`. Dataset/version/hash, uncapped shape, settings, route admission and consumed-output facts remain pending.

Prior work: New extension New extension; incumbent source linked below.

Remaining source/applicability scope:
- Sparse HDBSCAN component edge reduction uses canonical weight/lo/hi minimum with one block per component and scans candidate rows. Work may grow quadratically in components; no performance claim. Hierarchy uses 128-lane canonical row reduction.

Status: NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. Runtime reach: **not demonstrated**.

### C35 — IVF long-list chunking and short-list packing

| Selectable configuration | A compiler defines | B |
|---|---|---|
| `packed_lists` | `MOJOLEARN_C35_PACKED_LISTS` | Incumbent; no new C defines |
| `packed_lists_rows_128` | `MOJOLEARN_C35_PACKED_LISTS`, `MOJOLEARN_C35_ROWS_128` | Incumbent; no new C defines |

Production source: [ivf/impl/neighbors/ivf_flat/ivf_balanced_tasks.mojo](../../ivf/impl/neighbors/ivf_flat/ivf_balanced_tasks.mojo), [ivf/impl/neighbors/ivf_flat/ivf_flat_search.mojo](../../ivf/impl/neighbors/ivf_flat/ivf_flat_search.mojo), [ivf/resident.mojo](../../ivf/resident.mojo).

Harness: [C35/manifest.json](C35/manifest.json) → [full_workload.py](full_workload.py), selected through [tools/performance_ideas.py](../../tools/performance_ideas.py). Saved workload keys: `more:ivf`. Dataset/version/hash, uncapped shape, settings, route admission and consumed-output facts remain pending.

Prior work: Extend I16 scheduling with recorded scope; no duplication of already supplied staging arms. [I16](../performance_ideas/I16/manifest.json).

Remaining source/applicability scope:
- Existing KM and scan-dimension dispatch admission preserved; reach for unsupported large k/dim stays incumbent.

Status: NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. Runtime reach: **not demonstrated**.

### C36 — Reuse KMeans centroid tiles through assignment

| Selectable configuration | A compiler defines | B |
|---|---|---|
| `centroid_tiles` | `MOJOLEARN_C36_CENTROID_TILES` | Incumbent; no new C defines |
| `centroid_tiles_rows_4` | `MOJOLEARN_C36_CENTROID_TILES`, `MOJOLEARN_C36_ROWS_4` | Incumbent; no new C defines |

Production source: [cluster/impl/detail/classical_assignment.mojo](../../cluster/impl/detail/classical_assignment.mojo), [cluster/impl/detail/min_cluster_distance_compute.mojo](../../cluster/impl/detail/min_cluster_distance_compute.mojo), [x_cluster/device_ops.mojo](../../x_cluster/device_ops.mojo).

Harness: [C36/manifest.json](C36/manifest.json) → [full_workload.py](full_workload.py), selected through [tools/performance_ideas.py](../../tools/performance_ideas.py). Saved workload keys: `classical:kmeans`, `expanded:minibatch-kmeans`, `expanded:bisecting-kmeans`, `more:ivf`. Dataset/version/hash, uncapped shape, settings, route admission and consumed-output facts remain pending.

Prior work: New extension New extension; incumbent source linked below.

Remaining source/applicability scope:
- Two/four independent sample chains reuse centroid feature loads in main assignment and x_cluster nearest/mini-batch assignment. Large/register-pressure performance and runtime reach remain unmeasured.

Status: NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. Runtime reach: **not demonstrated**.

### C37 — Versioned fixed-row centroid accumulation

| Selectable configuration | A compiler defines | B |
|---|---|---|
| `row_panels` | `MOJOLEARN_C37_ROW_PANELS` | Incumbent; no new C defines |
| `row_panels_panel_128` | `MOJOLEARN_C37_ROW_PANELS`, `MOJOLEARN_C37_PANEL_128` | Incumbent; no new C defines |

Production source: [core/classical_centroid.mojo](../../core/classical_centroid.mojo), [cluster/impl/detail/classical_centroid.mojo](../../cluster/impl/detail/classical_centroid.mojo), [cluster/impl/detail/kmeans.mojo](../../cluster/impl/detail/kmeans.mojo), [cluster/host/kmeans_oracle.mojo](../../cluster/host/kmeans_oracle.mojo), [x_cluster/minibatch_cells.mojo](../../x_cluster/minibatch_cells.mojo), [x_cluster/minibatch.mojo](../../x_cluster/minibatch.mojo), [x_cluster/device_ops.mojo](../../x_cluster/device_ops.mojo), [x_cluster/host/host_ops.mojo](../../x_cluster/host/host_ops.mojo).

Harness: [C37/manifest.json](C37/manifest.json) → [full_workload.py](full_workload.py), selected through [tools/performance_ideas.py](../../tools/performance_ideas.py). Saved workload keys: `classical:kmeans`, `expanded:minibatch-kmeans`, `expanded:bisecting-kmeans`, `more:ivf`. Dataset/version/hash, uncapped shape, settings, route admission and consumed-output facts remain pending.

Prior work: A08 tested a geometry candidate; reuse evidence only when source/profile actually match. [A08](../performance_ideas/A08/manifest.json).

Remaining source/applicability scope:
- Main KMeans host still computes incumbent accumulators before replacing centroids with panel profile; source contract agrees but removed-work optimization is incomplete on host. MiniBatch online update uses ordered row panels merged into retained prior center/count; partial-fit weighted/reassignment semantics retained. Bisecting and IVF training call the migrated main KMeans source.

Status: NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. Runtime reach: **not demonstrated**.

### C38 — KMeans++ nearest-distance reuse

| Selectable configuration | A compiler defines | B |
|---|---|---|
| `reuse_nearest` | `MOJOLEARN_C38_REUSE_NEAREST` | Incumbent; no new C defines |
| `device_potential` | `MOJOLEARN_C38_DEVICE_POTENTIAL` | Incumbent; no new C defines |
| `reuse_nearest_device_potential` | `MOJOLEARN_C38_REUSE_NEAREST`, `MOJOLEARN_C38_DEVICE_POTENTIAL` | Incumbent; no new C defines |

Production source: [cluster/impl/detail/kmeans.mojo](../../cluster/impl/detail/kmeans.mojo), [x_cluster/common.mojo](../../x_cluster/common.mojo), [x_cluster/device_ops.mojo](../../x_cluster/device_ops.mojo), [x_cluster/host/host_ops.mojo](../../x_cluster/host/host_ops.mojo), [x_cluster/ops.mojo](../../x_cluster/ops.mojo).

Harness: [C38/manifest.json](C38/manifest.json) → [full_workload.py](full_workload.py), selected through [tools/performance_ideas.py](../../tools/performance_ideas.py). Saved workload keys: `classical:kmeans`, `more:ivf`, `expanded:bisecting-kmeans`. Dataset/version/hash, uncapped shape, settings, route admission and consumed-output facts remain pending.

Prior work: Existing incremental init/PSI switches need selective recipe coverage; do not reinvent them. New extension; incumbent source linked below.

Remaining source/applicability scope:
- REUSE_NEAREST has distinct duplicate-trial distance-row reuse in x_cluster generic init (all original RNG draws and potential order retained). DEVICE_POTENTIAL selects existing device-only greedy trial-adopt schedule for main IDENTICAL KMeans, newly selectable here. Scalable/default init may already retain nearest/device potential; recipes must select an affected initializer to claim reach.

Status: NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. Runtime reach: **not demonstrated**.

### C39 — Retain mini-batch and bisecting clustering state

| Selectable configuration | A compiler defines | B |
|---|---|---|
| `retain_state` | `MOJOLEARN_C39_RETAIN_STATE` | Incumbent; no new C defines |

Production source: [x_cluster/minibatch.mojo](../../x_cluster/minibatch.mojo), [x_cluster/bisect.mojo](../../x_cluster/bisect.mojo), [x_cluster/ops.mojo](../../x_cluster/ops.mojo), [x_cluster/device_ops.mojo](../../x_cluster/device_ops.mojo), [x_cluster/host/host_ops.mojo](../../x_cluster/host/host_ops.mojo).

Harness: [C39/manifest.json](C39/manifest.json) → [full_workload.py](full_workload.py), selected through [tools/performance_ideas.py](../../tools/performance_ideas.py). Saved workload keys: `expanded:minibatch-kmeans`, `expanded:bisecting-kmeans`. Dataset/version/hash, uncapped shape, settings, route admission and consumed-output facts remain pending.

Prior work: New extension New extension; incumbent source linked below.

Remaining source/applicability scope:
- Weighted bisect subsets retain incumbent host gather; unweighted subset arenas and minibatch rollback histories wired.

Status: NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. Runtime reach: **not demonstrated**.

### C40 — Share MeanShift distance tiles across seed updates

| Selectable configuration | A compiler defines | B |
|---|---|---|
| `seed_tiles` | `MOJOLEARN_C40_SEED_TILES` | Incumbent; no new C defines |
| `active_seeds` | `MOJOLEARN_C40_ACTIVE_SEEDS` | Incumbent; no new C defines |
| `seed_tiles_active_seeds` | `MOJOLEARN_C40_SEED_TILES`, `MOJOLEARN_C40_ACTIVE_SEEDS` | Incumbent; no new C defines |

Production source: [x_cluster/meanshift_idn.mojo](../../x_cluster/meanshift_idn.mojo), [x_cluster/device_ops.mojo](../../x_cluster/device_ops.mojo).

Harness: [C40/manifest.json](C40/manifest.json) → [full_workload.py](full_workload.py), selected through [tools/performance_ideas.py](../../tools/performance_ideas.py). Saved workload keys: `expanded:meanshift`. Dataset/version/hash, uncapped shape, settings, route admission and consumed-output facts remain pending.

Prior work: New extension New extension; incumbent source linked below.

Remaining source/applicability scope:
- Two-seed feature staging admits at most512 features by fixed shared storage; larger rows retain single-seed arithmetic. ACTIVE_SEEDS builds stable descriptors each shift; launch upper bound remains ns with inert excess blocks because no unsupported indirect launch is introduced.

Status: NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. Runtime reach: **not demonstrated**.

### C41 — OPTICS fused canonical reachability minima

| Selectable configuration | A compiler defines | B |
|---|---|---|
| `fused_minima` | `MOJOLEARN_C41_FUSED_MINIMA` | Incumbent; no new C defines |

Production source: [x_cluster/device_post.mojo](../../x_cluster/device_post.mojo), [x_cluster/device_ops.mojo](../../x_cluster/device_ops.mojo), [x_cluster/optics.mojo](../../x_cluster/optics.mojo).

Harness: [C41/manifest.json](C41/manifest.json) → [full_workload.py](full_workload.py), selected through [tools/performance_ideas.py](../../tools/performance_ideas.py). Saved workload keys: `expanded:optics`. Dataset/version/hash, uncapped shape, settings, route admission and consumed-output facts remain pending.

Prior work: New extension New extension; incumbent source linked below.

Remaining source/applicability scope:
- New one-visit relaxation/minimum in existing resident full distance graph; approximate ordering excluded.

Status: NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. Runtime reach: **not demonstrated**.

### C42 — Compact active agglomerative distance work

| Selectable configuration | A compiler defines | B |
|---|---|---|
| `active_triangle` | `MOJOLEARN_C42_ACTIVE_TRIANGLE` | Incumbent; no new C defines |

Production source: [x_cluster/device_ops.mojo](../../x_cluster/device_ops.mojo), [x_cluster/agglo.mojo](../../x_cluster/agglo.mojo).

Harness: [C42/manifest.json](C42/manifest.json) → [full_workload.py](full_workload.py), selected through [tools/performance_ideas.py](../../tools/performance_ideas.py). Saved workload keys: `more:agglomerative@active_linkage`. Dataset/version/hash, uncapped shape, settings, route admission and consumed-output facts remain pending.

Prior work: New extension New extension; incumbent source linked below.

Remaining source/applicability scope:
- Packed triangle is built from initial square and square storage is then released; preparation still includes full initial square. Saved hierarchy single-linkage recipe uses another production path; independent x_cluster AgglomerativeClustering full recipe remains pending.

Status: NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. Runtime reach: **not demonstrated**.

### C43 — Resident normalized graph operators

| Selectable configuration | A compiler defines | B |
|---|---|---|
| `resident_normalization` | `MOJOLEARN_C43_RESIDENT_NORMALIZATION` | Incumbent; no new C defines |

Production source: [spectral/impl/sparse/linalg/detail/laplacian.mojo](../../spectral/impl/sparse/linalg/detail/laplacian.mojo), [spectral/impl/sparse/solver/detail/lanczos.mojo](../../spectral/impl/sparse/solver/detail/lanczos.mojo), [x_neighbors/graph_par.mojo](../../x_neighbors/graph_par.mojo), [x_neighbors/lp_knn.mojo](../../x_neighbors/lp_knn.mojo), [x_neighbors/iter_device.mojo](../../x_neighbors/iter_device.mojo), [x_neighbors/iter_host.mojo](../../x_neighbors/iter_host.mojo), [python/mojolearn/_expansion_neighbors.py](../../python/mojolearn/_expansion_neighbors.py), [x_neighbors/classical_graph.mojo](../../x_neighbors/classical_graph.mojo), [bindings/_mojolearn_x_neighbors.mojo](../../bindings/_mojolearn_x_neighbors.mojo), [bindings/_mojolearn_x_neighbors_host.mojo](../../bindings/_mojolearn_x_neighbors_host.mojo).

Harness: [C43/manifest.json](C43/manifest.json) → [full_workload.py](full_workload.py), selected through [tools/performance_ideas.py](../../tools/performance_ideas.py). Saved workload keys: `more:spectral-embedding`, `more:spectral`, `expanded:label-propagation`, `expanded:label-spreading`, `expanded:pagerank`, `expanded:isomap`, `expanded:lle`. Dataset/version/hash, uncapped shape, settings, route admission and consumed-output facts remain pending.

Prior work: New extension New extension; incumbent source linked below.

Remaining source/applicability scope:
- Spectral, PageRank, compact kNN LabelPropagation, and dense LabelPropagation/Spreading raw-affinity bindings/consumers are source-integrated. Dense raw affinity keeps its dense topology and resident degree array; consumer computes the same quotient(s) before its fixed fold. Compact kNN LabelSpreading retains its normalized value format. Isomap/LLE use separate x_decomp dense representations without DeviceCoo/CSR normalization consumers.

Status: NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. Runtime reach: **not demonstrated**.

### C44 — Seeded UMAP graph and sampling-state reuse

| Selectable configuration | A compiler defines | B |
|---|---|---|
| `sampling_descriptors` | `MOJOLEARN_C44_SAMPLING_DESCRIPTORS` | Incumbent; no new C defines |

Production source: [umap/optimizer_identical_device.mojo](../../umap/optimizer_identical_device.mojo), [umap/sparse_optimizer.mojo](../../umap/sparse_optimizer.mojo), [umap/optimizer.mojo](../../umap/optimizer.mojo).

Harness: [C44/manifest.json](C44/manifest.json) → [full_workload.py](full_workload.py), selected through [tools/performance_ideas.py](../../tools/performance_ideas.py). Saved workload keys: `more:umap`. Dataset/version/hash, uncapped shape, settings, route admission and consumed-output facts remain pending.

Prior work: New extension New extension; incumbent source linked below.

Remaining source/applicability scope:
- Per-fit per-epoch descriptor arena is edges*(negative_sample_rate+1) Int32 cells and reused across epochs. Immutable graph already resident. Cross-fit cache requires explicit graph ownership/version invalidation API; no cache across unrelated fits is introduced.

Status: NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. Runtime reach: **not demonstrated**.

### C45 — Fuse tree split statistics and candidate scoring

Owned by **TREES IDENTICAL**. Plan/catalog entry retained without tree runtime edits. Builds on I17 frontier scheduling; different pass-fusion hypothesis.

[I17](../performance_ideas/I17/manifest.json).

### C46 — Exact histogram sibling subtraction

Owned by **TREES IDENTICAL**. Plan/catalog entry retained without tree runtime edits. Reuse I18 implementation; extend applicability only with explicit accumulator proofs.

[I18](../performance_ideas/I18/manifest.json).

### C47 — Tree frontier tasks by live work and histogram bytes

Owned by **TREES IDENTICAL**. Plan/catalog entry retained without tree runtime edits. Extends I17/A07/N07; attribute new geometry independently of existing candidates.

[A07](../performance_ideas/A07/manifest.json), [I17](../performance_ideas/I17/manifest.json), [N07](../performance_ideas/N07/manifest.json).

### C48 — Stable tree partition fused with child bookkeeping

Owned by **TREES IDENTICAL**. Plan/catalog entry retained without tree runtime edits. New extension

New extension; incumbent source linked below.

### C49 — Batch independent forest construction with logical RNG IDs

Owned by **TREES IDENTICAL**. Plan/catalog entry retained without tree runtime edits. I18 multi-tree frontier was still open; distinguish construction from inference batching.

[I18](../performance_ideas/I18/manifest.json).

### C50 — Packed forest inference and fixed tree-output reduction

Owned by **TREES IDENTICAL**. Plan/catalog entry retained without tree runtime edits. New extension

New extension; incumbent source linked below.

### C51 — TreeSHAP shared path metadata and bounded batching

Owned by **TREES IDENTICAL**. Plan/catalog entry retained without tree runtime edits. New extension

New extension; incumbent source linked below.

### C52 — Versioned stable pair-combine KDE reduction

| Selectable configuration | A compiler defines | B |
|---|---|---|
| `pair128` | `MOJOLEARN_C52_PAIR_ROWS=128` | Incumbent; no new C defines |
| `pair512` | `MOJOLEARN_C52_PAIR_ROWS=512` | Incumbent; no new C defines |

Production source: [kde/pair_lse.mojo](../../kde/pair_lse.mojo), [kde/impl/neighbors/kernel_density.mojo](../../kde/impl/neighbors/kernel_density.mojo), [kde/host/kde_oracle.mojo](../../kde/host/kde_oracle.mojo).

Caller chain: KernelDensity.score_samples; logsumexp_kernel; kde_chunk_lse_reduce_kernel; kde_chunk_lse_reduce_rowmax_kernel; oracle_logsumexp_row; _kde_lse_row_chunked.

Harness: [C52/manifest.json](C52/manifest.json) → [full_workload.py](full_workload.py), selected through [tools/performance_ideas.py](../../tools/performance_ideas.py). Saved workload keys: `classical:kde`. Dataset/version/hash, uncapped shape, settings, route admission and consumed-output facts remain pending.

Prior work: I20 schedule work exists; this is its separate arithmetic-profile proposal. [I20](../performance_ideas/I20/manifest.json).

Remaining source/applicability scope:
- All kernel/metric/weighted public variants need resolved full-dataset recipes and future reach evidence.
- Legacy explicitly disabled chunk-fold configurations are separate from the incumbent default A/B.

Status: NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. Runtime reach: **not demonstrated**.

### C53 — GMM reusable centered tiles and fused sufficient statistics

| Selectable configuration | A compiler defines | B |
|---|---|---|
| `center4` | `MOJOLEARN_C53_CENTER4` | Incumbent; no new C defines |
| `bgmm_center4` | `MOJOLEARN_C53_BGMM_STATS` | Incumbent; no new C defines |

Production source: [mixture/checks/mstep.mojo](../../mixture/checks/mstep.mojo), [x_cluster/device_ops.mojo](../../x_cluster/device_ops.mojo).

Caller chain: GaussianMixture.fit -> M-step center_pair_kernel[4]; BayesianGaussianMixture.fit -> DeviceOps._moments_gemm.

Harness: [C53/manifest.json](C53/manifest.json) → [full_workload.py](full_workload.py), selected through [tools/performance_ideas.py](../../tools/performance_ideas.py). Saved workload keys: `more:gmm`, `expanded:bayesian-gmm`. Dataset/version/hash, uncapped shape, settings, route admission and consumed-output facts remain pending.

Prior work: Reuse I21 centered-load/fusion work and retained loser evidence. [I21](../performance_ideas/I21/manifest.json).

Remaining source/applicability scope:
- Four-component staging requires the existing 16 MiB scratch budget; larger panels retain B. Full-recipe route admission is pending.
- General tiled full-size centering and additional fused sufficient-statistic arms are not implemented; do not treat CENTER4 as statistic fusion.

Status: NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. Runtime reach: **not demonstrated**.

### C54 — GP factor and batched prediction reuse

| Selectable configuration | A compiler defines | B |
|---|---|---|
| `kernel_tiles4` | `MOJOLEARN_C54_PREDICT_TILES` | Incumbent; no new C defines |

Production source: [gaussian_process/checks/kernels.mojo](../../gaussian_process/checks/kernels.mojo), [x_neighbors/iter_device.mojo](../../x_neighbors/iter_device.mojo).

Caller chain: GaussianProcessRegressor fit/predict; GaussianProcessClassifier fit/predict; GP likelihood optimizer -> gp_kernel_matrix -> gp_rbf_kernel/gp_matern_kernel; SVGP.predict -> op_svgp_predict -> _launch_scaled_rbf(prediction=True) -> _c54_svgp_predict_kernel.

Harness: [C54/manifest.json](C54/manifest.json) → [full_workload.py](full_workload.py), selected through [tools/performance_ideas.py](../../tools/performance_ideas.py). Saved workload keys: `more:gpr`, `more:gpc`, `expanded:svgp`. Dataset/version/hash, uncapped shape, settings, route admission and consumed-output facts remain pending.

Prior work: New extension New extension; incumbent source linked below.

Remaining source/applicability scope:
- C54 shares RBF/Matern/inducing-kernel inputs across four independent columns. Persistent fitted-factor ownership across public calls remains unimplemented; the pointer API still reuploads factors. No pointer-address cache is fabricated.
- GP mean, variance, predictive probabilities and optimizer paths require separate resolved full-workload settings; all runtime reach remains undemonstrated.

Status: NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. Runtime reach: **not demonstrated**.

### C55 — Naive Bayes class-statistic pass fusion

| Selectable configuration | A compiler defines | B |
|---|---|---|
| `class_group` | `MOJOLEARN_CLASSICAL_C55_CLASS_GROUP` | Incumbent; no new C defines |

Production source: [x_prep/prims.mojo](../../x_prep/prims.mojo), [x_prep/host/program.mojo](../../x_prep/host/program.mojo), [python/mojolearn/_expansion_prep.py](../../python/mojolearn/_expansion_prep.py), [bindings/_mojolearn_x_prep.mojo](../../bindings/_mojolearn_x_prep.mojo), [bindings/_mojolearn_x_prep_host.mojo](../../bindings/_mojolearn_x_prep_host.mojo).

Caller chain: GaussianNB/discrete NB _class_stats -> class_stats/class_stats_w -> class_stats_group_unit; host _groups bypass ensures candidate unit.

Harness: [C55/manifest.json](C55/manifest.json) → [full_workload.py](full_workload.py), selected through [tools/performance_ideas.py](../../tools/performance_ideas.py). Saved workload keys: `expanded:gaussian-nb`, `expanded:multinomial-nb`, `expanded:bernoulli-nb`, `expanded:complement-nb`, `expanded:categorical-nb`. Dataset/version/hash, uncapped shape, settings, route admission and consumed-output facts remain pending.

Prior work: New extension New extension; incumbent source linked below.

Remaining source/applicability scope:
- CategoricalNB histogram route not changed. CSR fast native NB callers may bypass _class_stats and are not claimed.
- Feature-selection and discriminant direct class_stats consumers also see the shared C55 unit; include their full recipes when their call paths select it.

Status: NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. Runtime reach: **not demonstrated**.

### C56 — Discriminant class solves and prediction fusion

| Selectable configuration | A compiler defines | B |
|---|---|---|
| `lda_input` | `MOJOLEARN_CLASSICAL_C56_LDA_INPUT` | Incumbent; no new C defines |
| `qda_project4` | `MOJOLEARN_C56_QDA_PROJECT4` | Incumbent; no new C defines |

Production source: [experiments/classical_identical_ideas/shared_controls.mojo](../../experiments/classical_identical_ideas/shared_controls.mojo), [bindings/_mojolearn_x_prep.mojo](../../bindings/_mojolearn_x_prep.mojo), [bindings/_mojolearn_x_prep_host.mojo](../../bindings/_mojolearn_x_prep_host.mojo), [python/mojolearn/_expansion_prep.py](../../python/mojolearn/_expansion_prep.py), [experiments/classical_identical_ideas/metric_workloads.py](../../experiments/classical_identical_ideas/metric_workloads.py), [naive_bayes/da.mojo](../../naive_bayes/da.mojo).

Caller chain: LinearDiscriminantAnalysis.decision_function binary branch -> one native matmul -> same coefficient-difference answer; QuadraticDiscriminantAnalysis.decision_function/predict_proba -> x_prep qda_dec_unit.

Harness: [C56/manifest.json](C56/manifest.json) → [full_workload.py](full_workload.py), selected through [tools/performance_ideas.py](../../tools/performance_ideas.py). Saved workload keys: `expanded:lda-clf@lda_outputs`, `expanded:qda`. Dataset/version/hash, uncapped shape, settings, route admission and consumed-output facts remain pending.

Prior work: New extension New extension; incumbent source linked below.

Remaining source/applicability scope:
- Candidate applies only to two-class decision_function; multiclass path is incumbent. Saved class count is pending. Ordinary classifier runner predict/proba does not exercise it; use explicit adapter.
- Class factorization/solve batching is not implemented; existing per-class rank thresholds remain intact.
- LDA input-reuse subarm is owned by shared_implementation.json and consolidated into this ID by the source index.

Status: NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. Runtime reach: **not demonstrated**.

### C57 — Robust covariance candidate state reuse

| Selectable configuration | A compiler defines | B |
|---|---|---|
| `candidate_mean` | `MOJOLEARN_C57_CANDIDATE_STATE` | Incumbent; no new C defines |

Production source: [x_decomp/mcd.mojo](../../x_decomp/mcd.mojo), [x_decomp/kit_device.mojo](../../x_decomp/kit_device.mojo).

Caller chain: MinCovDet.fit; EllipticEnvelope.fit; Mcd.c_step/DMcd.c_step -> emp_cov_at.

Harness: [C57/manifest.json](C57/manifest.json) → [full_workload.py](full_workload.py), selected through [tools/performance_ideas.py](../../tools/performance_ideas.py). Saved workload keys: `expanded:min-cov-det`, `expanded:elliptic-envelope`. Dataset/version/hash, uncapped shape, settings, route admission and consumed-output facts remain pending.

Prior work: New extension New extension; incumbent source linked below.

Remaining source/applicability scope:
- Retains a candidate subset mean rather than recomputing it for covariance. Independent robust candidates are not yet scheduled together.
- C23 interaction uses the common centered Gram panel profile; complete determinant/support/quality evidence pending.

Status: NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. Runtime reach: **not demonstrated**.

### C58 — Batch independent classical forecast candidates

| Selectable configuration | A compiler defines | B |
|---|---|---|
| `hw_scale_once` | `MOJOLEARN_C58_SHARED_PREP` | Incumbent; no new C defines |
| `hw_series4` | `MOJOLEARN_C58_SERIES4` | Incumbent; no new C defines |
| `forecast_series4` | `MOJOLEARN_C58_FORECAST4` | Incumbent; no new C defines |
| `team64` | `MOJOLEARN_C58_TEAM_MIB=64` | Incumbent; no new C defines |

Production source: [holtwinters/impl/internal/hw_estimate.mojo](../../holtwinters/impl/internal/hw_estimate.mojo), [holtwinters/impl/internal/hw_estimate_launch.mojo](../../holtwinters/impl/internal/hw_estimate_launch.mojo), [sequence/exec_device.mojo](../../sequence/exec_device.mojo), [sequence/fit_team_py.mojo](../../sequence/fit_team_py.mojo), [sequence/ets_team_py.mojo](../../sequence/ets_team_py.mojo).

Caller chain: ExponentialSmoothing.fit estimated -> holtwinters_estimate_gpu; Theta family -> theta_py -> DeviceExec.launch[OP_THETA] -> seq_kernel; ETS/GARCH scalar fallback -> seq_kernel; GARCH, ETS(A,A/Ad,N), Prophet fit -> _group -> owned team launch loop.

Harness: [C58/manifest.json](C58/manifest.json) → [full_workload.py](full_workload.py), selected through [tools/performance_ideas.py](../../tools/performance_ideas.py). Saved workload keys: `more:ets`, `expanded:theta`, `expanded:optimized-theta`, `expanded:dynamic-theta`, `expanded:dynamic-optimized-theta`, `expanded:auto-theta`, `expanded:damped-ets`, `expanded:garch`, `expanded:prophet`. Dataset/version/hash, uncapped shape, settings, route admission and consumed-output facts remain pending.

Prior work: Use I23 independent-work infrastructure; do not duplicate existing forecast arms. [I23](../performance_ideas/I23/manifest.json).

Remaining source/applicability scope:
- HW scale sharing applies parallel start trials; HW series4 applies the serial series route. Each route needs admission recorded from the saved settings.
- C58_FORECAST4 applies only scalar OP_THETA/OP_ETS/OP_GARCH. Cooperative GARCH/ETS/Prophet routes have the separate TEAM64 bounded state-budget arm; neither arm changes optimizer trial scheduling inside a series.
- Long grouped scalar fits have no demonstrated device completion, especially command-buffer lifetime; no vendor-specific workaround introduced.

Status: NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. Runtime reach: **not demonstrated**.

### C59 — ARIMA device-resident order and likelihood trials

| Selectable configuration | A compiler defines | B |
|---|---|---|
| `immutable_observations` | `MOJOLEARN_C59_TRIAL_STATE` | Incumbent; no new C defines |

Production source: [arima/impl/fast_eval_ws.mojo](../../arima/impl/fast_eval_ws.mojo), [arima/impl/batched_kalman.mojo](../../arima/impl/batched_kalman.mojo).

Caller chain: ARIMA/SARIMA likelihood-gradient fit -> FastEvalWS -> loglike_prepared -> fast_kalman_into -> batched_kalman_loop_kernel; AutoARIMA candidate ARIMA fits.

Harness: [C59/manifest.json](C59/manifest.json) → [full_workload.py](full_workload.py), selected through [tools/performance_ideas.py](../../tools/performance_ideas.py). Saved workload keys: `more:arima`, `expanded:autoarima`. Dataset/version/hash, uncapped shape, settings, route admission and consumed-output facts remain pending.

Prior work: Follow-up to I23 batching; profile-changing affine scans remain separate future work. [I23](../performance_ideas/I23/manifest.json).

Remaining source/applicability scope:
- Workspace admission remains the incumbent eval_ws_fits threshold; a recipe outside it keeps B.
- The independent gradient trial observation copy is removed; AutoARIMA order-search scheduling remains incumbent.
- Parent arena/CSS constructors retain their original allocation and input indexing.

Status: NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. Runtime reach: **not demonstrated**.

### C60 — Reuse classical stationarity preparation and lag work

| Selectable configuration | A compiler defines | B |
|---|---|---|
| `lag4` | `MOJOLEARN_C60_LAG4` | Incumbent; no new C defines |
| `diff_reuse` | `MOJOLEARN_C60_DIFF_REUSE` | Incumbent; no new C defines |

Production source: [tsa/impl/timeSeries/stationarity.mojo](../../tsa/impl/timeSeries/stationarity.mojo), [tsa/impl/select_d_fast.mojo](../../tsa/impl/select_d_fast.mojo).

Caller chain: kpss_test -> kpss_one_wait -> s2B_accumulation_kernel; AutoARIMA/select_d -> select_d_fast; classical_second_diff_reuse.

Harness: [C60/manifest.json](C60/manifest.json) → [full_workload.py](full_workload.py), selected through [tools/performance_ideas.py](../../tools/performance_ideas.py). Saved workload keys: `expanded:autoarima`. Dataset/version/hash, uncapped shape, settings, route admission and consumed-output facts remain pending.

Prior work: New extension New extension; incumbent source linked below.

Remaining source/applicability scope:
- Standalone KPSS/select_d full recipe is pending; no standalone saved board lane found.
- DIFF_REUSE needs D=0 and a requested d=2 trial; the standard d_max=2 recipe stops before that trial, so that setting does not exercise the subarm.
- ADF is not supported by this AutoARIMA API (test must be kpss); no ADF or autocorrelation runtime is invented.
- Fusing multiple requested reports requires a supported multi-report public interface; current KPSS call returns one test.

Status: NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. Runtime reach: **not demonstrated**.

## Programmed interactions

All combinations below have explicit A defines and incumbent B defines in their linked manifests. They require the union of their mapped full workloads. Mutually exclusive leaf/row/digit profiles are not combined. The complete configuration is an experiment only, remains OFF, and is not a proposed promotion without evidence.

| Selector ID | Configuration | Component arms |
|---|---|---|
| [C01](C01/manifest.json) | `interaction_fold_fusion` | C01/leaf64, C02/combined, C09/regression_bundle, C13/combined, C23/centered_panels, C37/combined, C52/pair128 |
| [C01](C01/manifest.json) | `interaction_c01_c02` | C01/leaf64, C02/paired_stats |
| [C01](C01/manifest.json) | `interaction_c01_c09` | C01/leaf64, C09/regression_bundle |
| [C01](C01/manifest.json) | `interaction_c01_c13` | C01/leaf64, C13/fold_stats |
| [C01](C01/manifest.json) | `interaction_c01_c23` | C01/leaf64, C23/centered_panels |
| [C01](C01/manifest.json) | `interaction_c01_c37` | C01/leaf64, C37/row_panels |
| [C01](C01/manifest.json) | `interaction_c01_c52` | C01/leaf64, C52/pair128 |
| [C01](C01/manifest.json) | `profile_c01_leaf64_c52_pair128` | C01/leaf64, C52/pair128 |
| [C01](C01/manifest.json) | `profile_c01_leaf64_c52_pair512` | C01/leaf64, C52/pair512 |
| [C01](C01/manifest.json) | `profile_c01_leaf128_c52_pair128` | C01/leaf128, C52/pair128 |
| [C01](C01/manifest.json) | `profile_c01_leaf128_c52_pair512` | C01/leaf128, C52/pair512 |
| [C01](C01/manifest.json) | `profile_c01_leaf64_c37_row_panels` | C01/leaf64, C37/row_panels |
| [C01](C01/manifest.json) | `profile_c01_leaf64_c37_row_panels_panel_128` | C01/leaf64, C37/row_panels_panel_128 |
| [C01](C01/manifest.json) | `profile_c01_leaf128_c37_row_panels` | C01/leaf128, C37/row_panels |
| [C01](C01/manifest.json) | `profile_c01_leaf128_c37_row_panels_panel_128` | C01/leaf128, C37/row_panels_panel_128 |
| [C01](C01/manifest.json) | `complete_classical` | C01/leaf64, C02/combined, C03/finite_extrema, C04/load_center, C05/combined, C06/rows2, C07/combined, C08/combined, C09/regression_bundle, C10/ranking_bundle, C11/draw_gather, C12/sparse_counts, C13/combined, C14/group_rhs, C15/factor_solve, C16/glm_fused, C17/combined, C18/combined, C19/ordered_128, C20/combined, C21/extrema_tree4, C22/symmetric_triangle, C23/centered_panels, C24/combined, C25/projection_reuse, C26/combined, C27/combined, C28/bucket_solves, C29/combined, C30/combined, C31/device_buckets, C32/combined, C33/combined, C34/parallel_edges, C35/combined, C36/combined, C37/combined, C38/combined, C39/retain_state, C40/combined, C41/fused_minima, C42/active_triangle, C43/resident_normalization, C44/sampling_descriptors, C52/pair128, C53/combined, C54/kernel_tiles4, C55/class_group, C56/combined, C57/candidate_mean, C58/combined, C59/immutable_observations, C60/combined |
| [C02](C02/manifest.json) | `combo_paired_stats_linear_pair` | C02/paired_stats, C02/linear_pair |
| [C02](C02/manifest.json) | `combined` | C02/paired_stats, C02/linear_pair |
| [C02](C02/manifest.json) | `interaction_covariance_consumers` | C02/combined, C04/load_center, C14/group_rhs, C23/centered_panels, C53/combined, C54/kernel_tiles4, C55/class_group, C56/combined, C57/candidate_mean |
| [C02](C02/manifest.json) | `interaction_c02_c04` | C02/paired_stats, C04/load_center |
| [C02](C02/manifest.json) | `interaction_c02_c14` | C02/paired_stats, C14/group_rhs |
| [C02](C02/manifest.json) | `interaction_c02_c23` | C02/paired_stats, C23/centered_panels |
| [C02](C02/manifest.json) | `interaction_c02_c53` | C02/paired_stats, C53/center4 |
| [C02](C02/manifest.json) | `interaction_c02_c54` | C02/paired_stats, C54/kernel_tiles4 |
| [C02](C02/manifest.json) | `interaction_c02_c55` | C02/paired_stats, C55/class_group |
| [C02](C02/manifest.json) | `interaction_c02_c56` | C02/paired_stats, C56/lda_input |
| [C02](C02/manifest.json) | `interaction_c02_c57` | C02/paired_stats, C57/candidate_mean |
| [C03](C03/manifest.json) | `interaction_preparation_lifetime` | C03/finite_extrema, C04/load_center, C05/combined, C08/combined |
| [C03](C03/manifest.json) | `interaction_c03_c04` | C03/finite_extrema, C04/load_center |
| [C03](C03/manifest.json) | `interaction_c03_c05` | C03/finite_extrema, C05/phase_scratch |
| [C03](C03/manifest.json) | `interaction_c03_c08` | C03/finite_extrema, C08/dictionary_inverse |
| [C05](C05/manifest.json) | `combo_phase_scratch_ols_phase` | C05/phase_scratch, C05/ols_phase |
| [C05](C05/manifest.json) | `combined` | C05/phase_scratch, C05/ols_phase |
| [C06](C06/manifest.json) | `interaction_distance_selection` | C06/rows2, C29/combined, C30/combined, C32/combined, C36/combined |
| [C06](C06/manifest.json) | `interaction_c06_c29` | C06/rows2, C29/stream_topk |
| [C06](C06/manifest.json) | `interaction_c06_c30` | C06/rows2, C30/direct_distance |
| [C06](C06/manifest.json) | `interaction_c06_c32` | C06/rows2, C32/count_fusion |
| [C06](C06/manifest.json) | `interaction_c06_c36` | C06/rows2, C36/centroid_tiles |
| [C06](C06/manifest.json) | `profile_c06_rows2_c30_direct_distance` | C06/rows2, C30/direct_distance |
| [C06](C06/manifest.json) | `profile_c06_rows2_c30_direct_distance_rows_4` | C06/rows2, C30/direct_distance_rows_4 |
| [C06](C06/manifest.json) | `profile_c06_rows4_c30_direct_distance` | C06/rows4, C30/direct_distance |
| [C06](C06/manifest.json) | `profile_c06_rows4_c30_direct_distance_rows_4` | C06/rows4, C30/direct_distance_rows_4 |
| [C07](C07/manifest.json) | `combo_digit4_keys1024` | C07/digit4, C07/keys1024 |
| [C07](C07/manifest.json) | `combo_digit6_keys1024` | C07/digit6, C07/keys1024 |
| [C07](C07/manifest.json) | `combo_digit4_keys4096` | C07/digit4, C07/keys4096 |
| [C07](C07/manifest.json) | `combo_digit6_keys4096` | C07/digit6, C07/keys4096 |
| [C07](C07/manifest.json) | `combined` | C07/digit4, C07/keys1024 |
| [C07](C07/manifest.json) | `interaction_canonical_keys` | C07/combined, C08/combined, C10/ranking_bundle, C12/sparse_counts, C31/device_buckets |
| [C07](C07/manifest.json) | `interaction_c07_c08` | C07/digit4, C08/dictionary_inverse |
| [C07](C07/manifest.json) | `interaction_c07_c10` | C07/digit4, C10/ranking_bundle |
| [C07](C07/manifest.json) | `interaction_c07_c12` | C07/digit4, C12/sparse_counts |
| [C07](C07/manifest.json) | `interaction_c07_c31` | C07/digit4, C31/device_buckets |
| [C08](C08/manifest.json) | `combo_dictionary_inverse_grouped_output` | C08/dictionary_inverse, C08/grouped_output |
| [C08](C08/manifest.json) | `combined` | C08/dictionary_inverse, C08/grouped_output |
| [C13](C13/manifest.json) | `combo_fold_stats_logcv_weights` | C13/fold_stats, C13/logcv_weights |
| [C13](C13/manifest.json) | `combined` | C13/fold_stats, C13/logcv_weights |
| [C13](C13/manifest.json) | `interaction_statistics_solve` | C13/combined, C14/group_rhs, C15/factor_solve, C18/combined, C23/centered_panels, C24/combined |
| [C13](C13/manifest.json) | `interaction_c13_c14` | C13/fold_stats, C14/group_rhs |
| [C13](C13/manifest.json) | `interaction_c13_c15` | C13/fold_stats, C15/factor_solve |
| [C13](C13/manifest.json) | `interaction_c13_c18` | C13/fold_stats, C18/next_residual |
| [C13](C13/manifest.json) | `interaction_c13_c23` | C13/fold_stats, C23/centered_panels |
| [C13](C13/manifest.json) | `interaction_c13_c24` | C13/fold_stats, C24/panel8 |
| [C16](C16/manifest.json) | `interaction_linear_trials` | C16/glm_fused, C17/combined, C19/ordered_128 |
| [C16](C16/manifest.json) | `interaction_c16_c17` | C16/glm_fused, C17/ovr_waves |
| [C16](C16/manifest.json) | `interaction_c16_c19` | C16/glm_fused, C19/ordered_128 |
| [C17](C17/manifest.json) | `combo_ovr_waves_line_search_pairs` | C17/ovr_waves, C17/line_search_pairs |
| [C17](C17/manifest.json) | `combined` | C17/ovr_waves, C17/line_search_pairs |
| [C18](C18/manifest.json) | `combo_next_residual_tile64` | C18/next_residual, C18/tile64 |
| [C18](C18/manifest.json) | `combo_next_residual_gram_prefetch` | C18/next_residual, C18/gram_prefetch |
| [C18](C18/manifest.json) | `combo_tile64_gram_prefetch` | C18/tile64, C18/gram_prefetch |
| [C18](C18/manifest.json) | `combo_next_residual_tile64_gram_prefetch` | C18/next_residual, C18/tile64, C18/gram_prefetch |
| [C18](C18/manifest.json) | `combined` | C18/next_residual, C18/tile64, C18/gram_prefetch |
| [C20](C20/manifest.json) | `combo_row_cache_pair_load` | C20/row_cache, C20/pair_load |
| [C20](C20/manifest.json) | `combined` | C20/row_cache, C20/pair_load |
| [C20](C20/manifest.json) | `interaction_svm_complete` | C20/combined, C21/extrema_tree4, C22/symmetric_triangle |
| [C20](C20/manifest.json) | `interaction_c20_c21` | C20/row_cache, C21/extrema_tree4 |
| [C20](C20/manifest.json) | `interaction_c20_c22` | C20/row_cache, C22/symmetric_triangle |
| [C24](C24/manifest.json) | `combo_panel8_rows2048` | C24/panel8, C24/rows2048 |
| [C24](C24/manifest.json) | `combo_panel8_tree4` | C24/panel8, C24/tree4 |
| [C24](C24/manifest.json) | `combo_rows2048_tree4` | C24/rows2048, C24/tree4 |
| [C24](C24/manifest.json) | `combo_panel8_rows2048_tree4` | C24/panel8, C24/rows2048, C24/tree4 |
| [C24](C24/manifest.json) | `combined` | C24/panel8, C24/rows2048, C24/tree4 |
| [C26](C26/manifest.json) | `combo_fixed_products_update_fused` | C26/fixed_products, C26/update_fused |
| [C26](C26/manifest.json) | `combined` | C26/fixed_products, C26/update_fused |
| [C27](C27/manifest.json) | `combo_components_factor_components` | C27/components, C27/factor_components |
| [C27](C27/manifest.json) | `combo_components_normalize_components` | C27/components, C27/normalize_components |
| [C27](C27/manifest.json) | `combo_factor_components_normalize_components` | C27/factor_components, C27/normalize_components |
| [C27](C27/manifest.json) | `combo_components_factor_components_normalize_components` | C27/components, C27/factor_components, C27/normalize_components |
| [C27](C27/manifest.json) | `combined` | C27/components, C27/factor_components, C27/normalize_components |
| [C29](C29/manifest.json) | `combined` | C29/stream_topk, C29/stream_topk_tile_128 |
| [C29](C29/manifest.json) | `interaction_graph_complete` | C29/combined, C31/device_buckets, C32/combined, C33/combined, C34/parallel_edges, C43/resident_normalization, C44/sampling_descriptors |
| [C29](C29/manifest.json) | `interaction_c29_c31` | C29/stream_topk, C31/device_buckets |
| [C29](C29/manifest.json) | `interaction_c29_c32` | C29/stream_topk, C32/count_fusion |
| [C29](C29/manifest.json) | `interaction_c29_c33` | C29/stream_topk, C33/frozen_chunks |
| [C29](C29/manifest.json) | `interaction_c29_c34` | C29/stream_topk, C34/parallel_edges |
| [C29](C29/manifest.json) | `interaction_c29_c43` | C29/stream_topk, C43/resident_normalization |
| [C29](C29/manifest.json) | `interaction_c29_c44` | C29/stream_topk, C44/sampling_descriptors |
| [C30](C30/manifest.json) | `combined` | C30/direct_distance, C30/direct_distance_rows_4 |
| [C32](C32/manifest.json) | `combined` | C32/count_fusion, C32/emit_fusion, C32/count_fusion_emit_fusion |
| [C33](C33/manifest.json) | `combined` | C33/frozen_chunks, C33/frozen_chunks_chunk_4 |
| [C35](C35/manifest.json) | `combined` | C35/packed_lists, C35/packed_lists_rows_128 |
| [C36](C36/manifest.json) | `combined` | C36/centroid_tiles, C36/centroid_tiles_rows_4 |
| [C36](C36/manifest.json) | `interaction_kmeans_complete` | C36/combined, C37/combined, C38/combined, C39/retain_state |
| [C36](C36/manifest.json) | `interaction_c36_c37` | C36/centroid_tiles, C37/row_panels |
| [C36](C36/manifest.json) | `interaction_c36_c38` | C36/centroid_tiles, C38/reuse_nearest |
| [C36](C36/manifest.json) | `interaction_c36_c39` | C36/centroid_tiles, C39/retain_state |
| [C37](C37/manifest.json) | `combined` | C37/row_panels, C37/row_panels_panel_128 |
| [C38](C38/manifest.json) | `combined` | C38/reuse_nearest, C38/device_potential, C38/reuse_nearest_device_potential |
| [C40](C40/manifest.json) | `combined` | C40/seed_tiles, C40/active_seeds, C40/seed_tiles_active_seeds |
| [C53](C53/manifest.json) | `combo_center4_bgmm_center4` | C53/center4, C53/bgmm_center4 |
| [C53](C53/manifest.json) | `combined` | C53/center4, C53/bgmm_center4 |
| [C56](C56/manifest.json) | `combo_lda_input_qda_project4` | C56/lda_input, C56/qda_project4 |
| [C56](C56/manifest.json) | `combined` | C56/lda_input, C56/qda_project4 |
| [C58](C58/manifest.json) | `combo_hw_scale_once_hw_series4` | C58/hw_scale_once, C58/hw_series4 |
| [C58](C58/manifest.json) | `combo_hw_scale_once_forecast_series4` | C58/hw_scale_once, C58/forecast_series4 |
| [C58](C58/manifest.json) | `combo_hw_series4_forecast_series4` | C58/hw_series4, C58/forecast_series4 |
| [C58](C58/manifest.json) | `combo_hw_scale_once_hw_series4_forecast_series4` | C58/hw_scale_once, C58/hw_series4, C58/forecast_series4 |
| [C58](C58/manifest.json) | `combo_hw_scale_once_team64` | C58/hw_scale_once, C58/team64 |
| [C58](C58/manifest.json) | `combo_hw_series4_team64` | C58/hw_series4, C58/team64 |
| [C58](C58/manifest.json) | `combo_hw_scale_once_hw_series4_team64` | C58/hw_scale_once, C58/hw_series4, C58/team64 |
| [C58](C58/manifest.json) | `combo_forecast_series4_team64` | C58/forecast_series4, C58/team64 |
| [C58](C58/manifest.json) | `combo_hw_scale_once_forecast_series4_team64` | C58/hw_scale_once, C58/forecast_series4, C58/team64 |
| [C58](C58/manifest.json) | `combo_hw_series4_forecast_series4_team64` | C58/hw_series4, C58/forecast_series4, C58/team64 |
| [C58](C58/manifest.json) | `combo_hw_scale_once_hw_series4_forecast_series4_team64` | C58/hw_scale_once, C58/hw_series4, C58/forecast_series4, C58/team64 |
| [C58](C58/manifest.json) | `combined` | C58/hw_scale_once, C58/hw_series4, C58/forecast_series4, C58/team64 |
| [C58](C58/manifest.json) | `interaction_forecast_complete` | C58/combined, C59/immutable_observations, C60/combined |
| [C58](C58/manifest.json) | `interaction_c58_c59` | C58/hw_scale_once, C59/immutable_observations |
| [C58](C58/manifest.json) | `interaction_c58_c60` | C58/hw_scale_once, C60/lag4 |
| [C60](C60/manifest.json) | `combo_lag4_diff_reuse` | C60/lag4, C60/diff_reuse |
| [C60](C60/manifest.json) | `combined` | C60/lag4, C60/diff_reuse |

## Older experiment references

These links explain infrastructure and prior A/B definitions. Their historical status and evidence are **not** evidence for any changed source in this inventory. Existing controls are not counted again as new C arms. No historical loser/winner record was replaced or converted into a new board measurement.

| Existing ID | Existing experiment | Original A controls | Original B controls | Source |
|---|---|---|---|---|
| [A06](../performance_ideas/A06/manifest.json) | Qualify AMD exact fused low-dimensional neighbor selection | No extra defines in existing manifest | No extra defines in existing manifest | [experiments/performance_ideas/A06/check.mojo](../../experiments/performance_ideas/A06/check.mojo), [neighbors/checks/fused_slot_merge_check.mojo](../../neighbors/checks/fused_slot_merge_check.mojo), [neighbors/estimator.mojo](../../neighbors/estimator.mojo), [gemm/experiments/native_build.py](../../gemm/experiments/native_build.py) |
| [A07](../performance_ideas/A07/manifest.json) | Qualify bounded degree graph and exact integer node histogram tasks | `MOJOLEARN_RBC_CANON_MERGE=1`, `MOJOLEARN_RBC_CANON_DEGREE_BUCKETS=1` | No extra defines in existing manifest | [experiments/performance_ideas/A07/check.mojo](../../experiments/performance_ideas/A07/check.mojo), [experiments/performance_ideas/A07/histogram_check.mojo](../../experiments/performance_ideas/A07/histogram_check.mojo), [experiments/performance_ideas/A07/histogram_tasks.mojo](../../experiments/performance_ideas/A07/histogram_tasks.mojo), [experiments/performance_ideas/N07/streamed_histogram.mojo](../../experiments/performance_ideas/N07/streamed_histogram.mojo), [gemm/experiments/profile_campaign.py](../../gemm/experiments/profile_campaign.py), [neighbors/checks/ball_cover_canonical_order.mojo](../../neighbors/checks/ball_cover_canonical_order.mojo), [ensemble/decisiontree/batched_levelalgo/builder.mojo](../../ensemble/decisiontree/batched_levelalgo/builder.mojo), [experiments/performance_ideas/A07/production_check.mojo](../../experiments/performance_ideas/A07/production_check.mojo) |
| [A08](../performance_ideas/A08/manifest.json) | Requalify promoted 256-row accumulation with fit interactions | `MOJOLEARN_IDN_KMEANS_ACC_ROWS_256_OFF=1`, `MOJOLEARN_IDN_KMEANS_ACC_ROWS_4096=1`, `MOJOLEARN_IDN_KMEANS_CONV_CHUNK_1=1`, `MOJOLEARN_IDN_KMEANS_ACC_ROWS_256_OFF=1`, `MOJOLEARN_IDN_KMEANS_CONV_CHUNK_1=1` | No extra defines in existing manifest | [experiments/performance_ideas/A08/check.mojo](../../experiments/performance_ideas/A08/check.mojo), [gemm/experiments/profile_campaign.py](../../gemm/experiments/profile_campaign.py), [cluster/checks/reduce_by_key.mojo](../../cluster/checks/reduce_by_key.mojo), [cluster/impl/detail/kmeans.mojo](../../cluster/impl/detail/kmeans.mojo) |
| [I02](../performance_ideas/I02/manifest.json) | Bound session GEMM scratch retention | `MOJOLEARN_STEP_PHASE_TIMERS=1` | No extra defines in existing manifest | [gemm/experiments/bounded_workspace.mojo](../../gemm/experiments/bounded_workspace.mojo), [gemm/experiments/bounded_workspace_check.mojo](../../gemm/experiments/bounded_workspace_check.mojo), [gemm/experiments/native_build.py](../../gemm/experiments/native_build.py) |
| [I04](../performance_ideas/I04/manifest.json) | Qualify a coherently shared opt-in 64-element IDENTICAL leaf version | `MOJOLEARN_IDN_GEMM_FOLD_LEAF_64=1` | No extra defines in existing manifest | [gemm/contract.mojo](../../gemm/contract.mojo), [gemm/experiments/fold_profile_probe.mojo](../../gemm/experiments/fold_profile_probe.mojo), [gemm/experiments/profile_check.mojo](../../gemm/experiments/profile_check.mojo), [gemm/experiments/profile_identity_check.mojo](../../gemm/experiments/profile_identity_check.mojo), [gemm/experiments/native_build.py](../../gemm/experiments/native_build.py) |
| [I12](../performance_ideas/I12/manifest.json) | gate solver pass fusion on accepted iterates and refusal semantics | `MOJOLEARN_IDN_QN_EXACT_TRIALS=1` | No extra defines in existing manifest | [experiments/performance_ideas/I12/check.mojo](../../experiments/performance_ideas/I12/check.mojo), [glm/impl/qn/glm_base.mojo](../../glm/impl/qn/glm_base.mojo), [glm/impl/qn/qn_linesearch.mojo](../../glm/impl/qn/qn_linesearch.mojo), [experiments/performance_ideas/I12/trials_check.mojo](../../experiments/performance_ideas/I12/trials_check.mojo), [experiments/performance_ideas/I12/sgd_check.mojo](../../experiments/performance_ideas/I12/sgd_check.mojo), [x_linear/device.mojo](../../x_linear/device.mojo), [x_linear/sgd.mojo](../../x_linear/sgd.mojo) |
| [I13](../performance_ideas/I13/manifest.json) | implement compact degree buckets for bounded RBC stable merges | `MOJOLEARN_RBC_CANON_DEGREE_BUCKETS=1` | `MOJOLEARN_RBC_CANON_MERGE=1` | [experiments/performance_ideas/I13/check.mojo](../../experiments/performance_ideas/I13/check.mojo), [neighbors/checks/ball_cover_canonical_order.mojo](../../neighbors/checks/ball_cover_canonical_order.mojo), [neighbors/checks/rbc_canonical_merge_check.mojo](../../neighbors/checks/rbc_canonical_merge_check.mojo) |
| [I14](../performance_ideas/I14/manifest.json) | qualify gated convergence at first fixed point on graph topologies | `MOJOLEARN_IDN_DBSCAN_CC_CHUNK16=1` | `MOJOLEARN_IDN_DBSCAN_CC_GATED_OFF=1`, `MOJOLEARN_IDN_HDB_ONE_SYNC_OFF=1`, `MOJOLEARN_IDN_HDB_CONDENSE_TWO_READS_OFF=1`, `MOJOLEARN_IDN_HDB_SELECT_ONE_READ_OFF=1`, `MOJOLEARN_IDN_HDB_MR_FUSED_GUARD_OFF=1`, `MOJOLEARN_IDN_HDB_SOFT_LEAN_OFF=1`, `MOJOLEARN_IDN_HDB_PREDICT_DEVICE_CAST_OFF=1`, `MOJOLEARN_IDN_HDB_PREDICT_LEAN_OFF=1` | [experiments/performance_ideas/I14/check.mojo](../../experiments/performance_ideas/I14/check.mojo), [experiments/performance_ideas/I14/hdbscan_stages.mojo](../../experiments/performance_ideas/I14/hdbscan_stages.mojo) |
| [I15](../performance_ideas/I15/manifest.json) | qualify streaming exact neighbor merge at ties and batch tails | `MOJOLEARN_KNN_SELECT_TRIAL=1`, `MOJOLEARN_IDN_KNN_CERTIFIED_REACH=1` | `MOJOLEARN_KNN_SELECT_TRIAL=1`, `MOJOLEARN_IDN_KNN_CERTIFIED_REACH=1`, `MOJOLEARN_KNN_CERTIFIED_MMA_OFF=1` | [experiments/performance_ideas/I15/check.mojo](../../experiments/performance_ideas/I15/check.mojo), [neighbors/impl/detail/knn_brute_force.mojo](../../neighbors/impl/detail/knn_brute_force.mojo), [experiments/performance_ideas/I15/certified_caller.mojo](../../experiments/performance_ideas/I15/certified_caller.mojo) |
| [I16](../performance_ideas/I16/manifest.json) | attribute IVF grouped and staged scans under skewed list occupancy | `MOJOLEARN_IVF_BALANCED_TASKS=1` | No extra defines in existing manifest | [experiments/performance_ideas/I16/check.mojo](../../experiments/performance_ideas/I16/check.mojo), [ivf/impl/neighbors/ivf_flat/ivf_balanced_tasks.mojo](../../ivf/impl/neighbors/ivf_flat/ivf_balanced_tasks.mojo), [ivf/impl/neighbors/ivf_flat/ivf_flat_search.mojo](../../ivf/impl/neighbors/ivf_flat/ivf_flat_search.mojo) |
| [I17](../performance_ideas/I17/manifest.json) | qualify loss-guide frontier retention at varying live leaf capacities | `MOJOLEARN_IDN_GBDT_FRONTIER_RESIDENT=1` | No extra defines in existing manifest | [experiments/performance_ideas/I17/check.mojo](../../experiments/performance_ideas/I17/check.mojo), [gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo](../../gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo), [experiments/performance_ideas/I17/caller_check.mojo](../../experiments/performance_ideas/I17/caller_check.mojo) |
| [I18](../performance_ideas/I18/manifest.json) | Retain bounded exact forest histograms and subtract compatible siblings | `MOJOLEARN_TREE_EXACT_SIBLING_HIST=1`, `MOJOLEARN_TREE_EXACT_SIBLING_HIST_AUDIT=1` | `MOJOLEARN_TREE_EXACT_SIBLING_HIST_AUDIT=1` | [experiments/performance_ideas/I18/check.mojo](../../experiments/performance_ideas/I18/check.mojo), [ensemble/decisiontree/batched_levelalgo/exact_histogram_subtract.mojo](../../ensemble/decisiontree/batched_levelalgo/exact_histogram_subtract.mojo), [ensemble/decisiontree/batched_levelalgo/retained_count_histograms.mojo](../../ensemble/decisiontree/batched_levelalgo/retained_count_histograms.mojo), [ensemble/decisiontree/batched_levelalgo/builder.mojo](../../ensemble/decisiontree/batched_levelalgo/builder.mojo), [experiments/performance_ideas/I18/forest_check.mojo](../../experiments/performance_ideas/I18/forest_check.mojo) |
| [I19](../performance_ideas/I19/manifest.json) | qualify reusable stable radix scratch and bounded key widths | `MOJOLEARN_IDN_RAGGED_FLOAT_RADIX=1` | No extra defines in existing manifest | [experiments/performance_ideas/I19/check.mojo](../../experiments/performance_ideas/I19/check.mojo), [experiments/performance_ideas/I19/ragged_float.mojo](../../experiments/performance_ideas/I19/ragged_float.mojo), [experiments/performance_ideas/I19/float_check.mojo](../../experiments/performance_ideas/I19/float_check.mojo), [x_prep/ragged_quantile.mojo](../../x_prep/ragged_quantile.mojo), [experiments/performance_ideas/I19/quantile_check.mojo](../../experiments/performance_ideas/I19/quantile_check.mojo), [x_prep/ragged_categories.mojo](../../x_prep/ragged_categories.mojo), [x_prep/ragged_select.mojo](../../x_prep/ragged_select.mojo), [core/stable_radix_digits.mojo](../../core/stable_radix_digits.mojo), [experiments/performance_ideas/I19/categories_check.mojo](../../experiments/performance_ideas/I19/categories_check.mojo), [experiments/performance_ideas/I19/select_check.mojo](../../experiments/performance_ideas/I19/select_check.mojo) |
| [I20](../performance_ideas/I20/manifest.json) | Retain bounded source-owned KDE partial storage under the existing chunked fold | `MOJOLEARN_IDN_KDE_PARTIAL_POOL=1` | No extra defines in existing manifest | [experiments/performance_ideas/I20/check.mojo](../../experiments/performance_ideas/I20/check.mojo), [kde/impl/chunk_workspace.mojo](../../kde/impl/chunk_workspace.mojo), [kde/impl/neighbors/kernel_density.mojo](../../kde/impl/neighbors/kernel_density.mojo), [kde/resident_fit.mojo](../../kde/resident_fit.mojo), [kde/checks/reused_workspace_check.mojo](../../kde/checks/reused_workspace_check.mojo) |
| [I21](../performance_ideas/I21/manifest.json) | gate fused GMM components on complete EM state and likelihood | `MOJOLEARN_IDN_GMM_COMPONENT_BATCH=1`, `MOJOLEARN_IDN_GMM_CENTER_PAIR=1` | No extra defines in existing manifest | [experiments/performance_ideas/I21/check.mojo](../../experiments/performance_ideas/I21/check.mojo), [mixture/checks/estep.mojo](../../mixture/checks/estep.mojo), [experiments/performance_ideas/I21/component_check.mojo](../../experiments/performance_ideas/I21/component_check.mojo), [mixture/checks/mstep.mojo](../../mixture/checks/mstep.mojo) |
| [I22](../performance_ideas/I22/manifest.json) | qualify blocked TSQR factors and retained reflector applies across tails | `MOJOLEARN_IDN_TSQR_REUSE=1`, `MOJOLEARN_IDN_TSQR_STRIP_UPDATE=1` | No extra defines in existing manifest | [experiments/performance_ideas/I22/check.mojo](../../experiments/performance_ideas/I22/check.mojo), [x_decomp/tsqr_device.mojo](../../x_decomp/tsqr_device.mojo), [experiments/performance_ideas/I22/reuse_check.mojo](../../experiments/performance_ideas/I22/reuse_check.mojo), [experiments/performance_ideas/I22/campaign.py](../../experiments/performance_ideas/I22/campaign.py) |
| [I23](../performance_ideas/I23/manifest.json) | qualify independent time-series fits and batch-gradient selection state | No extra defines in existing manifest | `MOJOLEARN_ARIMA_ID_BATCH_GRAD_OFF=1` | [experiments/performance_ideas/I23/check.mojo](../../experiments/performance_ideas/I23/check.mojo), [arima/impl/batched_arima.mojo](../../arima/impl/batched_arima.mojo) |
| [I24](../performance_ideas/I24/manifest.json) | fuse resident confusion and PRF input counts in one device pass | `MOJOLEARN_METRICS_JOINT_COUNTS=1` | No extra defines in existing manifest | [experiments/performance_ideas/I24/check.mojo](../../experiments/performance_ideas/I24/check.mojo), [metrics/impl/classification_joint.mojo](../../metrics/impl/classification_joint.mojo) |
| [N07](../performance_ideas/N07/manifest.json) | Stream bounded feature chunks with exact private integer histograms | `MOJOLEARN_IDN_RF_STREAM_REPLICAS=1` | No extra defines in existing manifest | [experiments/performance_ideas/N07/streamed_histogram.mojo](../../experiments/performance_ideas/N07/streamed_histogram.mojo), [experiments/performance_ideas/N07/check.mojo](../../experiments/performance_ideas/N07/check.mojo), [gemm/experiments/native_build.py](../../gemm/experiments/native_build.py), [ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo](../../ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo), [experiments/performance_ideas/N07/production_check.mojo](../../experiments/performance_ideas/N07/production_check.mojo) |
| [N08](../performance_ideas/N08/manifest.json) | Fuse canonical radius distance decisions into device CSR scan/fill | No extra defines in existing manifest | No extra defines in existing manifest | [experiments/performance_ideas/N08/fused_threshold.mojo](../../experiments/performance_ideas/N08/fused_threshold.mojo), [experiments/performance_ideas/N08/check.mojo](../../experiments/performance_ideas/N08/check.mojo), [gemm/experiments/native_build.py](../../gemm/experiments/native_build.py) |

## Handoff and execution status

The original [plan](../../docs/plans/CLASSICAL_IDENTICAL_AB_IDEAS_2026-10-06.md), [catalog](catalog.json), and [catalog source](catalog_source.py) were preserved. Implementation and this inventory belong only to this lane branch. No merge into main or another active lane is part of this handoff.

The harness source programs preparation, full fit/training, synchronization and consumed outputs, and separates inference, cold, excluded warmup and repeated use. Resolved full workload facts and a frozen artifact are required for future execution. [README](README.md) explains selection and recipe fields.

**Execution performed for these candidates: none.** No compilation, tests, candidate imports, identity/quality checks, linters, manifest validators, smoke runs, benchmarks, measurements, board writes or remote jobs. Commit/push use disabled local hooks. Git/source inspection and file editing are the only source-work operations.
