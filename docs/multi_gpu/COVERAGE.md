# Multi-GPU completion scope

This is an alpha implementation. API compatibility is not a constraint on the
remaining work. The constraints are arithmetic identity, correct estimator
semantics, and cloud-only validation. A driver selecting one GPU is not a
multi-GPU implementation. Running unrelated fits on different GPUs is not a
distributed fit of one estimator.

## Compute and memory are separate requirements

The implemented neural drivers replicate parameters and optimizer state.
Forests replicate training data; KMeans retains full-data work on the root.
These paths do not pool GPU memory. A model or dataset that exceeds one GPU's
memory needs additional partitioning and a memory-bounded replay mechanism.
Neither that capacity nor performance scaling has been qualified. ARIMA now
partitions independent series during fit, and scalers partition feature columns
during fit and transform; those workers receive only their assigned data.

## Public estimator inventory

Status refers to the implementation, not every configuration of the class.
Initial fixtures cover two RTX 4090s and two H100s. The continued classical
and histogram paths have two-H100 evidence only.

| Surface | Current multi-GPU coverage | Remaining numerical work |
| --- | --- | --- |
| SmallByteLanguageModelTrainer / LanguageModelTrainer | Concurrent microbatch waves with ordered replay and resident replicas | Memory partitioning and capacity qualification |
| SmallMLPTrainer | Concurrent microbatch gradients, ordered update | Larger shapes; memory partitioning |
| SambaStack | Concurrent microbatch gradients, ordered update | Broader block/configuration coverage; memory partitioning |
| RandomForestClassifier / RandomForestRegressor | Global tree-ID ranges over full data | Larger forests; data partitioning |
| ExtraTreesClassifier / ExtraTreesRegressor | Global tree-ID ranges over full data | Larger forests; data partitioning |
| GradientBoosting / classifier / regressor aliases | Greedy feature histograms; symmetric/depthwise/lossguide gates pass on two H100s | Pointwise searcher; root-state memory partitioning; cross-vendor qualification |
| OrderedRMSE | None | Above plus permutation and ordered-fold state |
| ExperimentalTwoLevelFeatureFreq | None | Categorical candidate generation, scores and both levels |
| KMeans | Parallel row-tile assignment | Resident staging; memory-bounded full-data updates |
| LinearRegression / Ridge | Original 128 Gram chunks distributed at 1..128 features; two-H100 gates pass | Wide OLS; larger feature counts; root-state partitioning |
| LogisticRegression | QN gradient feature columns; binary/multiclass two-H100 gates pass | Root-state partitioning; broader configurations and cross-vendor qualification |
| ElasticNet / Lasso | Original dot leaves across GPUs; cyclic fit and FP32 oracle gates pass on two H100s | Resident shard reuse; root-state partitioning and cross-vendor qualification |
| SVC / SVR | Linear/RBF kernel rows during fit/prediction; two-H100 cell/full-fit gates pass | Root-state partitioning, broader configurations and cross-vendor qualification |
| PCA / TruncatedSVD | Original Gram chunks at 1..128 features; covariance PCA and SVD gates pass | Other solver paths, larger widths and root-state partitioning |
| NearestNeighbors / RadiusNeighbors | Whole-query shards; brute/RBC KNN and ragged radius gates pass on two H100s | Pooled reference index; larger shapes/metrics and cross-vendor qualification |
| KNeighborsClassifier / KNeighborsRegressor | Whole-query shards with original voting/weighting; single/multi-target two-H100 gates pass | Pooled reference index; broader configurations and cross-vendor qualification |
| DBSCAN | Brute L2/L1 and RBC neighborhood rows; two-H100 adjacency/CSR, core-stage and full-fit gates pass | Root graph/index partitioning; larger shapes and cross-vendor qualification |
| KernelDensity | Whole-query shards; six kernels with/without positive weights pass on two H100s | Pooled reference index; broader metrics/shapes and cross-vendor qualification |
| AgglomerativeClustering | None | Distance tiles and global merge/tie order |
| SpectralClustering | None | Affinity tiles, eigensolver and downstream clustering |
| UMAP | None | Neighbor graph and globally ordered optimizer/RNG updates |
| IsolationForest | Global tree ranges during fit and score-time rebuild; two-H100 full-model and scoring gates pass | Full-data replication and assembled root model; broader configurations and cross-vendor qualification |
| GaussianProcessRegressor | Covariance/cross-covariance rows; two-H100 factor/dual/likelihood/mean/std gates pass | Root factorization/state partitioning; broader kernels/shapes and cross-vendor qualification |
| ARIMA | Independent-series fit; two-H100 fit/forecast equality gates passed | Broader orders, large-memory and cross-vendor qualification; distributed prediction |
| ExponentialSmoothing | Independent-series additive/multiplicative fit; two-H100 fit/forecast gates pass | Distributed prediction; broader configurations and capacity qualification |
| StandardScaler / MinMaxScaler | Column-sharded fit/transform/inverse; two-H100 gates passed | Large-memory and cross-vendor qualification |

Mamba1/2/3Block and TransformerBlock are forward/backward primitives, not
standalone fit drivers. Training-stack coverage must not be read as distributed
execution of every primitive. CPU host trainers/predictors are separate CPU
surfaces. Tokenization, metrics, model selection, linear algebra and native-only
lanes also need explicit operation-level scope; the estimator list does not
establish coverage of those capabilities.

## Completion evidence

For each implemented family, qualify its actual partition against the reference
execution using identical inputs, seeds and initial state. Record gradients or
intermediate statistics where applicable, full fitted state, predictions and
checkpoint/replay behavior. Cover weighted inputs, categorical handling, ties,
ragged partitions, errors and all exposed algorithm variants where applicable.

Cross-vendor claims require the same source and logical work on the named
vendors. Scaling claims require measurements; capacity claims require runs
whose state or data exceed a single device's memory. All builds, model runs and
tests remain on cloud hosts. The existing receipts establish none of the
unfinished rows above.
