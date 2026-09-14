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
Neither that capacity nor performance scaling has been qualified.

## Public estimator inventory

Status refers to the implementation, not every configuration of the class.
The existing cloud fixtures cover two RTX 4090s and two H100s only.

| Surface | Current multi-GPU coverage | Remaining numerical work |
| --- | --- | --- |
| SmallByteLanguageModelTrainer / LanguageModelTrainer | Ordered microbatch replay with resident replicas | Concurrent shard execution; memory partitioning and capacity qualification |
| SmallMLPTrainer | Concurrent microbatch gradients, ordered update | Larger shapes; memory partitioning |
| SambaStack | Concurrent microbatch gradients, ordered update | Broader block/configuration coverage; memory partitioning |
| RandomForestClassifier / RandomForestRegressor | Global tree-ID ranges over full data | Larger forests; data partitioning |
| ExtraTreesClassifier / ExtraTreesRegressor | Global tree-ID ranges over full data | Larger forests; data partitioning |
| GradientBoosting / classifier / regressor aliases | None | Feature/histogram partition preserving global scales, split order and sequential rounds |
| OrderedRMSE | None | Above plus permutation and ordered-fold state |
| ExperimentalTwoLevelFeatureFreq | None | Categorical candidate generation, scores and both levels |
| KMeans | Parallel row-tile assignment | Resident staging; memory-bounded full-data updates |
| LinearRegression / Ridge | None | Matrix output tiles, original contraction and solver order |
| LogisticRegression | None | Objective/gradient contractions, line search and solver state |
| ElasticNet / Lasso | None | Matrix work with original coordinate and convergence order |
| SVC / SVR | None | Kernel tiles with global working-set and solver order |
| PCA / TruncatedSVD | None | Matrix work and eigensolver trajectory/sign conventions |
| NearestNeighbors / RadiusNeighbors | None | Query/reference tiles, stable global neighbor ordering |
| KNeighborsClassifier / KNeighborsRegressor | None | Above plus original voting and weighting order |
| DBSCAN | None | Distance tiles and global connectivity/label semantics |
| KernelDensity | None | Query tiles with original reference-row reduction |
| AgglomerativeClustering | None | Distance tiles and global merge/tie order |
| SpectralClustering | None | Affinity tiles, eigensolver and downstream clustering |
| UMAP | None | Neighbor graph and globally ordered optimizer/RNG updates |
| IsolationForest | None | Global tree seed schedule, model merge and score order |
| GaussianProcessRegressor | None | Kernel tiles, factorization, solves and optimizer trajectory |
| ARIMA | None | Independent series with unchanged per-series initialization and solver state |
| ExponentialSmoothing | None | Independent series with unchanged optimization semantics |
| StandardScaler / MinMaxScaler | None | Column partition or original global reduction order |

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
