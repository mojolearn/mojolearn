# Multi-GPU completion scope

This is an alpha implementation. API compatibility is not a constraint on the
remaining work. The constraints are arithmetic identity, correct estimator
semantics, and cloud-only validation. A driver selecting one GPU is not a
multi-GPU implementation. Running unrelated fits on different GPUs is not a
distributed fit of one estimator.

## Compute and memory are separate requirements

The byte-LM replica driver pools AdamW moments, rollback copies and gradient-reduction
scratch in disjoint device ranges. A separate `PooledByteLanguageModelTrainer`
partitions decoder layers, parameters, moments, gradients and rollback storage,
with embedding/head on the first GPU. Its 958,746,624-parameter fixture completes
a step on two RTX 5090s while both the existing trainer and the same new
driver run out of memory on one. Nine new same-source H100/5090 groups match
complete state, gradient and loss hashes.
Each individual layer and embedding/head must still fit an owner, and portable
state construction/export requires full host arrays. SmallMLP
and Samba use host-staged disjoint optimizer ranges after the original global
clip. Their gradient computations still require a complete model per worker.
Forests replicate training data; KMeans retains full-data work on the root.
These remaining allocations limit capacity. A model or dataset that exceeds one GPU's
memory needs additional partitioning and a memory-bounded replay mechanism.
Other neural model capacities and performance scaling remain unqualified. A separate
reference-sharded KNN path has passed a 96 GiB host-staged index gate on two
80 GB H100s; it does not keep the full index resident in pooled VRAM. ARIMA now
partitions independent series during fit, and scalers partition feature columns
during fit and transform; those workers receive only their assigned data.

## Public estimator inventory

Status refers to the implementation, not every configuration of the class.
Initial fixtures cover two RTX 4090s and two H100s. A frozen-source replay on
two RTX 5090s matches all 16 H100 receipt groups for pooled neural optimizers,
MLP/Samba, boosting, wider Gram and tall full PCA. Pointwise histogram dump
bytes and OrderedRMSE trace records also match across those architectures.
Those receipts cover NVIDIA architectures. A separate earlier-source
[136-lane identity record](../../bench/results/identity_break/2026-09-14_136-lanes/README.md)
now covers sixteen parallel-driver lanes on two MI300X GPUs and two H100s,
matching their single-device AMD and Apple replay columns. Its 144 training
cells establish cross-vendor equality for those fixtures, not every later
pooling change or parameter combination. The new layer-pool and offload
implementations still require their own AMD/Apple qualification.

| Surface | Current multi-GPU coverage | Remaining numerical work |
| --- | --- | --- |
| SmallByteLanguageModelTrainer / LanguageModelTrainer | Replica training with pooled optimizer/reduction buffers; separate layer-owned model trainer with RTX 5090 capacity and H100/5090 ordered-replay gates; host-offloaded single-GPU replay | Broader shapes; RTX5090 and AMD/Apple qualification of host-offloaded replay; eight-device qualification |
| SmallMLPTrainer | Fixed 8→16→3 model (195 parameters), concurrent microbatch gradients, ordered sum and host-staged optimizer ranges | Broader admitted batch/optimizer fixtures and scheduling qualification; larger model architectures are not part of this estimator |
| SambaStack | Concurrent microbatch gradients, ordered sum, original global clipping and host-staged optimizer ranges | Broader configurations; resident state and model/activation pooling |
| RandomForestClassifier / RandomForestRegressor | Global tree-ID ranges over full data | Larger forests; data partitioning |
| ExtraTreesClassifier / ExtraTreesRegressor | Global tree-ID ranges over full data | Larger forests; data partitioning |
| GradientBoosting / GradientBoostingClassifier / GradientBoostingRegressor | Greedy and pointwise feature groups; full histogram bytes and adapter contracts pass on two H100s | Broader configurations; root-state memory partitioning; cross-vendor qualification |
| OrderedRMSE | Pointwise feature groups with original permutation/fold updates; trace/model gates pass on two H100s | Root-state pooling; broader configurations and cross-vendor qualification |
| ExperimentalTwoLevelFeatureFreq | Both levels use greedy feature histograms after original categorical generation; model/prediction gates pass on two H100s | Candidate/root-state pooling; broader configurations and cross-vendor qualification |
| KMeans | Parallel row-tile assignment | Resident staging; memory-bounded full-data updates |
| LinearRegression / Ridge | Original Gram chunks plus wider v1 output rows and minimum-norm OLS; two-H100 state/output gates pass | Larger shapes; root-state pooling; cross-vendor qualification |
| LogisticRegression | QN gradient feature columns; binary/multiclass two-H100 gates pass | Root-state partitioning; broader configurations and cross-vendor qualification |
| ElasticNet / Lasso | Original dot leaves across GPUs; cyclic fit and FP32 oracle gates pass on two H100s | Resident shard reuse; root-state partitioning and cross-vendor qualification |
| SVC / SVR | Linear/RBF kernel rows during fit/prediction; two-H100 cell/full-fit gates pass | Root-state partitioning, broader configurations and cross-vendor qualification |
| PCA / TruncatedSVD | Original Gram chunks and wider v1 output rows; covariance PCA/SVD through 257 features and tall full-PCA TSQR panels pass on two H100s | Wide full-PCA transpose QR; larger shapes, root-state pooling and cross-vendor qualification |
| NearestNeighbors / RadiusNeighbors | Whole-query brute/RBC and radius; brute KNN reference shards with a 96 GiB host-staged index gate on two H100s | Resident index pooling; RBC/radius reference partitioning; larger shapes/metrics and cross-vendor qualification |
| KNeighborsClassifier / KNeighborsRegressor | Query and brute reference shards with original voting/weighting; single/multi-target two-H100 gates pass | Resident index/target pooling; broader configurations and cross-vendor qualification |
| DBSCAN | Brute L2/L1 and RBC neighborhood rows; two-H100 adjacency/CSR, core-stage and full-fit gates pass | Root graph/index partitioning; larger shapes and cross-vendor qualification |
| KernelDensity | Whole-query shards; six kernels with/without positive weights pass on two H100s | Pooled reference index; broader metrics/shapes and cross-vendor qualification |
| AgglomerativeClustering | Native pairwise rows; full children/labels and raw-bit two-H100 gates pass | Root graph/state pooling; broader shapes and cross-vendor qualification |
| SpectralClustering | Native KNN rows and KMeans assignments; embedding/label two-H100 gates pass | Root affinity/eigensolver pooling; broader configurations and cross-vendor qualification |
| UMAP | Fit/transform native KNN rows; complete embedding/transform two-H100 gates pass | Root graph/optimizer pooling; broader configurations and cross-vendor qualification |
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
