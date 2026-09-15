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
and Samba use host-staged disjoint gradient columns and optimizer ranges.
Clipping places whole tensors on owners and retains the original small
cross-tensor norm on the first GPU. Their gradient computations still require
a complete host model per worker; Samba stages GPU work one block at a time.
Forests replicate training data; KMeans retains full-data work on the root.
These remaining allocations limit capacity. The new byte-LM offload driver
matches two updates of the 958.7M-parameter pooled model on one H100, with
2621 MiB observed on its selected GPU and full state on the host. Actual RTX5090
offload execution is still owed because RunPod had no stock. Other model/data paths
still need additional partitions or replay mechanisms.
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
Those receipts cover NVIDIA architectures. The
[166-lane identity record](../../bench/results/identity_break/2026-09-14_166-lanes/README.md)
covers 31 parallel-driver lanes, including the layer-pool and offload byte-LM
trainers, on two MI300X GPUs and two H100s (IDENTICAL=279 per vendor against
the single-device columns), and
[eight more par lanes](../../bench/results/identity_break/2026-09-15_par-lanes-new/README.md)
(the forest pool, GaussianMixture, resampling, HDBSCAN, Cholesky, KernelRidge,
Nystroem and RBFSampler drivers) read IDENTICAL on one and two devices of both
vendors. Those cells establish equality for the lanes' fixtures, not every
later pooling change, parameter combination or buffer size: on two MI300X a
device-to-device copy above 1 MiB can be read before it lands
([peer-copy-mi300x](../../bench/results/multi_gpu/2026-09-15/peer-copy-mi300x/README.md)),
and the [transport audit](../../bench/results/multi_gpu/2026-09-15/transport-audit/README.md)
lists which drivers have been asked above 1 MiB.

| Surface | Current multi-GPU coverage | Remaining numerical work |
| --- | --- | --- |
| SmallByteLanguageModelTrainer / LanguageModelTrainer | Replica training with pooled optimizer/reduction buffers; separate layer-owned model trainer with RTX 5090 capacity and H100/5090 ordered-replay gates; host-offloaded single-GPU replay. The replica pools' cross-device copies are staged through host memory on AMD (`core/multi_gpu.mojo::transfer_bytes`) after they diverged on two MI300X at 2.1 million parameters; the fixed par-byte-lm lanes at that width are equal on one and two devices on two MI300X and two H100s ([transport audit](../../bench/results/multi_gpu/2026-09-15/transport-audit/README.md)) | Broader shapes; RTX5090 and AMD/Apple qualification of host-offloaded replay; eight-device qualification |
| SmallMLPTrainer | Fixed 8→16→3 model (195 parameters), concurrent microbatch gradients, ordered sum and host-staged optimizer ranges | Broader admitted batch/optimizer fixtures and scheduling qualification; larger model architectures are not part of this estimator |
| SambaStack | Concurrent microbatch gradients, pooled gradient columns, whole clipping tensors and optimizer ranges; original norm and ordered sum; streamed host checkpoints | End-to-end large-model capacity qualification; host/IPC memory, int32 registry limit, individual block/tensor/activation capacity |
| RandomForestClassifier / RandomForestRegressor | Global tree-ID fit ranges over full data; separate resident-grove GPU prediction driver preserving the existing fixed32 engine; its native and 16 public prediction gates pass on two H100s and two MI300X with equal public receipts | Training-data partitioning; actual beyond-one-GPU model capacity, broader fixtures |
| ExtraTreesClassifier / ExtraTreesRegressor | Global tree-ID fit ranges over full data; shared resident-grove GPU prediction with original ET output precision, same two-H100 and two-MI300X gates | Training-data partitioning; actual beyond-one-GPU model capacity, broader fixtures |
| GradientBoosting / GradientBoostingClassifier / GradientBoostingRegressor | Greedy and pointwise feature groups; full histogram bytes and adapter contracts pass on two H100s | Broader configurations; root-state memory partitioning; cross-vendor qualification |
| OrderedRMSE | Pointwise feature groups with original permutation/fold updates; trace/model gates pass on two H100s | Root-state pooling; broader configurations and cross-vendor qualification |
| ExperimentalTwoLevelFeatureFreq | Both levels use greedy feature histograms after original categorical generation; model/prediction gates pass on two H100s | Candidate/root-state pooling; broader configurations and cross-vendor qualification |
| KMeans | Parallel row-tile assignment; one-device and two-device par-kmeans equal on two MI300X at 80000x16 rows (transport audit) | Resident staging; memory-bounded full-data updates |
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
| IsolationForest | Resident tree owners during fit and score-time rebuild; canonical per-row carry across owners; two-H100 model-buffer/path/score gates pass | Full-data replication, per-owner scratch/model limits and global int32 node admission; beyond-single-device capacity and cross-vendor qualification |
| GaussianProcessRegressor | Covariance/cross-covariance rows; two-H100 factor/dual/likelihood/mean/std gates pass | Root factorization/state partitioning; broader kernels/shapes and cross-vendor qualification |
| ARIMA | Independent-series fit; two-H100 fit/forecast equality gates passed | Broader orders, large-memory and cross-vendor qualification; distributed prediction |
| ExponentialSmoothing | Independent-series additive/multiplicative fit; two-H100 fit/forecast gates pass | Distributed prediction; broader configurations and capacity qualification |
| StandardScaler / MinMaxScaler | Column-sharded fit/transform/inverse; two-H100 gates passed | Large-memory and cross-vendor qualification |
| HDBSCAN | fit_hdbscan: core-distance k-NN query rows and dense pairwise distance rows through the neighbors and hierarchy row drivers; mutual reachability, MST, condensed tree and selection on the root; traces and attributes equal one device on two H100s and two MI300X with equal receipts across vendors; the par-hdbscan lane reads IDENTICAL on one and two devices of both vendors | MST, hierarchy and m x m graph pooling; sparse graph arm (not implemented in the lane); capacity |
| KernelRidge / Nystroem | fit_kernel_method / apply_kernel_method: kernel-matrix output rows through the SVM seam for linear, rbf, poly and sigmoid ('laplacian' refused by name); KernelRidge factor rows and target columns through the Cholesky driver; Nystroem basis and Jacobi on the root; state and outputs equal one device on two H100s and two MI300X with equal receipts across vendors; the par-kernel-ridge (600 rows, two targets) and par-nystroem identity_break lanes read IDENTICAL on one and two devices of both vendors | Kernel matrix, factor and eigensolver pooling; laplacian rows; capacity |
| RBFSampler | transform_rbf_sampler: whole query row ranges on one-device workers with the position-mapped fitted weights; transforms equal the one-call transform on two H100s and two MI300X with equal receipts; the par-rbf-sampler lane (1024 features, shards of 300 rows) reads IDENTICAL on one and two devices of both vendors | Fit has no data to partition; capacity |
| GaussianMixture | Row-sharded E-steps (fit, score_samples, predict_proba, predict) with the root M-step, Cholesky and convergence test; KMeans init through its row-tile driver; full covariance with kmeans and random init (the only exposed type; the others are refused by name); traces, state and outputs equal one device on two H100s and two MI300X, and the two vendors' receipts and trace files are equal; the par-gmm lane reads IDENTICAL on one and two devices of both vendors | M-step statistics and responsibilities on the root; capacity and throughput unqualified |
| Cholesky | fit_cholesky / solve_cholesky (and MOJOLEARN_CHOLESKY_DEVICE_COUNT inside potrf_lower and cho_solve): each panel's trailing-update product by whole output rows, the solve by whole right-hand-side columns staged through host memory (a device-to-device solve reads stale target memory on two MI300X, a platform behavior: [peer-copy-mi300x](../../bench/results/multi_gpu/2026-09-15/peer-copy-mi300x/README.md)); panel order, pivots and root matrix unchanged; traces equal one device on two H100s and two MI300X with equal digests across vendors, and the par-cholesky lane (600 x 600, three right-hand sides) reads IDENTICAL on one and two devices of both vendors | Panel order is sequential and not partitioned; a single right-hand side is not partitioned; matrix pooling and capacity |

The newer binding availability flags do not establish complete estimator
dispatch or qualification. IVF-FLAT is public as `mojolearn.IVFIndex` since
2026-09-14 and has no parallel driver; track its future distributed
build/search and index storage separately. Public `resample` bootstrap, permutation tests and Monte Carlo integration
now have distributed entries in `parallel_classical` that move whole global
replicate, permutation and 256-sample chunk ranges and keep the sort,
interval, p-value and fold on the root; their one-device equality passes on
two H100s and two MI300X with equal receipts across the two vendors
(`bench/results/multi_gpu/2026-09-14/resample-h100/`, `resample-mi300x/`).

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
