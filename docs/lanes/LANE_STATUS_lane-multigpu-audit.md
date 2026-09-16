# LANE STATUS: lane/multigpu-audit (2026-09-16)

Which shipped algorithms have a multi-GPU path and which do not, read from the
code rather than from the prose. Branch `lane/multigpu-audit`, cut from
origin/main `bfb8f725a`. Part 1 (this audit) is complete and is READ ONLY: no
build, no run, no Metal lock, no box rented. Part 2 (building the missing
paths) is NOT started and needs Andrew's word.

## What was read

- `python/mojolearn/_parallel_pool.py` (`DevicePool`, cooperative and
  non-cooperative modes, the 17 `MOJOLEARN_<X>_DEVICE_COUNT` switches a
  cooperative pool sets) and `python/mojolearn/_parallel_worker.py` (its 39
  operations plus the `cpu_reference` wrapper, and the `*_parallel_available`
  probe each one demands).
- The seven driver modules `python/mojolearn/parallel_classical.py`,
  `parallel_ensemble.py`, `parallel_graph.py`, `parallel_neighbors.py`,
  `parallel_neighbors_reference.py`, `parallel_preprocessing.py`,
  `parallel_training.py`, and the two native-session trainers
  `python/mojolearn/model_pool_training.py` and
  `python/mojolearn/offload_training.py`.
- Every driver's admission check. A driver's `type(estimator) is not X` /
  `type(estimator) not in (...)` line is what actually decides whether a
  shipped class has a path, and it is the line this audit classifies on.
- The 39 `@lane("par-*")` declarations in `/Users/andrewhendel/CascadeProjects/mojolearn/tools/identity_break.py`
  (176 lanes total).
- `python/mojolearn/host_surface.py`, `python/mojolearn/_backend.py` and
  `python/mojolearn/tests/test_expose_d_manifest.py` for which host bindings
  deliberately omit a parallel flag, and which bindings have no flag at all.
- `docs/multi_gpu/COVERAGE.md` and `docs/multi_gpu/README.md` as prose to check
  the code against, not as the source.
- `docs/lanes/LANE_STATUS_lane-cpu-training-par-wave3.md` (the 39-lane CPU
  triage) was read and is NOT redone here. That file answers whether a
  multi-GPU path can be VERIFIED on a CPU column. This file answers whether
  the path EXISTS.

## The universe being classified

The 69 algorithm surfaces in `python/mojolearn/__init__.py`'s `__all__`,
excluding submodules, configuration objects (`SambaConfig`,
`ByteLanguageModelConfig`), kernel objects (`RBF`, `Matern`, `ConstantKernel`,
`WhiteKernel`), state objects (`Mamba*State`, `TransformerState`), `Array`, and
the version and vendor helpers. `LanguageModelTrainer` is an alias of
`SmallByteLanguageModelTrainer` (`python/mojolearn/language_model.py:29`) and is
counted once. `resample` is counted once for its three public functions.

## Counts

| class | count |
|---|---|
| (a) has a multi-GPU path | 44 |
| (b) no path, plausibly should have one | 6 |
| (c) multi-GPU is not meaningful | 19 |

Of the 44 in (a), **33 have their own `par-*` identity lane** and **11 are
admitted by a driver's type check but have no dedicated lane** (see (a3)).
Seven further named refusals sit INSIDE families that have a path; they are
listed under "Named refusals" and are not counted as (b).

## (a) Has a multi-GPU path (44)

Mechanism is read from the pool the driver opens.

- **sharded** means `DevicePool(devices)` (not cooperative): the driver cuts
  the work into logical shards in Python, sends one request per shard, and
  merges the parts in shard order in Python.
- **cooperative** means `DevicePool(devices, cooperative=True)`: the whole fit
  goes to one worker and the partition happens inside the GPU binding, keyed by
  the `MOJOLEARN_<X>_DEVICE_COUNT` the pool sets.
- **native session** means no `DevicePool` at all; the trainer opens a
  multi-device session on `bindings/_mojolearn_byte_lm.mojo`.

### (a1) Sharded, with a lane (13 surfaces)

| surface | lane | driver | what is cut |
|---|---|---|---|
| RandomForestClassifier | par-forest | `parallel_ensemble.fit_forest` | global tree ID ranges over replicated data |
| ExtraTreesRegressor | par-forest-et | `parallel_ensemble.fit_forest` | the same |
| StandardScaler | par-scaler | `parallel_preprocessing.fit_scaler` / `transform_scaler` | feature columns, fit and both transforms |
| ARIMA | par-arima | `parallel_classical.fit_arima` | independent series |
| ExponentialSmoothing | par-holtwinters | `parallel_classical.fit_exponential_smoothing` | independent series |
| KNeighborsClassifier | par-queries-knn, par-reference-knn | `ParallelQueries`, `ReferenceShardedNeighbors` | query rows; reference rows merged by composite key |
| KNeighborsRegressor | par-reference-knn-reg | `ReferenceShardedNeighbors` | reference rows, then one vote request |
| RadiusNeighbors | par-queries-radius | `ParallelQueries` | query rows, ragged join in input order |
| KernelDensity | par-queries-kde | `ParallelQueries` | query rows |
| RBFSampler | par-rbf-sampler | `parallel_classical.transform_rbf_sampler` | transform rows (fit has no data to split) |
| SmallMLPTrainer | par-mlp | `ParallelNeuralTrainer` | logical gradient shards, ordered sum, cooperative update |
| SambaStack | par-samba, par-samba-clip | `ParallelNeuralTrainer` | the same, plus whole clipping tensors on owners |
| SmallByteLanguageModelTrainer | par-byte-lm, par-byte-lm-model-pool, par-byte-lm-offload | native session | replicas with pooled optimizer ranges; decoder layers owned per device; host-staged one-device replay |

### (a2) Cooperative, with a lane (20 surfaces)

| surface | lane | switch | what is cut inside the binding |
|---|---|---|---|
| GradientBoosting | par-boosting, par-boosting-pointwise | `MOJOLEARN_GBDT_DEVICE_COUNT` | packed feature groups in the greedy and pointwise histograms |
| OrderedRMSE | par-ordered-rmse | the same | pointwise feature groups, ordered folds on the root |
| ExperimentalTwoLevelFeatureFreq | par-feature-freq | the same | both levels' greedy histograms |
| IsolationForest | par-iforest | `MOJOLEARN_IFOREST_DEVICE_COUNT` | tree ranges at fit and at score-time rebuild |
| RandomForest / ExtraTrees inference | par-forest-pool | `MOJOLEARN_FOREST_DEVICE_COUNT` | 32 resident logical groves (`ParallelForestPredictor`) |
| KMeans | par-kmeans | `MOJOLEARN_KMEANS_DEVICE_COUNT` | row tiles during distance and assignment |
| Ridge | par-gram | `MOJOLEARN_GRAM_DEVICE_COUNT`, `MOJOLEARN_QR_DEVICE_COUNT` | pinned Gram chunks, wide output rows, TSQR panels |
| LogisticRegression | par-logistic | `MOJOLEARN_GLM_DEVICE_COUNT` | QN gradient feature columns |
| Lasso | par-cd | `MOJOLEARN_SOLVER_DEVICE_COUNT` | FP32-v1 dot leaves in cyclic coordinate descent |
| SVC | par-svm | `MOJOLEARN_SVM_DEVICE_COUNT` | kernel output rows at fit and prediction |
| GaussianProcessRegressor | par-gp | `MOJOLEARN_GP_DEVICE_COUNT` | covariance and cross-covariance rows |
| DBSCAN | par-dbscan | `MOJOLEARN_DBSCAN_DEVICE_COUNT` | neighborhood rows |
| AgglomerativeClustering | par-graph-agglomerative | `MOJOLEARN_HIERARCHY_DEVICE_COUNT` | native pairwise rows, MST and merges on the root |
| SpectralClustering | par-graph-spectral | `MOJOLEARN_NEIGHBORS_DEVICE_COUNT` | KNN rows and the KMeans assignment, eigensolver on the root |
| UMAP | par-graph-umap | the same | fit and transform neighbor rows |
| HDBSCAN | par-hdbscan | `MOJOLEARN_NEIGHBORS_DEVICE_COUNT` + `MOJOLEARN_HIERARCHY_DEVICE_COUNT` | core-distance k-NN rows and dense distance rows |
| GaussianMixture | par-gmm | `MOJOLEARN_GMM_DEVICE_COUNT` | every E-step's row ranges; M-step on the root |
| Cholesky | par-cholesky | `MOJOLEARN_CHOLESKY_DEVICE_COUNT` | trailing-update output rows, right-hand-side columns |
| KernelRidge, Nystroem | par-kernel-ridge, par-nystroem | `MOJOLEARN_SVM_DEVICE_COUNT` + the scoped Cholesky switch | kernel matrix output rows, factor rows, target columns |
| resample (bootstrap, permutation_test, monte_carlo_integrate) | par-resample | `MOJOLEARN_RESAMPLE_DEVICE_COUNT` | global replicate, permutation and 256-sample chunk ranges |

### (a3) Admitted by a driver, no dedicated `par-*` lane (11 surfaces)

These pass their driver's type check today, so the path EXISTS; what is missing
is a cell in the identity record, not code.

| surface | driver that admits it | standalone gate that exists |
|---|---|---|
| RandomForestRegressor | `fit_forest`, `ParallelForestPredictor` | `tools/parallel_forest_pool_check.py` |
| ExtraTreesClassifier | the same | the same |
| GradientBoostingClassifier | `fit_boosting` | `tools/parallel_boosting_adapters_check.py` |
| GradientBoostingRegressor | the same | the same |
| LinearRegression | `fit_gram_estimator` | `tools/parallel_gram_check.py` |
| PCA | `fit_gram_estimator` | `tools/parallel_gram_check.py`, `tools/parallel_pca_full_check.py` |
| TruncatedSVD | the same | `tools/parallel_gram_check.py` |
| ElasticNet | `fit_coordinate_descent` | `tools/parallel_solver_check.py` |
| SVR | `fit_svm`, `predict_svm` | `tools/parallel_svm_check.py` |
| MinMaxScaler | `fit_scaler`, `transform_scaler` | `tools/parallel_preprocessing_check.py` |
| NearestNeighbors | `ParallelQueries`, `ReferenceShardedNeighbors` | `tools/parallel_neighbors_check.py`, `tools/parallel_reference_neighbors_check.py` |

Cheapest honest work in the whole audit. Each is a handful of lines in
`tools/identity_break.py` beside the sibling lane that already exists, and the
cells ride whatever two-device leg the next record runs. No Mojo, no driver.

## (b) No multi-GPU path, plausibly should have one (6)

Ranked by value over cost. Every one of these owes a two-device box (two H100s
and two MI300X) which is NOT authorized today; all legs below are written as
OWED.

### b1. `model_selection.cross_val_score` folds. Cost: half a day, no Mojo.

`python/mojolearn/model_selection.py:184-185` raises
`NotImplementedError('cross_val_score supports n_jobs=1 only')`. Folds are
fully independent, the shape a non-cooperative `DevicePool` was built for, and
the merge is a list in fold order. This is the single cleanest second-device
win in the repo and it removes a refusal from a public API. Work: one
`fit_cross_val` style entry, one worker operation, one `par-cross-val` lane
beside the existing `cross-val` lane, one CPU admission in
`_parallel_pool.CPU_OPERATIONS` (it qualifies under wave 3's shape 1, so the
CPU column can verify it). Roughly 120 lines.

### b2. GaussianProcessClassifier. Cost: about a day, no Mojo on the cheap route.

`parallel_classical.fit_gaussian_process` refuses anything that is not
`GaussianProcessRegressor`. GPC (`python/mojolearn/_gpc_impl.py:144`) is one
binary Laplace fit for two classes and one per class past two, held in
`estimators_`; the per-class fits are independent, so the honest split is
sharded over `columns` in `fit`, merged in class order. The within-fit
covariance rows already have a driver (`gp_parallel_available`,
`MOJOLEARN_GP_DEVICE_COUNT`) if a cooperative arm is wanted later. Two lanes
exist to hold it to (`gpc`, `gpc-multiclass`). Roughly 200 lines plus a
`par-gpc` lane.

### b3. IVFIndex search, then build. Cost: about a day for search, two to three for build.

`docs/multi_gpu/COVERAGE.md` says it plainly and the code agrees:
`python/mojolearn/tests/test_expose_d_manifest.py` asserts the IVF binding
exports NO `*_parallel_available` flag, and no driver module mentions
`IVFIndex`. Search is query rows, exactly the shape `ParallelQueries` already
cuts, so a Python row driver over `ivf_flat_search_host` is the cheap half.
Build is the larger half: the coarse quantizer is `cluster/`'s k-means, which
already has a row-tile driver, but the list assignment and the CSR layout would
need a native row seam and a new flag. Three lanes exist to hold it to (`ivf`,
`ivf-euclidean`, `ivf-extend`).

### b4. `linalg.matmul` output row tiles. Cost: two to three days, highest regression risk.

There is no `MOJOLEARN_GEMM_DEVICE_COUNT` anywhere in the tree; the only GEMM
level switch is `MOJOLEARN_GRAM_DEVICE_COUNT` in `core/gram_multi_gpu.mojo`.
The machinery exists (`cholesky/multi_gpu.mojo::chol_trailing_rows` already
runs `identical_gemm_into` per owner and stages bytes through
`core/multi_gpu.mojo::transfer_bytes`), so this is a public door plus a switch
over output row tiles. High leverage because matmul sits under most other
paths, and for that same reason its gate has to be the most thorough. Do it
after b1 and b2, not before.

### b5. ARIMA and ExponentialSmoothing distributed prediction. Cost: half a day.

Both fits shard by series today; `forecast` and `predict` run on one device
(`docs/multi_gpu/COVERAGE.md` names it as remaining work for both rows). The
series split already exists in `parallel_classical`, so this reuses it. Small,
and it closes a row the coverage table already owns.

### b6. Embedding fold, and metrics row chunks. Cost: about a day each, lowest value.

`Embedding`'s gather is row-independent but its backward is an ordered
scatter-add, and the ordered fold IS the contract, so sharding it needs an
ordered merge rather than a concatenation. It is also already distributed
whenever it sits inside `SambaStack`, which has a path, so standalone value is
low. `metrics.silhouette_score` and `metrics.trustworthiness` are the only
metrics with enough pairwise work to matter; the rest are label counts.

### Named refusals inside families that DO have a path

These are gaps, but they are gaps in a covered family and each is small. None
is counted in (b).

| refusal | where |
|---|---|
| ARIMA with `exog` is not sharded | `parallel_classical.fit_arima` raises `NotImplementedError` by name |
| `kernel='laplacian'` has no distributed row seam | `parallel_classical._admit_kernel_method` |
| wide full PCA (`rows < columns`, `svd_solver='full'`) | `parallel_classical.fit_gram_estimator` |
| `GaussianMixture.sample` has no parallel entry | `python/mojolearn/mixture.py:333` |
| `GaussianProcessRegressor.sample_y` has no parallel entry | `python/mojolearn/_gp_impl.py:1059` |
| RadiusNeighbors and KernelDensity have query sharding only, no reference sharding | `ReferenceShardedNeighbors` admits three KNN classes only |
| IsolationForest refits its forest on every score call | `parallel_ensemble.score_isolation_forest` docstring |
| Cholesky panel order and a single right-hand side are sequential | `parallel_classical.solve_cholesky` docstring, by construction |

## (c) Multi-GPU is not meaningful (19)

This is a result, not a gap. Nothing below benefits from a second device.

| surface(s) | why not |
|---|---|
| Mamba1Block, Mamba2Block, Mamba3Block, TransformerBlock (4) | forward and backward primitives, not fit drivers. They ARE distributed already whenever they sit inside `SambaStack`, which has a path. A device split of one block at the shipped shapes is transport bound and partitions no shared state. |
| MLPInference, TransformerBlockInference, Mamba1/2/3BlockInference, SambaInference, LanguageModelInference (7) | one forward pass over independent rows. A caller already gets full throughput by running two processes, there is no shared state to partition, and a row split introduces no new arithmetic that a gate could check. |
| HostForest, HostGBDT, LanguageModelHostTrainer (3) | CPU host surfaces by definition. There is no device in the path. |
| GPT2Tokenizer (1) | integer table lookups on the host. No device arithmetic. |
| kpss_test (1) | one small statistic over one series, sequential partial sums. |
| clip_grad_norm_, cross_entropy (2) | single small tensors standalone; both are already partitioned inside `ParallelNeuralTrainer` (whole clipping tensors on owners, the cross-tensor norm on the first device). |
| SGD, Adam, AdamW (1, counted as the optimizer surface) | optimizer ranges are already pooled inside the neural and byte-LM trainers at `MOJOLEARN_OPTIMIZER_DEVICE_COUNT`. Standalone there is nothing to split. |

`OffloadedByteLanguageModelTrainer` deserves a line of its own. It admits
EXACTLY one device on purpose (`python/mojolearn/offload_training.py:40`). It
is a memory path, not a multi-GPU path, and it should never be counted as one.

## Cost basis

The three most recent full drivers on main landed at 614, 680 and 780 lines
across eight or nine files each (binding flag, a `*/multi_gpu.mojo`, the Python
driver, the worker operation, a native check, a `tools/parallel_*_check.py`, a
design note under `docs/multi_gpu/`), and each owed a two-H100 and a two-MI300X
leg:

    git show --stat 00037ee37   # Cholesky, 614 lines, 9 files
    git show --stat b9f219a9e   # resample,  680 lines, 8 files
    git show --stat dc0a6ab3c   # GaussianMixture, 780 lines, 9 files

A Python-only driver with no native seam (the shape b1, b2 and b3's search half
take) is far smaller: `transform_rbf_sampler` plus its worker operation is
about 60 lines.

## Done on this branch

- This audit. No code change.

## Resume commands for a session with none of this context

Read this file, then `docs/lanes/LANE_STATUS_lane-cpu-training-par-wave3.md`
(the CPU verification triage of the same 39 lanes) and
`docs/multi_gpu/COVERAGE.md` (the prose, which this audit found accurate on the
IVF and Mamba/Transformer statements).

**1. Take the worktree.**

    cd /Users/andrewhendel/CascadeProjects/mojolearn   # SHARED: never build or commit here
    git worktree add -b lane/multigpu-audit <scratch>/wt-multigpu-audit origin/main
    cd <scratch>/wt-multigpu-audit && git fetch origin main && git merge origin/main

**2. Reproduce the counts from the tree, not from this file.**

    grep -c '@lane("par-' tools/identity_break.py                    # 39
    grep -c '@lane(' tools/identity_break.py                         # 176
    git grep -n 'DevicePool(devices' python/mojolearn/parallel_*.py  # cooperative vs not, per driver
    git grep -ho 'MOJOLEARN_[A-Z_]*DEVICE_COUNT' -- '*.mojo' '*.py' | sort -u   # 17 switches

**3. Part 2, only with Andrew's word, in this order.** b1 (cross_val_score),
then the eleven (a3) lanes that need no driver, then b2 (GPC), then b3 (IVF
search), then b5 (ARIMA and Holt-Winters prediction), then b4 (matmul), then
b6. Each one owes, before it merges:

- the `par-*` identity lane, small fixtures, held to the plain one-device path
  with `_same_bytes`;
- the CPU verification where it is honest (b1 and b2's sharded route qualify
  under wave 3's shape 1; a cooperative arm does not and must refuse by name);
- the sabotage arm SEEN to fail, not assumed to.

**4. Gates before any push to main.**

    python3 tools/docs_facts.py --check
    python3 packaging/wheel_ci.py pins .
    python3 packaging/wheel_ci.py inventory python/mojolearn

`tools/docs_facts.py` scans only `README.md`, `CONTRIBUTING.md`,
`docs/RELEASE_CHECKLIST.md`, `SUPPORT_MATRIX.md` and
`docs/BYTE_LM_CPU_TRAINING.md`, so this lane status file is outside it.

## Rules this lane ran under

Read only. One core, `nice -n 19`, one process at a time, its own worktree. No
build, no run, no Metal lock (the 0.8.6 Apple column is recording). Nothing
rented. Never `git stash` in the shared checkout. Main only; the frozen release
commit `db9047b9f` and `release/0.8.6` were not touched.

## Pods

None rented on this branch.

## Owed

Every (b) item owes a two-device leg on two H100s and on two MI300X, and no box
is authorized. The eleven (a3) lanes owe nothing but a place in the next
record's two-device columns.
