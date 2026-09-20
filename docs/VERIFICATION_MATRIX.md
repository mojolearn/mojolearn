# The verification matrix

GENERATED. Do not edit by hand. `python3 tools/verification_matrix.py --write`
rebuilds it from the tree, and `--check` fails when this file is stale.
Every number below is read from `tools/identity_break.py`,
`python/mojolearn/host_surface.py` and the committed columns under
`bench/results/`. The tool's own docstring says how each cell is decided.

This is a historical coverage inventory, not qualification of the current wheel.
Public API entries include aliases, wrappers and helpers; they are not a count
of distinct algorithms. A lane with all four kinds still needs current, matching
release artifacts and all applicable backend/property checks.

The four kinds, for one lane:

1. **gpu**, a recorded GPU column carries the lane, on 1, 2 or 3 of the
   device classes apple, nvidia and amd.
2. **cpu**, a CPU verifier covers the lane, for training or for inference
   from a saved model, as `host_surface.py` declares it.
3. **sabotage**, a negative control that HAS BEEN SEEN to move this lane's
   bytes in a committed pair of columns. `declared` means a define exists
   and nobody has watched it fail here. `seen(harness)` means only the
   harness's own batch switch moved, which proves the probe can fail and
   says nothing about the implementation.
4. **batch**, a declared batch-invariance part, or a named `n/a:<reason>`.

## The numbers

- Lanes: **271** (212 single-device, 59 `par-*` multi-GPU drivers).
- Source public API entries enumerated from the public API: **255**.
- Source public API entries with ALL FOUR kinds on at least one lane: **212** of 255.
- Source public API entries with NO IDENTITY LANE AT ALL: **0**.
- Source public API entries with no lane of their own, but reached by the harness's
  CPU inference routing: **0**.

Per kind, over the public API entries:

| kind | API entries that have it | missing |
|---|---|---|
| gpu column | 250 | 5 |
| cpu verifier | 222 | 33 |
| sabotage seen to move a build | 217 | 38 |
| batch part or named n/a | 255 | 0 |

Per kind, over the lanes:

| kind | lanes that have it | missing |
|---|---|---|
| gpu column (any class) | 267 | 4 |
| gpu column on all three classes | 235 | 36 |
| cpu verifier declared | 237 | 34 |
| sabotage seen to move a build | 230 | 41 |
| batch part or named n/a | 271 | 0 |
| ALL FOUR | 227 | 44 |

Sabotage, split by what was actually watched:

| verdict | lanes | what it means |
|---|---|---|
| seen(build) | 230 | a sabotage BUILD moved the bytes; a real negative control |
| seen(harness) | 3 | only the harness batch switch moved; the probe can fail, the build is unproven |
| declared | 7 | the family declares a define; no committed pair moves this lane |
| none | 31 | no define reaches the lane and nothing has moved it |

## Source public API entries with no identity lane at all

None.

### Reached by the harness, with no lane of their own

`tools/identity_break.py` reaches these outside any lane body. `_public_est`
swaps the fitted estimator for the public CPU inference class on a CPU
column, so those answer the infer, reload and batch cells of the lanes named
in `NEURAL_PUBLIC_PART_LANES` (mamba1, mamba2, mamba3, mamba2-dtlimit, samba, samba-untied-dropout-accum, byte-lm, byte-lm-resident, mamba1-bf16w, mamba1-int8w, mamba2-bf16w, mamba2-int8w, mamba3-bf16w, mamba3-int8w, transformer, transformer-window);
the batchgrad part calls the accumulation helpers the same way.
They are verified, but no lane carries their name.

None.

## The algorithm matrix

`gpu` is the device classes that carry any of the algorithm's lanes.
A blank cell means no lane of this algorithm has that kind.

| algorithm | lanes | gpu | cpu | sabotage | batch | all four |
|---|---|---|---|---|---|---|
| `ARIMA` | 7 | amd,apple,nvidia | training | seen(build) | part | yes |
| `Adam` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `AdamW` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `AgglomerativeClustering` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `BpeTokenizer` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `ByteLanguageModelConfig` | 9 | amd,apple,nvidia | training | seen(build) | part | yes |
| `Cholesky` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `ConstantKernel` | 7 | amd,apple,nvidia | training | seen(build) | part | yes |
| `DBSCAN` | 4 | amd,apple,nvidia | training | seen(build) | part | yes |
| `DistributedIVFIndex` | 1 | nvidia | training | seen(build) | part | yes |
| `ElasticNet` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `Embedding` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `ExperimentalTwoLevelFeatureFreq` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `ExponentialSmoothing` | 4 | amd,apple,nvidia | training | seen(build) | part | yes |
| `ExtraTreesClassifier` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `ExtraTreesRegressor` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `GPT2Tokenizer` (alias of `BpeTokenizer`) | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `GaussianMixture` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `GaussianProcessClassifier` | 4 | amd,apple,nvidia | training | seen(build) | part | yes |
| `GaussianProcessRegressor` | 5 | amd,apple,nvidia | training | seen(build) | part | yes |
| `GradientBoosting` | 28 | amd,apple,nvidia | training | seen(build) | part | yes |
| `GradientBoostingClassifier` | 4 | amd,apple,nvidia | training | seen(build) | part | yes |
| `GradientBoostingRegressor` | 6 | amd,apple,nvidia | training | seen(build) | part | yes |
| `HDBSCAN` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `HostForest` | 1 |  | training | seen(build) | part | NO |
| `HostGBDT` | 1 |  | training | seen(build) | part | NO |
| `IVFIndex` | 4 | amd,apple,nvidia | training | seen(build) | part | yes |
| `IsolationForest` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `KMeans` | 9 | amd,apple,nvidia | training | seen(build) | part | yes |
| `KNeighborsClassifier` | 4 | amd,apple,nvidia | training | seen(build) | part | yes |
| `KNeighborsRegressor` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `KernelDensity` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `KernelRidge` | 8 | amd,apple,nvidia | training | seen(build) | part | yes |
| `LanguageModelConfig` (alias of `ByteLanguageModelConfig`) | 9 | amd,apple,nvidia | training | seen(build) | part | yes |
| `LanguageModelHostTrainer` | 1 | amd,apple,nvidia | training | seen(build) | n/a | yes |
| `LanguageModelInference` | 19 | amd,apple,nvidia | training | seen(build) | part | yes |
| `LanguageModelTrainer` (alias of `SmallByteLanguageModelTrainer`) | 5 | amd,apple,nvidia | training | seen(build) | part | yes |
| `Lasso` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `LinearRegression` | 4 | amd,apple,nvidia | training | seen(build) | part | yes |
| `LinearSVC` | 2 | nvidia | training | declared | part | NO |
| `LinearSVR` | 2 | nvidia | training | declared | part | NO |
| `LogisticRegression` | 6 | amd,apple,nvidia | training | seen(build) | part | yes |
| `MLPInference` | 16 | amd,apple,nvidia | training | seen(build) | part | yes |
| `Mamba1Block` | 10 | amd,apple,nvidia | training | seen(build) | part | yes |
| `Mamba1BlockInference` | 16 | amd,apple,nvidia | training | seen(build) | part | yes |
| `Mamba2Block` | 10 | amd,apple,nvidia | training | seen(build) | part | yes |
| `Mamba2BlockInference` | 16 | amd,apple,nvidia | training | seen(build) | part | yes |
| `Mamba3Block` | 9 | amd,apple,nvidia | training | seen(build) | part | yes |
| `Mamba3BlockInference` | 16 | amd,apple,nvidia | training | seen(build) | part | yes |
| `Matern` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `MinMaxScaler` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `NearestNeighbors` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `Nystroem` | 8 | amd,apple,nvidia | training | seen(build) | part | yes |
| `OffloadedByteLanguageModelTrainer` | 1 | amd,apple,nvidia |  | none | n/a | NO |
| `OrderedRMSE` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `PCA` | 4 | amd,apple,nvidia | training | seen(build) | part | yes |
| `ParallelByteLanguageModelTrainer` | 3 | amd,apple,nvidia |  | none | n/a | NO |
| `ParallelNeuralTrainer` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `PooledByteLanguageModelTrainer` | 1 | amd,apple,nvidia |  | none | n/a | NO |
| `QNRegressor` | 2 | nvidia | training | declared | part | NO |
| `RBF` | 6 | amd,apple,nvidia | training | seen(build) | part | yes |
| `RBFSampler` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `RadiusNeighbors` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `RandomForestClassifier` | 7 | amd,apple,nvidia | training | seen(build) | part | yes |
| `RandomForestRegressor` | 6 | amd,apple,nvidia | training | seen(build) | part | yes |
| `Ridge` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `SGD` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `SVC` | 4 | amd,apple,nvidia | training | seen(build) | part | yes |
| `SVR` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `SambaConfig` | 6 | amd,apple,nvidia | training | seen(build) | part | yes |
| `SambaInference` | 16 | amd,apple,nvidia | training | seen(build) | part | yes |
| `SambaStack` | 6 | amd,apple,nvidia | training | seen(build) | part | yes |
| `SmallByteLanguageModelTrainer` | 5 | amd,apple,nvidia | training | seen(build) | part | yes |
| `SmallMLPTrainer` | 4 | amd,apple,nvidia | training | seen(build) | part | yes |
| `SpectralClustering` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `SpectralEmbedding` | 1 | nvidia | training | seen(build) | n/a | yes |
| `StandardScaler` | 4 | amd,apple,nvidia | training | seen(build) | part | yes |
| `TransformerBlock` | 11 | amd,apple,nvidia | training | seen(build) | part | yes |
| `TransformerBlockInference` | 16 | amd,apple,nvidia | training | seen(build) | part | yes |
| `TruncatedSVD` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `UMAP` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `WhiteKernel` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `clip_grad_norm_` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `cross_entropy` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `cross_val_score` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `embedding.Embedding` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `hdbscan.HDBSCAN` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `hdbscan.all_points_membership_vectors` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `hdbscan.approximate_predict` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `hdbscan.membership_vector` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `host_model` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `host_predict` | 1 |  | training | seen(build) | part | NO |
| `host_predict_proba` | 1 |  | training | seen(build) | part | NO |
| `kernel_methods.KernelRidge` | 8 | amd,apple,nvidia | training | seen(build) | part | yes |
| `kernel_methods.Nystroem` | 8 | amd,apple,nvidia | training | seen(build) | part | yes |
| `kernel_methods.RBFSampler` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `kpss_test` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `language_model.ByteLanguageModelConfig` | 9 | amd,apple,nvidia | training | seen(build) | part | yes |
| `language_model.LanguageModelConfig` (alias of `ByteLanguageModelConfig`) | 9 | amd,apple,nvidia | training | seen(build) | part | yes |
| `language_model.LanguageModelHostTrainer` | 1 | amd,apple,nvidia | training | seen(build) | n/a | yes |
| `language_model.LanguageModelInference` | 19 | amd,apple,nvidia | training | seen(build) | part | yes |
| `language_model.LanguageModelTrainer` (alias of `SmallByteLanguageModelTrainer`) | 5 | amd,apple,nvidia | training | seen(build) | part | yes |
| `language_model.SmallByteLanguageModelTrainer` | 5 | amd,apple,nvidia | training | seen(build) | part | yes |
| `linalg.Cholesky` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `linalg.dequantize_int8` | 1 | amd,apple,nvidia | training | seen(build) | n/a | yes |
| `linalg.eigh` | 1 | amd,apple,nvidia | training | seen(build) | n/a | yes |
| `linalg.from_bf16` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `linalg.matmul` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `linalg.matmul_bf16` | 1 | amd,apple,nvidia | training | seen(build) | n/a | yes |
| `linalg.matmul_int8` | 1 | amd,apple,nvidia | training | seen(build) | n/a | yes |
| `linalg.qr` | 1 | amd,apple,nvidia | training | seen(build) | n/a | yes |
| `linalg.quantize_int8` | 1 | amd,apple,nvidia | training | seen(build) | n/a | yes |
| `linalg.svdvals` | 1 | amd,apple,nvidia | training | seen(build) | n/a | yes |
| `linalg.to_bf16` | 1 | amd,apple,nvidia | training | seen(build) | n/a | yes |
| `lm_corpus.TokenBatches` | 1 | amd,apple,nvidia | training | seen(build) | n/a | yes |
| `lm_corpus.TokenizedCorpus` | 1 | amd,apple,nvidia | training | seen(build) | n/a | yes |
| `lm_corpus.prepare` | 1 | amd,apple,nvidia | training | seen(build) | n/a | yes |
| `lm_corpus.require_vocabulary` | 1 | amd,apple,nvidia | training | seen(build) | n/a | yes |
| `lm_corpus.tokenizer_for` | 1 | amd,apple,nvidia | training | seen(build) | n/a | yes |
| `lowbit.format_of` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `lowbit.is_packed` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `lowbit.materialize_one` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `lowbit.pack` | 12 | amd,apple,nvidia | training | seen(build) | part | yes |
| `lowbit.pack_one` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `lowbit.unpack` | 4 | amd,apple,nvidia | training | seen(build) | n/a | yes |
| `lowbit.widen_bf16` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `mamba.Mamba1Block` | 10 | amd,apple,nvidia | training | seen(build) | part | yes |
| `mamba.Mamba1DecodeSession` | 1 | nvidia | training | seen(build) | n/a | yes |
| `mamba.Mamba2Block` | 10 | amd,apple,nvidia | training | seen(build) | part | yes |
| `mamba.Mamba3Block` | 9 | amd,apple,nvidia | training | seen(build) | part | yes |
| `manifold.SpectralEmbedding` | 1 | nvidia | training | seen(build) | n/a | yes |
| `manifold.UMAP` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `manifold.spectral_embedding` | 1 | nvidia | training | seen(build) | n/a | yes |
| `matmul` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `metrics.accuracy_score` | 1 | amd,apple,nvidia | training | seen(build) | n/a | yes |
| `metrics.adjusted_rand_score` | 1 | amd,apple,nvidia | training | seen(build) | n/a | yes |
| `metrics.completeness_score` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `metrics.confusion_matrix` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `metrics.entropy` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `metrics.f1_score` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `metrics.fowlkes_mallows_score` | 1 | amd,apple,nvidia | training | seen(build) | n/a | yes |
| `metrics.homogeneity_completeness_v_measure` | 1 | amd,apple,nvidia | training | seen(build) | n/a | yes |
| `metrics.homogeneity_score` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `metrics.kl_divergence` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `metrics.log_loss` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `metrics.mean_absolute_error` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `metrics.mean_squared_error` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `metrics.mutual_info_score` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `metrics.precision_recall_curve` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `metrics.precision_score` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `metrics.r2_score` | 1 | amd,apple,nvidia | training | seen(build) | n/a | yes |
| `metrics.rand_score` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `metrics.recall_score` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `metrics.roc_auc_score` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `metrics.root_mean_squared_error` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `metrics.silhouette_samples` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `metrics.silhouette_score` | 1 | amd,apple,nvidia | training | seen(build) | n/a | yes |
| `metrics.trustworthiness` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `metrics.v_measure_score` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `mixture.GaussianMixture` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `model_pool_training.PooledByteLanguageModelTrainer` | 1 | amd,apple,nvidia |  | none | n/a | NO |
| `model_selection.cross_val_score` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `model_selection.split_descriptor` | 1 |  | host-only | seen(build) | n/a | NO |
| `models.CausalLM` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `models.Checkpoint` | 1 | amd,apple,nvidia | training | seen(build) | n/a | yes |
| `models.ParallelCausalLM` | 1 | nvidia | training | seen(build) | part | yes |
| `models.SafetensorsFile` | 1 | amd,apple,nvidia | training | seen(build) | n/a | yes |
| `models.Tokenizer` | 1 | amd,apple,nvidia | training | seen(build) | n/a | yes |
| `models.plan_for` | 1 | amd,apple,nvidia | training | seen(build) | n/a | yes |
| `models.tokenizer.pattern_name` | 1 | amd,apple,nvidia | training | seen(build) | n/a | yes |
| `models.tokenizer.pretokenize` | 1 | amd,apple,nvidia | training | seen(build) | n/a | yes |
| `neural_network.SmallMLPTrainer` | 4 | amd,apple,nvidia | training | seen(build) | part | yes |
| `offload_training.OffloadedByteLanguageModelTrainer` | 1 | amd,apple,nvidia |  | none | n/a | NO |
| `parallel_classical.apply_kernel_method` | 2 | amd,apple,nvidia |  | none | part | NO |
| `parallel_classical.bootstrap` | 1 | amd,apple,nvidia |  | none | part | NO |
| `parallel_classical.fit_arima` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `parallel_classical.fit_cholesky` | 1 | amd,apple,nvidia |  | none | part | NO |
| `parallel_classical.fit_coordinate_descent` | 2 | amd,apple,nvidia |  | none | part | NO |
| `parallel_classical.fit_dbscan` | 1 | amd,apple,nvidia |  | seen(harness) | part | NO |
| `parallel_classical.fit_exponential_smoothing` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `parallel_classical.fit_gaussian_mixture` | 1 | amd,apple,nvidia |  | none | part | NO |
| `parallel_classical.fit_gaussian_process` | 1 | amd,apple,nvidia |  | none | part | NO |
| `parallel_classical.fit_gram_estimator` | 4 | amd,apple,nvidia |  | none | part | NO |
| `parallel_classical.fit_hdbscan` | 1 | amd,apple,nvidia |  | none | part | NO |
| `parallel_classical.fit_kernel_method` | 2 | amd,apple,nvidia |  | none | part | NO |
| `parallel_classical.fit_kmeans` | 1 | amd,apple,nvidia |  | seen(harness) | part | NO |
| `parallel_classical.fit_logistic` | 1 | amd,apple,nvidia |  | none | part | NO |
| `parallel_classical.fit_svm` | 2 | amd,apple,nvidia |  | none | part | NO |
| `parallel_classical.monte_carlo_integrate` | 1 | amd,apple,nvidia |  | none | part | NO |
| `parallel_classical.permutation_test` | 1 | amd,apple,nvidia |  | none | part | NO |
| `parallel_classical.predict_gaussian_mixture` | 1 | amd,apple,nvidia |  | none | part | NO |
| `parallel_classical.predict_gaussian_process` | 1 | amd,apple,nvidia |  | none | part | NO |
| `parallel_classical.predict_svm` | 2 | amd,apple,nvidia |  | none | part | NO |
| `parallel_classical.solve_cholesky` | 1 | amd,apple,nvidia |  | none | part | NO |
| `parallel_classical.transform_rbf_sampler` | 1 | amd,nvidia | training | seen(build) | part | yes |
| `parallel_ensemble.ParallelForestPredictor` | 1 | amd,nvidia |  | none | part | NO |
| `parallel_ensemble.fit_boosting` | 6 | amd,apple,nvidia |  | none | part | NO |
| `parallel_ensemble.fit_feature_freq` | 1 | amd,apple,nvidia |  | none | part | NO |
| `parallel_ensemble.fit_forest` | 4 | amd,apple,nvidia | training | seen(build) | part | yes |
| `parallel_ensemble.fit_isolation_forest` | 1 | amd,apple,nvidia |  | none | part | NO |
| `parallel_ensemble.fit_ordered_rmse` | 1 | amd,apple,nvidia |  | none | part | NO |
| `parallel_ensemble.score_isolation_forest` | 1 | amd,apple,nvidia |  | none | part | NO |
| `parallel_forecasting.forecast_arima` | 1 | nvidia | training | seen(build) | part | yes |
| `parallel_forecasting.forecast_exponential_smoothing` | 1 | nvidia | training | seen(build) | part | yes |
| `parallel_forecasting.predict_arima` | 1 | nvidia | training | seen(build) | part | yes |
| `parallel_forecasting.predict_exponential_smoothing` | 1 | nvidia | training | seen(build) | part | yes |
| `parallel_gaussian_process.fit_gaussian_process_classifier` | 1 | nvidia | training | seen(build) | part | yes |
| `parallel_gaussian_process.predict_gaussian_process_classifier` | 1 | nvidia | training | seen(build) | part | yes |
| `parallel_graph.fit_graph` | 3 | amd,apple,nvidia |  | seen(harness) | part | NO |
| `parallel_graph.transform_umap` | 1 | amd,apple,nvidia |  | none | part | NO |
| `parallel_ivf.DistributedIVFIndex` | 1 | nvidia | training | seen(build) | part | yes |
| `parallel_model_selection.cross_val_score` | 1 | nvidia | training | seen(build) | n/a | yes |
| `parallel_neighbors.ParallelQueries` | 4 | amd,apple,nvidia | training | seen(build) | part | yes |
| `parallel_neighbors_reference.ReferenceShardedNeighbors` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `parallel_preprocessing.fit_scaler` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `parallel_preprocessing.transform_scaler` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `parallel_training.ParallelByteLanguageModelTrainer` | 3 | amd,apple,nvidia |  | none | n/a | NO |
| `parallel_training.ParallelNeuralTrainer` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `parallel_training.ordered_sum_gradients` | 4 | amd,apple,nvidia | training | seen(build) | part | yes |
| `preprocessing.MinMaxScaler` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `preprocessing.StandardScaler` | 4 | amd,apple,nvidia | training | seen(build) | part | yes |
| `resample.bootstrap` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `resample.monte_carlo_integrate` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `resample.permutation_test` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `select_d` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `svm.LinearSVC` | 2 | nvidia | training | declared | part | NO |
| `svm.LinearSVR` | 2 | nvidia | training | declared | part | NO |
| `svm.SVC` | 4 | amd,apple,nvidia | training | seen(build) | part | yes |
| `svm.SVR` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `tokenizer.BpeTokenizer` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `tokenizer.BpeVocabularyTrainer` | 3 | amd,apple,nvidia | training | seen(build) | n/a | yes |
| `tokenizer.GPT2Tokenizer` (alias of `BpeTokenizer`) | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `tokenizer.TrainedBpeVocabulary` | 1 | amd,apple,nvidia | training | seen(build) | n/a | yes |
| `training.Adam` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `training.AdamW` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `training.ConstantLR` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `training.Generator` | 6 | amd,apple,nvidia | training | seen(build) | part | yes |
| `training.SGD` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `training.SambaConfig` | 6 | amd,apple,nvidia | training | seen(build) | part | yes |
| `training.SambaStack` | 6 | amd,apple,nvidia | training | seen(build) | part | yes |
| `training.WarmupCosineLR` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `training.WarmupLinearLR` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `training.accumulate_grads` | 1 | amd,apple,nvidia | training | seen(build) | n/a | yes |
| `training.accumulation_is_aligned` | 1 | amd,apple,nvidia | training | seen(build) | n/a | yes |
| `training.clip_grad_norm_` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `training.cross_entropy` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `training.embedding_backward` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `training.embedding_forward` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `training.linear_backward` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `training.linear_forward` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `training.rms_norm_backward` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `training.rms_norm_forward` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `transformer.TransformerBlock` | 11 | amd,apple,nvidia | training | seen(build) | part | yes |
| `transformer.TransformerDecodeSession` | 1 | nvidia | training | seen(build) | n/a | yes |
| `umap.UMAP` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |

## The lane matrix

| lane | gpu | cpu | sabotage | sabotage evidence | batch | all four |
|---|---|---|---|---|---|---|
| agglomerative | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-14_cpu-phase1b/cpu-apple-m4.sabotage.agglomerative-et.json` | part | yes |
| arima | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-forecast-umap-pca/cpu-apple-m4.sabotage.json` | part | yes |
| arima-011 | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-forecast-umap-pca/cpu-apple-m4.sabotage.json` | part | yes |
| arima-exog | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_arima-exog/x86-runpod/cpu-x86.exog-sabotage.json` | part | yes |
| arima-exog-seasonal | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_arima-exog/x86-runpod/cpu-x86.exog-sabotage.json` | part | yes |
| arima-seasonal-c | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-forecast-umap-pca/cpu-apple-m4.sabotage.json` | part | yes |
| bootstrap | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-16_sabotage-audit/r2-sab.json` | part | yes |
| bpe-trainer | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_sabotage-sweep/e-python-lanes/cpu-x86.sabotage.json` | n/a n/a:corpus-global-vocabulary-training (pair counts depend on the complete corpus; no per-row output) | yes |
| bpe-vocabulary | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-18_bpe-builder-native/m4.trainer.sabotage-build.json` | n/a n/a:corpus-global-vocabulary-training (pair counts depend on the complete corpus; no per-row output) | yes |
| byte-lm | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_neural-forward-inference/cpu-amd-epyc-4564p.host-sabotage.json` | part | yes |
| byte-lm-host-infer | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-native-nine/records/byte-lm-host-infer/byte-lm-host-infer/cpu-sabotage.json` | part | yes |
| byte-lm-host-infer-threaded | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-native-nine/records/byte-lm-host-infer-threaded/byte-lm-host-infer-threaded/cpu-sabotage.json` | part | yes |
| byte-lm-host-train | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-native-nine/records/byte-lm-host-train/byte-lm-host-train/cpu-sabotage.json` | n/a n/a:mean-reduction-fixed-batch (LanguageModelHostTrainer has train_step and loss only, and loss IS a step, python/mojolearn/_byte_lm_host.py:392-429; one loss and one update from mean cross-entropy over ids of the profile's fixed (batch, length + 1), so no output belongs to one sequence; the per-sequence logits are asked by byte-lm-host-infer; lane/batch-invariance-2's batchgrad records the same reason for this lane and its batchscale and ragged parts do not ask it) | yes |
| byte-lm-resident | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_neural-forward-inference/cpu-amd-epyc-4564p.host-sabotage.json` | part | yes |
| cholesky | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cholesky-cpu-inference/cpu-apple-m4.sabotage.json` | part | yes |
| cross-entropy-arms | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-native-nine/records/cross-entropy-arms/cross-entropy-arms/cpu-sabotage.json` | part | yes |
| cross-val | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-native-nine/records/cross-val/cross-val/cpu-sabotage.json` | part | yes |
| cross-val-folds | - | host-only | seen(build) | `bench/results/identity_break/2026-09-17_sabotage-sweep/e-python-lanes/cpu-x86.sabotage.json` | n/a n/a:function | NO |
| dbscan | amd,apple,nvidia | training | seen(build) | `bench/results/cpu-verification-completion-probe/2026-09-17/dbscan-repair-provenance-failure/cpu-sabotage.json (clean partner at another commit)` | part | yes |
| dbscan-brute-l1 | amd,apple,nvidia | training | seen(build) | `bench/results/cpu-verification-completion-probe/2026-09-17/dbscan-repair-provenance-failure/cpu-sabotage.json (clean partner at another commit)` | part | yes |
| dbscan-weighted | amd,apple,nvidia | training | seen(build) | `bench/results/cpu-verification-completion-probe/2026-09-17/dbscan-repair-provenance-failure/cpu-sabotage.json (clean partner at another commit)` | part | yes |
| elasticnet | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-linear-kernel/cpu-apple-m4-sabotage.json` | part | yes |
| elasticnet-l2end-no-intercept | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-linear-kernel/cpu-apple-m4-sabotage.json` | part | yes |
| embedding | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_ivf-embedding-cpu-inference/cpu-x86.sabotage.json` | part | yes |
| embedding-sort | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_ivf-embedding-cpu-inference/cpu-x86.sabotage.json` | part | yes |
| et-clf | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-14_cpu-phase1b/cpu-apple-m4.sabotage.et.json` | part | yes |
| et-clf-entropy-bestfirst | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-native-nine/records/et-clf-entropy-bestfirst/et-clf-entropy-bestfirst/cpu-sabotage.json` | part | yes |
| et-reg | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-14_cpu-phase1b/cpu-apple-m4.sabotage.et.json` | part | yes |
| et-reg-bootstrap-parallel | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-native-nine/records/et-reg-bootstrap-parallel/et-reg-bootstrap-parallel/cpu-sabotage.json` | part | yes |
| gbdt-adapter-clf | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-native-nine/records/gbdt-adapter-clf/gbdt-adapter-clf/cpu-sabotage.json` | part | yes |
| gbdt-adapter-reg | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-native-nine/records/gbdt-adapter-reg/gbdt-adapter-reg/cpu-sabotage.json` | part | yes |
| gbdt-adapter-score-weighted | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-classical-completion/gbdt-adapter-score-weighted/cpu-sabotage.json` | n/a n/a:scalar-reduction (score(X, y, sample_weight) returns one float over every row it is handed, python/mojolearn/_gbdt_adapters.py and _forest_protocol.py; the predict and predict_proba it wraps keep their batch parts on gbdt-adapter-clf, gbdt-adapter-reg, rf-clf and rf-reg) | yes |
| gbdt-bfa-quantile | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-19_gbdt-sabotage-pair/cpu-apple-m4.sabotage.json` | part | yes |
| gbdt-border-types | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-19_gbdt-sabotage-pair/cpu-apple-m4.sabotage.json` | part | yes |
| gbdt-catboost-defaults | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-19_gbdt-sabotage-pair/cpu-apple-m4.sabotage.json` | part | yes |
| gbdt-categorical-ctr | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-gbdt-modes/cpu-x86-forest-sabotage-host-infer.json` | part | yes |
| gbdt-categorical-ctr-tables | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-verifier-gaps-7/x86-runpod/cpu-sab.json` | part | yes |
| gbdt-depthwise | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-native-nine/records/gbdt-depthwise/gbdt-depthwise/cpu-sabotage.json` | part | yes |
| gbdt-exact-mae | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-native-nine/records/gbdt-exact-mae/gbdt-exact-mae/cpu-sabotage.json` | part | yes |
| gbdt-feature-freq | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-gbdt-modes/cpu-x86-forest-sabotage-host-infer.json` | part | yes |
| gbdt-lossguide | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-native-nine/records/gbdt-lossguide/gbdt-lossguide/cpu-sabotage.json` | part | yes |
| gbdt-lossguide-newtoncosine | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-classical-completion/gbdt-lossguide-newtoncosine/cpu-sabotage.json` | part | yes |
| gbdt-multiclass | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-native-nine/records/gbdt-multiclass/gbdt-multiclass/cpu-sabotage.json` | part | yes |
| gbdt-nan-modes | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-classical-completion/gbdt-nan-modes/cpu-sabotage.json` | part | yes |
| gbdt-onevsall | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-native-nine/records/gbdt-onevsall/gbdt-onevsall/cpu-sabotage.json` | part | yes |
| gbdt-ordered | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-19_gbdt-sabotage-pair/cpu-apple-m4.sabotage.json` | part | yes |
| gbdt-ordered-bayesian-noise | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-19_gbdt-sabotage-pair/cpu-apple-m4.sabotage.json` | part | yes |
| gbdt-ordered-rmse | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-gbdt-modes/cpu-x86-forest-sabotage-host-infer.json` | part | yes |
| gbdt-pair-logit | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_gbdt-pair-logit/confirm-3b1b2d6f6/cpu-x86-64-runpod.sabotage.json` | part | yes |
| gbdt-parametric-losses | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-classical-completion/gbdt-parametric-losses/cpu-sabotage.json` | part | yes |
| gbdt-pointwise-l2-bayesian-eval | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-gbdt-modes/cpu-x86-forest-sabotage-host-infer.json` | part | yes |
| gbdt-query-rmse | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_gbdt-pair-logit/confirm-3b1b2d6f6/cpu-x86-64-runpod.sabotage.json` | part | yes |
| gbdt-rmse | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-native-nine/records/gbdt-rmse/gbdt-rmse/cpu-sabotage.json` | part | yes |
| gbdt-stochastic-arms | - | training | declared | - | part | NO |
| gbdt-symmetric | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-native-nine/records/gbdt-symmetric/gbdt-symmetric/cpu-sabotage.json` | part | yes |
| gbdt-symmetric-eval | nvidia | training | seen(build) | `bench/results/identity_break/2026-09-20_gbdt-symmetric-eval/cpu-apple-m4-sabotage.json (clean partner at another commit)` | part | yes |
| gbdt-tensor-ctr-tables | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-verifier-gaps-7/x86-runpod/cpu-sab.json` | part | yes |
| gbdt-yeti-rank | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_gbdt-yeti-rank/cpu-apple-m4.sabotage.json` | part | yes |
| gemm-bf16 | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-neural-completion/records/gemm-bf16/cpu-sabotage.json` | n/a n/a:profile lane; the products are hashed whole | yes |
| gemm-int8 | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-neural-completion/records/gemm-int8/cpu-sabotage.json` | n/a n/a:profile lane; the products are hashed whole | yes |
| gemm-pinned | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-16_sabotage-audit/r2-sab.json` | part | yes |
| gemm-transposed | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-16_sabotage-audit/r2-sab.json` | part | yes |
| gmm | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-d-estimators/cpu-apple-m4.sabotage.json` | part | yes |
| gmm-random-init | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-d-estimators/cpu-apple-m4.sabotage.json` | part | yes |
| gmm-random-init-sample | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-verifier-gaps-7/x86-runpod/cpu-sab.json` | n/a n/a:no-batch-axis (GaussianMixture.sample takes no input rows) | yes |
| gmm-sample | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-verifier-gaps-7/x86-runpod/cpu-sab.json` | n/a n/a:no-batch-axis (GaussianMixture.sample takes no input rows) | yes |
| gp | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-d-estimators/cpu-apple-m4.sabotage.json` | part | yes |
| gp-matern12 | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-d-estimators/cpu-apple-m4.sabotage.json` | part | yes |
| gp-matern32 | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-d-estimators/cpu-apple-m4.sabotage.json` | part | yes |
| gp-matern52-ard | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-d-estimators/cpu-apple-m4.sabotage.json` | part | yes |
| gp-normalize-y | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_gp-sample-y/cpu-x86-epyc.host-sabotage.json` | part | yes |
| gp-optimize | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_gp-optimize/x86-runpod/cpu_gsab.json` | part | yes |
| gp-optimize-restarts | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_gp-optimize/x86-runpod/cpu_gsab.json` | part | yes |
| gp-sample-y | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-verifier-gaps-7/x86-runpod/cpu-sab.json` | n/a n/a:jointly-correlated (sample_y draws all rows of a call from one joint posterior) | yes |
| gp-sample-y-normalize | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-verifier-gaps-7/x86-runpod/cpu-sab.json` | n/a n/a:jointly-correlated (sample_y draws all rows of a call from one joint posterior) | yes |
| gpc | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_gpc/cpu-x86.host-sabotage.json` | part | yes |
| gpc-multiclass | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_gpc/cpu-x86.host-sabotage.json` | part | yes |
| grad-accumulation | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-19_laneless-public-classes/sabotage-training.json` | n/a n/a:accumulation-step (the A microbatches are the arithmetic the lane hashes; a call with fewer of them is a different accumulation, not a smaller batch of the same one) | yes |
| hdbscan | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-d-estimators/cpu-apple-m4.sabotage.json` | part | yes |
| hdbscan-leaf | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-d-estimators/cpu-apple-m4.sabotage.json` | part | yes |
| hf-causal-lm | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-19_models-namespace/m4.neural-host.sabotage.json (clean partner at another commit)` | part | yes |
| hf-checkpoint | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-19_models-namespace/m4.linalg-convert.sabotage.json (clean partner at another commit)` | n/a n/a:function (a header parse, a memoryview cast and the option matrix; the lane fits nothing and asks nothing for a row's answer) | yes |
| hf-tokenizer | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-19_models-namespace/m4.tokenizer-encoder.sabotage.json (clean partner at another commit)` | n/a n/a:no-batch-axis (mojolearn.models.Tokenizer encodes and decodes one document per call; it has no encode_batch, and a loop written here would test the harness, not the class) | yes |
| holtwinters | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_holtwinters-linesearch-fix/cpu-amd-epyc-9754.sabotage.json` | part | yes |
| holtwinters-multiplicative | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_holtwinters-linesearch-fix/cpu-amd-epyc-9754.sabotage.json` | part | yes |
| iforest | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-14_cpu-phase1b/cpu-apple-m4.sabotage.iforest.json` | part | yes |
| iforest-tuned | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-iforest-gmm-hdbscan/cpu-x86.iforest-gmm-hdbscan.sabotage.json` | part | yes |
| ivf | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_ivf-embedding-cpu-inference/cpu-x86.sabotage.json` | part | yes |
| ivf-euclidean | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_ivf-embedding-cpu-inference/cpu-x86.sabotage.json` | part | yes |
| ivf-extend | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_ivf-extend/cpu-x86.sabotage.json` | part | yes |
| kde | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-16_sabotage-audit/cpu-sab.json` | part | yes |
| kde-cosine-minkowski | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-neighbors-density/cpu-apple-m4.sabotage.json` | part | yes |
| kde-epanechnikov-l1 | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-neighbors-density/cpu-apple-m4.sabotage.json` | part | yes |
| kde-exponential-chebyshev | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-neighbors-density/cpu-apple-m4.sabotage.json` | part | yes |
| kde-linear-cosine | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-neighbors-density/cpu-apple-m4.sabotage.json` | part | yes |
| kde-tophat-sqeuclidean | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-neighbors-density/cpu-apple-m4.sabotage.json` | part | yes |
| kde-weighted | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-neighbors-density/cpu-apple-m4.sabotage.json` | part | yes |
| kernel-ridge | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-d-estimators/cpu-apple-m4.sabotage.json` | part | yes |
| kernel-ridge-laplacian | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-18-cpu-kernel-variants/sabotage-kernel-ridge-laplacian.json` | part | yes |
| kernel-ridge-poly | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-18-cpu-kernel-variants/sabotage-kernel-ridge-poly.json` | part | yes |
| kernel-ridge-sigmoid | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-18-cpu-kernel-variants/sabotage-kernel-ridge-sigmoid.json` | part | yes |
| kmeans | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_kmeans-predict/cpu-apple-m4.host-sabotage.json` | part | yes |
| kmeans-array | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_kmeans-predict/cpu-apple-m4.host-sabotage.json` | part | yes |
| kmeans-classic-pp | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_kmeans-predict/cpu-apple-m4.host-sabotage.json` | part | yes |
| kmeans-random | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_kmeans-predict/cpu-apple-m4.host-sabotage.json` | part | yes |
| kmeans-sqrt | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_kmeans-predict/cpu-apple-m4.host-sabotage.json` | part | yes |
| kmeans-weighted | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_kmeans-predict/cpu-apple-m4.host-sabotage.json` | part | yes |
| knn | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-16_sabotage-audit/cpu-sab.json` | part | yes |
| knn-chebyshev | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-neighbors-density/cpu-apple-m4.sabotage.json` | part | yes |
| knn-clf | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-16_sabotage-audit/cpu-sab.json` | part | yes |
| knn-clf-distance | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-neighbors-density/cpu-apple-m4.sabotage.json` | part | yes |
| knn-cosine | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-neighbors-density/cpu-apple-m4.sabotage.json` | part | yes |
| knn-manhattan | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-neighbors-density/cpu-apple-m4.sabotage.json` | part | yes |
| knn-minkowski-p3 | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-neighbors-density/cpu-apple-m4.sabotage.json` | part | yes |
| knn-rbc | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-neighbors-density/cpu-apple-m4.sabotage.json` | part | yes |
| knn-reg | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-16_sabotage-audit/cpu-sab.json` | part | yes |
| knn-reg-distance | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-neighbors-density/cpu-apple-m4.sabotage.json` | part | yes |
| knn-sqeuclidean | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-neighbors-density/cpu-apple-m4.sabotage.json` | part | yes |
| kpss | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_holtwinters-linesearch-fix/cpu-amd-epyc-9754.sabotage.json` | part | yes |
| language-model-config | - | training | seen(build) | `bench/results/identity_break/2026-09-19_language-model-config/cpu-sabotage.json` | n/a n/a:no-input-rows (a frozen config object; its shapes and offsets are functions of the config, and the lane passes no data) | NO |
| lasso | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-linear-kernel/cpu-apple-m4-sabotage.json` | part | yes |
| linalg-eigh | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-19_linalg-public/cpu-sabotage.json` | n/a n/a:whole-matrix-decomposition (a factorization reduces over every row; splitting the rows gives a different matrix, not a batch of the same call, and there is no per-row output) | yes |
| linalg-qr | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-19_linalg-public/cpu-sabotage.json` | n/a n/a:whole-matrix-decomposition (a factorization reduces over every row; splitting the rows gives a different matrix, not a batch of the same call, and there is no per-row output) | yes |
| linalg-svdvals | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-19_linalg-public/cpu-sabotage.json` | n/a n/a:whole-matrix-decomposition (a factorization reduces over every row; splitting the rows gives a different matrix, not a batch of the same call, and there is no per-row output) | yes |
| linear-svc | nvidia | training | declared | - | part | NO |
| linear-svc-squared-hinge | nvidia | training | declared | - | part | NO |
| linear-svr | nvidia | training | declared | - | part | NO |
| linear-svr-squared | nvidia | training | declared | - | part | NO |
| logistic | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-16_sabotage-audit/cpu-sab.json` | part | yes |
| logistic-elasticnet | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-linear-kernel/cpu-apple-m4-sabotage.json` | part | yes |
| logistic-l1 | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-linear-kernel/cpu-apple-m4-sabotage.json` | part | yes |
| logistic-multiclass | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-16_sabotage-audit/cpu-sab.json` | part | yes |
| logistic-unpenalized-no-intercept | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-linear-kernel/cpu-apple-m4-sabotage.json` | part | yes |
| lowbit-conversions | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-19_laneless-public-classes/sabotage-linalg-lowbit-convert.json` | part | yes |
| mamba1 | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-mamba/cpu-apple-m4.sabotage.json` | part | yes |
| mamba1-bf16w | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-complete-dependencies/records/mamba1-bf16w/cpu-sabotage.json` | part | yes |
| mamba1-decode-session | nvidia | training | seen(build) | `bench/results/identity_break/2026-09-20_cpu-routes-gpu-only-four/sabotage-mamba.json` | n/a n/a:fixed-session-batch (a decode session is opened for state.batch_size and its step refuses any other row count by name, so a row alone is a different session and not a smaller batch of this call; the per-call step the session is held to carries the batch part on the mamba1 and transformer lanes) | yes |
| mamba1-int8w | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-complete-dependencies/records/mamba1-int8w/cpu-sabotage.json` | part | yes |
| mamba2 | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-mamba/cpu-apple-m4.sabotage.json` | part | yes |
| mamba2-bf16w | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-complete-dependencies/records/mamba2-bf16w/cpu-sabotage.json` | part | yes |
| mamba2-dtlimit | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-mamba/cpu-apple-m4.sabotage.json` | part | yes |
| mamba2-int8w | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-complete-dependencies/records/mamba2-int8w/cpu-sabotage.json` | part | yes |
| mamba3 | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-mamba/cpu-apple-m4.sabotage.json` | part | yes |
| mamba3-bf16w | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-complete-dependencies/records/mamba3-bf16w/cpu-sabotage.json` | part | yes |
| mamba3-int8w | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-complete-dependencies/records/mamba3-int8w/cpu-sabotage.json` | part | yes |
| metrics | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_metrics-sabotage-coverage/cpu-sab-new.json` | n/a n/a:scalar-reduction (accuracy_score, adjusted_rand_score, v_measure_score, r2_score and silhouette_score each return one float over every row, python/mojolearn/_metrics_impl.py; the per-sample silhouette_samples is asked on metrics-classification) | yes |
| metrics-classification | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_metrics-sabotage-coverage/cpu-sab-new.json` | part | yes |
| metrics-fowlkes-mallows | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_metrics-sabotage-coverage/cpu-sab-new.json` | n/a n/a:function | yes |
| metrics-homogeneity-completeness | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-classical-completion/metrics-homogeneity-completeness/cpu-sabotage.json` | n/a n/a:global-contingency-reduction (three scalar scores over all labels; no per-row output) | yes |
| minmax-scaler | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-linear-kernel/cpu-apple-m4-sabotage.json` | part | yes |
| minmax-scaler-clip | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-linear-kernel/cpu-apple-m4-sabotage.json` | part | yes |
| mlp | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_neural-inference/cpu.neural-sabotage.json` | part | yes |
| mlp-bf16w | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-complete-dependencies/records/mlp-bf16w/cpu-sabotage.json` | n/a n/a:weight-format lane; the batch part is measured on its base lane | yes |
| mlp-int8w | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-complete-dependencies/records/mlp-int8w/cpu-sabotage.json` | n/a n/a:weight-format lane; the batch part is measured on its base lane | yes |
| monte-carlo | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-16_sabotage-audit/r2-sab.json` | n/a n/a:scalar-fold (resample.monte_carlo_integrate returns only integral, mean, volume and closed_form folded over [i_first, i_first + n_samples) by the pinned chunk tree, python/mojolearn/resample.py:196; no per-sample output exists to compare an i_first range against) | yes |
| nystroem | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-d-estimators/cpu-apple-m4.sabotage.json` | part | yes |
| nystroem-laplacian | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-18-cpu-kernel-variants/sabotage-nystroem-laplacian.json` | part | yes |
| nystroem-poly | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-18-cpu-kernel-variants/sabotage-nystroem-poly.json` | part | yes |
| nystroem-sigmoid | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-18-cpu-kernel-variants/sabotage-nystroem-sigmoid.json` | part | yes |
| ols | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-16_sabotage-audit/cpu-sab.json` | part | yes |
| ols-no-intercept | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-linear-kernel/cpu-apple-m4-sabotage.json` | part | yes |
| ols-weighted | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-linear-kernel/cpu-apple-m4-sabotage.json` | part | yes |
| optim-adam-clip | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-native-nine/records/optim-adam-clip/optim-adam-clip/cpu-sabotage.json` | part | yes |
| optim-sgd | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-native-nine/records/optim-sgd/optim-sgd/cpu-sabotage.json` | part | yes |
| ordered-gradient-sum | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-ordered-sum-completion/cpu-sabotage.json` | n/a n/a:ordered-shard-reduction (the specified shard order defines the sum; no per-row prediction) | yes |
| par-arima | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_sabotage-sweep/c-classical/cpu-x86.sabotage.json` | part | yes |
| par-boosting | amd,apple,nvidia | - | none | - | part | NO |
| par-boosting-clf | amd,nvidia | - | none | - | part | NO |
| par-boosting-pointwise | amd,apple,nvidia | - | none | - | part | NO |
| par-boosting-reg | amd,nvidia | - | none | - | part | NO |
| par-border-types | apple,nvidia | - | none | - | part | NO |
| par-byte-lm | amd,apple,nvidia | - | none | - | n/a n/a:driver-step (ParallelByteLanguageModelTrainer and its Pooled and Offloaded subclasses have train_step, state_dict, export_gradients and checkpoint only, python/mojolearn/parallel_training.py:89-172; train_step returns per-shard mean losses and one update from the shard-mean gradient, so no output belongs to one sequence, and the shard split is held to the replica trainer by the train column) | NO |
| par-byte-lm-model-pool | amd,apple,nvidia | - | none | - | n/a n/a:driver-step (ParallelByteLanguageModelTrainer and its Pooled and Offloaded subclasses have train_step, state_dict, export_gradients and checkpoint only, python/mojolearn/parallel_training.py:89-172; train_step returns per-shard mean losses and one update from the shard-mean gradient, so no output belongs to one sequence, and the shard split is held to the replica trainer by the train column) | NO |
| par-byte-lm-offload | amd,apple,nvidia | - | none | - | n/a n/a:driver-step (ParallelByteLanguageModelTrainer and its Pooled and Offloaded subclasses have train_step, state_dict, export_gradients and checkpoint only, python/mojolearn/parallel_training.py:89-172; train_step returns per-shard mean losses and one update from the shard-mean gradient, so no output belongs to one sequence, and the shard split is held to the replica trainer by the train column) | NO |
| par-causal-lm | nvidia | training | seen(build) | `bench/results/identity_break/2026-09-20_cpu-routes-gpu-only-four/sabotage-neural.json` | part | yes |
| par-cd | amd,apple,nvidia | - | none | - | part | NO |
| par-cd-elasticnet | amd,nvidia | - | none | - | part | NO |
| par-cholesky | amd,apple,nvidia | - | none | - | part | NO |
| par-cross-val | nvidia | training | seen(build) | `bench/results/identity_break/2026-09-20_cpu-routes-gpu-only-four/sabotage-gbdt.json` | n/a n/a:fold-reduction (one score per fold over that fold's whole test block; a call with fewer rows is a different cross-validation, not the same one in a smaller batch) | yes |
| par-dbscan | amd,apple,nvidia | - | seen(harness) | `bench/results/identity_break/2026-09-15_transductive-predict/apple-m4.batch-sabotage.json` | part | NO |
| par-feature-freq | amd,apple,nvidia | - | none | - | part | NO |
| par-forecast-arima | nvidia | training | seen(build) | `bench/results/identity_break/2026-09-20_broken-par-sabotage-arms/cpu-apple-m4.par-sweep.sabotage-core-only.before-fix.json` | part | yes |
| par-forecast-holtwinters | nvidia | training | seen(build) | `bench/results/identity_break/2026-09-20_broken-par-sabotage-arms/cpu-apple-m4.par-sweep.sabotage-core-only.before-fix.json` | part | yes |
| par-forest | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_sabotage-sweep/b-trees-gbdt/cpu-x86.sabotage.json` | part | yes |
| par-forest-et | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_sabotage-sweep/b-trees-gbdt/cpu-x86.sabotage.json` | part | yes |
| par-forest-et-clf | amd,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-20_broken-par-sabotage-arms/cpu-apple-m4.par-sweep.sabotage-core-only.before-fix.json` | part | yes |
| par-forest-pool | amd,nvidia | - | none | - | part | NO |
| par-forest-reg | amd,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-20_par-sabotage-arms/a-trees-gbdt/remote/leg_out/cpu-x86.sabotage.json` | part | yes |
| par-gmm | amd,apple,nvidia | - | none | - | part | NO |
| par-gp | amd,apple,nvidia | - | none | - | part | NO |
| par-gpc-fit | nvidia | training | seen(build) | `bench/results/identity_break/2026-09-20_broken-par-sabotage-arms/cpu-apple-m4.par-sweep.sabotage-core-only.before-fix.json` | part | yes |
| par-gpc-predict | nvidia | training | seen(build) | `bench/results/identity_break/2026-09-20_broken-par-sabotage-arms/cpu-apple-m4.par-sweep.sabotage-core-only.before-fix.json` | part | yes |
| par-gram | amd,apple,nvidia | - | none | - | part | NO |
| par-gram-ols | amd,nvidia | - | none | - | part | NO |
| par-gram-pca | amd,nvidia | - | none | - | part | NO |
| par-gram-tsvd | amd,nvidia | - | none | - | part | NO |
| par-graph-agglomerative | amd,apple,nvidia | - | seen(harness) | `bench/results/identity_break/2026-09-15_transductive-predict/apple-m4.batch-sabotage.json` | part | NO |
| par-graph-spectral | amd,apple,nvidia | - | none | - | n/a n/a:transductive (the par-graph-spectral lane fits without prediction_data; SpectralClustering.predict is the spectral lane's) | NO |
| par-graph-umap | amd,apple,nvidia | - | none | - | part | NO |
| par-hdbscan | amd,apple,nvidia | - | none | - | part | NO |
| par-holtwinters | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_holtwinters-linesearch-fix/cpu-amd-epyc-9754.sabotage.json` | part | yes |
| par-iforest | amd,apple,nvidia | - | none | - | part | NO |
| par-ivf | nvidia | training | seen(build) | `bench/results/identity_break/2026-09-20_broken-par-sabotage-arms/cpu-apple-m4.par-sweep.sabotage-core-only.before-fix.json` | part | yes |
| par-kernel-ridge | amd,apple,nvidia | - | none | - | part | NO |
| par-kmeans | amd,apple,nvidia | - | seen(harness) | `bench/results/identity_break/2026-09-15_kmeans-predict/apple-m4.batch-sabotage.json` | part | NO |
| par-logistic | amd,apple,nvidia | - | none | - | part | NO |
| par-mlp | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_sabotage-sweep/d-neural/cpu-x86.sabotage.json` | part | yes |
| par-nystroem | amd,apple,nvidia | - | none | - | part | NO |
| par-ordered | apple,nvidia | - | none | - | part | NO |
| par-ordered-rmse | amd,apple,nvidia | - | none | - | part | NO |
| par-queries-kde | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_sabotage-sweep/a-linear-neighbors/cpu-x86.sabotage.json` | part | yes |
| par-queries-knn | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_sabotage-sweep/a-linear-neighbors/cpu-x86.sabotage.json` | part | yes |
| par-queries-nn | amd,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-20_broken-par-sabotage-arms/cpu-apple-m4.par-sweep.sabotage-core-only.before-fix.json` | part | yes |
| par-queries-radius | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_sabotage-sweep/a-linear-neighbors/cpu-x86.sabotage.json` | part | yes |
| par-rbf-sampler | amd,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-20_broken-par-sabotage-arms/cpu-apple-m4.par-sweep.sabotage-core-only.before-fix.json` | part | yes |
| par-reference-knn | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_sabotage-sweep/a-linear-neighbors/cpu-x86.sabotage.json` | part | yes |
| par-reference-knn-reg | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_sabotage-sweep/a-linear-neighbors/cpu-x86.sabotage.json` | part | yes |
| par-resample | amd,apple,nvidia | - | none | - | part | NO |
| par-samba | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-16_cpu-par-samba/cpu-apple-m4.sabotage.json` | part | yes |
| par-samba-clip | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-16_cpu-par-samba/cpu-apple-m4.sabotage.json` | part | yes |
| par-scaler | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_sabotage-sweep/i-par-scaler-mismatch/cpu-x86.sabotage.json` | part | yes |
| par-scaler-minmax | amd,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_sabotage-sweep/i-par-scaler-mismatch/cpu-x86.sabotage.json` | part | yes |
| par-svm | amd,apple,nvidia | - | none | - | part | NO |
| par-svm-svr | amd,nvidia | - | none | - | part | NO |
| pca | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-16_sabotage-audit/cpu-sab.json` | part | yes |
| pca-full-whiten | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-forecast-umap-pca/cpu-apple-m4.sabotage.json` | part | yes |
| pca-whiten | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-16_sabotage-audit/cpu-sab.json` | part | yes |
| permutation-test | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-16_sabotage-audit/r2-sab.json` | part | yes |
| qn-absolute | nvidia | training | declared | - | part | NO |
| qn-squared | nvidia | training | declared | - | part | NO |
| radius | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-neighbors-density/cpu-apple-m4.sabotage.json` | part | yes |
| radius-chebyshev | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-neighbors-density/cpu-apple-m4.sabotage.json` | part | yes |
| radius-manhattan | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-neighbors-density/cpu-apple-m4.sabotage.json` | part | yes |
| radius-minkowski-p3 | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-neighbors-density/cpu-apple-m4.sabotage.json` | part | yes |
| rbf-sampler | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-d-estimators/cpu-apple-m4.sabotage.json` | part | yes |
| rf-clf | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-native-nine/records/rf-clf/rf-clf/cpu-sabotage.json` | part | yes |
| rf-clf-balanced-parallel | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-native-nine/records/rf-clf-balanced-parallel/rf-clf-balanced-parallel/cpu-sabotage.json` | part | yes |
| rf-clf-entropy-log2-noboot | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-native-nine/records/rf-clf-entropy-log2-noboot/rf-clf-entropy-log2-noboot/cpu-sabotage.json` | part | yes |
| rf-reg | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-native-nine/records/rf-reg/rf-reg/cpu-sabotage.json` | part | yes |
| rf-reg-gamma-ig | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-native-nine/records/rf-reg-gamma-ig/rf-reg-gamma-ig/cpu-sabotage.json` | part | yes |
| rf-reg-poisson | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-native-nine/records/rf-reg-poisson/rf-reg-poisson/cpu-sabotage.json` | part | yes |
| rf-score-weighted | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-classical-completion/rf-score-weighted/cpu-sabotage.json` | n/a n/a:scalar-reduction (score(X, y, sample_weight) returns one float over every row it is handed, python/mojolearn/_gbdt_adapters.py and _forest_protocol.py; the predict and predict_proba it wraps keep their batch parts on gbdt-adapter-clf, gbdt-adapter-reg, rf-clf and rf-reg) | yes |
| ridge | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-16_sabotage-audit/cpu-sab.json` | part | yes |
| ridge-no-intercept | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-linear-kernel/cpu-apple-m4-sabotage.json` | part | yes |
| samba | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-samba/cpu-apple-m4.sabotage.json` | part | yes |
| samba-bf16w | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-complete-dependencies/records/samba-bf16w/cpu-sabotage.json` | n/a n/a:weight-format lane; the batch part is measured on its base lane | yes |
| samba-int8w | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-complete-dependencies/records/samba-int8w/cpu-sabotage.json` | n/a n/a:weight-format lane; the batch part is measured on its base lane | yes |
| samba-untied-dropout-accum | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-samba/cpu-apple-m4.sabotage.json` | part | yes |
| saved-model-host-infer | - | training | seen(build) | `bench/results/identity_break/2026-09-19_laneless-public-classes/sabotage-forest-own-define.json` | part | NO |
| select-d | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_tsa-negative-controls/cpu-sabotage.json` | part | yes |
| spectral | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_spectral-predict/cpu-x86.host-sabotage.json (clean partner at another commit)` | part | yes |
| spectral-embedding | nvidia | training | seen(build) | `bench/results/identity_break/2026-09-20_spectral-embedding/cpu-x86-sabotage.json` | n/a n/a:transductive (SpectralEmbedding embeds the fitted rows only; it has no transform, in the reference and in scikit-learn alike) | yes |
| spectral-precomputed | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_spectral-predict/cpu-x86.host-sabotage.json (clean partner at another commit)` | part | yes |
| standard-scaler | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-linear-kernel/cpu-apple-m4-sabotage.json` | part | yes |
| standard-scaler-no-mean | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-linear-kernel/cpu-apple-m4-sabotage.json` | part | yes |
| standard-scaler-no-std | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-linear-kernel/cpu-apple-m4-sabotage.json` | part | yes |
| svc | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-svm/cpu-x86-sabotage.json` | part | yes |
| svc-linear | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-svm/cpu-x86-sabotage.json` | part | yes |
| svc-poly | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-svm/cpu-x86-sabotage.json` | part | yes |
| svr | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-svm/cpu-x86-sabotage.json` | part | yes |
| svr-linear | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-svm/cpu-x86-sabotage.json` | part | yes |
| tokenized-corpus | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-18_bpe-builder-native/m4.trainer.sabotage-build.json` | n/a n/a:corpus-global-vocabulary-training (pair counts depend on the complete corpus; no per-row output) | yes |
| tokenizer | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_clean-third-party/m4-owed/cpu-apple-m4.host-sabotage.json` | part | yes |
| training-primitives | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-native-nine/records/training-primitives/training-primitives/cpu-sabotage.json` | part | yes |
| transformer | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-transformer/cpu-apple-m4.sabotage.json` | part | yes |
| transformer-bf16w | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-complete-dependencies/records/transformer-bf16w/cpu-sabotage.json` | part | yes |
| transformer-decode-session | nvidia | training | seen(build) | `bench/results/identity_break/2026-09-20_cpu-routes-gpu-only-four/sabotage-neural-transformer-session.json` | n/a n/a:fixed-session-batch (a decode session is opened for state.batch_size and its step refuses any other row count by name, so a row alone is a different session and not a smaller batch of this call; the per-call step the session is held to carries the batch part on the mamba1 and transformer lanes) | yes |
| transformer-int8w | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-complete-dependencies/records/transformer-int8w/cpu-sabotage.json` | part | yes |
| transformer-window | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-transformer/cpu-apple-m4.sabotage.json` | part | yes |
| tsvd | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-16_sabotage-audit/cpu-sab.json` | part | yes |
| umap | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-forecast-umap-pca/cpu-apple-m4.sabotage.json` | part | yes |

## Lanes missing each kind, by name

A `par-*` lane IS NOT A CPU GAP AND NEVER WILL BE. It is a multi-device
driver whose whole claim, written in `identity_break._par_devices`, is
that a TWO-DEVICE column hashes equal to the one-device column cell for
cell. A CPU column has zero devices, so that claim there is not false,
it is NOT EXPRESSIBLE -- and an inexpressible claim listed beside real
gaps is noise that hides them. On 2026-09-19 it hid them at a ratio of
36 to 3. These lanes need a SECOND GPU, not another CPU run, and no CPU
work will ever close them; they are counted below under their own
heading and excluded from the two CPU-axis gap lists by construction.

AND A `par-*` LANE IS NOT A VENDOR-CLASS GAP EITHER (2026-09-19). The
rules make that count UNREACHABLE for them, in both directions at once:

  * `_verify_reference.py:314` REFUSES any column recording
    `par_devices != "0"`, and `gpu_coverage` above counts ADMITTED
    columns only -- so a two-device column is invisible to this number
    BY CONSTRUCTION;
  * and on ONE device every one of them is DEGENERATE, measured:
    `lane_applicability.degenerate('apple-metal')` holds all 13 of them,
    because with one shard the equality they assert is not false, it is
    not expressible.

So the claim is only STATEABLE on two devices and only ADMISSIBLE on
one. No run, on any hardware, ever, can take a `par-*` lane to three
vendor classes under these rules, and listing them as short of it has
been advertising 13 gaps that cannot be closed. A $3.34 two-device
MI300X leg was bought on 2026-09-19 before this was noticed; what it
proved is real and is reported under the driver heading below, not here.

**No GPU column at all: 1**

> gbdt-stochastic-arms

> (plus 0 `par-*` multi-GPU driver lanes, held out of this count: this count is unreachable for them in both directions. They are listed once below.)

**GPU column on fewer than three classes: 10**

> gbdt-symmetric-eval, linear-svc, linear-svc-squared-hinge, linear-svr, linear-svr-squared, mamba1-decode-session, qn-absolute, qn-squared, spectral-embedding, transformer-decode-session

> (plus 22 `par-*` multi-GPU driver lanes, held out of this count: this count is unreachable for them in both directions. They are listed once below.)

**No CPU verifier declared: 0**

> none

> (plus 34 `par-*` multi-GPU driver lanes, held out of this count: a CPU column cannot state their claim. They are listed once below.)

**Sabotage not seen to move a build: 7**

> gbdt-stochastic-arms, linear-svc, linear-svc-squared-hinge, linear-svr, linear-svr-squared, qn-absolute, qn-squared

> (plus 34 `par-*` multi-GPU driver lanes, held out of this count: a CPU column cannot state their claim. They are listed once below.)

**Batch undeclared: 0**

> none

## Lanes a GPU column cannot judge at all

6 lanes are DEGENERATE on every GPU column. Their arithmetic is
the CPU host route, or they stand on no Mojo binding at all, so handed
a GPU column they run that box's CPU and say nothing whatever about the
GPU. An NVIDIA and an AMD pod each printed that refusal verbatim on
2026-09-19; `lane_applicability` names three more. They are held out of
the two GPU-axis counts above because listing them there advertised six
gaps no run on any hardware can close -- the same unreachable count
already fixed for `par-*`. THEY ARE NOT UNVERIFIED: each is checked on
the cpu-host column, which is the one column its proposition is
stateable on.

### "Then why not just add the GPU column?"

Because the ALGORITHM already has one, in a different lane, and these
lanes are the other half of the same measurement.

mojolearn keeps TWO SPELLINGS of these algorithms on purpose. There are
the device kernels, and there is an independent host restatement of
them -- `gbdt/host/gbdt_oracle.mojo` says it in its own first line: "a
SECOND spelling of the device trainer", where "every device KERNEL the
fit reaches is RESTATED below, with the file and line it MIRRORS". Only
code that is host code ON THE DEVICE PATH TOO is reused; every kernel is
written again.

So the work divides: the device lane verifies the device spelling, one
of these lanes verifies the host spelling, and THE CROSS-VENDOR CLAIM IS
THAT THE TWO AGREE BITWISE. `saved-model-host-infer` is the host half of
what `rf-clf`, `rf-reg`, `et-clf` and `et-reg` cover on three vendors;
`language-model-config` is the host half of `byte-lm`, `transformer`,
`samba` and `mamba3`; `cross-val-folds` is the fold partition underneath
`cross-val`. None of those algorithms is missing a GPU column.

CONSOLIDATING THE TWO SPELLINGS WOULD DESTROY THE ORACLE. If the host
path called the device code, "the CPU agrees with the GPU" would be one
function agreeing with itself -- a check that cannot fail, and the thing
this whole document exists to prevent. The duplication IS the
measurement. It is also why nothing statically proves the two spellings
match: you cannot prove it, you compare their bits, and the lane that
compares them is the drift guard. A device kernel that changes without
its host restatement changing makes that lane's columns disagree.

Renting a GPU for a lane in this table buys a CPU run at GPU prices.

**RECORDED IS NOT THE SAME AS MEANINGFUL.** 3 of these
carry GPU columns anyway, recorded before this classification existed.
They are not extra assurance: each is a GPU box that ran its own CPU.
They are marked below so a reader does not count them as vendor
coverage, and they are not evidence for any vendor claim.

| lane | cpu | sabotage | GPU columns recorded |
|---|---|---|---|
| byte-lm-host-infer | training | seen(build) | amd,apple,nvidia (assert nothing) |
| byte-lm-host-infer-threaded | training | seen(build) | amd,apple,nvidia (assert nothing) |
| byte-lm-host-train | training | seen(build) | amd,apple,nvidia (assert nothing) |
| cross-val-folds | host-only | seen(build) | none, correctly |
| language-model-config | training | seen(build) | none, correctly |
| saved-model-host-infer | training | seen(build) | none, correctly |

## The multi-GPU driver lanes, which a CPU column cannot judge

59 `par-*` lanes exist. THEIR CLAIM IS ONLY STATEABLE ON TWO DEVICES -- that a two-device column hashes equal to the one-device column cell for cell -- so a one-device run of one is DEGENERATE: it compares a run against itself and passes whatever the code does. They are held out of the vendor-class counts above for that reason.

**59 of 59 now carry a TWO-DEVICE column**, read through `admit(..., par_axis=True)`. Until 2026-09-19 the default rule refused `par_devices != "0"`, so the only run that can state their claim was inadmissible and this evidence counted for nothing.

| par-arima | amd,nvidia | bench/results/identity_break/2026-09-14_136-lanes/amd-2xmi300x-gfx942.par-devices-0-1.json |
| par-boosting | amd,nvidia | bench/results/identity_break/2026-09-14_136-lanes/amd-2xmi300x-gfx942.par-devices-0-1.json |
| par-boosting-clf | amd,nvidia | bench/results/identity_break/2026-09-19_hardware-gaps/amd-par-boosting-clf-two.json |
| par-boosting-pointwise | amd,nvidia | bench/results/identity_break/2026-09-14_166-lanes/amd-2xmi300x-gfx942.par-devices-0-1.json |
| par-boosting-reg | amd,nvidia | bench/results/identity_break/2026-09-19_hardware-gaps/amd-par-boosting-reg-two.json |
| par-border-types | nvidia | bench/results/identity_break/2026-09-19_par-two-device-gaps/ordered-and-border-types/nvidia-nvidia-geforce-rtx-4090-sm_89.par-two-device-gaps.two-device.json |
| par-byte-lm | amd,nvidia | bench/results/identity_break/2026-09-14_136-lanes/amd-2xmi300x-gfx942.par-devices-0-1.json |
| par-byte-lm-model-pool | amd,nvidia | bench/results/identity_break/2026-09-14_166-lanes/amd-2xmi300x-gfx942.par-devices-0-1.json |
| par-byte-lm-offload | amd,nvidia | bench/results/identity_break/2026-09-14_166-lanes/amd-2xmi300x-gfx942.par-devices-0-1.json |
| par-causal-lm | nvidia | bench/results/identity_break/2026-09-20_takeover-last-gpu-columns/nvidia-nvidia-geforce-rtx-4090-sm_89.par-two-device-last-four-2026-09-20.two-device.json |
| par-cd | amd,nvidia | bench/results/identity_break/2026-09-14_136-lanes/amd-2xmi300x-gfx942.par-devices-0-1.json |
| par-cd-elasticnet | amd,nvidia | bench/results/identity_break/2026-09-19_hardware-gaps/amd-par-cd-elasticnet-two.json |
| par-cholesky | amd,nvidia | bench/results/identity_break/2026-09-15_par-lanes-new/amd-2xmi300x-new8/two.json |
| par-cross-val | nvidia | bench/results/identity_break/2026-09-20_takeover-last-gpu-columns/nvidia-nvidia-geforce-rtx-4090-sm_89.par-two-device-last-four-2026-09-20.two-device.json |
| par-dbscan | amd,nvidia | bench/results/identity_break/2026-09-14_136-lanes/amd-2xmi300x-gfx942.par-devices-0-1.json |
| par-feature-freq | amd,nvidia | bench/results/identity_break/2026-09-14_166-lanes/amd-2xmi300x-gfx942.par-devices-0-1.json |
| par-forecast-arima | nvidia | bench/results/identity_break/2026-09-19_par-two-device-final3b/nvidia-nvidia-geforce-rtx-4090-sm_89.par-two-device-final3b.two-device.json |
| par-forecast-holtwinters | nvidia | bench/results/identity_break/2026-09-19_par-two-device-final3b/nvidia-nvidia-geforce-rtx-4090-sm_89.par-two-device-final3b.two-device.json |
| par-forest | amd,nvidia | bench/results/identity_break/2026-09-14_136-lanes/amd-2xmi300x-gfx942.par-devices-0-1.json |
| par-forest-et | amd,nvidia | bench/results/identity_break/2026-09-14_136-lanes/amd-2xmi300x-gfx942.par-devices-0-1.json |
| par-forest-et-clf | amd,nvidia | bench/results/identity_break/2026-09-19_hardware-gaps/amd-par-forest-et-clf-two.json |
| par-forest-pool | amd,nvidia | bench/results/identity_break/2026-09-15_par-lanes-new/amd-2xmi300x-old4/two.json |
| par-forest-reg | amd,nvidia | bench/results/identity_break/2026-09-19_hardware-gaps/amd-par-forest-reg-two.json |
| par-gmm | amd,nvidia | bench/results/identity_break/2026-09-15_par-lanes-new/amd-2xmi300x-old4/two.json |
| par-gp | amd,nvidia | bench/results/identity_break/2026-09-14_136-lanes/amd-2xmi300x-gfx942.par-devices-0-1.json |
| par-gpc-fit | nvidia | bench/results/identity_break/2026-09-20_takeover-last-gpu-columns/nvidia-nvidia-geforce-rtx-4090-sm_89.par-two-device-last-four-2026-09-20.two-device.json |
| par-gpc-predict | nvidia | bench/results/identity_break/2026-09-20_takeover-last-gpu-columns/nvidia-nvidia-geforce-rtx-4090-sm_89.par-two-device-last-four-2026-09-20.two-device.json |
| par-gram | amd,nvidia | bench/results/identity_break/2026-09-14_136-lanes/amd-2xmi300x-gfx942.par-devices-0-1.json |
| par-gram-ols | amd,nvidia | bench/results/identity_break/2026-09-19_hardware-gaps/amd-par-gram-ols-two.json |
| par-gram-pca | amd,nvidia | bench/results/identity_break/2026-09-19_hardware-gaps/amd-par-gram-pca-two.json |
| par-gram-tsvd | amd,nvidia | bench/results/identity_break/2026-09-19_hardware-gaps/amd-par-gram-tsvd-two.json |
| par-graph-agglomerative | amd,nvidia | bench/results/identity_break/2026-09-14_166-lanes/amd-2xmi300x-gfx942.par-devices-0-1.json |
| par-graph-spectral | amd,nvidia | bench/results/identity_break/2026-09-14_166-lanes/amd-2xmi300x-gfx942.par-devices-0-1.json |
| par-graph-umap | amd,nvidia | bench/results/identity_break/2026-09-14_166-lanes/amd-2xmi300x-gfx942.par-devices-0-1.json |
| par-hdbscan | amd,nvidia | bench/results/identity_break/2026-09-15_par-lanes-new/amd-2xmi300x-old4/two.json |
| par-holtwinters | amd,nvidia | bench/results/identity_break/2026-09-14_166-lanes/amd-2xmi300x-gfx942.par-devices-0-1.json |
| par-iforest | amd,nvidia | bench/results/identity_break/2026-09-14_136-lanes/amd-2xmi300x-gfx942.par-devices-0-1.json |
| par-ivf | nvidia | bench/results/identity_break/2026-09-19_par-two-device-final3b/nvidia-nvidia-geforce-rtx-4090-sm_89.par-two-device-final3b.two-device.json |
| par-kernel-ridge | amd,nvidia | bench/results/identity_break/2026-09-15_par-lanes-new/amd-2xmi300x-new8/two.json |
| par-kmeans | amd,nvidia | bench/results/identity_break/2026-09-14_136-lanes/amd-2xmi300x-gfx942.par-devices-0-1.json |
| par-logistic | amd,nvidia | bench/results/identity_break/2026-09-14_136-lanes/amd-2xmi300x-gfx942.par-devices-0-1.json |
| par-mlp | amd,nvidia | bench/results/identity_break/2026-09-14_136-lanes/amd-2xmi300x-gfx942.par-devices-0-1.json |
| par-nystroem | amd,nvidia | bench/results/identity_break/2026-09-15_par-lanes-new/amd-2xmi300x-new8/two.json |
| par-ordered | nvidia | bench/results/identity_break/2026-09-19_par-two-device-gaps/ordered-and-border-types/nvidia-nvidia-geforce-rtx-4090-sm_89.par-two-device-gaps.two-device.json |
| par-ordered-rmse | amd,nvidia | bench/results/identity_break/2026-09-14_166-lanes/amd-2xmi300x-gfx942.par-devices-0-1.json |
| par-queries-kde | amd,nvidia | bench/results/identity_break/2026-09-14_166-lanes/amd-2xmi300x-gfx942.par-devices-0-1.json |
| par-queries-knn | amd,nvidia | bench/results/identity_break/2026-09-14_166-lanes/amd-2xmi300x-gfx942.par-devices-0-1.json |
| par-queries-nn | amd,nvidia | bench/results/identity_break/2026-09-19_hardware-gaps/amd-par-queries-nn-two.json |
| par-queries-radius | amd,nvidia | bench/results/identity_break/2026-09-14_166-lanes/amd-2xmi300x-gfx942.par-devices-0-1.json |
| par-rbf-sampler | amd,nvidia | bench/results/identity_break/2026-09-15_par-lanes-new/amd-2xmi300x-new8/two.json |
| par-reference-knn | amd,nvidia | bench/results/identity_break/2026-09-14_166-lanes/amd-2xmi300x-gfx942.par-devices-0-1.json |
| par-reference-knn-reg | amd,nvidia | bench/results/identity_break/2026-09-14_166-lanes/amd-2xmi300x-gfx942.par-devices-0-1.json |
| par-resample | amd,nvidia | bench/results/identity_break/2026-09-15_par-lanes-new/amd-2xmi300x-old4/two.json |
| par-samba | amd,nvidia | bench/results/identity_break/2026-09-14_136-lanes/amd-2xmi300x-gfx942.par-devices-0-1.json |
| par-samba-clip | amd,nvidia | bench/results/identity_break/2026-09-14_166-lanes/amd-2xmi300x-gfx942.par-devices-0-1.json |
| par-scaler | amd,nvidia | bench/results/identity_break/2026-09-14_136-lanes/amd-2xmi300x-gfx942.par-devices-0-1.json |
| par-scaler-minmax | amd,nvidia | bench/results/identity_break/2026-09-19_hardware-gaps/amd-par-scaler-minmax-two.json |
| par-svm | amd,nvidia | bench/results/identity_break/2026-09-14_136-lanes/amd-2xmi300x-gfx942.par-devices-0-1.json |
| par-svm-svr | amd,nvidia | bench/results/identity_break/2026-09-19_hardware-gaps/amd-par-svm-svr-two.json |

> par-boosting, par-boosting-clf, par-boosting-pointwise, par-boosting-reg, par-border-types, par-byte-lm, par-byte-lm-model-pool, par-byte-lm-offload, par-cd, par-cd-elasticnet, par-cholesky, par-dbscan, par-feature-freq, par-forest-pool, par-gmm, par-gp, par-gram, par-gram-ols, par-gram-pca, par-gram-tsvd, par-graph-agglomerative, par-graph-spectral, par-graph-umap, par-hdbscan, par-iforest, par-kernel-ridge, par-kmeans, par-logistic, par-nystroem, par-ordered, par-ordered-rmse, par-resample, par-svm, par-svm-svr

## Public names not counted as algorithms

Listed by name in `NOT_ALGORITHMS` in the tool, so the exclusion can be
argued with rather than hidden in a heuristic. These are process metadata,
tier switches, result containers, option lists and the caller-owned state
containers a block returns from `allocate_state`, whose buffers are hashed
through their block's own lanes.

> `Array`, `Mamba1State`, `Mamba2State`, `Mamba3State`, `TransformerState`, `__version__`, `gpu_arch`, `gpu_arch_how`, `linalg.PROFILE`, `linalg.PROFILE_BF16`, `linalg.PROFILE_FAMILY`, `linalg.PROFILE_INT8`, `linalg.PROFILE_VERSION`, `linalg.numeric_mode`, `linalg.profile`, `linalg.require_identical`, `lowbit.BF16Weight`, `lowbit.FORMATS`, `lowbit.Int8Weight`, `mamba.Mamba1State`, `mamba.Mamba2State`, `mamba.Mamba3State`, `models.CausalLMState`, `models.FAMILIES`, `models.HFConfig`, `models.ModelPlan`, `models.PATTERNS`, `models.UnsupportedModel`, `models.causal_lm.CausalLMState`, `models.config.FAMILIES`, `models.config.FIXED_TODAY`, `models.config.HFConfig`, `models.config.INTERFACE_DEFAULTS`, `models.config.ModelPlan`, `models.config.POSITION_CEILING`, `models.config.UnsupportedModel`, `models.safetensors.DTYPES`, `models.safetensors.INDEX_NAME`, `models.safetensors.SINGLE_NAME`, `models.safetensors.TensorInfo`, `models.tokenizer.PATTERNS`, `numeric_mode`, `resample.ALTERNATIVES`, `resample.BootstrapResult`, `resample.INTEGRANDS`, `resample.METHODS`, `resample.MonteCarloResult`, `resample.PermutationTestResult`, `resample.STATISTICS`, `set_numeric_mode`, `training.numeric_mode_used`, `training.vendor_used`, `transformer.TransformerState`, `vendor`

