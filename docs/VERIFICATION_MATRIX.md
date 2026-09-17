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

- Lanes: **229** (179 single-device, 50 `par-*` multi-GPU drivers).
- Source public API entries enumerated from the public API: **224**.
- Source public API entries with ALL FOUR kinds on at least one lane: **63** of 224.
- Source public API entries with NO IDENTITY LANE AT ALL: **19**.
- Source public API entries with no lane of their own, but reached by the harness's
  CPU inference routing: **2**.

Per kind, over the public API entries:

| kind | API entries that have it | missing |
|---|---|---|
| gpu column | 197 | 27 |
| cpu verifier | 167 | 57 |
| sabotage seen to move a build | 75 | 149 |
| batch part or named n/a | 203 | 21 |

Per kind, over the lanes:

| kind | lanes that have it | missing |
|---|---|---|
| gpu column (any class) | 208 | 21 |
| gpu column on all three classes | 171 | 58 |
| cpu verifier declared | 190 | 39 |
| sabotage seen to move a build | 63 | 166 |
| batch part or named n/a | 229 | 0 |
| ALL FOUR | 55 | 174 |

Sabotage, split by what was actually watched:

| verdict | lanes | what it means |
|---|---|---|
| seen(build) | 63 | a sabotage BUILD moved the bytes; a real negative control |
| seen(harness) | 0 | only the harness batch switch moved; the probe can fail, the build is unproven |
| declared | 127 | the family declares a define; no committed pair moves this lane |
| none | 39 | no define reaches the lane and nothing has moved it |

## Source public API entries with no identity lane at all

These are the most important gaps. An algorithm with no lane cannot
be missing a cell, so a lane census hides it entirely. `host family`
names the CPU host family that serves the class, where one does, which
means a CPU path exists and only the identity lane is missing.

| algorithm | kind | host family | defined in |
|---|---|---|---|
| `HostForest` | class | - | `python/mojolearn/_forest_host.py` |
| `HostGBDT` | class | - | `python/mojolearn/_gbdt_host.py` |
| `host_predict` | function | - | `python/mojolearn/_forest_host.py` |
| `host_predict_proba` | function | - | `python/mojolearn/_forest_host.py` |
| `linalg.PROFILE_BF16` | name | - | `python/mojolearn/linalg.py` |
| `linalg.PROFILE_INT8` | name | - | `python/mojolearn/linalg.py` |
| `linalg.from_bf16` | function | - | `python/mojolearn/_linalg_impl.py` |
| `lowbit.BF16Weight` | class | - | `python/mojolearn/lowbit.py` |
| `lowbit.FORMATS` | name | - | `python/mojolearn/lowbit.py` |
| `lowbit.Int8Weight` | class | - | `python/mojolearn/lowbit.py` |
| `lowbit.format_of` | function | - | `python/mojolearn/lowbit.py` |
| `lowbit.is_packed` | function | - | `python/mojolearn/lowbit.py` |
| `lowbit.materialize_one` | function | - | `python/mojolearn/lowbit.py` |
| `lowbit.pack_one` | function | - | `python/mojolearn/lowbit.py` |
| `lowbit.widen_bf16` | function | - | `python/mojolearn/lowbit.py` |
| `mamba.Mamba1DecodeSession` | class | - | `python/mojolearn/_mamba_impl.py` |
| `models` | name | - | `python/mojolearn/__init__.py` |
| `tokenizer.TrainedBpeVocabulary` | class | - | `python/mojolearn/tokenizer.py` |
| `transformer.TransformerDecodeSession` | class | - | `python/mojolearn/_transformer_impl.py` |

The saved-model host inference surface (`HostForest`, `HostGBDT`,
`host_model`, `host_predict`, `host_predict_proba`) has no identity_break
lane on purpose. It is measured by `tools/forest_host_gate.py` and
`tools/classical_host_gate.py` against committed recordings instead:
25 under `bench/results/forest_host/` and
15 classical recording directories named in
`host_surface.py`. That is a different gate, not a missing one, but it is
also not one of the four kinds counted here.

### Reached by the harness, with no lane of their own

`tools/identity_break.py` reaches these outside any lane body. `_public_est`
swaps the fitted estimator for the public CPU inference class on a CPU
column, so those answer the infer, reload and batch cells of the lanes named
in `NEURAL_PUBLIC_PART_LANES` (mamba1, mamba2, mamba3, mamba2-dtlimit, samba, samba-untied-dropout-accum, byte-lm, byte-lm-resident, mamba1-bf16w, mamba1-int8w, mamba2-bf16w, mamba2-int8w, mamba3-bf16w, mamba3-int8w, transformer, transformer-window);
the batchgrad part calls the accumulation helpers the same way.
They are verified, but no lane carries their name.

| algorithm | kind | host family | defined in |
|---|---|---|---|
| `training.accumulate_grads` | function | training | `python/mojolearn/_training_impl.py` |
| `training.accumulation_is_aligned` | function | - | `python/mojolearn/_training_impl.py` |

## The algorithm matrix

`gpu` is the device classes that carry any of the algorithm's lanes.
A blank cell means no lane of this algorithm has that kind.

| algorithm | lanes | gpu | cpu | sabotage | batch | all four |
|---|---|---|---|---|---|---|
| `ARIMA` | 6 | amd,apple,nvidia | training | seen(build) | part | NO |
| `Adam` | 1 | amd,apple,nvidia | training | declared | part | NO |
| `AdamW` | 1 | amd,apple,nvidia | training | declared | part | NO |
| `AgglomerativeClustering` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `ByteLanguageModelConfig` | 8 | amd,apple,nvidia | training | declared | part | NO |
| `Cholesky` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `ConstantKernel` | 5 | amd,apple,nvidia | training | seen(build) | part | yes |
| `DBSCAN` | 4 | amd,apple,nvidia | training | declared | part | NO |
| `ElasticNet` | 3 | amd,apple,nvidia | training | declared | part | NO |
| `Embedding` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `ExperimentalTwoLevelFeatureFreq` | 3 | amd,apple,nvidia | training | declared | part | NO |
| `ExponentialSmoothing` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `ExtraTreesClassifier` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `ExtraTreesRegressor` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `GPT2Tokenizer` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `GaussianMixture` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `GaussianProcessClassifier` | 2 | apple | training | declared | part | NO |
| `GaussianProcessRegressor` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `GradientBoosting` | 18 | amd,apple,nvidia | training | seen(build) | part | yes |
| `GradientBoostingClassifier` | 3 | amd,apple,nvidia | training | seen(build) | part | NO |
| `GradientBoostingRegressor` | 4 | amd,apple,nvidia | training | seen(build) | part | NO |
| `HDBSCAN` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `HostForest` | **0** |  |  |  |  | NO |
| `HostGBDT` | **0** |  |  |  |  | NO |
| `IVFIndex` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `IsolationForest` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `KMeans` | 10 | amd,apple,nvidia | training | seen(build) | part | yes |
| `KNeighborsClassifier` | 4 | amd,apple,nvidia | training | declared | part | NO |
| `KNeighborsRegressor` | 3 | amd,apple,nvidia | training | declared | part | NO |
| `KernelDensity` | 3 | amd,apple,nvidia | training | declared | part | NO |
| `KernelRidge` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `LanguageModelConfig` (alias of `ByteLanguageModelConfig`) | 8 | amd,apple,nvidia | training | declared | part | NO |
| `LanguageModelHostTrainer` | 1 | amd,apple,nvidia | training | declared | n/a | NO |
| `LanguageModelInference` | 17 | amd,apple,nvidia | training | seen(build) | part | yes |
| `LanguageModelTrainer` (alias of `SmallByteLanguageModelTrainer`) | 5 | amd,apple,nvidia | training | declared | part | NO |
| `Lasso` | 2 | amd,apple,nvidia | training | declared | part | NO |
| `LinearRegression` | 4 | amd,apple,nvidia | training | seen(build) | part | yes |
| `LogisticRegression` | 6 | amd,apple,nvidia | training | declared | part | NO |
| `MLPInference` | 15 | amd,apple,nvidia | training | seen(build) | part | yes |
| `Mamba1Block` | 9 | amd,apple,nvidia | training | seen(build) | part | yes |
| `Mamba1BlockInference` | 15 | amd,apple,nvidia | training | seen(build) | part | yes |
| `Mamba2Block` | 10 | amd,apple,nvidia | training | seen(build) | part | yes |
| `Mamba2BlockInference` | 15 | amd,apple,nvidia | training | seen(build) | part | yes |
| `Mamba3Block` | 9 | amd,apple,nvidia | training | seen(build) | part | yes |
| `Mamba3BlockInference` | 15 | amd,apple,nvidia | training | seen(build) | part | yes |
| `Matern` | 1 | apple | training | declared | part | NO |
| `MinMaxScaler` | 3 | amd,apple,nvidia | training | declared | part | NO |
| `NearestNeighbors` | 2 | amd,apple,nvidia | training | declared | part | NO |
| `Nystroem` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `OffloadedByteLanguageModelTrainer` | 1 | amd,apple,nvidia |  | none | n/a | NO |
| `OrderedRMSE` | 2 | amd,apple,nvidia | training | declared | part | NO |
| `PCA` | 4 | amd,apple,nvidia | training | declared | part | NO |
| `ParallelByteLanguageModelTrainer` | 3 | amd,apple,nvidia |  | none | n/a | NO |
| `ParallelNeuralTrainer` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `PooledByteLanguageModelTrainer` | 1 | amd,apple,nvidia |  | none | n/a | NO |
| `RBF` | 4 | amd,apple,nvidia | training | seen(build) | part | yes |
| `RBFSampler` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `RadiusNeighbors` | 2 | amd,apple,nvidia | training | declared | part | NO |
| `RandomForestClassifier` | 6 | amd,apple,nvidia | training | seen(build) | part | NO |
| `RandomForestRegressor` | 5 | amd,apple,nvidia | training | seen(build) | part | NO |
| `Ridge` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `SGD` | 1 | amd,apple,nvidia | training | declared | part | NO |
| `SVC` | 4 | amd,apple,nvidia | training | seen(build) | part | yes |
| `SVR` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `SambaConfig` | 6 | amd,apple,nvidia | training | seen(build) | part | yes |
| `SambaInference` | 15 | amd,apple,nvidia | training | seen(build) | part | yes |
| `SambaStack` | 6 | amd,apple,nvidia | training | seen(build) | part | yes |
| `SmallByteLanguageModelTrainer` | 5 | amd,apple,nvidia | training | declared | part | NO |
| `SmallMLPTrainer` | 4 | amd,apple,nvidia | training | declared | part | NO |
| `SpectralClustering` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `StandardScaler` | 4 | amd,apple,nvidia | training | declared | part | NO |
| `TransformerBlock` | 10 | amd,apple,nvidia | training | seen(build) | part | yes |
| `TransformerBlockInference` | 15 | amd,apple,nvidia | training | seen(build) | part | yes |
| `TruncatedSVD` | 2 | amd,apple,nvidia | training | declared | part | NO |
| `UMAP` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `WhiteKernel` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `clip_grad_norm_` | 1 | amd,apple,nvidia | training | declared | part | NO |
| `cross_entropy` | 1 | amd,apple,nvidia | training | declared | part | NO |
| `cross_val_score` | 1 | amd,apple,nvidia | training | declared | part | NO |
| `embedding.Embedding` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `hdbscan.HDBSCAN` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `hdbscan.all_points_membership_vectors` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `hdbscan.approximate_predict` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `hdbscan.membership_vector` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `host_model` | 2 | apple | training | declared | part | NO |
| `host_predict` | **0** |  |  |  |  | NO |
| `host_predict_proba` | **0** |  |  |  |  | NO |
| `kernel_methods.KernelRidge` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `kernel_methods.Nystroem` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `kernel_methods.RBFSampler` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `kpss_test` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `language_model.ByteLanguageModelConfig` | 8 | amd,apple,nvidia | training | declared | part | NO |
| `language_model.LanguageModelConfig` (alias of `ByteLanguageModelConfig`) | 8 | amd,apple,nvidia | training | declared | part | NO |
| `language_model.LanguageModelHostTrainer` | 1 | amd,apple,nvidia | training | declared | n/a | NO |
| `language_model.LanguageModelInference` | 17 | amd,apple,nvidia | training | seen(build) | part | yes |
| `language_model.LanguageModelTrainer` (alias of `SmallByteLanguageModelTrainer`) | 5 | amd,apple,nvidia | training | declared | part | NO |
| `language_model.SmallByteLanguageModelTrainer` | 5 | amd,apple,nvidia | training | declared | part | NO |
| `linalg.Cholesky` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `linalg.PROFILE_BF16` | **0** |  |  |  |  | NO |
| `linalg.PROFILE_INT8` | **0** |  |  |  |  | NO |
| `linalg.dequantize_int8` | 1 | apple | training | declared | n/a | NO |
| `linalg.from_bf16` | **0** |  |  |  |  | NO |
| `linalg.matmul` | 2 | amd,apple,nvidia | training | declared | part | NO |
| `linalg.matmul_bf16` | 1 | apple | training | declared | n/a | NO |
| `linalg.matmul_int8` | 1 | apple | training | declared | n/a | NO |
| `linalg.quantize_int8` | 1 | apple | training | declared | n/a | NO |
| `linalg.to_bf16` | 1 | apple | training | declared | n/a | NO |
| `lowbit.BF16Weight` | **0** |  |  |  |  | NO |
| `lowbit.FORMATS` | **0** |  |  |  |  | NO |
| `lowbit.Int8Weight` | **0** |  |  |  |  | NO |
| `lowbit.format_of` | **0** |  |  |  |  | NO |
| `lowbit.is_packed` | **0** |  |  |  |  | NO |
| `lowbit.materialize` (alias of `unpack`) | 4 | apple | training | declared | n/a | NO |
| `lowbit.materialize_one` | **0** |  |  |  |  | NO |
| `lowbit.pack` | 12 | apple | training | declared | part | NO |
| `lowbit.pack_one` | **0** |  |  |  |  | NO |
| `lowbit.unpack` | 4 | apple | training | declared | n/a | NO |
| `lowbit.widen_bf16` | **0** |  |  |  |  | NO |
| `mamba.Mamba1Block` | 9 | amd,apple,nvidia | training | seen(build) | part | yes |
| `mamba.Mamba1DecodeSession` | **0** |  |  |  |  | NO |
| `mamba.Mamba2Block` | 10 | amd,apple,nvidia | training | seen(build) | part | yes |
| `mamba.Mamba3Block` | 9 | amd,apple,nvidia | training | seen(build) | part | yes |
| `matmul` | 2 | amd,apple,nvidia | training | declared | part | NO |
| `metrics.accuracy_score` | 1 | amd,apple,nvidia | training | declared | n/a | NO |
| `metrics.adjusted_rand_score` | 1 | amd,apple,nvidia | training | declared | n/a | NO |
| `metrics.completeness_score` | 2 | amd,apple,nvidia | training | seen(build) | part | NO |
| `metrics.confusion_matrix` | 1 | amd,apple,nvidia | training | declared | part | NO |
| `metrics.entropy` | 1 | amd,apple,nvidia | training | declared | part | NO |
| `metrics.f1_score` | 1 | amd,apple,nvidia | training | declared | part | NO |
| `metrics.fowlkes_mallows_score` | 1 |  | training | seen(build) | n/a | NO |
| `metrics.homogeneity_completeness_v_measure` | 1 |  | training | seen(build) | n/a | NO |
| `metrics.homogeneity_score` | 2 | amd,apple,nvidia | training | seen(build) | part | NO |
| `metrics.kl_divergence` | 1 | amd,apple,nvidia | training | declared | part | NO |
| `metrics.log_loss` | 1 | amd,apple,nvidia | training | declared | part | NO |
| `metrics.mean_absolute_error` | 1 | amd,apple,nvidia | training | declared | part | NO |
| `metrics.mean_squared_error` | 1 | amd,apple,nvidia | training | declared | part | NO |
| `metrics.mutual_info_score` | 1 | amd,apple,nvidia | training | declared | part | NO |
| `metrics.precision_recall_curve` | 1 | amd,apple,nvidia | training | declared | part | NO |
| `metrics.precision_score` | 1 | amd,apple,nvidia | training | declared | part | NO |
| `metrics.r2_score` | 1 | amd,apple,nvidia | training | declared | n/a | NO |
| `metrics.rand_score` | 1 | amd,apple,nvidia | training | declared | part | NO |
| `metrics.recall_score` | 1 | amd,apple,nvidia | training | declared | part | NO |
| `metrics.roc_auc_score` | 1 | amd,apple,nvidia | training | declared | part | NO |
| `metrics.root_mean_squared_error` | 1 | amd,apple,nvidia | training | declared | part | NO |
| `metrics.silhouette_samples` | 1 | amd,apple,nvidia | training | declared | part | NO |
| `metrics.silhouette_score` | 1 | amd,apple,nvidia | training | declared | n/a | NO |
| `metrics.trustworthiness` | 1 | amd,apple,nvidia | training | declared | part | NO |
| `metrics.v_measure_score` | 3 | amd,apple,nvidia | training | seen(build) | part | NO |
| `mixture.GaussianMixture` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `model_pool_training.PooledByteLanguageModelTrainer` | 1 | amd,apple,nvidia |  | none | n/a | NO |
| `model_selection.cross_val_score` | 1 | amd,apple,nvidia | training | declared | part | NO |
| `model_selection.split_descriptor` | 1 |  |  | none | n/a | NO |
| `models` | **0** |  |  |  |  | NO |
| `neural_network.SmallMLPTrainer` | 4 | amd,apple,nvidia | training | declared | part | NO |
| `offload_training.OffloadedByteLanguageModelTrainer` | 1 | amd,apple,nvidia |  | none | n/a | NO |
| `parallel_classical.apply_kernel_method` | 2 | amd,nvidia |  | none | part | NO |
| `parallel_classical.bootstrap` | 1 | amd,nvidia |  | none | part | NO |
| `parallel_classical.fit_arima` | 1 | amd,apple,nvidia | training | declared | part | NO |
| `parallel_classical.fit_cholesky` | 1 | amd,nvidia |  | none | part | NO |
| `parallel_classical.fit_coordinate_descent` | 2 | amd,apple,nvidia |  | none | part | NO |
| `parallel_classical.fit_dbscan` | 1 | amd,apple,nvidia |  | none | part | NO |
| `parallel_classical.fit_exponential_smoothing` | 1 | amd,apple,nvidia | training | declared | part | NO |
| `parallel_classical.fit_gaussian_mixture` | 1 | amd,nvidia |  | none | part | NO |
| `parallel_classical.fit_gaussian_process` | 1 | amd,apple,nvidia |  | none | part | NO |
| `parallel_classical.fit_gram_estimator` | 4 | amd,apple,nvidia |  | none | part | NO |
| `parallel_classical.fit_hdbscan` | 1 | amd,apple,nvidia |  | none | part | NO |
| `parallel_classical.fit_kernel_method` | 2 | amd,nvidia |  | none | part | NO |
| `parallel_classical.fit_kmeans` | 1 | amd,apple,nvidia |  | none | part | NO |
| `parallel_classical.fit_logistic` | 1 | amd,apple,nvidia |  | none | part | NO |
| `parallel_classical.fit_svm` | 2 | amd,apple,nvidia |  | none | part | NO |
| `parallel_classical.monte_carlo_integrate` | 1 | amd,nvidia |  | none | part | NO |
| `parallel_classical.permutation_test` | 1 | amd,nvidia |  | none | part | NO |
| `parallel_classical.predict_gaussian_mixture` | 1 | amd,nvidia |  | none | part | NO |
| `parallel_classical.predict_gaussian_process` | 1 | amd,apple,nvidia |  | none | part | NO |
| `parallel_classical.predict_svm` | 2 | amd,apple,nvidia |  | none | part | NO |
| `parallel_classical.solve_cholesky` | 1 | amd,nvidia |  | none | part | NO |
| `parallel_classical.transform_rbf_sampler` | 1 | amd,nvidia |  | none | part | NO |
| `parallel_ensemble.ParallelForestPredictor` | 1 | amd,nvidia |  | none | part | NO |
| `parallel_ensemble.fit_boosting` | 4 | amd,apple,nvidia |  | none | part | NO |
| `parallel_ensemble.fit_feature_freq` | 1 | amd,apple,nvidia |  | none | part | NO |
| `parallel_ensemble.fit_forest` | 4 | amd,apple,nvidia | training | declared | part | NO |
| `parallel_ensemble.fit_isolation_forest` | 1 | amd,apple,nvidia |  | none | part | NO |
| `parallel_ensemble.fit_ordered_rmse` | 1 | amd,apple,nvidia |  | none | part | NO |
| `parallel_ensemble.score_isolation_forest` | 1 | amd,apple,nvidia |  | none | part | NO |
| `parallel_graph.fit_graph` | 3 | amd,apple,nvidia |  | none | part | NO |
| `parallel_graph.transform_umap` | 1 | amd,apple,nvidia |  | none | part | NO |
| `parallel_neighbors.ParallelQueries` | 4 | amd,apple,nvidia | training | declared | part | NO |
| `parallel_neighbors_reference.ReferenceShardedNeighbors` | 2 | amd,apple,nvidia | training | declared | part | NO |
| `parallel_preprocessing.fit_scaler` | 2 | amd,apple,nvidia | training | declared | part | NO |
| `parallel_preprocessing.transform_scaler` | 2 | amd,apple,nvidia | training | declared | part | NO |
| `parallel_training.ParallelByteLanguageModelTrainer` | 3 | amd,apple,nvidia |  | none | n/a | NO |
| `parallel_training.ParallelNeuralTrainer` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `parallel_training.ordered_sum_gradients` | 1 |  | training | seen(build) | n/a | NO |
| `preprocessing.MinMaxScaler` | 3 | amd,apple,nvidia | training | declared | part | NO |
| `preprocessing.StandardScaler` | 4 | amd,apple,nvidia | training | declared | part | NO |
| `resample.bootstrap` | 2 | amd,apple,nvidia | training | declared | part | NO |
| `resample.monte_carlo_integrate` | 2 | amd,apple,nvidia | training | declared | part | NO |
| `resample.permutation_test` | 2 | amd,apple,nvidia | training | declared | part | NO |
| `select_d` | 1 |  | training | seen(build) | part | NO |
| `tokenizer.BpeVocabularyTrainer` | 1 |  |  | none | n/a | NO |
| `tokenizer.GPT2Tokenizer` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `tokenizer.TrainedBpeVocabulary` | **0** |  |  |  |  | NO |
| `training.Adam` | 1 | amd,apple,nvidia | training | declared | part | NO |
| `training.AdamW` | 1 | amd,apple,nvidia | training | declared | part | NO |
| `training.ConstantLR` | 1 | amd,apple,nvidia | training | declared | part | NO |
| `training.Generator` | 6 | amd,apple,nvidia | training | seen(build) | part | yes |
| `training.SGD` | 1 | amd,apple,nvidia | training | declared | part | NO |
| `training.SambaConfig` | 6 | amd,apple,nvidia | training | seen(build) | part | yes |
| `training.SambaStack` | 6 | amd,apple,nvidia | training | seen(build) | part | yes |
| `training.WarmupCosineLR` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `training.WarmupLinearLR` | 1 | amd,apple,nvidia | training | declared | part | NO |
| `training.accumulate_grads` | routed |  |  |  |  | NO |
| `training.accumulation_is_aligned` | routed |  |  |  |  | NO |
| `training.clip_grad_norm_` | 1 | amd,apple,nvidia | training | declared | part | NO |
| `training.cross_entropy` | 1 | amd,apple,nvidia | training | declared | part | NO |
| `training.embedding_backward` | 1 | amd,apple,nvidia | training | declared | part | NO |
| `training.embedding_forward` | 1 | amd,apple,nvidia | training | declared | part | NO |
| `training.linear_backward` | 1 | amd,apple,nvidia | training | declared | part | NO |
| `training.linear_forward` | 1 | amd,apple,nvidia | training | declared | part | NO |
| `training.rms_norm_backward` | 1 | amd,apple,nvidia | training | declared | part | NO |
| `training.rms_norm_forward` | 1 | amd,apple,nvidia | training | declared | part | NO |
| `transformer.TransformerBlock` | 10 | amd,apple,nvidia | training | seen(build) | part | yes |
| `transformer.TransformerDecodeSession` | **0** |  |  |  |  | NO |
| `umap.UMAP` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |

## The lane matrix

| lane | gpu | cpu | sabotage | sabotage evidence | batch | all four |
|---|---|---|---|---|---|---|
| agglomerative | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-14_cpu-phase1b/cpu-apple-m4.sabotage.agglomerative-et.json` | part | yes |
| arima | amd,apple,nvidia | training | declared | - | part | NO |
| arima-011 | amd,apple,nvidia | training | declared | - | part | NO |
| arima-exog | - | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-public-promotion/cpu-sabotage.json` | part | NO |
| arima-exog-seasonal | - | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-public-promotion/cpu-sabotage.json` | part | NO |
| arima-seasonal-c | amd,apple,nvidia | training | declared | - | part | NO |
| bootstrap | amd,apple,nvidia | training | declared | - | part | NO |
| bpe-trainer | - | - | none | - | n/a n/a:corpus-global-vocabulary-training (pair counts depend on the complete corpus; no per-row output) | NO |
| byte-lm | amd,apple,nvidia | training | declared | - | part | NO |
| byte-lm-host-infer | amd,apple,nvidia | training | declared | - | part | NO |
| byte-lm-host-infer-threaded | amd,apple,nvidia | training | declared | - | part | NO |
| byte-lm-host-train | amd,apple,nvidia | training | declared | - | n/a n/a:mean-reduction-fixed-batch (LanguageModelHostTrainer has train_step and loss only, and loss IS a step, python/mojolearn/_byte_lm_host.py:392-429; one loss and one update from mean cross-entropy over ids of the profile's fixed (batch, length + 1), so no output belongs to one sequence; the per-sequence logits are asked by byte-lm-host-infer; lane/batch-invariance-2's batchgrad records the same reason for this lane and its batchscale and ragged parts do not ask it) | NO |
| byte-lm-resident | amd,apple,nvidia | training | declared | - | part | NO |
| cholesky | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cholesky-cpu-inference/cpu-apple-m4.sabotage.json` | part | yes |
| cross-entropy-arms | amd,apple,nvidia | training | declared | - | part | NO |
| cross-val | amd,apple,nvidia | training | declared | - | part | NO |
| cross-val-folds | - | - | none | - | n/a n/a:function | NO |
| dbscan | amd,apple,nvidia | training | declared | - | part | NO |
| dbscan-brute-l1 | amd,apple,nvidia | training | declared | - | part | NO |
| dbscan-weighted | amd,apple,nvidia | training | declared | - | part | NO |
| elasticnet | amd,apple,nvidia | training | declared | - | part | NO |
| elasticnet-l2end-no-intercept | amd,apple,nvidia | training | declared | - | part | NO |
| embedding | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_ivf-embedding-cpu-inference/cpu-x86.sabotage.json` | part | yes |
| embedding-sort | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_ivf-embedding-cpu-inference/cpu-x86.sabotage.json` | part | yes |
| et-clf | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-14_cpu-phase1b/cpu-apple-m4.sabotage.et.json` | part | yes |
| et-clf-entropy-bestfirst | amd,apple,nvidia | training | declared | - | part | NO |
| et-reg | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-14_cpu-phase1b/cpu-apple-m4.sabotage.et.json` | part | yes |
| et-reg-bootstrap-parallel | amd,apple,nvidia | training | declared | - | part | NO |
| gbdt-adapter-clf | amd,apple,nvidia | training | declared | - | part | NO |
| gbdt-adapter-reg | amd,apple,nvidia | training | declared | - | part | NO |
| gbdt-adapter-score-weighted | - | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-classical-completion/gbdt-adapter-score-weighted/cpu-sabotage.json` | n/a n/a:scalar-reduction (score(X, y, sample_weight) returns one float over every row it is handed, python/mojolearn/_gbdt_adapters.py and _forest_protocol.py; the predict and predict_proba it wraps keep their batch parts on gbdt-adapter-clf, gbdt-adapter-reg, rf-clf and rf-reg) | NO |
| gbdt-categorical-ctr | amd,apple,nvidia | training | declared | - | part | NO |
| gbdt-categorical-ctr-tables | apple | training | declared | - | part | NO |
| gbdt-depthwise | amd,apple,nvidia | training | declared | - | part | NO |
| gbdt-exact-mae | amd,apple,nvidia | training | declared | - | part | NO |
| gbdt-feature-freq | amd,apple,nvidia | training | declared | - | part | NO |
| gbdt-lossguide | amd,apple,nvidia | training | declared | - | part | NO |
| gbdt-lossguide-newtoncosine | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-classical-completion/gbdt-lossguide-newtoncosine/cpu-sabotage.json` | part | yes |
| gbdt-multiclass | amd,apple,nvidia | training | declared | - | part | NO |
| gbdt-nan-modes | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-classical-completion/gbdt-nan-modes/cpu-sabotage.json` | part | yes |
| gbdt-onevsall | amd,apple,nvidia | training | declared | - | part | NO |
| gbdt-ordered-rmse | amd,apple,nvidia | training | declared | - | part | NO |
| gbdt-pair-logit | apple | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-classical-completion/gbdt-pair-logit/cpu-sabotage.json` | part | yes |
| gbdt-parametric-losses | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-classical-completion/gbdt-parametric-losses/cpu-sabotage.json` | part | yes |
| gbdt-pointwise-l2-bayesian-eval | amd,apple,nvidia | training | declared | - | part | NO |
| gbdt-query-rmse | apple | training | declared | - | part | NO |
| gbdt-rmse | amd,apple,nvidia | training | declared | - | part | NO |
| gbdt-symmetric | amd,apple,nvidia | training | declared | - | part | NO |
| gbdt-tensor-ctr-tables | apple | training | declared | - | part | NO |
| gbdt-yeti-rank | apple | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-classical-completion/gbdt-yeti-rank/cpu-sabotage.json` | part | yes |
| gemm-bf16 | apple | training | declared | - | n/a n/a:profile lane; the products are hashed whole | NO |
| gemm-int8 | apple | training | declared | - | n/a n/a:profile lane; the products are hashed whole | NO |
| gemm-pinned | amd,apple,nvidia | training | declared | - | part | NO |
| gemm-transposed | amd,apple,nvidia | training | declared | - | part | NO |
| gmm | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-d-estimators/cpu-apple-m4.sabotage.json` | part | yes |
| gmm-random-init | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-d-estimators/cpu-apple-m4.sabotage.json` | part | yes |
| gmm-random-init-sample | apple | training | declared | - | n/a n/a:no-batch-axis (GaussianMixture.sample takes no input rows) | NO |
| gmm-sample | apple | training | declared | - | n/a n/a:no-batch-axis (GaussianMixture.sample takes no input rows) | NO |
| gp | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-d-estimators/cpu-apple-m4.sabotage.json` | part | yes |
| gp-matern12 | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-d-estimators/cpu-apple-m4.sabotage.json` | part | yes |
| gp-matern32 | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-d-estimators/cpu-apple-m4.sabotage.json` | part | yes |
| gp-matern52-ard | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-d-estimators/cpu-apple-m4.sabotage.json` | part | yes |
| gp-normalize-y | apple | training | declared | - | part | NO |
| gp-optimize | apple | training | declared | - | part | NO |
| gp-optimize-restarts | apple | training | declared | - | part | NO |
| gp-sample-y | apple | training | declared | - | n/a n/a:jointly-correlated (sample_y draws all rows of a call from one joint posterior) | NO |
| gp-sample-y-normalize | apple | training | declared | - | n/a n/a:jointly-correlated (sample_y draws all rows of a call from one joint posterior) | NO |
| gpc | apple | training | declared | - | part | NO |
| gpc-multiclass | apple | training | declared | - | part | NO |
| hdbscan | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-d-estimators/cpu-apple-m4.sabotage.json` | part | yes |
| hdbscan-leaf | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-d-estimators/cpu-apple-m4.sabotage.json` | part | yes |
| holtwinters | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_tsa-negative-controls/cpu-sabotage.json` | part | yes |
| holtwinters-multiplicative | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_tsa-negative-controls/cpu-sabotage.json` | part | yes |
| iforest | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-14_cpu-phase1b/cpu-apple-m4.sabotage.iforest.json` | part | yes |
| iforest-tuned | amd,apple,nvidia | training | declared | - | part | NO |
| ivf | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_ivf-embedding-cpu-inference/cpu-x86.sabotage.json` | part | yes |
| ivf-euclidean | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_ivf-embedding-cpu-inference/cpu-x86.sabotage.json` | part | yes |
| ivf-extend | apple | training | seen(build) | `bench/results/identity_break/2026-09-15_ivf-extend/cpu-x86.sabotage.json` | part | yes |
| kde | amd,apple,nvidia | training | declared | - | part | NO |
| kde-cosine-minkowski | amd,apple,nvidia | training | declared | - | part | NO |
| kde-epanechnikov-l1 | amd,apple,nvidia | training | declared | - | part | NO |
| kde-exponential-chebyshev | amd,apple,nvidia | training | declared | - | part | NO |
| kde-linear-cosine | amd,apple,nvidia | training | declared | - | part | NO |
| kde-tophat-sqeuclidean | amd,apple,nvidia | training | declared | - | part | NO |
| kde-weighted | amd,apple,nvidia | training | declared | - | part | NO |
| kernel-ridge | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-d-estimators/cpu-apple-m4.sabotage.json` | part | yes |
| kmeans | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-16_kmeans-and-spectral-cpu/nvidia-a100-sm_80.host-sabotage.json` | part | yes |
| kmeans-array | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-16_kmeans-and-spectral-cpu/nvidia-a100-sm_80.host-sabotage.json` | part | yes |
| kmeans-classic-pp | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-16_kmeans-and-spectral-cpu/nvidia-a100-sm_80.host-sabotage.json` | part | yes |
| kmeans-cosine | amd,apple,nvidia | training | declared | - | n/a n/a:fit-refused | NO |
| kmeans-random | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-16_kmeans-and-spectral-cpu/nvidia-a100-sm_80.host-sabotage.json` | part | yes |
| kmeans-sqrt | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-16_kmeans-and-spectral-cpu/nvidia-a100-sm_80.host-sabotage.json` | part | yes |
| kmeans-weighted | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-16_kmeans-and-spectral-cpu/nvidia-a100-sm_80.host-sabotage.json` | part | yes |
| knn | amd,apple,nvidia | training | declared | - | part | NO |
| knn-chebyshev | amd,apple,nvidia | training | declared | - | part | NO |
| knn-clf | amd,apple,nvidia | training | declared | - | part | NO |
| knn-clf-distance | amd,apple,nvidia | training | declared | - | part | NO |
| knn-cosine | amd,apple,nvidia | training | declared | - | part | NO |
| knn-manhattan | amd,apple,nvidia | training | declared | - | part | NO |
| knn-minkowski-p3 | amd,apple,nvidia | training | declared | - | part | NO |
| knn-rbc | amd,apple,nvidia | training | declared | - | part | NO |
| knn-reg | amd,apple,nvidia | training | declared | - | part | NO |
| knn-reg-distance | amd,apple,nvidia | training | declared | - | part | NO |
| knn-sqeuclidean | amd,apple,nvidia | training | declared | - | part | NO |
| kpss | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_tsa-negative-controls/cpu-sabotage.json` | part | yes |
| lasso | amd,apple,nvidia | training | declared | - | part | NO |
| logistic | amd,apple,nvidia | training | declared | - | part | NO |
| logistic-elasticnet | amd,apple,nvidia | training | declared | - | part | NO |
| logistic-l1 | amd,apple,nvidia | training | declared | - | part | NO |
| logistic-multiclass | amd,apple,nvidia | training | declared | - | part | NO |
| logistic-unpenalized-no-intercept | amd,apple,nvidia | training | declared | - | part | NO |
| mamba1 | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-mamba/cpu-apple-m4.sabotage.json` | part | yes |
| mamba1-bf16w | apple | training | declared | - | part | NO |
| mamba1-int8w | apple | training | declared | - | part | NO |
| mamba2 | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-mamba/cpu-apple-m4.sabotage.json` | part | yes |
| mamba2-bf16w | apple | training | declared | - | part | NO |
| mamba2-dtlimit | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-mamba/cpu-apple-m4.sabotage.json` | part | yes |
| mamba2-int8w | apple | training | declared | - | part | NO |
| mamba3 | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-mamba/cpu-apple-m4.sabotage.json` | part | yes |
| mamba3-bf16w | apple | training | declared | - | part | NO |
| mamba3-int8w | apple | training | declared | - | part | NO |
| metrics | amd,apple,nvidia | training | declared | - | n/a n/a:scalar-reduction (accuracy_score, adjusted_rand_score, v_measure_score, r2_score and silhouette_score each return one float over every row, python/mojolearn/_metrics_impl.py; the per-sample silhouette_samples is asked on metrics-classification) | NO |
| metrics-classification | amd,apple,nvidia | training | declared | - | part | NO |
| metrics-fowlkes-mallows | - | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-public-promotion/cpu-sabotage.json` | n/a n/a:function | NO |
| metrics-homogeneity-completeness | - | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-classical-completion/metrics-homogeneity-completeness/cpu-sabotage.json` | n/a n/a:global-contingency-reduction (three scalar scores over all labels; no per-row output) | NO |
| minmax-scaler | amd,apple,nvidia | training | declared | - | part | NO |
| minmax-scaler-clip | amd,apple,nvidia | training | declared | - | part | NO |
| mlp | amd,apple,nvidia | training | declared | - | part | NO |
| mlp-bf16w | apple | training | declared | - | n/a n/a:weight-format lane; the batch part is measured on its base lane | NO |
| mlp-int8w | apple | training | declared | - | n/a n/a:weight-format lane; the batch part is measured on its base lane | NO |
| monte-carlo | amd,apple,nvidia | training | declared | - | n/a n/a:scalar-fold (resample.monte_carlo_integrate returns only integral, mean, volume and closed_form folded over [i_first, i_first + n_samples) by the pinned chunk tree, python/mojolearn/resample.py:196; no per-sample output exists to compare an i_first range against) | NO |
| nystroem | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-d-estimators/cpu-apple-m4.sabotage.json` | part | yes |
| ols | amd,apple,nvidia | training | seen(build) | `bench/results/runpod_cpu/2026-09-15_proof/cold/leg_out/cpu-x86.host-sabotage.json` | part | yes |
| ols-no-intercept | amd,apple,nvidia | training | declared | - | part | NO |
| ols-weighted | amd,apple,nvidia | training | declared | - | part | NO |
| optim-adam-clip | amd,apple,nvidia | training | declared | - | part | NO |
| optim-sgd | amd,apple,nvidia | training | declared | - | part | NO |
| ordered-gradient-sum | - | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-ordered-sum-completion/cpu-sabotage.json` | n/a n/a:ordered-shard-reduction (the specified shard order defines the sum; no per-row prediction) | NO |
| par-arima | amd,apple,nvidia | training | declared | - | part | NO |
| par-boosting | amd,apple,nvidia | - | none | - | part | NO |
| par-boosting-clf | - | - | none | - | part | NO |
| par-boosting-pointwise | amd,apple,nvidia | - | none | - | part | NO |
| par-boosting-reg | - | - | none | - | part | NO |
| par-byte-lm | amd,apple,nvidia | - | none | - | n/a n/a:driver-step (ParallelByteLanguageModelTrainer and its Pooled and Offloaded subclasses have train_step, state_dict, export_gradients and checkpoint only, python/mojolearn/parallel_training.py:89-172; train_step returns per-shard mean losses and one update from the shard-mean gradient, so no output belongs to one sequence, and the shard split is held to the replica trainer by the train column) | NO |
| par-byte-lm-model-pool | amd,apple,nvidia | - | none | - | n/a n/a:driver-step (ParallelByteLanguageModelTrainer and its Pooled and Offloaded subclasses have train_step, state_dict, export_gradients and checkpoint only, python/mojolearn/parallel_training.py:89-172; train_step returns per-shard mean losses and one update from the shard-mean gradient, so no output belongs to one sequence, and the shard split is held to the replica trainer by the train column) | NO |
| par-byte-lm-offload | amd,apple,nvidia | - | none | - | n/a n/a:driver-step (ParallelByteLanguageModelTrainer and its Pooled and Offloaded subclasses have train_step, state_dict, export_gradients and checkpoint only, python/mojolearn/parallel_training.py:89-172; train_step returns per-shard mean losses and one update from the shard-mean gradient, so no output belongs to one sequence, and the shard split is held to the replica trainer by the train column) | NO |
| par-cd | amd,apple,nvidia | - | none | - | part | NO |
| par-cd-elasticnet | - | - | none | - | part | NO |
| par-cholesky | amd,nvidia | - | none | - | part | NO |
| par-dbscan | amd,apple,nvidia | - | none | - | part | NO |
| par-feature-freq | amd,apple,nvidia | - | none | - | part | NO |
| par-forest | amd,apple,nvidia | training | declared | - | part | NO |
| par-forest-et | amd,apple,nvidia | training | declared | - | part | NO |
| par-forest-et-clf | - | - | none | - | part | NO |
| par-forest-pool | amd,nvidia | - | none | - | part | NO |
| par-forest-reg | - | - | none | - | part | NO |
| par-gmm | amd,nvidia | - | none | - | part | NO |
| par-gp | amd,apple,nvidia | - | none | - | part | NO |
| par-gram | amd,apple,nvidia | - | none | - | part | NO |
| par-gram-ols | - | - | none | - | part | NO |
| par-gram-pca | - | - | none | - | part | NO |
| par-gram-tsvd | - | - | none | - | part | NO |
| par-graph-agglomerative | amd,apple,nvidia | - | none | - | part | NO |
| par-graph-spectral | amd,apple,nvidia | - | none | - | n/a n/a:transductive (the par-graph-spectral lane fits without prediction_data; SpectralClustering.predict is the spectral lane's) | NO |
| par-graph-umap | amd,apple,nvidia | - | none | - | part | NO |
| par-hdbscan | amd,apple,nvidia | - | none | - | part | NO |
| par-holtwinters | amd,apple,nvidia | training | declared | - | part | NO |
| par-iforest | amd,apple,nvidia | - | none | - | part | NO |
| par-kernel-ridge | amd,nvidia | - | none | - | part | NO |
| par-kmeans | amd,apple,nvidia | - | none | - | part | NO |
| par-logistic | amd,apple,nvidia | - | none | - | part | NO |
| par-mlp | amd,apple,nvidia | training | declared | - | part | NO |
| par-nystroem | amd,nvidia | - | none | - | part | NO |
| par-ordered-rmse | amd,apple,nvidia | - | none | - | part | NO |
| par-queries-kde | amd,apple,nvidia | training | declared | - | part | NO |
| par-queries-knn | amd,apple,nvidia | training | declared | - | part | NO |
| par-queries-nn | - | - | none | - | part | NO |
| par-queries-radius | amd,apple,nvidia | training | declared | - | part | NO |
| par-rbf-sampler | amd,nvidia | - | none | - | part | NO |
| par-reference-knn | amd,apple,nvidia | training | declared | - | part | NO |
| par-reference-knn-reg | amd,apple,nvidia | training | declared | - | part | NO |
| par-resample | amd,nvidia | - | none | - | part | NO |
| par-samba | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-16_cpu-par-samba/cpu-apple-m4.sabotage.json` | part | yes |
| par-samba-clip | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-16_cpu-par-samba/cpu-apple-m4.sabotage.json` | part | yes |
| par-scaler | amd,apple,nvidia | training | declared | - | part | NO |
| par-scaler-minmax | - | - | none | - | part | NO |
| par-svm | amd,apple,nvidia | - | none | - | part | NO |
| par-svm-svr | - | - | none | - | part | NO |
| pca | amd,apple,nvidia | training | declared | - | part | NO |
| pca-full-whiten | amd,apple,nvidia | training | declared | - | part | NO |
| pca-whiten | amd,apple,nvidia | training | declared | - | part | NO |
| permutation-test | amd,apple,nvidia | training | declared | - | part | NO |
| radius | amd,apple,nvidia | training | declared | - | part | NO |
| radius-chebyshev | amd,apple,nvidia | training | declared | - | part | NO |
| radius-manhattan | amd,apple,nvidia | training | declared | - | part | NO |
| radius-minkowski-p3 | amd,apple,nvidia | training | declared | - | part | NO |
| rbf-sampler | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-d-estimators/cpu-apple-m4.sabotage.json` | part | yes |
| rf-clf | amd,apple,nvidia | training | declared | - | part | NO |
| rf-clf-balanced-parallel | amd,apple,nvidia | training | declared | - | part | NO |
| rf-clf-entropy-log2-noboot | amd,apple,nvidia | training | declared | - | part | NO |
| rf-reg | amd,apple,nvidia | training | declared | - | part | NO |
| rf-reg-gamma-ig | amd,apple,nvidia | training | declared | - | part | NO |
| rf-reg-poisson | amd,apple,nvidia | training | declared | - | part | NO |
| rf-score-weighted | - | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-classical-completion/rf-score-weighted/cpu-sabotage.json` | n/a n/a:scalar-reduction (score(X, y, sample_weight) returns one float over every row it is handed, python/mojolearn/_gbdt_adapters.py and _forest_protocol.py; the predict and predict_proba it wraps keep their batch parts on gbdt-adapter-clf, gbdt-adapter-reg, rf-clf and rf-reg) | NO |
| ridge | amd,apple,nvidia | training | seen(build) | `bench/results/runpod_cpu/2026-09-15_proof/cold/leg_out/cpu-x86.host-sabotage.json` | part | yes |
| ridge-no-intercept | amd,apple,nvidia | training | declared | - | part | NO |
| samba | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-samba/cpu-apple-m4.sabotage.json` | part | yes |
| samba-bf16w | apple | training | declared | - | n/a n/a:weight-format lane; the batch part is measured on its base lane | NO |
| samba-int8w | apple | training | declared | - | n/a n/a:weight-format lane; the batch part is measured on its base lane | NO |
| samba-untied-dropout-accum | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-samba/cpu-apple-m4.sabotage.json` | part | yes |
| select-d | - | training | seen(build) | `bench/results/identity_break/2026-09-17_tsa-negative-controls/cpu-sabotage.json` | part | NO |
| spectral | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-public-promotion/cpu-sabotage.json` | part | yes |
| spectral-precomputed | amd,apple,nvidia | training | declared | - | part | NO |
| standard-scaler | amd,apple,nvidia | training | declared | - | part | NO |
| standard-scaler-no-mean | amd,apple,nvidia | training | declared | - | part | NO |
| standard-scaler-no-std | amd,apple,nvidia | training | declared | - | part | NO |
| svc | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-svm/cpu-x86-sabotage.json` | part | yes |
| svc-linear | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-svm/cpu-x86-sabotage.json` | part | yes |
| svc-poly | apple | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-svm/cpu-x86-sabotage.json` | part | yes |
| svr | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-svm/cpu-x86-sabotage.json` | part | yes |
| svr-linear | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-svm/cpu-x86-sabotage.json` | part | yes |
| tokenizer | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_clean-third-party/m4-owed/cpu-apple-m4.host-sabotage.json` | part | yes |
| training-primitives | amd,apple,nvidia | training | declared | - | part | NO |
| transformer | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-transformer/cpu-apple-m4.sabotage.json` | part | yes |
| transformer-bf16w | apple | training | declared | - | part | NO |
| transformer-int8w | apple | training | declared | - | part | NO |
| transformer-window | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-transformer/cpu-apple-m4.sabotage.json` | part | yes |
| tsvd | amd,apple,nvidia | training | declared | - | part | NO |
| umap | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-17_cpu-umap-completion/cpu-sabotage.json` | part | yes |

## Lanes missing each kind, by name

**No GPU column at all: 21**

> arima-exog, arima-exog-seasonal, bpe-trainer, cross-val-folds, gbdt-adapter-score-weighted, metrics-fowlkes-mallows, metrics-homogeneity-completeness, ordered-gradient-sum, par-boosting-clf, par-boosting-reg, par-cd-elasticnet, par-forest-et-clf, par-forest-reg, par-gram-ols, par-gram-pca, par-gram-tsvd, par-queries-nn, par-scaler-minmax, par-svm-svr, rf-score-weighted, select-d

**GPU column on fewer than three classes: 37**

> gbdt-categorical-ctr-tables, gbdt-pair-logit, gbdt-query-rmse, gbdt-tensor-ctr-tables, gbdt-yeti-rank, gemm-bf16, gemm-int8, gmm-random-init-sample, gmm-sample, gp-normalize-y, gp-optimize, gp-optimize-restarts, gp-sample-y, gp-sample-y-normalize, gpc, gpc-multiclass, ivf-extend, mamba1-bf16w, mamba1-int8w, mamba2-bf16w, mamba2-int8w, mamba3-bf16w, mamba3-int8w, mlp-bf16w, mlp-int8w, par-cholesky, par-forest-pool, par-gmm, par-kernel-ridge, par-nystroem, par-rbf-sampler, par-resample, samba-bf16w, samba-int8w, svc-poly, transformer-bf16w, transformer-int8w

**No CPU verifier declared: 39**

> bpe-trainer, cross-val-folds, par-boosting, par-boosting-clf, par-boosting-pointwise, par-boosting-reg, par-byte-lm, par-byte-lm-model-pool, par-byte-lm-offload, par-cd, par-cd-elasticnet, par-cholesky, par-dbscan, par-feature-freq, par-forest-et-clf, par-forest-pool, par-forest-reg, par-gmm, par-gp, par-gram, par-gram-ols, par-gram-pca, par-gram-tsvd, par-graph-agglomerative, par-graph-spectral, par-graph-umap, par-hdbscan, par-iforest, par-kernel-ridge, par-kmeans, par-logistic, par-nystroem, par-ordered-rmse, par-queries-nn, par-rbf-sampler, par-resample, par-scaler-minmax, par-svm, par-svm-svr

**Sabotage not seen to move a build: 166**

> arima, arima-011, arima-seasonal-c, bootstrap, bpe-trainer, byte-lm, byte-lm-host-infer, byte-lm-host-infer-threaded, byte-lm-host-train, byte-lm-resident, cross-entropy-arms, cross-val, cross-val-folds, dbscan, dbscan-brute-l1, dbscan-weighted, elasticnet, elasticnet-l2end-no-intercept, et-clf-entropy-bestfirst, et-reg-bootstrap-parallel, gbdt-adapter-clf, gbdt-adapter-reg, gbdt-categorical-ctr, gbdt-categorical-ctr-tables, gbdt-depthwise, gbdt-exact-mae, gbdt-feature-freq, gbdt-lossguide, gbdt-multiclass, gbdt-onevsall, gbdt-ordered-rmse, gbdt-pointwise-l2-bayesian-eval, gbdt-query-rmse, gbdt-rmse, gbdt-symmetric, gbdt-tensor-ctr-tables, gemm-bf16, gemm-int8, gemm-pinned, gemm-transposed, gmm-random-init-sample, gmm-sample, gp-normalize-y, gp-optimize, gp-optimize-restarts, gp-sample-y, gp-sample-y-normalize, gpc, gpc-multiclass, iforest-tuned, kde, kde-cosine-minkowski, kde-epanechnikov-l1, kde-exponential-chebyshev, kde-linear-cosine, kde-tophat-sqeuclidean, kde-weighted, kmeans-cosine, knn, knn-chebyshev, knn-clf, knn-clf-distance, knn-cosine, knn-manhattan, knn-minkowski-p3, knn-rbc, knn-reg, knn-reg-distance, knn-sqeuclidean, lasso, logistic, logistic-elasticnet, logistic-l1, logistic-multiclass, logistic-unpenalized-no-intercept, mamba1-bf16w, mamba1-int8w, mamba2-bf16w, mamba2-int8w, mamba3-bf16w, mamba3-int8w, metrics, metrics-classification, minmax-scaler, minmax-scaler-clip, mlp, mlp-bf16w, mlp-int8w, monte-carlo, ols-no-intercept, ols-weighted, optim-adam-clip, optim-sgd, par-arima, par-boosting, par-boosting-clf, par-boosting-pointwise, par-boosting-reg, par-byte-lm, par-byte-lm-model-pool, par-byte-lm-offload, par-cd, par-cd-elasticnet, par-cholesky, par-dbscan, par-feature-freq, par-forest, par-forest-et, par-forest-et-clf, par-forest-pool, par-forest-reg, par-gmm, par-gp, par-gram, par-gram-ols, par-gram-pca, par-gram-tsvd, par-graph-agglomerative, par-graph-spectral, par-graph-umap, par-hdbscan, par-holtwinters, par-iforest, par-kernel-ridge, par-kmeans, par-logistic, par-mlp, par-nystroem, par-ordered-rmse, par-queries-kde, par-queries-knn, par-queries-nn, par-queries-radius, par-rbf-sampler, par-reference-knn, par-reference-knn-reg, par-resample, par-scaler, par-scaler-minmax, par-svm, par-svm-svr, pca, pca-full-whiten, pca-whiten, permutation-test, radius, radius-chebyshev, radius-manhattan, radius-minkowski-p3, rf-clf, rf-clf-balanced-parallel, rf-clf-entropy-log2-noboot, rf-reg, rf-reg-gamma-ig, rf-reg-poisson, ridge-no-intercept, samba-bf16w, samba-int8w, spectral-precomputed, standard-scaler, standard-scaler-no-mean, standard-scaler-no-std, training-primitives, transformer-bf16w, transformer-int8w, tsvd

**Batch undeclared: 0**

> none

## Public names not counted as algorithms

Listed by name in `NOT_ALGORITHMS` in the tool, so the exclusion can be
argued with rather than hidden in a heuristic. These are process metadata,
tier switches, result containers, option lists and the caller-owned state
containers a block returns from `allocate_state`, whose buffers are hashed
through their block's own lanes.

> `Array`, `Mamba1State`, `Mamba2State`, `Mamba3State`, `TransformerState`, `__version__`, `gpu_arch`, `gpu_arch_how`, `linalg.PROFILE`, `linalg.PROFILE_FAMILY`, `linalg.PROFILE_VERSION`, `linalg.numeric_mode`, `linalg.profile`, `linalg.require_identical`, `mamba.Mamba1State`, `mamba.Mamba2State`, `mamba.Mamba3State`, `numeric_mode`, `resample.ALTERNATIVES`, `resample.BootstrapResult`, `resample.INTEGRANDS`, `resample.METHODS`, `resample.MonteCarloResult`, `resample.PermutationTestResult`, `resample.STATISTICS`, `set_numeric_mode`, `training.numeric_mode_used`, `training.vendor_used`, `transformer.TransformerState`, `vendor`

