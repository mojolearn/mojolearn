# The verification matrix

GENERATED. Do not edit by hand. `python3 tools/verification_matrix.py --write`
rebuilds it from the tree, and `--check` fails when this file is stale.
Every number below is read from `tools/identity_break.py`,
`python/mojolearn/host_surface.py` and the committed columns under
`bench/results/`. The tool's own docstring says how each cell is decided.

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

- Lanes: **199** (160 single-device, 39 `par-*` multi-GPU drivers).
- Shipped algorithms enumerated from the public API: **154**.
- Shipped algorithms with ALL FOUR kinds on at least one lane: **102** of 154.
- Shipped algorithms with NO IDENTITY LANE AT ALL: **6**.
- Shipped algorithms with no lane of their own, but reached by the harness's
  CPU inference routing: **8**.

Per kind, over the algorithms:

| kind | algorithms that have it | missing |
|---|---|---|
| gpu column | 139 | 15 |
| cpu verifier | 140 | 14 |
| sabotage seen to move a build | 103 | 51 |
| batch part or named n/a | 140 | 14 |

Per kind, over the lanes:

| kind | lanes that have it | missing |
|---|---|---|
| gpu column (any class) | 194 | 5 |
| gpu column on all three classes | 171 | 28 |
| cpu verifier declared | 169 | 30 |
| sabotage seen to move a build | 110 | 89 |
| batch part or named n/a | 199 | 0 |
| ALL FOUR | 107 | 92 |

Sabotage, split by what was actually watched:

| verdict | lanes | what it means |
|---|---|---|
| seen(build) | 110 | a sabotage BUILD moved the bytes; a real negative control |
| seen(harness) | 7 | only the harness batch switch moved; the probe can fail, the build is unproven |
| declared | 57 | the family declares a define; no committed pair moves this lane |
| none | 25 | no define reaches the lane and nothing has moved it |

## Shipped algorithms with no identity lane at all

These are the most important gaps. An algorithm with no lane cannot
be missing a cell, so a lane census hides it entirely. `host family`
names the CPU host family that serves the class, where one does, which
means a CPU path exists and only the identity lane is missing.

| algorithm | kind | host family | defined in |
|---|---|---|---|
| `HostForest` | class | - | `python/mojolearn/_forest_host.py` |
| `HostGBDT` | class | - | `python/mojolearn/_gbdt_host.py` |
| `host_model` | function | - | `python/mojolearn/_classical_host.py` |
| `host_predict` | function | - | `python/mojolearn/_forest_host.py` |
| `host_predict_proba` | function | - | `python/mojolearn/_forest_host.py` |
| `metrics.homogeneity_completeness_v_measure` | function | - | `python/mojolearn/_metrics_impl.py` |

The saved-model host inference surface (`HostForest`, `HostGBDT`,
`host_model`, `host_predict`, `host_predict_proba`) has no identity_break
lane on purpose. It is measured by `tools/forest_host_gate.py` and
`tools/classical_host_gate.py` against committed recordings instead:
25 under `bench/results/forest_host/` and
12 classical recording directories named in
`host_surface.py`. That is a different gate, not a missing one, but it is
also not one of the four kinds counted here.

### Reached by the harness, with no lane of their own

`tools/identity_break.py` reaches these outside any lane body. `_public_est`
swaps the fitted estimator for the public CPU inference class on a CPU
column, so those answer the infer, reload and batch cells of the lanes named
in `NEURAL_PUBLIC_PART_LANES` (mamba1, mamba2, mamba3, mamba2-dtlimit, samba, samba-untied-dropout-accum, byte-lm, byte-lm-resident, transformer, transformer-window);
the batchgrad part calls the accumulation helpers the same way.
They are verified, but no lane carries their name.

| algorithm | kind | host family | defined in |
|---|---|---|---|
| `MLPInference` | class | neural | `python/mojolearn/neural_inference.py` |
| `Mamba1BlockInference` | class | neural | `python/mojolearn/neural_inference.py` |
| `Mamba2BlockInference` | class | neural | `python/mojolearn/neural_inference.py` |
| `Mamba3BlockInference` | class | neural | `python/mojolearn/neural_inference.py` |
| `SambaInference` | class | neural | `python/mojolearn/neural_inference.py` |
| `TransformerBlockInference` | class | neural | `python/mojolearn/neural_inference.py` |
| `training.accumulate_grads` | function | training | `python/mojolearn/_training_impl.py` |
| `training.accumulation_is_aligned` | function | - | `python/mojolearn/_training_impl.py` |

## The algorithm matrix

`gpu` is the device classes that carry any of the algorithm's lanes.
A blank cell means no lane of this algorithm has that kind.

| algorithm | lanes | gpu | cpu | sabotage | batch | all four |
|---|---|---|---|---|---|---|
| `ARIMA` | 6 | amd,apple,nvidia | training | seen(build) | part | yes |
| `Adam` | 1 | amd,apple,nvidia | training | declared | part | NO |
| `AdamW` | 1 | amd,apple,nvidia | training | declared | part | NO |
| `AgglomerativeClustering` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `ByteLanguageModelConfig` | 5 | amd,apple,nvidia | training | seen(build) | part | yes |
| `Cholesky` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `ConstantKernel` | 5 | amd,apple,nvidia | training | seen(build) | part | yes |
| `DBSCAN` | 4 | amd,apple,nvidia | training | seen(build) | part | yes |
| `ElasticNet` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `Embedding` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `ExperimentalTwoLevelFeatureFreq` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `ExponentialSmoothing` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `ExtraTreesClassifier` | 2 | amd,apple,nvidia | training | declared | part | NO |
| `ExtraTreesRegressor` | 3 | amd,apple,nvidia | training | declared | part | NO |
| `GPT2Tokenizer` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `GaussianMixture` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `GaussianProcessClassifier` | 2 | apple | training | seen(build) | part | yes |
| `GaussianProcessRegressor` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `GradientBoosting` | 18 | amd,apple,nvidia | training | seen(build) | part | yes |
| `GradientBoostingClassifier` | 2 | amd,apple,nvidia | training | declared | part | NO |
| `GradientBoostingRegressor` | 3 | amd,apple,nvidia | training | declared | part | NO |
| `HDBSCAN` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `HostForest` | **0** |  |  |  |  | NO |
| `HostGBDT` | **0** |  |  |  |  | NO |
| `IVFIndex` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `IsolationForest` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `KMeans` | 10 | amd,apple,nvidia | training | seen(build) | part | yes |
| `KNeighborsClassifier` | 4 | amd,apple,nvidia | training | seen(build) | part | yes |
| `KNeighborsRegressor` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `KernelDensity` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `KernelRidge` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `LanguageModelConfig` (alias of `ByteLanguageModelConfig`) | 5 | amd,apple,nvidia | training | seen(build) | part | yes |
| `LanguageModelHostTrainer` | 1 | amd,apple,nvidia | training | declared | n/a | NO |
| `LanguageModelInference` | 2 | amd,apple,nvidia | training | seen(harness) | part | NO |
| `LanguageModelTrainer` (alias of `SmallByteLanguageModelTrainer`) | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `Lasso` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `LinearRegression` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `LogisticRegression` | 6 | amd,apple,nvidia | training | seen(build) | part | yes |
| `MLPInference` | routed |  |  |  |  | NO |
| `Mamba1Block` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `Mamba1BlockInference` | routed |  |  |  |  | NO |
| `Mamba2Block` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `Mamba2BlockInference` | routed |  |  |  |  | NO |
| `Mamba3Block` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `Mamba3BlockInference` | routed |  |  |  |  | NO |
| `Matern` | 1 | apple | training | seen(build) | part | yes |
| `MinMaxScaler` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `NearestNeighbors` | 1 | amd,apple,nvidia | training | declared | part | NO |
| `Nystroem` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `OrderedRMSE` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `PCA` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `RBF` | 4 | amd,apple,nvidia | training | seen(build) | part | yes |
| `RBFSampler` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `RadiusNeighbors` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `RandomForestClassifier` | 6 | amd,apple,nvidia | training | declared | part | NO |
| `RandomForestRegressor` | 4 | amd,apple,nvidia | training | declared | part | NO |
| `Ridge` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `SGD` | 1 | amd,apple,nvidia | training | declared | part | NO |
| `SVC` | 4 | amd,apple,nvidia | training | seen(build) | part | yes |
| `SVR` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `SambaConfig` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `SambaInference` | routed |  |  |  |  | NO |
| `SambaStack` | 4 | amd,apple,nvidia | training | seen(build) | part | yes |
| `SmallByteLanguageModelTrainer` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `SmallMLPTrainer` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `SpectralClustering` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `StandardScaler` | 4 | amd,apple,nvidia | training | seen(build) | part | yes |
| `TransformerBlock` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `TransformerBlockInference` | routed |  |  |  |  | NO |
| `TruncatedSVD` | 1 | amd,apple,nvidia | training | declared | part | NO |
| `UMAP` | 2 | amd,apple,nvidia | training | seen(build) | n/a | yes |
| `WhiteKernel` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `clip_grad_norm_` | 1 | amd,apple,nvidia | training | declared | part | NO |
| `cross_entropy` | 1 | amd,apple,nvidia | training | declared | part | NO |
| `cross_val_score` | 1 | amd,apple,nvidia | training | declared | part | NO |
| `embedding.Embedding` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `hdbscan.HDBSCAN` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `hdbscan.all_points_membership_vectors` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `hdbscan.approximate_predict` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `hdbscan.membership_vector` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `host_model` | **0** |  |  |  |  | NO |
| `host_predict` | **0** |  |  |  |  | NO |
| `host_predict_proba` | **0** |  |  |  |  | NO |
| `kernel_methods.KernelRidge` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `kernel_methods.Nystroem` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `kernel_methods.RBFSampler` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `kpss_test` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `language_model.ByteLanguageModelConfig` | 5 | amd,apple,nvidia | training | seen(build) | part | yes |
| `language_model.LanguageModelConfig` (alias of `ByteLanguageModelConfig`) | 5 | amd,apple,nvidia | training | seen(build) | part | yes |
| `language_model.LanguageModelHostTrainer` | 1 | amd,apple,nvidia | training | declared | n/a | NO |
| `language_model.LanguageModelInference` | 2 | amd,apple,nvidia | training | seen(harness) | part | NO |
| `language_model.LanguageModelTrainer` (alias of `SmallByteLanguageModelTrainer`) | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `language_model.SmallByteLanguageModelTrainer` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `linalg.Cholesky` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `linalg.matmul` | 2 | amd,apple,nvidia | training | declared | part | NO |
| `mamba.Mamba1Block` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `mamba.Mamba2Block` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `mamba.Mamba3Block` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `matmul` | 2 | amd,apple,nvidia | training | declared | part | NO |
| `metrics.accuracy_score` | 1 | amd,apple,nvidia | training | seen(build) | n/a | yes |
| `metrics.adjusted_rand_score` | 1 | amd,apple,nvidia | training | seen(build) | n/a | yes |
| `metrics.completeness_score` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `metrics.confusion_matrix` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `metrics.entropy` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `metrics.f1_score` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `metrics.fowlkes_mallows_score` | 1 |  | training | seen(build) | n/a | NO |
| `metrics.homogeneity_completeness_v_measure` | **0** |  |  |  |  | NO |
| `metrics.homogeneity_score` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
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
| `metrics.v_measure_score` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `mixture.GaussianMixture` | 3 | amd,apple,nvidia | training | seen(build) | part | yes |
| `model_selection.cross_val_score` | 1 | amd,apple,nvidia | training | declared | part | NO |
| `neural_network.SmallMLPTrainer` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `preprocessing.MinMaxScaler` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `preprocessing.StandardScaler` | 4 | amd,apple,nvidia | training | seen(build) | part | yes |
| `resample.bootstrap` | 2 | amd,apple,nvidia | training | declared | part | NO |
| `resample.monte_carlo_integrate` | 2 | amd,apple,nvidia | training | declared | part | NO |
| `resample.permutation_test` | 2 | amd,apple,nvidia | training | declared | part | NO |
| `tokenizer.GPT2Tokenizer` | 1 | amd,apple,nvidia | training | seen(build) | part | yes |
| `training.Adam` | 1 | amd,apple,nvidia | training | declared | part | NO |
| `training.AdamW` | 1 | amd,apple,nvidia | training | declared | part | NO |
| `training.ConstantLR` | 1 | amd,apple,nvidia | training | declared | part | NO |
| `training.Generator` | 4 | amd,apple,nvidia | training | seen(build) | part | yes |
| `training.SGD` | 1 | amd,apple,nvidia | training | declared | part | NO |
| `training.SambaConfig` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `training.SambaStack` | 4 | amd,apple,nvidia | training | seen(build) | part | yes |
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
| `transformer.TransformerBlock` | 2 | amd,apple,nvidia | training | seen(build) | part | yes |
| `umap.UMAP` | 2 | amd,apple,nvidia | training | seen(build) | n/a | yes |

## The lane matrix

| lane | gpu | cpu | sabotage | sabotage evidence | batch | all four |
|---|---|---|---|---|---|---|
| agglomerative | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-14_cpu-phase1b/cpu-apple-m4.sabotage.agglomerative-et.json` | part | yes |
| arima | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-forecast-umap-pca/cpu-apple-m4.sabotage.json` | part | yes |
| arima-011 | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-forecast-umap-pca/cpu-apple-m4.sabotage.json` | part | yes |
| arima-exog | - | training | declared | - | part | NO |
| arima-exog-seasonal | - | training | declared | - | part | NO |
| arima-seasonal-c | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-forecast-umap-pca/cpu-apple-m4.sabotage.json` | part | yes |
| bootstrap | amd,apple,nvidia | training | declared | - | part | NO |
| byte-lm | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_neural-forward-inference/cpu-amd-epyc-4564p.host-sabotage.json` | part | yes |
| byte-lm-host-infer | amd,apple,nvidia | training | seen(harness) | `bench/results/identity_break/2026-09-15_batch2/amd-mi325x-gfx942.sabotage.json` | part | NO |
| byte-lm-host-infer-threaded | amd,apple,nvidia | training | seen(harness) | `bench/results/identity_break/2026-09-15_rlpair/apple-m4.rlpair-sabotage.json` | part | NO |
| byte-lm-host-train | amd,apple,nvidia | training | declared | - | n/a n/a:mean-reduction-fixed-batch (LanguageModelHostTrainer has train_step and loss only, and loss IS a step, python/mojolearn/_byte_lm_host.py:392-429; one loss and one update from mean cross-entropy over ids of the profile's fixed (batch, length + 1), so no output belongs to one sequence; the per-sequence logits are asked by byte-lm-host-infer; lane/batch-invariance-2's batchgrad records the same reason for this lane and its batchscale and ragged parts do not ask it) | NO |
| byte-lm-resident | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_neural-forward-inference/cpu-amd-epyc-4564p.host-sabotage.json` | part | yes |
| cholesky | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cholesky-cpu-inference/cpu-apple-m4.sabotage.json` | part | yes |
| cross-entropy-arms | amd,apple,nvidia | training | declared | - | part | NO |
| cross-val | amd,apple,nvidia | training | declared | - | part | NO |
| dbscan | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_transductive-predict/cpu-x86.predict-sabotage.json` | part | yes |
| dbscan-brute-l1 | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_transductive-predict/cpu-x86.predict-sabotage.json` | part | yes |
| dbscan-weighted | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_transductive-predict/cpu-x86.predict-sabotage.json` | part | yes |
| elasticnet | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-linear-kernel/cpu-apple-m4-sabotage.json` | part | yes |
| elasticnet-l2end-no-intercept | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-linear-kernel/cpu-apple-m4-sabotage.json` | part | yes |
| embedding | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_ivf-embedding-cpu-inference/cpu-x86.sabotage.json` | part | yes |
| embedding-sort | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_ivf-embedding-cpu-inference/cpu-x86.sabotage.json` | part | yes |
| et-clf | amd,apple,nvidia | training | declared | - | part | NO |
| et-clf-entropy-bestfirst | amd,apple,nvidia | training | declared | - | part | NO |
| et-reg | amd,apple,nvidia | training | declared | - | part | NO |
| et-reg-bootstrap-parallel | amd,apple,nvidia | training | declared | - | part | NO |
| gbdt-adapter-clf | amd,apple,nvidia | training | declared | - | part | NO |
| gbdt-adapter-reg | amd,apple,nvidia | training | declared | - | part | NO |
| gbdt-adapter-score-weighted | - | training | declared | - | n/a n/a:scalar-reduction (score(X, y, sample_weight) returns one float over every row it is handed, python/mojolearn/_gbdt_adapters.py and _forest_protocol.py; the predict and predict_proba it wraps keep their batch parts on gbdt-adapter-clf, gbdt-adapter-reg, rf-clf and rf-reg) | NO |
| gbdt-categorical-ctr | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-gbdt-modes/cpu-x86-forest-sabotage-host-infer.json` | part | yes |
| gbdt-categorical-ctr-tables | apple | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-verifier-gaps-7/x86-runpod/cpu-sab.json` | part | yes |
| gbdt-depthwise | amd,apple,nvidia | training | declared | - | part | NO |
| gbdt-exact-mae | amd,apple,nvidia | training | declared | - | part | NO |
| gbdt-feature-freq | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-gbdt-modes/cpu-x86-forest-sabotage-host-infer.json` | part | yes |
| gbdt-lossguide | amd,apple,nvidia | training | declared | - | part | NO |
| gbdt-lossguide-newtoncosine | amd,apple,nvidia | training | declared | - | part | NO |
| gbdt-multiclass | amd,apple,nvidia | training | declared | - | part | NO |
| gbdt-nan-modes | amd,apple,nvidia | training | declared | - | part | NO |
| gbdt-onevsall | amd,apple,nvidia | training | declared | - | part | NO |
| gbdt-ordered-rmse | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-gbdt-modes/cpu-x86-forest-sabotage-host-infer.json` | part | yes |
| gbdt-pair-logit | apple | training | declared | - | part | NO |
| gbdt-parametric-losses | amd,apple,nvidia | training | declared | - | part | NO |
| gbdt-pointwise-l2-bayesian-eval | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-gbdt-modes/cpu-x86-forest-sabotage-host-infer.json` | part | yes |
| gbdt-query-rmse | apple | training | declared | - | part | NO |
| gbdt-rmse | amd,apple,nvidia | training | declared | - | part | NO |
| gbdt-symmetric | amd,apple,nvidia | training | declared | - | part | NO |
| gbdt-tensor-ctr-tables | apple | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-verifier-gaps-7/x86-runpod/cpu-sab.json` | part | yes |
| gbdt-yeti-rank | apple | training | declared | - | part | NO |
| gemm-pinned | amd,apple,nvidia | training | declared | - | part | NO |
| gemm-transposed | amd,apple,nvidia | training | declared | - | part | NO |
| gmm | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-d-estimators/cpu-apple-m4.sabotage.json` | part | yes |
| gmm-random-init | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-d-estimators/cpu-apple-m4.sabotage.json` | part | yes |
| gmm-random-init-sample | apple | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-verifier-gaps-7/x86-runpod/cpu-sab.json` | n/a n/a:no-batch-axis (GaussianMixture.sample takes no input rows) | yes |
| gmm-sample | apple | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-verifier-gaps-7/x86-runpod/cpu-sab.json` | n/a n/a:no-batch-axis (GaussianMixture.sample takes no input rows) | yes |
| gp | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-d-estimators/cpu-apple-m4.sabotage.json` | part | yes |
| gp-matern12 | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-d-estimators/cpu-apple-m4.sabotage.json` | part | yes |
| gp-matern32 | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-d-estimators/cpu-apple-m4.sabotage.json` | part | yes |
| gp-matern52-ard | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-d-estimators/cpu-apple-m4.sabotage.json` | part | yes |
| gp-normalize-y | apple | training | seen(build) | `bench/results/identity_break/2026-09-15_gp-sample-y/cpu-x86-epyc.host-sabotage.json` | part | yes |
| gp-optimize | apple | - | seen(build) | `bench/results/identity_break/2026-09-15_gp-optimize/x86-runpod/cpu_gsab.json` | part | NO |
| gp-optimize-restarts | apple | - | seen(build) | `bench/results/identity_break/2026-09-15_gp-optimize/x86-runpod/cpu_gsab.json` | part | NO |
| gp-sample-y | apple | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-verifier-gaps-7/x86-runpod/cpu-sab.json` | n/a n/a:jointly-correlated (sample_y draws all rows of a call from one joint posterior) | yes |
| gp-sample-y-normalize | apple | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-verifier-gaps-7/x86-runpod/cpu-sab.json` | n/a n/a:jointly-correlated (sample_y draws all rows of a call from one joint posterior) | yes |
| gpc | apple | training | seen(build) | `bench/results/identity_break/2026-09-15_gpc/cpu-x86.host-sabotage.json` | part | yes |
| gpc-multiclass | apple | training | seen(build) | `bench/results/identity_break/2026-09-15_gpc/cpu-x86.host-sabotage.json` | part | yes |
| hdbscan | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-d-estimators/cpu-apple-m4.sabotage.json` | part | yes |
| hdbscan-leaf | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-d-estimators/cpu-apple-m4.sabotage.json` | part | yes |
| holtwinters | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_holtwinters-linesearch-fix/cpu-amd-epyc-9754.sabotage.json` | part | yes |
| holtwinters-multiplicative | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_holtwinters-linesearch-fix/cpu-amd-epyc-9754.sabotage.json` | part | yes |
| iforest | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-iforest-gmm-hdbscan/cpu-x86.iforest-gmm-hdbscan.sabotage.json` | part | yes |
| iforest-tuned | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-iforest-gmm-hdbscan/cpu-x86.iforest-gmm-hdbscan.sabotage.json` | part | yes |
| ivf | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_ivf-embedding-cpu-inference/cpu-x86.sabotage.json` | part | yes |
| ivf-euclidean | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_ivf-embedding-cpu-inference/cpu-x86.sabotage.json` | part | yes |
| ivf-extend | apple | training | seen(build) | `bench/results/identity_break/2026-09-15_ivf-extend/cpu-x86.sabotage.json` | part | yes |
| kde | amd,apple,nvidia | training | declared | - | part | NO |
| kde-cosine-minkowski | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-neighbors-density/cpu-apple-m4.sabotage.json` | part | yes |
| kde-epanechnikov-l1 | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-neighbors-density/cpu-apple-m4.sabotage.json` | part | yes |
| kde-exponential-chebyshev | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-neighbors-density/cpu-apple-m4.sabotage.json` | part | yes |
| kde-linear-cosine | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-neighbors-density/cpu-apple-m4.sabotage.json` | part | yes |
| kde-tophat-sqeuclidean | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-neighbors-density/cpu-apple-m4.sabotage.json` | part | yes |
| kde-weighted | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-neighbors-density/cpu-apple-m4.sabotage.json` | part | yes |
| kernel-ridge | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-d-estimators/cpu-apple-m4.sabotage.json` | part | yes |
| kmeans | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_kmeans-predict/cpu-apple-m4.host-sabotage.json` | part | yes |
| kmeans-array | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_kmeans-predict/cpu-apple-m4.host-sabotage.json` | part | yes |
| kmeans-classic-pp | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_kmeans-predict/cpu-apple-m4.host-sabotage.json` | part | yes |
| kmeans-cosine | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_kmeans-predict/cpu-apple-m4.host-sabotage.json` | n/a n/a:fit-refused | yes |
| kmeans-random | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_kmeans-predict/cpu-apple-m4.host-sabotage.json` | part | yes |
| kmeans-sqrt | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_kmeans-predict/cpu-apple-m4.host-sabotage.json` | part | yes |
| kmeans-weighted | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_kmeans-predict/cpu-apple-m4.host-sabotage.json` | part | yes |
| knn | amd,apple,nvidia | training | declared | - | part | NO |
| knn-chebyshev | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-neighbors-density/cpu-apple-m4.sabotage.json` | part | yes |
| knn-clf | amd,apple,nvidia | training | declared | - | part | NO |
| knn-clf-distance | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-neighbors-density/cpu-apple-m4.sabotage.json` | part | yes |
| knn-cosine | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-neighbors-density/cpu-apple-m4.sabotage.json` | part | yes |
| knn-manhattan | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-neighbors-density/cpu-apple-m4.sabotage.json` | part | yes |
| knn-minkowski-p3 | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-neighbors-density/cpu-apple-m4.sabotage.json` | part | yes |
| knn-rbc | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-neighbors-density/cpu-apple-m4.sabotage.json` | part | yes |
| knn-reg | amd,apple,nvidia | training | declared | - | part | NO |
| knn-reg-distance | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-neighbors-density/cpu-apple-m4.sabotage.json` | part | yes |
| knn-sqeuclidean | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-neighbors-density/cpu-apple-m4.sabotage.json` | part | yes |
| kpss | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_holtwinters-linesearch-fix/cpu-amd-epyc-9754.sabotage.json` | part | yes |
| lasso | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-linear-kernel/cpu-apple-m4-sabotage.json` | part | yes |
| logistic | amd,apple,nvidia | training | declared | - | part | NO |
| logistic-elasticnet | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-linear-kernel/cpu-apple-m4-sabotage.json` | part | yes |
| logistic-l1 | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-linear-kernel/cpu-apple-m4-sabotage.json` | part | yes |
| logistic-multiclass | amd,apple,nvidia | training | declared | - | part | NO |
| logistic-unpenalized-no-intercept | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-linear-kernel/cpu-apple-m4-sabotage.json` | part | yes |
| mamba1 | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-mamba/cpu-apple-m4.sabotage.json` | part | yes |
| mamba2 | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-mamba/cpu-apple-m4.sabotage.json` | part | yes |
| mamba2-dtlimit | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-mamba/cpu-apple-m4.sabotage.json` | part | yes |
| mamba3 | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-mamba/cpu-apple-m4.sabotage.json` | part | yes |
| metrics | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_metrics-sabotage-coverage/cpu-sab-new.json` | n/a n/a:scalar-reduction (accuracy_score, adjusted_rand_score, v_measure_score, r2_score and silhouette_score each return one float over every row, python/mojolearn/_metrics_impl.py; the per-sample silhouette_samples is asked on metrics-classification) | yes |
| metrics-classification | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_metrics-sabotage-coverage/cpu-sab-new.json` | part | yes |
| metrics-fowlkes-mallows | - | training | seen(build) | `bench/results/identity_break/2026-09-15_metrics-sabotage-coverage/cpu-sab-new.json` | n/a n/a:function | NO |
| minmax-scaler | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-linear-kernel/cpu-apple-m4-sabotage.json` | part | yes |
| minmax-scaler-clip | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-linear-kernel/cpu-apple-m4-sabotage.json` | part | yes |
| mlp | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_neural-inference/cpu.neural-sabotage.json` | part | yes |
| monte-carlo | amd,apple,nvidia | training | declared | - | n/a n/a:scalar-fold (resample.monte_carlo_integrate returns only integral, mean, volume and closed_form folded over [i_first, i_first + n_samples) by the pinned chunk tree, python/mojolearn/resample.py:196; no per-sample output exists to compare an i_first range against) | NO |
| nystroem | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-d-estimators/cpu-apple-m4.sabotage.json` | part | yes |
| ols | amd,apple,nvidia | training | seen(build) | `bench/results/runpod_cpu/2026-09-15_proof/cold/leg_out/cpu-x86.host-sabotage.json` | part | yes |
| ols-no-intercept | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-linear-kernel/cpu-apple-m4-sabotage.json` | part | yes |
| ols-weighted | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-linear-kernel/cpu-apple-m4-sabotage.json` | part | yes |
| optim-adam-clip | amd,apple,nvidia | training | declared | - | part | NO |
| optim-sgd | amd,apple,nvidia | training | declared | - | part | NO |
| par-arima | amd,apple,nvidia | training | declared | - | part | NO |
| par-boosting | amd,apple,nvidia | - | none | - | part | NO |
| par-boosting-pointwise | amd,apple,nvidia | - | none | - | part | NO |
| par-byte-lm | amd,apple,nvidia | - | none | - | n/a n/a:driver-step (ParallelByteLanguageModelTrainer and its Pooled and Offloaded subclasses have train_step, state_dict, export_gradients and checkpoint only, python/mojolearn/parallel_training.py:89-172; train_step returns per-shard mean losses and one update from the shard-mean gradient, so no output belongs to one sequence, and the shard split is held to the replica trainer by the train column) | NO |
| par-byte-lm-model-pool | amd,apple,nvidia | - | none | - | n/a n/a:driver-step (ParallelByteLanguageModelTrainer and its Pooled and Offloaded subclasses have train_step, state_dict, export_gradients and checkpoint only, python/mojolearn/parallel_training.py:89-172; train_step returns per-shard mean losses and one update from the shard-mean gradient, so no output belongs to one sequence, and the shard split is held to the replica trainer by the train column) | NO |
| par-byte-lm-offload | amd,apple,nvidia | - | none | - | n/a n/a:driver-step (ParallelByteLanguageModelTrainer and its Pooled and Offloaded subclasses have train_step, state_dict, export_gradients and checkpoint only, python/mojolearn/parallel_training.py:89-172; train_step returns per-shard mean losses and one update from the shard-mean gradient, so no output belongs to one sequence, and the shard split is held to the replica trainer by the train column) | NO |
| par-cd | amd,apple,nvidia | - | none | - | part | NO |
| par-cholesky | amd,nvidia | - | none | - | part | NO |
| par-dbscan | amd,apple,nvidia | - | seen(harness) | `bench/results/identity_break/2026-09-15_transductive-predict/apple-m4.batch-sabotage.json` | part | NO |
| par-feature-freq | amd,apple,nvidia | - | none | - | part | NO |
| par-forest | amd,apple,nvidia | training | declared | - | part | NO |
| par-forest-et | amd,apple,nvidia | training | declared | - | part | NO |
| par-forest-pool | amd,nvidia | - | none | - | part | NO |
| par-gmm | amd,nvidia | - | none | - | part | NO |
| par-gp | amd,apple,nvidia | - | none | - | part | NO |
| par-gram | amd,apple,nvidia | - | none | - | part | NO |
| par-graph-agglomerative | amd,apple,nvidia | - | seen(harness) | `bench/results/identity_break/2026-09-15_transductive-predict/apple-m4.batch-sabotage.json` | part | NO |
| par-graph-spectral | amd,apple,nvidia | - | none | - | n/a n/a:transductive (the par-graph-spectral lane fits without prediction_data; SpectralClustering.predict is the spectral lane's) | NO |
| par-graph-umap | amd,apple,nvidia | - | none | - | n/a n/a:batch-dependent-by-contract (umap/transform.mojo:44,66 batch-mean sigma floor, :141 batch-max edge schedule, :146,154 batch-local RNG edge ordinal; cuML couples the same) | NO |
| par-hdbscan | amd,apple,nvidia | - | none | - | part | NO |
| par-holtwinters | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_holtwinters-linesearch-fix/cpu-amd-epyc-9754.sabotage.json` | part | yes |
| par-iforest | amd,apple,nvidia | - | none | - | part | NO |
| par-kernel-ridge | amd,nvidia | - | none | - | part | NO |
| par-kmeans | amd,apple,nvidia | - | seen(harness) | `bench/results/identity_break/2026-09-15_kmeans-predict/apple-m4.batch-sabotage.json` | part | NO |
| par-logistic | amd,apple,nvidia | - | none | - | part | NO |
| par-mlp | amd,apple,nvidia | training | declared | - | part | NO |
| par-nystroem | amd,nvidia | - | none | - | part | NO |
| par-ordered-rmse | amd,apple,nvidia | - | none | - | part | NO |
| par-queries-kde | amd,apple,nvidia | training | declared | - | part | NO |
| par-queries-knn | amd,apple,nvidia | training | seen(harness) | `bench/results/identity_break/2026-09-14_166-lanes/apple-m4.batch-sabotage.json` | part | NO |
| par-queries-radius | amd,apple,nvidia | training | declared | - | part | NO |
| par-rbf-sampler | amd,nvidia | - | none | - | part | NO |
| par-reference-knn | amd,apple,nvidia | training | declared | - | part | NO |
| par-reference-knn-reg | amd,apple,nvidia | training | declared | - | part | NO |
| par-resample | amd,nvidia | - | none | - | part | NO |
| par-samba | amd,apple,nvidia | - | none | - | part | NO |
| par-samba-clip | amd,apple,nvidia | - | none | - | part | NO |
| par-scaler | amd,apple,nvidia | training | declared | - | part | NO |
| par-svm | amd,apple,nvidia | - | none | - | part | NO |
| pca | amd,apple,nvidia | training | seen(harness) | `bench/results/identity_break/2026-09-15_batch2/amd-mi325x-gfx942.sabotage.json` | part | NO |
| pca-full-whiten | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-forecast-umap-pca/cpu-apple-m4.sabotage.json` | part | yes |
| pca-whiten | amd,apple,nvidia | training | declared | - | part | NO |
| permutation-test | amd,apple,nvidia | training | declared | - | part | NO |
| radius | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-neighbors-density/cpu-apple-m4.sabotage.json` | part | yes |
| radius-chebyshev | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-neighbors-density/cpu-apple-m4.sabotage.json` | part | yes |
| radius-manhattan | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-neighbors-density/cpu-apple-m4.sabotage.json` | part | yes |
| radius-minkowski-p3 | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-neighbors-density/cpu-apple-m4.sabotage.json` | part | yes |
| rbf-sampler | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-d-estimators/cpu-apple-m4.sabotage.json` | part | yes |
| rf-clf | amd,apple,nvidia | training | declared | - | part | NO |
| rf-clf-balanced-parallel | amd,apple,nvidia | training | declared | - | part | NO |
| rf-clf-entropy-log2-noboot | amd,apple,nvidia | training | declared | - | part | NO |
| rf-reg | amd,apple,nvidia | training | declared | - | part | NO |
| rf-reg-gamma-ig | amd,apple,nvidia | training | declared | - | part | NO |
| rf-reg-poisson | amd,apple,nvidia | training | declared | - | part | NO |
| rf-score-weighted | - | training | declared | - | n/a n/a:scalar-reduction (score(X, y, sample_weight) returns one float over every row it is handed, python/mojolearn/_gbdt_adapters.py and _forest_protocol.py; the predict and predict_proba it wraps keep their batch parts on gbdt-adapter-clf, gbdt-adapter-reg, rf-clf and rf-reg) | NO |
| ridge | amd,apple,nvidia | training | seen(build) | `bench/results/runpod_cpu/2026-09-15_proof/cold/leg_out/cpu-x86.host-sabotage.json` | part | yes |
| ridge-no-intercept | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-linear-kernel/cpu-apple-m4-sabotage.json` | part | yes |
| samba | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-samba/cpu-apple-m4.sabotage.json` | part | yes |
| samba-untied-dropout-accum | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-samba/cpu-apple-m4.sabotage.json` | part | yes |
| spectral | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_spectral-predict/cpu-x86.host-sabotage.json (clean partner at another commit)` | part | yes |
| spectral-precomputed | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_spectral-predict/cpu-x86.host-sabotage.json (clean partner at another commit)` | part | yes |
| standard-scaler | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-linear-kernel/cpu-apple-m4-sabotage.json` | part | yes |
| standard-scaler-no-mean | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-linear-kernel/cpu-apple-m4-sabotage.json` | part | yes |
| standard-scaler-no-std | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-linear-kernel/cpu-apple-m4-sabotage.json` | part | yes |
| svc | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-svm/cpu-x86-sabotage.json` | part | yes |
| svc-linear | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-svm/cpu-x86-sabotage.json` | part | yes |
| svc-poly | apple | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-svm/cpu-x86-sabotage.json` | part | yes |
| svr | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-svm/cpu-x86-sabotage.json` | part | yes |
| svr-linear | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-svm/cpu-x86-sabotage.json` | part | yes |
| tokenizer | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_clean-third-party/m4-owed/cpu-apple-m4.host-sabotage.json` | part | yes |
| training-primitives | amd,apple,nvidia | training | declared | - | part | NO |
| transformer | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-transformer/cpu-apple-m4.sabotage.json` | part | yes |
| transformer-window | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_cpu-transformer/cpu-apple-m4.sabotage.json` | part | yes |
| tsvd | amd,apple,nvidia | training | declared | - | part | NO |
| umap | amd,apple,nvidia | training | seen(build) | `bench/results/identity_break/2026-09-15_inference-forecast-umap-pca/cpu-apple-m4.sabotage.json` | n/a n/a:batch-dependent-by-contract (umap/transform.mojo:44,66 batch-mean sigma floor, :141 batch-max edge schedule, :146,154 batch-local RNG edge ordinal; cuML couples the same) | yes |

## Lanes missing each kind, by name

**No GPU column at all: 5**

> arima-exog, arima-exog-seasonal, gbdt-adapter-score-weighted, metrics-fowlkes-mallows, rf-score-weighted

**GPU column on fewer than three classes: 23**

> gbdt-categorical-ctr-tables, gbdt-pair-logit, gbdt-query-rmse, gbdt-tensor-ctr-tables, gbdt-yeti-rank, gmm-random-init-sample, gmm-sample, gp-normalize-y, gp-optimize, gp-optimize-restarts, gp-sample-y, gp-sample-y-normalize, gpc, gpc-multiclass, ivf-extend, par-cholesky, par-forest-pool, par-gmm, par-kernel-ridge, par-nystroem, par-rbf-sampler, par-resample, svc-poly

**No CPU verifier declared: 30**

> gp-optimize, gp-optimize-restarts, par-boosting, par-boosting-pointwise, par-byte-lm, par-byte-lm-model-pool, par-byte-lm-offload, par-cd, par-cholesky, par-dbscan, par-feature-freq, par-forest-pool, par-gmm, par-gp, par-gram, par-graph-agglomerative, par-graph-spectral, par-graph-umap, par-hdbscan, par-iforest, par-kernel-ridge, par-kmeans, par-logistic, par-nystroem, par-ordered-rmse, par-rbf-sampler, par-resample, par-samba, par-samba-clip, par-svm

**Sabotage not seen to move a build: 89**

> arima-exog, arima-exog-seasonal, bootstrap, byte-lm-host-infer, byte-lm-host-infer-threaded, byte-lm-host-train, cross-entropy-arms, cross-val, et-clf, et-clf-entropy-bestfirst, et-reg, et-reg-bootstrap-parallel, gbdt-adapter-clf, gbdt-adapter-reg, gbdt-adapter-score-weighted, gbdt-depthwise, gbdt-exact-mae, gbdt-lossguide, gbdt-lossguide-newtoncosine, gbdt-multiclass, gbdt-nan-modes, gbdt-onevsall, gbdt-pair-logit, gbdt-parametric-losses, gbdt-query-rmse, gbdt-rmse, gbdt-symmetric, gbdt-yeti-rank, gemm-pinned, gemm-transposed, kde, knn, knn-clf, knn-reg, logistic, logistic-multiclass, monte-carlo, optim-adam-clip, optim-sgd, par-arima, par-boosting, par-boosting-pointwise, par-byte-lm, par-byte-lm-model-pool, par-byte-lm-offload, par-cd, par-cholesky, par-dbscan, par-feature-freq, par-forest, par-forest-et, par-forest-pool, par-gmm, par-gp, par-gram, par-graph-agglomerative, par-graph-spectral, par-graph-umap, par-hdbscan, par-iforest, par-kernel-ridge, par-kmeans, par-logistic, par-mlp, par-nystroem, par-ordered-rmse, par-queries-kde, par-queries-knn, par-queries-radius, par-rbf-sampler, par-reference-knn, par-reference-knn-reg, par-resample, par-samba, par-samba-clip, par-scaler, par-svm, pca, pca-whiten, permutation-test, rf-clf, rf-clf-balanced-parallel, rf-clf-entropy-log2-noboot, rf-reg, rf-reg-gamma-ig, rf-reg-poisson, rf-score-weighted, training-primitives, tsvd

**Batch undeclared: 0**

> none

## Public names not counted as algorithms

Listed by name in `NOT_ALGORITHMS` in the tool, so the exclusion can be
argued with rather than hidden in a heuristic. These are process metadata,
tier switches, result containers, option lists and the caller-owned state
containers a block returns from `allocate_state`, whose buffers are hashed
through their block's own lanes.

> `Array`, `Mamba1State`, `Mamba2State`, `Mamba3State`, `TransformerState`, `__version__`, `gpu_arch`, `gpu_arch_how`, `linalg.PROFILE`, `linalg.PROFILE_FAMILY`, `linalg.PROFILE_VERSION`, `linalg.numeric_mode`, `linalg.profile`, `linalg.require_identical`, `mamba.Mamba1State`, `mamba.Mamba2State`, `mamba.Mamba3State`, `numeric_mode`, `resample.ALTERNATIVES`, `resample.BootstrapResult`, `resample.INTEGRANDS`, `resample.METHODS`, `resample.MonteCarloResult`, `resample.PermutationTestResult`, `resample.STATISTICS`, `select_d`, `set_numeric_mode`, `training.numeric_mode_used`, `training.vendor_used`, `transformer.TransformerState`, `vendor`

