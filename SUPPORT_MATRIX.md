# Support matrix

This page states what mojolearn supports: platforms, Python versions, devices,
numeric modes, the public API by family and what runs on a CPU. Release notes
are in [CHANGELOG.md](CHANGELOG.md), and `mojolearn doctor` reports what the
installed wheel supports on the machine it runs on.

## Platforms and wheels

| Wheel | GPU backend | GPU architectures carried | CPU |
|---|---|---|---|
| macOS arm64 | Apple Metal | Apple silicon (built at the M1 ISA floor) | Apple silicon host |
| Linux x86-64 | NVIDIA CUDA and AMD HIP in one wheel | CUDA `sm_89` and `sm_90a`, HIP `gfx942` | x86-64 host |

Both wheels carry every CPU host binding listed under
[The CPU surface](#the-cpu-surface), so both train and predict on a machine
with no GPU. Device code is architecture specific. On NVIDIA a device loads
the exact architecture the install carries, or the highest carried
architecture of the same family that does not exceed it. On AMD the match
must be exact. A GPU architecture the wheel does not carry runs from a source
build for that architecture.

## Python versions

CPython 3.10, 3.11, 3.12, 3.13 and 3.14 on both wheels.

## Verified devices

| Vendor | Device | Architecture | Backend |
|---|---|---|---|
| Apple | M4 | Apple silicon | Metal |
| NVIDIA | H100 | `sm_90a` | CUDA |
| NVIDIA | A100 | `sm_80` | CUDA |
| NVIDIA | RTX 4090 | `sm_89` | CUDA |
| AMD | MI325X | `gfx942` | HIP |
| AMD | MI300X | `gfx942` | HIP |
| CPU | x86-64 (Linux) | x86-64 | host bindings |
| CPU | Arm (Linux ARM64, Apple silicon macOS) | arm64 | host bindings |

## Numeric modes

| Mode | Contract |
|---|---|
| `identical` | The default. For a certified fixture and configuration, the raw result bits match across Apple Metal, NVIDIA CUDA, AMD HIP and the CPU host bindings. |
| `deterministic` | Repeated execution on the same device and build returns the same bits. No cross-vendor promise. |
| `fast` | Performance oriented. No repeatability or cross-vendor promise. |

| Family | `identical` | `deterministic` | `fast` |
|---|---|---|---|
| Gradient boosting | yes | yes | yes |
| Random Forest | yes | yes | yes |
| Extra Trees | yes | yes | yes |
| Every other family | yes | no | no |

Select a process mode with `mojolearn.set_numeric_mode(...)` or the
`MOJOLEARN_NUMERIC_MODE` environment variable. Estimators that accept
`numeric_mode=` override the process mode per call. Asking an
`identical`-only family for another mode raises a named error rather than
returning something weaker.

`identical` is a claim about certified fixtures and configurations, not about
arbitrary shapes, parameters, hardware, drivers or future builds. The enforced
identity paths are listed in the [identity-path ledger](IDENTITY_PATHS.md).

## Public API by family

Every name below is exported from `mojolearn` or one of its submodules.

| Family | Public API |
|---|---|
| Gradient boosting | `GradientBoosting`, `GradientBoostingClassifier`, `GradientBoostingRegressor`, `OrderedRMSE`, `ExperimentalTwoLevelFeatureFreq` |
| Random forests and Extra Trees | `RandomForestClassifier`, `RandomForestRegressor`, `ExtraTreesClassifier`, `ExtraTreesRegressor`, `IsolationForest` |
| Saved-model tree inference | `HostForest`, `HostGBDT`, `host_model`, `host_predict`, `host_predict_proba` |
| Neighbors | `NearestNeighbors`, `KNeighborsClassifier`, `KNeighborsRegressor`, `RadiusNeighbors` |
| Clustering | `KMeans` (including `metric=` and `oversampling_factor=`), `DBSCAN`, `AgglomerativeClustering`, `SpectralClustering`, `HDBSCAN`, `hdbscan.approximate_predict`, `hdbscan.membership_vector`, `hdbscan.all_points_membership_vectors` |
| Mixture models | `GaussianMixture` |
| Linear models | `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`, `LogisticRegression`, `QNRegressor` |
| Support vector machines | `SVC`, `SVR`, `LinearSVC`, `LinearSVR` |
| Kernel methods | `KernelRidge`, `Nystroem`, `RBFSampler` |
| Gaussian processes | `GaussianProcessRegressor`, `GaussianProcessClassifier`, kernels `RBF`, `Matern`, `ConstantKernel`, `WhiteKernel` |
| Decomposition and manifolds | `PCA`, `TruncatedSVD`, `UMAP`, `SpectralEmbedding`, `manifold.spectral_embedding` |
| Density | `KernelDensity` |
| Preprocessing | `StandardScaler`, `MinMaxScaler` |
| Linear algebra | `Cholesky`, `matmul`, `linalg`, `lowbit` (bf16 and int8 weight storage) |
| Nearest-neighbor search index | `IVFIndex`, `DistributedIVFIndex` |
| Resampling | `resample.bootstrap`, `resample.permutation_test`, `resample.monte_carlo_integrate` |
| Time series | `ARIMA`, `ExponentialSmoothing`, `kpss_test`, `select_d` |
| Metrics | `metrics` (classification, ranking, regression and clustering scores, `silhouette_score`, `trustworthiness` and more) |
| Model selection | `model_selection.cross_val_score` |
| Neural building blocks | `Embedding`, `TransformerBlock`, `Mamba1Block`, `Mamba2Block`, `Mamba3Block`, `SambaStack`, `SambaConfig`, with their state classes |
| Neural inference | `MLPInference`, `TransformerBlockInference`, `Mamba1BlockInference`, `Mamba2BlockInference`, `Mamba3BlockInference`, `SambaInference` |
| Training | `SmallMLPTrainer`, `SGD`, `Adam`, `AdamW`, `clip_grad_norm_`, `cross_entropy`, and the `training` primitives `embedding_forward`, `embedding_backward`, `rms_norm_forward`, `rms_norm_backward`, `linear_forward`, `linear_backward` |
| Language models | `LanguageModelTrainer`, `LanguageModelConfig`, `LanguageModelInference`, `LanguageModelHostTrainer`, `SmallByteLanguageModelTrainer`, `ByteLanguageModelConfig`, `ParallelByteLanguageModelTrainer`, `PooledByteLanguageModelTrainer`, `OffloadedByteLanguageModelTrainer` |
| Tokenizers | `BpeTokenizer` (the user supplies the vocabulary), `lm_corpus` |
| Multi-GPU | `ParallelNeuralTrainer`, `parallel_forecasting`, `parallel_gaussian_process`, `parallel_ivf`, `parallel_model_selection` |
| Cross-vendor training (experimental) | `mojolearn.cross_vendor` |
| Runtime | `numeric_mode`, `set_numeric_mode`, `vendor`, `gpu_arch` |

Querywise and pairwise gradient boosting losses and ranking metrics that are
not implemented are listed in [gbdt/NOT_IMPLEMENTED.tsv](gbdt/NOT_IMPLEMENTED.tsv).
Ordered multi-GPU training is described in
[docs/multi_gpu/README.md](docs/multi_gpu/README.md), and live training across
GPUs from different vendors in
[docs/CROSS_VENDOR_TRAINING.md](docs/CROSS_VENDOR_TRAINING.md).

## The CPU surface

A machine with a supported GPU trains on it. On a CPU-only install `fit` trains
on the CPU for every estimator with a CPU binding and refuses by name
otherwise, and saved models predict on the CPU. Every CPU binding runs in
`identical` mode, is checked bit for bit against the Apple, NVIDIA and AMD
columns with a sabotage build required to fail, and ships in both wheels under
`mojolearn/host/`. `python -m mojolearn verify --all` checks them on the
user's machine against the recorded GPU columns ([docs/VERIFY.md](docs/VERIFY.md)).
The surface is declared once in `python/mojolearn/host_surface.py`.

**Inference from a saved model.** <!--fact:host_inference_surfaces-->random forests, Extra Trees and eight gradient boosting variants; nearest neighbors on every metric and the ball cover, k-NN classification and k-NN regression with either weighting, radius neighbors and k-means assignment and distances; linear regression, ridge, truncated SVD, logistic regression, PCA with and without whitening (either solver), kernel density on every kernel, metric and weighting, the standard and min-max scalers, lasso, elasticnet, kernel ridge, the Nystroem approximation and random Fourier features, linear SVC and SVR and quasi-Newton regression on the squared and absolute losses; UMAP transform of a saved embedding (the GPU's bytes for a row, whatever else is asked in the same batch); SVC and the isolation forest; the Gaussian mixture's scores, probabilities, labels and samples; the Gaussian process regressor's predictive mean and std, normalized targets included, and the Gaussian process classifier's labels and probabilities; HDBSCAN's approximate_predict, membership_vector and all_points_membership_vectors; Embedding lookup in a saved table; IVF-Flat search over a saved index and extending it; batched ARIMA prediction, in sample and out of sample, and forecasts, with or without exogenous regressors, and Holt-Winters forecasts and in-sample one-step predictions, additive and multiplicative<!--/fact-->.

**Training.** <!--fact:host_training_lanes-->kernel ridge poly kernel variant, kernel ridge sigmoid kernel variant, kernel ridge laplacian kernel variant, nystroem poly kernel variant, nystroem sigmoid kernel variant, nystroem laplacian kernel variant, the row-sharded random Fourier feature transform, ordinary differencing order selection, pinned GEMM, kernel density, Holt-Winters, lasso, elasticnet, SVC, agglomerative clustering, the Extra Trees classifier, the Extra Trees regressor, the isolation forest, nearest neighbors, the k-NN classifier, the k-NN regressor, PCA, whitened PCA, truncated SVD, linear regression, ridge, DBSCAN, k-means, the metrics, spectral clustering, the standard scaler, the min-max scaler, logistic regression, the random forest classifier, the random forest regressor, k-means with a random start, k-means from given centroids, weighted k-means, the standard scaler without centering, the standard scaler without scaling, the clipped min-max scaler, spectral clustering on a precomputed affinity, spectral embedding (Laplacian eigenmaps), the Gaussian process with an RBF kernel, the Gaussian process with a Matern kernel at nu 0.5, the Gaussian process with a Matern kernel at nu 1.5, the Gaussian process with an ARD Matern kernel at nu 2.5, the Gaussian process with normalized targets, Gaussian process hyperparameter optimization, Gaussian process hyperparameter optimization with restarts, the binary Gaussian process classifier, the one-vs-rest Gaussian process classifier, the class-sharded Gaussian process classifier fit, the class-sharded Gaussian process classifier prediction, the resident Mamba-1 decode session, the resident Transformer decode session, the layer-owned causal language model, the fold-dispatched cross-validation, nearest neighbors under squared euclidean distance, the distance-weighted k-NN classifier, the distance-weighted k-NN regressor, the transposed GEMM ops, the Householder QR's R factor, both slice arms, the symmetric Jacobi eigendecomposition, ascending, the singular values, descending, brute-force DBSCAN under manhattan distance, kernel density with the tophat kernel under squared euclidean distance, kernel density with the Epanechnikov kernel under manhattan distance, kernel density with the exponential kernel under chebyshev distance, kernel density with the linear kernel under cosine distance, kernel density with the cosine kernel under minkowski distance, weighted kernel density, linear regression without an intercept, weighted linear regression, ridge without an intercept, unpenalized logistic regression without an intercept, elasticnet at the l2 end without an intercept, multiplicative Holt-Winters, the linear SVC, the polynomial SVC, the tuned isolation forest, nearest neighbors under manhattan distance, nearest neighbors under chebyshev distance, nearest neighbors under cosine distance, nearest neighbors under minkowski distance at p 3, nearest neighbors over the random ball cover, radius neighbors, radius neighbors under manhattan distance, radius neighbors under chebyshev distance, radius neighbors under minkowski distance at p 3, weighted DBSCAN, l1-penalized logistic regression, elasticnet-penalized logistic regression, multiclass logistic regression, linear SVC on the hinge loss, linear SVC on the squared hinge loss, linear SVR on the epsilon-insensitive loss, linear SVR on the squared epsilon-insensitive loss, quasi-Newton regression on the squared loss, quasi-Newton regression on the absolute loss, the KPSS stationarity test, SVR, the linear SVR, whitened PCA through the full SVD, the classification, ranking and regression metrics, the Fowlkes-Mallows index, the combined homogeneity, completeness and V-measure scores, the weighted scores of the gradient boosting classifier and regressor, the weighted scores of the random forest classifier and regressor, the small MLP, gradient boosting on symmetric trees with the Logloss loss, gradient boosting on symmetric trees with the RMSE loss, gradient boosting on depthwise trees with the Logloss loss, gradient boosting on lossguide trees with the Logloss loss, gradient boosting with the Min and Max NaN modes, the gradient boosting classifier, the gradient boosting regressor, gradient boosting with the Quantile, MAE, LogLinQuantile, MAPE, Poisson, Lq, Expectile, Tweedie, Huber and CrossEntropy losses, gradient boosting with Exact leaves and the Poisson bootstrap, gradient boosting on lossguide trees with the NewtonCosine score and the searcher options, multiclass gradient boosting, multiclass gradient boosting with public stochastic defaults, one-vs-all gradient boosting, ordered boosting with the RMSE loss (OrderedRMSE), gradient boosting with the six non-default feature border types, ordered boosting (boosting_type='Ordered') with the Logloss and RMSE losses, ordered boosting with the Bayesian bootstrap and score noise, gradient boosting on columns with exactly one border (binary flags), Plain and Ordered, boost from average on the MAE, Quantile and MAPE losses, gradient boosting at CatBoost's GPU defaults (auto learning rate, Bayesian bootstrap, score noise), gradient boosting with the bootstraps and the score noise on the RMSE loss and on Depthwise and Lossguide trees, gradient boosting with an eval set, the overfitting detector and best-model truncation, the two-level FeatureFreq estimator, gradient boosting with the pointwise searcher, L2 scores, the Bayesian bootstrap and an eval set, gradient boosting with one-hot categorical columns, gradient boosting ranking losses with default Bayesian bootstrap and score noise, gradient boosting with the QueryRMSE ranking loss on query groups, gradient boosting with the PairLogit ranking loss on generated and explicit pairs, gradient boosting with the YetiRank ranking loss on query groups, ARIMA, differenced ARIMA, seasonal ARIMA, ARIMA with exogenous regressors, differenced seasonal ARIMA with exogenous regressors, UMAP, k-means under the rooted euclidean metric, k-means from the classic k-means++ start, cross-validation of gradient boosting, the bootstrap, the permutation test, Monte Carlo integration, SGD with momentum, Nesterov and dampening, Adam and AdamW with the gradient clip and accumulation, the cross-entropy loss arms, the embedding, RMSNorm and linear training primitives, the ordered shard gradient reduction, the Cholesky factorization and solve, random Fourier features, kernel ridge, the Nystroem kernel approximation, the Gaussian mixture, the Gaussian mixture with a random start, HDBSCAN, HDBSCAN with leaf selection, the random forest classifier with entropy splits, log2 features and no bootstrap, the class-weighted random forest classifier with the parallel groves engine, the random forest regressor with the Poisson criterion, the random forest regressor with the gamma and inverse Gaussian criteria, the best-first Extra Trees classifier with entropy splits, the bootstrapped Extra Trees regressor with the parallel groves engine, the Mamba-2 block, the Mamba-2 block with an active dt clamp, the Mamba-1 block, the Mamba-3 block, the Transformer block, the sliding-window Transformer block, the Samba stack, the Samba stack with untied embeddings, dropout, accumulation, clipping and a cosine schedule, the byte LM forward pass on its reference path (inference), the byte LM forward pass on its threaded path (inference), the published byte LM host training step, the byte LM shape object at two non-default shapes, the column-sharded standard scaler, the column-sharded min-max scaler, series-sharded ARIMA, series-sharded Holt-Winters, the series-sharded ARIMA prediction and forecast drivers, the series-sharded Holt-Winters prediction and forecast drivers, query-sharded k-NN classification, query-sharded nearest-neighbor distances and indices, query-sharded radius neighbors, query-sharded kernel density, reference-sharded k-NN classification, reference-sharded k-NN regression, the tree-range-sharded random forest classifier, the tree-range-sharded Extra Trees regressor, the tree-range-sharded Extra Trees classifier, the tree-range-sharded random forest regressor, the small MLP trained over ordered logical gradient shards, the Samba stack trained over ordered logical gradient shards, the Samba stack trained over ordered logical gradient shards under a global norm clip, the Embedding layer, the Embedding layer on its sorted execution plan, the IVF-Flat index, the IVF-Flat index under euclidean distance, the shard-distributed IVF-Flat index, extending a built IVF-Flat index, the byte LM trainer, the byte LM trainer on its resident session, samples from the Gaussian mixture, samples from the Gaussian mixture with a random start, posterior draws from the Gaussian process, posterior draws from the Gaussian process with normalized targets, the byte-level BPE tokenizer (inference, host integers), byte-level BPE vocabulary training, a trained BPE vocabulary written, loaded back and used, a corpus tokenized once, cached and read back as batches, the Hugging Face checkpoint reader and the option matrix, the Hugging Face byte-level BPE tokenizer (three pre-tokenization patterns), a Hugging Face causal language model loaded and run, predictions of Metal-saved gradient boosting models with CTR tables (inference), predictions of Metal-saved gradient boosting models with tensor CTRs (inference), the bf16-storage GEMM profile, the int8 GEMM profile with power-of-two scales, the Transformer block with bf16-stored weights, the Transformer block with int8-stored weights, the Mamba-1 block with bf16-stored weights, the Mamba-1 block with int8-stored weights, the Mamba-2 block with bf16-stored weights, the Mamba-2 block with int8-stored weights, the Mamba-3 block with bf16-stored weights, the Mamba-3 block with int8-stored weights, the small MLP with bf16-stored weights, the small MLP with int8-stored weights, the Samba stack with bf16-stored weights, the Samba stack with int8-stored weights, saved forest and gradient boosting models predicted on the CPU (inference), the bf16 and int8 weight-storage conversions and gradient accumulation across microbatches<!--/fact-->.

**No CPU path.** The gradient boosting training configurations below have no
CPU path, and `fit` refuses each by name on a CPU-only install.
<!--fact:no_cpu_path-->(1) gradient boosting training with sample weights on any arm (`gbdt_fit` refuses `sample_weight`, and `class_weights` outside MultiClass and MultiClassOneVsAll, which reach the device through the same per-row weight column): the device's weighted target, histogram and partition-reduce kernels are a second launch arm (`has_weights`) and the gbdt/host oracles restate the unit-weight arm only; (2) gradient boosting training on a CTR categorical column, a `cat_features` column with more than `one_hot_max_size` categories: the CTR calcers build ordered target statistics over several permutations, with their online counters, grids and tables joined back into the compressed index, and the host path is pinned to one permutation with no calcer. One-hot categorical columns DO train, `ExperimentalTwoLevelFeatureFreq` has its own CPU route (gbdt/host/gbdt_oracle_feature_freq.mojo) except on a tree whose level winner is the tensor column itself, and CTR INFERENCE from a saved model is closed; it is the calcer tables' training that is not; (3) gradient boosting training with a categorical or one-hot column outside SymmetricTree with Logloss and Plain boosting: the one-hot grid and the `take_bin` equality split are restated in the symmetric searcher alone; (4) gradient boosting training with an eval set or the overfitting detector outside SymmetricTree with Logloss, Ordered boosting and the pointwise searcher's own lane: the held-out curve runs THAT arm's loss kernel (the multilogit and one-vs-all launches, `launch_approximate` at each pointwise objective) and the non-symmetric shapes put a tree on the cursor through a different apply, and neither is restated; (5) gradient boosting training with the pointwise searcher outside the gbdt-pointwise-l2-bayesian-eval configuration (L2 scores, the Bayesian bootstrap, Newton leaves, sample weights, an eval set with the Iter detector, boost_from_average on, GreedyLogSum borders, numeric columns): every other option selects a different launch shape of the pointwise kernels and one shape is restated; (6) gradient boosting training at a (loss, grow_policy, score_function, leaf_estimation_method, bootstrap_type) combination outside the ones the gbdt/host oracles restate, each refused by name: pointwise losses under Depthwise and Lossguide, score functions and leaf estimators outside each policy's covered pair, Depthwise's min_split_gain, min_child_hessian and min_data_in_leaf, feature_fraction outside Logloss, boost_from_average outside RMSE and the quantile family, and a NaN in X outside SymmetricTree with Logloss -- each one its own device kernel or its own searcher gate order<!--/fact-->.

**Byte-level language model.** `LanguageModelInference` runs the forward pass
([docs/BYTE_LM_CPU_INFERENCE.md](docs/BYTE_LM_CPU_INFERENCE.md)) and
`LanguageModelHostTrainer` runs one training step, forward, backward and the
AdamW update, reproducing the recorded GPU bytes
([docs/BYTE_LM_CPU_TRAINING.md](docs/BYTE_LM_CPU_TRAINING.md)). Identity is
claimed per model profile and batch shape. Saved forest and gradient boosting
engines are described in
[docs/FOREST_INFERENCE_ENGINES.md](docs/FOREST_INFERENCE_ENGINES.md).

The table below is generated from `python/mojolearn/host_surface.py` by
`tools/docs_facts.py --write`, and `pixi run check-docs-facts` fails when the
table and the manifest disagree. Every binding refuses by name any function it
does not list.

<!--fact:host_surface_table-->| family | binding under `mojolearn/host/` | routes (CPU-only install) | internal CPU reference lanes | predicts on a CPU from a saved model | gate | in a wheel |
|---|---|---|---|---|---|---|
| byte_lm | `_mojolearn_byte_lm_host.so` | loaded by path | byte-lm-host-infer, byte-lm-host-infer-threaded, byte-lm-host-train, byte-lm, byte-lm-resident, language-model-config | LanguageModelInference, LanguageModelHostTrainer, SmallByteLanguageModelTrainer | .github/workflows/byte-lm-cpu-gate.yml and tools/identity_break.py (cpu-identity-gate.yml) | yes |
| forest | `_mojolearn_forest_host.so` | loaded by path | gbdt-categorical-ctr-tables, gbdt-tensor-ctr-tables, saved-model-host-infer | RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor, GradientBoosting, OrderedRMSE, ExperimentalTwoLevelFeatureFreq (rf_classifier, rf_regressor, et_classifier, et_regressor, gbdt_symmetric, gbdt_depthwise, gbdt_lossguide, gbdt_rmse, gbdt_ordered_rmse, gbdt_feature_freq, gbdt_pointwise_bayesian_eval, gbdt_categorical_onehot) | tools/forest_host_gate.py (.github/workflows/forest-host-gate.yml) | yes |
| tokenizer | `_mojolearn_tokenizer_host.so` | loaded by path | tokenizer, bpe-trainer, bpe-vocabulary, tokenized-corpus, hf-tokenizer | BpeTokenizer | pixi run check-tokenizer and python/mojolearn/tests/test_tokenizer_surface.py | yes |
| neural | `_mojolearn_neural_host.so` | loaded by path | hf-causal-lm, par-causal-lm | MLPInference, TransformerBlockInference, Mamba1BlockInference, Mamba2BlockInference, Mamba3BlockInference, SambaInference | python/mojolearn/tests/test_neural_inference.py, tools/step_vs_full_check.py and tools/identity_break.py (mlp, transformer, transformer-window, transformer-decode, mamba1, mamba2, mamba3, mamba1-decode, mamba2-decode, mamba3-decode, mamba2-dtlimit, samba, samba-decode, samba-untied-dropout-accum) | yes |
| core | `_mojolearn_core_host.so` | `_mojolearn` | knn, knn-clf, knn-reg, kmeans, kmeans-random, kmeans-array, kmeans-weighted, knn-sqeuclidean, knn-clf-distance, knn-reg-distance, knn-manhattan, knn-chebyshev, knn-cosine, knn-minkowski-p3, knn-rbc, radius, radius-manhattan, radius-chebyshev, radius-minkowski-p3, kmeans-sqrt, kmeans-classic-pp, par-queries-knn, par-queries-nn, par-queries-radius, par-reference-knn, par-reference-knn-reg | NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor, KMeans, RadiusNeighbors (knn, knn-clf, knn-reg, knn-sqeuclidean, knn-manhattan, knn-chebyshev, knn-cosine, knn-minkowski-p3, knn-rbc, knn-clf-distance, knn-reg-distance, radius, radius-manhattan, radius-chebyshev, radius-minkowski-p3, kmeans, kmeans-random, kmeans-array, kmeans-weighted, kmeans-sqrt, kmeans-classic-pp) | tools/classical_host_gate.py (cpu-identity-gate.yml) | yes |
| linalg | `_mojolearn_linalg_host.so` | `_mojolearn_linalg` | gemm-pinned, gemm-transposed, cholesky, gemm-bf16, gemm-int8, linalg-qr, linalg-eigh, linalg-svdvals, lowbit-conversions, hf-checkpoint | no | tools/identity_break.py (cpu-identity-gate.yml) | yes |
| estimators | `_mojolearn_estimators_host.so` | `_mojolearn_estimators` | kde, pca, pca-whiten, tsvd, ols, ridge, dbscan, logistic, dbscan-brute-l1, kde-tophat-sqeuclidean, kde-epanechnikov-l1, kde-exponential-chebyshev, kde-linear-cosine, kde-cosine-minkowski, kde-weighted, ols-no-intercept, ols-weighted, ridge-no-intercept, logistic-unpenalized-no-intercept, dbscan-weighted, logistic-l1, logistic-elasticnet, logistic-multiclass, pca-full-whiten, par-queries-kde, linear-svc, linear-svc-squared-hinge, linear-svr, linear-svr-squared, qn-squared, qn-absolute | LinearRegression, Ridge, TruncatedSVD, LogisticRegression, PCA, KernelDensity, DBSCAN, StandardScaler, MinMaxScaler, Lasso, ElasticNet, KernelRidge, Nystroem, RBFSampler, AgglomerativeClustering, LinearSVC, LinearSVR, QNRegressor (ols, ridge, tsvd, logistic, logistic-multiclass, pca, pca-whiten, kde, ols-no-intercept, ols-weighted, ridge-no-intercept, logistic-l1, logistic-elasticnet, logistic-unpenalized-no-intercept, standard-scaler, standard-scaler-no-mean, standard-scaler-no-std, minmax-scaler, minmax-scaler-clip, lasso, elasticnet, elasticnet-l2end-no-intercept, kernel-ridge, nystroem, rbf-sampler, pca-full-whiten, kernel-ridge-poly, kernel-ridge-sigmoid, kernel-ridge-laplacian, nystroem-poly, nystroem-sigmoid, nystroem-laplacian, kde-tophat-sqeuclidean, kde-epanechnikov-l1, kde-exponential-chebyshev, kde-linear-cosine, kde-cosine-minkowski, kde-weighted, dbscan, agglomerative) | tools/identity_break.py and tools/classical_host_gate.py (cpu-identity-gate.yml) | yes |
| metrics | `_mojolearn_metrics_host.so` | `_mojolearn_metrics` | metrics, spectral, spectral-precomputed, umap, metrics-classification, metrics-fowlkes-mallows, metrics-homogeneity-completeness, spectral-embedding | SpectralClustering, SpectralEmbedding, manifold.spectral_embedding, UMAP, metrics.accuracy_score, metrics.adjusted_rand_score, metrics.entropy, metrics.mutual_info_score, metrics.homogeneity_score, metrics.completeness_score, metrics.v_measure_score, metrics.r2_score, metrics.silhouette_score, metrics.silhouette_samples, metrics.rand_score, metrics.precision_score, metrics.recall_score, metrics.f1_score, metrics.log_loss, metrics.roc_auc_score, metrics.confusion_matrix, metrics.precision_recall_curve, metrics.mean_squared_error, metrics.mean_absolute_error, metrics.root_mean_squared_error, metrics.kl_divergence, metrics.trustworthiness, metrics.fowlkes_mallows_score (umap, spectral, spectral-precomputed) | tools/identity_break.py (cpu-identity-gate.yml) | yes |
| preprocessing | `_mojolearn_preprocessing_host.so` | `_mojolearn_preprocessing` | standard-scaler, minmax-scaler, standard-scaler-no-mean, standard-scaler-no-std, minmax-scaler-clip, par-scaler, par-scaler-minmax | no | tools/identity_break.py (cpu-identity-gate.yml) | yes |
| tsa | `_mojolearn_tsa_host.so` | `_mojolearn_tsa` | holtwinters, holtwinters-multiplicative, kpss, select-d, par-holtwinters, par-forecast-holtwinters | no | tools/identity_break.py (cpu-identity-gate.yml) | yes |
| solver | `_mojolearn_solver_host.so` | `_mojolearn_solver` | lasso, elasticnet, agglomerative, elasticnet-l2end-no-intercept | no | tools/identity_break.py (cpu-identity-gate.yml) | yes |
| svm | `_mojolearn_svm_host.so` | `_mojolearn_svm` | svc, iforest, svc-linear, iforest-tuned, svr, svr-linear, svc-poly | SVC, IsolationForest, SVR (svc, svc-linear, svc-poly, svr, svr-linear, iforest, iforest-tuned) | tools/identity_break.py and tools/classical_host_gate.py (cpu-identity-gate.yml) | yes |
| trees | `_mojolearn_trees_host.so` | `_mojolearn_trees` | et-clf, et-reg, et-clf-entropy-bestfirst, et-reg-bootstrap-parallel, par-forest-et, par-forest-et-clf | no | tools/identity_break.py (cpu-identity-gate.yml) | yes |
| rf | `_mojolearn_rf_host.so` | `_mojolearn_rf` | rf-clf, rf-reg, rf-clf-entropy-log2-noboot, rf-clf-balanced-parallel, rf-reg-poisson, rf-reg-gamma-ig, par-forest, par-forest-reg, rf-score-weighted | no | tools/identity_break.py (cpu-identity-gate.yml) | yes |
| gp | `_mojolearn_gp_host.so` | `_mojolearn_gp` | gp, gp-matern12, gp-matern32, gp-matern52-ard, gp-normalize-y, gpc, gpc-multiclass, gp-sample-y, gp-sample-y-normalize, gp-optimize, gp-optimize-restarts, par-gpc-fit, par-gpc-predict | no | tools/identity_break.py (cpu-identity-gate.yml) | yes |
| kernel_methods | `_mojolearn_kernel_methods_host.so` | `_mojolearn_kernel_methods` | rbf-sampler, kernel-ridge, nystroem, par-rbf-sampler, kernel-ridge-poly, kernel-ridge-sigmoid, kernel-ridge-laplacian, nystroem-poly, nystroem-sigmoid, nystroem-laplacian | no | tools/identity_break.py (cpu-identity-gate.yml) | yes |
| mixture | `_mojolearn_mixture_host.so` | `_mojolearn_mixture` | gmm, gmm-random-init, gmm-sample, gmm-random-init-sample | no | tools/identity_break.py (cpu-identity-gate.yml) | yes |
| mixture_infer | `_mojolearn_mixture_infer_host.so` | loaded by path | no | GaussianMixture (gmm, gmm-random-init, gmm-sample, gmm-random-init-sample) | tools/classical_host_gate.py | yes |
| hdbscan | `_mojolearn_hdbscan_host.so` | `_mojolearn_hdbscan` | hdbscan, hdbscan-leaf | no | tools/identity_break.py (cpu-identity-gate.yml) | yes |
| gp_infer | `_mojolearn_gp_infer_host.so` | loaded by path | no | GaussianProcessRegressor, GaussianProcessClassifier (gp, gp-matern12, gp-matern32, gp-matern52-ard, gp-normalize-y, gpc, gpc-multiclass) | tools/classical_host_gate.py | yes |
| hdbscan_infer | `_mojolearn_hdbscan_infer_host.so` | loaded by path | no | hdbscan.approximate_predict, hdbscan.membership_vector, hdbscan.all_points_membership_vectors (hdbscan, hdbscan-leaf) | tools/classical_host_gate.py | yes |
| gbdt | `_mojolearn_gbdt_host.so` | `_mojolearn_gbdt` | gbdt-symmetric, gbdt-rmse, gbdt-depthwise, gbdt-lossguide, cross-val, par-cross-val, gbdt-nan-modes, gbdt-adapter-clf, gbdt-adapter-reg, gbdt-parametric-losses, gbdt-exact-mae, gbdt-lossguide-newtoncosine, gbdt-multiclass, gbdt-onevsall, gbdt-multiclass-defaults, gbdt-ordered-rmse, gbdt-feature-freq, gbdt-pointwise-l2-bayesian-eval, gbdt-categorical-ctr, gbdt-adapter-score-weighted, gbdt-query-rmse, gbdt-pair-logit, gbdt-yeti-rank, gbdt-ranking-defaults, gbdt-border-types, gbdt-ordered, gbdt-ordered-bayesian-noise, gbdt-binary-columns, gbdt-bfa-quantile, gbdt-catboost-defaults, gbdt-symmetric-eval, gbdt-stochastic-arms | no | tools/identity_break.py (cpu-identity-gate.yml) | yes |
| training | `_mojolearn_training_host.so` | `_mojolearn_training` | mlp, optim-sgd, optim-adam-clip, cross-entropy-arms, training-primitives, ordered-gradient-sum, par-mlp, samba, samba-untied-dropout-accum, par-samba, par-samba-clip, mlp-bf16w, mlp-int8w, samba-bf16w, samba-int8w, grad-accumulation | no | tools/identity_break.py (cpu-identity-gate.yml) | yes |
| resample | `_mojolearn_resample_host.so` | `_mojolearn_resample` | bootstrap, permutation-test, monte-carlo | no | tools/identity_break.py (cpu-identity-gate.yml) | yes |
| mamba | `_mojolearn_mamba_host.so` | `_mojolearn_mamba` | mamba2, mamba2-dtlimit, mamba1, mamba3, mamba1-bf16w, mamba1-int8w, mamba2-bf16w, mamba2-int8w, mamba3-bf16w, mamba3-int8w, mamba1-decode-session | no | tools/identity_break.py (cpu-identity-gate.yml) | yes |
| arima | `_mojolearn_arima_host.so` | `_mojolearn_arima` | arima, arima-011, arima-seasonal-c, par-arima, arima-exog, arima-exog-seasonal, par-forecast-arima | no | tools/identity_break.py (cpu-identity-gate.yml) | yes |
| embedding | `_mojolearn_embedding_host.so` | `_mojolearn_embedding` | embedding, embedding-sort | no | tools/identity_break.py (cpu-identity-gate.yml) | yes |
| embedding_infer | `_mojolearn_embedding_infer_host.so` | `_mojolearn_embedding` when its reference binding is not built | no | Embedding (embedding) | tools/classical_host_gate.py and tools/identity_break.py | yes |
| ivf | `_mojolearn_ivf_host.so` | `_mojolearn_ivf` | ivf, ivf-euclidean, ivf-extend, par-ivf | no | tools/identity_break.py (cpu-identity-gate.yml) | yes |
| ivf_search | `_mojolearn_ivf_search_host.so` | `_mojolearn_ivf` when its reference binding is not built | no | IVFIndex (ivf, ivf-euclidean, ivf-extend) | tools/classical_host_gate.py and tools/identity_break.py | yes |
| forecast | `_mojolearn_forecast_host.so` | `_mojolearn_arima`, `_mojolearn_tsa` when its reference binding is not built | no | ARIMA, ExponentialSmoothing, kpss_test (arima, arima-011, arima-seasonal-c, holtwinters, holtwinters-multiplicative, arima-exog, arima-exog-seasonal) | tools/classical_host_gate.py and tools/identity_break.py | yes |
| transformer | `_mojolearn_transformer_host.so` | `_mojolearn_transformer` | transformer, transformer-window, transformer-bf16w, transformer-int8w, transformer-decode-session | no | tools/identity_break.py (cpu-identity-gate.yml) | yes |<!--/fact-->

## What counts as certification

A cross-vendor result is accepted only when all of the following are recorded.

1. The same source state, fixture bytes, numeric profile and card schema were
   used on each claimed vendor.
2. Each hardware leg actually ran and records device and toolchain provenance.
3. Stage tags, dtypes, element counts and raw-bit hashes agree.
4. A sabotage arm demonstrates that the check detects the numerical mechanism
   it protects.
5. Refused configurations are reported as refusals, not passes.

`python -m mojolearn verify` checks the installed package against shipped
reference cards. See [verification](docs/VERIFY.md) and
[conformance bundles](docs/CONFORMANCE.md).
