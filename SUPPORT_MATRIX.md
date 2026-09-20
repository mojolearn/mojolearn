# Support and certification

mojolearn **0.8.5 is published on PyPI** as a Linux x86-64 wheel (CUDA sm_89,
CUDA sm_90a, HIP gfx942) and a macOS arm64 wheel, both from commit 8d16ce2f
(tags alpha-api-0.8.5-20260913 and v0.8.5), published 2026-09-14. Neither 0.8.4
nor 0.8.5 was installed and qualified on GPUs; both wheels install and import
from PyPI on a clean amd64 Linux container and on the Mac, and the identity
evidence for them is the source build on all three vendors. The last installed
GPU qualification is 0.8.3's: its Linux wheel passed its identical
qualification jobs on HIP gfx942 and CUDA sm_90a (29 smoke lanes, equal hashes
on both); sm_89 was not qualified installed. Only the three tree bindings (GBDT, Random Forest, Extra Trees) ship
the `fast` and `deterministic` tiers; every other binding builds and ships
`identical` only, and asking one of them for a lower tier raises a named error
(DEVIATION 2490, 0.8.0). The per-release detail, including what each patch
release changed and what was not qualified, is in [CHANGELOG.md](CHANGELOG.md).
Publication does not certify all features, architectures or current source
edits.

The [September 6 Apple 0.6.0 candidate](bench/results/resume/2026-09-06-feature-finish/README.md)
rebuilds all 45 extensions and passes all fifteen installed Python 3.10–3.14 /
numeric-mode jobs. This includes UMAP transform, OrderedRMSE and saved numeric
modes, plus all 102 Mamba and 44 Transformer surface checks per job. Those retained native-build checks do not qualify Linux or every later Python overlay. Corrected Apple native backward
passes all five baseline cases and Mamba1 L64; Mamba3 L65 remains RED on one
independent float32-reference intermediate despite matching the older AMD
capture's 93 shared native arrays. Python zero-state IDENTICAL backward is now exposed; exact installed-artifact and broader backward qualification remain open.

For the enforced identity surface, see the
[identity-path ledger](IDENTITY_PATHS.md). Fresh certification artifacts name
their commit, device, mode, and limitations; stale narrative reports are not a
support source.

## Numeric modes

| Mode | Contract |
|---|---|
| `fast` | Performance-oriented. No repeatability or cross-vendor bitwise promise. |
| `deterministic` | Repeated execution on the same device and build is intended to return the same bits. It makes no cross-vendor promise. |
| `identical` | For a specifically certified fixture and configuration, raw result bits match across the recorded Metal, CUDA, and HIP legs. |

Select a process default with `mojolearn.set_numeric_mode(...)` or
`MOJOLEARN_NUMERIC_MODE`. Estimators that accept `numeric_mode=` can override
the process default. All three modes are runtime-selectable binary sets, not
separate packages.

`identical` is not a blanket claim for arbitrary shapes, parameters, hardware,
drivers, or future builds. The claim is limited to completed evidence cards.
An unrun column is **pending**, never inferred from source inspection or another
device.

## Installation support

| Platform | Status | Qualification |
|---|---|---|
| macOS arm64 / Apple silicon | Published 0.8.5 wheel | Built at the Apple M1 ISA floor from commit 8d16ce2f; installs and imports from PyPI on the Mac. The tree bindings ship three tiers; every other binding ships `identical` only (DEVIATION 2490). The 0.8.3 SVC fix was verified on the Apple M4 and the 0.8.2 GBDT fix gave 36/36 identity cells equal to the H100 on the Apple M4 ([CHANGELOG.md](CHANGELOG.md)). |
| Linux x86-64 / NVIDIA CUDA | Published 0.8.5 wheel, `sm_89` and `sm_90a` | The 0.8.5 wheel was not installed and qualified on a GPU (it installs and imports on a clean amd64 container). The installed 0.8.3 wheel passed its identical qualification jobs on CUDA sm_90a (29 smoke lanes) and fit SVC at 400, 600 and 2,000 rows on an H100 with the source builds' bits; sm_89 was not qualified installed. Device code is architecture-specific. Certification on one NVIDIA architecture does not certify another. Release 0.3.0 had an AVX-512 host-code defect; it is historical and must not be used as current evidence. |
| Linux x86-64 / AMD HIP | Published 0.8.5 wheel, `gfx942` | The 0.8.5 wheel was not installed and qualified on a GPU; its HIP set was built on an AMD MI300X inside the 22.04 ROCm container. The installed 0.8.3 wheel passed its identical qualification jobs on HIP gfx942 (29 smoke lanes, hashes equal to sm_90a). The 0.8.2 GBDT sync fix (DEVIATION 2600) was verified 36/36 identity cells equal to the H100 on an MI300X. Measured chiefly on `gfx942`. That is not evidence for every AMD architecture. |
| CPU-only and other accelerators | Public saved-model inference; every CPU training binding ships too, for internal bitwise verification rather than production fitting | No estimator falls back to a CPU on its own; the library refuses. Named CPU paths exist and each is gated bit for bit against the Apple, NVIDIA and AMD columns with a sabotage build required to fail; the whole surface is declared once in `python/mojolearn/host_surface.py` and tabulated under [The CPU surface](#the-cpu-surface) below. Inference from a saved model: <!--fact:host_inference_surfaces-->random forests, Extra Trees and eight gradient boosting variants; nearest neighbors on every metric and the ball cover, k-NN classification and k-NN regression with either weighting, radius neighbors and k-means assignment and distances; linear regression, ridge, truncated SVD, logistic regression, PCA with and without whitening (either solver), kernel density on every kernel, metric and weighting, the standard and min-max scalers, lasso, elasticnet, kernel ridge, the Nystroem approximation and random Fourier features; UMAP transform of a saved embedding (the GPU's bytes for a row, whatever else is asked in the same batch); SVC and the isolation forest; the Gaussian mixture's scores, probabilities, labels and samples; the Gaussian process regressor's predictive mean and std, normalized targets included, and the Gaussian process classifier's labels and probabilities; HDBSCAN's approximate_predict, membership_vector and all_points_membership_vectors; Embedding lookup in a saved table; IVF-Flat search over a saved index and extending it; batched ARIMA prediction, in sample and out of sample, and forecasts, with or without exogenous regressors, and Holt-Winters forecasts and in-sample one-step predictions, additive and multiplicative<!--/fact--> (the forests: 24 three-GPU recordings reproduced on seven CPUs, [fixtures](bench/results/forest_host/README.md); the first five classical estimators: three-vendor recordings, 45 fixtures each; kernel density, SVC, whitened PCA and the k-NN classes: three-vendor recordings, 54 fixtures each, [fixtures](bench/results/classical_host/)). Training: <!--fact:host_training_lanes-->kernel ridge poly kernel variant, kernel ridge sigmoid kernel variant, kernel ridge laplacian kernel variant, nystroem poly kernel variant, nystroem sigmoid kernel variant, nystroem laplacian kernel variant, the row-sharded random Fourier feature transform, ordinary differencing order selection, pinned GEMM, kernel density, Holt-Winters, lasso, elasticnet, SVC, agglomerative clustering, the Extra Trees classifier, the Extra Trees regressor, the isolation forest, nearest neighbors, the k-NN classifier, the k-NN regressor, PCA, whitened PCA, truncated SVD, linear regression, ridge, DBSCAN, k-means, the metrics, spectral clustering, the standard scaler, the min-max scaler, logistic regression, the random forest classifier, the random forest regressor, k-means with a random start, k-means from given centroids, weighted k-means, the standard scaler without centering, the standard scaler without scaling, the clipped min-max scaler, spectral clustering on a precomputed affinity, the Gaussian process with an RBF kernel, the Gaussian process with a Matern kernel at nu 0.5, the Gaussian process with a Matern kernel at nu 1.5, the Gaussian process with an ARD Matern kernel at nu 2.5, the Gaussian process with normalized targets, Gaussian process hyperparameter optimization, Gaussian process hyperparameter optimization with restarts, the binary Gaussian process classifier, the one-vs-rest Gaussian process classifier, the class-sharded Gaussian process classifier fit, the class-sharded Gaussian process classifier prediction, nearest neighbors under squared euclidean distance, the distance-weighted k-NN classifier, the distance-weighted k-NN regressor, the transposed GEMM ops, the Householder QR's R factor, both slice arms, the symmetric Jacobi eigendecomposition, ascending, the singular values, descending, brute-force DBSCAN under manhattan distance, kernel density with the tophat kernel under squared euclidean distance, kernel density with the Epanechnikov kernel under manhattan distance, kernel density with the exponential kernel under chebyshev distance, kernel density with the linear kernel under cosine distance, kernel density with the cosine kernel under minkowski distance, weighted kernel density, linear regression without an intercept, weighted linear regression, ridge without an intercept, unpenalized logistic regression without an intercept, elasticnet at the l2 end without an intercept, multiplicative Holt-Winters, the linear SVC, the polynomial SVC, the tuned isolation forest, nearest neighbors under manhattan distance, nearest neighbors under chebyshev distance, nearest neighbors under cosine distance, nearest neighbors under minkowski distance at p 3, nearest neighbors over the random ball cover, radius neighbors, radius neighbors under manhattan distance, radius neighbors under chebyshev distance, radius neighbors under minkowski distance at p 3, weighted DBSCAN, l1-penalized logistic regression, elasticnet-penalized logistic regression, multiclass logistic regression, the KPSS stationarity test, SVR, the linear SVR, whitened PCA through the full SVD, the classification, ranking and regression metrics, the Fowlkes-Mallows index, the combined homogeneity, completeness and V-measure scores, the weighted scores of the gradient boosting classifier and regressor, the weighted scores of the random forest classifier and regressor, the small MLP, gradient boosting on symmetric trees with the Logloss loss, gradient boosting on symmetric trees with the RMSE loss, gradient boosting on depthwise trees with the Logloss loss, gradient boosting on lossguide trees with the Logloss loss, gradient boosting with the Min and Max NaN modes, the gradient boosting classifier, the gradient boosting regressor, gradient boosting with the Quantile, MAE, LogLinQuantile, MAPE, Poisson, Lq, Expectile, Tweedie, Huber and CrossEntropy losses, gradient boosting with Exact leaves and the Poisson bootstrap, gradient boosting on lossguide trees with the NewtonCosine score and the searcher options, multiclass gradient boosting, one-vs-all gradient boosting, ordered boosting with the RMSE loss (OrderedRMSE), gradient boosting with the six non-default feature border types, ordered boosting (boosting_type='Ordered') with the Logloss and RMSE losses, ordered boosting with the Bayesian bootstrap and score noise, boost from average on the MAE, Quantile and MAPE losses, gradient boosting at CatBoost's GPU defaults (auto learning rate, Bayesian bootstrap, score noise), gradient boosting with an eval set, the overfitting detector and best-model truncation, the two-level FeatureFreq estimator, gradient boosting with the pointwise searcher, L2 scores, the Bayesian bootstrap and an eval set, gradient boosting with one-hot categorical columns, gradient boosting with the QueryRMSE ranking loss on query groups, gradient boosting with the PairLogit ranking loss on generated and explicit pairs, gradient boosting with the YetiRank ranking loss on query groups, ARIMA, differenced ARIMA, seasonal ARIMA, ARIMA with exogenous regressors, differenced seasonal ARIMA with exogenous regressors, UMAP, k-means under the rooted euclidean metric, k-means from the classic k-means++ start, cross-validation of gradient boosting, the bootstrap, the permutation test, Monte Carlo integration, SGD with momentum, Nesterov and dampening, Adam and AdamW with the gradient clip and accumulation, the cross-entropy loss arms, the embedding, RMSNorm and linear training primitives, the ordered shard gradient reduction, the Cholesky factorization and solve, random Fourier features, kernel ridge, the Nystroem kernel approximation, the Gaussian mixture, the Gaussian mixture with a random start, HDBSCAN, HDBSCAN with leaf selection, the random forest classifier with entropy splits, log2 features and no bootstrap, the class-weighted random forest classifier with the parallel groves engine, the random forest regressor with the Poisson criterion, the random forest regressor with the gamma and inverse Gaussian criteria, the best-first Extra Trees classifier with entropy splits, the bootstrapped Extra Trees regressor with the parallel groves engine, the Mamba-2 block, the Mamba-2 block with an active dt clamp, the Mamba-1 block, the Mamba-3 block, the Transformer block, the sliding-window Transformer block, the Samba stack, the Samba stack with untied embeddings, dropout, accumulation, clipping and a cosine schedule, the byte LM forward pass on its reference path (inference), the byte LM forward pass on its threaded path (inference), the published byte LM host training step, the byte LM shape object at two non-default shapes, the column-sharded standard scaler, the column-sharded min-max scaler, series-sharded ARIMA, series-sharded Holt-Winters, the series-sharded ARIMA prediction and forecast drivers, the series-sharded Holt-Winters prediction and forecast drivers, query-sharded k-NN classification, query-sharded nearest-neighbor distances and indices, query-sharded radius neighbors, query-sharded kernel density, reference-sharded k-NN classification, reference-sharded k-NN regression, the tree-range-sharded random forest classifier, the tree-range-sharded Extra Trees regressor, the tree-range-sharded Extra Trees classifier, the tree-range-sharded random forest regressor, the small MLP trained over ordered logical gradient shards, the Samba stack trained over ordered logical gradient shards, the Samba stack trained over ordered logical gradient shards under a global norm clip, the Embedding layer, the Embedding layer on its sorted execution plan, the IVF-Flat index, the IVF-Flat index under euclidean distance, the shard-distributed IVF-Flat index, extending a built IVF-Flat index, the byte LM trainer, the byte LM trainer on its resident session, samples from the Gaussian mixture, samples from the Gaussian mixture with a random start, posterior draws from the Gaussian process, posterior draws from the Gaussian process with normalized targets, the byte-level BPE tokenizer (inference, host integers), byte-level BPE vocabulary training, a trained BPE vocabulary written, loaded back and used, a corpus tokenized once, cached and read back as batches, the Hugging Face checkpoint reader and the option matrix, the Hugging Face byte-level BPE tokenizer (three pre-tokenization patterns), a Hugging Face causal language model loaded and run, predictions of Metal-saved gradient boosting models with CTR tables (inference), predictions of Metal-saved gradient boosting models with tensor CTRs (inference), the bf16-storage GEMM profile, the int8 GEMM profile with power-of-two scales, the Transformer block with bf16-stored weights, the Transformer block with int8-stored weights, the Mamba-1 block with bf16-stored weights, the Mamba-1 block with int8-stored weights, the Mamba-2 block with bf16-stored weights, the Mamba-2 block with int8-stored weights, the Mamba-3 block with bf16-stored weights, the Mamba-3 block with int8-stored weights, the small MLP with bf16-stored weights, the small MLP with int8-stored weights, the Samba stack with bf16-stored weights, the Samba stack with int8-stored weights, saved forest and gradient boosting models predicted on the CPU (inference), the bf16 and int8 weight-storage conversions and gradient accumulation across microbatches<!--/fact-->, each identical to the three training GPU columns the manifest names in the full CPU reference [gate](.github/workflows/cpu-identity-gate.yml), with the sabotage host build required to read DIVERGENT, which the CPU reference gate builds and exercises on every run (agglomerative clustering, Extra Trees and the isolation forest were first read on the Apple M4 host path, [columns](bench/results/identity_break/2026-09-14_cpu-phase1b/README.md); for the random forest, k-means, scaler, spectral, Gaussian process, ARIMA and UMAP lanes the arm is declared and exercised by that gate, and no column recording it is committed in this tree). The published 0.8.5 wheels carry the byte LM host binding alone; 0.8.6 was folded into 0.8.7 and never published; and since 2026-09-16 (lane/ship-cpu-host-families) EVERY family in the table below ships in both wheels under `mojolearn/host/` (its last column), which 0.8.7 is the first published release to carry, so a lane's fit can be re-run where the user is rather than only in this repository, and a source checkout builds them with `bindings/build_*_host.sh`, each a shim over `bindings/build_host_family.sh`. Shipping the training bindings did not make ordinary CPU fitting public: it still refuses on a CPU-only install, and these bindings answer `python -m mojolearn verify --all`, which fits inside the verifier's own reference scope. Every other lane has no CPU path (<!--fact:no_cpu_path-->(1) gradient boosting training with sample weights on any arm (`gbdt_fit` refuses `sample_weight`, and `class_weights` outside MultiClass and MultiClassOneVsAll, which reach the device through the same per-row weight column): the device's weighted target, histogram and partition-reduce kernels are a second launch arm (`has_weights`) and the gbdt/host oracles restate the unit-weight arm only; (2) gradient boosting training on a CTR categorical column, a `cat_features` column with more than `one_hot_max_size` categories: the CTR calcers build ordered target statistics over several permutations, with their online counters, grids and tables joined back into the compressed index, and the host path is pinned to one permutation with no calcer. One-hot categorical columns DO train, `ExperimentalTwoLevelFeatureFreq` has its own CPU route (gbdt/host/gbdt_oracle_feature_freq.mojo) except on a tree whose level winner is the tensor column itself, and CTR INFERENCE from a saved model is closed; it is the calcer tables' training that is not; (3) gradient boosting training with a categorical or one-hot column outside SymmetricTree with Logloss and Plain boosting: the one-hot grid and the `take_bin` equality split are restated in the symmetric searcher alone; (4) gradient boosting training with an eval set or the overfitting detector outside SymmetricTree with Logloss, Ordered boosting and the pointwise searcher's own lane: the held-out curve runs THAT arm's loss kernel (the multilogit and one-vs-all launches, `launch_approximate` at each pointwise objective) and the non-symmetric shapes put a tree on the cursor through a different apply, and neither is restated; (5) gradient boosting training with the pointwise searcher outside the gbdt-pointwise-l2-bayesian-eval configuration (L2 scores, the Bayesian bootstrap, Newton leaves, sample weights, an eval set with the Iter detector, boost_from_average on, GreedyLogSum borders, numeric columns): every other option selects a different launch shape of the pointwise kernels and one shape is restated; (6) gradient boosting training at a (loss, grow_policy, score_function, leaf_estimation_method, bootstrap_type) combination outside the ones the gbdt/host oracles restate, each refused by name: RMSE and the pointwise losses under Depthwise and Lossguide, score functions and leaf estimators outside each policy's covered pair, most (bootstrap, loss) pairs, Depthwise's min_split_gain, min_child_hessian and min_data_in_leaf, random_strength and feature_fraction outside Logloss, boost_from_average outside RMSE and the quantile family, and a NaN in X outside SymmetricTree with Logloss -- each one its own device kernel or its own searcher gate order<!--/fact-->). The byte LM has two CPU surfaces needing no GPU: `LanguageModelInference` for the forward pass ([docs/BYTE_LM_CPU_INFERENCE.md](docs/BYTE_LM_CPU_INFERENCE.md)), and `LanguageModelHostTrainer` for one training step, forward, backward and the AdamW update, which reproduces the recorded GPU bytes of the retained three-vendor capture for all 128 of its steps, gradient and loss and post-step parameters and both Adam moments alike, on seven CPUs, with a wrong-gradient build required to fail the same gate ([docs/BYTE_LM_CPU_TRAINING.md](docs/BYTE_LM_CPU_TRAINING.md)). Both are one model profile at one batch shape; identity is claimed per shape, because the weight gradients contract over the token count. Reference path, one thread: a whole step measured 37 to 39 ms on the Linux CI runners and 69 ms on Apple M1. |

## The CPU surface

Generated from `python/mojolearn/host_surface.py` by `tools/docs_facts.py
--write`; `pixi run check-docs-facts` fails when this table and the manifest
disagree. Every binding below is IDENTICAL only, compiles as the kernel
matrix's CPU column (asserted at build time, read back before use), and
refuses by name every function it does not list.

<!--fact:host_surface_table-->| family | binding under `mojolearn/host/` | routes (CPU-only install) | internal CPU reference lanes | predicts on a CPU from a saved model | gate | in a wheel |
|---|---|---|---|---|---|---|
| byte_lm | `_mojolearn_byte_lm_host.so` | loaded by path | byte-lm-host-infer, byte-lm-host-infer-threaded, byte-lm-host-train, byte-lm, byte-lm-resident, language-model-config | LanguageModelInference, LanguageModelHostTrainer, SmallByteLanguageModelTrainer | .github/workflows/byte-lm-cpu-gate.yml and tools/identity_break.py (cpu-identity-gate.yml) | yes |
| forest | `_mojolearn_forest_host.so` | loaded by path | gbdt-categorical-ctr-tables, gbdt-tensor-ctr-tables, saved-model-host-infer | RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor, GradientBoosting, OrderedRMSE, ExperimentalTwoLevelFeatureFreq (rf_classifier, rf_regressor, et_classifier, et_regressor, gbdt_symmetric, gbdt_depthwise, gbdt_lossguide, gbdt_rmse, gbdt_ordered_rmse, gbdt_feature_freq, gbdt_pointwise_bayesian_eval, gbdt_categorical_onehot) | tools/forest_host_gate.py (.github/workflows/forest-host-gate.yml) | yes |
| tokenizer | `_mojolearn_tokenizer_host.so` | loaded by path | tokenizer, bpe-trainer, bpe-vocabulary, tokenized-corpus, hf-tokenizer | BpeTokenizer | pixi run check-tokenizer and python/mojolearn/tests/test_tokenizer_surface.py | yes |
| neural | `_mojolearn_neural_host.so` | loaded by path | hf-causal-lm | MLPInference, TransformerBlockInference, Mamba1BlockInference, Mamba2BlockInference, Mamba3BlockInference, SambaInference | python/mojolearn/tests/test_neural_inference.py, tools/step_vs_full_check.py and tools/identity_break.py (mlp, transformer, transformer-window, transformer-decode, mamba1, mamba2, mamba3, mamba1-decode, mamba2-decode, mamba3-decode, mamba2-dtlimit, samba, samba-decode, samba-untied-dropout-accum) | yes |
| core | `_mojolearn_core_host.so` | `_mojolearn` | knn, knn-clf, knn-reg, kmeans, kmeans-random, kmeans-array, kmeans-weighted, knn-sqeuclidean, knn-clf-distance, knn-reg-distance, knn-manhattan, knn-chebyshev, knn-cosine, knn-minkowski-p3, knn-rbc, radius, radius-manhattan, radius-chebyshev, radius-minkowski-p3, kmeans-sqrt, kmeans-classic-pp, par-queries-knn, par-queries-nn, par-queries-radius, par-reference-knn, par-reference-knn-reg | NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor, KMeans, RadiusNeighbors (knn, knn-clf, knn-reg, knn-sqeuclidean, knn-manhattan, knn-chebyshev, knn-cosine, knn-minkowski-p3, knn-rbc, knn-clf-distance, knn-reg-distance, radius, radius-manhattan, radius-chebyshev, radius-minkowski-p3, kmeans, kmeans-random, kmeans-array, kmeans-weighted, kmeans-sqrt, kmeans-classic-pp) | tools/classical_host_gate.py (cpu-identity-gate.yml) | yes |
| linalg | `_mojolearn_linalg_host.so` | `_mojolearn_linalg` | gemm-pinned, gemm-transposed, cholesky, gemm-bf16, gemm-int8, linalg-qr, linalg-eigh, linalg-svdvals, lowbit-conversions, hf-checkpoint | no | tools/identity_break.py (cpu-identity-gate.yml) | yes |
| estimators | `_mojolearn_estimators_host.so` | `_mojolearn_estimators` | kde, pca, pca-whiten, tsvd, ols, ridge, dbscan, logistic, dbscan-brute-l1, kde-tophat-sqeuclidean, kde-epanechnikov-l1, kde-exponential-chebyshev, kde-linear-cosine, kde-cosine-minkowski, kde-weighted, ols-no-intercept, ols-weighted, ridge-no-intercept, logistic-unpenalized-no-intercept, dbscan-weighted, logistic-l1, logistic-elasticnet, logistic-multiclass, pca-full-whiten, par-queries-kde | LinearRegression, Ridge, TruncatedSVD, LogisticRegression, PCA, KernelDensity, DBSCAN, StandardScaler, MinMaxScaler, Lasso, ElasticNet, KernelRidge, Nystroem, RBFSampler, AgglomerativeClustering (ols, ridge, tsvd, logistic, logistic-multiclass, pca, pca-whiten, kde, ols-no-intercept, ols-weighted, ridge-no-intercept, logistic-l1, logistic-elasticnet, logistic-unpenalized-no-intercept, standard-scaler, standard-scaler-no-mean, standard-scaler-no-std, minmax-scaler, minmax-scaler-clip, lasso, elasticnet, elasticnet-l2end-no-intercept, kernel-ridge, nystroem, rbf-sampler, pca-full-whiten, kernel-ridge-poly, kernel-ridge-sigmoid, kernel-ridge-laplacian, nystroem-poly, nystroem-sigmoid, nystroem-laplacian, kde-tophat-sqeuclidean, kde-epanechnikov-l1, kde-exponential-chebyshev, kde-linear-cosine, kde-cosine-minkowski, kde-weighted, dbscan, agglomerative) | tools/identity_break.py and tools/classical_host_gate.py (cpu-identity-gate.yml) | yes |
| metrics | `_mojolearn_metrics_host.so` | `_mojolearn_metrics` | metrics, spectral, spectral-precomputed, umap, metrics-classification, metrics-fowlkes-mallows, metrics-homogeneity-completeness | SpectralClustering, UMAP, metrics.accuracy_score, metrics.adjusted_rand_score, metrics.entropy, metrics.mutual_info_score, metrics.homogeneity_score, metrics.completeness_score, metrics.v_measure_score, metrics.r2_score, metrics.silhouette_score, metrics.silhouette_samples, metrics.rand_score, metrics.precision_score, metrics.recall_score, metrics.f1_score, metrics.log_loss, metrics.roc_auc_score, metrics.confusion_matrix, metrics.precision_recall_curve, metrics.mean_squared_error, metrics.mean_absolute_error, metrics.root_mean_squared_error, metrics.kl_divergence, metrics.trustworthiness, metrics.fowlkes_mallows_score (umap, spectral, spectral-precomputed) | tools/identity_break.py (cpu-identity-gate.yml) | yes |
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
| gbdt | `_mojolearn_gbdt_host.so` | `_mojolearn_gbdt` | gbdt-symmetric, gbdt-rmse, gbdt-depthwise, gbdt-lossguide, cross-val, gbdt-nan-modes, gbdt-adapter-clf, gbdt-adapter-reg, gbdt-parametric-losses, gbdt-exact-mae, gbdt-lossguide-newtoncosine, gbdt-multiclass, gbdt-onevsall, gbdt-ordered-rmse, gbdt-feature-freq, gbdt-pointwise-l2-bayesian-eval, gbdt-categorical-ctr, gbdt-adapter-score-weighted, gbdt-query-rmse, gbdt-pair-logit, gbdt-yeti-rank, gbdt-border-types, gbdt-ordered, gbdt-ordered-bayesian-noise, gbdt-bfa-quantile, gbdt-catboost-defaults, gbdt-symmetric-eval | no | tools/identity_break.py (cpu-identity-gate.yml) | yes |
| training | `_mojolearn_training_host.so` | `_mojolearn_training` | mlp, optim-sgd, optim-adam-clip, cross-entropy-arms, training-primitives, ordered-gradient-sum, par-mlp, samba, samba-untied-dropout-accum, par-samba, par-samba-clip, mlp-bf16w, mlp-int8w, samba-bf16w, samba-int8w, grad-accumulation | no | tools/identity_break.py (cpu-identity-gate.yml) | yes |
| resample | `_mojolearn_resample_host.so` | `_mojolearn_resample` | bootstrap, permutation-test, monte-carlo | no | tools/identity_break.py (cpu-identity-gate.yml) | yes |
| mamba | `_mojolearn_mamba_host.so` | `_mojolearn_mamba` | mamba2, mamba2-dtlimit, mamba1, mamba3, mamba1-bf16w, mamba1-int8w, mamba2-bf16w, mamba2-int8w, mamba3-bf16w, mamba3-int8w | no | tools/identity_break.py (cpu-identity-gate.yml) | yes |
| arima | `_mojolearn_arima_host.so` | `_mojolearn_arima` | arima, arima-011, arima-seasonal-c, par-arima, arima-exog, arima-exog-seasonal, par-forecast-arima | no | tools/identity_break.py (cpu-identity-gate.yml) | yes |
| embedding | `_mojolearn_embedding_host.so` | `_mojolearn_embedding` | embedding, embedding-sort | no | tools/identity_break.py (cpu-identity-gate.yml) | yes |
| embedding_infer | `_mojolearn_embedding_infer_host.so` | `_mojolearn_embedding` when its reference binding is not built | no | Embedding (embedding) | tools/classical_host_gate.py and tools/identity_break.py | yes |
| ivf | `_mojolearn_ivf_host.so` | `_mojolearn_ivf` | ivf, ivf-euclidean, ivf-extend, par-ivf | no | tools/identity_break.py (cpu-identity-gate.yml) | yes |
| ivf_search | `_mojolearn_ivf_search_host.so` | `_mojolearn_ivf` when its reference binding is not built | no | IVFIndex (ivf, ivf-euclidean, ivf-extend) | tools/classical_host_gate.py and tools/identity_break.py | yes |
| forecast | `_mojolearn_forecast_host.so` | `_mojolearn_arima`, `_mojolearn_tsa` when its reference binding is not built | no | ARIMA, ExponentialSmoothing, kpss_test (arima, arima-011, arima-seasonal-c, holtwinters, holtwinters-multiplicative, arima-exog, arima-exog-seasonal) | tools/classical_host_gate.py and tools/identity_break.py | yes |
| transformer | `_mojolearn_transformer_host.so` | `_mojolearn_transformer` | transformer, transformer-window, transformer-bf16w, transformer-int8w | no | tools/identity_break.py (cpu-identity-gate.yml) | yes |<!--/fact-->

Before publishing a release, validate the installed wheel rather than only the
source tree. Import it in a clean environment, load every shipped extension in
each tier it ships (three tiers for the tree bindings, `identical` only for the
rest), and run the public smoke suite. See [release instructions](docs/PYPI_RELEASE.md)
and the [release checklist](docs/RELEASE_CHECKLIST.md).

## Capability snapshot

The table groups public surfaces by the strongest evidence currently retained.
“Three-vendor card” means at least one named fixture has matching IDENTICAL
cards on Apple, NVIDIA, and AMD; it does not extend beyond that fixture. Each
row's evidence was recorded at the commit its cited card or brief names, and a
row that names no commit is certified only at the commit of its linked record,
not at the current source.

**Known IDENTICAL violation.** The Gram product `A^T A` that PCA, truncated
SVD and OLS run their first step through ([IDENTITY_PATHS.md](IDENTITY_PATHS.md)
row 27) falls back to the vendor matmul under IDENTICAL when the design has
more columns than the split-K kernel's capacity, instead of refusing. The
129-column fallback arm of `decomposition/checks/pca_check.mojo` completes
under IDENTICAL and returns a model that mode promises is vendor-independent
and is not. Recorded as pre-existing on NVIDIA and AMD in the
[AMD confirmations brief](docs/lanes/BRIEF_amd_confirmations_2026-09-12.md)
(finding 3); the refusal past capacity is still owed.

**A cross-vendor divergence found and fixed (DEVIATION 2710).** On 2026-09-13 the
46-lane run of `tools/identity_break.py` found `ExperimentalTwoLevelFeatureFreq`
different between every pair of an Apple M4, an NVIDIA H100 and an AMD MI325X
on 8 of 9 hostile fixtures while the other 43 lanes were identical. The cause
was memory, not arithmetic: the synchronized tensor drivers sized and zeroed
the fixed-point histogram accumulator by a literal dead flag, so under
IDENTICAL the histogram kernels wrote 264 cells into a 4-byte, never-zeroed
buffer and read back whatever each vendor's allocator had left there. Sized by
the live flag, all three vendors give the Apple column's bits on every fixture
and column (fixed arm base 7d9c56b51213cb42, hashed 9d97a55431b6f8e0; the old
code, selectable with `-D MOJOLEARN_2710_TENSOR_ACC_DEAD=1`, reproduces each
vendor's wrong hashes), see
`docs/lanes/BRIEF_feature_freq_divergence_2026-09-13.md`. The fix shipped in 0.8.5;
0.8.4 carries the defect on this estimator. Verified after the fix on 2026-09-14: the 46-lane
three-column rerun reads IDENTICAL x3 on every cell of every lane, this one included
(`bench/results/identity_break/2026-09-14_46-lanes/README.md`, 414 train and 459 infer and
model cells on an Apple M4, an NVIDIA H100 and an AMD MI300X, no one-column cell).

**Mamba-2 read past the end of X_d, 2026-09-14, DEVIATIONS 2712 and 2713, CLOSED together, one cause.**
`m2_ydiag_kernel` (`mamba/impl/modules/ssd_minimal.mojo`, S13 and S14, Y_diag = M . X_d) looped over
the chunk width Q = 256 and loaded the X_d row c * Q + jj for every jj, including rows at or past
T (X_d is [B, T, H, P]; T is 16 in the identity lanes and 1 in the step), a read past the end of the
allocation. The multiplier there is the structural +0.0, so 0 x finite garbage is 0 and every CUDA
and HIP run and every cold Metal run carried the same bits; 0 x NaN is NaN, which is what Metal
handed back in a warm process (DEVIATION 2712: the Apple M4 column of the 120-lane record at
65ae7612f, `step` and `backward` NaN on all nine mamba2 fixtures, first misattributed to AMD and
corrected the same day; the step reads farthest past since its T is 1); and the row past the last
allocation is an unmapped page on the MI325X 24.04 image (DEVIATION 2713: "Memory
access fault by GPU node-1" at the first forward launch, where the MI300X allocator
happened to back the page). Named by a poison-and-band build: every Mamba device allocation filled
with the canonical quiet NaN where nothing writes and given a 4096-element NaN band past its logical
length (`-D MOJOLEARN_MAMBA_POISON=1`), under which the block check reported `ydiag.out` MOVED 256
of 256 with every input stage OK. Fixed by reading X_d only below T (+0.0 past it, the bits every
record carries); the sixty Mamba-2 backward buffers that were allocated with no fill are filled
(`mamba_scratch`, zero in production). The gate that catches the next read of this class is
`pixi run check-mamba-poison` (`tools/mamba_poison_gate.sh`): the poison binding built into a copy of
the package, mamba1, mamba2, mamba2-dtlimit and mamba3 cold, all 36 training rows required IDENTICAL
against the three vendor columns of `2026-09-14_120-lanes-2711flip`, with three sabotages (the fix
removed, a planted over-read with the band, the same without it as the control). On the Apple M4:
36 of 36 with the fix, the fix-removed and planted-over-read sabotages fail, the control passes.
Owed: the gate on the H100 and the MI300X, the poison build on the MI325X image, and the 120-lane
three-column rerun that becomes the CPU gate's GPU columns (brief section 3.3).

**RTX 5090 (sm_120a), 2026-09-14, DEVIATION 2711, found and FIXED the same day.** The first leg on
a Blackwell consumer part refused four cells (pca, tsvd, ols, ridge on the 17-column `odd`
fixture): an sm_120a ahead-of-time pass miscompiled the strided-singles arm of the 17-wide split-K
Gram, so the Jacobi was handed an asymmetric matrix (a JIT build of the same source on the same GPU
was already right, and the wrong cells repeated bit for bit). `core/gram_splitk.mojo` ships the
scalar strided arm from 83380ca6d, the same products in the same order with per-cell scalar
accumulators; `-D MOJOLEARN_2711_GRAM_STRIDED_DEAD=1` restores the old arm for an A/B. Proof: the
120-lane record `bench/results/identity_break/2026-09-14_120-lanes-2711flip/README.md`, where the
Apple M4, the H100, the MI300X and the RTX 5090 read IDENTICAL on 1071 training and 1476 inference
and model cells, the four cells included, and the H100 and MI300X columns are identical to their
columns before the flip on every shared cell.

| Surface | Public availability | Strongest retained identity evidence | Important open work |
|---|---|---|---|
| Gradient boosting | Beta | Three-vendor cards for recorded configurations | Numeric single-permutation ordered RMSE passes AMD/NVIDIA at `6dd44ac5`, with 130 bitwise-matching records. Broader categorical/CTR coverage and external parity remain. |
| Random Forest / Extra Trees | Beta | Three-vendor cards for recorded configurations | Keep sklearn `max_leaf_nodes` semantics distinct from cuML-style level-order `max_leaves`; extend NVIDIA performance coverage. |
| k-means, DBSCAN | Beta | Three-vendor cards for recorded fixtures | Broaden shapes and public-surface wheel smoke. `SpectralClustering.predict` (2026-09-15, DEVIATION 2860, the Nystrom extension from a `prediction_data=True` fit) and `DBSCAN.predict` / `AgglomerativeClustering.predict` (DEVIATION 2740): their infer, model and batch cells now carry two NVIDIA columns (A100 sm_80 and RTX 2000 Ada sm_89, 2026-09-16) beside the Apple M4 and x86 CPU ones, and the saved-model route is recorded and checked at bench/results/classical_host/2026-09-16-nvidia-predict. The `spectral` lane was shrunk from 2000 rows to 512 after its 2026-09-15 columns were taken, so those two are superseded and its Apple column was retaken at the published size; the AMD cells are owed at the next release record. |
| k-NN | Beta | Three-vendor cards; the completed AMD four-arm layout outputs also match retained NVIDIA outputs and Apple hashes | Selector and transposed-distance flags remain opt-in. Broaden distributions and installed/external comparisons before changing normal dispatch. |
| PCA, truncated SVD, OLS, Ridge, logistic regression | Beta | Three-vendor matrix for recorded fixtures | Broaden shapes and public-surface wheel smoke. |
| FP32 matrix multiplication | Beta | Three-vendor frozen-profile sweep | Shapes and plans outside the recorded profile remain uncertified. |
| Isolation Forest | Beta | Three-vendor card for the recorded fixture | Re-run current bindings on NVIDIA architectures affected by earlier context-lifetime failures. |
| ARIMA filtering | Beta | Three-vendor card for the recorded filter fixture | The fitted estimator passes the frozen AMD/NVIDIA installed candidates in all three modes; final release qualification remains. |
| Holt-Winters and spectral/time-series helpers | Beta | Apple–AMD cards for recorded fixtures | NVIDIA column remains pending. `kpss_test` and `select_d` are public with no three-vendor card (`IDENTITY_PATHS.md` has no tsa row; the claim-surface census, 2026-09-14); `select_d` has no CPU path, and `kpss_test` runs on the CPU through the tsa host binding (the kpss lane, lane/cpu-training-batch3, 2026-09-14). A saved `ExponentialSmoothing` model forecasts and predicts in sample on a CPU with no GPU through the shipped forecast host binding (lane/inference-holtwinters, 2026-09-15); the in-sample predictions have an Apple recording only. |
| Gaussian process | Experimental | Regressor: Apple–AMD IDENTICAL card for recorded fixture. Classifier (`GaussianProcessClassifier`, the Laplace approximation with `optimizer=None`, one-vs-rest past two classes): the `gpc` and `gpc-multiclass` identity lanes on the Apple M4 Metal column and the CPU host columns; `sample_y` (2026-09-15, DEVIATION 2793): `gp-sample-y` and `gp-sample-y-normalize` IDENTICAL on Apple M4 Metal against the x86 CPU verifier; kernel hyperparameter optimization (2026-09-15, DEVIATIONS 2880 and 2881): `GaussianProcessRegressor(optimizer='fmin_l_bfgs_b', n_restarts_optimizer=k)` with the identical likelihood gradient and a projected L-BFGS, lanes `gp-optimize` and `gp-optimize-restarts` | Complete current-wheel and NVIDIA qualification; the classifier's NVIDIA and AMD columns are owed to the release record; no performance claim. Record `gp-sample-y`, `gp-sample-y-normalize`, `gp-optimize` and `gp-optimize-restarts` on NVIDIA and AMD at the release record; the classifier's optimizer stays refused (DEVIATION 1761); public CPU `sample_y` from a saved model is lane/inference-neighbors-density's. |
| Workstream D doors (2026-09-14), Cholesky (`Cholesky`, inside `_mojolearn_gp`), kernel methods (`KernelRidge`, `Nystroem`, `RBFSampler`, `_mojolearn_kernel_methods`), Gaussian mixture (`GaussianMixture`, `_mojolearn_mixture`), HDBSCAN (`HDBSCAN`, `_mojolearn_hdbscan`), resampling (`bootstrap`, `permutation_test`, `monte_carlo_integrate`, `_mojolearn_resample`) | Alpha, from source, lane `lane/expose-d`. The binding, the Python class and its surface test exist for each, and the four new bindings are registered in `_backend` and in every packaging list | What is measured is exactly this and nothing more. Each binding compiled once on one Apple M4 (`mojo build`, identical tier, apple column, 13 to 28 s each), and the Mojo checks each family already had (`pixi run check-cholesky`, `check-kernel-methods`, `check-mixture`, `check-hdbscan`, `check-resample`) are unchanged and stand where they stood (one Apple box in every case, AMD for none through these doors). The doors have since run: the 166-lane identity record (`bench/results/identity_break/2026-09-14_166-lanes/`) carries Apple M4, NVIDIA H100 and AMD MI325X train columns for cholesky, kernel-ridge, nystroem, rbf-sampler, gmm, hdbscan, bootstrap, permutation-test and monte-carlo; HDBSCAN's `prediction_data=True` and `mojolearn.hdbscan.approximate_predict` (2026-09-15) have Apple M4 Metal and CPU infer and batch cells only ([record](bench/results/identity_break/2026-09-15_hdbscan-predict/README.md)), their NVIDIA and AMD cells owed to the release record | Run the five surface tests on the M4 (`python3 -m mojolearn.tests.test_<x>_surface`), then the H100 and MI300X legs. Merge the lane bodies in `docs/lanes/LANE_BODY_*.py` into `tools/identity_break.py` and record three columns. Write a host twin for each family (none written). The `_NOT_YET` rows for these five families still stand in `python/mojolearn/__init__.py` and are the register owner's to delete. |
| Training primitives (`mojolearn.training.embedding_forward`, `embedding_backward`, `rms_norm_forward`, `rms_norm_backward`, `linear_forward`, `linear_backward`) and the k-means arms (`KMeans(metric=..., oversampling_factor=...)`) | Alpha (routed 2026-09-14) | The six primitives are the shipped training binding's own exports, previously unnamed by `mojolearn.training`. The embedding fold and the GEMM behind them are gated by their lanes' own checks, not by this routing. The k-means default's params list grew by one slot with its default value, so every recorded k-means cell keeps its bits. `metric='cosine'` was routed and refused by name on the Mojo host and was DELETED on 2026-09-18 (lane/kmeans-cosine-capability: cuVS refuses cosine k-means too, `kmeans_common.cuh:320`, and the arithmetic mean does not minimize cosine distance); an unsupported metric is still refused by name. `oversampling_factor=0.0` reaches the classic sequential k-means++ arm. Nothing here has run on any box since the routing | `python3 -m mojolearn.tests.test_training_primitives_surface` and `test_kmeans_metric_surface` on the M4, then lanes `training-primitives`, `kmeans-sqrt` and `kmeans-classic-pp` (`docs/lanes/LANE_BODY_training_primitives.py`, `LANE_BODY_kmeans.py`) on three columns. |
| IVF-FLAT (`IVFIndex`, `_mojolearn_ivf`) and the embedding table (`Embedding`, `_mojolearn_embedding`) | Alpha, from source (2026-09-14, lane `lane/expose-ivf-embedding`); both left `_NOT_YET` and are registered in `_backend` and every packaging list | Kernels: `pixi run check-ivf` reads ALL OK at IDENTICAL with one card (1e7c1702) on the Apple M4, an NVIDIA H100 and an AMD MI300X, and the embedding check's clause (a) card (c7f824c3, 6,887 cells) is byte-identical on the same three ([legs](bench/results/ivf_embed_km_legs_2026-09-14/README.md)). Embedding sabotage (`tools/embedding_sabotage_arm.sh`): after the five open findings of those legs were resolved in the check (two exact inert masks, the padding store's first stage, a by-add witness that separates by planted bits, a runnable clamp arm and a device flush probe), at 6c91eb1c3 all sixteen arms BIT on an NVIDIA H100 and an AMD MI300X, and on the Apple M4 (one core) fifteen BIT with `NO_FLUSH_ACC` asserted inert as contract 9.3 predicts ([verdicts](bench/results/embedding_sabotage_2026-09-14/README.md)). Through the doors: identity_break lanes `ivf` and `embedding` read IDENTICAL x3 on all 18 train, 18 held-out infer and 18 batch cells across the M4, the H100 and the MI300X, and `test_ivf_surface` and `test_embedding_surface` are GREEN on all three ([columns](bench/results/identity_break/2026-09-14_ivf-embedding/README.md)). On 2026-09-15 (lane `lane/embedding-owed`, ba4a108bb), the two remaining arms `EMB_FOLD_VIA_GEMM_ONEHOT` and `EMB_SORT_KEY_ID_ONLY_UNSTABLE` were built. All eighteen arms BIT on an H100 and an AMD MI325X; on the M4, seventeen BIT and `NO_FLUSH_ACC` is asserted inert. Clauses (a) to (f) of the embedding check PASS on all three, with the device nonfinite refusal (DEVIATION 1506) closed for the refusing entry points. `Embedding(plan="sort")` reaches PLAN_SORT, and lane `embedding-sort` reads IDENTICAL x3 on 54 of 54 cells ([evidence](bench/results/embedding_owed_2026-09-15/README.md)). `Embedding` refuses `max_norm`, `scale_grad_by_freq`, `sparse` and a missing `weight` by name; `IVFIndex` accepts L2Expanded and, since fix/ivf-l2sqrt (2026-09-14), L2SqrtExpanded (`metric='euclidean'`), which had returned all-zero distances because every norm was rooted: `check_l2_sqrt_is_the_root_of_l2` reads ALL OK with its own card (1250edd6) and the L2Expanded card still 1e7c1702 on the M4, an H100 and an MI300X, and the `ivf-euclidean` lane is IDENTICAL x3 on 9 train, 9 infer and 9 batch cells ([columns](bench/results/identity_break/2026-09-14_ivf-euclidean/README.md)) | A GPU-installed wheel. No speed claim |
| Byte-level BPE tokenizer (`BpeTokenizer`) | Alpha, from source: `bindings/build_tokenizer_host.sh` builds the CPU binding with the Unicode classes compiled in; mojolearn ships no vocabulary, the user loads one (`BpeTokenizer.from_files(encoder_json, vocab_bpe)`, `from_ranks_file`, `from_token_bytes`) | Held id for id to a second Python encoder over a synthetic vocabulary mojolearn trains itself (`python/mojolearn/_tokenizer_synthetic.py`), through `pixi run check-tokenizer` and `python/mojolearn/tests/test_tokenizer_surface.py` (2026-09-15, lane/clean-third-party); a reversed-ids sabotage build must fail the surface test. Host integers and tables only, no float arithmetic, so cross-vendor identity is by construction and is not a claim. The `tokenizer` identity_break lane moved to the synthetic vocabulary at `LANE_REVISIONS["tokenizer"]`, so its cells from the 166-lane record hashed older input and are owed | The `tokenizer` lane's Apple, NVIDIA and AMD cells at the new revision, at the next release record. |
| Mamba 1 | Experimental | Three-vendor operator/backward evidence at `718495cd`; baseline and L64 pass and match AMD/NVIDIA at `395d9421` | Frozen AMD installed checks pass all modes; NVIDIA IDENTICAL passes, with non-IDENTICAL sequence accuracy work remaining. Python zero-state IDENTICAL backward is now exposed; exact installed-artifact and broader backward qualification remain open. |
| Mamba 2 | Experimental | Three-vendor backward certificate at `718495cd`; baseline, L257 and incoming state pass and match AMD/NVIDIA at `395d9421` | Newer NVIDIA/AMD source forward/state checks passed, including the AMD binding fix. Frozen AMD installed checks pass all modes and NVIDIA IDENTICAL passes. Python zero-state IDENTICAL backward is now exposed; broader fixtures and exact installed-artifact qualification remain open. |
| Mamba 3 | Experimental | Baseline and L65 pass AMD/NVIDIA at `395d9421`; public gradients and complete retained diagnostics match by bits | L65 uses the explicit [compositional arithmetic contract](mamba/BACKWARD_CERTIFICATION.md), with independent whole-forward public gradients. Current Apple installed checks pass all modes/interpreters and corrected native baseline passes; Apple L65 remains RED as described above. Later NVIDIA source passes FAST/IDENTICAL surface checks; the frozen failing wheel is not relabeled. Zero-state IDENTICAL Python backward is exposed; final Linux wheel and broader backward qualification remain open. |
| Transformer block | Experimental | Three-vendor operator card for an earlier fixture | Frozen AMD installed API checks pass. NVIDIA lifetime fix passes IDENTICAL; the subsequent full-FP32 patch passes FAST/DETERMINISTIC. The current Apple wheel passes the full surface in all fifteen interpreter/mode jobs. Final Linux qualification and an independent corpus API oracle remain pending. |
| UMAP | Published 0.6.0 alpha API: `fit`, `fit_transform`, `transform` and CSR graph storage | Original three-vendor fixtures; expanded six-case IDENTICAL inputs/embeddings and 876 native stage cells match AMD/NVIDIA at `d88c7883` | All six expanded quality cases pass in all modes on AMD/NVIDIA after the self-neighbor fix. The frozen Linux candidates pass all installed UMAP modes and match six IDENTICAL input/embedding fixtures. The published alpha wheel has inherited native provenance; combined CUDA/HIP wheel qualification and larger-scale coverage remain open. |
| Training primitives and checkpointing | Experimental | Local correctness gates | Cross-vendor public-surface qualification remains open. |

The [AMD k-NN layout record](bench/results/e1/2026-09-05_215006-mojolearn-e2-amd/README.md)
retains all four arms, correctness bytes, rotating timing rounds and the
comparison with earlier NVIDIA and Apple evidence. Its speedups are scoped
to those fixtures and devices, not a general dispatch or scalability claim.

The [weighted CTR slice](bench/results/e1/2026-09-05_235251-amd-catboost-fixed-partition/README.md)
passes on AMD, including the production leaf estimator on fixed occupied
zero-mass partitions at L2=0 and L2=3. This is focused native correctness;
the numeric single-permutation [ordered RMSE path](gbdt/ORDERED_RMSE.md) now
passes both GPUs at `6dd44ac5`, including fresh weighted-CTR checks. The
[comparison](bench/results/resume/2026-09-06-ordered-mamba-knn/cross-device-continued.json)
also matches all 5,440 new adversarial kNN records in all four flag arms.

The [installed comparison](bench/results/resume/2026-09-06-installed-gap-closure/installed-lane-comparison.json)
adds six UMAP fixtures (24 arrays) and the full OrderedRMSE model plus 72
prediction cells matching across AMD/NVIDIA at `eb835021`. Both complete
candidate dispositions remain visible; lane agreement does not approve the
failed NVIDIA candidate for release. GBDT/OrderedRMSE saves now persist the
effective numeric mode at `29a8c848`, tested with changed process defaults in
lightweight checks and a separate NVIDIA native wrapper overlay. That newer
wrapper is exposed in the 0.6.0 alpha Python overlay; that does not recertify inherited native binaries.

`GradientBoosting.fit` accepts `group_id` (string or integer ids, each group's rows
consecutive). `loss="QueryRMSE"` reads it, on SymmetricTree with the greedy searcher, with no
bootstrap, categorical features or eval set; every other loss refuses a grouping by name in both
the GPU binding and the GBDT host binding, and `subgroup_id` and `pairs` are refused by name in
Python. Its identity lane `gbdt-query-rmse` has Apple M4 Metal and CPU columns; the NVIDIA and AMD
columns are owed to the next release record. The other querywise and pairwise losses and the
ranking metrics are listed in [gbdt/NOT_IMPLEMENTED.tsv](gbdt/NOT_IMPLEMENTED.tsv).

## What counts as certification

A cross-vendor result is accepted only when all of the following are recorded:

1. The same source state, fixture bytes, numeric profile, and card schema were
   used on each claimed vendor.
2. Each hardware leg actually ran and records device and toolchain provenance.
3. Stage tags, dtypes, element counts, and raw-bit hashes agree.
4. A sabotage or alternate-spelling arm demonstrates that the check detects the
   numerical mechanism it is intended to protect.
5. Refused configurations are reported as refusals, not passes.

`python -m mojolearn verify` is a useful local check against a shipped
reference card, but one local run cannot independently establish a
cross-vendor claim. See [verification](docs/VERIFY.md) and
[conformance bundles](docs/CONFORMANCE.md).

## Current priorities (2026-09-13)

- 0.8.5 is published (2026-09-14) with the `ExperimentalTwoLevelFeatureFreq` fix (DEVIATION 2710)
  and the AMD GEMM row; the next release carries whatever the lanes below merge.
- CPU training phase 1 (`docs/lanes/BRIEF_cpu_training_2026-09-13.md`): gemm-pinned first, then
  kde, holtwinters, lasso and elasticnet, svc (merged 2026-09-13, seven runners green), then
  agglomerative, et-clf, et-reg, iforest (phase 1b, 2026-09-14, IDENTICAL x4 on the Apple M4 host
  path, the seven-runner run owed); a lane passes only when the fourth column reads IDENTICAL on
  every cell. Still owed there: the isolation forest's GPU-side model export.
- DONE 2026-09-14: the 46-lane three columns rerun with every byte level language model cell
  filled (`bench/results/identity_break/2026-09-14_46-lanes/`).
- CPU inference for the classical lanes, per the costing in
  `docs/lanes/BRIEF_forest_host_inference_2026-09-13.md`: ols, ridge, tsvd, logistic and pca
  merged 2026-09-13 with three-vendor recordings 2026-09-14; kde, svc, the whitened pca and
  knn, knn-clf, knn-reg merged 2026-09-14 (Apple M4 measured, NVIDIA and AMD recordings owed);
  iforest not started.

The project-level sequencing lives in [ROADMAP.md](ROADMAP.md). Update this
page only from recorded evidence; do not turn planned or in-progress runs into
support claims.

## UMAP and Mamba closure (2026-09-05)

`UMAP.fit` and `fit_transform` expose the supported dense Euclidean 2D/3D
slice in the published macOS 0.5.0 wheel. Its installed API passed in all
three modes on Python 3.10–3.14. The downloaded PyPI artifact matched the
publication digest and passed additional API checks; see the
[post-publication record](bench/results/wheels/2026-09-05-umap-api/postpublish/results.json).
A refreshed Linux wheel and NVIDIA/AMD installed-artifact qualification remain pending.

At `718495cd`, Apple M4, NVIDIA RTX 4090, and AMD MI300X matched all 54
native Mamba backward gradient tensors across five cases and all 186 cells
in the named UMAP fixture. See the [three-vendor record](bench/results/e1g/2026-09-05_042552-amd-mamba/cross-device.json).
These source certificates do not certify other shapes or installed artifacts.
The Mamba3 correctness claim from that historical certificate is superseded:
its staged reference shared a missing scale/gamma chain-rule contribution.
The [corrected AMD/NVIDIA record](bench/results/resume/2026-09-05-next-certification/corrected-backward-cross-device.json)
passes all five baseline cases and matches 54 gradient tensors, while requiring
independent whole-forward gradients for every Mamba3 public leaf. Its long
profile was RED on intermediate checks despite matching public tensors. The
[September 6 certificate](bench/results/resume/2026-09-06-ordered-mamba-knn/README.md)
now passes both profiles on AMD/NVIDIA under the documented compositional
arithmetic policy, retaining all intermediate outputs and direct differences.
The later AMD allocation fix has its own successful source Python API rerun;
see the [AMD source qualification record](bench/results/e1/2026-09-05_134041-mojolearn-e2-amd/README.md).
That later binding evidence is separate from the original native certificate.

## UMAP 0.6.0 release candidate (historical, 2026-09-05)

0.6.0 was published on 2026-09-06 and the wheels since (0.7.0 through 0.8.4) supersede this
candidate; the section stays as the record of what that candidate's evidence covered.

The source candidate implements `fit`, `fit_transform` and `transform` with
dense Euclidean input and two- or three-dimensional output. Public fitting
now stores the fuzzy graph in CSR form using O(n_samples × n_neighbors)
space. This does not add sparse input support or approximate neighbors:
exact neighbor search still performs quadratic pair comparisons.

`transform` embeds new samples against the frozen fitted embedding. Private
training and embedding copies add O(n_samples × (n_features + n_components))
storage. Changes to parameters or numeric mode require refitting; changing
query batching does not change results. Supervised targets and alternate
metrics or initialization remain unsupported.

| Evidence layer | Completed evidence | Still pending |
|---|---|---|
| Published 0.5.0 macOS artifact | Dense fit/fit_transform installed-wheel checks | Transform and CSR are not in this published artifact |
| Source transform and sparse native paths before public CSR integration | Named held-out transform embeddings match Apple, NVIDIA and AMD; separate sparse native gates passed | This earlier source does not certify the later public integration |
| Integrated public CSR fit/transform source | All three numeric modes passed API and held-out quality checks on Apple, NVIDIA and AMD; named IDENTICAL held-out layouts match | Broader shapes and metrics remain outside this evidence |
| 0.6.0 release candidate artifact | macOS candidate passed Python 3.10–3.14/mode smoke and clean Python 3.12 UMAP fit/transform/quality checks in all modes | Publication and Linux installed-artifact checks |

Evidence: [Apple public integration](bench/results/umap/2026-09-05-public-sparse/README.md),
[AMD integrated API results](bench/results/e1/2026-09-05_142553-mojolearn-e2-amd/diag/followup/results.tsv),
and [earlier NVIDIA transform and sparse qualification](bench/results/e1g/2026-09-05_095508-nvidia-mamba/README.md).
The latter records frozen source `72da212b`, whose public fit remained dense;
public CSR integration landed separately at `274ba7c0`. The later AMD
integration ran source `e2cfe1ac`. Keep these revisions and source hashes with
their results rather than relabeling them as one release certificate.

The original 186-cell UMAP certificate and 54-tensor Mamba backward
certificate above retain their original counts and scope. New transform
quality/identity checks are separate evidence, not additions to those totals.
The k-NN optimization remains experimental and is not enabled in normal
wheel builds.

Latest evidence: [integrated NVIDIA qualification and native public k-NN pricing](bench/results/e1g/2026-09-05_103918-nvidia-mamba/README.md),
and [exact macOS 0.6.0 candidate qualification](bench/results/wheels/2026-09-05-umap-060/README.md).
The flag-gated k-NN selector improved paired median native request times by
2.99x/5.99x/18.86x on the named NVIDIA 32/128/1000-query fixtures; these are
not cuML comparisons or a claim that every algorithm improved.

## September 6 bounded follow-ups and alpha exposure

The [public alpha guide](python/mojolearn/ALPHA_API.md) lists ordinary module
imports, existing optimizers/losses, fixed trainers and required native bindings.
Alpha exposure does not certify missing binaries or unfinished algorithms.
Ordered RMSE is not a ranking objective; see the
[tree feature inventory](docs/TREE_ALPHA_FEATURE_STATUS.md).

The [NVIDIA MLP campaign](bench/results/resume/2026-09-06-root-training-nvidia/README.md)
passed all 18 jobs at `8f6ed41`, including independent reference/edge checks,
16-step learning and complete same-device checkpoint continuation. The
[September 7 AMD MLP run](bench/results/resume/2026-09-07-root-mlp-amd/README.md)
also passed all 18 jobs; all 16 complete raw training steps match NVIDIA.
MLP Metal and foreign-vendor MLP resume remain open. The two-block,
34,944-parameter byte LM passed
all 12 single-vendor jobs on NVIDIA and AMD at `d921eade`; all 128 retained
steps, heldout bytes and final checkpoints match between these two runs.
Heldout loss fell 5.5413 → 2.8436. The final
[raw comparator](bench/results/resume/2026-09-06-root-byte-lm-cross-vendor/README.md)
now admits checkpoint continuation in both NVIDIA/AMD directions with effective
optimizer-state controls. Metal and step costs remain open.

The [NVIDIA digits experiment](bench/results/resume/2026-09-06-root-umap-nvidia/README.md)
measured real-data neighborhood trustworthiness and retention against
umap-learn, plus a separate matched cuML timing workload. All 15 jobs passed.
This closes that bounded real-data experiment, not arbitrary dataset or
cross-vendor coverage. The
[missing-vendor inventory](docs/MISSING_VENDOR_EVIDENCE_QUEUE.md) distinguishes
existing ARIMA fitted API passes and other smoke/timing results from the full
identity cells still owed.
