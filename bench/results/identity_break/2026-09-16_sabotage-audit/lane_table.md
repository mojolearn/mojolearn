| lane | cat | host family | CPU-covered | evidence / reason |
|---|---|---|---|---|
| `rf-clf` | b | rf | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `rf-reg` | b | rf | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `et-clf` | a | trees | yes | moved: infer,model,train; UNMOVED: infer,model,train |
| `et-reg` | a | trees | yes | moved: infer,model,train; UNMOVED: infer,model,train |
| `gbdt-symmetric` | b | gbdt | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `gbdt-depthwise` | b | gbdt | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `gbdt-lossguide` | b | gbdt | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `gbdt-rmse` | b | gbdt | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `kmeans` | a | core | yes | moved: batch,infer,train |
| `knn` | a | core | yes | moved: batch,infer,train; UNMOVED: model |
| `knn-clf` | a | core | yes | moved: batch,infer,train; UNMOVED: model |
| `knn-reg` | a | core | yes | moved: batch,infer,train; UNMOVED: model |
| `dbscan` | a | estimators | yes | moved: batch,infer; UNMOVED: model,train |
| `pca` | a | estimators | yes | moved: batch,infer,model,train |
| `pca-whiten` | a | estimators | yes | moved: batch,infer,model,train |
| `tsvd` | a | estimators | yes | moved: batch,infer,model,train; UNMOVED: model |
| `ols` | a | estimators | yes | moved: batch,infer,model,train |
| `ridge` | a | estimators | yes | moved: batch,infer,model,train |
| `logistic` | a | estimators | yes | moved: batch,infer,model,train |
| `lasso` | a | estimators,solver | yes | moved: batch,infer,model,train |
| `elasticnet` | a | estimators,solver | yes | moved: batch,infer,model,train |
| `svc` | a | svm | yes | moved: batch,infer,model,train |
| `kde` | a | estimators | yes | moved: batch,infer,train; UNMOVED: model |
| `agglomerative` | a | solver | yes | moved: batch,infer,train; UNMOVED: model,train |
| `spectral` | a | metrics | yes | moved: batch,infer; UNMOVED: model,train |
| `holtwinters` | a | forecast,tsa | yes | moved: batch,infer,train; UNMOVED: batch,infer,train |
| `gemm-pinned` | a | linalg | yes | moved: batch,train; UNMOVED: batch,train |
| `metrics` | a | metrics | yes | moved: train; UNMOVED: train |
| `svr` | a | svm | yes | moved: batch,infer,model,train |
| `arima` | a | arima,forecast | yes | moved: batch,infer,model,train |
| `gp` | a | gp,gp_infer | yes | moved: batch,infer,train |
| `gpc` | a | gp,gp_infer | yes | moved: batch,infer,model,train |
| `gpc-multiclass` | a | gp,gp_infer | yes | moved: batch,infer,model,train |
| `umap` | a | metrics | yes | moved: infer,model,train |
| `radius` | a | core | yes | moved: batch,infer,train; UNMOVED: batch,infer,model,train |
| `standard-scaler` | a | estimators,preprocessing | yes | moved: batch,infer,model,train |
| `minmax-scaler` | a | estimators,preprocessing | yes | moved: batch,infer,model,train |
| `gbdt-ordered-rmse` | a | gbdt | yes | moved: infer,model; UNMOVED: batch,model,train |
| `gbdt-feature-freq` | a | gbdt | yes | moved: infer,model; UNMOVED: batch,model,train |
| `mlp` | a | training | yes | moved: batch,infer; UNMOVED: model,train |
| `byte-lm` | a | byte_lm | yes | moved: batch,infer,model,train |
| `byte-lm-host-infer` | b | byte_lm | yes | arm exists (MOJOLEARN_BYTE_LM_HOST_SABOTAGE); no committed column shows it move; harness batch-env only |
| `byte-lm-host-train` | b | byte_lm | yes | arm exists (MOJOLEARN_BYTE_LM_HOST_SABOTAGE); no committed column shows it move |
| `mamba1` | a | mamba | yes | moved: batch,infer,train; UNMOVED: infer,train |
| `mamba2` | a | mamba | yes | moved: batch,infer,train; UNMOVED: train |
| `mamba3` | a | mamba | yes | moved: batch,infer,train; UNMOVED: train |
| `transformer` | a | transformer | yes | moved: batch,infer,train; UNMOVED: train |
| `samba` | a | training | yes | moved: batch,infer,model,train; UNMOVED: model,train |
| `rf-clf-entropy-log2-noboot` | b | rf | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `rf-clf-balanced-parallel` | b | rf | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `rf-reg-poisson` | b | rf | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `rf-reg-gamma-ig` | b | rf | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `et-clf-entropy-bestfirst` | b | trees | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `et-reg-bootstrap-parallel` | b | trees | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `gbdt-multiclass` | b | gbdt | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `gbdt-onevsall` | b | gbdt | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `gbdt-parametric-losses` | b | gbdt | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `gbdt-lossguide-newtoncosine` | b | gbdt | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `gbdt-pointwise-l2-bayesian-eval` | a | gbdt | yes | moved: infer,model; UNMOVED: batch,model,train |
| `gbdt-exact-mae` | b | gbdt | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `gbdt-categorical-ctr` | a | gbdt | yes | moved: infer,model; UNMOVED: batch,model,train |
| `gbdt-categorical-ctr-tables` | a | forest | yes | moved: batch,infer,train; UNMOVED: model |
| `gbdt-tensor-ctr-tables` | a | forest | yes | moved: batch,infer,train; UNMOVED: model |
| `gbdt-nan-modes` | b | gbdt | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `gbdt-adapter-clf` | b | gbdt | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `gbdt-adapter-reg` | b | gbdt | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `gbdt-query-rmse` | a | gbdt | yes | moved: batch,infer,model,train |
| `gbdt-pair-logit` | a | gbdt | yes | moved: batch,infer,model,train |
| `gbdt-yeti-rank` | a | gbdt | yes | moved: batch,infer,model,train |
| `gbdt-adapter-score-weighted` | b | gbdt | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `rf-score-weighted` | b | rf | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `mamba2-dtlimit` | a | mamba | yes | moved: batch,infer,train; UNMOVED: train |
| `transformer-window` | a | transformer | yes | moved: batch,infer,train; UNMOVED: train |
| `byte-lm-resident` | a | byte_lm | yes | moved: batch,infer,model,train |
| `byte-lm-host-infer-threaded` | b | byte_lm | yes | arm exists (MOJOLEARN_BYTE_LM_HOST_SABOTAGE); no committed column shows it move |
| `samba-untied-dropout-accum` | a | training | yes | moved: batch,infer,model,train; UNMOVED: model,train |
| `optim-sgd` | b | training | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `optim-adam-clip` | b | training | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `cross-entropy-arms` | b | training | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `kmeans-random` | a | core | yes | moved: batch,infer,train |
| `kmeans-array` | a | core | yes | moved: batch,infer,train |
| `kmeans-weighted` | a | core | yes | moved: batch,infer,train |
| `dbscan-brute-l1` | a | estimators | yes | moved: batch,infer; UNMOVED: model,train |
| `dbscan-weighted` | a | estimators | yes | moved: batch,infer; UNMOVED: model,train |
| `kde-tophat-sqeuclidean` | a | estimators | yes | moved: batch,infer,train; UNMOVED: model |
| `kde-epanechnikov-l1` | a | estimators | yes | moved: batch,infer,train; UNMOVED: model |
| `kde-exponential-chebyshev` | a | estimators | yes | moved: batch,infer,train; UNMOVED: model |
| `kde-linear-cosine` | a | estimators | yes | moved: batch,infer,train; UNMOVED: model |
| `kde-cosine-minkowski` | a | estimators | yes | moved: batch,infer,train; UNMOVED: model |
| `kde-weighted` | a | estimators | yes | moved: batch,infer,train; UNMOVED: model |
| `pca-full-whiten` | a | estimators | yes | moved: batch,infer,model,train |
| `ols-no-intercept` | a | estimators | yes | moved: batch,infer,model,train |
| `ols-weighted` | a | estimators | yes | moved: batch,infer,model,train |
| `ridge-no-intercept` | a | estimators | yes | moved: batch,infer,model,train |
| `logistic-l1` | a | estimators | yes | moved: batch,infer,model,train |
| `logistic-multiclass` | a | estimators | yes | moved: batch,infer,model,train |
| `logistic-elasticnet` | a | estimators | yes | moved: batch,infer,model,train |
| `logistic-unpenalized-no-intercept` | a | estimators | yes | moved: batch,infer,model,train |
| `elasticnet-l2end-no-intercept` | a | estimators,solver | yes | moved: batch,infer,model,train |
| `svc-linear` | a | svm | yes | moved: batch,infer,model,train |
| `svc-poly` | a | svm | yes | moved: batch,infer,model,train |
| `svr-linear` | a | svm | yes | moved: batch,infer,model,train |
| `knn-sqeuclidean` | a | core | yes | moved: batch,infer,train; UNMOVED: model |
| `knn-manhattan` | a | core | yes | moved: batch,infer,train; UNMOVED: model |
| `knn-chebyshev` | a | core | yes | moved: batch,infer,train; UNMOVED: model |
| `knn-cosine` | a | core | yes | moved: batch,infer,train; UNMOVED: batch,infer,model |
| `knn-minkowski-p3` | a | core | yes | moved: batch,infer,train; UNMOVED: model |
| `knn-rbc` | a | core | yes | moved: batch,infer,train; UNMOVED: batch,infer,model,train |
| `knn-clf-distance` | a | core | yes | moved: batch,infer,train; UNMOVED: model |
| `knn-reg-distance` | a | core | yes | moved: batch,infer,train; UNMOVED: model |
| `radius-manhattan` | a | core | yes | moved: batch,infer,train; UNMOVED: batch,infer,model,train |
| `radius-chebyshev` | a | core | yes | moved: batch,infer,train; UNMOVED: model |
| `radius-minkowski-p3` | a | core | yes | moved: batch,infer,train; UNMOVED: model |
| `standard-scaler-no-mean` | a | estimators,preprocessing | yes | moved: batch,infer,model,train |
| `standard-scaler-no-std` | a | estimators,preprocessing | yes | moved: batch,infer,model,train |
| `minmax-scaler-clip` | a | estimators,preprocessing | yes | moved: batch,infer,model,train |
| `spectral-precomputed` | a | metrics | yes | moved: batch,infer; UNMOVED: model,train |
| `holtwinters-multiplicative` | a | forecast,tsa | yes | moved: batch,infer,train; UNMOVED: batch,infer,train |
| `kpss` | a | tsa | yes | moved: batch,train |
| `arima-011` | a | arima,forecast | yes | moved: batch,infer,model,train |
| `arima-seasonal-c` | a | arima,forecast | yes | moved: batch,infer,model,train |
| `arima-exog` | b | arima,forecast | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `arima-exog-seasonal` | b | arima,forecast | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `gp-normalize-y` | a | gp,gp_infer | yes | moved: batch,infer,train |
| `gp-sample-y` | a | gp | yes | moved: infer,train |
| `gp-sample-y-normalize` | a | gp | yes | moved: infer,train |
| `gp-optimize` | c | - | no | no host family and no host sabotage define |
| `gp-optimize-restarts` | c | - | no | no host family and no host sabotage define |
| `gp-matern12` | a | gp,gp_infer | yes | moved: batch,infer,train |
| `gp-matern32` | a | gp,gp_infer | yes | moved: batch,infer,train |
| `gp-matern52-ard` | a | gp,gp_infer | yes | moved: batch,infer,train |
| `gemm-transposed` | a | linalg | yes | moved: batch,train; UNMOVED: batch,train |
| `metrics-classification` | a | metrics | yes | moved: batch,train; UNMOVED: batch |
| `metrics-fowlkes-mallows` | a | metrics | yes | moved: train |
| `tokenizer` | a | tokenizer | yes | moved: batch; UNMOVED: infer,train |
| `cross-val` | b | gbdt | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `cholesky` | a | linalg | yes | moved: batch,infer,model,train; UNMOVED: batch,infer |
| `kernel-ridge` | a | estimators,kernel_methods | yes | moved: batch,infer,model,train |
| `nystroem` | a | estimators,kernel_methods | yes | moved: batch,infer,model,train |
| `rbf-sampler` | a | estimators,kernel_methods | yes | moved: batch,infer,train; UNMOVED: model |
| `gmm` | a | mixture,mixture_infer | yes | moved: batch,infer,model,train |
| `gmm-random-init` | a | mixture,mixture_infer | yes | moved: batch,infer,model,train; UNMOVED: batch,infer |
| `gmm-sample` | a | mixture,mixture_infer | yes | moved: infer,train |
| `gmm-random-init-sample` | a | mixture,mixture_infer | yes | moved: infer,train |
| `hdbscan` | a | hdbscan,hdbscan_infer | yes | moved: batch,infer,model,train |
| `hdbscan-leaf` | a | hdbscan,hdbscan_infer | yes | moved: batch,infer,model,train |
| `bootstrap` | a | resample | yes | moved: batch,train |
| `permutation-test` | a | resample | yes | moved: batch,train |
| `monte-carlo` | a | resample | yes | moved: train |
| `training-primitives` | b | training | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `ivf` | a | ivf,ivf_search | yes | moved: batch,infer,model,train; UNMOVED: batch,infer,train |
| `ivf-euclidean` | a | ivf,ivf_search | yes | moved: batch,infer,model,train |
| `ivf-extend` | a | ivf,ivf_search | yes | moved: batch,infer,model,train |
| `embedding` | a | embedding,embedding_infer | yes | moved: batch,infer,train |
| `embedding-sort` | a | embedding | yes | moved: batch,infer,train |
| `kmeans-sqrt` | a | core | yes | moved: batch,infer,train |
| `kmeans-classic-pp` | a | core | yes | moved: batch,infer,train |
| `kmeans-cosine` | b | core | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move; arm RAN, did NOT move: train |
| `par-forest` | b | rf | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `par-forest-et` | b | trees | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `par-boosting` | c | - | no | no host family and no host sabotage define |
| `par-kmeans` | c | - | no | no host family and no host sabotage define; harness batch-env move only |
| `par-gram` | c | - | no | no host family and no host sabotage define |
| `par-logistic` | c | - | no | no host family and no host sabotage define |
| `par-cd` | c | - | no | no host family and no host sabotage define |
| `par-svm` | c | - | no | no host family and no host sabotage define |
| `par-gp` | c | - | no | no host family and no host sabotage define |
| `par-dbscan` | c | - | no | no host family and no host sabotage define; harness batch-env move only |
| `par-scaler` | b | preprocessing | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `par-arima` | b | arima | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `par-mlp` | b | training | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `par-samba` | c | - | no | no host family and no host sabotage define |
| `par-byte-lm` | c | - | no | no host family and no host sabotage define |
| `par-queries-knn` | b | core | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move; harness batch-env only |
| `par-queries-radius` | b | core | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `par-queries-kde` | b | estimators | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `par-reference-knn` | b | core | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `par-reference-knn-reg` | b | core | yes | arm exists (MOJOLEARN_HOST_SABOTAGE); no committed column shows it move |
| `par-graph-agglomerative` | c | - | no | no host family and no host sabotage define; harness batch-env move only |
| `par-graph-spectral` | c | - | no | no host family and no host sabotage define |
| `par-graph-umap` | c | - | no | no host family and no host sabotage define |
| `par-ordered-rmse` | c | - | no | no host family and no host sabotage define |
| `par-feature-freq` | c | - | no | no host family and no host sabotage define |
| `par-boosting-pointwise` | c | - | no | no host family and no host sabotage define |
| `par-holtwinters` | a | tsa | yes | moved: batch,infer,train |
| `par-byte-lm-model-pool` | c | - | no | no host family and no host sabotage define |
| `par-byte-lm-offload` | c | - | no | no host family and no host sabotage define |
| `par-samba-clip` | c | - | no | no host family and no host sabotage define |
| `iforest` | a | svm | yes | moved: batch,infer,train; UNMOVED: model |
| `iforest-tuned` | a | svm | yes | moved: batch,infer,model,train |
| `par-iforest` | c | - | no | no host family and no host sabotage define |
| `par-forest-pool` | c | - | no | no host family and no host sabotage define |
| `par-gmm` | c | - | no | no host family and no host sabotage define |
| `par-resample` | c | - | no | no host family and no host sabotage define |
| `par-hdbscan` | c | - | no | no host family and no host sabotage define |
| `par-cholesky` | c | - | no | no host family and no host sabotage define |
| `par-kernel-ridge` | c | - | no | no host family and no host sabotage define |
| `par-nystroem` | c | - | no | no host family and no host sabotage define |
| `par-rbf-sampler` | c | - | no | no host family and no host sabotage define |
