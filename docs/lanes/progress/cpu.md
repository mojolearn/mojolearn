# cpu: progress

Lane `cpu` (CURRENT DIRECTIVES 1c + LANE CHARTER): the CPU-path audit, gap
fixes and shared CPU infrastructure. Per-family CPU speed belongs to each
family lane's phase 5 (handed off below).

## Where this lane stands (2026-09-28, session 2)

**NOT YET ON MAIN (session 3, 2026-09-28 ~05Z).** lane/cpu f1cf0a323 holds
the session-1 fixes plus the host FP-environment unification, merged with
origin/main 3fa29cd1f (IDENTITY_PATHS row renumbered 199 -> 202: the
cluster lane took 199-201). The site check on the merged tree: PASS (no new
raw `sync_parallelize`, no `host_fp_env` user; lane/trees-cpu's
`flush_subnormals` border search is pinned arithmetic, not an environment
module). The H100 proof below is of 41f60919d; the merge with main is the
OWED gate.

**OWED GATE (RunPod out of money 2026-09-28 ~04:50Z, pod dpse0qp7knpu44
gone, no pod can be rented).** Running instead on the central AMD box
(`tools/amd_central.sh`, trees `/root/mojolearn-cpu` = lane/cpu and
`/root/mojolearn-cpu-base` = origin/main 3fa29cd1f, scripts and outputs in
`/root/ev-cpu/`, copies in ~/mojolearn-evidence/cpu/amd_gate_*.sh):
1. `amd_gate_cpu.sh` (under `sh`, no slot): builds both trees, then CPU
   columns base default, new default / 1 / 3 threads (`cpu_done` when done).
   Pass = new vs base IDENTICAL except the known logistic cell; new at
   1 / 3 / default IDENTICAL.
2. `amd_gate_gpu.sh` under ONE slot (`amd_central.sh run cpu ...`): HIP
   columns base then new (`gpu_done`). Pass = HIP new vs base IDENTICAL; HIP
   new vs CPU new AGREE.
3. `pixi run check-host-parallel` PASS and the sabotage
   (patches/host_parallel_fp_env.patch) FAILS, on the new tree.
4. test_host_surface.
5. STILL OWED WHEN RUNPOD IS FUNDED: the NVIDIA column (CUDA new vs base,
   CUDA vs CPU). The gate rule is NVIDIA + CPU; merge waits on the
   orchestrator's call whether AMD + CPU stands in.

The WIP par-* fallback (below) is parked on branch lane/cpu-par-fallback
(pushed, unrun) so it stays out of this gate.

| commit | what |
|---|---|
| d49fb66b + 059d3661f | `PCA(whiten=False).inverse_transform` and `TruncatedSVD.inverse_transform` on every CPU-only install (`core/classical_host_predict.mojo::host_inverse_transform_into`); lanes `pca-inverse`, `tsvd-inverse`, `pca-whiten-inverse` (PENDING "no reference"). Sabotage arms bit in session 1 (patches in ~/mojolearn-evidence/cpu/patches/). |
| 762f811cc, 81b443e9e | DEVIATION 5900, `core/host_parallel.mojo::host_parallelize` (tasks run in the caller's MXCSR/FPCR; Mojo's workers run FTZ+DAZ). |
| 41f60919d | **ONE module.** `core/host_parallel.mojo` is the only host thread split and absorbs lane/algos-linear's `core/host_fp_env.mojo` (it reached main at 0b7b6d5c1; the merge of lane/cpu removed it and its ten `host_ieee_fp_enter/leave` lines in classical_host_predict and glm/estimator, which were inert inside `host_parallelize`: the caller's environment is already IEEE). `host_parallelize` everywhere, now also svm_parameter's finite scan, the byte LM host rows and its exp check, and the multi-GPU drivers' per-device tasks. `host_parallelize_pool_env` (= the worker's FTZ/DAZ) for the GBDT fit's host regions only (gbdt/train.mojo, gbdt/resident_model.mojo, gbdt/host/gbdt_oracle.mojo): their recorded columns carry the pool's bits. `tools/check_host_parallel_sites.py` refuses a raw `sync_parallelize` anywhere else and runs first in `pixi run check-host-parallel`. IDENTITY_PATHS row 202. |

**Proof on the H100 pod (dpse0qp7knpu44), session 2.** Two trees built from
scratch: `/root/base` = merge base f237f1996 (main), `/root/mojolearn` =
lane/cpu 41f60919d. identity_break, every lane, repeats 1 (482 lanes; CPU
columns include par-*, the CUDA column excludes them). Records in
`/root/audit3/` on the pod (`d_*.txt` are the diffs).
- **CPU new vs CPU main (default threads):** every cell IDENTICAL except
  `logistic-unpenalized-no-intercept/dupes` infer + batch (DEVIATION 5900's
  cell) and the three new inverse lanes (ONE-COLUMN).
- **That cell was the ONLY thread-count-dependent cell on main:** main's CPU
  column at MOJOLEARN_CPU_THREADS=1 vs default differs there and nowhere
  else (train 4014, infer/model 5075, batch 3239 IDENTICAL).
- **CPU new at 1, 3 and default threads:** every cell IDENTICAL (train 4041,
  infer/model 5130, batch 3267).
- **CUDA new vs CUDA main:** every cell IDENTICAL (train 3789, infer/model
  4680, batch 3024, rlpair 180, batchgrad 144). The 74 GBDT `denormal` cells
  that moved at 762f811cc are back: that was the OWED re-run of 81b443e9e.
- **CUDA new vs CPU new:** every compared cell IDENTICAL; the logistic cell
  now agrees (it was DIVERGENT on main). pca-inverse, tsvd-inverse,
  pca-whiten-inverse: CPU == CUDA on all 9 fixtures.
- Seam `pixi run check-host-parallel`: PASS (caller 0x1fa0). Sabotage (drop
  `host_fp_env_set(env)`, patches/host_parallel_fp_env.patch): FAIL (task
  results 0.0 != 8.33e-309, env 0x9ff0 != 0x1fa0); reversed: PASS. Site
  sabotage (svm_parameter back to `sync_parallelize`): the site check FAILS
  naming svm_parameter.mojo:317. On origin/main the site check names 37 raw
  sites.
- byte LM exhaustive exp check (2^32 patterns, now on host_parallelize): PASS.
- test_host_surface 200 passed; test_lane_select 69 passed (six of them
  with MOJOLEARN_LANE_SELECT_TEST_FORCE=1: the pod tree's host_surface.py
  differs from its merge base, which is the lane's own change; the manifest
  was byte-identical after the run).

**For lane/algos-linear:** merge main. `core/host_fp_env.mojo` is gone;
`core/classical_host_predict.mojo` and `glm/estimator.mojo::qn_softmax_host`
split through `host_parallelize`. Any new host loop uses it; the site check
refuses anything else.

## Next session, in order

0. The OWED GATE above; merge lane/cpu to main and push; tell main.
1. The one-device plain fallback for the 31 par-* lanes that refuse on a
   CPU-only install (unrun WIP on branch lane/cpu-par-fallback: pool
   `CPU_SINGLE_DEVICE_PLAIN`, worker `_cpu_plain`, host_surface names;
   merge it onto lane/cpu after the gate).
2. Probe-recipe gaps in tools/cpu_path_audit.py (82 rows fail on BOTH
   columns; list in "The audit").
3. Keep the audit table current.
4. The GBDT environment question (lane trees' call, with a column
   re-record): the GBDT fit's host regions keep the pool's FTZ/DAZ through
   `host_parallelize_pool_env`; gbdt_oracle's serial small-fit arm
   (`n_rows * n_features < 2^18`) runs on the calling thread (IEEE). On the
   recorded fixtures GPU and CPU agree; a small fit whose border search reads
   a subnormal feature is exposed.

## The audit (phase 1, step 1)

Tool: `tools/cpu_path_audit.py` (record / diff / table). It fits every public
name (mojolearn.__all__, the expansion doors, and the public submodules'
`__all__`) on a small non-uniform fixture on the GPU and on the CPU host route
(MOJOLEARN_VENDOR=cpu, the lane check's CPU env), calls every inference method,
and compares sha256 of the outputs. Raw records:
~/mojolearn-evidence/cpu/probe_{gpu,cpu}.jsonl (RTX 4090 pod, commit 993e58c9).

**Findings, read against the rule "a gap is CPU-fails-where-GPU-works":**

- **CPU gaps in existing code: one, fixed.** PCA / TruncatedSVD `inverse_transform` (above).
- **CPU != GPU in existing code: one, fixed.** LogisticRegression predict_proba at the default thread count (above; found by the full harness columns, not the probe).
- **Every other existing public algorithm** with a fit or inference call trains and infers on the CPU route and its probe bits EQUAL the GPU's (146 names EQUAL, 0 DIFFER). Existing names with no probe (neural blocks, trainers, optimizers, parallel drivers, tokenizers, host_* helpers) are covered by their admitted lanes (column "lanes"); the full harness CPU column read every such lane STABLE and IDENTICAL to CUDA.
- **par-* lanes on a CPU-only install** (59 lanes, one device): 25 run and read STABLE (par-arima, par-forest*, par-gpc-*, par-ivf, par-mlp, par-samba*, par-scaler*, the par-queries/reference neighbor drivers, par-causal-lm, par-cross-val, par-forecast-*, par-holtwinters, par-rbf-sampler). 31 refuse BY NAME ("no CPU implementation of the cooperative multi-GPU driver <op> yet": gbdt_fit, solver_fit, cholesky_fit, dbscan_fit, forest_prepare, gmm_fit, gp_fit, gram_fit, graph_fit, hdbscan_fit, iforest_fit, km_fit, kmeans_fit, glm_fit, ordered_rmse_fit, resample, svm_fit), and the three par-byte-lm* lanes refuse ("rebuild bindings/build_byte_lm.sh for parallel training"). These are MULTI-DEVICE drivers; `PUBLIC_INAPPLICABLE_PREFIX_REASONS["par-"]` already states their claim needs two devices. A CPU-only user calling e.g. `parallel_classical.fit_kmeans(..., devices=[0])` gets the refusal instead of the plain host fit. **Open, for the next session:** route a ONE-device cooperative call on a CPU-only install to the plain estimator's host fit (bits equal to the plain fit by the par lanes' own `_mismatch_bytes` invariant), the way `CPU_SINGLE_DEVICE_COOPERATIVE` already does for mlp_update/samba_update.
- **Rows failing on BOTH columns are probe-recipe bugs, never CPU gaps** (the orchestrator fixed four in e73eb1bd6). Remaining such rows: ExperimentalTwoLevelFeatureFreq, OrderedRMSE, and most x_* rows (constructor args, 3-D inputs, endog). They say nothing about the CPU.

### Expansion (x_*) findings, for the owning lanes (not fixed here)

- **x_cluster GPU binding hangs on its second call in a process** (MiniBatchKMeans, BisectingKMeans, MeanShift, AffinityPropagation, BayesianGaussianMixture, OPTICS twice): `_expansion_cluster.py:58` never returns; `x_cluster/device_ops.mojo` builds a DeviceContext per call. Reported to main. Same TIMEOUT on the GPU for x_neighbors (LocalOutlierFactor, NearestCentroid, OneClassSVM, KernelPCA, LabelPropagation, LabelSpreading, SVGP), x_sequence (MLPClassifier, MLPRegressor, CrostonOptimized) and x_trees (DecisionTreeClassifier) when run after other names in one process. The CPU route ran all of them.
- sequence: `Theta`, `OptimizedTheta`, `DynamicTheta`, `DynamicOptimizedTheta`, `AutoTheta`, `CrostonClassic/Optimized/SBA`, `ETS`, `DampedETS`: `predict(X)` raises `TypeError: only 0-dimensional arrays can be converted to Python scalars` on both columns (a sklearn-shaped call with an array; their API is forecast(h), so possibly a probe mismatch; the lane should say which).
- sequence: `layer_norm_backward`, `Theta`/`OptimizedTheta`/`DynamicTheta` and decomp `johnson_lindenstrauss_min_dim`, `sparse_encode` have no identity lane naming them.
- On the expansion lanes' CPU columns: the full CPU harness column read every x-*/sequence-* cell STABLE; their CUDA columns were not run here because of the hang above.

## Next phases (lane cpu)

- Phase 2, shared CPU infrastructure: `core/host_parallel.mojo` is the one host thread split (FP environment pinned); family lanes use it for their CPU speed phase. Candidates: a shared blocked host GEMM cell loop (`gemm/host/gemm_oracle.mojo` and `core/classical_host_predict.mojo::host_pinned_cell_ptr` are serial per cell), a row-task helper that every `_rows(c)` closure repeats, and thread-count proof tooling (`MOJOLEARN_CPU_THREADS=1` vs default column diff, as run above).
- **Identity at every thread count is a rule, and now proven once**: reductions keep a fixed order independent of thread count; every speed change re-runs the 1-thread vs N-thread column diff (`identity_break.py --require-cpu` with and without MOJOLEARN_CPU_THREADS=1, then `--diff`).

### Per-family CPU speed (each family lane's phase 5), serial host modules today

From host_surface's `host_modules`, the modules with no thread split (targets
to rank by CPU wall time at realistic large shapes on R2 data):
- **linear:** glm_oracle, qn_oracle (estimators), cd_oracle (solver).
- **cluster:** kmeans_oracle (core, metrics, mixture), dbscan_oracle, linkage_oracle + condense/extract (hdbscan), spectral_oracle, gmm chol/gemm.
- **neighbors:** gp (gpr_grad_oracle, gpc_oracle), kernel_methods random_features, svm gemm_oracle.
- **decomp:** pca_full_oracle, umap_oracle, linalg (gemm_oracle, chol_oracle, linalg_public: all serial).
- **prep:** already threaded (scaler_oracle, resample_host).
- **sequence:** arima_oracle, hw_oracle, kpss_oracle (tsa, forecast).
- **trees:** gbdt oracles (18 of 20 modules serial), trees estimator.
- **ann:** ivf_host, ivf_host_search, ivf_flat_index (all serial).
- **neural:** mlp/loss/optimizer/embedding/chunked_lm_head oracles, mamba and transformer oracles (serial), tokenizer (bpe serial; encoding threaded).
- **metrics:** curve, graph (serial).

## Audit table

Status: ADMITTED = a covered lane of the name is in `public_reference_lanes()`;
PENDING = its lanes are pending with that reason; NO LANE = no identity lane
touches the name. Columns are from the probe at 993e58c9 (before the fixes:
PCA and TruncatedSVD inverse_transform are fixed on lane/cpu).

| name | GPU probe | CPU fit | CPU infer | CPU = GPU bits | lanes (CPU column) | status | why |
|---|---|---|---|---|---|---|---|
| DistributedIVFIndex | - | - | - | - | none | NO LANE |  |
| ParallelByteLanguageModelTrainer | - | - | - | - | none | NO LANE |  |
| ParallelNeuralTrainer | - | - | - | - | none | NO LANE |  |
| PooledByteLanguageModelTrainer | - | - | - | - | none | NO LANE |  |
| OffloadedByteLanguageModelTrainer | - | - | - | - | none | NO LANE |  |
| cross_val_score | OK | OK | - | EQUAL | 1: cross-val | ADMITTED |  |
| MinMaxScaler | OK | OK | OK | EQUAL | 2: minmax-scaler, minmax-scaler-clip | ADMITTED |  |
| StandardScaler | OK | OK | OK | EQUAL | 3: standard-scaler, standard-scaler-no-mean, standard-scaler-no-std | ADMITTED |  |
| SGD | - | - | - | - | 1: optim-sgd | ADMITTED |  |
| Adam | - | - | - | - | 1: optim-adam-clip | ADMITTED |  |
| AdamW | - | - | - | - | 1: optim-adam-clip | ADMITTED |  |
| SambaStack | - | - | - | - | 4: samba, samba-bf16w, samba-int8w ... | ADMITTED |  |
| ARIMA | OK | OK | OK | EQUAL | 5: arima, arima-011, arima-exog ... | ADMITTED |  |
| AgglomerativeClustering | predict ERROR | OK | predict ERROR | EQUAL | 1: agglomerative | ADMITTED | ValueError: mojolearn AgglomerativeClustering.predict: prediction data was not stored. Fit with AgglomerativeClustering(prediction_data=True |
| Cholesky | OK | OK | OK | EQUAL | 1: cholesky | ADMITTED |  |
| KernelRidge | OK | OK | OK | EQUAL | 7: kernel-ridge, kernel-ridge-laplacian, kernel-ridge-poly ... | ADMITTED |  |
| Nystroem | OK | OK | OK | EQUAL | 7: kernel-ridge-laplacian, kernel-ridge-poly, kernel-ridge-sigmoid ... | ADMITTED |  |
| RBFSampler | OK | OK | OK | EQUAL | 1: rbf-sampler | ADMITTED |  |
| GaussianMixture | OK | OK | OK | EQUAL | 5: gmm, gmm-random-init, gmm-random-init-sample ... | ADMITTED |  |
| HDBSCAN | OK | OK | - | EQUAL | 2: hdbscan, hdbscan-leaf | ADMITTED |  |
| IVFIndex | OK | OK | OK | EQUAL | 3: ivf, ivf-euclidean, ivf-extend | ADMITTED |  |
| Embedding | OK | OK | OK | EQUAL | 2: embedding, embedding-sort | ADMITTED |  |
| DBSCAN | predict ERROR | OK | predict ERROR | EQUAL | 3: dbscan, dbscan-brute-l1, dbscan-weighted | ADMITTED | ValueError: mojolearn DBSCAN.predict: prediction data was not stored. Fit with DBSCAN(prediction_data=True), which keeps the core samples th |
| GaussianProcessClassifier | OK | OK | OK | EQUAL | 2: gpc, gpc-multiclass | ADMITTED |  |
| GaussianProcessRegressor | OK | OK | OK | EQUAL | 9: gp, gp-matern12, gp-matern32 ... | ADMITTED |  |
| KernelDensity | OK | OK | OK | EQUAL | 7: kde, kde-cosine-minkowski, kde-epanechnikov-l1 ... | ADMITTED |  |
| ExtraTreesClassifier | OK | OK | OK | EQUAL | 2: et-clf, et-clf-entropy-bestfirst | ADMITTED |  |
| ExtraTreesRegressor | OK | OK | OK | EQUAL | 2: et-reg, et-reg-bootstrap-parallel | ADMITTED |  |
| ElasticNet | OK | OK | OK | EQUAL | 2: elasticnet, elasticnet-l2end-no-intercept | ADMITTED |  |
| ExperimentalTwoLevelFeatureFreq | fit ERROR | fit ERROR | - | - | 2: gbdt-feature-freq, gbdt-tensor-ctr-tables | ADMITTED | TypeError: ExperimentalTwoLevelFeatureFreq.__init__() missing 1 required positional argument: 'sources' |
| GradientBoosting | predict_proba ERROR | OK | predict_proba ERROR | EQUAL | 27: gbdt-bfa-quantile, gbdt-binary-columns, gbdt-border-types ... | ADMITTED | ValueError: mojolearn: predict_proba is defined for Logloss, CrossEntropy, MultiClass and MultiClassOneVsAll; this model was fitted with 'RM |
| GradientBoostingClassifier | OK | OK | OK | EQUAL | 3: gbdt-adapter-clf, gbdt-adapter-score-weighted, gbdt-catboost-defaults | ADMITTED |  |
| GradientBoostingRegressor | OK | OK | OK | EQUAL | 4: cross-val, gbdt-adapter-reg, gbdt-adapter-score-weighted ... | ADMITTED |  |
| OrderedRMSE | fit ERROR | fit ERROR | - | - | 2: gbdt-binary-columns, gbdt-ordered-rmse | ADMITTED | TypeError: OrderedRMSE.fit() missing 1 required positional argument: 'y' |
| ExponentialSmoothing | OK | OK | OK | EQUAL | 2: holtwinters, holtwinters-multiplicative | ADMITTED |  |
| BpeTokenizer | - | - | - | - | 3: bpe-vocabulary, tokenized-corpus, tokenizer | ADMITTED |  |
| GPT2Tokenizer | - | - | - | - | none | NO LANE |  |
| MLPInference | - | - | - | - | 16: mamba1-bf16w, mamba1-int8w, mamba2-bf16w ... | ADMITTED |  |
| TransformerBlockInference | - | - | - | - | 16: mamba1-bf16w, mamba1-int8w, mamba2-bf16w ... | ADMITTED |  |
| Mamba1BlockInference | - | - | - | - | 16: mamba1-bf16w, mamba1-int8w, mamba2-bf16w ... | ADMITTED |  |
| Mamba2BlockInference | - | - | - | - | 16: mamba1-bf16w, mamba1-int8w, mamba2-bf16w ... | ADMITTED |  |
| Mamba3BlockInference | - | - | - | - | 16: mamba1-bf16w, mamba1-int8w, mamba2-bf16w ... | ADMITTED |  |
| SambaInference | - | - | - | - | 16: mamba1-bf16w, mamba1-int8w, mamba2-bf16w ... | ADMITTED |  |
| IsolationForest | OK | OK | OK | EQUAL | 2: iforest, iforest-tuned | ADMITTED |  |
| KMeans | OK | OK | OK | EQUAL | 8: kmeans, kmeans-array, kmeans-classic-pp ... | ADMITTED |  |
| KNeighborsClassifier | OK | OK | OK | EQUAL | 2: knn-clf, knn-clf-distance | ADMITTED |  |
| KNeighborsRegressor | OK | OK | OK | EQUAL | 2: knn-reg, knn-reg-distance | ADMITTED |  |
| Lasso | OK | OK | OK | EQUAL | 1: lasso | ADMITTED |  |
| LinearRegression | OK | OK | OK | EQUAL | 3: ols, ols-no-intercept, ols-weighted | ADMITTED |  |
| LinearSVC | OK | OK | OK | EQUAL | 2: linear-svc, linear-svc-squared-hinge | ADMITTED |  |
| LinearSVR | OK | OK | OK | EQUAL | 2: linear-svr, linear-svr-squared | ADMITTED |  |
| LogisticRegression | OK | OK | OK | EQUAL | 5: logistic, logistic-elasticnet, logistic-l1 ... | ADMITTED |  |
| QNRegressor | OK | OK | OK | EQUAL | 2: qn-absolute, qn-squared | ADMITTED |  |
| Mamba1Block | - | - | - | - | 10: mamba1, mamba1-bf16w, mamba1-decode-session ... | ADMITTED |  |
| Mamba2Block | - | - | - | - | 10: mamba1-bf16w, mamba1-int8w, mamba2 ... | ADMITTED |  |
| Mamba3Block | - | - | - | - | 9: mamba1-bf16w, mamba1-int8w, mamba2-bf16w ... | ADMITTED |  |
| SVC | predict_proba REFUSED | OK | predict_proba REFUSED | EQUAL | 3: svc, svc-linear, svc-poly | ADMITTED | NotImplementedError: mojolearn SVC: predict_proba is not available; Platt scaling is not in cuML's C++ surface at all (svm/NOT_IMPLEMENTED.t |
| SVR | OK | OK | OK | EQUAL | 2: svr, svr-linear | ADMITTED |  |
| SpectralClustering | predict ERROR | OK | predict ERROR | EQUAL | 2: spectral, spectral-precomputed | ADMITTED | ValueError: mojolearn SpectralClustering.predict: prediction data was not stored. Fit with SpectralClustering(prediction_data=True), which k |
| SpectralEmbedding | OK | OK | - | EQUAL | 2: spectral-embedding, x-decomp-spectral-rbf | ADMITTED |  |
| TransformerBlock | - | - | - | - | 11: mamba1-bf16w, mamba1-int8w, mamba2-bf16w ... | ADMITTED |  |
| NearestNeighbors | OK | OK | OK | EQUAL | 7: knn, knn-chebyshev, knn-cosine ... | ADMITTED |  |
| PCA | OK | OK | inverse_transform REFUSED | EQUAL | 4: pca, pca-full-whiten, pca-whiten ... | ADMITTED | ImportError: mojolearn: no CPU implementation of _mojolearn_estimators.inverse_transform yet; see SUPPORT_MATRIX.md (the host binding _mojol |
| RadiusNeighbors | OK | OK | OK | EQUAL | 4: radius, radius-chebyshev, radius-manhattan ... | ADMITTED |  |
| RandomForestClassifier | OK | OK | OK | EQUAL | 9: rf-clf, rf-clf-balanced-parallel, rf-clf-entropy-log2-noboot ... | ADMITTED |  |
| RandomForestRegressor | OK | OK | OK | EQUAL | 7: rf-reg, rf-reg-gamma-ig, rf-reg-poisson ... | ADMITTED |  |
| Ridge | OK | OK | OK | EQUAL | 2: ridge, ridge-no-intercept | ADMITTED |  |
| TruncatedSVD | inverse_transform ERROR | OK | inverse_transform ERROR | EQUAL | 2: tsvd, x-decomp-pca-randomized | ADMITTED | ValueError: mojolearn TruncatedSVD component count differs from fit |
| UMAP | OK | OK | OK | EQUAL | 1: umap | ADMITTED |  |
| SmallMLPTrainer | - | - | - | - | 3: mlp, mlp-bf16w, mlp-int8w | ADMITTED |  |
| SmallByteLanguageModelTrainer | - | - | - | - | 2: byte-lm, byte-lm-resident | ADMITTED |  |
| LanguageModelTrainer | - | - | - | - | none | NO LANE |  |
| LanguageModelInference | - | - | - | - | 19: byte-lm-host-infer, byte-lm-host-infer-threaded, language-model-config ... | ADMITTED |  |
| LanguageModelHostTrainer | - | - | - | - | 1: byte-lm-host-train | ADMITTED |  |
| HostForest | - | - | - | - | 1: saved-model-host-infer | ADMITTED |  |
| HostGBDT | - | - | - | - | 1: saved-model-host-infer | ADMITTED |  |
| host_model | - | - | - | - | 2: gbdt-categorical-ctr-tables, gbdt-tensor-ctr-tables | ADMITTED |  |
| host_predict | - | - | - | - | 1: saved-model-host-infer | ADMITTED |  |
| host_predict_proba | - | - | - | - | 1: saved-model-host-infer | ADMITTED |  |
| kpss_test | OK | OK | - | EQUAL | 1: kpss | ADMITTED |  |
| matmul | OK | OK | - | EQUAL | 2: gemm-pinned, gemm-transposed | ADMITTED |  |
| clip_grad_norm_ | - | - | - | - | 1: optim-adam-clip | ADMITTED |  |
| cross_entropy | - | - | - | - | 1: cross-entropy-arms | ADMITTED |  |
| select_d | OK | OK | - | EQUAL | 1: select-d | ADMITTED |  |
| SGDClassifier | predict_proba ERROR | OK | predict_proba ERROR | EQUAL | 1: x-sgd-clf | PENDING (no reference) | AttributeError: probability estimates are not available for loss='hinge' |
| SGDRegressor | OK | OK | OK | EQUAL | 1: x-sgd-reg | PENDING (no reference) |  |
| PoissonRegressor | fit ERROR | fit ERROR | - | - | 2: x-glm-poisson, x-glm-poisson-sw | PENDING (no reference) | ValueError: Some value(s) of y are out of the valid range of the loss 'HalfPoissonLoss'. |
| GammaRegressor | fit ERROR | fit ERROR | - | - | 1: x-glm-gamma | PENDING (no reference) | ValueError: Some value(s) of y are out of the valid range of the loss 'HalfGammaLoss'. |
| TweedieRegressor | OK | OK | OK | EQUAL | 1: x-glm-tweedie | PENDING (no reference) |  |
| HuberRegressor | OK | OK | OK | EQUAL | 1: x-huber | PENDING (no reference) |  |
| BayesianRidge | OK | OK | OK | EQUAL | 1: x-bayes-ridge | PENDING (no reference) |  |
| ARDRegression | OK | OK | OK | EQUAL | 1: x-ard | PENDING (no reference) |  |
| Lars | OK | OK | OK | EQUAL | 1: x-lars | PENDING (no reference) |  |
| LassoLars | OK | OK | OK | EQUAL | 1: x-lasso-lars | PENDING (no reference) |  |
| QuantileRegressor | OK | OK | OK | EQUAL | 1: x-quantile | PENDING (no reference) |  |
| Perceptron | OK | OK | OK | EQUAL | 1: x-perceptron | PENDING (no reference) |  |
| PassiveAggressiveClassifier | OK | OK | OK | EQUAL | 1: x-pa-clf | PENDING (no reference) |  |
| PassiveAggressiveRegressor | OK | OK | OK | EQUAL | 1: x-pa-reg | PENDING (no reference) |  |
| SGDOneClassSVM | OK | OK | OK | EQUAL | 1: x-sgd-ocsvm | PENDING (no reference) |  |
| RidgeClassifier | OK | OK | OK | EQUAL | 1: x-ridge-clf | PENDING (no reference) |  |
| RidgeCV | OK | OK | OK | EQUAL | 1: x-ridge-cv | PENDING (no reference) |  |
| LassoCV | OK | OK | OK | EQUAL | 1: x-lasso-cv | PENDING (no reference) |  |
| ElasticNetCV | OK | OK | OK | EQUAL | 1: x-enet-cv | PENDING (no reference) |  |
| LogisticRegressionCV | OK | OK | OK | EQUAL | 1: x-logistic-cv | PENDING (no reference) |  |
| IsotonicRegression | fit ERROR | fit ERROR | - | - | 1: x-isotonic | PENDING (no reference) | ValueError: mojolearn IsotonicRegression: X must be 1-D or of shape (n, 1) |
| MiniBatchKMeans | TIMEOUT | OK | OK | - | 3: x-cluster-minibatch-kmeans, x-cluster-minibatch-options, x-cluster-minibatch-partial | PENDING (no reference) |  |
| BisectingKMeans | TIMEOUT | OK | OK | - | 2: x-cluster-bisecting-kmeans, x-cluster-bisecting-options | PENDING (no reference) |  |
| MeanShift | TIMEOUT | OK | OK | - | 2: x-cluster-meanshift, x-cluster-meanshift-binned | PENDING (no reference) |  |
| OPTICS | OK | OK | - | EQUAL | 2: x-cluster-optics, x-cluster-optics-metrics | PENDING (no reference) |  |
| AffinityPropagation | TIMEOUT | OK | OK | - | 2: x-cluster-affinity-propagation, x-cluster-ap-precomputed | PENDING (no reference) |  |
| BayesianGaussianMixture | TIMEOUT | OK | OK | - | 3: x-cluster-bgmm, x-cluster-bgmm-covtypes, x-cluster-bgmm-inits | PENDING (no reference) |  |
| LocalOutlierFactor | TIMEOUT | OK | predict ERROR, decision_function ERROR, score_samples ERROR | - | 1: x-neighbors-lof | PENDING (no reference) | AttributeError: predict is not available when novelty=False; use fit_predict |
| NearestCentroid | TIMEOUT | OK | OK | - | 1: x-neighbors-nearest-centroid | PENDING (no reference) |  |
| OneClassSVM | TIMEOUT | OK | OK | - | 1: x-neighbors-ocsvm | PENDING (no reference) |  |
| KernelPCA | TIMEOUT | OK | inverse_transform REFUSED | - | 1: x-neighbors-kpca | PENDING (no reference) | NotImplementedError: KernelPCA: inverse_transform needs fit_inverse_transform, which is not implemented |
| PolynomialCountSketch | fit ERROR | fit ERROR | - | - | 1: x-neighbors-poly-sketch | PENDING (no reference) | ValueError: random_state=None draws from the OS; pass an int for a reproducible fit |
| AdditiveChi2Sampler | OK | OK | OK | EQUAL | 1: x-neighbors-additive-chi2 | PENDING (no reference) |  |
| SkewedChi2Sampler | fit ERROR | fit ERROR | - | - | 1: x-neighbors-skewed-chi2 | PENDING (no reference) | ValueError: random_state=None draws from the OS; pass an int for a reproducible fit |
| LabelPropagation | TIMEOUT | OK | OK | - | 1: x-neighbors-label-propagation | PENDING (no reference) |  |
| LabelSpreading | TIMEOUT | OK | OK | - | 1: x-neighbors-label-spreading | PENDING (no reference) |  |
| KNNImputer | OK | OK | OK | EQUAL | 1: x-neighbors-knn-imputer | PENDING (no reference) |  |
| PageRank | fit ERROR | fit ERROR | - | - | 1: x-neighbors-pagerank | PENDING (no reference) | ValueError: the adjacency matrix must be square |
| connected_components | - | - | - | - | 1: x-neighbors-connected-components | PENDING (no reference) |  |
| Louvain | fit ERROR | fit ERROR | - | - | 1: x-neighbors-louvain | PENDING (no reference) | ValueError: the adjacency matrix must be square |
| SVGP | TIMEOUT | OK | OK | - | 1: x-neighbors-svgp | PENDING (no reference) |  |
| IncrementalPCA | OK | OK | OK | EQUAL | 1: x-decomp-ipca | PENDING (no reference) |  |
| GaussianRandomProjection | fit ERROR | fit ERROR | - | - | 1: x-decomp-grp | PENDING (no reference) | ValueError: eps=0.1 and n_samples=160 lead to a target dimension of 4350 which is larger than the original space with n_features=6 |
| SparseRandomProjection | fit ERROR | fit ERROR | - | - | 1: x-decomp-srp | PENDING (no reference) | ValueError: eps=0.1 and n_samples=160 lead to a target dimension of 4350 which is larger than the original space with n_features=6 |
| johnson_lindenstrauss_min_dim | - | - | - | - | none | NO LANE |  |
| NMF | OK | OK | OK | EQUAL | 1: x-decomp-nmf | PENDING (no reference) |  |
| FastICA | OK | OK | OK | EQUAL | 1: x-decomp-fastica | PENDING (no reference) |  |
| FactorAnalysis | OK | OK | OK | EQUAL | 1: x-decomp-factor-analysis | PENDING (no reference) |  |
| lu_factor | - | - | - | - | 1: x-decomp-lu | PENDING (no reference) |  |
| lu_solve | - | - | - | - | 1: x-decomp-lu | PENDING (no reference) |  |
| solve | - | - | - | - | 2: cholesky, x-decomp-lu | ADMITTED |  |
| lstsq | - | - | - | - | 1: x-decomp-lstsq-rsvd | PENDING (no reference) |  |
| randomized_svd | - | - | - | - | 1: x-decomp-lstsq-rsvd | PENDING (no reference) |  |
| PLSRegression | inverse_transform ERROR | OK | inverse_transform ERROR | EQUAL | 1: x-decomp-pls | PENDING (no reference) | ValueError: x_decomp: gemm inner dimensions 6 and 2 differ |
| PLSCanonical | fit ERROR | fit ERROR | - | - | 1: x-decomp-pls | PENDING (no reference) | ValueError: n_components == 2, while 1 <= n_components <= 1 is required |
| CCA | fit ERROR | fit ERROR | - | - | 1: x-decomp-pls | PENDING (no reference) | ValueError: n_components == 2, while 1 <= n_components <= 1 is required |
| DictionaryLearning | OK | OK | OK | EQUAL | 1: x-decomp-dict-learning | PENDING (no reference) |  |
| MiniBatchDictionaryLearning | OK | OK | OK | EQUAL | 1: x-decomp-dict-learning | PENDING (no reference) |  |
| SparsePCA | OK | OK | OK | EQUAL | 1: x-decomp-sparse-pca | PENDING (no reference) |  |
| MiniBatchSparsePCA | OK | OK | OK | EQUAL | 1: x-decomp-sparse-pca | PENDING (no reference) |  |
| sparse_encode | - | - | - | - | none | NO LANE |  |
| SparseCoder | fit ERROR | fit ERROR | - | - | 1: x-decomp-dict-learning | PENDING (no reference) | TypeError: SparseCoder.__init__() missing 1 required positional argument: 'dictionary' |
| LatentDirichletAllocation | OK | OK | OK | EQUAL | 1: x-decomp-lda | PENDING (no reference) |  |
| Isomap | OK | OK | OK | EQUAL | 1: x-decomp-manifold | PENDING (no reference) |  |
| MDS | OK | OK | - | EQUAL | 1: x-decomp-manifold | PENDING (no reference) |  |
| ClassicalMDS | OK | OK | - | EQUAL | 1: x-decomp-manifold | PENDING (no reference) |  |
| LocallyLinearEmbedding | OK | OK | OK | EQUAL | 1: x-decomp-manifold | PENDING (no reference) |  |
| MinCovDet | OK | OK | - | - | 1: x-decomp-robust-cov | PENDING (no reference) |  |
| EllipticEnvelope | OK | OK | OK | EQUAL | 1: x-decomp-robust-cov | PENDING (no reference) |  |
| AlternatingLeastSquares | OK | OK | - | - | 1: x-decomp-als | PENDING (no reference) |  |
| f_classif | - | - | - | - | 1: x-prep-select-kbest | PENDING (no reference) |  |
| f_regression | - | - | - | - | 1: x-prep-select-kbest | PENDING (no reference) |  |
| chi2 | - | - | - | - | 1: x-prep-select-kbest | PENDING (no reference) |  |
| mutual_info_classif | - | - | - | - | 1: x-prep-mutual-info | PENDING (no reference) |  |
| mutual_info_regression | - | - | - | - | 1: x-prep-mutual-info | PENDING (no reference) |  |
| RobustScaler | OK | OK | OK | EQUAL | 2: x-prep-robust-scaler, x-prep-robust-scaler-unit-variance | PENDING (no reference) |  |
| MaxAbsScaler | OK | OK | OK | EQUAL | 1: x-prep-maxabs-scaler | PENDING (no reference) |  |
| OrdinalEncoder | transform ERROR, inverse_transform ERROR | OK | transform ERROR, inverse_transform ERROR | - | 2: x-prep-encoder-options, x-prep-ordinal-encoder | PENDING (no reference) | ValueError: mojolearn: OrdinalEncoder found unknown categories in column(s) [0, 1, 2, 3, 4, 5] during transform |
| OneHotEncoder | transform ERROR, inverse_transform ERROR | OK | transform ERROR, inverse_transform ERROR | - | 2: x-prep-encoder-options, x-prep-onehot-encoder | PENDING (no reference) | ValueError: mojolearn: OneHotEncoder found unknown categories in column(s) [0, 1, 2, 3, 4, 5] during transform |
| TargetEncoder | OK | OK | OK | EQUAL | 1: x-prep-target-encoder | PENDING (no reference) |  |
| SimpleImputer | OK | OK | OK | EQUAL | 2: x-prep-simple-imputer, x-prep-simple-imputer-indicator | PENDING (no reference) |  |
| KBinsDiscretizer | inverse_transform ERROR | OK | inverse_transform ERROR | EQUAL | 2: x-prep-inverse-transforms, x-prep-kbins | PENDING (no reference) | ValueError: mojolearn: X has 6 columns, expected 29 |
| GaussianNB | OK | OK | OK | EQUAL | 2: x-prep-gaussian-nb, x-prep-priors | PENDING (no reference) |  |
| MultinomialNB | OK | OK | OK | EQUAL | 2: x-prep-multinomial-nb, x-prep-priors | PENDING (no reference) |  |
| BernoulliNB | OK | OK | OK | EQUAL | 1: x-prep-bernoulli-nb | PENDING (no reference) |  |
| LinearDiscriminantAnalysis | OK | OK | OK | EQUAL | 3: x-prep-lda, x-prep-priors, x-prep-rfe | PENDING (no reference) |  |
| QuadraticDiscriminantAnalysis | OK | OK | OK | EQUAL | 2: x-prep-priors, x-prep-qda | PENDING (no reference) |  |
| QuantileTransformer | OK | OK | OK | EQUAL | 2: x-prep-inverse-transforms, x-prep-quantile-transformer | PENDING (no reference) |  |
| PowerTransformer | OK | OK | OK | EQUAL | 2: x-prep-inverse-transforms, x-prep-power-transformer | PENDING (no reference) |  |
| Normalizer | OK | OK | OK | EQUAL | 1: x-prep-normalizer | PENDING (no reference) |  |
| PolynomialFeatures | OK | OK | OK | EQUAL | 1: x-prep-polynomial-features | PENDING (no reference) |  |
| SplineTransformer | OK | OK | OK | EQUAL | 1: x-prep-spline-transformer | PENDING (no reference) |  |
| Binarizer | OK | OK | OK | EQUAL | 1: x-prep-binarizer | PENDING (no reference) |  |
| LabelEncoder | transform ERROR, inverse_transform ERROR | OK | transform ERROR, inverse_transform ERROR | - | 1: x-prep-label-encoder | PENDING (no reference) | ValueError: mojolearn: y contains previously unseen labels |
| LabelBinarizer | inverse_transform ERROR | OK | inverse_transform ERROR | EQUAL | 2: x-prep-inverse-transforms, x-prep-label-binarizer | PENDING (no reference) | ValueError: mojolearn: Y has 6 columns, expected 796 |
| MultiLabelBinarizer | OK | OK | OK | EQUAL | 1: x-prep-multilabel-binarizer | PENDING (no reference) |  |
| IterativeImputer | OK | OK | OK | EQUAL | 1: x-prep-iterative-imputer | PENDING (no reference) |  |
| VarianceThreshold | OK | OK | OK | EQUAL | 1: x-prep-variance-threshold | PENDING (no reference) |  |
| SelectKBest | fit ERROR | fit ERROR | - | - | 2: x-prep-mutual-info, x-prep-select-kbest | PENDING (no reference) | ValueError: mojolearn: label None is neither a number nor a str |
| RFE | fit ERROR | fit ERROR | - | - | 1: x-prep-rfe | PENDING (no reference) | TypeError: RFE.__init__() missing 1 required positional argument: 'estimator' |
| ComplementNB | OK | OK | OK | EQUAL | 1: x-prep-complement-nb | PENDING (no reference) |  |
| CategoricalNB | OK | OK | OK | EQUAL | 1: x-prep-categorical-nb | PENDING (no reference) |  |
| LSTMRegressor | fit ERROR | fit ERROR | - | - | 3: sequence-lion, sequence-lr-schedulers, sequence-lstm | PENDING (no reference) | ValueError: X must be 3-D: (n_samples, n_timesteps, n_features) |
| LSTMClassifier | fit ERROR | fit ERROR | - | - | 1: sequence-lstm | PENDING (no reference) | ValueError: X must be 3-D: (n_samples, n_timesteps, n_features) |
| GRURegressor | fit ERROR | fit ERROR | - | - | 2: sequence-adamax, sequence-gru | PENDING (no reference) | ValueError: X must be 3-D: (n_samples, n_timesteps, n_features) |
| GRUClassifier | fit ERROR | fit ERROR | - | - | 1: sequence-gru | PENDING (no reference) | ValueError: X must be 3-D: (n_samples, n_timesteps, n_features) |
| RMSprop | - | - | - | - | 1: sequence-rmsprop | PENDING (no reference) |  |
| Adagrad | - | - | - | - | 1: sequence-adagrad | PENDING (no reference) |  |
| AutoARIMA | fit ERROR | fit ERROR | - | - | 1: sequence-autoarima | PENDING (no reference) | TypeError: AutoARIMA.__init__() missing 1 required positional argument: 'endog' |
| STL | fit ERROR | fit ERROR | - | - | 1: sequence-stl | PENDING (no reference) | TypeError: STL.__init__() missing 1 required positional argument: 'endog' |
| VAR | fit ERROR | fit ERROR | - | - | 1: sequence-var | PENDING (no reference) | TypeError: VAR.__init__() missing 1 required positional argument: 'endog' |
| MLPClassifier | TIMEOUT | OK | OK | - | 1: sequence-mlp | PENDING (no reference) |  |
| MLPRegressor | TIMEOUT | OK | OK | - | 1: sequence-mlp | PENDING (no reference) |  |
| RNNRegressor | fit ERROR | fit ERROR | - | - | 1: sequence-rnn | PENDING (no reference) | ValueError: X must be 3-D: (n_samples, n_timesteps, n_features) |
| RNNClassifier | fit ERROR | fit ERROR | - | - | 2: sequence-nadam, sequence-rnn | PENDING (no reference) | ValueError: X must be 3-D: (n_samples, n_timesteps, n_features) |
| Lion | - | - | - | - | 1: sequence-lion | PENDING (no reference) |  |
| Adafactor | - | - | - | - | 1: sequence-adafactor | PENDING (no reference) |  |
| LAMB | - | - | - | - | 1: sequence-lamb | PENDING (no reference) |  |
| Adamax | - | - | - | - | 1: sequence-adamax | PENDING (no reference) |  |
| NAdam | - | - | - | - | 2: sequence-lr-schedulers, sequence-nadam | PENDING (no reference) |  |
| StepLR | - | - | - | - | 1: sequence-lr-schedulers | PENDING (no reference) |  |
| ExponentialLR | - | - | - | - | 1: sequence-lr-schedulers | PENDING (no reference) |  |
| OneCycleLR | - | - | - | - | 1: sequence-lr-schedulers | PENDING (no reference) |  |
| LayerNorm | - | - | - | - | 1: sequence-layernorm | PENDING (no reference) |  |
| layer_norm_forward | - | - | - | - | 1: sequence-layernorm | PENDING (no reference) |  |
| layer_norm_backward | - | - | - | - | none | NO LANE |  |
| Theta | predict ERROR | OK | predict ERROR | - | none | NO LANE | TypeError: only 0-dimensional arrays can be converted to Python scalars |
| OptimizedTheta | predict ERROR | OK | predict ERROR | - | none | NO LANE | TypeError: only 0-dimensional arrays can be converted to Python scalars |
| DynamicTheta | predict ERROR | OK | predict ERROR | - | none | NO LANE | TypeError: only 0-dimensional arrays can be converted to Python scalars |
| DynamicOptimizedTheta | predict ERROR | OK | predict ERROR | - | 1: sequence-theta | PENDING (no reference) | TypeError: only 0-dimensional arrays can be converted to Python scalars |
| AutoTheta | predict ERROR | OK | predict ERROR | - | 1: sequence-theta | PENDING (no reference) | TypeError: only 0-dimensional arrays can be converted to Python scalars |
| CrostonClassic | predict ERROR | OK | predict ERROR | - | 1: sequence-croston | PENDING (no reference) | TypeError: only 0-dimensional arrays can be converted to Python scalars |
| CrostonOptimized | TIMEOUT | OK | predict ERROR | - | 1: sequence-croston | PENDING (no reference) | TypeError: only 0-dimensional arrays can be converted to Python scalars |
| CrostonSBA | predict ERROR | OK | predict ERROR | - | 1: sequence-croston | PENDING (no reference) | TypeError: only 0-dimensional arrays can be converted to Python scalars |
| ETS | predict ERROR | OK | predict ERROR | - | 1: sequence-ets | PENDING (no reference) | TypeError: only 0-dimensional arrays can be converted to Python scalars |
| DampedETS | predict ERROR | OK | predict ERROR | - | 1: sequence-ets | PENDING (no reference) | TypeError: only 0-dimensional arrays can be converted to Python scalars |
| GARCH | fit ERROR | fit ERROR | - | - | 1: sequence-garch | PENDING (no reference) | ValueError: GARCH: each series needs at least 10 observations |
| ProphetForecaster | fit ERROR | fit ERROR | - | - | 1: sequence-prophet | PENDING (no reference) | ValueError: operands could not be broadcast together with shapes (160,6) (160,) |
| MoEBlock | - | - | - | - | 1: sequence-moe | PENDING (no reference) |  |
| DecisionTreeClassifier | TIMEOUT | OK | OK | - | 11: trees-adaboost-clf, trees-bagging-clf, trees-calibrated ... | PENDING (no reference) |  |
| DecisionTreeRegressor | OK | OK | OK | EQUAL | 8: trees-adaboost-reg, trees-bagging-reg, trees-dt-random ... | PENDING (no reference) |  |
| BaggingClassifier | OK | OK | OK | EQUAL | 2: trees-bagging-clf, trees-voting-clf | PENDING (no reference) |  |
| BaggingRegressor | OK | OK | OK | EQUAL | 1: trees-bagging-reg | PENDING (no reference) |  |
| AdaBoostClassifier | OK | OK | OK | EQUAL | 1: trees-adaboost-clf | PENDING (no reference) |  |
| AdaBoostRegressor | OK | OK | OK | EQUAL | 1: trees-adaboost-reg | PENDING (no reference) |  |
| DARTRegressor | OK | OK | OK | EQUAL | 2: trees-dart-reg, trees-shap-tree | PENDING (no reference) |  |
| DARTClassifier | OK | OK | OK | EQUAL | 1: trees-dart-clf | PENDING (no reference) |  |
| RandomTreesEmbedding | OK | OK | OK | EQUAL | 1: trees-random-embedding | PENDING (no reference) |  |
| VotingClassifier | fit ERROR | fit ERROR | - | - | 1: trees-voting-clf | PENDING (no reference) | TypeError: VotingClassifier.__init__() missing 1 required positional argument: 'estimators' |
| VotingRegressor | fit ERROR | fit ERROR | - | - | 1: trees-voting-reg | PENDING (no reference) | TypeError: VotingRegressor.__init__() missing 1 required positional argument: 'estimators' |
| StackingClassifier | fit ERROR | fit ERROR | - | - | 1: trees-stacking-clf | PENDING (no reference) | TypeError: StackingClassifier.__init__() missing 1 required positional argument: 'estimators' |
| StackingRegressor | fit ERROR | fit ERROR | - | - | 1: trees-stacking-reg | PENDING (no reference) | TypeError: StackingRegressor.__init__() missing 1 required positional argument: 'estimators' |
| MultiOutputClassifier | fit ERROR | fit ERROR | - | - | 1: trees-multioutput | PENDING (no reference) | TypeError: MultiOutputClassifier.__init__() missing 1 required positional argument: 'estimator' |
| MultiOutputRegressor | fit ERROR | fit ERROR | - | - | 1: trees-multioutput | PENDING (no reference) | TypeError: MultiOutputRegressor.__init__() missing 1 required positional argument: 'estimator' |
| OneVsRestClassifier | fit ERROR | fit ERROR | - | - | 1: trees-onevsrest | PENDING (no reference) | TypeError: OneVsRestClassifier.__init__() missing 1 required positional argument: 'estimator' |
| CalibratedClassifierCV | OK | OK | OK | EQUAL | 1: trees-calibrated | PENDING (no reference) |  |
| TreeExplainer | - | - | - | - | 1: trees-shap-tree | PENDING (no reference) |  |
| KernelExplainer | - | - | - | - | 1: trees-shap-kernel | PENDING (no reference) |  |
| PermutationExplainer | - | - | - | - | 1: trees-shap-permutation | PENDING (no reference) |  |
| Conv2d | fit ERROR | fit ERROR | - | - | 2: x-cnn-conv-options, x-cnn-conv2d | PENDING (no reference) | TypeError: Conv2d.__init__() missing 3 required positional arguments: 'in_channels', 'out_channels', and 'kernel_size' |
| Conv1d | fit ERROR | fit ERROR | - | - | 1: x-cnn-conv1d | PENDING (no reference) | TypeError: Conv1d.__init__() missing 3 required positional arguments: 'in_channels', 'out_channels', and 'kernel_size' |
| MaxPool2d | fit ERROR | fit ERROR | - | - | 2: x-cnn-pool, x-cnn-pool-options | PENDING (no reference) | TypeError: _Pool2d.__init__() missing 1 required positional argument: 'kernel_size' |
| AvgPool2d | fit ERROR | fit ERROR | - | - | 2: x-cnn-pool, x-cnn-pool-options | PENDING (no reference) | TypeError: AvgPool2d.__init__() missing 1 required positional argument: 'kernel_size' |
| MaxPool1d | fit ERROR | fit ERROR | - | - | 1: x-cnn-pool | PENDING (no reference) | TypeError: MaxPool1d.__init__() missing 1 required positional argument: 'kernel_size' |
| AvgPool1d | fit ERROR | fit ERROR | - | - | 1: x-cnn-pool | PENDING (no reference) | TypeError: AvgPool1d.__init__() missing 1 required positional argument: 'kernel_size' |
| CNNClassifier | fit ERROR | fit ERROR | - | - | 2: x-cnn-trainer, x-cnn-trainer-options | PENDING (no reference) | TypeError: CNNClassifier.__init__() missing 1 required positional argument: 'input_shape' |
| BatchNorm2d | fit ERROR | fit ERROR | - | - | 2: x-cnn-batchnorm, x-cnn-bn-options | PENDING (no reference) | TypeError: BatchNorm2d.__init__() missing 1 required positional argument: 'num_features' |
| BatchNorm1d | fit ERROR | fit ERROR | - | - | 1: x-cnn-batchnorm | PENDING (no reference) | TypeError: BatchNorm2d.__init__() missing 1 required positional argument: 'num_features' |
| Dropout2d | transform ERROR | OK | transform ERROR | - | 1: x-cnn-dropout2d | PENDING (no reference) | ValueError: mojolearn: Dropout2d.forward takes (N, C, H, W) or (C, H, W) |
| AdaptiveAvgPool2d | transform ERROR | OK | transform ERROR | - | 2: x-cnn-globalpool, x-cnn-pool-options | PENDING (no reference) | ValueError: mojolearn: AdaptiveAvgPool2d.forward takes (N, C, H, W) |
| AdaptiveMaxPool2d | transform ERROR | OK | transform ERROR | - | 2: x-cnn-globalpool, x-cnn-pool-options | PENDING (no reference) | ValueError: mojolearn: AdaptiveMaxPool2d.forward takes (N, C, H, W) |
| BasicBlock | fit ERROR | fit ERROR | - | - | 1: x-cnn-resnet-block | PENDING (no reference) | TypeError: BasicBlock.__init__() missing 2 required positional arguments: 'inplanes' and 'planes' |
| GCNConv | - | - | - | - | 1: x-cnn-gcn | PENDING (no reference) |  |
| SAGEConv | - | - | - | - | 2: x-cnn-gnn-options, x-cnn-sage | PENDING (no reference) |  |
| IVFPQIndex | fit ERROR | fit ERROR | - | - | 2: x-ann-filter, x-ann-ivf-pq | PENDING (no reference) | TypeError: IVFPQIndex.__init__() missing 2 required positional arguments: 'n_lists' and 'n_probes' |
| TSNE | OK | OK | - | EQUAL | 1: x-ann-tsne | PENDING (no reference) |  |
| CagraIndex | OK | OK | - | - | 1: x-ann-cagra | PENDING (no reference) |  |
| IVFSQIndex | fit ERROR | fit ERROR | - | - | 2: x-ann-filter, x-ann-ivf-sq | PENDING (no reference) | TypeError: IVFSQIndex.__init__() missing 2 required positional arguments: 'n_lists' and 'n_probes' |
| IVFRaBitQIndex | fit ERROR | fit ERROR | - | - | 1: x-ann-ivf-rabitq | PENDING (no reference) | TypeError: IVFRaBitQIndex.__init__() missing 2 required positional arguments: 'n_lists' and 'n_probes' |
| refine | - | - | - | - | 1: x-ann-refine | PENDING (no reference) |  |
| metrics.accuracy_score | OK | OK | - | EQUAL | 1: metrics | ADMITTED |  |
| metrics.confusion_matrix | OK | OK | - | EQUAL | 1: metrics-classification | ADMITTED |  |
| metrics.precision_score | OK | OK | - | EQUAL | 1: metrics-classification | ADMITTED |  |
| metrics.recall_score | OK | OK | - | EQUAL | 1: metrics-classification | ADMITTED |  |
| metrics.f1_score | OK | OK | - | EQUAL | 1: metrics-classification | ADMITTED |  |
| metrics.log_loss | call ERROR | call ERROR | - | - | 1: metrics-classification | ADMITTED | TypeError: log_loss probabilities must have dtype float32 |
| metrics.roc_auc_score | call ERROR | call ERROR | - | - | 1: metrics-classification | ADMITTED | TypeError: binary ranking scores must have dtype float32 |
| metrics.precision_recall_curve | call ERROR | call ERROR | - | - | 1: metrics-classification | ADMITTED | TypeError: binary ranking scores must have dtype float32 |
| metrics.adjusted_rand_score | OK | OK | - | EQUAL | 1: metrics | ADMITTED |  |
| metrics.completeness_score | OK | OK | - | EQUAL | 2: metrics-classification, metrics-homogeneity-completeness | ADMITTED |  |
| metrics.entropy | OK | OK | - | EQUAL | 1: metrics-classification | ADMITTED |  |
| metrics.fowlkes_mallows_score | OK | OK | - | EQUAL | 1: metrics-fowlkes-mallows | ADMITTED |  |
| metrics.homogeneity_completeness_v_measure | OK | OK | - | EQUAL | 1: metrics-homogeneity-completeness | ADMITTED |  |
| metrics.homogeneity_score | OK | OK | - | EQUAL | 2: metrics-classification, metrics-homogeneity-completeness | ADMITTED |  |
| metrics.kl_divergence | call ERROR | call ERROR | - | - | 1: metrics-classification | ADMITTED | ValueError: mojolearn metrics: P must be 1-D, got shape (20, 6) |
| metrics.mutual_info_score | OK | OK | - | EQUAL | 1: metrics-classification | ADMITTED |  |
| metrics.mean_squared_error | OK | OK | - | EQUAL | 1: metrics-classification | ADMITTED |  |
| metrics.mean_absolute_error | OK | OK | - | EQUAL | 1: metrics-classification | ADMITTED |  |
| metrics.root_mean_squared_error | OK | OK | - | EQUAL | 1: metrics-classification | ADMITTED |  |
| metrics.r2_score | OK | OK | - | EQUAL | 1: metrics | ADMITTED |  |
| metrics.rand_score | OK | OK | - | EQUAL | 1: metrics-classification | ADMITTED |  |
| metrics.silhouette_samples | OK | OK | - | EQUAL | 1: metrics-classification | ADMITTED |  |
| metrics.silhouette_score | OK | OK | - | EQUAL | 1: metrics | ADMITTED |  |
| metrics.trustworthiness | OK | OK | - | EQUAL | 1: metrics-classification | ADMITTED |  |
| metrics.v_measure_score | OK | OK | - | EQUAL | 3: metrics, metrics-classification, metrics-homogeneity-completeness | ADMITTED |  |
| linalg.matmul | OK | OK | - | EQUAL | 2: gemm-pinned, gemm-transposed | ADMITTED |  |
| linalg.qr | OK | OK | - | EQUAL | 1: linalg-qr | ADMITTED |  |
| linalg.eigh | OK | OK | - | EQUAL | 1: linalg-eigh | ADMITTED |  |
| linalg.svdvals | OK | OK | - | EQUAL | 1: linalg-svdvals | ADMITTED |  |
| linalg.Cholesky | OK | OK | OK | EQUAL | 1: cholesky | ADMITTED |  |
| linalg.matmul_bf16 | call ERROR | call ERROR | - | - | 1: gemm-bf16 | ADMITTED | TypeError: mojolearn.linalg: b has dtype float32; the bf16 profile takes bf16 BITS in a uint16 buffer, which to_bf16() produces. Refused rat |
| linalg.matmul_int8 | call ERROR | call ERROR | - | - | 1: gemm-int8 | ADMITTED | ValueError: mojolearn.linalg.matmul_int8: contracted extents differ, a gives k=6 and b gives k=40 |
| linalg.to_bf16 | OK | OK | - | EQUAL | 1: gemm-bf16 | ADMITTED |  |
| linalg.from_bf16 | OK | OK | - | EQUAL | 1: lowbit-conversions | ADMITTED |  |
| linalg.quantize_int8 | OK | OK | - | EQUAL | 1: gemm-int8 | ADMITTED |  |
| linalg.dequantize_int8 | OK | OK | - | EQUAL | 1: gemm-int8 | ADMITTED |  |
| manifold.SpectralEmbedding | OK | OK | - | EQUAL | 2: spectral-embedding, x-decomp-spectral-rbf | ADMITTED |  |
| manifold.UMAP | OK | OK | OK | EQUAL | 1: umap | ADMITTED |  |
| manifold.spectral_embedding | OK | OK | - | EQUAL | 1: spectral-embedding | ADMITTED |  |
| resample.bootstrap | OK | OK | - | EQUAL | 1: bootstrap | ADMITTED |  |
| resample.permutation_test | OK | OK | - | EQUAL | 1: permutation-test | ADMITTED |  |
| resample.monte_carlo_integrate | OK | OK | - | - | 1: monte-carlo | ADMITTED |  |
| hdbscan.HDBSCAN | OK | OK | - | EQUAL | 2: hdbscan, hdbscan-leaf | ADMITTED |  |
| hdbscan.approximate_predict | OK | OK | OK | EQUAL | 2: hdbscan, hdbscan-leaf | ADMITTED |  |
| hdbscan.membership_vector | OK | OK | OK | EQUAL | 2: hdbscan, hdbscan-leaf | ADMITTED |  |
| hdbscan.all_points_membership_vectors | OK | OK | OK | EQUAL | 2: hdbscan, hdbscan-leaf | ADMITTED |  |
| model_selection.cross_val_score | OK | OK | - | EQUAL | 1: cross-val | ADMITTED |  |
| training.SGD | - | - | - | - | 1: optim-sgd | ADMITTED |  |
| training.Adam | - | - | - | - | 1: optim-adam-clip | ADMITTED |  |
| training.AdamW | - | - | - | - | 1: optim-adam-clip | ADMITTED |  |
| training.clip_grad_norm_ | - | - | - | - | 1: optim-adam-clip | ADMITTED |  |
| training.cross_entropy | - | - | - | - | 1: cross-entropy-arms | ADMITTED |  |
| training.ConstantLR | - | - | - | - | 1: optim-sgd | ADMITTED |  |
| training.WarmupLinearLR | - | - | - | - | 1: optim-adam-clip | ADMITTED |  |
| training.WarmupCosineLR | - | - | - | - | 1: samba-untied-dropout-accum | ADMITTED |  |
| training.Generator | - | - | - | - | 4: samba, samba-bf16w, samba-int8w ... | ADMITTED |  |
| training.accumulate_grads | - | - | - | - | 1: grad-accumulation | ADMITTED |  |
| training.SambaConfig | - | - | - | - | 4: samba, samba-bf16w, samba-int8w ... | ADMITTED |  |
| training.SambaStack | - | - | - | - | 4: samba, samba-bf16w, samba-int8w ... | ADMITTED |  |
| training.embedding_forward | - | - | - | - | 1: training-primitives | ADMITTED |  |
| training.embedding_backward | - | - | - | - | 1: training-primitives | ADMITTED |  |
| training.rms_norm_forward | - | - | - | - | 1: training-primitives | ADMITTED |  |
| training.rms_norm_backward | - | - | - | - | 1: training-primitives | ADMITTED |  |
| training.linear_forward | - | - | - | - | 1: training-primitives | ADMITTED |  |
| training.linear_backward | - | - | - | - | 1: training-primitives | ADMITTED |  |
| training.chunked_lm_head_loss | - | - | - | - | none | NO LANE |  |
| parallel_forecasting.predict_arima | - | - | - | - | none | NO LANE |  |
| parallel_forecasting.forecast_arima | - | - | - | - | none | NO LANE |  |
| parallel_forecasting.predict_exponential_smoothing | - | - | - | - | none | NO LANE |  |
| parallel_forecasting.forecast_exponential_smoothing | - | - | - | - | none | NO LANE |  |
| parallel_gaussian_process.fit_gaussian_process_classifier | - | - | - | - | none | NO LANE |  |
| parallel_gaussian_process.predict_gaussian_process_classifier | - | - | - | - | none | NO LANE |  |
| parallel_model_selection.cross_val_score | call ERROR | call ERROR | - | - | 1: cross-val | ADMITTED | TypeError: cross_val_score() missing 1 required keyword-only argument: 'devices' |
