# BRIEF, the claim surface versus the code surface (2026-09-14)

Read-only census, three parallel sweeps at main `d0cee8e13`, Andrew's question of Sep 14:
"are all of our estimators verified bitwise identical, and does every feature flag
multiply what we claim?" The answer is that the identity claim covers one pinned
configuration per public class (46 lanes at the time of the sweep, 47 with pca-whiten),
while the public parameter surface selects roughly 130 numeric paths the harness never
runs, and about 50,000 lines of identity-gated Mojo has no public door at all. Nothing
here is a defect in a measured cell; it is the size of the unmeasured space.

Definitions. NUMERIC-PATH: a parameter value that selects a different kernel, objective,
sampler, solver, metric, kernel function, reduction or an `if value == ...` in Mojo or
Python that changes arithmetic. SHAPE-ONLY: sizes, seeds, tolerances, iteration caps,
output layout. REFUSE: the class raises by name, so the value is outside the claim by
design (the contract's third move). Every count below came from reading the branch site;
the per-parameter tables with file:line live in the session that wrote this brief and are
summarized here by their actionable half, the lane list.

## 1. Counts

| family | numeric-path parameters | uncovered (parameter, value) pairs | new lanes proposed |
|---|---|---|---|
| trees and forests (RF, ET, GBDT x3 classes, iforest) | 47 | about 65 | 15 |
| neural, optimizers, loss, schedules | 24 | about 30 | 8 |
| classical (31 public surfaces) | 57 | about 92 | 34 |
| total | about 128 | about 187 | 57 |

Fully covered or fully refused today, nothing to add: TruncatedSVD, AgglomerativeClustering,
UMAP, GBDT `grow_policy` (three lanes), Transformer `n_kv_heads` (MHA in samba, GQA in
transformer), StandardScaler and MinMaxScaler `inverse` (both hashed).

No lane at all today: `kpss_test`, `cross_val_score`, `GradientBoostingClassifier` and
`GradientBoostingRegressor` (the sklearn adapters, the only callers of
`gbdt_binary_probabilities` and `gbdt_binary_classes`), `SGD`, `Adam`, `clip_grad_norm_`,
every schedule, `cross_entropy` called directly, and the whole host inference surface
(`HostForest`, `HostGBDT`, `host_model`, `host_predict`, the ten `_classical_host`
classes). The host surface is measured by `tools/forest_host_gate.py` and
`tools/classical_host_gate.py` instead, on recordings, not by identity_break.

## 2. Three findings that are defects in the claim, not gaps

1. `byte-lm-host-infer` pins `threaded=False`. The shipped default of
   `LanguageModelInference` is `threaded=True`, a different host kernel
   (`python/mojolearn/_byte_lm_host.py:148-166`, `:190-198`), measured by no lane. Its
   only gate is `tools/byte_lm_host_path_sweep.py`.
2. GBDT `nan_mode` cannot be exercised by any fixture. `compute_nan_mode`
   (`gbdt/data/quantization.mojo:88-99`) collapses Min and Max to Forbidden on a NaN-free
   column, and no identity_break fixture contains a NaN (`tools/identity_break.py:311-347`).
   A tenth fixture with NaN-bearing columns is needed before any lane can cover it.
3. Accepted and silently ignored, which the contract says must be a refusal by name:
   GBDT `bagging_temperature` and `subsample` at the default `bootstrap_type=None`
   (`gbdt/train.mojo:1656`, `boot_kind` stays -1); NearestNeighbors `p` under any metric
   but minkowski (documented at `python/mojolearn/neighbors.py:439-442`, so this one is
   stated); `cross_val_score(groups=)` (warned and ignored,
   `python/mojolearn/model_selection.py:135-137`).
   RESOLVED (2026-09-14): all three now refuse by name. GBDT and `cross_val_score` in
   1217ed7bb; NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor and
   RadiusNeighbors refuse a `p` other than the default 2 under a metric that does not read
   it (`_refuse_inert_p` in `python/mojolearn/neighbors.py`), all
   covered by `cd python && python3 -m mojolearn.tests.test_refuse_ignored_knobs`.

## 3. Proposed lanes, 57

Each bundles several uncovered values into one fit where they share a kernel. Fixture
cautions at the end.

### Trees, 15

```python
@lane("rf-clf-entropy-log2-noboot")
  ml.RandomForestClassifier(n_estimators=16, max_depth=8, random_state=7,
      criterion="entropy", max_features="log2", bootstrap=False, max_leaves=64)
@lane("rf-clf-balanced-parallel")   # class_weight -> the weighted fit entry; parallel_groves predict kernel
  ml.RandomForestClassifier(n_estimators=16, max_depth=8, random_state=7,
      class_weight="balanced", inference_engine="parallel_groves")
@lane("rf-reg-poisson")             # y = |yr| + 1
  ml.RandomForestRegressor(n_estimators=16, max_depth=8, random_state=7,
      criterion="poisson", max_features="sqrt", max_samples=0.6)
@lane("rf-reg-gamma-ig")            # two fits, y > 0
  ml.RandomForestRegressor(criterion="gamma", ...); ml.RandomForestRegressor(criterion="inverse_gaussian", ...)
@lane("et-clf-entropy-bestfirst")   # max_leaf_nodes selects the best-first grower, a different builder
  ml.ExtraTreesClassifier(n_estimators=16, random_state=7, criterion="entropy",
      max_leaf_nodes=32, max_features=4)
@lane("et-reg-bootstrap-parallel")
  ml.ExtraTreesRegressor(n_estimators=16, max_depth=8, random_state=7,
      max_features="log2", bootstrap=True, max_samples=0.5, inference_engine="parallel_groves")
@lane("gbdt-multiclass")
  ml.GradientBoosting(n_estimators=20, max_depth=6, loss="MultiClass", class_weights=[1.0, 2.0, 0.5])
@lane("gbdt-onevsall")
  ml.GradientBoosting(n_estimators=20, max_depth=6, loss="MultiClassOneVsAll")
@lane("gbdt-parametric-losses")     # ten fits: Quantile, MAE, LogLinQuantile, MAPE, Poisson, CrossEntropy,
                                    # Lq(q=3), Expectile(alpha=.3), Tweedie(vp=1.5), Huber(delta=1.0)
@lane("gbdt-lossguide-newtoncosine")
  ml.GradientBoosting(n_estimators=20, max_leaves=32, grow_policy="Lossguide", loss="Logloss",
      score_function="NewtonCosine", min_child_hessian=1.0, min_split_gain=0.01, min_data_in_leaf=8,
      feature_fraction=0.5, random_strength=1.0, bootstrap_type="Bernoulli", subsample=0.7,
      leaf_estimation_method="Gradient", leaf_estimation_iterations=3)
@lane("gbdt-pointwise-l2-bayesian-eval")
  ml.GradientBoosting(n_estimators=20, max_depth=6, loss="Logloss", score_function="L2",
      use_pointwise_searcher=True, bootstrap_type="Bayesian", bagging_temperature=0.5,
      boost_from_average=True, od_type="Iter", od_wait=5, use_best_model=True
  ).fit(X, yc, sample_weight=w, eval_set=(Xh, ych))
@lane("gbdt-exact-mae")
  ml.GradientBoosting(n_estimators=20, max_depth=6, loss="MAE",
      leaf_estimation_method="Exact", bootstrap_type="Poisson", subsample=0.6)
@lane("gbdt-categorical-ctr")
  ml.GradientBoosting(n_estimators=20, max_depth=6, loss="Logloss", cat_features=[0, 1],
      one_hot_features=[1], permutation_count=2, ctr_estimation_permutation_id=0).fit(coded(X), yc)
@lane("gbdt-adapter-clf") / @lane("gbdt-adapter-reg")
  ml.GradientBoostingClassifier(n_estimators=20, max_depth=6)   # predict_proba, decision_function, score
  ml.GradientBoostingRegressor(n_estimators=20, max_depth=6)
@lane("iforest-tuned")
  ml.IsolationForest(n_estimators=16, random_state=5, max_samples=512, max_features=0.5,
      bootstrap=True, contamination=0.1, max_depth=6)
```

Plus a tenth FIXTURE `nan` (columns 5 to 7 hold NaN) read by a `gbdt-nanmax` lane.

### Neural, 8

```python
@lane("mamba2-dtlimit")            ml.Mamba2Block(w, dt_limit=(0.01, 0.1))
@lane("transformer-window")        ml.TransformerBlock(w, n_heads=2, n_kv_heads=1, window=8)   # ring KV cache
@lane("byte-lm-resident")          ml.SmallByteLanguageModelTrainer(..., resident=True, step_result="lean")
@lane("byte-lm-host-infer-threaded")  ml.LanguageModelInference(flat, shape=shape, threaded=True, threads=4)
@lane("samba-untied-dropout-accum")
  cfg = ml.SambaConfig(vocab=256, d_model=32, layers=("mamba3", "attention"), n_heads=2,
                       intermediate=64, tie_embeddings=False, dropout=0.1)
  ml.SambaStack(cfg, generator=ml.training.Generator(1), lr=1e-3, max_norm=1.0, accumulation_steps=4,
      lr_schedule=ml.training.WarmupCosineLR(1e-3, warmup_steps=2, total_steps=8, min_lr=1e-5))
@lane("optim-sgd")
  ml.training.SGD(params, lr=1e-2, momentum=0.9, dampening=0.0, nesterov=True, weight_decay=0.01,
      lr_schedule=ml.training.ConstantLR(1e-2, warmup_steps=2))   # and a second at dampening=0.5
@lane("optim-adam-clip")
  ml.training.Adam(params, lr=1e-3, weight_decay=0.01,
      lr_schedule=ml.training.WarmupLinearLR(1e-3, warmup_steps=2, total_steps=8, min_lr=0.0))
  ml.training.AdamW(params2, lr=1e-3, weight_decay=0.0); ml.training.clip_grad_norm_(grads, max_norm=1.0)
@lane("cross-entropy-arms")
  cross_entropy(logits, y, reduction="none"); cross_entropy(logits, y, reduction="sum", num_items=17)
  cross_entropy(logits, y, label_smoothing=0.1, return_grad=True)   # label_smoothing > 0 is a DIFFERENT kernel
  cross_entropy(logits, y_masked, ignore_index=0)
```

### Classical, 34

```python
@lane("kmeans-random")      ml.KMeans(n_clusters=8, init="random", n_init=3, random_state=3)
@lane("kmeans-array")       ml.KMeans(n_clusters=8, init="array", init_centroids=X[:8].copy(), random_state=3)
@lane("kmeans-weighted")    ml.KMeans(n_clusters=8, random_state=3).fit(X, sample_weight=w)
@lane("dbscan-brute-l1")    ml.DBSCAN(eps=0.9, min_samples=5, metric="manhattan", algorithm="brute")
@lane("dbscan-weighted")    ml.DBSCAN(eps=0.9, min_samples=5).fit(X, sample_weight=w)
@lane("kde-tophat-sqeuclidean")    ml.KernelDensity(bandwidth=0.7, kernel="tophat", metric="sqeuclidean")
@lane("kde-epanechnikov-l1")       ml.KernelDensity(bandwidth=0.7, kernel="epanechnikov", metric="l1")
@lane("kde-exponential-chebyshev") ml.KernelDensity(bandwidth=0.7, kernel="exponential", metric="chebyshev")
@lane("kde-linear-cosine")         ml.KernelDensity(bandwidth=0.7, kernel="linear", metric="cosine")
@lane("kde-cosine-minkowski")      ml.KernelDensity(bandwidth=0.7, kernel="cosine", metric="minkowski")
@lane("kde-weighted")              ml.KernelDensity(bandwidth=0.7).fit(X, sample_weight=w)
@lane("pca-full-whiten")    ml.PCA(n_components=4, whiten=True, svd_solver="full")   # pca_fit_full, a different algorithm
@lane("ols-no-intercept")   ml.LinearRegression(fit_intercept=False)
@lane("ols-weighted")       ml.LinearRegression().fit(X, yr, sample_weight=w)
@lane("ridge-no-intercept") ml.Ridge(alpha=1.0, fit_intercept=False)
@lane("logistic-l1")        ml.LogisticRegression(penalty="l1", C=1.0, max_iter=50)          # OWL-QN, a different solver
@lane("logistic-elasticnet")ml.LogisticRegression(penalty="elasticnet", l1_ratio=0.5, max_iter=50)
@lane("logistic-unpenalized-no-intercept")  ml.LogisticRegression(penalty=None, fit_intercept=False, max_iter=50)
@lane("elasticnet-l2end-no-intercept")      ml.ElasticNet(alpha=0.01, l1_ratio=0.0, fit_intercept=False, max_iter=200)
@lane("svc-linear")         ml.SVC(C=1.0, kernel="linear", max_iter=200)
@lane("svr-linear")         ml.SVR(C=1.0, kernel="linear", epsilon=0.1, max_iter=200)
@lane("knn-sqeuclidean")    ml.NearestNeighbors(n_neighbors=8, metric="sqeuclidean")
@lane("knn-manhattan")      ml.NearestNeighbors(n_neighbors=8, metric="manhattan")
@lane("knn-chebyshev")      ml.NearestNeighbors(n_neighbors=8, metric="chebyshev")
@lane("knn-cosine")         ml.NearestNeighbors(n_neighbors=8, metric="cosine")
@lane("knn-minkowski-p3")   ml.NearestNeighbors(n_neighbors=8, metric="minkowski", p=3)
@lane("knn-rbc")            ml.NearestNeighbors(n_neighbors=8, algorithm="rbc")   # a different entry AND L2SqrtUnexpanded
@lane("knn-clf-distance")   ml.KNeighborsClassifier(n_neighbors=8, weights="distance")
@lane("knn-reg-distance")   ml.KNeighborsRegressor(n_neighbors=8, weights="distance")
@lane("radius-minkowski-p3")ml.RadiusNeighbors(radius=r, metric="minkowski", p=3)
@lane("radius-manhattan")   ml.RadiusNeighbors(radius=r, metric="manhattan")
@lane("radius-chebyshev")   ml.RadiusNeighbors(radius=r, metric="chebyshev")
@lane("standard-scaler-no-mean")  ml.StandardScaler(with_mean=False)
@lane("standard-scaler-no-std")   ml.StandardScaler(with_std=False)
@lane("minmax-scaler-clip")       ml.MinMaxScaler(feature_range=(-1.0, 1.0), clip=True)
@lane("spectral-precomputed")     ml.SpectralClustering(n_clusters=4, affinity="precomputed", random_state=3).fit(A)
@lane("holtwinters-multiplicative") ml.ExponentialSmoothing(series_pos, seasonal="multiplicative", seasonal_periods=12)
@lane("kpss")                     kpss_test(y, d=0); kpss_test(y, d=1, D=1, s=12, return_statistic=True)
@lane("arima-011")                ml.ARIMA(order=(0, 1, 1))                                    # d>0, q>0, k=0
@lane("arima-seasonal-c")         ml.ARIMA(order=(1, 0, 0), seasonal_order=(1, 0, 0, 12), trend="c")
@lane("gp-matern12")   ml.ConstantKernel(1.0) * ml.Matern(1.0, nu=0.5) + ml.WhiteKernel(0.1)
@lane("gp-matern32")   ml.ConstantKernel(1.0) * ml.Matern(1.0, nu=1.5) + ml.WhiteKernel(0.1)
@lane("gp-matern52-ard") ml.ConstantKernel(1.0) * ml.Matern([1.0, 2.0, 0.5, 4.0], nu=2.5) + ml.WhiteKernel(0.1)
@lane("gemm-transposed")  matmul(a, b, transpose_b=True); matmul(a, b, transpose_a=True)      # OP_NT, OP_TN
@lane("metrics-classification")
  precision/recall/f1 at average in {None,'binary','micro','macro','weighted'} x zero_division in {0,1};
  log_loss(normalize=True/False); entropy(base=None/2); roc_auc_score; confusion_matrix;
  precision_recall_curve; mse/mae/rmse; rand_score; mutual_info_score; homogeneity/completeness;
  kl_divergence; trustworthiness; silhouette_samples; v_measure_score(beta=2.0)      # 19 functions with no cell today
@lane("cross-val")   ml.model_selection.cross_val_score(ml.Ridge(alpha=1.0), X, yr, cv=3)
```

Host inference lanes (`host-forest`, `host-gbdt`, `host-classical`) belong in the gate
tools, not here: on a GPU box they duplicate what `forest_host_gate.py` and
`classical_host_gate.py` already compare, and elsewhere they must record the by-name
`no CPU implementation of <binding>.<function> yet` sentence, never a hash.

Fixture cautions. `kde-linear-cosine` and `knn-cosine` divide by a row norm; the `dupes`
fixture has an all-zero COLUMN, not row, and DEVIATION 553 refuses an all-zero row, so any
fixture producing one reads REFUSED. `spectral-precomputed` must derive its affinity
matrix in fixed-order host arithmetic or the input itself becomes vendor dependent.
`holtwinters-multiplicative` and the poisson, gamma and inverse-gaussian criteria need
positive targets; shift the fixture, do not clamp on the device.

## 4. Built and identity-gated, no public door

| lane | Mojo lines | gate | vendor evidence | path |
|---|---|---|---|---|
| HDBSCAN | 7,147 | oracle + sabotage, pixi task | none | `hdbscan/checks/hdbscan_check.mojo` |
| Gaussian mixture | 7,888 | E-step, M-step, sabotage | none, ledger rows 86-91 reserved | `mixture/checks/gmm_check.mojo` |
| kernel methods | 7,719 | kernel matrix, random features, sabotage | none, rows 78-85 reserved | `kernel_methods/checks/km_check.mojo` |
| resampling, bootstrap intervals | 7,159 | index map, intervals, statistics | none | `resample/checks/resample_check.mojo` |
| Cholesky potrf, trsm, logdet, solve | 6,275 | sabotage | only as a blob inside the GP build | `cholesky/checks/cholesky_check.mojo`, `bindings/build_gp.sh:233` |
| IVF index | 5,458 | layout sabotage, large-k, large-probe | Apple + NVIDIA + AMD, IDENTICAL card byte-identical (2026-09-14) | `ivf/checks/ivf_check.mojo` |
| embedding lane | 6,449 | check; sabotage arms run 2026-09-14, eleven of sixteen bite on NVIDIA and AMD, ten on Apple (bench/results/ivf_embed_km_legs_2026-09-14) | Apple + NVIDIA + AMD card byte-identical, no pixi task | `embedding/checks/embedding_check.mojo:12` |
| GPT-2 byte BPE tokenizer | 1,861 | 43/43 exact tiktoken ids | no float arithmetic, no identity arm by construction | `tokenizer/checks/tokenizer_check.mojo` |

Inside shipped bindings, implemented, gated, never routed from Python:

- Multinomial (softmax) logistic regression: `glm/impl/qn/glm_softmax.mojo`, dispatched at
  `glm/impl/qn/qn.mojo:140-141`, blocked at `glm/estimator.mojo:320` (`pams.loss =
  QN_LOSS_LOGISTIC` hardcoded) and refused at `python/mojolearn/linear_model.py:1013-1018`.
  Its five-arm bitwise gate `glm/checks/multinomial_check.mojo` has NO INVOKER anywhere
  (no pixi task, no tools/ script, no workflow). Same for the six other QN losses and
  `glm/checks/qn_losses_check.mojo`.
- Six training primitives exported by the shipped training binding and implemented in
  `python/mojolearn/_training_impl.py:1905-1983` (embedding, rms_norm, linear, forward and
  backward) but omitted from `python/mojolearn/training.py:21-42`.
- `forest_resident_layout`, `forest_vector_groves` (`bindings/_mojolearn_rf.mojo:947,954`),
  `gbdt_per_round_paths`, `trees_stage_copy_policy`, `trees_shared_counts_mask`: gated,
  reachable only from bench scripts.
- KMeans `metric` (cosine, sqrt) and `oversampling_factor` (the classic sequential
  k-means++ arm) exist in `cluster/impl/kmeans_params.mojo` and are unreachable from Python.
- `parked/deviation-250-partitions-reduce.patch` is unapplied and `IDENTITY_PATHS.md:39`
  and `:189` still say two GBDT leaf folds depend on it. `parked/glm-qn-losses-wip...patch`
  targets a path that no longer exists and is dead.

The package's own register of named absences, `_NOT_YET` in
`python/mojolearn/__init__.py:365`, is EMPTY. Every row above belongs in it or in a
binding.

Exposed, not shipped: seven of the eight `bindings/_mojolearn_*_host.mojo` are in no wheel
on either platform (only `_mojolearn_byte_lm_host` is named in any packaging script). The
forest host classes are in `__all__` and raise ImportError naming the build script; the
classical host classes are not in `__all__` at all. `kpss_test` and `select_d` are public
with no three-vendor record (`IDENTITY_PATHS.md:421`, "tsa has no row") and no CPU twin.

## 5. What to do, ranked by claim per hour

1. Add the 57 lanes and the `nan` fixture to `tools/identity_break.py`, run the three
   columns once. About 100 lanes, roughly 900 train cells, two rented boxes plus the Mac.
   Every uncovered numeric path enters the claim in one rerun.
2. Fix the three defects in section 2: flip `byte-lm-host-infer` to the shipped default and
   add the single-thread arm as a second lane; refuse `bagging_temperature` and `subsample`
   by name without a bootstrap type; refuse `cross_val_score(groups=)`.
3. Give `glm/checks/multinomial_check.mojo` and `qn_losses_check.mojo` a pixi task and run
   them on three vendors. If green, route `LogisticRegression` for more than two classes.
   That is the cheapest new capability in the tree: the code and the gate exist.
4. Fill `_NOT_YET` with every row of section 4, or give the row a binding. Never both
   absent.
5. Ship the seven host bindings in the wheels (the host-surface manifest lane, in flight on
   a peer branch Sep 14), so the CPU claim is checkable with pip alone.
6. Expose or park: HDBSCAN, GMM, kernel methods, resample, IVF, Cholesky, tokenizer. Each
   is one binding, one Python class, one lane, three columns. Or move its checks to
   parked/ and say so.
7. Ship the harness and the three columns in the wheel so `python -m mojolearn identity`
   reproduces the diff on any supported GPU with no clone (the "verify from outside" item).
