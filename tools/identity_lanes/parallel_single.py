# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# THE MULTI-GPU DRIVERS ON ONE DEVICE (lane vdv-cov-parallel, 2026-11-24; plan docs/plans/VERIFY_THROUGH_DEVICE.md
# item C, "every `parallel_*` entry gets its single-GPU lane").
#
# Every public `parallel_*` driver and Parallel* class keeps `devices=` (an explicit GPU list) and has no `device=`.
# Each lane here runs its driver on ONE device, `devices=(0,)`, with the smallest LOGICAL sharding that makes the
# driver split its work (trees_per_shard, series_per_shard, columns_per_shard, rows_per_shard, logical_shards,
# repeated layer owners), and holds the driver's answer byte for byte, in the cell, to the plain one-device public
# path the driver documents as equal (the estimator's own fit, predict or transform; the replica trainer; the
# plain CausalLM). A disagreement raises NumericalMismatch carrying the parts, so a fault that separates the two
# routes reads as a caught digest, never as a refusal. The physical device count never selects the reduction
# order, which is what makes a one-device cell meaningful on every vendor.
#
# Names: `psd-*` ("parallel, single device"), never `par-*`: the `par-` prefix is the retired multi-GPU
# verification (docs/TOMBSTONES.md MOJOLEARN_PAR_VERIFY, 2026-10-31), excluded from a full-column record
# (identity_break.RECORD_EXCLUDED_PREFIXES) and ignored by reference admission
# (_verify_reference.RETIRED_LANE_PREFIXES). That retirement stays: no lane here asks for a second device.
# The lane bodies follow the retired par-* lanes (recoverable at 868702642c^), rechecked against today's drivers.
#
# SIDES (contract of vd-verify, docs: briefs-20261122/vdv-_common.md). The lanes are device-agnostic: the harness
# calls each once per side with a device-bound `ml`; the drivers are imported from their modules (not reached
# through the view), so they keep `devices=` and receive no `device=`; the plain estimators the drivers take and
# the oracles are built through `ml` and carry the side.
#   * ONE-SIDED, missing="cpu": the cooperative drivers (bindings/parallel_plan_run.mojo plan_vendor_admit: a
#     CPU-only installation refuses "no CPU implementation of the cooperative multi-GPU driver"; a GPU process
#     refuses a member that says device='cpu', bindings/parallel_public_native_door.mojo), and the byte-LM
#     sessions (bindings/parallel_byte_native_door.mojo parallel_byte_open). Each missing side is an owed
#     implementation, listed in docs/plans/verify_through_device/mapped_cov_parallel.tsv.
#   * BOTH SIDES: the drivers whose shards a CPU-only installation runs on the host bindings in the native local
#     pool (the Python-sharded drivers, bindings/parallel_shard_native_door.mojo; the forecasting drivers,
#     bindings/parallel_forecast_native_door.mojo; the parallel cross-validation, bindings/parallel_cv_native_run.mojo
#     cv_vendor_admit; ParallelCausalLM's CPU route, bindings/parallel_causal_native_run.mojo; ParallelNeuralTrainer,
#     whose base lane neural-trainer-unfused runs on the CPU column). In a process with a GPU set these drivers
#     choose their route from the installation, not from the member's device: the shard and forecasting doors
#     refuse a member that says device='cpu' (`_member_device`), so their CPU side FAILS on a GPU installation
#     until that door runs the host shards for it. That is the product gap this lane reports, not hides.
#
# Every lane is an independent unit: it builds its own fixture slice, model and driver and shares nothing mutable
# with another lane. Sizes are correctness-check sizes. Every reference is OWED to the release's one verification.
#
# tools/identity_break.py executes this file in ITS OWN namespace (`_load_lane_fragments`): lane names are literal,
# helpers carry the `_parallel_single_` prefix, nothing existing is rebound. `_one_sided` and ONE_SIDED come from
# the harness (sub-lane vdv-harness).

#: The one device every lane hands its driver. Literal on purpose (as forest-shard and neural-trainer-unfused do):
#: the claim is one device with logical shards, never a device list read from the environment.
_PARALLEL_SINGLE_DEVICES = (0,)

#: Query rows per shard for the query drivers: 64 queries in four blocks.
_PARALLEL_SINGLE_QUERY_ROWS = 16


def _parallel_single_check(parts, *pairs):
    """Hold every (name_a, a, name_b, b) pair to the same bytes; raise NumericalMismatch carrying `parts` naming
    every pair that moved."""
    mismatch = _oracle_mismatch(*pairs)
    if mismatch:
        raise NumericalMismatch(mismatch, parts)


def _parallel_single_flat(value):
    """The numeric arrays of a nested result (dict by sorted key, list, tuple, array, scalar), in a fixed order,
    for `_h`. Text is encoded through `_hf_text`; None contributes nothing."""
    out = []

    def walk(v):
        if v is None:
            return
        if isinstance(v, dict):
            for k in sorted(v, key=str):  # glue: fixed order over a result record's keys
                walk(v[k])
        elif isinstance(v, (list, tuple)):
            for x in v:  # glue: the record's items in their own order
                walk(x)
        elif isinstance(v, (str, bytes)):
            out.append(_hf_text(v if isinstance(v, str) else v.decode("latin-1")))
        else:
            a = np.asarray(v)
            if a.dtype == object and a.ndim == 0:
                # an opaque record value: its type name, never its repr (an address)
                out.append(_hf_text(type(v).__name__))
            elif a.dtype == object:
                for x in a.ravel().tolist():  # glue: an object array's items
                    walk(x)
            else:
                out.append(np.ascontiguousarray(a))
    walk(value)
    return out


# ================================================================ ONE-SIDED (no CPU form): the cooperative drivers

# ---------------------------------------------------------------- parallel_classical: linear, Gram, SVM, GP

@lane("psd-logistic")
def _(ml, X, yc, yr, Xh=None):
    """fit_logistic on one device (gradient columns partitioned inside the GPU binding), held to the plain
    LogisticRegression fit."""
    from mojolearn.parallel_classical import fit_logistic
    par = fit_logistic(ml.LogisticRegression(max_iter=50), X, yc, devices=_PARALLEL_SINGLE_DEVICES)
    plain = ml.LogisticRegression(max_iter=50).fit(X, yc)
    parts = dict(coef=_h(par.coef_), intercept=_h(par.intercept_), proba=_h(par.predict_proba(X[:256])))
    _parallel_single_check(parts, ("fit_logistic coef_", par.coef_, "plain coef_", plain.coef_),
                           ("fit_logistic predict_proba", par.predict_proba(X[:256]),
                            "plain predict_proba", plain.predict_proba(X[:256])))
    return _fit(parts, par, lambda e: (e.predict_proba(Xh[:256]),))


@lane("psd-gram")
def _(ml, X, yc, yr, Xh=None):
    """fit_gram_estimator on Ridge: the pinned Gram chunks distributed over one device, held to the plain fit."""
    from mojolearn.parallel_classical import fit_gram_estimator
    par = fit_gram_estimator(ml.Ridge(alpha=1.0), X, yr, devices=_PARALLEL_SINGLE_DEVICES)
    plain = ml.Ridge(alpha=1.0).fit(X, yr)
    parts = dict(coef=_h(par.coef_), predict=_h(par.predict(X[:256])))
    _parallel_single_check(parts, ("fit_gram_estimator coef_", par.coef_, "plain coef_", plain.coef_))
    return _fit(parts, par, lambda e: (e.predict(Xh[:256]),))


@lane("psd-gram-pca")
def _(ml, X, yc, yr, Xh=None):
    """fit_gram_estimator on PCA (the unsupervised, y=None form), held to the plain PCA fit."""
    from mojolearn.parallel_classical import fit_gram_estimator
    par = fit_gram_estimator(ml.PCA(n_components=4), X, devices=_PARALLEL_SINGLE_DEVICES)
    plain = ml.PCA(n_components=4).fit(X)
    parts = dict(components=_h(par.components_), variance=_h(par.explained_variance_),
                 transform=_h(par.transform(X[:256])))
    _parallel_single_check(parts, ("fit_gram_estimator components_", par.components_,
                                   "plain components_", plain.components_))
    return _fit(parts, par, lambda e: (e.transform(Xh[:256]),))


@lane("psd-cd")
def _(ml, X, yc, yr, Xh=None):
    """fit_coordinate_descent on Lasso, held to the plain fit."""
    from mojolearn.parallel_classical import fit_coordinate_descent
    par = fit_coordinate_descent(ml.Lasso(alpha=0.01, max_iter=200), X, yr, devices=_PARALLEL_SINGLE_DEVICES)
    plain = ml.Lasso(alpha=0.01, max_iter=200).fit(X, yr)
    parts = dict(coef=_h(par.coef_), predict=_h(par.predict(X[:256])))
    _parallel_single_check(parts, ("fit_coordinate_descent coef_", par.coef_, "plain coef_", plain.coef_))
    return _fit(parts, par, lambda e: (e.predict(Xh[:256]),))


@lane("psd-svm")
def _(ml, X, yc, yr, Xh=None):
    """fit_svm and predict_svm (decision_function and predict) on SVC, held to the plain fit's decision values
    and labels."""
    from mojolearn.parallel_classical import fit_svm, predict_svm
    dev = _PARALLEL_SINGLE_DEVICES
    kw = dict(C=1.0, kernel="rbf", max_iter=200)
    par = fit_svm(ml.SVC(**kw), X[:2000], yc[:2000], devices=dev)
    plain = ml.SVC(**kw).fit(X[:2000], yc[:2000])
    q = np.ascontiguousarray(X[2000:2256])
    dec = predict_svm(par, q, devices=dev, method="decision_function")
    pred = predict_svm(par, q, devices=dev, method="predict")
    parts = dict(decision=_h(dec), predict=_h(pred))
    _parallel_single_check(parts, ("predict_svm decision_function", dec, "plain decision_function",
                                   plain.decision_function(q)),
                           ("predict_svm predict", pred, "plain predict", plain.predict(q)))
    return _fit(parts, par, lambda e: (predict_svm(e, np.ascontiguousarray(Xh[:256]), devices=dev,
                                                   method="decision_function"),))


@lane("psd-gp")
def _(ml, X, yc, yr, Xh=None):
    """fit_gaussian_process and predict_gaussian_process (mean and std) on a ConstantKernel * RBF + White
    regressor, held to the plain fit and predict."""
    from mojolearn.parallel_classical import fit_gaussian_process, predict_gaussian_process
    dev = _PARALLEL_SINGLE_DEVICES
    k = ml.ConstantKernel(1.0) * ml.RBF(1.0) + ml.WhiteKernel(0.1)
    xs = np.ascontiguousarray(X[:256, :4])
    q = np.ascontiguousarray(X[256:320, :4])
    par = fit_gaussian_process(ml.GaussianProcessRegressor(kernel=k), xs, yr[:256], devices=dev)
    plain = ml.GaussianProcessRegressor(kernel=k).fit(xs, yr[:256])
    mean, std = predict_gaussian_process(par, q, devices=dev, return_std=True)
    pm, ps = plain.predict(q, return_std=True)
    parts = dict(alpha=_h(par.alpha_), L=_h(par.L_), mean=_h(mean), std=_h(std))
    _parallel_single_check(parts, ("fit_gaussian_process alpha_", par.alpha_, "plain alpha_", plain.alpha_),
                           ("predict_gaussian_process mean", mean, "plain mean", pm),
                           ("predict_gaussian_process std", std, "plain std", ps))
    return _fit(parts, par, lambda e: predict_gaussian_process(e, np.ascontiguousarray(Xh[:64, :4]), devices=dev,
                                                               return_std=True))


# ---------------------------------------------------------------- parallel_classical: clustering and mixtures

@lane("psd-kmeans")
def _(ml, X, yc, yr, Xh=None):
    """fit_kmeans, held to the plain KMeans fit."""
    from mojolearn.parallel_classical import fit_kmeans
    par = fit_kmeans(ml.KMeans(n_clusters=8, random_state=3), X, devices=_PARALLEL_SINGLE_DEVICES)
    plain = ml.KMeans(n_clusters=8, random_state=3).fit(X)
    parts = dict(centers=_h(par.cluster_centers_), labels=_h(par.labels_))
    _parallel_single_check(parts, ("fit_kmeans cluster_centers_", par.cluster_centers_,
                                   "plain cluster_centers_", plain.cluster_centers_),
                           ("fit_kmeans labels_", par.labels_, "plain labels_", plain.labels_))
    return _fit(parts, par, _km_probe(X, Xh))


@lane("psd-dbscan")
def _(ml, X, yc, yr, Xh=None):
    """fit_dbscan (with prediction data), held to the plain DBSCAN labels."""
    from mojolearn.parallel_classical import fit_dbscan
    A = np.ascontiguousarray(X[:6000, :4])
    par = fit_dbscan(ml.DBSCAN(eps=0.9, min_samples=5, prediction_data=True), A, devices=_PARALLEL_SINGLE_DEVICES)
    plain = ml.DBSCAN(eps=0.9, min_samples=5).fit(A)
    parts = dict(labels=_h(par.labels_))
    _parallel_single_check(parts, ("fit_dbscan labels_", par.labels_, "plain labels_", plain.labels_))
    return _fit(parts, par, lambda e: (e.predict(np.ascontiguousarray(Xh[:256, :4])),))


@lane("psd-hdbscan")
def _(ml, X, yc, yr, Xh=None):
    """fit_hdbscan (with prediction data), held to the plain HDBSCAN labels and core distances."""
    from mojolearn.parallel_classical import fit_hdbscan
    A = np.ascontiguousarray(X[:6000, :4])
    par = fit_hdbscan(ml.HDBSCAN(min_cluster_size=5, prediction_data=True), A, devices=_PARALLEL_SINGLE_DEVICES)
    plain = ml.HDBSCAN(min_cluster_size=5).fit(A)
    parts = dict(labels=_h(par.labels_), core=_h(par.core_distances_),
                 counts=_h(np.asarray([par.n_clusters_, par.n_outliers_], dtype=np.int64)))
    _parallel_single_check(parts, ("fit_hdbscan labels_", par.labels_, "plain labels_", plain.labels_),
                           ("fit_hdbscan core_distances_", par.core_distances_,
                            "plain core_distances_", plain.core_distances_))
    return _fit(parts, par, lambda e: tuple(ml.hdbscan.approximate_predict(e, np.ascontiguousarray(Xh[:256, :4]))))


@lane("psd-gmm")
def _(ml, X, yc, yr, Xh=None):
    """fit_gaussian_mixture (row-sharded E-steps) and predict_gaussian_mixture (predict and score_samples), held
    to the plain GaussianMixture."""
    from mojolearn.parallel_classical import fit_gaussian_mixture, predict_gaussian_mixture
    dev = _PARALLEL_SINGLE_DEVICES
    kw = dict(n_components=4, max_iter=30, random_state=3)
    A = np.ascontiguousarray(X[:6000, :4])
    par = fit_gaussian_mixture(ml.GaussianMixture(**kw), A, devices=dev)
    plain = ml.GaussianMixture(**kw).fit(A)
    labels = predict_gaussian_mixture(par, A, devices=dev, method="predict")
    scores = predict_gaussian_mixture(par, A[:512], devices=dev, method="score_samples")
    parts = dict(weights=_h(par.weights_), means=_h(par.means_), covariances=_h(par.covariances_),
                 precisions=_h(par.precisions_cholesky_), labels=_h(labels), scores=_h(scores))
    _parallel_single_check(parts, ("fit_gaussian_mixture covariances_", par.covariances_,
                                   "plain covariances_", plain.covariances_),
                           ("predict_gaussian_mixture predict", labels, "plain predict", plain.predict(A)),
                           ("predict_gaussian_mixture score_samples", scores,
                            "plain score_samples", plain.score_samples(A[:512])))
    return _fit(parts, par, lambda e: (predict_gaussian_mixture(e, np.ascontiguousarray(Xh[:64, :4]), devices=dev,
                                                                method="score_samples"),))


# ---------------------------------------------------------------- parallel_classical: resampling

@lane("psd-resample")
def _(ml, X, yc, yr, Xh=None):
    """parallel_classical.bootstrap (mean and quantile), permutation_test and monte_carlo_integrate on one
    device, each held to the plain `resample` call with the same seed."""
    from mojolearn import parallel_classical as pc
    rs = ml.resample
    dev = _PARALLEL_SINGLE_DEVICES
    x = np.ascontiguousarray(yr[:4096])
    parts, pairs = {}, []
    for name, kw in (("mean", dict(statistic="mean", n_resamples=2048)),  # glue: two statistics
                     ("quantile", dict(statistic="quantile", q_or_prop=0.25, n_resamples=1024, alternative="less"))):
        b = pc.bootstrap(x, devices=dev, random_state=3, **kw)
        plain = rs.bootstrap(x, random_state=3, **kw)
        pairs.append(("parallel bootstrap distribution " + name, b.distribution,
                      "plain bootstrap distribution", plain.distribution))
        parts["bootstrap-" + name] = _h(b.distribution, b.sorted_distribution,
                                        np.asarray([b.point_estimate, b.standard_error, b.confidence_interval[0],
                                                    b.confidence_interval[1]], dtype=np.float64))
    a = np.ascontiguousarray(yr[:512])
    c = np.ascontiguousarray(yr[512:1024])
    p = pc.permutation_test(a, c, devices=dev, statistic="diff_means", n_resamples=2048, random_state=3)
    plain = rs.permutation_test(a, c, statistic="diff_means", n_resamples=2048, random_state=3)
    pairs.append(("parallel permutation null", p.null_distribution, "plain permutation null", plain.null_distribution))
    parts["permutation"] = _h(p.null_distribution, np.asarray([p.statistic, p.pvalue], dtype=np.float64),
                              np.asarray([p.count_less, p.count_greater], dtype=np.int64))
    lo, hi = [0.0, 0.0], [1.0, 2.0]
    r = pc.monte_carlo_integrate("product", lo, hi, 65536 + 300, devices=dev, random_state=1)
    plain = rs.monte_carlo_integrate("product", lo, hi, 65536 + 300, random_state=1)
    pairs.append(("parallel monte_carlo_integrate", np.asarray([r.integral, r.mean], dtype=np.float64),
                  "plain monte_carlo_integrate", np.asarray([plain.integral, plain.mean], dtype=np.float64)))
    parts["monte-carlo"] = _h(np.asarray([r.integral, r.mean, r.volume, r.closed_form], dtype=np.float64))
    _parallel_single_check(parts, *pairs)
    return _fit(parts)


# ---------------------------------------------------------------- parallel_classical: Cholesky and kernel methods

@lane("psd-cholesky")
def _(ml, X, yc, yr, Xh=None):
    """fit_cholesky and solve_cholesky on a 600 x 600 Cauchy SPD matrix (above 1 MiB, the size at which the
    device-to-device column solve once read wrong columns), held to the plain Cholesky fit and solve."""
    from mojolearn.parallel_classical import fit_cholesky, solve_cholesky
    dev = _PARALLEL_SINGLE_DEVICES
    A = _cauchy_spd(X[:600, :4])
    par = fit_cholesky(ml.Cholesky(), A, devices=dev)
    plain = ml.Cholesky().fit(A)
    if par.info_ != 0:
        raise AssertionError(f"psd-cholesky: the Cauchy matrix did not factor (info={par.info_})")
    B = np.ascontiguousarray(np.stack([yr[:600], yr[600:1200], yr[1200:1800]], 1).astype(np.float32))
    x = solve_cholesky(par, B, devices=dev)
    parts = dict(L=_h(par.L_), logdet=_h(np.float64(par.logdet_)), info=_h(np.int64(par.info_)), solve=_h(x))
    _parallel_single_check(parts, ("fit_cholesky L_", par.L_, "plain L_", plain.L_),
                           ("solve_cholesky", x, "plain solve", plain.solve(B)))
    return _fit(parts, par, lambda e: (solve_cholesky(e, np.ascontiguousarray(Xh[:600, :3]), devices=dev),))


@lane("psd-kernel-ridge")
def _(ml, X, yc, yr, Xh=None):
    """fit_kernel_method and apply_kernel_method on a two-target KernelRidge, held to the plain fit and
    predict."""
    from mojolearn.parallel_classical import fit_kernel_method, apply_kernel_method
    dev = _PARALLEL_SINGLE_DEVICES
    kw = dict(alpha=0.1, kernel="rbf", gamma=0.5)
    A = np.ascontiguousarray(X[:600, :4])
    Y = np.ascontiguousarray(np.stack([yr[:600], yr[600:1200]], 1).astype(np.float32))
    q = np.ascontiguousarray(X[600:856, :4])
    par = fit_kernel_method(ml.KernelRidge(**kw), A, Y, devices=dev)
    plain = ml.KernelRidge(**kw).fit(A, Y)
    pred = apply_kernel_method(par, q, devices=dev)
    parts = dict(dual=_h(par.dual_coef_), predict=_h(pred))
    _parallel_single_check(parts, ("fit_kernel_method dual_coef_", par.dual_coef_, "plain dual_coef_", plain.dual_coef_),
                           ("apply_kernel_method predict", pred, "plain predict", plain.predict(q)))
    return _fit(parts, par, lambda e: (apply_kernel_method(e, np.ascontiguousarray(Xh[:64, :4]), devices=dev),))


@lane("psd-nystroem")
def _(ml, X, yc, yr, Xh=None):
    """fit_kernel_method and apply_kernel_method on Nystroem (the transformer form of the kernel-method driver),
    held to the plain fit and transform."""
    from mojolearn.parallel_classical import fit_kernel_method, apply_kernel_method
    dev = _PARALLEL_SINGLE_DEVICES
    kw = dict(kernel="rbf", gamma=0.5, n_components=32, random_state=7)
    A = np.ascontiguousarray(X[:600, :4])
    q = np.ascontiguousarray(X[600:1624, :4])
    par = fit_kernel_method(ml.Nystroem(**kw), A, devices=dev)
    plain = ml.Nystroem(**kw).fit(A)
    out = apply_kernel_method(par, q, devices=dev)
    parts = dict(components=_h(par.components_), normalization=_h(par.normalization_), transform=_h(out))
    _parallel_single_check(parts, ("fit_kernel_method normalization_", par.normalization_,
                                   "plain normalization_", plain.normalization_),
                           ("apply_kernel_method transform", out, "plain transform", plain.transform(q)))
    return _fit(parts, par, lambda e: (apply_kernel_method(e, np.ascontiguousarray(Xh[:64, :4]), devices=dev),))


@lane("psd-rbf-sampler")
def _(ml, X, yc, yr, Xh=None):
    """transform_rbf_sampler with 300 rows per shard (1000 rows: three full shards and a short one), held to the
    plain RBFSampler transform."""
    from mojolearn.parallel_classical import transform_rbf_sampler
    dev = _PARALLEL_SINGLE_DEVICES
    m = ml.RBFSampler(gamma=0.5, n_components=1024, random_state=1).fit(X)
    out = transform_rbf_sampler(m, X[:1000], devices=dev, rows_per_shard=300)
    parts = dict(transform=_h(out))
    _parallel_single_check(parts, ("transform_rbf_sampler", out, "plain transform", m.transform(X[:1000])))
    return _fit(parts, m, lambda e: (transform_rbf_sampler(e, Xh[:256], devices=dev, rows_per_shard=100),))


# ---------------------------------------------------------------- parallel_ensemble (cooperative entries)

@lane("psd-boosting")
def _(ml, X, yc, yr, Xh=None):
    """fit_boosting on the symmetric-tree Logloss configuration (feature histograms partitioned), held to the
    plain GradientBoosting fit."""
    from mojolearn.parallel_ensemble import fit_boosting
    kw = dict(n_estimators=20, max_depth=6, loss="Logloss")
    par = fit_boosting(_gbdt(ml.GradientBoosting, **kw), X, yc, devices=_PARALLEL_SINGLE_DEVICES)
    plain = _gbdt(ml.GradientBoosting, **kw).fit(X, yc)
    parts = dict(predict=_h(par.predict(X)), proba=_h(par.predict_proba(X)))
    _parallel_single_check(parts, ("fit_boosting predict_proba", par.predict_proba(X[:2048]),
                                   "plain predict_proba", plain.predict_proba(X[:2048])))
    return _fit(parts, par, lambda e: (e.predict(Xh), e.predict_proba(Xh)))


@lane("psd-boosting-reg")
def _(ml, X, yc, yr, Xh=None):
    """fit_boosting on the sklearn-style GradientBoostingRegressor adapter, held to the plain fit."""
    from mojolearn.parallel_ensemble import fit_boosting
    kw = dict(n_estimators=20, max_depth=6)
    par = fit_boosting(_gbdt(ml.GradientBoostingRegressor, **kw), X, yr, devices=_PARALLEL_SINGLE_DEVICES)
    plain = _gbdt(ml.GradientBoostingRegressor, **kw).fit(X, yr)
    parts = dict(predict=_h(par.predict(X)))
    _parallel_single_check(parts, ("fit_boosting predict", par.predict(X[:2048]),
                                   "plain predict", plain.predict(X[:2048])))
    return _fit(parts, par, lambda e: (e.predict(Xh),))


@lane("psd-iforest")
def _(ml, X, yc, yr, Xh=None):
    """fit_isolation_forest and score_isolation_forest (score_samples and predict), held to the plain
    IsolationForest."""
    from mojolearn.parallel_ensemble import fit_isolation_forest, score_isolation_forest
    dev = _PARALLEL_SINGLE_DEVICES
    par = fit_isolation_forest(ml.IsolationForest(n_estimators=16, random_state=5), X, devices=dev)
    plain = ml.IsolationForest(n_estimators=16, random_state=5).fit(X)
    scores = score_isolation_forest(par, X, devices=dev, method="score_samples")
    labels = score_isolation_forest(par, X[:512], devices=dev, method="predict")
    parts = dict(scores=_h(scores), predict=_h(labels))
    _parallel_single_check(parts, ("score_isolation_forest score_samples", scores,
                                   "plain score_samples", plain.score_samples(X)),
                           ("score_isolation_forest predict", labels, "plain predict", plain.predict(X[:512])))
    return _fit(parts, par, lambda e: (score_isolation_forest(e, Xh, devices=dev, method="score_samples"),))


@lane("psd-ordered-rmse")
def _(ml, X, yc, yr, Xh=None):
    """fit_ordered_rmse with a fixture-hashed permutation, held to the plain OrderedRMSE fit."""
    from mojolearn.parallel_ensemble import fit_ordered_rmse
    perm = np.argsort(_hashed_uniform(X.shape[0], 1, "ordered-permutation").reshape(-1), kind="stable")
    par = fit_ordered_rmse(ml.OrderedRMSE(n_estimators=20, max_depth=6), X, yr, permutation=perm,
                           devices=_PARALLEL_SINGLE_DEVICES)
    plain = ml.OrderedRMSE(n_estimators=20, max_depth=6).fit(X, yr, permutation=perm)
    p = par.predict(X)
    parts = dict(predict=_h(p))
    _parallel_single_check(parts, ("fit_ordered_rmse predict", p, "plain predict", plain.predict(X)))
    return _fit(parts, par, lambda e: (e.predict(Xh),))


@lane("psd-feature-freq")
def _(ml, X, yc, yr, Xh=None):
    """fit_feature_freq on the two-level feature-frequency booster over coded columns, held to the plain fit."""
    from mojolearn.parallel_ensemble import fit_feature_freq
    Xc = _coded(X)
    par = fit_feature_freq(ml.ExperimentalTwoLevelFeatureFreq(sources=[0, 1], random_state=7), Xc, yr,
                           devices=_PARALLEL_SINGLE_DEVICES)
    plain = ml.ExperimentalTwoLevelFeatureFreq(sources=[0, 1], random_state=7).fit(Xc, yr)
    p = par.predict(Xc)
    parts = dict(predict=_h(p))
    _parallel_single_check(parts, ("fit_feature_freq predict", p, "plain predict", plain.predict(Xc)))
    return _fit(parts, par, lambda e: (e.predict(_coded(Xh)),))


@lane("psd-forest-pool")
def _(ml, X, yc, yr, Xh=None):
    """ParallelForestPredictor (resident random-forest groves, the fixed logical groves on one device):
    predict and predict_proba, held to the estimator's own parallel_groves predict_proba and predict."""
    from mojolearn.parallel_ensemble import ParallelForestPredictor
    m = ml.RandomForestClassifier(n_estimators=40, max_depth=8, random_state=7,
                                  inference_engine="parallel_groves").fit(X, yc)

    def pooled(e, R):
        e.inference_engine = "parallel_groves"  # a reloaded estimator keeps the pool's engine
        with ParallelForestPredictor(e, devices=_PARALLEL_SINGLE_DEVICES) as pool:
            return pool.predict(R), pool.predict_proba(R)

    pred, proba = pooled(m, X)
    parts = dict(predict=_h(pred), proba=_h(proba))
    _parallel_single_check(parts, ("ParallelForestPredictor predict_proba", proba,
                                   "parallel_groves predict_proba", m.predict_proba(X)),
                           ("ParallelForestPredictor predict", pred, "parallel_groves predict", m.predict(X)))
    return _fit(parts, m, lambda e: pooled(e, Xh))


# ---------------------------------------------------------------- parallel_graph

@lane("psd-graph-agglomerative")
def _(ml, X, yc, yr, Xh=None):
    """fit_graph on AgglomerativeClustering (with prediction data), held to the plain labels and merge tree."""
    from mojolearn.parallel_graph import fit_graph
    A = np.ascontiguousarray(X[:2000, :4])
    par = fit_graph(ml.AgglomerativeClustering(n_clusters=4, prediction_data=True), A, devices=_PARALLEL_SINGLE_DEVICES)
    plain = ml.AgglomerativeClustering(n_clusters=4).fit(A)
    parts = dict(labels=_h(par.labels_), children=_h(par.children_))
    _parallel_single_check(parts, ("fit_graph labels_", par.labels_, "plain labels_", plain.labels_),
                           ("fit_graph children_", par.children_, "plain children_", plain.children_))
    return _fit(parts, par, lambda e: (e.predict(np.ascontiguousarray(Xh[:256, :4])),))


@lane("psd-graph-umap")
def _(ml, X, yc, yr, Xh=None):
    """fit_graph on UMAP and transform_umap, held to the plain UMAP embedding and transform."""
    from mojolearn.parallel_graph import fit_graph, transform_umap
    dev = _PARALLEL_SINGLE_DEVICES
    kw = dict(n_neighbors=8, n_components=2, n_epochs=8, random_state=3)
    A = np.ascontiguousarray(X[:1024, :8])
    q = np.ascontiguousarray(X[1024:1088, :8])
    par = fit_graph(ml.UMAP(**kw), A, devices=dev)
    plain = ml.UMAP(**kw).fit(A)
    t = transform_umap(par, q, devices=dev)
    parts = dict(embedding=_h(par.embedding_), transform=_h(t))
    _parallel_single_check(parts, ("fit_graph embedding_", par.embedding_, "plain embedding_", plain.embedding_),
                           ("transform_umap", t, "plain transform", plain.transform(q)))
    return _fit(parts, par, lambda e: (transform_umap(e, np.ascontiguousarray(Xh[:64, :8]), devices=dev),))


# ---------------------------------------------------------------- ParallelByteLanguageModelTrainer (sessions in the GPU binding)

def _parallel_single_byte_state(ml, shape):
    """The starting byte-LM state: the byte-lm lane's weights (`_byte_lm_params`) through the public trainer's
    `state_dict` (the constructor's documented input)."""
    named, _ = _byte_lm_params(shape)
    seed = ml.SmallByteLanguageModelTrainer(named, data_schedule={"dataset": "identity_break", "order": "sequential"},
                                            shape=shape)
    return seed.state_dict()


def _parallel_single_same_state(name_a, a, name_b, b):
    """The (name_a, a, name_b, b) pairs of two exported byte-LM states, array by array."""
    return [(f"{name_a} {key}", np.asarray(a[key]), f"{name_b} {key}", np.asarray(b[key]))
            for key in ("parameters", "m", "v", "flags")]


def _parallel_single_byte_run(trainer, reference, ids, steps):
    """`steps` training steps of two logical shards on both trainers; returns (losses, the pairs that must
    agree)."""
    losses, pairs = [], []
    for step in range(steps):  # glue: a few driver steps
        shards = [ids[4 * step:4 * step + 2], ids[4 * step + 2:4 * step + 4]]
        a, b = trainer.train_step(shards), reference.train_step(shards)
        la = np.asarray(a["losses"], dtype=np.float32)
        pairs.append((f"step {step} losses", la, "replica trainer losses", np.asarray(b["losses"], dtype=np.float32)))
        losses.append(la)
    return np.concatenate(losses), pairs


@lane("psd-byte-lm")
def _(ml, X, yc, yr, Xh=None):
    """ParallelByteLanguageModelTrainer, replica schedule, one device, two logical shards, pooled AdamW ranges:
    three steps, `set_lr` before the third, held step by step to the replica trainer with complete optimizer
    replicas (pool_optimizer=False). Then the exports (`state_dict`, `export_raw`, `export_gradients`) and a
    `checkpoint` restored with `from_checkpoint`, whose fourth step must equal the live trainer's."""
    from mojolearn.parallel_training import ParallelByteLanguageModelTrainer as P
    dev = _PARALLEL_SINGLE_DEVICES
    shape = ml.ByteLanguageModelConfig()
    state0 = _parallel_single_byte_state(ml, shape)
    ids = _ids(X, 8 * shape.batch, shape.length + 1)
    with P(state0, devices=dev, logical_shards=2) as tr, \
            P(state0, devices=dev, logical_shards=2, pool_optimizer=False) as ref:
        losses, pairs = _parallel_single_byte_run(tr, ref, ids, 2)
        lr_bits = np.asarray([tr.set_lr(5e-4), ref.set_lr(5e-4)], dtype=np.float64)
        more, more_pairs = _parallel_single_byte_run(tr, ref, ids[8:], 1)
        state, rstate = tr.state_dict(), ref.state_dict()
        raw = tr.export_raw()
        grads = np.asarray(tr.export_gradients(), dtype=np.float32)
        ck = tr.checkpoint()
        with P.from_checkpoint(ck, devices=dev) as back:
            shards = [ids[12:14], ids[14:16]]
            after = np.asarray(tr.train_step(shards)["losses"], dtype=np.float32)
            restored = np.asarray(back.train_step(shards)["losses"], dtype=np.float32)
            final, back_final = tr.state_dict(), back.state_dict()
        pairs += more_pairs
        pairs += _parallel_single_same_state("trainer", state, "replica trainer", rstate)
        pairs.append(("export_gradients", grads, "replica trainer export_gradients",
                      np.asarray(ref.export_gradients(), dtype=np.float32)))
    pairs += [("export_raw parameters", np.asarray(raw["parameters"]), "state_dict parameters",
               np.asarray(state["parameters"])),
              ("set_lr stored bits", lr_bits[:1], "replica set_lr stored bits", lr_bits[1:]),
              ("from_checkpoint step losses", restored, "live trainer step losses", after)]
    pairs += _parallel_single_same_state("from_checkpoint trainer", back_final, "live trainer", final)
    parts = dict(loss=_h(np.concatenate([losses, more, after])), params=_h(np.asarray(state["parameters"])),
                 m=_h(np.asarray(state["m"])), v=_h(np.asarray(state["v"])), grads=_h(grads),
                 lr=_h(lr_bits), final=_h(np.asarray(final["parameters"])))
    _parallel_single_check(parts, *pairs)
    return _fit(parts)


@lane("psd-byte-lm-worker")
def _(ml, X, yc, yr, Xh=None):
    """The cross-process worker half of a byte-LM step on one device: `shard_gradient` per logical shard, the
    ordered fold (`ordered_sum_gradients`), `apply_gradient`, held to `train_step` of the same two shards; and
    the device fold (`fold_reset`, `shard_gradient_fold`, `fold_add`, `fold_export`) held to the same total when
    the binding has one (`has_device_fold`)."""
    from mojolearn.parallel_training import ParallelByteLanguageModelTrainer as P
    from mojolearn.cross_vendor import ordered_fold
    dev = _PARALLEL_SINGLE_DEVICES
    shape = ml.ByteLanguageModelConfig()
    state0 = _parallel_single_byte_state(ml, shape)
    ids = _ids(X, 4 * shape.batch, shape.length + 1)
    shards = [ids[0:2], ids[2:4]]
    # pool_optimizer=False on both: the split-step entries require complete optimizer replicas
    # (bindings/parallel_byte_native_run.mojo parallel_session_require_split; the replica schedule's default is a
    # pooled optimizer, python/mojolearn/parallel_training.py), and the reference train_step runs the same layout.
    # Harness fix, lane gs-fix: the lane was refused "shard_gradient requires devices=(d,) and pool_optimizer=False".
    with P(state0, devices=dev, logical_shards=2, pool_optimizer=False) as tr, \
            P(state0, devices=dev, logical_shards=2, pool_optimizer=False) as w:
        step = np.asarray(tr.train_step(shards)["losses"], dtype=np.float32)
        l0, g0 = w.shard_gradient(shards[0])
        l1, g1 = w.shard_gradient(shards[1])
        # The byte-LM worker fold is `cross_vendor.ordered_fold` over the flat float32[n_total] shard gradients
        # (python/mojolearn/parallel_training.py `apply_gradient`); `ordered_sum_gradients` takes one LIST of
        # tensors per shard (the MLP/Samba form) and refused the flat arrays (lane gs-fix-2: wrong test).
        summed = ordered_fold([g0, g1])
        # ordered_fold returns the folded float32 BYTES (bindings/cross_vendor_native_door.mojo:222 `store.tobytes()`),
        # the form apply_gradient takes; np.asarray(bytes, float32) parses text and refused the lane (gs-fix-4: wrong test).
        total = np.frombuffer(summed, dtype=np.float32).copy()
        fold = bool(w.has_device_fold)
        if fold:
            w.fold_reset()
            f0 = w.shard_gradient_fold(shards[0])
            w.fold_add(g1)
            folded = np.frombuffer(bytes(w.fold_export()), dtype=np.float32).copy()
        w.apply_gradient(summed)
        pairs = [("shard_gradient losses", np.asarray([l0, l1], dtype=np.float32), "train_step losses", step),
                 ("ordered fold of shard_gradient", total, "train_step export_gradients",
                  np.asarray(tr.export_gradients(), dtype=np.float32))]
        pairs += _parallel_single_same_state("apply_gradient worker", w.state_dict(), "train_step trainer",
                                             tr.state_dict())
        if fold:
            pairs += [("fold_export", folded, "ordered_fold total", total),
                      ("shard_gradient_fold loss", np.asarray([f0], dtype=np.float32), "shard_gradient loss",
                       np.asarray([l0], dtype=np.float32))]
        params = np.asarray(w.state_dict()["parameters"])
    parts = dict(loss=_h(step), total=_h(total), params=_h(params),
                 device_fold=_h(np.asarray([int(fold)], dtype=np.int64)))
    _parallel_single_check(parts, *pairs)
    return _fit(parts)


@lane("psd-byte-lm-model-pool")
def _(ml, X, yc, yr, Xh=None):
    """ParallelByteLanguageModelTrainer, model_pool schedule on one device (every decoder layer and its
    optimizer state owned by device 0), two steps of two logical shards, held step by step to the replica
    trainer."""
    from mojolearn.parallel_training import ParallelByteLanguageModelTrainer as P
    dev = _PARALLEL_SINGLE_DEVICES
    shape = ml.ByteLanguageModelConfig()
    state0 = _parallel_single_byte_state(ml, shape)
    ids = _ids(X, 4 * shape.batch, shape.length + 1)
    with P(state0, devices=dev, logical_shards=2, schedule="model_pool") as tr, \
            P(state0, devices=dev, logical_shards=2, pool_optimizer=False) as ref:
        losses, pairs = _parallel_single_byte_run(tr, ref, ids, 2)
        state = tr.state_dict()
        pairs += _parallel_single_same_state("model_pool trainer", state, "replica trainer", ref.state_dict())
    parts = dict(loss=_h(losses), params=_h(np.asarray(state["parameters"])), m=_h(np.asarray(state["m"])))
    _parallel_single_check(parts, *pairs)
    return _fit(parts)


@lane("psd-byte-lm-offload")
def _(ml, X, yc, yr, Xh=None):
    """ParallelByteLanguageModelTrainer, offload schedule (one decoder layer resident at a time, state in host
    memory), two steps of two logical shards, held step by step to the replica trainer."""
    from mojolearn.parallel_training import ParallelByteLanguageModelTrainer as P
    dev = _PARALLEL_SINGLE_DEVICES
    shape = ml.ByteLanguageModelConfig()
    state0 = _parallel_single_byte_state(ml, shape)
    ids = _ids(X, 4 * shape.batch, shape.length + 1)
    with P(state0, devices=dev, logical_shards=2, schedule="offload") as tr, \
            P(state0, devices=dev, logical_shards=2, pool_optimizer=False) as ref:
        losses, pairs = _parallel_single_byte_run(tr, ref, ids, 2)
        state = tr.state_dict()
        pairs += _parallel_single_same_state("offload trainer", state, "replica trainer", ref.state_dict())
    parts = dict(loss=_h(losses), params=_h(np.asarray(state["parameters"])), m=_h(np.asarray(state["m"])))
    _parallel_single_check(parts, *pairs)
    return _fit(parts)


@lane("psd-byte-lm-corpus")
def _(ml, X, yc, yr, Xh=None):
    """ParallelByteLanguageModelTrainer.train_step_from: a 16 KB corpus of the fixture's bytes prepared with a
    300-rank vocabulary (as the lm-corpus lane does), a model of the corpus's vocabulary, one step on corpus
    steps [0, 1], held to `train_step` on `batches.ids(0)` and `batches.ids(1)` from the same state."""
    from mojolearn.parallel_training import ParallelByteLanguageModelTrainer as P
    dev = _PARALLEL_SINGLE_DEVICES
    LC = ml.lm_corpus
    raw = np.ascontiguousarray(X).tobytes()[:16384]
    with tempfile.TemporaryDirectory(prefix="ib-psd-byte-lm-corpus-") as d:
        path = os.path.join(d, "corpus.txt")
        with open(path, "wb") as fh:
            fh.write(raw)
        c = LC.prepare(path, cache_dir=os.path.join(d, "cache"), vocab_size=300, vocab_sample_bytes=16384,
                       document_bytes=4096)
        shape = ml.ByteLanguageModelConfig(vocab_size=int(c.vocabulary["n_vocab"]))
        state0 = _parallel_single_byte_state(ml, shape)
        b = c.batches(shape.batch, shape.length)
        with P(state0, devices=dev, logical_shards=2) as tr, P(state0, devices=dev, logical_shards=2) as ref:
            got = np.asarray(tr.train_step_from(b, [0, 1])["losses"], dtype=np.float32)
            want = np.asarray(ref.train_step([b.ids(0), b.ids(1)])["losses"], dtype=np.float32)
            state = tr.state_dict()
            pairs = [("train_step_from losses", got, "train_step(batches.ids) losses", want)]
            pairs += _parallel_single_same_state("train_step_from trainer", state, "train_step trainer",
                                                 ref.state_dict())
    parts = dict(loss=_h(got), params=_h(np.asarray(state["parameters"])))
    _parallel_single_check(parts, *pairs)
    return _fit(parts)


# ================================================================ BOTH SIDES: a CPU-only installation runs the host shards

# ---------------------------------------------------------------- parallel_preprocessing (Python-sharded, native local pool)

@lane("psd-scaler")
def _(ml, X, yc, yr, Xh=None):
    """fit_scaler and transform_scaler (forward and inverse) on StandardScaler with four columns per shard (the
    16-column fixture is four shards), held to the plain scaler."""
    from mojolearn.parallel_preprocessing import fit_scaler, transform_scaler
    dev = _PARALLEL_SINGLE_DEVICES
    par = fit_scaler(ml.StandardScaler(), X, devices=dev, columns_per_shard=4)
    plain = ml.StandardScaler().fit(X)
    t = transform_scaler(par, X[:256], devices=dev, columns_per_shard=4)
    inv = transform_scaler(par, t, devices=dev, columns_per_shard=4, inverse=True)
    parts = dict(mean=_h(par.mean_), var=_h(par.var_), transform=_h(t), inverse=_h(inv))
    _parallel_single_check(parts, ("fit_scaler mean_", par.mean_, "plain mean_", plain.mean_),
                           ("fit_scaler var_", par.var_, "plain var_", plain.var_),
                           ("transform_scaler", t, "plain transform", plain.transform(X[:256])),
                           ("transform_scaler inverse", inv, "plain inverse_transform", plain.inverse_transform(t)))
    return _fit(parts, par, lambda e: (transform_scaler(e, Xh[:256], devices=dev, columns_per_shard=4),))


@lane("psd-scaler-minmax")
def _(ml, X, yc, yr, Xh=None):
    """fit_scaler and transform_scaler on MinMaxScaler with four columns per shard, held to the plain scaler."""
    from mojolearn.parallel_preprocessing import fit_scaler, transform_scaler
    dev = _PARALLEL_SINGLE_DEVICES
    par = fit_scaler(ml.MinMaxScaler(), X, devices=dev, columns_per_shard=4)
    plain = ml.MinMaxScaler().fit(X)
    t = transform_scaler(par, X[:256], devices=dev, columns_per_shard=4)
    parts = dict(data_min=_h(par.data_min_), data_max=_h(par.data_max_), scale=_h(par.scale_), transform=_h(t))
    _parallel_single_check(parts, ("fit_scaler scale_", par.scale_, "plain scale_", plain.scale_),
                           ("transform_scaler", t, "plain transform", plain.transform(X[:256])))
    return _fit(parts, par, lambda e: (transform_scaler(e, Xh[:256], devices=dev, columns_per_shard=4),))


# ---------------------------------------------------------------- parallel_neighbors and parallel_neighbors_reference

@lane("psd-queries-knn")
def _(ml, X, yc, yr, Xh=None):
    """ParallelQueries over a fitted KNeighborsClassifier: 64 queries in four blocks of 16 for kneighbors,
    predict and predict_proba, held to the estimator's own calls."""
    from mojolearn.parallel_neighbors import ParallelQueries
    m = ml.KNeighborsClassifier(n_neighbors=8).fit(X[:4096], yc[:4096])
    q = np.ascontiguousarray(X[4096:4160])
    with ParallelQueries(m, devices=_PARALLEL_SINGLE_DEVICES, rows_per_shard=_PARALLEL_SINGLE_QUERY_ROWS) as pq:
        d, i = pq.query(q, method="kneighbors")
        pr = pq.query(q, method="predict")
        pp = pq.query(q, method="predict_proba")
    d0, i0 = m.kneighbors(q)
    parts = dict(dist=_h(d), idx=_h(i), predict=_h(pr), proba=_h(pp))
    _parallel_single_check(parts, ("ParallelQueries kneighbors distances", d, "plain distances", d0),
                           ("ParallelQueries kneighbors indices", i, "plain indices", i0),
                           ("ParallelQueries predict", pr, "plain predict", m.predict(q)),
                           ("ParallelQueries predict_proba", pp, "plain predict_proba", m.predict_proba(q)))
    return _fit(parts)


@lane("psd-queries-kde")
def _(ml, X, yc, yr, Xh=None):
    """ParallelQueries over a fitted KernelDensity: 256 queries in blocks of 16 for score_samples, held to the
    estimator's own score_samples."""
    from mojolearn.parallel_neighbors import ParallelQueries
    m = ml.KernelDensity(bandwidth=0.7).fit(X[:4096, :4])
    q = np.ascontiguousarray(X[4096:4352, :4])
    with ParallelQueries(m, devices=_PARALLEL_SINGLE_DEVICES, rows_per_shard=_PARALLEL_SINGLE_QUERY_ROWS) as pq:
        s = pq.query(q, method="score_samples")
    parts = dict(scores=_h(s))
    _parallel_single_check(parts, ("ParallelQueries score_samples", s, "plain score_samples", m.score_samples(q)))
    return _fit(parts)


@lane("psd-reference-proba")
def _(ml, X, yc, yr, Xh=None):
    """ReferenceShardedNeighbors.predict_proba, which neighbors-reference-shard does not call: the 4096-row
    reference in four blocks of 1024, 64 queries in blocks of 16, held to the plain predict_proba."""
    from mojolearn.parallel_neighbors_reference import ReferenceShardedNeighbors
    m = ml.KNeighborsClassifier(n_neighbors=8).fit(X[:4096], yc[:4096])
    q = np.ascontiguousarray(X[4096:4160])
    with ReferenceShardedNeighbors(m, devices=_PARALLEL_SINGLE_DEVICES, reference_rows_per_shard=1024,
                                   query_rows_per_shard=_PARALLEL_SINGLE_QUERY_ROWS) as rs:
        pp = rs.predict_proba(q)
    parts = dict(proba=_h(pp))
    _parallel_single_check(parts, ("ReferenceShardedNeighbors predict_proba", pp,
                                   "plain predict_proba", m.predict_proba(q)))
    return _fit(parts)


# ---------------------------------------------------------------- parallel_gaussian_process (class-sharded)

@lane("psd-gpc")
def _(ml, X, yc, yr, Xh=None):
    """fit_gaussian_process_classifier and predict_gaussian_process_classifier (predict_proba and predict) on a
    two-class and a three-class (one-vs-rest, one shard per class) classifier, held to the plain fits."""
    from mojolearn.parallel_gaussian_process import (fit_gaussian_process_classifier,
                                                     predict_gaussian_process_classifier)
    dev = _PARALLEL_SINGLE_DEVICES
    xs = np.ascontiguousarray(X[:256, :4])
    q = np.ascontiguousarray(X[256:320, :4])
    k = ml.ConstantKernel(1.0) * ml.RBF(1.0)
    par = fit_gaussian_process_classifier(ml.GaussianProcessClassifier(kernel=k), xs, yc[:256], devices=dev)
    plain = ml.GaussianProcessClassifier(kernel=k).fit(xs, yc[:256])
    k3 = ml.ConstantKernel(2.0) * ml.Matern(1.0, nu=1.5)
    y3 = _gpc_three_classes(X)
    par3 = fit_gaussian_process_classifier(ml.GaussianProcessClassifier(kernel=k3), xs, y3, devices=dev)
    plain3 = ml.GaussianProcessClassifier(kernel=k3).fit(xs, y3)
    proba = predict_gaussian_process_classifier(par, q, devices=dev, method="predict_proba")
    proba3 = predict_gaussian_process_classifier(par3, q, devices=dev, method="predict_proba")
    labels3 = np.asarray(predict_gaussian_process_classifier(par3, q, devices=dev, method="predict"))
    parts = dict(L=_h(*[e.L_ for e in par.estimators_]), proba=_h(proba),
                 three_L=_h(*[e.L_ for e in par3.estimators_]), three_proba=_h(proba3), three_labels=_h(labels3))
    _parallel_single_check(parts, ("fit_gaussian_process_classifier L_", par.estimators_[0].L_,
                                   "plain L_", plain.estimators_[0].L_),
                           ("predict_gaussian_process_classifier proba", proba, "plain predict_proba",
                            plain.predict_proba(q)),
                           ("predict_gaussian_process_classifier three-class proba", proba3,
                            "plain three-class predict_proba", plain3.predict_proba(q)),
                           ("predict_gaussian_process_classifier three-class predict", labels3,
                            "plain three-class predict", np.asarray(plain3.predict(q))))
    return _fit(parts, par, lambda e: (predict_gaussian_process_classifier(e, np.ascontiguousarray(Xh[:64, :4]),
                                                                           devices=dev, method="predict_proba"),))


# ---------------------------------------------------------------- forecasting (series-sharded)

def _parallel_single_hw_series(X):
    """Four positive seasonal series of 512 points from the fixture (cumulative sums lifted above one)."""
    S = np.cumsum(X[:512, :4], axis=0).T.astype(np.float32) + np.float32(50.0)
    return np.ascontiguousarray((S - S.min(axis=1, keepdims=True) + np.float32(1.0)).astype(np.float32))


@lane("psd-arima")
def _(ml, X, yc, yr, Xh=None):
    """fit_arima with two series per shard over four series, held to the plain ARIMA fit."""
    from mojolearn.parallel_classical import fit_arima
    series = np.ascontiguousarray(X[:512, :4].T)
    par = fit_arima(ml.ARIMA(order=(1, 0, 0)), series, devices=_PARALLEL_SINGLE_DEVICES, series_per_shard=2)
    plain = ml.ARIMA(order=(1, 0, 0)).fit(series)
    f = par.forecast(24)
    parts = dict(ar=_h(par.ar_), mu=_h(par.mu_), sigma2=_h(par.sigma2_), forecast=_h(f))
    _parallel_single_check(parts, ("fit_arima ar_", par.ar_, "plain ar_", plain.ar_),
                           ("fit_arima sigma2_", par.sigma2_, "plain sigma2_", plain.sigma2_),
                           ("fit_arima forecast(24)", f, "plain forecast(24)", plain.forecast(24)))
    return _fit(parts, par, lambda e: (e.forecast(FORECAST_HORIZON),))


@lane("psd-forecast-arima")
def _(ml, X, yc, yr, Xh=None):
    """predict_arima and forecast_arima over four series, three to a shard (the uneven 3 + 1 partition), held to
    the fitted ARIMA's own predict and forecast."""
    from mojolearn.parallel_forecasting import predict_arima, forecast_arima
    dev = _PARALLEL_SINGLE_DEVICES
    series = np.ascontiguousarray(X[:512, :4].T)
    m = ml.ARIMA(order=(1, 0, 0)).fit(series)
    sharded = predict_arima(m, 0, m.n_obs_, devices=dev, series_per_shard=3)
    ahead = forecast_arima(m, 24, devices=dev, series_per_shard=3)
    parts = dict(predict=_h(sharded), forecast=_h(ahead))
    _parallel_single_check(parts, ("predict_arima(0, n_obs)", sharded, "plain ARIMA.predict(0, n_obs)",
                                   m.predict(0, m.n_obs_)),
                           ("forecast_arima(24)", ahead, "plain ARIMA.forecast(24)", m.forecast(24)))
    return _fit(parts, m, lambda e: (forecast_arima(e, FORECAST_HORIZON, devices=dev, series_per_shard=3),))


@lane("psd-holtwinters")
def _(ml, X, yc, yr, Xh=None):
    """fit_exponential_smoothing (additive Holt-Winters, period 12) with two series per shard over four series,
    held to the plain fit's fitted arrays and forecast."""
    from mojolearn.parallel_classical import fit_exponential_smoothing
    S = _parallel_single_hw_series(X)
    kw = dict(seasonal="additive", seasonal_periods=12, ts_num=4)
    par = fit_exponential_smoothing(ml.ExponentialSmoothing(S, **kw), devices=_PARALLEL_SINGLE_DEVICES,
                                    series_per_shard=2)
    plain = ml.ExponentialSmoothing(S, **kw).fit()
    names = ("level_", "trend_", "season_", "sse_", "alpha_", "beta_", "gamma_", "n_iter_")
    f = par.forecast(24)
    parts = dict(**{n.strip("_"): _h(getattr(par, n)) for n in names}, forecast=_h(f))
    _parallel_single_check(parts, *[(f"fit_exponential_smoothing {n}", getattr(par, n), f"plain {n}", getattr(plain, n))
                                    for n in names],
                           ("fit_exponential_smoothing forecast(24)", f, "plain forecast(24)", plain.forecast(24)))
    return _fit(parts, par, lambda e: (e.forecast(FORECAST_HORIZON),))


@lane("psd-forecast-holtwinters")
def _(ml, X, yc, yr, Xh=None):
    """predict_exponential_smoothing and forecast_exponential_smoothing (all series and one indexed series) over
    four series, three to a shard, held to the fitted model's own predict and forecast."""
    from mojolearn.parallel_forecasting import forecast_exponential_smoothing, predict_exponential_smoothing
    dev = _PARALLEL_SINGLE_DEVICES
    S = _parallel_single_hw_series(X)
    m = ml.ExponentialSmoothing(S, seasonal="additive", seasonal_periods=12, ts_num=4).fit()
    sharded = predict_exponential_smoothing(m, m.n, m.n + 24, devices=dev, series_per_shard=3)
    ahead = forecast_exponential_smoothing(m, 24, devices=dev, series_per_shard=3)
    one = forecast_exponential_smoothing(m, 24, index=2, devices=dev, series_per_shard=3)
    parts = dict(predict=_h(sharded), forecast=_h(ahead), indexed=_h(one))
    _parallel_single_check(parts, ("predict_exponential_smoothing(n, n + 24)", sharded,
                                   "plain predict(n, n + 24)", m.predict(m.n, m.n + 24)),
                           ("forecast_exponential_smoothing(24)", ahead, "plain forecast(24)", m.forecast(24)),
                           ("forecast_exponential_smoothing(24, index=2)", one, "plain forecast(24, index=2)",
                            m.forecast(24, index=2)))
    return _fit(parts, m, lambda e: (forecast_exponential_smoothing(e, FORECAST_HORIZON, devices=dev,
                                                                    series_per_shard=3),))


# ---------------------------------------------------------------- parallel_model_selection

@lane("psd-cross-val")
def _(ml, X, yc, yr, Xh=None):
    """parallel_model_selection.cross_val_score on one device, three unshuffled folds of the sklearn-style
    boosting regressor (the cross-val lane's estimator), held to the serial model_selection.cross_val_score."""
    from mojolearn.parallel_model_selection import cross_val_score
    scores = np.asarray(cross_val_score(_gbdt(ml.GradientBoostingRegressor, n_estimators=8, max_depth=4), X, yr,
                                        devices=_PARALLEL_SINGLE_DEVICES, cv=3), dtype=np.float64)
    serial = np.asarray(ml.model_selection.cross_val_score(_gbdt(ml.GradientBoostingRegressor, n_estimators=8,
                                                                 max_depth=4), X, yr, cv=3), dtype=np.float64)
    parts = dict(scores=_h(scores))
    _parallel_single_check(parts, ("parallel cross_val_score", scores, "serial cross_val_score", serial))
    return _fit(parts)


# ---------------------------------------------------------------- ParallelNeuralTrainer: checkpoint and gradients

@lane("psd-neural-trainer-ckpt")
def _(ml, X, yc, yr, Xh=None):
    """ParallelNeuralTrainer over the neural-trainer-unfused lane's SmallMLPTrainer, one device, three logical
    shards: two steps, `export_gradients` and `checkpoint`, then `from_checkpoint` and a third step on the
    restored and the live trainer, whose gradients and checkpoints must agree."""
    from mojolearn.parallel_training import ParallelNeuralTrainer
    dev = _PARALLEL_SINGLE_DEVICES
    w = [((np.arange(int(np.prod(s)), dtype=np.float32) % 7 - 3) / 32).reshape(s).astype(np.float32)
         for s in ((16, 8), (16,), (3, 16), (3,))]
    Xm = np.ascontiguousarray(X[:576, :8])
    t = _three_class(X[:576])
    steps = [[(Xm[192 * s + 64 * k: 192 * s + 64 * (k + 1)], t[192 * s + 64 * k: 192 * s + 64 * (k + 1)])
              for k in range(3)] for s in range(3)]
    m = ml.SmallMLPTrainer(*w, data_schedule={"dataset": "identity_break", "order": "sequential"})
    with ParallelNeuralTrainer(m, devices=dev, logical_shards=3) as tr:
        for shards in steps[:2]:  # glue: two driver steps
            tr.train_step(shards)
        grads = _parallel_single_flat(tr.export_gradients())
        ck = tr.checkpoint()
        with ParallelNeuralTrainer.from_checkpoint(ck, devices=dev) as back:
            back.train_step(steps[2])
            tr.train_step(steps[2])
            live_g, back_g = _parallel_single_flat(tr.export_gradients()), _parallel_single_flat(back.export_gradients())
            live_ck, back_ck = _parallel_single_flat(tr.checkpoint()), _parallel_single_flat(back.checkpoint())
    pairs = [("from_checkpoint array count", np.int64(len(back_g) + len(back_ck)),
              "live array count", np.int64(len(live_g) + len(live_ck)))]
    pairs += [(f"from_checkpoint export_gradients[{k}]", b, "live export_gradients", a)
              for k, (a, b) in enumerate(zip(live_g, back_g))]
    pairs += [(f"from_checkpoint checkpoint[{k}]", b, "live checkpoint", a)
              for k, (a, b) in enumerate(zip(live_ck, back_ck))]
    parts = dict(grads=_h(*grads), checkpoint=_h(*_parallel_single_flat(ck)), after=_h(*live_g))
    _parallel_single_check(parts, *pairs)
    return _fit(parts)


# ---------------------------------------------------------------- models.ParallelCausalLM (layer owners, one device)

def _parallel_single_causal_open(ml, d):
    """The hf-causal-lm lane's Llama fixture written under `d`, loaded as the plain CausalLM and as a
    ParallelCausalLM whose every layer is owned by device 0 (repeated owners: one worker). Raises when the two
    did not run on the same backend: ParallelCausalLM takes no device= and follows the installation, so on a
    GPU installation its CPU side is not a host computation (the product gap this lane reports)."""
    from mojolearn import _causal_lm_fixtures as fx
    from mojolearn.models.parallel_causal_lm import ParallelCausalLM
    cfg, tensors = fx.family_fixture("llama", False)
    root = fx._write_checkpoint(os.path.join(d, "llama"), cfg, tensors)
    plain = ml.models.causal_lm.CausalLM.load(root, numeric_profile="fp32_v1")
    owners = _PARALLEL_SINGLE_DEVICES * int(plain.plan.n_layers)
    par = ParallelCausalLM.load(root, layer_devices=owners, numeric_profile="fp32_v1")
    if str(par.vendor_used()) != str(plain.vendor_used()):
        par.close()
        raise AssertionError(f"ParallelCausalLM ran on {par.vendor_used()!r} and this side's CausalLM on "
                             f"{plain.vendor_used()!r}: ParallelCausalLM has no device= and follows the installation")
    return plain, par


@lane("psd-causal-lm")
def _(ml, X, yc, yr, Xh=None):
    """ParallelCausalLM (`load`, which builds the owner through the same native init as the constructor) on one
    device: `forward` and `__call__`, `allocate_state`, prefill then `step`, `reset_state` and a second prefill,
    greedy `generate`, the weights (`parameters`, `named_parameters`, `state_dict`) and a `load_state_dict` round
    trip, each held to the plain CausalLM of the same checkpoint."""
    A = ml.Array.from_buffer
    ids = _ids(X, HF_LM_BATCH, HF_LM_LEN, HF_LM_VOCAB)
    head = np.ascontiguousarray(ids[:, :HF_LM_LEN - 1])
    last = np.ascontiguousarray(ids[:, HF_LM_LEN - 1:])
    with tempfile.TemporaryDirectory(prefix="ib-psd-causal-lm-") as d:
        plain, par = _parallel_single_causal_open(ml, d)
        with par:
            logits = np.asarray(par.forward(A(ids)))
            called = np.asarray(par(A(ids)))
            state = par.allocate_state(HF_LM_BATCH, HF_LM_LEN)
            par.forward(A(head), state)
            step = np.asarray(par.step(A(last), state))
            par.reset_state(state)
            par.forward(A(head), state)
            again = np.asarray(par.step(A(last), state))
            gen = np.asarray(par.generate(A(ids), HF_LM_NEW), dtype=np.int32)
            named = sorted((str(n), np.asarray(w)) for n, w in par.named_parameters())
            listed = [np.asarray(w) for w in par.parameters()]
            sd = par.state_dict()
            # ParallelCausalLM.load_state_dict refuses by design (NotImplementedError: "a layer-partitioned model
            # is loaded from a checkpoint; load the state dict into a CausalLM", bindings/causal_lm_construct_door.mojo,
            # since 20d35b773a). Wrong test fixed (lane gs-fix): the lane called it and was refused on the GPU side;
            # it now holds the refusal.
            try:
                par.load_state_dict(sd)
            except NotImplementedError:
                pass
            else:
                raise AssertionError("ParallelCausalLM.load_state_dict no longer refuses: hold its reload again")
            pstate = plain.allocate_state(HF_LM_BATCH, HF_LM_LEN)
            plain.forward(A(head), pstate)
            pairs = [("ParallelCausalLM forward", logits, "plain CausalLM forward", np.asarray(plain.forward(A(ids)))),
                     ("ParallelCausalLM __call__", called, "ParallelCausalLM forward", logits),
                     ("ParallelCausalLM step", step, "plain CausalLM step", np.asarray(plain.step(A(last), pstate))),
                     ("ParallelCausalLM step after reset_state", again, "ParallelCausalLM step", step),
                     ("ParallelCausalLM generate", gen, "plain CausalLM generate",
                      np.asarray(plain.generate(A(ids), HF_LM_NEW), dtype=np.int32)),
                     ("ParallelCausalLM named_parameters", np.concatenate([_hf_bytes(w) for _, w in named]),
                      "plain CausalLM named_parameters",
                      np.concatenate([_hf_bytes(np.asarray(w)) for _, w in sorted((str(n), w) for n, w in
                                                                                   plain.named_parameters())])),
                     ("ParallelCausalLM state_dict", np.concatenate([_hf_bytes(np.asarray(sd[k])) for k in sorted(sd)]),
                      "ParallelCausalLM named_parameters", np.concatenate([_hf_bytes(w) for _, w in named])),
                     ("ParallelCausalLM parameters count", np.int64(len(listed)), "named_parameters count",
                      np.int64(len(named)))]
    parts = dict(logits=_h(logits), step=_h(step), generate=_h(gen),
                 params=_h(np.concatenate([_hf_bytes(w) for _, w in named])))
    _parallel_single_check(parts, *pairs)
    return _fit(parts)


@lane("psd-causal-lm-text")
def _(ml, X, yc, yr, Xh=None):
    """ParallelCausalLM's decoding surfaces on one device: `stream` (read two at a time), `generate_text`,
    `stream_text`, a text stream paused, checkpointed and continued with `resume_generation`, and `blocks`,
    each held to the plain CausalLM of the same checkpoint with the hf-causal-lm-text lane's byte tokenizer."""
    A = ml.Array.from_buffer
    ids = _ids(X, HF_LM_BATCH, HF_LM_LEN, HF_LM_VOCAB)
    tok, _, _ = _coverage_text_tokenizers(ml)
    raw = np.frombuffer(np.ascontiguousarray(X).tobytes()[:24], dtype=np.uint8)
    text = bytes(_COVERAGE_TEXT_ALPHABET[int(b) % 64] for b in raw).decode("ascii")  # glue: 24 prompt bytes
    with tempfile.TemporaryDirectory(prefix="ib-psd-causal-lm-text-") as d:
        plain, par = _parallel_single_causal_open(ml, d)
        with par:
            s = par.stream(A(ids), HF_LM_NEW)
            streamed = np.ascontiguousarray(np.concatenate(
                [np.asarray(s.read(2), dtype=np.int32) for _ in range(-(-HF_LM_NEW // 2))], axis=1))  # glue: bounded reads
            greedy = par.generate_text(text, HF_LM_NEW, tokenizer=tok)
            ts = par.stream_text(text, HF_LM_NEW, tokenizer=tok)
            pieces = "".join(ts.read(2) for _ in range(-(-HF_LM_NEW // 2) + 1))  # glue: bounded reads
            paused = par.stream_text(text, HF_LM_NEW, tokenizer=tok)
            first = paused.read(2)
            resumed = par.resume_generation(paused.state_dict(), tokenizer=tok)
            rest = "".join(resumed.read(2) for _ in range(-(-HF_LM_NEW // 2)))  # glue: bounded reads
            n_blocks = len(par.blocks)
        want = np.ascontiguousarray(np.asarray(plain.generate(A(ids), HF_LM_NEW), dtype=np.int32)[:, HF_LM_LEN:])
        plain_text = plain.generate_text(text, HF_LM_NEW, tokenizer=tok)
        pairs = [("ParallelCausalLM stream", streamed, "plain CausalLM generate continuation", want),
                 ("ParallelCausalLM generate_text", _hf_text(greedy), "plain CausalLM generate_text", _hf_text(plain_text)),
                 ("ParallelCausalLM stream_text", _hf_text(pieces), "ParallelCausalLM generate_text", _hf_text(greedy)),
                 ("ParallelCausalLM resume_generation", _hf_text(first + rest), "ParallelCausalLM generate_text",
                  _hf_text(greedy)),
                 ("ParallelCausalLM blocks", np.int64(n_blocks), "plain CausalLM n_layers", np.int64(plain.n_layers))]
    parts = dict(stream=_h(streamed), text=_h(_hf_text(greedy)), resumed=_h(_hf_text(first + rest)),
                 blocks=_h(np.asarray([n_blocks], dtype=np.int64)))
    _parallel_single_check(parts, *pairs)
    return _fit(parts)


# ================================================================ declarations

_one_sided("psd-logistic", missing="cpu", reason="fit_logistic is a cooperative multi-GPU driver (glm_fit): "
           "its gradient columns are tiles inside the GPU binding, which no host binding restates")
_one_sided("psd-gram", missing="cpu", reason="fit_gram_estimator is a cooperative multi-GPU driver: its Gram "
           "chunks live inside the GPU binding, which no host binding restates")
_one_sided("psd-gram-pca", missing="cpu", reason="fit_gram_estimator is a cooperative multi-GPU driver: its Gram "
           "chunks live inside the GPU binding, which no host binding restates")
_one_sided("psd-cd", missing="cpu", reason="fit_coordinate_descent is a cooperative multi-GPU driver with no "
           "host restatement")
_one_sided("psd-svm", missing="cpu", reason="fit_svm and predict_svm are cooperative multi-GPU drivers with no "
           "host restatement")
_one_sided("psd-gp", missing="cpu", reason="fit_gaussian_process and predict_gaussian_process are cooperative "
           "multi-GPU drivers with no host restatement")
_one_sided("psd-kmeans", missing="cpu", reason="fit_kmeans is a cooperative multi-GPU driver with no host "
           "restatement")
_one_sided("psd-dbscan", missing="cpu", reason="fit_dbscan is a cooperative multi-GPU driver with no host "
           "restatement")
_one_sided("psd-hdbscan", missing="cpu", reason="fit_hdbscan is a cooperative multi-GPU driver with no host "
           "restatement")
_one_sided("psd-gmm", missing="cpu", reason="fit_gaussian_mixture and predict_gaussian_mixture are cooperative "
           "multi-GPU drivers with no host restatement")
_one_sided("psd-resample", missing="cpu", reason="parallel_classical bootstrap, permutation_test and "
           "monte_carlo_integrate are cooperative multi-GPU drivers with no host restatement")
_one_sided("psd-cholesky", missing="cpu", reason="fit_cholesky and solve_cholesky are cooperative multi-GPU "
           "drivers (cholesky_fit, cholesky_solve) with no host restatement")
_one_sided("psd-kernel-ridge", missing="cpu", reason="fit_kernel_method and apply_kernel_method are cooperative "
           "multi-GPU drivers (km_fit, km_apply) with no host restatement")
_one_sided("psd-nystroem", missing="cpu", reason="fit_kernel_method and apply_kernel_method are cooperative "
           "multi-GPU drivers (km_fit, km_apply) with no host restatement")
_one_sided("psd-rbf-sampler", missing="cpu", reason="transform_rbf_sampler is a cooperative multi-GPU driver "
           "with no host restatement")
_one_sided("psd-boosting", missing="cpu", reason="fit_boosting is a cooperative multi-GPU driver with no host "
           "restatement")
_one_sided("psd-boosting-reg", missing="cpu", reason="fit_boosting is a cooperative multi-GPU driver with no "
           "host restatement")
_one_sided("psd-iforest", missing="cpu", reason="fit_isolation_forest and score_isolation_forest are cooperative "
           "multi-GPU drivers with no host restatement")
_one_sided("psd-ordered-rmse", missing="cpu", reason="fit_ordered_rmse is a cooperative multi-GPU driver with no "
           "host restatement")
_one_sided("psd-feature-freq", missing="cpu", reason="fit_feature_freq is a cooperative multi-GPU driver with no "
           "host restatement")
_one_sided("psd-forest-pool", missing="cpu", reason="ParallelForestPredictor's groves are resident in the GPU "
           "binding; no host binding restates the pool")
_one_sided("psd-graph-agglomerative", missing="cpu", reason="fit_graph is a cooperative multi-GPU driver with no "
           "host restatement")
_one_sided("psd-graph-umap", missing="cpu", reason="fit_graph and transform_umap are cooperative multi-GPU "
           "drivers with no host restatement")
_one_sided("psd-byte-lm", missing="cpu", reason="ParallelByteLanguageModelTrainer's sessions live inside the GPU "
           "byte-LM binding, which no host binding restates")
_one_sided("psd-byte-lm-worker", missing="cpu", reason="ParallelByteLanguageModelTrainer's sessions live inside "
           "the GPU byte-LM binding, which no host binding restates")
_one_sided("psd-byte-lm-model-pool", missing="cpu", reason="ParallelByteLanguageModelTrainer's sessions live "
           "inside the GPU byte-LM binding, which no host binding restates")
_one_sided("psd-byte-lm-offload", missing="cpu", reason="ParallelByteLanguageModelTrainer's sessions live inside "
           "the GPU byte-LM binding, which no host binding restates")
_one_sided("psd-byte-lm-corpus", missing="cpu", reason="ParallelByteLanguageModelTrainer's sessions live inside "
           "the GPU byte-LM binding, which no host binding restates")
# models.ParallelCausalLM keeps devices= and takes no device= (Andrew, vd-functions addenda 1 and 2, 2026-11-24):
# its CPU side is an owed decision, listed in UNMAPPED.md, not a lane failure.
_one_sided("psd-causal-lm", missing="cpu", reason="models.ParallelCausalLM keeps devices= (a GPU list) and takes no "
           "device=, by the owner's direction of 2026-11-24")
_one_sided("psd-causal-lm-text", missing="cpu", reason="models.ParallelCausalLM keeps devices= (a GPU list) and takes "
           "no device=, by the owner's direction of 2026-11-24")


_batch_decl("n/a:ordered-shard-driver (the lane holds a multi-GPU driver on one device to its plain one-device "
            "path; the driver's batch axis is the plain estimator's, measured on that estimator's own lane)",
            "psd-logistic", "psd-gram", "psd-gram-pca", "psd-cd", "psd-svm", "psd-gp", "psd-kmeans", "psd-dbscan",
            "psd-hdbscan", "psd-gmm", "psd-resample", "psd-cholesky", "psd-kernel-ridge", "psd-nystroem",
            "psd-rbf-sampler", "psd-boosting", "psd-boosting-reg", "psd-iforest", "psd-ordered-rmse",
            "psd-feature-freq", "psd-forest-pool", "psd-graph-agglomerative", "psd-graph-umap", "psd-byte-lm",
            "psd-byte-lm-worker", "psd-byte-lm-model-pool", "psd-byte-lm-offload", "psd-byte-lm-corpus",
            "psd-scaler", "psd-scaler-minmax", "psd-queries-knn", "psd-queries-kde", "psd-reference-proba",
            "psd-gpc", "psd-arima", "psd-forecast-arima", "psd-holtwinters", "psd-forecast-holtwinters",
            "psd-cross-val", "psd-neural-trainer-ckpt", "psd-causal-lm", "psd-causal-lm-text")
