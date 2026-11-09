# SPDX-License-Identifier: Apache-2.0
"""cross_val_score through the DeviceRows door equals the host path, bit for bit.

Lane ne1-b-door (plan docs/plans/NATIVE_ENTRY_POINTS_AFTER_0866.md workstream
B, first release). Model selection hands a fold's X to an estimator as
`_arena_io.DeviceRows` when the estimator's plan says `device_rows`; the
shared doors (`bindings/device_rows_door.mojo`, used by `classical_check_x`,
the classical admission and `family_lifecycle_begin`) resolve it. For one
estimator per family this runs `cross_val_score` twice on the same data, once
on the host-array path and once with the plan's `device_rows` switched on,
and compares the score digests. The same rows reach the same kernels in the
same order, so the bytes must be equal in `identical` mode on every vendor.

Runs on a box with a built wheel (never on the laptop):
    python -m pytest -q tools/test_device_rows_door_cv.py
"""
import hashlib

import pytest

np = pytest.importorskip("numpy")
mojolearn = pytest.importorskip("mojolearn")

from mojolearn import model_selection  # noqa: E402


def _estimators():
    # One estimator per family door, small settings (a correctness check,
    # not a measurement). (id, factory, task)
    from mojolearn import linear_model, tree, randomforest, svm
    from mojolearn import _expansion_prep as prep
    from mojolearn import _gp_impl as gp
    from mojolearn import kernel_methods as km
    from mojolearn import _gbdt_adapters as boosting
    from mojolearn import _svm_impl as svm_impl, neighbors
    return [
        ("classical-ols", lambda: linear_model.LinearRegression(), "reg"),
        ("classical-ridge", lambda: linear_model.Ridge(alpha=1.0), "reg"),
        ("logistic", lambda: linear_model.LogisticRegression(max_iter=50), "clf"),
        ("lda", lambda: prep.LinearDiscriminantAnalysis(), "clf"),
        ("gaussian-nb", lambda: prep.GaussianNB(), "clf"),
        ("tree", lambda: tree.DecisionTreeRegressor(max_depth=3, random_state=0), "reg"),
        ("forest", lambda: randomforest.RandomForestClassifier(n_estimators=3, max_depth=3, random_state=0), "clf"),
        ("boosting", lambda: boosting.GradientBoostingRegressor(n_estimators=3, max_depth=2, random_state=0), "reg"),
        ("linear-svc", lambda: svm.LinearSVC(max_iter=200), "clf"),
        ("gp", lambda: gp.GaussianProcessRegressor(), "reg"),
        ("kernel-ridge", lambda: km.KernelRidge(alpha=1.0), "reg"),
        # lane a6: the family / classical opt-ins
        ("svc", lambda: svm_impl.SVC(), "clf"),
        ("svr", lambda: svm_impl.SVR(), "reg"),
        ("linear-svr", lambda: svm.LinearSVR(max_iter=200), "reg"),
        ("knn-clf", lambda: neighbors.KNeighborsClassifier(n_neighbors=3), "clf"),
        ("knn-reg", lambda: neighbors.KNeighborsRegressor(n_neighbors=3), "reg"),
        ("qn", lambda: linear_model.QNRegressor(), "reg"),
        # lane a7c: the x_linear door's opt-ins
        ("lasso", lambda: _cd().Lasso(alpha=0.1), "reg"),
        ("elastic-net", lambda: _cd().ElasticNet(alpha=0.1), "reg"),
        ("huber", lambda: _xl().HuberRegressor(max_iter=50), "reg"),
        ("sgd-clf", lambda: _xl().SGDClassifier(max_iter=5, random_state=0), "clf"),
        ("ridge-clf", lambda: _xl().RidgeClassifier(), "clf"),
    ]


def _cd():
    from mojolearn import _solver_impl
    return _solver_impl


def _xl():
    from mojolearn import _expansion_linear
    return _expansion_linear


def _data(task):
    rng = np.random.default_rng(11)
    x = rng.normal(size=(60, 5)).astype("f4")
    if task == "clf":
        y = (x[:, 0] + 0.5 * x[:, 1] > 0).astype("i8")
    else:
        y = (2 * x[:, 0] - x[:, 1] + 0.1 * x[:, 2] ** 2).astype("f4")
    return x, y


def _plan(cls):
    registry = model_selection._NATIVE_CONTEXT["family_msel_estimators"]
    if cls not in registry:
        pytest.skip(f"{cls.__name__} has no native model-selection plan in this build")
    return registry[cls]


def _digest(scores):
    return hashlib.sha256(np.asarray(scores, dtype="f8").tobytes()).hexdigest()


@pytest.mark.parametrize("name,factory,task", _estimators(), ids=[row[0] for row in _estimators()])
def test_device_rows_door_matches_host_path(name, factory, task):
    x, y = _data(task)
    estimator = factory()
    plan = _plan(type(estimator))
    before = plan["device_rows"]
    try:
        plan["device_rows"] = False
        host = model_selection.cross_val_score(factory(), x, y, cv=5)
        plan["device_rows"] = True
        door = model_selection.cross_val_score(factory(), x, y, cv=5)
    finally:
        plan["device_rows"] = before
    assert _digest(door) == _digest(host), (name, list(host), list(door))


def test_device_rows_door_recognizes_only_the_exact_class():
    # The door's exact class check: the connected class is what model
    # selection allocates, and every owner context it calls carries it.
    from mojolearn import _arena_io, linear_model
    plan = _plan(linear_model.LinearRegression)
    context = plan["context"]
    assert context["door_rows_class"] is _arena_io.DeviceRows
    assert context["door_rows_context"] is _arena_io._ROWS_CONTEXT


@pytest.mark.parametrize("n,k", [(1, 1), (7, 2), (41, 5), (1000, 7)])
def test_split_plan_matches_per_fold_select(n, k):
    # N051: select_folds_i64 (one split plan) gives every fold the arrays k
    # select_fold_i64 calls give, word for word, empty folds included.
    from mojolearn._buffer import _native
    plan, single = _native("select_folds_i64"), _native("select_fold_i64")
    rng = np.random.default_rng(n * 31 + k)
    fold = rng.integers(0, k + 1, size=n).astype("i4")  # fold id k is never selected: an empty-test case
    sizes = np.array([(fold == f).sum() for f in range(k)], dtype="i8")
    scratch = np.zeros(n, dtype="i8")
    tests = [np.zeros(max(int(s), 1), dtype="i8") for s in sizes]
    trains = [np.zeros(max(n - int(s), 1), dtype="i8") for s in sizes]
    test_addrs = np.array([t.ctypes.data if s else scratch.ctypes.data for t, s in zip(tests, sizes)], dtype="i8")
    train_addrs = np.array([t.ctypes.data if n - s else scratch.ctypes.data for t, s in zip(trains, sizes)], dtype="i8")
    assert plan(fold.ctypes.data, n, k, sizes.ctypes.data, test_addrs.ctypes.data, train_addrs.ctypes.data) == -1
    for f in range(k):
        size = int(sizes[f])
        want_test = np.zeros(max(size, 1), dtype="i8")
        want_train = np.zeros(max(n - size, 1), dtype="i8")
        got = single(fold.ctypes.data, n, f, want_test.ctypes.data if size else scratch.ctypes.data,
                     want_train.ctypes.data if n - size else scratch.ctypes.data)
        assert got == size
        assert tests[f][:size].tobytes() == want_test[:size].tobytes()
        assert trains[f][:n - size].tobytes() == want_train[:n - size].tobytes()


def test_split_plan_reports_a_count_that_disagrees():
    from mojolearn._buffer import _native
    plan = _native("select_folds_i64")
    fold = np.array([0, 1, 0, 1, 0], dtype="i4")
    sizes = np.array([2, 2], dtype="i8")  # fold 0 has three rows
    buf = np.zeros(5, dtype="i8")
    addrs = np.array([buf.ctypes.data, buf.ctypes.data], dtype="i8")
    assert plan(fold.ctypes.data, 5, 2, sizes.ctypes.data, addrs.ctypes.data, addrs.ctypes.data) == 0


def test_family_and_classical_estimators_opt_in():
    # Lane a6: model selection hands these classes DeviceRows folds.
    from mojolearn import decomposition, kernel_methods as km, neighbors, linear_model
    from mojolearn import _svm_impl as svm_impl, svm, _gp_impl as gp
    for cls in (decomposition.PCA, svm_impl.SVC, svm_impl.SVR, svm.LinearSVC, svm.LinearSVR,
                gp.GaussianProcessRegressor, km.KernelRidge, neighbors.KNeighborsClassifier,
                neighbors.KNeighborsRegressor, linear_model.LogisticRegression, linear_model.QNRegressor):
        assert _plan(cls)["device_rows"] is True, cls.__name__


def test_nan_fold_reaches_a_nan_accepting_door():
    # Lane a6: a door that accepts NaN in X (its spec does not scan) gets a
    # NaN fold unrefused; a scanning door refuses it with admit_x's wording.
    from mojolearn import linear_model
    x, y = _data("reg")
    x = x.copy()
    x[4, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        model_selection.cross_val_score(linear_model.Ridge(alpha=1.0), x, y, cv=3, error_score="raise")


def test_x_linear_estimators_opt_in():
    # Lane a7c: Lasso, ElasticNet and the x_linear door's estimators take
    # DeviceRows folds; the *CV estimators (own splitter) do not.
    xl, cd = _xl(), _cd()
    for cls in (cd.Lasso, cd.ElasticNet, xl.SGDClassifier, xl.SGDRegressor, xl.HuberRegressor,
                xl.BayesianRidge, xl.RidgeClassifier, xl.QuantileRegressor):
        assert _plan(cls)["device_rows"] is True, cls.__name__
    for cls in (xl.RidgeCV, xl.LassoCV, xl.ElasticNetCV, xl.LogisticRegressionCV):
        assert not _plan(cls).get("device_rows"), cls.__name__


def _kmeans(n_clusters=3):
    from mojolearn import cluster
    return cluster.KMeans(n_clusters=n_clusters, n_init=1, max_iter=20, random_state=0)


@pytest.mark.parametrize("method", ["predict", "transform"])
def test_kmeans_fold_entries_match_host_path(method):
    # Lane a8d (follow-up of a7d): KMeans fit and predict / transform of a
    # resident fold run on the device fold entries (kmeans_fit_fold,
    # kmeans_predict_fold, kmeans_transform_fold); the host-array path must
    # give the same bytes.
    x, _ = _data("reg")
    plan = _plan(type(_kmeans()))
    before = plan["device_rows"]
    try:
        plan["device_rows"] = False
        host = model_selection.cross_val_predict(_kmeans(), x, cv=5, method=method)
        plan["device_rows"] = True
        door = model_selection.cross_val_predict(_kmeans(), x, cv=5, method=method)
    finally:
        plan["device_rows"] = before
    host = np.asarray(host)
    door = np.asarray(door)
    assert host.shape == door.shape
    assert host.tobytes() == door.tobytes(), method


def test_kmeans_nan_fold_is_refused():
    # The fold entry's finite word refuses a NaN fold with the admission's
    # wording.
    x, _ = _data("reg")
    x = x.copy()
    x[7, 2] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        model_selection.cross_val_predict(_kmeans(), x, cv=3)


def test_decomp_cluster_manifold_estimators_registered_and_opt_in():
    # Lanes a7d / a8d: connected to native model selection (no longer
    # pending migration) and handed DeviceRows folds.
    from mojolearn import cluster, mixture, decomposition
    from mojolearn import _expansion_cluster as xcluster, _expansion_decomp as xdecomp
    from mojolearn import _spectral_impl as spectral
    for cls in (cluster.KMeans, xcluster.MiniBatchKMeans, xcluster.BisectingKMeans, mixture.GaussianMixture,
                xcluster.BayesianGaussianMixture, decomposition.TruncatedSVD, xdecomp.IncrementalPCA,
                xdecomp.NMF, xdecomp.SparsePCA, xcluster.OPTICS, xdecomp.Isomap,
                xdecomp.LocallyLinearEmbedding, xdecomp.MDS, spectral.SpectralEmbedding):
        assert _plan(cls)["device_rows"] is True, cls.__name__
