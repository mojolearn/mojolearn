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
    ]


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
