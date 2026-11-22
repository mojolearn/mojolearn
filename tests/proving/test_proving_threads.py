# SPDX-License-Identifier: Apache-2.0
"""Two Python threads on the member routes (lane fx-gil, rounds 1 to 4).

Each family is fitted and predicted from two threads started together; every thread must get the result of the
same call made alone (same parameters, same seed). Then two threads read ONE fitted model (predict, save), and two
threads run model selection and bagging over a member at once. A crash, a deadlock, a refusal ("unknown or
released") or a different result is a failure.
"""
import numpy as np
import pytest

from _proving import DEVICES, assert_same, binary, cls, data, in_threads, make, member

X, Y, C = data()
X_NEW = data(rows=40, seed=7)[0]

#: (class, parameters, target): the families fx-gil names: gbdt, forest, tree, ET, DART, linear, cd, sgd, xlinear,
#: logistic, qn_regressor, linear_svm.
FAMILIES = [
    ("GradientBoostingRegressor", dict(n_estimators=8, max_depth=3, random_state=0), "r"),
    ("GradientBoostingClassifier", dict(n_estimators=8, max_depth=3, random_state=0), "c"),
    ("RandomForestRegressor", dict(n_estimators=6, random_state=0), "r"),
    ("RandomForestClassifier", dict(n_estimators=6, random_state=0), "c"),
    ("DecisionTreeRegressor", dict(random_state=0), "r"),
    ("ExtraTreesRegressor", dict(n_estimators=6, random_state=0), "r"),
    ("ExtraTreesClassifier", dict(n_estimators=6, random_state=0), "c"),
    ("DARTRegressor", dict(n_estimators=6, random_state=0), "r"),
    ("DARTClassifier", dict(n_estimators=6, random_state=0), "c"),
    ("LinearRegression", {}, "r"),
    ("Ridge", dict(alpha=1.0), "r"),
    ("Lasso", dict(alpha=0.01), "r"),
    ("ElasticNet", dict(alpha=0.01), "r"),
    ("SGDRegressor", dict(random_state=0, max_iter=20), "r"),
    ("SGDClassifier", dict(random_state=0, max_iter=20), "c"),
    ("HuberRegressor", {}, "r"),
    ("PoissonRegressor", {}, "r"),
    ("LogisticRegression", {}, "c"),
    ("QNRegressor", {}, "r"),
    ("LinearSVC", {}, "c"),
    ("LinearSVR", {}, "r"),
]
IDS = [f[0] for f in FAMILIES]


def _target(target):
    return Y if target == "r" else C


def _fit_predict(name, params, target, device):
    model = make(name, device, **params).fit(X, _target(target))
    return np.asarray(model.predict(X_NEW))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name,params,target", FAMILIES, ids=IDS)
def test_two_threads_fit_and_predict_like_one(device, name, params, target):
    alone = _fit_predict(name, params, target, device)
    for _ in range(3):
        both = in_threads(lambda i: _fit_predict(name, params, target, device))
        for i, result in enumerate(both):
            assert_same(result, alone, f"{name} thread {i}")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name,params,target", FAMILIES, ids=IDS)
def test_two_threads_predict_one_fitted_model(device, name, params, target):
    model = make(name, device, **params).fit(X, _target(target))
    alone = np.asarray(model.predict(X_NEW))
    for _ in range(3):
        for i, result in enumerate(in_threads(lambda i: np.asarray(model.predict(X_NEW)), threads=4)):
            assert_same(result, alone, f"{name} reader {i}")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", ["RandomForestRegressor", "ExtraTreesRegressor"])
def test_two_threads_save_one_fitted_forest(device, name, tmp_path):
    # fx-gil round 4: two threads saving / exporting ONE fitted forest at once both succeed (before: the second
    # raised "unknown or released"), and both files load to the same predictions.
    model = make(name, device, n_estimators=6, random_state=0).fit(X, Y)
    expected = np.asarray(model.predict(X_NEW))
    paths = [str(tmp_path / f"{name}-{i}.npz") for i in range(2)]
    in_threads(lambda i: model.save(paths[i]))
    for path in paths:
        assert_same(type(model).load(path).predict(X_NEW), expected, path)


#: Members of model selection and bagging from two threads: worker-mode members (gbdt, forest, ET, DART, the
#: linear kinds on a CPU route) and door-mode members (SVC, k-NN, PLS, MLP).
MEMBERS = [
    ("Ridge", dict(alpha=1.0), "r"),
    ("LinearRegression", {}, "r"),
    ("Lasso", dict(alpha=0.01), "r"),
    ("ElasticNet", dict(alpha=0.01), "r"),
    ("SGDRegressor", dict(random_state=0, max_iter=20), "r"),
    ("HuberRegressor", {}, "r"),
    ("LogisticRegression", {}, "c"),
    ("QNRegressor", {}, "r"),
    ("LinearSVC", {}, "c"),
    ("LinearSVR", {}, "r"),
    ("RandomForestRegressor", dict(n_estimators=6, random_state=0), "r"),
    ("ExtraTreesClassifier", dict(n_estimators=6, random_state=0), "c"),
    ("GradientBoostingRegressor", dict(n_estimators=8, max_depth=3, random_state=0), "r"),
    ("DARTRegressor", dict(n_estimators=6, random_state=0), "r"),
    ("SVC", {}, "c"),
    ("KNeighborsClassifier", {}, "c"),
    ("PLSRegression", dict(n_components=2), "r"),
    ("MLPRegressor", dict(max_iter=5, random_state=0), "r"),
]
MEMBER_IDS = [m[0] for m in MEMBERS]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name,params,target", MEMBERS, ids=MEMBER_IDS)
def test_two_threads_cross_val_score_like_one(device, name, params, target):
    def score(_):
        return np.asarray(cls("cross_val_score")(make(name, device, **params), X, _target(target), cv=3))

    alone = score(0)
    for i, result in enumerate(in_threads(score)):
        assert_same(result, alone, f"cross_val_score {name} thread {i}")
    # n_jobs=-1 (all cores, worker mode on a CPU route) gives the n_jobs=1 scores.
    many = cls("cross_val_score")(make(name, device, **params), X, _target(target), cv=3, n_jobs=-1)
    assert_same(many, alone, f"cross_val_score {name} n_jobs=-1")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name,params", [
    ("RandomForestRegressor", dict(n_estimators=4, random_state=0)),
    ("ExtraTreesRegressor", dict(n_estimators=4, random_state=0)),
    ("DecisionTreeRegressor", dict(random_state=0)),
])
def test_two_threads_bagging_over_a_member_like_one(device, name, params):
    # Bagging members are tree estimators (_proving.NOT_BAGGING_MEMBERS: the others are refused by name); the device
    # is the ensemble's (_proving.member).
    def bag(_):
        model = make("BaggingRegressor", device, estimator=member(name, **params), n_estimators=4, random_state=0)
        return np.asarray(model.fit(X, Y).predict(X_NEW))

    alone = bag(0)
    for i, result in enumerate(in_threads(bag)):
        assert_same(result, alone, f"bagging {name} thread {i}")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", ["RandomForestClassifier", "ExtraTreesClassifier"])
def test_two_threads_bagging_classifier_over_a_forest(device, name):
    def bag(_):
        model = make("BaggingClassifier", device, estimator=member(name, n_estimators=4, random_state=0),
                     n_estimators=3, random_state=0)
        return np.asarray(model.fit(X, binary(C)).predict_proba(X_NEW))

    alone = bag(0)
    for i, result in enumerate(in_threads(bag)):
        assert_same(result, alone, f"bagging {name} thread {i}")


@pytest.mark.gpu
def test_two_threads_fit_linear_regression_on_the_gpu():
    # fx-gil round 4: the kept TSQR state of a GPU LinearRegression fit (before: a double free was possible).
    alone = _fit_predict("LinearRegression", {}, "r", "gpu")
    for _ in range(5):
        for i, result in enumerate(in_threads(lambda i: _fit_predict("LinearRegression", {}, "r", "gpu"), 4)):
            assert_same(result, alone, f"LinearRegression thread {i}")
