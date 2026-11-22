# SPDX-License-Identifier: Apache-2.0
"""Use-after-free and lifetime at the boundary (lanes fx-borrow, ma-core, sweep-b).

A fitted state kept by model selection or an ensemble must own what it reads: after the caller's X (and the fold
buffers made from it) are gone and the memory is reused, the kept estimators predict exactly as before, and as an
estimator fitted by hand on the same rows.
"""
import gc

import numpy as np
import pytest

from _proving import DEVICES, assert_same, binary, churn, cls, data, make

X, Y, C = data()
X_NEW = data(rows=40, seed=7)[0]


def _folds(n_splits=3):
    return list(cls("KFold")(n_splits=n_splits).split(X))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name,target,read", [
    ("KernelDensity", None, "score_samples"),
    ("Ridge", "r", "predict"),
    ("LogisticRegression", "c", "predict"),
])
def test_cross_validate_kept_estimators_outlive_the_fold_buffers(device, name, target, read):
    # fx-borrow: cross_validate(..., return_estimator=True); ma-core round 3: each returned estimator predicts as
    # one fitted by hand on that fold's train rows, and classes_ has the hand fit's type.
    target_values = {"r": Y, "c": C, None: None}[target]
    temporary = X.copy()
    out = cls("cross_validate")(make(name, device), temporary, None if target is None else target_values.copy(),
                                cv=cls("KFold")(n_splits=3), return_estimator=True)
    del temporary
    churn()
    kept = out["estimator"]
    assert len(kept) == 3
    for fold, (train, _) in enumerate(_folds()):
        hand = make(name, device)
        hand.fit(X[train]) if target is None else hand.fit(X[train], target_values[train])
        assert_same(getattr(kept[fold], read)(X_NEW), getattr(hand, read)(X_NEW), f"{name} fold {fold}")
        if target == "c":
            assert type(kept[fold].classes_) is type(hand.classes_)
            assert_same(kept[fold].classes_, hand.classes_, "classes_")


@pytest.mark.parametrize("device", DEVICES)
def test_bagging_over_knn_predicts_the_same_after_the_input_is_freed(device):
    # fx-borrow: BaggingClassifier(KNeighborsClassifier()) holds member states that reference no freed fold.
    def fit():
        source = X.copy()
        model = cls("BaggingClassifier")(make("KNeighborsClassifier", device), n_estimators=4, random_state=0)
        model.fit(source, C.copy())
        return model, source

    model, source = fit()
    first = model.predict(X_NEW)
    del source
    churn()
    assert_same(model.predict(X_NEW), first, "after the free")
    assert_same(fit()[0].predict(X_NEW), first, "a second fit")


@pytest.mark.parametrize("device", DEVICES)
def test_gaussian_process_search_refit_predicts_after_del_x(device):
    # fx-borrow: a GP search refit, then predict after the training X is deleted.
    source = X.copy()
    search = cls("GridSearchCV")(make("GaussianProcessRegressor", device), {"alpha": [1e-2, 1e-1]}, cv=3)
    search.fit(source, Y.copy())
    first = search.predict(X_NEW)
    del source
    churn()
    assert_same(search.predict(X_NEW), first, "search predict after del X")
    hand = make("GaussianProcessRegressor", device, alpha=search.best_params_["alpha"]).fit(X, Y)
    assert_same(hand.predict(X_NEW), first, "hand refit")


@pytest.mark.parametrize("device", DEVICES)
def test_isolation_forest_held_predict_after_the_input_is_freed(device):
    # fx-borrow: IsolationForest's held predict (IF_OP_FOREST_RUN after adopt) equals its door, after the free.
    source = X.copy()
    model = make("IsolationForest", device, n_estimators=20, random_state=0).fit(source)
    first = (model.predict(X_NEW), model.decision_function(X_NEW), model.score_samples(X_NEW))
    del source
    churn()
    again = (model.predict(X_NEW), model.decision_function(X_NEW), model.score_samples(X_NEW))
    for a, b, what in zip(first, again, ("predict", "decision_function", "score_samples")):
        assert_same(b, a, what)
    twin = make("IsolationForest", device, n_estimators=20, random_state=0).fit(X)
    assert_same(twin.predict(X_NEW), first[0], "a second fit")


@pytest.mark.parametrize("device", DEVICES)
def test_search_refit_of_a_forest_classifier_has_classes(device):
    # sweep-b: a classifier forest member packed through member_pack_labeled has classes_ (was missing).
    labels = np.array([3, 7, 9], dtype=np.int64)[C]
    search = cls("GridSearchCV")(make("RandomForestClassifier", device, n_estimators=5, random_state=0),
                                 {"max_depth": [2, 4]}, cv=3).fit(X, labels)
    best = search.best_estimator_
    hand = make("RandomForestClassifier", device, n_estimators=5, random_state=0,
                max_depth=search.best_params_["max_depth"]).fit(X, labels)
    assert_same(best.classes_, hand.classes_, "classes_")
    assert_same(best.predict(X_NEW), hand.predict(X_NEW), "predict")


@pytest.mark.parametrize("device", DEVICES)
def test_bagging_gaussian_nb_member_route_is_repeatable(device):
    # fx-audit session 2: bagging GaussianNB (int32 draws) on the member route.
    def fit():
        return cls("BaggingClassifier")(make("GaussianNB", device), n_estimators=5, random_state=0).fit(X, binary(C))

    first = fit()
    gc.collect()
    assert_same(fit().predict_proba(X_NEW), first.predict_proba(X_NEW), "bagging GaussianNB")
