# SPDX-License-Identifier: Apache-2.0
"""A class_weight dict naming a label y lacks, while leaving a class of y unweighted, is refused in scikit-learn
1.9.1's words on every route (lane mf-gate6, 2026-11-23): sklearn/utils/class_weight.py:114-120 (compute_class_weight,
reached by LogisticRegression, LinearSVC, SVC, SGDClassifier and RidgeClassifier), which list `int(c)` per unweighted
class of y (:105-108). The member route of an IterativeImputer classifier step (x_prep/iterimp_run.mojo,
core/member_class_weight.mojo `member_cw_values_refusal`) raised its family's fixed words before this lane ("class_weight
contains labels absent from y" / "names a label which is not in y's classes"); the public doors' words are pinned here
too (bindings/class_weight_absent_unpack.mojo). The configuration is tools/mf_left4_digests.py `imp_foreign_*`.
"""
import numpy as np
import pytest

import mojolearn
from mojolearn.impute import IterativeImputer

WORDS = "The classes, [0, 2], are not in class_weight"


def _member(name, class_weight):
    if name == "lr":
        return mojolearn.LogisticRegression(max_iter=200, class_weight=class_weight)
    if name == "svc":
        return mojolearn.LinearSVC(C=0.5, class_weight=class_weight)
    if name == "svk":
        return mojolearn.SVC(C=1.0, gamma=0.5, probability=False, class_weight=class_weight)
    if name == "sgd":
        return mojolearn.SGDClassifier(max_iter=20, random_state=3, class_weight=class_weight)
    return mojolearn.RidgeClassifier(alpha=0.7, class_weight=class_weight)


def _imputer_rows():
    X = np.random.default_rng(7).integers(0, 3, size=(120, 4)).astype(np.float32)
    X[np.random.default_rng(8).random(X.shape) < 0.15] = np.nan
    return X


@pytest.mark.parametrize("name", ["lr", "svc", "svk", "sgd", "ridgec"])
def test_iterative_imputer_member_names_unweighted_classes(name):
    # the step's classes are its target's values 0.0, 1.0, 2.0: the dict weights 1.0 and names 'a'
    with pytest.raises(ValueError) as refused:
        IterativeImputer(estimator=_member(name, {"a": 2.0, 1.0: 0.5}), max_iter=2, random_state=0).fit(
            _imputer_rows())
    assert str(refused.value) == WORDS


@pytest.mark.parametrize("name", ["lr", "svc", "svk", "sgd", "ridgec"])
@pytest.mark.parametrize("labels", ["int", "float", "str"])
def test_public_fit_names_unweighted_classes(name, labels):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(60, 3)).astype(np.float32)
    codes = np.arange(60) % 3
    X[:, 0] += codes
    if labels == "str":
        y, weight, words = np.array(["a", "b", "c"])[codes], {"b": 0.5, "z": 2.0}, "The classes, ['a', 'c'], are " \
            "not in class_weight"
    else:
        y = codes.astype(np.float64 if labels == "float" else np.int64)
        weight, words = {1: 0.5, 9: 2.0}, WORDS
    with pytest.raises(ValueError) as refused:
        _member(name, weight).fit(X, y)
    assert str(refused.value) == words


def test_iterative_imputer_words_equal_scikit_learn():
    sklearn = pytest.importorskip("sklearn")
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer as SkIterativeImputer
    from sklearn.linear_model import LogisticRegression as SkLogisticRegression
    del sklearn
    with pytest.raises(ValueError) as refused:
        SkIterativeImputer(estimator=SkLogisticRegression(max_iter=200, class_weight={"a": 2.0, 1.0: 0.5}),
                           max_iter=2, random_state=0).fit(_imputer_rows())
    assert str(refused.value) == WORDS
