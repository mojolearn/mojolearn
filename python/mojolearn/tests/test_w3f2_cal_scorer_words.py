# SPDX-License-Identifier: Apache-2.0
"""A search's probability scorer on a NON-classifier prints y_true as scikit-learn prints it (lane w3f2-cal;
core/msel_scorer_run.mojo `_unknown_label_words`, `_label_text`): a continuous y in LabelBinarizer's words with its
NumPy repr, an integer y's one label as an integer. The oracle is scikit-learn's own GridSearchCV over its
GaussianMixture with the same folds (cv=2, unshuffled): the same first fold fails with the same words.
Fail-proof: set `scored.given = 0` in core/msel_search_run.mojo: the first test's words gain ", dtype=float32".
"""
import re

import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")
from sklearn.mixture import GaussianMixture as SkGMM  # noqa: E402
from sklearn.model_selection import GridSearchCV as SkGrid  # noqa: E402

import mojolearn as ml  # noqa: E402
from mojolearn import model_selection as ms  # noqa: E402


def _x(n=40, seed=2):
    return np.random.default_rng(seed).standard_normal((n, 2)).astype(np.float32)


def _words(make, X, y):
    with pytest.raises(ValueError) as caught:
        make().fit(X, y)
    return str(caught.value)


@pytest.mark.parametrize("y_kind", ["continuous", "one_int_label"])
def test_probability_scorer_prints_y_true_as_sklearn(y_kind):
    X = _x()
    if y_kind == "continuous":
        y = np.linspace(0.5, 4.0, len(X))
    else:
        y = np.ones(len(X), dtype=np.int64)
    theirs = _words(lambda: SkGrid(SkGMM(n_components=2, random_state=0), {"n_components": [2]}, cv=2,
                                   scoring="neg_log_loss", error_score="raise"), X, y)
    ours = _words(lambda: ms.GridSearchCV(ml.GaussianMixture(n_components=2, random_state=0), {"n_components": [2]},
                                          cv=2, scoring="neg_log_loss", error_score="raise"), X, y)
    assert re.sub(r"^ValueError: ", "", ours) == theirs
