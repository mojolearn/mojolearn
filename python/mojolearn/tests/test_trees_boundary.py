# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Python API boundary tests of the trees family: refusals, exception types and words, returned types,
buffer admission and framework interop. Numerical results are held by the verify lanes."""


# moved from test_x_trees_sanity.py (vdv-pytest-b, 2026-11-24)
import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")
from sklearn.datasets import make_classification  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

import mojolearn as ml  # noqa: E402


def _xts_clf(n_classes=3, seed=0):
    X, y = make_classification(n_samples=1500, n_features=10, n_informative=6, n_classes=n_classes,
                               random_state=seed)
    return train_test_split(X.astype(np.float32), y, test_size=0.3, random_state=seed)


def test_cv_splitter_objects():
    # The meta-estimators admit mojolearn's native splitters and lists of
    # index pairs; a Python splitter's split() would be a callback inside the
    # native fit (bindings/trees_public_meta_door.mojo _cv_value).
    from mojolearn.model_selection import KFold, StratifiedKFold
    Xa, Xb, ya, yb = _xts_clf()
    est = [("dt", ml.DecisionTreeClassifier(max_depth=4))]
    a = ml.StackingClassifier(est, final_estimator=ml.DecisionTreeClassifier(max_depth=3), cv=3).fit(Xa, ya)
    b = ml.StackingClassifier(est, final_estimator=ml.DecisionTreeClassifier(max_depth=3),
                              cv=StratifiedKFold(3)).fit(Xa, ya)
    np.testing.assert_array_equal(np.asarray(a.transform(Xb)), np.asarray(b.transform(Xb)))
    pairs = [(np.asarray(tr).tolist(), np.asarray(te).tolist())
             for tr, te in KFold(4, shuffle=True, random_state=0).split(Xa)]
    c = ml.CalibratedClassifierCV(ml.DecisionTreeClassifier(max_depth=4), cv=pairs).fit(Xa, ya)
    assert len(c.calibrated_classifiers_) == 4
    d = ml.CalibratedClassifierCV(ml.DecisionTreeClassifier(max_depth=4), cv=KFold(3, shuffle=True, random_state=1),
                                  method="isotonic", ensemble=False).fit(Xa, ya)
    assert accuracy_score(yb, np.asarray(d.predict(Xb))) > 0.5
