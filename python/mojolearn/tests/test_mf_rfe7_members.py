# SPDX-License-Identifier: Apache-2.0
"""RFE rows lane mf-rfe7 (2026-11-23): the last users of the estimator's own doors (`_fit_clone`, the door's
rounds over its fit and attribute reads) moved onto the member RUN, or refused natively in the words that
door raised. Each test compares the member route with what the deleted door computed:
  * SVC / SVR with kernel='linear': the rounds read the member's coef_ block (the SVM coef_ operation's GEMV
    rows, which the door read): ranking_ equals an elimination written out over the estimator's own
    `coef_`, binary, one-vs-one multiclass and the regressor;
  * a named feature-wide attribute (GaussianNB theta_, MultinomialNB feature_log_prob_, LDA means_): ranking_
    equals the elimination over the estimator's own attribute, squared and summed over rows;
  * a META estimator with a plain getter: with no round the final fit runs (estimator_ predicts as the search
    fitted on the columns); with a round, the door's words;
  * refusals in the deleted rounds' words: a non-linear SVC, ExtraTrees 'auto', an unknown name, a step RFE
    cannot follow, RandomForestRegressor(oob_score=True) with sample_weight.

Runs with the built package (the python/mojolearn/tests suite on a box, never on the laptop). Written, not run.
"""
import numpy as np
import pytest

import mojolearn
from mojolearn.model_selection import GridSearchCV


def _data(n=72, d=7, seed=11, classes=2):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)).astype(np.float32)
    score = x[:, 0] + 0.7 * x[:, 2] - 0.4 * x[:, 5]
    if classes == 2:
        y = (score > 0).astype(np.int64)
    else:
        y = np.digitize(score, np.quantile(score, [1 / 3, 2 / 3])).astype(np.int64)
    t = (2.0 * x[:, 0] - x[:, 3] + 0.3 * x[:, 6]).astype(np.float32)
    return x, y, t


def _written_out(make, attribute, x, y, n_select, step=1):
    """RFE's elimination written out over the estimator's own fitted attribute (the deleted door's reads):
    fit on the supported columns, square, sum over rows, drop the `step` smallest (ties by column order, as
    c2_rfe_step's stable order), rank."""
    d = x.shape[1]
    support = np.ones(d, dtype=bool)
    ranking = np.ones(d, dtype=np.int64)
    while support.sum() > n_select:
        cols = np.flatnonzero(support)
        est = make().fit(x[:, cols], y)
        values = np.asarray(getattr(est, attribute), dtype=np.float64)
        keys = values ** 2 if values.ndim == 1 else (values ** 2).sum(axis=0)
        drop = min(step, len(cols) - n_select)
        gone = cols[np.argsort(keys, kind="stable")[:drop]]
        support[gone] = False
        ranking[~support] += 1
    return support, ranking


@pytest.mark.parametrize("classes", [2, 3])
def test_linear_svc_rounds_read_the_member_coef(classes):
    x, y, _ = _data(classes=classes)
    make = lambda: mojolearn.SVC(kernel="linear", C=1.0)
    rfe = mojolearn.RFE(make(), n_features_to_select=3).fit(x, y)
    support, ranking = _written_out(make, "coef_", x, y, 3)
    assert list(rfe.support_) == list(support)
    np.testing.assert_array_equal(np.asarray(rfe.ranking_), ranking)
    np.testing.assert_array_equal(rfe.predict(x), make().fit(x[:, support], y).predict(x[:, support]))


def test_linear_svr_rounds_read_the_member_coef():
    x, _, t = _data()
    make = lambda: mojolearn.SVR(kernel="linear", C=1.0)
    rfe = mojolearn.RFE(make(), n_features_to_select=2, importance_getter="coef_").fit(x, t)
    support, ranking = _written_out(make, "coef_", x, t, 2)
    np.testing.assert_array_equal(np.asarray(rfe.ranking_), ranking)


@pytest.mark.parametrize("cls, getter, positive", [("GaussianNB", "theta_", False),
                                                   ("MultinomialNB", "feature_log_prob_", True),
                                                   ("LinearDiscriminantAnalysis", "means_", False)])
def test_named_feature_wide_attribute(cls, getter, positive):
    x, y, _ = _data(classes=3)
    if positive:
        x = np.abs(x)
    make = lambda: getattr(mojolearn, cls)()
    rfe = mojolearn.RFE(make(), n_features_to_select=3, importance_getter=getter).fit(x, y)
    support, ranking = _written_out(make, getter, x, y, 3)
    np.testing.assert_array_equal(np.asarray(rfe.ranking_), ranking)


def test_meta_with_a_plain_getter():
    x, y, _ = _data()
    every = np.arange(len(x))
    search = lambda: GridSearchCV(mojolearn.LinearDiscriminantAnalysis(), {"tol": [1e-4]}, cv=[(every, every)])
    kept = mojolearn.RFE(search(), n_features_to_select=x.shape[1]).fit(x, y)
    assert all(kept.support_)
    np.testing.assert_array_equal(kept.predict(x), search().fit(x, y).predict(x))
    with pytest.raises(Exception, match="RFE needs an estimator with coef_ or feature_importances_"):
        mojolearn.RFE(search(), n_features_to_select=3).fit(x, y)


def test_refusals_in_the_door_words():
    x, y, t = _data()
    with pytest.raises(AttributeError, match="coef_ is only available for kernel='linear'"):
        mojolearn.RFE(mojolearn.SVC(kernel="rbf"), n_features_to_select=3).fit(x, y)
    with pytest.raises(Exception, match="RFE needs an estimator with coef_ or feature_importances_"):
        mojolearn.RFE(mojolearn.ExtraTreesRegressor(n_estimators=3), n_features_to_select=3).fit(x, t)
    with pytest.raises(AttributeError, match="'GaussianNB' object has no attribute 'sigma_'"):
        mojolearn.RFE(mojolearn.GaussianNB(), n_features_to_select=3, importance_getter="sigma_").fit(x, y)
    with pytest.raises(TypeError, match="is not one fitted estimator RFE can follow natively"):
        mojolearn.RFE(mojolearn.BaggingClassifier(mojolearn.LinearDiscriminantAnalysis(), n_estimators=2),
                      n_features_to_select=3, importance_getter="estimators_.coef_").fit(x, y)
    weights = np.ones(len(x), dtype=np.float32)
    with pytest.raises(NotImplementedError, match="sample_weight is not implemented together with oob_score"):
        mojolearn.RFE(mojolearn.RandomForestRegressor(n_estimators=4, oob_score=True, random_state=0),
                      n_features_to_select=3).fit(x, t, sample_weight=weights)
    # No round reads importances: the final fit runs (as scikit-learn's RFE), even for a kind with neither block.
    kept = mojolearn.RFE(mojolearn.ExtraTreesRegressor(n_estimators=3, random_state=0),
                         n_features_to_select=x.shape[1]).fit(x, t)
    assert all(kept.support_)
