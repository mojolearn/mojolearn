# SPDX-License-Identifier: Apache-2.0
"""Meta-estimator public boundary (search, RFE, voting, bagging and their members): refusals in the doors' and
scikit-learn's words, warnings, the targets a member route admits, and member parameters. The member-versus-door
agreements and the oracle checks stay in the test_mf_*_members.py, test_meta_members_native.py and
test_mf_r9d_scorer_rows.py files, whose helpers are imported here."""
import warnings

import numpy as np
import pytest

import mojolearn
import mojolearn as ml
from mojolearn import metrics
from mojolearn import model_selection as ms
from mojolearn.tests.test_meta_members_native import _clf as _meta_clf
from mojolearn.tests.test_mf_enc6_members import _both as _enc6_both, _data as _enc6_data, _member as _enc6_member
from mojolearn.tests.test_mf_lin6_members import _data as _lin6_data, _member as _lin6_member
from mojolearn.tests.test_mf_member8_members import (
    _NO_ROUND as _member8_NO_ROUND, _data as _member8_data, _same as _member8_same)
from mojolearn.tests.test_mf_nbr6_members import _data as _nbr6_data, _member as _nbr6_member
from mojolearn.tests.test_mf_r9d_scorer_rows import (
    _folds as _r9d_folds, _search as _r9d_search, _split_scores as _r9d_split_scores, _x as _r9d_x)
from mojolearn.tests.test_mf_rfe6_members import _data as _rfe6_data


# ---------------------------------------------------------------- moved from test_meta_members_native.py (vdv-pytest-a)


def test_bagging_members_hold_their_own_seeds_and_the_validated_parameters():
    X, y = _meta_clf()
    bag = ml.BaggingClassifier(ml.DecisionTreeClassifier(max_depth=3), n_estimators=5, random_state=3).fit(X, y)
    members = bag.estimators_
    seeds = [m.get_params()["random_state"] for m in members]
    assert len(set(seeds)) == len(seeds)
    for m in members:
        assert m.get_params()["max_depth"] == 3
        np.testing.assert_array_equal(np.asarray(m.predict(X)).shape, (len(X),))


# ---------------------------------------------------------------- moved from test_mf_lin6_members.py (vdv-pytest-a)
# The listed-cv, LDA and HostLasso refusals keep the door's words through the member route (lane mf-lin6).


@pytest.mark.parametrize("cv,words", [
    ([], "cv produced no \\(train, test\\) split"),
    ([(np.arange(10), np.arange(10, 20), np.arange(5))], "too many values to unpack \\(expected 2, got 3\\)"),
    ([(np.arange(10),)], "not enough values to unpack \\(expected 2, got 1\\)"),
])
def test_listed_cv_malformed_keeps_door_words(cv, words):
    X, y = _lin6_data()
    with pytest.raises(Exception, match=words):
        mojolearn.LassoCV(cv=cv).fit(X, y)
    with pytest.raises(Exception, match=words):
        _lin6_member(mojolearn.LassoCV(cv=cv), X, y, scoring="r2")


def test_lda_n_components_zero_is_sklearn_refusal():
    X, y = _lin6_data()
    labels = (y > 0).astype(np.int64)
    words = "The 'n_components' parameter of LinearDiscriminantAnalysis must be an int in the range \\[1, inf\\)"
    with pytest.raises(Exception, match=words):
        mojolearn.LinearDiscriminantAnalysis(n_components=0).fit(X, labels)
    with pytest.raises(Exception, match=words):
        _lin6_member(mojolearn.LinearDiscriminantAnalysis(n_components=0), X, labels, scoring="accuracy")


def test_host_lasso_member_fit_keeps_door_words():
    from mojolearn._classical_host import HostLasso
    X, y = _lin6_data()
    words = "no CPU implementation of the HostLasso fit on the host class"
    with pytest.raises(Exception, match=words):
        HostLasso().fit(X, y)
    with pytest.raises(Exception, match=words):
        _lin6_member(HostLasso(), X, y, scoring="r2")


# ---------------------------------------------------------------- moved from test_mf_member8_members.py (vdv-pytest-a)
# RFE refusals in RFE's, Python's and scikit-learn's words, and the targets RFE admits (lane mf-member8).


@pytest.mark.parametrize("name,make,attributes", _member8_NO_ROUND, ids=[row[0] for row in _member8_NO_ROUND])
def test_a_round_is_refused_in_rfe_words(name, make, attributes):
    x, y = _member8_data()
    with pytest.raises(Exception, match="RFE needs an estimator with coef_ or feature_importances_"):
        mojolearn.RFE(make(), n_features_to_select=2).fit(x, y)


def test_cholesky_refused_in_python_words():
    rng = np.random.default_rng(2)
    m = rng.random((6, 6)).astype(np.float32)
    a = (m @ m.T + 6 * np.eye(6, dtype=np.float32)).astype(np.float32)
    with pytest.raises(TypeError, match=r"Cholesky.fit\(\) takes 2 positional arguments but 3 were given"):
        mojolearn.RFE(mojolearn.Cholesky(jitter=0.0), n_features_to_select=6).fit(a, np.zeros(6, np.float32))


def test_affinity_propagation_precomputed_has_no_centers():
    x, y = _member8_data(n=12, d=12)
    s = -np.abs(x - x.T).astype(np.float32)
    with pytest.raises(AttributeError, match="has no attribute 'cluster_centers_'"):
        mojolearn.RFE(mojolearn.AffinityPropagation(affinity="precomputed", random_state=0),
                      n_features_to_select=6, importance_getter="cluster_centers_").fit(s, y)


@pytest.mark.parametrize("name,make,attributes", _member8_NO_ROUND, ids=[row[0] for row in _member8_NO_ROUND])
@pytest.mark.parametrize("target", ["two_d", "labels"])
def test_no_round_accepts_what_check_X_y_accepts(name, make, attributes, target):
    x, y = _member8_data()
    yy = np.stack([y, -y], axis=1) if target == "two_d" else np.where(y > 0, "pos", "neg")
    rfe = mojolearn.RFE(make(), n_features_to_select=x.shape[1]).fit(x, yy)
    own = make().fit(x)
    _member8_same(getattr(rfe.estimator_, attributes[0]), getattr(own, attributes[0]))


def test_no_round_y_refusals_in_scikit_learn_words():
    x, y = _member8_data()
    with pytest.raises(ValueError, match="RFE requires y to be passed, but the target y is None"):
        mojolearn.RFE(mojolearn.NearestNeighbors(), n_features_to_select=x.shape[1]).fit(x, None)
    bad = y.copy()
    bad[3] = np.nan
    with pytest.raises(ValueError, match="Input y contains NaN"):
        mojolearn.RFE(mojolearn.OPTICS(min_samples=4), n_features_to_select=x.shape[1]).fit(x, bad)
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        mojolearn.RFE(mojolearn.MDS(random_state=0), n_features_to_select=x.shape[1]).fit(x, y[:-1])


def test_meta_fits_a_target_free_member_and_predict_fails_as_scikit_learn():
    x, y = _member8_data()
    labels = (y > 0).astype(np.int64)
    vote = mojolearn.VotingClassifier([("nn", mojolearn.NearestNeighbors(n_neighbors=3))]).fit(x, labels)
    _member8_same(vote.estimators_[0]._index, mojolearn.NearestNeighbors(n_neighbors=3).fit(x)._index)
    with pytest.raises(AttributeError, match="has no attribute 'predict'"):
        vote.predict(x)


def test_mincovdet_precision_getter_refused_without_store_precision():
    x, y = _member8_data(n=80, d=6)
    with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'ndim'"):
        mojolearn.RFE(mojolearn.MinCovDet(store_precision=False, random_state=0), n_features_to_select=3,
                      importance_getter="precision_").fit(x, y)


# ---------------------------------------------------------------- moved from test_mf_nbr6_members.py (vdv-pytest-a)
# The KNNImputer and DictionaryLearning callback refusals through the member route (lane mf-nbr6).


def test_knn_imputer_zero_neighbors_fits_then_refuses():
    X, _ = _nbr6_data()
    member = _nbr6_member(mojolearn.KNNImputer(n_neighbors=0), X)
    with pytest.raises(Exception, match="n_neighbors must be >= 1"):
        member.transform(X)


def test_dictionary_learning_callback_refused_in_one_text():
    X, _ = _nbr6_data(d=4)
    words = "callback is not carried"
    with pytest.raises(Exception, match=words):
        mojolearn.DictionaryLearning(n_components=3, callback=lambda *a: None).fit(X)
    with pytest.raises(Exception, match=words):
        _nbr6_member(mojolearn.DictionaryLearning(n_components=3, callback=lambda *a: None), X)


# ---------------------------------------------------------------- moved from test_mf_r9d_scorer_rows.py (vdv-pytest-a)
# The scorer rows' refusals in scikit-learn's words, beside scikit-learn's own search (lane mf-r9d).


def test_roc_auc_over_a_mixture_models_columns():
    sk_ms = pytest.importorskip('sklearn.model_selection')
    sk_mixture = pytest.importorskip('sklearn.mixture')
    # Fail-proof: in core/msel_scorer_run.mojo `_float_roc_value` drop the two-column refusal (two components
    # score instead of raising).
    X = _r9d_x()
    y = (np.arange(12) % 2).astype(np.float64)
    one = _r9d_search(ml.GaussianMixture(random_state=0, numeric_mode='identical'), {'n_components': [1]}, X, y,
                      'roc_auc')
    np.testing.assert_allclose(_r9d_split_scores(one), [0.5, 0.5, 0.5])
    with pytest.raises(ValueError, match=r'y should be a 1d array, got an array of shape \(4, 2\) instead'):
        _r9d_search(ml.GaussianMixture(random_state=0, numeric_mode='identical'), {'n_components': [2]}, X, y,
                    'roc_auc')
    with pytest.raises(ValueError, match=r'y should be a 1d array, got an array of shape \(4, 2\) instead'):
        sk_ms.GridSearchCV(sk_mixture.GaussianMixture(random_state=0), {'n_components': [2]}, cv=_r9d_folds(12),
                           scoring='roc_auc', error_score='raise', refit=False).fit(X, y)
    tolerated = _r9d_search(ml.GaussianMixture(random_state=0, numeric_mode='identical'), {'n_components': [2]},
                            X, y, 'roc_auc', error_score=-3.0)
    np.testing.assert_array_equal(_r9d_split_scores(tolerated), [-3.0, -3.0, -3.0])


def test_roc_auc_over_a_mixture_model_with_continuous_targets_is_sklearns_words():
    # Fail-proof: in `_float_roc_value` code a continuous y as labels (the side scores).
    X = _r9d_x()
    y = np.linspace(0.0, 1.0, 12) + 0.25
    with pytest.raises(ValueError, match='continuous format is not supported'):
        _r9d_search(ml.GaussianMixture(random_state=0, numeric_mode='identical'), {'n_components': [1]}, X, y,
                    'roc_auc')


def test_log_loss_over_a_mixture_model_with_more_target_values_than_components_is_sklearns_words():
    sk_ms = pytest.importorskip('sklearn.model_selection')
    sk_mixture = pytest.importorskip('sklearn.mixture')
    # ROUND b. Fail-proof: in `_float_proba_value` drop log_loss's class-count check (the side scores).
    X = _r9d_x()
    y = (np.arange(12) % 3).astype(np.float64)
    with pytest.raises(ValueError, match='y_true and y_prob contain different number of classes: 3 vs 2'):
        _r9d_search(ml.GaussianMixture(random_state=0, numeric_mode='identical'), {'n_components': [2]}, X, y,
                    'neg_log_loss')
    with pytest.raises(ValueError, match='y_true and y_prob contain different number of classes: 3 vs 2'):
        sk_ms.GridSearchCV(sk_mixture.GaussianMixture(random_state=0), {'n_components': [2]}, cv=_r9d_folds(12),
                           scoring='neg_log_loss', error_score='raise', refit=False).fit(X, y)


def test_a_label_metric_whose_tuple_resolves_to_predict_proba_is_sklearns_mix_words():
    sk_ms = pytest.importorskip('sklearn.model_selection')
    # ROUND b. Fail-proof: in `_wrong_response_value` drop the binary column pick (the words name
    # 'continuous-multioutput').
    sk_linear = pytest.importorskip('sklearn.linear_model')
    sk_m = pytest.importorskip('sklearn.metrics')
    X = _r9d_x()
    y = (np.arange(12) % 2).astype(np.int64)
    ours = ms.make_scorer(metrics.accuracy_score, response_method=('predict_proba', 'predict'))
    with pytest.raises(ValueError, match="can't handle a mix of binary and continuous targets"):
        _r9d_search(ml.LogisticRegression(max_iter=30), {'C': [1.0]}, X, y, ours)
    with pytest.raises(ValueError, match="can't handle a mix of binary and continuous targets"):
        sk_ms.GridSearchCV(sk_linear.LogisticRegression(max_iter=30), {'C': [1.0]}, cv=_r9d_folds(12),
                           scoring=sk_m.make_scorer(sk_m.accuracy_score, response_method=('predict_proba', 'predict')),
                           error_score='raise', refit=False).fit(X, y)


def test_top_k_with_a_bad_k_is_the_doors_words_per_side():
    # ROUND b. Fail-proof: in `_scorer_name` drop the k refusal (the search keeps no row for it).
    X = _r9d_x()
    y = (np.arange(12) % 3).astype(np.int64)
    scorer = ms.make_scorer(metrics.top_k_accuracy_score, response_method=('decision_function', 'predict_proba'),
                            k=0)
    with pytest.raises(Exception, match='k must be a positive integer'):
        _r9d_search(ml.LogisticRegression(max_iter=30), {'C': [1.0]}, X, y, scorer)
    tolerated = _r9d_search(ml.LogisticRegression(max_iter=30), {'C': [1.0]}, X, y, scorer, error_score=-3.0)
    np.testing.assert_array_equal(_r9d_split_scores(tolerated), [-3.0, -3.0, -3.0])


# ---------------------------------------------------------------- moved from test_mf_rfe6_members.py (vdv-pytest-a)
# A refused target raises the estimator's own fit's words through RFE (lane mf-rfe6).


def _rfe6_words(call):
    try:
        call()
    except Exception as error:  # noqa: BLE001 - the words are what is compared
        return type(error).__name__, str(error)
    raise AssertionError("no refusal")


@pytest.mark.parametrize("make", [
    lambda: mojolearn.Ridge(alpha=1.0),
    lambda: mojolearn.GradientBoostingRegressor(n_estimators=3, max_depth=2, random_state=0),
], ids=["ridge", "gbdt"])
def test_refused_target_raises_the_estimators_own_words(make):
    x, _, t = _rfe6_data()
    bad = t.copy()
    bad[3] = np.nan
    own = _rfe6_words(lambda: make().fit(x, bad))
    rfe = _rfe6_words(lambda: mojolearn.RFE(make(), n_features_to_select=3).fit(x, bad))
    assert rfe == own


def test_short_labels_raise_before_any_fit():
    x, y, _ = _rfe6_data()
    with pytest.raises(ValueError, match=r"inconsistent numbers of samples: \[64, 63\]"):
        mojolearn.RFE(mojolearn.LogisticRegression(max_iter=50), n_features_to_select=3).fit(x, y[:-1])


# ---------------------------------------------------------------- moved from test_mf_enc6_members.py (vdv-pytest-a)
# The OneHotEncoder drop refusals and unknown-category warnings through the member route (lane mf-enc6).


def test_onehot_drop_refusals_are_the_doors():
    X = _enc6_data()
    with pytest.raises(Exception, match=r"were not found in the training data.\nCategory: 9.0, Feature: 0"):
        _enc6_member(mojolearn.OneHotEncoder(drop=[9.0, 10.0, 5.0], sparse_output=False), X)
    with pytest.raises(Exception, match="because it is infrequent"):
        _enc6_member(mojolearn.OneHotEncoder(drop=[4.0, 10.0, 5.0], min_frequency=4, sparse_output=False), X)
    with pytest.raises(Exception, match="should have length equal to the number of features"):
        _enc6_member(mojolearn.OneHotEncoder(drop=[0.0, 10.0], sparse_output=False), X)


@pytest.mark.parametrize("kwargs", [{"handle_unknown": "warn"}, {"handle_unknown": "ignore", "drop": "first"},
                                    {"handle_unknown": "infrequent_if_exist", "drop": "first", "min_frequency": 4}])
def test_onehot_warning_settings(kwargs):
    X = _enc6_data()
    member, door = _enc6_both(lambda: mojolearn.OneHotEncoder(sparse_output=False, **kwargs), X, inverse=False)
    unknown = X[:4].copy()
    unknown[:, 0] = 99.0
    with warnings.catch_warnings(record=True) as seen_door:
        warnings.simplefilter("always")
        want = np.asarray(door.transform(unknown))
    with warnings.catch_warnings(record=True) as seen_member:
        warnings.simplefilter("always")
        got = np.asarray(member.transform(unknown))
    np.testing.assert_array_equal(got, want)
    assert [str(w.message) for w in seen_member] == [str(w.message) for w in seen_door]
    assert any("Found unknown categories in columns [0]" in str(w.message) for w in seen_door)
