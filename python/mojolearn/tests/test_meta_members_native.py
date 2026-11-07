# SPDX-License-Identifier: Apache-2.0
"""Meta-estimator members, fit loops and per-permutation folds on the native
path (lane ne2-b-members, rows N049 N055 N061 N064 N071-N076 N080-N085 of
docs/plans/native_entry_points_after_0866/owed.tsv). Small inputs; every
expected value comes from the library's own public calls on the same data,
so each check is a contract between two public routes, not a timing."""
import numpy as np
import pytest

import mojolearn as ml
import mojolearn.metrics as mm
import mojolearn.model_selection as ms


def _clf(k=3, n=90, d=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n, d), dtype=np.float32)
    y = (X[:, 0] * k).astype(np.int64).clip(0, k - 1)
    return X, y


def _forest():
    return ml.RandomForestClassifier(n_estimators=4, max_depth=3, random_state=0)


# ---------------------------------------------------------------- N075 / N076


def test_hard_vote_of_forest_members_matches_their_predictions():
    X, y = _clf()
    vote = ml.VotingClassifier([("a", _forest()), ("b", ml.DecisionTreeClassifier(max_depth=2))],
                               voting="hard").fit(X, y)
    votes = np.zeros((len(X), 3))
    for member in vote.estimators_:
        votes[np.arange(len(X)), np.asarray(member.predict(X)).astype(np.int64)] += 1.0
    expected = np.asarray(vote.classes_)[votes.argmax(axis=1)]
    np.testing.assert_array_equal(np.asarray(vote.predict(X)), expected)


def test_binary_ovr_and_multioutput_forest_members_answer_their_labels():
    X, y = _clf(k=2)
    ovr = ml.OneVsRestClassifier(_forest()).fit(X, y)
    np.testing.assert_array_equal(np.asarray(ovr.predict(X)),
                                  np.asarray(ovr.classes_)[np.asarray(ovr.estimators_[0].predict(X)).astype(np.int64)])
    Y = np.stack([y, 1 - y], axis=1)
    mo = ml.MultiOutputClassifier(_forest()).fit(X, Y)
    got = np.asarray(mo.predict(X))
    for j, member in enumerate(mo.estimators_):
        np.testing.assert_array_equal(got[:, j], np.asarray(mo.classes_[j])[np.asarray(member.predict(X)).astype(np.int64)])


# ---------------------------------------------------------------- N082


@pytest.mark.parametrize("make", [
    lambda: ml.VotingClassifier([("a", _forest()), ("b", ml.DecisionTreeClassifier(max_depth=2))], voting="soft"),
    lambda: ml.VotingClassifier([("a", _forest()), ("b", ml.DecisionTreeClassifier(max_depth=2))], voting="hard"),
    lambda: ml.StackingClassifier([("a", _forest())], cv=2),
    lambda: ml.OneVsRestClassifier(_forest()),
    lambda: ml.CalibratedClassifierCV(_forest(), cv=2),
])
@pytest.mark.parametrize("labels", ["int", "str"])
def test_meta_classifier_score_is_the_accuracy_of_its_predict(make, labels):
    X, y = _clf()
    if labels == "str":
        y = np.array(["ant", "bee", "cat"])[y]
    model = make().fit(X, y)
    w = np.linspace(0.5, 1.5, len(X)).astype(np.float32)
    pred = model.predict(X)
    assert model.score(X, y) == mm.accuracy_score(y, pred)
    assert model.score(X, y, sample_weight=w) == mm.accuracy_score(y, pred, sample_weight=w)
    if labels == "str":
        # a truth label the model never saw never matches
        y_unseen = np.array(y, dtype=object)
        y_unseen[0] = "zzz"
        y_unseen = y_unseen.astype(str)
        assert model.score(X, y_unseen) == mm.accuracy_score(y_unseen, pred)


# ---------------------------------------------------------------- N081 / N084


def test_multioutput_score_is_the_all_outputs_match_fraction():
    X, y = _clf()
    Y = np.stack([y, (y + 1) % 3], axis=1)
    mo = ml.MultiOutputClassifier(_forest()).fit(X, Y)
    pred = np.asarray(mo.predict(X))
    hit = np.all(pred == Y, axis=1)
    w = np.linspace(0.5, 1.5, len(X)).astype(np.float32)
    assert mo.score(X, Y) == mm.accuracy_score(np.ones(len(X), dtype=np.int32), hit.astype(np.int32))
    assert mo.score(X, Y, sample_weight=w) == mm.accuracy_score(np.ones(len(X), dtype=np.int32), hit.astype(np.int32),
                                                                sample_weight=w)


def test_multioutput_string_labels_predict_and_score():
    X, y = _clf()
    names = np.array(["ant", "bee", "cat"])
    Y = np.stack([names[y], names[(y + 1) % 3]], axis=1)
    mo = ml.MultiOutputClassifier(_forest()).fit(X, Y)
    pred = mo.predict(X)
    rows = [list(r) for r in pred]
    assert len(rows) == len(X) and all(len(r) == 2 for r in rows)
    hit = np.array([list(p) == list(t) for p, t in zip(rows, Y.tolist())])
    expected = float(hit.mean())
    assert mo.score(X, Y) == pytest.approx(expected, abs=0.0)
    flipped = Y.copy()
    flipped[:, 1] = "zzz"  # an unseen label in one output: no row matches
    assert mo.score(X, flipped) == 0.0
    with pytest.raises(Exception):
        mo.score(X, Y[:, :1])


# ---------------------------------------------------------------- N071 / N072 / N073


def test_meta_configuration_changes_are_seen_after_the_admission_stamp():
    X, y = _clf()
    vote = ml.VotingClassifier([("a", _forest()), ("b", ml.DecisionTreeClassifier(max_depth=2))]).fit(X, y)
    first = np.asarray(vote.predict(X))
    np.testing.assert_array_equal(np.asarray(vote.predict(X)), first)   # stamped admission: same answer
    vote.set_params(voting="bogus")
    with pytest.raises(Exception):
        vote.predict(X)
    vote.set_params(voting="hard")
    np.testing.assert_array_equal(np.asarray(vote.predict(X)), first)
    # an in-place edit of the member list is seen too
    vote.estimators.append(("c", "not an estimator"))
    with pytest.raises(Exception):
        vote.predict(X)


def test_staged_member_tables_follow_a_member_refit():
    X, y = _clf()
    vote = ml.VotingClassifier([("a", _forest()), ("b", _forest())], voting="soft").fit(X, y)
    before = np.asarray(vote.predict_proba(X))
    np.testing.assert_array_equal(np.asarray(vote.predict_proba(X)), before)
    vote.estimators_[0].set_params(inference_engine="sequential")
    np.testing.assert_array_equal(np.asarray(vote.predict_proba(X)), before)


# ---------------------------------------------------------------- N083


def test_soft_vote_is_the_weighted_mean_of_member_probabilities():
    X, y = _clf()
    vote = ml.VotingClassifier([("a", _forest()), ("b", ml.DecisionTreeClassifier(max_depth=2))], voting="soft",
                               weights=[2.0, 1.0]).fit(X, y)
    acc = np.zeros((len(X), 3))
    for member, w in zip(vote.estimators_, [2.0, 1.0]):
        p = np.asarray(member.predict_proba(X), dtype=np.float64)
        acc[:, np.asarray(member.classes_).astype(np.int64)] += w * p
    np.testing.assert_allclose(np.asarray(vote.predict_proba(X)), acc / 3.0, rtol=0, atol=1e-6)


# ---------------------------------------------------------------- N049 / N061 / N064


def test_bagging_members_hold_their_own_seeds_and_the_validated_parameters():
    X, y = _clf()
    bag = ml.BaggingClassifier(ml.DecisionTreeClassifier(max_depth=3), n_estimators=5, random_state=3).fit(X, y)
    members = bag.estimators_
    seeds = [m.get_params()["random_state"] for m in members]
    assert len(set(seeds)) == len(seeds)
    for m in members:
        assert m.get_params()["max_depth"] == 3
        np.testing.assert_array_equal(np.asarray(m.predict(X)).shape, (len(X),))


def test_dart_and_adaboost_fits_repeat_bit_for_bit():
    rng = np.random.default_rng(1)
    X = rng.random((120, 4), dtype=np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1]).astype(np.float32)
    a = ml.DARTRegressor(n_estimators=8, random_state=0).fit(X, y)
    b = ml.DARTRegressor(n_estimators=8, random_state=0).fit(X, y)
    np.testing.assert_array_equal(np.asarray(a.predict(X)), np.asarray(b.predict(X)))
    seeds = [t.get_params()["random_state"] for t in a.trees_]
    assert len(set(seeds)) == len(seeds)
    yc = (X[:, 0] > 0.5).astype(np.int64)
    c1 = ml.AdaBoostClassifier(n_estimators=6, random_state=0).fit(X, yc)
    c2 = ml.AdaBoostClassifier(n_estimators=6, random_state=0).fit(X, yc)
    np.testing.assert_array_equal(np.asarray(c1.predict(X)), np.asarray(c2.predict(X)))
    r1 = ml.AdaBoostRegressor(n_estimators=6, random_state=0).fit(X, y)
    r2 = ml.AdaBoostRegressor(n_estimators=6, random_state=0).fit(X, y)
    np.testing.assert_array_equal(np.asarray(r1.predict(X)), np.asarray(r2.predict(X)))


# ---------------------------------------------------------------- N085


def test_dart_tree_explainer_is_additive():
    rng = np.random.default_rng(2)
    X = rng.random((150, 4), dtype=np.float32)
    y = (X[:, 0] - X[:, 2]).astype(np.float32)
    d = ml.DARTRegressor(n_estimators=10, random_state=0).fit(X, y)
    e = ml.TreeExplainer(d, data=X[:60])
    p = np.asarray(e.shap_values(X[:20]))
    np.testing.assert_allclose(p.sum(1) + e.expected_value, np.asarray(d.predict(X[:20])), rtol=1e-5, atol=1e-3)


# ---------------------------------------------------------------- N055


def test_permutation_test_score_with_redrawn_folds_repeats():
    X, y = _clf(k=2, n=60)
    cv = ml.KFold(n_splits=3, shuffle=True)   # unseeded: folds are redrawn per permutation
    s1 = ms.permutation_test_score(ml.DecisionTreeClassifier(max_depth=2), X, y, cv=cv, n_permutations=4,
                                   random_state=0)
    assert len(s1[1]) == 4 and 0.0 < s1[2] <= 1.0
    cv_int = ms.permutation_test_score(ml.DecisionTreeClassifier(max_depth=2), X, y, cv=3, n_permutations=4,
                                       random_state=0)
    again = ms.permutation_test_score(ml.DecisionTreeClassifier(max_depth=2), X, y, cv=3, n_permutations=4,
                                      random_state=0)
    assert cv_int[0] == again[0]
    np.testing.assert_array_equal(np.asarray(cv_int[1]), np.asarray(again[1]))


# ---------------------------------------------------------------- N080


def test_roc_auc_ovr_micro_is_the_raveled_binary_auc():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 3, 80)
    s = rng.random((80, 3))
    s = s / s.sum(axis=1, keepdims=True)
    onehot = (y[:, None] == np.arange(3)[None, :]).astype(np.int64)
    got = mm.roc_auc_score(y, s, multi_class="ovr", average="micro")
    assert got == mm.roc_auc_score(onehot.ravel(), s.ravel())
    w = np.linspace(0.5, 1.5, 80)
    got_w = mm.roc_auc_score(y, s, multi_class="ovr", average="micro", sample_weight=w)
    assert got_w == mm.roc_auc_score(onehot.ravel(), s.ravel(), sample_weight=np.repeat(w, 3))
    bad = s.copy()
    bad[0] *= 2.0   # rows must sum to 1: refused as before
    with pytest.raises(Exception):
        mm.roc_auc_score(y, bad, multi_class="ovr", average="micro")
