"""IsolationForest `save()` and pickle write the fitted forest, not the
training matrix (lane fix-iforest-save-device-forest, 2026-11-13).

A fit on a resident binary borrows a float32 C-order X, so a caller who
edits that X in place after fit must not change what a saved, pickled or
copied model scores. Public API boundary tests on small inputs."""
import copy
import pickle

import numpy as np
import pytest

from mojolearn import IsolationForest


def _data():
    rng = np.random.default_rng(11)
    X = np.ascontiguousarray(rng.random((256, 4), dtype=np.float32))
    query = np.ascontiguousarray(rng.random((17, 4), dtype=np.float32))
    return X, query


def _scores(model, query):
    return (np.asarray(model.score_samples(query)).copy(),
            np.asarray(model.decision_function(query)).copy(),
            np.asarray(model.predict(query)).copy())


def _assert_same(got, want):
    for g, w in zip(got, want):
        np.testing.assert_array_equal(g, w)


def _mutate(X):
    X[:] = X[::-1] * np.float32(3.0) + np.float32(1.0)


@pytest.mark.parametrize("contamination", ["auto", 0.1])
def test_save_after_in_place_edit_scores_the_fitted_forest(tmp_path, contamination):
    X, query = _data()
    model = IsolationForest(n_estimators=8, max_samples=64, random_state=3, contamination=contamination).fit(X)
    before = _scores(model, query)
    _mutate(X)
    path = tmp_path / "iforest.npz"
    model.save(str(path))
    loaded = IsolationForest.load(str(path))
    assert loaded.offset_ == model.offset_
    assert loaded.max_samples_ == model.max_samples_
    assert loaded.n_features_in_ == 4
    _assert_same(_scores(loaded, query), before)
    # The saved file holds the trees, not the training matrix.
    with np.load(str(path)) as members:
        assert "x" not in members.files
        assert str(members["format"]) == "mojolearn-iforest-2"
        assert members["tree_offsets"].shape == (9,)


def test_pickle_and_copy_after_in_place_edit_score_the_fitted_forest():
    X, query = _data()
    model = IsolationForest(n_estimators=8, max_samples=64, random_state=5).fit(X)
    before = _scores(model, query)
    _mutate(X)
    _assert_same(_scores(pickle.loads(pickle.dumps(model)), query), before)
    _assert_same(_scores(copy.deepcopy(model), query), before)
    _assert_same(_scores(copy.copy(model), query), before)
    # The estimator itself still scores its fitted forest.
    _assert_same(_scores(model, query), before)


def test_loaded_forest_saves_and_pickles_again(tmp_path):
    X, query = _data()
    model = IsolationForest(n_estimators=4, max_samples=32, random_state=9).fit(X)
    before = _scores(model, query)
    first = tmp_path / "a.npz"
    model.save(str(first))
    loaded = IsolationForest.load(str(first))
    second = tmp_path / "b.npz"
    loaded.save(str(second))
    _assert_same(_scores(IsolationForest.load(str(second)), query), before)
    _assert_same(_scores(pickle.loads(pickle.dumps(loaded)), query), before)


def test_refit_of_a_loaded_forest_uses_the_new_data(tmp_path):
    X, query = _data()
    model = IsolationForest(n_estimators=4, max_samples=32, random_state=9).fit(X)
    path = tmp_path / "a.npz"
    model.save(str(path))
    loaded = IsolationForest.load(str(path))
    X2 = np.ascontiguousarray(X[::-1] * np.float32(2.0))
    fresh = IsolationForest(n_estimators=4, max_samples=32, random_state=9).fit(X2)
    loaded.fit(X2)
    _assert_same(_scores(loaded, query), _scores(fresh, query))


def test_unfitted_save_is_refused(tmp_path):
    with pytest.raises(RuntimeError, match="not fitted"):
        IsolationForest().save(str(tmp_path / "never.npz"))


def test_host_column_loads_scores_and_resaves_a_saved_forest(tmp_path):
    # Save (mojolearn-iforest-2), load on the host column (HostIsolationForest,
    # no training matrix), score: equal to the scores before the save. The
    # loaded host model saves its forest again and that file scores the same.
    from mojolearn import _classical_host as host
    X, query = _data()
    model = IsolationForest(n_estimators=8, max_samples=64, random_state=13, contamination=0.1).fit(X)
    before = _scores(model, query)
    _mutate(X)
    path = tmp_path / "forest.npz"
    model.save(str(path))
    loaded = host.HostIsolationForest.load(str(path))
    assert "_x" not in vars(loaded)
    _assert_same(_scores(loaded, query), before)
    again = tmp_path / "again.npz"
    loaded.save(str(again))
    with np.load(str(again)) as members:
        assert str(members["format"]) == "mojolearn-iforest-2"
        assert "x" not in members.files
    _assert_same(_scores(host.HostIsolationForest.load(str(again)), query), before)


def test_fitted_estimator_pickles_its_resident_forest():
    # `fit` is one native call (no read-out at fit end unless the binary is
    # built with -D MOJOLEARN_IDN_IF_FIT_EXPORT); pickle reads the forest
    # the scores use when it runs, and writes it in place of `_x`.
    X, _ = _data()
    model = IsolationForest(n_estimators=4, max_samples=32, random_state=2).fit(X)
    state = model.__getstate__()
    forest = state["_if_forest"]
    assert "_x" not in state
    assert len(forest) == 5 and np.asarray(forest[0]).shape == (5,)
    # A second read gives the same forest words (no refit between them).
    again = model.__getstate__()["_if_forest"]
    _assert_same(tuple(np.asarray(a) for a in again), tuple(np.asarray(a) for a in forest))
