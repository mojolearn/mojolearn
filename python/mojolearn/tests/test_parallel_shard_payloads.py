# SPDX-License-Identifier: Apache-2.0
"""What the sharded drivers send their workers, checked through the results.

Lane ne2-i-j (2026-11-07, plan rows N450-N452 of
docs/plans/NATIVE_ENTRY_POINTS_AFTER_0866.md):

* `fit_forest` sends a classifier's labels as int32 codes plus one class
  table (not one Python object per row); the joined forest must publish the
  plain fit's `classes_` and predict the plain fit's bytes, for numeric,
  string and boolean labels.
* With more tree ranges (or GPC classes) than workers, X and y are stored
  once per worker under a call token; the results must not change, and a
  second call (a new token on the same idle pool) must not see the first
  call's operands.
* `ParallelQueries.query(..., method="radius_neighbors")` gets each shard's
  CSR triple and builds the ragged rows in the parent; the rows must be the
  plain method's, with and without distances, sorted or not.

Devices come from MOJOLEARN_PAR_DEVICES (default "0"), as in
test_parallel_public_drivers.py.

    cd python && python3 -m pytest mojolearn/tests/test_parallel_shard_payloads.py
"""
import os

import numpy as np
import pytest

import mojolearn


def _devices():
    raw = os.environ.get("MOJOLEARN_PAR_DEVICES", "0")
    return tuple(int(v) for v in raw.split(",") if v.strip())


def _fixture(rows=96, features=5, seed=3):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((rows, features)).astype(np.float32)
    score = X[:, 0] + 0.5 * X[:, 1]
    return X, score


def _labels(kind, score):
    bins = np.digitize(score, [-0.5, 0.5])  # three classes
    if kind == "int":
        return (bins * 7 - 3).astype(np.int64)
    if kind == "str":
        return np.array(["low", "mid", "high"])[bins]
    if kind == "bool":
        return score > 0
    if kind == "list":
        return [int(v) for v in bins]
    raise AssertionError(kind)


@pytest.mark.parametrize("kind", ["int", "str", "bool", "list"])
@pytest.mark.parametrize("cls_name", ["RandomForestClassifier", "ExtraTreesClassifier"])
def test_forest_labels_as_codes_publish_the_plain_classes(kind, cls_name):
    from mojolearn.parallel_ensemble import fit_forest
    X, score = _fixture()
    y = _labels(kind, score)
    cls = getattr(mojolearn, cls_name)
    kw = dict(n_estimators=6, max_depth=4, random_state=5)
    # trees_per_shard=2: three tree ranges, more than the workers of a one- or two-device run
    parallel = fit_forest(cls(**kw), X, y, devices=_devices(), trees_per_shard=2)
    plain = cls(**kw).fit(X, y)
    assert list(parallel.classes_) == list(plain.classes_)
    assert type(parallel.classes_) is type(plain.classes_)
    assert parallel.n_classes_ == plain.n_classes_
    for method in ("predict", "predict_proba"):
        got = np.asarray(getattr(parallel, method)(X))
        expected = np.asarray(getattr(plain, method)(X))
        assert got.dtype == expected.dtype and got.shape == expected.shape, method
        assert got.tobytes() == expected.tobytes(), method


def test_stored_operands_belong_to_one_call():
    # two calls in a row reuse the idle pool; each stores and releases its own X and y
    from mojolearn.parallel_ensemble import fit_forest
    X, score = _fixture()
    y = _labels("int", score)
    kw = dict(n_estimators=4, max_depth=3, random_state=1)
    first = fit_forest(mojolearn.RandomForestClassifier(**kw), X, y, devices=_devices(), trees_per_shard=1)
    X2 = np.ascontiguousarray(X[::-1])
    y2 = np.ascontiguousarray(y[::-1])
    second = fit_forest(mojolearn.RandomForestClassifier(**kw), X2, y2, devices=_devices(), trees_per_shard=1)
    plain2 = mojolearn.RandomForestClassifier(**kw).fit(X2, y2)
    assert np.asarray(second.predict_proba(X)).tobytes() == np.asarray(plain2.predict_proba(X)).tobytes()
    plain1 = mojolearn.RandomForestClassifier(**kw).fit(X, y)
    assert np.asarray(first.predict_proba(X)).tobytes() == np.asarray(plain1.predict_proba(X)).tobytes()


def test_forest_regressor_targets_are_unchanged():
    from mojolearn.parallel_ensemble import fit_forest
    X, score = _fixture()
    kw = dict(n_estimators=6, max_depth=4, random_state=2)
    parallel = fit_forest(mojolearn.RandomForestRegressor(**kw), X, score, devices=_devices(), trees_per_shard=2)
    plain = mojolearn.RandomForestRegressor(**kw).fit(X, score)
    assert np.asarray(parallel.predict(X)).tobytes() == np.asarray(plain.predict(X)).tobytes()


@pytest.mark.parametrize("return_distance,sort_results", [(True, False), (True, True), (False, False)])
def test_radius_rows_are_built_from_the_shards_csr(return_distance, sort_results):
    from mojolearn.neighbors import RadiusNeighbors
    from mojolearn.parallel_neighbors import ParallelQueries
    X, _ = _fixture(rows=200, features=4, seed=9)
    m = RadiusNeighbors(radius=1.2).fit(X[:150])
    q = np.ascontiguousarray(X[150:])
    kw = dict(return_distance=return_distance, sort_results=sort_results)
    with ParallelQueries(m, devices=_devices(), rows_per_shard=8) as pq:
        got = pq.query(q, method="radius_neighbors", **kw)
    expected = m.radius_neighbors(q, **kw)
    if return_distance:
        got_d, got_i = got
        exp_d, exp_i = expected
    else:
        got_i, exp_i = got, expected
    assert len(got_i) == len(exp_i) == len(q)
    for row in range(len(q)):
        assert np.asarray(got_i[row]).tobytes() == np.asarray(exp_i[row]).tobytes(), row
        if return_distance:
            assert np.asarray(got_d[row]).tobytes() == np.asarray(exp_d[row]).tobytes(), row
