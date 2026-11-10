# SPDX-License-Identifier: Apache-2.0
"""LogisticRegression public packaging, writable state and archive seams.

Original fit workloads remain here to check Python classes, dtypes and shapes.
Numerical argmax/probability/accuracy oracles execute directly through Mojo QN
requests in checks/native_tests/logistic_numeric_{cpu,gpu}_suite.mojo. This
module does not claim numerical or cross-vendor verification by itself.
"""
import os
import sys
import tempfile

import numpy as np

from mojolearn import Array, linear_model
from mojolearn.linear_model import LogisticRegression


def _fixture(n=600, d=4, seed=0):
    """Three planted classes, one center per class three sigma out along
    its own axis, standard normal noise: linearly separable enough for
    training accuracy above 0.9 and never exactly tied."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, size=n).astype(np.int64)
    centers = np.zeros((3, d), dtype=np.float32)
    for c in range(3):
        centers[c, c] = 3.0
    X = (rng.standard_normal((n, d)).astype(np.float32) + centers[y]).astype(np.float32)
    return X, y


def test_three_class_fit_public_layout():
    X, y = _fixture()
    m = LogisticRegression(max_iter=100).fit(X, y)
    assert m.classes_ == [0, 1, 2]
    assert np.asarray(m.coef_).shape == (3, 4) and np.asarray(m.coef_).dtype == np.float32
    assert np.asarray(m.intercept_).shape == (3,)
    scores = np.asarray(m.decision_function(X))
    assert scores.shape == (600, 3) and scores.dtype == np.float32
    proba = np.asarray(m.predict_proba(X))
    assert proba.shape == (600, 3) and proba.dtype == np.float64
    pred = np.asarray(m.predict(X))
    assert pred.shape == (600,) and pred.dtype == np.int64
    assert int(m.retcode_) == 0
    return m, X, y


def test_tie_rule_first_maximum_wins():
    X, y = _fixture(n=64)
    m = LogisticRegression(max_iter=5).fit(X, y)
    # every logit zero: every row an exact three-way tie
    m._w = Array.from_list([0.0] * m._w.size, "<f4")
    scores = np.asarray(m.decision_function(X))
    assert scores.shape == (64, 3) and not scores.any()
    pred = np.asarray(m.predict(X))
    assert np.array_equal(pred, np.zeros(64, dtype=np.int64)), "a tie must go to classes_[0]"
    proba = np.asarray(m.predict_proba(X))
    assert np.all(proba == 1.0 / 3.0)


def test_three_classes_refuse_l1_by_name():
    X, y = _fixture(n=64)
    for penalty, kw in (("l1", {}), ("elasticnet", {"l1_ratio": 0.5})):
        try:
            LogisticRegression(penalty=penalty, max_iter=5, **kw).fit(X, y)
        except NotImplementedError as exc:
            assert "OWL-QN" in str(exc) and "3 classes" in str(exc)
        else:
            raise AssertionError(f"penalty={penalty!r} with three classes did not refuse")


def test_binary_path_keeps_its_shapes():
    X, y = _fixture(n=200)
    yb = (y == 1).astype(np.int64)
    m = LogisticRegression(max_iter=50).fit(X, yb)
    assert m.classes_ == [0, 1]
    assert np.asarray(m.coef_).shape == (1, 4)
    assert np.asarray(m.intercept_).shape == (1,)
    assert np.asarray(m.decision_function(X)).shape == (200,)
    proba = np.asarray(m.predict_proba(X))
    assert proba.shape == (200, 2)
    pred = np.asarray(m.predict(X))
    assert pred.shape == (200,) and pred.dtype == np.int64


def test_save_load_round_trip_carries_the_class_count():
    X, y = _fixture(n=300)
    m = LogisticRegression(max_iter=50).fit(X, y)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "multi.npz")
        m.save(path)
        back = LogisticRegression.load(path)
        assert back.classes_ == [0, 1, 2]
        assert np.array_equal(np.asarray(back.coef_), np.asarray(m.coef_))
        assert np.array_equal(np.asarray(back.intercept_), np.asarray(m.intercept_))
        a = np.asarray(m.predict_proba(X)).tobytes()
        b = np.asarray(back.predict_proba(X)).tobytes()
        assert a == b, "reload must give the same bytes"
        assert np.array_equal(np.asarray(back.predict(X)), np.asarray(m.predict(X)))
        # the host subclass, when this checkout carries a host binding
        # exporting the softmax link; otherwise the refusal is by name
        from mojolearn import _classical_host
        try:
            host = _classical_host.host_model(path)
            hp = np.asarray(host.predict_proba(X)).tobytes()
        except Exception as exc:  # noqa: BLE001
            print(f"  host path not checked here: {type(exc).__name__}: {exc}")
        else:
            print("  host predict_proba bytes equal the GPU bytes:", hp == a)


def test_string_labels_decode_to_a_list():
    X, y = _fixture(n=90)
    names = np.array(["cat", "dog", "emu"])
    m = LogisticRegression(max_iter=50).fit(X, [names[i] for i in y])
    assert m.classes_ == ["cat", "dog", "emu"]
    pred = m.predict(X)
    assert isinstance(pred, list) and set(pred) <= {"cat", "dog", "emu"}


def main(out=sys.stdout):
    arms = [
        test_three_class_fit_public_layout,
        test_tie_rule_first_maximum_wins,
        test_three_classes_refuse_l1_by_name,
        test_binary_path_keeps_its_shapes,
        test_save_load_round_trip_carries_the_class_count,
        test_string_labels_decode_to_a_list,
    ]
    for arm in arms:
        arm()
        out.write(f"  {arm.__name__} OK\n")
    out.write(
        "test_logistic_multiclass_surface: public boundary checks passed on %s.\n"
        "Original fit layouts, writable zero-state tie handling, named L1\n"
        "refusal, archive class count and label decoding are covered here.\n"
        "Native numerical suites and cross-vendor verification are separate.\n"
        % linear_model._backend.vendor())
    return 0


if __name__ == "__main__":
    sys.exit(main())
