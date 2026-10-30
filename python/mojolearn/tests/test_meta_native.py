# SPDX-License-Identifier: Apache-2.0
"""The meta-estimators on the real public path (one native call per method;
small inputs): a member that is not a mojolearn estimator is refused by name
during native admission at fit, and mojolearn members fit and predict."""
import numpy as np
import pytest
import mojolearn as ml
from mojolearn._verify_native_runtime import audit_native_call


class _OtherEstimator:
    """A scikit-learn-style estimator that is not a mojolearn class."""
    __module__ = "user_code"
    calls = []

    def get_params(self, deep=True):
        self.calls.append('get_params')
        return {}

    def set_params(self, **params):
        self.calls.append('set_params')
        return self

    def fit(self, X, y, sample_weight=None):
        self.calls.append('fit')
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        self.calls.append('predict')
        return np.zeros(len(X), dtype=np.int64)

    def predict_proba(self, X):
        self.calls.append('predict_proba')
        return np.full((len(X), 2), 0.5)

    def decision_function(self, X):
        self.calls.append('decision_function')
        return np.zeros(len(X))


def _data(classifier, outputs=1):
    rng = np.random.default_rng(0)
    X = rng.random((40, 4), dtype=np.float32)
    if classifier:
        y = (X[:, 0] > 0.5).astype(np.int32)
    else:
        y = X[:, 0].astype(np.float32)
    if outputs > 1:
        y = np.stack([y] * outputs, axis=1)
    return X, y


def _refusing():
    other = _OtherEstimator()
    dt_c, dt_r = ml.DecisionTreeClassifier(max_depth=2), ml.DecisionTreeRegressor(max_depth=2)
    return [
        (ml.VotingClassifier([("a", dt_c), ("b", other)]), True, 1),
        (ml.VotingRegressor([("a", dt_r), ("b", other)]), False, 1),
        (ml.StackingClassifier([("a", other)], cv=2), True, 1),
        (ml.StackingClassifier([("a", dt_c)], final_estimator=other, cv=2), True, 1),
        (ml.StackingRegressor([("a", other)], cv=2), False, 1),
        (ml.MultiOutputRegressor(other), False, 2),
        (ml.MultiOutputClassifier(other), True, 2),
        (ml.OneVsRestClassifier(other), True, 1),
        (ml.CalibratedClassifierCV(other, cv=2), True, 1),
    ]


@pytest.mark.parametrize("case", range(9))
def test_non_mojolearn_member_refused(case):
    model, classifier, outputs = _refusing()[case]
    X, y = _data(classifier, outputs)
    _OtherEstimator.calls.clear()
    observed = audit_native_call(model.fit, X, y)
    # Native state admission rejects user objects before clone/get_params/fit;
    # the older Python-level NotImplementedError is no longer this boundary.
    assert type(observed.exception) is TypeError
    assert str(observed.exception) == "Sequential requires native input types; Python-defined protocols and subclasses are not supported"
    report = observed.report()
    assert not any(report[key] for key in (
        "python_frames", "forbidden_native_calls", "unresolved_native_calls",
        "wrapper_work", "unobserved_existing_threads")), report
    assert _OtherEstimator.calls == []


def test_mojolearn_members_fit_and_predict():
    X, yc = _data(True)
    _, yr = _data(False)
    dt_c, dt_r = ml.DecisionTreeClassifier(max_depth=2), ml.DecisionTreeRegressor(max_depth=2)
    v = ml.VotingClassifier([("a", dt_c), ("b", ml.DecisionTreeClassifier(max_depth=3))], voting="soft").fit(X, yc)
    assert v.predict(X).shape == (40,)
    assert v.predict_proba(X).shape == (40, 2)
    assert v.transform(X).shape == (40, 4)
    assert 0.0 <= v.score(X, yc) <= 1.0
    s = ml.StackingRegressor([("a", dt_r)], cv=2).fit(X, yr)
    assert s.predict(X).shape == (40,)
    assert s.transform(X).shape == (40, 1)
    o = ml.OneVsRestClassifier(dt_c).fit(X, yc)
    assert o.predict_proba(X).shape == (40, 2)
    c = ml.CalibratedClassifierCV(dt_c, cv=2).fit(X, yc)
    assert c.predict_proba(X).shape == (40, 2)
    assert len(c.calibrated_classifiers_) == 2
    m = ml.MultiOutputClassifier(dt_c).fit(X, np.stack([yc, 1 - yc], axis=1))
    assert m.predict(X).shape == (40, 2)
    assert 0.0 <= m.score(X, np.stack([yc, 1 - yc], axis=1)) <= 1.0


def test_unfitted_and_parameter_refusals_keep_their_wording():
    X, yc = _data(True)
    # the package's fitted check answers first, with scikit-learn's NotFittedError
    from mojolearn.exceptions import NotFittedError
    with pytest.raises(NotFittedError, match="not fitted"):
        ml.OneVsRestClassifier(ml.DecisionTreeClassifier()).predict(X)
    with pytest.raises(ValueError, match="voting must be 'hard' or 'soft'"):
        ml.VotingClassifier([("a", ml.DecisionTreeClassifier())], voting="x").fit(X, yc)
    with pytest.raises(ValueError, match="estimator names must be unique"):
        ml.VotingClassifier([("a", ml.DecisionTreeClassifier()), ("a", ml.DecisionTreeClassifier())]).fit(X, yc)
    with pytest.raises(NotImplementedError, match="n_jobs"):
        ml.MultiOutputRegressor(ml.Ridge(), n_jobs=2).fit(X, np.stack([yc, yc], 1).astype(np.float32))
    with pytest.raises(ValueError, match="method must be 'sigmoid' or 'isotonic'"):
        ml.CalibratedClassifierCV(ml.DecisionTreeClassifier(), method="x").fit(X, yc)


def test_host_predict_matches_host_model(tmp_path):
    X, yr = _data(False)
    path = str(tmp_path / "rf.npz")
    ml.RandomForestRegressor(n_estimators=2, random_state=0).fit(X, yr).save(path)
    assert ml.host_predict(path, X).tobytes() == ml.host_model(path).predict(X).tobytes()
