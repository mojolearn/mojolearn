# SPDX-License-Identifier: Apache-2.0
"""tools/parallel_gram_check.py: equal bytes are not enough, the two-device fit must spread (plan N455). Fake
estimators and a fake binding stand in for the wheel (no device, no mojolearn import)."""
import pathlib
import sys
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import parallel_gram_check as G  # noqa: E402

np = pytest.importorskip("numpy")


class _Fake:
    """A Gram estimator whose state is fixed bytes; `skew` perturbs the coefficients of a parallel fit."""

    def __init__(self, skew=0.0, **params):
        self.skew = skew

    def fit(self, X, y=None, **kw):
        self.coef_ = np.arange(3, dtype="<f4") + np.float32(self.skew)
        self.intercept_ = 0.5
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype="<f4") + self.coef_[0]


@pytest.fixture
def binding(monkeypatch):
    """A loaded 'binding' whose device_plan_log reports the ranks the next parallel fit set."""
    state = dict(ranks=0, pending=0)

    def device_plan_log():
        got = (0, state["ranks"])
        state["ranks"] = 0
        return got

    module = types.ModuleType("_fake_plan_binding")
    module.device_plan_log = device_plan_log
    monkeypatch.setitem(sys.modules, "_fake_plan_binding", module)
    return state


def fit_on(state, ranks, skew=0.0):
    def fit_gram_estimator(est, X, y, devices=(), **kw):
        state["ranks"] = ranks
        est.skew = skew
        return est.fit(X, y, **kw)
    return fit_gram_estimator


X = np.zeros((4, 3), dtype="<f4")
Y = np.zeros(4, dtype="<f4")


def test_a_fit_that_spreads_and_matches_is_a_match(binding):
    verdict, reason, ranks, digest = G.run_case(_Fake, {}, X, Y, {}, fit_on(binding, 2))
    assert (verdict, ranks) == ("MATCH", 2) and digest


def test_equal_bytes_on_one_rank_is_a_fail_not_a_trivial_pass(binding):
    verdict, reason, ranks, _ = G.run_case(_Fake, {}, X, Y, {}, fit_on(binding, 1))
    assert verdict == "FAIL" and ranks == 1 and "spread" in reason


def test_a_byte_mismatch_is_a_differ(binding):
    verdict, reason, _, _ = G.run_case(_Fake, {}, X, Y, {}, fit_on(binding, 2, skew=1.0))
    assert verdict == "DIFFER" and "coef_" in reason


def test_no_rank_record_is_a_fail():
    assert G.case_verdict(None, None, [])[0] == "FAIL"
    verdict, reason = G.case_verdict(None, None, ["_x.device_plan_log: RuntimeError: gone"])
    assert verdict == "FAIL" and "device_plan_log raised" in reason


def test_ridge_and_linear_regression_cases_are_kept_with_and_without_weights():
    text = (ROOT / "tools" / "parallel_gram_check.py").read_text()
    body = text[text.index("cases = ["):text.index("for cls, params, target, kwargs in cases")]
    assert body.count("(LinearRegression,") == 3 and body.count("(Ridge,") == 2
    assert "dict(sample_weight=weights)" in body and "(PCA," in body and "(TruncatedSVD," in body
