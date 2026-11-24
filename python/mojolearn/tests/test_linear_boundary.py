# SPDX-License-Identifier: Apache-2.0
"""Linear models public boundary: returned shapes, parameters and the words of a refusal.

The numerical answers on each side are verify lanes (ols*, ridge*, lasso, elasticnet, logistic*); what stays here is
what a digest cannot see.
"""
import pytest


# ---------------------------------------------------------------- moved from test_admitted_call_doors.py (vdv-pytest-a g2)
# One admission record per public call (`AdmittedCall`): the fitted coefficient's shape, the parameter read back and
# the AttributeError of a missing fitted attribute. Ridge's fit and predict bits (and a second predict) are the
# ridge lane's.

def test_ridge_answers_through_the_record():
    np = pytest.importorskip("numpy")
    import mojolearn

    rng = np.random.default_rng(0)
    X = rng.standard_normal((64, 5)).astype(np.float32)
    y = (X @ np.arange(1, 6, dtype=np.float32) + 0.5).astype(np.float32)
    try:
        model = mojolearn.Ridge(alpha=1.0).fit(X, y)
    except (ImportError, NotImplementedError, RuntimeError) as exc:
        pytest.skip(f"no classical binding in this process: {type(exc).__name__}: {exc}")
    assert np.asarray(model.coef_).shape == (5,)
    assert model.get_params()["alpha"] == 1.0
    with pytest.raises(AttributeError):
        model.no_such_attribute_
