# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Weighted accuracy and R2 and the estimators' score(sample_weight)
(lane/cpu-training-small-gaps, 2026-09-15).

Refusal checks always run. The definitions are scikit-learn's in Float32 on
the pinned-sum path; their comparison with the Float64 definition (to a
Float32 tolerance, not bitwise) is native:
checks/native_tests/metrics_weighted_score_{cpu,gpu}.mojo. The forests'
weighted score is test_forest_protocol.py; the gradient-boosting adapters'
is below and needs a built trees binding.
"""

import numpy as np
import pytest

from mojolearn import metrics


@pytest.mark.parametrize("weights, match", [
    ([1.0, -1.0, 1.0], "nonnegative"),
    ([1.0, float("nan"), 1.0], "nonnegative"),
    ([0.0, 0.0, 0.0], "positive total"),
    ([1.0, 1.0], "entries for 3"),
    ([[1.0], [1.0], [1.0]], "1-D"),
])
def test_weight_refusals_before_the_binding(weights, match):
    with pytest.raises(ValueError, match=match):
        metrics.accuracy_score([0, 1, 1], [0, 1, 0], sample_weight=weights)
    with pytest.raises(ValueError, match=match):
        metrics.r2_score(np.float32([0, 1, 2]), np.float32([0, 1, 1]), sample_weight=weights)


def test_normalize_false_sums_correct_sample_weights():
    # Exact dyadic weights distinguish weighted count, unweighted count,
    # and normalized accuracy without a large fixture or tolerance.
    true, pred, weights = [0, 1, 1], [0, 1, 0], [0.5, 2.0, 4.0]
    assert metrics.accuracy_score(true, pred, normalize=False, sample_weight=weights) == 2.5
    assert metrics.accuracy_score(true, true, normalize=False, sample_weight=weights) == 6.5


def test_gradient_boosting_adapters_score_takes_sample_weight():
    # score(X, y, sample_weight) is scikit-learn's weighted metric of predict.
    import mojolearn
    rng = np.random.default_rng(3)
    X = rng.standard_normal((64, 3)).astype(np.float32)
    yc = (X[:, 0] > 0).astype(np.int64)
    yr = (X[:, 0] + 2.0 * X[:, 1]).astype(np.float32)
    w = rng.uniform(0.25, 2.0, 64).astype(np.float32)
    try:
        clf = mojolearn.GradientBoostingClassifier(n_estimators=3, max_depth=2, learning_rate=0.3).fit(X, yc)
        reg = mojolearn.GradientBoostingRegressor(n_estimators=3, max_depth=2, learning_rate=0.3).fit(X, yr)
    except (ImportError, NotImplementedError, RuntimeError) as exc:
        pytest.skip(f"no trees binding fits gradient boosting here: {type(exc).__name__}: {exc}")
    expected = metrics.accuracy_score(yc, clf.predict(X), sample_weight=w)
    assert clf.score(X, yc, sample_weight=w) == pytest.approx(expected, rel=1e-6, abs=1e-6)
    expected = metrics.r2_score(yr, np.asarray(reg.predict(X), dtype=np.float32), sample_weight=w)
    assert reg.score(X, yr, sample_weight=w) == pytest.approx(expected, rel=1e-6, abs=1e-6)
