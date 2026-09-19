# SPDX-License-Identifier: Apache-2.0
"""Regression tests for single-pass linear-model archive loading."""

import numpy as np

from mojolearn import Ridge
from mojolearn import linear_model


def test_ridge_load_reads_archive_once_and_restores_exact_state(tmp_path, monkeypatch):
    model = Ridge(alpha=0.25, fit_intercept=True)
    model.coef_ = linear_model.Array.from_list([1.0, -2.0, 3.0], "<f4")
    model.intercept_ = -0.125
    model.n_features_in_ = 3
    path = tmp_path / "ridge.npz"
    model.save(path)

    original = linear_model._serialize.read_npz
    calls = []

    def counted(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(linear_model._serialize, "read_npz", counted)
    restored = Ridge.load(path)
    assert len(calls) == 1
    assert restored.alpha == model.alpha
    assert restored.fit_intercept is model.fit_intercept
    assert restored.intercept_ == model.intercept_
    assert restored.n_features_in_ == model.n_features_in_
    assert np.array_equal(np.asarray(restored.coef_).view(np.uint32),
                          np.asarray(model.coef_).view(np.uint32))
