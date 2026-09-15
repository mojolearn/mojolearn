# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""GaussianProcessRegressor(normalize_y=True) (lane/cpu-training-small-gaps,
2026-09-15).

The refusal and source checks always run; the value checks run through
whichever GP and preprocessing bindings this install loads and are skipped,
saying so, without them. The statistics are Float32 (StandardScaler's pinned
folds) where scikit-learn's reference is Float64, so values are compared to a
tolerance and repeated runs bitwise.
"""

from pathlib import Path

import numpy as np
import pytest

from mojolearn import host_surface
from mojolearn._gp_impl import RBF, ConstantKernel, GaussianProcessRegressor, WhiteKernel

ROOT = Path(__file__).resolve().parents[3]


def _data():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(96, 3)).astype(np.float32)
    y = (50.0 + 12.0 * np.sin(x[:, 0]) + 3.0 * x[:, 1]).astype(np.float32)
    q = rng.normal(size=(24, 3)).astype(np.float32)
    return x, y, q


def _fit_or_skip(x, y, normalize_y=True):
    kernel = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.1)
    try:
        from mojolearn._cpu_reference import reference_training
        with reference_training():
            return GaussianProcessRegressor(kernel=kernel, normalize_y=normalize_y).fit(x, y)
    except (ImportError, NotImplementedError) as exc:
        pytest.skip(f"no GP binding that fits on this install: {exc}")


def test_non_bool_is_refused_by_name():
    with pytest.raises(TypeError, match="normalize_y must be a bool"):
        GaussianProcessRegressor(normalize_y="yes")


def test_the_absence_row_and_the_refusal_are_gone():
    tsv = (ROOT / "gaussian_process/NOT_IMPLEMENTED.tsv").read_text()
    assert "sklearn normalize_y" not in tsv
    src = (ROOT / "python/mojolearn/_gp_impl.py").read_text()
    assert "normalize_y=True is \"\n                \"refused" not in src
    assert "gp-normalize-y" in host_surface.family("gp")["training_lanes"]


def test_normalized_prediction_repeats_bitwise_and_tracks_sklearn():
    x, y, q = _data()
    m = _fit_or_skip(x, y)
    mean, std = m.predict(q, return_std=True)
    mean2, std2 = _fit_or_skip(x, y).predict(q, return_std=True)
    assert np.asarray(mean).tobytes() == np.asarray(mean2).tobytes()
    assert np.asarray(std).tobytes() == np.asarray(std2).tobytes()
    assert abs(m._y_train_mean - float(np.mean(y.astype(np.float64)))) < 1e-3
    gp = pytest.importorskip("sklearn.gaussian_process")
    kernels = pytest.importorskip("sklearn.gaussian_process.kernels")
    k = kernels.ConstantKernel(1.0, "fixed") * kernels.RBF(1.0, "fixed") + kernels.WhiteKernel(0.1, "fixed")
    ref = gp.GaussianProcessRegressor(kernel=k, alpha=2.0 ** -20, optimizer=None, normalize_y=True).fit(
        x.astype(np.float64), y.astype(np.float64))
    rmean, rstd = ref.predict(q.astype(np.float64), return_std=True)
    np.testing.assert_allclose(np.asarray(mean, np.float64), rmean, rtol=2e-4, atol=2e-3)
    np.testing.assert_allclose(np.asarray(std, np.float64), rstd, rtol=2e-3, atol=2e-3)


def test_constant_target_scales_by_one():
    x, _, q = _data()
    y = np.full(x.shape[0], 7.0, np.float32)
    m = _fit_or_skip(x, y)
    assert m._y_train_std == 1.0 and m._y_train_mean == 7.0
    mean = np.asarray(m.predict(q))
    assert np.all(np.isfinite(mean))
