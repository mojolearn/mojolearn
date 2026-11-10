# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of Mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Python API coverage for admitted finite/nonfinite input layouts.

The large word-position sweep lives in checks/native_tests/finite_scan.mojo
and calls the native scans directly. These small fixtures retain the Python
buffer admission, estimator fit/query refusal, and equivalent-layout behavior
that a pure Mojo test cannot exercise.
"""

import pytest

np = pytest.importorskip("numpy")

import mojolearn as ml

ROWS, COLS = 32, 8
LAYOUTS = ("c", "f", "f64", "strided")


def _x(dtype=np.float32):
    rng = np.random.default_rng(7)
    return rng.standard_normal((ROWS, COLS)).astype(dtype)


def _layout(x, layout):
    if layout == "c":
        return np.ascontiguousarray(x)
    if layout == "f":
        return np.asfortranarray(x)
    if layout == "f64":
        return np.ascontiguousarray(x, dtype=np.float64)
    wide = np.zeros((ROWS * 2, COLS), dtype=x.dtype)
    wide[::2] = x
    return wide[::2]


def _fitters():
    y = np.arange(ROWS, dtype=np.float32) % 7
    return {
        "ridge": lambda X: ml.Ridge().fit(X, y),
        "kmeans": lambda X: ml.KMeans(n_clusters=3, n_init=1, random_state=0).fit(X),
        "pca": lambda X: ml.PCA(n_components=4).fit(X),
        # StandardScaler accepts NaN like scikit-learn (allow-nan); it never refused non-finite input, so it is not a finite-scan door
    }


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("estimator", sorted(_fitters()))
def test_fit_refuses_nonfinite_admitted_layout(estimator, layout, bad):
    fit = _fitters()[estimator]
    base = _x()
    base[ROWS // 2, COLS // 2] = bad
    with pytest.raises(Exception, match=r"(?i)nan|inf|finite"):
        fit(_layout(base, layout))


@pytest.mark.parametrize("estimator", ["kmeans", "pca"])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_query_refuses_nonfinite_admitted_layout(estimator, layout):
    X = _x()
    model = _fitters()[estimator](X)
    Q = X.copy()
    Q[ROWS // 2, COLS // 2] = np.nan
    with pytest.raises(Exception, match=r"(?i)nan|inf|finite"):
        (model.predict if estimator == "kmeans" else model.transform)(_layout(Q, layout))


@pytest.mark.parametrize("layout", ["c", "f", "strided"])
def test_finite_layouts_fit_the_same(layout):
    X = _x()
    y = np.arange(ROWS, dtype=np.float32) % 7
    ref = np.asarray(ml.Ridge().fit(X, y).coef_)
    # A float64 input may preserve float64 fitted state; its admission and
    # refusals are covered above, without requiring a float32 fitted dtype.
    got = np.asarray(ml.Ridge().fit(_layout(X, layout), y).coef_)
    assert np.array_equal(got.view(np.uint32), ref.view(np.uint32)), layout
