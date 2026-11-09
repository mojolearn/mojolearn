# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The public doors refuse NaN and infinity wherever the bad word sits, now
that the finiteness test is one SIMD scan split across the host's cores
(core/host_finite_scan.mojo; 0.8.66 regressions C1 and C2).

The inputs are large enough (262,144 words) to be split into several tasks,
and the bad word is placed at the first word, inside a task, on a task
boundary region and at the last word, in float32 C order (the borrowed scan),
float64 (the parallel cast), Fortran order and a row-strided view. A finite
input still fits and gives the same result as its C-order float32 copy.
Run under pytest: `cd python && python3 -m pytest mojolearn/tests/test_parallel_finite_scan.py`.
"""

import pytest

np = pytest.importorskip("numpy")

import mojolearn as ml

ROWS, COLS = 4096, 64
POSITIONS = [0, 1, ROWS * COLS // 3, ROWS * COLS // 2 - 1, ROWS * COLS // 2, ROWS * COLS - 1]


def _x(dtype=np.float32):
    rng = np.random.default_rng(7)
    return rng.standard_normal((ROWS, COLS)).astype(dtype)


def _layouts(x):
    yield "c", np.ascontiguousarray(x)
    yield "f", np.asfortranarray(x)
    yield "f64", np.ascontiguousarray(x, dtype=np.float64)
    wide = np.zeros((ROWS * 2, COLS), dtype=x.dtype)
    wide[::2] = x
    yield "strided", wide[::2]


def _fitters():
    y = np.arange(ROWS, dtype=np.float32) % 7
    return {
        "ridge": lambda X: ml.Ridge().fit(X, y),
        "kmeans": lambda X: ml.KMeans(n_clusters=3, n_init=1, random_state=0).fit(X),
        "pca": lambda X: ml.PCA(n_components=4).fit(X),
        # StandardScaler accepts NaN like scikit-learn (allow-nan); it never refused non-finite input, so it is not a finite-scan door
    }


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("position", POSITIONS)
@pytest.mark.parametrize("estimator", sorted(_fitters()))
def test_nonfinite_refused_at_every_position(estimator, position, bad):
    fit = _fitters()[estimator]
    base = _x()
    base.reshape(-1)[position] = bad
    for layout, X in _layouts(base):
        with pytest.raises(Exception, match=r"(?i)nan|inf|finite"):
            fit(X)


@pytest.mark.parametrize("estimator", ["kmeans", "pca"])
def test_nonfinite_refused_at_predict(estimator):
    X = _x()
    model = _fitters()[estimator](X)
    for position in POSITIONS:
        Q = X.copy()
        Q.reshape(-1)[position] = np.nan
        with pytest.raises(Exception, match=r"(?i)nan|inf|finite"):
            (model.predict if estimator == "kmeans" else model.transform)(Q)


def test_finite_layouts_fit_the_same():
    X = _x()
    y = np.arange(ROWS, dtype=np.float32) % 7
    ref = np.asarray(ml.Ridge().fit(X, y).coef_)
    for layout, Xl in _layouts(X):
        if layout == "f64":
            continue  # a float64 input may keep float64 fitted state; its refusals are tested above
        got = np.asarray(ml.Ridge().fit(Xl, y).coef_)
        assert np.array_equal(got.view(np.uint32), ref.view(np.uint32)), layout
