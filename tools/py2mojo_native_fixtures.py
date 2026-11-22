# SPDX-License-Identifier: Apache-2.0
"""Write the inputs and scikit-learn reference values of the Mojo checks that replaced numeric Python tests.

Lane py2mojo-pilot (2026-11-22, docs/plans/PY2MOJO_CONVERSION.md). The original tests draw their inputs from NumPy's
PCG64 streams (`np.random.default_rng(seed)`) and compare against scikit-learn at run time. Mojo can neither
regenerate those streams nor call scikit-learn, so both are stored as raw little-endian words: `.f32` (float32),
`.i32` (int32) and `.f64` (binary64, the reference values scikit-learn returns). Matrices are row-major. Every array is
the one the original test built (same seed, draw order and dtype) and every reference is the scikit-learn call the
original test made; the originating test is named beside each set.

    checks/native_tests/fixtures/prep_sanity/  prep_scalers_sklearn.mojo  (test_x_prep_scalers.py,
                                               test_x_prep_normalizer.py, test_x_prep_binarizer.py)
    checks/native_tests/fixtures/metrics_fm/   metrics_fowlkes_mallows.mojo  (test_fowlkes_mallows_score.py)

Generated with numpy 2.5.5 and scikit-learn 1.9.1 (2026-11-22). Regenerate and repack:

    python tools/py2mojo_native_fixtures.py && python3 tools/fixture_packs.py pack checks/native_tests/fixtures

The sets are independent and are written one per process across the host cores (`os.cpu_count()` workers).
"""
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "checks" / "native_tests" / "fixtures"

# test_x_prep_scalers.py::test_robust_scaler, the five keyword sets in order
ROBUST_KWARGS = ({}, dict(quantile_range=(10.0, 90.0)), dict(with_centering=False),
                 dict(unit_variance=True), dict(unit_variance=True, quantile_range=(5.0, 80.0)))
# test_fowlkes_mallows_score.py::test_matches_sklearn_bitwise_on_small_cases, the seven pairs in order
FM_SMALL = (([0, 0, 1, 1], [0, 0, 1, 1]), ([0, 0, 1, 1], [1, 1, 0, 0]), ([0, 0, 0, 0], [0, 1, 2, 3]),
            ([0, 0, 0, 0], [0, 0, 0, 0]), ([5], [9]), ([0, 1], [0, 0]), ([-3, 7, 7, 100000, -3], [2, 2, 2, 1, 1]))


def _put(family, name, array, dtype):
    (ROOT / family).mkdir(parents=True, exist_ok=True)
    data = np.ascontiguousarray(np.asarray(array, dtype=dtype)).ravel(order="C")
    (ROOT / family / name).write_bytes(data.astype(np.dtype(dtype).newbyteorder("<")).tobytes())
    return f"{family}/{name} {np.asarray(array).shape}"


def _scaler_data(seed=0, n=301, d=6):
    # test_x_prep_scalers.py::_data
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32) * np.float32(3)
    X[:, 1] = np.round(X[:, 1])
    X[:, 2] = np.float32(2.5)
    X[:, 3] = np.float32(0)
    X[::7, 4] = np.nan
    return X


def robust():
    import sklearn.preprocessing as sk
    X = _scaler_data()
    out = [_put("prep_sanity", "scalers-x0.f32", X, np.float32)]
    for k, kw in enumerate(ROBUST_KWARGS):
        r = sk.RobustScaler(**kw).fit(X.astype(np.float64))
        center = r.center_ if r.center_ is not None else np.zeros(X.shape[1])
        out.append(_put("prep_sanity", f"robust-{k}-center.f64", center, np.float64))
        out.append(_put("prep_sanity", f"robust-{k}-scale.f64", r.scale_, np.float64))
        out.append(_put("prep_sanity", f"robust-{k}-transform.f64", r.transform(X.astype(np.float64)), np.float64))
    return out


def maxabs():
    import sklearn.preprocessing as sk
    # test_x_prep_scalers.py::test_maxabs_scaler
    X = _scaler_data(1)
    r = sk.MaxAbsScaler().fit(X.astype(np.float64))
    return [_put("prep_sanity", "scalers-x1.f32", X, np.float32),
            _put("prep_sanity", "maxabs-scale.f64", r.scale_, np.float64),
            _put("prep_sanity", "maxabs-transform.f64", r.transform(X.astype(np.float64)), np.float64)]


def normalizer():
    from sklearn.preprocessing import Normalizer
    # test_x_prep_normalizer.py::test_normalizer
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 7)).astype(np.float32)
    X[3] = 0
    out = [_put("prep_sanity", "normalizer-x.f32", X, np.float32)]
    for norm in ("l1", "l2", "max"):
        want = Normalizer(norm=norm).fit(X).transform(X.astype(np.float64))
        out.append(_put("prep_sanity", f"normalizer-{norm}.f64", want, np.float64))
    return out


def binarizer():
    from sklearn.preprocessing import Binarizer
    # test_x_prep_binarizer.py::test_binarizer (exact; scikit-learn keeps the float32 dtype)
    rng = np.random.default_rng(0)
    X = np.round(rng.standard_normal((60, 5)), 1).astype(np.float32)
    out = [_put("prep_sanity", "binarizer-x.f32", X, np.float32)]
    for k, th in enumerate((0.0, 0.3, -1.0)):
        want = Binarizer(threshold=th).fit(X).transform(X)
        assert want.dtype == np.float32
        out.append(_put("prep_sanity", f"binarizer-{k}.f32", want, np.float32))
    return out


def fowlkes():
    import sklearn.metrics as skm
    # test_fowlkes_mallows_score.py::test_matches_sklearn_bitwise_on_small_cases (references only: the inputs are
    # literals in the Mojo check) and ::test_matches_sklearn_bitwise_on_a_random_pair
    small = [skm.fowlkes_mallows_score(t, p) for t, p in FM_SMALL]
    rng = np.random.default_rng(3)
    true = rng.integers(0, 6, 5000).astype(np.int32)
    pred = rng.integers(-2, 9, 5000).astype(np.int32)
    return [_put("metrics_fm", "fm-small-want.f64", small, np.float64),
            _put("metrics_fm", "fm-rand-yt.i32", true, np.int32),
            _put("metrics_fm", "fm-rand-yp.i32", pred, np.int32),
            _put("metrics_fm", "fm-rand-want.f64", [skm.fowlkes_mallows_score(true, pred)], np.float64)]


SETS = (robust, maxabs, normalizer, binarizer, fowlkes)


def _run(index):
    return SETS[index]()


def main():
    with ProcessPoolExecutor(max_workers=min(len(SETS), os.cpu_count() or 1)) as pool:
        for lines in pool.map(_run, range(len(SETS))):
            for line in lines:
                print(line)


if __name__ == "__main__":
    main()
