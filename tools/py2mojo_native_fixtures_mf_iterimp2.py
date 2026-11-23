# SPDX-License-Identifier: Apache-2.0
"""The inputs and reference words of lane mf-iterimp2's Mojo tests (2026-11-23), as raw little-endian words (`_put`
of tools/py2mojo_native_fixtures.py: `.f32` float32, `.f64` binary64, row-major).

    checks/native_tests/fixtures/mf_iterimp2/       (checks/native_tests/prep_iterative_imputer.mojo
                                                     `test_iterative_estimator_transform`; run on the BASE build,
                                                     origin/lane/member-fits d3a8864541, CPU column: the door's
                                                     per-step transform rounds over the public fitted estimators)
        x.f32, xh.f32         _data(2), _data(3) of test_x_prep_iterative.py (400 x 5, 15% NaN)
        est-<k>-tr.f32        IterativeImputer(estimator=E_k).fit(X).transform(Xh) words, E_k = 0 Ridge(alpha=1),
                              1 LinearRegression(), 2 BayesianRidge(), every other keyword at its default

    checks/native_tests/fixtures/mf_bayes_std/      (checks/native_tests/x_linear_bayes_std.mojo; scikit-learn only,
                                                     fitted in float64 on the float32 inputs)
        bayes-x.f32 (200 x 5), bayes-y.f32, bayes-xq.f32 (50 x 5)
        ard-x.f32 (200 x 6), ard-y.f32 (reads columns 0 and 1 only), ard-xq.f32 (50 x 6)
        u<k>-dk.f64           sigma_'s side, k = 0 BayesianRidge(), 1 BayesianRidge(fit_intercept=False),
                              2 ARDRegression(), 3 ARDRegression(fit_intercept=False)
        u<k>-offset.f64       X_offset_
        u<k>-sigma.f64        sigma_ (dk x dk)
        u<k>-mean.f64, u<k>-std.f64   predict(xq, return_std=True)

Regenerate and repack (owed: not run by the lane):

    PYTHONPATH=<base build>/python python tools/py2mojo_native_fixtures_mf_iterimp2.py && \\
        python3 tools/fixture_packs.py pack checks/native_tests/fixtures/mf_iterimp2 && \\
        python3 tools/fixture_packs.py pack checks/native_tests/fixtures/mf_bayes_std

The jobs (the inputs, three estimators, four scikit-learn fits) are independent: one per process across the host
cores (`os.cpu_count()` workers, at most the job count).
"""
from concurrent.futures import ProcessPoolExecutor
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from py2mojo_native_fixtures import _put  # noqa: E402

FAM = "mf_iterimp2"
STD = "mf_bayes_std"
F32 = np.float32
F64 = np.float64


def _data(seed=0, n=400, d=5):
    # test_x_prep_iterative.py::_data
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, 2))
    X = (Z @ rng.standard_normal((2, d)) + 0.1 * rng.standard_normal((n, d))).astype(np.float32)
    X[rng.random((n, d)) < 0.15] = np.nan
    return X


def _bayes_data():
    rng = np.random.default_rng(11)
    X = rng.standard_normal((200, 5)).astype(F32) + F32(0.5)
    y = (X @ np.array([1.5, -2.0, 0.5, 0.0, 3.0], F32) + F32(0.7) + F32(0.3) * rng.standard_normal(200).astype(F32))
    xq = rng.standard_normal((50, 5)).astype(F32)
    return X, y.astype(F32), xq


def _ard_data():
    rng = np.random.default_rng(12)
    X = rng.standard_normal((200, 6)).astype(F32) - F32(0.25)
    y = (F32(3) * X[:, 0] - F32(2) * X[:, 1] + F32(0.4) + F32(0.1) * rng.standard_normal(200).astype(F32))
    xq = rng.standard_normal((50, 6)).astype(F32)
    return X, y.astype(F32), xq


def _estimator(k):
    import mojolearn as ml
    return (ml.Ridge(alpha=1.0), ml.LinearRegression(), ml.BayesianRidge())[k]


def inputs():
    out = [_put(FAM, "x.f32", _data(2), F32), _put(FAM, "xh.f32", _data(3), F32)]
    X, y, xq = _bayes_data()
    out += [_put(STD, "bayes-x.f32", X, F32), _put(STD, "bayes-y.f32", y, F32), _put(STD, "bayes-xq.f32", xq, F32)]
    X, y, xq = _ard_data()
    out += [_put(STD, "ard-x.f32", X, F32), _put(STD, "ard-y.f32", y, F32), _put(STD, "ard-xq.f32", xq, F32)]
    return out


def transform(k):
    import mojolearn as ml
    m = ml.IterativeImputer(estimator=_estimator(k)).fit(_data(2))
    return [_put(FAM, f"est-{k}-tr.f32", np.asarray(m.transform(_data(3))), F32)]


def bayes(k):
    from sklearn.linear_model import ARDRegression, BayesianRidge
    X, y, xq = _bayes_data() if k < 2 else _ard_data()
    cls = BayesianRidge if k < 2 else ARDRegression
    m = cls(fit_intercept=(k % 2 == 0)).fit(X.astype(F64), y.astype(F64))
    mean, std = m.predict(xq.astype(F64), return_std=True)
    sigma = np.asarray(m.sigma_)
    return [_put(STD, f"u{k}-dk.f64", [sigma.shape[0]], F64),
            _put(STD, f"u{k}-offset.f64", np.asarray(m.X_offset_), F64),
            _put(STD, f"u{k}-sigma.f64", sigma, F64),
            _put(STD, f"u{k}-mean.f64", mean, F64),
            _put(STD, f"u{k}-std.f64", std, F64)]


def _job(item):
    name, arg = item
    fn = globals()[name]
    return fn() if arg is None else fn(arg)


def main():
    jobs = [("inputs", None)] + [("transform", k) for k in range(3)] + [("bayes", k) for k in range(4)]
    with ProcessPoolExecutor(max_workers=min(len(jobs), os.cpu_count() or 1)) as pool:
        for lines in pool.map(_job, jobs):
            for line in lines:
                print(line)


if __name__ == "__main__":
    main()
