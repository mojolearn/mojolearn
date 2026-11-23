# SPDX-License-Identifier: Apache-2.0
"""The inputs and door reference words of checks/native_tests/prep_iterative_imputer.mojo
`test_iterative_estimator` (lane mf-iterimp, 2026-11-23): IterativeImputer with a supplied estimator, as the
door's per-step estimator rounds computed it BEFORE the member route (run this on the BASE build,
origin/lane/member-fits 4d76a9e16e, CPU column), as raw little-endian words (`_put` of
tools/py2mojo_native_fixtures.py: `.f32` float32, `.f64` binary64, row-major).

    checks/native_tests/fixtures/mf_iterimp/
        x.f32                 _data(2) of test_x_prep_iterative.py (400 x 5, 15% NaN)
        est-<k>-xt.f32        fit_transform(X) words, estimator k = 0 Ridge(alpha=1), 1 LinearRegression(),
                              2 BayesianRidge(), every other keyword at its default
        est-<k>-meta.f64      [n_iter_, len(imputation_sequence_)]
        est-<k>-j.f64         each step's feat_idx
        est-<k>-coef.f32      each step's coef_ (4 words a step), as float32
        est-<k>-inter.f32     each step's intercept_, as float32

Regenerate and repack (owed: not run by the lane):

    PYTHONPATH=<base build>/python python tools/py2mojo_native_fixtures_mf_iterimp.py && \\
        python3 tools/fixture_packs.py pack checks/native_tests/fixtures/mf_iterimp

The estimators are independent fits, one per process across the host cores (`os.cpu_count()` workers, at most
the job count).
"""
from concurrent.futures import ProcessPoolExecutor
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from py2mojo_native_fixtures import _put  # noqa: E402

FAM = "mf_iterimp"
F32 = np.float32
F64 = np.float64


def _data(seed=0, n=400, d=5):
    # test_x_prep_iterative.py::_data
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, 2))
    X = (Z @ rng.standard_normal((2, d)) + 0.1 * rng.standard_normal((n, d))).astype(np.float32)
    X[rng.random((n, d)) < 0.15] = np.nan
    return X


def _estimator(k):
    import mojolearn as ml
    return (ml.Ridge(alpha=1.0), ml.LinearRegression(), ml.BayesianRidge())[k]


def inputs():
    return [_put(FAM, "x.f32", _data(2), F32)]


def estimator(k):
    import mojolearn as ml
    m = ml.IterativeImputer(estimator=_estimator(k))
    xt = m.fit_transform(_data(2))
    seq = m.imputation_sequence_
    out = [_put(FAM, f"est-{k}-xt.f32", np.asarray(xt), F32)]
    out.append(_put(FAM, f"est-{k}-meta.f64", [m.n_iter_, len(seq)], F64))
    out.append(_put(FAM, f"est-{k}-j.f64", [s[0] for s in seq], F64))
    out.append(_put(FAM, f"est-{k}-coef.f32", np.concatenate([np.ravel(np.asarray(s[2].coef_)) for s in seq]), F32))
    out.append(_put(FAM, f"est-{k}-inter.f32", [float(np.ravel(np.asarray(s[2].intercept_))[0]) for s in seq], F32))
    return out


def _job(item):
    name, arg = item
    fn = globals()[name]
    return fn() if arg is None else fn(arg)


def main():
    jobs = [("inputs", None)] + [("estimator", k) for k in range(3)]
    with ProcessPoolExecutor(max_workers=min(len(jobs), os.cpu_count() or 1)) as pool:
        for lines in pool.map(_job, jobs):
            for line in lines:
                print(line)


if __name__ == "__main__":
    main()
