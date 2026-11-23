# SPDX-License-Identifier: Apache-2.0
"""The inputs and scikit-learn reference values of
checks/native_tests/prep_iterative_imputer.mojo (lane bm-rest-iterimp,
2026-11-23), ported from python/mojolearn/tests/test_x_prep_iterative.py
`test_iterative` and the built-in-estimator parts of `test_iterative_options`,
as raw little-endian words (`_put` of tools/py2mojo_native_fixtures.py: `.f32`
float32, `.f64` binary64, row-major). Every input is the array the original
test built (`_data`, same seed, draw order and dtype) and every reference is
the oracle call the original test made (sklearn IterativeImputer on the
float64 copy).

    checks/native_tests/fixtures/prep_iterimp/
        x<s>.f32                  _data(s), s = 0, 1, 2, 3 (400 x 5, 15% NaN)
        it-<k>-ft.f64 / -tr.f64   test_iterative keyword set k: fit_transform(X0), transform(X1)
        it-<k>-niter.f64          its n_iter_
        ind-ft.f64 / -tr.f64      add_indicator=True: fit_transform(X2), transform(X3) (400 x (5 + m))
        ind-width.f64, ind-niter.f64
        rnd-<k>-b.f64 / -b2.f64   the random / n_nearest_features sets k on X2: random_state and random_state + 1
        post-<k>-stats.f64        the sample_posterior sets k on X2 over seeds 0..11: [B.mean(), B.std(0).mean()]

Regenerate and repack (owed: not run by the lane):

    python tools/py2mojo_native_fixtures_bm_rest_iterimp.py && \
        python3 tools/fixture_packs.py pack checks/native_tests/fixtures/prep_iterimp

The sets are independent and are written one per process across the host
cores (`os.cpu_count()` workers).
"""
from concurrent.futures import ProcessPoolExecutor
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from py2mojo_native_fixtures import _put  # noqa: E402

FAM = "prep_iterimp"
F32 = np.float32
F64 = np.float64

ITERATIVE = (dict(), dict(max_iter=3, imputation_order="descending", initial_strategy="median"),
             dict(min_value=-1.0, max_value=1.0))
RANDOM = (dict(imputation_order="random", random_state=3), dict(n_nearest_features=3, random_state=3),
          dict(n_nearest_features=2, imputation_order="random", random_state=5, max_iter=20))
POSTERIOR = (dict(sample_posterior=True), dict(sample_posterior=True, min_value=-0.5, max_value=0.5))


def _data(seed=0, n=400, d=5):
    # test_x_prep_iterative.py::_data
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, 2))
    X = (Z @ rng.standard_normal((2, d)) + 0.1 * rng.standard_normal((n, d))).astype(np.float32)
    X[rng.random((n, d)) < 0.15] = np.nan
    return X


def _sk(**kw):
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    return IterativeImputer(**kw)


def inputs():
    return [_put(FAM, f"x{s}.f32", _data(s), F32) for s in range(4)]


def iterative(k):
    X, Xh = _data(0), _data(1)
    r = _sk(**ITERATIVE[k])
    out = [_put(FAM, f"it-{k}-ft.f64", r.fit_transform(X.astype(F64)), F64)]
    out.append(_put(FAM, f"it-{k}-niter.f64", [r.n_iter_], F64))
    out.append(_put(FAM, f"it-{k}-tr.f64", r.transform(Xh.astype(F64)), F64))
    return out


def indicator():
    X, Xh = _data(2), _data(3)
    r = _sk(add_indicator=True)
    a = r.fit_transform(X.astype(F64))
    return [_put(FAM, "ind-ft.f64", a, F64), _put(FAM, "ind-width.f64", [a.shape[1]], F64),
            _put(FAM, "ind-niter.f64", [r.n_iter_], F64),
            _put(FAM, "ind-tr.f64", r.transform(Xh.astype(F64)), F64)]


def random(k):
    X = _data(2).astype(F64)
    kw = RANDOM[k]
    b = _sk(**kw).fit_transform(X)
    b2 = _sk(**dict(kw, random_state=kw["random_state"] + 1)).fit_transform(X)
    return [_put(FAM, f"rnd-{k}-b.f64", b, F64), _put(FAM, f"rnd-{k}-b2.f64", b2, F64)]


def posterior(k):
    X = _data(2)
    miss = np.isnan(X)
    B = np.array([_sk(random_state=s, **POSTERIOR[k]).fit_transform(X.astype(F64))[miss] for s in range(12)])
    return [_put(FAM, f"post-{k}-stats.f64", [B.mean(), B.std(0).mean()], F64)]


def _job(item):
    name, arg = item
    fn = globals()[name]
    return fn() if arg is None else fn(arg)


def main():
    jobs = [("inputs", None), ("indicator", None)]
    jobs += [("iterative", k) for k in range(len(ITERATIVE))]
    jobs += [("random", k) for k in range(len(RANDOM))]
    jobs += [("posterior", k) for k in range(len(POSTERIOR))]
    with ProcessPoolExecutor(max_workers=min(len(jobs), os.cpu_count() or 1)) as pool:
        for lines in pool.map(_job, jobs):
            for line in lines:
                print(line)


if __name__ == "__main__":
    main()
