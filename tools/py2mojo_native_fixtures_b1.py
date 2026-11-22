# SPDX-License-Identifier: Apache-2.0
"""Batch 1 of docs/plans/PY2MOJO_CONVERSION.md (lane py2mojo-A, 2026-11-22): the inputs and the NumPy and scikit-learn
reference values of the prep Mojo checks that replace numeric Python tests, as raw little-endian words (`_put` of
tools/py2mojo_native_fixtures.py: `.f32` float32, `.f64` binary64, row-major). Every input is the array the original
test built (same seed, draw order and dtype) and every reference is the oracle call the original test made; the
originating test is named beside each set.

    checks/native_tests/fixtures/prep_scalers/   prep_standard_scaler.mojo (checks/standard_scaler_smoke.py),
                                                 prep_scalers.mojo (checks/minmax_scaler_smoke.py,
                                                 test_cpu_training_e2.py::test_scalers_run_on_the_host_when_built),
                                                 prep_scaler_options.mojo (test_x_prep_scaler_options.py)
    checks/native_tests/fixtures/prep_members/   prep_simple_imputer.mojo, prep_polynomial.mojo, prep_power.mojo,
                                                 prep_quantile.mojo, prep_spline.mojo (test_x_prep_imputer.py,
                                                 test_x_prep_polynomial.py, test_x_prep_power.py,
                                                 test_x_prep_quantile.py, test_x_prep_spline.py)

Generated with numpy 2.5.5 and scikit-learn 1.9.1 (2026-11-22). Regenerate and repack:

    python tools/py2mojo_native_fixtures_b1.py && python3 tools/fixture_packs.py pack checks/native_tests/fixtures

The sets are independent and are written one per process across the host cores (`os.cpu_count()` workers).
"""
from concurrent.futures import ProcessPoolExecutor
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from py2mojo_native_fixtures import _put  # noqa: E402

SC = "prep_scalers"
PM = "prep_members"
F32 = np.float32
F64 = np.float64


# ---------------------------------------------------------------- checks/standard_scaler_smoke.py
def standard_smoke():
    rng = np.random.default_rng(20260910)
    cases = [np.array([[-1, 2, 7], [0, 8, 7], [1, 18, 7]], F32), np.array([[3, -2, 0]], F32),
             rng.uniform(-8, 8, (513, 7)).astype(F32)]
    repeated = np.zeros((513, 4), dtype=F32)  # the original's np.empty: only columns 0 and 2 are read
    repeated[:, 0] = F32(.1)
    repeated[:, 2] = np.linspace(-2, 2, 513, dtype=F32)
    cases.append(np.ascontiguousarray(repeated[:, ::2]))
    out = []
    for k, x in enumerate(cases):
        mu = x.astype(F64).mean(axis=0)
        var = ((x.astype(F64) - mu) ** 2).mean(axis=0)
        out += [_put(SC, f"std-x{k}.f32", x, F32), _put(SC, f"std-mu{k}.f64", mu, F64),
                _put(SC, f"std-var{k}.f64", var, F64)]
    return out


# ---------------------------------------------------------------- checks/minmax_scaler_smoke.py
def _mm_reference(x, feature_range):
    low, high = map(F32, feature_range)
    minimum, maximum = x.min(axis=0), x.max(axis=0)
    span = maximum - minimum
    divisor = np.where(span < 10 * np.finfo(F32).eps, F32(1), span)
    scale = (high - low) / divisor
    offset = low - minimum * scale
    return minimum, maximum, span, scale, offset


def minmax_smoke():
    rng = np.random.default_rng(20260910)
    cases = [np.array([[-1, 2, 7], [0, 8, 7], [1, 18, 7]], F32), np.array([[3, -2, 0]], F32),
             rng.uniform(-8, 8, (513, 7)).astype(F32), rng.uniform(-8, 8, (7, 259)).astype(F32).T,
             np.array([[0, 0], [2**-22, 2**-18]], F32)]
    out = []
    for k, x in enumerate(cases):
        out.append(_put(SC, f"mm-x{k}.f32", x, F32))
        for b, bounds in enumerate(((0, 1), (-2, 3))):
            stats = _mm_reference(x, bounds)
            expected = x * stats[3] + stats[4]
            words = np.concatenate([np.asarray(s, F64).ravel() for s in stats] + [expected.astype(F64).ravel()])
            out.append(_put(SC, f"mm-want{k}-{b}.f64", words, F64))
    return out


# ---------------------------------------------------------------- test_cpu_training_e2.py
def e2_scalers():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((700, 5)).astype(F32)
    x[:, 4] = F32(2.5)
    return [_put(SC, "e2-x.f32", x, F32), _put(SC, "e2-mean.f64", x[:, :4].mean(0), F64)]


# ---------------------------------------------------------------- test_x_prep_scaler_options.py
def _opt_data(seed=3, n=257, d=5, nan_every=7):
    rng = np.random.default_rng(seed)
    X = (rng.normal(size=(n, d)) * [1, 10, 0.1, 3, 100] + [0, 5, -2, 0, 1000]).astype(F32)[:, :d]
    X[:, 3] = 2.5
    Xn = X.copy()
    Xn[::nan_every, 0] = np.nan
    Xn[1::nan_every + 4, 2] = np.nan
    return X, Xn


def _std_words(ref):
    # mean_, var_, scale_, n_samples_seen_ broadcast to d
    d = ref.mean_.shape[0]
    seen = np.broadcast_to(np.asarray(ref.n_samples_seen_, F64), (d,))
    return np.concatenate([ref.mean_, ref.var_, ref.scale_, seen])


def scaler_options():
    from sklearn import preprocessing as skp
    X, Xn = _opt_data()
    out = [_put(SC, "opt-x.f32", X, F32), _put(SC, "opt-xn.f32", Xn, F32)]
    # test_standard_nan_fit_and_transform
    ref = skp.StandardScaler().fit(Xn.astype(F64))
    out.append(_put(SC, "opt-nan-stats.f64", _std_words(ref), F64))
    out.append(_put(SC, "opt-nan-transform.f64", ref.transform(Xn.astype(F64)), F64))
    # test_standard_sample_weight
    w = np.random.default_rng(5).uniform(0, 3, size=X.shape[0]).astype(F32)
    w[::11] = 0
    out.append(_put(SC, "opt-w.f32", w, F32))
    for k, data in enumerate((X, Xn)):
        ref = skp.StandardScaler().fit(data.astype(F64), sample_weight=w.astype(F64))
        out.append(_put(SC, f"opt-w{k}-stats.f64", _std_words(ref), F64))
    out.append(_put(SC, "opt-unit-mean.f64", skp.StandardScaler().fit(X.astype(F64)).mean_, F64))
    # test_standard_partial_fit
    P, Pn = _opt_data(n=600)
    pw = np.random.default_rng(9).uniform(0.5, 2, size=P.shape[0]).astype(F32)
    out += [_put(SC, "opt-p.f32", P, F32), _put(SC, "opt-pn.f32", Pn, F32), _put(SC, "opt-pw.f32", pw, F32)]
    for k, (data, weight) in enumerate(((P, None), (Pn, None), (P, pw), (Pn, pw))):
        ref = skp.StandardScaler()
        for a, b in ((0, 100), (100, 350), (350, 600)):
            sw = None if weight is None else weight[a:b].astype(F64)
            ref.partial_fit(data[a:b].astype(F64), sample_weight=sw)
        out.append(_put(SC, f"opt-p{k}-stats.f64", _std_words(ref), F64))
    out.append(_put(SC, "opt-p-mean.f64", P.astype(F64).mean(0), F64))
    # test_standard_all_nan_feature
    Y, _ = _opt_data(seed=4)
    out += [_put(SC, "opt-y.f32", Y, F32),
            _put(SC, "opt-y-mean.f64", skp.StandardScaler().fit(Y.astype(F64)).mean_, F64)]
    # test_minmax_nan_and_partial_fit
    for k, clip in enumerate((False, True)):
        ref = skp.MinMaxScaler(feature_range=(-1, 2), clip=clip).fit(Xn.astype(F64))
        words = np.concatenate([ref.data_min_, ref.data_max_, ref.data_range_, ref.scale_, ref.min_])
        out.append(_put(SC, f"opt-mm{k}-stats.f64", words, F64))
        out.append(_put(SC, f"opt-mm{k}-transform.f64", ref.transform(Xn.astype(F64)), F64))
    out.append(_put(SC, "opt-mm-fill.f32", np.where(np.isnan(Xn), np.nanmin(Xn, axis=0), Xn).astype(F32), F32))
    Z = X[:20].copy()
    Z[:, 4] = np.nan
    r = skp.MinMaxScaler().fit(X[20:].astype(F64)).partial_fit(Z.astype(F64))
    assert np.isnan(r.scale_[4])
    out.append(_put(SC, "opt-mm-z.f32", Z, F32))
    return out


# ---------------------------------------------------------------- test_x_prep_imputer.py
IMPUTER_KW = (dict(strategy="mean"), dict(strategy="median"), dict(strategy="most_frequent"),
              dict(strategy="constant", fill_value=-3.0), dict(strategy="median", keep_empty_features=True),
              dict(strategy="mean", add_indicator=True))


def _imp_data(seed=0, n=203, d=5):
    rng = np.random.default_rng(seed)
    X = np.round(rng.standard_normal((n, d)) * 2).astype(F32)
    X[rng.random((n, d)) < 0.2] = np.nan
    X[:, 3] = np.nan
    return X


def imputer():
    from sklearn.impute import SimpleImputer as SkSI
    X, Xh = _imp_data(), _imp_data(1)
    out = [_put(PM, "imp-x.f32", X, F32), _put(PM, "imp-xh.f32", Xh, F32)]
    for k, kw in enumerate(IMPUTER_KW):
        r = SkSI(**kw).fit(X.astype(F64))
        t = r.transform(Xh.astype(F64))
        out += [_put(PM, f"imp-{k}-stats.f64", np.asarray(r.statistics_, F64), F64),
                _put(PM, f"imp-{k}-transform.f64", t, F64),
                _put(PM, f"imp-{k}-width.f64", [t.shape[1]], F64)]
    Xm = np.nan_to_num(_imp_data(2), nan=-1.0)
    r = SkSI(missing_values=-1.0, strategy="mean").fit(Xm.astype(F64))
    t = r.transform(Xm.astype(F64))
    out += [_put(PM, "imp-num-x.f32", Xm, F32), _put(PM, "imp-num-transform.f64", t, F64),
            _put(PM, "imp-num-width.f64", [t.shape[1]], F64)]
    return out


# ---------------------------------------------------------------- test_x_prep_polynomial.py
POLY_KW = (dict(), dict(degree=3), dict(degree=(2, 3), interaction_only=True, include_bias=False))


def polynomial():
    from sklearn.preprocessing import PolynomialFeatures as SkP
    X = np.random.default_rng(0).standard_normal((40, 4)).astype(F32)
    out = [_put(PM, "poly-x.f32", X, F32)]
    for k, kw in enumerate(POLY_KW):
        r = SkP(**kw).fit(X)
        out += [_put(PM, f"poly-{k}-powers.f64", r.powers_, F64),
                _put(PM, f"poly-{k}-width.f64", [r.powers_.shape[0]], F64),
                _put(PM, f"poly-{k}-transform.f64", r.transform(X.astype(F64)), F64)]
    return out


# ---------------------------------------------------------------- test_x_prep_power.py
POWER_KW = (dict(), dict(standardize=False), dict(method="box-cox"))


def _pw_data(seed=0, n=600):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.standard_normal(n), rng.exponential(2.0, n), rng.gamma(2.0, 1.5, n),
                            rng.lognormal(0, 0.7, n)]).astype(F32)


def power():
    from sklearn.preprocessing import PowerTransformer as SkPT
    X, Xh = _pw_data(), _pw_data(1)
    out = []
    for k, kw in enumerate(POWER_KW):
        if kw.get("method") == "box-cox":
            X, Xh = np.abs(X) + F32(0.1), np.abs(Xh) + F32(0.1)
        r = SkPT(**kw).fit(X.astype(F64))
        out += [_put(PM, f"pw-{k}-x.f32", X, F32), _put(PM, f"pw-{k}-xh.f32", Xh, F32),
                _put(PM, f"pw-{k}-lambdas.f64", r.lambdas_, F64),
                _put(PM, f"pw-{k}-transform.f64", r.transform(Xh.astype(F64)), F64)]
    return out


# ---------------------------------------------------------------- test_x_prep_quantile.py
QUANTILE_KW = (dict(n_quantiles=100), dict(n_quantiles=50, output_distribution="normal"))


def _qt_data(seed=0, n=700, d=4):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(F32)
    X[:, 1] = np.round(X[:, 1] * 2)
    X[::11, 2] = np.nan
    X[:, 3] = np.exp(X[:, 3])
    return X


def quantile():
    from sklearn.preprocessing import QuantileTransformer as SkQT
    X, Xh = _qt_data(), _qt_data(1)
    out = [_put(PM, "qt-x.f32", X, F32), _put(PM, "qt-xh.f32", Xh, F32)]
    for k, kw in enumerate(QUANTILE_KW):
        r = SkQT(**kw).fit(X.astype(F64))
        b = r.transform(Xh.astype(F64))
        normal = kw.get("output_distribution") == "normal"
        Z = b.astype(F32)
        Z[:5] = -5.2 if normal else 0.0
        Z[5:10] = 5.2 if normal else 1.0
        Z[10:15] = F32(-7.0)
        Z[15:20] = F32(7.0)
        Z[20:30] = F32(0.37)
        out += [_put(PM, f"qt-{k}-quantiles.f64", r.quantiles_, F64), _put(PM, f"qt-{k}-transform.f64", b, F64),
                _put(PM, f"qt-{k}-z.f32", Z, F32), _put(PM, f"qt-{k}-inverse.f64", r.inverse_transform(Z.astype(F64)), F64)]
    return out


# ---------------------------------------------------------------- test_x_prep_spline.py
SPLINE_KN = np.array([[-2.0, -1.5, -1.0], [-0.5, 0.0, 0.1], [0.4, 0.3, 0.8], [2.0, 1.1, 1.5]])
# (keywords, weighted, feature columns): test_spline's four, test_spline_options' eleven, the two degree <= 1
# 'linear' single-feature cases and the three handle_missing='zeros' cases, in the original's order
SPLINE_CASES = (
    (dict(), False, 3), (dict(n_knots=6, degree=2, knots="quantile"), False, 3), (dict(extrapolation="continue"), False, 3),
    (dict(include_bias=False, degree=1), False, 3),
    (dict(knots="kn"), False, 3), (dict(knots="kn", degree=2, extrapolation="linear"), False, 3),
    (dict(extrapolation="linear"), False, 3),
    (dict(extrapolation="periodic"), False, 3), (dict(extrapolation="periodic", degree=2, n_knots=4), False, 3),
    (dict(extrapolation="periodic", include_bias=False, knots="kn"), False, 3),
    (dict(extrapolation="periodic", degree=3, n_knots=4), False, 3),
    (dict(), True, 3), (dict(knots="quantile", n_knots=6), True, 3), (dict(knots="quantile", n_knots=4), False, 3),
    (dict(order="F", degree=2), False, 3),
    (dict(extrapolation="linear", degree=1), False, 1), (dict(extrapolation="linear", degree=0), False, 1),
    (dict(handle_missing="zeros"), False, 3), (dict(handle_missing="zeros", knots="quantile"), False, 3),
    (dict(handle_missing="zeros", extrapolation="periodic"), False, 3),
)


def spline():
    from sklearn.preprocessing import SplineTransformer as SkS
    rng = np.random.default_rng(0)
    X0 = rng.standard_normal((300, 3)).astype(F32)
    Xh0 = (rng.standard_normal((100, 3)) * 1.5).astype(F32)
    rng = np.random.default_rng(1)
    X1 = rng.standard_normal((300, 3)).astype(F32)
    Xh1 = (rng.standard_normal((100, 3)) * 1.7).astype(F32)
    sw = rng.integers(0, 4, 300).astype(F64)
    Xn, Xhn = X1.copy(), Xh1.copy()
    Xn[::7, 1] = np.nan
    Xhn[::5, 0] = np.nan
    out = [_put(PM, "spl-x0.f32", X0, F32), _put(PM, "spl-xh0.f32", Xh0, F32), _put(PM, "spl-x1.f32", X1, F32),
           _put(PM, "spl-xh1.f32", Xh1, F32), _put(PM, "spl-xn.f32", Xn, F32), _put(PM, "spl-xhn.f32", Xhn, F32),
           _put(PM, "spl-sw.f32", sw, F32), _put(PM, "spl-kn.f32", SPLINE_KN, F32)]
    for k, (kw, weighted, cols) in enumerate(SPLINE_CASES):
        kw = {key: (SPLINE_KN if value == "kn" else value) for key, value in kw.items()}
        if k < 4:
            X, Xh = X0, Xh0
        elif "handle_missing" in kw:
            X, Xh = Xn, Xhn
        else:
            X, Xh = X1[:, :cols], Xh1[:, :cols]
        r = SkS(**kw).fit(X.astype(F64), sample_weight=sw if weighted else None)
        t = r.transform(Xh.astype(F64))
        out += [_put(PM, f"spl-{k}-transform.f64", t, F64), _put(PM, f"spl-{k}-width.f64", [t.shape[1]], F64)]
    return out


SETS = (standard_smoke, minmax_smoke, e2_scalers, scaler_options, imputer, polynomial, power, quantile, spline)


def _run(index):
    return SETS[index]()


def main():
    with ProcessPoolExecutor(max_workers=min(len(SETS), os.cpu_count() or 1)) as pool:
        for lines in pool.map(_run, range(len(SETS))):
            for line in lines:
                print(line)


if __name__ == "__main__":
    main()
