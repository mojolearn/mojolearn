# SPDX-License-Identifier: Apache-2.0
"""Write the inputs and scikit-learn reference values of the mutual-information and RFE Mojo checks (lane
bm-rest-sel, 2026-11-23; ported from python/mojolearn/tests/test_x_prep_selection.py `test_mutual_info`,
`test_mutual_info_discrete`, `test_rfe` and the RFE-getter half of `test_score_edges_and_rfe_getter`).

Every array is the one the original test built (same seed, draw order and dtype) and every reference is the
scikit-learn call the original test made, stored as raw little-endian words by `_put` of
`tools/py2mojo_native_fixtures.py`. Class labels are stored as their codes (the position in the sorted classes,
which is what the door's label encoding hands the programs).

    checks/native_tests/fixtures/prep_sel_bm/   prep_feature_selection.mojo (test_mutual_info, test_rfe, ...)

Regenerate and repack (the repository's pixi `test` environment):

    .pixi/envs/test/bin/python tools/py2mojo_native_fixtures_bm_rest_sel.py && \\
        python3 tools/fixture_packs.py pack checks/native_tests/fixtures

The four sets are independent and are written one per process across the host cores (`os.cpu_count()` workers).
"""
from concurrent.futures import ProcessPoolExecutor
import os
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from py2mojo_native_fixtures import _put  # noqa: E402

FAMILY = "prep_sel_bm"


def _codes(y):
    return np.unique(y, return_inverse=True)[1].astype(np.int32)


def mutual_info():
    # test_mutual_info
    import sklearn.feature_selection as skfs
    out = []
    rng = np.random.default_rng(5)
    n, d = 500, 4
    y = rng.integers(0, 3, n)
    X = (rng.standard_normal((n, d)) + y[:, None] * np.array([0.0, 0.3, 1.0, 2.0])).astype(np.float32)
    yr = (X[:, 2] + 0.5 * rng.standard_normal(n)).astype(np.float32)
    out.append(_put(FAMILY, "mi-x.f32", X, np.float32))
    out.append(_put(FAMILY, "mi-y.i32", _codes(y), np.int32))
    out.append(_put(FAMILY, "mi-yr.f32", yr, np.float32))
    out.append(_put(FAMILY, "mi-c.f64", skfs.mutual_info_classif(X, y, random_state=0), np.float64))
    out.append(_put(FAMILY, "mi-r.f64", skfs.mutual_info_regression(X, yr, random_state=0), np.float64))
    return out


def mutual_info_discrete():
    # test_mutual_info_discrete
    import sklearn.feature_selection as skfs
    out = []
    rng = np.random.default_rng(6)
    n = 500
    y = rng.integers(0, 3, n)
    Xd = np.stack([rng.integers(0, 4, n), (y + rng.integers(0, 2, n)) % 4, y * 2.5,
                   np.full(n, 7)], axis=1).astype(np.float32)
    Xc = (rng.standard_normal((n, 2)) + y[:, None] * np.array([0.2, 1.0])).astype(np.float32)
    X = np.concatenate([Xd[:, :2], Xc[:, :1], Xd[:, 2:], Xc[:, 1:]], axis=1)
    yr = (X[:, 1] + 0.7 * rng.standard_normal(n)).astype(np.float32)
    Xd4 = X[:, [0, 1, 3, 4]]
    out.append(_put(FAMILY, "mid-x.f32", X, np.float32))
    out.append(_put(FAMILY, "mid-y.i32", _codes(y), np.int32))
    out.append(_put(FAMILY, "mid-yr.f32", yr, np.float32))
    cases = ((Xd4, True), (X, [0, 1, 3, 4]), (X, np.array([True, True, False, True, True, False])),
             (X, [-6, 1, 3, -2]), (X, True))
    for j, (A, df) in enumerate(cases):
        out.append(_put(FAMILY, f"mid-c{j}.f64",
                        skfs.mutual_info_classif(A, y, discrete_features=df, random_state=0), np.float64))
        if j == 4:      # continuous columns as categories: the regression is refused
            continue
        out.append(_put(FAMILY, f"mid-r{j}.f64",
                        skfs.mutual_info_regression(A, yr, discrete_features=df, random_state=0), np.float64))
    # discrete_features=False: the seed means over random_state 0..9
    out.append(_put(FAMILY, "mid-tc.f64", np.mean(
        [skfs.mutual_info_classif(X, y, discrete_features=False, random_state=s) for s in range(10)], axis=0),
        np.float64))
    out.append(_put(FAMILY, "mid-tr.f64", np.mean(
        [skfs.mutual_info_regression(X, yr, discrete_features=False, random_state=s) for s in range(10)], axis=0),
        np.float64))
    return out


def rfe():
    # test_rfe
    import sklearn.feature_selection as skfs
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SkLDA
    out = []
    rng = np.random.default_rng(7)
    n, d = 400, 8
    y = rng.integers(0, 3, n)
    X = (rng.standard_normal((n, d)) + y[:, None] * np.linspace(0, 1.4, d)).astype(np.float32)
    r = skfs.RFE(SkLDA(), n_features_to_select=3, step=2).fit(X.astype(np.float64), y)
    out.append(_put(FAMILY, "rfe-x.f32", X, np.float32))
    out.append(_put(FAMILY, "rfe-y.i32", _codes(y), np.int32))
    out.append(_put(FAMILY, "rfe-rank.i32", r.ranking_, np.int32))
    out.append(_put(FAMILY, "rfe-pred.i32", _codes(r.predict(X.astype(np.float64))), np.int32))
    return out


def rfe_getter():
    # test_score_edges_and_rfe_getter, the RFE half (importance_getter='coef_'); the draws before Xg are the
    # score half's, made in the same order
    import sklearn.feature_selection as skfs
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SkLDA
    out = []
    rng = np.random.default_rng(11)
    n = 300
    y = rng.integers(0, 3, n)
    rng.standard_normal((n, 5))
    rng.standard_normal(n)
    Xg = (rng.standard_normal((n, 8)) + y[:, None] * np.linspace(0, 1.4, 8)).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = skfs.RFE(SkLDA(), n_features_to_select=3, step=2, importance_getter="coef_").fit(
            Xg.astype(np.float64), y)
    out.append(_put(FAMILY, "rg-x.f32", Xg, np.float32))
    out.append(_put(FAMILY, "rg-y.i32", _codes(y), np.int32))
    out.append(_put(FAMILY, "rg-rank.i32", r.ranking_, np.int32))
    return out


SETS = (mutual_info, mutual_info_discrete, rfe, rfe_getter)


def _run(index):
    return SETS[index]()


def main():
    with ProcessPoolExecutor(max_workers=min(len(SETS), os.cpu_count() or 1)) as pool:
        for lines in pool.map(_run, range(len(SETS))):
            for line in lines:
                print(line)


if __name__ == "__main__":
    main()
