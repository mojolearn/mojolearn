# SPDX-License-Identifier: Apache-2.0
"""Write the inputs and reference values of the batch 3 Mojo checks that replaced numeric Python tests.

Lane py2mojo-C (2026-11-22, docs/plans/PY2MOJO_CONVERSION.md batch 3: python/mojolearn/tests/test_x_trees_sanity.py).
Same storage as `tools/py2mojo_native_fixtures.py` (raw little-endian words: `.f32`, `.i32`, `.f64`; matrices
row-major); every array is the one the original test built (same seed, draw order and dtype) and every reference is
the scikit-learn call it made. The originating test is named beside each set.

    checks/native_tests/fixtures/trees_glue/  trees_entry_glue.mojo  (test_stacking_folds_match_sklearn and the raw
                                              Platt / isotonic arm of test_calibration)
    checks/native_tests/fixtures/trees_sk/    trees_sklearn_quality.mojo, trees_shap.mojo  (make_classification /
                                              make_regression splits of `_clf` / `_reg` and the scikit-learn scores)

Generated with numpy 2.5.5 and scikit-learn 1.9.1 (2026-11-22). Regenerate and repack:

    python tools/py2mojo_native_fixtures_b3.py && python3 tools/fixture_packs.py pack checks/native_tests/fixtures

The sets are independent and are written one per process across the host cores (`os.cpu_count()` workers).
"""
from concurrent.futures import ProcessPoolExecutor
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from py2mojo_native_fixtures import _put  # noqa: E402


def glue():
    from sklearn.calibration import _sigmoid_calibration
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import KFold, StratifiedKFold
    # test_stacking_folds_match_sklearn: the codes and each fold's test rows (stratified: sorted), fold sizes first
    y = np.random.RandomState(0).randint(0, 4, size=103)
    strat = [np.sort(te) for _, te in StratifiedKFold(5).split(np.zeros((103, 1)), y)]
    plain = [te for _, te in KFold(4).split(np.zeros((103, 1)))]
    out = [_put("trees_glue", "folds-codes.i32", y, np.int32),
           _put("trees_glue", "folds-strat-sizes.i32", [len(t) for t in strat], np.int32),
           _put("trees_glue", "folds-strat-rows.i32", np.concatenate(strat), np.int32),
           _put("trees_glue", "folds-kfold-sizes.i32", [len(t) for t in plain], np.int32),
           _put("trees_glue", "folds-kfold-rows.i32", np.concatenate(plain), np.int32)]
    # test_calibration, the raw-binding arm
    rs = np.random.RandomState(0)
    f = rs.normal(size=400)
    yc = (f + rs.normal(size=400) > 0).astype(np.int32)
    a_ref, b_ref = _sigmoid_calibration(f, yc)
    t = np.linspace(-4, 4, 97)
    iso = IsotonicRegression(out_of_bounds="clip").fit(f, yc.astype(np.float64)).predict(t)
    out += [_put("trees_glue", "cal-f.f64", f, np.float64),
            _put("trees_glue", "cal-y.i32", yc, np.int32),
            _put("trees_glue", "cal-ab.f64", [a_ref, b_ref], np.float64),
            _put("trees_glue", "cal-t.f64", t, np.float64),
            _put("trees_glue", "cal-iso.f64", iso, np.float64)]
    return out


def _clf(n_classes=3, seed=0):
    # test_x_trees_sanity.py::_clf
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    X, y = make_classification(n_samples=1500, n_features=10, n_informative=6, n_classes=n_classes,
                               random_state=seed)
    return train_test_split(X.astype(np.float32), y, test_size=0.3, random_state=seed)


def _reg(seed=0):
    # test_x_trees_sanity.py::_reg
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    X, y = make_regression(n_samples=1500, n_features=10, n_informative=6, noise=5.0, random_state=seed)
    return train_test_split(X.astype(np.float32), y.astype(np.float32), test_size=0.3, random_state=seed)


def sk():
    from sklearn.metrics import accuracy_score, r2_score
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    Xa, Xb, ya, yb = _clf()
    Ra, Rb, ra, rb = _reg()
    # test_decision_tree_classifier / test_decision_tree_regressor: scikit-learn's held-out scores
    acc = accuracy_score(yb, DecisionTreeClassifier(max_depth=6, random_state=0).fit(Xa, ya).predict(Xb))
    r2 = r2_score(rb, DecisionTreeRegressor(max_depth=8, random_state=0).fit(Ra, ra).predict(Rb))
    return [_put("trees_sk", "clf-xa.f32", Xa, np.float32), _put("trees_sk", "clf-xb.f32", Xb, np.float32),
            _put("trees_sk", "clf-ya.i32", ya, np.int32), _put("trees_sk", "clf-yb.i32", yb, np.int32),
            _put("trees_sk", "reg-xa.f32", Ra, np.float32), _put("trees_sk", "reg-xb.f32", Rb, np.float32),
            _put("trees_sk", "reg-ya.f32", ra, np.float32), _put("trees_sk", "reg-yb.f32", rb, np.float32),
            _put("trees_sk", "dt-scores.f64", [acc, r2], np.float64)]


SETS = [glue, sk]


def _run(index):
    return SETS[index]()


def main():
    with ProcessPoolExecutor(max_workers=min(len(SETS), os.cpu_count() or 1)) as pool:
        for lines in pool.map(_run, range(len(SETS))):
            for line in lines:
                print(line)


if __name__ == "__main__":
    main()
