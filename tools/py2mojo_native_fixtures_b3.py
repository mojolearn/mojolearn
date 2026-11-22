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


SETS = [glue]


def _run(index):
    return SETS[index]()


def main():
    with ProcessPoolExecutor(max_workers=min(len(SETS), os.cpu_count() or 1)) as pool:
        for lines in pool.map(_run, range(len(SETS))):
            for line in lines:
                print(line)


if __name__ == "__main__":
    main()
