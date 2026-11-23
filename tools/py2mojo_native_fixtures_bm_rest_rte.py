# SPDX-License-Identifier: Apache-2.0
"""Write the inputs and reference values of lane bm-rest-rte's Mojo checks that replaced numeric Python tests.

Lane bm-rest-rte (2026-11-23; python/mojolearn/tests/test_x_trees_sanity.py and test_meta_members_native.py). Same
storage as `tools/py2mojo_native_fixtures.py` (raw little-endian words; matrices row-major); every array is the one
the original test built (the three-class `_clf()` and the `_reg()` splits are already in `trees_sk/`, written by
`tools/py2mojo_native_fixtures_b3.py`) and every reference is the LightGBM call it made.

    checks/native_tests/fixtures/trees_rte/  trees_boost_quality.mojo  (test_dart, test_dart_options: the two-class
                                             `_clf(n_classes=2)` split and LightGBM's five held-out scores)
                                             trees_entry_dart.mojo     (the repeat fits' X and y)
                                             xtrees_shap_additive.mojo (the explained DART fit's X and y)

Regenerate and repack:

    python tools/py2mojo_native_fixtures_bm_rest_rte.py && python3 tools/fixture_packs.py pack checks/native_tests/fixtures

The sets are independent and are written one per process across the host cores (`os.cpu_count()` workers).
"""
from concurrent.futures import ProcessPoolExecutor
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from py2mojo_native_fixtures import _put  # noqa: E402
from py2mojo_native_fixtures_b3 import _clf, _reg  # noqa: E402

FAMILY = "trees_rte"


def dart_scores():
    # test_dart and test_dart_options: LightGBM's held-out R^2 / accuracy of its DART fits, in the original's order
    import lightgbm
    from sklearn.metrics import accuracy_score, r2_score
    scores = []
    Xa, Xb, ya, yb = _reg()
    ref = lightgbm.LGBMRegressor(boosting_type="dart", n_estimators=60, learning_rate=0.1, verbose=-1).fit(Xa, ya)
    scores.append(r2_score(yb, ref.predict(Xb)))
    Ca, Cb, ca, cb = _clf(n_classes=2)
    ref = lightgbm.LGBMClassifier(boosting_type="dart", n_estimators=60, verbose=-1).fit(Ca, ca)
    scores.append(accuracy_score(cb, ref.predict(Cb)))
    Ta, Tb, ta, tb = _clf(n_classes=3)
    kw = dict(n_estimators=60, reg_alpha=0.5, max_delta_step=0.7, colsample_bytree=0.8, subsample=0.8,
              subsample_freq=1)
    ref = lightgbm.LGBMClassifier(boosting_type="dart", verbose=-1, **kw).fit(Ta, ta)
    scores.append(accuracy_score(tb, ref.predict(Tb)))
    kw = dict(n_estimators=60, reg_alpha=2.0, colsample_bytree=0.6, subsample=0.7, subsample_freq=2)
    ref = lightgbm.LGBMRegressor(boosting_type="dart", verbose=-1, **kw).fit(Xa, ya)
    scores.append(r2_score(yb, ref.predict(Xb)))
    kw = dict(n_estimators=60, max_delta_step=40.0)
    ref = lightgbm.LGBMRegressor(boosting_type="dart", verbose=-1, **kw).fit(Xa, ya)
    scores.append(r2_score(yb, ref.predict(Xb)))
    return [_put(FAMILY, "dart-scores.f64", scores, np.float64)]


def binary_split():
    # test_dart: `_clf(n_classes=2)`
    Ca, Cb, ca, cb = _clf(n_classes=2)
    return [_put(FAMILY, "clf2-xa.f32", Ca, np.float32), _put(FAMILY, "clf2-xb.f32", Cb, np.float32),
            _put(FAMILY, "clf2-ya.i32", ca, np.int32), _put(FAMILY, "clf2-yb.i32", cb, np.int32)]


def repeat_data():
    # test_dart_and_adaboost_fits_repeat_bit_for_bit
    rng = np.random.default_rng(1)
    X = rng.random((120, 4), dtype=np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1]).astype(np.float32)
    return [_put(FAMILY, "rep-x.f32", X, np.float32), _put(FAMILY, "rep-y.f32", y, np.float32)]


def shap_data():
    # test_dart_tree_explainer_is_additive
    rng = np.random.default_rng(2)
    X = rng.random((150, 4), dtype=np.float32)
    y = (X[:, 0] - X[:, 2]).astype(np.float32)
    return [_put(FAMILY, "shap-x.f32", X, np.float32), _put(FAMILY, "shap-y.f32", y, np.float32)]


SETS = [dart_scores, binary_split, repeat_data, shap_data]


def _run(index):
    return SETS[index]()


def main():
    with ProcessPoolExecutor(max_workers=min(len(SETS), os.cpu_count() or 1)) as pool:
        for lines in pool.map(_run, range(len(SETS))):
            for line in lines:
                print(line)


if __name__ == "__main__":
    main()
