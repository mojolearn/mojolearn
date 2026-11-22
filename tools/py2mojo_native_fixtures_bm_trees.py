# SPDX-License-Identifier: Apache-2.0
"""Write the inputs and reference values of lane bm-trees' Mojo checks that replaced numeric Python tests.

Lane bm-trees (2026-11-22, python/mojolearn/tests/test_x_trees_sanity.py). Same storage as
`tools/py2mojo_native_fixtures.py` (raw little-endian words; matrices row-major); every array is the one the original
test built (the `_clf()` / `_reg()` splits are already in `trees_sk/`, written by `tools/py2mojo_native_fixtures_b3.py`)
and every reference is the scikit-learn call it made.

    checks/native_tests/fixtures/trees_bm/  trees_et_quality.mojo  (test_decision_tree_random_splitter,
                                            test_extratrees_deviance_criteria)

Regenerate and repack:

    python tools/py2mojo_native_fixtures_bm_trees.py && python3 tools/fixture_packs.py pack checks/native_tests/fixtures

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


def random_splitter():
    # test_decision_tree_random_splitter: scikit-learn's held-out accuracy and R^2
    from sklearn.metrics import accuracy_score, r2_score
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    Xa, Xb, ya, yb = _clf()
    acc = accuracy_score(yb, DecisionTreeClassifier(splitter="random", max_depth=10, random_state=0)
                         .fit(Xa, ya).predict(Xb))
    Ra, Rb, ra, rb = _reg()
    r2 = r2_score(rb, DecisionTreeRegressor(splitter="random", max_leaf_nodes=20, random_state=0)
                  .fit(Ra, ra).predict(Rb))
    return [_put("trees_bm", "dt-random-scores.f64", [acc, r2], np.float64)]


def deviance():
    # test_extratrees_deviance_criteria: the exp-transformed targets and scikit-learn's Poisson R^2
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.metrics import r2_score
    Xa, Xb, ya, yb = _reg()
    ya, yb = np.exp(ya / np.std(ya)).astype(np.float32), np.exp(yb / np.std(ya)).astype(np.float32)
    r = r2_score(yb, ExtraTreesRegressor(n_estimators=40, max_depth=10, criterion="poisson", random_state=0)
                 .fit(Xa, ya).predict(Xb))
    return [_put("trees_bm", "pos-ya.f32", ya, np.float32), _put("trees_bm", "pos-yb.f32", yb, np.float32),
            _put("trees_bm", "et-poisson-r2.f64", [r], np.float64)]


SETS = [random_splitter, deviance]


def _run(index):
    return SETS[index]()


def main():
    with ProcessPoolExecutor(max_workers=min(len(SETS), os.cpu_count() or 1)) as pool:
        for lines in pool.map(_run, range(len(SETS))):
            for line in lines:
                print(line)


if __name__ == "__main__":
    main()
