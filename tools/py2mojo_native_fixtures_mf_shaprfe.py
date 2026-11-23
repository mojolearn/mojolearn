# SPDX-License-Identifier: Apache-2.0
"""Write the inputs and reference values of the RFE-over-LDA Mojo check (lane mf-shaprfe, 2026-11-23):
checks/native_tests/prep_feature_selection.mojo `test_rfe_lda`.

The references are the CURRENT DOOR's answers: `mojolearn.RFE(LinearDiscriminantAnalysis(solver=...), ...)`
fitted through the installed package (the own-door path at the base commit, origin/lane/member-fits 4d76a9e16e;
the member path at the lane's head must give the same words). Run it against a build of the BASE commit so the
references are the old path's; the Mojo check then holds the new path to them. Class labels are stored as their
codes (the position in the sorted classes, what the door's label encoding hands the member).

    checks/native_tests/fixtures/prep_sel_mf/   prep_feature_selection.mojo test_rfe_lda

Regenerate and repack (the repository's pixi `test` environment, a build of the base commit installed):

    .pixi/envs/test/bin/python tools/py2mojo_native_fixtures_mf_shaprfe.py && \\
        python3 tools/fixture_packs.py pack checks/native_tests/fixtures

The configurations are independent and are written one per process across the host cores (`os.cpu_count()`
workers).
"""
from concurrent.futures import ProcessPoolExecutor
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from py2mojo_native_fixtures import _put  # noqa: E402

FAMILY = "prep_sel_mf"

#: (name, data set, solver, step, n_features_to_select); the Mojo check's table `RFE_LDA_CASES` in the same order.
CASES = (
    ("svd-s1-none", "k3", "svd", 1, None),
    ("svd-s2-two", "k3", "svd", 2, 2),
    ("lsqr-s1-two", "k3", "lsqr", 1, 2),
    ("lsqr-s2-none", "k3", "lsqr", 2, None),
    ("svd-s1-three-binary", "k2", "svd", 1, 3),
)


def _codes(y):
    return np.unique(y, return_inverse=True)[1].astype(np.int32)


def _data(which):
    # n = 300 rows, d = 8 features; three informative columns of different strength, the rest noise.
    rng = np.random.default_rng(11 if which == "k3" else 12)
    n, d = 300, 8
    K = 3 if which == "k3" else 2
    y = rng.integers(0, K, n)
    shift = np.array([0.0, 1.5, 0.0, 0.8, 0.0, 0.3, 0.0, 0.0])
    X = (rng.standard_normal((n, d)) + y[:, None] * shift).astype(np.float32)
    return X, y


def data_sets():
    out = []
    for which in ("k3", "k2"):  # glue: two fixed data sets
        X, y = _data(which)
        out.append(_put(FAMILY, "lda-" + which + "-x.f32", X, np.float32))
        out.append(_put(FAMILY, "lda-" + which + "-y.i32", _codes(y), np.int32))
    return out


def _case(index):
    import mojolearn as ml
    name, which, solver, step, nsel = CASES[index]
    X, y = _data(which)
    r = ml.RFE(ml.LinearDiscriminantAnalysis(solver=solver), n_features_to_select=nsel, step=step).fit(X, y)
    return [
        _put(FAMILY, "rfe-" + name + "-rank.i32", np.asarray(r.ranking_), np.int32),
        _put(FAMILY, "rfe-" + name + "-sup.i32", np.asarray(r.support_, dtype=np.int32), np.int32),
    ]


def _run(index):
    if index == 0:
        return data_sets()
    return _case(index - 1)


def main():
    units = 1 + len(CASES)
    with ProcessPoolExecutor(max_workers=min(units, os.cpu_count() or 1)) as pool:
        for lines in pool.map(_run, range(units)):
            for line in lines:
                print(line)


if __name__ == "__main__":
    main()
