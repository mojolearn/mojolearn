# SPDX-License-Identifier: Apache-2.0
"""Write the inputs and expected words of checks/native_tests/trees_calibrated_fit.mojo (lane mf-cal, 2026-11-23).

The expected words are the CURRENT DOOR's: run this against the base build (origin/lane/member-fits 4d76a9e16e,
before the typed RUN), so that the Mojo test compares the RUN with what the door wrote. Same storage as
`tools/py2mojo_native_fixtures.py` (raw little-endian words, matrices row-major).

    checks/native_tests/fixtures/mf_cal/
        x.f32                  X, 60 x 3 float32
        y2.i32, y3.i32         binary and three-class codes (0 .. k-1)
        y3gap.i32              three-class codes whose class 2 sits only in fold 0's test rows
        <cfg>-ab.f64           per calibrator, in pair order then class column: Platt (a, b); (0, 0) for isotonic
        <cfg>-m.i32            per calibrator: the isotonic knot count (0 for Platt)
        <cfg>-knots.f64        per isotonic calibrator: knots_x[0:m] then knots_y[0:m]
    <cfg> = {logreg, linsvc}-{sigmoid, isotonic}-k{2, 3}-e{1, 0} (e: ensemble)

The cv is the explicit partition the Mojo test builds: fold f's test rows are i % 3 == f, its train rows the others,
ascending. The members: LogisticRegression(C=1.0, max_iter=100, linesearch_max_iter=20, tol=1e-4) and
LinearSVC(C=1.0) (defaults otherwise); the test's member words are those estimators' admitted words.

Regenerate and repack:

    python tools/py2mojo_native_fixtures_mf_cal.py && python3 tools/fixture_packs.py pack checks/native_tests/fixtures

The configurations are independent and are written one per process across the host cores.
"""
from concurrent.futures import ProcessPoolExecutor
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from py2mojo_native_fixtures import _put  # noqa: E402

N = 60
D = 3
FOLDS = 3


def _x():
    rng = np.random.default_rng(23)
    return rng.normal(size=(N, D)).astype(np.float32)


def _y(k):
    x = _x()
    score = x[:, 0] + 0.5 * x[:, 1] - 0.25 * x[:, 2]
    if k == 2:
        return (score > 0.0).astype(np.int32)
    return np.digitize(score, np.quantile(score, [1.0 / 3.0, 2.0 / 3.0])).astype(np.int32)


def _y3gap():
    # class 2 only at rows i % 3 == 0 (fold 0's test rows): fold 0's train rows lack it
    y = _y(2).copy()
    y[0::6] = 2
    return y.astype(np.int32)


def _cv():
    rows = np.arange(N)
    return [(rows[rows % FOLDS != f], rows[rows % FOLDS == f]) for f in range(FOLDS)]


CONFIGS = [(family, method, k, ensemble)
           for family in ("logreg", "linsvc")
           for method in ("sigmoid", "isotonic")
           for k in (2, 3)
           for ensemble in (True, False)]


def _name(family, method, k, ensemble):
    return f"{family}-{method}-k{k}-e{1 if ensemble else 0}"


def inputs():
    return [_put("mf_cal", "x.f32", _x(), np.float32), _put("mf_cal", "y2.i32", _y(2), np.int32),
            _put("mf_cal", "y3.i32", _y(3), np.int32), _put("mf_cal", "y3gap.i32", _y3gap(), np.int32)]


def config(index):
    from mojolearn.calibration import CalibratedClassifierCV
    from mojolearn.linear_model import LogisticRegression
    from mojolearn.svm import LinearSVC
    family, method, k, ensemble = CONFIGS[index]
    base = (LogisticRegression(C=1.0, max_iter=100, linesearch_max_iter=20, tol=1e-4) if family == "logreg"
            else LinearSVC(C=1.0))
    model = CalibratedClassifierCV(base, method=method, cv=_cv(), ensemble=ensemble).fit(_x(), _y(k))
    ab, ms, knots = [], [], []
    for _, cals in model.calibrated_classifiers_:
        for kind, params in cals:
            if kind == "sigmoid":
                ab += [float(params[0]), float(params[1])]
                ms.append(0)
            else:
                m = int(params[2])
                ab += [0.0, 0.0]
                ms.append(m)
                knots += list(np.asarray(params[0], dtype=np.float64)[:m])
                knots += list(np.asarray(params[1], dtype=np.float64)[:m])
    name = _name(family, method, k, ensemble)
    return [_put("mf_cal", name + "-ab.f64", ab, np.float64), _put("mf_cal", name + "-m.i32", ms, np.int32),
            _put("mf_cal", name + "-knots.f64", knots, np.float64)]


def _run(index):
    return inputs() if index == len(CONFIGS) else config(index)


def main():
    with ProcessPoolExecutor(max_workers=min(len(CONFIGS) + 1, os.cpu_count() or 1)) as pool:
        for lines in pool.map(_run, range(len(CONFIGS) + 1)):
            for line in lines:
                print(line)


if __name__ == "__main__":
    main()
