# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Non-finite inputs and unimplemented knobs in the tree family are refused
by name, with the exception type the rest of the library uses.

Found by the pip smoke of 0.8.13 (2026-09-22):

- RandomForest* and ExtraTrees* fitted a NaN or inf in X without a word. The
  builder has no missing-value arm, so the value was quantized and split on
  like any number.
- IsolationForest.fit on a NaN raised a bare `Exception` from the native fit.
- GradientBoosting.fit on a NaN label reached the searcher and surfaced as a
  bare `Exception` ("All splits have infinite score"); a NaN in the eval
  set's labels was not checked either.
- ExtraTreesClassifier(class_weight=...) raised a bare `Exception` from the
  native builder instead of NotImplementedError.

Every check here runs before any device work, so it needs the bindings to
import and nothing more. Run as a module:
`cd python && python3 -m mojolearn.tests.test_trees_nonfinite_refusals`
or under pytest.
"""

import math

import mojolearn as ml


def _rows(n=24, f=3):
    return [[float((i * 7 + j * 3) % 11) for j in range(f)] for i in range(n)]


def _with(value):
    X = _rows()
    X[5][1] = value
    return X


def _labels(n=24):
    return [i % 2 for i in range(n)]


def _raises(fn, exc_type, needle):
    try:
        fn()
    except exc_type as e:
        assert needle in str(e), f"wrong message: {e}"
        return
    except Exception as e:  # the wrong type is the defect this file closes
        raise AssertionError(f"raised {type(e).__name__}, wanted {exc_type.__name__}: {e}")
    raise AssertionError(f"accepted; wanted {exc_type.__name__}")


def test_forests_refuse_nonfinite_X():
    for bad in (math.nan, math.inf, -math.inf):
        X = _with(bad)
        _raises(lambda: ml.RandomForestClassifier(n_estimators=2).fit(X, _labels()),
                ValueError, "NaN or infinity")
        _raises(lambda: ml.RandomForestRegressor(n_estimators=2).fit(X, [float(v) for v in _labels()]),
                ValueError, "NaN or infinity")
        _raises(lambda: ml.ExtraTreesClassifier(n_estimators=2).fit(X, _labels()),
                ValueError, "NaN or infinity")
        _raises(lambda: ml.ExtraTreesRegressor(n_estimators=2).fit(X, [float(v) for v in _labels()]),
                ValueError, "NaN or infinity")


def test_isolation_forest_refuses_nonfinite_X():
    _raises(lambda: ml.IsolationForest(n_estimators=2).fit(_with(math.nan)),
            ValueError, "NaN or infinity")


def test_extra_trees_class_weight_is_not_implemented():
    _raises(lambda: ml.ExtraTreesClassifier(n_estimators=2, class_weight="balanced").fit(_rows(), _labels()),
            NotImplementedError, "class_weight is not implemented")


def test_gradient_boosting_refuses_nonfinite_labels():
    X = _rows()
    for bad in (math.nan, math.inf):
        y = [float(i) for i in range(24)]
        y[3] = bad
        _raises(lambda: ml.GradientBoosting(n_estimators=2).fit(X, y),
                ValueError, "y must be finite")
        good = [float(i) for i in range(24)]
        _raises(lambda: ml.GradientBoosting(n_estimators=2).fit(X, good, eval_set=(X, y)),
                ValueError, "eval_set y must be finite")


_NATIVE_NAN_CHILD = r"""
import math, sys
import mojolearn as ml
import mojolearn.randomforest as rfm
rfm.all_finite = lambda _x: True  # reach the native builder past the Python refusal
X = [[float((i * 7 + j * 3) % 11) for j in range(4)] for i in range(200)]
for i in range(200):
    for j in range(4):
        if X[i][j] > 6.0:
            X[i][j] = math.nan
y = [i % 2 for i in range(200)]
for est, yy in ((ml.RandomForestClassifier, y), (ml.RandomForestRegressor, [float(v) for v in y])):
    try:
        est(n_estimators=2, random_state=0).fit(X, yy)
    except Exception as e:
        if "would not terminate" not in str(e):
            sys.exit(f"{est.__name__}: wrong error {type(e).__name__}: {e}")
    else:
        sys.exit(f"{est.__name__}: the native builder accepted NaN")
X = [[float((i * 7 + j * 3) % 11) for j in range(4)] for i in range(200)]
for i in range(0, 200, 9):
    X[i][1] = math.inf
    X[i][3] = -math.inf
ml.RandomForestClassifier(n_estimators=2, random_state=0).fit(X, y)
print("ok")
"""


def test_forest_native_builder_refuses_nan():
    """The native RF builder itself (pip smoke 2026-09-22): a NaN in X made
    the fit loop forever at the default unlimited depth, a NaN binning left
    in the histogram but partitioning right. The builder now refuses it by
    name; +-inf still fits (it bins and partitions consistently). Run in a
    child with a timeout, because the defect this guards is a hang."""
    import subprocess
    import sys
    try:
        r = subprocess.run([sys.executable, "-c", _NATIVE_NAN_CHILD],
                           capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        raise AssertionError("the native forest fit on NaN did not return in 300 s")
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stdout + r.stderr


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
