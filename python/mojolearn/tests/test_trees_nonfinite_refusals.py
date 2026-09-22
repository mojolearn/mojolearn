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


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
