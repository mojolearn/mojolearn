# SPDX-License-Identifier: Apache-2.0
"""Write the inputs and the door's reference values of lane mf-ada's Mojo checks (AdaBoost fit's typed RUN).

Lane mf-ada (2026-11-23; python/mojolearn/tests/test_meta_members_native.py
`test_dart_and_adaboost_fits_repeat_bit_for_bit`, its AdaBoost half, and the fitted state of the AdaBoost door).
Same storage as `tools/py2mojo_native_fixtures.py` (raw little-endian words; matrices row-major). The reference
values are the PUBLIC estimators' (`mojolearn.AdaBoostClassifier` / `AdaBoostRegressor`, CPU column), so run this
with the wheel or build of the BASE commit (origin/lane/member-fits 4d76a9e16e, the door before `ada_fit_run`): the
checks then compare the new RUN against the old door, word for word.

    checks/native_tests/fixtures/trees_ada/  trees_ada_fit.mojo    (every case below)
                                             trees_entry_dart.mojo (the repeat fits read ada-x / ada-y)

Cases (X = default_rng(1).random((120, 4), float32), the original test's X; y = X0 + 0.5 X1):
  csam  AdaBoostClassifier(n_estimators=25, random_state=0) on (X0 + X1 > 1): estimator_weights_, estimator_errors_,
        predict codes
  cstop AdaBoostClassifier(n_estimators=25, random_state=0) on (X0 > 0.5): the first stump is perfect, one member
  rlin / rsq / rexp  AdaBoostRegressor(n_estimators=25, random_state=0, loss=...) on y: weights, errors, predict
  The worse-than-random refusal (X all zero, y alternating) is asserted here against the door's words; the Mojo
  check expects the same words.
  Lane mf-ada2 (the member route), n_estimators=5, random_state=0:
  rfc / etc / dtr  AdaBoostClassifier over RandomForestClassifier(n_estimators=5, max_depth=2), ExtraTreesClassifier(
        n_estimators=5, max_depth=2), DecisionTreeClassifier(splitter='random', max_depth=2), each with
        class_weight='balanced', on csam's labels
  rfr / etr / dtp  AdaBoostRegressor over RandomForestRegressor(n_estimators=5, max_depth=3), ExtraTreesRegressor(
        n_estimators=5, max_depth=3), DecisionTreeRegressor(max_depth=3, criterion='poisson') on y
  The Poisson refusal on y - 0.5 is asserted here against the door's words (the Mojo check's case dtneg).
  Lane mf-oob3 (the out-of-bag bases, which the base build fits through its door rounds `_ada_door_rounds`):
  rfco  AdaBoostClassifier over RandomForestClassifier(n_estimators=5, max_depth=2, oob_score=True) on csam's labels
  rfro  AdaBoostRegressor over RandomForestRegressor(n_estimators=5, max_depth=3, oob_score=True) on y
        each also records every member's oob_decision_function_ (n x 2) / oob_prediction_ (n), float64,
        concatenated in member order (`<name>-oob.f64`), and their oob_score_ (`<name>-oobs.f64`). For these two
        the BASE is origin/lane/member-fits 3298f7f582 (the last head with the door rounds).

Regenerate and repack:

    python tools/py2mojo_native_fixtures_mf_ada.py && python3 tools/fixture_packs.py pack checks/native_tests/fixtures

The cases are independent and are written one per process across the host cores (`os.cpu_count()` workers).
"""
from concurrent.futures import ProcessPoolExecutor
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from py2mojo_native_fixtures import _put  # noqa: E402

FAMILY = "trees_ada"
WORSE = "BaseClassifier in AdaBoostClassifier ensemble is worse than random, ensemble can not be fit."


def _data():
    rng = np.random.default_rng(1)
    X = rng.random((120, 4), dtype=np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1]).astype(np.float32)
    return X, y


def inputs():
    X, y = _data()
    return [_put(FAMILY, "ada-x.f32", X, np.float32), _put(FAMILY, "ada-y.f32", y, np.float32)]


def _record(name, model, X, classifier):
    out = [_put(FAMILY, f"{name}-count.i32", [len(model.estimator_weights_)], np.int32),
           _put(FAMILY, f"{name}-alpha.f64", np.asarray(model.estimator_weights_, dtype=np.float64), np.float64),
           _put(FAMILY, f"{name}-error.f64", np.asarray(model.estimator_errors_, dtype=np.float64), np.float64)]
    p = np.asarray(model.predict(X))
    out.append(_put(FAMILY, f"{name}-pred.{'i32' if classifier else 'f32'}", p,
                    np.int32 if classifier else np.float32))
    return out


def classifier_cases():
    import mojolearn as ml
    X, _ = _data()
    out = []
    ys = (X[:, 0] + X[:, 1] > 1.0).astype(np.int64)
    out += _record("csam", ml.AdaBoostClassifier(n_estimators=25, random_state=0).fit(X, ys), X, True)
    yt = (X[:, 0] > 0.5).astype(np.int64)
    stop = ml.AdaBoostClassifier(n_estimators=25, random_state=0).fit(X, yt)
    assert len(stop.estimators_) < 25, "cstop: the first stump was expected to be perfect"
    out += _record("cstop", stop, X, True)
    try:
        ml.AdaBoostClassifier(n_estimators=3, random_state=0).fit(np.zeros((8, 2), np.float32),
                                                                   np.array([0, 1] * 4, np.int64))
    except Exception as problem:  # the door's refusal, which the Mojo check expects word for word
        assert WORSE in str(problem), str(problem)
    else:
        raise AssertionError("the worse-than-random fit did not refuse")
    return out


def regressor_case(loss):
    import mojolearn as ml
    X, y = _data()
    name = {"linear": "rlin", "square": "rsq", "exponential": "rexp"}[loss]
    return _record(name, ml.AdaBoostRegressor(n_estimators=25, random_state=0, loss=loss).fit(X, y), X, False)


POISSON = "criterion='poisson' requires y >= 0 with a positive sum"


def _record_oob(name, model, classifier):
    """Lane mf-oob3: every member's out-of-bag outputs, in member order."""
    key = "oob_decision_function_" if classifier else "oob_prediction_"
    words = np.concatenate([np.asarray(getattr(m, key), dtype=np.float64).ravel() for m in model.estimators_])
    scores = np.asarray([float(m.oob_score_) for m in model.estimators_], dtype=np.float64)
    return [_put(FAMILY, f"{name}-oob.f64", words, np.float64), _put(FAMILY, f"{name}-oobs.f64", scores, np.float64)]


def member_case(name):
    """Lane mf-ada2: the member route's bases (module docstring)."""
    import mojolearn as ml
    X, y = _data()
    if name == "rfco":
        ys = (X[:, 0] + X[:, 1] > 1.0).astype(np.int64)
        model = ml.AdaBoostClassifier(estimator=ml.RandomForestClassifier(n_estimators=5, max_depth=2, oob_score=True),
                                      n_estimators=5, random_state=0).fit(X, ys)
        return _record(name, model, X, True) + _record_oob(name, model, True)
    if name == "rfro":
        model = ml.AdaBoostRegressor(estimator=ml.RandomForestRegressor(n_estimators=5, max_depth=3, oob_score=True),
                                     n_estimators=5, random_state=0).fit(X, y)
        return _record(name, model, X, False) + _record_oob(name, model, False)
    if name in ("rfc", "etc", "dtr"):
        base = {"rfc": lambda: ml.RandomForestClassifier(n_estimators=5, max_depth=2, class_weight="balanced"),
                "etc": lambda: ml.ExtraTreesClassifier(n_estimators=5, max_depth=2, class_weight="balanced"),
                "dtr": lambda: ml.DecisionTreeClassifier(splitter="random", max_depth=2,
                                                         class_weight="balanced")}[name]()
        ys = (X[:, 0] + X[:, 1] > 1.0).astype(np.int64)
        return _record(name, ml.AdaBoostClassifier(estimator=base, n_estimators=5, random_state=0).fit(X, ys), X, True)
    if name == "dtneg":
        try:
            ml.AdaBoostRegressor(estimator=ml.DecisionTreeRegressor(max_depth=3, criterion="poisson"),
                                 n_estimators=5, random_state=0).fit(X, y - np.float32(0.5))
        except Exception as problem:  # the door's refusal, which the Mojo check expects
            assert POISSON in str(problem), str(problem)
        else:
            raise AssertionError("dtneg: the negative Poisson targets did not refuse")
        return []
    base = {"rfr": lambda: ml.RandomForestRegressor(n_estimators=5, max_depth=3),
            "etr": lambda: ml.ExtraTreesRegressor(n_estimators=5, max_depth=3),
            "dtp": lambda: ml.DecisionTreeRegressor(max_depth=3, criterion="poisson")}[name]()
    return _record(name, ml.AdaBoostRegressor(estimator=base, n_estimators=5, random_state=0).fit(X, y), X, False)


def _run(job):
    kind, arg = job
    if kind == "inputs":
        return inputs()
    if kind == "classifier":
        return classifier_cases()
    if kind == "member":
        return member_case(arg)
    return regressor_case(arg)


def main():
    jobs = [("inputs", None), ("classifier", None), ("regressor", "linear"), ("regressor", "square"),
            ("regressor", "exponential")] + [("member", name) for name in
                                             ("rfc", "etc", "dtr", "rfr", "etr", "dtp", "dtneg", "rfco", "rfro")]
    with ProcessPoolExecutor(max_workers=min(len(jobs), os.cpu_count() or 1)) as pool:
        for lines in pool.map(_run, jobs):
            for line in lines:
                print(line)


if __name__ == "__main__":
    main()
