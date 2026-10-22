# SPDX-License-Identifier: Apache-2.0
"""Whole-operation CPU tree predictions against frozen independent oracles."""
import ast
from pathlib import Path

import numpy as np
import pytest

from mojolearn import _forest_host as forest, _gbdt_host as gbdt
from mojolearn._buffer import frombytes
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests._legacy_host_trees import LegacyHostForest, LegacyHostGBDT
from mojolearn.tests.test_host_archive_native import _state
from mojolearn.tests.test_host_model_native_router import _forest_archive, GBDT_CASES, FLOAT_TEXT


def _forest_pair(tmp_path, estimator, engine="sequential", labels=None):
    classifier = forest._ESTIMATORS[estimator][1]
    # One actual split and two distinct leaves exercise both traversals and
    # class selection, rather than only the constant-output leaf case.
    changes = dict(offsets=np.asarray([0, 3], dtype="<i4"), colid=np.asarray([0, -1, -1], dtype="<i4"),
        quesval=np.asarray([.5, 0, 0], dtype="<f4"), left_child=np.asarray([1, -1, -1], dtype="<i4"),
        leaves=np.asarray([0, 0, .75, .25, .25, .75] if classifier else [0, -.5, .75], dtype="<f4"))
    if labels is not None:
        changes["classes"] = np.asarray(labels)
    _, members = _forest_archive(tmp_path, estimator, engine, changes)
    arrays = {name: frombytes(members[name].tobytes(), dtype, members[name].shape)
              for name, dtype in forest._MODEL_ARRAYS}
    options = dict(estimator=estimator, n_features=2, n_trees=1, num_outputs=2 if classifier else 1,
        classes=list(labels) if labels is not None else members["classes"].tolist() if classifier else None,
        device="fixture", inference_engine=engine, **arrays)
    return forest.HostForest(**options), LegacyHostForest(**options)


def _queries(kind, columns=2):
    data = np.arange(6 * columns, dtype="<f8").reshape(6, columns) / 8
    data[0, 0] = -.25
    return _layout(data, kind)


def _layout(data, kind):
    if kind == "list":
        return data.tolist()
    if kind == "fortran":
        return np.asfortranarray(data)
    if kind == "strided":
        return np.repeat(data, 2, axis=0)[::2]
    if kind == "big_endian":
        return data.astype(">f8")
    return data.astype("<f4")


@pytest.mark.parametrize("estimator", forest._ESTIMATORS)
@pytest.mark.parametrize("engine", ["sequential", "parallel_groves"])
@pytest.mark.parametrize("kind", ["float32", "list", "fortran", "strided", "big_endian"])
def test_forest_prediction_matches_existing_bits_without_python(tmp_path, estimator, engine, kind):
    actual, expected = _forest_pair(tmp_path, estimator, engine)
    x = _queries(kind)
    reference = expected.predict(x)
    assert _state(audit_native_call(actual.predict, x).assert_native()) == _state(reference)
    assert _state(audit_native_call(actual.predict, x).assert_native()) == _state(reference)
    if actual.is_classifier:
        assert _state(audit_native_call(actual.predict_proba, x).assert_native()) == _state(expected.predict_proba(x))
        assert _state(audit_native_call(actual._predict_with_proba, x).assert_native()) == _state(expected._predict_with_proba(x))


@pytest.mark.parametrize("labels", [[2, 7], [.5, 2.5], [False, True], ["left", "right"], [2**70, 2**70 + 1]])
@pytest.mark.parametrize("estimator", ["RandomForestClassifier", "ExtraTreesClassifier"])
def test_native_forest_label_type_and_large_integer_semantics(tmp_path, labels, estimator):
    actual, expected = _forest_pair(tmp_path, estimator, labels=labels)
    x = _queries("float32")
    assert _state(audit_native_call(actual.predict, x).assert_native()) == _state(expected.predict(x))


@pytest.mark.parametrize("case", GBDT_CASES)
@pytest.mark.parametrize("kind", ["float32", "list", "fortran", "strided", "big_endian"])
def test_gbdt_prediction_and_ctr_layout_match_without_python(case, kind):
    text, features, queries, _ = GBDT_CASES[case]
    options = dict(loss="RMSE", text=text, n_features_in=features, approx_dim=1,
                   numeric_mode="identical", bias=0.)
    actual, expected = gbdt.HostGBDT(**options), LegacyHostGBDT(**options)
    x = _layout(np.asarray(queries, dtype="<f8"), kind)
    assert _state(audit_native_call(actual.predict, x).assert_native()) == _state(expected.predict(x))


@pytest.mark.parametrize("loss", ["Logloss", "CrossEntropy"])
def test_gbdt_probability_output_retains_float64_bits(loss):
    options = dict(loss=loss, text=FLOAT_TEXT, n_features_in=1, approx_dim=1)
    actual, expected = gbdt.HostGBDT(**options), LegacyHostGBDT(**options)
    x = [[0.], [.5], [1.]]
    got = audit_native_call(actual.predict_proba, x).assert_native()
    assert got.dtype == "<f8"
    assert _state(got) == _state(expected.predict_proba(x))


@pytest.mark.parametrize("nan_mode", ["as_false", "as_true"])
def test_gbdt_nan_and_signed_zero_input_bits_match_the_existing_path(nan_mode):
    options = dict(loss="RMSE", text=FLOAT_TEXT.replace("nan as_is", "nan " + nan_mode), n_features_in=1, approx_dim=1)
    actual, expected = gbdt.HostGBDT(**options), LegacyHostGBDT(**options)
    words = np.asarray([0x7fc01234, 0x80000000, 0x00000001, 0x3f800000], dtype="<u4")
    x = frombytes(words.tobytes(), "<f4", (4, 1))
    assert _state(audit_native_call(actual.predict, x).assert_native()) == _state(expected.predict(x))


@pytest.mark.parametrize("loss,error", [("RMSE", ValueError), ("MultiClass", NotImplementedError),
                                        ("MultiClassOneVsAll", NotImplementedError)])
def test_gbdt_probability_refusals_stay_inside_native_boundary(loss, error):
    actual = gbdt.HostGBDT(loss=loss, text=FLOAT_TEXT, n_features_in=1, approx_dim=1)
    report = audit_native_call(actual.predict_proba, [[0.]])
    assert isinstance(report.exception, error), report.report()
    _assert_no_callbacks(report)


def _assert_no_callbacks(audit):
    report = audit.report()
    assert not any(report[key] for key in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads")), report


@pytest.mark.parametrize("kind", ["forest", "gbdt"])
def test_tree_input_protocols_are_refused_before_hooks(tmp_path, kind):
    calls = []
    class CallbackArray:
        def __array__(self, *args, **kwargs):
            calls.append("array")
            return np.ones((2, 2), dtype="<f4")
        def __iter__(self):
            calls.append("iter")
            return iter([[0., 1.]])
    actual = _forest_pair(tmp_path, "RandomForestRegressor")[0] if kind == "forest" else gbdt.HostGBDT(
        loss="RMSE", text=FLOAT_TEXT, n_features_in=1, approx_dim=1)
    audit = audit_native_call(actual.predict, CallbackArray())
    assert isinstance(audit.exception, TypeError), audit.report()
    _assert_no_callbacks(audit)
    assert calls == []


def test_forest_grove_state_hash_properties_and_actual_native_release(tmp_path):
    actual, expected = _forest_pair(tmp_path, "ExtraTreesRegressor", "parallel_groves")
    assert audit_native_call(actual.model_sha256).assert_native() == expected.model_sha256()
    assert audit_native_call(type(actual).n_trees.fget, actual).assert_native() == 1
    assert audit_native_call(type(actual).inference_engine.fget, actual).assert_native() == "parallel_groves"
    audit_native_call(actual.predict, _queries("float32")).assert_native()
    handle = vars(actual)["_groves"][0]
    state = audit_native_call(actual.__getstate__).assert_native()
    assert state["_groves"] is None and "_binding" not in state
    restored = object.__new__(type(actual))
    audit_native_call(restored.__setstate__, state).assert_native()
    assert _state(audit_native_call(restored.predict, _queries("float32")).assert_native()) == _state(expected.predict(_queries("float32")))
    audit_native_call(type(actual).__del__, actual).assert_native()
    assert vars(actual)["_groves"] is None
    native = forest._load()
    x = frombytes(np.zeros((1, 2), dtype="<f4").tobytes(), "<f4", (1, 2))
    out = frombytes(np.zeros(1, dtype="<f4").tobytes(), "<f4", (1,))
    with pytest.raises(Exception, match="unknown or released host grove"):
        native.forest_host_groves_predict(handle, x._addr, out._addr, [1, 2, 1], 1)


def test_gbdt_hash_and_properties_are_native():
    options = dict(loss="RMSE", text=FLOAT_TEXT, n_features_in=1, approx_dim=1)
    actual, expected = gbdt.HostGBDT(**options), LegacyHostGBDT(**options)
    assert audit_native_call(actual.model_sha256).assert_native() == expected.model_sha256()
    assert audit_native_call(type(actual).n_trees.fget, actual).assert_native() == 1
    assert audit_native_call(type(actual).inference_engine.fget, actual).assert_native() == "sequential"


def test_tree_prediction_and_state_methods_are_one_native_call():
    for module, clsname, names in ((forest, "HostForest", {"predict", "predict_proba", "_predict_with_proba", "_vote",
            "_groves_handle", "__getstate__", "__setstate__", "model_sha256", "n_trees", "inference_engine"}),
            (gbdt, "HostGBDT", {"predict", "predict_proba", "model_sha256", "n_trees", "inference_engine"})):
        cls = next(n for n in ast.parse(Path(module.__file__).read_text()).body if isinstance(n, ast.ClassDef) and n.name == clsname)
        for method in cls.body:
            if not isinstance(method, ast.FunctionDef) or method.name not in names:
                continue
            body = method.body[1:] if ast.get_docstring(method) is not None else method.body
            assert len(body) == 1 and isinstance(body[0], ast.Return), method.name
            assert sum(isinstance(n, ast.Call) for n in ast.walk(body[0])) == 1, method.name
