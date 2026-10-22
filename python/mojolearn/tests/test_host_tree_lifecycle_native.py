# SPDX-License-Identifier: Apache-2.0
"""Native tree construction and parsing against frozen pre-migration oracles."""
import ast
from pathlib import Path

import numpy as np
import pytest

from mojolearn import _forest_host as forest, _gbdt_host as gbdt
from mojolearn._buffer import frombytes
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests._legacy_host_trees import LegacyHostForest, LegacyHostGBDT, parse_model_text as legacy_parse
from mojolearn.tests.test_host_archive_native import _state
from mojolearn.tests.test_host_model_native_router import GBDT_CASES, FLOAT_TEXT
from mojolearn.tests.test_host_tree_native import _forest_options, _queries


def _refused(audit, exception=ValueError):
    assert isinstance(audit.exception, exception), audit.report()
    report = audit.report()
    assert not any(report[key] for key in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads")), report


@pytest.mark.parametrize("estimator", forest._ESTIMATORS)
@pytest.mark.parametrize("engine", ["sequential", "parallel_groves"])
def test_forest_constructor_keeps_exact_state_and_predictions(tmp_path, estimator, engine):
    options = _forest_options(tmp_path, estimator, engine)
    expected = LegacyHostForest(**options)
    actual = audit_native_call(forest.HostForest, **options).assert_native()
    assert _state(vars(actual)) == _state(vars(expected))
    assert _state(actual.predict(_queries("float32"))) == _state(expected.predict(_queries("float32")))


@pytest.mark.parametrize("labels", [[2, 7], [.5, 2.5], [False, True], ["left", "right"], [2**70, 2**70 + 1]])
@pytest.mark.parametrize("container", ["list", "tuple", "numpy"])
def test_constructor_preserves_native_class_sequence_semantics(tmp_path, labels, container):
    options = _forest_options(tmp_path, "ExtraTreesClassifier")
    options["classes"] = np.asarray(labels) if container == "numpy" else tuple(labels) if container == "tuple" else labels
    expected = LegacyHostForest(**options)
    actual = audit_native_call(forest.HostForest, **options).assert_native()
    assert _state(vars(actual)) == _state(vars(expected))
    assert _state(audit_native_call(actual.predict, _queries("float32")).assert_native()) == _state(expected.predict(_queries("float32")))


@pytest.mark.parametrize("case", GBDT_CASES)
@pytest.mark.parametrize("estimator", gbdt.GBDT_ESTIMATORS)
def test_gbdt_constructor_and_parser_preserve_every_saved_layout(case, estimator):
    text, features, queries, _ = GBDT_CASES[case]
    options = dict(loss="RMSE", text=text, n_features_in=features, approx_dim=1,
                   n_classes=-1, numeric_mode=None, bias=None, estimator=estimator)
    expected = LegacyHostGBDT(**options)
    actual = audit_native_call(gbdt.HostGBDT, **options).assert_native()
    assert _state(vars(actual)) == _state(vars(expected))
    assert _state(audit_native_call(gbdt.parse_model_text, text).assert_native()) == _state(legacy_parse(text))
    assert _state(actual.predict(queries)) == _state(expected.predict(queries))


@pytest.mark.parametrize("case", GBDT_CASES)
def test_ctr_helpers_are_one_complete_native_operation(case):
    text, features, queries, _ = GBDT_CASES[case]
    options = dict(loss="RMSE", text=text, n_features_in=features, approx_dim=1,
                   n_classes=None, estimator="GradientBoosting")
    expected = LegacyHostGBDT(**options)
    assert audit_native_call(gbdt.gbdt_ctr_model_dim, text).assert_native() == 1
    got = audit_native_call(gbdt.gbdt_ctr_predict, queries, **options).assert_native()
    assert _state(got) == _state(expected.predict(queries))


@pytest.mark.parametrize("text,features,dim,bias", [
    (FLOAT_TEXT.split("tree 0 depth")[0].replace("trees 1", "trees 0"), 1, 1, 0.),
    (FLOAT_TEXT.replace("losses 0\n", "losses 0\nbias -0.5/bfe0000000000000\n"), 1, 1, -.5),
    (FLOAT_TEXT.replace("depth 1 dim 1", "depth 1 dim 2") + "leaf 0 2 0.5/3f000000\nleaf 0 3 1/3f800000\n", 1, 2, 0.),
])
def test_parser_preserves_empty_ensemble_bias_and_multioutput(text, features, dim, bias):
    options = dict(loss="RMSE", text=text, n_features_in=str(features), approx_dim=np.int64(dim),
                   n_classes="-3", numeric_mode="custom-retained-value", bias=str(bias))
    expected = LegacyHostGBDT(**options)
    actual = audit_native_call(gbdt.HostGBDT, **options).assert_native()
    assert _state(vars(actual)) == _state(vars(expected))
    assert _state(audit_native_call(gbdt.parse_model_text, text).assert_native()) == _state(legacy_parse(text))


@pytest.mark.parametrize("change", [
    {"estimator": "Unknown"}, {"inference_engine": "unknown"}, {"n_features": True},
    {"n_trees": 0}, {"num_outputs": 0}, {"classes": [0]}, {"classes": None},
])
def test_invalid_forest_metadata_is_refused_without_python(tmp_path, change):
    options = _forest_options(tmp_path, "RandomForestClassifier") | change
    with pytest.raises(ValueError):
        LegacyHostForest(**options)
    _refused(audit_native_call(forest.HostForest, **options))


@pytest.mark.parametrize("field,dtype,shape", [("offsets", "<f4", (2,)), ("colid", "<i4", (1, 3)),
                                               ("leaves", "<f4", (1,))])
def test_invalid_forest_buffers_are_refused_without_python(tmp_path, field, dtype, shape):
    options = _forest_options(tmp_path, "RandomForestClassifier")
    options[field] = frombytes(np.zeros(shape, dtype=dtype).tobytes(), dtype, shape)
    with pytest.raises(ValueError):
        LegacyHostForest(**options)
    _refused(audit_native_call(forest.HostForest, **options))


@pytest.mark.parametrize("change", [{"estimator": "Unknown"}, {"estimator": "OrderedRMSE", "loss": "Logloss"},
    {"n_features_in": 2}, {"approx_dim": 2}, {"bias": 1.}, {"text": "corrupt model"}])
def test_invalid_gbdt_metadata_is_refused_without_python(change):
    options = dict(loss="RMSE", text=FLOAT_TEXT, n_features_in=1, approx_dim=1) | change
    with pytest.raises(ValueError):
        LegacyHostGBDT(**options)
    _refused(audit_native_call(gbdt.HostGBDT, **options))


@pytest.mark.parametrize("field", ["text", "n_features_in", "bias", "n_classes"])
def test_constructor_never_invokes_python_conversion_hooks(field):
    touched = []
    class Hook:
        def __str__(self):
            touched.append("str")
            return FLOAT_TEXT
        def __int__(self):
            touched.append("int")
            return 1
        def __float__(self):
            touched.append("float")
            return 0.
    options = dict(loss="RMSE", text=FLOAT_TEXT, n_features_in=1, approx_dim=1)
    options[field] = Hook()
    _refused(audit_native_call(gbdt.HostGBDT, **options), TypeError)
    assert touched == []


def test_forest_never_invokes_class_label_iterators(tmp_path):
    touched = []
    class Labels:
        def __iter__(self):
            touched.append("iter")
            return iter([0, 1])
        def __len__(self):
            touched.append("len")
            return 2
    options = _forest_options(tmp_path, "RandomForestClassifier")
    options["classes"] = Labels()
    _refused(audit_native_call(forest.HostForest, **options), TypeError)
    assert touched == []


def test_groves_reinitialization_releases_previous_native_snapshot(tmp_path):
    options = _forest_options(tmp_path, "ExtraTreesRegressor", "parallel_groves")
    actual = forest.HostForest(**options)
    actual.predict(_queries("float32"))
    handle = vars(actual)["_groves"][0]
    audit_native_call(actual.__init__, **options).assert_native()
    assert vars(actual)["_groves"] is None
    x = frombytes(np.zeros((1, 2), dtype="<f4").tobytes(), "<f4", (1, 2))
    out = frombytes(np.zeros(1, dtype="<f4").tobytes(), "<f4", (1,))
    with pytest.raises(Exception, match="unknown or released host grove"):
        forest._load().forest_host_groves_predict(handle, x._addr, out._addr, [1, 2, 1], 1)
    assert _state(actual.predict(_queries("float32"))) == _state(LegacyHostForest(**options).predict(_queries("float32")))


def test_constructor_parser_and_ctr_wrappers_are_single_native_calls():
    for module, names in ((forest, {"__init__"}),
                          (gbdt, {"__init__", "parse_model_text", "gbdt_ctr_model_dim", "gbdt_ctr_predict"})):
        for method in ast.walk(ast.parse(Path(module.__file__).read_text())):
            if not isinstance(method, ast.FunctionDef) or method.name not in names:
                continue
            body = method.body[1:] if ast.get_docstring(method) is not None else method.body
            assert len(body) == 1 and isinstance(body[0], ast.Return), method.name
            assert sum(isinstance(node, ast.Call) for node in ast.walk(body[0])) == 1, method.name


def test_constructor_refuses_dictionary_descriptor_hooks(tmp_path):
    touched = []
    class CustomForest(forest.HostForest):
        @property
        def __dict__(self):
            touched.append("dict")
            return {}
    options = _forest_options(tmp_path, "RandomForestRegressor")
    _refused(audit_native_call(CustomForest, **options), TypeError)
    assert touched == []


def test_constructor_refuses_dictionary_key_hooks_before_comparing(tmp_path):
    touched = []
    class Key(str):
        __hash__ = str.__hash__
        def __eq__(self, other):
            touched.append("eq")
            return super().__eq__(other)
    actual = object.__new__(forest.HostForest)
    # An existing shared-key instance dictionary may compare against the
    # class's interned "estimator" key during fixture insertion. A fresh
    # dictionary installs the hostile key without invoking it in setup.
    object.__setattr__(actual, "__dict__", {Key("estimator"): "RandomForestRegressor"})
    options = _forest_options(tmp_path, "RandomForestRegressor")
    _refused(audit_native_call(actual.__init__, **options), TypeError)
    assert touched == []
