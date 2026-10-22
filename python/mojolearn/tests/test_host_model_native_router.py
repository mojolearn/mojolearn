# SPDX-License-Identifier: Apache-2.0
"""Audit the public whole-archive route, preserving saved state and outputs.

Constructors and parser calls below are independent legacy test oracles.
Only the public load operation is audited; unchanged prediction methods are
called separately to verify that native reconstruction retains their state.
"""
import ast
from pathlib import Path

import numpy as np
import pytest

import mojolearn
from mojolearn import _classical_host as classical, _forest_host as forest, _gbdt_host as gbdt
from mojolearn._buffer import frombytes
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests.test_host_archive_native import CASES, _archive, _state
from mojolearn.tests.test_gbdt_host_ctr import CTR_TEXT, TENSOR_TEXT
from mojolearn.tests._legacy_host_trees import LegacyHostForest, LegacyHostGBDT


@pytest.mark.parametrize("name", CASES)
def test_public_host_model_routes_classical_archive_without_python(tmp_path, name):
    path = _archive(tmp_path, name)
    cls = getattr(classical, "Host" + name)
    expected = getattr(classical, name).load.__func__(cls, path)
    vars(expected)["estimator"] = name
    actual = audit_native_call(mojolearn.host_model, path).assert_native()
    assert type(actual) is cls
    assert _state(vars(actual)) == _state(vars(expected))


@pytest.mark.parametrize("name", ["LinearRegression", "Ridge", "StandardScaler"])
def test_public_router_preserves_existing_linear_and_scaler_loads(tmp_path, name):
    if name == "StandardScaler":
        members = dict(format="mojolearn-scaler-1", meta=np.asarray([2, 4, 1, 1], dtype="<i8"),
            mean=np.asarray([1., 2.], dtype="<f4"), var=np.asarray([4., 9.], dtype="<f4"),
            scale=np.asarray([2., 3.], dtype="<f4"))
    else:
        members = dict(format="mojolearn-linear-1", meta=np.asarray([2, 1], dtype="<i8"),
            coef=np.asarray([.25, .75], dtype="<f4"), intercept=np.asarray([.5], dtype="<f8"))
        if name == "Ridge":
            members["alpha"] = np.asarray([1.25], dtype="<f8")
    path = tmp_path / (name + ".npz")
    np.savez(path, **members, estimator=name, numeric_mode="identical")
    cls = getattr(classical, "Host" + name)
    expected = cls.load(path)
    vars(expected)["estimator"] = name
    actual = audit_native_call(mojolearn.host_model, path).assert_native()
    assert type(actual) is cls
    assert _state(vars(actual)) == _state(vars(expected))
    if name == "StandardScaler":
        np.testing.assert_array_equal(actual.mean_.tolist(), [1., 2.])
        assert actual.n_samples_seen_ == 4
    else:
        np.testing.assert_array_equal(actual.coef_.tolist(), [.25, .75])
        assert actual.intercept_ == .5


def test_public_router_preserves_second_arima_format(tmp_path):
    path = _archive(tmp_path, "ARIMA", override=dict(format="mojolearn-arima-2",
        meta=np.asarray([1, 0, 0, 0, 0, 0, 0, 1, 2, 4, 30, 1, 1], dtype="<i8"),
        exog=np.ones((2, 4, 1), dtype="<f4"), params=np.ones((2, 4), dtype="<f4"),
        x=np.ones((2, 4), dtype="<f4"), x0=np.ones((2, 4), dtype="<f4")))
    expected = classical.ARIMA.load.__func__(classical.HostARIMA, path)
    vars(expected)["estimator"] = "ARIMA"
    actual = audit_native_call(mojolearn.host_model, path).assert_native()
    assert _state(vars(actual)) == _state(vars(expected))


def test_public_router_fixture_matrix_covers_every_registered_archive_entry():
    covered = {(fmt, name) for name, (fmt, _) in CASES.items()}
    covered.update({("mojolearn-linear-1", "LinearRegression"), ("mojolearn-linear-1", "Ridge"),
                    ("mojolearn-scaler-1", "StandardScaler"), ("mojolearn-arima-2", "ARIMA")})
    covered.update((fmt + suffix, name) for fmt, family in forest._FORMATS.items()
                   for name, info in forest._ESTIMATORS.items() if family == info[0]
                   for suffix in ("", forest._GROVES_SUFFIX))
    covered.update((gbdt.GBDT_FORMAT, name) for name in gbdt.GBDT_ESTIMATORS)
    connected = classical._CLASSICAL["host_archive_context"]["host_archive_formats"]
    assert covered == {(fmt, name) for fmt, classes in connected.items() for name in classes}


def _forest_archive(tmp_path, estimator, engine="sequential", override=None):
    family, classifier = forest._ESTIMATORS[estimator]
    fmt = next(fmt for fmt, value in forest._FORMATS.items() if value == family)
    if engine == "parallel_groves":
        fmt += forest._GROVES_SUFFIX
    members = dict(format=fmt, estimator=estimator, numeric_mode="identical", device="fixture",
        meta=np.asarray([2, 1, 2 if classifier else 1], dtype="<i8"),
        offsets=np.asarray([0, 1], dtype="<i4"), colid=np.asarray([-1], dtype="<i4"),
        quesval=np.asarray([0], dtype="<f4"), left_child=np.asarray([-1], dtype="<i4"),
        leaves=np.asarray([.25, .75] if classifier else [.5], dtype="<f4"))
    if classifier:
        members["classes"] = np.asarray(["cold", "warm"])
    if override:
        members.update(override)
    path = tmp_path / "forest.npz"
    np.savez(path, **members)
    return path, members


def _legacy_forest(members, engine):
    arrays = {name: frombytes(members[name].tobytes(), dtype, members[name].shape)
              for name, dtype in forest._MODEL_ARRAYS}
    return LegacyHostForest(estimator=members["estimator"], n_features=int(members["meta"][0]),
        n_trees=int(members["meta"][1]), num_outputs=int(members["meta"][2]),
        classes=members["classes"].tolist() if "classes" in members else None,
        device=members.get("device"), inference_engine=engine, **arrays)


@pytest.mark.parametrize("estimator", forest._ESTIMATORS)
@pytest.mark.parametrize("engine", ["sequential", "parallel_groves"])
def test_public_forest_archive_routes_preserve_state_and_predictions(tmp_path, estimator, engine):
    path, members = _forest_archive(tmp_path, estimator, engine)
    expected = _legacy_forest(members, engine)
    actual = audit_native_call(mojolearn.host_model, path).assert_native()
    direct = audit_native_call(forest.HostForest.from_file, path).assert_native()
    assert _state(vars(actual)) == _state(vars(expected)) == _state(vars(direct))
    queries = [[0.0, 1.0], [2.0, 3.0]]
    assert _state(actual.predict(queries)) == _state(expected.predict(queries))


FLOAT_TEXT = ("format mojolearn-model 2\nfeatures 1 0\ntrees 1\nlosses 0\n"
    "feature 0 folds 1 one_hot 0 type float nan as_is borders 1 0.5/3f000000\n"
    "tree 0 depth 1 dim 1 weights 0\nsplit 0 0 0 0\n"
    "leaf 0 0 0.25/3e800000\nleaf 0 1 0.75/3f400000\n")
NONSYMMETRIC_TEXT = FLOAT_TEXT.replace("tree 0 depth 1", "ntree 0 nodes 1").replace(
    "split 0 0 0 0", "node 0 0 0 0 1 1")
ONEHOT_TEXT = FLOAT_TEXT.replace("features 1 0", "features 1 1").replace(
    "folds 1 one_hot 0 type float", "folds 1 one_hot 1 type cat"
).replace("split 0 0 0 0", "split 0 0 0 0 split_type take_bin")
NONSYMMETRIC_ONEHOT_TEXT = ONEHOT_TEXT.replace("tree 0 depth 1", "ntree 0 nodes 1").replace(
    "split 0 0 0 0 split_type", "node 0 0 0 0 1 1 split_type")
GBDT_CASES = {
    "float": (FLOAT_TEXT, 1, [[0.], [1.]], [.25, .75]),
    "non_symmetric": (NONSYMMETRIC_TEXT, 1, [[0.], [1.]], [.25, .75]),
    "one_hot": (ONEHOT_TEXT, 1, [[0.], [1.]], [.75, .25]),
    "non_symmetric_one_hot": (NONSYMMETRIC_ONEHOT_TEXT, 1, [[0.], [1.]], [.75, .25]),
    "simple_ctr": (CTR_TEXT, 1, [[0.], [2.], [3.]], [1., .5, .25]),
    "tensor_ctr": (TENSOR_TEXT, 2, [[0., 0.], [1., 1.], [2., 0.]], [1., .5, .25]),
}


def _gbdt_archive(tmp_path, text, features, estimator="GradientBoosting", loss="RMSE", override=None):
    fields = dict(format=gbdt.GBDT_FORMAT, estimator=estimator, numeric_mode="identical", loss=loss,
        model=np.frombuffer(text.encode("utf-8"), dtype="<u1"),
        meta=np.asarray([features, 1, -1], dtype="<i8"), bias=np.asarray([0], dtype="<f8"))
    if override:
        fields.update(override)
    path = tmp_path / "gbdt.npz"
    np.savez(path, **fields)
    return path


@pytest.mark.parametrize("case", GBDT_CASES)
@pytest.mark.parametrize("estimator", gbdt.GBDT_ESTIMATORS)
def test_public_gbdt_routes_preserve_full_layout_and_consumed_predictions(tmp_path, case, estimator):
    text, features, queries, predictions = GBDT_CASES[case]
    path = _gbdt_archive(tmp_path, text, features, estimator)
    expected = LegacyHostGBDT(loss="RMSE", text=text, n_features_in=features, approx_dim=1,
                            numeric_mode="identical", bias=0., estimator=estimator)
    actual = audit_native_call(mojolearn.host_model, path).assert_native()
    direct = audit_native_call(gbdt.HostGBDT.from_file, path).assert_native()
    assert _state(vars(actual)) == _state(vars(expected)) == _state(vars(direct))
    assert _state(actual.predict(queries)) == _state(expected.predict(queries))
    np.testing.assert_array_equal(actual.predict(queries).tolist(), predictions)


@pytest.mark.parametrize("changes", [
    {"meta": np.asarray([2, 2, 1], dtype="<i8")},
    {"offsets": np.asarray([0, 2], dtype="<i4")},
    {"leaves": np.asarray([], dtype="<f4")},
    {"classes": np.asarray(["only-one"])},
])
def test_forest_bad_archives_fail_inside_native_boundary(tmp_path, changes):
    path, _ = _forest_archive(tmp_path, "RandomForestClassifier", override=changes)
    audit = audit_native_call(mojolearn.host_model, path)
    assert isinstance(audit.exception, ValueError), audit.report()
    report = audit.report()
    assert not any(report[k] for k in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads")), report


@pytest.mark.parametrize("changes", [
    {"meta": np.asarray([2, 1, -1], dtype="<i8")},
    {"bias": np.asarray([1], dtype="<f8")},
    {"estimator": "OrderedRMSE", "loss": "Logloss"},
    {"model": np.frombuffer(FLOAT_TEXT.replace("trees 1", "trees 2").encode(), dtype="<u1")},
])
def test_gbdt_bad_archives_fail_inside_native_boundary(tmp_path, changes):
    path = _gbdt_archive(tmp_path, FLOAT_TEXT, 1, override=changes)
    audit = audit_native_call(mojolearn.host_model, path)
    assert isinstance(audit.exception, ValueError), audit.report()
    report = audit.report()
    assert not any(report[k] for k in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads")), report


def test_public_archive_entries_are_each_one_native_binding_call():
    for module, owner, name in ((forest, None, "host_model"), (classical, None, "host_model"),
                                (forest, "HostForest", "from_file"), (gbdt, "HostGBDT", "from_file")):
        tree = ast.parse(Path(module.__file__).read_text())
        parent = tree if owner is None else next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner)
        method = next(n for n in parent.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert len(method.body) == 2
        assert isinstance(method.body[-1], ast.Return)
        assert sum(isinstance(n, ast.Call) for n in ast.walk(method.body[-1])) == 1
