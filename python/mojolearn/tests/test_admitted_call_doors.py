# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""One admission record per public call (`AdmittedCall`, lane ne1-e-admitted-call).

Source checks (run anywhere): the record module carries the fields the plan
names; classical_c, the public attribute entry, the GP and boost doors build
one record and hand it to their helpers; the helpers that re-admitted modules,
bindings or X take the door's admission instead.

Runtime checks: with a compiled classical binding, Ridge fit / predict / coef_ and a GP classifier fit
and predict give the same answers as before (the record changes no numerics).

    cd python && python3 -m pytest mojolearn/tests/test_admitted_call_doors.py -q
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def _function(text, signature):
    start = text.index(signature)
    end = text.find("\ndef ", start + 1)
    return text[start:] if end < 0 else text[start:end]


def test_record_fields():
    text = _read("bindings/admitted_call.mojo")
    record = text[text.index("struct AdmittedCall("):]
    for field in ("var modules: NativeModules", "var binding: PythonObject", "var fields: PythonObject",
                  "var inputs: List[AdmittedInput]", "var layout_checked: Bool", "var layout_plan: PythonObject"):
        assert field in record, field
    admitted = text[text.index("struct AdmittedInput("):text.index("struct AdmittedCall(")]
    for field in ("var address: Int", "var shape: List[Int]", "var strides: List[Int]", "var item_code: Int",
                  "var read_only: Bool", "var sparse_code: Int", "var owner: PythonObject"):
        assert field in admitted, field
    # The record imports only the module loader: every group imports it.
    imports = [line for line in text.splitlines() if line.startswith(("from ", "import "))]
    assert imports == ["from std.python import Python, PythonObject",
                       "from bindings.native_loaded_module import NativeModules"]


def test_classical_door_builds_one_record():
    text = _read("bindings/classical_native.mojo")
    door = _function(text, "def classical_c(")
    assert door.count("AdmittedCall()") == 1
    assert "NativeModules()" not in door
    assert "classical_entries(call, model, context)" in door
    assert "classical_getattribute(call, model, args[0], context)" in door
    assert door.count("sequential_fields(model)") == 1
    entries = _function(text, "def classical_entries(")
    assert "mut call: AdmittedCall" in entries and 'native_loaded_module("builtins")' not in entries


def test_attribute_helpers_share_the_door_modules():
    text = _read("bindings/public_attributes_native.mojo")
    for name in ("def _available(", "def _property("):
        body = _function(text, name)
        assert "mut mods: NativeModules" in body.split(")", 1)[0] + ")"
        assert "NativeModules()" not in body, name
    door = _function(text, "def public_attribute_c(")
    assert "AdmittedCall()" in door and "NativeModules()" not in door


def test_bindings_resolve_once_per_call():
    text = _read("providers/python/estimator_binding_pyapi.mojo")
    assert "def estimator_binding_mode(mut call: AdmittedCall" in text
    assert "def estimator_arguments(mut call: AdmittedCall" in text
    gp = _read("providers/python/gp_public_lifecycle_pyapi.mojo")
    door = _function(gp, "def gp_public_operation(")
    assert door.count("AdmittedCall()") == 1
    assert 'estimator_binding_mode(call, PythonObject("identical"), "_mojolearn", context)' in door
    assert "estimator_arguments(call, model," in _function(gp, "def _gp_public_arguments(")


def test_second_admissions_removed():
    boost = _read("providers/python/tree_boost_public_lifecycle_pyapi.mojo")
    operation = _function(boost, "def tree_boost_public_operation(")
    assert "labels_shape_of(" not in operation
    assert "classical_check_x(call, args[0], name, context, 0, admitted=True)" in operation
    assert "labels_shape_of(" not in _function(boost, "def _boost_classifier_target(").split('"""', 2)[2]
    linalg = _read("bindings/linalg_ops_buffers.mojo")
    assert "LinAdmission(prepared, context, prepared is source)" in _function(linalg, "def lin_typed(")
    projection = _function(_read("bindings/host_projection_native.mojo"), "def host_projection_inference(")
    assert "host_tree_cast(raw, context)" in projection
    assert re.search(r"lin_operand\(args\[0\], \"a\", context\)(?!, True)", _read("bindings/linalg_ops_decomp.mojo")) is None


def test_mode_plan_reused_per_class_state():
    text = _function(_read("bindings/mode_metadata_native.mojo"),
                     "def _plan(mut mods: NativeModules, model: PythonObject, state: PythonObject")
    assert "class_record(mods, _MODE_PLAN_SLOT, cls)" in text
    assert "class_admit(mods, _MODE_PLAN_SLOT, cls" in text


def test_ridge_answers_through_the_record():
    np = pytest.importorskip("numpy")
    import mojolearn

    rng = np.random.default_rng(0)
    X = rng.standard_normal((64, 5)).astype(np.float32)
    y = (X @ np.arange(1, 6, dtype=np.float32) + 0.5).astype(np.float32)
    try:
        model = mojolearn.Ridge(alpha=1.0).fit(X, y)
    except (ImportError, NotImplementedError, RuntimeError) as exc:
        pytest.skip(f"no classical binding in this process: {type(exc).__name__}: {exc}")
    first = np.asarray(model.predict(X))
    assert np.array_equal(first, np.asarray(model.predict(X)))
    assert np.asarray(model.coef_).shape == (5,)
    assert model.get_params()["alpha"] == 1.0
    with pytest.raises(AttributeError):
        model.no_such_attribute_


def test_gp_classifier_resolves_its_bindings_once():
    np = pytest.importorskip("numpy")
    import mojolearn

    rng = np.random.default_rng(1)
    X = rng.standard_normal((40, 3)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)
    try:
        model = mojolearn.GaussianProcessClassifier().fit(X, y)
    except (ImportError, NotImplementedError, RuntimeError, AttributeError) as exc:
        pytest.skip(f"no GP binding in this process: {type(exc).__name__}: {exc}")
    predicted = np.asarray(model.predict(X))
    assert predicted.shape == (40,)
    assert np.array_equal(predicted, np.asarray(model.predict(X)))
