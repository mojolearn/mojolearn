# SPDX-License-Identifier: Apache-2.0
"""Public density digests preserve stored bytes and own the complete hash walk."""
import array
import builtins
import gc
import hashlib
import json
import os
import subprocess
import sys
import weakref

import numpy as np
import pytest

from mojolearn import Array
from mojolearn import _classical_host as host
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests._connected_exports import assert_connected_export_refused


# Frozen pre-migration operation, independent of the new native hash owner.
# This is test-only product-reference code; callers prepare it outside audits.
def _legacy_digest(model):
    digest = hashlib.sha256()
    for name in sorted(model._HOST_ARRAYS):
        value = getattr(model, name)
        if value is not None:
            digest.update(value.tobytes())
    return digest.hexdigest()


def _model(family, value, weighted=False, cls=None):
    cls = cls or (host.HostKernelDensity if family == "kde" else host.HostIsolationForest)
    model = cls(numeric_mode="identical")
    vars(model).update(_x=value, _mojolearn_fitted=True, n_features_in_=2)
    if family == "kde":
        vars(model).update(_w=Array.from_list([.5, 1., 2.], "<f4") if weighted else None)
    return model


PAYLOADS = ("array", "array_fortran", "ndarray", "fortran", "strided", "reversed",
            "readonly", "big_endian", "uint64", "float_words", "memoryview",
            "native_array", "empty", "zero_dim", "scalar", "complex_scalar",
            "string_scalar", "unicode_scalar", "structured", "datetime_scalar", "timedelta_scalar",
            "datetime_strided", "timedelta", "void_scalar")


def _payload(kind):
    x = np.asarray([[.25, -.75], [1.5, .125], [-.5, .875]], dtype="<f4")
    if kind == "array": return Array.from_buffer(x)
    if kind == "array_fortran": return Array.from_buffer(np.asfortranarray(x))
    if kind == "ndarray": return x
    if kind == "fortran": return np.asfortranarray(x)
    if kind == "strided": return np.repeat(x, 2, axis=1)[:, ::2]
    if kind == "reversed": return x[::-1, ::-1]
    if kind == "readonly":
        x.flags.writeable = False
        return x
    if kind == "big_endian": return x.astype(">f4")
    if kind == "uint64": return np.asarray([0, 2**63 + 1, 2**64 - 1], dtype="<u8")
    if kind == "float_words":
        return Array.from_buffer(np.asarray([0, 0x80000000, 1, 0x7f800001, 0xffc00023], dtype="<u4").view("<f4"))
    if kind == "memoryview": return memoryview(x)[::-1]
    if kind == "native_array": return array.array("f", [1.5, -.25, -0.0])
    if kind == "empty": return Array.from_buffer(np.empty((0, 2), dtype="f4"))
    if kind == "zero_dim": return np.asarray(-0., dtype="f4")
    if kind == "scalar": return np.float32(-0.)
    if kind == "complex_scalar": return np.complex128(complex(-0., 2.5))
    if kind == "string_scalar": return np.bytes_(b"a\x00b")
    if kind == "unicode_scalar": return np.str_("\u00e9\U0001f680")
    if kind == "structured": return np.asarray([(1, -.25), (4, .5)], dtype=[("i", "i4"), ("x", "f4")])
    if kind == "datetime_scalar": return np.datetime64("2020-01-02")
    if kind == "timedelta_scalar": return np.timedelta64(-4, "D")
    if kind == "datetime_strided": return np.asarray(["2020-01-02", "2022-03-04"], dtype="M8[D]")[::-1]
    if kind == "timedelta": return np.asarray([1, 3, -5], dtype="m8[ns]")
    if kind == "void_scalar": return np.void(b"\x01\x03")
    raise AssertionError(kind)


def _native_error(audit, expected):
    assert type(audit.exception) is type(expected), audit.report()
    assert str(audit.exception) == str(expected)
    report = audit.report()
    assert not any(report[key] for key in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads")), report


@pytest.mark.parametrize("family,weighted", [("kde", False), ("kde", True), ("iforest", False)])
@pytest.mark.parametrize("kind", PAYLOADS)
def test_density_hash_preserves_consumed_physical_and_logical_bytes(family, weighted, kind):
    value = _payload(kind)
    model = _model(family, value, weighted)
    expected = _legacy_digest(model)
    assert audit_native_call(model.model_sha256).assert_native() == expected
    assert vars(model)["_x"] is value


@pytest.mark.parametrize("family", ["kde", "iforest"])
def test_density_hash_keeps_fortran_array_physical_order(family):
    physical = _model(family, _payload("array_fortran"))
    logical = _model(family, _payload("fortran"))
    assert _legacy_digest(physical) != _legacy_digest(logical)
    assert audit_native_call(physical.model_sha256).assert_native() == _legacy_digest(physical)
    assert audit_native_call(logical.model_sha256).assert_native() == _legacy_digest(logical)


@pytest.mark.parametrize("family", ["kde", "iforest"])
@pytest.mark.parametrize("kind", ["missing", "missing_weights", "none", "unfitted", "integer", "bytes", "list", "released"])
def test_density_hash_preserves_missing_none_and_native_errors(family, kind):
    model = _model(family, _payload("array"))
    if kind == "missing": del vars(model)["_x"]
    elif kind == "missing_weights":
        vars(model).pop("_w", None)
        model._HOST_ARRAYS = ("_w", "_x")
    elif kind == "none": model._x = None
    elif kind == "unfitted": model = type(model)(numeric_mode="identical")
    elif kind == "integer": model._x = 7
    elif kind == "bytes": model._x = b"raw"
    elif kind == "list": model._x = [1., 2.]
    elif kind == "released":
        model._x = memoryview(b"raw")
        model._x.release()
    before = vars(model).copy()
    try:
        expected = _legacy_digest(model)
    except Exception as failure:
        _native_error(audit_native_call(model.model_sha256), failure)
    else:
        assert audit_native_call(model.model_sha256).assert_native() == expected
    assert vars(model).keys() == before.keys()
    assert all(vars(model)[name] is value for name, value in before.items())


FIELD_FORMS = ((), ("_x", "_x"), ["_z", "_x"], {"_z", "_x"}, frozenset(("_x", "_z")),
               {"_z": None, "_x": None}, "", b"", range(0), [1], [1, "_x"], None)


@pytest.mark.parametrize("family", ["kde", "iforest"])
@pytest.mark.parametrize("fields", FIELD_FORMS)
def test_density_hash_preserves_native_field_collections(family, fields):
    model = _model(family, _payload("array"))
    model._HOST_ARRAYS = fields
    model._z = _payload("native_array")
    try:
        expected = _legacy_digest(model)
    except Exception as failure:
        _native_error(audit_native_call(model.model_sha256), failure)
    else:
        assert audit_native_call(model.model_sha256).assert_native() == expected


@pytest.mark.parametrize("family", ["kde", "iforest"])
def test_density_hash_preserves_registered_subclass_native_slots(family):
    base = host.HostKernelDensity if family == "kde" else host.HostIsolationForest
    class Slotted(base):
        __slots__ = ("_x", "_w")
        _HOST_ARRAYS = ("_x", "_w", "_x")
    model = _model(family, _payload("fortran"), cls=Slotted)
    model._x = vars(model).pop("_x")
    model._w = vars(model).pop("_w", None)
    assert audit_native_call(model.model_sha256).assert_native() == _legacy_digest(model)


@pytest.mark.parametrize("family", ["kde", "iforest"])
@pytest.mark.parametrize("kind", ["tobytes", "array_protocol", "ndarray_subclass", "field_iterator", "field_compare", "property"])
def test_density_hash_refuses_python_protocols_before_hooks(family, kind):
    touched = []
    class Foreign:
        def tobytes(self): touched.append("tobytes"); return b"bad"
        def __array__(self, *args, **kwargs): touched.append("array"); return np.ones(1)
        def __iter__(self): touched.append("iter"); return iter(("_x",))
        def __lt__(self, other): touched.append("compare"); return True
    class ForeignArray(np.ndarray):
        def tobytes(self, *args, **kwargs): touched.append("array_tobytes"); return b"bad"
    model = _model(family, _payload("array"))
    if kind in ("tobytes", "array_protocol"): model._x = Foreign()
    elif kind == "ndarray_subclass": model._x = np.ones(2).view(ForeignArray)
    elif kind == "field_iterator": model._HOST_ARRAYS = Foreign()
    elif kind == "field_compare": model._HOST_ARRAYS = [Foreign(), Foreign()]
    elif kind == "property":
        base = type(model)
        class Derived(base):
            @property
            def _x(self): touched.append("property"); return np.ones(1)
        model = _model(family, _payload("array"), cls=Derived)
    audit = audit_native_call(model.model_sha256)
    assert type(audit.exception) is TypeError, audit.report()
    report = audit.report()
    assert not any(report[key] for key in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads")), report
    assert touched == []


@pytest.mark.parametrize("family", ["kde", "iforest"])
def test_density_hash_rejects_replaced_array_buffer_property(family, monkeypatch):
    model = _model(family, _payload("array"))
    touched = []
    def replacement(self):
        touched.append("buffer")
        return memoryview(b"bad")
    with monkeypatch.context() as patch:
        patch.setattr(Array, "_mv", property(replacement))
        audit = audit_native_call(model.model_sha256)
    assert type(audit.exception) is TypeError, audit.report()
    assert not audit.report()["python_frames"]
    assert touched == []


@pytest.mark.parametrize("family", ["kde", "iforest"])
@pytest.mark.parametrize("target", ["hashlib", "sorted", "memoryview", "array_tobytes"])
def test_density_hash_uses_connected_native_entries(family, target, monkeypatch):
    model = _model(family, _payload("array"))
    expected = _legacy_digest(model)
    touched = []
    def forbidden(*args, **kwargs):
        touched.append(target)
        raise AssertionError("replacement executed")
    # Pin only observer globals; the product sees the replacements throughout
    # the audited operation. This prevents inspect/dis from calling the poison.
    import dis
    import mojolearn._verify_native_runtime as verifier
    monkeypatch.setitem(dis.__dict__, "sorted", sorted)
    monkeypatch.setitem(verifier.__dict__, "sorted", sorted)
    with monkeypatch.context() as patch:
        if target == "hashlib": patch.setattr(hashlib, "sha256", forbidden)
        elif target == "array_tobytes": patch.setattr(Array, "tobytes", forbidden)
        else: patch.setattr(builtins, target, forbidden)
        audit = audit_native_call(model.model_sha256)
    if target in ("sorted", "memoryview"):
        # Connected builtins exports refuse when replaced (see _connected_exports).
        assert_connected_export_refused(audit, "builtins", target)
        assert touched == []
        audit = audit_native_call(model.model_sha256)
    assert audit.assert_native() == expected
    assert touched == []


@pytest.mark.parametrize("family", ["kde", "iforest"])
def test_density_hash_consumes_storage_without_retaining_owners(family):
    value = _payload("strided")
    reference = weakref.ref(value)
    model = _model(family, value, weighted=True)
    expected = _legacy_digest(model)
    result = audit_native_call(model.model_sha256).assert_native()
    del model, value
    gc.collect()
    assert reference() is None
    assert result == expected
    assert len(bytes.fromhex(result)) == 32


@pytest.mark.parametrize("family", ["kde", "iforest"])
def test_density_hash_without_numpy_has_exact_digest(family):
    # Standalone child fixture: installed wheels intentionally omit test files.
    code = r'''
import hashlib, json, sys
sys.modules["numpy"] = None
from mojolearn import Array
from mojolearn import _classical_host as host
from mojolearn._verify_native_runtime import audit_native_call
family = sys.argv[1]
cls = host.HostKernelDensity if family == "kde" else host.HostIsolationForest
model = cls(numeric_mode="identical")
x = Array.from_list([[.25, -.75], [1.5, .125], [-.5, .875]], "<f4")
vars(model)["_x"] = x
if family == "kde": vars(model)["_w"] = None
expected = hashlib.sha256(x.tobytes()).hexdigest()
audit = audit_native_call(model.model_sha256)
assert audit.assert_native() == expected, audit.report()
print(json.dumps({"digest": expected, "numpy_loaded": sys.modules["numpy"] is not None}))
'''
    result = subprocess.run([sys.executable, "-c", code, family], text=True, capture_output=True,
                            env={**os.environ, "MOJOLEARN_VENDOR": "cpu"}, check=False)
    assert result.returncode == 0, result.stderr
    record = json.loads(result.stdout)
    assert not record["numpy_loaded"]
    assert len(record["digest"]) == 64
