# SPDX-License-Identifier: Apache-2.0
"""Native standard buffers and sequence inference versus the former asarray ABI."""
from array import array
from types import BuiltinFunctionType

import numpy as np
import pytest

from mojolearn import Array, Sequential, _backend
from mojolearn._cast_in import describe
from mojolearn._verify_native_runtime import audit_native_call


def binding():
    value = _backend.binding("_mojolearn_x_sequence")
    assert isinstance(value.model_fit, BuiltinFunctionType)
    assert value.x_sequence_vendor() == _backend.vendor()
    assert value.x_sequence_numeric_mode() == {"fast": 0, "identical": 1}[_backend.default_mode()]
    return value


def reference(x, y):
    # np.asarray is the exact previous public admission, used only in this
    # independent testing controller. Numerical execution calls the raw ABI.
    x, y = np.asarray(x), np.asarray(y)
    # The old public preparation materialized noncontiguous standard buffers;
    # an overlapping broadcast view cannot be passed directly to raw CastIn.
    if not (x.flags.c_contiguous or x.flags.f_contiguous):
        x = np.ascontiguousarray(x)
    classes, codes = np.unique(y, return_inverse=True)
    codes = codes.astype("i4")
    count = 3 * len(classes) + len(classes)
    params, m, v = (np.zeros(count, "f4") for _ in range(3))
    state = np.zeros(12, "i8")
    curve = np.zeros(2, "f4")
    spec = [1, 3, 1, 1, len(classes), 1, 0]
    b = binding()
    b.model_fit([describe(x, "X"), describe(codes, "y"), params.ctypes.data,
                 curve.ctypes.data, m.ctypes.data, v.ctypes.data, state.ctypes.data, 0],
                spec, [len(x), 1, 1, 0, 0, 0, 1, 4, 2, 1, 123, 0, 0, 1, 0],
                [0.03, 0.0, 0.0, 1e-8, 0.0, 0.0, 0.0, 0.1, 1.0, 0.0])
    predicted = np.zeros((len(x), 1), "f4")
    b.model_predict([describe(x, "X"), params.ctypes.data, predicted.ctypes.data],
                    spec, [0.0], [len(x), 4096, 4])
    return params, m, v, state, curve, classes, classes[predicted[:, 0].astype("i4")]


def supplied(kind):
    x = np.arange(18, dtype="f4").reshape(6, 3) / 17 - 0.5
    y = np.array([2, 7, 2, 7, 7, 2], "i2")
    if kind == "contiguous":
        return memoryview(x), y
    if kind == "readonly":
        return memoryview(x).toreadonly(), y
    if kind == "fortran":
        return memoryview(np.asfortranarray(x)), y
    if kind == "reversed":
        return memoryview(x[::-1, ::-1]), y
    if kind == "broadcast":
        return memoryview(np.broadcast_to(x[:1], x.shape)), y
    if kind == "array_buffer":
        return memoryview(array("f", x.ravel())).cast("B").cast("f", shape=x.shape), y
    if kind == "target_buffer":
        return x, memoryview(y[::-1])
    if kind == "nested_rows":
        return list(x), y
    if kind == "nested_strided_rows":
        return [row[::-1] for row in x], y
    if kind == "nested_numpy_scalars":
        return [[np.float32(v) for v in row] for row in x], list(y)
    if kind == "nested_half":
        return list(x.astype("f2")), list(y.astype("f2"))
    if kind == "nested_big_endian":
        return list(x.astype(">f4")), list(y)
    if kind == "nested_array_owners":
        return [Array.from_list(row.tolist(), "<f4") for row in x], y
    if kind == "ranges":
        return [range(k, k + 3) for k in range(6)], y
    raise AssertionError(kind)


@pytest.mark.parametrize("kind", ["contiguous", "readonly", "fortran", "reversed", "broadcast",
                                    "array_buffer", "target_buffer", "nested_rows", "nested_strided_rows",
                                    "nested_numpy_scalars", "nested_half", "nested_big_endian",
                                    "nested_array_owners", "ranges"])
def test_native_standard_buffers_and_nested_arrays_preserve_existing_fit_and_prediction(kind):
    x, y = supplied(kind)
    expected = reference(x, y)
    out = Sequential().linear(2).compile(loss="cross_entropy", optimizer="sgd", lr=0.03)
    assert audit_native_call(out.fit, x, y, epochs=2, batch_size=4, seed=123).assert_native() is out
    for actual, wanted in zip((out.params_, out._opt_m, out._opt_v, out._fit_state,
                               np.asarray(out.loss_curve_, "f4"), out.classes_), expected[:6]):
        assert actual.dtype == wanted.dtype
        assert actual.tobytes() == wanted.tobytes()
    actual = audit_native_call(out.predict, x).assert_native()
    assert actual.dtype == expected[6].dtype
    assert actual.tobytes() == expected[6].tobytes()


def test_native_nested_scalar_dtype_promotion_matches_all_numeric_dtype_pairs():
    types = [np.dtype(s) for s in ("?", "i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8", "f2", "f4", "f8")]
    out = Sequential().linear(2)
    for left in types:
        for right in types:
            supplied_y = [left.type(0), right.type(1), left.type(1), right.type(0)]
            expected = np.asarray(supplied_y)
            wanted_classes, wanted_codes = np.unique(expected, return_inverse=True)
            codes, descriptor, target = audit_native_call(out._targets, supplied_y, 4, 2,
                                                          "cross_entropy").assert_native()
            assert target == 1 and descriptor[3:5] == [4, 1]
            assert out.classes_.dtype == expected.dtype, (left, right)
            assert out.classes_.tobytes() == wanted_classes.tobytes(), (left, right)
            assert codes.tobytes() == wanted_codes.astype("i4").tobytes(), (left, right)


@pytest.mark.parametrize("labels", [
    [np.uint64(2**63 + 7), np.uint64(2**63 + 9)],
    [np.int64(-3), np.uint64(2**63 + 9)],
    [2**63 + 7, 2**63 + 9],
    [np.float16(-0.0), np.int8(1)],
])
def test_native_sequence_dtype_promotion_preserves_integer_range_and_signed_zero(labels):
    supplied_y = labels * 2
    expected = np.asarray(supplied_y)
    classes, codes = np.unique(expected, return_inverse=True)
    out = Sequential().linear(len(classes))
    actual, _, _ = audit_native_call(out._targets, supplied_y, 4, len(classes), "cross_entropy").assert_native()
    assert out.classes_.dtype == expected.dtype
    assert out.classes_.tobytes() == classes.tobytes()
    assert actual.tobytes() == codes.astype("i4").tobytes()


def test_native_buffer_and_sequence_admission_do_not_call_numpy_conversion(monkeypatch):
    x, y = supplied("nested_half")
    expected = reference(x, y)
    out = Sequential().linear(2).compile(loss="cross_entropy", optimizer="sgd", lr=0.03)

    def forbidden(*args, **kwargs):
        raise AssertionError("NumPy preparation reached in native buffer admission")

    with monkeypatch.context() as patch:
        for name in ("asarray", "array", "empty", "zeros", "ascontiguousarray", "promote_types", "result_type"):
            patch.setattr(np, name, forbidden)
        audit_native_call(out.fit, x, y, epochs=2, batch_size=4, seed=123).assert_native()
    assert out.params_.tobytes() == expected[0].tobytes()
    assert out.classes_.tobytes() == expected[5].tobytes()


@pytest.mark.parametrize("kind", ["non_native", "rank3_fortran", "ragged_rows", "complex", "object"])
def test_native_buffer_compatibility_retains_existing_refusals(kind):
    x = np.arange(18, dtype="f4").reshape(6, 3)
    expected = ValueError
    if kind == "non_native":
        x = memoryview(x.astype(">f4"))
        expected = TypeError
    elif kind == "rank3_fortran":
        x = memoryview(np.asfortranarray(x.reshape(6, 1, 3)))
    elif kind == "ragged_rows":
        x = [x[0], x[1, :2]]
    elif kind == "complex":
        x = [row.astype("c8") for row in x]
        expected = TypeError
    else:
        x = [row.astype(object) for row in x]
        expected = TypeError
    out = Sequential().linear(1).compile()
    trace = audit_native_call(out.fit, x, np.zeros(6, "f4"), epochs=1)
    assert isinstance(trace.exception, expected), trace.report()
    for key in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads"):
        assert not trace.report()[key], trace.report()


def test_native_half_labels_preserve_every_non_nan_encoding():
    # Exhaust every binary16 encoding, including signed zeros, subnormals and
    # infinities. np.unique is the independent label/codes oracle.
    labels = np.arange(65536, dtype="u2").view("f2")
    labels = labels[~np.isnan(labels)]
    classes, expected = np.unique(labels, return_inverse=True)
    out = Sequential().linear(len(classes))
    codes, _, _ = audit_native_call(out._targets, labels, len(labels), len(classes),
                                    "cross_entropy").assert_native()
    assert out.classes_.dtype == np.dtype("f2")
    assert out.classes_.tobytes() == classes.tobytes()
    assert codes.tobytes() == expected.astype("i4").tobytes()


def test_native_uint64_label_order_preserves_both_halves_and_extremes():
    labels = np.array([2**64 - 1, 0, 2**63, 2**63 - 1, 1, 2**63 + 1,
                       2**64 - 1, 0, 2**63], "u8")
    classes, expected = np.unique(labels, return_inverse=True)
    out = Sequential().linear(len(classes))
    codes, _, _ = audit_native_call(out._targets, labels, len(labels), len(classes),
                                    "cross_entropy").assert_native()
    assert out.classes_.tobytes() == classes.tobytes()
    assert codes.tobytes() == expected.astype("i4").tobytes()


@pytest.mark.parametrize("train_dtype,validation_dtype", [
    ("f2", "f2"), ("f2", "f4"), ("f4", "f2"),
    ("u8", "u8"), ("u8", "i8"), ("i8", "u8"),
])
def test_native_label_new_dtypes_preserve_validation_codes(train_dtype, validation_dtype):
    x = np.arange(18, dtype="f4").reshape(6, 3) / 17
    y = np.array([2, 7, 2, 7, 7, 2], train_dtype)
    vy = y[::-1].astype(validation_dtype)
    reference_model = Sequential().linear(2).compile(loss="cross_entropy", optimizer="sgd", lr=0.03)
    audit_native_call(reference_model.fit, x, y.astype("i8"), epochs=2, batch_size=4, seed=123,
                      validation_data=(x, vy.astype("i8"))).assert_native()
    model = Sequential().linear(2).compile(loss="cross_entropy", optimizer="sgd", lr=0.03)
    audit_native_call(model.fit, x, y, epochs=2, batch_size=4, seed=123,
                      validation_data=(x, vy)).assert_native()
    assert model.params_.tobytes() == reference_model.params_.tobytes()
    assert np.asarray(model.validation_loss_curve_, "f4").tobytes() == np.asarray(
        reference_model.validation_loss_curve_, "f4").tobytes()
    assert model.classes_.dtype == np.dtype(train_dtype)


@pytest.mark.parametrize("train_dtype,unknown", [
    ("u8", np.array([-1, 7, 2, 7, 7, 2], "i8")),
    ("i8", np.array([2**64 - 1, 7, 2, 7, 7, 2], "u8")),
    ("f2", np.array([np.nan, 7, 2, 7, 7, 2], "f2")),
])
def test_native_validation_refuses_out_of_domain_label_without_wrapping(train_dtype, unknown):
    x = np.arange(18, dtype="f4").reshape(6, 3) / 17
    y = np.array([2, 7, 2, 7, 7, 2], train_dtype)
    model = Sequential().linear(2).compile(loss="cross_entropy", optimizer="sgd", lr=0.03)
    trace = audit_native_call(model.fit, x, y, epochs=1, validation_data=(x, unknown))
    assert isinstance(trace.exception, ValueError), trace.report()
    assert "unknown class or NaN" in str(trace.exception)
    for key in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads"):
        assert not trace.report()[key], trace.report()
