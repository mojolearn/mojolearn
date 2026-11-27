# SPDX-License-Identifier: Apache-2.0
# One retained metadata capture for the classification metric vector loaders.
from std.python import Python, PythonObject
from bindings.py_balanced import py_type, py_get
from bindings.classical_arrays_door import classical_is_array
from bindings.admitted_input_unpack import AdmittedTypedPin, admitted_typed_once, admitted_typed_c, admitted_zeros, admitted_shape_tuple
from bindings.tensor_view import base_format, TensorView
from bindings.family_common_unpack import family_repr, FAMILY_F32P, FAMILY_I64P
from core.host_finite_scan import host_all_finite_f32
from metrics.metrics_vector_input_run import metric_i32_narrow
from bindings.metrics_keys_unpack import metrics_keys


def metric_vector_exact(value: PythonObject, context: PythonObject) raises -> Bool:
    var nk = metrics_keys()
    return classical_is_array(value, context) or py_type(value) is py_get(context, nk[].k_ndarray)


def _metric_vector_shape(input: TensorView, name: String) raises -> Int:
    ref shape = input.shape
    var rank = len(shape)
    if rank != 1 and not (rank == 2 and shape[1] == 1):
        raise Error("mojolearn metrics: " + name + " must be 1-D, got shape " + family_repr(admitted_shape_tuple(shape)))
    if shape[0] == 0:
        raise Error("mojolearn metrics: " + name + " is empty")
    return shape[0]


def metric_i32_vector(value: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    var input_pin = AdmittedTypedPin()
    var input = admitted_typed_once(value, context, input_pin)
    var count = _metric_vector_shape(input, name)
    ref view = input
    var code = base_format(view.fmt)
    var signed = String("bhilqn").find(code) >= 0 and code.byte_length() == 1
    var unsigned = String("BHILQN").find(code) >= 0 and code.byte_length() == 1
    if view.is_array:
        var letter = String(view.dtype[byte=1:2])
        signed = letter == "i"
        unsigned = letter == "u"
    if (not signed and not unsigned) or (view.itemsize != 1 and view.itemsize != 2 and view.itemsize != 4 and view.itemsize != 8):
        raise Error("mojolearn metrics: " + name + " must be an integer label array, got "
                    + (view.dtype if view.is_array else view.fmt) + "; the ported kernels take int32 labels")
    # admit_ids: one parallel read narrowed to int32 in the same pass (it was
    # a serial int64 read into a zero-filled store, then a serial narrowing).
    var output = admitted_zeros([count], "<i4", context)
    metric_i32_narrow(view.addr, view.strides[0], count, view.itemsize, signed, output.addr, name)
    var result = output.keep
    _ = input_pin^
    _ = output^
    return result


def metric_f32_vector(value: PythonObject, name: String, require_finite: Bool,
                      regression: Bool, context: PythonObject) raises -> PythonObject:
    var input_pin = AdmittedTypedPin()
    var input = admitted_typed_once(value, context, input_pin)
    var count = _metric_vector_shape(input, name)
    var code = base_format(input.fmt)
    var f32 = input.dtype == "<f4" if input.is_array else code == "f"
    var f64 = input.dtype == "<f8" if input.is_array else code == "d"
    # float64 is converted once below by the shared admission narrowing
    # (admitted_typed_c; Andrew, 2026-11-27).
    if regression and not f32 and not f64:
        raise Error("TypeError: mojolearn regression metrics: " + name + " must have dtype float32 or float64")
    # The numerical owners consume an Array, so keep an operation-owned copy
    # independent of mutable source Array slots after this snapshot returns.
    var output = admitted_typed_c(input, input_pin^, "<f4", name, context, True)
    if require_finite and not host_all_finite_f32(FAMILY_F32P(unsafe_from_address=output.addr), count):
        raise Error("mojolearn metrics: " + name + " contains NaN or infinity; refused "
                    + "here (metrics/README.md HAND-OFF ask 3) rather than letting it "
                    + "reach a kernel whose answer for it is a washed sentinel")
    var result = output.keep
    _ = output^
    return result
