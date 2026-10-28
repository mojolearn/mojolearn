# SPDX-License-Identifier: Apache-2.0
# One retained metadata capture for the classification metric vector loaders.
from std.python import Python, PythonObject
from bindings.py_balanced import py_type, py_get
from bindings.classical_arrays import classical_is_array
from bindings.linalg_ops_buffers import LinAdmission, lin_admitted_c, lin_new, lin_tuple
from bindings.optimizer_public_door import base_format
from bindings.family_common import family_repr, FAMILY_F32P, FAMILY_I32P, FAMILY_I64P
from core.family_cast_impl import family_all_finite_f32
from core.mcls_impl import mcls_read_i64, mcls_narrow_i32


def metric_vector_exact(value: PythonObject, context: PythonObject) raises -> Bool:
    return classical_is_array(value, context) or py_type(value) is py_get(context, "ndarray")


def _metric_vector_shape(input: LinAdmission, name: String) raises -> Int:
    ref shape = input.view.shape
    var rank = len(shape)
    if rank != 1 and not (rank == 2 and shape[1] == 1):
        raise Error("mojolearn metrics: " + name + " must be 1-D, got shape " + family_repr(lin_tuple(shape)))
    if shape[0] == 0:
        raise Error("mojolearn metrics: " + name + " is empty")
    return shape[0]


def _metric_i32_range(name: String) raises:
    raise Error("mojolearn metrics: " + name + " does not fit in int32; the implemented "
                + "kernels are the int32 instantiation (the int64 overload of "
                + "adjusted_rand_index is the same code at a wider type and is "
                + "not instantiated, metrics/NOT_IMPLEMENTED.tsv)")


def metric_i32_vector(value: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    var input = LinAdmission(value, context)
    var count = _metric_vector_shape(input, name)
    ref view = input.view
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
    var words = lin_new([count], "<i8", context)
    if not mcls_read_i64(view.addr, view.strides[0], count, view.itemsize, signed,
                         FAMILY_I64P(unsafe_from_address=words.addr)):
        _metric_i32_range(name)
    var output = lin_new([count], "<i4", context)
    if not mcls_narrow_i32(FAMILY_I64P(unsafe_from_address=words.addr), count,
                           FAMILY_I32P(unsafe_from_address=output.addr)):
        _metric_i32_range(name)
    var result = output.keep
    _ = input^
    _ = words^
    _ = output^
    return result


def metric_f32_vector(value: PythonObject, name: String, require_finite: Bool,
                      regression: Bool, context: PythonObject) raises -> PythonObject:
    var input = LinAdmission(value, context)
    var count = _metric_vector_shape(input, name)
    var code = base_format(input.view.fmt)
    var f32 = input.view.dtype == "<f4" if input.view.is_array else code == "f"
    var f64 = input.view.dtype == "<f8" if input.view.is_array else code == "d"
    if regression and not f32:
        raise Error("TypeError: mojolearn regression metrics: " + name
                    + " must have dtype float32; cast explicitly before scoring")
    if f64:
        raise Error("TypeError: mojolearn metrics: " + name + " is float64; the float64 overloads of "
                    + "r2_score, kl_divergence and silhouette_score are NOT implemented "
                    + "(no float64 on this GPU). Cast to float32 yourself so the "
                    + "precision you run at is the one you chose.")
    # The numerical owners consume an Array, so keep an operation-owned copy
    # independent of mutable source Array slots after this snapshot returns.
    var output = lin_admitted_c(input^, "<f4", name, context, True)
    if require_finite and not family_all_finite_f32(FAMILY_F32P(unsafe_from_address=output.addr), count):
        raise Error("mojolearn metrics: " + name + " contains NaN or infinity; refused "
                    + "here (metrics/README.md HAND-OFF ask 3) rather than letting it "
                    + "reach a kernel whose answer for it is a washed sentinel")
    var result = output.keep
    _ = output^
    return result
