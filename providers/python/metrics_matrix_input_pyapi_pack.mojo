# SPDX-License-Identifier: Apache-2.0
# Exact native metric target/score/weight buffers, captured once per input.
# The PACK half of the former `providers/python/metrics_matrix_input_pyapi.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `providers/python/metrics_matrix_input_pyapi_unpack.mojo`.
from std.python import Python, PythonObject
from bindings.admitted_input_unpack import AdmittedTypedPin, admitted_typed_once, admitted_typed_c, admitted_shape_tuple
from bindings.tensor_view import base_format, TensorView
from bindings.family_common_unpack import family_repr, FAMILY_F32P
from core.family_cast_impl import family_min_f32, family_max_f32
from core.host_finite_scan import host_all_finite_f32


def _metric_f32(input: TensorView) -> Bool:
    return input.dtype == "<f4" if input.is_array else base_format(input.fmt) == "f"


def metric_reg_target(source: PythonObject, name: String, caller: String,
                      context: PythonObject) raises -> PythonObject:
    var input_pin = AdmittedTypedPin()
    var input = admitted_typed_once(source, context, input_pin)
    if not _metric_f32(input):
        raise Error("TypeError: mojolearn " + caller + ": " + name + " must have dtype float32; cast explicitly before scoring")
    var rank = len(input.shape)
    var rows = input.shape[0] if rank >= 1 else 0
    var cols = input.shape[1] if rank == 2 else 1
    if (rank != 1 and rank != 2) or rows == 0 or cols == 0:
        raise Error("mojolearn " + caller + ": " + name + " must be a nonempty (n,) or (n, n_outputs) array")
    var output = admitted_typed_c(input, input_pin^, "<f4", name, context, True)
    if not host_all_finite_f32(FAMILY_F32P(unsafe_from_address=output.addr), rows * cols):
        raise Error("mojolearn " + caller + ": " + name + " contains NaN or infinity")
    var result = Python.tuple(output.keep, rows, cols)
    _ = output^
    return result


def metric_score_matrix(source: PythonObject, n: Int, caller: String, ndim: Int,
                        name: String, context: PythonObject) raises -> PythonObject:
    var input_pin = AdmittedTypedPin()
    var input = admitted_typed_once(source, context, input_pin)
    if not _metric_f32(input):
        raise Error("TypeError: mojolearn " + caller + ": " + name + " must have dtype float32; cast explicitly")
    var rank = len(input.shape)
    var rows = input.shape[0] if rank >= 1 else 0
    var cols = input.shape[1] if rank >= 2 else 1
    var nd = 1 if rank == 2 and cols == 1 and ndim == 1 else rank
    if ndim >= 0 and nd != ndim:
        raise Error("mojolearn " + caller + ": " + name + " must be " + String(ndim) + "-D, got shape "
                    + family_repr(admitted_shape_tuple(input.shape)))
    if rows != n:
        raise Error("mojolearn " + caller + ": " + name + " has " + String(rows) + " rows for " + String(n) + " samples")
    if rank > 2:
        raise Error("mojolearn: " + name + " must be at most 2-D, got " + String(rank) + "-D")
    if input.count() == 0:
        var what = "feature(s)" if rank == 2 and rows != 0 else "sample(s)"
        var shape = family_repr(admitted_shape_tuple(input.shape))
        raise Error("mojolearn: " + name + " is empty, shape " + shape + ": Found array with 0 " + what
                    + " (shape=" + shape + ") while a minimum of 1 is required.")
    var output = admitted_typed_c(input, input_pin^, "<f4", name, context, True)
    if not host_all_finite_f32(FAMILY_F32P(unsafe_from_address=output.addr), rows * cols):
        raise Error("mojolearn " + caller + ": " + name + " must be finite")
    var result = Python.tuple(output.keep, rows, cols, nd)
    _ = output^
    return result
