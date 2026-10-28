# SPDX-License-Identifier: Apache-2.0
# Exact native metric target/score/weight buffers, captured once per input.
from std.python import Python, PythonObject
from bindings.linalg_ops_buffers import LinAdmission, lin_admitted_c, lin_tuple
from bindings.optimizer_public_door import base_format
from bindings.family_common import family_repr, FAMILY_F32P
from core.family_cast_impl import family_all_finite_f32, family_min_f32, family_max_f32


def _metric_f32(input: LinAdmission) -> Bool:
    return input.view.dtype == "<f4" if input.view.is_array else base_format(input.view.fmt) == "f"


def metric_reg_target(source: PythonObject, name: String, caller: String,
                      context: PythonObject) raises -> PythonObject:
    var input = LinAdmission(source, context)
    if not _metric_f32(input):
        raise Error("TypeError: mojolearn " + caller + ": " + name + " must have dtype float32; cast explicitly before scoring")
    var rank = len(input.view.shape)
    var rows = input.view.shape[0] if rank >= 1 else 0
    var cols = input.view.shape[1] if rank == 2 else 1
    if (rank != 1 and rank != 2) or rows == 0 or cols == 0:
        raise Error("mojolearn " + caller + ": " + name + " must be a nonempty (n,) or (n, n_outputs) array")
    var output = lin_admitted_c(input^, "<f4", name, context, True)
    if not family_all_finite_f32(FAMILY_F32P(unsafe_from_address=output.addr), rows * cols):
        raise Error("mojolearn " + caller + ": " + name + " contains NaN or infinity")
    var result = Python.tuple(output.keep, rows, cols)
    _ = output^
    return result


def metric_score_matrix(source: PythonObject, n: Int, caller: String, ndim: Int,
                        name: String, context: PythonObject) raises -> PythonObject:
    var input = LinAdmission(source, context)
    if not _metric_f32(input):
        raise Error("TypeError: mojolearn " + caller + ": " + name + " must have dtype float32; cast explicitly")
    var rank = len(input.view.shape)
    var rows = input.view.shape[0] if rank >= 1 else 0
    var cols = input.view.shape[1] if rank >= 2 else 1
    var nd = 1 if rank == 2 and cols == 1 and ndim == 1 else rank
    if ndim >= 0 and nd != ndim:
        raise Error("mojolearn " + caller + ": " + name + " must be " + String(ndim) + "-D, got shape "
                    + family_repr(lin_tuple(input.view.shape)))
    if rows != n:
        raise Error("mojolearn " + caller + ": " + name + " has " + String(rows) + " rows for " + String(n) + " samples")
    if rank > 2:
        raise Error("mojolearn: " + name + " must be at most 2-D, got " + String(rank) + "-D")
    if input.view.count() == 0:
        var what = "feature(s)" if rank == 2 and rows != 0 else "sample(s)"
        var shape = family_repr(lin_tuple(input.view.shape))
        raise Error("mojolearn: " + name + " is empty, shape " + shape + ": Found array with 0 " + what
                    + " (shape=" + shape + ") while a minimum of 1 is required.")
    var output = lin_admitted_c(input^, "<f4", name, context, True)
    if not family_all_finite_f32(FAMILY_F32P(unsafe_from_address=output.addr), rows * cols):
        raise Error("mojolearn " + caller + ": " + name + " must be finite")
    var result = Python.tuple(output.keep, rows, cols, nd)
    _ = output^
    return result


def metric_sample_weights(source: PythonObject, n: Int, caller: String,
                          context: PythonObject) raises -> PythonObject:
    var input = LinAdmission(source, context)
    if len(input.view.shape) != 1:
        raise Error("mojolearn " + caller + ": sample_weight must be 1-D, got shape "
                    + family_repr(lin_tuple(input.view.shape)))
    var count = input.view.shape[0]
    if count != n:
        raise Error("mojolearn " + caller + ": sample_weight has " + String(count) + " entries for " + String(n) + " samples")
    var output = lin_admitted_c(input^, "<f4", "sample_weight", context, True)
    var words = FAMILY_F32P(unsafe_from_address=output.addr)
    if not family_all_finite_f32(words, n) or (n > 0 and family_min_f32(words, n) < 0):
        raise Error("mojolearn " + caller + ": sample_weight must have finite nonnegative entries")
    if n == 0 or not family_max_f32(words, n) > 0:
        raise Error("mojolearn " + caller + ": sample_weight must have positive total weight")
    var result = output.keep
    _ = output^
    return result
