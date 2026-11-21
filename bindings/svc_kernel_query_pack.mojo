# SPDX-License-Identifier: Apache-2.0
# PACK of the kernel SVC's inference and of `coef_`
# (bindings/svc_kernel_native_door.mojo sequences the operations): the
# public result array RUN's last launch writes in place, allocated before
# RUN, and what remains after it (the row argmax of break_ties).
from std.python import Python, PythonObject
from bindings.family_common_pack import family_array
from bindings.svc_common_pack import svc_argmax_rows
from bindings.svc_kernel_rows_unpack import _addr
from bindings.svc_kernel_query_unpack import SvcQueryUnpacked, SVC_RESULT_F32_ROWS, SVC_RESULT_F32_PAIRS, SVC_RESULT_CODES, SVC_RESULT_ARGMAX


def svc_query_result(mut unpacked: SvcQueryUnpacked, context: PythonObject) raises -> PythonObject:
    """The public array of the operation, and the plan pointed at it: the
    binary decisions are written by the machine itself, every other result
    by the epilogue."""
    var n = unpacked.plan.rows
    if unpacked.result == SVC_RESULT_F32_ROWS:
        var decisions = family_array(Python.tuple(PythonObject(n)), "<f4", context)
        unpacked.plan.decisions = _addr(decisions)
        return decisions
    var result: PythonObject
    if unpacked.result == SVC_RESULT_F32_PAIRS:
        result = family_array(Python.tuple(PythonObject(n), PythonObject(unpacked.plan.pairs)), "<f4", context)
    elif unpacked.result == SVC_RESULT_CODES:
        result = family_array(Python.tuple(PythonObject(n)), "<i8", context)
    else:
        result = family_array(Python.tuple(PythonObject(n), PythonObject(unpacked.plan.classes)), "<f8", context)
    unpacked.plan.dst = _addr(result)
    return result


def svc_query_pack(unpacked: SvcQueryUnpacked, result: PythonObject, context: PythonObject) raises -> PythonObject:
    """After RUN: the 'ovr' scores of break_ties become their row argmax
    (int64 codes); every other result is already the public array."""
    if unpacked.result == SVC_RESULT_ARGMAX:
        return svc_argmax_rows(result, context)
    return result


def svc_coef_output(machines: Int, features: Int, context: PythonObject) raises -> PythonObject:
    """`coef_`: float32 (machines x features), one GEMV row per machine."""
    return family_array(Python.tuple(machines, features), "<f4", context)
