# SPDX-License-Identifier: Apache-2.0
# The compiled owner of the linalg functions of python/mojolearn/_linalg_impl.py (quantize/dequantize, eigh, svd, ...), resample.py and parallel_graph.py (family_lin_, family_rs_, family_pg_ operations): one native call per public method,
# reached from bindings/family_native.mojo for every `family_lin_*`, `family_rs_*`
# and `family_pg_*` operation. The functions live in bindings/linalg_ops_*.mojo.
from std.python import Python, PythonObject
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.linalg_ops_lowbit_door import lin_lowbit_operation
from bindings.linalg_ops_resample_door import rs_operation
from bindings.linalg_ops_decomp_door import lin_decomp_operation
from providers.python.linalg_result_pyapi_door import linalg_result_operation
from bindings.operation_names_linops import (
    LINOPS_OPN_FAMILY_LIN_QR, LINOPS_OPN_FAMILY_LIN_NUMERIC_MODE, LINOPS_OPN_FAMILY_RS_RESULT_INITIALIZE,
    LINOPS_OPN_FAMILY_LIN_FROM_BF16, LINOPS_OPN_FAMILY_LIN_DEQUANTIZE_INT8)


def linalg_ops_unknown_operation(operation: String) raises:
    """The error of a `family_lin_*` / `family_rs_*` / `family_pg_*` name that
    is not an operation (it has no leaf code): the text of the owner its
    prefix selected, as before. Always raises."""
    if operation.startswith("family_lin_result_"):
        raise Error("RuntimeError: unknown linalg result operation")
    if operation.startswith("family_lin_"):
        raise Error("RuntimeError: unknown linalg operation " + operation)
    if operation.startswith("family_rs_"):
        raise Error("RuntimeError: unknown resample operation " + operation)
    raise Error("RuntimeError: unknown family operation " + operation)


def linalg_ops_operation(model: PythonObject, operation: String, args: PythonObject,
         context: PythonObject) raises -> PythonObject:
    """The named form for the one caller that still holds the name: the byte
    LM host trainer's weight widening
    (bindings/byte_lm_host_trainer_native_door.mojo, the byte lane's file;
    only these two names reach it). It maps the name to its leaf code (at
    most two compares; the chain it replaces made up to sixteen) and runs the
    coded form. Delete it when that door passes the code."""
    var code = -1
    if operation == "family_lin_from_bf16":
        code = LINOPS_OPN_FAMILY_LIN_FROM_BF16
    elif operation == "family_lin_dequantize_int8":
        code = LINOPS_OPN_FAMILY_LIN_DEQUANTIZE_INT8
    else:
        classical_connect_numpy(context)
        linalg_ops_unknown_operation(operation)
    return linalg_ops_operation(model, code, args, context)


def linalg_ops_operation(model: PythonObject, operation: Int, args: PythonObject,
         context: PythonObject) raises -> PythonObject:
    # bw2-linalg: `operation` is the leaf code of bindings/operation_names_linops.mojo
    # (results 0-1, decompositions 2-5, low-bit and GEMM 6-20, resampling
    # 21-26), resolved once by the family door; the owner is a range test (it
    # was two prefix tests, four name compares and two more prefix tests).
    classical_connect_numpy(context)
    if operation < LINOPS_OPN_FAMILY_LIN_QR:
        return linalg_result_operation(operation, args, context)
    if operation < LINOPS_OPN_FAMILY_LIN_NUMERIC_MODE:
        return lin_decomp_operation(operation, args, context)
    if operation < LINOPS_OPN_FAMILY_RS_RESULT_INITIALIZE:
        return lin_lowbit_operation(operation, args, context)
    return rs_operation(operation, args, context)
