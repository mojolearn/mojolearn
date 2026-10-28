# SPDX-License-Identifier: Apache-2.0
"""Borrowed-pointer admission for neural-only NN/NT/TN model products."""
from std.python import PythonObject
from checks.numerics import GLOBAL_NUMERIC_MODE,NUMERIC_IDENTICAL
from training.neural_gemm_admission import neural_gemm_shape


def neural_gemm_params(params: PythonObject,a: Int,b: Int,c: Int) raises -> Tuple[Int,Int,Int,Int]:
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("neural_gemm requires the IDENTICAL training binding")
    if len(params)!=4:
        raise Error("neural_gemm params must be [m,n,k,op]")
    var m = Int(py=params[0])
    var n = Int(py=params[1])
    var k = Int(py=params[2])
    var op = Int(py=params[3])
    return neural_gemm_shape(m, n, k, op, a, b, c)
