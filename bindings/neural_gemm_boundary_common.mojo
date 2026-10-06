# SPDX-License-Identifier: Apache-2.0
"""Borrowed-pointer admission for neural-only NN/NT/TN model products."""
from std.python import PythonObject
from checks.numerics import GLOBAL_NUMERIC_MODE,NUMERIC_IDENTICAL
from gemm.experiments.neural_profile import neural_validate


def neural_gemm_params(params: PythonObject,a: Int,b: Int,c: Int) raises -> Tuple[Int,Int,Int,Int]:
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("neural_gemm requires the IDENTICAL training binding")
    if len(params)!=4:
        raise Error("neural_gemm params must be [m,n,k,op]")
    var m = Int(py=params[0])
    var n = Int(py=params[1])
    var k = Int(py=params[2])
    var op = Int(py=params[3])
    neural_validate(m,n,k,op)
    if m<1 or n<1 or k<1:
        raise Error("neural_gemm requires positive matrix dimensions")
    if a==0 or b==0 or c==0:
        raise Error("neural_gemm refuses null borrowed buffers")
    if (c<a+4*m*k and a<c+4*m*n) or (c<b+4*n*k and b<c+4*m*n):
        raise Error("neural_gemm output must not overlap an input")
    return (m,n,k,op)
