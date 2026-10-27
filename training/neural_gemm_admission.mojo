# SPDX-License-Identifier: Apache-2.0
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from gemm.experiments.neural_profile import neural_validate

def neural_gemm_shape(m: Int, n: Int, k: Int, op: Int, a: Int, b: Int, c: Int) raises -> Tuple[Int, Int, Int, Int]:
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("neural_gemm requires the IDENTICAL training binding")
    neural_validate(m,n,k,op)
    if m<1 or n<1 or k<1:
        raise Error("neural_gemm requires positive matrix dimensions")
    if a==0 or b==0 or c==0:
        raise Error("neural_gemm refuses null borrowed buffers")
    if (c<a+4*m*k and a<c+4*m*n) or (c<b+4*n*k and b<c+4*m*n):
        raise Error("neural_gemm output must not overlap an input")
    return (m,n,k,op)
