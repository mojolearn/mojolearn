# SPDX-License-Identifier: Apache-2.0
"""CPU-only training GEMM boundary; no GPU module is imported."""
from std.python import Python,PythonObject
from std.python._cpython import GILReleased
from std.math import isfinite
from bindings.hostptr import f32_ptr
from bindings.neural_gemm_boundary_common import neural_gemm_params
from gemm.host.neural_gemm import neural_gemm_host_into


def neural_gemm_binding(a_addr: PythonObject,b_addr: PythonObject,c_addr: PythonObject,
                        params: PythonObject) raises -> PythonObject:
    """Addresses A,B,C; params=[m,n,k,op], op NN=0,NT=1,TN=2. Retain nothing."""
    var aa = Int(py=a_addr)
    var ba = Int(py=b_addr)
    var ca = Int(py=c_addr)
    var shape = neural_gemm_params(params,aa,ba,ca)
    var ap = f32_ptr(aa)
    var bp = f32_ptr(ba)
    var cp = f32_ptr(ca)
    var m = shape[0]
    var n = shape[1]
    var k = shape[2]
    with GILReleased(Python()):
        for i in range(m*k):
            if not isfinite(ap.unsafe_load(i)):
                raise Error("SmallMLPTrainer GEMM refuses nonfinite inputs")
        for i in range(n*k):
            if not isfinite(bp.unsafe_load(i)):
                raise Error("SmallMLPTrainer GEMM refuses nonfinite inputs")
        neural_gemm_host_into(cp,ap,bp,m,n,k,shape[3])
        for i in range(m*n):
            if not isfinite(cp.unsafe_load(i)):
                raise Error("SmallMLPTrainer GEMM returned a nonfinite result")
    return PythonObject(m*n)
