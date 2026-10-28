# SPDX-License-Identifier: Apache-2.0
"""Training binding's neural-only GEMM; preparation and output stay in-call."""
from std.python import Python,PythonObject
from std.python._cpython import GILReleased
from training.neural_gemm_native import neural_gemm_native
from bindings.neural_gemm_boundary_common import neural_gemm_params

def neural_gemm_binding(a_addr: PythonObject,b_addr: PythonObject,c_addr: PythonObject,
                        params: PythonObject) raises -> PythonObject:
    """Addresses A,B,C; params=[m,n,k,op], op NN=0,NT=1,TN=2. Retain nothing."""
    var aa = Int(py=a_addr)
    var ba = Int(py=b_addr)
    var ca = Int(py=c_addr)
    var shape = neural_gemm_params(params,aa,ba,ca)
    var count = 0
    with GILReleased(Python()):
        count = neural_gemm_native(aa, ba, ca, shape)
    return PythonObject(count)
