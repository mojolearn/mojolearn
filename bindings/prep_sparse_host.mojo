# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""CPython adapter only; native CPU API is x_prep.host.sparse_input."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.prep_sparse_common import sparse_shape, sparse_view, sparse_check_outputs
from x_prep.sparse_input import IP, FP
from x_prep.host.sparse_input import sparse_to_csr_host, sparse_dense_nnz_host


def sparse_csr_binding(sizes: PythonObject, am: PythonObject, bm: PythonObject, vm: PythonObject,
                       outputs: PythonObject) raises -> PythonObject:
    sparse_check_outputs(outputs)
    var s = sparse_shape(sizes)
    var a = sparse_view(am, True)
    var b = sparse_view(bm, True)
    var v = sparse_view(vm)
    var ip = IP(unsafe_from_address=Int(py=outputs[0]))
    var ix = IP(unsafe_from_address=Int(py=outputs[1]))
    var dv = FP(unsafe_from_address=Int(py=outputs[2]))
    var capacity = Int(py=outputs[3])
    var nnz = 0
    with GILReleased(Python()):
        nnz = sparse_to_csr_host(s, a, b, v, ip, ix, dv, capacity)
    return PythonObject(nnz)


def sparse_capacity_binding(sizes: PythonObject, vm: PythonObject) raises -> PythonObject:
    var s = sparse_shape(sizes)
    var v = sparse_view(vm)
    var nnz = 0
    with GILReleased(Python()):
        nnz = sparse_dense_nnz_host(s, v)
    return PythonObject(nnz)
