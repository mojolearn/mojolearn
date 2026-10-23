# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python door of `x_cnn/host/ops_host.mojo`'s graph builders for the
host binding (`bindings/_mojolearn_x_cnn_host.mojo`): PythonObject arguments
to typed pointers, the GIL released around the typed call. HOST ONLY."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from core.hostptr import i32_ptr, f32_ptr
from x_cnn.host.ops_host import csr_build_host, gcn_loops_host, perm_gather_host, vals_t_host


def csr_build_host_binding(rows_addr: PythonObject, cols_addr: PythonObject, csr_addr: PythonObject, order_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    """`x_cnn_csr_build` for the host binding (`bindings/_mojolearn_x_cnn.mojo`'s
    `csr_build_binding`, same arguments, same words)."""
    var n = Int(py=params[0])
    var nnz = Int(py=params[1])
    if n <= 0 or nnz < 0:
        raise Error("x_cnn csr_build: positive n and nnz >= 0 required")
    var pc = i32_ptr(Int(py=csr_addr)).unsafe_origin_cast[MutAnyOrigin]()
    var rows = i32_ptr(Int(py=rows_addr)).unsafe_origin_cast[MutAnyOrigin]() if nnz > 0 else pc
    var cols = i32_ptr(Int(py=cols_addr)).unsafe_origin_cast[MutAnyOrigin]() if nnz > 0 else pc
    var order = i32_ptr(Int(py=order_addr)).unsafe_origin_cast[MutAnyOrigin]() if nnz > 0 else pc
    with GILReleased(Python()):
        csr_build_host(rows, cols, nnz, n, pc, order)
    return PythonObject(n + 1 + 2 * nnz)


def gcn_loops_host_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """`x_cnn_gcn_loops` for the host binding (`bindings/_mojolearn_x_cnn.mojo`'s
    `gcn_loops_binding`, same arguments, same words)."""
    var n = Int(py=params[0])
    var nnz = Int(py=params[1])
    var fill = Float32(2.0) if Int(py=params[2]) != 0 else Float32(1.0)
    if n <= 0 or nnz < 0 or Int(py=len(addrs)) != 6:
        raise Error("x_cnn gcn_loops: addrs [src, dst, w, src_out, dst_out, w_out], positive n and nnz >= 0 required")
    var so = i32_ptr(Int(py=addrs[3])).unsafe_origin_cast[MutAnyOrigin]()
    var dso = i32_ptr(Int(py=addrs[4])).unsafe_origin_cast[MutAnyOrigin]()
    var wo = f32_ptr(Int(py=addrs[5])).unsafe_origin_cast[MutAnyOrigin]()
    var src = i32_ptr(Int(py=addrs[0])).unsafe_origin_cast[MutAnyOrigin]() if nnz > 0 else so
    var dst = i32_ptr(Int(py=addrs[1])).unsafe_origin_cast[MutAnyOrigin]() if nnz > 0 else so
    var w = f32_ptr(Int(py=addrs[2])).unsafe_origin_cast[MutAnyOrigin]() if nnz > 0 else wo
    var k = 0
    with GILReleased(Python()):
        k = gcn_loops_host(src, dst, w, nnz, n, fill, so, dso, wo)
    return PythonObject(k)


def perm_gather_host_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """`x_cnn_perm_gather` for the host binding: addrs = [src f32, idx i32,
    out f32], params = [n_src, n]; out[i] = src[idx[i]]. Returns n."""
    var n_src = Int(py=params[0])
    var n = Int(py=params[1])
    if n < 0 or n_src < 0:
        raise Error("x_cnn perm_gather: n >= 0 required")
    if n == 0:
        return PythonObject(0)
    var src = f32_ptr(Int(py=addrs[0])).unsafe_origin_cast[MutAnyOrigin]()
    var idx = i32_ptr(Int(py=addrs[1])).unsafe_origin_cast[MutAnyOrigin]()
    var out = f32_ptr(Int(py=addrs[2])).unsafe_origin_cast[MutAnyOrigin]()
    with GILReleased(Python()):
        perm_gather_host(src, idx, out, n)
    return PythonObject(n)


def vals_t_host_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """`x_cnn_vals_t` for the host binding: addrs = [vals_f f32, order_f i32,
    order_t i32, out f32], params = [nnz]. Returns nnz."""
    var nnz = Int(py=params[0])
    if nnz < 0:
        raise Error("x_cnn vals_t: nnz >= 0 required")
    if nnz == 0:
        return PythonObject(0)
    var vals = f32_ptr(Int(py=addrs[0])).unsafe_origin_cast[MutAnyOrigin]()
    var of = i32_ptr(Int(py=addrs[1])).unsafe_origin_cast[MutAnyOrigin]()
    var ot = i32_ptr(Int(py=addrs[2])).unsafe_origin_cast[MutAnyOrigin]()
    var out = f32_ptr(Int(py=addrs[3])).unsafe_origin_cast[MutAnyOrigin]()
    with GILReleased(Python()):
        vals_t_host(vals, of, ot, out, nnz)
    return PythonObject(nnz)
