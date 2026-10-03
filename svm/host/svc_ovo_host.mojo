# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""SVC's one-vs-one attribute layout and `coef_`, THE HOST COLUMN
(lane/apple-fast-py2mojo-linear): `svm/impl/svc_ovo_layout.mojo`'s rules as
plain loops for the CPU binding, the same words (integer placement, and the
same ascending binary64 chain per feature for `coef_`)."""

from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from svm.impl.svc_ovo_layout import (
    OvoPairs,
    dual_gemv_column,
    ovo_check_result,
    ovo_dual_row,
    ovo_pairs_from_python,
    ovo_side,
)

comptime _I32P = MutPointer[Int32, MutAnyOrigin]
comptime _F32P = MutPointer[Float32, MutAnyOrigin]


def ovo_layout_host(
    pairs: OvoPairs, k: Int, n_bound: Int, cap: Int, support_addr: Int, nsup_addr: Int, dual_addr: Int,
) -> Int:
    var cls = List[Int](length=max(1, n_bound), fill=-1)
    for q in range(len(pairs.m)):
        if pairs.m[q] == 0:
            continue
        var sp = _I32P(unsafe_from_address=pairs.sup[q])
        var dp = _F32P(unsafe_from_address=pairs.dual[q])
        for e in range(pairs.m[q]):
            var r = Int(sp[e])
            if r < 0 or r >= n_bound:
                return -1
            cls[r] = ovo_side(dp[e], pairs.ci[q], pairs.cj[q])
    var colof = List[Int](length=max(1, n_bound), fill=-1)
    var support = _I32P(unsafe_from_address=support_addr)
    var nsup = _I32P(unsafe_from_address=nsup_addr)
    var n_sv = 0
    for c in range(k):
        var start = n_sv
        for r in range(n_bound):
            if cls[r] == c:
                if n_sv >= cap:
                    return -2
                colof[r] = n_sv
                support[n_sv] = Int32(r)
                n_sv += 1
        nsup[c] = Int32(n_sv - start)
    var dual = _F32P(unsafe_from_address=dual_addr)
    for t in range((k - 1) * n_sv):
        dual[t] = Float32(0.0)
    for q in range(len(pairs.m)):
        if pairs.m[q] == 0:
            continue
        var sp = _I32P(unsafe_from_address=pairs.sup[q])
        var dp = _F32P(unsafe_from_address=pairs.dual[q])
        for e in range(pairs.m[q]):
            var d = dp[e]
            var side = ovo_side(d, pairs.ci[q], pairs.cj[q])
            var row = ovo_dual_row(side, pairs.ci[q], pairs.cj[q])
            dual[row * n_sv + colof[Int(sp[e])]] = -d
    return n_sv


def svc_ovo_layout_host_binding(
    sup_addrs: PythonObject, dual_addrs: PythonObject, meta: PythonObject, out_addrs: PythonObject,
) raises -> PythonObject:
    """The CPU binding's `svc_ovo_layout`, the GPU binding's contract."""
    var pairs = ovo_pairs_from_python(sup_addrs, dual_addrs, meta)
    var k = Int(py=meta[0])
    var n_bound = Int(py=meta[1])
    if len(out_addrs) != 4:
        raise Error("svc_ovo_layout: out_addrs [support, n_support, dual_coef, cap]")
    var sa = Int(py=out_addrs[0])
    var na = Int(py=out_addrs[1])
    var da = Int(py=out_addrs[2])
    var cap = Int(py=out_addrs[3])
    if k < 2 or n_bound < 0 or sa == 0 or na == 0 or da == 0:
        raise Error("svc_ovo_layout: needs two classes and three output buffers")
    var n_sv = 0
    with GILReleased(Python()):
        n_sv = ovo_layout_host(pairs, k, n_bound, cap, sa, na, da)
    ovo_check_result(n_sv)
    return PythonObject(n_sv)


def svc_dual_gemv_host_binding(
    dual_addr: PythonObject, sv_addr: PythonObject, dims: PythonObject, out_addr: PythonObject,
) raises -> PythonObject:
    """The CPU binding's `svc_dual_gemv`."""
    var n_sv = Int(py=dims[0])
    var d = Int(py=dims[1])
    var oa = Int(py=out_addr)
    if n_sv < 0 or d < 0 or oa == 0:
        raise Error("svc_dual_gemv: bad dims or null output")
    if d == 0:
        return PythonObject(0)
    var dl = Int(py=dual_addr)
    var sl = Int(py=sv_addr)
    if n_sv > 0 and (dl == 0 or sl == 0):
        raise Error("svc_dual_gemv: null input")
    var dst = _F32P(unsafe_from_address=oa)
    with GILReleased(Python()):
        var dp = _F32P(unsafe_from_address=dl if n_sv > 0 else oa)
        var sp = _F32P(unsafe_from_address=sl if n_sv > 0 else oa)
        for j in range(d):
            dst[j] = dual_gemv_column(dp, sp, n_sv, d, j)
    return PythonObject(d)
