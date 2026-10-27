# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Python entry points for `svm/host/svc_ovo_host.mojo` (the shell): they
unpack PythonObject arguments, call the typed host kernels and box the result."""

from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from svm.host.svc_ovo_host import ovo_dual_gemv_host, ovo_layout_host
from svm.impl.svc_ovo_layout import ovo_check_result
from providers.python.svm_impl_svc_ovo_layout_pyapi import ovo_pairs_from_python


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
    """The CPU binding's `svc_dual_gemv` (dims [n_sv, d(, negate)])."""
    var n_sv = Int(py=dims[0])
    var d = Int(py=dims[1])
    var neg = Int(py=dims[2]) if len(dims) > 2 else 0
    var oa = Int(py=out_addr)
    if n_sv < 0 or d < 0 or oa == 0:
        raise Error("svc_dual_gemv: bad dims or null output")
    if d == 0:
        return PythonObject(0)
    var dl = Int(py=dual_addr)
    var sl = Int(py=sv_addr)
    if n_sv > 0 and (dl == 0 or sl == 0):
        raise Error("svc_dual_gemv: null input")
    with GILReleased(Python()):
        ovo_dual_gemv_host(dl, sl, n_sv, d, neg, oa)
    return PythonObject(d)
