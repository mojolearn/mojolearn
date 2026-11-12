# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Python entry points for `svm/impl/svc_ovo_layout.mojo` (the shell): they
unpack PythonObject arguments, call the typed device kernels and box the
result. `ovo_pairs_from_python` is shared with the host column's shell."""

# The PACK half of the former `providers/python/svm_impl_svc_ovo_layout_pyapi.mojo` (fence-flip split, 2026-11-12);
# the UNPACK half is `providers/python/svm_impl_svc_ovo_layout_pyapi_unpack.mojo`.
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from max.gpu.host import DeviceContext

from svm.impl.svc_ovo_layout import (
    OvoPairs,
    ovo_check_result,
    ovo_dual_gemv_device,
    ovo_layout_device,
)


from providers.python.svm_impl_svc_ovo_layout_pyapi_unpack import (
    ovo_pairs_from_python,
)


def svc_ovo_layout_device_binding(
    ctx: DeviceContext, sup_addrs: PythonObject, dual_addrs: PythonObject, meta: PythonObject,
    out_addrs: PythonObject,
) raises -> PythonObject:
    """`svc_ovo_layout(sup_addrs, dual_addrs, meta, out_addrs)`: meta
    [k, n_bound, P, (i, j, n_support) per pair]; out_addrs [support (int32,
    cap), n_support (int32, k), dual_coef (float32, (k - 1) * cap), cap].
    Returns n_sv (dual_coef_ written with row stride n_sv)."""
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
        n_sv = ovo_layout_device(ctx, pairs, k, n_bound, cap, sa, na, da)
    ovo_check_result(n_sv)
    return PythonObject(n_sv)


def svc_dual_gemv_device_binding(
    ctx: DeviceContext, dual_addr: PythonObject, sv_addr: PythonObject, dims: PythonObject,
    out_addr: PythonObject,
) raises -> PythonObject:
    """`svc_dual_gemv(dual, sv, [n_sv, d(, negate)], out)`: coef_ = dual @ sv
    (one row), float32 out of d, negated (exact) when `negate` (the OvO
    pair rows in scikit-learn's orientation; lane pyglue-numeric: Python
    negated them). Returns d."""
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
    with GILReleased(Python()):
        ovo_dual_gemv_device(ctx, dl, sl, n_sv, d, neg, oa)
    return PythonObject(d)
