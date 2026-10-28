# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""SVC's one-vs-one attribute layout and `coef_`, THE HOST COLUMN
(lane/apple-fast-py2mojo-linear): `svm/impl/svc_ovo_layout.mojo`'s rules as
plain loops for the CPU binding, the same words (integer placement, and the
same ascending binary64 chain per feature for `coef_`). The Python entry points
are in `providers/python/svm_host_svc_ovo_host_pyapi.mojo`."""

from svm.impl.svc_ovo_layout import (
    OvoPairs,
    dual_gemv_column,
    ovo_check_result,
    ovo_dual_row,
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


def ovo_dual_gemv_host(dl: Int, sl: Int, n_sv: Int, d: Int, neg: Int, oa: Int):
    """The host column's coef_ = dual @ sv (one row), the device's chain."""
    var dst = _F32P(unsafe_from_address=oa)
    var dp = _F32P(unsafe_from_address=dl if n_sv > 0 else oa)
    var sp = _F32P(unsafe_from_address=sl if n_sv > 0 else oa)
    for j in range(d):
        var v = dual_gemv_column(dp, sp, n_sv, d, j)
        dst[j] = -v if neg != 0 else v
