# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host column of x_neighbors/ocsvm_dev.mojo: OneClassSVM's SMO as the
one sequential item `ocsvm_smo_item`. CPU-only installs and the
verification digests only. The device's grid scans pick the item's index
(a total order) and its rho is the same blocked fold, so the two agree."""
from x_neighbors.items import FP, IP, ocsvm_smo_item
from x_neighbors.host_ops import X_NEIGHBORS_HOST_SABOTAGE
from x_neighbors.ocsvm_init import (
    XN_OCSVM_DEV_INIT, oci_chunks, oci_part_item, oci_scan_item, oci_alpha_item, oci_nu_hi, oci_nu_lo,
    XN_UNIT_DEV, unit_ff_item,
)


def op_ocsvm_alpha_init(cv: Int, alpha: Int, n: Int, nu_hi: Float32, nu_lo: Float32) raises:
    """The host column of x_neighbors/ocsvm_dev.mojo `op_ocsvm_alpha_init`:
    the same three stages, each item in turn."""
    if n <= 0:
        return
    var nc = oci_chunks(n)
    var s_p = List[Float32](length=2 * nc, fill=Float32(0))
    var s_o = List[Float32](length=2 * (nc + 1), fill=Float32(0))
    var ph = FP(unsafe_from_address=Int(s_p.unsafe_ptr()))
    var pl = ph + nc
    var oh = FP(unsafe_from_address=Int(s_o.unsafe_ptr()))
    var ol = oh + (nc + 1)
    var p_cv = FP(unsafe_from_address=cv)
    var p_alpha = FP(unsafe_from_address=alpha)
    for c in range(nc):
        oci_part_item(c, p_cv, ph, pl, n)
    for c in range(nc + 1):
        oci_scan_item(c, ph, pl, oh, ol, n, nu_hi, nu_lo)
    for i in range(n):
        oci_alpha_item(i, p_cv, oh, ol, p_alpha, n)
    _ = s_p^
    _ = s_o^


def op_ocsvm(q: Int, cv: Int, alpha: Int, info: Int, iters: Int, n: Int, eps: Float32, max_iter: Int) raises:
    var s_g = List[Float32](length=n if n > 0 else 1, fill=Float32(0))
    ocsvm_smo_item(
        0, FP(unsafe_from_address=q), FP(unsafe_from_address=cv), FP(unsafe_from_address=alpha),
        FP(unsafe_from_address=Int(s_g.unsafe_ptr())), FP(unsafe_from_address=info),
        IP(unsafe_from_address=iters), n, eps, max_iter,
    )
    comptime if X_NEIGHBORS_HOST_SABOTAGE:
        if n > 0:
            var pa = FP(unsafe_from_address=alpha)
            pa.unsafe_store(0, pa.unsafe_load(0) + Float32(1e-3))
    _ = s_g^


def op_unit_ff(v: Int, res: Int, info: Int, n: Int) raises:
    """The host column of x_neighbors/ocsvm_dev.mojo `op_unit_ff`: the same
    items, each in turn."""
    var pinfo = IP(unsafe_from_address=info)
    pinfo.unsafe_store(0, Int32(0))
    pinfo.unsafe_store(1, Int32(0))
    if n <= 0:
        pinfo.unsafe_store(1, Int32(1))
        return
    var nc = oci_chunks(n)
    var s_p = List[Float32](length=2 * nc, fill=Float32(0))
    var s_o = List[Float32](length=2 * (nc + 1), fill=Float32(0))
    var ph = FP(unsafe_from_address=Int(s_p.unsafe_ptr()))
    var pl = ph + nc
    var oh = FP(unsafe_from_address=Int(s_o.unsafe_ptr()))
    var ol = oh + (nc + 1)
    var p_v = FP(unsafe_from_address=v)
    var p_res = FP(unsafe_from_address=res)
    for c in range(nc):
        oci_part_item(c, p_v, ph, pl, n)
    for c in range(nc + 1):
        oci_scan_item(c, ph, pl, oh, ol, n, Float32(1), Float32(0))
    for i in range(n):
        unit_ff_item(i, p_v, oh, ol, p_res, pinfo, n)
    _ = s_p^
    _ = s_o^
