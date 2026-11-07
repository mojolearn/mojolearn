# SPDX-License-Identifier: Apache-2.0
# Typed auxiliary transport to the existing backend owners; no numerical work
# moves between devices. The common descriptor provider controls GIL/lifetimes.
from x_neighbors.ocsvm_init import XN_OCSVM_DEV_INIT, XN_UNIT_DEV, oci_nu_hi, oci_nu_lo
from x_neighbors.ocsvm_dev import op_ocsvm_alpha_init, op_unit_ff, op_ocsvm_x, OCSVM_IDN_RES
from x_neighbors.kfeat_rng import XN_IDN_SKETCH_CTR
from x_neighbors.kfeat_dev import kfeat_pcs_draw_idn
from x_neighbors.iter_device import op_ocsvm_labels
# Capability bit 4: this binary carries the ocsvm_labels auxiliary operation.
comptime XN_AUX_OCSVM_LABELS = True


def xn_aux_capabilities() -> Int:
    return (Int(OCSVM_IDN_RES) | (Int(XN_OCSVM_DEV_INIT) << 1) | (Int(XN_UNIT_DEV) << 2)
            | (Int(XN_IDN_SKETCH_CTR) << 3) | (Int(XN_AUX_OCSVM_LABELS) << 4))


def xn_aux_execute(op: Int, a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    if op == 3:
        return xn_aux_capabilities()
    if op == 4:
        comptime if not XN_IDN_SKETCH_CTR:
            raise Error("AttributeError: x_neighbors_kfeat_pcs_draw_idn is not exported by this binding")
        # The unchanged device owner validates dimensions, owns its uploads,
        # counter-based kernel, downloads and final synchronization.
        if a[0] == 0 or a[1] == 0:
            raise Error("x_neighbors_kfeat_pcs_draw_idn: null buffer address")
        kfeat_pcs_draw_idn(p[0], p[1], p[2], a[0], a[1])
        return 0
    if op == 0:
        comptime if not XN_OCSVM_DEV_INIT:
            raise Error("AttributeError: x_neighbors_ocsvm_alpha_init is not exported by this binding")
        var cv = a[0]
        var alpha = a[1]
        var n = p[0]
        if n < 0:
            raise Error("x_neighbors: a negative size was passed")
        if n > 0 and (cv == 0 or alpha == 0):
            raise Error("x_neighbors: null buffer address")
        var nu = f[0]
        op_ocsvm_alpha_init(cv, alpha, n, oci_nu_hi(nu), oci_nu_lo(nu))
        return 0
    if op == 1:
        comptime if not XN_UNIT_DEV:
            raise Error("AttributeError: x_neighbors_unit_ff is not exported by this binding")
        var v = a[0]
        var res = a[1]
        var info = a[2]
        var n = p[0]
        if n < 0:
            raise Error("x_neighbors: a negative size was passed")
        if info == 0 or (n > 0 and (v == 0 or res == 0)):
            raise Error("x_neighbors: null buffer address")
        op_unit_ff(v, res, info, n)
        return 0
    if op == 2:
        comptime if not OCSVM_IDN_RES:
            raise Error("AttributeError: x_neighbors_ocsvm_resident_idn is not exported by this binding")
        for k in range(5):  # small-loop(five admitted address and dimension words)
            if a[k] == 0:
                raise Error("x_neighbors: null buffer address")
            if p[k] < 0:
                raise Error("x_neighbors: a negative size was passed")
        var gamma = Float32(f[0])
        var coef0 = Float32(f[1])
        var eps = Float32(f[2])
        op_ocsvm_x(a[0], a[1], a[2], a[3], a[4], p[0], p[1], p[2], gamma, coef0, p[3], eps, p[4])
        return 0
    if op == 5:
        # ocsvm_labels (lane ne3-g-svc-gp-neighbors, N369): addresses
        # (q, support rows, dual column, int32 labels), ints (n, m, d, kind,
        # degree, sign mode), floats (gamma, coef0, scale, shift, threshold).
        for k in range(4):  # small-loop(four admitted address words)
            if a[k] == 0:
                raise Error("x_neighbors: null buffer address")
        for k in range(6):  # small-loop(six admitted dimension words)
            if p[k] < 0:
                raise Error("x_neighbors: a negative size was passed")
        op_ocsvm_labels(a[0], a[1], a[2], a[3], p[0], p[1], p[2], p[3], p[4], Float32(f[0]), Float32(f[1]),
                        Float32(f[2]), Float32(f[3]), p[5], Float32(f[4]))
        return 0
    raise Error("x_neighbors auxiliary native ABI: unknown operation")
