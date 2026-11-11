# SPDX-License-Identifier: Apache-2.0
# Typed auxiliary transport to the existing backend owners; no numerical work
# moves between devices. The common descriptor provider controls GIL/lifetimes.
from x_neighbors.ocsvm_init import XN_OCSVM_DEV_INIT, XN_UNIT_DEV, oci_nu_hi, oci_nu_lo
from x_neighbors.ocsvm_host import op_ocsvm_alpha_init, op_unit_ff
from x_neighbors.kfeat_rng import XN_IDN_SKETCH_CTR, op_kfeat_pcs_draw_idn_host
from x_neighbors.items import IP
from x_neighbors.iter_host import op_ocsvm_labels
# Capability bit 4: this binary carries the ocsvm_labels auxiliary operation.
comptime XN_AUX_OCSVM_LABELS = True
from x_neighbors.host_ops import op_nc_labels, op_nc_proba, XN_NC_SOFTMAX64, op_lp_dense_proba
from x_neighbors.proba64_nc import NC_PROBA64
# Capability bit 5: this binary carries nc_labels and nc_proba.
comptime XN_AUX_NC_FUSED = True
# Capability bit 6: this binary carries lp_dense_proba.
comptime XN_AUX_LP_DENSE = True


# Capability bit 7 retains the existing NearestCentroid Float64 probability switch.
def xn_aux_capabilities() -> Int:
    return (0 | (Int(XN_OCSVM_DEV_INIT) << 1) | (Int(XN_UNIT_DEV) << 2)
            | (Int(XN_IDN_SKETCH_CTR) << 3) | (Int(XN_AUX_OCSVM_LABELS) << 4)
            | (Int(XN_AUX_NC_FUSED) << 5) | (Int(XN_AUX_LP_DENSE) << 6) | (Int(NC_PROBA64) << 7))


def xn_aux_execute(op: Int, a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    if op == 3:
        return xn_aux_capabilities()
    if op == 4:
        comptime if not XN_IDN_SKETCH_CTR:
            raise Error("AttributeError: x_neighbors_kfeat_pcs_draw_idn is not exported by this binding")
        var seed = UInt32(p[0] & 0xFFFFFFFF)
        var nc = p[1]
        var total = p[2]
        # Preserve the retained host wrapper's admission, including its
        # components * total bound. The device owner has its own total bound.
        if nc <= 0 or total <= 0:
            raise Error("x_neighbors_kfeat_pcs_draw_idn: sizes must be positive")
        if nc * total > 2147483647:
            raise Error("x_neighbors_kfeat_pcs_draw_idn: more than 2^31 - 1 draws")
        if a[0] == 0 or a[1] == 0:
            raise Error("x_neighbors_kfeat_pcs_draw_idn: null buffer address")
        op_kfeat_pcs_draw_idn_host(seed, nc, total,
                                 IP(unsafe_from_address=a[0]), IP(unsafe_from_address=a[1]))
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
        raise Error("AttributeError: x_neighbors_ocsvm_resident_idn is not exported by this host binding")
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
    if op == 6 or op == 7:
        # nc_labels (6) / nc_proba (7) (lane ne3-g-svc-gp-neighbors, N327):
        # addresses (q, centroids, std, prior, out), ints (n, d, classes,
        # mode). std and prior are read by the decision modes only; the
        # caller passes a live address for each.
        for k in range(5):  # small-loop(five admitted address words)
            if a[k] == 0:
                raise Error("x_neighbors: null buffer address")
        for k in range(4):  # small-loop(four admitted dimension words)
            if p[k] < 0:
                raise Error("x_neighbors: a negative size was passed")
        # The optional fifth int (lane a8b-device-x-neighbors-rest): the host
        # address of q's finite status word (0: not asked).
        var status = p[4] if len(p) > 4 else 0
        if op == 6:
            if p[3] > 2:
                raise Error("x_neighbors nc_labels: unknown mode")
            op_nc_labels(a[0], a[1], a[2], a[3], a[4], p[0], p[1], p[2], p[3], status)
            return 0
        if p[3] > XN_NC_SOFTMAX64:
            raise Error("x_neighbors nc_proba: unknown mode")
        if p[3] == XN_NC_SOFTMAX64:
            comptime if not NC_PROBA64:
                raise Error("AttributeError: xn_softmax64 is not exported by this binding")
        op_nc_proba(a[0], a[1], a[2], a[3], a[4], p[0], p[1], p[2], p[3], status)
        return 0
    if op == 8:
        # lp_dense_proba (lane ne3-g-svc-gp-neighbors, N324): addresses
        # (q, X_, label_distributions_, proba out, int32 codes out), ints
        # (nq, n, d, classes, kind, want_codes), floats (gamma, coef0). With
        # want_codes only the codes are written, else only the proba.
        for k in range(5):  # small-loop(five admitted address words)
            if a[k] == 0:
                raise Error("x_neighbors: null buffer address")
        for k in range(6):  # small-loop(six admitted dimension words)
            if p[k] < 0:
                raise Error("x_neighbors: a negative size was passed")
        # The optional seventh int (lane a8b-device-x-neighbors-rest): the
        # host address of q's finite status word (0: not asked).
        var status = p[6] if len(p) > 6 else 0
        op_lp_dense_proba(a[0], a[1], a[2], a[3], a[4], p[0], p[1], p[2], p[3], p[4], Float32(f[0]), Float32(f[1]),
                          p[5], status)
        return 0
    raise Error("x_neighbors auxiliary native ABI: unknown operation")
