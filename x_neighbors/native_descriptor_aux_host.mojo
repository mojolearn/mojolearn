# SPDX-License-Identifier: Apache-2.0
# Typed auxiliary transport to the existing backend owners; no numerical work
# moves between devices. The common descriptor provider controls GIL/lifetimes.
from x_neighbors.ocsvm_init import XN_OCSVM_DEV_INIT, XN_UNIT_DEV, oci_nu_hi, oci_nu_lo
from x_neighbors.ocsvm_host import op_ocsvm_alpha_init, op_unit_ff


def xn_aux_capabilities() -> Int:
    return (0 | (Int(XN_OCSVM_DEV_INIT) << 1) | (Int(XN_UNIT_DEV) << 2))


def xn_aux_execute(op: Int, a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    if op == 3:
        return xn_aux_capabilities()
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
    raise Error("x_neighbors auxiliary native ABI: unknown operation")
