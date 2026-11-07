# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python door of the x_neighbors host kernels the CPU binding
(`bindings/_mojolearn_x_neighbors_host.mojo`) exports by hand: PythonObject
arguments to typed values and back, and argument checks. No arithmetic on
data; every number comes from the typed host kernels in x_neighbors/.
HOST ONLY."""
from std.python import PythonObject
from x_neighbors.items import FP, IP
from x_neighbors.iter_host import lp_fast_resident
from x_neighbors.kapprox_host import kpca_resident
from x_neighbors.kfeat_rng import op_kfeat_pcs_draw_idn_host, op_kfeat_schi2_fit_idn_host
from x_neighbors.ocsvm_host import op_ocsvm_alpha_init, op_unit_ff
from x_neighbors.ocsvm_init import oci_nu_hi, oci_nu_lo
from x_neighbors.native_descriptor_aux_host import xn_aux_execute


def kpca_resident_binding() raises -> PythonObject:
    """0: the host column never takes the resident KernelPCA path."""
    return PythonObject(kpca_resident())


def lp_fast_resident_binding() raises -> PythonObject:
    """0: the host column never takes the resident kNN-graph loop."""
    return PythonObject(lp_fast_resident())


def _kfeat_args(p: PythonObject, what: String) raises -> Tuple[UInt32, Int, Int]:
    var seed = UInt32(Int(py=p[0]) & 0xFFFFFFFF)
    var a = Int(py=p[1])
    var b = Int(py=p[2])
    if a <= 0 or b <= 0:
        raise Error("x_neighbors_" + what + ": sizes must be positive")
    if a * b > 2147483647:
        raise Error("x_neighbors_" + what + ": more than 2^31 - 1 draws")
    return (seed, a, b)


def kfeat_schi2_fit_idn_host_binding(p: PythonObject, w_out: PythonObject, off_out: PythonObject) raises -> PythonObject:
    """p = [seed & 2^32-1, d, n_components]: `random_weights_` (d x nc float32
    at w_out) and `random_offset_` (nc at off_out). Returns 0."""
    var a = _kfeat_args(p, "kfeat_schi2_fit_idn")
    var w = FP(unsafe_from_address=Int(py=w_out))
    var off = FP(unsafe_from_address=Int(py=off_out))
    if Int(w) == 0 or Int(off) == 0:
        raise Error("x_neighbors_kfeat_schi2_fit_idn: null buffer address")
    op_kfeat_schi2_fit_idn_host(a[0], a[1], a[2], w, off)
    return PythonObject(0)


def kfeat_pcs_draw_idn_host_binding(p: PythonObject, idx_out: PythonObject, sgn_out: PythonObject) raises -> PythonObject:
    """p = [seed & 2^32-1, n_components, degree * n_features]: `indexHash_`
    and `bitHash_` (int32, flat) at idx_out / sgn_out. Returns 0."""
    var a = _kfeat_args(p, "kfeat_pcs_draw_idn")
    var idx = IP(unsafe_from_address=Int(py=idx_out))
    var sgn = IP(unsafe_from_address=Int(py=sgn_out))
    if Int(idx) == 0 or Int(sgn) == 0:
        raise Error("x_neighbors_kfeat_pcs_draw_idn: null buffer address")
    op_kfeat_pcs_draw_idn_host(a[0], a[1], a[2], idx, sgn)
    return PythonObject(0)



def ocsvm_alpha_init_binding(a_: PythonObject, i_: PythonObject, f_: PythonObject) raises -> PythonObject:
    """x_neighbors_ocsvm_alpha_init (host column): addresses (cv, alpha),
    ints (n,), floats (nu,)."""
    var cv = Int(py=a_[0])
    var alpha = Int(py=a_[1])
    var n = Int(py=i_[0])
    if n < 0:
        raise Error("x_neighbors: a negative size was passed")
    if n > 0 and (cv == 0 or alpha == 0):
        raise Error("x_neighbors: null buffer address")
    var nu = Float64(py=f_[0])
    op_ocsvm_alpha_init(cv, alpha, n, oci_nu_hi(nu), oci_nu_lo(nu))
    return PythonObject(None)


def unit_ff_binding(a_: PythonObject, i_: PythonObject, f_: PythonObject) raises -> PythonObject:
    """x_neighbors_unit_ff (host column): addresses (v, res, info), ints (n,)."""
    var v = Int(py=a_[0])
    var res = Int(py=a_[1])
    var info = Int(py=a_[2])
    var n = Int(py=i_[0])
    if n < 0:
        raise Error("x_neighbors: a negative size was passed")
    if info == 0 or (n > 0 and (v == 0 or res == 0)):
        raise Error("x_neighbors: null buffer address")
    op_unit_ff(v, res, info, n)
    return PythonObject(None)


def _aux_call(op: Int, a_: PythonObject, i_: PythonObject, f_: PythonObject, na: Int, ni: Int,
              nf: Int) raises:
    """One auxiliary descriptor operation (x_neighbors/native_descriptor_aux_*)
    from the Python door: the fixed address, dimension and scalar words of
    the named operation, then its typed admission and owner."""
    var a = List[Int]()
    var p = List[Int]()
    var f = List[Float64]()
    for k in range(na):  # small-loop(the fixed address words of the named operation)
        a.append(Int(py=a_[k]))
    for k in range(ni):  # small-loop(the fixed dimension words of the named operation)
        p.append(Int(py=i_[k]))
    for k in range(nf):  # small-loop(the fixed scalar words of the named operation)
        f.append(Float64(py=f_[k]))
    _ = xn_aux_execute(op, a, p, f)


def ocsvm_labels_binding(a_: PythonObject, i_: PythonObject, f_: PythonObject) raises -> PythonObject:
    """x_neighbors_ocsvm_labels (host column): OneClassSVM.predict's kernel_matmul,
    unary and sign-label ops in one native session. Addresses (q, support
    rows, dual column, int32 labels), ints (n, m, d, kind, degree, sign
    mode), floats (gamma, coef0, scale, shift, threshold)."""
    _aux_call(5, a_, i_, f_, 4, 6, 5)
    return PythonObject(None)


def nc_labels_binding(a_: PythonObject, i_: PythonObject, f_: PythonObject) raises -> PythonObject:
    """x_neighbors_nc_labels (host column): NearestCentroid.predict's codes in one
    native session. Addresses (q, centroids, std, prior, int32 codes), ints
    (n, d, classes, mode: 0 sqdist, 1 l1dist, 2 decision)."""
    _aux_call(6, a_, i_, f_, 5, 4, 0)
    return PythonObject(None)


def nc_proba_binding(a_: PythonObject, i_: PythonObject, f_: PythonObject) raises -> PythonObject:
    """x_neighbors_nc_proba (host column): NearestCentroid.predict_proba /
    predict_log_proba in one native session. Addresses (q, centroids, std,
    prior, out), ints (n, d, classes, mode: 0 softmax, 1 log-softmax,
    2 float64 softmax)."""
    _aux_call(7, a_, i_, f_, 5, 4, 0)
    return PythonObject(None)
