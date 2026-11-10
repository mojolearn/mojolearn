# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python door of the x_neighbors kernels the GPU binding
(`bindings/_mojolearn_x_neighbors.mojo`) exports by hand: PythonObject
arguments to typed values and back, argument checks, the GIL released
around the typed call. No arithmetic on data; every number comes from the
typed kernels in x_neighbors/."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from x_neighbors.kapprox_dev import kpca_resident
from x_neighbors.kfeat_dev import (
    kfeat_first_negative,
    kfeat_flags,
    kfeat_pcs_draw_idn,
    kfeat_schi2_draw,
    kfeat_schi2_fit,
    kfeat_schi2_fit_idn,
    kfeat_schi2_transform,
    kfeat_schi2_weights,
)
from x_neighbors.nan_cells_device import nc_fit_lean
from x_neighbors.ocsvm_dev import op_ocsvm_alpha_init, op_ocsvm_x, op_unit_ff
from x_neighbors.ocsvm_init import oci_nu_hi, oci_nu_lo
from x_neighbors.native_descriptor_aux_device import xn_aux_execute


def kpca_resident_binding() raises -> PythonObject:
    """1 when KernelPCA.fit takes the resident kit route."""
    return PythonObject(kpca_resident())


# TOMBSTONE: x_neighbors_lp_fast_resident (dead export: always 1, the resident kNN-graph loop is the only path) deleted 2026-11-09 by lane/n3b-sgd-on-program-fit; code recoverable at 5b5490960d.
# Restore: git apply -3 experiments/removed/N3B_SGD_HOST_EPOCH_LOOPS.patch


def nc_fit_lean_binding() raises -> PythonObject:
    """1 when this binary's fit-time count op ignores the cell list."""
    return PythonObject(nc_fit_lean())


def kfeat_flags_binding() raises -> PythonObject:
    return PythonObject(kfeat_flags())


def kfeat_first_negative_binding(xaddr: PythonObject, n: PythonObject) raises -> PythonObject:
    var addr = Int(py=xaddr)
    var cnt = Int(py=n)
    var r = 0
    with GILReleased(Python()):
        r = kfeat_first_negative(addr, cnt)
    return PythonObject(r)


def kfeat_schi2_fit_binding(p: PythonObject, w_out: PythonObject, off_out: PythonObject) raises -> PythonObject:
    """p = [seed & 2^32-1, d, n_components]. Returns 0."""
    var seed_word = Int(py=p[0])
    var d = Int(py=p[1])
    var nc = Int(py=p[2])
    var w = Int(py=w_out)
    var off = Int(py=off_out)
    with GILReleased(Python()):
        kfeat_schi2_fit(seed_word, d, nc, w, off)
    return PythonObject(0)


def kfeat_schi2_draw_binding(p: PythonObject, z_out: PythonObject, off_out: PythonObject) raises -> PythonObject:
    """p = [seed & 2^32-1, d, n_components]. Returns 0."""
    var seed_word = Int(py=p[0])
    var d = Int(py=p[1])
    var nc = Int(py=p[2])
    var z = Int(py=z_out)
    var off = Int(py=off_out)
    with GILReleased(Python()):
        kfeat_schi2_draw(seed_word, d, nc, z, off)
    return PythonObject(0)


def kfeat_schi2_weights_binding(z_addr: PythonObject, w_addr: PythonObject, n: PythonObject) raises -> PythonObject:
    """Returns 0."""
    var z = Int(py=z_addr)
    var w = Int(py=w_addr)
    var count = Int(py=n)
    with GILReleased(Python()):
        kfeat_schi2_weights(z, w, count)
    return PythonObject(0)


def kfeat_schi2_transform_binding(p: PythonObject, a: PythonObject, f: PythonObject) raises -> PythonObject:
    """p = [n, d, nc, pending]; a = [x, wz, w_out, off, out]; f = [skewedness].
    Returns 0."""
    var n = Int(py=p[0])
    var d = Int(py=p[1])
    var nc = Int(py=p[2])
    var pending = Int(py=p[3])
    var x = Int(py=a[0])
    var wz = Int(py=a[1])
    var wo = Int(py=a[2])
    var off = Int(py=a[3])
    var out = Int(py=a[4])
    var skew = Float64(py=f[0])
    with GILReleased(Python()):
        kfeat_schi2_transform(n, d, nc, pending, x, wz, wo, off, out, skew)
    return PythonObject(0)


def kfeat_schi2_fit_idn_binding(p: PythonObject, w_out: PythonObject, off_out: PythonObject) raises -> PythonObject:
    """p = [seed & 2^32-1, d, n_components]. Returns 0."""
    var seed_word = Int(py=p[0])
    var d = Int(py=p[1])
    var nc = Int(py=p[2])
    var w = Int(py=w_out)
    var off = Int(py=off_out)
    with GILReleased(Python()):
        kfeat_schi2_fit_idn(seed_word, d, nc, w, off)
    return PythonObject(0)


def kfeat_pcs_draw_idn_binding(p: PythonObject, idx_out: PythonObject, sgn_out: PythonObject) raises -> PythonObject:
    """p = [seed & 2^32-1, n_components, degree * n_features]. Returns 0."""
    var seed_word = Int(py=p[0])
    var nc = Int(py=p[1])
    var total = Int(py=p[2])
    var idx = Int(py=idx_out)
    var sgn = Int(py=sgn_out)
    with GILReleased(Python()):
        kfeat_pcs_draw_idn(seed_word, nc, total, idx, sgn)
    return PythonObject(0)


def ocsvm_resident_binding(a_: PythonObject, i_: PythonObject, f_: PythonObject) raises -> PythonObject:
    """xn_ocsvm_x: addresses (x, cv, alpha, info, iters), ints (n, d, kind,
    degree, max_iter), floats (gamma, coef0, eps)."""
    var a = List[Int]()
    for k in range(5):  # small-loop(the five address words)
        var v = Int(py=a_[k])
        if v == 0:
            raise Error("x_neighbors: null buffer address")
        a.append(v)
    var iv = List[Int]()
    for k in range(5):  # small-loop(the five size words)
        var v = Int(py=i_[k])
        if v < 0:
            raise Error("x_neighbors: a negative size was passed")
        iv.append(v)
    var gamma = Float32(Float64(py=f_[0]))
    var coef0 = Float32(Float64(py=f_[1]))
    var eps = Float32(Float64(py=f_[2]))
    op_ocsvm_x(a[0], a[1], a[2], a[3], a[4], iv[0], iv[1], iv[2], gamma, coef0, iv[3], eps, iv[4])
    return PythonObject(None)


def ocsvm_alpha_init_binding(a_: PythonObject, i_: PythonObject, f_: PythonObject) raises -> PythonObject:
    """x_neighbors_ocsvm_alpha_init: addresses (cv, alpha), ints (n,),
    floats (nu,). nu is read as binary64 and split into a float-float pair
    (a scalar of the call)."""
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
    """x_neighbors_unit_ff: addresses (v, res, info), ints (n,)."""
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
              nf: Int, optional_ints: Int = 0) raises:
    """One auxiliary descriptor operation (x_neighbors/native_descriptor_aux_*)
    from the Python door: the fixed address, dimension and scalar words of
    the named operation, then its typed admission and owner, with the GIL released."""
    var a = List[Int]()
    var p = List[Int]()
    var f = List[Float64]()
    for k in range(na):  # small-loop(the fixed address words of the named operation)
        a.append(Int(py=a_[k]))
    for k in range(ni):  # small-loop(the fixed dimension words of the named operation)
        p.append(Int(py=i_[k]))
    # `optional_ints` trailing words the caller may pass (lane
    # a8b-device-x-neighbors-rest: the finite status address).
    for k in range(ni, min(ni + optional_ints, len(i_))):  # small-loop(at most the named operation's optional words)
        p.append(Int(py=i_[k]))
    for k in range(nf):  # small-loop(the fixed scalar words of the named operation)
        f.append(Float64(py=f_[k]))
    with GILReleased(Python()):
        _ = xn_aux_execute(op, a, p, f)


def ocsvm_labels_binding(a_: PythonObject, i_: PythonObject, f_: PythonObject) raises -> PythonObject:
    """x_neighbors_ocsvm_labels: OneClassSVM.predict's kernel_matmul,
    unary and sign-label ops in one native session. Addresses (q, support
    rows, dual column, int32 labels), ints (n, m, d, kind, degree, sign
    mode), floats (gamma, coef0, scale, shift, threshold)."""
    _aux_call(5, a_, i_, f_, 4, 6, 5)
    return PythonObject(None)


def nc_labels_binding(a_: PythonObject, i_: PythonObject, f_: PythonObject) raises -> PythonObject:
    """x_neighbors_nc_labels: NearestCentroid.predict's codes in one
    native session. Addresses (q, centroids, std, prior, int32 codes), ints
    (n, d, classes, mode: 0 sqdist, 1 l1dist, 2 decision)."""
    _aux_call(6, a_, i_, f_, 5, 4, 0, 1)
    return PythonObject(None)


def nc_proba_binding(a_: PythonObject, i_: PythonObject, f_: PythonObject) raises -> PythonObject:
    """x_neighbors_nc_proba: NearestCentroid.predict_proba /
    predict_log_proba in one native session. Addresses (q, centroids, std,
    prior, out), ints (n, d, classes, mode: 0 softmax, 1 log-softmax,
    2 float64 softmax)."""
    _aux_call(7, a_, i_, f_, 5, 4, 0, 1)
    return PythonObject(None)


def lp_dense_proba_binding(a_: PythonObject, i_: PythonObject, f_: PythonObject) raises -> PythonObject:
    """x_neighbors_lp_dense_proba: LabelPropagation / LabelSpreading
    predict_proba or predict with kernel='rbf' in one native session.
    Addresses (q, X_, label_distributions_, proba out, int32 codes out),
    ints (nq, n, d, classes, kernel kind, want_codes), floats (gamma, coef0)."""
    _aux_call(8, a_, i_, f_, 5, 6, 2, 1)
    return PythonObject(None)
