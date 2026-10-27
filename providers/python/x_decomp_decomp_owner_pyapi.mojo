# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The doors of the decomposition owners' drivers (x_decomp/ica_owner.mojo,
...): each unpacks one list of addresses and integers (and one of floats),
releases the GIL and calls one typed driver; no data work happens here. The
native owners in bindings/ (ica_native.mojo, ...) are their only callers."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from x_decomp.cells import F32Ptr
from x_decomp.exec_trait import Exec
from x_decomp.pca_randomized import RDev
from x_decomp.decomp_owner_kit import DecompDev
from x_decomp.ica_owner import ica_fit_run, ica_project_run
from x_decomp.nmf import nmf_args_checked
from x_decomp.nmf_owner import nmf_inverse_run, nmf_run
from x_decomp.fa_owner import fa_fit_run, fa_query_run
from x_decomp.lda_fit import LdaArgs
from x_decomp.lda_owner import lda_run
from x_decomp.kpca_owner import kpca_eig_run
from x_decomp.rp_owner import rp_fit_run, rp_project_run


def door_ptr(p: PythonObject, i: Int) raises -> F32Ptr:
    var a = Int(py=p[i])
    if a == 0:
        raise Error("x_decomp: null float32 buffer address")
    return F32Ptr(unsafe_from_address=a)


def door_dim(p: PythonObject, i: Int) raises -> Int:
    var v = Int(py=p[i])
    if v < 0 or v > 2147483647:
        raise Error("x_decomp: dimension out of range")
    return v


def ica_fit_py[E: Exec, D: RDev, RES: Bool, X: DecompDev](p: PythonObject, f: PythonObject) raises -> PythonObject:
    """FastICA fit / fit_transform. p = [x, n, d, nc, whiten (0, 1 unit,
    2 arbitrary), w_init (0: drawn), seed, algorithm, fun, max_iter,
    project, components, mixing, whitening, mean, sources]; f = [alpha,
    tol]. Returns n_iter_."""
    if len(p) != 16 or len(f) != 2:
        raise Error("x_decomp_ica_fit: params must hold the input, the options and five outputs")
    var x = door_ptr(p, 0)
    var n = door_dim(p, 1)
    var d = door_dim(p, 2)
    var nc = door_dim(p, 3)
    var whiten = Int(py=p[4])
    var winit_address = Int(py=p[5])
    var winit = F32Ptr(unsafe_from_address=winit_address)  # read only when nonzero
    var seed = Int(py=p[6])
    var algo = Int(py=p[7])
    var fun = Int(py=p[8])
    var max_iter = Int(py=p[9])
    var project = Int(py=p[10]) != 0
    var pc = door_ptr(p, 11)
    var pm = door_ptr(p, 12)
    var pw = door_ptr(p, 13)
    var pu = door_ptr(p, 14)
    var ps = door_ptr(p, 15)
    var alpha = Float64(py=f[0])
    var tol = Float64(py=f[1])
    var it = 0
    with GILReleased(Python()):
        it = ica_fit_run[E, D, RES, X](x, n, d, nc, whiten, winit, winit_address != 0, seed, algo, fun, max_iter,
                                       alpha, tol, project, pc, pm, pw, pu, ps)
    return PythonObject(it)


def ica_project_py[E: Exec, D: RDev, RES: Bool](p: PythonObject) raises -> PythonObject:
    """FastICA transform (p[0] 0) or inverse_transform (1). p = [op, x, n,
    width, matrix, rows, mean, whiten, out]."""
    if len(p) != 9:
        raise Error("x_decomp_ica_project: params must hold op, x, n, width, the matrix, rows, mean, whiten and out")
    var inverse = Int(py=p[0]) != 0
    var x = door_ptr(p, 1)
    var n = door_dim(p, 2)
    var width = door_dim(p, 3)
    var matrix = door_ptr(p, 4)
    var rows = door_dim(p, 5)
    var mean = door_ptr(p, 6)
    var whiten = Int(py=p[7]) != 0
    var out = door_ptr(p, 8)
    with GILReleased(Python()):
        ica_project_run[E, D, RES](x, n, width, inverse, matrix, rows, mean, whiten, out)
    return PythonObject(n)


def nmf_fit_py[E: Exec, D: RDev, RES: Bool, X: DecompDev](p: PythonObject, f: PythonObject) raises -> PythonObject:
    """NMF fit_transform (init 0..4) or transform (5). p = [x, n, d, nc,
    init, W in (read for init 4), H in (read for init 4 and 5), seed mod
    2^32, solver (0 cd, 1 mu), update_H, max_iter, shuffle, W out, H out];
    f = [beta, tol, l1W, l1H, l2W, l2H]. Returns (n_iter_, error)."""
    if len(p) != 14 or len(f) != 6:
        raise Error("x_decomp_nmf_fit: params must hold the input, the options and two outputs")
    var x = door_ptr(p, 0)
    var n = door_dim(p, 1)
    var d = door_dim(p, 2)
    var nc = door_dim(p, 3)
    var init = Int(py=p[4])
    var w_in = door_ptr(p, 5)
    var h_in = door_ptr(p, 6)
    var seed = Int(py=p[7])
    var solver = Int(py=p[8])
    var a = nmf_args_checked(Float64(py=f[0]), Int(py=p[9]) != 0, Int(py=p[10]), Float64(py=f[1]),
                             Float64(py=f[2]), Float64(py=f[3]), Float64(py=f[4]), Float64(py=f[5]),
                             Int(py=p[11]) != 0, seed)
    var w_out = door_ptr(p, 12)
    var h_out = door_ptr(p, 13)
    var got = Tuple[Int, Float64](0, 0.0)
    with GILReleased(Python()):
        got = nmf_run[E, D, RES, X](x, n, d, nc, init, w_in, h_in, seed, solver, a, w_out, h_out)
    return Python.tuple(PythonObject(got[0]), PythonObject(got[1]))


def nmf_inverse_py[E: Exec, D: RDev, RES: Bool](p: PythonObject) raises -> PythonObject:
    """NMF inverse_transform. p = [W, n, nc, H, d, out]."""
    if len(p) != 6:
        raise Error("x_decomp_nmf_inverse: params must hold W, n, nc, H, d and out")
    var w = door_ptr(p, 0)
    var n = door_dim(p, 1)
    var nc = door_dim(p, 2)
    var h = door_ptr(p, 3)
    var d = door_dim(p, 4)
    var out = door_ptr(p, 5)
    with GILReleased(Python()):
        nmf_inverse_run[E, D, RES](w, n, nc, h, d, out)
    return PythonObject(n)


def fa_fit_py[E: Exec, D: RDev, RES: Bool, X: DecompDev](p: PythonObject, f: PythonObject) raises -> PythonObject:
    """FactorAnalysis fit. p = [x, n, d, nc, max_iter, psi0 (d), rotation
    (0 None, 1 varimax, 2 quartimax), components, noise variances, mean,
    log-likelihoods (float64, max(max_iter, 1))]; f = [tol]. Returns n_iter_."""
    if len(p) != 11 or len(f) != 1:
        raise Error("x_decomp_fa_fit: params must hold the input, the options and four outputs")
    var x = door_ptr(p, 0)
    var n = door_dim(p, 1)
    var d = door_dim(p, 2)
    var nc = door_dim(p, 3)
    var max_iter = Int(py=p[4])
    var psi0 = door_ptr(p, 5)
    var rotation = Int(py=p[6])
    var pc = door_ptr(p, 7)
    var ps = door_ptr(p, 8)
    var pm = door_ptr(p, 9)
    var ll_address = Int(py=p[10])
    if ll_address == 0:
        raise Error("x_decomp: null float64 buffer address")
    var ll = MutPointer[Float64, MutAnyOrigin](unsafe_from_address=ll_address)
    var tol = Float64(py=f[0])
    var it = 0
    with GILReleased(Python()):
        it = fa_fit_run[E, D, RES, X](x, n, d, nc, max_iter, tol, psi0, rotation, pc, ps, pm, ll)
    return PythonObject(it)


def fa_query_py[E: Exec, D: RDev, RES: Bool, X: DecompDev](p: PythonObject) raises -> PythonObject:
    """A fitted FactorAnalysis query. p = [op (0 transform, 1 covariance, 2
    precision, 3 score_samples, 4 score), x, n, d, nc, components, noise
    variances, mean, out]. Returns the score (op 4), else 0."""
    if len(p) != 9:
        raise Error("x_decomp_fa_query: params must hold op, x, n, d, nc, the fitted arrays and out")
    var op = Int(py=p[0])
    var x = door_ptr(p, 1)
    var n = door_dim(p, 2)
    var d = door_dim(p, 3)
    var nc = door_dim(p, 4)
    var pc = door_ptr(p, 5)
    var ps = door_ptr(p, 6)
    var pm = door_ptr(p, 7)
    var out = door_ptr(p, 8)
    var value = Float64(0.0)
    with GILReleased(Python()):
        value = fa_query_run[E, D, RES, X](op, x, n, d, nc, pc, ps, pm, out)
    return PythonObject(value)


def lda_py[E: Exec, D: RDev, RES: Bool, X: DecompDev](p: PythonObject, f: PythonObject, message: PythonObject) raises -> PythonObject:
    """One LatentDirichletAllocation operation. p = [op (0 fit, 1
    partial_fit, 2 transform, 3 score, 4 perplexity), x, n, v, nc, fresh,
    flag (normalize / sub_sampling), components in, exp_dir in, components
    out, exp_dir out, out, max_iter, online, batch_size,
    max_doc_update_iter, seed mod 2^32, draw, n_batch_iter, evaluate_every];
    f = [doc prior, topic prior, offset, decay, mean_change_tol, total
    samples, perp_tol]; message the negativity refusal. Returns (n_iter_,
    draw, n_batch_iter, value)."""
    if len(p) != 20 or len(f) != 7:
        raise Error("x_decomp_lda: params must hold op, the input, the state, the options and outputs")
    var op = Int(py=p[0])
    var x = door_ptr(p, 1)
    var n = door_dim(p, 2)
    var v = door_dim(p, 3)
    var nc = door_dim(p, 4)
    var fresh = Int(py=p[5]) != 0
    var flag = Int(py=p[6]) != 0
    var ci = door_ptr(p, 7)
    var ei = door_ptr(p, 8)
    var co = door_ptr(p, 9)
    var eo = door_ptr(p, 10)
    var out = door_ptr(p, 11)
    var a = LdaArgs(Int(py=p[12]), Int(py=p[13]) != 0, Int(py=p[14]), Int(py=p[15]), Int(py=p[16]), Int(py=p[19]),
                    Float64(py=f[0]), Float64(py=f[1]), Float64(py=f[2]), Float64(py=f[3]), Float64(py=f[4]),
                    Float64(py=f[5]), Float64(py=f[6]))
    var draw = Int(py=p[17])
    var nbi = Int(py=p[18])
    var text = String(message)
    var got = Tuple[Int, Float64](0, 0.0)
    with GILReleased(Python()):
        got = lda_run[E, D, RES, X](op, x, n, v, nc, fresh, a, draw, nbi, flag, ci, ei, co, eo, out, text)
    return Python.tuple(PythonObject(got[0]), PythonObject(draw), PythonObject(nbi), PythonObject(got[1]))


def kpca_eig_py[E: Exec, D: RDev, RES: Bool, X: DecompDev](p: PythonObject, f: PythonObject) raises -> PythonObject:
    """KernelPCA.fit's eigenpairs. p = [centered kernel (n x n; unread when
    resident), x (n x d; read when resident), resident, n, d, n_components
    (-1: None), lanczos, remove_zero_eig, values (n room), vectors (n x
    kept room), cols (n), all (1)]; f = [gamma]. Returns the kept count."""
    if len(p) != 12 or len(f) != 1:
        raise Error("x_decomp_kpca_eig: params must hold the kernel, the input, the options and four outputs")
    var kc = door_ptr(p, 0)
    var x = door_ptr(p, 1)
    var resident = Int(py=p[2]) != 0
    var n = door_dim(p, 3)
    var d = door_dim(p, 4)
    var requested = Int(py=p[5])
    var lanczos = Int(py=p[6]) != 0
    var remove_zero = Int(py=p[7]) != 0
    var pv = door_ptr(p, 8)
    var pe = door_ptr(p, 9)
    var pc = door_ptr(p, 10)
    var pa = door_ptr(p, 11)
    var gamma = Float64(py=f[0])
    var kept = 0
    with GILReleased(Python()):
        kept = kpca_eig_run[E, D, RES, X](kc, x, resident, n, d, gamma, requested, lanczos, remove_zero, pv, pe, pc, pa)
    return PythonObject(kept)


def rp_fit_py[E: Exec, D: RDev, RES: Bool, X: DecompDev](p: PythonObject, f: PythonObject) raises -> PythonObject:
    """A random projection's fit. p = [x, n, d, n_components, sparse, seed
    mod 2^32, inverse, fused_ok, components (kc x d), inverse (d x kc)];
    f = [density]. Returns the fused scan's first non-finite index or -1."""
    if len(p) != 10 or len(f) != 1:
        raise Error("x_decomp_rp_fit: params must hold the input, the options and two outputs")
    var x = door_ptr(p, 0)
    var n = door_dim(p, 1)
    var d = door_dim(p, 2)
    var kc = door_dim(p, 3)
    var sparse = Int(py=p[4]) != 0
    var seed = Int(py=p[5])
    var inverse = Int(py=p[6]) != 0
    var fused_ok = Int(py=p[7]) != 0
    var pc = door_ptr(p, 8)
    var pi = door_ptr(p, 9)
    var density = Float64(py=f[0])
    var bad = -1
    with GILReleased(Python()):
        bad = rp_fit_run[E, D, RES, X](x, n, d, kc, sparse, density, seed, inverse, fused_ok, pc, pi)
    return PythonObject(bad)


def rp_project_py[E: Exec, D: RDev, RES: Bool, X: DecompDev](p: PythonObject) raises -> PythonObject:
    """X matrix^T. p = [x, m, d, matrix, rows, tiled, out]. Returns 1 when
    the tiled projection flagged a non-finite X."""
    if len(p) != 7:
        raise Error("x_decomp_rp_project: params must hold x, m, d, the matrix, rows, tiled and out")
    var x = door_ptr(p, 0)
    var m = door_dim(p, 1)
    var d = door_dim(p, 2)
    var matrix = door_ptr(p, 3)
    var rows = door_dim(p, 4)
    var tiled = Int(py=p[5]) != 0
    var out = door_ptr(p, 6)
    var flagged = 0
    with GILReleased(Python()):
        flagged = rp_project_run[E, D, RES, X](x, m, d, matrix, rows, tiled, out)
    return PythonObject(flagged)
