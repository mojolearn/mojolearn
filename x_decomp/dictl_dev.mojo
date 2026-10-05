# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""x_decomp/dictl.mojo's dictionary learning loops on resident device
matrices (lane py-runtime-b): the driver text of dictl.mojo on `DKit` (the
row solvers' kernels as DevExec launches them; the FAST fused atom update,
x_decomp/dict_fast.mojo, where Python's `_dict_dev` took it). GPU binding
only."""
from std.math import sqrt
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from x_decomp.api import _f, _n
from x_decomp.dictl import DlArgs, ENC_LARS, ENC_LASSO_CD, ENC_OMP, ENC_THRESHOLD, _enc_nnz
from x_decomp.kit import (
    Mat, OP_ABS, OP_ADD, OP_DIV, OP_MAXS, OP_SCALE, OP_SOFT, OP_SQ, OP_SQDIFF, OP_SQRT, OP_SUB, mat_from,
)
from x_decomp.kit_device import DKit, DMat


def sparse_encode_dev(mut k: DKit, X: DMat, D: DMat, algo: Int, alpha: Float64, nnz: Int, init: DMat,
                           has_init: Bool, max_iter: Int, positive: Bool) raises -> DMat:
    """`_sparse_encode` (alpha given)."""
    var n = X.r
    var m = X.c
    var kc = D.r
    if algo == ENC_LARS:
        return k.lars_rows(k.mm(D, D, False, True), k.mm(X, D, False, True), m, _enc_nnz(m, kc, nnz))
    var Q = k.mm(X, D, False, True)
    if algo == ENC_THRESHOLD:
        var code = k.ew1(OP_SOFT, Q, alpha)
        if positive:
            return k.ew1(OP_MAXS, code, 0.0)
        return code^
    var G = k.mm(D, D, False, True)
    if algo == ENC_OMP:
        return k.omp_rows(G, Q, _enc_nnz(m, kc, nnz))
    var W = k.copy(init) if (algo == ENC_LASSO_CD and has_init) else k.zeros(n, kc)
    k.lasso_rows(G, Q, W, alpha, max_iter, 1e-8, positive)
    return W^


def resample_atom_dev(mut k: DKit, Y: DMat, seed: Int, c: Int) raises -> DMat:
    """`_resample_atom`."""
    var u = k.word(k.rand(1, 1, seed, 40 + 2 * c, 0))
    var idx = min(Int(u * Float64(Y.r)), Y.r - 1)
    var row = k.rows(Y, idx, idx + 1)
    var mean = k.ew1(OP_SCALE, k.total(row), 1.0 / Float64(row.c))
    var std = k.word(k.ew1(OP_SQRT, k.ew1(OP_SCALE, k.total(k.ew2(OP_SQDIFF, row, mean)), 1.0 / Float64(row.c)), 0.0))
    var noise = k.ew1(OP_SCALE, k.rand(1, row.c, seed, 41 + 2 * c, 1), 0.01 * (std if std != 0.0 else 1.0))
    return k.ew2(OP_ADD, row, noise)


def update_dict_dev(mut k: DKit, D: DMat, Y: DMat, code: DMat, A: DMat, B: DMat, positive: Bool, seed: Int,
                         mut counter: Int, mut code_out: DMat, mut zeroed: Bool) raises -> DMat:
    """`_update_dict(k, D, Y, code, A, B, positive, seed, counter)`: the new
    D returned; when an atom is resampled, code_out is code with its column
    zeroed (zeroed set)."""
    var nc = D.r
    var dg = k.diag(A)
    var used = k.count_gt(dg, 1e-6)
    zeroed = False
    if not positive and D.r * D.c > 0 and used == nc:
        var Dn = k.zeros(0, 0)
        if k.dict_fused(D, A, B, Dn):
            return Dn^
    var Dm = k.copy(D)
    for j in range(nc):  # small-loop(nc: atoms): sklearn's atom order, each a chain of kit cells
        var ajj = k.cols(dg, j, j + 1)
        var row: DMat
        if used == nc or k.word(ajj) > 1e-6:
            var upd = k.ew2(OP_SUB, k.vec_t(k.cols(B, j, j + 1)), k.mm(k.rows(A, j, j + 1), Dm, False, False))
            row = k.ew2(OP_ADD, k.rows(Dm, j, j + 1), k.ew2(OP_DIV, upd, ajj))
        else:
            row = resample_atom_dev(k, Y, seed, counter)
            counter += 1
            if not zeroed:
                code_out = k.copy(code)
                zeroed = True
            k.fill0(code_out, j, code_out.c, code_out.r)
        if positive:
            row = k.ew1(OP_MAXS, row, 0.0)
        var nrm = k.ew1(OP_SQRT, k.total(k.ew1(OP_SQ, row, 0.0)), 0.0)
        row = k.ew2(OP_DIV, row, k.ew1(OP_MAXS, nrm, 1.0))
        k.place_row(Dm, row, j)
    return Dm^


def cost_dev(mut k: DKit, X: DMat, code: DMat, D: DMat, alpha: Float64) raises -> Float64:
    """`_cost`."""
    var r = k.word(k.total(k.ew2(OP_SQDIFF, X, k.mm(code, D, False, False))))
    var l1 = k.word(k.total(k.ew1(OP_ABS, code, 0.0)))
    return 0.5 * r + alpha * l1


def dict_learning_loop_dev(mut k: DKit, X: DMat, mut code: DMat, mut D: DMat, a: DlArgs,
                                mut errors: List[Float64]) raises -> Int:
    """`_dict_learning`'s loop: code and D replaced, errors appended; returns ii."""
    var counter = 0
    var ii = 0
    for i in range(1, a.max_iter + 1):
        ii = i
        code = sparse_encode_dev(k, X, D, a.algo, a.alpha, a.nnz, code, True, a.enc_iter, a.pos_code)
        var A = k.mm(code, code, True, False)
        var B = k.mm(X, code, True, False)
        var code2 = k.zeros(0, 0)
        var zeroed = False
        D = update_dict_dev(k, D, X, code, A, B, a.pos_dict, a.seed, counter, code2, zeroed)
        if zeroed:
            code = code2^
        errors.append(cost_dev(k, X, code, D, a.alpha))
        var ne = len(errors)
        if ne > 1 and errors[ne - 2] - errors[ne - 1] < a.tol * errors[ne - 1]:
            break
    return ii


def minibatch_loop_dev(mut k: DKit, Xt: DMat, mut D: DMat, a: DlArgs, bs: Int, n_steps: Int) raises -> Int:
    """MiniBatchDictionaryLearning.fit's step loop over Xt (the shuffled
    rows): D replaced. Returns the number of steps taken (step + 1)."""
    var n = Xt.r
    var m = Xt.c
    var nc = D.r
    var A = k.zeros(nc, nc)
    var B = k.zeros(m, nc)
    var nb = n // bs + (1 if n % bs != 0 else 0)
    var ewa = 0.0
    var have_ewa = False
    var ewa_min = 0.0
    var have_min = False
    var no_imp = 0
    var counter = 0
    var step = -1
    for st in range(n_steps):
        step = st
        var bi = st % nb
        var a0 = bi * bs
        var b0 = min(a0 + bs, n)
        var Xb = k.rows(Xt, a0, b0)
        var b_n = Xb.r
        var none = k.zeros(0, 0)
        var code = sparse_encode_dev(k, Xb, D, a.algo, a.alpha, -1, none, False, a.enc_iter, a.pos_code)
        var c = cost_dev(k, Xb, code, D, a.alpha) / Float64(b_n)
        var theta = (st + 1) * b_n if st < b_n - 1 else b_n * b_n + st + 1 - b_n
        var beta = Float64(theta + 1 - b_n) / Float64(theta + 1)
        A = k.ew2(OP_ADD, k.ew1(OP_SCALE, A, beta), k.ew1(OP_SCALE, k.mm(code, code, True, False), 1.0 / Float64(b_n)))
        B = k.ew2(OP_ADD, k.ew1(OP_SCALE, B, beta), k.ew1(OP_SCALE, k.mm(Xb, code, True, False), 1.0 / Float64(b_n)))
        var old = k.copy(D)
        var code2 = k.zeros(0, 0)
        var zeroed = False
        D = update_dict_dev(k, D, Xb, code, A, B, a.pos_dict, a.seed, counter, code2, zeroed)
        var s1 = st + 1
        var lim = Float64(n) / Float64(b_n)
        if not (lim < 100.0):
            lim = 100.0
        if Float64(s1) <= lim:
            continue
        if not have_ewa:
            ewa = c
            have_ewa = True
        else:
            var al = Float64(b_n) / Float64(n + 1)
            if 1.0 < al:
                al = 1.0
            ewa = ewa * (1 - al) + c * al
        var diff = sqrt(k.word(k.total(k.ew2(OP_SQDIFF, D, old)))) / Float64(nc)
        if a.tol > 0 and diff <= a.tol:
            break
        if not have_min or ewa < ewa_min:
            no_imp = 0
            ewa_min = ewa
            have_min = True
        else:
            no_imp += 1
        if a.max_no_imp >= 0 and no_imp >= a.max_no_imp:
            break
    return step + 1


def dict_learning_dev_py(
    x: PythonObject, code: PythonObject, d: PythonObject, errs: PythonObject, p: PythonObject, f: PythonObject
) raises -> PythonObject:
    """`dict_learning_py` on the resident kit."""
    var n = _n(p, 0)
    var m = _n(p, 1)
    var nc = _n(p, 2)
    if nc < 1 or n * m > 2147483647 or n * nc > 2147483647 or nc * m > 2147483647:
        raise Error("x_decomp: dict learning shape out of range")
    var a = DlArgs(p, f)
    var px = _f(x)
    var pc = _f(code)
    var pd = _f(d)
    var pe = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=errs))
    var ii = 0
    var ne = 0
    with GILReleased(Python()):
        var k = DKit()
        var X = k.upload(mat_from(px, n, m))
        var C = k.upload(mat_from(pc, n, nc))
        var D = k.upload(mat_from(pd, nc, m))
        var el = List[Float64]()
        ii = dict_learning_loop_dev(k, X, C, D, a, el)
        var hc = k.get(C)
        var hd = k.get(D)
        k.sync()
        for i in range(n * nc):
            pc.unsafe_store(i, hc.d[i])
        for i in range(nc * m):
            pd.unsafe_store(i, hd.d[i])
        ne = len(el)
        for i in range(ne):
            pe.unsafe_store(i, el[i])
    return Python.tuple(ii, ne)


def minibatch_dev_py(x: PythonObject, d: PythonObject, p: PythonObject, f: PythonObject) raises -> PythonObject:
    """`minibatch_py` on the resident kit."""
    var n = _n(p, 0)
    var m = _n(p, 1)
    var nc = _n(p, 2)
    var bs = _n(p, 11)
    var n_steps = _n(p, 12)
    if nc < 1 or bs < 1 or n * m > 2147483647 or nc * m > 2147483647:
        raise Error("x_decomp: minibatch dict shape out of range")
    var a = DlArgs(p, f)
    var px = _f(x)
    var pd = _f(d)
    var steps = 0
    with GILReleased(Python()):
        var k = DKit()
        var X = k.upload(mat_from(px, n, m))
        var D = k.upload(mat_from(pd, nc, m))
        steps = minibatch_loop_dev(k, X, D, a, bs, n_steps)
        var hd = k.get(D)
        k.sync()
        for i in range(nc * m):
            pd.unsafe_store(i, hd.d[i])
    return PythonObject(steps)
