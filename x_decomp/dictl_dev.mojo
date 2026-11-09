# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""x_decomp/dictl.mojo's dictionary learning loops on resident device
matrices (lane py-runtime-b): the driver text of dictl.mojo on `DKit` (the
row solvers' kernels as DevExec launches them; the FAST fused atom update,
x_decomp/dict_fast.mojo, where Python's `_dict_dev` took it). GPU binding
only.

Lane n10-dictl-arima-loops (2026-11-09): each loop is a device program read
once per DL_CHECK_EVERY iterations. The body is the encode (the lasso / LARS
/ OMP row solvers as kernels, no drain), the two gemms and the atom update
with the unused-atom resample as a kernel (x_decomp/dictl_resample.mojo),
its state written to a ring slot and its cost words to a device array; the
float64 stop tests run on the host over the chunk's words in order, so the
stopping iteration, the errors and every word are the per-iteration loop's
(the host column, dictl.mojo). It read 3 + 2 nc words per iteration before
(the used-atom count, A[j, j] per atom and its column move, two cost words)."""
from std.math import sqrt
from x_decomp.cells import F32Ptr
from x_decomp.dictl import DlArgs, ENC_LARS, ENC_LASSO_CD, ENC_OMP, ENC_THRESHOLD, _enc_nnz
from x_decomp.kit import (
    Mat, OP_ABS, OP_ADD, OP_DIV, OP_MAXS, OP_SCALE, OP_SOFT, OP_SQ, OP_SQDIFF, OP_SQRT, OP_SUB, mat_from,
)
from x_decomp.kit_device import DKit, DMat
from x_decomp.dictl_resample import enqueue_dict_resample


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


# TOMBSTONE: resample_atom_dev, cost_dev and the per-iteration host reads of the dictionary loops (A[j, j] per atom, two
# draws per resample, the used-atom count, two cost words per iteration and a diff word per step) deleted 2026-11-09 by
# lane/n10-dictl-arima-loops; code recoverable at the lane's base f59d913b0b.
# Restore: git apply experiments/removed/DICTL_HOST_READS.patch

#: iterations (steps) enqueued between two reads of the per-iteration cost
#: words: the loop keeps that many states in a ring and the host replays its
#: float64 stop tests over the chunk (the ARIMA solver's ASYNC_READ_EVERY
#: shape). The stop iteration and every word are the per-iteration loop's.
comptime DL_CHECK_EVERY = 4
#: floats of ring state the batch loop holds at most (1 GiB) before it takes
#: fewer slots: the ring is K copies of the code (n x nc) and the dictionary
comptime DL_RING_MAX_FLOATS = 1 << 28


def _row_view(A: DMat, i: Int) -> DMat:
    """Row i of A, in place (1 x A.c)."""
    return DMat(rows_of=A, row0=i, rows=1)


def _elem_view(A: DMat, i: Int, j: Int) -> DMat:
    """A[i, j], in place (1 x 1)."""
    var v = DMat(rows_of=A, row0=i, rows=1)
    v.off += j
    v.c = 1
    return v^


def row_std_dev(mut k: DKit, Y: DMat) raises -> DMat:
    """The std of every row of Y (Y.r x 1): `_resample_atom`'s cells
    (mean = total / m, std = sqrt(total((row - mean)^2) / m)) for all rows at
    once, the same per-row folds (`rowsum` folds each row alone)."""
    var inv = 1.0 / Float64(Y.c)
    var mean = k.ew1(OP_SCALE, k.rowsum(Y), inv)
    return k.ew1(OP_SQRT, k.ew1(OP_SCALE, k.rowsum(k.ew2(OP_SQDIFF, Y, mean)), inv), 0.0)


def update_dict_dev(mut k: DKit, D: DMat, Y: DMat, ystd: DMat, code: DMat, A: DMat, B: DMat, positive: Bool,
                    seed: Int, counter: DMat) raises -> DMat:
    """`_update_dict(k, D, Y, code, A, B, positive, seed, counter)`: the new
    D returned. An unused atom (A[j, j] <= 1e-6) is resampled from a row of Y
    (`ystd` its rows' std) on the device (x_decomp/dictl_resample.mojo):
    code's column j is zeroed in place and the resample count word `counter`
    advanced. Nothing is read home."""
    var nc = D.r
    var m = D.c
    if not positive and D.n() > 0:
        var Dn = k.zeros(0, 0)
        if k.dict_fused(D, A, B, Y, ystd, code, counter, seed, Dn):
            return Dn^
    var Dm = k.copy(D)
    var Bt = k.t(B)
    for j in range(nc):  # small-loop(nc: atoms): sklearn's atom order, each a chain of kit cells
        var ajj = _elem_view(A, j, j)
        var upd = k.ew2(OP_SUB, _row_view(Bt, j), k.mm(k.rows(A, j, j + 1), Dm, False, False))
        var row = k.ew2(OP_ADD, _row_view(Dm, j), k.ew2(OP_DIV, upd, ajj))
        enqueue_dict_resample(
            k.ctx, row.p(), ajj.p(), Y.p(), ystd.p(), code.p(), counter.p(), Y.r, m, code.r, nc, j, seed, row.p(), 0,
        )
        if positive:
            row = k.ew1(OP_MAXS, row, 0.0)
        var nrm = k.ew1(OP_SQRT, k.total(k.ew1(OP_SQ, row, 0.0)), 0.0)
        row = k.ew2(OP_DIV, row, k.ew1(OP_MAXS, nrm, 1.0))
        k.place_row(Dm, row, j)
    return Dm^


def _cost_words(mut k: DKit, X: DMat, code: DMat, D: DMat, words: DMat, at: Int) raises:
    """`_cost`'s two float32 words (the residual's and code's L1 totals) into
    words[at], words[at + 1], on the device; the host forms
    0.5 r + alpha l1 in float64 after the chunk's read."""
    k.put(_row_view(words, at), k.total(k.ew2(OP_SQDIFF, X, k.mm(code, D, False, False))))
    k.put(_row_view(words, at + 1), k.total(k.ew1(OP_ABS, code, 0.0)))


def _dl_step(mut k: DKit, X: DMat, ystd: DMat, Cp: DMat, Dp: DMat, a: DlArgs, counter: DMat, Cs: DMat, Ds: DMat,
             words: DMat, s: Int) raises:
    """One `_dict_learning` iteration from (Cp, Dp) into the ring slot
    (Cs, Ds), its cost words into words[2 s], words[2 s + 1]. Enqueued only."""
    var c = sparse_encode_dev(k, X, Dp, a.algo, a.alpha, a.nnz, Cp, True, a.enc_iter, a.pos_code)
    var A = k.mm(c, c, True, False)
    var B = k.mm(X, c, True, False)
    var Dn = update_dict_dev(k, Dp, X, ystd, c, A, B, a.pos_dict, a.seed, counter)
    _cost_words(k, X, c, Dn, words, 2 * s)
    k.put(Cs, c)
    k.put(Ds, Dn)


def dict_learning_loop_dev(mut k: DKit, X: DMat, mut code: DMat, mut D: DMat, a: DlArgs,
                                mut errors: List[Float64]) raises -> Int:
    """`_dict_learning`'s loop: code and D replaced, errors appended; returns
    ii. The iterations run on the device DL_CHECK_EVERY at a time (each into
    its ring slot), then one read of their cost words; the host forms each
    error and takes the convergence test in order, and the answer is the
    stopping iteration's slot. Same words, errors and ii as the
    per-iteration loop."""
    var n = X.r
    var nc = D.r
    var m = D.c
    if a.max_iter < 1:
        return 0
    var K = DL_CHECK_EVERY
    var per = max(1, n * nc + nc * m)
    if K * per > DL_RING_MAX_FLOATS:
        K = max(1, DL_RING_MAX_FLOATS // per)
    var ystd = row_std_dev(k, X)
    var counter = k.zeros(1, 1)
    var ringC = DMat(K * n, nc)
    var ringD = DMat(K * nc, m)
    var words = DMat(2 * K, 1)
    var ii = 0
    var i = 1
    var stopped = False
    while i <= a.max_iter and not stopped:  # driver-loop: over flag-read chunks of the bounded iteration loop
        var last = min(i + K - 1, a.max_iter)
        for it in range(i, last + 1):  # driver-loop: over the iterations enqueued between two reads
            var s = (it - 1) % K
            var Cs = DMat(rows_of=ringC, row0=s * n, rows=n)
            var Ds = DMat(rows_of=ringD, row0=s * nc, rows=nc)
            if it == 1:
                _dl_step(k, X, ystd, code, D, a, counter, Cs, Ds, words, s)
            else:
                var p = (it - 2) % K
                var Cp = DMat(rows_of=ringC, row0=p * n, rows=n)
                var Dp = DMat(rows_of=ringD, row0=p * nc, rows=nc)
                _dl_step(k, X, ystd, Cp, Dp, a, counter, Cs, Ds, words, s)
        var hw = k.get(words)
        k.sync()
        for it in range(i, last + 1):  # driver-loop: over the chunk's iterations, the stop test in their order
            var s = (it - 1) % K
            ii = it
            errors.append(0.5 * Float64(hw.d[2 * s]) + a.alpha * Float64(hw.d[2 * s + 1]))
            var ne = len(errors)
            if ne > 1 and errors[ne - 2] - errors[ne - 1] < a.tol * errors[ne - 1]:
                stopped = True
                break
        i = last + 1
    var fs = (ii - 1) % K
    code = k.copy(DMat(rows_of=ringC, row0=fs * n, rows=n))
    D = k.copy(DMat(rows_of=ringD, row0=fs * nc, rows=nc))
    return ii


def _mb_lim(n: Int, b_n: Int) -> Float64:
    """The step after which MiniBatchDictionaryLearning's stop tests run."""
    var lim = Float64(n) / Float64(b_n)
    if not (lim < 100.0):
        lim = 100.0
    return lim


def _mb_rows(n: Int, bs: Int, st: Int) -> Int:
    """Batch `st`'s first row."""
    var nb = n // bs + (1 if n % bs != 0 else 0)
    return (st % nb) * bs


def minibatch_loop_dev(mut k: DKit, Xt: DMat, mut D: DMat, a: DlArgs, bs: Int, n_steps: Int) raises -> Int:
    """MiniBatchDictionaryLearning.fit's step loop over Xt (the shuffled
    rows): D replaced. Returns the number of steps taken (step + 1). The
    steps run on the device DL_CHECK_EVERY at a time, (D, A, B) of each in a
    ring slot and its cost and dictionary-change words in `words`; the host
    then replays the float64 forgetting average and both stop tests over the
    chunk in order. Same words and step count as the per-step loop."""
    var n = Xt.r
    var m = Xt.c
    var nc = D.r
    if n_steps < 1:
        return 0
    var K = DL_CHECK_EVERY
    var ystd_all = row_std_dev(k, Xt)
    var counter = k.zeros(1, 1)
    var ringD = DMat(K * nc, m)
    var ringA = DMat(K * nc, nc)
    var ringB = DMat(K * m, nc)
    var words = DMat(3 * K, 1)
    var A0 = k.zeros(nc, nc)
    var B0 = k.zeros(m, nc)
    var ewa = 0.0
    var have_ewa = False
    var ewa_min = 0.0
    var have_min = False
    var no_imp = 0
    var step = -1
    var st0 = 0
    var stopped = False
    while st0 < n_steps and not stopped:  # driver-loop: over flag-read chunks of the bounded step loop
        var last = min(st0 + K, n_steps)
        for st in range(st0, last):  # driver-loop: over the steps enqueued between two reads
            var s = st % K
            var a0 = _mb_rows(n, bs, st)
            var b0 = min(a0 + bs, n)
            var b_n = b0 - a0
            var need = Float64(st + 1) > _mb_lim(n, b_n)
            var Ds = DMat(rows_of=ringD, row0=s * nc, rows=nc)
            var As = DMat(rows_of=ringA, row0=s * nc, rows=nc)
            var Bs = DMat(rows_of=ringB, row0=s * m, rows=m)
            if st == 0:
                _mb_step(k, Xt, ystd_all, D, A0, B0, a, a0, b0, st, need, counter, Ds, As, Bs, words, s)
            else:
                var p = (st - 1) % K
                var Dp = DMat(rows_of=ringD, row0=p * nc, rows=nc)
                var Ap = DMat(rows_of=ringA, row0=p * nc, rows=nc)
                var Bp = DMat(rows_of=ringB, row0=p * m, rows=m)
                _mb_step(k, Xt, ystd_all, Dp, Ap, Bp, a, a0, b0, st, need, counter, Ds, As, Bs, words, s)
        var hw = k.get(words)
        k.sync()
        for st in range(st0, last):  # driver-loop: over the chunk's steps, the stop tests in their order
            step = st
            var s = st % K
            var a0 = _mb_rows(n, bs, st)
            var b_n = min(a0 + bs, n) - a0
            if Float64(st + 1) <= _mb_lim(n, b_n):
                continue
            var c = (0.5 * Float64(hw.d[3 * s]) + a.alpha * Float64(hw.d[3 * s + 1])) / Float64(b_n)
            if not have_ewa:
                ewa = c
                have_ewa = True
            else:
                var al = Float64(b_n) / Float64(n + 1)
                if 1.0 < al:
                    al = 1.0
                ewa = ewa * (1 - al) + c * al
            var diff = sqrt(Float64(hw.d[3 * s + 2])) / Float64(nc)
            if a.tol > 0 and diff <= a.tol:
                stopped = True
                break
            if not have_min or ewa < ewa_min:
                no_imp = 0
                ewa_min = ewa
                have_min = True
            else:
                no_imp += 1
            if a.max_no_imp >= 0 and no_imp >= a.max_no_imp:
                stopped = True
                break
        st0 = last
    D = k.copy(DMat(rows_of=ringD, row0=(step % K) * nc, rows=nc))
    return step + 1


def _mb_step(mut k: DKit, Xt: DMat, ystd_all: DMat, Dp: DMat, Ap: DMat, Bp: DMat, a: DlArgs, a0: Int, b0: Int,
             st: Int, need: Bool, counter: DMat, Ds: DMat, As: DMat, Bs: DMat, words: DMat, s: Int) raises:
    """MiniBatch step `st` on rows [a0, b0) of Xt from (Dp, Ap, Bp) into the
    ring slot (Ds, As, Bs). When the stop tests will read this step (`need`)
    its cost words (at the old D, as the loop takes them) go to
    words[3 s], words[3 s + 1] and the dictionary change's total to
    words[3 s + 2]. Enqueued only."""
    var Xb = k.rows(Xt, a0, b0)
    var b_n = Xb.r
    var ystd = DMat(rows_of=ystd_all, row0=a0, rows=b_n)
    var none = k.zeros(0, 0)
    var code = sparse_encode_dev(k, Xb, Dp, a.algo, a.alpha, -1, none, False, a.enc_iter, a.pos_code)
    if need:
        _cost_words(k, Xb, code, Dp, words, 3 * s)
    var theta = (st + 1) * b_n if st < b_n - 1 else b_n * b_n + st + 1 - b_n
    var beta = Float64(theta + 1 - b_n) / Float64(theta + 1)
    var An = k.ew2(OP_ADD, k.ew1(OP_SCALE, Ap, beta), k.ew1(OP_SCALE, k.mm(code, code, True, False), 1.0 / Float64(b_n)))
    var Bn = k.ew2(OP_ADD, k.ew1(OP_SCALE, Bp, beta), k.ew1(OP_SCALE, k.mm(Xb, code, True, False), 1.0 / Float64(b_n)))
    var Dn = update_dict_dev(k, Dp, Xb, ystd, code, An, Bn, a.pos_dict, a.seed, counter)
    if need:
        k.put(_row_view(words, 3 * s + 2), k.total(k.ew2(OP_SQDIFF, Dn, Dp)))
    k.put(Ds, Dn)
    k.put(As, An)
    k.put(Bs, Bn)


def dict_learning_dev_run(
    px: F32Ptr, pc: F32Ptr, pd: F32Ptr, pe: MutPointer[Float64, MutUntrackedOrigin], n: Int, m: Int, nc: Int, a: DlArgs,
    mut ne: Int,
) raises -> Int:
    """`dict_learning_run` on the resident kit."""
    if nc < 1 or n * m > 2147483647 or n * nc > 2147483647 or nc * m > 2147483647:
        raise Error("x_decomp: dict learning shape out of range")
    var k = DKit()
    var X = k.upload(mat_from(px, n, m))
    var C = k.upload(mat_from(pc, n, nc))
    var D = k.upload(mat_from(pd, nc, m))
    var el = List[Float64]()
    var ii = dict_learning_loop_dev(k, X, C, D, a, el)
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
    return ii


def minibatch_dev_run(
    px: F32Ptr, pd: F32Ptr, n: Int, m: Int, nc: Int, bs: Int, n_steps: Int, a: DlArgs
) raises -> Int:
    """`minibatch_run` on the resident kit."""
    if nc < 1 or bs < 1 or n * m > 2147483647 or nc * m > 2147483647:
        raise Error("x_decomp: minibatch dict shape out of range")
    var k = DKit()
    var X = k.upload(mat_from(px, n, m))
    var D = k.upload(mat_from(pd, nc, m))
    var steps = minibatch_loop_dev(k, X, D, a, bs, n_steps)
    var hd = k.get(D)
    k.sync()
    for i in range(nc * m):
        pd.unsafe_store(i, hd.d[i])
    return steps
