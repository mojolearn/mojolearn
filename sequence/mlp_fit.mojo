# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The multi-layer perceptron's epoch loop and prediction over `Exec`
(`sequence/mlp.mojo` holds the element bodies and the reference citations)."""
from sequence.exec_trait import Exec
from sequence.mlp import (
    ACT_SOFTMAX,
    EPI_ACT_BWD,
    EPI_BIAS_ACT,
    EPI_L2GRAD,
    SplitMix,
    fisher_yates,
)
from sequence.ops import (
    FP,
    OP_ONE_HOT,
    OP_PROBA2,
    OP_ACT,
    OP_ACT_BWD,
    OP_COLSUM_DIV,
    OP_COPY,
    OP_DIVS,
    OP_GATHER_ROWS,
    OP_GEMM_EPI,
    OP_L2GRAD,
    OP_MLP_BLOSS,
    OP_MLP_ROWLOSS,
    OP_OPT,
    OP_SOFTMAX,
    OP_SUMSQ,
    OPT_SK_ADAM,
    OPT_SK_SGD,
    Args,
)
from sequence.recurrent import bias_rows, colsum, gather_rows, gemm
from checks.numerics import ftz, identical_div, identical_mul, identical_pow64, identical_sqrt

comptime SOLVER_ADAM = 0
comptime SOLVER_SGD = 1
comptime LR_CONSTANT = 0
comptime LR_INVSCALING = 1
comptime LR_ADAPTIVE = 2


struct MLPNet(Copyable, Movable):
    var sizes: List[Int]   # D, h1, ..., O
    var act: Int
    var out_act: Int

    def __init__(out self, var sizes: List[Int], act: Int, out_act: Int):
        self.sizes = sizes^
        self.act = act
        self.out_act = out_act

    def n_layers(self) -> Int:
        return len(self.sizes) - 1

    def w_off(self, i: Int) -> Int:
        """Offset of layer i's coefs (i = 0 .. n_layers - 1); its intercepts
        follow them."""
        var off = 0
        for k in range(i):
            off += self.sizes[k] * self.sizes[k + 1] + self.sizes[k + 1]
        return off

    def b_off(self, i: Int) -> Int:
        return self.w_off(i) + self.sizes[i] * self.sizes[i + 1]

    def n_params(self) -> Int:
        return self.w_off(self.n_layers())


def gemm_epi[E: Exec](
    mut ex: E, A: FP, Bm: FP, C: FP, M: Int, N: Int, K: Int,
    sam: Int, sak: Int, sbk: Int, sbn: Int, epi: Int, p3: FP, i10: Int, f0: Float32, f1: Float32,
) raises:
    """`gemm` (dense C, no accumulate) with its follower in the same launch
    (mlp.mojo::op_gemm_epi)."""
    var a = Args()
    a.p0 = A
    a.p1 = Bm
    a.p2 = C
    a.p3 = p3
    a.i0 = M
    a.i1 = N
    a.i2 = K
    a.i3 = sam
    a.i4 = sak
    a.i5 = sbk
    a.i6 = sbn
    a.i8 = N
    a.i9 = epi
    a.i10 = i10
    a.f0 = f0
    a.f1 = f1
    ex.launch[OP_GEMM_EPI](a, M * N)


def _act_launch[E: Exec](mut ex: E, z: FP, rows: Int, cols: Int, act: Int) raises:
    var a = Args()
    a.p0 = z
    a.p1 = z
    a.i0 = act
    a.i1 = cols
    if act == ACT_SOFTMAX:
        ex.launch[OP_SOFTMAX](a, rows)
    elif act != 0:
        ex.launch[OP_ACT](a, rows * cols)


def mlp_forward[E: Exec](mut ex: E, net: MLPNet, P: FP, acts: List[FP], B: Int) raises:
    """acts[0] holds the batch [B, D]; acts[i] <- act(acts[i-1] @ W_i + b_i)."""
    var L = net.n_layers()
    for i in range(L):
        var fi = net.sizes[i]
        var fo = net.sizes[i + 1]
        # one launch: the GEMM, its bias and (not softmax) its activation
        var act = net.out_act if i == L - 1 else net.act
        gemm_epi(ex, acts[i], P + net.w_off(i), acts[i + 1], B, fo, fi, fi, 1, fo, 1,
                 EPI_BIAS_ACT, P + net.b_off(i), 0 if act == ACT_SOFTMAX else act, Float32(0.0), Float32(0.0))
        if act == ACT_SOFTMAX:
            _act_launch(ex, acts[i + 1], B, fo, act)


def mlp_fit[E: Exec](
    mut ex: E, net: MLPNet, X: FP, Y: FP, N: Int, Pio: FP, curve: FP,
    loss_kind: Int, solver: Int, lr_sched: Int, nesterov: Bool, batch: Int, max_iter: Int,
    shuffle: Bool, seed: UInt64, n_iter_no_change: Int,
    lr_init: Float32, b1: Float32, b2: Float32, eps: Float32, momentum: Float32,
    power_t: Float64, alpha: Float32, tol: Float64, y_codes: Int = 0,
) raises -> Int:
    """Returns n_iter; curve[0:n_iter] is the epoch loss (float32 of the
    float64 accumulation). `y_codes` (cpu2-l11-neural): 0, `Y` is the
    dense (N, O) target; 1, `Y` holds N int32 class codes and the (N, O)
    one-hot target is built on the executor (`OP_ONE_HOT`); 2 (O = 1), the
    codes themselves as the float target. Not in host NumPy either way."""
    var L = net.n_layers()
    var D = net.sizes[0]
    var O = net.sizes[L]
    var np_ = net.n_params()
    var bs = batch if batch < N else N
    var dX = ex.alloc(N * D)
    ex.upload(dX, X, N * D)
    var dY = ex.alloc(N * O)
    if y_codes == 1 or (y_codes == 2 and O == 1):
        var dC = ex.alloc(N)
        ex.upload(dC, Y, N)
        var oh = Args()
        oh.p0 = dC
        oh.p1 = dY
        oh.i0 = O
        oh.i1 = 1 if y_codes == 2 else 0
        ex.launch[OP_ONE_HOT](oh, N * O)
    else:
        ex.upload(dY, Y, N * O)
    var P = ex.alloc(np_)
    ex.upload(P, Pio, np_)
    var Gr = ex.alloc(np_)
    var s1 = ex.alloc(np_)
    var s2 = ex.alloc(np_)
    var s3 = ex.alloc(np_)
    var acts = List[FP]()
    var deltas = List[FP]()
    for i in range(L + 1):
        acts.append(ex.alloc(bs * net.sizes[i]))
        deltas.append(ex.alloc(bs * net.sizes[i]))
    var yb = ex.alloc(bs * O)
    var rowloss = ex.alloc(bs)
    var l2 = ex.alloc(L)
    var n_batches = (N + bs - 1) // bs
    var dloss = ex.alloc(n_batches)
    var didx = ex.alloc(N)
    var perm = List[Float32]()
    for i in range(N):
        perm.append(Float32(i))
    var rng = SplitMix(seed)
    var host_loss = List[Float32](length=n_batches, fill=Float32(0.0))
    var best = Float64(1.0e308)
    var no_imp = 0
    var lr = lr_init
    var t_upd = 0
    var pw1 = Float32(1.0)
    var pw2 = Float32(1.0)
    var t_samples = 0
    var n_iter = 0
    for it in range(max_iter):
        if shuffle:
            fisher_yates(rng, perm)
        ex.upload(didx, FP(unsafe_from_address=Int(perm.unsafe_ptr())), N)
        for bi in range(n_batches):
            var off = bi * bs
            var B = bs if off + bs <= N else N - off
            # X and y rows in one launch (apple2; the same copies)
            var gx = Args()
            gx.p0 = dX
            gx.p1 = didx
            gx.p2 = acts[0]
            gx.p3 = dY
            gx.p4 = yb
            gx.i0 = B
            gx.i1 = D
            gx.i2 = off
            gx.i3 = O
            ex.launch[OP_GATHER_ROWS](gx, B * (D + O))
            mlp_forward(ex, net, P, acts, B)
            var a = Args()
            a.p0 = acts[L]
            a.p1 = yb
            a.p2 = deltas[L]
            a.p3 = rowloss
            a.i0 = loss_kind
            a.i1 = O
            ex.launch[OP_MLP_ROWLOSS](a, B)
            if L <= 4:
                # every layer's ||W||^2 in one launch, a thread each (apple2)
                var q = Args()
                q.p0 = P
                q.p1 = l2
                q.i2 = L
                for i in range(L):
                    var o_ = net.w_off(i)
                    var c_ = net.sizes[i] * net.sizes[i + 1]
                    if i == 0:
                        q.i4 = o_
                        q.i5 = c_
                    elif i == 1:
                        q.i6 = o_
                        q.i7 = c_
                    elif i == 2:
                        q.i8 = o_
                        q.i9 = c_
                    else:
                        q.i10 = o_
                        q.i11 = c_
                ex.launch[OP_SUMSQ](q, L)
            else:
                for i in range(L):
                    var q = Args()
                    q.p0 = P + net.w_off(i)
                    q.p1 = l2
                    q.i0 = i
                    q.i1 = net.sizes[i] * net.sizes[i + 1]
                    ex.launch[OP_SUMSQ](q, 1)
            var bl = Args()
            bl.p0 = rowloss
            bl.p1 = l2
            bl.p2 = dloss
            bl.i0 = bi
            bl.i1 = B
            bl.i2 = O
            bl.i3 = L
            bl.i4 = loss_kind
            bl.f0 = ftz(identical_mul(Float32(0.5), alpha))
            ex.launch[OP_MLP_BLOSS](bl, 1)
            var i = L - 1
            while i >= 0:
                var fi = net.sizes[i]
                var fo = net.sizes[i + 1]
                # one launch each: GEMM + L2 term, column sum + mean, GEMM + act'
                gemm_epi(ex, acts[i], deltas[i + 1], Gr + net.w_off(i), fi, fo, B, 1, fi, fo, 1,
                         EPI_L2GRAD, P + net.w_off(i), 0, alpha, Float32(B))
                var cs = Args()
                cs.p0 = deltas[i + 1]
                cs.p1 = Gr + net.b_off(i)
                cs.i0 = B
                cs.i2 = fo
                cs.f0 = Float32(B)
                ex.launch[OP_COLSUM_DIV](cs, fo)
                if i > 0:
                    gemm_epi(ex, deltas[i + 1], P + net.w_off(i), deltas[i], B, fi, fo, fo, 1, 1, fo,
                             EPI_ACT_BWD, acts[i], net.act, Float32(0.0), Float32(0.0))
                i -= 1
            var o = Args()
            o.p0 = P
            o.p1 = Gr
            o.p2 = s1
            o.p3 = s2
            o.p4 = s3
            if solver == SOLVER_ADAM:
                t_upd += 1
                pw1 = ftz(identical_mul(pw1, b1))
                pw2 = ftz(identical_mul(pw2, b2))
                o.i0 = OPT_SK_ADAM
                o.f1 = b1
                o.f2 = b2
                o.f3 = eps
                o.f5 = ftz(identical_div(ftz(identical_mul(lr, ftz(identical_sqrt(Float32(1.0) - pw2)))),
                                         Float32(1.0) - pw1))
            else:
                o.i0 = OPT_SK_SGD
                o.i2 = 1 if nesterov else 0
                o.f0 = lr
                o.f1 = momentum
            ex.launch[OP_OPT](o, np_)
        ex.sync()
        ex.download(FP(unsafe_from_address=Int(host_loss.unsafe_ptr())), dloss, n_batches)
        var acc = Float64(0.0)
        for bi in range(n_batches):
            var off = bi * bs
            var B = bs if off + bs <= N else N - off
            acc += Float64(host_loss[bi]) * Float64(B)
        var loss = acc / Float64(N)
        curve.unsafe_store(it, Float32(loss))
        n_iter = it + 1
        t_samples += N
        if loss > best - tol:
            no_imp += 1
        else:
            no_imp = 0
        if loss < best:
            best = loss
        if solver == SOLVER_SGD and lr_sched == LR_INVSCALING:
            lr = Float32(Float64(lr_init) / identical_pow64(Float64(t_samples + 1), power_t))
        if no_imp > n_iter_no_change:
            if solver == SOLVER_SGD and lr_sched == LR_ADAPTIVE and lr > Float32(1e-6):
                lr = ftz(identical_div(lr, Float32(5.0)))
                no_imp = 0
            else:
                break
    ex.sync()
    ex.download(Pio, P, np_)
    _ = perm^
    _ = host_loss^
    return n_iter


def mlp_predict[E: Exec](mut ex: E, net: MLPNet, X: FP, N: Int, Pin: FP, dst: FP, chunk: Int,
                         proba2: Bool = False) raises:
    """`proba2` (cpu2-l11-neural): a one-output logistic net writes the
    (N, 2) probability `[1 - p, p]` (`OP_PROBA2` on the executor)."""
    var L = net.n_layers()
    var D = net.sizes[0]
    var O = net.sizes[L]
    var bs = chunk if chunk < N else N
    var dX = ex.alloc(N * D)
    ex.upload(dX, X, N * D)
    var P = ex.alloc(net.n_params())
    ex.upload(P, Pin, net.n_params())
    var acts = List[FP]()
    for i in range(L + 1):
        acts.append(ex.alloc(bs * net.sizes[i]))
    var dout = ex.alloc(N * O)
    var b0 = 0
    while b0 < N:
        var B = bs if b0 + bs <= N else N - b0
        var c = Args()
        c.p0 = dX + b0 * D
        c.p1 = acts[0]
        ex.launch[OP_COPY](c, B * D)
        mlp_forward(ex, net, P, acts, B)
        var d = Args()
        d.p0 = acts[L]
        d.p1 = dout + b0 * O
        ex.launch[OP_COPY](d, B * O)
        b0 += B
    if proba2 and O == 1:
        var two = ex.alloc(N * 2)
        var q = Args()
        q.p0 = dout
        q.p1 = two
        ex.launch[OP_PROBA2](q, N * 2)
        ex.sync()
        ex.download(dst, two, N * 2)
        return
    ex.sync()
    ex.download(dst, dout, N * O)
