# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RECURRENT NETWORKS: RNN (tanh, relu), LSTM and GRU layers under a linear
head, trained by backpropagation through time with one of the lane's
optimizers, written once against `sequence/exec.mojo::Exec`.

References: PyTorch `torch/nn/modules/rnn.py` (`nn.RNN`, `nn.LSTM`, `nn.GRU`:
the parameter layout `weight_ih_l{k}` [G*H, D_in], `weight_hh_l{k}` [G*H, H],
`bias_ih_l{k}`, `bias_hh_l{k}`, the gate orders i,f,g,o and r,z,n, and the GRU
update h' = n + z (h - n) of `aten/src/ATen/native/cuda/RNN.cu`).
`sequence/NOT_IMPLEMENTED.tsv` names what is not carried.

The arithmetic order is ours and fixed: the input projection of every step is
one GEMM over the whole sequence plus b_ih; each step adds h @ W_hh^T + b_hh
(GH) to it as GX + GH; every weight gradient is ONE GEMM whose reduction runs
over (time, batch) rows in ascending order; bias gradients are column sums in
the same row order.
"""
from sequence.exec_trait import Exec
from sequence.ops import (
    FP,
    Args,
    OP_BIAS,
    OP_CE,
    OP_CELL_BWD,
    OP_CELL_FWD,
    OP_CELL_BWD_H,
    OP_CELL_FWD_H,
    OP_COLSUM,
    OP_FILL,
    OP_GATHER_ROWS,
    OP_GATHER_SEQ,
    OP_GEMM,
    OP_GEMM_SPLITK,
    OP_MSE,
    OP_OPT,
    OP_SEQ_OUT,
    OP_SOFTMAX,
    OP_SUM,
    OPT_ADAGRAD,
    OPT_ADAM,
    OPT_ADAMW,
    OPT_ADAMAX,
    OPT_NADAM,
    gates_of,
)
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, identical_div, identical_mul, identical_pow64, identical_sqrt, ftz

comptime TASK_MSE = 0
comptime TASK_CE = 1


@fieldwise_init
struct Net(ImplicitlyCopyable, Movable):
    var cell: Int
    var D: Int
    var H: Int
    var L: Int
    var O: Int

    def G(self) -> Int:
        return gates_of(self.cell)

    def din(self, l: Int) -> Int:
        return self.D if l == 0 else self.H

    def layer_size(self, l: Int) -> Int:
        var gh = self.G() * self.H
        return gh * self.din(l) + gh * self.H + 2 * gh

    def w_ih(self, l: Int) -> Int:
        var off = 0
        for k in range(l):
            off += self.layer_size(k)
        return off

    def w_hh(self, l: Int) -> Int:
        return self.w_ih(l) + self.G() * self.H * self.din(l)

    def b_ih(self, l: Int) -> Int:
        return self.w_hh(l) + self.G() * self.H * self.H

    def b_hh(self, l: Int) -> Int:
        return self.b_ih(l) + self.G() * self.H

    def head(self) -> Int:
        return self.w_ih(self.L)

    def n_params(self) -> Int:
        return self.head() + self.O * self.H + self.O


# ------------------------------------------------------------------ launch helpers
def gemm[E: Exec](
    mut ex: E, A: FP, B: FP, C: FP, M: Int, N: Int, K: Int,
    sam: Int, sak: Int, sbk: Int, sbn: Int, accumulate: Bool, ldc: Int,
) raises:
    var a = Args()
    a.p0 = A
    a.p1 = B
    a.p2 = C
    a.i0 = M
    a.i1 = N
    a.i2 = K
    a.i3 = sam
    a.i4 = sak
    a.i5 = sbk
    a.i6 = sbn
    a.i7 = 1 if accumulate else 0
    a.i8 = ldc
    # FAST (apple2): a product of few cells over a long K (VAR's Z^T Z) ran
    # on M N threads, each one K-long chain; split K so about 8192 threads
    # fold blocks of >= 4096, then one ordered sum per cell. IDENTICAL keeps
    # the one chain (its bits); so does every product outside that shape.
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        var MN = M * N
        if MN <= 1024 and K >= 32768:
            var S = min(8192 // MN, K // 4096)
            if S > 1:
                var KS = (K + S - 1) // S
                S = (K + KS - 1) // KS
                a.p3 = ex.alloc(S * MN)
                a.i9 = S
                a.i10 = KS
                a.i11 = 0
                ex.launch[OP_GEMM_SPLITK](a, S * MN)
                a.i11 = 1
                ex.launch[OP_GEMM_SPLITK](a, MN)
                return
    ex.launch[OP_GEMM](a, M * N)


def bias_rows[E: Exec](mut ex: E, X: FP, b: FP, Y: FP, R: Int, C: Int) raises:
    var a = Args()
    a.p0 = X
    a.p1 = b
    a.p2 = Y
    a.i0 = R
    a.i1 = C
    a.i2 = C
    a.i3 = C
    ex.launch[OP_BIAS](a, R * C)


def colsum[E: Exec](mut ex: E, X: FP, dst: FP, R: Int, C: Int) raises:
    var a = Args()
    a.p0 = X
    a.p1 = dst
    a.i0 = R
    a.i1 = C
    a.i2 = C
    ex.launch[OP_COLSUM](a, C)


def fill[E: Exec](mut ex: E, p: FP, n: Int, v: Float32) raises:
    var a = Args()
    a.p0 = p
    a.f0 = v
    ex.launch[OP_FILL](a, n)


def gather_seq[E: Exec](mut ex: E, X: FP, idx: FP, dst: FP, T: Int, B: Int, D: Int, off: Int) raises:
    var a = Args()
    a.p0 = X
    a.p1 = idx
    a.p2 = dst
    a.i0 = T
    a.i1 = B
    a.i2 = D
    a.i3 = off
    ex.launch[OP_GATHER_SEQ](a, T * B * D)


def gather_rows[E: Exec](mut ex: E, Y: FP, idx: FP, dst: FP, B: Int, O: Int, off: Int) raises:
    var a = Args()
    a.p0 = Y
    a.p1 = idx
    a.p2 = dst
    a.i0 = B
    a.i1 = O
    a.i2 = off
    ex.launch[OP_GATHER_ROWS](a, B * O)


def sum_into[E: Exec](mut ex: E, X: FP, n: Int, dst: FP, at: Int, scale: Float32) raises:
    var a = Args()
    a.p0 = X
    a.p1 = dst
    a.i0 = at
    a.i1 = n
    a.f0 = scale
    ex.launch[OP_SUM](a, 1)


# ------------------------------------------------------------------ work buffers
struct Work(Movable):
    var gx: List[FP]
    var gh: List[FP]
    var act: List[FP]
    var hall: List[FP]
    var call: List[FP]
    var dgx: FP
    var dgh: FP
    var dh: FP
    var dhn: FP
    var dc: FP
    var dhout0: FP
    var dhout1: FP
    var xb: FP
    var yb: FP
    var yhat: FP
    var dy: FP
    var sq: FP

    def __init__[E: Exec](out self, mut ex: E, net: Net, T: Int, B: Int, train: Bool) raises:
        var GH = net.G() * net.H
        var H = net.H
        self.gx = List[FP]()
        self.gh = List[FP]()
        self.act = List[FP]()
        self.hall = List[FP]()
        self.call = List[FP]()
        for _ in range(net.L):
            self.gx.append(ex.alloc(T * B * GH))
            self.gh.append(ex.alloc(T * B * GH))
            self.act.append(ex.alloc(T * B * GH))
            self.hall.append(ex.alloc((T + 1) * B * H))
            self.call.append(ex.alloc((T + 1) * B * H))
        var tb = T * B if train else 1
        self.dgx = ex.alloc(tb * GH)
        self.dgh = ex.alloc(tb * GH)
        self.dh = ex.alloc(B * H if train else 1)
        self.dhn = ex.alloc(B * H if train else 1)
        self.dc = ex.alloc(B * H if train else 1)
        self.dhout0 = ex.alloc(tb * H)
        self.dhout1 = ex.alloc(tb * H)
        self.xb = ex.alloc(T * B * net.D)
        self.yb = ex.alloc(B * net.O)
        self.yhat = ex.alloc(B * net.O)
        self.dy = ex.alloc(B * net.O)
        self.sq = ex.alloc(B * net.O)


def forward[E: Exec](mut ex: E, net: Net, P: FP, x: FP, T: Int, B: Int, w: Work) raises -> FP:
    """Every layer over the time-major input x [T, B, D]; returns h_T of the
    top layer [B, H]. Hall[l] holds h_0 = 0 then h_1..h_T."""
    var H = net.H
    var GH = net.G() * H
    var inp = x
    for l in range(net.L):
        var din = net.din(l)
        gemm(ex, inp, P + net.w_ih(l), w.gx[l], T * B, GH, din, din, 1, 1, din, False, GH)
        bias_rows(ex, w.gx[l], P + net.b_ih(l), w.gx[l], T * B, GH)
        fill(ex, w.hall[l], B * H, Float32(0.0))
        fill(ex, w.call[l], B * H, Float32(0.0))
        for s in range(T):
            var hprev = w.hall[l] + s * B * H
            var ghs = w.gh[l] + s * B * GH
            # one launch: h_prev @ W_hh^T + b_hh into GH, then the cell
            # (ops.mojo::op_cell_fwd_h; the same bits as the three launches)
            var a = Args()
            a.p0 = w.gx[l] + s * B * GH
            a.p1 = ghs
            a.p2 = w.act[l] + s * B * GH
            a.p3 = hprev
            a.p4 = w.call[l] + s * B * H
            a.p5 = w.hall[l] + (s + 1) * B * H
            a.p6 = w.call[l] + (s + 1) * B * H
            a.p7 = P + net.w_hh(l)
            a.p8 = P + net.b_hh(l)
            a.i0 = net.cell
            a.i1 = B
            a.i2 = H
            ex.launch[OP_CELL_FWD_H](a, B * H)
        inp = w.hall[l] + B * H
    return w.hall[net.L - 1] + T * B * H


def head[E: Exec](mut ex: E, net: Net, P: FP, hT: FP, B: Int, yhat: FP) raises:
    var H = net.H
    var O = net.O
    gemm(ex, hT, P + net.head(), yhat, B, O, H, H, 1, 1, H, False, O)
    bias_rows(ex, yhat, P + net.head() + O * H, yhat, B, O)


def backward[E: Exec](mut ex: E, net: Net, P: FP, Gr: FP, x: FP, T: Int, B: Int, w: Work) raises:
    """Gradients of every parameter into Gr, given dy [B, O] in w.dy."""
    var H = net.H
    var O = net.O
    var GH = net.G() * H
    var hT = w.hall[net.L - 1] + T * B * H
    gemm(ex, w.dy, hT, Gr + net.head(), O, H, B, 1, O, H, 1, False, H)
    colsum(ex, w.dy, Gr + net.head() + O * H, B, O)
    var cur = w.dhout0
    var other = w.dhout1
    fill(ex, cur, T * B * H, Float32(0.0))
    gemm(ex, w.dy, P + net.head(), cur + (T - 1) * B * H, B, H, O, O, 1, H, 1, False, H)
    var l = net.L - 1
    while l >= 0:
        var din = net.din(l)
        var dh = w.dh
        var dhn = w.dhn
        fill(ex, dh, B * H, Float32(0.0))
        fill(ex, w.dc, B * H, Float32(0.0))
        var s = T - 1
        while s >= 0:
            var a = Args()
            a.p0 = w.act[l] + s * B * GH
            a.p1 = w.gh[l] + s * B * GH
            a.p2 = w.hall[l] + s * B * H
            a.p3 = w.call[l] + s * B * H
            a.p4 = w.call[l] + (s + 1) * B * H
            a.p5 = dh
            a.p6 = cur + s * B * H
            a.p7 = w.dc
            a.p8 = w.dgx + s * B * GH
            a.p9 = w.dgh + s * B * GH
            a.p10 = dhn
            a.p11 = P + net.w_hh(l)
            a.i0 = net.cell
            a.i1 = B
            a.i2 = H
            # one launch: the later step's recurrent GEMM dh += dGH_{s+1} W_hh
            # (it was a launch after that step's cell backward), then this
            # step's cell backward (ops.mojo::op_cell_bwd_h). After s = 0 that
            # GEMM would only produce the gradient into the zero h_0, which
            # nothing reads, so it is not run.
            a.i3 = 1 if s < T - 1 else 0
            a.i4 = B * GH
            ex.launch[OP_CELL_BWD_H](a, B * H)
            var t = dh
            dh = dhn
            dhn = t
            s -= 1
        var inp = x if l == 0 else w.hall[l - 1] + B * H
        gemm(ex, w.dgx, inp, Gr + net.w_ih(l), GH, din, T * B, 1, GH, din, 1, False, din)
        gemm(ex, w.dgh, w.hall[l], Gr + net.w_hh(l), GH, H, T * B, 1, GH, H, 1, False, H)
        colsum(ex, w.dgx, Gr + net.b_ih(l), T * B, GH)
        colsum(ex, w.dgh, Gr + net.b_hh(l), T * B, GH)
        if l > 0:
            gemm(ex, w.dgx, P + net.w_ih(l), other, T * B, din, GH, GH, 1, din, 1, False, din)
            var t2 = cur
            cur = other
            other = t2
        l -= 1


# ------------------------------------------------------------------ optimizer
@fieldwise_init
struct OptConfig(ImplicitlyCopyable, Movable):
    """kind (sequence/ops.mojo OPT_*), flags, beta1 / momentum / lr_decay,
    beta2 / alpha, eps, weight_decay, f7 (dampening / RMSprop momentum)."""

    var kind: Int
    var flags: Int
    var f1: Float32
    var f2: Float32
    var eps: Float32
    var wd: Float32
    var f7: Float32


struct OptState(Movable):
    """The host scalars' running state: beta1^t and beta2^t (one product per
    step, identical_mul, so a step's bias correction is a function of t) and
    NAdam's mu product."""

    var pw1: Float32
    var pw2: Float32
    var mu_prod: Float32

    def __init__(out self):
        self.pw1 = Float32(1.0)
        self.pw2 = Float32(1.0)
        self.mu_prod = Float32(1.0)


def opt_scalars(cfg: OptConfig, mut st: OptState, t: Int, lr: Float32) -> Tuple[Float32, Float32, Float32]:
    """(f5, f6, f7) of step t (one-based), advancing the running state. The
    caller runs every step in order, so the state is a function of t."""
    var f5 = Float32(0.0)
    var f6 = Float32(0.0)
    var f7 = cfg.f7
    if cfg.kind == OPT_ADAM or cfg.kind == OPT_ADAMW or cfg.kind == OPT_ADAMAX or cfg.kind == OPT_NADAM:
        st.pw1 = ftz(identical_mul(st.pw1, cfg.f1))
        st.pw2 = ftz(identical_mul(st.pw2, cfg.f2))
    if cfg.kind == OPT_ADAM or cfg.kind == OPT_ADAMW:
        f5 = ftz(identical_div(lr, Float32(1.0) - st.pw1))
        f6 = ftz(identical_sqrt(Float32(1.0) - st.pw2))
    elif cfg.kind == OPT_ADAMAX:
        f5 = ftz(identical_div(lr, Float32(1.0) - st.pw1))
    elif cfg.kind == OPT_NADAM:
        # torch _single_tensor_nadam: mu_t = b1 (1 - 0.5 0.96^(t md)),
        # mu_{t+1} likewise, mu_product *= mu_t (float32 state);
        # c1 = -lr (1 - mu_t) / (1 - mu_product),
        # c2 = -lr mu_{t+1} / (1 - mu_product mu_{t+1}) (host float64)
        var md = Float64(cfg.f7)
        var b1 = Float64(cfg.f1)
        var mu = b1 * (1.0 - 0.5 * identical_pow64(0.96, Float64(t) * md))
        var mu_next = b1 * (1.0 - 0.5 * identical_pow64(0.96, Float64(t + 1) * md))
        st.mu_prod = ftz(identical_mul(st.mu_prod, Float32(mu)))
        var mp = Float64(st.mu_prod)
        var lr64 = Float64(lr)
        f5 = Float32(1.0) - st.pw2
        f6 = Float32(-lr64 * (1.0 - mu) / (1.0 - mp))
        f7 = Float32(-lr64 * mu_next / (1.0 - mp * mu_next))
    elif cfg.kind == OPT_ADAGRAD:
        f5 = ftz(identical_div(lr, Float32(1.0) + ftz(identical_mul(Float32(t - 1), cfg.f1))))
    return (f5, f6, f7)


def opt_step[E: Exec](
    mut ex: E, cfg: OptConfig, mut st: OptState, t: Int, lr: Float32,
    P: FP, Gr: FP, s1: FP, s2: FP, s3: FP, n: Int,
) raises:
    var a = Args()
    a.p0 = P
    a.p1 = Gr
    a.p2 = s1
    a.p3 = s2
    a.p4 = s3
    a.i0 = cfg.kind
    a.i1 = t
    a.i2 = cfg.flags
    a.f0 = lr
    a.f1 = cfg.f1
    a.f2 = cfg.f2
    a.f3 = cfg.eps
    a.f4 = cfg.wd
    var sc = opt_scalars(cfg, st, t, lr)
    a.f5 = sc[0]
    a.f6 = sc[1]
    a.f7 = sc[2]
    ex.launch[OP_OPT](a, n)


# ------------------------------------------------------------------ fit / predict
def rnn_fit[E: Exec](
    mut ex: E, net: Net, task: Int,
    X: FP, Y: FP, N: Int, T: Int,
    order: FP, n_order: Int, steps: MutPointer[Int32, MutUntrackedOrigin], n_steps: Int,
    Pio: FP, losses: FP, lrs: FP, cfg: OptConfig, init_acc: Float32,
) raises:
    """`n_steps` optimizer steps; step k trains on the samples
    order[steps[2k] : steps[2k] + steps[2k+1]] (indices as floats). Pio holds
    the starting parameters and receives the trained ones; losses[k] is the
    mean loss of step k before its update."""
    var np_ = net.n_params()
    var ycols = net.O if task == TASK_MSE else 1
    var bmax = 1
    for k in range(n_steps):
        var c = Int(steps.unsafe_load(2 * k + 1))
        if c > bmax:
            bmax = c
    var dX = ex.alloc(N * T * net.D)
    ex.upload(dX, X, N * T * net.D)
    var dY = ex.alloc(N * ycols)
    ex.upload(dY, Y, N * ycols)
    var dord = ex.alloc(n_order)
    ex.upload(dord, order, n_order)
    var P = ex.alloc(np_)
    ex.upload(P, Pio, np_)
    var Gr = ex.alloc(np_)
    var s1 = ex.alloc(np_)
    var s2 = ex.alloc(np_)
    var s3 = ex.alloc(np_)
    if init_acc != Float32(0.0):
        fill(ex, s2, np_, init_acc)
    var dloss = ex.alloc(n_steps)
    var w = Work(ex, net, T, bmax, True)
    var st = OptState()
    for k in range(n_steps):
        var off = Int(steps.unsafe_load(2 * k))
        var B = Int(steps.unsafe_load(2 * k + 1))
        gather_seq(ex, dX, dord, w.xb, T, B, net.D, off)
        var hT = forward(ex, net, P, w.xb, T, B, w)
        head(ex, net, P, hT, B, w.yhat)
        if task == TASK_MSE:
            gather_rows(ex, dY, dord, w.yb, B, net.O, off)
            var a = Args()
            a.p0 = w.yhat
            a.p1 = w.yb
            a.p2 = w.dy
            a.p3 = w.sq
            a.f0 = Float32(2.0) / Float32(B * net.O)
            ex.launch[OP_MSE](a, B * net.O)
            sum_into(ex, w.sq, B * net.O, dloss, k, Float32(1.0) / Float32(B * net.O))
        else:
            gather_rows(ex, dY, dord, w.yb, B, 1, off)
            var a = Args()
            a.p0 = w.yhat
            a.p1 = w.yb
            a.p2 = w.dy
            a.p3 = w.sq
            a.i0 = B
            a.i1 = net.O
            a.f0 = Float32(1.0) / Float32(B)
            ex.launch[OP_CE](a, B)
            sum_into(ex, w.sq, B, dloss, k, Float32(1.0) / Float32(B))
        backward(ex, net, P, Gr, w.xb, T, B, w)
        opt_step(ex, cfg, st, k + 1, ftz(lrs.unsafe_load(k)), P, Gr, s1, s2, s3, np_)
    ex.sync()
    ex.download(Pio, P, np_)
    ex.download(losses, dloss, n_steps)


def rnn_predict[E: Exec](
    mut ex: E, net: Net, task: Int, X: FP, N: Int, T: Int, Pin: FP,
    dst: FP, seq: FP, want_seq: Bool, chunk: Int,
) raises:
    """dst [N, O]: the head's output (task MSE) or its softmax (task CE);
    with want_seq, seq [N, T, H] is the top layer's h_1..h_T batch-first."""
    var np_ = net.n_params()
    var H = net.H
    var bmax = chunk if chunk < N else N
    var dX = ex.alloc(N * T * net.D)
    ex.upload(dX, X, N * T * net.D)
    var idx = List[Float32]()
    for i in range(N):
        idx.append(Float32(i))
    var didx = ex.alloc(N)
    ex.upload(didx, FP(unsafe_from_address=Int(idx.unsafe_ptr())), N)
    var P = ex.alloc(np_)
    ex.upload(P, Pin, np_)
    var dout = ex.alloc(N * net.O)
    var dseq = ex.alloc(N * T * H if want_seq else 1)
    var w = Work(ex, net, T, bmax, False)
    var b0 = 0
    while b0 < N:
        var B = bmax if b0 + bmax <= N else N - b0
        gather_seq(ex, dX, didx, w.xb, T, B, net.D, b0)
        var hT = forward(ex, net, P, w.xb, T, B, w)
        head(ex, net, P, hT, B, dout + b0 * net.O)
        if task == TASK_CE:
            var a = Args()
            a.p0 = dout + b0 * net.O
            a.p1 = dout + b0 * net.O
            a.i1 = net.O
            ex.launch[OP_SOFTMAX](a, B)
        if want_seq:
            var a = Args()
            a.p0 = w.hall[net.L - 1] + B * H
            a.p1 = dseq
            a.i0 = T
            a.i1 = B
            a.i2 = H
            a.i3 = b0
            ex.launch[OP_SEQ_OUT](a, B * T * H)
        b0 += B
    ex.sync()
    ex.download(dst, dout, N * net.O)
    if want_seq:
        ex.download(seq, dseq, N * T * H)
    _ = idx^
