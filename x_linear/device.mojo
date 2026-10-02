# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE LINEAR LANE ON THE DEVICE (lane/algos-linear, 2026-09-27).

Pass 1: every fit is x_linear/dispatch.mojo's `fit_dispatch`, run by ONE
device thread, so the GPU executes the host's exact sequence of operations.
Speed phase: a fit `team_fit` names runs on ONE BLOCK of LINEAR_TPB threads
(x_linear/team.mojo): each stored value still comes from one thread's
one-thread sequence, so the bits are the host's.
Scoring is one thread per (row, output) pair. A parallel fit schedule with
the same fold order is pass 2's speed work.

Every entry runs on ONE process-lifetime DeviceContext (`linear_ctx`, the
x_cnn `_Global` pattern; CURRENT DIRECTIVES 2026-09-27: a context per call
exhausts Metal's per-process command queues, and x_cluster/x_neighbors hung
on the second GPU call of a process). Each entry's buffers are released
before it returns; the context stays.
"""
from std.gpu import block_idx, block_dim, thread_idx
from std.ffi import _Global
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, NUMERIC_FAST
from x_linear.ops import FP, IP
from x_linear.dispatch import fit_dispatch, decision_one, team_fit, team_rows, team_own, ALGO_SGD, ALGO_LARS, ALGO_RIDGE_KFOLD, ALGO_BAYES, ALGO_ARD
from x_linear.ops import ld, st, ldi, sti, fd, i2f, fa, fs, fm, fabs, fmad, fmax, fmin, shuffle
from x_linear.witness import Witness, witness_end, WITNESS_TRIES
from x_linear.sgd import sgd_mb_on, mb_sub_size, mb_dblk, mb_row, mb_row_dot, mb_rowsq, mb_block_dot, MB_DBLK, LR_PA1, LR_PA2, mb_part, mb_step, mb_bias_step, mb_subs, mb_eta, mb_optimal_init, mb_penalty, LR_OPTIMAL, LR_ADAPTIVE, P_L2, P_L1
from x_linear.sgd import (
    sgd_loss, sgd_dloss, sgd_reg_block, _sgd_target, _clip_one,
    ws_mul, ws_div, ws_decay, ws_clip, WS_RESET, ff_add, oc_hinge, oc_offset,
    L_HINGE, LR_INVSCALING, P_NONE, P_EN,
)
from checks.numerics import identical_pow
from x_linear.ridgecv import kf_start, kf_end, kf_mean, kf_cross, kf_solve, kf_pred, kf_score
from x_linear.tops import upper_cell, fold_fa, chain_cfmad
from std.os import getenv
from x_linear.team import LINEAR_TPB, team_work, device_team, solo, team_barrier


struct _LinearContext(Defaultable, Movable):
    """The slot `linear_ctx` fills on first use; one per numeric tier so a
    FAST and an IDENTICAL .so in one process never share it."""
    var ctx: Optional[DeviceContext]

    def __init__(out self):
        self.ctx = Optional[DeviceContext]()


comptime _CTX_NAME = "MojoXLinearContextIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoXLinearContextFast"
comptime X_LINEAR_CONTEXT = _Global[StorageType=_LinearContext, name=_CTX_NAME, init_fn=_LinearContext.__init__]


def linear_ctx() raises -> DeviceContext:
    """The shared context, created on first use."""
    var slot = X_LINEAR_CONTEXT.get_or_create_ptr()
    if not slot[].ctx:
        slot[].ctx = DeviceContext()
    return slot[].ctx.value().copy()


#: FAST on Apple (lane/linear-apple3): the fits x_linear/blocks.mojo names run
#: their row passes on n / 1024 blocks, the control on the host, instead of
#: one program on ONE block. WIP: opt-in (`-D MOJOLEARN_X_LINEAR_BLOCKS=1`)
#: until its A/B and paired quality check are on record. IDENTICAL and the
#: other vendors never compile the branch.
comptime X_LINEAR_BLOCKS = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and is_defined["MOJOLEARN_X_LINEAR_BLOCKS"]()
)
#: The same for the fits that work from the centered Gram
#: (x_linear/blocks_gram.mojo; WIP, opt-in `-D MOJOLEARN_X_LINEAR_BLOCKS_GRAM=1`).
comptime X_LINEAR_BLOCKS_GRAM = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and is_defined["MOJOLEARN_X_LINEAR_BLOCKS_GRAM"]()
)


def fit_kernel(
    algo: Int32, x: FP, y: FP, n: Int32, d: Int32, ip: IP, fp: FP, res: FP, fw: FP, iw: IP, tw: FP,
):
    """ONE block. A team fit runs on every thread of it; any other fit on
    thread 0 alone, as a team of one (x_linear/team.mojo)."""
    var a = Int(algo)
    var bufs = team_rows(a, ip)
    var own = team_own(a, Int(d))
    if team_fit(a):
        fit_dispatch(device_team(tw, Int(n), bufs, own), a, x, y, Int(n), Int(d), ip, fp, res, fw, iw)
    elif Int(thread_idx.x) == 0:
        fit_dispatch(solo(tw, Int(n), bufs, own), a, x, y, Int(n), Int(d), ip, fp, res, fw, iw)


def decision_kernel(x: FP, wb: FP, n: Int32, d: Int32, k: Int32, link: Int32, res: FP):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(n) * Int(k):
        var i = t // Int(k)
        var c = t % Int(k)
        res.unsafe_store(t, decision_one(x, i, Int(d), wb, c, Int(link)))


comptime XG_TPB = 256


def _xg_blocks(count: Int) -> Int:
    return (count + XG_TPB - 1) // XG_TPB


def xg_means_kernel(x: FP, n: Int32, d: Int32, fi: Int32, fw: FP):
    """`t_col_means` as a grid: thread j folds column j ascending
    (`fold_fa`) and divides by n, the same statements; zeros without an
    intercept, as `lars_fit` fills them."""
    var j = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    if j < Int(d):
        if fi != 0:
            st(fw, j, fd(fold_fa(x, j, Int(d), Int(n)), i2f(Int(n))))
        else:
            st(fw, j, Float32(0))


def xg_gram_kernel(x: FP, n: Int32, d: Int32, fw: FP):
    """`t_centered_gram` as a grid (lane/neural-net-experiment, 2026-09-30,
    the classical pass): one thread per upper-triangle cell, each the same
    `chain_cfmad` over the rows ascending from the means in fw[0, d), into
    G at fw[d, d + d*d). The team form ran the same chains on ONE block of
    256 threads, 96 chains of a million rows per thread at 220 features:
    15.4 s on an L40S for `lars` on istella (bench_board 0.8.25) against
    cuML's 0.064. Per cell the chain is unchanged, so the bits are the
    team form's; only the thread that runs it differs."""
    var dd = Int(d)
    var cells = dd * (dd + 1) // 2
    var c = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    if c < cells:
        var jk = upper_cell(c, dd)
        var j = jk[0]
        var k = jk[1]
        var acc = chain_cfmad(x, j, dd, ld(fw, j), x, k, dd, ld(fw, k), Int(n))
        st(fw, dd + j * dd + k, acc)
        st(fw, dd + k * dd + j, acc)


def _sgd_on_host() -> Bool:
    """`MOJOLEARN_X_LINEAR_SGD_HOST=1` runs the per-sample SGD fit on the host
    inside the device binding (the A/B arm); by default (lane/neural-pass139,
    GPU-only rule) it runs on the device, `_sgd_ps_grid`."""
    return String(getenv("MOJOLEARN_X_LINEAR_SGD_HOST")) == "1"


def _lars_grid_gram() -> Bool:
    """`MOJOLEARN_X_LINEAR_LARS_GRID_GRAM=0` keeps the Gram on the team
    (the A/B arm); default the grid kernel."""
    return String(getenv("MOJOLEARN_X_LINEAR_LARS_GRID_GRAM")) != "0"


def _bayes_grid_gram() -> Bool:
    """`MOJOLEARN_X_LINEAR_BAYES_GRID_GRAM=0` keeps BayesianRidge's and
    ARD's Gram on the team (the A/B arm); default the grid kernel."""
    return String(getenv("MOJOLEARN_X_LINEAR_BAYES_GRID_GRAM")) != "0"


def _fit_on_host(
    algo: Int, x: FP, y: FP, n: Int, d: Int,
    ip: List[Int32], fp: List[Float32], n_out: Int, n_fw: Int, n_iw: Int, res: FP,
) raises:
    """The host form of a fit, from the device binding (lane/neural-net-
    experiment, 2026-09-30, the classical pass): exactly what
    bindings/_mojolearn_x_linear_host.mojo runs, a `solo` team on host
    scratch. For SGD, whose program is one sample after the next (their
    plain SGD, the order is the algorithm), the device ran that sequence on
    ONE GPU THREAD: 630 s for a million rows of istella on an L40S against
    sklearn's 55 s (bench_board 0.8.25, `sgd-reg`). The host form is the
    same program, the identical tier's own reference, on a CPU thread; the
    one-vs-rest problems of a classifier run as independent units."""
    var hip = ip.copy()
    var hfp = fp.copy()
    var fw = List[Float32](length=max(n_fw, 1), fill=Float32(0))
    var iw = List[Int32](length=max(n_iw, 1), fill=Int32(0))
    var bufs = team_rows(algo, IP(unsafe_from_address=Int(hip.unsafe_ptr())))
    var own = team_own(algo, d)
    var tw = List[Float32](length=team_work(n, bufs, own), fill=Float32(0))
    for i in range(n_out):
        res.unsafe_store(i, Float32(0))
    fit_dispatch(
        solo(FP(unsafe_from_address=Int(tw.unsafe_ptr())), n, bufs, own), algo, x, y, n, d,
        IP(unsafe_from_address=Int(hip.unsafe_ptr())), FP(unsafe_from_address=Int(hfp.unsafe_ptr())),
        res, FP(unsafe_from_address=Int(fw.unsafe_ptr())), IP(unsafe_from_address=Int(iw.unsafe_ptr())),
    )
    _ = hip^
    _ = hfp^
    _ = fw^
    _ = iw^
    _ = tw^



# ------------------------------------------------ minibatch SGD on the grid (lane/neural-pass103)
# x_linear/sgd.mojo `sgd_mb_one` with each batch as three launches: the rows'
# loss derivatives (one thread a row), the gradient partials (one thread a
# (sub-block, column), sub-block-major so neighbouring threads read one row's
# words), the step (one thread a weight, the intercept and the objective).
# The shuffle, the rate schedule and the stopping run on the host with the
# same statements; an epoch's batches are enqueued without a sync.
@always_inline
def _sgd_mb_rows_kernel_body(x: FP, ys: FP, idx: IP, start: Int32, bs: Int32, d: Int32, w: FP, bias: FP, loss: Int32,
                       eps: Float32, swp: FP, has_sw: Int32, wpos: Float32, wneg: Float32, has_cw: Int32,
                       dlv: FP, lv: FP, lr: Int32, eta0: Float32, dblk: Int32):
    var r = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    if r < Int(bs):
        var i = Int(idx.unsafe_load(Int(start) + r))
        var o = mb_row(x, ys, i, Int(d), w, 0, ld(bias, 0), Int(loss), eps, swp, has_sw != 0, wpos, wneg, has_cw != 0,
                       Int(lr), eta0, Int(dblk))
        st(dlv, r, o[0])
        st(lv, r, o[1])


def sgd_mb_rows_kernel(x: FP, ys: FP, idx: IP, start: Int32, bs: Int32, d: Int32, w: FP, bias: FP, loss: Int32,
                       eps: Float32, swp: FP, has_sw: Int32, wpos: Float32, wneg: Float32, has_cw: Int32,
                       dlv: FP, lv: FP, lr: Int32, eta0: Float32, dblk: Int32, wf: IP, woff: Int32, nonce: Int32):
    _sgd_mb_rows_kernel_body(x, ys, idx, start, bs, d, w, bias, loss, eps, swp, has_sw, wpos, wneg, has_cw, dlv, lv, lr, eta0,
                             dblk)
    witness_end(wf, woff, nonce)

@always_inline
def _sgd_mb_parts_kernel_body(x: FP, d: Int32, idx: IP, start: Int32, dlv: FP, lv: FP, bs: Int32, nsub: Int32, parts: FP,
                              sub: Int32):
    var q = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    var dd = Int(d)
    var subs = mb_subs(Int(bs), Int(sub))
    var s = q // (dd + 2)
    var j = q - s * (dd + 2)
    if s < subs:
        st(parts, j * Int(nsub) + s, mb_part(x, dd, idx, Int(start), dlv, lv, j, s, Int(bs), Int(sub)))


def sgd_mb_parts_kernel(x: FP, d: Int32, idx: IP, start: Int32, dlv: FP, lv: FP, bs: Int32, nsub: Int32, parts: FP,
                        sub: Int32, wf: IP, woff: Int32, nonce: Int32):
    _sgd_mb_parts_kernel_body(x, d, idx, start, dlv, lv, bs, nsub, parts, sub)
    witness_end(wf, woff, nonce)

@always_inline
def _sgd_mb_step_kernel_body(parts: FP, nsub: Int32, bs: Int32, d: Int32, w: FP, bias: FP, obj: FP, eta: Float32,
                       alpha: Float32, l1r: Float32, penalty: Int32, fi: Int32, need_obj: Int32, one_class: Int32,
                       bsum: Int32, sub: Int32):
    var j = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    var dd = Int(d)
    var subs = mb_subs(Int(bs), Int(sub))
    if j > dd + 1:
        return
    var g = Float32(0)
    for s in range(subs):
        g = fa(g, ld(parts, j * Int(nsub) + s))
    if j < dd:
        st(w, j, mb_step(ld(w, j), g, Int(bs), eta, alpha, l1r, Int(penalty), bsum != 0))
    elif j == dd:
        if fi != 0:
            st(bias, 0, mb_bias_step(ld(bias, 0), g, Int(bs), eta, alpha, one_class != 0, bsum != 0))
    elif need_obj != 0:
        st(obj, 0, fa(ld(obj, 0), g))


def sgd_mb_step_kernel(parts: FP, nsub: Int32, bs: Int32, d: Int32, w: FP, bias: FP, obj: FP, eta: Float32,
                       alpha: Float32, l1r: Float32, penalty: Int32, fi: Int32, need_obj: Int32, one_class: Int32,
                       bsum: Int32, sub: Int32, wf: IP, woff: Int32, nonce: Int32):
    _sgd_mb_step_kernel_body(parts, nsub, bs, d, w, bias, obj, eta, alpha, l1r, penalty, fi, need_obj, one_class, bsum, sub)
    witness_end(wf, woff, nonce)


# lane/neural-pass134 (2026-10-02): K batches in ONE launch on one block. The
# three per-batch kernels each ran as a single block at batch 256 (the peer's
# MI325X: perceptron istella 24.6 s against 7.7 on the L40S, launch-bound);
# here one block runs each batch's rows, partials and step in order with
# device-ordering barriers between them, the same helpers in the same order,
# and the rate of each batch from the same schedule (`mb_eta` with t
# counting as the host does). `MOJOLEARN_X_LINEAR_SGD_CHUNK=1` restores one
# batch a launch (the three kernels).
comptime SGD_CHUNK_DEFAULT = 64


def sgd_rowsq_kernel(x: FP, n: Int32, d: Int32, sqp: FP, wf: IP, woff: Int32, nonce: Int32):
    """sqp[i] = `mb_rowsq` of row i: the PA rates' |x_i|^2, the chain
    `mb_row` walks, once an epoch instead of once a visit."""
    var i = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    if i < Int(n):
        st(sqp, i, mb_rowsq(x, i, Int(d)))
    witness_end(wf, woff, nonce)


def _sgd_mb_chunk_kernel_body(
    x: FP, ys: FP, idx: IP, w: FP, bias: FP, swp: FP, dlv: FP, lv: FP, parts: FP, obj: FP, ci: IP, cf: FP,
    dotp: FP, sqp: FP,
    start0: Int32, nbat: Int32, t0: Int32,
):
    """ci: n, batch, d, loss, has_sw, has_cw, lr, nsub, penalty, fi, need_obj,
    one_class, bsum; cf: eps, wpos, wneg, eta0, eta, alpha, l1r, power_t,
    opt_init (Metal takes at most 31 kernel arguments)."""
    var nn = ldi(ci, 0)
    var batch = ldi(ci, 1)
    var dd = ldi(ci, 2)
    var loss = ldi(ci, 3)
    var has_sw = ldi(ci, 4) != 0
    var has_cw = ldi(ci, 5) != 0
    var lr = ldi(ci, 6)
    var nsub = ldi(ci, 7)
    var penalty = ldi(ci, 8)
    var fi = ldi(ci, 9) != 0
    var need_obj = ldi(ci, 10) != 0
    var one_class = ldi(ci, 11) != 0
    var bsum = ldi(ci, 12) != 0
    var sub = ldi(ci, 13)
    var dblk = ldi(ci, 14)
    var has_sq = ldi(ci, 15) != 0
    var eps = ld(cf, 0)
    var wpos = ld(cf, 1)
    var wneg = ld(cf, 2)
    var eta0 = ld(cf, 3)
    var eta = ld(cf, 4)
    var alpha = ld(cf, 5)
    var l1r = ld(cf, 6)
    var power_t = ld(cf, 7)
    var opt_init = ld(cf, 8)
    var tid = Int(thread_idx.x)
    var nt = Int(block_dim.x)
    var start = Int(start0)
    var t = Int(t0)
    for _q in range(Int(nbat)):
        var bs = min(batch, nn - start)
        if bs <= 0:
            break
        var et = mb_eta(lr, eta, eta0, alpha, power_t, opt_init, t)
        if dblk == MB_DBLK:
            # one task per (row, MB_DBLK-column block), its loads all in
            # flight before its chain, then each row's blocks folded
            # ascending: `mb_dot`'s words (lane/neural-pass132: one thread
            # walking a whole random row missed cache every step on the
            # MI325X, istella 220 columns)
            var nb = (dd + MB_DBLK - 1) // MB_DBLK
            for q in range(tid, bs * nb, nt):
                var r = q // nb
                st(dotp, q, mb_block_dot(x, Int(idx.unsafe_load(start + r)), dd, w, 0, q - r * nb))
            team_barrier()
            for r in range(tid, bs, nt):
                var i = Int(idx.unsafe_load(start + r))
                var dot = Float32(0)
                for bb in range(nb):
                    dot = fa(dot, ld(dotp, r * nb + bb))
                var o = mb_row_dot(x, ys, i, dd, dot, ld(bias, 0), loss, eps, swp, has_sw, wpos, wneg, has_cw, lr, eta0,
                                   sqp, has_sq)
                st(dlv, r, o[0])
                st(lv, r, o[1])
        else:
            for r in range(tid, bs, nt):
                var i = Int(idx.unsafe_load(start + r))
                var o = mb_row(x, ys, i, dd, w, 0, ld(bias, 0), loss, eps, swp, has_sw, wpos, wneg, has_cw, lr, eta0, dblk)
                st(dlv, r, o[0])
                st(lv, r, o[1])
        team_barrier()
        var subs = mb_subs(bs, sub)
        for q in range(tid, (dd + 2) * subs, nt):
            var sb = q // (dd + 2)
            var j = q - sb * (dd + 2)
            st(parts, j * nsub + sb, mb_part(x, dd, idx, start, dlv, lv, j, sb, bs, sub))
        team_barrier()
        for j in range(tid, dd + 2, nt):
            var g = Float32(0)
            for sb in range(subs):
                g = fa(g, ld(parts, j * nsub + sb))
            if j < dd:
                st(w, j, mb_step(ld(w, j), g, bs, et, alpha, l1r, penalty, bsum))
            elif j == dd:
                if fi:
                    st(bias, 0, mb_bias_step(ld(bias, 0), g, bs, et, alpha, one_class, bsum))
            elif need_obj:
                st(obj, 0, fa(ld(obj, 0), g))
        team_barrier()
        t += bs if bsum else 1
        start += bs


def sgd_mb_chunk_kernel(
    x: FP, ys: FP, idx: IP, w: FP, bias: FP, swp: FP, dlv: FP, lv: FP, parts: FP, obj: FP, ci: IP, cf: FP,
    dotp: FP, sqp: FP, start0: Int32, nbat: Int32, t0: Int32, wf: IP, woff: Int32, nonce: Int32,
):
    _sgd_mb_chunk_kernel_body(x, ys, idx, w, bias, swp, dlv, lv, parts, obj, ci, cf, dotp, sqp, start0, nbat, t0)
    witness_end(wf, woff, nonce)


def _sgd_chunk() -> Int:
    var v = String(getenv("MOJOLEARN_X_LINEAR_SGD_CHUNK"))
    if v == "":
        return SGD_CHUNK_DEFAULT
    try:
        return max(1, Int(v))
    except:
        return SGD_CHUNK_DEFAULT

def _sgd_mb_grid(x: FP, n_x: Int, y: FP, n_y: Int, n: Int, d: Int, ip: List[Int32], fp: List[Float32],
                 n_out: Int, res: FP) raises:
    var ctx = linear_ctx()
    var k = Int(ip[0])
    var loss = Int(ip[1])
    var penalty = Int(ip[2])
    var lr = Int(ip[3])
    var fi = Int(ip[4]) != 0
    var max_iter = Int(ip[5])
    var nic = Int(ip[6])
    var do_shuffle = Int(ip[7]) != 0
    var seed = (UInt64(UInt32(ip[9])) << 32) | UInt64(UInt32(ip[8]))
    var has_sw = Int(ip[10]) != 0
    var has_cw = Int(ip[11]) != 0
    var batch = Int(ip[12])
    var bsum = len(ip) > 13 and Int(ip[13]) != 0
    var alpha = fp[0]
    var l1r = fp[1]
    var eta0 = fp[2]
    var power_t = fp[3]
    var eps = fp[4]
    var tol = fp[5]
    if penalty == P_L2:
        l1r = Float32(0)
    elif penalty == P_L1:
        l1r = Float32(1)
    var problems = k if k > 2 else 1
    var one_class = k == 1
    var sub = mb_sub_size(batch)
    var dblk = mb_dblk(batch)
    var nsub = mb_subs(batch, sub)
    var dx = ctx.enqueue_create_buffer[DType.float32](max(n_x, 1))
    var dys = ctx.enqueue_create_buffer[DType.float32](max(n, 1))
    var dsw = ctx.enqueue_create_buffer[DType.float32](max(n, 1))
    var didx = ctx.enqueue_create_buffer[DType.int32](max(n, 1))
    var ddl = ctx.enqueue_create_buffer[DType.float32](batch)
    var dlv = ctx.enqueue_create_buffer[DType.float32](batch)
    var dparts = ctx.enqueue_create_buffer[DType.float32]((d + 2) * nsub)
    var dw = ctx.enqueue_create_buffer[DType.float32](max(d, 1))
    var dbias = ctx.enqueue_create_buffer[DType.float32](1)
    var dobj = ctx.enqueue_create_buffer[DType.float32](1)
    var dws = ctx.enqueue_create_buffer[DType.float32](max(d, 1))
    var dbs = ctx.enqueue_create_buffer[DType.float32](1)
    var wcap = 0
    var ws0 = 0
    while ws0 < n:
        var wbs = min(batch, n - ws0)
        wcap += _xg_blocks(wbs) + _xg_blocks((d + 2) * mb_subs(wbs, sub)) + _xg_blocks(d + 2)
        ws0 += wbs
    wcap += _xg_blocks(n)
    var wit = Witness(ctx, max(wcap, 1))
    var chunk = _sgd_chunk()
    var dci = ctx.enqueue_create_buffer[DType.int32](16)
    var ddotp = ctx.enqueue_create_buffer[DType.float32](max(batch * ((d + MB_DBLK - 1) // MB_DBLK), 1))
    var pa_rate = lr == LR_PA1 or lr == LR_PA2
    var dsq = ctx.enqueue_create_buffer[DType.float32](max(n, 1) if pa_rate else 1)
    var dcf = ctx.enqueue_create_buffer[DType.float32](9)
    var hci = List[Int32](length=16, fill=Int32(0))
    var hcf = List[Float32](length=9, fill=Float32(0))
    hci[0] = Int32(n)
    hci[1] = Int32(batch)
    hci[2] = Int32(d)
    hci[3] = Int32(loss)
    hci[4] = Int32(1 if has_sw else 0)
    hci[5] = Int32(1 if has_cw else 0)
    hci[6] = Int32(lr)
    hci[7] = Int32(nsub)
    hci[8] = Int32(penalty)
    hci[9] = Int32(1 if fi else 0)
    hci[12] = Int32(1 if bsum else 0)
    hci[13] = Int32(sub)
    hci[14] = Int32(dblk)
    hci[15] = Int32(1 if pa_rate else 0)
    hcf[0] = eps
    hcf[3] = eta0
    hcf[5] = alpha
    hcf[6] = l1r
    hcf[7] = power_t
    if n_x > 0:
        ctx.enqueue_copy(dst_buf=dx, src_ptr=x)
    if has_sw:
        ctx.enqueue_copy(dst_buf=dsw, src_ptr=y + n)
    var ys = List[Float32](length=max(n, 1), fill=Float32(0))
    var idx = List[Int32](length=max(n, 1), fill=Int32(0))
    var hw = List[Float32](length=max(d, 1), fill=Float32(0))
    var hb = List[Float32](length=1, fill=Float32(0))
    var ho = List[Float32](length=1, fill=Float32(0))
    var need_obj = tol > Float32(-3.0e38)
    var max_epochs = 0
    var status = 0
    for c in range(problems):
        for i in range(n):
            if one_class:
                ys[i] = Float32(1)  # sgd_fit's one-class target (y is not read)
            else:
                var v = y.unsafe_load(i)
                if k == 0:
                    ys[i] = v
                elif k == 2:
                    ys[i] = Float32(1) if v == Float32(1) else Float32(-1)
                else:
                    ys[i] = Float32(1) if v == i2f(c) else Float32(-1)
            idx[i] = Int32(i)
        ctx.enqueue_copy(dst_buf=dys, src_ptr=ys.unsafe_ptr())
        dw.enqueue_fill(Float32(0))
        dbias.enqueue_fill(Float32(1) if one_class else Float32(0))
        var wpos = fp[6 + c] if has_cw else Float32(1)
        var wneg = fp[6 + problems + c] if has_cw else Float32(1)
        var rng = seed + UInt64(1000003) * UInt64(c)
        var eta = eta0
        var opt_init = mb_optimal_init(loss, alpha, eps) if lr == LR_OPTIMAL else Float32(0)
        var t = 1
        var best = Float32(3.0e38)
        var no_improve = 0
        var epochs = 0
        var failed = False
        var ipp = IP(unsafe_from_address=Int(idx.unsafe_ptr()))
        for epoch in range(max_iter):
            epochs = epoch + 1
            if do_shuffle:
                shuffle(ipp, n, rng)
            # the epoch as ONE guarded unit (x_linear/witness.mojo): its steps
            # update the weights in place, so a cut epoch restores the weights
            # it started from and replays the same batches
            ctx.enqueue_copy(dst_buf=dws, src_buf=dw)
            ctx.enqueue_copy(dst_buf=dbs, src_buf=dbias)
            var t_start = t
            var tries = 0
            while True:
                var nonce = wit.begin()
                var wo = 0
                t = t_start
                ctx.enqueue_copy(dst_buf=didx, src_ptr=idx.unsafe_ptr())
                dobj.enqueue_fill(Float32(0))
                var start = 0
                if chunk > 1 and batch <= XG_TPB and d + 2 <= XG_TPB:
                    if pa_rate:
                        ctx.enqueue_function[sgd_rowsq_kernel](
                            dx.unsafe_ptr(), Int32(n), Int32(d), dsq.unsafe_ptr(), wit.p(), Int32(wo), nonce,
                            grid_dim=_xg_blocks(n), block_dim=XG_TPB,
                        )
                        wo += _xg_blocks(n)
                    hcf[4] = eta
                    hcf[1] = wpos
                    hcf[2] = wneg
                    hcf[8] = opt_init
                    hci[10] = Int32(1 if need_obj else 0)
                    hci[11] = Int32(1 if one_class else 0)
                    ctx.enqueue_copy(dst_buf=dci, src_ptr=hci.unsafe_ptr())
                    ctx.enqueue_copy(dst_buf=dcf, src_ptr=hcf.unsafe_ptr())
                    while start < n:
                        var nbt = 0
                        var tt = t
                        var s1 = start
                        while nbt < chunk and s1 < n:
                            var bs1 = min(batch, n - s1)
                            tt += bs1 if bsum else 1
                            s1 += bs1
                            nbt += 1
                        ctx.enqueue_function[sgd_mb_chunk_kernel](
                            dx.unsafe_ptr(), dys.unsafe_ptr(), didx.unsafe_ptr(), dw.unsafe_ptr(), dbias.unsafe_ptr(),
                            dsw.unsafe_ptr(), ddl.unsafe_ptr(), dlv.unsafe_ptr(), dparts.unsafe_ptr(), dobj.unsafe_ptr(),
                            dci.unsafe_ptr(), dcf.unsafe_ptr(), ddotp.unsafe_ptr(), dsq.unsafe_ptr(), Int32(start), Int32(nbt), Int32(t),
                            wit.p(), Int32(wo), nonce, grid_dim=1, block_dim=XG_TPB,
                        )
                        wo += 1
                        t = tt
                        start = s1
                else:
                    while start < n:
                        var bs = min(batch, n - start)
                        var et = mb_eta(lr, eta, eta0, alpha, power_t, opt_init, t)
                        ctx.enqueue_function[sgd_mb_rows_kernel](
                            dx.unsafe_ptr(), dys.unsafe_ptr(), didx.unsafe_ptr(), Int32(start), Int32(bs), Int32(d),
                            dw.unsafe_ptr(), dbias.unsafe_ptr(), Int32(loss), eps, dsw.unsafe_ptr(), Int32(1 if has_sw else 0),
                            wpos, wneg, Int32(1 if has_cw else 0), ddl.unsafe_ptr(), dlv.unsafe_ptr(), Int32(lr), eta0, Int32(dblk),
                            wit.p(), Int32(wo), nonce, grid_dim=_xg_blocks(bs), block_dim=XG_TPB,
                        )
                        wo += _xg_blocks(bs)
                        ctx.enqueue_function[sgd_mb_parts_kernel](
                            dx.unsafe_ptr(), Int32(d), didx.unsafe_ptr(), Int32(start), ddl.unsafe_ptr(), dlv.unsafe_ptr(),
                            Int32(bs), Int32(nsub), dparts.unsafe_ptr(), Int32(sub), wit.p(), Int32(wo), nonce,
                            grid_dim=_xg_blocks((d + 2) * mb_subs(bs, sub)), block_dim=XG_TPB,
                        )
                        wo += _xg_blocks((d + 2) * mb_subs(bs, sub))
                        ctx.enqueue_function[sgd_mb_step_kernel](
                            dparts.unsafe_ptr(), Int32(nsub), Int32(bs), Int32(d), dw.unsafe_ptr(), dbias.unsafe_ptr(),
                            dobj.unsafe_ptr(), et, alpha, l1r, Int32(penalty), Int32(1 if fi else 0), Int32(1 if need_obj else 0),
                            Int32(1 if one_class else 0), Int32(1 if bsum else 0), Int32(sub), wit.p(), Int32(wo), nonce,
                            grid_dim=_xg_blocks(d + 2), block_dim=XG_TPB,
                        )
                        wo += _xg_blocks(d + 2)
                        t += bs if bsum else 1
                        start += bs
                ctx.enqueue_copy(dst_ptr=hw.unsafe_ptr(), src_buf=dw)
                ctx.enqueue_copy(dst_ptr=hb.unsafe_ptr(), src_buf=dbias)
                ctx.enqueue_copy(dst_ptr=ho.unsafe_ptr(), src_buf=dobj)
                if wit.ok(ctx, wo, "SGD epoch"):
                    break
                tries += 1
                if tries >= WITNESS_TRIES:
                    wit.fail()
                ctx.enqueue_copy(dst_buf=dw, src_buf=dws)
                ctx.enqueue_copy(dst_buf=dbias, src_buf=dbs)
            ctx.synchronize()
            var bias = hb[0]
            var finite = bias == bias and fabs(bias) < Float32(3.0e38)
            for j in range(d):
                var wj = hw[j]
                if not (wj == wj and fabs(wj) < Float32(3.0e38)):
                    finite = False
            if not finite:
                failed = True
                break
            if need_obj:
                var mean_obj = fa(fd(ho[0], i2f(n)), mb_penalty(FP(unsafe_from_address=Int(hw.unsafe_ptr())), 0, d, alpha, l1r, penalty))
                if one_class:
                    mean_obj = fa(mean_obj, fm(alpha, bias))
                if mean_obj > fs(best, tol):
                    no_improve += 1
                else:
                    no_improve = 0
                if mean_obj < best:
                    best = mean_obj
                if no_improve >= nic:
                    if lr == LR_ADAPTIVE and eta > Float32(1e-6):
                        eta = fd(eta, Float32(5))
                        no_improve = 0
                    else:
                        break
        if failed:
            status = -1
            for j in range(d):
                res.unsafe_store(c * d + j, Float32(0))
            res.unsafe_store(problems * d + c, Float32(0))
        else:
            for j in range(d):
                res.unsafe_store(c * d + j, hw[j])
            # one-class: the slot holds offset_ = 1 - intercept
            res.unsafe_store(problems * d + c, fs(Float32(1), hb[0]) if one_class else hb[0])
            if epochs > max_epochs:
                max_epochs = epochs
    res.unsafe_store(problems * d + problems, i2f(max_epochs))
    res.unsafe_store(problems * d + problems + 1, i2f(status))
    _ = ys^
    _ = idx^
    _ = hw^
    _ = hb^
    _ = ho^
    _ = dx^
    _ = dys^
    _ = dsw^
    _ = didx^
    _ = ddl^
    _ = dlv^
    _ = dparts^
    _ = dw^
    _ = dbias^
    _ = dobj^
    _ = dws^
    _ = dbs^
    _ = wit^


# ------------------------------------------------ per-sample SGD on the grid (lane/neural-pass139)
# x_linear/sgd.mojo `sgd_one` (their plain SGD: one sample after the next, the
# order is the algorithm) with each sample spread over a block (GPU-only
# rule: the one-thread team form took 630 s for a million istella rows on an
# L40S, so the device binding ran it on the host). One block a one-vs-rest
# problem (grid_dim = P); per sample, in order:
#   A. one task a MB_DBLK-column block of the predictor (`mb_block_dot`, its
#      loads all in flight before its chain) and, when `tol` reads the
#      objective, one a block of the penalty norms (`sgd_reg_block`);
#      the partials to device words, then a device-ordering barrier;
#   B. EVERY thread folds the blocks ascending from zero (`mb_dot`'s words)
#      and runs sgd_one's scalar statements (the rate, the loss, the PA step
#      from the |x_i|^2 of `mb_rowsq`, the clip, the class and sample
#      weights, the decay factor, the intercept and the cumulative L1 u) in
#      registers: the same words in every thread, so no barrier is needed
#      before C and the control flow is uniform;
#   C. thread j applies sgd_one's per-weight statements to v_j (the fold
#      `ws_mul(v_j, hi, lo)` when wscale fell below 1e-9, the step
#      `fmad(update / wscale, x_ij, v_j)`, Tsuruoka's clip `ws_clip` with
#      q_j), then a barrier.
# The weights are their WeightVector's v with wscale = (hi, lo) a float-float
# in the scalar state (x_linear/sgd.mojo `ws_decay`); w = wscale * v leaves
# the device only at the finite check and the result.
# A launch runs SGD_PS_CHUNK samples of the epoch; the epoch is ONE witness-
# guarded unit (x_linear/witness.mojo): w, q, the scalar state (wscale
# included) and t are
# restored from their epoch-start copies and the epoch replays on a rerun.
# The shuffle, the finite check, the stopping and the adaptive rate run on
# the host with sgd_one's statements. `MOJOLEARN_X_LINEAR_SGD_HOST=1` runs
# the fit on the host instead (the A/B arm).
comptime SGD_PS_CHUNK = 2048
# the per-problem scalar state ps[SGD_PS_ST c ..]: intercept, u, objective,
# the one-class intercept's low word, wscale hi, wscale lo
comptime SGD_PS_ST = 6


def _sgd_ps_kernel_body(
    x: FP, lab: FP, swp: FP, idx: IP, w: FP, q: FP, sqp: FP, part: FP,
    ps: FP, pt: IP, act: IP, pf: FP, ci: IP, cf: FP, start: Int32, cnt: Int32,
):
    """ci: n, d, loss, penalty, lr, fit_intercept, need_obj, one_class, has_sw,
    has_cw, k, has_sq; cf: alpha, l1_ratio (the penalty's), eta0, power_t,
    eps, optimal_init, decay_factor; per problem c: pf[3c..] eta, wpos, wneg;
    ps[SGD_PS_ST c..] intercept, u, objective, intercept lo (one-class), wscale hi, lo; pt[c] t; act[c] 1 while it trains."""
    var c = Int(block_idx.x)
    if ldi(act, c) == 0:
        return
    var nn = ldi(ci, 0)
    var d = ldi(ci, 1)
    var loss = ldi(ci, 2)
    var penalty = ldi(ci, 3)
    var lr = ldi(ci, 4)
    var fi = ldi(ci, 5) != 0
    var need_obj = ldi(ci, 6) != 0
    var one_class = ldi(ci, 7) != 0
    var has_sw = ldi(ci, 8) != 0
    var has_cw = ldi(ci, 9) != 0
    var k = ldi(ci, 10)
    var alpha = ld(cf, 0)
    var l1_ratio = ld(cf, 1)
    var eta0 = ld(cf, 2)
    var power_t = ld(cf, 3)
    var eps = ld(cf, 4)
    var optimal_init = ld(cf, 5)
    var decay_factor = ld(cf, 6)
    var pa_rate = lr == LR_PA1 or lr == LR_PA2
    var tid = Int(thread_idx.x)
    var nt = Int(block_dim.x)
    var nb = (d + MB_DBLK - 1) // MB_DBLK
    var need_reg = need_obj and penalty != P_NONE and not pa_rate
    var ntask = 2 * nb if need_reg else nb
    var woff = c * d
    var poff = c * 3 * nb
    var do_decay = penalty == P_L2 or penalty == P_EN
    var do_l1 = penalty == P_L1 or penalty == P_EN
    var intercept = ld(ps, SGD_PS_ST * c)
    var u = ld(ps, SGD_PS_ST * c + 1)
    var objective = Float32(0) if Int(start) == 0 else ld(ps, SGD_PS_ST * c + 2)
    var il = ld(ps, SGD_PS_ST * c + 3)
    var whi = ld(ps, SGD_PS_ST * c + 4)
    var wlo = ld(ps, SGD_PS_ST * c + 5)
    var t = ldi(pt, c)
    var eta = ld(pf, 3 * c)
    var wpos = ld(pf, 3 * c + 1)
    var wneg = ld(pf, 3 * c + 2)
    for r in range(Int(start), Int(start) + Int(cnt)):
        var i = ldi(idx, c * nn + r)
        # A: the blocks
        for qq in range(tid, ntask, nt):
            if qq < nb:
                st(part, poff + qq, mb_block_dot(x, i, d, w, woff, qq))
            else:
                var rb = sgd_reg_block(w, woff, d, qq - nb)
                st(part, poff + qq, rb[0])
                st(part, poff + nb + qq, rb[1])
        team_barrier()
        # B: sgd_one's scalar statements, in every thread
        var dot = Float32(0)
        for b in range(nb):
            dot = fa(dot, ld(part, poff + b))
        var y = _sgd_target(k, c, ld(lab, i))
        var dotw = ws_mul(dot, whi, wlo)
        var p = fa(fa(dotw, il), intercept) if one_class else fa(dotw, intercept)
        if lr == LR_OPTIMAL:
            eta = fd(Float32(1), fm(alpha, fs(fa(optimal_init, i2f(t)), Float32(1))))
        elif lr == LR_INVSCALING:
            eta = fd(eta0, identical_pow(i2f(t), power_t))
        var oc = one_class and loss == L_HINGE
        var och = oc_hinge(dotw, intercept, il)
        var cur = och[0] if oc else sgd_loss(loss, y, p, eps)
        if need_obj:
            objective = fa(objective, cur)
            if not pa_rate:
                if penalty != P_NONE:
                    var n2 = Float32(0)
                    var n1 = Float32(0)
                    for b in range(nb):
                        n2 = fa(n2, ld(part, poff + nb + b))
                        n1 = fa(n1, ld(part, poff + 2 * nb + b))
                    n2 = fm(fm(whi, whi), n2)
                    n1 = fm(whi, n1)
                    var reg = fa(fm(fm(fs(Float32(1), l1_ratio), Float32(0.5)), n2), fm(l1_ratio, n1))
                    objective = fa(objective, fm(alpha, reg))
                if one_class:
                    objective = fa(objective, fm(intercept, alpha))
        var skip = False
        var update = Float32(0)
        if pa_rate:
            var sq = ld(sqp, i)
            if lr == LR_PA1:
                if sq == 0:
                    skip = True
                else:
                    update = fmin(eta0, fd(cur, sq))
            else:
                update = fd(cur, fa(sq, fd(Float32(0.5), eta0)))
            if not skip:
                if loss == L_HINGE:
                    update = fm(update, y)
                elif fs(y, p) < 0:
                    update = -update
        else:
            var dl = och[1] if oc else sgd_dloss(loss, y, p, eps)
            if dl < Float32(-1e12):
                dl = Float32(-1e12)
            elif dl > Float32(1e12):
                dl = Float32(1e12)
            update = fm(-eta, dl)
        if not skip:
            if has_cw or has_sw:
                var cwv = Float32(1)
                if has_cw:
                    cwv = wpos if y > 0 else wneg
                var swi = ld(swp, i) if has_sw else Float32(1)
                update = fm(update, fm(cwv, swi))
            # their w.scale on wscale, the fold into v below 1e-9 (C)
            var fold = False
            var fhi = Float32(1)
            var flo = Float32(0)
            if do_decay:
                var ws = ws_decay(whi, wlo, fm(decay_factor, eta))
                whi = ws[0]
                wlo = ws[1]
                if whi < WS_RESET:
                    fold = True
                    fhi = whi
                    flo = wlo
                    whi = Float32(1)
                    wlo = Float32(0)
            var cu = Float32(0)
            if update != 0:
                cu = ws_div(update, whi, wlo)
            if fi:
                var iu = update
                if one_class:
                    iu = fs(iu, fm(eta, alpha))
                if iu != 0:
                    if one_class:
                        var ia = ff_add(intercept, il, iu)
                        intercept = ia[0]
                        il = ia[1]
                    else:
                        intercept = fa(intercept, iu)
            if do_l1:
                u = fa(u, fm(fm(l1_ratio, eta), alpha))
            # C: the weights, thread j
            var ixd = i * d
            for j in range(tid, d, nt):
                var wj = ld(w, woff + j)
                if fold:
                    wj = ws_mul(wj, fhi, flo)
                if update != 0:
                    wj = fmad(cu, ld(x, ixd + j), wj)
                if do_l1:
                    var rq = ws_clip(wj, u, ld(q, woff + j), whi)
                    st(q, woff + j, rq[1])
                    wj = rq[0]
                st(w, woff + j, wj)
            t += 1
        team_barrier()
    if tid == 0:
        st(ps, SGD_PS_ST * c, intercept)
        st(ps, SGD_PS_ST * c + 1, u)
        st(ps, SGD_PS_ST * c + 2, objective)
        st(ps, SGD_PS_ST * c + 3, il)
        st(ps, SGD_PS_ST * c + 4, whi)
        st(ps, SGD_PS_ST * c + 5, wlo)
        sti(pt, c, t)


def sgd_ps_kernel(
    x: FP, lab: FP, swp: FP, idx: IP, w: FP, q: FP, sqp: FP, part: FP,
    ps: FP, pt: IP, act: IP, pf: FP, ci: IP, cf: FP, start: Int32, cnt: Int32,
    wf: IP, woff: Int32, nonce: Int32,
):
    _sgd_ps_kernel_body(x, lab, swp, idx, w, q, sqp, part, ps, pt, act, pf, ci, cf, start, cnt)
    witness_end(wf, woff, nonce)


def _sgd_ps_grid(x: FP, n_x: Int, y: FP, n_y: Int, n: Int, d: Int, ip: List[Int32], fp: List[Float32],
                 n_out: Int, res: FP) raises:
    """The per-sample fit (batch_size 0) on the device: `sgd_fit` with every
    problem's `sgd_one` as `sgd_ps_kernel` launches; the same words."""
    var ctx = linear_ctx()
    var k = Int(ip[0])
    var loss = Int(ip[1])
    var penalty = Int(ip[2])
    var lr = Int(ip[3])
    var fi = Int(ip[4]) != 0
    var max_iter = Int(ip[5])
    var nic = Int(ip[6])
    var do_shuffle = Int(ip[7]) != 0
    var seed = (UInt64(UInt32(ip[9])) << 32) | UInt64(UInt32(ip[8]))
    var has_sw = Int(ip[10]) != 0
    var has_cw = Int(ip[11]) != 0
    var alpha = fp[0]
    var l1_ratio = fp[1]
    var eta0 = fp[2]
    var power_t = fp[3]
    var eps = fp[4]
    var tol = fp[5]
    if penalty == P_L2:
        l1_ratio = Float32(0)
    elif penalty == P_L1:
        l1_ratio = Float32(1)
    var problems = k if k > 2 else 1
    var one_class = k == 1
    var pa_rate = lr == LR_PA1 or lr == LR_PA2
    var need_obj = tol > Float32(-3.0e38)
    var nb = (d + MB_DBLK - 1) // MB_DBLK
    var optimal_init = mb_optimal_init(loss, alpha, eps) if lr == LR_OPTIMAL else Float32(0)
    var decay_factor = fm(fs(Float32(1), l1_ratio), alpha)
    var chunk = SGD_PS_CHUNK
    var dx = ctx.enqueue_create_buffer[DType.float32](max(n_x, 1))
    var dlab = ctx.enqueue_create_buffer[DType.float32](max(n, 1))
    var dsw = ctx.enqueue_create_buffer[DType.float32](max(n, 1))
    var didx = ctx.enqueue_create_buffer[DType.int32](max(problems * n, 1))
    var dw = ctx.enqueue_create_buffer[DType.float32](max(problems * d, 1))
    var dq = ctx.enqueue_create_buffer[DType.float32](max(problems * d, 1))
    var dsq = ctx.enqueue_create_buffer[DType.float32](max(n, 1) if pa_rate else 1)
    var dpart = ctx.enqueue_create_buffer[DType.float32](max(problems * 3 * nb, 1))
    var dps = ctx.enqueue_create_buffer[DType.float32](SGD_PS_ST * problems)
    var dpt = ctx.enqueue_create_buffer[DType.int32](problems)
    var dact = ctx.enqueue_create_buffer[DType.int32](problems)
    var dpf = ctx.enqueue_create_buffer[DType.float32](3 * problems)
    var dci = ctx.enqueue_create_buffer[DType.int32](12)
    var dcf = ctx.enqueue_create_buffer[DType.float32](7)
    var dws = ctx.enqueue_create_buffer[DType.float32](max(problems * d, 1))
    var dqs = ctx.enqueue_create_buffer[DType.float32](max(problems * d, 1))
    var dpss = ctx.enqueue_create_buffer[DType.float32](SGD_PS_ST * problems)
    var dpts = ctx.enqueue_create_buffer[DType.int32](problems)
    var launches = (n + chunk - 1) // chunk
    var wit = Witness(ctx, max(launches * problems + (_xg_blocks(n) if pa_rate else 0), 1))
    var hci = List[Int32](length=12, fill=Int32(0))
    hci[0] = Int32(n)
    hci[1] = Int32(d)
    hci[2] = Int32(loss)
    hci[3] = Int32(penalty)
    hci[4] = Int32(lr)
    hci[5] = Int32(1 if fi else 0)
    hci[6] = Int32(1 if need_obj else 0)
    hci[7] = Int32(1 if one_class else 0)
    hci[8] = Int32(1 if has_sw else 0)
    hci[9] = Int32(1 if has_cw else 0)
    hci[10] = Int32(k)
    hci[11] = Int32(1 if pa_rate else 0)
    var hcf = List[Float32](length=7, fill=Float32(0))
    hcf[0] = alpha
    hcf[1] = l1_ratio
    hcf[2] = eta0
    hcf[3] = power_t
    hcf[4] = eps
    hcf[5] = optimal_init
    hcf[6] = decay_factor
    var hidx = List[Int32](length=max(problems * n, 1), fill=Int32(0))
    var hw = List[Float32](length=max(problems * d, 1), fill=Float32(0))
    var hps = List[Float32](length=SGD_PS_ST * problems, fill=Float32(0))
    var hpt = List[Int32](length=problems, fill=Int32(1))
    var hact = List[Int32](length=problems, fill=Int32(1))
    var hpf = List[Float32](length=3 * problems, fill=Float32(0))
    var rngs = List[UInt64](length=problems, fill=UInt64(0))
    var etas = List[Float32](length=problems, fill=eta0)
    var bests = List[Float32](length=problems, fill=Float32(3.0e38))
    var no_imp = List[Int](length=problems, fill=0)
    var epochs = List[Int](length=problems, fill=0)
    var failed = List[Bool](length=problems, fill=False)
    for c in range(problems):
        for i in range(n):
            hidx[c * n + i] = Int32(i)
        hps[SGD_PS_ST * c] = Float32(1) if one_class else Float32(0)
        hps[SGD_PS_ST * c + 4] = Float32(1)
        rngs[c] = seed + UInt64(1000003) * UInt64(c)
        hpf[3 * c + 1] = fp[6 + c] if has_cw else Float32(1)
        hpf[3 * c + 2] = fp[6 + problems + c] if has_cw else Float32(1)
    if n_x > 0:
        ctx.enqueue_copy(dst_buf=dx, src_ptr=x)
    ctx.enqueue_copy(dst_buf=dlab, src_ptr=y)
    if has_sw:
        ctx.enqueue_copy(dst_buf=dsw, src_ptr=y + n)
    ctx.enqueue_copy(dst_buf=dci, src_ptr=hci.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dcf, src_ptr=hcf.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dps, src_ptr=hps.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dpt, src_ptr=hpt.unsafe_ptr())
    dw.enqueue_fill(Float32(0))
    dq.enqueue_fill(Float32(0))
    for epoch in range(max_iter):
        var live = False
        for c in range(problems):
            if hact[c] != 0:
                live = True
                epochs[c] = epoch + 1
                if do_shuffle:
                    shuffle(IP(unsafe_from_address=Int(hidx.unsafe_ptr())) + c * n, n, rngs[c])
                hpf[3 * c] = etas[c]
        if not live:
            break
        ctx.enqueue_copy(dst_buf=dact, src_ptr=hact.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=dpf, src_ptr=hpf.unsafe_ptr())
        # the epoch as ONE guarded unit: it updates w, q, the scalar state
        # and t in place, so a cut epoch restores them and replays
        ctx.enqueue_copy(dst_buf=dws, src_buf=dw)
        ctx.enqueue_copy(dst_buf=dqs, src_buf=dq)
        ctx.enqueue_copy(dst_buf=dpss, src_buf=dps)
        ctx.enqueue_copy(dst_buf=dpts, src_buf=dpt)
        var tries = 0
        while True:
            var nonce = wit.begin()
            var wo = 0
            ctx.enqueue_copy(dst_buf=didx, src_ptr=hidx.unsafe_ptr())
            if pa_rate:
                ctx.enqueue_function[sgd_rowsq_kernel](
                    dx.unsafe_ptr(), Int32(n), Int32(d), dsq.unsafe_ptr(), wit.p(), Int32(wo), nonce,
                    grid_dim=_xg_blocks(n), block_dim=XG_TPB,
                )
                wo += _xg_blocks(n)
            var start = 0
            while start < n:
                var cnt = min(chunk, n - start)
                ctx.enqueue_function[sgd_ps_kernel](
                    dx.unsafe_ptr(), dlab.unsafe_ptr(), dsw.unsafe_ptr(), didx.unsafe_ptr(), dw.unsafe_ptr(),
                    dq.unsafe_ptr(), dsq.unsafe_ptr(), dpart.unsafe_ptr(), dps.unsafe_ptr(), dpt.unsafe_ptr(),
                    dact.unsafe_ptr(), dpf.unsafe_ptr(), dci.unsafe_ptr(), dcf.unsafe_ptr(), Int32(start), Int32(cnt),
                    wit.p(), Int32(wo), nonce, grid_dim=problems, block_dim=XG_TPB,
                )
                wo += problems
                start += cnt
            ctx.enqueue_copy(dst_ptr=hw.unsafe_ptr(), src_buf=dw)
            ctx.enqueue_copy(dst_ptr=hps.unsafe_ptr(), src_buf=dps)
            if wit.ok(ctx, wo, "SGD per-sample epoch"):
                break
            tries += 1
            if tries >= WITNESS_TRIES:
                wit.fail()
            ctx.enqueue_copy(dst_buf=dw, src_buf=dws)
            ctx.enqueue_copy(dst_buf=dq, src_buf=dqs)
            ctx.enqueue_copy(dst_buf=dps, src_buf=dpss)
            ctx.enqueue_copy(dst_buf=dpt, src_buf=dpts)
        ctx.synchronize()
        for c in range(problems):
            if hact[c] == 0:
                continue
            # sgd_one's floating-point under-/overflow check
            var intercept = hps[SGD_PS_ST * c]
            var finite = intercept == intercept and fabs(intercept) < Float32(3.0e38)
            for j in range(d):
                var wj = ws_mul(hw[c * d + j], hps[SGD_PS_ST * c + 4], hps[SGD_PS_ST * c + 5])
                if not (wj == wj and fabs(wj) < Float32(3.0e38)):
                    finite = False
            if not finite:
                failed[c] = True
                hact[c] = 0
                continue
            var mean_obj = fd(hps[SGD_PS_ST * c + 2], i2f(n))
            if need_obj and mean_obj > fs(bests[c], tol):
                no_imp[c] += 1
            else:
                no_imp[c] = 0
            if mean_obj < bests[c]:
                bests[c] = mean_obj
            if no_imp[c] >= nic:
                if lr == LR_ADAPTIVE and etas[c] > Float32(1e-6):
                    etas[c] = fd(etas[c], Float32(5))
                    no_imp[c] = 0
                else:
                    hact[c] = 0
    var max_epochs = 0
    var status = 0
    for c in range(problems):
        if failed[c]:
            status = -1
            for j in range(d):
                res.unsafe_store(c * d + j, Float32(0))
            res.unsafe_store(problems * d + c, Float32(0))
        else:
            # coef = wscale * v
            for j in range(d):
                res.unsafe_store(c * d + j, ws_mul(hw[c * d + j], hps[SGD_PS_ST * c + 4], hps[SGD_PS_ST * c + 5]))
            # one-class: the slot holds offset_ = 1 - intercept (`oc_offset`)
            res.unsafe_store(problems * d + c, oc_offset(hps[SGD_PS_ST * c], hps[SGD_PS_ST * c + 3])
                             if one_class else hps[SGD_PS_ST * c])
            if epochs[c] > max_epochs:
                max_epochs = epochs[c]
    res.unsafe_store(problems * d + problems, i2f(max_epochs))
    res.unsafe_store(problems * d + problems + 1, i2f(status))
    _ = hci^
    _ = hcf^
    _ = hidx^
    _ = hw^
    _ = hps^
    _ = hpt^
    _ = hact^
    _ = hpf^
    _ = dx^
    _ = dlab^
    _ = dsw^
    _ = didx^
    _ = dw^
    _ = dq^
    _ = dsq^
    _ = dpart^
    _ = dps^
    _ = dpt^
    _ = dact^
    _ = dpf^
    _ = dci^
    _ = dcf^
    _ = dws^
    _ = dqs^
    _ = dpss^
    _ = dpts^
    _ = wit^


# ------------------------------------------------ k-fold RidgeCV on the grid (lane/neural-pass91)
# x_linear/ridgecv.mojo's chains, one thread each: the training means (d + 1
# threads), the centered Gram cells and X'y (one thread a cell), the solves
# (one thread an alpha, its own scratch), the held-out predictions (one
# thread a row and alpha) and the fold scores (one thread an alpha, summed
# folds ascending). The same helpers as the host fit, so the same words.
def kf_means_kernel(x: FP, y: FP, n: Int32, d: Int32, s: Int32, e: Int32, fi: Int32, xm: FP):
    var j = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    var dd = Int(d)
    if j < dd:
        st(xm, j, kf_mean(x, dd, j, Int(n), Int(s), Int(e)) if fi != 0 else Float32(0))
    elif j == dd:
        st(xm, dd, kf_mean(y, 1, 0, Int(n), Int(s), Int(e)) if fi != 0 else Float32(0))


def kf_cells_kernel(x: FP, y: FP, n: Int32, d: Int32, s: Int32, e: Int32, xm: FP, g: FP, xty: FP):
    var c = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    var dd = Int(d)
    var cells = dd * (dd + 1) // 2
    if c < cells:
        var jk = upper_cell(c, dd)
        var j = jk[0]
        var k = jk[1]
        var v = kf_cross(x, dd, j, ld(xm, j), x, dd, k, ld(xm, k), Int(n), Int(s), Int(e))
        st(g, j * dd + k, v)
        st(g, k * dd + j, v)
    elif c < cells + dd:
        var j = c - cells
        st(xty, j, kf_cross(x, dd, j, ld(xm, j), y, 1, 0, ld(xm, dd), Int(n), Int(s), Int(e)))


def kf_solve_kernel(g: FP, xty: FP, xm: FP, d: Int32, alphas: FP, na: Int32, fi: Int32, aw: FP, w: FP, b: FP):
    var a = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    var dd = Int(d)
    if a < Int(na):
        st(b, a, kf_solve(g, xty, xm, ld(xm, dd), dd, ld(alphas, a), fi != 0, aw + a * dd * dd, w + a * dd))


def kf_pred_kernel(x: FP, d: Int32, s: Int32, e: Int32, na: Int32, w: FP, b: FP, p: FP, stride: Int32):
    var q = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    var nt = Int(e) - Int(s)
    if q < nt * Int(na):
        var a = q // nt
        var r = q - a * nt
        st(p, a * Int(stride) + r, kf_pred(x, Int(s) + r, Int(d), w + a * Int(d), ld(b, a)))


def kf_score_kernel(y: FP, p: FP, s: Int32, e: Int32, na: Int32, stride: Int32, sums: FP):
    var a = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    if a < Int(na):
        st(sums, a, fa(ld(sums, a), kf_score(y, p + a * Int(stride), Int(s), Int(e))))


def _ridge_kfold_grid(x: FP, n_x: Int, y: FP, n_y: Int, n: Int, d: Int, ip: List[Int32], fp: List[Float32],
                      res: FP) raises:
    var ctx = linear_ctx()
    var k = Int(ip[0])
    var fi = Int(ip[1])
    var na = Int(ip[2])
    var stride = n // k + 1
    var dx = ctx.enqueue_create_buffer[DType.float32](max(n_x, 1))
    var dy = ctx.enqueue_create_buffer[DType.float32](max(n_y, 1))
    var dal = ctx.enqueue_create_buffer[DType.float32](max(na, 1))
    var dxm = ctx.enqueue_create_buffer[DType.float32](d + 1)
    var dg = ctx.enqueue_create_buffer[DType.float32](max(d * d, 1))
    var dxty = ctx.enqueue_create_buffer[DType.float32](max(d, 1))
    var daw = ctx.enqueue_create_buffer[DType.float32](max(na * d * d, 1))
    var dw = ctx.enqueue_create_buffer[DType.float32](max(na * d, 1))
    var db = ctx.enqueue_create_buffer[DType.float32](max(na, 1))
    var dp = ctx.enqueue_create_buffer[DType.float32](max(na * stride, 1))
    var dsum = ctx.enqueue_create_buffer[DType.float32](max(na, 1))
    var hfp = fp.copy()
    if n_x > 0:
        ctx.enqueue_copy(dst_buf=dx, src_ptr=x)
    if n_y > 0:
        ctx.enqueue_copy(dst_buf=dy, src_ptr=y)
    ctx.enqueue_copy(dst_buf=dal, src_ptr=hfp.unsafe_ptr())
    dsum.enqueue_fill(Float32(0))
    var cells = d * (d + 1) // 2
    for f in range(k):
        var s = Int32(kf_start(n, k, f))
        var e = Int32(kf_end(n, k, f))
        ctx.enqueue_function[kf_means_kernel](dx.unsafe_ptr(), dy.unsafe_ptr(), Int32(n), Int32(d), s, e, Int32(fi),
                                              dxm.unsafe_ptr(), grid_dim=_xg_blocks(d + 1), block_dim=XG_TPB)
        ctx.enqueue_function[kf_cells_kernel](dx.unsafe_ptr(), dy.unsafe_ptr(), Int32(n), Int32(d), s, e, dxm.unsafe_ptr(),
                                              dg.unsafe_ptr(), dxty.unsafe_ptr(), grid_dim=_xg_blocks(cells + d), block_dim=XG_TPB)
        ctx.enqueue_function[kf_solve_kernel](dg.unsafe_ptr(), dxty.unsafe_ptr(), dxm.unsafe_ptr(), Int32(d), dal.unsafe_ptr(),
                                              Int32(na), Int32(fi), daw.unsafe_ptr(), dw.unsafe_ptr(), db.unsafe_ptr(),
                                              grid_dim=_xg_blocks(na), block_dim=XG_TPB)
        var nt = Int(e) - Int(s)
        ctx.enqueue_function[kf_pred_kernel](dx.unsafe_ptr(), Int32(d), s, e, Int32(na), dw.unsafe_ptr(), db.unsafe_ptr(),
                                             dp.unsafe_ptr(), Int32(stride), grid_dim=_xg_blocks(nt * na), block_dim=XG_TPB)
        ctx.enqueue_function[kf_score_kernel](dy.unsafe_ptr(), dp.unsafe_ptr(), s, e, Int32(na), Int32(stride),
                                              dsum.unsafe_ptr(), grid_dim=_xg_blocks(na), block_dim=XG_TPB)
    var hs = List[Float32](length=max(na, 1), fill=Float32(0))
    ctx.enqueue_copy(dst_ptr=hs.unsafe_ptr(), src_buf=dsum)
    ctx.synchronize()
    for a in range(na):
        res.unsafe_store(a, fd(hs[a], i2f(k)))
    _ = hfp^
    _ = hs^
    _ = dx^
    _ = dy^
    _ = dal^
    _ = dxm^
    _ = dg^
    _ = dxty^
    _ = daw^
    _ = dw^
    _ = db^
    _ = dp^
    _ = dsum^


def fit_device(
    algo: Int, x: FP, n_x: Int, y: FP, n_y: Int, n: Int, d: Int,
    ip: List[Int32], fp: List[Float32], n_out: Int, n_fw: Int, n_iw: Int, res: FP,
) raises:
    if algo == ALGO_RIDGE_KFOLD:
        _ridge_kfold_grid(x, n_x, y, n_y, n, d, ip, fp, res)
        return
    if algo == ALGO_SGD and len(ip) > 12 and sgd_mb_on(Int(ip[12]), Int(ip[0]), Int(ip[3])):
        _sgd_mb_grid(x, n_x, y, n_y, n, d, ip, fp, n_out, res)
        return
    if algo == ALGO_SGD:
        if _sgd_on_host():
            _fit_on_host(algo, x, y, n, d, ip, fp, n_out, n_fw, n_iw, res)
        else:
            _sgd_ps_grid(x, n_x, y, n_y, n, d, ip, fp, n_out, res)
        return
    var ctx = linear_ctx()
    comptime if X_LINEAR_BLOCKS:
        from x_linear.blocks import blocks_handles, blocks_fit

        if blocks_handles(algo, n):
            blocks_fit(ctx, algo, x, n_x, y, n_y, n, d, ip, fp, n_out, res)
            return
    comptime if X_LINEAR_BLOCKS_GRAM:
        from x_linear.blocks import XB_MIN_ROWS
        from x_linear.blocks_gram import gram_handles, gram_fit

        if n >= XB_MIN_ROWS and gram_handles(algo):
            gram_fit(ctx, algo, x, n_x, y, n_y, n, d, ip, fp, n_out, res)
            return
    var dx = ctx.enqueue_create_buffer[DType.float32](max(n_x, 1))
    var dy = ctx.enqueue_create_buffer[DType.float32](max(n_y, 1))
    var dfp = ctx.enqueue_create_buffer[DType.float32](max(len(fp), 1))
    var dout = ctx.enqueue_create_buffer[DType.float32](max(n_out, 1))
    var dfw = ctx.enqueue_create_buffer[DType.float32](max(n_fw, 1))
    var diw = ctx.enqueue_create_buffer[DType.int32](max(n_iw, 1))
    var hip = ip.copy()
    # LARS reads ip[4] on the device: 1 when the Gram is already in fw
    # (`xg_gram_kernel` below), 0 when the team computes it.
    var grid_gram = algo == ALGO_LARS and _lars_grid_gram() and d > 0
    # lane/neural-pass87 (2026-10-01): BayesianRidge (unweighted) and ARD read
    # the same layout (xm at 0, G at d, ip[1] fit_intercept) and the same
    # centered Gram chains, which the team ran on ONE block (24,310 chains
    # of every row at 220 features over 256 threads).
    var bayes_like = (algo == ALGO_BAYES and len(ip) > 2 and ip[2] == 0) or algo == ALGO_ARD
    if bayes_like and _bayes_grid_gram() and d > 0:
        grid_gram = True
    if algo == ALGO_LARS or bayes_like:
        while len(hip) < 5:
            hip.append(Int32(0))
        hip[4] = Int32(1 if grid_gram else 0)
    var dip = ctx.enqueue_create_buffer[DType.int32](max(len(hip), 1))
    var dtw = ctx.enqueue_create_buffer[DType.float32](
        team_work(n, team_rows(algo, IP(unsafe_from_address=Int(hip.unsafe_ptr()))), team_own(algo, d)))
    var hfp = fp.copy()
    if n_x > 0:
        ctx.enqueue_copy(dst_buf=dx, src_ptr=x)
    if n_y > 0:
        ctx.enqueue_copy(dst_buf=dy, src_ptr=y)
    if len(hip) > 0:
        ctx.enqueue_copy(dst_buf=dip, src_ptr=hip.unsafe_ptr())
    if len(hfp) > 0:
        ctx.enqueue_copy(dst_buf=dfp, src_ptr=hfp.unsafe_ptr())
    dout.enqueue_fill(Float32(0))
    dfw.enqueue_fill(Float32(0))
    diw.enqueue_fill(Int32(0))
    dtw.enqueue_fill(Float32(0))
    if grid_gram:
        # The means then the centered Gram into fw[0, d + d*d), the layout
        # `lars_fit` reads (xm at 0, G at d); the team recomputes the means
        # itself (the same statements, the same values) and skips the Gram.
        ctx.enqueue_function[xg_means_kernel](
            dx.unsafe_ptr(), Int32(n), Int32(d), Int32(hip[1]), dfw.unsafe_ptr(),
            grid_dim=_xg_blocks(d), block_dim=XG_TPB,
        )
        ctx.enqueue_function[xg_gram_kernel](
            dx.unsafe_ptr(), Int32(n), Int32(d), dfw.unsafe_ptr(),
            grid_dim=_xg_blocks(d * (d + 1) // 2), block_dim=XG_TPB,
        )
    ctx.enqueue_function[fit_kernel](
        Int32(algo), dx.unsafe_ptr(), dy.unsafe_ptr(), Int32(n), Int32(d),
        dip.unsafe_ptr(), dfp.unsafe_ptr(), dout.unsafe_ptr(), dfw.unsafe_ptr(), diw.unsafe_ptr(),
        dtw.unsafe_ptr(),
        grid_dim=1, block_dim=LINEAR_TPB if team_fit(algo) else 1,
    )
    if n_out > 0:
        ctx.enqueue_copy(dst_ptr=res, src_buf=dout)
    ctx.synchronize()
    _ = hip^
    _ = hfp^
    _ = dx^
    _ = dy^
    _ = dip^
    _ = dfp^
    _ = dout^
    _ = dfw^
    _ = diw^
    _ = dtw^
    _ = ctx^


def decision_device(x: FP, wb: FP, n: Int, d: Int, k: Int, link: Int, res: FP) raises:
    var ctx = linear_ctx()
    var dx = ctx.enqueue_create_buffer[DType.float32](max(n * d, 1))
    var dwb = ctx.enqueue_create_buffer[DType.float32](k * (d + 1))
    var dout = ctx.enqueue_create_buffer[DType.float32](max(n * k, 1))
    if n * d > 0:
        ctx.enqueue_copy(dst_buf=dx, src_ptr=x)
    ctx.enqueue_copy(dst_buf=dwb, src_ptr=wb)
    var total = n * k
    if total > 0:
        ctx.enqueue_function[decision_kernel](
            dx.unsafe_ptr(), dwb.unsafe_ptr(), Int32(n), Int32(d), Int32(k), Int32(link), dout.unsafe_ptr(),
            grid_dim=(total + 127) // 128, block_dim=128,
        )
        ctx.enqueue_copy(dst_ptr=res, src_buf=dout)
    ctx.synchronize()
    _ = dx^
    _ = dwb^
    _ = dout^
    _ = ctx^
