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
from max.gpu.host import DeviceContext, DeviceBuffer
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, NUMERIC_FAST
from x_linear.ops import FP, IP
from x_linear.dispatch import fit_dispatch, decision_one, team_fit, team_rows, team_own, ALGO_SGD, ALGO_LARS, ALGO_GLM, ALGO_RIDGE_KFOLD, ALGO_RIDGE, ALGO_BAYES, ALGO_ARD
from x_linear.ops import ld, st, ldi, fd, i2f, fa, fs, fm, fabs, shuffle, fmad, flog, fill, copy, row_dot, mean_of
from x_linear.witness import Witness, witness_end, WITNESS_TRIES
from x_linear.sgd import sgd_mb_on, mb_sub_size, mb_dblk, mb_row, mb_row_dot, mb_rowsq, mb_block_dot, MB_DBLK, LR_PA1, LR_PA2, mb_part, mb_step, mb_bias_step, mb_subs, mb_eta, mb_optimal_init, mb_penalty, LR_OPTIMAL, LR_ADAPTIVE, P_L2, P_L1
from x_linear.ridgecv import kf_start, kf_end, kf_mean, kf_cross, kf_solve, kf_pred, kf_score, kf_ff_solve
from x_linear.ridge import ridge_ff_unit, ridge_ff_units, ridge_ff_solve
from x_linear.tops import t_fold_fa_staged, t_fold_fa_blocked, fold_parts, fold_blocks, FOLD_BLOCK, X_LINEAR_SERIAL_FOLDS
from x_linear.glm import (
    _unit, _glm_deriv_row, _glm_cell, _glm_cell_rows, _glm_slot_count, _glm_slot_cell, _glm_step, GLM_LINK_LOG, GLM_STALL_ITERS,
    _glm_cell_part, _glm_cell_store,
)
from x_linear.dispatch import fit_dispatch, decision_one, team_fit, team_rows, team_own, ALGO_SGD, ALGO_LARS, ALGO_ISOTONIC, ALGO_ISOTONIC_PREDICT
from x_linear.isotonic import iso_predict_one, iso_gather_one, iso_bounds, iso_bounds_from, iso_group, iso_after_unique
from std.memory import bitcast
from x_linear.dispatch import fit_dispatch, decision_one, team_fit, team_rows, team_own, ALGO_SGD, ALGO_LARS, ALGO_ENETCV
from x_linear.cd_grid import enetcv_fit_grid
from x_linear.moments_grid import MOMENTS_GRID, MG_NT, mg_means_kernel, mg_cross_kernel, mg_tiles
from x_linear.dispatch import ALGO_RIDGE
from x_linear.tops import upper_cell, fold_fa, chain_cfmad
from std.os import getenv
from x_linear.logcv_grid import logcv_fit_grid
from x_linear.huber_grid import huber_fit_grid
from x_linear.dispatch import ALGO_HUBER, ALGO_ENETCV
from x_linear.enetcv_fast import enetcv_fast
from x_linear.tops import X_LINEAR_SERIAL_FOLDS
from x_linear.dispatch import ALGO_LOGCV
from x_linear.team import LINEAR_TPB, team_work, device_team, solo, team_barrier
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.kernel_matrix import TARGET_COLUMN, lib_smem_page_fits_for


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
    """`MOJOLEARN_X_LINEAR_SGD_HOST=0` keeps SGD on its one device thread
    (the A/B arm); default the host."""
    return String(getenv("MOJOLEARN_X_LINEAR_SGD_HOST")) != "0"


def _lars_grid_gram() -> Bool:
    """`MOJOLEARN_X_LINEAR_LARS_GRID_GRAM=0` keeps the Gram on the team
    (the A/B arm); default the grid kernel."""
    return String(getenv("MOJOLEARN_X_LINEAR_LARS_GRID_GRAM")) != "0"


def _bayes_grid_gram() -> Bool:
    """`MOJOLEARN_X_LINEAR_BAYES_GRID_GRAM=0` keeps BayesianRidge's and
    ARD's Gram on the team (the A/B arm); default the grid kernel."""
    return String(getenv("MOJOLEARN_X_LINEAR_BAYES_GRID_GRAM")) != "0"


def _enetcv_grid() -> Bool:
    """`MOJOLEARN_X_LINEAR_ENETCV_GRID=0` keeps LassoCV / ElasticNetCV on
    the one-block team fit (the A/B arm); default the grid form
    (x_linear/cd_grid.mojo)."""
    return String(getenv("MOJOLEARN_X_LINEAR_ENETCV_GRID")) != "0"


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
            res.unsafe_store(problems * d + c, hb[0])
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



# ------------------------------------------------ GLM on the grid (lane/neural-pass89)
# The GLM Newton loop driven from the host, each pass a kernel over the whole
# grid: the rows' derivatives and loss terms (one thread a row) and the
# gradient / Hessian cells (one thread a cell chain over the rows ascending,
# in glm_fit's warp-uniform slots). The folds that define the objective run on
# one block (`t_fold_fa_staged`), the m x m step on one thread
# (`_glm_step`), and the control (the line search, the stall count) on the
# host with the same fa / fm words. Every value is glm_fit's: the same
# helpers, the same order. glm_fit ran all of it on ONE block: istella's
# 24,531 cells were ~96 million-row chains a thread (the L40S board timed
# out). `MOJOLEARN_X_LINEAR_GLM_GRID=0` restores the one-block fit.
def _glm_grid() -> Bool:
    return String(getenv("MOJOLEARN_X_LINEAR_GLM_GRID")) != "0"


def glm_init_kernel(y: FP, n: Int32, d: Int32, fi: Int32, link: Int32, sw: Int32, res: FP, sc: FP, wf: IP, woff: Int32, nonce: Int32):
    """glm_fit's prologue on one thread: den (sc[0]) and the intercept start."""
    var nn = Int(n)
    var dd = Int(d)
    var den = i2f(nn)
    if sw != 0:
        den = Float32(0)
        for i in range(nn):
            den = fa(den, ld(y, nn + i))
    st(sc, 0, den)
    fill(res, 0, dd + 3, Float32(0))
    if fi != 0:
        var ym = mean_of(y, nn)
        if sw != 0:
            var acc = Float32(0)
            for i in range(nn):
                acc = fmad(ld(y, nn + i), ld(y, i), acc)
            ym = fd(acc, den)
        st(res, dd, flog(ym) if Int(link) == GLM_LINK_LOG else ym)
    witness_end(wf, woff, nonce)

def glm_obj_map_kernel(x: FP, y: FP, n: Int32, d: Int32, fi: Int32, power: Float32, link: Int32, sw: Int32,
                       theta: FP, eta: FP, lt: FP, wf: IP, woff: Int32, nonce: Int32):
    """`_objective_team`'s row statements, one thread a row."""
    var i = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    var nn = Int(n)
    if i < nn:
        var dd = Int(d)
        var b = ld(theta, dd) if fi != 0 else Float32(0)
        var e = fa(row_dot(x, i, dd, theta, 0), b)
        st(eta, i, e)
        var l = _unit(power, Int(link), ld(y, i), e, 0)
        if sw != 0:
            l = fm(ld(y, nn + i), l)
        st(lt, i, l)
    witness_end(wf, woff, nonce)

def glm_obj_fold_kernel(tw: FP, lt: FP, n: Int32, d: Int32, theta: FP, alpha: Float32, sc: FP, slot: Int32, wf: IP, woff: Int32, nonce: Int32):
    """`_objective_team`'s fold and value on one block: sc[slot] = f."""
    var t = device_team(tw, Int(n), 3, 0)
    var acc = t_fold_fa_blocked(t, lt, Int(n), t.row(1))
    if t.lead():
        var reg = Float32(0)
        for j in range(Int(d)):
            var w = ld(theta, j)
            reg = fmad(w, w, reg)
        st(sc, Int(slot), fa(fd(acc, ld(sc, 0)), fm(fm(Float32(0.5), alpha), reg)))
    witness_end(wf, woff, nonce)

def glm_deriv_kernel(y: FP, n: Int32, power: Float32, link: Int32, sw: Int32, eta: FP, gr: FP, hr: FP, wf: IP, woff: Int32, nonce: Int32):
    var i = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    if i < Int(n):
        _glm_deriv_row(y, Int(n), i, power, Int(link), eta, sw != 0, gr, hr)
    witness_end(wf, woff, nonce)

def glm_cells_kernel(x: FP, gr: FP, hr: FP, lo: Int32, cnt: Int32, d: Int32, m: Int32, g: FP, h: FP):
    """One thread a slot of glm_fit's warp-uniform layout, rows [lo, lo + cnt)."""
    var sl = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    var dd = Int(d)
    var mm = Int(m)
    if sl < _glm_slot_count(dd, mm):
        var c = _glm_slot_cell(sl, dd, mm)
        if c >= 0:
            _glm_cell_rows(c, x, gr, hr, Int(lo), Int(cnt), dd, mm, g, h)


#: Apple: macOS silently aborts a command buffer that holds the GPU for
#: seconds and leaves its output partly stale (the M2's GLM grid digests
#: differed run to run on istella, 24,531 cells of 1M-row chains in ONE
#: launch, and on taxi under a second Metal job). The cells run in row
#: slices of at most GLM_APPLE_SLICE_MACS chain steps a launch, each waited
#: on, every chain resuming from its stored value (the same words).
comptime GLM_APPLE_SLICE_MACS = 1 << 29


def _glm_rows_slice(n: Int, slots: Int) -> Int:
    comptime if has_apple_gpu_accelerator():
        return max(64, min(n, (GLM_APPLE_SLICE_MACS // max(slots, 1)) // 64 * 64))
    return n


def _glm_blocks_slice(nb: Int, slots: Int) -> Int:
    """The row blocks a parts launch takes: all of them off Apple."""
    comptime if has_apple_gpu_accelerator():
        return max(1, min(nb, GLM_APPLE_SLICE_MACS // (max(slots, 1) * FOLD_BLOCK)))
    return nb


def glm_cell_parts_kernel(x: FP, gr: FP, hr: FP, n: Int32, d: Int32, m: Int32, nb: Int32, b0: Int32, bcnt: Int32,
                          parts: FP, wf: IP, woff: Int32, nonce: Int32):
    """lane/neural-pass97: one thread a (slot, row block): the slot's cell
    chain over the block from zero into parts[slot * nb + block]; the row
    blocks [b0, b0 + bcnt) only (the Apple slices, `_glm_blocks_slice`)."""
    var q = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    var dd = Int(d)
    var mm = Int(m)
    var nbb = Int(nb)
    # block-major: neighbouring threads are neighbouring slots over the same
    # rows, so their x words share cache lines (and a warp keeps one kind)
    var slots = _glm_slot_count(dd, mm)
    var bq = q // slots
    var sl = q - bq * slots
    var b = Int(b0) + bq
    if bq < Int(bcnt) and b < nbb:
        var c = _glm_slot_cell(sl, dd, mm)
        if c >= 0:
            var lo = b * FOLD_BLOCK
            st(parts, sl * nbb + b, _glm_cell_part(c, x, gr, hr, dd, mm, lo, min(FOLD_BLOCK, Int(n) - lo)))
    witness_end(wf, woff, nonce)

def glm_cell_combine_kernel(parts: FP, d: Int32, m: Int32, nb: Int32, g: FP, h: FP, wf: IP, woff: Int32, nonce: Int32):
    """One thread a slot: its partials folded blocks ascending."""
    var sl = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    var dd = Int(d)
    var mm = Int(m)
    if sl < _glm_slot_count(dd, mm):
        var c = _glm_slot_cell(sl, dd, mm)
        if c >= 0:
            _glm_cell_store(c, fold_parts(parts, sl * Int(nb), Int(nb)), g, h, mm)
    witness_end(wf, woff, nonce)

def glm_step_kernel(g: FP, h: FP, step: FP, res: FP, m: Int32, d: Int32, alpha: Float32, tol: Float32, sc: FP, wf: IP, woff: Int32, nonce: Int32):
    """The dense step on one thread: sc[2] flag, sc[3] slope."""
    var fs_ = _glm_step(g, h, step, res, Int(m), Int(d), alpha, ld(sc, 0), tol, 0, Float32(0))
    st(sc, 2, i2f(fs_[0]))
    st(sc, 3, fs_[1])
    witness_end(wf, woff, nonce)

def glm_trial_kernel(step: FP, res: FP, trial: FP, m: Int32, tt: Float32):
    for j in range(Int(m)):
        st(trial, j, fmad(tt, ld(step, j), ld(res, j)))


def glm_accept_kernel(res: FP, trial: FP, m: Int32):
    copy(res, 0, trial, 0, Int(m))


def _glm_grid_objective(
    mut ctx: DeviceContext, theta: FP, x: FP, y: FP, eta: FP, lt: FP, tw: FP, sc: FP,
    sc_buf: DeviceBuffer[DType.float32], hsc: FP,
    n: Int, d: Int, fi: Int, power: Float32, link: Int, sw: Int, alpha: Float32, rows_grid: Int,
    mut wit: Witness,
) raises -> Float32:
    """`_objective_team` at theta on the grid (map) and one block (fold); eta
    holds its linear predictor afterwards."""
    var tries = 0
    while True:
        var nonce = wit.begin()
        ctx.enqueue_function[glm_obj_map_kernel](x, y, Int32(n), Int32(d), Int32(fi), power, Int32(link), Int32(sw),
                                                 theta, eta, lt, wit.p(), Int32(0), nonce,
                                                 grid_dim=rows_grid, block_dim=XG_TPB)
        ctx.enqueue_function[glm_obj_fold_kernel](tw, lt, Int32(n), Int32(d), theta, alpha, sc, Int32(1),
                                                  wit.p(), Int32(rows_grid), nonce, grid_dim=1, block_dim=LINEAR_TPB)
        if wit.ok(ctx, rows_grid + 1, "GLM objective"):
            break
        tries += 1
        if tries >= WITNESS_TRIES:
            wit.fail()
    ctx.synchronize()
    var v = Float32(0)
    ctx.enqueue_copy(dst_ptr=hsc, src_buf=sc_buf)
    ctx.synchronize()
    v = hsc.unsafe_load(1)
    return v


def _glm_fit_grid(x: FP, n_x: Int, y: FP, n_y: Int, n: Int, d: Int, ip: List[Int32], fp: List[Float32],
                  n_out: Int, res: FP) raises:
    var ctx = linear_ctx()
    var max_iter = Int(ip[0])
    var fi = Int(ip[1])
    var link = Int(ip[2])
    var sw = Int(ip[3])
    var power = fp[0]
    var alpha = fp[1]
    var tol = fp[2]
    var m = d + 1 if fi != 0 else d
    var dx = ctx.enqueue_create_buffer[DType.float32](max(n_x, 1))
    var dy = ctx.enqueue_create_buffer[DType.float32](max(n_y, 1))
    var dres = ctx.enqueue_create_buffer[DType.float32](max(n_out, d + 3))
    var deta = ctx.enqueue_create_buffer[DType.float32](max(n, 1))
    var dlt = ctx.enqueue_create_buffer[DType.float32](max(n, 1))
    var dgr = ctx.enqueue_create_buffer[DType.float32](max(n, 1))
    var dhr = ctx.enqueue_create_buffer[DType.float32](max(n, 1))
    var dg = ctx.enqueue_create_buffer[DType.float32](m)
    var dh = ctx.enqueue_create_buffer[DType.float32](m * m)
    var dstep = ctx.enqueue_create_buffer[DType.float32](m)
    var dtrial = ctx.enqueue_create_buffer[DType.float32](m)
    var dsc = ctx.enqueue_create_buffer[DType.float32](8)
    var dtw = ctx.enqueue_create_buffer[DType.float32](team_work(n, 3, 0))
    var hsc_l = List[Float32](length=8, fill=Float32(0))
    var hscp = FP(unsafe_from_address=Int(hsc_l.unsafe_ptr()))
    if n_x > 0:
        ctx.enqueue_copy(dst_buf=dx, src_ptr=x)
    if n_y > 0:
        ctx.enqueue_copy(dst_buf=dy, src_ptr=y)
    dsc.enqueue_fill(Float32(0))
    dh.enqueue_fill(Float32(0))
    dtw.enqueue_fill(Float32(0))
    var rows_grid = _xg_blocks(n)
    var slot_ub = m + m * (m + 1) // 2 + 4 * 64
    var slot_grid = _xg_blocks(slot_ub)
    var nb = fold_blocks(n)
    var dparts = ctx.enqueue_create_buffer[DType.float32](max(slot_ub * nb, 1))
    var rows_slice = _glm_rows_slice(n, _glm_slot_count(d, m))
    var blocks_slice = _glm_blocks_slice(nb, _glm_slot_count(d, m))
    var wit = Witness(ctx, max(rows_grid + 1, max(_xg_blocks(slot_ub * blocks_slice), slot_grid + 1)))
    var tries0 = 0
    while True:
        var nonce = wit.begin()
        ctx.enqueue_function[glm_init_kernel](dy.unsafe_ptr(), Int32(n), Int32(d), Int32(fi), Int32(link), Int32(sw),
                                              dres.unsafe_ptr(), dsc.unsafe_ptr(), wit.p(), Int32(0), nonce,
                                              grid_dim=1, block_dim=1)
        if wit.ok(ctx, 1, "GLM init"):
            break
        tries0 += 1
        if tries0 >= WITNESS_TRIES:
            wit.fail()

    var iters = 0
    var converged = False
    var f = _glm_grid_objective(ctx, FP(unsafe_from_address=Int(dres.unsafe_ptr())), FP(unsafe_from_address=Int(dx.unsafe_ptr())), FP(unsafe_from_address=Int(dy.unsafe_ptr())), FP(unsafe_from_address=Int(deta.unsafe_ptr())), FP(unsafe_from_address=Int(dlt.unsafe_ptr())), FP(unsafe_from_address=Int(dtw.unsafe_ptr())), FP(unsafe_from_address=Int(dsc.unsafe_ptr())), dsc, hscp, n, d, fi, power, link, sw, alpha, rows_grid, wit)
    var stall = 0
    for it in range(max_iter):
        var tries1 = 0
        while True:
            var nonce = wit.begin()
            ctx.enqueue_function[glm_deriv_kernel](dy.unsafe_ptr(), Int32(n), power, Int32(link), Int32(sw), deta.unsafe_ptr(),
                                                   dgr.unsafe_ptr(), dhr.unsafe_ptr(), wit.p(), Int32(0), nonce,
                                                   grid_dim=rows_grid, block_dim=XG_TPB)
            if wit.ok(ctx, rows_grid, "GLM derivatives"):
                break
            tries1 += 1
            if tries1 >= WITNESS_TRIES:
                wit.fail()
        comptime if X_LINEAR_SERIAL_FOLDS:
            var lo = 0
            while lo < n:
                var cnt = min(rows_slice, n - lo)
                ctx.enqueue_function[glm_cells_kernel](dx.unsafe_ptr(), dgr.unsafe_ptr(), dhr.unsafe_ptr(), Int32(lo), Int32(cnt),
                                                       Int32(d), Int32(m), dg.unsafe_ptr(), dh.unsafe_ptr(),
                                                       grid_dim=slot_grid, block_dim=XG_TPB)
                lo += cnt
                if lo < n and rows_slice < n:
                    ctx.synchronize()
        else:
            var b0 = 0
            while b0 < nb:
                var bc = min(blocks_slice, nb - b0)
                var tries2 = 0
                while True:
                    var nonce = wit.begin()
                    ctx.enqueue_function[glm_cell_parts_kernel](dx.unsafe_ptr(), dgr.unsafe_ptr(), dhr.unsafe_ptr(), Int32(n), Int32(d),
                                                                Int32(m), Int32(nb), Int32(b0), Int32(bc), dparts.unsafe_ptr(),
                                                                wit.p(), Int32(0), nonce,
                                                                grid_dim=_xg_blocks(slot_ub * bc), block_dim=XG_TPB)
                    if wit.ok(ctx, _xg_blocks(slot_ub * bc), "GLM cell parts"):
                        break
                    tries2 += 1
                    if tries2 >= WITNESS_TRIES:
                        wit.fail()
                b0 += bc
                if b0 < nb and blocks_slice < nb:
                    ctx.synchronize()
        # the combine rebuilds g and h from the parts, so the step (which
        # scales them in place) reruns with it
        var tries3 = 0
        while True:
            var nonce = wit.begin()
            comptime if not X_LINEAR_SERIAL_FOLDS:
                ctx.enqueue_function[glm_cell_combine_kernel](dparts.unsafe_ptr(), Int32(d), Int32(m), Int32(nb), dg.unsafe_ptr(),
                                                              dh.unsafe_ptr(), wit.p(), Int32(0), nonce,
                                                              grid_dim=slot_grid, block_dim=XG_TPB)
            ctx.enqueue_function[glm_step_kernel](dg.unsafe_ptr(), dh.unsafe_ptr(), dstep.unsafe_ptr(), dres.unsafe_ptr(),
                                                  Int32(m), Int32(d), alpha, tol, dsc.unsafe_ptr(), wit.p(),
                                                  Int32(slot_grid), nonce, grid_dim=1, block_dim=1)
            comptime if X_LINEAR_SERIAL_FOLDS:
                break  # the serial cells are in place: no rerun (opt-in form)
            if wit.ok(ctx, slot_grid + 1, "GLM step"):
                break
            tries3 += 1
            if tries3 >= WITNESS_TRIES:
                wit.fail()
        ctx.enqueue_copy(dst_ptr=hscp, src_buf=dsc)
        ctx.synchronize()
        var flag = Int(hscp.unsafe_load(2))
        var slope = hscp.unsafe_load(3)
        if flag == 1:
            converged = True
            break
        iters = it + 1
        if flag == 2:
            break
        var tt = Float32(1)
        var accepted = False
        for _ in range(40):
            ctx.enqueue_function[glm_trial_kernel](dstep.unsafe_ptr(), dres.unsafe_ptr(), dtrial.unsafe_ptr(), Int32(m), tt,
                                                   grid_dim=1, block_dim=1)
            var ft = _glm_grid_objective(ctx, FP(unsafe_from_address=Int(dtrial.unsafe_ptr())), FP(unsafe_from_address=Int(dx.unsafe_ptr())), FP(unsafe_from_address=Int(dy.unsafe_ptr())), FP(unsafe_from_address=Int(deta.unsafe_ptr())), FP(unsafe_from_address=Int(dlt.unsafe_ptr())), FP(unsafe_from_address=Int(dtw.unsafe_ptr())), FP(unsafe_from_address=Int(dsc.unsafe_ptr())), dsc, hscp, n, d, fi, power, link, sw, alpha, rows_grid, wit)
            if ft == ft and ft <= fa(f, fm(fm(Float32(1e-4), tt), slope)):
                ctx.enqueue_function[glm_accept_kernel](dres.unsafe_ptr(), dtrial.unsafe_ptr(), Int32(m),
                                                        grid_dim=1, block_dim=1)
                if ft == f:
                    stall += 1
                else:
                    stall = 0
                f = ft
                accepted = True
                break
            tt = fm(tt, Float32(0.5))
        if not accepted:
            f = _glm_grid_objective(ctx, FP(unsafe_from_address=Int(dres.unsafe_ptr())), FP(unsafe_from_address=Int(dx.unsafe_ptr())), FP(unsafe_from_address=Int(dy.unsafe_ptr())), FP(unsafe_from_address=Int(deta.unsafe_ptr())), FP(unsafe_from_address=Int(dlt.unsafe_ptr())), FP(unsafe_from_address=Int(dtw.unsafe_ptr())), FP(unsafe_from_address=Int(dsc.unsafe_ptr())), dsc, hscp, n, d, fi, power, link, sw, alpha, rows_grid, wit)
            break
        if stall >= GLM_STALL_ITERS:
            converged = True
            break
    ctx.enqueue_copy(dst_ptr=res, src_buf=dres)
    ctx.synchronize()
    if fi == 0:
        res.unsafe_store(d, Float32(0))
    res.unsafe_store(d + 1, i2f(iters))
    res.unsafe_store(d + 2, Float32(1) if converged else Float32(0))
    _ = dx^
    _ = dy^
    _ = dres^
    _ = deta^
    _ = dlt^
    _ = dgr^
    _ = dhr^
    _ = dg^
    _ = dh^
    _ = dstep^
    _ = dtrial^
    _ = dsc^
    _ = dtw^
    _ = dparts^
    _ = hsc_l^
    _ = wit^



# ------------------------------------------------ k-fold RidgeCV on the grid (lane/neural-pass91)
# x_linear/ridgecv.mojo's chains, one thread each: the training means (d + 1
# threads), the centered Gram cells and X'y (one thread a cell), the solves
# (one thread an alpha, its own scratch), the held-out predictions (one
# thread a row and alpha) and the fold scores (one thread an alpha, summed
# folds ascending). The same helpers as the host fit, so the same words.
@always_inline
def _kf_means_kernel_body(x: FP, y: FP, n: Int32, d: Int32, s: Int32, e: Int32, fi: Int32, xm: FP):
    var j = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    var dd = Int(d)
    if j < dd:
        st(xm, j, kf_mean(x, dd, j, Int(n), Int(s), Int(e)) if fi != 0 else Float32(0))
    elif j == dd:
        st(xm, dd, kf_mean(y, 1, 0, Int(n), Int(s), Int(e)) if fi != 0 else Float32(0))


def kf_means_kernel(x: FP, y: FP, n: Int32, d: Int32, s: Int32, e: Int32, fi: Int32, xm: FP, wf: IP, woff: Int32, nonce: Int32):
    _kf_means_kernel_body(x, y, n, d, s, e, fi, xm)
    witness_end(wf, woff, nonce)

@always_inline
def _kf_cells_kernel_body(x: FP, y: FP, n: Int32, d: Int32, s: Int32, e: Int32, xm: FP, g: FP, xty: FP):
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


def kf_cells_kernel(x: FP, y: FP, n: Int32, d: Int32, s: Int32, e: Int32, xm: FP, g: FP, xty: FP, wf: IP, woff: Int32, nonce: Int32):
    _kf_cells_kernel_body(x, y, n, d, s, e, xm, g, xty)
    witness_end(wf, woff, nonce)

@always_inline
def _kf_solve_kernel_body(g: FP, xty: FP, xm: FP, d: Int32, alphas: FP, na: Int32, fi: Int32, aw: FP, w: FP, b: FP,
                    trust: FP):
    var a = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    var dd = Int(d)
    if a < Int(na):
        var r = kf_solve(g, xty, xm, ld(xm, dd), dd, ld(alphas, a), fi != 0, aw + a * dd * dd, w + a * dd)
        st(b, a, r[0])
        st(trust, a, Float32(1) if r[1] else Float32(0))


def kf_solve_kernel(g: FP, xty: FP, xm: FP, d: Int32, alphas: FP, na: Int32, fi: Int32, aw: FP, w: FP, b: FP,
                    trust: FP, wf: IP, woff: Int32, nonce: Int32):
    _kf_solve_kernel_body(g, xty, xm, d, alphas, na, fi, aw, w, b, trust)
    witness_end(wf, woff, nonce)

@always_inline
def _kf_ff_unit_kernel_body(x: FP, y: FP, n: Int32, d: Int32, fi: Int32, s: Int32, e: Int32, u0: Int32, count: Int32,
                      sh: FP, sl: FP):
    var u = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    if u < Int(count):
        ridge_ff_unit(Int(u0) + u, x, y, Int(n), Int(d), 1, fi != 0, False, Int(n), sh, sl, Int(s), Int(e))


def kf_ff_unit_kernel(x: FP, y: FP, n: Int32, d: Int32, fi: Int32, s: Int32, e: Int32, u0: Int32, count: Int32,
                      sh: FP, sl: FP, wf: IP, woff: Int32, nonce: Int32):
    _kf_ff_unit_kernel_body(x, y, n, d, fi, s, e, u0, count, sh, sl)
    witness_end(wf, woff, nonce)

@always_inline
def _kf_ff_solve_kernel_body(d: Int32, fi: Int32, alpha: Float32, sh: FP, sl: FP, bh: FP, bl: FP, fh: FP, fl: FP, tmp: FP,
                       w: FP, b: FP, a: Int32):
    var dd = Int(d)
    st(b, Int(a), kf_ff_solve(dd, fi != 0, alpha, sh, sl, bh, bl, fh, fl, tmp, w + Int(a) * dd))


def kf_ff_solve_kernel(d: Int32, fi: Int32, alpha: Float32, sh: FP, sl: FP, bh: FP, bl: FP, fh: FP, fl: FP, tmp: FP,
                       w: FP, b: FP, a: Int32, wf: IP, woff: Int32, nonce: Int32):
    _kf_ff_solve_kernel_body(d, fi, alpha, sh, sl, bh, bl, fh, fl, tmp, w, b, a)
    witness_end(wf, woff, nonce)

@always_inline
def _kf_pred_kernel_body(x: FP, d: Int32, s: Int32, e: Int32, na: Int32, w: FP, b: FP, p: FP, stride: Int32):
    var q = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    var nt = Int(e) - Int(s)
    if q < nt * Int(na):
        var a = q // nt
        var r = q - a * nt
        st(p, a * Int(stride) + r, kf_pred(x, Int(s) + r, Int(d), w + a * Int(d), ld(b, a)))


def kf_pred_kernel(x: FP, d: Int32, s: Int32, e: Int32, na: Int32, w: FP, b: FP, p: FP, stride: Int32, wf: IP, woff: Int32, nonce: Int32):
    _kf_pred_kernel_body(x, d, s, e, na, w, b, p, stride)
    witness_end(wf, woff, nonce)

@always_inline
def _kf_score_kernel_body(y: FP, p: FP, s: Int32, e: Int32, na: Int32, stride: Int32, sums: FP):
    var a = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    if a < Int(na):
        st(sums, a, fa(ld(sums, a), kf_score(y, p + a * Int(stride), Int(s), Int(e))))


def kf_score_kernel(y: FP, p: FP, s: Int32, e: Int32, na: Int32, stride: Int32, sums: FP, wf: IP, woff: Int32, nonce: Int32):
    _kf_score_kernel_body(y, p, s, e, na, stride, sums)
    witness_end(wf, woff, nonce)

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
    var dtr = ctx.enqueue_create_buffer[DType.float32](max(na, 1))
    var htr = List[Float32](length=max(na, 1), fill=Float32(0))
    var ffw = d + 1 + d * d + d
    var ff_units = ridge_ff_units(n, d, 1)
    var dsh = ctx.enqueue_create_buffer[DType.float32](ffw)
    var dsl = ctx.enqueue_create_buffer[DType.float32](ffw)
    var dbh = ctx.enqueue_create_buffer[DType.float32](max(d, 1))
    var dbl = ctx.enqueue_create_buffer[DType.float32](max(d, 1))
    var dfh = ctx.enqueue_create_buffer[DType.float32](max(d * d, 1))
    var dfl = ctx.enqueue_create_buffer[DType.float32](max(d * d, 1))
    var dtmp = ctx.enqueue_create_buffer[DType.float32](d + 1)
    var hfp = fp.copy()
    if n_x > 0:
        ctx.enqueue_copy(dst_buf=dx, src_ptr=x)
    if n_y > 0:
        ctx.enqueue_copy(dst_buf=dy, src_ptr=y)
    ctx.enqueue_copy(dst_buf=dal, src_ptr=hfp.unsafe_ptr())
    dsum.enqueue_fill(Float32(0))
    var cells = d * (d + 1) // 2
    # lane/neural-pass93 + the Metal witness (x_linear/witness.mojo): per
    # fold, unit A (means, Gram, solves and the trust flags home) rebuilds
    # from the data; unit B (float-float re-solves, predictions, the score
    # added into the sums) reruns from the sums as they stood before it
    var nt_max = n // k + 1
    var wcap = max(_xg_blocks(d + 1) + _xg_blocks(cells + d) + _xg_blocks(na),
                   _xg_blocks(d + 1) + _xg_blocks(max(ff_units - (d + 1), 1)) + na + _xg_blocks(nt_max * na) + _xg_blocks(na))
    var wit = Witness(ctx, wcap)
    var dsum_save = ctx.enqueue_create_buffer[DType.float32](max(na, 1))
    for f in range(k):
        var s = Int32(kf_start(n, k, f))
        var e = Int32(kf_end(n, k, f))
        var ta = 0
        while True:
            var nonce = wit.begin()
            var wo = 0
            ctx.enqueue_function[kf_means_kernel](dx.unsafe_ptr(), dy.unsafe_ptr(), Int32(n), Int32(d), s, e, Int32(fi),
                                                  dxm.unsafe_ptr(), wit.p(), Int32(wo), nonce,
                                                  grid_dim=_xg_blocks(d + 1), block_dim=XG_TPB)
            wo += _xg_blocks(d + 1)
            ctx.enqueue_function[kf_cells_kernel](dx.unsafe_ptr(), dy.unsafe_ptr(), Int32(n), Int32(d), s, e, dxm.unsafe_ptr(),
                                                  dg.unsafe_ptr(), dxty.unsafe_ptr(), wit.p(), Int32(wo), nonce,
                                                  grid_dim=_xg_blocks(cells + d), block_dim=XG_TPB)
            wo += _xg_blocks(cells + d)
            ctx.enqueue_function[kf_solve_kernel](dg.unsafe_ptr(), dxty.unsafe_ptr(), dxm.unsafe_ptr(), Int32(d), dal.unsafe_ptr(),
                                                  Int32(na), Int32(fi), daw.unsafe_ptr(), dw.unsafe_ptr(), db.unsafe_ptr(),
                                                  dtr.unsafe_ptr(), wit.p(), Int32(wo), nonce,
                                                  grid_dim=_xg_blocks(na), block_dim=XG_TPB)
            wo += _xg_blocks(na)
            # lane/neural-pass93: the alphas whose float32 factor is not trusted, in float-float
            ctx.enqueue_copy(dst_ptr=htr.unsafe_ptr(), src_buf=dtr)
            if wit.ok(ctx, wo, "RidgeCV fold"):
                break
            ta += 1
            if ta >= WITNESS_TRIES:
                wit.fail()
        ctx.synchronize()
        ctx.enqueue_copy(dst_buf=dsum_save, src_buf=dsum)
        var tb = 0
        while True:
            var nonce = wit.begin()
            var wo = 0
            var have_ff = False
            for a in range(na):
                if htr[a] == Float32(1):
                    continue
                if not have_ff:
                    ctx.enqueue_function[kf_ff_unit_kernel](dx.unsafe_ptr(), dy.unsafe_ptr(), Int32(n), Int32(d), Int32(fi), s, e,
                                                            Int32(0), Int32(d + 1), dsh.unsafe_ptr(), dsl.unsafe_ptr(),
                                                            wit.p(), Int32(wo), nonce,
                                                            grid_dim=_xg_blocks(d + 1), block_dim=XG_TPB)
                    wo += _xg_blocks(d + 1)
                    ctx.enqueue_function[kf_ff_unit_kernel](dx.unsafe_ptr(), dy.unsafe_ptr(), Int32(n), Int32(d), Int32(fi), s, e,
                                                            Int32(d + 1), Int32(ff_units - (d + 1)), dsh.unsafe_ptr(), dsl.unsafe_ptr(),
                                                            wit.p(), Int32(wo), nonce,
                                                            grid_dim=_xg_blocks(ff_units - (d + 1)), block_dim=XG_TPB)
                    wo += _xg_blocks(ff_units - (d + 1))
                    have_ff = True
                ctx.enqueue_function[kf_ff_solve_kernel](Int32(d), Int32(fi), hfp[a], dsh.unsafe_ptr(), dsl.unsafe_ptr(),
                                                         dbh.unsafe_ptr(), dbl.unsafe_ptr(), dfh.unsafe_ptr(), dfl.unsafe_ptr(),
                                                         dtmp.unsafe_ptr(), dw.unsafe_ptr(), db.unsafe_ptr(), Int32(a),
                                                         wit.p(), Int32(wo), nonce, grid_dim=1, block_dim=1)
                wo += 1
            var nt = Int(e) - Int(s)
            ctx.enqueue_function[kf_pred_kernel](dx.unsafe_ptr(), Int32(d), s, e, Int32(na), dw.unsafe_ptr(), db.unsafe_ptr(),
                                                 dp.unsafe_ptr(), Int32(stride), wit.p(), Int32(wo), nonce,
                                                 grid_dim=_xg_blocks(nt * na), block_dim=XG_TPB)
            wo += _xg_blocks(nt * na)
            ctx.enqueue_function[kf_score_kernel](dy.unsafe_ptr(), dp.unsafe_ptr(), s, e, Int32(na), Int32(stride),
                                                  dsum.unsafe_ptr(), wit.p(), Int32(wo), nonce,
                                                  grid_dim=_xg_blocks(na), block_dim=XG_TPB)
            wo += _xg_blocks(na)
            if wit.ok(ctx, wo, "RidgeCV scores"):
                break
            tb += 1
            if tb >= WITNESS_TRIES:
                wit.fail()
            ctx.enqueue_copy(dst_buf=dsum, src_buf=dsum_save)
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
    _ = dtr^
    _ = htr^
    _ = dsh^
    _ = dsl^
    _ = dbh^
    _ = wit^
    _ = dsum_save^
    _ = dbl^
    _ = dfh^
    _ = dfl^
    _ = dtmp^



# ------------------------------------------------ Ridge's float-float refit on the grid (lane/neural-pass93)
def ridge_ff_unit_kernel(x: FP, y: FP, n: Int32, d: Int32, t_n: Int32, fi: Int32, sw: Int32, u0: Int32, count: Int32,
                         sh: FP, sl: FP, wf: IP, woff: Int32, nonce: Int32):
    var u = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    if u < Int(count):
        ridge_ff_unit(Int(u0) + u, x, y, Int(n), Int(d), Int(t_n), fi != 0, sw != 0, Int(n) * Int(t_n), sh, sl)
    witness_end(wf, woff, nonce)


def ridge_ff_solve_kernel(d: Int32, t_n: Int32, fi: Int32, alpha: Float32, sh: FP, sl: FP, bh: FP, bl: FP, dst: FP,
                          fh: FP, fl: FP, wf: IP, woff: Int32, nonce: Int32):
    """dst: coef T*d | intercept T | ok (1 / 0)."""
    var ok = ridge_ff_solve(Int(d), Int(t_n), fi != 0, alpha, sh, sl, bh, bl, dst, fh, fl)
    st(dst, Int(t_n) * Int(d) + Int(t_n), Float32(1) if ok else Float32(0))
    witness_end(wf, woff, nonce)


def _ridge_ff_grid(mut ctx: DeviceContext, x: FP, y: FP, n: Int, d: Int, t_n: Int, fi: Bool, sw: Bool, alpha: Float32,
                   res: FP, sidx: Int) raises:
    var nm = d + t_n
    var units = ridge_ff_units(n, d, t_n)
    var words = d + t_n + d * d + d * t_n
    var dsh = ctx.enqueue_create_buffer[DType.float32](words)
    var dsl = ctx.enqueue_create_buffer[DType.float32](words)
    var dbh = ctx.enqueue_create_buffer[DType.float32](max(d, 1))
    var dbl = ctx.enqueue_create_buffer[DType.float32](max(d, 1))
    var dout = ctx.enqueue_create_buffer[DType.float32](t_n * d + t_n + 1)
    var dfh = ctx.enqueue_create_buffer[DType.float32](max(d * d, 1))
    var dfl = ctx.enqueue_create_buffer[DType.float32](max(d * d, 1))
    # lane/neural-xlw: the refit as ONE guarded unit from zeroed sums (the M2
    # under load returned an all-zero RidgeClassifier from a cut unit launch)
    var h = List[Float32](length=t_n * d + t_n + 1, fill=Float32(0))
    var b1 = _xg_blocks(nm)
    var b2 = _xg_blocks(units - nm)
    var wit = Witness(ctx, b1 + b2 + 1)
    var tries = 0
    while True:
        var nonce = wit.begin()
        dsh.enqueue_fill(Float32(0))
        dsl.enqueue_fill(Float32(0))
        ctx.enqueue_function[ridge_ff_unit_kernel](x, y, Int32(n), Int32(d), Int32(t_n), Int32(1 if fi else 0),
                                                   Int32(1 if sw else 0), Int32(0), Int32(nm), dsh.unsafe_ptr(),
                                                   dsl.unsafe_ptr(), wit.p(), Int32(0), nonce,
                                                   grid_dim=b1, block_dim=XG_TPB)
        ctx.enqueue_function[ridge_ff_unit_kernel](x, y, Int32(n), Int32(d), Int32(t_n), Int32(1 if fi else 0),
                                                   Int32(1 if sw else 0), Int32(nm), Int32(units - nm), dsh.unsafe_ptr(),
                                                   dsl.unsafe_ptr(), wit.p(), Int32(b1), nonce,
                                                   grid_dim=b2, block_dim=XG_TPB)
        ctx.enqueue_function[ridge_ff_solve_kernel](Int32(d), Int32(t_n), Int32(1 if fi else 0), alpha, dsh.unsafe_ptr(),
                                                    dsl.unsafe_ptr(), dbh.unsafe_ptr(), dbl.unsafe_ptr(), dout.unsafe_ptr(),
                                                    dfh.unsafe_ptr(), dfl.unsafe_ptr(), wit.p(), Int32(b1 + b2), nonce,
                                                    grid_dim=1, block_dim=1)
        ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=dout)
        if wit.ok(ctx, b1 + b2 + 1, "Ridge float-float refit"):
            break
        tries += 1
        if tries >= WITNESS_TRIES:
            wit.fail()
    ctx.synchronize()
    _ = wit^
    var ok = h[t_n * d + t_n] == Float32(1)
    if ok:
        for i in range(t_n * d + t_n):
            res.unsafe_store(i, h[i])
    res.unsafe_store(sidx, Float32(0) if ok else Float32(2))
    _ = h^
    _ = dsh^
    _ = dsl^
    _ = dbh^
    _ = dbl^
    _ = dout^
    _ = dfh^
    _ = dfl^



# ------------------------------------------------ isotonic on the grid (lane/neural-pass107)
# The fit's sort as an LSD radix sort on the device: the rows start in index
# order and are stably sorted by y's key, then by x's key (8-bit digits, four
# passes each; a pass is a count per tile of ISO_TILE rows, one scan and a
# stable scatter per tile). The keys order exactly as the host's comparison
# (`_iso_less`: x, then y, then the row): IEEE order with -0 folded onto +0
# (the comparison has -0 == +0). The order (x, y, row) is total, so the
# permutation is the host sort's whatever the algorithm, and so is every
# word after it (`iso_fit_sorted` on one thread). Predict: one thread a query.
comptime ISO_TILE = 4096
comptime ISO_RADIX = 256


@always_inline
def _iso_key(v: Float32) -> UInt32:
    var b = bitcast[DType.uint32](v)
    if (b & UInt32(0x7FFFFFFF)) == UInt32(0):
        b = UInt32(0)
    if (b & UInt32(0x80000000)) != UInt32(0):
        return ~b
    return b | UInt32(0x80000000)


@always_inline
def _iso_keys_kernel_body(x: FP, y: FP, kx: IP, ky: IP, perm: IP, n: Int32):
    var i = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    if i < Int(n):
        kx.unsafe_store(i, bitcast[DType.int32](_iso_key(x.unsafe_load(i))))
        ky.unsafe_store(i, bitcast[DType.int32](_iso_key(y.unsafe_load(i))))
        perm.unsafe_store(i, Int32(i))


def iso_keys_kernel(x: FP, y: FP, kx: IP, ky: IP, perm: IP, n: Int32, wf: IP, woff: Int32, nonce: Int32):
    _iso_keys_kernel_body(x, y, kx, ky, perm, n)
    witness_end(wf, woff, nonce)

@always_inline
def _digit(keys: IP, row: Int, shift: Int) -> Int:
    return Int((bitcast[DType.uint32](keys.unsafe_load(row)) >> UInt32(shift)) & UInt32(ISO_RADIX - 1))


@always_inline
def _iso_count_kernel_body(keys: IP, perm: IP, n: Int32, shift: Int32, counts: IP, ntiles: Int32):
    var tl = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    if tl < Int(ntiles):
        for dg in range(ISO_RADIX):
            counts.unsafe_store(dg * Int(ntiles) + tl, Int32(0))
        var lo = tl * ISO_TILE
        var hi = min(Int(n), lo + ISO_TILE)
        for i in range(lo, hi):
            var dg = _digit(keys, Int(perm.unsafe_load(i)), Int(shift))
            var o = dg * Int(ntiles) + tl
            counts.unsafe_store(o, counts.unsafe_load(o) + 1)


def iso_count_kernel(keys: IP, perm: IP, n: Int32, shift: Int32, counts: IP, ntiles: Int32, wf: IP, woff: Int32, nonce: Int32):
    _iso_count_kernel_body(keys, perm, n, shift, counts, ntiles)
    witness_end(wf, woff, nonce)

@always_inline
def _iso_scan_kernel_body(counts: IP, total: Int32):
    """counts (digit-major, tile-minor) -> their exclusive prefix, in place: one thread."""
    if Int(block_idx.x) == 0 and Int(thread_idx.x) == 0:
        var acc = Int32(0)
        for i in range(Int(total)):
            var c = counts.unsafe_load(i)
            counts.unsafe_store(i, acc)
            acc += c


def iso_scan_kernel(counts: IP, total: Int32, wf: IP, woff: Int32, nonce: Int32):
    _iso_scan_kernel_body(counts, total)
    witness_end(wf, woff, nonce)

@always_inline
def _iso_scatter_kernel_body(keys: IP, src: IP, dst: IP, n: Int32, shift: Int32, offs: IP, ntiles: Int32):
    var tl = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    if tl < Int(ntiles):
        var lo = tl * ISO_TILE
        var hi = min(Int(n), lo + ISO_TILE)
        for i in range(lo, hi):
            var r = src.unsafe_load(i)
            var o = _digit(keys, Int(r), Int(shift)) * Int(ntiles) + tl
            var at = offs.unsafe_load(o)
            dst.unsafe_store(Int(at), r)
            offs.unsafe_store(o, at + 1)


def iso_scatter_kernel(keys: IP, src: IP, dst: IP, n: Int32, shift: Int32, offs: IP, ntiles: Int32, wf: IP, woff: Int32, nonce: Int32):
    _iso_scatter_kernel_body(keys, src, dst, n, shift, offs, ntiles)
    witness_end(wf, woff, nonce)

# ---------------------------------------------- block radix (lane/neural-pass118)
# The sort above ran one thread per 4096-row tile with its digit counters
# in device memory (M4, 1M rows: 70 ms). Here a pass is three launches:
# block b (RS_NT threads x RS_IPT contiguous rows) counts each thread's
# rows per 4-bit digit in threadgroup memory and publishes the block's
# per-digit totals; one thread scans those (digit-major, block-minor) and
# notes a pass whose rows all share one digit; block b recounts, scans its
# thread counters (digit-major, thread-minor) and writes each row to its
# digit's global offset + its thread's offset + its rank within the
# thread: the stable order, so the LSD passes give the (x, y, row) order
# (a total order: any correct sort gives this one permutation, so the
# words downstream are unchanged). A one-digit pass is a copy.
# `MOJOLEARN_X_LINEAR_ISO_BLOCK_RADIX=0` restores the tile sort.
comptime RS_BITS = 4
comptime RS_D = 1 << RS_BITS
comptime RS_NT = 256
comptime RS_IPT = 16
comptime RS_TILE = RS_NT * RS_IPT
comptime RS_BYTES = (RS_D * RS_NT + RS_NT) * 4
comptime ISO_BLOCK_RADIX = lib_smem_page_fits_for[TARGET_COLUMN, RS_BYTES]()


@always_inline
def _rdig(keys: IP, row: Int, shift: Int) -> Int:
    return Int((bitcast[DType.uint32](keys.unsafe_load(row)) >> UInt32(shift)) & UInt32(RS_D - 1))


@always_inline
def _rs_count_kernel_body(keys: IP, src: IP, n: Int32, shift: Int32, btot: IP, nb: Int32):
    var cnt = stack_allocation[RS_D * RS_NT, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    for d in range(RS_D):
        cnt[d * RS_NT + tid] = Int32(0)
    var lo0 = Int(block_idx.x) * RS_TILE + tid * RS_IPT
    for u in range(RS_IPT):
        var i = lo0 + u
        if i < Int(n):
            var d = _rdig(keys, Int(src.unsafe_load(i)), Int(shift))
            cnt[d * RS_NT + tid] = cnt[d * RS_NT + tid] + 1
    barrier()
    if tid < RS_D:
        var c = Int32(0)
        for t in range(RS_NT):
            c += cnt[tid * RS_NT + t]
        btot.unsafe_store(tid * Int(nb) + Int(block_idx.x), c)


def rs_count_kernel(keys: IP, src: IP, n: Int32, shift: Int32, btot: IP, nb: Int32, wf: IP, woff: Int32, nonce: Int32):
    _rs_count_kernel_body(keys, src, n, shift, btot, nb)
    witness_end(wf, woff, nonce)

@always_inline
def _rs_scan_kernel_body(btot: IP, nb: Int32, n: Int32, flag: IP):
    """Exclusive prefix of btot (digit-major, block-minor) in place; flag[0]
    = 1 when one digit holds every row (the pass is a copy)."""
    if Int(block_idx.x) == 0 and Int(thread_idx.x) == 0:
        var acc = Int32(0)
        var one = Int32(0)
        for d in range(RS_D):
            var dt = Int32(0)
            for b in range(Int(nb)):
                var c = btot.unsafe_load(d * Int(nb) + b)
                btot.unsafe_store(d * Int(nb) + b, acc)
                acc += c
                dt += c
            if dt == n:
                one = 1
        flag.unsafe_store(0, one)


def rs_scan_kernel(btot: IP, nb: Int32, n: Int32, flag: IP, wf: IP, woff: Int32, nonce: Int32):
    _rs_scan_kernel_body(btot, nb, n, flag)
    witness_end(wf, woff, nonce)

@always_inline
def _rs_scatter_kernel_body(keys: IP, src: IP, dst: IP, n: Int32, shift: Int32, btot: IP, nb: Int32, flag: IP):
    var nn = Int(n)
    var tid = Int(thread_idx.x)
    var lo = Int(block_idx.x) * RS_TILE + tid * RS_IPT
    if flag.unsafe_load(0) != 0:
        for u in range(RS_IPT):
            var i = lo + u
            if i < nn:
                dst.unsafe_store(i, src.unsafe_load(i))
        return
    var cnt = stack_allocation[RS_D * RS_NT, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var part = stack_allocation[RS_NT, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    for d in range(RS_D):
        cnt[d * RS_NT + tid] = Int32(0)
    var lo0 = Int(block_idx.x) * RS_TILE + tid * RS_IPT
    for u in range(RS_IPT):
        var i = lo0 + u
        if i < Int(n):
            var d = _rdig(keys, Int(src.unsafe_load(i)), Int(shift))
            cnt[d * RS_NT + tid] = cnt[d * RS_NT + tid] + 1
    barrier()
    # exclusive scan of cnt (digit-major, thread-minor) within the block:
    # thread t owns the RS_D consecutive entries [t * RS_D, t * RS_D + RS_D)
    var own = SIMD[DType.int32, RS_D]()
    var s = Int32(0)
    comptime for q in range(RS_D):
        own[q] = cnt[tid * RS_D + q]
        s += own[q]
    part[tid] = s
    barrier()
    var off = 1
    while off < RS_NT:
        var v = part[tid] + (part[tid - off] if tid >= off else Int32(0))
        barrier()
        part[tid] = v
        barrier()
        off *= 2
    var base = part[tid] - s
    barrier()
    comptime for q in range(RS_D):
        cnt[tid * RS_D + q] = base
        base += own[q]
    barrier()
    # the block's in-tile offset of (digit d, thread t) is cnt[d * RS_NT + t]
    # minus the block's first offset of digit d, which is cnt[d * RS_NT]
    var run = SIMD[DType.int32, RS_D](0)
    for u in range(RS_IPT):
        var i = lo + u
        if i < nn:
            var r = src.unsafe_load(i)
            var d = _rdig(keys, Int(r), Int(shift))
            var at = btot.unsafe_load(d * Int(nb) + Int(block_idx.x)) + (cnt[d * RS_NT + tid] - cnt[d * RS_NT]) + run[d]
            dst.unsafe_store(Int(at), r)
            run[d] += 1


def rs_scatter_kernel(keys: IP, src: IP, dst: IP, n: Int32, shift: Int32, btot: IP, nb: Int32, flag: IP, wf: IP, woff: Int32, nonce: Int32):
    _rs_scatter_kernel_body(keys, src, dst, n, shift, btot, nb, flag)
    witness_end(wf, woff, nonce)

@always_inline
def _iso_gather_kernel_body(x: FP, y: FP, n: Int32, perm: IP, fw: FP):
    var j = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    var nn = Int(n)
    if j < nn:
        iso_gather_one(j, x, y, nn, False, perm, fw + 3 * nn, fw + 4 * nn, fw + 5 * nn)


def iso_gather_kernel(x: FP, y: FP, n: Int32, perm: IP, fw: FP, wf: IP, woff: Int32, nonce: Int32):
    _iso_gather_kernel_body(x, y, n, perm, fw)
    witness_end(wf, woff, nonce)

# The group bounds (lane/neural-pass117): one thread walked all n sorted
# rows (M4, istella's 87-value column at 1M rows: 109 ms). A row whose x
# equals the row before never starts a group, so the walk only needs the
# rows where x changes: flagged and compacted in order by the grid, then
# `iso_bounds_from` walks them on one thread (the same starts).
@always_inline
def _iso_changed(xs: FP, j: Int, n: Int) -> Int:
    if j >= 1 and j < n and xs.unsafe_load(j) != xs.unsafe_load(j - 1):
        return 1
    return 0


@always_inline
def _iso_flag_count_kernel_body(fw: FP, n: Int32, bcnt: IP):
    var nn = Int(n)
    var tid = Int(thread_idx.x)
    var j = Int(block_idx.x) * XG_TPB + tid
    var sh = stack_allocation[XG_TPB, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    sh[tid] = Int32(_iso_changed(fw + 3 * nn, j, nn))
    barrier()
    if tid == 0:
        var c = Int32(0)
        for u in range(XG_TPB):
            c += sh[u]
        bcnt.unsafe_store(Int(block_idx.x), c)


def iso_flag_count_kernel(fw: FP, n: Int32, bcnt: IP, wf: IP, woff: Int32, nonce: Int32):
    _iso_flag_count_kernel_body(fw, n, bcnt)
    witness_end(wf, woff, nonce)

@always_inline
def _iso_flag_scan_kernel_body(bcnt: IP, nb: Int32):
    """Exclusive prefix of the block counts in place; the total at [nb]."""
    if Int(block_idx.x) == 0 and Int(thread_idx.x) == 0:
        var acc = Int32(0)
        for b in range(Int(nb)):
            var c = bcnt.unsafe_load(b)
            bcnt.unsafe_store(b, acc)
            acc += c
        bcnt.unsafe_store(Int(nb), acc)


def iso_flag_scan_kernel(bcnt: IP, nb: Int32, wf: IP, woff: Int32, nonce: Int32):
    _iso_flag_scan_kernel_body(bcnt, nb)
    witness_end(wf, woff, nonce)

@always_inline
def _iso_flag_write_kernel_body(fw: FP, n: Int32, bcnt: IP, cand: IP):
    var nn = Int(n)
    var tid = Int(thread_idx.x)
    var j = Int(block_idx.x) * XG_TPB + tid
    var sh = stack_allocation[XG_TPB, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var f = _iso_changed(fw + 3 * nn, j, nn)
    sh[tid] = Int32(f)
    barrier()
    if f != 0:
        var r = Int32(0)
        for u in range(tid):
            r += sh[u]
        cand.unsafe_store(Int(bcnt.unsafe_load(Int(block_idx.x)) + r), Int32(j))


def iso_flag_write_kernel(fw: FP, n: Int32, bcnt: IP, cand: IP, wf: IP, woff: Int32, nonce: Int32):
    _iso_flag_write_kernel_body(fw, n, bcnt, cand)
    witness_end(wf, woff, nonce)

@always_inline
def _iso_bounds_kernel_body(fw: FP, n: Int32, iw: IP, mslot: IP, cand: IP, bcnt: IP, nb: Int32):
    if Int(block_idx.x) == 0 and Int(thread_idx.x) == 0:
        var nn = Int(n)
        var nc = Int(bcnt.unsafe_load(Int(nb)))
        mslot.unsafe_store(0, Int32(iso_bounds_from(fw + 3 * nn, nn, cand, nc, iw + nn)))


def iso_bounds_kernel(fw: FP, n: Int32, iw: IP, mslot: IP, cand: IP, bcnt: IP, nb: Int32, wf: IP, woff: Int32, nonce: Int32):
    _iso_bounds_kernel_body(fw, n, iw, mslot, cand, bcnt, nb)
    witness_end(wf, woff, nonce)

@always_inline
def _iso_group_kernel_body(fw: FP, n: Int32, iw: IP, mslot: IP):
    var g = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    var nn = Int(n)
    if g < Int(mslot.unsafe_load(0)):
        iso_group(g, fw + 3 * nn, fw + 4 * nn, fw + 5 * nn, iw + nn, fw, nn)


def iso_group_kernel(fw: FP, n: Int32, iw: IP, mslot: IP, wf: IP, woff: Int32, nonce: Int32):
    _iso_group_kernel_body(fw, n, iw, mslot)
    witness_end(wf, woff, nonce)

@always_inline
def _iso_after_kernel_body(n: Int32, ip: IP, fp: FP, res: FP, fw: FP, iw: IP, mslot: IP):
    if Int(block_idx.x) == 0 and Int(thread_idx.x) == 0:
        iso_after_unique(Int(mslot.unsafe_load(0)), Int(n), ip, fp, res, fw, iw)


def iso_after_kernel(n: Int32, ip: IP, fp: FP, res: FP, fw: FP, iw: IP, mslot: IP, wf: IP, woff: Int32, nonce: Int32):
    _iso_after_kernel_body(n, ip, fp, res, fw, iw, mslot)
    witness_end(wf, woff, nonce)

@always_inline
def _iso_predict_kernel_body(x: FP, thr: FP, n: Int32, m: Int32, oob: Int32, fp: FP, res: FP):
    var q = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    if q < Int(n):
        iso_predict_one(q, x, thr, Int(m), Int(oob), fp, res)


def iso_predict_kernel(x: FP, thr: FP, n: Int32, m: Int32, oob: Int32, fp: FP, res: FP, wf: IP, woff: Int32, nonce: Int32):
    _iso_predict_kernel_body(x, thr, n, m, oob, fp, res)
    witness_end(wf, woff, nonce)

def ISO_WIT_CAP(n: Int) -> Int:
    """The witness words of one isotonic fit (an upper bound over both sorts)."""
    var nb = max((n + RS_TILE - 1) // RS_TILE, 1)
    var nt = (n + ISO_TILE - 1) // ISO_TILE
    return 2 * (32 // RS_BITS) * (2 * nb + 1) + 8 * (2 * _xg_blocks(nt) + 1) + 6 * _xg_blocks(n) + 16


def _iso_fit_grid(x: FP, n_x: Int, y: FP, n_y: Int, n: Int, ip: List[Int32], fp: List[Float32], n_out: Int,
                  n_fw: Int, n_iw: Int, res: FP) raises:
    var ctx = linear_ctx()
    var hip = ip.copy()
    var hfp = fp.copy()
    var dx = ctx.enqueue_create_buffer[DType.float32](max(n_x, 1))
    var dy = ctx.enqueue_create_buffer[DType.float32](max(n_y, 1))
    var dip = ctx.enqueue_create_buffer[DType.int32](max(len(hip), 1))
    var dfp = ctx.enqueue_create_buffer[DType.float32](max(len(hfp), 1))
    var dout = ctx.enqueue_create_buffer[DType.float32](max(n_out, 1))
    var dfw = ctx.enqueue_create_buffer[DType.float32](max(n_fw, 1))
    var diw = ctx.enqueue_create_buffer[DType.int32](max(n_iw, 1))
    var dkx = ctx.enqueue_create_buffer[DType.int32](max(n, 1))
    var dky = ctx.enqueue_create_buffer[DType.int32](max(n, 1))
    var dpa = ctx.enqueue_create_buffer[DType.int32](max(n, 1))
    var dpb = ctx.enqueue_create_buffer[DType.int32](max(n, 1))
    var ntiles = (n + ISO_TILE - 1) // ISO_TILE
    var dcnt = ctx.enqueue_create_buffer[DType.int32](ISO_RADIX * ntiles)
    if n_x > 0:
        ctx.enqueue_copy(dst_buf=dx, src_ptr=x)
    if n_y > 0:
        ctx.enqueue_copy(dst_buf=dy, src_ptr=y)
    ctx.enqueue_copy(dst_buf=dip, src_ptr=hip.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dfp, src_ptr=hfp.unsafe_ptr())
    var dm = ctx.enqueue_create_buffer[DType.int32](1)
    var nbk = _xg_blocks(n)
    var dbc = ctx.enqueue_create_buffer[DType.int32](nbk + 1)
    var dcand = ctx.enqueue_create_buffer[DType.int32](max(n, 1))
    # the whole isotonic fit as ONE guarded unit from zeroed scratch (x_linear/witness.mojo)
    var wit = Witness(ctx, ISO_WIT_CAP(n))
    var tries = 0
    while True:
        var nonce = wit.begin()
        var wo = 0
        dout.enqueue_fill(Float32(0))
        dfw.enqueue_fill(Float32(0))
        diw.enqueue_fill(Int32(0))
        ctx.enqueue_function[iso_keys_kernel](dx.unsafe_ptr(), dy.unsafe_ptr(), dkx.unsafe_ptr(), dky.unsafe_ptr(),
                                              dpa.unsafe_ptr(), Int32(n), wit.p(), Int32(wo), nonce, grid_dim=_xg_blocks(n), block_dim=XG_TPB)
        wo += _xg_blocks(n)
        var cur_a = True
        var block_radix = False
        comptime if ISO_BLOCK_RADIX:
            block_radix = String(getenv("MOJOLEARN_X_LINEAR_ISO_BLOCK_RADIX")) != "0"
        if block_radix:
            var nb = max((n + RS_TILE - 1) // RS_TILE, 1)
            var dbt = ctx.enqueue_create_buffer[DType.int32](RS_D * nb)
            var dfl = ctx.enqueue_create_buffer[DType.int32](1)
            for key in range(2):
                for pas in range(32 // RS_BITS):
                    var shift = Int32(RS_BITS * pas)
                    var kp = IP(unsafe_from_address=Int(dky.unsafe_ptr()) if key == 0 else Int(dkx.unsafe_ptr()))
                    var src = IP(unsafe_from_address=Int(dpa.unsafe_ptr()) if cur_a else Int(dpb.unsafe_ptr()))
                    var dst = IP(unsafe_from_address=Int(dpb.unsafe_ptr()) if cur_a else Int(dpa.unsafe_ptr()))
                    ctx.enqueue_function[rs_count_kernel](kp, src, Int32(n), shift, dbt.unsafe_ptr(), Int32(nb),
                                                          wit.p(), Int32(wo), nonce, grid_dim=nb, block_dim=RS_NT)
                    wo += nb
                    ctx.enqueue_function[rs_scan_kernel](dbt.unsafe_ptr(), Int32(nb), Int32(n), dfl.unsafe_ptr(),
                                                         wit.p(), Int32(wo), nonce, grid_dim=1, block_dim=1)
                    wo += 1
                    ctx.enqueue_function[rs_scatter_kernel](kp, src, dst, Int32(n), shift, dbt.unsafe_ptr(), Int32(nb),
                                                            dfl.unsafe_ptr(), wit.p(), Int32(wo), nonce, grid_dim=nb, block_dim=RS_NT)
                    wo += nb
                    cur_a = not cur_a
            ctx.synchronize()
            _ = dbt^
            _ = dfl^
        for key in range(2 if not block_radix else 0):
            for pas in range(4):
                var shift = Int32(8 * pas)
                var kp = IP(unsafe_from_address=Int(dky.unsafe_ptr()) if key == 0 else Int(dkx.unsafe_ptr()))
                var src = IP(unsafe_from_address=Int(dpa.unsafe_ptr()) if cur_a else Int(dpb.unsafe_ptr()))
                var dst = IP(unsafe_from_address=Int(dpb.unsafe_ptr()) if cur_a else Int(dpa.unsafe_ptr()))
                ctx.enqueue_function[iso_count_kernel](kp, src, Int32(n), shift, dcnt.unsafe_ptr(), Int32(ntiles),
                                                       wit.p(), Int32(wo), nonce, grid_dim=_xg_blocks(ntiles), block_dim=XG_TPB)
                wo += _xg_blocks(ntiles)
                ctx.enqueue_function[iso_scan_kernel](dcnt.unsafe_ptr(), Int32(ISO_RADIX * ntiles), wit.p(), Int32(wo), nonce, grid_dim=1, block_dim=1)
                wo += 1
                ctx.enqueue_function[iso_scatter_kernel](kp, src, dst, Int32(n), shift, dcnt.unsafe_ptr(), Int32(ntiles),
                                                         wit.p(), Int32(wo), nonce, grid_dim=_xg_blocks(ntiles), block_dim=XG_TPB)
                wo += _xg_blocks(ntiles)
                cur_a = not cur_a
        var pp = IP(unsafe_from_address=Int(dpa.unsafe_ptr()) if cur_a else Int(dpb.unsafe_ptr()))
        ctx.enqueue_function[iso_gather_kernel](dx.unsafe_ptr(), dy.unsafe_ptr(), Int32(n), pp, dfw.unsafe_ptr(),
                                                wit.p(), Int32(wo), nonce, grid_dim=_xg_blocks(n), block_dim=XG_TPB)
        wo += _xg_blocks(n)
        ctx.enqueue_function[iso_flag_count_kernel](dfw.unsafe_ptr(), Int32(n), dbc.unsafe_ptr(), wit.p(), Int32(wo), nonce, grid_dim=nbk, block_dim=XG_TPB)
        wo += nbk
        ctx.enqueue_function[iso_flag_scan_kernel](dbc.unsafe_ptr(), Int32(nbk), wit.p(), Int32(wo), nonce, grid_dim=1, block_dim=1)
        wo += 1
        ctx.enqueue_function[iso_flag_write_kernel](dfw.unsafe_ptr(), Int32(n), dbc.unsafe_ptr(), dcand.unsafe_ptr(),
                                                    wit.p(), Int32(wo), nonce, grid_dim=nbk, block_dim=XG_TPB)
        wo += nbk
        ctx.enqueue_function[iso_bounds_kernel](dfw.unsafe_ptr(), Int32(n), diw.unsafe_ptr(), dm.unsafe_ptr(),
                                                dcand.unsafe_ptr(), dbc.unsafe_ptr(), Int32(nbk), wit.p(), Int32(wo), nonce, grid_dim=1, block_dim=1)
        wo += 1
        ctx.enqueue_function[iso_group_kernel](dfw.unsafe_ptr(), Int32(n), diw.unsafe_ptr(), dm.unsafe_ptr(),
                                               wit.p(), Int32(wo), nonce, grid_dim=_xg_blocks(n), block_dim=XG_TPB)
        wo += _xg_blocks(n)
        ctx.enqueue_function[iso_after_kernel](Int32(n), dip.unsafe_ptr(), dfp.unsafe_ptr(), dout.unsafe_ptr(), dfw.unsafe_ptr(),
                                               diw.unsafe_ptr(), dm.unsafe_ptr(), wit.p(), Int32(wo), nonce, grid_dim=1, block_dim=1)
        wo += 1
        if n_out > 0:
            ctx.enqueue_copy(dst_ptr=res, src_buf=dout)
        if wit.ok(ctx, wo, "isotonic fit"):
            break
        tries += 1
        if tries >= WITNESS_TRIES:
            wit.fail()
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
    _ = dkx^
    _ = dky^
    _ = dpa^
    _ = dbc^
    _ = dcand^
    _ = dpb^
    _ = dcnt^
    _ = dm^
    _ = wit^


def _iso_predict_grid(x: FP, n_x: Int, thr: FP, n_thr: Int, n: Int, ip: List[Int32], fp: List[Float32], res: FP) raises:
    var ctx = linear_ctx()
    var hfp = fp.copy()
    var dx = ctx.enqueue_create_buffer[DType.float32](max(n_x, 1))
    var dt = ctx.enqueue_create_buffer[DType.float32](max(n_thr, 1))
    var dfp = ctx.enqueue_create_buffer[DType.float32](max(len(hfp), 1))
    var dout = ctx.enqueue_create_buffer[DType.float32](max(n, 1))
    ctx.enqueue_copy(dst_buf=dx, src_ptr=x)
    ctx.enqueue_copy(dst_buf=dt, src_ptr=thr)
    ctx.enqueue_copy(dst_buf=dfp, src_ptr=hfp.unsafe_ptr())
    # the whole isotonic predict as ONE guarded unit from zeroed scratch (x_linear/witness.mojo)
    var wit = Witness(ctx, _xg_blocks(n))
    var tries = 0
    while True:
        var nonce = wit.begin()
        var wo = 0
        ctx.enqueue_function[iso_predict_kernel](dx.unsafe_ptr(), dt.unsafe_ptr(), Int32(n), ip[0], ip[1], dfp.unsafe_ptr(),
                                                 dout.unsafe_ptr(), wit.p(), Int32(wo), nonce, grid_dim=_xg_blocks(n), block_dim=XG_TPB)
        wo += _xg_blocks(n)
        ctx.enqueue_copy(dst_ptr=res, src_buf=dout)
        if wit.ok(ctx, wo, "isotonic predict"):
            break
        tries += 1
        if tries >= WITNESS_TRIES:
            wit.fail()
    ctx.synchronize()
    _ = hfp^
    _ = dx^
    _ = dt^
    _ = dfp^
    _ = dout^
    _ = wit^


def fit_device(
    algo: Int, x: FP, n_x: Int, y: FP, n_y: Int, n: Int, d: Int,
    ip: List[Int32], fp: List[Float32], n_out: Int, n_fw: Int, n_iw: Int, res: FP,
) raises:
    if algo == ALGO_GLM and _glm_grid() and n > 0:
        _glm_fit_grid(x, n_x, y, n_y, n, d, ip, fp, n_out, res)
        return
    if algo == ALGO_RIDGE_KFOLD:
        _ridge_kfold_grid(x, n_x, y, n_y, n, d, ip, fp, res)
        return
    if algo == ALGO_SGD and len(ip) > 12 and sgd_mb_on(Int(ip[12]), Int(ip[0]), Int(ip[3])):
        _sgd_mb_grid(x, n_x, y, n_y, n, d, ip, fp, n_out, res)
        return
    # lane/neural-pass107: isotonic on the grid (the radix sort, then the fit on
    # one thread; one thread a predicted query); weighted fits keep the team form
    if algo == ALGO_ISOTONIC and len(ip) > 3 and Int(ip[3]) == 0 and n > 0:
        _iso_fit_grid(x, n_x, y, n_y, n, ip, fp, n_out, n_fw, n_iw, res)
        return
    if algo == ALGO_ISOTONIC_PREDICT and n > 0:
        _iso_predict_grid(x, n_x, y, n_y, n, ip, fp, res)
        return
    if algo == ALGO_SGD and _sgd_on_host():
        _fit_on_host(algo, x, y, n, d, ip, fp, n_out, n_fw, n_iw, res)
        return
    var ctx = linear_ctx()
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator():
        # LassoCV / ElasticNetCV with every row pass on the grid
        # (x_linear/enetcv_fast.mojo, lane/apple-fast-classical); `=0` is the A/B arm
        if algo == ALGO_ENETCV and n > 0 and String(getenv("MOJOLEARN_X_LINEAR_ENETCV_FAST")) != "0":
            if enetcv_fast(ctx, x, n_x, y, n_y, n, d, ip, fp, n_out, res):
                return
    comptime if not X_LINEAR_SERIAL_FOLDS:
        if algo == ALGO_LOGCV and n > 0 and String(getenv("MOJOLEARN_X_LINEAR_LOGCV_GRID")) != "0":
            logcv_fit_grid(ctx, algo, x, n_x, y, n_y, n, d, ip, fp, n_out, n_fw, n_iw, res)
            return
        if algo == ALGO_HUBER and n > 0 and String(getenv("MOJOLEARN_X_LINEAR_HUBER_GRID")) != "0":
            huber_fit_grid(ctx, x, n_x, y, n_y, n, d, ip, fp, n_out, n_fw, n_iw, res)
            return
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
    if algo == ALGO_ENETCV and d > 0 and len(ip) >= 7 and _enetcv_grid():
        enetcv_fit_grid(ctx, x, n_x, y, n_y, n, d, ip, fp, n_out, res)
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
    # Ridge: the moments of [X | Y] on the grid (lane/neural-pass120);
    # ip[4] tells the team they are in fw
    var ridge_pre = False
    comptime if MOMENTS_GRID:
        ridge_pre = (algo == ALGO_RIDGE and d > 0 and n > 0 and len(ip) >= 4 and Int(ip[3]) == 0
                     and String(getenv("MOJOLEARN_X_LINEAR_MOMENTS_GRID")) != "0")
    if algo == ALGO_RIDGE:
        while len(hip) < 5:
            hip.append(Int32(0))
        hip[4] = Int32(1 if ridge_pre else 0)
    # lane/neural-pass87 (2026-10-01): BayesianRidge (unweighted) and ARD read
    # the same layout (xm at 0, G at d, ip[1] fit_intercept) and the same
    # centered Gram chains, which the team ran on ONE block (24,310 chains
    # of every row at 220 features over 256 threads).
    var bayes_like = (algo == ALGO_BAYES and len(ip) > 2 and ip[2] == 0) or algo == ALGO_ARD
    if bayes_like and _bayes_grid_gram() and d > 0:
        grid_gram = True
    var lars_pre = False
    comptime if MOMENTS_GRID:
        lars_pre = (algo == ALGO_LARS and grid_gram and n > 0
                    and String(getenv("MOJOLEARN_X_LINEAR_MOMENTS_GRID")) != "0")
    if algo == ALGO_LARS or bayes_like:
        while len(hip) < 5:
            hip.append(Int32(0))
        hip[4] = Int32(2 if lars_pre else (1 if grid_gram else 0))
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator():
        # x_linear/bayes.mojo `X_LINEAR_GRAM_SSE`: ip[5], the sse from the
        # normal equations (lane/apple-fast-classical); `=0` is the A/B arm
        if bayes_like:
            while len(hip) < 6:
                hip.append(Int32(0))
            hip[5] = Int32(0 if String(getenv("MOJOLEARN_X_LINEAR_GRAM_SSE")) == "0" else 1)
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
    if lars_pre:
        # lane/neural-pass120's moments of [X | y] (fw: xm 0, G d, X'y
        # d + d*d, y's mean parked in prev = 2d + d*d)
        ctx.enqueue_function[mg_means_kernel](
            dx.unsafe_ptr(), dy.unsafe_ptr(), Int32(n), Int32(d), Int32(1), Int32(hip[1]), dfw.unsafe_ptr(),
            Int32(0), Int32(2 * d + d * d), grid_dim=mg_tiles(d, 1), block_dim=MG_NT,
        )
        var tl = mg_tiles(d, 1)
        ctx.enqueue_function[mg_cross_kernel](
            dx.unsafe_ptr(), dy.unsafe_ptr(), Int32(n), Int32(d), Int32(1), dfw.unsafe_ptr(),
            Int32(0), Int32(2 * d + d * d), Int32(d), Int32(d + d * d), grid_dim=tl * (tl + 1) // 2, block_dim=MG_NT,
        )
    elif grid_gram and bayes_like and MOMENTS_GRID and String(getenv("MOJOLEARN_X_LINEAR_MOMENTS_GRID")) != "0":
        # lane/neural-pass130: BayesianRidge / ARD's means and centered Gram
        # from the staged moments kernels (no Y columns), into the layout
        # `xg_gram_kernel` fills (xm at 0, G at d): the same chains, staged
        var tlb = mg_tiles(d, 0)
        ctx.enqueue_function[mg_means_kernel](
            dx.unsafe_ptr(), dy.unsafe_ptr(), Int32(n), Int32(d), Int32(0), Int32(hip[1]), dfw.unsafe_ptr(),
            Int32(0), Int32(0), grid_dim=tlb, block_dim=MG_NT,
        )
        ctx.enqueue_function[mg_cross_kernel](
            dx.unsafe_ptr(), dy.unsafe_ptr(), Int32(n), Int32(d), Int32(0), dfw.unsafe_ptr(),
            Int32(0), Int32(0), Int32(d), Int32(0), grid_dim=tlb * (tlb + 1) // 2, block_dim=MG_NT,
        )
    elif grid_gram:
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
    if ridge_pre:
        var t_n = Int(hip[0])
        var r_xm = 0
        var r_gg = d
        var r_ym = d + 2 * d * d + d
        var r_xty = r_ym + t_n
        ctx.enqueue_function[mg_means_kernel](
            dx.unsafe_ptr(), dy.unsafe_ptr(), Int32(n), Int32(d), Int32(t_n), Int32(hip[1]), dfw.unsafe_ptr(),
            Int32(r_xm), Int32(r_ym), grid_dim=mg_tiles(d, t_n), block_dim=MG_NT,
        )
        var tl = mg_tiles(d, t_n)
        ctx.enqueue_function[mg_cross_kernel](
            dx.unsafe_ptr(), dy.unsafe_ptr(), Int32(n), Int32(d), Int32(t_n), dfw.unsafe_ptr(),
            Int32(r_xm), Int32(r_ym), Int32(r_gg), Int32(r_xty), grid_dim=tl * (tl + 1) // 2, block_dim=MG_NT,
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
    if algo == ALGO_RIDGE:
        # lane/neural-pass93: the float-float refit when the float32 factor
        # was not trusted (x_linear/ridge.mojo, status 1), on the grid
        var t_n = Int(ip[0])
        var a_n = Int(ip[2])
        var sidx = t_n * d + t_n + 2 + a_n
        if n_out > sidx and res.unsafe_load(sidx) == Float32(1):
            _ridge_ff_grid(ctx, FP(unsafe_from_address=Int(dx.unsafe_ptr())), FP(unsafe_from_address=Int(dy.unsafe_ptr())), n, d, t_n, Int(ip[1]) != 0, len(ip) > 3 and Int(ip[3]) != 0,
                           res.unsafe_load(t_n * d + t_n), res, sidx)
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
