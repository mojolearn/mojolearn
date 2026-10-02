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
from x_linear.dispatch import fit_dispatch, decision_one, team_fit, team_rows, team_own, ALGO_SGD, ALGO_LARS, ALGO_RIDGE_KFOLD, ALGO_RIDGE, ALGO_BAYES, ALGO_ARD
from x_linear.ops import ld, st, fd, i2f, fa
from x_linear.ridge import ridge_ff_unit, ridge_ff_units, ridge_ff_solve
from x_linear.witness import Witness, witness_end, WITNESS_TRIES
from x_linear.ridgecv import kf_start, kf_end, kf_mean, kf_cross, kf_solve, kf_pred, kf_score, kf_ff_solve
from x_linear.tops import upper_cell, fold_fa, chain_cfmad
from std.os import getenv
from x_linear.team import LINEAR_TPB, team_work, device_team, solo


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
                         sh: FP, sl: FP):
    var u = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    if u < Int(count):
        ridge_ff_unit(Int(u0) + u, x, y, Int(n), Int(d), Int(t_n), fi != 0, sw != 0, Int(n) * Int(t_n), sh, sl)


def ridge_ff_solve_kernel(d: Int32, t_n: Int32, fi: Int32, alpha: Float32, sh: FP, sl: FP, bh: FP, bl: FP, dst: FP,
                          fh: FP, fl: FP):
    """dst: coef T*d | intercept T | ok (1 / 0)."""
    var ok = ridge_ff_solve(Int(d), Int(t_n), fi != 0, alpha, sh, sl, bh, bl, dst, fh, fl)
    st(dst, Int(t_n) * Int(d) + Int(t_n), Float32(1) if ok else Float32(0))


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
    dsh.enqueue_fill(Float32(0))
    dsl.enqueue_fill(Float32(0))
    ctx.enqueue_function[ridge_ff_unit_kernel](x, y, Int32(n), Int32(d), Int32(t_n), Int32(1 if fi else 0), Int32(1 if sw else 0),
                                               Int32(0), Int32(nm), dsh.unsafe_ptr(), dsl.unsafe_ptr(),
                                               grid_dim=_xg_blocks(nm), block_dim=XG_TPB)
    ctx.enqueue_function[ridge_ff_unit_kernel](x, y, Int32(n), Int32(d), Int32(t_n), Int32(1 if fi else 0), Int32(1 if sw else 0),
                                               Int32(nm), Int32(units - nm), dsh.unsafe_ptr(), dsl.unsafe_ptr(),
                                               grid_dim=_xg_blocks(units - nm), block_dim=XG_TPB)
    ctx.enqueue_function[ridge_ff_solve_kernel](Int32(d), Int32(t_n), Int32(1 if fi else 0), alpha, dsh.unsafe_ptr(),
                                                dsl.unsafe_ptr(), dbh.unsafe_ptr(), dbl.unsafe_ptr(), dout.unsafe_ptr(),
                                                dfh.unsafe_ptr(), dfl.unsafe_ptr(), grid_dim=1, block_dim=1)
    var h = List[Float32](length=t_n * d + t_n + 1, fill=Float32(0))
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=dout)
    ctx.synchronize()
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


def fit_device(
    algo: Int, x: FP, n_x: Int, y: FP, n_y: Int, n: Int, d: Int,
    ip: List[Int32], fp: List[Float32], n_out: Int, n_fw: Int, n_iw: Int, res: FP,
) raises:
    if algo == ALGO_RIDGE_KFOLD:
        _ridge_kfold_grid(x, n_x, y, n_y, n, d, ip, fp, res)
        return
    if algo == ALGO_SGD and _sgd_on_host():
        _fit_on_host(algo, x, y, n, d, ip, fp, n_out, n_fw, n_iw, res)
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
