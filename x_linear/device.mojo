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
from x_linear.dispatch import fit_dispatch, decision_one, team_fit, team_rows, team_own, ALGO_SGD, ALGO_LARS
from x_linear.ops import ld, st, fd, i2f, fa, fs, fm, fabs, shuffle
from x_linear.witness import Witness, witness_end, WITNESS_TRIES
from x_linear.sgd import sgd_mb_on, mb_row, mb_part, mb_step, mb_bias_step, mb_subs, mb_eta, mb_optimal_init, mb_penalty, LR_OPTIMAL, LR_ADAPTIVE, P_L2, P_L1
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
                       dlv: FP, lv: FP, lr: Int32, eta0: Float32):
    var r = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    if r < Int(bs):
        var i = Int(idx.unsafe_load(Int(start) + r))
        var o = mb_row(x, ys, i, Int(d), w, 0, ld(bias, 0), Int(loss), eps, swp, has_sw != 0, wpos, wneg, has_cw != 0,
                       Int(lr), eta0)
        st(dlv, r, o[0])
        st(lv, r, o[1])


def sgd_mb_rows_kernel(x: FP, ys: FP, idx: IP, start: Int32, bs: Int32, d: Int32, w: FP, bias: FP, loss: Int32,
                       eps: Float32, swp: FP, has_sw: Int32, wpos: Float32, wneg: Float32, has_cw: Int32,
                       dlv: FP, lv: FP, lr: Int32, eta0: Float32, wf: IP, woff: Int32, nonce: Int32):
    _sgd_mb_rows_kernel_body(x, ys, idx, start, bs, d, w, bias, loss, eps, swp, has_sw, wpos, wneg, has_cw, dlv, lv, lr, eta0)
    witness_end(wf, woff, nonce)

@always_inline
def _sgd_mb_parts_kernel_body(x: FP, d: Int32, idx: IP, start: Int32, dlv: FP, lv: FP, bs: Int32, nsub: Int32, parts: FP):
    var q = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    var dd = Int(d)
    var subs = mb_subs(Int(bs))
    var s = q // (dd + 2)
    var j = q - s * (dd + 2)
    if s < subs:
        st(parts, j * Int(nsub) + s, mb_part(x, dd, idx, Int(start), dlv, lv, j, s, Int(bs)))


def sgd_mb_parts_kernel(x: FP, d: Int32, idx: IP, start: Int32, dlv: FP, lv: FP, bs: Int32, nsub: Int32, parts: FP, wf: IP, woff: Int32, nonce: Int32):
    _sgd_mb_parts_kernel_body(x, d, idx, start, dlv, lv, bs, nsub, parts)
    witness_end(wf, woff, nonce)

@always_inline
def _sgd_mb_step_kernel_body(parts: FP, nsub: Int32, bs: Int32, d: Int32, w: FP, bias: FP, obj: FP, eta: Float32,
                       alpha: Float32, l1r: Float32, penalty: Int32, fi: Int32, need_obj: Int32, one_class: Int32,
                       bsum: Int32):
    var j = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    var dd = Int(d)
    var subs = mb_subs(Int(bs))
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
                       bsum: Int32, wf: IP, woff: Int32, nonce: Int32):
    _sgd_mb_step_kernel_body(parts, nsub, bs, d, w, bias, obj, eta, alpha, l1r, penalty, fi, need_obj, one_class, bsum)
    witness_end(wf, woff, nonce)

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
    var nsub = mb_subs(batch)
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
        wcap += _xg_blocks(wbs) + _xg_blocks((d + 2) * mb_subs(wbs)) + _xg_blocks(d + 2)
        ws0 += wbs
    var wit = Witness(ctx, max(wcap, 1))
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
                while start < n:
                    var bs = min(batch, n - start)
                    var et = mb_eta(lr, eta, eta0, alpha, power_t, opt_init, t)
                    ctx.enqueue_function[sgd_mb_rows_kernel](
                        dx.unsafe_ptr(), dys.unsafe_ptr(), didx.unsafe_ptr(), Int32(start), Int32(bs), Int32(d),
                        dw.unsafe_ptr(), dbias.unsafe_ptr(), Int32(loss), eps, dsw.unsafe_ptr(), Int32(1 if has_sw else 0),
                        wpos, wneg, Int32(1 if has_cw else 0), ddl.unsafe_ptr(), dlv.unsafe_ptr(), Int32(lr), eta0,
                        wit.p(), Int32(wo), nonce, grid_dim=_xg_blocks(bs), block_dim=XG_TPB,
                    )
                    wo += _xg_blocks(bs)
                    ctx.enqueue_function[sgd_mb_parts_kernel](
                        dx.unsafe_ptr(), Int32(d), didx.unsafe_ptr(), Int32(start), ddl.unsafe_ptr(), dlv.unsafe_ptr(),
                        Int32(bs), Int32(nsub), dparts.unsafe_ptr(), wit.p(), Int32(wo), nonce,
                        grid_dim=_xg_blocks((d + 2) * mb_subs(bs)), block_dim=XG_TPB,
                    )
                    wo += _xg_blocks((d + 2) * mb_subs(bs))
                    ctx.enqueue_function[sgd_mb_step_kernel](
                        dparts.unsafe_ptr(), Int32(nsub), Int32(bs), Int32(d), dw.unsafe_ptr(), dbias.unsafe_ptr(),
                        dobj.unsafe_ptr(), et, alpha, l1r, Int32(penalty), Int32(1 if fi else 0), Int32(1 if need_obj else 0),
                        Int32(1 if one_class else 0), Int32(1 if bsum else 0), wit.p(), Int32(wo), nonce,
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


def fit_device(
    algo: Int, x: FP, n_x: Int, y: FP, n_y: Int, n: Int, d: Int,
    ip: List[Int32], fp: List[Float32], n_out: Int, n_fw: Int, n_iw: Int, res: FP,
) raises:
    if algo == ALGO_SGD and len(ip) > 12 and sgd_mb_on(Int(ip[12]), Int(ip[0]), Int(ip[3])):
        _sgd_mb_grid(x, n_x, y, n_y, n, d, ip, fp, n_out, res)
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
    if algo == ALGO_LARS:
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
