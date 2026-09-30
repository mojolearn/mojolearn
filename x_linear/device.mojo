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
from x_linear.ops import ld, st, fd, i2f
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


def fit_device(
    algo: Int, x: FP, n_x: Int, y: FP, n_y: Int, n: Int, d: Int,
    ip: List[Int32], fp: List[Float32], n_out: Int, n_fw: Int, n_iw: Int, res: FP,
) raises:
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
