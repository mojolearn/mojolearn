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
from x_linear.dispatch import fit_dispatch, decision_one, team_fit, team_rows, team_own, ALGO_SGD, ALGO_LARS, ALGO_ISOTONIC, ALGO_ISOTONIC_PREDICT
from x_linear.isotonic import iso_predict_one, iso_gather_one, iso_bounds, iso_group, iso_after_unique
from std.memory import bitcast
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


def iso_keys_kernel(x: FP, y: FP, kx: IP, ky: IP, perm: IP, n: Int32):
    var i = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    if i < Int(n):
        kx.unsafe_store(i, bitcast[DType.int32](_iso_key(x.unsafe_load(i))))
        ky.unsafe_store(i, bitcast[DType.int32](_iso_key(y.unsafe_load(i))))
        perm.unsafe_store(i, Int32(i))


@always_inline
def _digit(keys: IP, row: Int, shift: Int) -> Int:
    return Int((bitcast[DType.uint32](keys.unsafe_load(row)) >> UInt32(shift)) & UInt32(ISO_RADIX - 1))


def iso_count_kernel(keys: IP, perm: IP, n: Int32, shift: Int32, counts: IP, ntiles: Int32):
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


def iso_scan_kernel(counts: IP, total: Int32):
    """counts (digit-major, tile-minor) -> their exclusive prefix, in place: one thread."""
    if Int(block_idx.x) == 0 and Int(thread_idx.x) == 0:
        var acc = Int32(0)
        for i in range(Int(total)):
            var c = counts.unsafe_load(i)
            counts.unsafe_store(i, acc)
            acc += c


def iso_scatter_kernel(keys: IP, src: IP, dst: IP, n: Int32, shift: Int32, offs: IP, ntiles: Int32):
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


def iso_gather_kernel(x: FP, y: FP, n: Int32, perm: IP, fw: FP):
    var j = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    var nn = Int(n)
    if j < nn:
        iso_gather_one(j, x, y, nn, False, perm, fw + 3 * nn, fw + 4 * nn, fw + 5 * nn)


def iso_bounds_kernel(fw: FP, n: Int32, iw: IP, mslot: IP):
    if Int(block_idx.x) == 0 and Int(thread_idx.x) == 0:
        var nn = Int(n)
        mslot.unsafe_store(0, Int32(iso_bounds(fw + 3 * nn, nn, iw + nn)))


def iso_group_kernel(fw: FP, n: Int32, iw: IP, mslot: IP):
    var g = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    var nn = Int(n)
    if g < Int(mslot.unsafe_load(0)):
        iso_group(g, fw + 3 * nn, fw + 4 * nn, fw + 5 * nn, iw + nn, fw, nn)


def iso_after_kernel(n: Int32, ip: IP, fp: FP, res: FP, fw: FP, iw: IP, mslot: IP):
    if Int(block_idx.x) == 0 and Int(thread_idx.x) == 0:
        iso_after_unique(Int(mslot.unsafe_load(0)), Int(n), ip, fp, res, fw, iw)


def iso_predict_kernel(x: FP, thr: FP, n: Int32, m: Int32, oob: Int32, fp: FP, res: FP):
    var q = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    if q < Int(n):
        iso_predict_one(q, x, thr, Int(m), Int(oob), fp, res)


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
    dout.enqueue_fill(Float32(0))
    dfw.enqueue_fill(Float32(0))
    diw.enqueue_fill(Int32(0))
    ctx.enqueue_function[iso_keys_kernel](dx.unsafe_ptr(), dy.unsafe_ptr(), dkx.unsafe_ptr(), dky.unsafe_ptr(),
                                          dpa.unsafe_ptr(), Int32(n), grid_dim=_xg_blocks(n), block_dim=XG_TPB)
    var cur_a = True
    for key in range(2):
        for pas in range(4):
            var shift = Int32(8 * pas)
            var kp = IP(unsafe_from_address=Int(dky.unsafe_ptr()) if key == 0 else Int(dkx.unsafe_ptr()))
            var src = IP(unsafe_from_address=Int(dpa.unsafe_ptr()) if cur_a else Int(dpb.unsafe_ptr()))
            var dst = IP(unsafe_from_address=Int(dpb.unsafe_ptr()) if cur_a else Int(dpa.unsafe_ptr()))
            ctx.enqueue_function[iso_count_kernel](kp, src, Int32(n), shift, dcnt.unsafe_ptr(), Int32(ntiles),
                                                   grid_dim=_xg_blocks(ntiles), block_dim=XG_TPB)
            ctx.enqueue_function[iso_scan_kernel](dcnt.unsafe_ptr(), Int32(ISO_RADIX * ntiles), grid_dim=1, block_dim=1)
            ctx.enqueue_function[iso_scatter_kernel](kp, src, dst, Int32(n), shift, dcnt.unsafe_ptr(), Int32(ntiles),
                                                     grid_dim=_xg_blocks(ntiles), block_dim=XG_TPB)
            cur_a = not cur_a
    var pp = IP(unsafe_from_address=Int(dpa.unsafe_ptr()) if cur_a else Int(dpb.unsafe_ptr()))
    var dm = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.enqueue_function[iso_gather_kernel](dx.unsafe_ptr(), dy.unsafe_ptr(), Int32(n), pp, dfw.unsafe_ptr(),
                                            grid_dim=_xg_blocks(n), block_dim=XG_TPB)
    ctx.enqueue_function[iso_bounds_kernel](dfw.unsafe_ptr(), Int32(n), diw.unsafe_ptr(), dm.unsafe_ptr(), grid_dim=1, block_dim=1)
    ctx.enqueue_function[iso_group_kernel](dfw.unsafe_ptr(), Int32(n), diw.unsafe_ptr(), dm.unsafe_ptr(),
                                           grid_dim=_xg_blocks(n), block_dim=XG_TPB)
    ctx.enqueue_function[iso_after_kernel](Int32(n), dip.unsafe_ptr(), dfp.unsafe_ptr(), dout.unsafe_ptr(), dfw.unsafe_ptr(),
                                           diw.unsafe_ptr(), dm.unsafe_ptr(), grid_dim=1, block_dim=1)
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
    _ = dkx^
    _ = dky^
    _ = dpa^
    _ = dpb^
    _ = dcnt^
    _ = dm^


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
    ctx.enqueue_function[iso_predict_kernel](dx.unsafe_ptr(), dt.unsafe_ptr(), Int32(n), ip[0], ip[1], dfp.unsafe_ptr(),
                                             dout.unsafe_ptr(), grid_dim=_xg_blocks(n), block_dim=XG_TPB)
    ctx.enqueue_copy(dst_ptr=res, src_buf=dout)
    ctx.synchronize()
    _ = hfp^
    _ = dx^
    _ = dt^
    _ = dfp^
    _ = dout^


def fit_device(
    algo: Int, x: FP, n_x: Int, y: FP, n_y: Int, n: Int, d: Int,
    ip: List[Int32], fp: List[Float32], n_out: Int, n_fw: Int, n_iw: Int, res: FP,
) raises:
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
