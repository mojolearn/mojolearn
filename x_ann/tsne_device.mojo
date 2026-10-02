# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""t-SNE on the device: one thread per cell of `x_ann/tsne_core.mojo`; the
symmetrization is the shared host function (`tsne_symmetrize`)."""

from std.gpu import block_idx, block_dim, thread_idx
from std.memory import stack_allocation
from std.time import perf_counter_ns
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from max.gpu.host import DeviceBuffer, DeviceContext
from x_ann.device_ctx import x_ann_ctx
from x_ann.stage_timer import AnnStages
from x_ann.knn_device import knn_enqueue

from metrics.checks.device_io import upload_f32, upload_i32, download_f32, download_i32
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, ftz, identical_log
from std.sys.info import has_apple_gpu_accelerator
from x_ann.switches import ANN3_TSNE_RB32, ANN3_TSNE_RB64, ANN3_TSNE_STEP_ROWS, ANN3_TSNE_ZSUM
from x_ann.fast_env import tsne_fast_split, tsne_fast_zsum
from checks.numerics import identical_div, identical_mul
from x_ann.tsne_core import ts_q
from x_ann.tsne_core import (
    F32P, I32P, ts_kl_cell, ts_perplexity_cell, ts_repulse_fold, ts_repulse_pair, ts_repulse_terms, ts_step_cell,
    ts_sum_cell, tsne_nn, tsne_symmetrize, tsne_validate,
)

comptime TPB = 128


def _tid() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


def _grid(count: Int) -> Int:
    return (count + TPB - 1) // TPB


def perplexity_kernel(n: Int32, nn_d: F32P, nn: Int32, log_perp: Float32, p: F32P):
    var i = _tid()
    if i < Int(n):
        ts_perplexity_cell(i, nn_d, Int(nn), log_perp, p)


comptime _FAST_APPLE = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
#: rows per threadgroup and staged candidate rows per tile of the repulsion
#: (lane ann-apple3, OPT-IN trials under FAST on Apple: 32 or 64 rows per
#: threadgroup, x_ann/switches.mojo)
comptime RTB = 32 if (_FAST_APPLE and ANN3_TSNE_RB32) else (64 if (_FAST_APPLE and ANN3_TSNE_RB64) else 128)
comptime RTJ = 256


def repulse_tiled_kernel(n: Int32, y: F32P, row_z: F32P, rep: F32P):
    """`ts_repulse_cell` with the candidate rows staged in threadgroup memory
    (lane ann-apple): tiles and rows ascending, so row i folds j = 0, 1, ...,
    n - 1 in the cell's order, each j through the cell's own
    `ts_repulse_pair` on ftz(y) (the staged values are flushed once; ftz is
    idempotent). The same bits."""
    var t = Int(thread_idx.x)
    var nr = Int(n)
    var i = Int(block_idx.x) * RTB + t
    var tile = stack_allocation[2 * RTJ, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var live = i < nr
    var y0 = Float32(0.0)
    var y1 = Float32(0.0)
    if live:
        y0 = ftz(y.unsafe_load(2 * i))
        y1 = ftz(y.unsafe_load(2 * i + 1))
    var z = Float32(0.0)
    var r0 = Float32(0.0)
    var r1 = Float32(0.0)
    var j0 = 0
    while j0 < nr:
        for e in range(t, 2 * RTJ, RTB):
            var v = Float32(0.0)
            if 2 * j0 + e < 2 * nr:
                v = ftz(y.unsafe_load(2 * j0 + e))
            tile[e] = v
        barrier()
        if live:
            var jn = RTJ if nr - j0 > RTJ else nr - j0
            # four j at a time: their terms are independent, so they are
            # formed first and folded after in ascending j (the same
            # statements in the same fold order; only the independent work
            # overlaps). The group holding row i itself takes the plain loop.
            var r = 0
            while r + 4 <= jn:
                if i >= j0 + r and i < j0 + r + 4:
                    for u in range(4):
                        if j0 + r + u != i:
                            ts_repulse_pair(y0, y1, tile[2 * (r + u)], tile[2 * (r + u) + 1], z, r0, r1)
                else:
                    var ta = ts_repulse_terms(y0, y1, tile[2 * r], tile[2 * r + 1])
                    var tb = ts_repulse_terms(y0, y1, tile[2 * r + 2], tile[2 * r + 3])
                    var tc = ts_repulse_terms(y0, y1, tile[2 * r + 4], tile[2 * r + 5])
                    var td = ts_repulse_terms(y0, y1, tile[2 * r + 6], tile[2 * r + 7])
                    ts_repulse_fold(ta, z, r0, r1)
                    ts_repulse_fold(tb, z, r0, r1)
                    ts_repulse_fold(tc, z, r0, r1)
                    ts_repulse_fold(td, z, r0, r1)
                r += 4
            while r < jn:
                if j0 + r != i:
                    ts_repulse_pair(y0, y1, tile[2 * r], tile[2 * r + 1], z, r0, r1)
                r += 1
        barrier()
        j0 += RTJ
    if live:
        row_z.unsafe_store(i, z)
        rep.unsafe_store(2 * i, r0)
        rep.unsafe_store(2 * i + 1, r1)


def sum_kernel(n: Int32, row_z: F32P, z: F32P):
    if _tid() == 0:
        ts_sum_cell(row_z, Int(n), z)


#: threads of the FAST Z sum
comptime ZT = 128
comptime TS_ZSUM = _FAST_APPLE and ANN3_TSNE_ZSUM
comptime TS_STEP_ROWS = _FAST_APPLE and ANN3_TSNE_STEP_ROWS


def sum_team_kernel(n: Int32, row_z: F32P, z: F32P):
    """FAST on Apple, OPT-IN (lane ann-apple3): Z from ONE threadgroup of ZT
    threads. Thread t adds rows t, t + ZT, ...; the ZT sums are joined by a
    halving tree in threadgroup memory; thread 0 stores Z."""
    var t = Int(thread_idx.x)
    var part = stack_allocation[ZT, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var acc = Float32(0.0)
    for i in range(t, Int(n), ZT):
        acc = acc + row_z.unsafe_load(i)
    part[t] = acc
    barrier()
    var step = ZT // 2
    while step > 0:
        var v = Float32(0.0)
        if t < step:
            v = part[t] + part[t + step]
        barrier()
        if t < step:
            part[t] = v
        barrier()
        step = step // 2
    if t == 0:
        z.unsafe_store(0, part[0])


#: stripes of candidate rows per row in the split repulsion
comptime TS_SPLIT = 8


def repulse_split_kernel(n: Int32, y: F32P, part: F32P):
    """FAST on Apple, `MOJOLEARN_TSNE_FAST_SPLIT=1` (lane/apple-fast-ann,
    2026-10-02): `repulse_tiled_kernel` with each row's candidate rows j
    split into TS_SPLIT stripes, one threadgroup per (row block, stripe):
    threadgroup (b, s) folds j in [s n / TS_SPLIT, (s + 1) n / TS_SPLIT) for
    rows b RTB .. b RTB + RTB - 1 (tiles of RTJ staged rows, the cell's
    `ts_repulse_pair` on ftz(y)) and stores its three partial sums at
    part[(s n + i) 3 ..]; `repulse_join_kernel` adds the stripes in order.
    Cause: `_ts_iter`'s repulsion ran n / RTB threadgroups (157 at the
    board's 20,000 rows), each thread walking every row, for all 1000
    iterations, so the GPU was mostly idle; this runs TS_SPLIT times as many
    threadgroups over the same pairs. FAST bits move (a fold per stripe,
    then across stripes): paired trustworthiness / KL check."""
    var t = Int(thread_idx.x)
    var nr = Int(n)
    var s = Int(block_idx.y)
    var i = Int(block_idx.x) * RTB + t
    var j_lo = (s * nr) // TS_SPLIT
    var j_hi = ((s + 1) * nr) // TS_SPLIT
    var tile = stack_allocation[2 * RTJ, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var live = i < nr
    var y0 = Float32(0.0)
    var y1 = Float32(0.0)
    if live:
        y0 = ftz(y.unsafe_load(2 * i))
        y1 = ftz(y.unsafe_load(2 * i + 1))
    var z = Float32(0.0)
    var r0 = Float32(0.0)
    var r1 = Float32(0.0)
    var j0 = j_lo
    while j0 < j_hi:
        var jn = RTJ if j_hi - j0 > RTJ else j_hi - j0
        for e in range(t, 2 * RTJ, RTB):
            var v = Float32(0.0)
            if e < 2 * jn:
                v = ftz(y.unsafe_load(2 * j0 + e))
            tile[e] = v
        barrier()
        if live:
            var r = 0
            while r + 4 <= jn:
                if i >= j0 + r and i < j0 + r + 4:
                    for u in range(4):
                        if j0 + r + u != i:
                            ts_repulse_pair(y0, y1, tile[2 * (r + u)], tile[2 * (r + u) + 1], z, r0, r1)
                else:
                    var ta = ts_repulse_terms(y0, y1, tile[2 * r], tile[2 * r + 1])
                    var tb = ts_repulse_terms(y0, y1, tile[2 * r + 2], tile[2 * r + 3])
                    var tc = ts_repulse_terms(y0, y1, tile[2 * r + 4], tile[2 * r + 5])
                    var td = ts_repulse_terms(y0, y1, tile[2 * r + 6], tile[2 * r + 7])
                    ts_repulse_fold(ta, z, r0, r1)
                    ts_repulse_fold(tb, z, r0, r1)
                    ts_repulse_fold(tc, z, r0, r1)
                    ts_repulse_fold(td, z, r0, r1)
                r += 4
            while r < jn:
                if j0 + r != i:
                    ts_repulse_pair(y0, y1, tile[2 * r], tile[2 * r + 1], z, r0, r1)
                r += 1
        barrier()
        j0 += RTJ
    if live:
        var o = (s * nr + i) * 3
        part.unsafe_store(o, z)
        part.unsafe_store(o + 1, r0)
        part.unsafe_store(o + 2, r1)


def repulse_join_kernel(n: Int32, part: F32P, row_z: F32P, rep: F32P):
    """Row i's z, r0, r1 as the sum of its TS_SPLIT stripe partials in
    stripe order (the split repulsion's second launch)."""
    var i = _tid()
    if i < Int(n):
        var nr = Int(n)
        var z = Float32(0.0)
        var r0 = Float32(0.0)
        var r1 = Float32(0.0)
        for s in range(TS_SPLIT):
            var o = (s * nr + i) * 3
            z = z + part.unsafe_load(o)
            r0 = ftz(r0 + part.unsafe_load(o + 1))
            r1 = ftz(r1 + part.unsafe_load(o + 2))
        row_z.unsafe_store(i, z)
        rep.unsafe_store(2 * i, r0)
        rep.unsafe_store(2 * i + 1, r1)


def _ts_repulse(
    ctx: DeviceContext, mut ycur: DeviceBuffer[DType.float32], n: Int, mut drz: DeviceBuffer[DType.float32],
    mut drep: DeviceBuffer[DType.float32], split: Bool, mut dpart: DeviceBuffer[DType.float32],
) raises:
    """The repulsion launch: split over stripes under `split` (FAST on
    Apple, lane/apple-fast-ann), the tiled kernel otherwise."""
    comptime if _FAST_APPLE:
        if split:
            ctx.enqueue_function[repulse_split_kernel](
                Int32(n), ycur.unsafe_ptr(), dpart.unsafe_ptr(), grid_dim=((n + RTB - 1) // RTB, TS_SPLIT),
                block_dim=RTB,
            )
            ctx.enqueue_function[repulse_join_kernel](
                Int32(n), dpart.unsafe_ptr(), drz.unsafe_ptr(), drep.unsafe_ptr(), grid_dim=_grid(n), block_dim=TPB,
            )
            return
    ctx.enqueue_function[repulse_tiled_kernel](Int32(n), ycur.unsafe_ptr(), drz.unsafe_ptr(), drep.unsafe_ptr(),
                                               grid_dim=(n + RTB - 1) // RTB, block_dim=RTB)


def _ts_zsum(ctx: DeviceContext, n: Int, mut drz: DeviceBuffer[DType.float32], mut dz: DeviceBuffer[DType.float32],
             zteam: Bool) raises:
    """The Z launch: `sum_team_kernel` under `zteam` (the ann-apple3 define
    or `MOJOLEARN_TSNE_FAST_ZSUM=1`, lane/apple-fast-ann, 2026-10-02; cause:
    `sum_kernel` is ONE thread adding the n row sums, 20,000 dependent adds
    per iteration for 1000 iterations), the one-thread sum otherwise. FAST
    bits move (the order of a float sum): paired trustworthiness / KL check."""
    comptime if _FAST_APPLE:
        if zteam:
            ctx.enqueue_function[sum_team_kernel](Int32(n), drz.unsafe_ptr(), dz.unsafe_ptr(), grid_dim=1, block_dim=ZT)
            return
    ctx.enqueue_function[sum_kernel](Int32(n), drz.unsafe_ptr(), dz.unsafe_ptr(), grid_dim=1, block_dim=1)


@always_inline
def _step_tail(
    e: Int, yi: Float32, attr: Float32, y_new: F32P, rep: F32P, z: F32P, update: F32P, gains: F32P,
    exaggeration: Float32, momentum: Float32, learning_rate: Float32,
):
    """`ts_step_cell` after its attraction sum, statement for statement."""
    var neg = ftz(identical_div(rep.unsafe_load(e), z.unsafe_load(0)))
    var grad = ftz(identical_mul(Float32(4.0), ftz(ftz(identical_mul(exaggeration, attr)) - neg)))
    var upd = update.unsafe_load(e)
    var gain = gains.unsafe_load(e)
    if ftz(identical_mul(upd, grad)) < Float32(0.0):
        gain = ftz(gain + Float32(0.2))
    else:
        gain = ftz(identical_mul(gain, Float32(0.8)))
    if gain < Float32(0.01):
        gain = Float32(0.01)
    grad = ftz(identical_mul(grad, gain))
    upd = ftz(ftz(identical_mul(momentum, upd)) - ftz(identical_mul(learning_rate, grad)))
    gains.unsafe_store(e, gain)
    update.unsafe_store(e, upd)
    y_new.unsafe_store(e, ftz(yi + upd))


def step_rows_kernel(
    n: Int32, y: F32P, y_new: F32P, indptr: I32P, indices: I32P, values: F32P, rep: F32P,
    z: F32P, update: F32P, gains: F32P, exaggeration: Float32, momentum: Float32, lr: Float32,
):
    """FAST on Apple, OPT-IN (lane ann-apple3): `ts_step_cell` for both
    coordinates of row i in one thread. The attraction walks the row's CSR
    entries once and forms each neighbor's `ts_q` once (the cell forms it once
    per coordinate); each coordinate's sum and update are the cell's
    statements in the cell's order."""
    var i = _tid()
    if i < Int(n):
        var y0 = ftz(y.unsafe_load(2 * i))
        var y1 = ftz(y.unsafe_load(2 * i + 1))
        var a0 = Float32(0.0)
        var a1 = Float32(0.0)
        for s in range(Int(indptr.unsafe_load(i)), Int(indptr.unsafe_load(i + 1))):
            var j = Int(indices.unsafe_load(s))
            var pq = ftz(identical_mul(ftz(values.unsafe_load(s)), ts_q(y, i, j)))
            a0 = ftz(a0 + ftz(identical_mul(pq, ftz(y0 - ftz(y.unsafe_load(2 * j))))))
            a1 = ftz(a1 + ftz(identical_mul(pq, ftz(y1 - ftz(y.unsafe_load(2 * j + 1))))))
        _step_tail(2 * i, y0, a0, y_new, rep, z, update, gains, exaggeration, momentum, lr)
        _step_tail(2 * i + 1, y1, a1, y_new, rep, z, update, gains, exaggeration, momentum, lr)


def step_kernel(
    count: Int32, y: F32P, y_new: F32P, indptr: I32P, indices: I32P, values: F32P, rep: F32P,
    z: F32P, update: F32P, gains: F32P, exaggeration: Float32, momentum: Float32, lr: Float32,
):
    var e = _tid()
    if e < Int(count):
        ts_step_cell(e, y, y_new, indptr, indices, values, rep, z, update, gains, exaggeration, momentum, lr)


def kl_kernel(n: Int32, y: F32P, indptr: I32P, indices: I32P, values: F32P, z: F32P, kl: F32P):
    var i = _tid()
    if i < Int(n):
        ts_kl_cell(i, y, indptr, indices, values, z, kl)


def _ts_iter(
    ctx: DeviceContext, mut ycur: DeviceBuffer[DType.float32], mut ynext: DeviceBuffer[DType.float32], n: Int,
    mut dptr: DeviceBuffer[DType.int32], mut dind: DeviceBuffer[DType.int32], mut dval: DeviceBuffer[DType.float32],
    mut drz: DeviceBuffer[DType.float32], mut drep: DeviceBuffer[DType.float32], mut dz: DeviceBuffer[DType.float32],
    mut dupd: DeviceBuffer[DType.float32], mut dgain: DeviceBuffer[DType.float32], ex: Float32, mom: Float32,
    lr: Float32, split: Bool, zteam: Bool, mut dpart: DeviceBuffer[DType.float32],
) raises:
    _ts_repulse(ctx, ycur, n, drz, drep, split, dpart)
    _ts_zsum(ctx, n, drz, dz, zteam)
    comptime if TS_STEP_ROWS:
        ctx.enqueue_function[step_rows_kernel](
            Int32(n), ycur.unsafe_ptr(), ynext.unsafe_ptr(), dptr.unsafe_ptr(), dind.unsafe_ptr(),
            dval.unsafe_ptr(), drep.unsafe_ptr(), dz.unsafe_ptr(), dupd.unsafe_ptr(), dgain.unsafe_ptr(), ex,
            mom, lr, grid_dim=_grid(n), block_dim=TPB,
        )
    else:
        ctx.enqueue_function[step_kernel](
            Int32(2 * n), ycur.unsafe_ptr(), ynext.unsafe_ptr(), dptr.unsafe_ptr(), dind.unsafe_ptr(),
            dval.unsafe_ptr(), drep.unsafe_ptr(), dz.unsafe_ptr(), dupd.unsafe_ptr(), dgain.unsafe_ptr(), ex,
            mom, lr, grid_dim=_grid(2 * n), block_dim=TPB,
        )


def _ts_iter_timed(
    ctx: DeviceContext, mut ycur: DeviceBuffer[DType.float32], mut ynext: DeviceBuffer[DType.float32], n: Int,
    mut dptr: DeviceBuffer[DType.int32], mut dind: DeviceBuffer[DType.int32], mut dval: DeviceBuffer[DType.float32],
    mut drz: DeviceBuffer[DType.float32], mut drep: DeviceBuffer[DType.float32], mut dz: DeviceBuffer[DType.float32],
    mut dupd: DeviceBuffer[DType.float32], mut dgain: DeviceBuffer[DType.float32], ex: Float32, mom: Float32,
    lr: Float32, mut t_rep: Int, mut t_sum: Int, mut t_step: Int, split: Bool, zteam: Bool,
    mut dpart: DeviceBuffer[DType.float32],
) raises:
    """`_ts_iter` for the stage pass only (MOJOLEARN_ANN_STAGES, lane
    ann-apple3): the same three launches, drained one by one, their wall
    times added to t_rep / t_sum / t_step (ns)."""
    var t0 = Int(perf_counter_ns())
    _ts_repulse(ctx, ycur, n, drz, drep, split, dpart)
    ctx.synchronize()
    var t1 = Int(perf_counter_ns())
    _ts_zsum(ctx, n, drz, dz, zteam)
    ctx.synchronize()
    var t2 = Int(perf_counter_ns())
    comptime if TS_STEP_ROWS:
        ctx.enqueue_function[step_rows_kernel](
            Int32(n), ycur.unsafe_ptr(), ynext.unsafe_ptr(), dptr.unsafe_ptr(), dind.unsafe_ptr(),
            dval.unsafe_ptr(), drep.unsafe_ptr(), dz.unsafe_ptr(), dupd.unsafe_ptr(), dgain.unsafe_ptr(), ex,
            mom, lr, grid_dim=_grid(n), block_dim=TPB,
        )
    else:
        ctx.enqueue_function[step_kernel](
            Int32(2 * n), ycur.unsafe_ptr(), ynext.unsafe_ptr(), dptr.unsafe_ptr(), dind.unsafe_ptr(),
            dval.unsafe_ptr(), drep.unsafe_ptr(), dz.unsafe_ptr(), dupd.unsafe_ptr(), dgain.unsafe_ptr(), ex,
            mom, lr, grid_dim=_grid(2 * n), block_dim=TPB,
        )
    ctx.synchronize()
    var t3 = Int(perf_counter_ns())
    t_rep += t1 - t0
    t_sum += t2 - t1
    t_step += t3 - t2


def _ts_kl(
    ctx: DeviceContext, mut y: DeviceBuffer[DType.float32], n: Int, mut dptr: DeviceBuffer[DType.int32],
    mut dind: DeviceBuffer[DType.int32], mut dval: DeviceBuffer[DType.float32], mut drz: DeviceBuffer[DType.float32],
    mut drep: DeviceBuffer[DType.float32], mut dz: DeviceBuffer[DType.float32], mut dkl: DeviceBuffer[DType.float32],
) raises:
    ctx.enqueue_function[repulse_tiled_kernel](Int32(n), y.unsafe_ptr(), drz.unsafe_ptr(), drep.unsafe_ptr(),
                                               grid_dim=(n + RTB - 1) // RTB, block_dim=RTB)
    ctx.enqueue_function[sum_kernel](Int32(n), drz.unsafe_ptr(), dz.unsafe_ptr(), grid_dim=1, block_dim=1)
    ctx.enqueue_function[kl_kernel](Int32(n), y.unsafe_ptr(), dptr.unsafe_ptr(), dind.unsafe_ptr(),
                                    dval.unsafe_ptr(), dz.unsafe_ptr(), dkl.unsafe_ptr(), grid_dim=_grid(n),
                                    block_dim=TPB)


def tsne_fit_device(
    x: List[Float32], n: Int, d: Int, y0: List[Float32], perplexity: Float32, exaggeration: Float32,
    learning_rate: Float32, max_iter: Int, exploration: Int, mut y_out: List[Float32], mut kl_out: Float32,
) raises:
    tsne_validate(n, d, perplexity, max_iter, exploration)
    var nn = tsne_nn(n, perplexity)
    var st = AnnStages("tsne_fit")
    var ctx = x_ann_ctx()
    var dx = upload_f32(ctx, x)
    st.mark(ctx, "upload")
    var dnd = ctx.enqueue_create_buffer[DType.float32](n * nn)
    var dni = ctx.enqueue_create_buffer[DType.int32](n * nn)
    var dp = ctx.enqueue_create_buffer[DType.float32](n * nn)
    knn_enqueue(ctx, dx, n, d, nn, dnd, dni)
    ctx.enqueue_function[perplexity_kernel](Int32(n), dnd.unsafe_ptr(), Int32(nn), identical_log(perplexity),
                                            dp.unsafe_ptr(), grid_dim=_grid(n), block_dim=TPB)
    ctx.synchronize()
    st.host("knn_perplexity")
    var nn_i = download_i32(ctx, dni, n * nn)
    var p_cond = download_f32(ctx, dp, n * nn)
    var indptr = List[Int32]()
    var indices = List[Int32]()
    var values = List[Float32]()
    tsne_symmetrize(n, nn, nn_i, p_cond, indptr, indices, values)
    st.host("symmetrize")

    var dptr = upload_i32(ctx, indptr)
    var dind = upload_i32(ctx, indices)
    var dval = upload_f32(ctx, values)
    var dy = upload_f32(ctx, y0)
    var dy2 = upload_f32(ctx, y0)
    var dupd = upload_f32(ctx, List[Float32](length=2 * n, fill=Float32(0.0)))
    var dgain = upload_f32(ctx, List[Float32](length=2 * n, fill=Float32(1.0)))
    var drz = ctx.enqueue_create_buffer[DType.float32](n)
    var drep = ctx.enqueue_create_buffer[DType.float32](2 * n)
    var dz = ctx.enqueue_create_buffer[DType.float32](1)
    var dkl = ctx.enqueue_create_buffer[DType.float32](n)
    # lane/apple-fast-ann (2026-10-02): the two env switches of the
    # iteration (x_ann/fast_env.mojo), read once here on the host; the
    # stripe partials buffer exists only under the split
    var split = tsne_fast_split()
    var zteam = tsne_fast_zsum()
    comptime if TS_ZSUM:
        zteam = True
    var dpart = ctx.enqueue_create_buffer[DType.float32]((TS_SPLIT * n * 3) if split else 1)
    st.mark(ctx, "upload_graph")
    var t_rep = 0
    var t_sum = 0
    var t_step = 0
    for it in range(max_iter):
        var ex = exaggeration if it < exploration else Float32(1.0)
        var mom = Float32(0.5) if it < exploration else Float32(0.8)
        if st.on:
            # the stage pass: each launch drained and timed (ann-apple3)
            if it % 2 == 0:
                _ts_iter_timed(ctx, dy, dy2, n, dptr, dind, dval, drz, drep, dz, dupd, dgain, ex, mom,
                               learning_rate, t_rep, t_sum, t_step, split, zteam, dpart)
            else:
                _ts_iter_timed(ctx, dy2, dy, n, dptr, dind, dval, drz, drep, dz, dupd, dgain, ex, mom,
                               learning_rate, t_rep, t_sum, t_step, split, zteam, dpart)
        elif it % 2 == 0:
            _ts_iter(ctx, dy, dy2, n, dptr, dind, dval, drz, drep, dz, dupd, dgain, ex, mom, learning_rate,
                     split, zteam, dpart)
        else:
            _ts_iter(ctx, dy2, dy, n, dptr, dind, dval, drz, drep, dz, dupd, dgain, ex, mom, learning_rate,
                     split, zteam, dpart)
    st.mark(ctx, "iterations")
    if st.on:
        print("ANN-STAGE tsne_iter repulse", Float64(t_rep) / 1.0e6)
        print("ANN-STAGE tsne_iter sum", Float64(t_sum) / 1.0e6)
        print("ANN-STAGE tsne_iter step", Float64(t_step) / 1.0e6)
    if max_iter % 2 == 0:
        _ts_kl(ctx, dy, n, dptr, dind, dval, drz, drep, dz, dkl)
    else:
        _ts_kl(ctx, dy2, n, dptr, dind, dval, drz, drep, dz, dkl)
    ctx.synchronize()
    if max_iter % 2 == 0:
        y_out = download_f32(ctx, dy, 2 * n)
    else:
        y_out = download_f32(ctx, dy2, 2 * n)
    var kl = download_f32(ctx, dkl, n)
    var total = Float32(0.0)
    for i in range(n):
        total = total + kl[i]
    kl_out = total
    st.host("kl_download")
    _ = dpart^
    _ = dkl^
    _ = dz^
    _ = drep^
    _ = drz^
    _ = dgain^
    _ = dupd^
    _ = dy2^
    _ = dy^
    _ = dval^
    _ = dind^
    _ = dptr^
    _ = dp^
    _ = dni^
    _ = dnd^
    _ = dx^
    _ = ctx^
