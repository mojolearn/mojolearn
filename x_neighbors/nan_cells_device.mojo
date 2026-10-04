# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`op_nan_cells` on the device (lane hr-small-passes, 2026-10-02): the NaN
cells of an n x d float32 matrix, flat indices ascending, the NaN count of
every column and the total. The same integers as the host pass of
x_neighbors/nan_cells.mojo (the CPU column keeps it), so no bit moves.

A deterministic stream compaction, all of it parallel:
  1. `nan_count_kernel`: block b counts the NaN cells of its chunk of
     NC_CHUNK consecutive flat cells.
  2. `nan_group_sum_kernel`: block g sums NC_CHUNK consecutive block counts.
  3. `nan_top_scan_kernel`: the exclusive scan of the few group sums (one
     group per NC_CHUNK^2 = 4,194,304 cells) and the total.
  4. `nan_down_scan_kernel`: block g scans its block counts from its group
     offset, in rounds of NC_TPB in index order: each block's offset.
  5. `nan_scatter_kernel`: block b writes its NaN cells from its offset, in
     rounds of NC_TPB cells in flat order, each round a block scan.
Every offset is an integer sum, so the list is the host pass's list in the
host pass's order on every column. The column counts (`nan_colmiss_kernel`)
are integer adds of per-(row block, column) counts: exact in any order.
A cell is missing when its exponent is all ones and its mantissa is not
zero (a NaN), tested on the bits so no fast-math can fold `v != v`."""
from std.atomic import Atomic, Ordering
from std.python import PythonObject
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast, stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.kernel_matrix import TARGET_COLUMN, lib_smem_page_fits_for
from x_neighbors.items import FP, IP
from x_neighbors.device_ops import xn_ctx, _down_i
from core.device_pool import pool_give, pool_take
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from std.sys.info import has_apple_gpu_accelerator
from std.sys.compile import is_defined


comptime NC_TPB = 256
comptime NC_PER = 8
comptime NC_CHUNK = NC_TPB * NC_PER
#: rows per thread of the column counts
comptime NC_RB = 64
#: one Int32 per thread of threadgroup memory: 1 KB
comptime NC_SMEM_FITS = lib_smem_page_fits_for[TARGET_COLUMN, NC_TPB * 4]()


@always_inline
def _is_nan(v: Float32) -> Bool:
    var b = bitcast[DType.uint32](v)
    return (b & UInt32(0x7F800000)) == UInt32(0x7F800000) and (b & UInt32(0x007FFFFF)) != UInt32(0)


def nan_count_kernel(x: FP, bcount: IP, total_: Int64):
    var total = Int(total_)
    var t = Int(thread_idx.x)
    var base = Int(block_idx.x) * NC_CHUNK
    var sh = stack_allocation[NC_TPB, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var c = Int32(0)
    for k in range(NC_PER):
        var j = base + k * NC_TPB + t
        if j < total and _is_nan(x.unsafe_load(j)):
            c += 1
    sh[t] = c
    barrier()
    var s = NC_TPB // 2
    while s > 0:
        if t < s:
            sh[t] = sh[t] + sh[t + s]
        barrier()
        s //= 2
    if t == 0:
        bcount.unsafe_store(Int(block_idx.x), sh[0])


def nan_group_sum_kernel(bcount: IP, gsum: IP, nb_: Int64):
    var nb = Int(nb_)
    var t = Int(thread_idx.x)
    var base = Int(block_idx.x) * NC_CHUNK
    var sh = stack_allocation[NC_TPB, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var c = Int32(0)
    for k in range(NC_PER):
        var j = base + k * NC_TPB + t
        if j < nb:
            c += bcount.unsafe_load(j)
    sh[t] = c
    barrier()
    var s = NC_TPB // 2
    while s > 0:
        if t < s:
            sh[t] = sh[t] + sh[t + s]
        barrier()
        s //= 2
    if t == 0:
        gsum.unsafe_store(Int(block_idx.x), sh[0])


def nan_top_scan_kernel(gsum: IP, goff: IP, info: IP, ng_: Int64):
    """One block over the group sums: ng = cells / 4,194,304 rounded up, so
    each thread's run is a handful of entries at any size this op takes
    (int32 cell indices)."""
    var ng = Int(ng_)
    var t = Int(thread_idx.x)
    var per = (ng + NC_TPB - 1) // NC_TPB
    var lo = t * per
    var hi = min(lo + per, ng)
    var sh = stack_allocation[NC_TPB, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var c = Int32(0)
    for i in range(lo, hi):
        c += gsum.unsafe_load(i)
    sh[t] = c
    barrier()
    var off = 1
    while off < NC_TPB:
        var add = Int32(0)
        if t >= off:
            add = sh[t - off]
        barrier()
        sh[t] = sh[t] + add
        barrier()
        off *= 2
    var run = sh[t] - c
    for i in range(lo, hi):
        var g = gsum.unsafe_load(i)
        goff.unsafe_store(i, run)
        run += g
    if t == NC_TPB - 1:
        info.unsafe_store(0, sh[t])


def nan_down_scan_kernel(bcount: IP, goff: IP, boff: IP, nb_: Int64):
    var nb = Int(nb_)
    var t = Int(thread_idx.x)
    var base = Int(block_idx.x) * NC_CHUNK
    var sh = stack_allocation[NC_TPB, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var running = goff.unsafe_load(Int(block_idx.x))
    for k in range(NC_PER):
        var j = base + k * NC_TPB + t
        var v = Int32(0)
        if j < nb:
            v = bcount.unsafe_load(j)
        sh[t] = v
        barrier()
        var off = 1
        while off < NC_TPB:
            var add = Int32(0)
            if t >= off:
                add = sh[t - off]
            barrier()
            sh[t] = sh[t] + add
            barrier()
            off *= 2
        var incl = sh[t]
        var tot = sh[NC_TPB - 1]
        barrier()
        if j < nb:
            boff.unsafe_store(j, running + incl - v)
        running += tot


def nan_scatter_kernel(x: FP, boff: IP, cells: IP, total_: Int64):
    var total = Int(total_)
    var t = Int(thread_idx.x)
    var base = Int(block_idx.x) * NC_CHUNK
    var sh = stack_allocation[NC_TPB, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var running = boff.unsafe_load(Int(block_idx.x))
    for k in range(NC_PER):
        var j = base + k * NC_TPB + t
        var v = Int32(0)
        if j < total and _is_nan(x.unsafe_load(j)):
            v = 1
        sh[t] = v
        barrier()
        var off = 1
        while off < NC_TPB:
            var add = Int32(0)
            if t >= off:
                add = sh[t - off]
            barrier()
            sh[t] = sh[t] + add
            barrier()
            off *= 2
        var incl = sh[t]
        var tot = sh[NC_TPB - 1]
        barrier()
        if v != 0:
            cells.unsafe_store(Int(running + incl - v), Int32(j))
        running += tot


def nan_colmiss_kernel(x: FP, colmiss: IP, n_: Int64, d_: Int64):
    """Thread (row block, column): the NaN count of NC_RB rows of one
    column, added to the column's total (an integer add: any order)."""
    var n = Int(n_)
    var d = Int(d_)
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var nrb = (n + NC_RB - 1) // NC_RB
    if tid >= nrb * d:
        return
    var f = tid % d
    var r0 = (tid // d) * NC_RB
    var r1 = min(r0 + NC_RB, n)
    var c = Int32(0)
    for r in range(r0, r1):
        if _is_nan(x.unsafe_load(r * d + f)):
            c += 1
    if c > 0:
        _ = Atomic.fetch_add[ordering = Ordering.RELAXED](colmiss + f, c)


# lane/apple-fast-gap-manprep (2026-10-03): KNNImputer.fit needs only the
# column counts; the cell list (four scan kernels, a sync and the list's
# download) is the transform's. With colmiss_only=1 the op runs the column
# count kernel alone and stores their sum as the total (integer adds, the
# same count). FAST + Apple default since the M3 A/B gmp-imp-taxi (with
# TIE_MEAN); -D MOJOLEARN_XN_FAST_NAN_COLMISS_ONLY_OFF restores the full pass.
comptime NC_COLMISS_ONLY = (GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
                            and not is_defined["MOJOLEARN_XN_FAST_NAN_COLMISS_ONLY_OFF"]())


# lane/apple-fast-w4-small, opt-in (-D MOJOLEARN_XN_FAST_NAN_FIT_LEAN), on
# NC_COLMISS_ONLY (FAST + Apple): KNNImputer.fit's column counts without
# fresh allocations. Main's fit path creates x's device buffer (n d floats)
# and the count buffer on every call and the Python layer zero-fills an
# n d int32 cell list the op never writes; here x and the counts live in
# pooled buffers (core/device_pool, exact size: the board's rounds reuse
# them), the counts come down straight into the caller's array, and the
# Python layer passes a one-slot cell list (`x_neighbors_nc_fit_lean`).
# The same kernel and integer adds: the same counts.
comptime NC_FIT_LEAN = NC_COLMISS_ONLY and is_defined["MOJOLEARN_XN_FAST_NAN_FIT_LEAN"]()


def nc_fit_lean_binding() raises -> PythonObject:
    """1 when this binary's fit-time count op ignores the cell list (the
    Python layer then passes a one-slot one)."""
    comptime if NC_FIT_LEAN:
        return PythonObject(1)
    return PythonObject(0)


def nan_cells_device(x: Int, cells: Int, colmiss: Int, info: Int, n: Int, d: Int, colmiss_only: Int = 0) raises:
    """The device op: x (n x d) uploaded once, the three outputs downloaded
    (cells: the first `count` slots)."""
    comptime assert NC_SMEM_FITS, "nan_cells_device: a 1 KB threadgroup page must fit"
    var ctx = xn_ctx()
    var total = n * d
    comptime if NC_FIT_LEAN:
        if colmiss_only == 1:
            var bx = pool_take["MojoXNeighborsNanFitX"](ctx, total)
            var bc = pool_take["MojoXNeighborsNanFitCm"](ctx, d)
            if total > 0:
                ctx.enqueue_copy(dst_buf=bx, src_ptr=FP(unsafe_from_address=x))
            ctx.enqueue_memset(bc, Float32(0.0))   # the int32 zero's bits
            if total > 0:
                var nrb2 = (n + NC_RB - 1) // NC_RB
                var threads2 = nrb2 * d
                ctx.enqueue_function[nan_colmiss_kernel](
                    FP(unsafe_from_address=Int(bx.unsafe_ptr())), IP(unsafe_from_address=Int(bc.unsafe_ptr())),
                    Int64(n), Int64(d), grid_dim=(threads2 + NC_TPB - 1) // NC_TPB, block_dim=NC_TPB)
            if d > 0:
                ctx.enqueue_copy(dst_ptr=FP(unsafe_from_address=colmiss), src_buf=bc)
            ctx.synchronize()
            var cnt2 = 0
            var cmp2 = IP(unsafe_from_address=colmiss)
            for f in range(d):
                cnt2 += Int(cmp2.unsafe_load(f))
            IP(unsafe_from_address=info).unsafe_store(0, Int32(cnt2))
            pool_give["MojoXNeighborsNanFitX"](bx^)
            pool_give["MojoXNeighborsNanFitCm"](bc^)
            return
    comptime if NC_COLMISS_ONLY:
        if colmiss_only == 1:
            var d_x1 = ctx.enqueue_create_buffer[DType.float32](max(total, 1))
            if total > 0:
                ctx.enqueue_copy(dst_buf=d_x1, src_ptr=FP(unsafe_from_address=x))
            var d_cm1 = ctx.enqueue_create_buffer[DType.int32](max(d, 1))
            ctx.enqueue_memset(d_cm1, Int32(0))
            if total > 0:
                var nrb1 = (n + NC_RB - 1) // NC_RB
                var threads1 = nrb1 * d
                ctx.enqueue_function[nan_colmiss_kernel](d_x1.unsafe_ptr(), d_cm1.unsafe_ptr(), Int64(n), Int64(d),
                                                         grid_dim=(threads1 + NC_TPB - 1) // NC_TPB, block_dim=NC_TPB)
            if d > 0:
                _down_i(ctx, d_cm1, colmiss, d)
            ctx.synchronize()
            var cnt = 0
            var cmp = IP(unsafe_from_address=colmiss)
            for f in range(d):
                cnt += Int(cmp.unsafe_load(f))
            IP(unsafe_from_address=info).unsafe_store(0, Int32(cnt))
            _ = d_x1^
            _ = d_cm1^
            return
    # x straight to the device (the copy engine; no host-thread staging)
    var d_x = ctx.enqueue_create_buffer[DType.float32](max(total, 1))
    if total > 0:
        ctx.enqueue_copy(dst_buf=d_x, src_ptr=FP(unsafe_from_address=x))
    var nb = max((total + NC_CHUNK - 1) // NC_CHUNK, 1)
    var ng = (nb + NC_CHUNK - 1) // NC_CHUNK
    var d_bc = ctx.enqueue_create_buffer[DType.int32](nb)
    var d_bo = ctx.enqueue_create_buffer[DType.int32](nb)
    var d_gs = ctx.enqueue_create_buffer[DType.int32](ng)
    var d_go = ctx.enqueue_create_buffer[DType.int32](ng)
    var d_cells = ctx.enqueue_create_buffer[DType.int32](max(total, 1))
    var d_cm = ctx.enqueue_create_buffer[DType.int32](max(d, 1))
    var d_info = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.enqueue_memset(d_cm, Int32(0))
    ctx.enqueue_memset(d_info, Int32(0))
    var xp = d_x.unsafe_ptr()
    ctx.enqueue_function[nan_count_kernel](xp, d_bc.unsafe_ptr(), Int64(total), grid_dim=nb, block_dim=NC_TPB)
    ctx.enqueue_function[nan_group_sum_kernel](d_bc.unsafe_ptr(), d_gs.unsafe_ptr(), Int64(nb), grid_dim=ng, block_dim=NC_TPB)
    ctx.enqueue_function[nan_top_scan_kernel](d_gs.unsafe_ptr(), d_go.unsafe_ptr(), d_info.unsafe_ptr(), Int64(ng),
                                              grid_dim=1, block_dim=NC_TPB)
    ctx.enqueue_function[nan_down_scan_kernel](d_bc.unsafe_ptr(), d_go.unsafe_ptr(), d_bo.unsafe_ptr(), Int64(nb),
                                               grid_dim=ng, block_dim=NC_TPB)
    ctx.enqueue_function[nan_scatter_kernel](xp, d_bo.unsafe_ptr(), d_cells.unsafe_ptr(), Int64(total),
                                             grid_dim=nb, block_dim=NC_TPB)
    if total > 0:
        var nrb = (n + NC_RB - 1) // NC_RB
        var threads = nrb * d
        ctx.enqueue_function[nan_colmiss_kernel](xp, d_cm.unsafe_ptr(), Int64(n), Int64(d),
                                                 grid_dim=(threads + NC_TPB - 1) // NC_TPB, block_dim=NC_TPB)
    var hinfo = List[Int32](length=1, fill=Int32(0))
    ctx.enqueue_copy(dst_ptr=hinfo.unsafe_ptr(), src_buf=d_info)
    if d > 0:
        _down_i(ctx, d_cm, colmiss, d)
    ctx.synchronize()
    var count = Int(hinfo[0])
    IP(unsafe_from_address=info).unsafe_store(0, Int32(count))
    if count > 0:
        # only the used prefix crosses back
        ctx.enqueue_copy(dst_ptr=IP(unsafe_from_address=cells), src_buf=d_cells.create_sub_buffer[DType.int32](0, count))
        ctx.synchronize()
    _ = d_x^
    _ = d_bc^
    _ = d_bo^
    _ = d_gs^
    _ = d_go^
    _ = d_cells^
    _ = d_cm^
    _ = d_info^
    _ = hinfo^
