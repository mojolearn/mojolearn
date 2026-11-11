# SPDX-License-Identifier: Apache-2.0
"""Whole Gram output cells with the original contraction order on every GPU.

THE NEGATIVE CONTROL. `-D MOJOLEARN_GRAM_PARALLEL_SABOTAGE=1` makes every
owner above rank 0 contract from one output row early: the packed operand of
the `tn` path is copied from `first - 1`, and the `nt` path's kernel is handed
a first-cell index one row low. The destination sub-buffer keeps the true
`first`, so nothing about a buffer length, an allocation or a validation
changes -- only which values the shard contracts. It is a `comptime if`, so no
production bit can move, and it is INERT AT ONE DEVICE: the shift is guarded by
`rank > 0` and a one-device column has only rank 0, which is what makes a moved
`par-gram`, `par-gram-ols`, `par-gram-pca` or `par-gram-tsvd` cell attributable
to the define rather than to the second device. Owed a two-device column
(`MOJOLEARN_PAR_DEVICES=0,1`); no host binding restates this driver.

THE PLAN VALUE (multi-GPU encapsulation, 2026-11-03,
docs/plans/MULTI_GPU_ENCAPSULATION_20261103.md). Every Gram owner path takes a
`DevicePlan`; nothing below this file reads the thread plan. The split-K shard
path (`gram_splitk_parallel`, formerly `core/gram_splitk.mojo` `_gram_parallel`)
lives here, behind `gram_splitk_launch_dispatch` /
`gram_splitk_launch_centered_dispatch`, which run the unchanged single-device
launch (`_splitk_launch_one` / `_splitk_launch_centered_one`) when
`plan.single()`. Thin entries without a plan read it ONCE through
`gram_plan_current()` and forward.
"""
from core.device_plan import DevicePlan, device_plan_current
from core.device_zero import enqueue_fill
from core.gram_splitk import (
    GRAM_MAX_CELLS_PER_THREAD,
    _enqueue_partial,
    _enqueue_partial_centered,
    _enqueue_reduce,
    _splitk_chunk_rows,
    _splitk_launch_centered_one,
    _splitk_launch_one,
    gram_splitk_cells_for,
    gram_splitk_chunk_count,
)
from std.os import getenv
from std.sys.compile import is_defined
from max.gpu import block_idx, block_dim, thread_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from core.host_parallel import host_parallelize
from core.multi_gpu import peer_clone, copy_columns_kernel
from core.step_phase import STEP_PHASE_TIMERS
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_mul_add
from checks.rtf_seam import rtf_mul_add
from gemm.checks.gemm_identical import identical_gemm_into, identical_gemm_workspace_max_floats
from gemm.contract import OP_TN


def pinned_gemm_nt_gram_kernel(
    z: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
    first_in: Int32,
    count_in: Int32,
):
    """`z[m x n] = x[m x k] ."""
    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var count = Int(count_in) if count_in > 0 else m*n
    if cell >= count:
        return
    var global_cell = cell + Int(first_in)
    var i = global_cell // n
    var j = global_cell % n
    var acc = Float32(0.0)
    for p in range(k):
        acc = rtf_mul_add(
            ftz(x.unsafe_load(i * k + p)),
            ftz(x.unsafe_load(j * k + p)),
            acc,
        )
    z.unsafe_store(cell, ftz(Float32(0.0) + ftz(acc)))


@fieldwise_init
struct GramOutputShard(Movable):
    var ctx: DeviceContext
    var a: DeviceBuffer[DType.float32]
    var b: DeviceBuffer[DType.float32]
    var output: DeviceBuffer[DType.float32]
    var workspace: DeviceBuffer[DType.float32]
    var first: Int
    #: the row this owner CONTRACTS FROM. Equal to `first` in every production
    #: build; MOJOLEARN_GRAM_PARALLEL_SABOTAGE is the only thing that separates
    #: them, and only above rank 0.
    var read_first: Int
    var width: Int

    def __deinit__(deinit self):
        _ = self.workspace^
        _ = self.output^
        _ = self.b^
        _ = self.a^
        try:
            self.ctx.synchronize()
        except:
            pass
        _ = self.ctx^


def gram_plan_current() raises -> DevicePlan:
    """The Gram family's plan, read ONCE from the thread plan (or
    MOJOLEARN_GRAM_DEVICE_COUNT outside one). Only the plan-less thin entries
    call this; every owner path below takes the value."""
    return device_plan_current("MOJOLEARN_GRAM_DEVICE_COUNT")


def gram_plan_owner_count(plan: DevicePlan) raises -> Int:
    """The owner count the Gram/QR/tn-centered gates parsed before the plan
    value: `Int(device_plan_setting(name, "1"))`. `plan.single()` (setting ""
    or "1") is 1; any other setting is parsed exactly as before, so "01" is
    still one owner and a non-integer still raises. Same truth table, same
    route, same bits."""
    if plan.single():
        return 1
    return Int(plan.setting)


def parallel_gram_outputs[tn: Bool](ctx: DeviceContext,
    mut z: DeviceBuffer[DType.float32], mut x: DeviceBuffer[DType.float32],
    m: Int, k: Int,
) raises -> Bool:
    """Plan-less entry (checks): reads the Gram plan once and forwards."""
    return parallel_gram_outputs[tn](gram_plan_current(), ctx, z, x, m, k)


def parallel_gram_outputs[tn: Bool](plan: DevicePlan, ctx: DeviceContext,
    mut z: DeviceBuffer[DType.float32], mut x: DeviceBuffer[DType.float32],
    m: Int, k: Int,
) raises -> Bool:
    var count = gram_plan_owner_count(plan)
    if count == 1:
        return False
    if count < 1 or count > 64 or GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("parallel Gram outputs require IDENTICAL and 1..64 devices")
    comptime if STEP_PHASE_TIMERS:
        raise Error("parallel Gram cannot use process-global GEMM phase counters")
    if m < 1 or k < 1 or m > 2147483647//m:
        raise Error("parallel Gram output exceeds signed 32-bit cell indexing")
    comptime if tn:
        if k > 2147483647//m:
            raise Error("parallel Gram input exceeds signed 32-bit copy indexing")
    count = min(count,m)
    ctx.synchronize()
    var shards = List[GramOutputShard]()
    for rank in range(count):
        var first = m*rank//count
        var width = m*(rank+1)//count-first
        var source = first
        comptime if is_defined["MOJOLEARN_GRAM_PARALLEL_SABOTAGE"]():
            # Check-only arm: later owners contract from one row early.
            if rank > 0:
                source = first - 1
        var device = DeviceContext(device_id=plan.device(rank))
        var a: DeviceBuffer[DType.float32]
        var b: DeviceBuffer[DType.float32]
        var need = 1
        comptime if tn:
            var packed = ctx.enqueue_create_buffer[DType.float32](k*width)
            ctx.enqueue_function[copy_columns_kernel[False]](x.unsafe_ptr(),packed.unsafe_ptr(),
                Int32(m),Int32(source),Int32(width),Int32(k*width),
                grid_dim=((k*width+255)//256,1,1),block_dim=(256,1,1))
            ctx.synchronize()
            a = peer_clone(ctx,device,packed)
            b = peer_clone(ctx,device,x)
            _ = packed^
            need = max(1,identical_gemm_workspace_max_floats(width,m,k))
        else:
            a = peer_clone(ctx,device,x)
            b = device.enqueue_create_buffer[DType.float32](1)
        var output = device.enqueue_create_buffer[DType.float32](width*m)
        var workspace = device.enqueue_create_buffer[DType.float32](need)
        device.synchronize()
        shards.append(GramOutputShard(device^,a^,b^,output^,workspace^,first,source,width))
    var failed = List[Int](length=count,fill=0)
    var sp = rebind[MutPointer[GramOutputShard, MutUntrackedOrigin]](shards.unsafe_ptr())
    var fp = rebind[MutPointer[Int, MutUntrackedOrigin]](failed.unsafe_ptr())
    def task(rank: Int) {imm sp, imm fp, imm m, imm k}:
        try:
            ref s = sp[rank]
            comptime if tn:
                identical_gemm_into(s.ctx,s.output,s.a,s.b,s.workspace,s.width,m,k,OP_TN)
            else:
                s.ctx.enqueue_function[pinned_gemm_nt_gram_kernel](s.output.unsafe_ptr(),s.a.unsafe_ptr(),
                    Int32(m),Int32(m),Int32(k),Int32(s.read_first*m),Int32(s.width*m),
                    grid_dim=((s.width*m+255)//256,1,1),block_dim=(256,1,1))
            s.ctx.synchronize()
        except:
            fp[rank] = 1
    host_parallelize(task,count)
    for rank in range(count):
        if failed[rank] != 0:
            raise Error("Gram output shard failed: " + String(rank))
    for rank in range(count):
        ref s = shards[rank]
        s.ctx.synchronize()
        var target = z.create_sub_buffer[DType.float32](s.first*m,s.width*m)
        s.output.enqueue_copy_to(target)
        s.ctx.synchronize()
    _ = shards^
    ctx.synchronize()
    return True


@fieldwise_init
struct GramShard(Movable):
    var ctx: DeviceContext
    var x: DeviceBuffer[DType.float32]
    var mu: DeviceBuffer[DType.float32]
    var partials: DeviceBuffer[DType.float32]
    var first: Int
    var chunks: Int
    var rows: Int

    def __deinit__(deinit self):
        _ = self.x^
        _ = self.mu^
        _ = self.partials^
        try:
            self.ctx.synchronize()
        except:
            pass
        _ = self.ctx^


def _gram_shard_launch[CELLS: Int, centered: Bool](mut shard: GramShard, m: Int, kc: Int) raises:
    comptime if centered:
        _enqueue_partial_centered[CELLS](shard.ctx, shard.partials, shard.x, shard.mu,
            m, shard.rows, kc, shard.chunks)
    else:
        _enqueue_partial[CELLS](shard.ctx, shard.partials, shard.x,
            m, shard.rows, kc, shard.chunks)


def gram_splitk_parallel[centered: Bool](
    plan: DevicePlan,
    ctx: DeviceContext, mut partials: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32], mut mu: DeviceBuffer[DType.float32],
    m: Int, k: Int, kc: Int, n_chunks: Int,
) raises -> Bool:
    """Assign existing pinned K chunks to devices; do not invent new partials.

    Each GPU receives whole original row chunks. The original kernel sees the
    same row sequence, same chunk_rows and same register width. Only its source
    address and output slot origin change. The caller still folds all 128
    partials, including the original zero tail, in its original order.
    """
    # The exact old test (`requested == "" or requested == "1"`) is
    # `plan.single()` by definition.
    if plan.single():
        return False
    var count = Int(plan.setting)
    if count < 1 or count > 64 or GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("parallel Gram requires IDENTICAL and 1..64 devices")
    var live_chunks = min(n_chunks, (k + kc - 1) // kc)
    var active = min(count, live_chunks)
    if active < 1:
        raise Error("parallel Gram requires nonempty rows")
    ctx.synchronize()
    enqueue_fill(ctx, partials, Float32(0))
    ctx.synchronize()
    var shards = List[GramShard]()
    for rank in range(active):
        var first = live_chunks * rank // active
        var end = live_chunks * (rank + 1) // active
        var begin_row = first * kc
        var rows = min(k, end * kc) - begin_row
        var device = DeviceContext(device_id=plan.device(rank))
        var source_row = begin_row
        comptime if is_defined["MOJOLEARN_GRAM_PARALLEL_SABOTAGE"]():
            # Check-only arm: later owners read their rows one row early. The
            # length `rows * m`, the partial-output size and the write-back
            # offset `shard.first * m * m` all keep the true `begin_row`, so no
            # allocation and no validation moves. INERT AT ONE DEVICE.
            if rank > 0:
                source_row = begin_row - 1
        var view = x.create_sub_buffer[DType.float32](source_row * m, rows * m)
        var local_x = peer_clone(ctx, device, view)
        var local_mu: DeviceBuffer[DType.float32]
        comptime if centered:
            local_mu = peer_clone(ctx, device, mu)
        else:
            local_mu = device.enqueue_create_buffer[DType.float32](1)
        var out = device.enqueue_create_buffer[DType.float32]((end - first) * m * m)
        device.synchronize()
        shards.append(GramShard(device^, local_x^, local_mu^, out^, first, end - first, rows))
    var cells = gram_splitk_cells_for(m)
    for rank in range(active):
        if cells == 4:
            _gram_shard_launch[4, centered](shards[rank], m, kc)
        elif cells == 16:
            _gram_shard_launch[16, centered](shards[rank], m, kc)
        else:
            _gram_shard_launch[GRAM_MAX_CELLS_PER_THREAD, centered](shards[rank], m, kc)
    for rank in range(active):
        ref shard = shards[rank]
        var destination = partials.create_sub_buffer[DType.float32](shard.first * m * m, shard.chunks * m * m)
        shard.partials.enqueue_copy_to(destination)
        shard.ctx.synchronize()
    _ = shards^
    ctx.synchronize()
    return True


def gram_splitk_launch_dispatch(
    plan: DevicePlan,
    ctx: DeviceContext,
    mut z: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut partials: DeviceBuffer[DType.float32],
    m: Int,
    k: Int,
) raises:
    """The plain split-K Gram over `plan`. One device: the unchanged
    single-device launch (`_splitk_launch_one`). Otherwise whole original K
    chunks per device (`gram_splitk_parallel`), then the original reduce on
    `ctx` in the original fold order."""
    if plan.single():
        _splitk_launch_one(ctx, z, x, partials, m, k)
        return
    var n_chunks = gram_splitk_chunk_count()
    var kc = _splitk_chunk_rows(k, n_chunks)
    var mn = m * m
    var unused_mu = x
    if gram_splitk_parallel[False](plan, ctx, partials, x, unused_mu, m, k, kc, n_chunks):
        _enqueue_reduce(ctx, z, partials, mn, n_chunks)
        return
    _splitk_launch_one(ctx, z, x, partials, m, k)


def gram_splitk_launch_centered_dispatch(
    plan: DevicePlan,
    ctx: DeviceContext,
    mut z: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut mu: DeviceBuffer[DType.float32],
    mut partials: DeviceBuffer[DType.float32],
    m: Int,
    k: Int,
) raises:
    """`gram_splitk_launch_dispatch` with the centered tile read."""
    if plan.single():
        _splitk_launch_centered_one(ctx, z, x, mu, partials, m, k)
        return
    var n_chunks = gram_splitk_chunk_count()
    var kc = _splitk_chunk_rows(k, n_chunks)
    var mn = m * m
    if gram_splitk_parallel[True](plan, ctx, partials, x, mu, m, k, kc, n_chunks):
        _enqueue_reduce(ctx, z, partials, mn, n_chunks)
        return
    _splitk_launch_centered_one(ctx, z, x, mu, partials, m, k)
