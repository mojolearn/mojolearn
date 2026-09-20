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
"""
from std.os import getenv
from std.sys.compile import is_defined
from std.gpu import block_idx, block_dim, thread_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from max.algorithm import sync_parallelize
from core.multi_gpu import peer_clone, copy_columns_kernel
from core.step_phase import STEP_PHASE_TIMERS
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_mul_add
from checks.rtf_seam import rtf_mul_add
from gemm.checks.gemm_identical import identical_gemm_into, identical_gemm_workspace_max_floats
from gemm.checks.gemm_oracle import OP_TN


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


def parallel_gram_outputs[tn: Bool](ctx: DeviceContext,
    mut z: DeviceBuffer[DType.float32], mut x: DeviceBuffer[DType.float32],
    m: Int, k: Int,
) raises -> Bool:
    var count = Int(getenv("MOJOLEARN_GRAM_DEVICE_COUNT", "1"))
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
        var device = DeviceContext(device_id=rank)
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
    sync_parallelize(task,count)
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
