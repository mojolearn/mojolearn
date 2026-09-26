# SPDX-License-Identifier: Apache-2.0
"""Whole Gram output cells with the original contraction order on every GPU."""
from std.os import getenv
from max.gpu import block_idx, block_dim, thread_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from max.algorithm import sync_parallelize
from core.multi_gpu import peer_clone, copy_columns_kernel
from core.step_phase import STEP_PHASE_TIMERS
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_mul_add
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
        acc = ftz(
            identical_mul_add(
                ftz(x.unsafe_load(i * k + p)),
                ftz(x.unsafe_load(j * k + p)),
                acc,
            )
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
    count = min(count,m)
    ctx.synchronize()
    var shards = List[GramOutputShard]()
    for rank in range(count):
        var first = m*rank//count
        var width = m*(rank+1)//count-first
        var device = DeviceContext(device_id=rank)
        var a: DeviceBuffer[DType.float32]
        var b: DeviceBuffer[DType.float32]
        var need = 1
        comptime if tn:
            var packed = ctx.enqueue_create_buffer[DType.float32](k*width)
            ctx.enqueue_function[copy_columns_kernel[False]](x.unsafe_ptr(),packed.unsafe_ptr(),
                Int32(m),Int32(first),Int32(width),Int32(k*width),
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
        shards.append(GramOutputShard(device^,a^,b^,output^,workspace^,first,width))
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
                    Int32(m),Int32(m),Int32(k),Int32(s.first*m),Int32(s.width*m),
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
