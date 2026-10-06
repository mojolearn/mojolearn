# SPDX-License-Identifier: Apache-2.0
"""Supported NVIDIA cp.async operand pipeline with exact leaf/fold order.

Four-byte naturally aligned loads support NN/NT/TN and ragged contractions.
Each thread owns its two staging slots: completion waits are per-thread and
no other thread reads those slots. A next-operand copy is issued before the
current FMA; no assembly or compiler-output edits are involved. Plain staged
loads are a separate compile-time control. Tile reuse is deliberately kept
separate from this initial supported-primitive experiment.
"""
from std.gpu import block_idx,block_dim,thread_idx
from std.memory import stack_allocation
from std.sys.info import is_nvidia_gpu
from max.gpu.memory import async_copy,async_copy_commit_group,async_copy_wait_all,AddressSpace
from max.gpu.host import DeviceBuffer,DeviceContext
from gemm.checks.gemm_identical import GEMM_FOLD_SLOTS,_fold_push,_fold_drain,contract_partition,gemm_operand_strides
from checks.numerics import ftz
from checks.rtf_seam import rtf_mul_add


def pipeline_kernel[ASYNC: Bool](c: MutPointer[Float32,MutAnyOrigin],a: MutPointer[Float32,MutAnyOrigin],
    b: MutPointer[Float32,MutAnyOrigin],m: Int32,n: Int32,k: Int32,leaf: Int32,leaves: Int32,
    asi: Int32,asp: Int32,bsp: Int32,bsj: Int32):
    var tid = Int(thread_idx.x)
    var cell = Int(block_idx.x)*Int(block_dim.x)+tid
    if cell >= Int(m)*Int(n):
        return
    var row = cell//Int(n)
    var col = cell%Int(n)
    var as_ = stack_allocation[256,Float32,alignment=16,address_space=AddressSpace.SHARED]()
    var bs_ = stack_allocation[256,Float32,alignment=16,address_space=AddressSpace.SHARED]()
    var stack = SIMD[DType.float32,GEMM_FOLD_SLOTS](0.0)
    var occ = 0
    for t in range(Int(leaves)):
        var begin = t*Int(leaf)
        var end = min(begin+Int(leaf),Int(k))
        var acc = Float32(0.0)
        var slot = 0
        var ap = a.unsafe_offset(row*Int(asi)+begin*Int(asp))
        var bp = b.unsafe_offset(begin*Int(bsp)+col*Int(bsj))
        comptime if ASYNC and is_nvidia_gpu():
            async_copy[size=4](ap.address_space_cast[AddressSpace.GLOBAL](),as_.unsafe_offset(tid))
            async_copy[size=4](bp.address_space_cast[AddressSpace.GLOBAL](),bs_.unsafe_offset(tid))
            async_copy_commit_group()
            async_copy_wait_all()
        else:
            as_[tid]=ap.unsafe_load()
            bs_[tid]=bp.unsafe_load()
        for p in range(begin,end):
            var nextslot = 1-slot
            if p+1 < end:
                ap = a.unsafe_offset(row*Int(asi)+(p+1)*Int(asp))
                bp = b.unsafe_offset((p+1)*Int(bsp)+col*Int(bsj))
                comptime if ASYNC and is_nvidia_gpu():
                    async_copy[size=4](ap.address_space_cast[AddressSpace.GLOBAL](),as_.unsafe_offset(nextslot*128+tid))
                    async_copy[size=4](bp.address_space_cast[AddressSpace.GLOBAL](),bs_.unsafe_offset(nextslot*128+tid))
                    async_copy_commit_group()
                else:
                    as_[nextslot*128+tid]=ap.unsafe_load()
                    bs_[nextslot*128+tid]=bp.unsafe_load()
            acc = rtf_mul_add(ftz(as_[slot*128+tid]),ftz(bs_[slot*128+tid]),acc)
            comptime if ASYNC and is_nvidia_gpu():
                if p+1 < end:
                    async_copy_wait_all()
            slot = nextslot
        _ = _fold_push(stack,occ,ftz(acc))
    c.unsafe_store(cell,ftz(_fold_drain(stack,occ)))


# N03 experiment: NEVER RUN — PENDING VALIDATION; incumbent defaults retained.
# N03 PENDING: compile evidence alone does not qualify device correctness, quality or speed.
# Explicit NVIDIA-only async adapter; no production default or unsupported vendor fallback.
def pipeline_gemm[ASYNC: Bool](ctx: DeviceContext,mut c: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],mut b: DeviceBuffer[DType.float32],m: Int,n: Int,k: Int,op: Int) raises:
    if m<1 or n<1 or k<1 or len(c)<m*n or len(a)<m*k or len(b)<n*k:
        raise Error("invalid async pipeline geometry")
    var part = contract_partition(k)
    var st = gemm_operand_strides(op,m,n,k)
    ctx.enqueue_function[pipeline_kernel[ASYNC]](c,a,b,Int32(m),Int32(n),Int32(k),
        Int32(part[0]),Int32(part[1]),Int32(st[0]),Int32(st[1]),Int32(st[2]),Int32(st[3]),
        grid_dim=((m*n+127)//128,1,1),block_dim=(128,1,1))
