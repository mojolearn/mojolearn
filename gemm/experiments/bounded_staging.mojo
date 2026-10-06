# SPDX-License-Identifier: Apache-2.0
"""A02 bounded one/two/four-plane staging with unchanged FP32 leaves.

Existing packed production bodies admit only one/two pages. This separate
supported adapter tests a four-plane resource choice without silently
widening those bodies' template contract. Every plane holds one ascending
contraction step for each independent output lane, and two uniform barriers
protect the whole staging group. Increasing depth amortizes those barriers
over more terms while charging its extra shared storage. It does not claim
asynchronous load overlap or replace the production tiled-reuse dispatcher.
"""
from std.gpu import block_idx,block_dim,thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext,DeviceBuffer
from gemm.checks.gemm_identical import GEMM_FOLD_SLOTS,_fold_push,_fold_drain,contract_partition,gemm_operand_strides
from checks.numerics import ftz
from checks.rtf_seam import rtf_mul_add


def staging_kernel[DEPTH: Int](c: MutPointer[Float32,MutAnyOrigin],a: MutPointer[Float32,MutAnyOrigin],
    b: MutPointer[Float32,MutAnyOrigin],m: Int32,n: Int32,k: Int32,leaf: Int32,leaves: Int32,
    asi: Int32,asp: Int32,bsp: Int32,bsj: Int32):
    comptime assert DEPTH==1 or DEPTH==2 or DEPTH==4,"bounded staging depth"
    var tid=Int(thread_idx.x);var cell=Int(block_idx.x)*Int(block_dim.x)+tid
    var valid=cell<Int(m)*Int(n)
    var row=cell//Int(n);var col=cell%Int(n)
    var as_=stack_allocation[DEPTH*128,Float32,address_space=AddressSpace.SHARED]()
    var bs_=stack_allocation[DEPTH*128,Float32,address_space=AddressSpace.SHARED]()
    var stack=SIMD[DType.float32,GEMM_FOLD_SLOTS](0.0);var occ=0
    for t in range(Int(leaves)):
        var begin=t*Int(leaf);var end=min(begin+Int(leaf),Int(k));var acc=Float32(0)
        for start in range(begin,end,DEPTH):
            comptime for page in range(DEPTH):
                var av=Float32(0);var bv=Float32(0)
                if valid and start+page<end:
                    av=ftz(a.unsafe_load(row*Int(asi)+(start+page)*Int(asp)))
                    bv=ftz(b.unsafe_load((start+page)*Int(bsp)+col*Int(bsj)))
                as_[page*128+tid]=av;bs_[page*128+tid]=bv
            # No masked output lane may return before these uniform barriers.
            barrier()
            comptime for page in range(DEPTH):
                if start+page<end:
                    acc=rtf_mul_add(as_[page*128+tid],bs_[page*128+tid],acc)
            barrier()
        _=_fold_push(stack,occ,ftz(acc))
    if valid:c.unsafe_store(cell,ftz(_fold_drain(stack,occ)))


# A02 experiment: NEVER RUN — PENDING MEASUREMENT; incumbent defaults retained.
# A02 PENDING: compile evidence alone does not qualify device correctness, quality or speed.
# Explicit depth1/2/4 staging adapter only; existing library depth stays unchanged.
def bounded_staging_gemm[DEPTH: Int](ctx: DeviceContext,mut c: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],mut b: DeviceBuffer[DType.float32],m: Int,n: Int,k: Int,op: Int) raises:
    if m<1 or n<1 or k<1 or len(c)<m*n or len(a)<m*k or len(b)<n*k:
        raise Error("invalid bounded staging geometry")
    var part=contract_partition(k);var st=gemm_operand_strides(op,m,n,k)
    ctx.enqueue_function[staging_kernel[DEPTH]](c,a,b,Int32(m),Int32(n),Int32(k),Int32(part[0]),Int32(part[1]),
        Int32(st[0]),Int32(st[1]),Int32(st[2]),Int32(st[3]),grid_dim=((m*n+127)//128,1,1),block_dim=(128,1,1))
