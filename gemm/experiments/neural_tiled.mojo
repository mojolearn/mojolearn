# SPDX-License-Identifier: Apache-2.0
"""NN09/NN15: physical staging layouts under the selected neural profile.

128 threads own an 8x16 output tile. Shared operands feed ascending scalar
FMA chains; no native matrix instruction defines an undocumented sum. Page
depth, padded rows, XOR addresses and thread mapping are independent arms.
Neural-only dispatch reaches these opt-in arms. No build or verification run.
"""
from std.sys.compile import is_defined
from std.gpu import block_idx,thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from max.gpu.host import DeviceBuffer,DeviceContext
from checks.numerics import ftz
from checks.rtf_seam import rtf_mul_add
from gemm.contract import CONTRACT_K_LEAF_MIN
from gemm.experiments.neural_profile import (
    NEURAL_EXPERIMENTS_ALLOWED, NEURAL_LEAF, NEURAL_CHAINS,neural_partition,neural_strides,neural_validate,
    neural_fold_push,neural_fold_drain,neural_merge_chains,
)

comptime NN09 = NEURAL_EXPERIMENTS_ALLOWED and is_defined["MOJOLEARN_IDN_NEURAL_NN09"]()
comptime NN15 = NEURAL_EXPERIMENTS_ALLOWED and is_defined["MOJOLEARN_IDN_NEURAL_NN15"]()


@always_inline
def _neural_b_stage_addr[PAD: Int,SWIZZLE: Bool](p: Int,col: Int) -> Int:
    # XOR with p's low 4 bits is self-inverse over all 16 logical columns.
    # PAD adds a bank-row gap, without changing the logical operand index.
    return p*(16+PAD)+(col ^ (p & 15) if SWIZZLE else col)


def neural_tiled_kernel[DEPTH: Int,PAD: Int,SWIZZLE: Bool,TRANSPOSE_THREADS: Bool](
    c: MutPointer[Float32,MutAnyOrigin],a: MutPointer[Float32,MutAnyOrigin],
    b: MutPointer[Float32,MutAnyOrigin],m: Int32,n: Int32,k: Int32,
    leaf: Int32,leaves: Int32,asi: Int32,asp: Int32,bsp: Int32,bsj: Int32,
):
    comptime assert DEPTH == 1 or DEPTH == 2, "one or two staging pages"
    comptime assert PAD == 0 or PAD == 1, "unpadded or one-word padded rows"
    comptime assert DEPTH*24*(16+PAD)*4 <= 32768, "bounded shared page footprint"
    var tid = Int(thread_idx.x)
    var tiles_n = (Int(n)+15)//16
    var row0 = (Int(block_idx.x)//tiles_n)*8
    var col0 = (Int(block_idx.x)%tiles_n)*16
    var ri = tid%8 if TRANSPOSE_THREADS else tid//16
    var ci = tid//8 if TRANSPOSE_THREADS else tid%16
    var row = row0+ri
    var col = col0+ci
    var live = row<Int(m) and col<Int(n)
    var as_ = stack_allocation[DEPTH*8*(16+PAD),Float32,address_space=AddressSpace.SHARED]()
    var bs_ = stack_allocation[DEPTH*16*(16+PAD),Float32,address_space=AddressSpace.SHARED]()
    var stack = SIMD[DType.float32,16](0.0)
    var occupied = 0
    for t in range(Int(leaves)):
        var begin = t*Int(leaf)
        var end = min((t+1)*Int(leaf),Int(k))
        var acc = SIMD[DType.float32,NEURAL_CHAINS](0.0)
        for start in range(begin,end,DEPTH*16):
            comptime for page in range(DEPTH):
                var p0 = start+page*16
                var ar = tid//16
                var ak = tid%16
                var av = Float32(0)
                if row0+ar<Int(m) and p0+ak<end:
                    av = ftz(a.unsafe_load((row0+ar)*Int(asi)+(p0+ak)*Int(asp)))
                as_[page*8*(16+PAD)+ar*(16+PAD)+ak] = av
                # Two coalesced staging passes cover the 16x16 right tile.
                comptime for half in range(2):
                    var q = tid+half*128
                    var bk = q//16
                    var bc = q%16
                    var bv = Float32(0)
                    if p0+bk<end and col0+bc<Int(n):
                        bv = ftz(b.unsafe_load((p0+bk)*Int(bsp)+(col0+bc)*Int(bsj)))
                    bs_[page*16*(16+PAD)+_neural_b_stage_addr[PAD,SWIZZLE](bk,bc)] = bv
            barrier()
            comptime for page in range(DEPTH):
                comptime for p in range(16):
                    # Mask ARITHMETIC tails. Multiplying padded zero by an
                    # infinity would alter NaN behavior; do not do that.
                    if live and start+page*16+p<end:
                        var chain = (start+page*16+p-begin)%NEURAL_CHAINS
                        acc[chain] = rtf_mul_add(as_[page*8*(16+PAD)+ri*(16+PAD)+p],
                            bs_[page*16*(16+PAD)+_neural_b_stage_addr[PAD,SWIZZLE](p,ci)],acc[chain])
            barrier()
        neural_fold_push[16](stack,occupied,neural_merge_chains[NEURAL_CHAINS](acc))
    if live:
        c.unsafe_store(row*Int(n)+col,neural_fold_drain[16](stack,occupied))


def neural_tiled_ab[
    CANDIDATE: Bool = False,DEPTH: Int = 1,PAD: Int = 0,
    SWIZZLE: Bool = False,TRANSPOSE_THREADS: Bool = False,
](ctx: DeviceContext,mut c: DeviceBuffer[DType.float32],
  mut a: DeviceBuffer[DType.float32],mut b: DeviceBuffer[DType.float32],
  m: Int,n: Int,k: Int,op: Int) raises:
    """B one-page row-major layout; A changes requested enabled dimensions.

    NN09 gates depth/padding/swizzle; NN15 gates output-thread mapping. Each
    may be isolated before combinations. This is synchronous shared staging,
    not a claim of asynchronous overlap. Existing A02/A03 evidence does not
    validate this new tile. Host mirrors use the selected leaf/chain profile.
    """
    neural_validate(m,n,k,op)
    if len(c)<m*n or len(a)<m*k or len(b)<n*k:
        raise Error("neural tiled storage too short")
    if m == 0 or n == 0:
        return
    comptime D = DEPTH if CANDIDATE and NN09 else 1
    comptime P = PAD if CANDIDATE and NN09 else 0
    comptime S = SWIZZLE and CANDIDATE and NN09
    comptime T = TRANSPOSE_THREADS and CANDIDATE and NN15
    var part = neural_partition[NEURAL_LEAF](k)
    var st = neural_strides(op,m,n,k)
    ctx.enqueue_function[neural_tiled_kernel[D,P,S,T]](c,a,b,
        Int32(m),Int32(n),Int32(k),Int32(part[0]),Int32(part[1]),
        Int32(st[0]),Int32(st[1]),Int32(st[2]),Int32(st[3]),
        grid_dim=(((m+7)//8)*((n+15)//16),1,1),block_dim=(128,1,1))
