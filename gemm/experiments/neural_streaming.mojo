# SPDX-License-Identifier: Apache-2.0
"""NN02/NN11/NN16: bounded leaf batches, exact stack capacity, write ownership.

Both arms compute the selected common FP32 profile. The bounded arm reuses a
GROUP-leaf plane and saves the canonical streaming stack between launches.
The control writes ALL leaf partials once and folds them in a single pass.
Scratch ownership is one in-order context; no allocation is shared globally.
Neural model dispatch reaches these opt-in arms. Uncompiled and unverified.
"""
from gemm.experiments.neural_profile_device import neural_profile_device
from std.sys.compile import is_defined
from std.gpu import block_idx, block_dim, thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import ftz
from gemm.contract import CONTRACT_K_LEAF_MIN
from gemm.experiments.neural_switches import (
    NEURAL_GEMM_SCHEDULE,SCHED_GEOMETRY,SCHED_STREAM,SCHED_STREAM_EXACT,
    SCHED_PAGES,SCHED_COST,SCHED_FOLD_EXACT,SCHED_THREADMAP,SCHED_PAGES_THREADMAP,
)
from gemm.experiments.neural_profile import (
    NEURAL_EXPERIMENTS_ALLOWED, NEURAL_LEAF, NEURAL_CHAINS, neural_partition, neural_strides, neural_validate,
    neural_leaf, neural_fold_push, neural_fold_drain, 
)

# No measured winners: all switches OFF. A full-workload A/B on NVIDIA and
# AMD plus same-version host/Apple identity and neural quality remains pending.
# Arms of MOJOLEARN_IDN_NEURAL_GEMM_SCHEDULE (gemm/experiments/neural_switches.mojo):
# 2 stream, 4 stream + exact slots, 11 exact slots alone.
comptime NN02 = NEURAL_EXPERIMENTS_ALLOWED and (NEURAL_GEMM_SCHEDULE == SCHED_STREAM or NEURAL_GEMM_SCHEDULE == SCHED_STREAM_EXACT)
comptime NN11 = NEURAL_EXPERIMENTS_ALLOWED and (NEURAL_GEMM_SCHEDULE == SCHED_STREAM_EXACT or NEURAL_GEMM_SCHEDULE == SCHED_FOLD_EXACT)
comptime NN16 = NEURAL_EXPERIMENTS_ALLOWED and is_defined["MOJOLEARN_IDN_NEURAL_NN16"]()


def neural_fold_levels(leaves: Int) -> Int:
    """floor(log2(P))+1: a power-of-two leaf count occupies that top slot."""
    var levels = 1
    var remaining = leaves
    while remaining > 1:
        remaining = remaining >> 1
        levels += 1
    return levels


def neural_streaming_floats[GROUP: Int = 8, SPECIALIZE: Bool = False](k: Int, cells: Int) -> Int:
    comptime assert GROUP > 0, "positive bounded leaf group"
    var leaves = neural_partition[NEURAL_LEAF](k)[1]
    var levels = neural_fold_levels(leaves) if SPECIALIZE and NN11 else 16
    return max(1,(min(GROUP,leaves)+levels)*cells)


def _neural_clear_kernel(p: MutPointer[Float32,MutAnyOrigin], count: Int32):
    var i = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if i < Int(count):
        p.unsafe_store(i,Float32(0))


def _neural_leaf_batch_kernel(
    partials: MutPointer[Float32,MutAnyOrigin], a: MutPointer[Float32,MutAnyOrigin],
    b: MutPointer[Float32,MutAnyOrigin], m: Int32,n: Int32,k: Int32,
    leaf: Int32, first: Int32, count: Int32, asi: Int32,asp: Int32,bsp: Int32,bsj: Int32,
):
    var i = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    var cells = Int(m)*Int(n)
    if i < cells*Int(count):
        var t = Int(first)+i//cells
        var cell = i%cells
        var value = neural_leaf[NEURAL_CHAINS](a,b,cell//Int(n),cell%Int(n),
            t*Int(leaf),min((t+1)*Int(leaf),Int(k)),Int(asi),Int(asp),Int(bsp),Int(bsj))
        partials.unsafe_store(i,value)


def _neural_stream_fold_kernel(
    c: MutPointer[Float32,MutAnyOrigin], partials: MutPointer[Float32,MutAnyOrigin],
    state: MutPointer[Float32,MutAnyOrigin], cells: Int32,first: Int32,count: Int32,
    leaves: Int32,levels: Int32,
):
    var cell = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if cell >= Int(cells):
        return
    # Binary-counter tree occupancy after exactly first leaves is first.
    # Never read an unoccupied slot. Thus every read below has a dominating
    # producer in this launch or an earlier launch on this in-order stream.
    var occupied = Int(first)
    for t in range(Int(count)):
        var value = partials.unsafe_load(t*Int(cells)+cell)
        var level = 0
        while (occupied & (1 << level)) != 0:
            value = ftz(ftz(state.unsafe_load(level*Int(cells)+cell))+ftz(value))
            occupied -= 1 << level
            level += 1
        state.unsafe_store(level*Int(cells)+cell,value)
        occupied += 1 << level
    if Int(first)+Int(count) == Int(leaves):
        var result = Float32(0)
        var have = False
        for level in range(Int(levels)):
            if (occupied & (1 << level)) != 0:
                var value = state.unsafe_load(level*Int(cells)+cell)
                if have:
                    result = ftz(ftz(value)+ftz(result))
                else:
                    result = value
                    have = True
        c.unsafe_store(cell,ftz(result))


def _neural_full_fold_kernel(
    c: MutPointer[Float32,MutAnyOrigin], partials: MutPointer[Float32,MutAnyOrigin],
    cells: Int32,leaves: Int32,
):
    var cell = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if cell < Int(cells):
        var stack = SIMD[DType.float32,16](0.0)
        var occupied = 0
        for t in range(Int(leaves)):
            neural_fold_push[16](stack,occupied,partials.unsafe_load(t*Int(cells)+cell))
        c.unsafe_store(cell,neural_fold_drain[16](stack,occupied))


def _neural_shared_fold_kernel[SLOTS: Int](
    c: MutPointer[Float32,MutAnyOrigin],a: MutPointer[Float32,MutAnyOrigin],
    b: MutPointer[Float32,MutAnyOrigin],m: Int32,n: Int32,k: Int32,
    leaf: Int32,leaves: Int32,asi: Int32,asp: Int32,bsp: Int32,bsj: Int32,
):
    # Every thread owns one column of this level-major shared array. There
    # is no interthread read and consequently no barrier or physical-lane
    # participation assumption. At 16 levels this uses 8 KiB per block.
    var state = stack_allocation[SLOTS*128,Float32,address_space=AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var cell = Int(block_idx.x)*128+tid
    if cell>=Int(m)*Int(n):
        return
    var occupied = 0
    for t in range(Int(leaves)):
        var value = neural_leaf[NEURAL_CHAINS](a,b,cell//Int(n),cell%Int(n),
            t*Int(leaf),min((t+1)*Int(leaf),Int(k)),Int(asi),Int(asp),Int(bsp),Int(bsj))
        var level = 0
        while (occupied & (1 << level)) != 0:
            value = ftz(ftz(state[level*128+tid])+ftz(value))
            occupied -= 1 << level
            level += 1
        state[level*128+tid] = value
        occupied += 1 << level
    var result = Float32(0)
    var have = False
    comptime for level in range(SLOTS):
        if (occupied & (1 << level)) != 0:
            if have:
                result = ftz(ftz(state[level*128+tid])+ftz(result))
            else:
                result = state[level*128+tid]
                have = True
    c.unsafe_store(cell,ftz(result))


def _neural_capacity_launch[SLOTS: Int,SHARED: Bool](ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],mut a: DeviceBuffer[DType.float32],mut b: DeviceBuffer[DType.float32],
    m: Int,n: Int,k: Int,op: Int) raises:
    comptime if not SHARED:
        neural_profile_device[NEURAL_LEAF,NEURAL_CHAINS,SLOTS](ctx,c,a,b,m,n,k,op)
    else:
        neural_validate(m,n,k,op)
        if len(c)<m*n or len(a)<m*k or len(b)<n*k:
            raise Error("neural shared fold operand too short")
        if m==0 or n==0:
            return
        var part = neural_partition[NEURAL_LEAF](k)
        if part[1]>=(1 << SLOTS):
            raise Error("neural shared fold stack too small")
        var st = neural_strides(op,m,n,k)
        ctx.enqueue_function[_neural_shared_fold_kernel[SLOTS]](c,a,b,
            Int32(m),Int32(n),Int32(k),Int32(part[0]),Int32(part[1]),
            Int32(st[0]),Int32(st[1]),Int32(st[2]),Int32(st[3]),
            grid_dim=((m*n+127)//128,1,1),block_dim=(128,1,1))


def neural_streaming_ab[
    CANDIDATE: Bool = False, GROUP: Int = 8,
    SPECIALIZE: Bool = False, ELIDE_CLEAR: Bool = False,
](ctx: DeviceContext, mut c: DeviceBuffer[DType.float32],
  mut a: DeviceBuffer[DType.float32],mut b: DeviceBuffer[DType.float32],
  mut scratch: DeviceBuffer[DType.float32],m: Int,n: Int,k: Int,op: Int) raises:
    """Caller owns scratch through completion; group size changes only storage.

    NN02 B is full partial materialization, A bounded partial groups. NN11
    toggles exact global state slots independently. NN16 toggles the redundant
    scratch clear independently in BOTH arms. NN16 has an intentionally
    explicit component B; this does not claim a redundant clear was found in
    a production model. Public NN16 instead removes the existing ByteLMScratch
    head-workspace clear. NN03/NN04 select the shared leaf/chain profile here.
    """
    comptime assert GROUP > 0, "positive bounded leaf group"
    neural_validate(m,n,k,op)
    if len(c)<m*n or len(a)<m*k or len(b)<n*k:
        raise Error("neural streaming operand storage too short")
    if m == 0 or n == 0:
        return
    var part = neural_partition[NEURAL_LEAF](k)
    var cells = m*n
    var levels = neural_fold_levels(part[1]) if SPECIALIZE and NN11 else 16
    var batch = part[1]
    comptime if CANDIDATE and NN02:
        batch = min(GROUP,part[1])
    var required = max(1,batch*cells)
    comptime if CANDIDATE and NN02:
        required = max(1,(batch+levels)*cells)
    if required > 2147483647 or len(scratch)<required:
        raise Error("neural streaming scratch too short or beyond Int32 launch range")
    if part[1] == 0:
        # k=0 owns and stores +0 for every logical output; no uninitialized C.
        ctx.enqueue_function[_neural_clear_kernel](c,Int32(cells),
            grid_dim=((cells+127)//128,1,1),block_dim=(128,1,1))
        return
    comptime if not (ELIDE_CLEAR and NN16):
        ctx.enqueue_function[_neural_clear_kernel](scratch,Int32(required),
            grid_dim=((required+127)//128,1,1),block_dim=(128,1,1))
    var st = neural_strides(op,m,n,k)
    for first in range(0,part[1],batch):
        var count = min(batch,part[1]-first)
        ctx.enqueue_function[_neural_leaf_batch_kernel](scratch,a,b,Int32(m),Int32(n),Int32(k),
            Int32(part[0]),Int32(first),Int32(count),Int32(st[0]),Int32(st[1]),Int32(st[2]),Int32(st[3]),
            grid_dim=((count*cells+127)//128,1,1),block_dim=(128,1,1))
        comptime if CANDIDATE and NN02:
            # The allocation has disjoint leaf [0,batch*cells) and fold-state
            # tails. One raw view describes both ranges for this queued launch.
            var scratch_ptr = scratch.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
            ctx.enqueue_function[_neural_stream_fold_kernel](c.unsafe_ptr(),scratch_ptr,
                scratch_ptr+batch*cells,Int32(cells),Int32(first),Int32(count),
                Int32(part[1]),Int32(levels),grid_dim=((cells+127)//128,1,1),block_dim=(128,1,1))
        else:
            ctx.enqueue_function[_neural_full_fold_kernel](c,scratch,Int32(cells),Int32(part[1]),
                grid_dim=((cells+127)//128,1,1),block_dim=(128,1,1))


def neural_fold_capacity_ab[CANDIDATE: Bool = False,SHARED: Bool = False](
    ctx: DeviceContext,mut c: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],mut b: DeviceBuffer[DType.float32],
    m: Int,n: Int,k: Int,op: Int,
) raises:
    """NN11 register arm: smallest supported SIMD extent that holds the tree.

    Extents are 1/2/4/8/16 (SIMD power-of-two constraint). Selection follows
    only the mathematical tree height; m/n, dataset and vendor are absent.
    """
    comptime if CANDIDATE and NN11:
        var levels = neural_fold_levels(neural_partition[NEURAL_LEAF](k)[1])
        if levels <= 1:
            _neural_capacity_launch[1,SHARED](ctx,c,a,b,m,n,k,op)
        elif levels <= 2:
            _neural_capacity_launch[2,SHARED](ctx,c,a,b,m,n,k,op)
        elif levels <= 4:
            _neural_capacity_launch[4,SHARED](ctx,c,a,b,m,n,k,op)
        elif levels <= 8:
            _neural_capacity_launch[8,SHARED](ctx,c,a,b,m,n,k,op)
        else:
            _neural_capacity_launch[16,SHARED](ctx,c,a,b,m,n,k,op)
    else:
        neural_profile_device[NEURAL_LEAF,NEURAL_CHAINS,16](ctx,c,a,b,m,n,k,op)
