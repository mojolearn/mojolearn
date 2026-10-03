"""FAST-only working-set selection without host round trips (Apple).

`WorkingSet.simple_select` gathers from the sorted order three times
(upper set from the front, lower set from the back, a fill from the front)
and each gather reads its count back to the host, which on Metal is a
drain per gather, two or three per SMO outer iteration. Here every count
stays on the device:

  * `fws_flags_kernel` (grid): the candidate flag of each SORTED position,
    `in_upper` / `in_lower` / any, and not yet selected (`selmap`, one
    byte per training index, written by earlier launches only);
  * `fws_count_kernel` (grid): each FWS_T-wide chunk of the sorted order
    (from the back for the lower set) counts its candidates;
  * `fws_pick_kernel` (grid): ranks each candidate by the chunk counts
    before it plus a ballot per warp, and writes the same indices at the
    same working-set slots the host-counted gather writes; the stage's
    count goes to `state`, which the next launch reads. (Until
    lane/cgr-kernel one block walked the whole sorted list.)

A thread only reads device memory an EARLIER launch wrote, plus
threadgroup memory behind a barrier, so no in-kernel device-memory
ordering is relied on. The selection is the same list in the same order:
the first `need` candidates in sorted order (the last `need`, kept in
ascending order, for the lower set).
"""

from std.bit import pop_count
from std.gpu import WARP_SIZE, block_dim, block_idx, thread_idx
from std.gpu.primitives.warp import vote
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from std.sys.info import has_apple_gpu_accelerator
from svm.impl.smo_sets import in_lower, in_upper

#: SCHEDULING (the walk's selection is ranked over the whole sorted list, so
#: the chunk width moves no bit). 256 on Apple: at 1024 the M2 Pro's
#: pipeline limit for the one-block walk (now `fws_pick_kernel`) was 832 threads (Metal validation,
#: steward 1790601522115), no Dynamic Caching, and the over-limit dispatch
#: is dropped with no error: svc / svr moved on M2 Metal only.
comptime FWS_T = 256 if has_apple_gpu_accelerator() else 1024
comptime FWS_WARPS = FWS_T // WARP_SIZE
#: The ballot word: one bit per hardware lane (AMD's wave is 64 wide).
comptime FWS_VOTE = DType.uint64 if WARP_SIZE == 64 else DType.uint32

comptime FWS_UPPER = 0
comptime FWS_LOWER = 1
comptime FWS_ANY = 2


def fws_mark_kernel(
    selmap: MutPointer[UInt8, MutAnyOrigin],
    idx: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
):
    """`selmap[idx[t]] = 1` for the FIFO-kept slots."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(n_in):
        selmap.unsafe_store(Int(idx.unsafe_load(t)), UInt8(1))


def fws_flags_kernel(
    flags: MutPointer[UInt8, MutAnyOrigin],
    sorted_idx: MutPointer[UInt32, MutAnyOrigin],
    nt_in: Int32,
    alpha: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    C: MutPointer[Float32, MutAnyOrigin],
    selmap: MutPointer[UInt8, MutAnyOrigin],
    mode: Int32,
):
    var j = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if j < Int(nt_in):
        var e = Int(sorted_idx.unsafe_load(j))
        var ok = selmap.unsafe_load(e) == UInt8(0)
        if ok and mode == Int32(FWS_UPPER):
            ok = in_upper(alpha.unsafe_load(e), y.unsafe_load(e), C.unsafe_load(e))
        elif ok and mode == Int32(FWS_LOWER):
            ok = in_lower(alpha.unsafe_load(e), y.unsafe_load(e), C.unsafe_load(e))
        flags.unsafe_store(j, UInt8(1) if ok else UInt8(0))


@always_inline
def _fws_need(state: MutPointer[Int32, MutAnyOrigin], n_fifo: Int, n_ws: Int, stage: Int32) -> Tuple[Int, Int]:
    """(slots already filled, candidates this stage takes). Stage 0: the
    upper half (`(n_ws - n_fifo) // 2`) from the front. Stage 1: the lower
    set from the back, up to the rest. Stage 2: the fill from the front."""
    var n_already = n_fifo
    if stage == Int32(0):
        return (n_already, (n_ws - n_already) // 2)
    if stage == Int32(1):
        n_already += Int(state.unsafe_load(0))
        return (n_already, n_ws - n_already)
    n_already += Int(state.unsafe_load(0)) + Int(state.unsafe_load(1))
    return (n_already, n_ws - n_already)


def fws_count_kernel[
    FROM_END: Bool
](flags: MutPointer[UInt8, MutAnyOrigin], nt_in: Int32, bc: MutPointer[Int32, MutAnyOrigin]):
    """Grid, one block per FWS_T-wide chunk of the sorted order (from the
    back for the lower set): `bc[block]` = the chunk's candidate count."""
    var t = Int(thread_idx.x)
    var b = Int(block_idx.x)
    var lane = t % WARP_SIZE
    var warp = t // WARP_SIZE
    var nt = Int(nt_in)
    var r = b * FWS_T + t
    var flag = False
    if r < nt:
        var j = nt - 1 - r if FROM_END else r
        flag = flags.unsafe_load(j) != UInt8(0)
    var wsum = stack_allocation[
        FWS_WARPS, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var m = vote[FWS_VOTE](flag)
    if lane == 0:
        wsum[warp] = Int32(pop_count(m))
    barrier()
    if t == 0:
        var tot = 0
        for w in range(FWS_WARPS):
            tot += Int(wsum[w])
        bc.unsafe_store(b, Int32(tot))


def fws_pick_kernel[
    FROM_END: Bool
](
    flags: MutPointer[UInt8, MutAnyOrigin],
    sorted_idx: MutPointer[UInt32, MutAnyOrigin],
    nt_in: Int32,
    nb_in: Int32,
    bc: MutPointer[Int32, MutAnyOrigin],
    idx: MutPointer[Int32, MutAnyOrigin],
    selmap: MutPointer[UInt8, MutAnyOrigin],
    state: MutPointer[Int32, MutAnyOrigin],
    n_fifo_in: Int32,
    n_ws_in: Int32,
    stage: Int32,
):
    """Grid, the same chunks as `fws_count_kernel`: a candidate's rank is
    the counts of the chunks before its own (`bc`, an earlier launch's)
    plus its rank inside the chunk (a ballot per warp). The first `need`
    ranks are written at the working-set slots the walk wrote (the lower
    set's in ascending sorted order); block 0 stores the stage's count in
    `state[stage]`. Integer ranks only: the same list in the same order as
    the one-block walk it replaced (lane/cgr-kernel)."""
    var t = Int(thread_idx.x)
    var b = Int(block_idx.x)
    var lane = t % WARP_SIZE
    var warp = t // WARP_SIZE
    var nt = Int(nt_in)
    var nb = Int(nb_in)
    var nn = _fws_need(state, Int(n_fifo_in), Int(n_ws_in), stage)
    var n_already = nn[0]
    var need = nn[1]
    if need <= 0:
        if b == 0 and t == 0:
            state.unsafe_store(Int(stage), Int32(0))
        return
    # the chunks before this one and all of them, folded by the block
    var red_pre = stack_allocation[
        FWS_T, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var red_tot = stack_allocation[
        FWS_T, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var my_pre = 0
    var my_tot = 0
    var i = t
    while i < nb:
        var v = Int(bc.unsafe_load(i))
        if i < b:
            my_pre += v
        my_tot += v
        i += FWS_T
    red_pre[t] = Int32(my_pre)
    red_tot[t] = Int32(my_tot)
    barrier()
    var step = FWS_T // 2
    while step > 0:
        if t < step:
            red_pre[t] = red_pre[t] + red_pre[t + step]
            red_tot[t] = red_tot[t] + red_tot[t + step]
        barrier()
        step //= 2
    var pre = Int(red_pre[0])
    var total = Int(red_tot[0])
    var n_copy = total if total < need else need
    var r = b * FWS_T + t
    var j = nt - 1 - r if FROM_END else r
    var flag = False
    if r < nt:
        flag = flags.unsafe_load(j) != UInt8(0)
    var wsum = stack_allocation[
        FWS_WARPS, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var m = vote[FWS_VOTE](flag)
    var below = Int(pop_count(m & ((Scalar[FWS_VOTE](1) << Scalar[FWS_VOTE](lane)) - 1)))
    if lane == 0:
        wsum[warp] = Int32(pop_count(m))
    barrier()
    var wpre = 0
    for w in range(FWS_WARPS):
        if w < warp:
            wpre += Int(wsum[w])
    var rank = pre + wpre + below
    if flag and rank < need:
        var e = Int32(sorted_idx.unsafe_load(j))
        var slot = n_copy - 1 - rank if FROM_END else rank
        idx.unsafe_store(n_already + slot, e)
        selmap.unsafe_store(Int(e), UInt8(1))
    if b == 0 and t == 0:
        state.unsafe_store(Int(stage), Int32(n_copy))
