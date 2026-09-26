"""FAST-only working-set selection without host round trips (Apple).

`WorkingSet.simple_select` gathers from the sorted order three times
(upper set from the front, lower set from the back, a fill from the front)
and each gather reads its count back to the host, which on Metal is a
drain per gather, two or three per SMO outer iteration. Here every count
stays on the device:

  * `fws_flags_kernel` (grid): the candidate flag of each SORTED position,
    `in_upper` / `in_lower` / any, and not yet selected (`selmap`, one
    byte per training index, written by earlier launches only);
  * `fws_walk_kernel` (ONE block): walks the flags in sorted order (from
    the back for the lower set), ranks candidates with a ballot per warp,
    and writes the same indices at the same working-set slots the
    host-counted gather writes; the running counts go to `state`, which
    the next launch reads.

A thread only reads device memory an EARLIER launch wrote, plus
threadgroup memory behind a barrier, so no in-kernel device-memory
ordering is relied on. The selection is the same list in the same order:
the first `need` candidates in sorted order (the last `need`, kept in
ascending order, for the lower set).
"""

from std.bit import pop_count
from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.primitives.warp import vote
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from svm.impl.smo_sets import in_lower, in_upper

comptime FWS_T = 1024
comptime FWS_WARPS = FWS_T // 32
comptime FWS_MAX_WS = 2048

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


def fws_walk_kernel[
    FROM_END: Bool
](
    flags: MutPointer[UInt8, MutAnyOrigin],
    sorted_idx: MutPointer[UInt32, MutAnyOrigin],
    nt_in: Int32,
    idx: MutPointer[Int32, MutAnyOrigin],
    selmap: MutPointer[UInt8, MutAnyOrigin],
    state: MutPointer[Int32, MutAnyOrigin],
    n_fifo_in: Int32,
    n_ws_in: Int32,
    stage: Int32,
):
    """Stage 0: the upper half (`(n_ws - n_fifo) // 2`) from the front.
    Stage 1: the lower set from the back, up to the rest. Stage 2: the
    fill from the front, if anything is still missing. `state[stage]`
    receives the count this stage selected."""
    var t = Int(thread_idx.x)
    var lane = t & 31
    var warp = t >> 5
    var nt = Int(nt_in)
    var n_ws = Int(n_ws_in)
    var n_already = Int(n_fifo_in)
    var need: Int
    if stage == Int32(0):
        need = (n_ws - n_already) // 2
    elif stage == Int32(1):
        n_already += Int(state.unsafe_load(0))
        need = n_ws - n_already
    else:
        n_already += Int(state.unsafe_load(0)) + Int(state.unsafe_load(1))
        need = n_ws - n_already
    if need <= 0:
        if t == 0:
            state.unsafe_store(Int(stage), Int32(0))
        return

    var wsum = stack_allocation[
        FWS_WARPS, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var picked = stack_allocation[
        FWS_MAX_WS, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var cnt = 0
    var base = 0
    while base < nt and cnt < need:
        var r = base + t
        var j = nt - 1 - r if FROM_END else r
        var flag = False
        if r < nt:
            flag = flags.unsafe_load(j) != UInt8(0)
        var m = vote[DType.uint32](flag)
        var below = Int(pop_count(m & ((UInt32(1) << UInt32(lane)) - 1)))
        if lane == 0:
            wsum[warp] = Int32(pop_count(m))
        barrier()
        var pre = 0
        var tot = 0
        for w in range(FWS_WARPS):
            var v = Int(wsum[w])
            if w < warp:
                pre += v
            tot += v
        barrier()
        var rank = cnt + pre + below
        if flag and rank < need:
            var e = Int32(sorted_idx.unsafe_load(j))
            comptime if FROM_END:
                picked[rank] = e
            else:
                idx.unsafe_store(n_already + rank, e)
                selmap.unsafe_store(Int(e), UInt8(1))
        cnt += tot
        base += FWS_T
    var n_copy = cnt if cnt < need else need
    comptime if FROM_END:
        barrier()
        var i = t
        while i < n_copy:
            var e = picked[n_copy - 1 - i]
            idx.unsafe_store(n_already + i, e)
            selmap.unsafe_store(Int(e), UInt8(1))
            i += FWS_T
    if t == 0:
        state.unsafe_store(Int(stage), Int32(n_copy))
