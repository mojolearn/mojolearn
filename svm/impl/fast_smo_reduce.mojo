"""FAST-only block reductions for the SMO block solve (Apple).

The block solve spends each inner iteration in three 1024-wide
arg-reductions. The pinned schedules fold the 32 warp partials with a
31-step serial loop and protect each shared slot with a trailing barrier.
These fold both levels with `shuffle_xor` butterflies (the second over the
warp slots, log2(warps) steps, every warp redundantly, no broadcast), fuse
the `f_u` argmin and the `f_max` argmax into one pass, and take their
shared slots from the caller so each call site owns distinct memory and
needs no trailing barrier (the next write to a slot is always behind a
later barrier every reader has passed).

The order is the same total order as `pinned_argreduce` (value, then the
smaller key; keys are unique for active threads) with the thread as a last
tie-break, so the same element wins as in every pinned schedule.
"""

from std.gpu import thread_idx, lane_id, WARP_SIZE
from std.gpu.primitives.warp import shuffle_xor
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier


@always_inline
def _better[
    MAX: Bool
](ov: Float32, ok: Int32, ot: Int32, mv: Float32, mk: Int32, mt: Int32) -> Bool:
    """Does (ov, ok, ot) beat (mv, mk, mt)? Value, then smaller key, then
    smaller thread (only padding entries tie on value and key)."""
    var tie = ov == mv and (ok < mk or (ok == mk and ot < mt))
    comptime if MAX:
        return ov > mv or tie
    else:
        return ov < mv or tie


@always_inline
def _warp_fold2[
    STEPS_TO: Int
](
    mut nv: Float32, mut nk: Int32, mut nt: Int32,
    mut xv: Float32, mut xk: Int32, mut xt: Int32,
):
    """Butterfly over offsets 1 .. STEPS_TO/2: min triple and max triple
    folded together (independent chains, interleaved)."""
    comptime for s in range(6):
        comptime off = 1 << s
        comptime if off < STEPS_TO:
            var anv = shuffle_xor(nv, UInt32(off))
            var ank = shuffle_xor(nk, UInt32(off))
            var ant = shuffle_xor(nt, UInt32(off))
            var axv = shuffle_xor(xv, UInt32(off))
            var axk = shuffle_xor(xk, UInt32(off))
            var axt = shuffle_xor(xt, UInt32(off))
            if _better[False](anv, ank, ant, nv, nk, nt):
                nv = anv
                nk = ank
                nt = ant
            if _better[True](axv, axk, axt, xv, xk, xt):
                xv = axv
                xk = axk
                xt = axt


@always_inline
def _warp_fold1[
    MAX: Bool, STEPS_TO: Int
](mut v: Float32, mut k: Int32, mut t: Int32):
    comptime for s in range(6):
        comptime off = 1 << s
        comptime if off < STEPS_TO:
            var ov = shuffle_xor(v, UInt32(off))
            var ok = shuffle_xor(k, UInt32(off))
            var ot = shuffle_xor(t, UInt32(off))
            if _better[MAX](ov, ok, ot, v, k, t):
                v = ov
                k = ok
                t = ot


@always_inline
def fast_argmin_argmax[
    o1: MutOrigin, o2: MutOrigin, o3: MutOrigin, //, block_size: Int
](
    vmin: Float32,
    vmax: Float32,
    key: Int32,
    sv: MutPointer[Float32, o1, address_space = AddressSpace.SHARED],
    sk: MutPointer[Int32, o2, address_space = AddressSpace.SHARED],
    st: MutPointer[Int32, o3, address_space = AddressSpace.SHARED],
) -> Tuple[Float32, Int32, Float32]:
    """(f_u, thread of f_u, f_max) to every thread. `sv`, `sk`, `st` hold
    2 * block_size / WARP_SIZE slots owned by this call site."""
    comptime WARPS = block_size // WARP_SIZE
    var tid = Int32(thread_idx.x)
    var nv = vmin
    var nk = key
    var nt = tid
    var xv = vmax
    var xk = key
    var xt = tid
    _warp_fold2[WARP_SIZE](nv, nk, nt, xv, xk, xt)
    var lane = Int(lane_id())
    if lane == 0:
        var w = Int(tid) // WARP_SIZE
        sv[w] = nv
        sk[w] = nk
        st[w] = nt
        sv[WARPS + w] = xv
        sk[WARPS + w] = xk
        st[WARPS + w] = xt
    barrier()
    comptime if WARPS > 1:
        var j = lane % WARPS
        nv = sv[j]
        nk = sk[j]
        nt = st[j]
        xv = sv[WARPS + j]
        xk = sk[WARPS + j]
        xt = st[WARPS + j]
        _warp_fold2[WARPS](nv, nk, nt, xv, xk, xt)
    else:
        nv = sv[0]
        nt = st[0]
        xv = sv[1]
    return (nv, nt, xv)


@always_inline
def fast_argext[
    o1: MutOrigin, o2: MutOrigin, o3: MutOrigin, //, block_size: Int, MAX: Bool
](
    value: Float32,
    key: Int32,
    sv: MutPointer[Float32, o1, address_space = AddressSpace.SHARED],
    sk: MutPointer[Int32, o2, address_space = AddressSpace.SHARED],
    st: MutPointer[Int32, o3, address_space = AddressSpace.SHARED],
) -> Tuple[Float32, Int32]:
    """(value, thread) of the arg-extremum to every thread; `block_size /
    WARP_SIZE` slots owned by this call site."""
    comptime WARPS = block_size // WARP_SIZE
    var tid = Int32(thread_idx.x)
    var v = value
    var k = key
    var t = tid
    _warp_fold1[MAX, WARP_SIZE](v, k, t)
    var lane = Int(lane_id())
    if lane == 0:
        var w = Int(tid) // WARP_SIZE
        sv[w] = v
        sk[w] = k
        st[w] = t
    barrier()
    comptime if WARPS > 1:
        var j = lane % WARPS
        v = sv[j]
        k = sk[j]
        t = st[j]
        _warp_fold1[MAX, WARPS](v, k, t)
    else:
        v = sv[0]
        t = st[0]
    return (v, t)
