# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""OPTICS ordering kernels for the FAST tier on Apple (lane/apple-fast-optics2,
2026-10-03). `DeviceOps.optics_fast` (x_cluster/device_ops.mojo) launches them
under the `OPTICS_*` switches of x_cluster/optics.mojo; every build without a
define, IDENTICAL and the host binding never instantiate a kernel here. Only
the GPU binding imports this file.

The profile (docs/apple-fast/notes/optics2.md): main's ordering loop is two
launches per step (`optics_part_kernel`, `optics_step_kernel`), 20,000 for
the board's 10,000 rows, and at ~20 us a Metal launch that is the whole fit.

Every kernel keeps main's contract: the step's point is the unprocessed row
with the lowest reachability, the LOWEST INDEX on a tie (`order_key`: value
bits, then index), and an unprocessed row within `max_eps` takes
`max(dist, core)` when that is strictly lower (`optics_relax_cell`'s test).
Same picks, same words as `optics_order` and the host column.

`optics_batch_kernel` (MOJOLEARN_OPTICS_STEP_BATCH): OBG threadgroups of
OB_TPB threads run OB_STEPS steps per launch. Thread g (grid-wide) owns rows
g, g + OB_T, ...: their reach, pred and done words are read and written by
that thread only, so nothing n-sized crosses between threadgroups. A step
is: the block's min key over its unprocessed rows (`_block_min_key`: warp
shuffles, one 32-entry threadgroup pass), published to the step's parity
slot by thread 0 with a RELEASE store of the epoch (step + 1) to the
block's flag; the first warp ACQUIRE-spins on the OBG flags, reads the OBG
keys, and the min is the step's point (svm/impl/smoblocksolve.mojo's
`_publish` / `_await` protocol: the same thread writes the payload and
releases, so the release orders it; no block runs more than one step ahead
of the slowest, so a parity slot is never overwritten before every block
has read it). The spin is bounded (OB_SPIN_CAP polls): a block that waits
that long sets `fail` and leaves, every other block then does the same, and
`DeviceOps.optics_fast` reads the word and reruns the ordering with main's
two launches a step, so a grid that is not co-resident costs time, never a
wrong word. n / OB_STEPS launches instead of 2n.

`optics_fused_kernel` (MOJOLEARN_OPTICS_FRONTIER_DEVICE): one launch per
step. A block relaxes its SCAN_PER rows against the step's point (taken from
the partial minima the previous launch wrote) and emits the block's min key
over its still unprocessed rows for the next step. The partials are
double-buffered (step s reads part[s % 2], writes part[(s + 1) % 2]), so no
block can overwrite a partial another block has yet to read. n + 2 launches
instead of 2n + 1.

`SQ` (MOJOLEARN_OPTICS_CORE_SQ): the matrix holds SQUARED distances and the
cell is rooted at use with `sqrt_cell`'s expression (same input, same
function: the same word the sqrt pass would have stored).
"""
from std.atomic import Atomic, Ordering
from std.gpu import WARP_SIZE, block_dim, block_idx, thread_idx
from std.gpu.primitives.warp import shuffle_xor
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.numerics import identical_sqrt
from x_cluster.bodies import FPtr, IPtr
from x_cluster.device_post import RTPB, SCAN_PER, _block_min_u64
from x_cluster.post_bodies import KEY_NONE, order_key

comptime UPtr = MutPointer[UInt64, MutAnyOrigin]
comptime SharedU64 = UnsafePointer[UInt64, MutUntrackedOrigin, address_space = AddressSpace.SHARED]

#: the batch kernel's threadgroups: few enough to be co-resident on every
#: Apple GPU (the M1 has 8 cores), enough threads for one row each at the
#: board's 10k rows
comptime OBG = 8
#: threads of each (Apple's maximum)
comptime OB_TPB = 1024
#: warps of a threadgroup (32 at Apple's simdgroup width)
comptime OB_WARPS = OB_TPB // WARP_SIZE
#: grid-wide threads
comptime OB_T = OBG * OB_TPB
#: ACQUIRE polls before a block gives up on a flag and sets `fail`
comptime OB_SPIN_CAP = 1 << 22
#: ordering steps per batch launch: far under the ~4 s command-buffer cut at
#: any board size (a step is a few us), and few launches (20 at 10k rows)
comptime OB_STEPS = 512
#: the fused kernel's block: RTPB threads over SCAN_PER rows, the shape of
#: `optics_part_kernel`, so both write the same number of partials
comptime OF_TPB = RTPB
comptime OF_PER = SCAN_PER


@always_inline
def _dd[SQ: Bool](dist: FPtr, idx: Int) -> Float32:
    """The distance cell, rooted at use when the matrix holds squares
    (`bodies.sqrt_cell`: `identical_sqrt(max(v, 0))`)."""
    var v = dist[idx]
    comptime if SQ:
        return identical_sqrt(v if v > Float32(0) else Float32(0))
    else:
        return v


@always_inline
def _key_lt(ah: UInt32, al: UInt32, bh: UInt32, bl: UInt32) -> Bool:
    """(ah, al) < (bh, bl): `order_key`'s order on its two halves."""
    return ah < bh or (ah == bh and al < bl)


@always_inline
def _warp_min_key(mut hi: UInt32, mut lo: UInt32):
    """Butterfly min of a 64-bit key held as two 32-bit halves; every lane
    ends with the warp's minimum. Every lane reaches every shuffle: the bound
    is comptime and nothing here is conditional."""
    var off = 1
    while off < WARP_SIZE:
        var oh = shuffle_xor(hi, UInt32(off))
        var ol = shuffle_xor(lo, UInt32(off))
        if _key_lt(oh, ol, hi, lo):
            hi = oh
            lo = ol
        off *= 2


@always_inline
def _block_min_key(red: SharedU64, mine: UInt64) -> UInt64:
    """The min key over the OB_TPB threads: warps by shuffles, the OB_WARPS
    warp minima by the first warp; every thread returns it."""
    var tid = Int(thread_idx.x)
    var lane = tid % WARP_SIZE
    var wid = tid // WARP_SIZE
    var hi = UInt32(mine >> 32)
    var lo = UInt32(mine & UInt64(0xFFFFFFFF))
    _warp_min_key(hi, lo)
    if lane == 0:
        red[wid] = (UInt64(hi) << 32) | UInt64(lo)
    barrier()
    if wid == 0:
        var k = red[lane] if lane < OB_WARPS else KEY_NONE
        var h2 = UInt32(k >> 32)
        var l2 = UInt32(k & UInt64(0xFFFFFFFF))
        _warp_min_key(h2, l2)
        if lane == 0:
            red[0] = (UInt64(h2) << 32) | UInt64(l2)
    barrier()
    var r = red[0]
    barrier()  # every thread has read red[0] before the next round's lane-0 writes
    return r


def optics_init2_kernel(
    core: FPtr, n: Int32, max_eps: Float32, reach: FPtr, pred: IPtr, done: IPtr, core_copy: FPtr, copy: Int32
):
    """`optics_init_kernel` (the core clamp to +inf past `max_eps`, +inf
    reachabilities, -1 predecessors, nothing processed) plus, with `copy`,
    the clamped core written into `core_copy` (OPTICS_LIVEBUF's paired
    readback buffer)."""
    var i = Int(block_idx.x) * OF_TPB + Int(thread_idx.x)
    if i < Int(n):
        var inf = Float32.MAX * Float32(2)
        var c = core[i]
        if c > max_eps:
            c = inf
            core[i] = inf
        reach[i] = inf
        pred[i] = -1
        done[i] = 0
        if copy != Int32(0):
            core_copy[i] = c


@always_inline
def _warp_all(ok: Bool) -> Bool:
    """Every lane's `ok` (a butterfly AND; every lane reaches every shuffle)."""
    var v = UInt32(1) if ok else UInt32(0)
    var off = 1
    while off < WARP_SIZE:
        v = v & shuffle_xor(v, UInt32(off))
        off *= 2
    return v != UInt32(0)


def optics_batch_kernel[SQ: Bool](
    dist: FPtr, core: FPtr, n: Int32, max_eps: Float32, done: IPtr, reach: FPtr, pred: IPtr,
    ordering: IPtr, step0: Int32, steps: Int32, xk: UPtr, flags: IPtr, fail: IPtr,
):
    """Steps step0 .. step0 + steps - 1 of the ordering by OBG threadgroups
    (grid OBG, block OB_TPB). Thread g owns rows i == g (mod OB_T). `xk`:
    2 * OBG key slots (parity, block); `flags`: OBG epochs, zero before the
    first launch and only growing; `fail`: one word, set when a wait hits
    OB_SPIN_CAP."""
    var red = stack_allocation[OB_WARPS, UInt64, address_space = AddressSpace.SHARED]()
    var good = stack_allocation[1, Int32, address_space = AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var blk = Int(block_idx.x)
    var g = blk * OB_TPB + tid
    var N = Int(n)
    var inf = Float32.MAX * Float32(2)
    for s in range(Int(steps)):
        var step = Int(step0) + s
        var epoch = Int32(step + 1)
        var mine = KEY_NONE
        for i in range(g, N, OB_T):
            if done[i] == Int32(0):
                mine = min(mine, order_key(reach[i], i))
        var r = _block_min_key(red, mine)
        if tid == 0:
            # the block's key into this step's parity slot, then the epoch:
            # the same thread writes the payload and releases
            xk[(step & 1) * OBG + blk] = r
            Atomic.store[ordering = Ordering.RELEASE](flags.unsafe_offset(blk), epoch)
        if tid < WARP_SIZE:
            var ok = True
            var gk = KEY_NONE
            if tid < OBG:
                var polls = 0
                while Atomic.load[ordering = Ordering.ACQUIRE](flags.unsafe_offset(tid)) < epoch:
                    polls += 1
                    if polls >= OB_SPIN_CAP:
                        ok = False
                        break
                if ok:
                    gk = xk[(step & 1) * OBG + tid]
            var hi = UInt32(gk >> 32)
            var lo = UInt32(gk & UInt64(0xFFFFFFFF))
            _warp_min_key(hi, lo)
            var all_ok = _warp_all(ok)
            if tid == 0:
                red[0] = (UInt64(hi) << 32) | UInt64(lo)
                good[0] = Int32(1) if all_ok else Int32(0)
        barrier()
        var r2 = red[0]
        var is_good = good[0] != Int32(0)
        barrier()
        if not is_good:
            if tid == 0:
                Atomic.store[ordering = Ordering.RELEASE](fail, Int32(1))
            return  # uniform in the block; every other block times out the same way
        if r2 == KEY_NONE:
            return  # every row is processed
        var point = Int(UInt32(r2 & UInt64(0xFFFFFFFF)))
        if g == 0:
            ordering[step] = Int32(point)
        if point % OB_T == g:
            done[point] = Int32(1)  # the owner marks its own row
        var cp = core[point]
        if cp != inf:
            var row = point * N
            for i in range(g, N, OB_T):
                if i != point and done[i] == Int32(0):
                    var dd = _dd[SQ](dist, row + i)
                    if dd <= max_eps:
                        var rd = dd if dd > cp else cp
                        if rd < reach[i]:
                            reach[i] = rd
                            pred[i] = Int32(point)


def optics_step2_kernel[SQ: Bool](
    part: UPtr, nb: Int32, dist: FPtr, core: FPtr, n: Int32, max_eps: Float32, done: IPtr, reach: FPtr,
    pred: IPtr, ordering: IPtr, step: Int32,
):
    """`optics_step_kernel` with the cell rooted at use (OPTICS_CORE_SQ
    without STEP_BATCH or FRONTIER_DEVICE): the point from the partials,
    booked by thread 0, one row relaxed per thread. Grid pgrid(n), block PTPB."""
    var o = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var r = KEY_NONE
    for q in range(Int(nb)):
        r = min(r, part[q])
    if r == KEY_NONE:
        return
    var point = Int(UInt32(r & UInt64(0xFFFFFFFF)))
    if o == 0:
        done[point] = Int32(1)
        ordering[Int(step)] = Int32(point)
    var N = Int(n)
    if o >= N or o == point or done[o] != Int32(0):
        return
    var cp = core[point]
    if cp == Float32.MAX * Float32(2):
        return
    var dd = _dd[SQ](dist, point * N + o)
    if not (dd <= max_eps):
        return
    var rd = dd if dd > cp else cp
    if rd < reach[o]:
        reach[o] = rd
        pred[o] = Int32(point)


def optics_fused_kernel[SQ: Bool](
    part_in: UPtr, part_out: UPtr, nb: Int32, dist: FPtr, core: FPtr, n: Int32, max_eps: Float32,
    done: IPtr, reach: FPtr, pred: IPtr, ordering: IPtr, step: Int32,
):
    """One step: the point from `part_in` (every thread re-reduces the nb
    partials, as `optics_step_kernel` does), block 0's thread 0 books it,
    block b relaxes rows [b * OF_PER, (b + 1) * OF_PER) and writes
    part_out[b] = the min key over those still unprocessed (the next step's
    `part_in`). Grid nb = ceil(n / OF_PER), block OF_TPB."""
    var red = stack_allocation[OF_TPB, UInt64, address_space = AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var b = Int(block_idx.x)
    var N = Int(n)
    var r = KEY_NONE
    for q in range(Int(nb)):
        r = min(r, part_in[q])
    var mine = KEY_NONE
    if r != KEY_NONE:
        var point = Int(UInt32(r & UInt64(0xFFFFFFFF)))
        if b == 0 and tid == 0:
            done[point] = Int32(1)
            ordering[Int(step)] = Int32(point)
        var cp = core[point]
        var inf = Float32.MAX * Float32(2)
        var row = point * N
        var base = b * OF_PER
        var end = min(base + OF_PER, N)
        for i in range(base + tid, end, OF_TPB):
            if i != point and done[i] == Int32(0):
                var rr = reach[i]
                if cp != inf:
                    var dd = _dd[SQ](dist, row + i)
                    if dd <= max_eps:
                        var rd = dd if dd > cp else cp
                        if rd < rr:
                            rr = rd
                            reach[i] = rd
                            pred[i] = Int32(point)
                mine = min(mine, order_key(rr, i))
    var m = _block_min_u64(red, mine)
    if tid == 0:
        part_out[b] = m
