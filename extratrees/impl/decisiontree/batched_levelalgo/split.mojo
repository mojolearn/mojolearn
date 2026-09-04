# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The split record and its total order."""




@fieldwise_init
struct Split(ImplicitlyCopyable, Movable):
    """All info pertaining to splitting a node. `split.cuh:32-45`."""

    var quesval: Float32
    """Threshold to compare in this node. Theirs is `DataT quesval`."""

    var colid: Int32
    """Feature index. Theirs is `IdxT colid`."""

    var best_metric_val: Float32
    """Best info gain on this node. Theirs is `DataT best_metric_val`."""

    var n_left: Int32
    """Number of samples in the left child. Theirs is `int nLeft`."""

    comptime Min = Float32.MIN_FINITE
    """`split.cuh:36`: `-std::numeric_limits<DataT>::max()`."""

    def __init__(out self):
        """`split.cuh:54-59`, the default constructor."""
        self.quesval = Self.Min
        self.best_metric_val = Self.Min
        self.colid = -1
        self.n_left = 0

    def update(mut self, other: Self) -> Bool:
        """Updates the current split if the input gain is better."""
        var update_result = False
        if other.best_metric_val > self.best_metric_val:
            update_result = True
        elif other.best_metric_val == self.best_metric_val:
            if other.colid > self.colid:
                update_result = True
            elif other.colid == self.colid:
                if other.quesval > self.quesval:
                    update_result = True
        if update_result:
            self = other
        return update_result

    def is_valid(self) -> Bool:
        """Whether a candidate was ever recorded here."""
        return self.colid != -1



from std.atomic import Atomic, Ordering
from std.gpu import WARP_SIZE, block_dim, block_idx, grid_dim, lane_id, thread_idx
from std.gpu.primitives import warp
from std.memory import stack_allocation
from std.sys.compile import is_defined
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from extratrees.checks.pcg_rng import FNV1A32_BASIS, fmix32, fnv1a32



comptime ET_TIE_BREAK_KEYED = not is_defined["MOJOLEARN_ET_TIE_MAX_COLID"]()
"""DEVIATION 463's build switch."""

comptime SPLIT_TIE_SALT: UInt32 = 0x7B31C853
"""DEVIATION 463's salt."""


def split_tie_salt_for(tree_id: UInt32, node_id: UInt32) -> UInt32:
    """The per-node half of DEVIATION 463's rank key, computed ONCE per node. Host and device must call THIS function -- the builder stages one value per work item (`node_tie_salt`), the host splitter computes it at the top of the node loop -- so the rule cannot drift between the two."""
    var h = fnv1a32(FNV1A32_BASIS, SPLIT_TIE_SALT)
    h = fnv1a32(h, tree_id)
    return fnv1a32(h, node_id)


def split_tie_rank(tie_salt: UInt32, colid: Int32) -> UInt32:
    """DEVIATION 463's rank: one more fnv1a32 round for the column, then the `fmix32` avalanche (DEVIATION 464's lesson: an un-finalized fnv chain on a small word is a Weyl rotation, not a permutation)."""
    return fmix32(fnv1a32(tie_salt, colid.cast[DType.uint32]()))


def keyed_tie_wins(tie_salt: UInt32, cand: Split, best: Split) -> Bool:
    """Does `cand` beat `best` under DEVIATION 463's tie order?"""
    var rc = split_tie_rank(tie_salt, cand.colid)
    var rb = split_tie_rank(tie_salt, best.colid)
    if rc != rb:
        return rc > rb
    if cand.colid > best.colid:
        return True
    if cand.colid == best.colid:
        return cand.quesval > best.quesval
    return False


comptime SPLIT_SAB_NONE = 0
"""No sabotage. The shipping path."""

comptime SPLIT_SAB_FLOAT_KEY = 1
"""Drop the exact rational and reduce on `best_metric_val` alone -- cuML's own key, which DEVIATION 145 says is the wrong one for classification."""

comptime SPLIT_SAB_NO_COLID_ARM = 2
"""Drop `update`'s second arm (`other.colid > colid`, `split.cuh:85`)."""

comptime SPLIT_SAB_NO_QUESVAL_ARM = 3
"""Drop `update`'s third arm (`other.quesval > quesval`, `split.cuh:88`)."""

comptime SPLIT_SAB_NO_CROSS_WARP = 4
"""Warp 0 reads slot 0 for every lane, so only warp 0's own result survives the cross-warp combine (`split.cuh:126-127`)."""

comptime SPLIT_SAB_INVERT_GUARD = 5
"""Invert the publish guard at `split.cuh:143-146` -- write when `split_reg.update()` says NOT to, and not when it says to. Inverting is deterministic: the first block to reach a freshly seeded cell always improves it, so the inverted guard never writes, and every non-empty cell keeps its seed no matter which block gets there first."""

comptime SPLIT_SAB_NO_LOCK = 6
"""Publish without taking the mutex at all (`split.cuh:133-134`), with the read-modify-write window deliberately widened so the race is observable rather than merely possible."""

comptime SPLIT_SAB_BLOCK0_ONLY = 7
"""Publish only from block 0, so a node served by several blocks sees one block's slice."""

comptime SPLIT_SAB_MAX_COLID_TIE = 8
"""Restore the pre-463 tie-break: on an exact key tie the HIGHEST column id wins outright, no keyed rank."""


@fieldwise_init
struct ExactKey(ImplicitlyCopyable, Movable):
    """The exact rational a classification candidate is ordered by."""

    var num: Int64
    """`GiniProxyExact.num`. `sq_L*nR + sq_R*nL` for Gini."""

    var den: Int64
    """`GiniProxyExact.den`. `nL*nR`, strictly positive when `valid != 0`."""

    var valid: Int32
    """`GiniProxyExact.valid` as 0/1."""

    def __init__(out self):
        """The absent key."""
        self.num = 0
        self.den = 0
        self.valid = 0


def _mul_wide_u64(a: UInt64, b: UInt64, mut hi: UInt64, mut lo: UInt64):
    """The 128-bit product of two `UInt64`s as two `UInt64` limbs. DEVIATION BLOCK 167: `Int128` is what the host uses and what the Metal backend refuses to compile."""
    var a0 = a & 0xFFFFFFFF
    var a1 = a >> 32
    var b0 = b & 0xFFFFFFFF
    var b1 = b >> 32

    var t = a0 * b0
    var w0 = t & 0xFFFFFFFF
    var k = t >> 32

    t = a1 * b0 + k
    var w1 = t & 0xFFFFFFFF
    var w2 = t >> 32

    t = a0 * b1 + w1
    k = t >> 32

    lo = (t << 32) + w0
    hi = a1 * b1 + w2 + k


def _cmp_wide_products(a: UInt64, b: UInt64, c: UInt64, d: UInt64) -> Int32:
    """`a*b` against `c*d`, exactly, in 64-bit arithmetic. -1 / 0 / +1."""
    var hi1 = UInt64(0)
    var lo1 = UInt64(0)
    var hi2 = UInt64(0)
    var lo2 = UInt64(0)
    _mul_wide_u64(a, b, hi1, lo1)
    _mul_wide_u64(c, d, hi2, lo2)
    if hi1 != hi2:
        return Int32(1) if hi1 > hi2 else Int32(-1)
    if lo1 != lo2:
        return Int32(1) if lo1 > lo2 else Int32(-1)
    return Int32(0)


def compare_exact_key(a: ExactKey, b: ExactKey) -> Int32:
    """`GiniObjectiveFunction.CompareProxyExact`, without `Int128`."""
    if a.valid == 0 and b.valid == 0:
        return Int32(0)
    if a.valid == 0:
        return Int32(-1)
    if b.valid == 0:
        return Int32(1)

    var sa = Int32(0)
    if a.num > 0:
        sa = Int32(1)
    elif a.num < 0:
        sa = Int32(-1)
    var sb = Int32(0)
    if b.num > 0:
        sb = Int32(1)
    elif b.num < 0:
        sb = Int32(-1)
    if sa != sb:
        return Int32(1) if sa > sb else Int32(-1)
    if sa == 0:
        return Int32(0)

    var an = UInt64(a.num) if a.num > 0 else UInt64(-a.num)
    var bn = UInt64(b.num) if b.num > 0 else UInt64(-b.num)
    var mag = _cmp_wide_products(an, UInt64(b.den), bn, UInt64(a.den))
    if sa > 0:
        return mag
    return -mag


@fieldwise_init
struct SplitExact(ImplicitlyCopyable, Movable):
    """The record the device reduction actually moves."""

    var split: Split
    var key: ExactKey

    def __init__(out self):
        """`split.cuh:54-59` extended: their default `Split`, and the absent key."""
        self.split = Split()
        self.key = ExactKey()

    def update(mut self, other: Self, sabotage: Int32, tie_salt: UInt32) -> Bool:
        """Their `update` (`split.cuh:76-90`) with the exact key ahead of it."""
        var c = compare_exact_key(other.key, self.key)
        if sabotage == SPLIT_SAB_FLOAT_KEY:
            c = Int32(0)

        var update_result = False
        if c > 0:
            update_result = True
        elif c == 0:
            if (
                sabotage == SPLIT_SAB_NO_COLID_ARM
                or sabotage == SPLIT_SAB_NO_QUESVAL_ARM
            ):
                if other.split.best_metric_val > self.split.best_metric_val:
                    update_result = True
                elif other.split.best_metric_val == self.split.best_metric_val:
                    if (
                        other.split.colid > self.split.colid
                        and sabotage != SPLIT_SAB_NO_COLID_ARM
                    ):
                        update_result = True
                    elif other.split.colid == self.split.colid:
                        if (
                            other.split.quesval > self.split.quesval
                            and sabotage != SPLIT_SAB_NO_QUESVAL_ARM
                        ):
                            update_result = True
            else:
                if (
                    other.key.valid != 0
                    and self.key.valid != 0
                    and sabotage != SPLIT_SAB_FLOAT_KEY
                ):
                    var keyed = False
                    comptime if ET_TIE_BREAK_KEYED:
                        keyed = sabotage != SPLIT_SAB_MAX_COLID_TIE
                    if keyed:
                        update_result = keyed_tie_wins(
                            tie_salt, other.split, self.split
                        )
                    elif other.split.colid > self.split.colid:
                        update_result = True
                    elif other.split.colid == self.split.colid:
                        if other.split.quesval > self.split.quesval:
                            update_result = True
                else:
                    var probe = self.split
                    update_result = probe.update(other.split)

        if update_result:
            self = other
        return update_result


def split_warp_reduce(
    s: SplitExact, sabotage: Int32, tie_salt: UInt32
) -> SplitExact:
    """`Split::warpReduce`, `split.cuh:92-105`. Every lane of the warp must reach this function: the shuffles carry their own membership mask and a lane that never arrives hangs the rest."""
    var acc = s
    var off = UInt32(WARP_SIZE // 2)
    while off >= 1:
        var other = SplitExact(
            Split(
                warp.shuffle_xor(acc.split.quesval, off),
                warp.shuffle_xor(acc.split.colid, off),
                warp.shuffle_xor(acc.split.best_metric_val, off),
                warp.shuffle_xor(acc.split.n_left, off),
            ),
            ExactKey(
                warp.shuffle_xor(acc.key.num, off),
                warp.shuffle_xor(acc.key.den, off),
                warp.shuffle_xor(acc.key.valid, off),
            ),
        )
        _ = acc.update(other, sabotage, tie_salt)
        off //= 2
    return acc


def split_reduce_shared_bytes(tpb: Int, warp_size: Int) -> Int:
    """Shared memory `split_reduce_kernel[tpb]` needs, in bytes."""
    return (tpb // warp_size) * 36


@always_inline
def split_reduce_seed_at(
    out_quesval: MutPointer[Float32, MutAnyOrigin],
    out_colid: MutPointer[Int32, MutAnyOrigin],
    out_metric: MutPointer[Float32, MutAnyOrigin],
    out_nleft: MutPointer[Int32, MutAnyOrigin],
    out_num: MutPointer[Int64, MutAnyOrigin],
    out_den: MutPointer[Int64, MutAnyOrigin],
    out_valid: MutPointer[Int32, MutAnyOrigin],
    out_n_merges: MutPointer[Int32, MutAnyOrigin],
    out_n_warps: MutPointer[Int32, MutAnyOrigin],
    idx: Int,
):
    """Seed ONE reduce cell: `split_reduce_init_kernel`'s per-index body."""
    out_quesval[unsafe_offset=idx] = Split.Min
    out_colid[unsafe_offset=idx] = Int32(-1)
    out_metric[unsafe_offset=idx] = Split.Min
    out_nleft[unsafe_offset=idx] = Int32(0)
    out_num[unsafe_offset=idx] = Int64(0)
    out_den[unsafe_offset=idx] = Int64(0)
    out_valid[unsafe_offset=idx] = Int32(0)
    out_n_merges[unsafe_offset=idx] = Int32(0)
    out_n_warps[unsafe_offset=idx] = Int32(0)


def split_reduce_init_kernel(
    out_quesval: MutPointer[Float32, MutAnyOrigin],
    out_colid: MutPointer[Int32, MutAnyOrigin],
    out_metric: MutPointer[Float32, MutAnyOrigin],
    out_nleft: MutPointer[Int32, MutAnyOrigin],
    out_num: MutPointer[Int64, MutAnyOrigin],
    out_den: MutPointer[Int64, MutAnyOrigin],
    out_valid: MutPointer[Int32, MutAnyOrigin],
    out_n_merges: MutPointer[Int32, MutAnyOrigin],
    out_n_warps: MutPointer[Int32, MutAnyOrigin],
    len: Int32,
):
    """`initSplit`, `split.cuh:158-166`, as a grid-stride write-only map."""
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while idx < Int(len):
        split_reduce_seed_at(
            out_quesval,
            out_colid,
            out_metric,
            out_nleft,
            out_num,
            out_den,
            out_valid,
            out_n_merges,
            out_n_warps,
            idx,
        )
        idx += stride


def split_reduce_kernel[
    TPB: Int
](
    out_quesval: MutPointer[Float32, MutAnyOrigin],
    out_colid: MutPointer[Int32, MutAnyOrigin],
    out_metric: MutPointer[Float32, MutAnyOrigin],
    out_nleft: MutPointer[Int32, MutAnyOrigin],
    out_num: MutPointer[Int64, MutAnyOrigin],
    out_den: MutPointer[Int64, MutAnyOrigin],
    out_valid: MutPointer[Int32, MutAnyOrigin],
    out_n_merges: MutPointer[Int32, MutAnyOrigin],
    out_n_warps: MutPointer[Int32, MutAnyOrigin],
    mutexes: MutPointer[Int32, MutAnyOrigin],
    cand_quesval: MutPointer[Float32, MutAnyOrigin],
    cand_colid: MutPointer[Int32, MutAnyOrigin],
    cand_metric: MutPointer[Float32, MutAnyOrigin],
    cand_nleft: MutPointer[Int32, MutAnyOrigin],
    cand_num: MutPointer[Int64, MutAnyOrigin],
    cand_den: MutPointer[Int64, MutAnyOrigin],
    cand_valid: MutPointer[Int32, MutAnyOrigin],
    node_begin: MutPointer[Int32, MutAnyOrigin],
    node_count: MutPointer[Int32, MutAnyOrigin],
    node_tie_salt: MutPointer[UInt32, MutAnyOrigin],
    blocks_per_node_in: Int32,
    sabotage_in: Int32,
):
    """The best candidate per node, reduced on device. The caller prices it against the device's queried budget; a `comptime assert` below refuses a `TPB` that is not a whole number of warps, because `nWarps = blockDim.x / WarpSize` (`split.cuh:124`) silently becomes zero otherwise on a 64-wide wavefront."""
    comptime assert (
        TPB % WARP_SIZE == 0
    ), "split_reduce_kernel: TPB must be a whole number of warps on this target"
    comptime NW = TPB // WARP_SIZE

    var nid = Int(block_idx.y)
    var blk = Int(block_idx.x)
    var sab = sabotage_in
    var begin = Int(node_begin[unsafe_offset=nid])
    var count = Int(node_count[unsafe_offset=nid])
    var tie_salt = node_tie_salt[unsafe_offset=nid]
    var bpn = Int(blocks_per_node_in)

    var acc = SplitExact()
    var stride = TPB * bpn
    var end = begin + count
    var i = begin + Int(thread_idx.x) + blk * TPB
    while i < end:
        var cand = SplitExact(
            Split(
                cand_quesval[unsafe_offset=i],
                cand_colid[unsafe_offset=i],
                cand_metric[unsafe_offset=i],
                cand_nleft[unsafe_offset=i],
            ),
            ExactKey(
                cand_num[unsafe_offset=i],
                cand_den[unsafe_offset=i],
                cand_valid[unsafe_offset=i],
            ),
        )
        _ = acc.update(cand, sab, tie_salt)
        i += stride

    acc = split_warp_reduce(acc, sab, tie_salt)

    var lane = Int(lane_id())
    var warp_id = Int(thread_idx.x) // WARP_SIZE

    var s_quesval = stack_allocation[
        NW, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var s_colid = stack_allocation[
        NW, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var s_metric = stack_allocation[
        NW, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var s_nleft = stack_allocation[
        NW, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var s_num = stack_allocation[
        NW, Scalar[DType.int64], address_space = AddressSpace.SHARED
    ]()
    var s_den = stack_allocation[
        NW, Scalar[DType.int64], address_space = AddressSpace.SHARED
    ]()
    var s_valid = stack_allocation[
        NW, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()

    if lane == 0:
        s_quesval[unsafe_offset=warp_id] = acc.split.quesval
        s_colid[unsafe_offset=warp_id] = acc.split.colid
        s_metric[unsafe_offset=warp_id] = acc.split.best_metric_val
        s_nleft[unsafe_offset=warp_id] = acc.split.n_left
        s_num[unsafe_offset=warp_id] = acc.key.num
        s_den[unsafe_offset=warp_id] = acc.key.den
        s_valid[unsafe_offset=warp_id] = acc.key.valid

    barrier()

    if warp_id == 0:
        var src = lane
        if sab == SPLIT_SAB_NO_CROSS_WARP:
            src = 0
        if lane < NW:
            acc = SplitExact(
                Split(
                    s_quesval[unsafe_offset=src],
                    s_colid[unsafe_offset=src],
                    s_metric[unsafe_offset=src],
                    s_nleft[unsafe_offset=src],
                ),
                ExactKey(
                    s_num[unsafe_offset=src],
                    s_den[unsafe_offset=src],
                    s_valid[unsafe_offset=src],
                ),
            )
        else:
            acc = SplitExact()

        var mine = Int32(0)
        if lane < NW:
            if s_colid[unsafe_offset=lane] != Int32(-1):
                mine = Int32(1)
        var n_contrib = warp.sum(mine)

        acc = split_warp_reduce(acc, sab, tie_salt)

        if Int(thread_idx.x) == 0 and acc.split.is_valid():
            if not (sab == SPLIT_SAB_BLOCK0_ONLY and blk != 0):
                var slot = nid

                if sab != SPLIT_SAB_NO_LOCK:
                    while True:
                        if (
                            Atomic.load[ordering = Ordering.ACQUIRE](
                                mutexes.unsafe_offset(slot)
                            )
                            != Int32(0)
                        ):
                            continue
                        var expected = Int32(0)
                        if Atomic.compare_exchange[
                            success_ordering = Ordering.RELAXED,
                            failure_ordering = Ordering.RELAXED,
                            weak=True,
                        ](mutexes.unsafe_offset(slot), expected, Int32(1)):
                            break

                var cur = SplitExact(
                    Split(
                        out_quesval[unsafe_offset=slot],
                        out_colid[unsafe_offset=slot],
                        out_metric[unsafe_offset=slot],
                        out_nleft[unsafe_offset=slot],
                    ),
                    ExactKey(
                        out_num[unsafe_offset=slot],
                        out_den[unsafe_offset=slot],
                        out_valid[unsafe_offset=slot],
                    ),
                )

                if sab == SPLIT_SAB_NO_LOCK:
                    var spin = 0
                    while spin < 4096:
                        if Atomic.load[ordering = Ordering.RELAXED](
                            out_n_merges.unsafe_offset(slot)
                        ) == Int32(-424242):
                            spin += 1 << 20
                        spin += 1

                var moved = cur.update(acc, sab, tie_salt)
                if sab == SPLIT_SAB_INVERT_GUARD:
                    cur = acc
                    moved = not moved

                if moved:
                    out_quesval[unsafe_offset=slot] = cur.split.quesval
                    out_colid[unsafe_offset=slot] = cur.split.colid
                    out_metric[unsafe_offset=slot] = cur.split.best_metric_val
                    out_nleft[unsafe_offset=slot] = cur.split.n_left
                    out_num[unsafe_offset=slot] = cur.key.num
                    out_den[unsafe_offset=slot] = cur.key.den
                    out_valid[unsafe_offset=slot] = cur.key.valid

                out_n_merges[unsafe_offset=slot] = (
                    out_n_merges[unsafe_offset=slot] + Int32(1)
                )
                if n_contrib > out_n_warps[unsafe_offset=slot]:
                    out_n_warps[unsafe_offset=slot] = n_contrib

                if sab != SPLIT_SAB_NO_LOCK:
                    Atomic.store[ordering = Ordering.RELEASE](
                        mutexes.unsafe_offset(slot), Int32(0)
                    )


def split_tie_count_kernel(
    out_ties: MutPointer[Int32, MutAnyOrigin],
    win_colid: MutPointer[Int32, MutAnyOrigin],
    win_num: MutPointer[Int64, MutAnyOrigin],
    win_den: MutPointer[Int64, MutAnyOrigin],
    win_valid: MutPointer[Int32, MutAnyOrigin],
    cand_num: MutPointer[Int64, MutAnyOrigin],
    cand_den: MutPointer[Int64, MutAnyOrigin],
    cand_valid: MutPointer[Int32, MutAnyOrigin],
    node_begin: MutPointer[Int32, MutAnyOrigin],
    node_count: MutPointer[Int32, MutAnyOrigin],
    len: Int32,
):
    """DEVIATION 463's exact-tie counter, per node, AFTER the reduce."""
    var nid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while nid < Int(len):
        var ties = Int32(0)
        if win_colid[unsafe_offset=nid] >= 0:
            var wkey = ExactKey(
                win_num[unsafe_offset=nid],
                win_den[unsafe_offset=nid],
                win_valid[unsafe_offset=nid],
            )
            var b = Int(node_begin[unsafe_offset=nid])
            var e = b + Int(node_count[unsafe_offset=nid])
            for i in range(b, e):
                if cand_valid[unsafe_offset=i] == 0:
                    continue
                var ck = ExactKey(
                    cand_num[unsafe_offset=i],
                    cand_den[unsafe_offset=i],
                    cand_valid[unsafe_offset=i],
                )
                if compare_exact_key(ck, wkey) == Int32(0):
                    ties += 1
        out_ties[unsafe_offset=nid] = ties
        nid += stride
