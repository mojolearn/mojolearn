"""The split candidate, and the total order that makes a forest reproducible.

MIRRORS `cpp/src/decisiontree/batched-levelalgo/split.cuh` at rapidsai/cuml
`v26.08.00` (`265b9da6a0e75dbef071a3168398b993a5ff6f0e`), checked out
read-only at `~/CascadeProjects/upstream/cuml-v26.08.00`.

This is the smallest file in the learner and the one the whole
reproducibility story rests on, so it is worth saying plainly what it does
before reading it.

The histogram kernel produces, per (node, column, bin), a gain. Those
candidates are reduced -- across lanes of a warp, across warps of a block,
and finally across BLOCKS through a mutex-guarded global slot -- down to one
winner per node. A reduction is only reproducible if its operator does not
care about the order the operands arrive in, and `Split::update`
(`split.cuh:142-191`) is written to be exactly that: a lexicographic maximum

    1. higher `best_metric_val` wins                        (`:150-157`)
    2. tie -> higher `colid` wins                           (`:161-168`)
    3. tie -> merge the equivalent threshold RANGE if the
       two candidates route the same training rows          (`:174-177`)
    4. tie -> higher `quesval` wins                          (`:181-188`)

Steps 1, 2 and 4 are a plain maximum on a total order, and a maximum is
associative, commutative and idempotent, so the answer does not depend on
grouping. **Step 3 is not**, and that is a real property of their code
rather than of this port -- see the DEVIATION BLOCK below, which carries a
worked counterexample and an OPEN item.

The other half of the story is that the histogram counts feeding these
gains are INTEGER counts accumulated with an integer atomic
(`bins.cuh:29-32`), so the histogram itself is already order-independent.
Integer histogram plus total-order tie-break is why the unweighted
classification path can be bit-identical across Metal, CUDA and HIP with no
fixed point and no dither.

ONE SEMANTIC WORTH KNOWING BEFORE READING THE CODE, because reasoning about
it produced the wrong answer here and a run produced the right one.
`select_split_range_midpoint` (`:126-134`) does not only choose a bin, it
then ASSIGNS `quesval = quantiles[bin]`. So on a unit range `[b, b]` the
rule is the identity on the BIN INDEX and is NOT the identity on the
THRESHOLD. What follows is that the `quesval` a candidate carries through
the reduction exists only as a tie-break key (`:181-188`); the threshold
finally published for the node is always read back out of the quantiles
array. `ensemble/mojo_only/split_check.mojo` arm D(iii) holds that to a
per-node count -- it was written predicting no movement, measured 35 of 37,
and the measurement was right.

================= DEVIATION BLOCK (whole file) =================

DEVIATION 104. `raft::WarpSize` is a hardcoded 32 in their source
(`split.cuh:210, 236-238`). This port does not transcribe the constant; it
uses Mojo's queried `WARP_SIZE`, per this repository's standing rule that
no wavefront width may be assumed (32 on NVIDIA and Apple, 64 on AMD CDNA,
32 on AMD RDNA).

The reduction SHAPE is transcribed exactly: their loop is
`for (i = WarpSize/2; i >= 1; i /= 2) { update(shfl(field, lane + i)); }`
(`:209-219`), a rotate-and-reduce where `lane + i` wraps modulo the warp
width, so after the loop EVERY lane holds the reduction of the whole warp.
`std.gpu.primitives.warp.shuffle_idx` is the Mojo spelling of `raft::shfl`
-- a language-level counterpart to `__shfl_sync`, not a library standing in
for an algorithm -- and this repository's `vendor_correctness_check`
already holds it to `raft::shfl`'s behaviour.

PRICE, and it is not zero: at width 32 this reproduces their grouping
exactly. At width 64 the grouping differs, and by the non-associativity
recorded under DEVIATION 105 below, a different grouping can select a
different split in one narrow tie class. Everywhere outside that class the
answer is identical at any width.

DEVIATION 105. THEIR REDUCTION OPERATOR IS NOT ASSOCIATIVE, AND THEIR OWN
CROSS-BLOCK REDUCTION ORDER IS ARBITRARY. Recorded here because it bounds
what this port is allowed to claim.

`update` merges an equivalent split RANGE (`:174-177`) only when the two
candidates agree on `global_nLeft`; otherwise it falls through to the
`quesval` maximum (`:181-188`). Different groupings of three candidates can
therefore reach different answers. A worked counterexample, all with equal
gain and equal `colid`:

    X  : global_nLeft 5, range [1,1], quesval 10
    X2 : global_nLeft 5, range [3,3], quesval 30
    Y  : global_nLeft 7, range [2,2], quesval 20

    ((X + X2) + Y) : X and X2 merge to range [1,3] quesval 30; Y's
                     nLeft differs and its quesval 20 loses.
                     -> range [1,3], and `select_split_range_midpoint`
                        (`:126-134`) picks bin 1 + (3-1+1)/2 = 2.
    ((X + Y) + X2) : Y beats X on quesval; X2's nLeft differs from Y's
                     and its quesval 30 wins outright.
                     -> range [3,3], midpoint picks bin 3.

Two different chosen thresholds from the same set of candidates. Reaching
it needs equal gain and equal column with BOTH an equal-`nLeft` pair and an
unequal-`nLeft` third -- rare, but constructible, and `evalBestSplit`
(`:232-274`) merges blocks into the global slot under
`while (atomicCAS(mutex, 0, 1));`, which orders blocks by arrival. Their
comment at `:123-125` says the midpoint rule exists "so deterministic
tie-breaking does not pick an edge", so determinism is plainly their
INTENT.

THIS PORT DOES NOT ACT ON THIS. Their structure is transcribed verbatim,
non-associativity included, because copying is the charter and because a
"fix" here would be an invention that silently diverges from their answer
in the common case too. It is recorded as an OPEN item in
`ensemble/PLAN.md` to be settled the only way this repository settles
anything -- by running THEIR binary on a constructed fixture in this tie
class on the NVIDIA column and reading their per-cell output -- not by
reasoning, which is what produced the paragraph you are reading.

DEVIATION 106. Their `atomicCAS` / `__threadfence()` / `atomicExch` mutex
(`:251`, `:270-271`) is not expressible on Metal in that spelling: Mojo 1.0
comptime-asserts that `threadfence` "is only implemented on NVIDIA GPUs",
the Apple backend rejects strong compare-exchange by name, and it rejects
`acquire` success ordering on a compare-exchange. This port therefore uses
the translation this repository already established and enqueued for
cuVS's own cross-block mutex (`neighbors/mutex_probe_main.mojo`): spin on
an ACQUIRE load until the mutex reads free, claim it with a WEAK RELAXED
compare-exchange, hand it back with a RELEASE store. The synchronizes-with
edge is a load-acquire observing a store-release, which is the C++11
statement of what their CAS-spin plus `__threadfence` accomplishes -- CUDA
defines `__threadfence()` as `cuda::atomic_thread_fence(seq_cst,
thread_scope_device)`. Only the mutex HOLDER ever writes the release value
and their own code discards the exchanged value too, so no ABA hides in the
relaxed claim. This changes HOW the handoff is said, never WHAT is said.

DEVIATION 107. `printSplits` (`:291-308`) is a debug printer built on
`raft::linalg::writeOnlyUnaryOp` and is NOT ported. Price of declining it:
one debug aid, replaceable by a host-side copy and print at any call site
that wants it. `initSplit` (`:284-289`) IS ported, because the builder
needs it every level.

NOT A DEVIATION, recorded so a reader does not go looking: `local_nLeft` is
`std::int64_t` in their struct (`:51`) and is written by an `atomicAdd` on
`unsigned long long` in `countLocalLeftKernel`
(`kernels/builder_kernels_impl.cuh:79-81`). 64-bit integer atomics are a
hard COMPILE error on Apple GPU (measured this session:
`ensemble/mojo_only/atomic_width_probe.mojo`). The field keeps its Int64
width HERE, because nothing in this file atomically adds to it; the atomic
width question belongs to the kernel that does, and is that lane's to
record. What matters for this file is that `local_nLeft` takes no part in
split SELECTION at all -- their comment at `:139-140` says so: "local_nLeft
is intentionally not accepted here: split selection is based on global
counts, and local counts are filled just before partitioning."
=================================================================
"""

from std.atomic import Atomic, Ordering
from max.gpu.memory import AddressSpace

# `split.cuh:8` includes `bins.cuh`, because `detail::CountLeft` is
# bin-typed. Same direction here; `bins.mojo` imports nothing from this
# file, so there is no cycle -- exactly their arrangement.
from ensemble.decisiontree.batched_levelalgo.bins import Bin
from std.gpu import WARP_SIZE, block_dim, thread_idx
from std.gpu.primitives.id import lane_id
from std.gpu.primitives.warp import shuffle_idx
from max.gpu.sync import barrier


@always_inline
def count_left[
    BinT: Bin, ho: MutOrigin, aspace: AddressSpace, //
](
    hist: MutPointer[BinT, ho, address_space=aspace],
    i: Int32,
    n_bins: Int32,
    n_outputs: Int32,
) -> Int64:
    """`detail::CountLeft`, `split.cuh:19-27`.

    Their loop sums `hist[n_bins * j + i].Count()` over the output classes,
    starting from `j = 0` and adding `j = 1 ..< n_outputs`. The histogram is
    CUMULATIVE by the time this runs, so cell `i` already holds the left
    partition's count.

    IT IS BIN-TYPED, and the first version of this file got that wrong. It
    took a plane of `UInt32` counts, which no caller could use: their
    `CountLeft` takes `BinT const*` and calls `.Count()`, and `Gain`
    (`objectives.cuh:168`, `:369`) hands it the bin histogram directly. The
    wrong-shaped version sat here unreached while `objectives.mojo` carried
    a private duplicate of the right one -- an unported file is visible, a
    MIS-ported one is not, and an unreached one hides the mismatch. Their
    layout is restored: `CountLeft` lives in `split.cuh` (which
    `#include`s `bins.cuh` at `:8`) and `objectives.cuh` calls it.

    Their accumulator is `BinCountT` (`unsigned long long`) widened to
    `std::int64_t` at `:26`. The sum over classes at one bin is the row
    count of that partition, bounded by `n_sampled_rows`.
    """
    var n_left = UInt64(Int(hist[unsafe_offset = Int(i)].Count()))
    for j in range(1, Int(n_outputs)):
        n_left += UInt64(
            Int(hist[unsafe_offset = j * Int(n_bins) + Int(i)].Count())
        )
    return Int64(Int(n_left))


struct Split[dtype: DType](TrivialRegisterPassable):
    """`ML::DT::Split<DataT, IdxT>`, `split.cuh:36-275`, with IdxT = Int32.

    IdxT is instantiated as `int` throughout cuML's RF, so every `IdxT`
    field below is Int32 and every `std::int64_t` field is Int64. The
    widths are theirs.
    """

    # `split.cuh:43` -- threshold to compare in this node
    var quesval: Scalar[Self.dtype]
    # `split.cuh:45` -- feature index
    var colid: Int32
    # `split.cuh:47` -- best info gain on this node
    var best_metric_val: Scalar[Self.dtype]
    # `split.cuh:49` -- global number of samples in the left child
    var global_nLeft: Int64
    # `split.cuh:51` -- rank-local number of samples in the left child
    var local_nLeft: Int64
    # `split.cuh:53` -- first quantile index of an inclusive range of
    # training-equivalent splits
    var split_start: Int32
    # `split.cuh:55` -- last quantile index of that range
    var split_end: Int32

    @staticmethod
    @always_inline
    def Min() -> Scalar[Self.dtype]:
        """`split.cuh:40`, `-std::numeric_limits<DataT>::max()`.

        Note this is `-max()`, NOT `lowest()` and NOT `-infinity`. For an
        IEEE float the two are the same value, but the objectives return
        this exact expression on every rejected candidate
        (`objectives.cuh:53`, `:88`, `:130`, ...), so the comparison
        `other_best_metric_val > best_metric_val` at `:150` is an equality
        between two identical bit patterns, not a near-miss. Spelling it
        the same way is what keeps that true.
        """
        return -Scalar[Self.dtype].MAX_FINITE

    @always_inline
    def __init__(out self):
        """`split.cuh:57-65`, their default constructor."""
        self.quesval = Self.Min()
        self.best_metric_val = Self.Min()
        self.colid = -1
        self.global_nLeft = 0
        self.local_nLeft = 0
        self.split_start = -1
        self.split_end = -1

    @always_inline
    def __init__(
        out self,
        quesval: Scalar[Self.dtype],
        colid: Int32,
        best_metric_val: Scalar[Self.dtype],
        global_nLeft: Int64,
        local_nLeft: Int64,
        split_start: Int32,
        split_end: Int32,
    ):
        self.quesval = quesval
        self.colid = colid
        self.best_metric_val = best_metric_val
        self.global_nLeft = global_nLeft
        self.local_nLeft = local_nLeft
        self.split_start = split_start
        self.split_end = split_end

    @always_inline
    def IsValid(self) -> Bool:
        """`split.cuh:79`."""
        return self.colid != Int32(-1)

    @always_inline
    def has_valid_split_range(self) -> Bool:
        """`split.cuh:81-84`."""
        return self.split_start >= Int32(0) and self.split_end >= self.split_start

    @always_inline
    def can_merge_equivalent_split_range(
        self,
        other_global_nLeft: Int64,
        other_split_start: Int32,
        other_split_end: Int32,
    ) -> Bool:
        """`split.cuh:86-92`."""
        return (
            self.global_nLeft == other_global_nLeft
            and self.has_valid_split_range()
            and other_split_start >= Int32(0)
            and other_split_end >= other_split_start
        )

    @always_inline
    def merge_equivalent_split_range(
        mut self,
        other_quesval: Scalar[Self.dtype],
        other_split_start: Int32,
        other_split_end: Int32,
    ):
        """`split.cuh:97-104`.

        Their comment at `:94-96`: "`quesval` tracks the upper edge only to
        preserve the existing threshold tie-break against candidates
        outside this equivalent range."
        """
        self.split_start = (
            other_split_start if other_split_start
            < self.split_start else self.split_start
        )
        self.split_end = (
            other_split_end if other_split_end
            > self.split_end else self.split_end
        )
        if other_quesval > self.quesval:
            self.quesval = other_quesval

    @always_inline
    def replace_with(
        mut self,
        other_quesval: Scalar[Self.dtype],
        other_colid: Int32,
        other_best_metric_val: Scalar[Self.dtype],
        other_global_nLeft: Int64,
        other_split_start: Int32,
        other_split_end: Int32,
    ) -> Bool:
        """`split.cuh:106-121`. Note `local_nLeft` is reset to 0 at `:117`.
        """
        self.quesval = other_quesval
        self.colid = other_colid
        self.best_metric_val = other_best_metric_val
        self.global_nLeft = other_global_nLeft
        self.local_nLeft = 0
        self.split_start = other_split_start
        self.split_end = other_split_end
        return True

    @always_inline
    def select_split_range_midpoint[
        qo: MutOrigin, //
    ](
        mut self,
        quantiles: MutPointer[Scalar[Self.dtype], qo],
        n_bins: Int32,
    ):
        """`split.cuh:126-134`.

        Their comment at `:123-125`: "Several thresholds can be equally
        good for the training data while still routing future inference
        values differently. Select the middle split in that equivalent
        range so deterministic tie-breaking does not pick an edge."

        The `(end - start + 1) / 2` is C++ integer division on values that
        are non-negative here (the `has_valid_split_range()` guard), so
        Mojo's `//` matches it.
        """
        if self.has_valid_split_range() and self.split_end < n_bins:
            var bin = self.split_start + (
                self.split_end - self.split_start + Int32(1)
            ) // Int32(2)
            self.quesval = quantiles[unsafe_offset = Int(bin)]
            self.split_start = bin
            self.split_end = bin

    @always_inline
    def update(
        mut self,
        other_quesval: Scalar[Self.dtype],
        other_colid: Int32,
        other_best_metric_val: Scalar[Self.dtype],
        other_global_nLeft: Int64,
        other_split_start: Int32,
        other_split_end: Int32,
    ) -> Bool:
        """`split.cuh:142-191`. THE TOTAL ORDER. Transcribed branch for
        branch, in their order; see the module docstring for why the order
        is the algorithm and not a formatting choice.
        """
        # `:149-157` -- Primary ordering: higher gain wins; lower or
        # unordered gain loses.
        if other_best_metric_val > self.best_metric_val:
            return self.replace_with(
                other_quesval,
                other_colid,
                other_best_metric_val,
                other_global_nLeft,
                other_split_start,
                other_split_end,
            )
        if other_best_metric_val != self.best_metric_val:
            return False

        # `:160-169` -- Equal gain: preserve the existing deterministic
        # feature tie-break.
        if other_colid > self.colid:
            return self.replace_with(
                other_quesval,
                other_colid,
                other_best_metric_val,
                other_global_nLeft,
                other_split_start,
                other_split_end,
            )
        if other_colid != self.colid:
            return False

        # `:171-177` -- Equal gain and feature: multiple thresholds can
        # send the same training rows left and right. Merge that range and
        # choose its representative after reduction.
        if self.can_merge_equivalent_split_range(
            other_global_nLeft, other_split_start, other_split_end
        ):
            self.merge_equivalent_split_range(
                other_quesval, other_split_start, other_split_end
            )
            return True

        # `:179-188` -- Equal gain and feature, but a different training
        # partition: keep the existing deterministic threshold tie-break.
        if other_quesval > self.quesval:
            return self.replace_with(
                other_quesval,
                other_colid,
                other_best_metric_val,
                other_global_nLeft,
                other_split_start,
                other_split_end,
            )

        return False

    @always_inline
    def update_bin(
        mut self,
        other_quesval: Scalar[Self.dtype],
        other_colid: Int32,
        other_best_metric_val: Scalar[Self.dtype],
        other_global_nLeft: Int64,
        other_bin: Int32,
    ) -> Bool:
        """`split.cuh:193-201`, the one-bin overload: a degenerate range
        `[bin, bin]`. Spelled as a distinct name because Mojo resolves
        overloads on arity and their two `update`s differ only in the last
        argument's meaning.
        """
        return self.update(
            other_quesval,
            other_colid,
            other_best_metric_val,
            other_global_nLeft,
            other_bin,
            other_bin,
        )

    @always_inline
    def warp_reduce(mut self):
        """`split.cuh:206-220`, the in-warp reduction.

        Their loop, verbatim:

            auto lane = raft::laneId();
            for (int i = raft::WarpSize / 2; i >= 1; i /= 2) {
              auto id  = lane + i;
              auto qu  = raft::shfl(quesval, id);
              ... update(qu, co, be, gnl, bs, bn);
            }

        `raft::shfl(v, id)` is `__shfl_sync(mask, v, id)`, whose source
        lane is taken modulo the warp width, so `lane + i` WRAPS -- this is
        a rotate-and-reduce and every lane ends holding the whole warp's
        reduction, which is what `evalBestSplit` then relies on when only
        lane 0 publishes. See DEVIATION 104 for the width.

        `global_nLeft` is Int64 and is shuffled as two Int32 halves. That
        is a spelling change with no value change: `raft::shfl` on a 64-bit
        type does the same split internally.
        """
        var lane = lane_id()
        var i = WARP_SIZE // 2
        while i >= 1:
            var src = UInt32((lane + i) % WARP_SIZE)
            var qu = shuffle_idx(self.quesval, src)
            var co = shuffle_idx(self.colid, src)
            var be = shuffle_idx(self.best_metric_val, src)
            var gnl_lo = shuffle_idx(
                Int32(Int(self.global_nLeft & 0xFFFFFFFF)), src
            )
            var gnl_hi = shuffle_idx(Int32(Int(self.global_nLeft >> 32)), src)
            var bs = shuffle_idx(self.split_start, src)
            var bn = shuffle_idx(self.split_end, src)
            var gnl = (Int64(Int(gnl_hi)) << 32) | Int64(
                Int(UInt32(Int(gnl_lo)))
            )
            _ = self.update(qu, co, be, gnl, bs, bn)
            i //= 2

    @always_inline
    def eval_best_split[
        so: MutOrigin,
        sas: AddressSpace,
        go: MutOrigin,
        mo: MutOrigin,
        qo: MutOrigin, //
    ](
        mut self,
        split_scratch: MutPointer[Self, so, address_space=sas],
        split: MutPointer[Self, go],
        mutex: MutPointer[Int32, mo],
        quantiles: MutPointer[Scalar[Self.dtype], qo],
        n_bins: Int32,
    ):
        """`split.cuh:232-274`. Warp reduce, block reduce, then a
        mutex-guarded merge into the node's global slot.

        Their contract at `:229-230`: "all threads in the block must enter
        this function together. At the end thread0 will contain the best
        split."

        The mutex handoff is DEVIATION 106; see the module docstring. The
        rest is transcribed: only warp 0 does the second reduction, lanes
        beyond `nWarps` seed a default `Split`, and only thread 0 takes the
        lock, applies `select_split_range_midpoint`, and publishes.
        """
        self.warp_reduce()
        var warp = Int(thread_idx.x) // WARP_SIZE
        var n_warps = Int(block_dim.x) // WARP_SIZE
        var lane = lane_id()
        if lane == 0:
            split_scratch[unsafe_offset=warp] = self.copy()
        barrier()
        if warp == 0:
            if lane < n_warps:
                self = split_scratch[unsafe_offset=lane].copy()
            else:
                self = Self()
            self.warp_reduce()
            # `:249` -- only the first thread publishes for this node.
            if thread_idx.x == 0 and self.IsValid():
                self.select_split_range_midpoint(quantiles, n_bins)

                # `:251-252` -- their `while (atomicCAS(mutex, 0, 1));`
                # as an acquire-load spin plus a weak relaxed claim.
                while True:
                    if (
                        Atomic.load[ordering = Ordering.ACQUIRE](mutex)
                        != Int32(0)
                    ):
                        continue
                    var expected = Int32(0)
                    if Atomic.compare_exchange[
                        success_ordering = Ordering.RELAXED,
                        failure_ordering = Ordering.RELAXED,
                        weak=True,
                    ](mutex, expected, Int32(1)):
                        break

                # `:253-259` -- read the current global split into a
                # register copy. Their field-by-field read exists because
                # `split` is `volatile`; ours is a plain load of the same
                # seven fields, in their order.
                var split_reg = split[unsafe_offset=0].copy()
                var update_result = split_reg.update(
                    self.quesval,
                    self.colid,
                    self.best_metric_val,
                    self.global_nLeft,
                    self.split_start,
                    self.split_end,
                )
                # `:262-269`
                if update_result:
                    split[unsafe_offset=0] = split_reg.copy()

                # `:270-271` -- their `__threadfence(); atomicExch(mutex,
                # 0);` folded into one RELEASE store, which orders the
                # publish above before the handback.
                Atomic.store[ordering = Ordering.RELEASE](mutex, Int32(0))


def init_split_kernel[
    dtype: DType
](splits: MutPointer[Split[dtype], MutAnyOrigin], len: Int32):
    """`initSplit`, `split.cuh:284-289`.

    Theirs is `raft::linalg::writeOnlyUnaryOp` over a lambda that assigns a
    default-constructed `Split`. RAFT's `writeOnlyUnaryOp` is a plain
    grid-stride write-only map with no read of the destination, so the
    kernel below IS that map with their lambda inlined -- there is no
    vendor primitive being stood in for and nothing is unfused by writing
    it out.
    """
    from std.gpu import block_dim, block_idx, grid_dim, thread_idx

    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while idx < Int(len):
        splits[unsafe_offset=idx] = Split[dtype]()
        idx += stride
