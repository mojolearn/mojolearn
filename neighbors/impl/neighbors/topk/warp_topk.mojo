# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""A warp-level, register-resident top-k selection queue.

WRITTEN FROM THE ALGORITHM. The sorting network underneath is Batcher's,

    Batcher, K. E., "Sorting networks and their applications",
    AFIPS Spring Joint Computer Conference, 1968,

and lives in `neighbors/impl/neighbors/topk/bitonic.mojo`; the queue
discipline layered on top of it is derived here from the same construction.
No implementation was consulted.

WHAT THIS IS FOR
----------------
One warp streams an arbitrarily long row of candidates past this object and
ends holding the best `k` of them, SORTED, without the row or the answer ever
touching shared or global memory. That is what lets a distance kernel fuse
its selection into its own epilogue: the accumulators are already in
registers and the answer stays there.

THE TWO QUEUES
--------------
  * The WARP QUEUE: `num_warp_q` pairs, `R = num_warp_q / 32` per lane in
    registers, held SORTED best-first at all times in the layout
    `i = r * 32 + lane`. It is the running answer.
  * The THREAD QUEUE: `num_thread_q` pairs per lane, unsorted, purely a
    staging buffer. Filling it is cheap; draining it is not.

`add_thread_q` stages a candidate only if it could still make the cut, which
is decided against `warp_k_top`, the WORST key currently in the warp queue,
broadcast to every lane. Rejecting on that bound is safe because the warp
queue already holds `num_warp_q >= k` candidates that are all at least as
good, so a candidate worse than every one of them cannot enter the best `k`.
The test is deliberately non-strict (`<=` for `dir == False`): being too
permissive only costs a staged slot, being too strict loses an answer.

`check_thread_q` votes across the warp and, if ANY lane has filled its
staging array, drains every lane. The vote is why the CALL CONTRACT below is
not optional.

`merge_warp_q` is the drain. It reads the staging arrays as a second
warp-wide array, sorts it, merges the two sorted arrays keeping the better
half, and leaves the warp queue sorted again. `reduce()` is exactly this
call, since a drain already ends in the sorted state the caller wants.

THE CALL CONTRACT
-----------------
EVERY LANE OF THE WARP MUST CALL `add` (or `check_thread_q`, or
`merge_warp_q`, or `reduce`) THE SAME NUMBER OF TIMES. Those methods contain
a warp vote and a network full of `shuffle_xor`; a lane that skips one hangs
the others. A caller with a short or ragged row pads it by calling `add` with
the sentinel key, which is rejected by the bound and costs nothing. The only
per-lane divergence this object permits is `num_vals`, which is why the
vote exists.

`write_out` takes no vote and may be called under a warp-uniform predicate.

OUTPUT LAYOUT
-------------
After `reduce()`, register `i` of lane `l` is output slot `i * 32 + l`, and
`write_out` stores only the slots below `k`. This is the layout the bitonic
network already produces, so `write_out` is a straight store with no
permutation.

TIES
----
Inherited unchanged from the network: the comparison is the total order

    (ka, va) before (kb, vb)  iff  ka < kb, or (ka == kb and va < vb)

so equal keys come back ordered by ASCENDING PAYLOAD for `dir == False`. The
answer is a deterministic function of the input multiset. It is NOT the same
tie order another selector would produce, and callers that compare index
arrays against a different selector on a tied fixture are asserting something
no selector promises.

THE SENTINEL
------------
`init_k_val` / `init_v_val` fill both queues at construction and refill the
staging array after every drain, and they pad the working arrays when
`num_thread_q` is not a power of two or the two capacities differ. They must
be the WORST pair under `dir`: for `dir == False` that is a large key such as
`FLT_MAX` together with a large payload such as `0xFFFFFFFF`, and no
candidate the caller adds may sort after it. A candidate with key `+inf`
would violate that and could be displaced by a sentinel.
"""

from std.gpu.primitives.id import lane_id
from std.gpu.primitives.warp import max as warp_max
from std.gpu.primitives.warp import shuffle_idx

from neighbors.impl.neighbors.topk.bitonic import (
    bitonic_merge_two_warp,
    bitonic_sort_warp,
)


struct WarpSelect[num_warp_q: Int, num_thread_q: Int, dir: Bool](
    TrivialRegisterPassable
):
    """The queue. `dir == False` keeps the SMALLEST `k` and returns them
    ascending, which is what a k-NN caller wants; `dir == True` keeps the
    largest and returns them descending."""

    # Registers per lane in the warp queue. `num_warp_q` is a power-of-two
    # multiple of the warp size, so `R` is a power of two and the network
    # over `32 * R` elements is a legal bitonic width.
    comptime R = Self.num_warp_q // 32

    # The staging array is padded to a power of two so the SAME network can
    # sort it. `num_thread_q == 3` is a SHIPPED configuration, not a corner
    # case, so this padding is on the main path. Written as a comptime
    # conditional chain rather than a bit trick so it folds with certainty.
    comptime TQP = 1 if Self.num_thread_q <= 1 else (
        2 if Self.num_thread_q <= 2 else (
            4 if Self.num_thread_q <= 4 else (
                8 if Self.num_thread_q <= 8 else 16
            )
        )
    )

    # Width of the working arrays inside a drain. Batcher's merge-and-keep
    # needs its two inputs to be the SAME length, so both the warp queue and
    # the staged candidates are padded up to the larger of the two. `R` and
    # `TQP` are both powers of two, so their max is one too.
    comptime MP = Self.R if Self.R > Self.TQP else Self.TQP

    # The running answer, always sorted best-first, layout `r * 32 + lane`.
    var warp_k: SIMD[DType.float32, Self.R]
    var warp_v: SIMD[DType.uint32, Self.R]

    # Per-lane staging. Slots at or above `num_vals` hold the sentinel; the
    # drain restores that, which is what lets `add_thread_q` write a slot
    # without first clearing it.
    var thread_k: SIMD[DType.float32, Self.TQP]
    var thread_v: SIMD[DType.uint32, Self.TQP]

    # How many staged pairs this lane holds. THE ONE PER-LANE DIVERGENT
    # QUANTITY in the object; every other field is warp-uniform or indexed
    # by lane. Public because callers vote on it before deciding to reduce.
    var num_vals: Int

    # The worst key in the warp queue, broadcast to every lane. Warp-uniform
    # by construction (`shuffle_idx` from lane 31), which is required: an
    # admission test that disagreed between lanes would still be correct,
    # but it would make `num_vals` diverge for a reason the vote cannot see.
    var warp_k_top: Float32

    # The sentinel, kept because the staging array is refilled with it after
    # every drain and the working arrays are padded with it.
    var init_k: Float32
    var init_v: UInt32

    # The caller's requested `k`. Recorded for completeness; the bound this
    # object actually rejects against is the worst of all `num_warp_q`
    # slots, not the worst of the first `k`, because the latter would need a
    # runtime-indexed broadcast for a strictly weaker filter. `write_out`'s
    # own `k` argument is what decides how much is stored.
    var k: Int

    @always_inline
    def __init__(out self, init_k_val: Float32, init_v_val: UInt32, k: Int):
        """Both queues start full of the sentinel, so the warp queue is
        trivially sorted and `warp_k_top` is the sentinel key: every early
        candidate is admitted until real values have displaced it."""
        comptime assert Self.num_warp_q % 32 == 0, (
            "num_warp_q must be a whole number of warp rows"
        )
        comptime assert Self.R > 0, "num_warp_q must be at least 32"
        comptime assert (Self.R & (Self.R - 1)) == 0, (
            "num_warp_q / 32 must be a power of two: the bitonic network is"
            " only defined on power-of-two widths"
        )
        comptime assert Self.num_thread_q > 0, "num_thread_q must be positive"
        comptime assert Self.num_thread_q <= 16, (
            "num_thread_q above 16 would outgrow the padding chain above"
        )

        self.init_k = init_k_val
        self.init_v = init_v_val
        self.k = k
        self.warp_k = SIMD[DType.float32, Self.R](init_k_val)
        self.warp_v = SIMD[DType.uint32, Self.R](init_v_val)
        self.thread_k = SIMD[DType.float32, Self.TQP](init_k_val)
        self.thread_v = SIMD[DType.uint32, Self.TQP](init_v_val)
        self.num_vals = 0
        self.warp_k_top = init_k_val

    @always_inline
    def add_thread_q(mut self, key: Float32, val: UInt32):
        """Stage one candidate, or drop it.

        No shuffle, no vote, no touch of the warp queue: this is the hot
        path and it must stay a compare plus a predicated register write.
        The write is a comptime-unrolled scatter rather than
        `self.thread_k[self.num_vals] = key`, because a runtime index into a
        SIMD register array is not a register write at all -- it forces the
        staging array into local memory, which is exactly the memory traffic
        this queue exists to avoid. `TQP` is 2 to 4 in practice, so the
        unrolled form is a couple of selects.

        The `num_vals < num_thread_q` guard is defensive. Under the
        documented call sequence `check_thread_q` has already drained a full
        lane, so it never fires; it is here so that a caller that skips the
        drain loses candidates instead of corrupting the array.
        """
        var admit: Bool
        comptime if Self.dir:
            # Keeping the largest: a candidate must be at least as large as
            # the smallest thing already held.
            admit = key >= self.warp_k_top
        else:
            admit = key <= self.warp_k_top

        if admit and self.num_vals < Self.num_thread_q:
            comptime for t in range(Self.TQP):
                if t == self.num_vals:
                    self.thread_k[t] = key
                    self.thread_v[t] = val
            self.num_vals += 1

    @always_inline
    def check_thread_q(mut self):
        """Drain if ANY lane has filled its staging array.

        The vote is the reason the whole warp has to reach this call: the
        drain is a bitonic network, every stage of it shuffles, and a lane
        that sat the merge out would leave the others waiting. `warp_max`
        over a 0/1 flag is an any-vote and it is warp-uniform on every
        vendor, which a ballot mask spelling would not be.
        """
        var full = Int32(0)
        if self.num_vals >= Self.num_thread_q:
            full = Int32(1)
        if warp_max(full) != Int32(0):
            self.merge_warp_q()

    @always_inline
    def merge_warp_q(mut self):
        """Fold the staged candidates into the warp queue.

        Four steps, and the ordering matters:

        1. Read the staging arrays as a warp-wide array `B` of `32 * MP`
           pairs under the same `r * 32 + lane` map, padded with the
           sentinel above `TQP`. Slots at or above each lane's `num_vals`
           already hold the sentinel, so nothing has to be cleared first.
        2. Sort `B`. It arrives in whatever order candidates showed up, so
           this is the full Batcher sort, not a merge.
        3. Pad the warp queue to the same `MP` width -- appending sentinels
           at the END of a best-first array keeps it sorted -- and merge the
           two sorted arrays, keeping the better `32 * MP`. That single
           reverse-and-compare pass plus one bitonic merge is a complete
           merge-and-truncate, which is why this is not a re-sort of
           everything.
        4. Take the first `R` registers back as the new warp queue. Because
           the merged array is sorted best-first in the same layout, its
           first `32 * R` elements ARE registers 0..R-1, with no movement.

        Then the staging arrays go back to the sentinel and `warp_k_top` is
        re-read, since the cut has just tightened.
        """
        # 1. the staged candidates, as a warp-wide array
        var bk = SIMD[DType.float32, Self.MP](self.init_k)
        var bv = SIMD[DType.uint32, Self.MP](self.init_v)

        comptime for t in range(Self.TQP):
            bk[t] = self.thread_k[t]
            bv[t] = self.thread_v[t]

        # 2. sort it
        bitonic_sort_warp[Self.MP, Self.dir](bk, bv)

        # 3. pad the answer to the same width and merge
        var ak = SIMD[DType.float32, Self.MP](self.init_k)
        var av = SIMD[DType.uint32, Self.MP](self.init_v)

        comptime for r in range(Self.R):
            ak[r] = self.warp_k[r]
            av[r] = self.warp_v[r]

        bitonic_merge_two_warp[Self.MP, Self.dir](ak, av, bk, bv)

        # 4. keep the best `num_warp_q`
        comptime for r in range(Self.R):
            self.warp_k[r] = ak[r]
            self.warp_v[r] = av[r]

        self.thread_k = SIMD[DType.float32, Self.TQP](self.init_k)
        self.thread_v = SIMD[DType.uint32, Self.TQP](self.init_v)
        self.num_vals = 0

        # The worst slot is index `num_warp_q - 1`, which under
        # `i = r * 32 + lane` is register `R - 1` of LANE 31. Broadcast it
        # so every lane admits and rejects identically.
        self.warp_k_top = shuffle_idx(self.warp_k[Self.R - 1], UInt32(31))

    @always_inline
    def add(mut self, key: Float32, val: UInt32):
        """Stage one candidate and drain if the warp needs it. The whole
        warp must reach this the same number of times; see the call
        contract in the module docstring."""
        self.add_thread_q(key, val)
        self.check_thread_q()

    @always_inline
    def reduce(mut self):
        """Finish: drain whatever is staged and leave the warp queue sorted.

        This is `merge_warp_q` and nothing else, because a drain already
        ends with the warp queue fully sorted in the output layout. It is
        idempotent -- a second call re-merges an empty staging array, which
        cannot change the answer -- and the object stays usable afterwards,
        so a caller may `reduce`, hand the result on, `add` more, and
        `reduce` again.
        """
        self.merge_warp_q()

    @always_inline
    def write_out(
        self,
        out_k: MutPointer[Float32, MutAnyOrigin],
        out_v: MutPointer[UInt32, MutAnyOrigin],
        k: Int,
    ):
        """Store the answer. Register `i` of lane `l` goes to slot
        `i * 32 + l`, and only slots below `k` are written.

        No shuffle and no vote here, so this may sit under a warp-uniform
        predicate. It reads the warp queue without changing it, which is
        sound at any time because that queue is sorted between operations,
        not merely after `reduce`: `add_thread_q` never touches it, and
        `merge_warp_q` leaves it sorted. A queue that has never seen a
        candidate writes `k` sentinels.

        Slots at or above `num_warp_q` are NOT written. A caller that asks
        for `k > num_warp_q` owns filling the remainder.
        """
        var lane = Int(lane_id())

        comptime for r in range(Self.R):
            var idx = r * 32 + lane
            if idx < k:
                out_k.unsafe_store(idx, self.warp_k[r])
                out_v.unsafe_store(idx, self.warp_v[r])
