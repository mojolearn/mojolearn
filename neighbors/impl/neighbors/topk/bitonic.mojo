# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Batcher's bitonic sorting network, laid out across one warp's registers.

WRITTEN FROM THE ALGORITHM, not from anybody's source. The network is
Batcher's, published as

    Batcher, K. E., "Sorting networks and their applications",
    AFIPS Spring Joint Computer Conference, 1968.

Nothing here is transliterated from an implementation; the stage/stride
recurrence, the index-to-(register, lane) map and the direction derivation
below are each rederived from that paper's construction.

THE ALGORITHM, RESTATED SO THE CODE CAN BE READ AGAINST IT
-----------------------------------------------------------
A *bitonic* sequence rises then falls, or is a rotation of one. Batcher's
merge sorts a bitonic sequence of length `n` in `log2(n)` stages. At the
stage with stride `s` (taking `s = n/2, n/4, ..., 1`) every index `i` whose
bit `s` is clear is compare-exchanged with `i | s`. A full sort of `n`
elements is built by sorting the two halves in OPPOSITE directions, which
makes their concatenation bitonic, then merging. Unrolled, that is

    for kk in 2, 4, ..., n:          # size of the bitonic block being merged
        for s in kk/2, kk/4, ..., 1: # stride inside that block
            compare-exchange (i, i|s) for every i with (i & s) == 0,
            ascending where (i & kk) == 0 and descending elsewhere

and the `kk == n` pass leaves the whole array sorted, because `(i & n) == 0`
holds for every `i < n`.

THE LANE / REGISTER MAP, AND WHY IT IS THIS ONE
------------------------------------------------
`W` registers per lane over a 32-lane warp hold `n = 32 * W` elements at

    i = r * 32 + lane          (r is the register, lane is `lane_id()`)

so register `r` of lane `l` is element `r * 32 + l`. That map is not a free
choice: the queue built on top of this file must hand the caller output slot
`i * 32 + l` from register `i` of lane `l`, so the sort has to produce its
answer in exactly this order or `write_out` would have to shuffle.

The map splits every compare-exchange into two cases that need different
hardware:

  * `s >= 32`. `i` and `i | s` differ only in the register bits, so the
    partner is register `r | (s / 32)` OF THE SAME LANE. Pure register
    traffic, no communication, and the pairing is fully comptime.
  * `s < 32`. `i` and `i | s` differ only in the lane bits, so the partner is
    lane `lane ^ s` in the SAME register, fetched with one `shuffle_xor` per
    array (keys and values).

DERIVING THE DIRECTION OF A BLOCK
----------------------------------
A block's direction is bit `kk` of the index. With `i = r * 32 + lane` and
`lane < 32`:

  * `kk >= 32`: `i & kk` depends only on `r`, and equals `r & (kk / 32)`
    scaled. So the direction is a COMPTIME constant per register. Note this
    covers `kk == 32` too, where `kk / 32 == 1` and the direction alternates
    register by register.
  * `kk < 32`: `i & kk` depends only on `lane`. It is a runtime value, but it
    is the SAME runtime value on both lanes of any pair, because a pair
    differs only in bit `s` and `s < kk`.

Both lanes of a `s < 32` exchange therefore agree on the direction. That
agreement is what makes the exchange safe with no explicit handshake.

`dir` inverts the whole thing: `dir == False` sorts best-first with "best"
meaning SMALLEST, `dir == True` best-first with "best" meaning LARGEST. The
block direction is `not dir` where `(i & kk) == 0` and `dir` elsewhere,
which at `kk == n` collapses to `not dir` everywhere, i.e. smallest-first
for `dir == False`.

TIES, STATED EXACTLY BECAUSE CORRECTNESS DEPENDS ON IT
-------------------------------------------------------
The comparison is a TOTAL order on the PAIR, never on the key alone:

    (ka, va) precedes (kb, vb)  iff  ka < kb, or (ka == kb and va < vb)

`key_is_before` is that predicate and it is the only comparison in this
file. Equal keys are broken by the SMALLER PAYLOAD first, so `dir == False`
returns the lowest-numbered of a set of tied candidates and the result is a
deterministic function of the input, not of the schedule.

Every exchange assigns the two elements FIXED roles -- `A` is whatever sits
at the low index `i`, `B` at the high index `i | s` -- and both sides compute
one `swap` flag from `(A, B)` in that order. The low side keeps `A` unless
`swap`, the high side keeps `B` unless `swap`. Since the flag is one
expression over one ordered pair, the two sides can never both keep or both
discard, including when the two elements are EQUAL as pairs: then
`key_is_before` is false in both directions, `swap` is false, and each side
keeps what it had. That is the property a shuffle-based exchange needs and
that a key-only comparison would not give.

NaN keys are outside the contract. `key_is_before` is false in both
directions for a NaN, so a NaN compares as equal-to-everything and its final
position is unspecified. Callers feed distances, which are non-NaN.
"""

from std.bit import log2_floor
from std.gpu.primitives.id import lane_id
from std.gpu.primitives.warp import shuffle_xor


@always_inline
def key_is_before(
    ka: Float32, va: UInt32, kb: Float32, vb: UInt32
) -> Bool:
    """The total order over (key, payload). See the module docstring.

    Strictly ascending in the key, with the payload breaking exact key ties
    in favor of the SMALLER payload. Never returns True for two pairs that
    are equal in both components, which is what keeps the two sides of an
    exchange complementary.
    """
    return ka < kb or (ka == kb and va < vb)


@always_inline
def is_better[dir: Bool](
    ka: Float32, va: UInt32, kb: Float32, vb: UInt32
) -> Bool:
    """`(ka, va)` outranks `(kb, vb)` under the queue's direction.

    `dir == False` keeps the SMALLEST keys, so "better" is "earlier" in the
    total order. `dir == True` keeps the largest, so "better" is "later".
    """
    comptime if dir:
        return key_is_before(kb, vb, ka, va)
    else:
        return key_is_before(ka, va, kb, vb)


@always_inline
def _needs_swap(
    asc: Bool, ak: Float32, av: UInt32, bk: Float32, bv: UInt32
) -> Bool:
    """Should the pair at (low, high) trade places?

    `A` is the element at the LOW index and `B` the one at the high index;
    the roles are fixed by the caller and never by which lane is asking.
    Ascending wants the earlier element low, so it swaps when `B` precedes
    `A`; descending wants the later element low, so it swaps when `A`
    precedes `B`. Both clauses are strict, so equal pairs never swap.
    """
    if asc:
        return key_is_before(bk, bv, ak, av)
    return key_is_before(ak, av, bk, bv)


@always_inline
def _cex_stage[W: Int, dir: Bool, S: Int, KK: Int](
    mut keys: SIMD[DType.float32, W],
    mut vals: SIMD[DType.uint32, W],
    lane: Int,
):
    """One stage of the network: every compare-exchange at stride `S`, for
    blocks of size `KK`.

    `S` and `KK` are comptime, so the whole stage unrolls into straight-line
    register code with `S < 32` contributing exactly two shuffles per
    register and `S >= 32` contributing none.
    """
    comptime if S >= 32:
        # ---- register-resident exchange -------------------------------
        # `S` is a whole number of warps, so the partner of register `r` is
        # register `r ^ RS` in this same lane. Visiting only the `r` whose
        # `RS` bit is clear touches each pair once.
        comptime RS = S // 32
        comptime for r in range(W):
            comptime if (r & RS) == 0:
                # `S >= 32` and `S < KK` force `KK >= 64`, so `KK // 32` is
                # a real register-bit mask and the direction is comptime.
                comptime ASC = (not dir) if ((r & (KK // 32)) == 0) else dir
                var ak = keys[r]
                var av = vals[r]
                var bk = keys[r | RS]
                var bv = vals[r | RS]
                if _needs_swap(ASC, ak, av, bk, bv):
                    keys[r] = bk
                    vals[r] = bv
                    keys[r | RS] = ak
                    vals[r | RS] = av
    else:
        # ---- cross-lane exchange --------------------------------------
        # Same register, partner lane `lane ^ S`. Every lane reaches the
        # shuffles unconditionally: a lane that skipped one would hang the
        # rest of the warp.
        comptime for r in range(W):
            var ok = keys[r]
            var ov = vals[r]
            var pk = shuffle_xor(ok, UInt32(S))
            var pv = shuffle_xor(ov, UInt32(S))

            # FIXED ROLES. `A` is the low index of the pair on BOTH lanes,
            # so both lanes feed `_needs_swap` the same ordered pair and get
            # the same flag out.
            var is_low = (lane & S) == 0
            var ak = pk
            var av = pv
            var bk = ok
            var bv = ov
            if is_low:
                ak = ok
                av = ov
                bk = pk
                bv = pv

            var asc: Bool
            comptime if KK >= 32:
                # Direction lives in the register bits; comptime per `r`.
                asc = (not dir) if ((r & (KK // 32)) == 0) else dir
            else:
                # Direction lives in the lane bits. `S < KK < 32`, so bit
                # `KK` of `lane` is identical on both lanes of the pair and
                # this runtime value agrees across the exchange.
                asc = (not dir) if ((lane & KK) == 0) else dir

            var do_swap = _needs_swap(asc, ak, av, bk, bv)
            # Low keeps A unless swapping, high keeps B unless swapping.
            # Complementary by construction, so nothing is duplicated or
            # dropped.
            var take_b = do_swap if is_low else (not do_swap)
            if take_b:
                keys[r] = bk
                vals[r] = bv
            else:
                keys[r] = ak
                vals[r] = av


@always_inline
def bitonic_merge_warp[W: Int, dir: Bool](
    mut keys: SIMD[DType.float32, W],
    mut vals: SIMD[DType.uint32, W],
):
    """Batcher's merge: a BITONIC array of `32 * W` pairs becomes sorted.

    PRECONDITION, and it is not checked anywhere: the array must already be
    bitonic under the direction `dir` implies. Feeding an arbitrary array
    here returns garbage, quietly.

    This is the final `kk == n` pass of the full sort, so `(i & n) == 0`
    everywhere and every block runs in the same direction.
    """
    comptime N = 32 * W
    comptime STAGES = log2_floor(N)
    var lane = Int(lane_id())

    comptime for step in range(STAGES):
        # N/2, N/4, ..., 1
        comptime S = N >> (step + 1)
        _cex_stage[W, dir, S, N](keys, vals, lane)


@always_inline
def bitonic_sort_warp[W: Int, dir: Bool](
    mut keys: SIMD[DType.float32, W],
    mut vals: SIMD[DType.uint32, W],
):
    """Full Batcher sort of the `32 * W` pairs the warp holds.

    Blocks of size `KK` are merged after their two halves have been sorted
    in opposite directions, which is what makes each block bitonic before
    its merge runs. No precondition on the input.
    """
    comptime N = 32 * W
    comptime LEVELS = log2_floor(N)
    var lane = Int(lane_id())

    comptime for lk in range(LEVELS):
        comptime KK = 2 << lk  # 2, 4, ..., N
        # The inner bound is `lk + 1`, but it is spelled as a fixed range
        # plus a comptime guard so the loop bound never depends on the outer
        # induction variable. Same unrolled stages, one fewer thing for the
        # comptime evaluator to have to agree with us about.
        comptime for lj in range(LEVELS):
            comptime if lj <= lk:
                comptime S = KK >> (lj + 1)  # KK/2, ..., 1
                _cex_stage[W, dir, S, KK](keys, vals, lane)


@always_inline
def bitonic_merge_two_warp[W: Int, dir: Bool](
    mut ak: SIMD[DType.float32, W],
    mut av: SIMD[DType.uint32, W],
    bk: SIMD[DType.float32, W],
    bv: SIMD[DType.uint32, W],
):
    """Merge two SORTED warp-wide arrays and keep the better `32 * W`.

    Both inputs hold `32 * W` pairs already sorted best-first under `dir`.
    On return `ak`/`av` hold, sorted, exactly the best `32 * W` of the `64 *
    W` pairs that went in; the other half is dropped, which is the whole
    point -- a bounded queue never wants it.

    WHY ONE ELEMENTWISE PASS SUFFICES BEFORE THE MERGE. Batcher's
    construction says that for sorted `A` and sorted `B` of equal length
    `n`, the sequence `L[i] = better(A[i], B[n-1-i])` is bitonic AND is
    exactly the best `n` of the union, with `H[i] = worse(...)` the other
    `n`. So reversing `B`, taking the elementwise better, and running one
    bitonic merge is a complete merge-and-truncate. It costs one pass plus
    `log2(32 W)` stages rather than a full re-sort.

    Reversing `B` is one shuffle: element `i = r * 32 + lane` reverses to
    `32 W - 1 - i = (W - 1 - r) * 32 + (31 - lane)`, and `31 - lane` is
    `lane ^ 31` for a 32-lane warp, so `shuffle_xor(..., 31)` on register
    `W - 1 - r` fetches it.

    Duplicate pairs are harmless: `is_better` is strict, so on a tie the
    `A` side is kept, and since the two are equal as pairs the resulting
    multiset is the same either way.
    """
    comptime for r in range(W):
        # Register `W - 1 - r` of lane `31 - lane` is the reversal partner.
        var rk = shuffle_xor(bk[W - 1 - r], UInt32(31))
        var rv = shuffle_xor(bv[W - 1 - r], UInt32(31))
        if is_better[dir](rk, rv, ak[r], av[r]):
            ak[r] = rk
            av[r] = rv

    # `ak` is now bitonic and holds the winning half; sort it.
    bitonic_merge_warp[W, dir](ak, av)
