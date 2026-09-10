# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Block-wide argmin / argmax with an EXPLICIT tie-break, no lane primitive.

NO REFERENCE FILE. `smoblocksolve.cuh` reduces `KVPair<math_t, int>` with
`cub::BlockReduce<Pair, WSIZE>::Reduce(pair, cuda::minimum{} / maximum{},
n_ws)`, and `KVPair::operator<` / `operator>` compare `val` ONLY
(`cpp/src_prims/selection/kselection.cuh:66-79`, both carrying the comment
"///@todo: should we also consider the key when values are the same?").
On equal `val` the surviving key is whichever pair CUB's warp-then-block
fold happens to keep, which is a function of the fold topology and therefore
of the lane width and the vendor (IDENTITY_PATHS rows 8 and 20).

DEVIATION 633 (svm/README.md, identity content section 3): the tie resolves
to the SMALLER KEY, and the key the block solve passes is the TRAINING INDEX
`ws_idx[tid]`, not `tid`, so the winner is a pure function of the working
SET, not of the order the kernel cache permuted it into. DEVIATION 635
(`smoblocksolve.mojo`): the `f_max` fold is the SAME argmax, so all three
reductions of the block solve tie on the training index.

The shape is `core/pinned_reduce.mojo`'s halving tree through threadgroup
memory with no warp primitive, so no lane width can reach it. A selection
over a TOTAL order (value, then key) is exactly commutative and associative
away from NaN, so the result is independent of `block_size` as long as the
padding threads pass the identity (`+inf` for argmin, `-inf` for argmax,
with any key). NaN is never RECORDED: the block solve masks with
`+-INFINITY`, `svc_impl.mojo` refuses non-finite inputs by name (DEVIATION
636), and a NaN that float overflow leaves in `alpha`/`f` raises in
`smosolver.mojo` before any record (DEVIATION 637); their
`CheckStoppingCondition` throws on a NaN `diff` at the same place.

SIGNED ZERO (IDENTITY_PATHS row 39). `f` can be `+0.0` (a sample exactly on
the margin) and `-0.0` (a negative subnormal flushed at a seam, row 10), and
both can sit in one working set. The compare here is `ov < mv or (ov == mv
and ok < mk)`: on a `(+0.0, -0.0)` pair `<` is FALSE and `==` is TRUE on
every vendor (IEEE 754 comparison predicates are not implementation-
defined, unlike `max`/`min`), so the KEY decides, and the value returned is
the winner's own bits, `-0.0` or `+0.0` as that sample holds it. No
hardware `max`/`min`, no `SIMD.reduce_*`, no `.clamp()` is used anywhere in
this file. Gated by `svc_check.mojo::check_block_solve_signed_zero_tie`
(both zeros planted in one working set, both orders) and its sabotages.

CONTRACT (the same as `pinned_block_sum`'s): every thread of the block
calls it; `block_size` is a power of two; the result is returned to EVERY
thread (theirs broadcasts `u` and `l` through `__shared__`); the trailing
`barrier()` protects the slab for back-to-back calls. Mode-free: there is
no FAST arm to preserve, because the library call's tie order is not a
spelling anyone wrote down.

SABOTAGE ARMS (the foot of this file) are not called by the implementation; they are
the spellings the row-39 gate must reject.
"""

from std.gpu import thread_idx, lane_id, WARP_SIZE
from std.gpu.primitives.warp import shuffle_xor
from std.math import max
from std.memory import stack_allocation
from std.sys.compile import is_defined
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier


#: SABOTAGE (svc_check "tie to the HIGHER key"): the tie-break direction is
#: reversed in BOTH arg-reductions. Must FAIL device-vs-oracle on F3.dup
#: (equal f) and the signed-zero gate (order A).
comptime SAB_ARG_TIE_HIGH = is_defined["MOJOLEARN_SVM_SABOTAGE_ARG_TIE_HIGH"]()


@always_inline
def pinned_block_argmin[
    block_size: Int
](value: Float32, key: Int32) -> Tuple[Float32, Int32]:
    """`min` over `(value, key)` lexicographically; returned to all threads."""
    var tid = Int(thread_idx.x)
    var vals = stack_allocation[
        block_size,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var keys = stack_allocation[
        block_size,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    vals[tid] = value
    keys[tid] = key
    barrier()
    var step = block_size // 2
    while step > 0:
        if tid < step:
            var ov = vals[tid + step]
            var ok = keys[tid + step]
            var mv = vals[tid]
            var mk = keys[tid]
            var tie: Bool
            comptime if SAB_ARG_TIE_HIGH:
                tie = ov == mv and ok > mk
            else:
                tie = ov == mv and ok < mk
            if ov < mv or tie:
                vals[tid] = ov
                keys[tid] = ok
        barrier()
        step //= 2
    var rv = vals[0]
    var rk = keys[0]
    barrier()
    return (rv, rk)


@always_inline
def pinned_block_argmax[
    block_size: Int
](value: Float32, key: Int32) -> Tuple[Float32, Int32]:
    """`max` over value, ties to the SMALLER key; returned to all threads."""
    var tid = Int(thread_idx.x)
    var vals = stack_allocation[
        block_size,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var keys = stack_allocation[
        block_size,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    vals[tid] = value
    keys[tid] = key
    barrier()
    var step = block_size // 2
    while step > 0:
        if tid < step:
            var ov = vals[tid + step]
            var ok = keys[tid + step]
            var mv = vals[tid]
            var mk = keys[tid]
            var tie: Bool
            comptime if SAB_ARG_TIE_HIGH:
                tie = ov == mv and ok > mk
            else:
                tie = ov == mv and ok < mk
            if ov > mv or tie:
                vals[tid] = ov
                keys[tid] = ok
        barrier()
        step //= 2
    var rv = vals[0]
    var rk = keys[0]
    barrier()
    return (rv, rk)


# ---------------------------------------------------------------------------
# DEVIATION 2491 (2026-09-10): THE SAME SELECTION, TWO BARRIERS INSTEAD OF
# TWELVE. The halving trees above are kept as the reference spelling (and
# for a block narrower than a warp); the block solve calls the functions
# below. A warp folds its (value, key) pairs with `shuffle_xor` butterflies
# (no barrier), lane 0 of each warp posts the warp's winner to threadgroup
# memory, one barrier, and EVERY thread folds the WARPS posted winners in
# ascending warp order, one barrier to protect the slots for the next call.
#
# WHY THE BITS CANNOT MOVE. The compare is the module docstring's `ov < mv
# or (ov == mv and ok < mk)` (`>` for the max), a TOTAL order on (value,
# key) away from NaN, and keys are unique training indices. The minimum of
# a finite set under a total order is one element; every fold topology, any
# lane width, any warp count returns that element. `+0.0 == -0.0` is TRUE
# on every vendor, so the key decides the signed-zero tie exactly as the
# tree does (row 39), and the returned value is that element's own bits.
# NaN never wins (both predicates false) and never reaches a record
# (DEVIATION 637). So this is the pinned selection with a cheaper schedule,
# not a new arm; IDENTICAL takes it too, and the SVC card gates it.
#
# The third return is the THREAD holding the winner, which the block solve
# used to recover by a ballot through threadgroup memory (two more
# barriers per reduction); it rides along in the fold for free.
# ---------------------------------------------------------------------------


@always_inline
def _arg_better[MAX: Bool](
    ov: Float32, ok: Int32, mv: Float32, mk: Int32
) -> Bool:
    """Does (ov, ok) beat (mv, mk)? Strict on value, smaller key on a tie."""
    var tie: Bool
    comptime if SAB_ARG_TIE_HIGH:
        tie = ov == mv and ok > mk
    else:
        tie = ov == mv and ok < mk
    comptime if MAX:
        return ov > mv or tie
    else:
        return ov < mv or tie


@always_inline
def block_argext[
    block_size: Int, MAX: Bool
](value: Float32, key: Int32) -> Tuple[Float32, Int32, Int32]:
    """`min` (or `max` when `MAX`) over `(value, key)` lexicographically,
    returned to all threads as (value, key, thread of the winner)."""
    var tid = Int(thread_idx.x)
    var v = value
    var k = key
    var t = Int32(tid)
    comptime if block_size < WARP_SIZE or block_size % WARP_SIZE != 0:
        # Narrower than a warp (or ragged): the reference tree, then one
        # ballot for the thread.
        var slot = stack_allocation[
            1, Scalar[DType.int32], address_space = AddressSpace.SHARED
        ]()
        var r: Tuple[Float32, Int32]
        comptime if MAX:
            r = pinned_block_argmax[block_size](v, k)
        else:
            r = pinned_block_argmin[block_size](v, k)
        if k == r[1]:
            slot[0] = t
        barrier()
        var win = slot[0]
        barrier()
        return (r[0], r[1], win)
    else:
        comptime WARPS = block_size // WARP_SIZE
        var offset = 1
        while offset < WARP_SIZE:
            var ov = shuffle_xor(v, UInt32(offset))
            var ok = shuffle_xor(k, UInt32(offset))
            var ot = shuffle_xor(t, UInt32(offset))
            if _arg_better[MAX](ov, ok, v, k):
                v = ov
                k = ok
                t = ot
            offset *= 2
        var vals = stack_allocation[
            WARPS, Scalar[DType.float32], address_space = AddressSpace.SHARED
        ]()
        var keys = stack_allocation[
            WARPS, Scalar[DType.int32], address_space = AddressSpace.SHARED
        ]()
        var tids = stack_allocation[
            WARPS, Scalar[DType.int32], address_space = AddressSpace.SHARED
        ]()
        var w = tid // WARP_SIZE
        if Int(lane_id()) == 0:
            vals[w] = v
            keys[w] = k
            tids[w] = t
        barrier()
        var rv = vals[0]
        var rk = keys[0]
        var rt = tids[0]
        comptime for i in range(1, WARPS):
            var ov = vals[i]
            var ok = keys[i]
            if _arg_better[MAX](ov, ok, rv, rk):
                rv = ov
                rk = ok
                rt = tids[i]
        barrier()
        return (rv, rk, rt)


# ---------------------------------------------------------------------------
# SABOTAGE ARMS (row 39): NOT called by the implementation. `smoblocksolve.mojo` routes
# `f_max` through one of these under the matching `-D` define so the
# signed-zero gate can show each spelling FAIL (or be Apple-inert).
# ---------------------------------------------------------------------------


@always_inline
def sabotage_block_max_nokey[block_size: Int](value: Float32) -> Float32:
    """SAB_FMAX_NOKEY: the pre-DEVIATION-635 `f_max`, a strict-`>` halving
    tree with no key. On a `(+0.0, -0.0)` tie nothing updates, so the
    survivor is a function of tree POSITION, not of the working set, and
    it disagrees with the oracle's smallest-key rule. Vendor-invariant but
    wrong against the oracle: must FAIL the signed-zero gate (order A)."""
    var tid = Int(thread_idx.x)
    var vals = stack_allocation[
        block_size,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    vals[tid] = value
    barrier()
    var step = block_size // 2
    while step > 0:
        if tid < step:
            var ov = vals[tid + step]
            if ov > vals[tid]:
                vals[tid] = ov
        barrier()
        step //= 2
    var rv = vals[0]
    barrier()
    return rv


@always_inline
def sabotage_block_max_hw[
    block_size: Int, swap: Bool
](value: Float32) -> Float32:
    """SAB_FMAX_HWMAX (`max(mine, other)`) / SAB_FMAX_HWMAX_SWAP
    (`max(other, mine)`): the halving tree through the HARDWARE `max`,
    whose answer on `(+0.0, -0.0)` is the vendor's (row 39: Apple returns
    the SECOND operand, NVIDIA and AMD return +0.0). Which zero survives
    therefore depends on the vendor AND on the operand order; the gate
    records which spelling is Apple-inert and why that is not evidence."""
    var tid = Int(thread_idx.x)
    var vals = stack_allocation[
        block_size,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    vals[tid] = value
    barrier()
    var step = block_size // 2
    while step > 0:
        if tid < step:
            var ov = vals[tid + step]
            var mv = vals[tid]
            comptime if swap:
                vals[tid] = max(ov, mv)
            else:
                vals[tid] = max(mv, ov)
        barrier()
        step //= 2
    var rv = vals[0]
    barrier()
    return rv
