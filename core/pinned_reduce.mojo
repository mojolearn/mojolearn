# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The within-block float fold, with ONE shape on every vendor.

DEVIATION 504 (IDENTITY_PATHS row 20). Reached only under
`NUMERIC_IDENTICAL`; the FAST arm IS the library call.

NO REFERENCE FILE. cuVS, cuML and RAFT each ship one GPU backend and reduce with
whatever CUB gives them; the question this file answers -- *does the fold
combine the same partials in the same order on Metal, CUDA and HIP* -- only
exists because we ship three backends from one source.

WHY A LIBRARY `block.sum` IS NOT ENOUGH
---------------------------------------
`max.gpu.primitives.block.sum` is correct, tuned, and NOT machine
independent: its internal cross-lane stage folds at the HARDWARE'S warp
width, which is 32 on Apple and NVIDIA and **64 on AMD's CDNA wavefront**.
Float addition is not associative, so a 64-wide fold and a 32-wide fold of
the same 128 values are two different sums of the same multiset, and they
differ in the last bits. That is IDENTITY_PATHS row 8's AMD residue, found
by the GBDT lane on `pointwise_targets.mojo` and closed there at 14 of 16
producer sites with a fold of exactly this shape (DEVIATION 251).

The unsupervised sections had the same residue at every `block.sum` they
call -- `core/row_norms.mojo`, `cluster/checks/reduce_by_key.mojo`,
`cluster/checks/plus_plus.mojo` -- and no equivalent. This file is that
equivalent, and rows 19-24 of the ledger route through it.

ONE FOLD, TWO CALLERS
---------------------
`gbdt/targets/kernel/pointwise_targets.mojo`'s `pinned_block_sum` used to
carry its own copy of the halving tree (a lane boundary, not a judgement
that two folds were fine). Since 2026-09-09 its IDENTICAL arm imports
`two_phase_halving_sum` from here, so the identity claim rides on one
implementation; `check_pinned_fold_shape` in
`cluster/checks/kmeans_identity_check.mojo` gates that the fold is a pure
function of the value vector and not of the lane width.

THE CONTRACT, and it is the same one `block.sum` already carries
-----------------------------------------------------------------
- EVERY thread of the block must call this. Threads with no data pass 0.0.
- Only thread 0's return value is meaningful.
- `block_size` must be a power of two, so the halving fold is exact.
- The trailing `barrier()` protects the shared slab so back-to-back calls
  in one kernel are safe.

Under `NUMERIC_FAST` this is the library call, bit for bit, and compiles to
exactly what was there before.
"""

from std.bit import log2_floor
from std.gpu import thread_idx, WARP_SIZE
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.primitives.block import sum as block_sum
from max.gpu.primitives.block import prefix_sum as block_prefix_sum
from max.gpu.sync import barrier

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL


@always_inline
def pinned_block_sum[block_size: Int](value: Float32) -> Float32:
    """`block.sum` under FAST; a lane-width-independent halving tree under
    IDENTICAL.

    The IDENTICAL arm writes every thread's value into a `block_size` slab
    of threadgroup memory and folds it `red[t] += red[t + step]` for
    `step = block_size/2, ..., 1`. No warp primitive appears anywhere in
    it, so nothing in the fold can consult the hardware's lane width, and
    the sequence of additions is a pure function of `block_size` -- the
    same on Metal, PTX and AMDGPU.

    It is NOT the same sum as the library call: a halving tree and CUB's
    warp-then-block shape combine different partials. IDENTICAL bits
    therefore differ from FAST bits on Apple BY DESIGN, exactly as
    `identical_mul_add` does, and what is purchased is that IDENTICAL's
    bits are the same bits everywhere.
    """
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return halving_block_sum[block_size](value)
    else:
        return block_sum[block_size=block_size](value)


def halving_block_sum[block_size: Int](value: Float32) -> Float32:
    """The IDENTICAL arm of `pinned_block_sum`, callable in EVERY mode.

    DEVIATION 2291 (2026-09-08): `block_sum` refuses block sizes that are
    not greater than the warp size ("Block size must be a greater than warp
    size", block.mojo:186), so a 32-thread block cannot use the library
    fold on a 64-wide wavefront at all: the FAST and DETERMINISTIC builds of
    the estimators binding failed to instantiate `qr_panel_kernel` on
    gfx942 (the 0.7.0 release build, 2026-09-08). A kernel whose block is
    narrower than every vendor's widest warp calls this tree directly; the
    tree is the same sequence of additions on every backend, so IDENTICAL
    bits are untouched and FAST bits on that kernel move to the same value.
    """
    return two_phase_halving_sum[block_size](value)


@always_inline
def two_phase_halving_sum[block_size: Int](value: Float32) -> Float32:
    """The halving tree `red[t] += red[t + step]`, `step = block_size/2 ..
    1`, folded in THREE barriers instead of `log2(block_size) + 2`.

    Same additions, same operands, same order, so the same bits: the tree's
    first `log2(G)` steps only ever combine elements that share `t mod P`
    (`P = min(block_size, 16)`, `G = block_size / P`), so thread `t < P`
    folds its own strided group in registers (phase 1); the remaining
    `log2(P)` steps read only the `P` partials, so every thread folds those
    redundantly in registers (phase 2) and no broadcast is needed. The
    trailing barrier keeps the slab safe across back-to-back calls.
    """
    comptime assert block_size > 0 and (block_size & (block_size - 1)) == 0, (
        "the halving tree needs a power-of-two block"
    )
    comptime P = 16 if block_size > 16 else block_size
    comptime G = block_size // P
    var tid = Int(thread_idx.x)
    var red = stack_allocation[
        block_size,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    red[tid] = value
    barrier()
    if tid < P:
        var v = InlineArray[Float32, G](fill=Float32(0.0))
        comptime for j in range(G):
            v[j] = red[tid + j * P]
        comptime for k in range(log2_floor(G)):
            comptime S = G >> (k + 1)
            comptime for j in range(S):
                v[j] = v[j] + v[j + S]
        # in place: red[tid] is read by this thread alone (j == 0 above)
        red[tid] = v[0]
    barrier()
    var w = InlineArray[Float32, P](fill=Float32(0.0))
    comptime for t in range(P):
        w[t] = red[t]
    comptime for k in range(log2_floor(P)):
        comptime S = P >> (k + 1)
        comptime for t in range(S):
            w[t] = w[t] + w[t + S]
    var total = w[0]
    barrier()
    return total


# ===========================================================================
# THE SELECTIONS (DEVIATION 528, IDENTITY_PATHS row 30's compile half)
# ===========================================================================
#
# `pinned_block_sum` above exists because a SUM's fold shape changes its
# bits. A MAX or a MIN is a selection over a total order, exactly associative
# and commutative, so no fold shape can move it and by that reasoning these
# two should not need to exist at all.
#
# THEY EXIST FOR A DIFFERENT REASON, AND IT IS NOT NUMERIC: the library
# primitive REFUSES TO COMPILE at these block sizes on a 64-wide wavefront.
#
#     max/mojo/max/gpu/primitives/block.mojo:186:
#     constraint failed: Block size must be a greater than warp size
#
# Measured on an MI325X, 2026-08-23, by the E2 lane's AMD leg:
# `bindings/build_estimators.sh` FAILED there, and the two call sites were
# `block_max`/`block_min` at `SIGNFLIP_TPB = 32` in
# `decomposition/impl/linalg/detail/pca.mojo`. So **PCA and truncated SVD
# did not build on AMD at all**, in either mode, while every gate on this
# side was green -- a whole-section failure that one M4 cannot see and that
# no amount of bit-comparison would have found, because there were no bits.
#
# 32 is not an arbitrary choice there either: the sign rule's tie-break is
# stated over lanes and the fixtures plant ties both across and within a
# 32-lane group, so widening the block to satisfy the constraint would move
# what the check is checking. Replacing the primitive is the smaller change.
#
# NO MODE GATE, deliberately, and this is the difference from
# `pinned_block_sum`. A halving selection returns exactly what CUB's shape
# returns, so there is no FAST arm to preserve and no IDENTICAL arm to buy
# anything: swapping these in moves ZERO bits on every column, which
# `check_sign_flip_matches_host_rule` asserts against a fold-free host scan.
# A mode gate here would imply a difference that does not exist.
#
# THE ±0.0 AND NaN CAVEAT, because a selection is only exactly commutative
# away from them (IDENTITY_PATHS row 13). `max(+0.0, -0.0)` may return
# either, so WHICH zero survives is a fold-order question after all, and a
# NaN operand makes `>` false in both directions. Neither reaches these at
# their one call site: pass 1 folds `abs(...)`, and `abs(-0.0)` is `+0.0`, so
# no negative zero is ever compared; and the per-thread partial is seeded
# `+0.0` and only updated on a strict `>`, so a NaN never enters it. A future
# caller that cannot say the same about its inputs must state why before
# using these.


@always_inline
def pinned_block_max[block_size: Int](value: Float32) -> Float32:
    """Block-wide max through threadgroup memory, no lane primitive.

    Same contract as `pinned_block_sum`: EVERY thread of the block calls
    it, threads with no data pass the identity (`+0.0` for a fold over
    magnitudes), only thread 0's return is meaningful, `block_size` is a
    power of two, and the trailing `barrier()` protects the slab so
    back-to-back calls in one kernel are safe.

    Unlike the sum, this is bit-for-bit the library's answer on any width;
    see the block comment above for why it exists anyway.
    """
    var tid = Int(thread_idx.x)
    var red = stack_allocation[
        block_size,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    red[tid] = value
    barrier()
    var step = block_size // 2
    while step > 0:
        if tid < step:
            var other = red[tid + step]
            if other > red[tid]:
                red[tid] = other
        barrier()
        step //= 2
    var total = red[0]
    barrier()
    return total


@always_inline
def pinned_block_min[block_size: Int](value: Float32) -> Float32:
    """Block-wide min through threadgroup memory, no lane primitive.

    The twin of `pinned_block_max`; same contract, and the identity a
    dataless thread passes is whatever sentinel the caller uses as its "no
    candidate" value (at the sign-flip call site that is `Float32(n)`).
    """
    var tid = Int(thread_idx.x)
    var red = stack_allocation[
        block_size,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    red[tid] = value
    barrier()
    var step = block_size // 2
    while step > 0:
        if tid < step:
            var other = red[tid + step]
            if other < red[tid]:
                red[tid] = other
        barrier()
        step //= 2
    var total = red[0]
    barrier()
    return total


# ===========================================================================
# THE FLOAT BLOCK SCAN (host-cpu-identity lane, 2026-09-30)
# ===========================================================================
#
# `max.gpu.primitives.block.prefix_sum` has the same residue as `block.sum`
# above: a Hillis-Steele scan inside each HARDWARE warp
# (`res += shuffle_up(res, 1, 2, 4, ...)`, std/gpu/primitives/warp.mojo:1084
# at max 26.5.0), the warp totals scanned by warp 0 the same way, and the
# previous warps' inclusive prefix added (max/gpu/primitives/block.mojo:672).
# The warp is 32 lanes on Apple and NVIDIA and 64 on AMD CDNA, so on Float32
# the AMD scan is a different association of the same values and differs in
# the last bits. In k-means++ (`cluster/checks/plus_plus.mojo`) that scan
# is the cumulative d^2 weight the draw binary-searches, and on the MI325X
# one draw of the IVF 40000-row quantizer (pick 890 of 1024) landed on the
# other side of a boundary: every later centroid, the index and the search
# moved (bench/results/host-cpu-identity-20260930). The host oracle
# (`cluster/host/kmeans_oracle.mojo::host_block_prefix_sum`) has always
# replayed the 32-wide shape; this is the device half of that claim.


@always_inline
def pinned_block_prefix_sum[
    block_size: Int, exclusive: Bool = False
](value: Float32) -> Float32:
    """`block.prefix_sum` at a 32-LANE warp on every vendor, under IDENTICAL.

    Where the hardware warp is 32 (Apple, NVIDIA) this IS the library call,
    so those columns keep their bits by construction. Elsewhere (AMD CDNA,
    64) the library's 32-wide shape is replayed through threadgroup memory,
    step for step: the in-warp Hillis-Steele scan (lane l adds lane
    l - offset's previous value when l >= offset, offsets 1 .. 16), the
    exclusive shift (lane 0 gets 0), the warp total (`inclusive`, or
    `exclusive + value` for an exclusive scan, as the library forms it),
    warp 0's inclusive scan of the totals, then the previous warps' prefix
    added to every warp past the first. Under FAST it is the library call.

    Same contract as the library: EVERY thread of the block calls it, and
    `block_size` is a multiple of 32 no larger than 32 * 32.
    """
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL or WARP_SIZE == 32:
        return block_prefix_sum[block_size=block_size, exclusive=exclusive](
            value
        )
    else:
        return warp32_block_prefix_sum[block_size, exclusive](value)


def warp32_block_prefix_sum[
    block_size: Int, exclusive: Bool = False
](value: Float32) -> Float32:
    """The 32-lane replay of `pinned_block_prefix_sum`, callable at any
    warp width and in every mode (its check, `cluster/checks/
    pinned_scan_check.mojo`, compares it with the library scan on a 32-lane
    GPU and with the host oracle anywhere)."""
    comptime W = 32
    comptime assert block_size % W == 0 and block_size <= W * W, (
        "the 32-lane scan needs a block that is a multiple of 32, at"
        " most 1024 threads"
    )
    comptime n_warps = block_size // W
    var tid = Int(thread_idx.x)
    var lane = tid % W
    var wid = tid // W
    var slab = stack_allocation[
        block_size,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var tot = stack_allocation[
        W,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    # Step 1: the in-warp Hillis-Steele scan, each step reading the
    # previous step's values (a shuffle_up reads before anyone writes).
    var res = value
    slab[unsafe_offset=tid] = res
    barrier()
    comptime for i in range(5):
        comptime offset = 1 << i
        var n = Float32(0.0)
        if lane >= offset:
            n = slab[unsafe_offset=tid - offset]
        barrier()
        if lane >= offset:
            res += n
        slab[unsafe_offset=tid] = res
        barrier()
    comptime if exclusive:
        if lane == 0:
            res = Float32(0.0)
        else:
            res = slab[unsafe_offset=tid - 1]
    # Step 2: the last lane of each warp stores the warp's INCLUSIVE sum.
    if lane == W - 1:
        var inclusive_warp_sum = res
        comptime if exclusive:
            inclusive_warp_sum += value
        tot[unsafe_offset=wid] = inclusive_warp_sum
    if tid < W and tid >= n_warps:
        # lanes past the warp count read nothing a lower lane uses; zero
        # keeps them finite
        tot[unsafe_offset=tid] = Float32(0.0)
    barrier()
    # Step 3: warp 0 scans the warp totals, inclusive, the same way.
    var p = Float32(0.0)
    if tid < W:
        p = tot[unsafe_offset=tid]
    comptime for i in range(5):
        comptime offset = 1 << i
        var n = Float32(0.0)
        if tid < W and tid >= offset:
            n = tot[unsafe_offset=tid - offset]
        barrier()
        if tid < W and tid >= offset:
            p += n
        if tid < W:
            tot[unsafe_offset=tid] = p
        barrier()
    # Step 4: the previous warps' prefix.
    if wid > 0:
        res += tot[unsafe_offset=wid - 1]
    barrier()
    return res
