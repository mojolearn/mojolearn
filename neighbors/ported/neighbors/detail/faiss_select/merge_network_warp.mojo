# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's PORTED_MAP.tsv and in this file's own docstring. See NOTICE.
# Portions of this file are derived from FAISS, via RAFT's vendored copy at
# raft/neighbors/detail/faiss_select/, and are used under the MIT license.
#
#   Copyright (c) Facebook, Inc. and its affiliates.
#
#   Permission is hereby granted, free of charge, to any person obtaining a
#   copy of this software and associated documentation files (the
#   "Software"), to deal in the Software without restriction, including
#   without limitation the rights to use, copy, modify, merge, publish,
#   distribute, sublicense, and/or sell copies of the Software, and to
#   permit persons to whom the Software is furnished to do so, subject to
#   the following conditions:
#
#   The above copyright notice and this permission notice shall be included
#   in all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
#   IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
#   CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
#   TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
#   SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# See NOTICE, section "Derived from FAISS", for why this is the one
# obligation here that is not Apache-2.0.

"""FAISS WarpSelect's register-resident merge networks.

PORT OF, at RAFT `661a3b8` (branch-25.08):
  * `raft/neighbors/detail/faiss_select/Comparators.cuh` (the `float`
    specialization only; `half` has no counterpart here)
  * `raft/neighbors/detail/faiss_select/MergeNetworkUtils.cuh` in full
  * `raft/neighbors/detail/faiss_select/MergeNetworkWarp.cuh`, the
    power-of-two path in full

Do not improve.

THIS IS NOT `matrix/detail/select_warpsort.cuh` AND NOT A RENAME OF IT
----------------------------------------------------------------------
RAFT ships the FAISS WarpSelect design TWICE, as two unrelated files:

  `matrix/detail/select_warpsort.cuh` + `util/bitonic_sort.cuh`
      -> ported at `neighbors/gbdt/matrix/detail/select_warpsort.mojo`.
      A block-wide `block_kernel` that reads a materialized distance row
      from GLOBAL memory and tree-merges per-warp queues through SHARED
      memory. `select_k-inl.cuh:38` routes `2 < k <= 256` here.

  `neighbors/detail/faiss_select/{Select,MergeNetworkWarp}.cuh`  <- THIS
      A queue with NO shared memory and no block phase at all. It is a
      struct a kernel keeps in REGISTERS and feeds one value at a time.
      `cuvs/src/neighbors/detail/fused_l2_knn.cuh:221-222` instantiates it
      inside the distance kernel, which is why that kernel never writes a
      distance matrix.

Neither subsumes the other. They differ in the network too, not just the
plumbing: `bitonic_sort.cuh` sorts data STRIDED across the subwarp with one
element per lane per array slot and takes `warp_width` as a RUNTIME argument;
this file sorts `WarpSize * N` elements with `N` registers per lane, pins
`WarpSize == 32` in a `static_assert` (`MergeNetworkWarp.cuh:501`), and
expands the whole network at compile time. `bitonic_sort.cuh` compares
strictly and keeps the incumbent on a tie; this file deliberately runs BOTH
`<` and `>` on the two sides of an exchange so that the two lanes of a pair
agree, which their comment at `MergeNetworkWarp.cuh:38-64` explains at
length and which is NOT the same tie behavior.

THE ONE CONSTRUCT THAT DOES NOT PORT ELSEWHERE DOES NOT APPEAR HERE
--------------------------------------------------------------------
`select_warpsort.mojo` records that `set_k_th_` (`select_warpsort.cuh:329`)
is untranslatable: it is `__shfl_sync` with an explicit `width` AND a source
lane `k - 1` that their own comment says is deliberately allowed to exceed
that width, so the `% width` inside the intrinsic is load-bearing. Mojo's
`shuffle_idx` has no `width`.

**That blocker does not reach this file, and it does not reach
`fused_l2_knn` either.** Every shuffle here is one of:

  * `shfl_xor(v, stride)` with `stride < 32` -- exactly Mojo's
    `shuffle_xor(v, stride)`.
  * `shfl_xor(v, WarpSize - 1)`, i.e. stride 31 -- still inside one warp.
  * `shfl(warpK[...], kLane)` in `Select.cuh:427`, where
    `kLane = (k - 1) % WarpSize` is ALREADY reduced modulo the warp size by
    their own constructor (`Select.cuh:363`). The lane is in `[0, 31]`, there
    is no width argument, and Mojo's `shuffle_idx(v, lane)` is that call.

So the register-resident queue is reachable in Mojo, and the fusion it
enables is not blocked by the `set_k_th_` wall.

DEVIATIONS
----------
1. **Register arrays are `SIMD` values, and a non-power-of-two `N` is
   PADDED.** RAFT holds `K k[N]` and nvcc keeps it in registers; Mojo's
   `stack_allocation` without an address space is thread-local MEMORY
   (`PORTING.md 26`), so `SIMD` is the only register form. Mojo's `SIMD`
   width must be a power of two and `NumThreadQ` is 3 for the k>32 fused
   instantiation (`fused_l2_knn.cuh:760`), so the array is widened to 4 and
   element 3 is NEVER read or written: every loop in this file runs
   `range(N)` with the true `N`. Padding lanes are initialized to the same
   sentinel as the rest so that a stray read would be inert rather than
   garbage.

2. **`swap`/`assign` are written as `if`, not as a conditional
   expression.** `MergeNetworkUtils.cuh:13-24` writes `x = swap ? y : x`.
   Identical arithmetic; `PORTING.md 19` forbids the conditional form for
   pointers in this tree and an explicit branch is the same value here.

3. **`@always_inline` everywhere.** Theirs is `inline __device__` on every
   function in this file. Dropping it is not cosmetic: with these functions
   left as callable device functions the APPLE METAL BACKEND CRASHES rather
   than diagnosing, which is what happened to `select_warpsort.mojo` before
   the annotation was restored there. Their annotation, restored.

NOT PORTED, and each is a row in `UNPORTED.tsv`
------------------------------------------------
  * `Comparator<half>` (`Comparators.cuh:23-27`). Metal has no `half`
    comparison intrinsics to mirror and nothing asks for it.
  * The NON-power-of-two `BitonicMergeStep` specializations,
    `MergeNetworkWarp.cuh:221-304` (Low) and `:305-388` (High). These are
    UNREACHABLE for every instantiation `fused_l2_knn.cuh` makes, and that
    is checked rather than assumed: its only two instantiations are
    `NumWarpQ=32, NumThreadQ=2` and `NumWarpQ=64, NumThreadQ=3`
    (`fused_l2_knn.cuh:743-771`), so `kNumWarpQRegisters` is 1 or 2 and every
    `BitonicMergeStep` reached has `N` in {1, 2}. `BitonicSortStep` DOES see
    the odd `N = 3` and is ported in full, because its generic case splits
    `N` into `N/2` and `N - N/2` and handles odd sizes without ever calling a
    non-power-of-two merge. A `comptime assert` makes the assumption fail
    at compile time rather than silently.
"""

from std.gpu.primitives.warp import lane_id, shuffle_xor


#: `static const int WarpSize = 32;`, `util/cuda_dev_essentials.cuh:83`, and
#: `static_assert(WarpSize == 32)` at `MergeNetworkWarp.cuh:501`. Theirs,
#: pinned, and wrong on AMD -- the same cross-vendor hazard
#: `select_warpsort.mojo` DEVIATION 9 names.
comptime WARP_LANES = 32


@always_inline
def is_pow2(n: Int) -> Bool:
    """`utils::isPowerOf2`, `StaticUtils.h`."""
    return n > 0 and (n & (n - 1)) == 0


@always_inline
def pow2_ceil(n: Int) -> Int:
    """The SIMD width that holds `n` registers. See DEVIATION 1. This is a
    PORT ARTIFACT: RAFT needs no such thing because a C array has no width
    constraint."""
    var w = 1
    while w < n:
        w = w * 2
    return w


# =========================================================================
# `Comparators.cuh` and `MergeNetworkUtils.cuh`
# =========================================================================


@always_inline
def comp_lt(a: Float32, b: Float32) -> Bool:
    """`Comparator<T>::lt`, `Comparators.cuh:17`."""
    return a < b


@always_inline
def comp_gt(a: Float32, b: Float32) -> Bool:
    """`Comparator<T>::gt`, `Comparators.cuh:19`."""
    return a > b


# =========================================================================
# `warpBitonicMergeLE16`, `MergeNetworkWarp.cuh:84-137`
#
# Merges `WarpSize / 2L` lists in parallel using warp shuffles. One element
# per lane. `L <= 16` because 32 threads are needed for the shuffle merge.
# If `IsBitonic` is false the first stage is reversed, so the input does not
# have to be sorted directionally.
# =========================================================================


@always_inline
def warp_bitonic_merge_le16[
    L: Int, dir: Bool, is_bitonic: Bool
](mut key: Float32, mut val: UInt32):
    """`warpBitonicMergeLE16<K, V, L, Dir, Comp, IsBitonic>`, `:83-136`."""
    comptime assert is_pow2(L), "L must be a power-of-2"
    comptime assert L <= WARP_LANES // 2, "merge list size must be <= 16"

    var lane = Int(lane_id())

    @parameter
    if not is_bitonic:
        # `:91-113`. Reverse the first comparison stage: merging a list of
        # size 8 has the exchanges 0 <-> 15, 1 <-> 14, ...
        var other_k = shuffle_xor(key, UInt32(2 * L - 1))
        var other_v = shuffle_xor(val, UInt32(2 * L - 1))
        var small = (lane & L) == 0
        var s: Bool

        @parameter
        if dir:
            # Both comparisons are run on purpose; see their comment at
            # `:38-63` on why this beats a lexicographic ordering.
            if small:
                s = comp_gt(key, other_k)
            else:
                s = comp_lt(key, other_k)
        else:
            if small:
                s = comp_lt(key, other_k)
            else:
                s = comp_gt(key, other_k)
        if s:
            key = other_k
            val = other_v

    # `#pragma unroll for (stride = IsBitonic ? L : L / 2; stride > 0;
    #  stride /= 2)`, `:115-135`. Comptime because their `#pragma unroll`
    # over a comptime bound is comptime, and because a `SIMD` caller wants
    # no runtime trip count here.
    comptime FIRST = L if is_bitonic else L // 2

    @parameter
    for step in range(32):
        comptime STRIDE = FIRST >> step

        @parameter
        if STRIDE > 0:
            var other_k = shuffle_xor(key, UInt32(STRIDE))
            var other_v = shuffle_xor(val, UInt32(STRIDE))
            var small = (lane & STRIDE) == 0
            var s: Bool

            @parameter
            if dir:
                if small:
                    s = comp_gt(key, other_k)
                else:
                    s = comp_lt(key, other_k)
            else:
                if small:
                    s = comp_lt(key, other_k)
                else:
                    s = comp_gt(key, other_k)
            if s:
                key = other_k
                val = other_v


# =========================================================================
# `BitonicMergeStep`, power-of-two specializations,
# `MergeNetworkWarp.cuh:140-219`
#
# `BASE` is this port's addition and carries no meaning of its own: it is
# where their copy of a sub-array into `newK[N/2]` and back went, because a
# `SIMD` element index must be comptime and a `SIMD` cannot be offset by a
# pointer. Their copy in and copy out is a no-op on the values.
#
# Their `Low` template parameter is NOT used by the power-of-two
# specialization at all (compare `:157` and `:181` with the non-power-of-two
# ones at `:218` and `:306`, which do use it). It is carried here anyway so
# the call sites read like theirs.
# =========================================================================


@always_inline
def bitonic_merge_step[
    N: Int, BASE: Int, ARR: Int, dir: Bool, low: Bool
](mut keys: SIMD[DType.float32, ARR], mut vals: SIMD[DType.uint32, ARR]):
    """`BitonicMergeStep<K, V, N, Dir, Comp, Low, true>::merge`, `:156-212`.
    """
    comptime assert is_pow2(N), "must be power of 2"

    @parameter
    if N == 1:
        # `:157-163`. All merges eventually call this.
        var key = keys[BASE]
        var val = vals[BASE]
        warp_bitonic_merge_le16[16, dir, True](key, val)
        keys[BASE] = key
        vals[BASE] = val
    else:
        # `:171-184`, the in-register compare-exchange across the halves.
        @parameter
        for i in range(N // 2):
            var ka = keys[BASE + i]
            var kb = keys[BASE + i + N // 2]
            var s: Bool

            @parameter
            if dir:
                s = comp_gt(ka, kb)
            else:
                s = comp_lt(ka, kb)
            if s:
                keys[BASE + i] = kb
                keys[BASE + i + N // 2] = ka
                var va = vals[BASE + i]
                vals[BASE + i] = vals[BASE + i + N // 2]
                vals[BASE + i + N // 2] = va

        # `:186-197` and `:199-211`, the two halves. Theirs copies each half
        # into `newK`/`newV` and back; `BASE` does that without the copy.
        bitonic_merge_step[N // 2, BASE, ARR, dir, True](keys, vals)
        bitonic_merge_step[N // 2, BASE + N // 2, ARR, dir, False](keys, vals)


# =========================================================================
# `warpMergeAnyRegisters`, `MergeNetworkWarp.cuh:390-442`
# =========================================================================


@always_inline
def warp_merge_any_registers[
    N1: Int, N2: Int, ARR1: Int, ARR2: Int, dir: Bool, full_merge: Bool
](
    mut k1: SIMD[DType.float32, ARR1],
    mut v1: SIMD[DType.uint32, ARR1],
    mut k2: SIMD[DType.float32, ARR2],
    mut v2: SIMD[DType.uint32, ARR2],
):
    """`warpMergeAnyRegisters<K, V, N1, N2, Dir, Comp, FullMerge>`,
    `:391-436`.

    Merges a sorted k/v list of `WarpSize * N1` with a sorted k/v list of
    `WarpSize * N2`, for any `N1, N2 >= 1`.
    """
    # The non-power-of-two `BitonicMergeStep` specializations
    # (`MergeNetworkWarp.cuh:221`, `:305`) are deliberately not ported;
    # `fused_l2_knn.cuh` instantiates only N1 in {1, 2}. Fail loudly rather
    # than silently if that ever stops being true.
    comptime assert is_pow2(N1), "warp_merge_any_registers: N1 must be a power of two"

    comptime SMALLEST = N1 if N1 < N2 else N2

    @parameter
    for i in range(SMALLEST):
        comptime IA = N1 - 1 - i
        var ka = k1[IA]
        var va = v1[IA]
        var kb = k2[i]
        var vb = v2[i]

        var other_ka = ka
        var other_va = va

        @parameter
        if full_merge:
            # `:411-414`. Only needed when the second list must survive.
            other_ka = shuffle_xor(ka, UInt32(WARP_LANES - 1))
            other_va = shuffle_xor(va, UInt32(WARP_LANES - 1))

        var other_kb = shuffle_xor(kb, UInt32(WARP_LANES - 1))
        var other_vb = shuffle_xor(vb, UInt32(WARP_LANES - 1))

        # `ka` is always first in the list, so our own lane does not enter
        # this comparison. Theirs, `:419-422`.
        var swap_a: Bool

        @parameter
        if dir:
            swap_a = comp_gt(ka, other_kb)
        else:
            swap_a = comp_lt(ka, other_kb)
        if swap_a:
            k1[IA] = other_kb
            v1[IA] = other_vb

        @parameter
        if full_merge:
            var swap_b: Bool

            @parameter
            if dir:
                swap_b = comp_lt(kb, other_ka)
            else:
                swap_b = comp_gt(kb, other_ka)
            if swap_b:
                k2[i] = other_ka
                v2[i] = other_va

    bitonic_merge_step[N1, 0, ARR1, dir, True](k1, v1)

    @parameter
    if full_merge:
        comptime assert is_pow2(N2), "warp_merge_any_registers: FULL merge needs a power-of-two N2"
        bitonic_merge_step[N2, 0, ARR2, dir, False](k2, v2)


# =========================================================================
# `BitonicSortStep` / `warpSortAnyRegisters`, `MergeNetworkWarp.cuh:444-517`
#
# THIS one does handle an odd `N`, and it must: `fused_l2_knn.cuh:760`
# instantiates the queue with `NumThreadQ = 3`. Its generic case splits `N`
# into `N/2` and `N - N/2` and merges them, so an odd size never reaches a
# non-power-of-two BitonicMergeStep.
# =========================================================================


@always_inline
def _copy_out[
    N: Int, BASE: Int, ARR: Int, SUB: Int
](
    keys: SIMD[DType.float32, ARR],
    vals: SIMD[DType.uint32, ARR],
    mut sk: SIMD[DType.float32, SUB],
    mut sv: SIMD[DType.uint32, SUB],
):
    """Their `newK[i] = k[i + off]` copy, `:456-460` / `:466-470`."""

    @parameter
    for i in range(N):
        sk[i] = keys[BASE + i]
        sv[i] = vals[BASE + i]


@always_inline
def _copy_in[
    N: Int, BASE: Int, ARR: Int, SUB: Int
](
    mut keys: SIMD[DType.float32, ARR],
    mut vals: SIMD[DType.uint32, ARR],
    sk: SIMD[DType.float32, SUB],
    sv: SIMD[DType.uint32, SUB],
):
    """Their `k[i + off] = newK[i]` copy back, `:477-486`."""

    @parameter
    for i in range(N):
        keys[BASE + i] = sk[i]
        vals[BASE + i] = sv[i]


@always_inline
def bitonic_sort_step[
    N: Int, BASE: Int, ARR: Int, dir: Bool
](mut keys: SIMD[DType.float32, ARR], mut vals: SIMD[DType.uint32, ARR]):
    """`BitonicSortStep<K, V, N, Dir, Comp>::sort`, `:442-486`, with the
    `N == 1` specialization at `:490-503`."""

    @parameter
    if N == 1:
        # `:493-502`. 1 -> WarpSize in multiples of 2. Their
        # `static_assert(WarpSize == 32)` is why this list is five calls and
        # not a loop.
        var key = keys[BASE]
        var val = vals[BASE]
        warp_bitonic_merge_le16[1, dir, False](key, val)
        warp_bitonic_merge_le16[2, dir, False](key, val)
        warp_bitonic_merge_le16[4, dir, False](key, val)
        warp_bitonic_merge_le16[8, dir, False](key, val)
        warp_bitonic_merge_le16[16, dir, False](key, val)
        keys[BASE] = key
        vals[BASE] = val
    else:
        comptime SIZE_A = N // 2
        comptime SIZE_B = N - SIZE_A
        comptime WA = pow2_ceil(SIZE_A)
        comptime WB = pow2_ceil(SIZE_B)

        var ak = SIMD[DType.float32, WA](keys[BASE])
        var av = SIMD[DType.uint32, WA](vals[BASE])
        _copy_out[SIZE_A, BASE, ARR, WA](keys, vals, ak, av)
        bitonic_sort_step[SIZE_A, 0, WA, dir](ak, av)

        var bk = SIMD[DType.float32, WB](keys[BASE])
        var bv = SIMD[DType.uint32, WB](vals[BASE])
        _copy_out[SIZE_B, BASE + SIZE_A, ARR, WB](keys, vals, bk, bv)
        bitonic_sort_step[SIZE_B, 0, WB, dir](bk, bv)

        warp_merge_any_registers[SIZE_A, SIZE_B, WA, WB, dir, True](
            ak, av, bk, bv
        )

        _copy_in[SIZE_A, BASE, ARR, WA](keys, vals, ak, av)
        _copy_in[SIZE_B, BASE + SIZE_A, ARR, WB](keys, vals, bk, bv)


@always_inline
def warp_sort_any_registers[
    N: Int, ARR: Int, dir: Bool
](mut keys: SIMD[DType.float32, ARR], mut vals: SIMD[DType.uint32, ARR]):
    """`warpSortAnyRegisters<K, V, N, Dir, Comp>`, `:508-512`."""
    bitonic_sort_step[N, 0, ARR, dir](keys, vals)
