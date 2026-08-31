# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CatBoost's ONE pointwise histogram loop, written once and shared.

PORT OF `catboost/cuda/methods/kernel/compute_point_hist2_loop.cuh` at
CatBoost `54a8143a`. Transliterated. Do not improve.

This is the spine of the pointwise family (`PORTING.md` 91 B, `NEXT_TWO.md`
rung 1). Every one of their `pointwise_hist2*` kernels is this loop
instantiated with a different accumulator; the loop knows nothing about bin
widths, shared-memory layouts or writeback, and the accumulator knows nothing
about striping, alignment or block-per-feature splitting.

WHY IT IS SHARED HERE AND DUPLICATED IN THE OTHER FAMILY
--------------------------------------------------------
`PORTING.md` 13 records that the greedy-subsets family carries a COPY of its
loop in every kernel, and that the duplication once shipped a silently wrong
histogram because a fix landed in one copy and not the other. The stated
reason was that Mojo could not pass a shared-memory pointer across a function
boundary.

That was false, and `checks/shared_pointer_probe.mojo` measures it: the
callee simply has to take the origin as a PARAMETER rather than assert
`MutAnyOrigin`, which is a different origin and not a wildcard. So this
family is written CatBoost's way from the first line -- one loop, one
`PointHist2` trait standing in for their `THist` template argument -- and the
class of bug item 13 describes cannot occur in it.

THEIR THREE ENTRY POINTS, AND WHAT PICKS BETWEEN THEM
------------------------------------------------------
    ComputeHistogram    scalar, `N` points per thread per iteration (`:12`)
    ComputeHistogram2   `uint2`/`float2` loads, 2 points   (`:123`)
    ComputeHistogram4   `uint4`/`float4` loads, 4 points   (`:245`)

All three GATHER through `indices` -- the pointwise family has no
direct-load variant, unlike the greedy-subsets one. The wider variants are
not a different algorithm: they load the same points in the same per-lane
order and hand them to `AddPoint2` / `AddPoint4`, which their accumulators
implement as two or four `AddPoint`s in a fixed order. Vector width is a
memory-transaction choice, not a numeric one, and this port keeps that true
by requiring the same of every accumulator it admits.

THE ALIGNMENT PEEL, WHICH IS THE PART THAT LOOKS LIKE NOISE AND IS NOT
-----------------------------------------------------------------------
Each entry point peels an unaligned HEAD and an unaligned TAIL and gives both
to the blocks with `blockIdx.x % BLOCKS_PER_FEATURE == 0`, so that the main
loop sees a whole number of aligned strides and every load in it is
coalesced. The head length is `min(dsSize, Q - (offset & (Q-1)))` for a
quantum `Q` of 32, 128 and 128 respectively, and the tail is
`dsSize & (Q'-1)` for 32, 64 and 128.

**That peel decides WHICH block adds WHICH point, so it is a float summation
ORDER**, and this port therefore pins its quantum rather than deriving it
from the device (see `ALIGN_LANES`).

GPU-AGNOSTIC
------------
Their `threadIdx.x & 31` / `threadIdx.x / 32` is a thread-to-column
permutation, not a wavefront assumption: at `HIST_BLOCK_COUNT == 1` it is the
identity for any block size, and at any wavefront width it is a bijection.
It is transcribed with the pinned 32 rather than a queried lane width, which
keeps the column each thread reads -- and therefore the order the additions
land in -- the same on Apple, NVIDIA and AMD. There are no lane intrinsics in
this file.
"""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.gpu.intrinsics import ldg
from max.gpu.sync import barrier

from checks.kernel_matrix import (
    TARGET_COLUMN,
    requires_uniform_iteration_for,
)


#: PORTING.md 11's row, and here it is a PRECONDITION rather than a tuning
#: knob. CatBoost's accumulators sync a `tiled_partition<8>` INSIDE
#: `AddPoint` (`pointwise_hist2_one_byte_5bit.cu:79`, `:108`, `:147`), which
#: is lane-local, so lanes with different iteration counts never wait on
#: each other. Mojo exposes only a threadgroup `barrier()`, and a
#: threadgroup barrier some threads reach and others skip is undefined --
#: item 11 records that this exact mistake made every feature's histogram
#: read 0.0 in the other family.
#:
#: So the body loops below run ONE iteration count for the whole block and
#: let threads past their own count contribute a zero point. Adding 0.0
#: changes no sum, which is what makes this a scheduling change and not a
#: numeric one.
comptime UNIFORM_ITERATION = requires_uniform_iteration_for[TARGET_COLUMN]()


#: NUMERIC ROW. The alignment quantum, pinned at CatBoost's 32.
#:
#: It is NOT `lane_width_for[...]`, and the difference is deliberate. Their
#: 32 does two jobs at once -- it is the warp width AND the coalescing
#: quantum -- and only the second is what the peel is for. Deriving it from a
#: 64-wide wavefront would move points between the head, the body and the
#: tail, which are summed by different blocks, which changes the float
#: result. Pinning it keeps one summation shape on every vendor at the cost
#: of nothing but a scheduling preference.
comptime ALIGN_LANES = 32


trait PointHist2(Movable):
    """Their `THist` template argument (`:12`, `:123`, `:245`).

    CatBoost passes an accumulator TYPE and calls four methods on it. The
    loop never looks inside, which is exactly what makes one loop able to
    serve every bin width.

    THE CONTRACT THE LOOP RELIES ON, and every implementor owes it:

    * `add_point_2` must be `add_point` twice, in the order (x, y), and
      `add_point_4` must be `add_point` four times in the order (x, y, z, w).
      The wide variants exist to widen the LOAD, not to change the sum.
      An accumulator that reorders them makes vector width a numeric knob
      and breaks the identity `NEXT_TWO.md` rung 2 is gated on.
    * `reduce` is called with every thread of the block present and after a
      `barrier()`, because theirs is (`:126`, `:234`, `:344`).
    * a bin of 0 with a target and weight of 0 must be a no-op, since that
      is what the peel feeds threads outside the head or tail (`:36-39`).
    """

    def add_point(mut self, ci: UInt32, t: Float32, w: Float32, row: UInt32):
        """`AddPoint(ci, wt, w)`, plus a row id CatBoost does not pass.

        ============== DEVIATION BLOCK (PORTING.md 93) ==============
        `row` is `indices[position]` -- the gathered document id, NOT the
        position. It exists for exactly one implementor: `PointHist8`
        accumulates in FIXED POINT because its design calls `atomicAdd` on
        threadgroup memory and Metal has none ("Unsupported local float
        atomic operation for given target", probed 2026-08-21), and the
        dither that makes fixed-point quantization unbiased has to be keyed
        on something stable per row.

        IT MUST BE THE ROW AND NOT THE POSITION. A document's position in
        the index array is reordered at every level; its id is not. Key the
        dither on the position and the same row quantizes differently at
        different depths, which breaks `parent == child + sibling` -- and
        that identity is what lets the partial pass compute one child and
        subtract for the other.

        The 5-, 6- and 7-bit accumulators ignore it entirely: they add
        floats into private or turn-taken slots and need no atomic.
        =============================================================
        """
        ...

    def add_point_2(
        mut self,
        ci: SIMD[DType.uint32, 2],
        t: SIMD[DType.float32, 2],
        w: SIMD[DType.float32, 2],
        rows: SIMD[DType.uint32, 2],
    ):
        """`AddPoint2(bin, localTarget, localWeight)`."""
        ...

    def add_point_4(
        mut self,
        ci: SIMD[DType.uint32, 4],
        t: SIMD[DType.float32, 4],
        w: SIMD[DType.float32, 4],
        rows: SIMD[DType.uint32, 4],
    ):
        """`AddPoint4(bin, localTarget, localWeight)`."""
        ...

    def reduce(mut self):
        """`Reduce()`."""
        ...


@always_inline
def _column_of_thread[hist_block_count: Int](tid: Int) -> Int:
    """Their `(threadIdx.x & 31) + (threadIdx.x / 32 / HIST_BLOCK_COUNT) * 32`.

    A permutation of the thread id, not a lane trick. At
    `hist_block_count == 1` it IS the identity; above 1 it folds several
    thread groups onto the same column set so that each group accumulates
    into its own private copy of the histogram.
    """
    return (tid & (ALIGN_LANES - 1)) + (
        tid // ALIGN_LANES // hist_block_count
    ) * ALIGN_LANES


@always_inline
def _peel[
    THist: PointHist2, //, hist_block_count: Int
](
    mut hist: THist,
    span: Int,
    valid: Int,
    at: Int,
    indices: MutPointer[UInt32, MutAnyOrigin],
    target: MutPointer[Float32, MutAnyOrigin],
    weight: MutPointer[Float32, MutAnyOrigin],
    cindex: MutPointer[UInt32, MutAnyOrigin],
    col_step: Int,
):
    """Their striding peel loop (`:150-157`, `:176-184`, `:262-269`,
    `:288-296`), CONVERGED for the whole block.

    ============== DEVIATION BLOCK (PORTING.md 11 and 92) ==============
    Theirs is `for (; colId < span; colId += blockDim.x / HIST_BLOCK_COUNT)`,
    and at any block wider than `span` the threads with `colId >= span`
    never enter it at all. Under an 8-lane tile sync that is harmless: the
    lanes that skip are whole tiles, and a tile only ever waits on itself.
    Under a threadgroup barrier it is not, because `AddPoint` takes eight of
    them.

    THIS WAS NOT A THEORETICAL CONCERN AND THE UNIFORM BODY DID NOT COVER
    IT. When `compute_histogram`'s body was converged, these peel loops were
    left as theirs -- they looked like peels, not like body loops. At
    `BLOCK_SIZE = 256` and `span = 128`, exactly half the block enters, and
    `checks/pointwise_hist2_5bit_check.mojo` came back with 115 wrong
    cells for the `uint2` form and 73 for `uint4` while the scalar forms
    were exact -- points silently missing, totals low.

    The loop's own check could not see it: `TallyHist` gives every thread a
    PRIVATE slot, so it measures coverage and coverage was never wrong. It
    took a real accumulator, where eight threads share an inner copy and the
    barrier is what holds their writes apart, to make it visible. That is
    the whole argument for gating a kernel against a real accumulator rather
    than a convenient one.

    So every thread runs `ceil(span / col_step)` iterations and contributes
    a zero point when its column is outside `span` or outside `valid`.
    Adding 0.0 changes no sum.
    ====================================================================
    """
    var col = _column_of_thread[hist_block_count](Int(thread_idx.x))
    var n_peel = (span + col_step - 1) // col_step
    for _ in range(n_peel):
        var ci = UInt32(0)
        var w = Float32(0.0)
        var wt = Float32(0.0)
        var row = UInt32(0)
        if col < span and col < valid:
            row = ldg(indices.unsafe_offset(at + col))
            ci = ldg(cindex.unsafe_offset(Int(row)))
            w = ldg(weight.unsafe_offset(at + col))
            wt = ldg(target.unsafe_offset(at + col))
        hist.add_point(ci, wt, w, row)
        col += col_step


@always_inline
def compute_histogram[
    THist: PointHist2, //,
    stripe_size: Int,
    outer_unroll: Int,
    n: Int,
    hist_block_count: Int,
    blocks_per_feature: Int,
](
    mut hist: THist,
    indices: MutPointer[UInt32, MutAnyOrigin],
    offset_in: UInt32,
    ds_size_in: UInt32,
    target: MutPointer[Float32, MutAnyOrigin],
    weight: MutPointer[Float32, MutAnyOrigin],
    cindex: MutPointer[UInt32, MutAnyOrigin],
):
    """`ComputeHistogram` (`:12-131`), copied.

    `N` points per thread per iteration, gathered one at a time. Their
    pointer arithmetic is kept as explicit CURSORS here because Mojo has no
    `p += k` on a borrowed pointer argument; the cursors carry the same
    values their pointers would, step for step.
    """
    var base = Int(offset_in)
    var ds_size = Int(ds_size_in)
    if ds_size == 0:
        return

    var i = _column_of_thread[hist_block_count](Int(thread_idx.x))
    var bpf_id = Int(block_idx.x) % blocks_per_feature

    # --- their head peel (`:31-45`) ------------------------------------
    var last_id = ALIGN_LANES - (base & (ALIGN_LANES - 1))
    if ds_size < last_id:
        last_id = ds_size

    if bpf_id == 0:
        var index = UInt32(0)
        var ci = UInt32(0)
        var w = Float32(0.0)
        var wt = Float32(0.0)
        if i < last_id:
            index = ldg(indices.unsafe_offset(base + i))
            ci = ldg(cindex.unsafe_offset(Int(index)))
            w = ldg(weight.unsafe_offset(base + i))
            wt = ldg(target.unsafe_offset(base + i))
        # UNCONDITIONAL, exactly as theirs is: a thread outside the head
        # still calls AddPoint, with a zero bin and zero stats. Guarding it
        # would drop a barrier that the accumulators take inside AddPoint.
        hist.add_point(ci, wt, w, index)

    ds_size = ds_size - last_id if ds_size > last_id else 0
    base += last_id

    # --- their tail peel (`:48-60`) ------------------------------------
    var unaligned_tail = ds_size & (ALIGN_LANES - 1)
    if unaligned_tail != 0:
        if bpf_id == 0:
            var tail_offset = ds_size - unaligned_tail
            var index = UInt32(0)
            var ci = UInt32(0)
            var w = Float32(0.0)
            var wt = Float32(0.0)
            if i < unaligned_tail:
                index = ldg(indices.unsafe_offset(base + tail_offset + i))
                ci = ldg(cindex.unsafe_offset(Int(index)))
                w = ldg(weight.unsafe_offset(base + tail_offset + i))
                wt = ldg(target.unsafe_offset(base + tail_offset + i))
            hist.add_point(ci, wt, w, index)
    ds_size -= unaligned_tail

    if bpf_id == 0 and ds_size <= 0:
        barrier()
        hist.reduce()
        return

    # --- their strided body (`:68-129`) --------------------------------
    base += bpf_id * stripe_size
    var consumed = bpf_id * stripe_size
    ds_size = ds_size - consumed if ds_size > consumed else 0
    comptime stripe = stripe_size * blocks_per_feature

    if ds_size != 0:
        # ============== DEVIATION BLOCK (PORTING.md 11 and 92) ==========
        # THEIRS IS PER-THREAD, OURS IS PER-BLOCK, and this is forced:
        #
        #     iteration_count        = (dsSize - i + stripe - 1) / stripe
        #     blocked_iteration_count = (dsSize - (i|31) + stripe - 1)
        #                               / stripe / N
        #
        # Both depend on `i`, so threads run different counts, and their
        # `AddPoint` syncs an 8-lane tile that does not care. Ours takes a
        # threadgroup barrier, which every thread must reach the same
        # number of times.
        #
        # `max_iters` is `iteration_count` evaluated at `i == 0`, which is
        # its maximum over the block, so no thread loses an iteration. A
        # thread past its OWN count contributes `(bin 0, 0.0, 0.0)`, and
        # adding 0.0 changes no sum.
        #
        # THE COST IS A BOUNDS TEST PER POINT, which their unrolled loop
        # does not need precisely because their count is per-thread. That
        # is the trade: their divergence control (`i | 31`) becomes our
        # convergence requirement.
        # ================================================================
        comptime assert UNIFORM_ITERATION, (
            "compute_point_hist2_loop only has the uniform-iteration path"
            " written. A column whose sync_granularity_for is not"
            " SYNC_BLOCK could run CatBoost's per-thread counts directly,"
            " but that path does not exist here -- write it rather than"
            " letting this fall through (PORTING.md 11)."
        )
        var max_iters = (ds_size + (stripe - 1)) // stripe
        var own_iters = 0
        if ds_size > i:
            own_iters = (ds_size - i + (stripe - 1)) // stripe
        var blocked_iters = max_iters // n

        var cur = base + i
        var done = 0

        for _ in range(blocked_iters):
            var local_ci = SIMD[DType.uint32, n](0)
            var local_w = SIMD[DType.float32, n](0)
            var local_wt = SIMD[DType.float32, n](0)
            var local_row = SIMD[DType.uint32, n](0)

            comptime for k in range(n):
                if done + k < own_iters:
                    var idx = ldg(
                        indices.unsafe_offset(cur + stripe * k)
                    )
                    local_row[k] = idx
                    local_ci[k] = ldg(cindex.unsafe_offset(Int(idx)))
                    local_w[k] = ldg(
                        weight.unsafe_offset(cur + stripe * k)
                    )
                    local_wt[k] = ldg(
                        target.unsafe_offset(cur + stripe * k)
                    )

            cur += stripe * n
            done += n

            comptime for k in range(n):
                hist.add_point(
                    local_ci[k], local_wt[k], local_w[k], local_row[k]
                )

        for _ in range(blocked_iters * n, max_iters):
            var ci = UInt32(0)
            var w = Float32(0.0)
            var wt = Float32(0.0)
            var index = UInt32(0)
            if done < own_iters:
                index = ldg(indices.unsafe_offset(cur))
                ci = ldg(cindex.unsafe_offset(Int(index)))
                w = ldg(weight.unsafe_offset(cur))
                wt = ldg(target.unsafe_offset(cur))
            cur += stripe
            done += 1
            hist.add_point(ci, wt, w, index)

        barrier()
        hist.reduce()


@always_inline
def compute_histogram_2[
    THist: PointHist2, //,
    stripe_size: Int,
    outer_unroll: Int,
    hist_block_count: Int,
    blocks_per_feature: Int,
](
    mut hist: THist,
    indices: MutPointer[UInt32, MutAnyOrigin],
    offset_in: UInt32,
    ds_size_in: UInt32,
    target: MutPointer[Float32, MutAnyOrigin],
    weight: MutPointer[Float32, MutAnyOrigin],
    cindex: MutPointer[UInt32, MutAnyOrigin],
):
    """`ComputeHistogram2` (`:123-240`), copied.

    Two points per thread per iteration through `uint2`/`float2` loads.
    Note the peel quanta CHANGE from the scalar version and are not
    symmetric: the head aligns to 128 (`:147`) and the tail to 64 (`:171`),
    and both peel loops STRIDE (`for (; colId < 128; colId += blockDim.x /
    HIST_BLOCK_COUNT)`) rather than covering one column each. Transcribed as
    written; the asymmetry is theirs.
    """
    var base = Int(offset_in)
    var ds_size = Int(ds_size_in)
    if ds_size == 0:
        return

    var bpf_id = Int(block_idx.x) % blocks_per_feature
    var col_step = Int(block_dim.x) // hist_block_count

    # --- head, aligned to 128 (`:146-168`) -----------------------------
    var last_id = 128 - (base & 127)
    if ds_size < last_id:
        last_id = ds_size

    if bpf_id == 0:
        _peel[hist_block_count](hist, 128, last_id, base, indices, target,
                                weight, cindex, col_step)

    ds_size = ds_size - last_id if ds_size > last_id else 0
    base += last_id

    # --- tail, aligned to 64 (`:170-187`) ------------------------------
    var unaligned_tail = ds_size & 63
    if unaligned_tail != 0:
        if bpf_id == 0:
            var tail_offset = ds_size - unaligned_tail
            _peel[hist_block_count](hist, 64, unaligned_tail,
                                    base + tail_offset, indices, target,
                                    weight, cindex, col_step)
    ds_size -= unaligned_tail

    if ds_size <= 0:
        if bpf_id == 0:
            barrier()
            hist.reduce()
        return

    # --- strided body (`:199-236`) -------------------------------------
    var consumed = bpf_id * stripe_size * 2
    base += consumed
    ds_size = ds_size - consumed if ds_size > consumed else 0
    comptime stripe = stripe_size * blocks_per_feature * 2

    if ds_size != 0:
        # per-block iteration count; see the DEVIATION BLOCK in
        # `compute_histogram` -- same reason, same zero-point filler
        comptime assert UNIFORM_ITERATION, (
            "compute_histogram_2 only has the uniform-iteration path"
            " written (PORTING.md 11)"
        )
        var i = 2 * _column_of_thread[hist_block_count](Int(thread_idx.x))
        var max_iters = (ds_size + (stripe - 1)) // stripe
        var own_iters = 0
        if ds_size > i:
            own_iters = (ds_size - i + (stripe - 1)) // stripe
        var cur = base + i

        for j in range(max_iters):
            var bins = SIMD[DType.uint32, 2](0)
            var lt = SIMD[DType.float32, 2](0)
            var lw = SIMD[DType.float32, 2](0)
            var li = SIMD[DType.uint32, 2](0)
            if j < own_iters:
                li = SIMD[DType.uint32, 2](
                    ldg(indices.unsafe_offset(cur)),
                    ldg(indices.unsafe_offset(cur + 1)),
                )
                bins = SIMD[DType.uint32, 2](
                    ldg(cindex.unsafe_offset(Int(li[0]))),
                    ldg(cindex.unsafe_offset(Int(li[1]))),
                )
                lt = SIMD[DType.float32, 2](
                    ldg(target.unsafe_offset(cur)),
                    ldg(target.unsafe_offset(cur + 1)),
                )
                lw = SIMD[DType.float32, 2](
                    ldg(weight.unsafe_offset(cur)),
                    ldg(weight.unsafe_offset(cur + 1)),
                )
            cur += stripe
            hist.add_point_2(bins, lt, lw, li)

        barrier()
        hist.reduce()


@always_inline
def compute_histogram_4[
    THist: PointHist2, //,
    stripe_size: Int,
    outer_unroll: Int,
    hist_block_count: Int,
    blocks_per_feature: Int,
](
    mut hist: THist,
    indices: MutPointer[UInt32, MutAnyOrigin],
    offset_in: UInt32,
    ds_size_in: UInt32,
    target: MutPointer[Float32, MutAnyOrigin],
    weight: MutPointer[Float32, MutAnyOrigin],
    cindex: MutPointer[UInt32, MutAnyOrigin],
):
    """`ComputeHistogram4` (`:245-352`), copied.

    Four points per thread per iteration. Head AND tail both align to 128
    here (`:257`, `:282`), which is the one place the three entry points
    agree with each other rather than each choosing its own quanta.

    Their extra `__syncthreads()` before the body (`:320`) has no
    counterpart in the other two and is transcribed rather than tidied.
    """
    var base = Int(offset_in)
    var ds_size = Int(ds_size_in)
    if ds_size == 0:
        return

    var bpf_id = Int(block_idx.x) % blocks_per_feature
    var col_step = Int(block_dim.x) // hist_block_count

    # --- head, 128 (`:256-280`) ----------------------------------------
    var last_id = 128 - (base & 127)
    if ds_size < last_id:
        last_id = ds_size

    if bpf_id == 0:
        _peel[hist_block_count](hist, 128, last_id, base, indices, target,
                                weight, cindex, col_step)

    ds_size = ds_size - last_id if ds_size > last_id else 0
    base += last_id

    # --- tail, 128 (`:282-300`) ----------------------------------------
    var unaligned_tail = ds_size & 127
    if unaligned_tail != 0:
        if bpf_id == 0:
            var tail_offset = ds_size - unaligned_tail
            _peel[hist_block_count](hist, 128, unaligned_tail,
                                    base + tail_offset, indices, target,
                                    weight, cindex, col_step)
    ds_size -= unaligned_tail

    if ds_size <= 0:
        if bpf_id == 0:
            barrier()
            hist.reduce()
        return

    # --- strided body (`:311-348`) -------------------------------------
    var consumed = bpf_id * stripe_size * 4
    base += consumed
    ds_size = ds_size - consumed if ds_size > consumed else 0
    comptime stripe = stripe_size * blocks_per_feature * 4

    barrier()  # theirs, `:320`, and only in this variant

    if ds_size != 0:
        # per-block iteration count; see the DEVIATION BLOCK in
        # `compute_histogram`
        comptime assert UNIFORM_ITERATION, (
            "compute_histogram_4 only has the uniform-iteration path"
            " written (PORTING.md 11)"
        )
        var i = 4 * _column_of_thread[hist_block_count](Int(thread_idx.x))
        var max_iters = (ds_size + (stripe - 1)) // stripe
        var own_iters = 0
        if ds_size > i:
            own_iters = (ds_size - i + (stripe - 1)) // stripe
        var cur = base + i

        for j in range(max_iters):
            var bins = SIMD[DType.uint32, 4](0)
            var lt = SIMD[DType.float32, 4](0)
            var lw = SIMD[DType.float32, 4](0)
            var li = SIMD[DType.uint32, 4](0)
            if j < own_iters:
                li = SIMD[DType.uint32, 4](
                    ldg(indices.unsafe_offset(cur)),
                    ldg(indices.unsafe_offset(cur + 1)),
                    ldg(indices.unsafe_offset(cur + 2)),
                    ldg(indices.unsafe_offset(cur + 3)),
                )
                bins = SIMD[DType.uint32, 4](
                    ldg(cindex.unsafe_offset(Int(li[0]))),
                    ldg(cindex.unsafe_offset(Int(li[1]))),
                    ldg(cindex.unsafe_offset(Int(li[2]))),
                    ldg(cindex.unsafe_offset(Int(li[3]))),
                )
                lt = SIMD[DType.float32, 4](
                    ldg(target.unsafe_offset(cur)),
                    ldg(target.unsafe_offset(cur + 1)),
                    ldg(target.unsafe_offset(cur + 2)),
                    ldg(target.unsafe_offset(cur + 3)),
                )
                lw = SIMD[DType.float32, 4](
                    ldg(weight.unsafe_offset(cur)),
                    ldg(weight.unsafe_offset(cur + 1)),
                    ldg(weight.unsafe_offset(cur + 2)),
                    ldg(weight.unsafe_offset(cur + 3)),
                )
            cur += stripe
            hist.add_point_4(bins, lt, lw, li)

        barrier()
        hist.reduce()
