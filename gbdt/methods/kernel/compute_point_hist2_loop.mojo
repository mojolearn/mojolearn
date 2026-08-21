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

That was false, and `mojo_only/shared_pointer_probe.mojo` measures it: the
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

    def add_point(mut self, ci: UInt32, t: Float32, w: Float32):
        """`AddPoint(ci, wt, w)`."""
        ...

    def add_point_2(
        mut self,
        ci: SIMD[DType.uint32, 2],
        t: SIMD[DType.float32, 2],
        w: SIMD[DType.float32, 2],
    ):
        """`AddPoint2(bin, localTarget, localWeight)`."""
        ...

    def add_point_4(
        mut self,
        ci: SIMD[DType.uint32, 4],
        t: SIMD[DType.float32, 4],
        w: SIMD[DType.float32, 4],
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
        hist.add_point(ci, wt, w)

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
            hist.add_point(ci, wt, w)
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
        var iteration_count = 0
        if ds_size > i:
            iteration_count = (ds_size - i + (stripe - 1)) // stripe
        # `(i | 31)` is theirs (`:80`) and it is DIVERGENCE CONTROL, not a
        # bounds guard. It rounds `i` up to the largest column in its
        # 32-group, so every lane in the group agrees on how many unrolled
        # iterations to run and the unrolled loop never diverges; the
        # remainder loop below picks up whatever each lane still owes.
        #
        # It cannot change the RESULT and that was measured, not assumed.
        # Replacing it with a bare `i` leaves all 160 cases of
        # `mojo_only/pointwise_loop_check.mojo` bit-identical across all
        # three `n`, and the arithmetic says why: `blocked * n` is
        # `(iterCount // n) * n <= iterCount` for either spelling, so the
        # two loops always deliver `iterCount` points in the same per-lane
        # order. The check records this as a mechanism it deliberately does
        # NOT gate, because there is nothing there to catch.
        var blocked_iteration_count = 0
        if ds_size > (i | (ALIGN_LANES - 1)):
            blocked_iteration_count = (
                (ds_size - (i | (ALIGN_LANES - 1)) + (stripe - 1)) // stripe
            ) // n

        var cur = base + i

        for _ in range(blocked_iteration_count):
            var local_index = SIMD[DType.uint32, n](0)

            comptime for k in range(n):
                local_index[k] = ldg(indices.unsafe_offset(cur + stripe * k))

            var local_ci = SIMD[DType.uint32, n](0)
            var local_w = SIMD[DType.float32, n](0)
            var local_wt = SIMD[DType.float32, n](0)

            comptime for k in range(n):
                local_ci[k] = ldg(
                    cindex.unsafe_offset(Int(local_index[k]))
                )
                local_w[k] = ldg(weight.unsafe_offset(cur + stripe * k))
                local_wt[k] = ldg(target.unsafe_offset(cur + stripe * k))

            cur += stripe * n

            comptime for k in range(n):
                hist.add_point(local_ci[k], local_wt[k], local_w[k])

        for _ in range(blocked_iteration_count * n, iteration_count):
            var index = ldg(indices.unsafe_offset(cur))
            var ci = ldg(cindex.unsafe_offset(Int(index)))
            var w = ldg(weight.unsafe_offset(cur))
            var wt = ldg(target.unsafe_offset(cur))
            cur += stripe
            hist.add_point(ci, wt, w)

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
        var col = _column_of_thread[hist_block_count](Int(thread_idx.x))
        while col < 128:
            var index = UInt32(0)
            var ci = UInt32(0)
            var w = Float32(0.0)
            var wt = Float32(0.0)
            if col < last_id:
                index = ldg(indices.unsafe_offset(base + col))
                ci = ldg(cindex.unsafe_offset(Int(index)))
                w = ldg(weight.unsafe_offset(base + col))
                wt = ldg(target.unsafe_offset(base + col))
            hist.add_point(ci, wt, w)
            col += col_step

    ds_size = ds_size - last_id if ds_size > last_id else 0
    base += last_id

    # --- tail, aligned to 64 (`:170-187`) ------------------------------
    var unaligned_tail = ds_size & 63
    if unaligned_tail != 0:
        if bpf_id == 0:
            var col = _column_of_thread[hist_block_count](Int(thread_idx.x))
            var tail_offset = ds_size - unaligned_tail
            while col < 64:
                var index = UInt32(0)
                var ci = UInt32(0)
                var w = Float32(0.0)
                var wt = Float32(0.0)
                if col < unaligned_tail:
                    index = ldg(
                        indices.unsafe_offset(base + tail_offset + col)
                    )
                    ci = ldg(cindex.unsafe_offset(Int(index)))
                    w = ldg(weight.unsafe_offset(base + tail_offset + col))
                    wt = ldg(target.unsafe_offset(base + tail_offset + col))
                hist.add_point(ci, wt, w)
                col += col_step
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
        var i = 2 * _column_of_thread[hist_block_count](Int(thread_idx.x))
        var iter_count = 0
        if ds_size > i:
            iter_count = (ds_size - i + (stripe - 1)) // stripe
        var cur = base + i

        for _ in range(iter_count):
            var li = SIMD[DType.uint32, 2](
                ldg(indices.unsafe_offset(cur)),
                ldg(indices.unsafe_offset(cur + 1)),
            )
            var bins = SIMD[DType.uint32, 2](
                ldg(cindex.unsafe_offset(Int(li[0]))),
                ldg(cindex.unsafe_offset(Int(li[1]))),
            )
            var lt = SIMD[DType.float32, 2](
                ldg(target.unsafe_offset(cur)),
                ldg(target.unsafe_offset(cur + 1)),
            )
            var lw = SIMD[DType.float32, 2](
                ldg(weight.unsafe_offset(cur)),
                ldg(weight.unsafe_offset(cur + 1)),
            )
            cur += stripe
            hist.add_point_2(bins, lt, lw)

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
        var col = _column_of_thread[hist_block_count](Int(thread_idx.x))
        while col < 128:
            var index = UInt32(0)
            var ci = UInt32(0)
            var w = Float32(0.0)
            var wt = Float32(0.0)
            if col < last_id:
                index = ldg(indices.unsafe_offset(base + col))
                ci = ldg(cindex.unsafe_offset(Int(index)))
                w = ldg(weight.unsafe_offset(base + col))
                wt = ldg(target.unsafe_offset(base + col))
            hist.add_point(ci, wt, w)
            col += col_step

    ds_size = ds_size - last_id if ds_size > last_id else 0
    base += last_id

    # --- tail, 128 (`:282-300`) ----------------------------------------
    var unaligned_tail = ds_size & 127
    if unaligned_tail != 0:
        if bpf_id == 0:
            var col = _column_of_thread[hist_block_count](Int(thread_idx.x))
            var tail_offset = ds_size - unaligned_tail
            while col < 128:
                var index = UInt32(0)
                var ci = UInt32(0)
                var w = Float32(0.0)
                var wt = Float32(0.0)
                if col < unaligned_tail:
                    index = ldg(
                        indices.unsafe_offset(base + tail_offset + col)
                    )
                    ci = ldg(cindex.unsafe_offset(Int(index)))
                    w = ldg(weight.unsafe_offset(base + tail_offset + col))
                    wt = ldg(target.unsafe_offset(base + tail_offset + col))
                hist.add_point(ci, wt, w)
                col += col_step
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
        var i = 4 * _column_of_thread[hist_block_count](Int(thread_idx.x))
        var iter_count = 0
        if ds_size > i:
            iter_count = (ds_size - i + (stripe - 1)) // stripe
        var cur = base + i

        for _ in range(iter_count):
            var li = SIMD[DType.uint32, 4](
                ldg(indices.unsafe_offset(cur)),
                ldg(indices.unsafe_offset(cur + 1)),
                ldg(indices.unsafe_offset(cur + 2)),
                ldg(indices.unsafe_offset(cur + 3)),
            )
            var bins = SIMD[DType.uint32, 4](
                ldg(cindex.unsafe_offset(Int(li[0]))),
                ldg(cindex.unsafe_offset(Int(li[1]))),
                ldg(cindex.unsafe_offset(Int(li[2]))),
                ldg(cindex.unsafe_offset(Int(li[3]))),
            )
            var lt = SIMD[DType.float32, 4](
                ldg(target.unsafe_offset(cur)),
                ldg(target.unsafe_offset(cur + 1)),
                ldg(target.unsafe_offset(cur + 2)),
                ldg(target.unsafe_offset(cur + 3)),
            )
            var lw = SIMD[DType.float32, 4](
                ldg(weight.unsafe_offset(cur)),
                ldg(weight.unsafe_offset(cur + 1)),
                ldg(weight.unsafe_offset(cur + 2)),
                ldg(weight.unsafe_offset(cur + 3)),
            )
            cur += stripe
            hist.add_point_4(bins, lt, lw)

        barrier()
        hist.reduce()
