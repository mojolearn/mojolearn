# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# This lane MIRRORS CatBoost (pinned commit 54a8143a). Per-file provenance is in the root DERIVATION_MAP.tsv, in this file's own docstring, and in NOTICE; files with no CatBoost counterpart say so.
"""Add a grown tree's leaf values to the running predictions.

PORT OF the `AddModelValue` step of
`catboost/cuda/methods/doc_parallel_boosting.h:265` (`AppendModels`), whose
device side is CatBoost's add-model-value kernel family
(`catboost/cuda/models/kernel/add_model_value.cu`).

Their `AppendModels` runs once per boosting iteration and is what turns a
tree into a change in the cursor. `Rescale(step)` multiplies the leaf values
by the learning rate first (`doc_parallel_boosting.h:388`), which is why the
rate arrives here as a plain multiplier rather than being baked into the leaf
estimate.

================= DEVIATION BLOCK =================
CatBoost applies the model by EVALUATING it, on the learn set as much as
anywhere else. `TAddModelDocParallel<TObliviousTreeModel>::Append` calls
`AddObliviousTree` (`add_oblivious_tree_model_doc_parallel.cpp:191-192`),
which launches `AddObliviousTreeImpl` (`add_model_value.cu:70-120`): each row
reads its bin out of the compressed index at every level and builds the leaf
index bit by bit. There is NO partition-indexed path in their doc-parallel
weak learner. Their other kernel, `AddBinModelValueImpl` (`:14-53`), takes a
precomputed per-row bin array and belongs to the leaves estimator's `MoveTo`
(`pointwise_oracle.cpp:50-52`), not to `AppendModels`.

On the LEARN set during training the partition is already known: growth left
`row_index` permuted so that every row of leaf `L` sits in
`[offset[L], offset[L] + size[L])`. So the leaf assignment is read off the
partition instead of recomputed. It is exact rather than approximate and is
the same answer their evaluation produces, which `boosting_check` asserts by
comparing this cursor against `predict`'s rather than assuming it.

`add_bin_values.mojo` beside this file IS the port of `AddObliviousTreeImpl`,
so the evaluating form is not missing; what is missing is a test cursor to
point it at. Recorded so that gap is visible.
===================================================
"""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx

from original.numerics import identical_mul_add


def add_model_value_kernel(
    part_offset: MutPointer[UInt32, MutAnyOrigin],
    part_size: MutPointer[UInt32, MutAnyOrigin],
    row_index: MutPointer[UInt32, MutAnyOrigin],
    leaf_values: MutPointer[Float32, MutAnyOrigin],
    learning_rate: Float32,
    cursor: MutPointer[Float32, MutAnyOrigin],
    dim_count_in: Int32,
    cursor_stride_in: Int32,
):
    """Grid y is the LEAF, x strides that leaf's rows, **z is the APPROX
    DIMENSION**.

    THE TWO LAYOUTS ARE DIFFERENT AND THAT IS THEIRS. Their `shift` is
    BIN-MAJOR -- `newPoint[bin * cursorDim + dim]`
    (`pointwise_oracle.cpp:41-47`) -- while the cursor is PLANE-MAJOR,
    one contiguous column per class. So the value for (leaf, dim) is read
    at `leaf * dimCount + dim` and written into `dim * cursorStride + row`.
    Reading both as the same layout is the kind of mistake that trains a
    model with the classes rotated and no assertion fires.

    `dim_count_in == 1` with `cursor_stride_in == 0` is the single-
    dimensional path every pointwise loss takes, and it produces exactly
    the arithmetic this kernel had before the z axis existed: `block_idx.z`
    is 0, so the read is `leaf * 1 + 0` and the write offset is 0.

    `Rescale(step)` is folded in as `learning_rate` rather than mutating the
    stored leaf values, so the model keeps the UNSCALED estimate and the rate
    stays a property of the boosting loop. Their `Rescale` mutates the weak
    model in place; ours does not, because nothing here serialises a model
    yet and an unscaled leaf is the more useful thing to inspect.
    """
    var leaf = Int(block_idx.y)
    var dim = Int(block_idx.z)
    var dim_count = Int(dim_count_in)
    var plane = dim * Int(cursor_stride_in)
    var offset = Int(part_offset.unsafe_load(leaf))
    var size = Int(part_size.unsafe_load(leaf))
    var raw = leaf_values.unsafe_load(leaf * dim_count + dim)

    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while i < size:
        var row = plane + Int(row_index.unsafe_load(offset + i))
        # IDENTITY_PATHS row 9, the CURSOR UPDATE seam (E1 2026-08-22:
        # with tree 0 fully bit-identical Apple<->AMD under RMSE, the
        # first divergence moved to tree 1's FIRST histogram -- the only
        # arithmetic in between is this `cursor + leaf*rate`, which HIP's
        # default contraction may fuse per row while Metal's baseline is
        # measured unfused). `identical_mul_add` is an explicit fma under
        # IDENTICAL (one rounding, every vendor) and the naive chain
        # under FAST -- same operands, same ops, Apple FAST bits
        # unchanged.
        cursor.unsafe_store(
            row,
            identical_mul_add(
                raw, learning_rate, cursor.unsafe_load(row)
            ),
        )
        i += stride


comptime ABMV_BLOCK = 256
"""`const ui32 blockSize = 256` (`add_model_value.cu:60`)."""

comptime ABMV_ELEMENTS = 4
"""`const ui32 elementsPerThreads = 4` (`add_model_value.cu:61`)."""


def add_bin_model_value_kernel(
    bin_values: MutPointer[Float32, MutAnyOrigin],
    bins: MutPointer[UInt32, MutAnyOrigin],
    size_in: Int32,
    cursor_dim_in: Int32,
    cursor_align_in: Int32,
    cursor: MutPointer[Float32, MutAnyOrigin],
):
    """`AddBinModelValueImpl` (`add_model_value.cu:14-53`), the kernel the
    leaves estimator's `MoveTo` actually launches on their side
    (`pointwise_oracle.cpp:50-52`) -- FLAT over rows, `bins[i]` names row
    i's leaf, no per-leaf grid axis at all.

    ================= DEVIATION 210b (scheduling repair) ==============
    This port's `MoveTo` used to launch `add_model_value_kernel` -- the
    PARTITION-indexed AppendModels-style kernel -- on a grid of
    `ceil(max_leaf/256)` blocks x `bin_count` leaves: every leaf priced
    at the WIDEST leaf's block count, which on a skewed depth-8 higgs
    tree is ~100M+ empty threads per call. Measured 2026-08-22 on higgs
    2M rows: 13.62 ms/call against 1.0 ms for the ENTIRE Logloss
    evaluation kernel, at ~11 calls per tree = 47% of the whole fit.
    Their design has no such kernel in `MoveTo` at all; this is their
    kernel, restored. Grid: `CeilDivide(size, blockSize *
    elementsPerThreads)` (`:62`), one dimension.

    Their `readIndices` / `writeIndices` arms are NULL on this call path
    (the estimation cursor is bin-ordered and indices mean identity, the
    `readIndices ? ... : idx` branch), so the identity arms are compiled
    in and the pointers are not taken; the day a caller needs the gather
    arms it adds them as comptime flags rather than loading an identity
    array their null skips.
    ===================================================================

    The register-staging shape is theirs: each thread stages
    `ELEMENTS_PER_THREAD` (bins, then values, then adds) with the
    `i + j * BLOCK_SIZE` layout of their `#pragma unroll` loops.
    """
    var size = Int(size_in)
    var cursor_dim = Int(cursor_dim_in)
    var cursor_align = Int(cursor_align_in)
    var i = Int(block_idx.x) * ABMV_BLOCK * ABMV_ELEMENTS + Int(
        thread_idx.x
    )

    var bins_local = InlineArray[UInt32, ABMV_ELEMENTS](fill=UInt32(0))

    comptime for j in range(ABMV_ELEMENTS):
        var idx = i + j * ABMV_BLOCK
        if idx < size:
            bins_local[j] = bins.unsafe_load(idx)

    for dim in range(cursor_dim):
        var vals_local = InlineArray[Float32, ABMV_ELEMENTS](
            fill=Float32(0.0)
        )

        comptime for j in range(ABMV_ELEMENTS):
            var idx = i + j * ABMV_BLOCK
            if idx < size:
                vals_local[j] = bin_values.unsafe_load(
                    Int(bins_local[j]) * cursor_dim + dim
                )

        comptime for j in range(ABMV_ELEMENTS):
            var idx = i + j * ABMV_BLOCK
            if idx < size:
                var at = idx + dim * cursor_align
                cursor.unsafe_store(
                    at, cursor.unsafe_load(at) + vals_local[j]
                )


def fill_bins_from_partition_kernel(
    part_offset: MutPointer[UInt32, MutAnyOrigin],
    part_size: MutPointer[UInt32, MutAnyOrigin],
    bins: MutPointer[UInt32, MutAnyOrigin],
):
    """The per-row `bins` array `TBinOptimizedOracle` carries (their ctor
    receives it ready-made from the searcher; this port's searchers hand
    over a partition instead, so the bins are read off it ONCE per tree).
    Grid y is the leaf, x strides its rows -- the shape is tolerable here
    because it runs once per tree, not once per Newton evaluation, and
    the launcher machine-sizes x."""
    var leaf = Int(block_idx.y)
    var offset = Int(part_offset.unsafe_load(leaf))
    var size = Int(part_size.unsafe_load(leaf))
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while i < size:
        bins.unsafe_store(offset + i, UInt32(leaf))
        i += stride
