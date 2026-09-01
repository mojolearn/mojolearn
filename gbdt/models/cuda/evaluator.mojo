# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CatBoost's GPU model evaluator, ported: quantize once, then every tree
over cache-resident buckets.

PORT OF `catboost/libs/model/cuda/evaluator.cu` + `.cuh` at CatBoost
`54a8143a`. Transliterated. Do not improve.

WHY THIS EXISTS BESIDE `predict`. `doc_parallel_boosting.predict` walks the
TRAINING-side apply (`AddObliviousTreeImpl`), one kernel per tree over the
full compressed index: 100 trees re-stream ~19 MB of cindex each, 1.9 GB of
DRAM traffic for one model application, and it measured 2.9x BEHIND
CatBoost's CPU evaluator on 800k x 100 (bench/interleaved, 2026-08-20).
CatBoost's own inference path is THIS file: quantize the raw floats ONCE
into a doc-tiled bucket layout, then evaluate blocks of trees against
buckets that stay resident in cache -- the traffic no longer multiplies by
the tree count. That is their design for exactly the problem we measured,
so it is ported rather than reinvented (vendor rule).

THE BUCKET LAYOUT (`TCudaQuantizationBucket = uchar4`): docs are tiled in
groups of 128 = 32 lanes x 4 docs; one packed 4-byte word holds one
feature's bin for a lane's 4 docs:

    bucket[u32 units] = buckets_count * 32 * (doc // 128)
                      + bucket_idx * 32 + (doc % 32)
    byte j of it      = doc (doc//128)*128 + j*32 + (doc%32)

`TGPURepackedBin.FeatureIdx` arrives PRE-MULTIPLIED by 32
(`evaluator.cpp:32`, `cpuRepackedBin.FeatureIndex * WarpSize`), so the
kernel adds it to the tile base directly. `FeatureVal` is the CPU
repack's `SplitIdx` = training bin index + 1, because the split predicate
is `bucket >= SplitIdx` where training stored `bin > bin_idx`.

THE ONE-HOT ARM, AND WHOSE PREDICATE IT IS. `TGPURepackedBin` carries a
third member, `FeatureXorMask` (`evaluator.cuh:14`), and their GPU kernel
never reads it -- because their GPU evaluator REFUSES a categorical model
outright: `CB_ENSURE(!model.HasCategoricalFeatures(), "Model contains
categorical features, GPU evaluation impossible")` (`evaluator.cpp:22`).
The mask is not a gap in their kernel, it is a member their CPU evaluator
uses and their GPU one inherits unused.

So the predicate is TAKEN FROM THEIR CPU EVALUATOR rather than invented.
Their repack (`libs/model/model.cpp:566-572`):

    if (feature.Type != ESplitType::OneHotFeature) {
        rb.SplitIdx = featureIndex.SplitIdx;
    } else {
        rb.XorMask = ((~featureIndex.SplitIdx) & 0xff);
        rb.SplitIdx = 0xff;
    }

and their predicate (`libs/model/cpu/evaluator_impl.cpp:38`):

    indexesVec[docId] |= ((binFeaturePtr[docId] ^ xorMask) >= borderVal)
                         << depth;

`(b ^ (~v & 0xff)) >= 0xff` holds exactly when `b == v` for a byte, so
one branch-free form covers both predicates and a float split keeps the
identical arithmetic under `xorMask == 0`. Their CPU dispatch templates
the mask away when no split needs it (`NeedXorMask`,
`evaluator_impl.cpp:16`, `:257`); this file mirrors that with a comptime
kernel parameter picked from the model, so a float-only model runs the
byte-for-byte kernel it ran before this arm existed. Recorded as a
deviation in `PORTING.md`, because their GPU evaluator declines the case
and ours takes it.

DEVIATIONS, all stated:
* by-value structs -> parallel scalar arrays (as every kernel in this
  port): repacked bins arrive as `bin_feature_idx` (u32, pre-scaled) +
  `bin_feature_val` (u32) + `bin_feature_xor` (u32) arrays; the uchar4 is
  a `UInt32` unpacked by byte shifts.
* their results pipeline converts to FLOAT64 with model scale/bias
  (`ProcessResultsImpl`); Metal has no float64, so `ProcessResults` is
  not carried as a post-pass. Scale is always 1 here; BIAS IS NOT since
  boost_from_average landed (2026-08-22), and it enters as the
  accumulator's SEED -- `launch_eval` clears `results` to
  `Float32(model.bias)` instead of zero. That is bias-then-add in
  float32 where theirs is add-then-bias in float64: at most one float32
  rounding of association apart, inside the reorder tolerance the
  tree-block atomicAdd already imposes, and `model_io_check` gates the
  evaluator against the tree-wise apply on a BIASED model since the same
  date. The in-kernel accumulator is `float` on their side too
  (`TCudaEvaluatorLeafType = float`, evaluator.cuh:26).
* the one atomic is their `TAtomicAdd<float>` on global memory; Metal has
  global float atomicAdd (measured), so it ports directly. Same
  non-determinism class as their own GPU evaluator.

RESULTS PADDING IS A CONTRACT: the reduce writes every doc slot of a
128-doc block, in-range or not (theirs pads `ResultsFloatBuf` with
`AlignBy<2048>`), so `results` must hold `ceil(docs/128)*128` floats and
the caller reads only the first `docs`.
"""

from std.atomic import Atomic

from checks.kernel_matrix import (
    QUANTIZE_SEARCH_TWO_LEVEL,
    TARGET_COLUMN,
    quantize_search_for,
)
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.gpu.intrinsics import ldg
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

#: `evaluator.cu:47-53`, their constants verbatim.
comptime EVAL_OBJECTS_PER_THREAD = 4
comptime EVAL_TREE_SUB_BLOCK_WIDTH = 8
comptime EVAL_EXT_TREE_BLOCK_WIDTH = 128
comptime EVAL_QUANT_DOC_BLOCK = 256
comptime EVAL_BLOCK_WIDTH = 256
comptime EVAL_DOC_BLOCK_SIZE = EVAL_BLOCK_WIDTH // EVAL_TREE_SUB_BLOCK_WIDTH
comptime EVAL_WARP = 32

comptime NEG_INFTY = Float32(-3.4028234663852886e38)


def gpu_binarize_kernel[search_mode: Int = quantize_search_for[TARGET_COLUMN]()](
    values: MutPointer[Float32, MutAnyOrigin],
    value_stride_in: Int32,
    feature_count_in: Int32,
    object_count_in: Int32,
    borders: MutPointer[Float32, MutAnyOrigin],
    feature_border_offsets: MutPointer[UInt32, MutAnyOrigin],
    feature_border_counts: MutPointer[UInt32, MutAnyOrigin],
    float_feature_for_bucket: MutPointer[UInt32, MutAnyOrigin],
    buckets_count_in: Int32,
    target: MutPointer[UInt32, MutAnyOrigin],
):
    """`Binarize<ColumnFirst>` (`evaluator.cu:84-126`), copied.

    grid = (ceil(docs / 1024), buckets); block = 256. Each thread bins 4
    docs of one bucket's feature against the bucket's borders staged in
    shared memory; out-of-range docs read `-inf` (their accessor's
    `NegativeInfty()`), which no border exceeds, and the lane-guarded
    store drops them.
    """
    var tid = Int(thread_idx.x)
    var object_count = Int(object_count_in)
    var buckets_count = Int(buckets_count_in)
    var stride = Int(value_stride_in)

    var blockby32 = Int(block_idx.x) * (
        EVAL_QUANT_DOC_BLOCK // EVAL_WARP
    ) + tid // EVAL_WARP
    var first_doc = blockby32 * EVAL_WARP * EVAL_OBJECTS_PER_THREAD + (
        tid % EVAL_WARP
    )

    var bucket = Int(block_idx.y)
    var border_base = Int(feature_border_offsets.unsafe_load(bucket))
    var border_count = Int(feature_border_counts.unsafe_load(bucket))
    var feature = Int(float_feature_for_bucket.unsafe_load(bucket))

    var borders_local = stack_allocation[
        EVAL_QUANT_DOC_BLOCK,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    if tid < border_count:
        borders_local[tid] = ldg(borders + (border_base + tid))
    barrier()

    var f = InlineArray[Float32, EVAL_OBJECTS_PER_THREAD](fill=NEG_INFTY)

    @parameter
    for j in range(EVAL_OBJECTS_PER_THREAD):
        var doc = first_doc + j * EVAL_WARP
        if doc < object_count:
            f[j] = ldg(values + (feature * stride + doc))

    var bins = InlineArray[UInt32, EVAL_OBJECTS_PER_THREAD](fill=0)

    @parameter
    if search_mode == QUANTIZE_SEARCH_TWO_LEVEL:
        # The Apple arm of `quantize_search_for`: a PIVOT pass over every
        # 8th border, then one 8-wide segment pass, both branchless
        # independent compare-adds -- the linear scan's pipelined shape at
        # ~a fifth of its compares. (A lower_bound binary search was tried
        # first and measured 3.5x WORSE than the full scan: its eight
        # dependent divergent loads stall where compare-adds pipeline.
        # The matrix row carries both prices.)
        var seg = InlineArray[UInt32, EVAL_OBJECTS_PER_THREAD](fill=0)
        var p = 7
        while p < border_count:
            var pivot = borders_local[p]

            @parameter
            for j in range(EVAL_OBJECTS_PER_THREAD):
                seg[j] += UInt32(Int(f[j] > pivot))
            p += 8

        @parameter
        for j in range(EVAL_OBJECTS_PER_THREAD):
            var base = Int(seg[j]) * 8
            var count = UInt32(base)

            @parameter
            for u in range(8):
                if base + u < border_count:
                    count += UInt32(
                        Int(f[j] > borders_local[base + u])
                    )
            # `seg` cleared pivots at indices 8s-1, so ALL borders below
            # index 8s are below the value (sorted) and the segment pass
            # covers exactly [8s, 8s+8); the next pivot bounds the value
            # from above, so nothing past the segment can be cleared
            bins[j] = count
    else:
        # `#pragma unroll 8` + `bins.x += features.x > border`
        # (`evaluator.cu:117-121`): branchless, borders in unrolled chunks
        # of 8. A branchy scalar first translation of this loop measured
        # 40-46 ms; the shape is load-bearing, not style.
        var border_id = 0
        while border_id + 8 <= border_count:

            @parameter
            for u in range(8):
                var border = borders_local[border_id + u]

                @parameter
                for j in range(EVAL_OBJECTS_PER_THREAD):
                    bins[j] += UInt32(Int(f[j] > border))
            border_id += 8
        while border_id < border_count:
            var border = borders_local[border_id]

            @parameter
            for j in range(EVAL_OBJECTS_PER_THREAD):
                bins[j] += UInt32(Int(f[j] > border))
            border_id += 1

    if first_doc < object_count:
        var packed = (
            bins[0]
            | (bins[1] << 8)
            | (bins[2] << 16)
            | (bins[3] << 24)
        )
        target.unsafe_store(
            buckets_count * EVAL_WARP * blockby32
            + bucket * EVAL_WARP
            + (tid % EVAL_WARP),
            packed,
        )


def _calc_tree_index[
    unroll_depth: Int, need_xor_mask: Bool
](
    depth: Int,
    bin_feature_idx: MutPointer[UInt32, MutAnyOrigin],
    bin_feature_val: MutPointer[UInt32, MutAnyOrigin],
    bin_feature_xor: MutPointer[UInt32, MutAnyOrigin],
    split_base: Int,
    quantized: MutPointer[UInt32, MutAnyOrigin],
    quant_base: Int,
    mut idx: InlineArray[UInt32, EVAL_OBJECTS_PER_THREAD],
):
    """`CalcIndexesUnwrapped<TreeDepth>` / `CalcIndexesBase`
    (`evaluator.cu:129-156`): the leaf index of 4 docs at once, one packed
    bucket load per level. `unroll_depth > 0` is their comptime-unrolled
    6/7/8 arm; 0 is the generic loop. Their `+=` instead of `|=` is an
    A100 compiler workaround (MLTOOLS-6839) and either is exact here;
    `+=` is kept verbatim.

    `need_xor_mask` is their CPU evaluator's template parameter
    (`cpu/evaluator_impl.cpp:16`, `:257`): under `False` this is their GPU
    kernel unchanged, under `True` every compare becomes
    `(bucket ^ xorMask) >= FeatureVal` (`:38`), which is the one-hot
    predicate. See the file docstring for whose predicate it is and why
    their own GPU arm does not carry it.
    """

    @parameter
    for k in range(EVAL_OBJECTS_PER_THREAD):
        idx[k] = 0

    @parameter
    if unroll_depth > 0:

        @parameter
        for d in range(unroll_depth):
            var fi = Int(ldg(bin_feature_idx + (split_base + d)))
            var fv = ldg(bin_feature_val + (split_base + d))
            var buckets = ldg(quantized + (quant_base + fi))
            var xm = UInt32(0)

            @parameter
            if need_xor_mask:
                xm = ldg(bin_feature_xor + (split_base + d))

            @parameter
            for k in range(EVAL_OBJECTS_PER_THREAD):
                var b = ((buckets >> UInt32(8 * k)) & UInt32(255)) ^ xm
                if b >= fv:
                    idx[k] += UInt32(1 << d)
    else:
        for d in range(depth):
            var fi = Int(ldg(bin_feature_idx + (split_base + d)))
            var fv = ldg(bin_feature_val + (split_base + d))
            var buckets = ldg(quantized + (quant_base + fi))
            var xm = UInt32(0)

            @parameter
            if need_xor_mask:
                xm = ldg(bin_feature_xor + (split_base + d))

            @parameter
            for k in range(EVAL_OBJECTS_PER_THREAD):
                var b = ((buckets >> UInt32(8 * k)) & UInt32(255)) ^ xm
                if b >= fv:
                    idx[k] += UInt32(1 << d)


def eval_oblivious_trees_kernel[need_xor_mask: Bool = False](
    quantized: MutPointer[UInt32, MutAnyOrigin],
    tree_sizes: MutPointer[UInt32, MutAnyOrigin],
    tree_count_in: Int32,
    tree_start_offsets: MutPointer[UInt32, MutAnyOrigin],
    bin_feature_idx: MutPointer[UInt32, MutAnyOrigin],
    bin_feature_val: MutPointer[UInt32, MutAnyOrigin],
    bin_feature_xor: MutPointer[UInt32, MutAnyOrigin],
    first_leaf_offset: MutPointer[UInt32, MutAnyOrigin],
    ext_tree_block_width_in: Int32,
    buckets_count_in: Int32,
    leaf_values: MutPointer[Float32, MutAnyOrigin],
    document_count_in: Int32,
    results: MutPointer[Float32, MutAnyOrigin],
):
    """`EvalObliviousTrees` (`evaluator.cu:172-249`), copied.

    block = (32, 8): 32 doc-lanes x 8 tree sub-blocks; each thread owns 4
    docs and walks its sub-block's tree range two at a time. The shared
    reduce and the float atomicAdd into `results` are theirs line for
    line, including the `[tid]` + `[tid + 128]` pairing of the final add.

    ================= DEVIATION BLOCK =================
    Their `ExtTreeBlockWidth` is a compile-time 128: a tree sub-block owns
    `8 * 128 = 1024` trees, which is sized for their epsilon-class
    8000-tree models. At a 100-tree model that leaves SEVEN of the eight
    sub-blocks with `firstTreeIdx >= treeCount` -- 7/8 of every block's
    compute idle, measured 2.0-2.7x behind CatBoost CPU on 800k x 100
    (2026-08-20). The width is therefore a RUNTIME argument and the host
    picks `ceil(tree_count / 64)`, which fills all eight sub-blocks at
    any model size and reproduces their shape (W=125 vs their 128) at
    8000 trees. Scheduling only: the per-tree walk, the reduce and the
    atomic are unchanged, and the harness asserts this arm equals the
    tree-wise apply bit for bit.
    ===================================================
    """
    var tx = Int(thread_idx.x)
    var ty = Int(thread_idx.y)
    var tree_count = Int(tree_count_in)
    var buckets_count = Int(buckets_count_in)
    var document_count = Int(document_count_in)

    var inner_block_by32 = tx // EVAL_WARP
    var blockby32 = Int(block_idx.y) * (
        EVAL_DOC_BLOCK_SIZE // EVAL_WARP
    ) + inner_block_by32
    var in_block_id = tx % EVAL_WARP
    var first_doc = blockby32 * EVAL_WARP * EVAL_OBJECTS_PER_THREAD + in_block_id

    var quant_base = buckets_count * EVAL_WARP * blockby32 + tx % EVAL_WARP

    var ext_width = Int(ext_tree_block_width_in)
    var first_tree = EVAL_TREE_SUB_BLOCK_WIDTH * ext_width * (
        ty + EVAL_TREE_SUB_BLOCK_WIDTH * Int(block_idx.x)
    )
    var last_tree = first_tree + (
        EVAL_TREE_SUB_BLOCK_WIDTH * ext_width
    )
    if last_tree > tree_count:
        last_tree = tree_count

    var local_result = InlineArray[Float32, EVAL_OBJECTS_PER_THREAD](
        fill=Float32(0.0)
    )

    if first_tree < last_tree and first_doc < document_count:
        var split_base = Int(ldg(tree_start_offsets + first_tree))
        var leaf_base = Int(ldg(first_leaf_offset + first_tree))
        var idx = InlineArray[UInt32, EVAL_OBJECTS_PER_THREAD](fill=0)
        for tree in range(first_tree, last_tree):
            var depth = Int(ldg(tree_sizes + tree))
            if depth == 6:
                _calc_tree_index[6, need_xor_mask](
                    depth, bin_feature_idx, bin_feature_val, bin_feature_xor,
                    split_base, quantized, quant_base, idx,
                )
            elif depth == 7:
                _calc_tree_index[7, need_xor_mask](
                    depth, bin_feature_idx, bin_feature_val, bin_feature_xor,
                    split_base, quantized, quant_base, idx,
                )
            elif depth == 8:
                _calc_tree_index[8, need_xor_mask](
                    depth, bin_feature_idx, bin_feature_val, bin_feature_xor,
                    split_base, quantized, quant_base, idx,
                )
            else:
                _calc_tree_index[0, need_xor_mask](
                    depth, bin_feature_idx, bin_feature_val, bin_feature_xor,
                    split_base, quantized, quant_base, idx,
                )

            @parameter
            for k in range(EVAL_OBJECTS_PER_THREAD):
                local_result[k] += ldg(
                    leaf_values + (leaf_base + Int(idx[k]))
                )
            split_base += depth
            leaf_base += 1 << depth

    # `reduceVals[...]` (`evaluator.cu:228-247`), their exact slots.
    var reduce_vals = stack_allocation[
        EVAL_DOC_BLOCK_SIZE
        * EVAL_OBJECTS_PER_THREAD
        * EVAL_TREE_SUB_BLOCK_WIDTH,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()

    @parameter
    for k in range(EVAL_OBJECTS_PER_THREAD):
        reduce_vals[
            inner_block_by32 * EVAL_WARP * EVAL_OBJECTS_PER_THREAD
            + EVAL_WARP * k
            + in_block_id
            + ty * EVAL_DOC_BLOCK_SIZE * EVAL_OBJECTS_PER_THREAD
        ] = local_result[k]
    barrier()
    var flat = tx + ty * EVAL_DOC_BLOCK_SIZE
    var lr = reduce_vals[flat]

    @parameter
    for j in range(1, EVAL_OBJECTS_PER_THREAD):
        lr += reduce_vals[j * 256 + flat]
    reduce_vals[flat] = lr
    barrier()
    if ty < EVAL_OBJECTS_PER_THREAD:
        _ = Atomic.fetch_add(
            results.unsafe_offset(
                blockby32 * EVAL_WARP * EVAL_OBJECTS_PER_THREAD
                + tx
                + ty * EVAL_DOC_BLOCK_SIZE
            ),
            reduce_vals[flat] + reduce_vals[flat + 128],
        )


# ---------------------------------------------------------------------------
# HOST SIDE, their `TGPUCatboostEvaluationContext` (`evaluator.cpp`).
# The model arrays are built ONCE, at model-pack time, exactly as their
# `TGPUModelData` is built when the evaluator context is created -- never
# per predict call.
# ---------------------------------------------------------------------------

from max.gpu.host import DeviceBuffer, DeviceContext

from gbdt.methods.doc_parallel_boosting import model_approx_dim
from gbdt.models.oblivious_model import (
    BIN_SPLIT_TAKE_BIN,
    TAdditiveModel,
)


@fieldwise_init
struct GpuEvaluatorModel(Movable):
    """`TGPUModelData`, the members this port reaches: tree geometry,
    repacked splits as two parallel arrays, flat leaves."""

    var tree_sizes: DeviceBuffer[DType.uint32]
    var tree_starts: DeviceBuffer[DType.uint32]
    var first_leaf: DeviceBuffer[DType.uint32]
    var bf_idx: DeviceBuffer[DType.uint32]
    var bf_val: DeviceBuffer[DType.uint32]
    var bf_xor: DeviceBuffer[DType.uint32]
    var leaves: DeviceBuffer[DType.float32]
    var tree_count: Int
    var need_xor_mask: Bool
    """Their `NeedXorMask` dispatch (`cpu/evaluator_impl.cpp:257`,
    `:428-437`): true iff some split in this model is one-hot. A float-only
    model runs the kernel it ran before the one-hot arm existed."""
    var bias: Float32
    """`model.bias` narrowed once; the accumulator's seed (see the
    ProcessResults deviation in the module docstring)."""


def pack_model_for_evaluator(
    ctx: DeviceContext, model: TAdditiveModel
) raises -> GpuEvaluatorModel:
    """Their model repack (`evaluator.cpp:29-58`): `FeatureIdx` pre-scaled
    by the warp width for the bucket tile stride, `FeatureVal` =
    training bin index + 1 because `bucket >= val` must reproduce
    training's `bin > bin_idx`. Buckets are our layout features one to
    one (no unused-feature compaction yet: every kept feature has
    borders).

    ## ONE-DIMENSIONAL ONLY, AND THAT IS THEIRS RATHER THAN OURS

    Their GPU evaluator refuses a multi-output model outright:

        CB_ENSURE(ModelTrees->GetDimensionsCount() == 1,
                  "Model is not one-dimensional, GPU evaluation is not
                   supported yet");
                                    -- `libs/model/cuda/evaluator.cpp:28`

    and their kernel says the same thing structurally: `EvalObliviousTrees`
    advances `leafValues += (1 << curTreeDepth)` per tree
    (`evaluator.cu:222`), ONE value per leaf with no dimension stride, and
    accumulates into a scalar per document. `ApproxDimension` appears in
    that file only in `ProcessResults`, the prediction-type
    post-processing, never in the tree walk.

    So there is no multi-dimensional path of theirs to port here. This
    refuses where they refuse, with their message, RATHER THAN silently
    predicting the first class's approxes -- which is what a model whose
    `leaf_values` is `n_leaves * dim` would get from a walk that reads
    `leaf_values[leaf]`.

    MULTICLASS PREDICTION GOES THROUGH THE OTHER APPLY:
    `gbdt/train.mojo`'s `predict_multi_floats`, over
    `compute_bins_and_add_kernel`, which IS multi-dimensional
    (DEVIATION 81). `checks/multiclass_train_check.mojo` gates it.
    """
    var approx_dim = model_approx_dim(model)
    if approx_dim != 1:
        raise Error(
            "Model is not one-dimensional (dim " + String(approx_dim)
            + "), GPU evaluation is not supported yet -- their own"
            " `libs/model/cuda/evaluator.cpp:28` refuses the same case."
            " Use predict_multi_floats, which goes through the"
            " multi-dimensional apply."
        )
    # OBLIVIOUS ONLY, AND THAT TOO IS THEIRS: `CB_ENSURE(model.IsOblivious(),
    # "Model is not oblivious, GPU evaluation impossible")`
    # (`libs/model/cuda/evaluator.cpp:25`). A Depthwise/Lossguide model
    # (DEVIATION 259) predicts through `predict_floats`, whose
    # non-symmetric arm is their `TAddModelDocParallel<TNonSymmetricTree>`.
    if not model.is_oblivious():
        raise Error(
            "Model is not oblivious, GPU evaluation impossible -- their own"
            " `libs/model/cuda/evaluator.cpp:25` refuses the same case."
            " Use predict_floats, whose non-symmetric apply is"
            " add_non_symmetric_tree_doc_parallel.mojo."
        )

    var total_levels = 0
    var total_leaves = 0
    for t in range(model.size()):
        total_levels += model.weak_models[t].structure.get_depth()
        total_leaves += 1 << model.weak_models[t].structure.get_depth()
    if total_levels == 0:
        raise Error("empty model has nothing to pack")

    var n_trees = model.size()
    var tree_sizes = ctx.enqueue_create_buffer[DType.uint32](n_trees)
    var tree_starts = ctx.enqueue_create_buffer[DType.uint32](n_trees)
    var first_leaf = ctx.enqueue_create_buffer[DType.uint32](n_trees)
    var bf_idx = ctx.enqueue_create_buffer[DType.uint32](total_levels)
    var bf_val = ctx.enqueue_create_buffer[DType.uint32](total_levels)
    var bf_xor = ctx.enqueue_create_buffer[DType.uint32](total_levels)
    var leaves = ctx.enqueue_create_buffer[DType.float32](total_leaves)

    var h1 = ctx.enqueue_create_host_buffer[DType.uint32](n_trees)
    var h2 = ctx.enqueue_create_host_buffer[DType.uint32](n_trees)
    var h3 = ctx.enqueue_create_host_buffer[DType.uint32](n_trees)
    var h4 = ctx.enqueue_create_host_buffer[DType.uint32](total_levels)
    var h5 = ctx.enqueue_create_host_buffer[DType.uint32](total_levels)
    var h7 = ctx.enqueue_create_host_buffer[DType.uint32](total_levels)
    var h6 = ctx.enqueue_create_host_buffer[DType.float32](total_leaves)

    var lvl = 0
    var leaf = 0
    var need_xor_mask = False
    for t in range(n_trees):
        ref weak = model.weak_models[t]
        var depth = weak.structure.get_depth()
        h1.unsafe_ptr().unsafe_store(t, UInt32(depth))
        h2.unsafe_ptr().unsafe_store(t, UInt32(lvl))
        h3.unsafe_ptr().unsafe_store(t, UInt32(leaf))
        for level in range(depth):
            h4.unsafe_ptr().unsafe_store(
                lvl,
                UInt32(Int(weak.structure.splits[level].feature_id))
                * UInt32(EVAL_WARP),
            )
            var bin_idx = Int(weak.structure.splits[level].bin_idx)
            if (
                Int(weak.structure.splits[level].split_type)
                == BIN_SPLIT_TAKE_BIN
            ):
                # their `model.cpp:569-570` for an `ESplitType::OneHotFeature`
                if bin_idx < 0 or bin_idx > 254:
                    raise Error(
                        "a one-hot split on category " + String(bin_idx)
                        + " cannot be repacked: the bucket is one byte and"
                        " 255 is the predicate's own sentinel"
                    )
                h5.unsafe_ptr().unsafe_store(lvl, UInt32(0xFF))
                h7.unsafe_ptr().unsafe_store(
                    lvl, UInt32((~bin_idx) & 0xFF)
                )
                need_xor_mask = True
            else:
                h5.unsafe_ptr().unsafe_store(lvl, UInt32(bin_idx + 1))
                h7.unsafe_ptr().unsafe_store(lvl, UInt32(0))
            lvl += 1
        var n_leaves = 1 << depth
        for i in range(n_leaves):
            var v = Float32(0.0)
            if i < len(weak.leaf_values):
                v = weak.leaf_values[i]
            h6.unsafe_ptr().unsafe_store(leaf + i, v)
        leaf += n_leaves
    ctx.enqueue_copy(dst_buf=tree_sizes, src_ptr=h1.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=tree_starts, src_ptr=h2.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=first_leaf, src_ptr=h3.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=bf_idx, src_ptr=h4.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=bf_val, src_ptr=h5.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=bf_xor, src_ptr=h7.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=leaves, src_ptr=h6.unsafe_ptr())
    ctx.synchronize()
    # past the drain, every staging buffer the pack enqueued from: their
    # last uses were the seven enqueues above, which freed each of them
    # with copies still queued (the step-33 race class -- h1 died with six
    # copies not yet enqueued behind it)
    _ = h1^
    _ = h2^
    _ = h3^
    _ = h4^
    _ = h5^
    _ = h7^
    _ = h6^
    return GpuEvaluatorModel(
        tree_sizes^, tree_starts^, first_leaf^, bf_idx^, bf_val^, bf_xor^,
        leaves^, n_trees, need_xor_mask, Float32(model.bias),
    )


def quantized_buffer_u32s(buckets_count: Int, n_rows: Int) -> Int:
    """`TCudaQuantizedData::SetDimensions` (`evaluator.cu:59-66`): 32
    packed words per (128-doc block, bucket), 4 blocks of 32 docs each."""
    return (
        EVAL_WARP
        * buckets_count
        * ((n_rows + 127) // 128)
        * EVAL_OBJECTS_PER_THREAD
    )


def padded_results(n_rows: Int) -> Int:
    """The reduce writes whole 128-doc blocks (their `AlignBy<2048>` pad)."""
    return ((n_rows + 127) // 128) * 128


def launch_quantize(
    ctx: DeviceContext,
    mut xdev: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_feats: Int,
    mut borders: DeviceBuffer[DType.float32],
    mut border_offsets: DeviceBuffer[DType.uint32],
    mut border_counts: DeviceBuffer[DType.uint32],
    mut bucket_feature: DeviceBuffer[DType.uint32],
    mut quantized: DeviceBuffer[DType.uint32],
) raises:
    """`QuantizeData` (`evaluator.cu:379-401`), the ColumnFirst arm; our
    colmajor X has stride `n_rows`."""
    ctx.enqueue_function[
        gpu_binarize_kernel[quantize_search_for[TARGET_COLUMN]()]
    ](
        xdev.unsafe_ptr(), Int32(n_rows), Int32(n_feats), Int32(n_rows),
        borders.unsafe_ptr(), border_offsets.unsafe_ptr(),
        border_counts.unsafe_ptr(), bucket_feature.unsafe_ptr(),
        Int32(n_feats), quantized.unsafe_ptr(),
        grid_dim=(
            (n_rows + EVAL_QUANT_DOC_BLOCK * EVAL_OBJECTS_PER_THREAD - 1)
            // (EVAL_QUANT_DOC_BLOCK * EVAL_OBJECTS_PER_THREAD),
            n_feats,
            1,
        ),
        block_dim=(EVAL_QUANT_DOC_BLOCK, 1, 1),
    )


def launch_eval(
    ctx: DeviceContext,
    mut m: GpuEvaluatorModel,
    mut quantized: DeviceBuffer[DType.uint32],
    n_feats: Int,
    n_rows: Int,
    mut results: DeviceBuffer[DType.float32],
) raises:
    """`EvalQuantizedData` (`evaluator.cu:344-368`): clear results, one
    kernel over (tree blocks, doc blocks). The clear value is the model's
    bias -- the ProcessResults deviation in the module docstring."""
    ctx.enqueue_memset(results, m.bias)
    # the adaptive sub-block width of the kernel's DEVIATION BLOCK: fill
    # all eight tree sub-blocks at any model size
    var ext_width = (m.tree_count + 63) // 64
    if ext_width < 1:
        ext_width = 1
    var tree_grid = (
        m.tree_count
        + EVAL_TREE_SUB_BLOCK_WIDTH * ext_width
        - 1
    ) // (EVAL_TREE_SUB_BLOCK_WIDTH * ext_width)
    var doc_grid = (
        n_rows + EVAL_DOC_BLOCK_SIZE * EVAL_OBJECTS_PER_THREAD - 1
    ) // (EVAL_DOC_BLOCK_SIZE * EVAL_OBJECTS_PER_THREAD)
    # their `NeedXorMask` dispatch: one kernel or the other, never a
    # runtime test inside the level loop
    if m.need_xor_mask:
        ctx.enqueue_function[eval_oblivious_trees_kernel[True]](
            quantized.unsafe_ptr(), m.tree_sizes.unsafe_ptr(),
            Int32(m.tree_count), m.tree_starts.unsafe_ptr(),
            m.bf_idx.unsafe_ptr(), m.bf_val.unsafe_ptr(),
            m.bf_xor.unsafe_ptr(),
            m.first_leaf.unsafe_ptr(), Int32(ext_width), Int32(n_feats),
            m.leaves.unsafe_ptr(), Int32(n_rows), results.unsafe_ptr(),
            grid_dim=(tree_grid, doc_grid, 1),
            block_dim=(EVAL_DOC_BLOCK_SIZE, EVAL_TREE_SUB_BLOCK_WIDTH, 1),
        )
    else:
        ctx.enqueue_function[eval_oblivious_trees_kernel[False]](
            quantized.unsafe_ptr(), m.tree_sizes.unsafe_ptr(),
            Int32(m.tree_count), m.tree_starts.unsafe_ptr(),
            m.bf_idx.unsafe_ptr(), m.bf_val.unsafe_ptr(),
            m.bf_xor.unsafe_ptr(),
            m.first_leaf.unsafe_ptr(), Int32(ext_width), Int32(n_feats),
            m.leaves.unsafe_ptr(), Int32(n_rows), results.unsafe_ptr(),
            grid_dim=(tree_grid, doc_grid, 1),
            block_dim=(EVAL_DOC_BLOCK_SIZE, EVAL_TREE_SUB_BLOCK_WIDTH, 1),
        )
