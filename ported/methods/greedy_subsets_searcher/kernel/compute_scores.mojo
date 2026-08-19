"""Score every candidate split of a level and pick the best.

PORT OF `catboost/cuda/methods/greedy_subsets_searcher/kernel/
compute_scores.cu` at CatBoost `54a8143a`. Transliterated. Do not improve.

**The parallel axis is the BIN-FEATURE, and the leaf loop is SERIAL inside
the thread.** That is the whole shape:

    for (offset = blockIdx.x * BlockSize; offset < binFeatureCount; ...)
        binFeatureId = offset + tid
        for (i = 0; i < pCount; i++)          <- leaves, serially
            leafId = partIds[i]
            ...

An oblivious level scores ONE split across all its leaves, so a candidate's
score is a sum over leaves. Putting the candidate on the parallel axis makes
that sum thread-local: no cross-thread reduction over leaves at all, and the
per-leaf reads are a strided walk one thread owns. Putting the LEAF on the
parallel axis instead would force a reduction across threads for every
candidate.

The reason this shape is available to them and not, historically, to us: the
bin prefix scan already ran in its own kernel (`histogram_utils.scan`), so a
thread reads a leaf's cumulative sum at one bin directly. A kernel that
re-derives the running prefix per candidate must walk bins in order, which
forces the bin loop innermost, which pushes the leaf loop outward. **The scan
is what buys the parallel shape**, not a micro-optimization on top of it.

Their right-side arithmetic, copied: the right child is never accumulated,
only derived as `partStat - sumLeft`. One histogram, both sides.
"""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.memory import stack_allocation


#: `compute_scores.cu:167`.
comptime SCORE_BLOCK_SIZE = 128


def compute_optimal_splits_kernel(
    bf_skip: MutPointer[UInt8, MutAnyOrigin],
    bin_feature_count_in: Int32,
    histograms: MutPointer[Float32, MutAnyOrigin],
    part_stats: MutPointer[Float32, MutAnyOrigin],
    stat_count_in: Int32,
    part_ids: MutPointer[UInt32, MutAnyOrigin],
    p_count_in: Int32,
    lambda_l2: Float32,
    out_score: MutPointer[Float32, MutAnyOrigin],
    out_bin: MutPointer[UInt32, MutAnyOrigin],
):
    """`ComputeOptimalSplits`, with the block argmax that follows it.

    One block reduces its candidates to a single best and writes one record;
    the host takes the minimum over at most 64 of those in a plain loop
    (`greedy_search_helper.cpp:513-532`). Copied: an argmax over 64 structs
    is not worth a second kernel and a second synchronization.
    """
    var bin_feature_count = Int(bin_feature_count_in)
    var stat_count = Int(stat_count_in)
    var p_count = Int(p_count_in)
    var tid = Int(thread_idx.x)

    var best_score = Float32(0.0)
    var best_bin = UInt32(0xFFFFFFFF)

    var offset = Int(block_idx.x) * SCORE_BLOCK_SIZE
    while offset < bin_feature_count:
        var bin_feature_id = offset + tid
        if bin_feature_id >= bin_feature_count:
            break
        if bf_skip.unsafe_load(bin_feature_id) != 0:
            offset += SCORE_BLOCK_SIZE * Int(grid_dim.x)
            continue

        var score = Float32(0.0)

        # THE SERIAL LEAF LOOP. See the module docstring for why it is inside
        # the thread and not across threads.
        for i in range(p_count):
            var leaf_id = Int(part_ids.unsafe_load(i))
            var leaf_base = leaf_id * stat_count * bin_feature_count

            # stat 0 is the weight plane.
            var weight_left = max(
                histograms.unsafe_load(leaf_base + bin_feature_id),
                Float32(0.0),
            )
            var weight_right = max(
                part_stats.unsafe_load(leaf_id * stat_count) - weight_left,
                Float32(0.0),
            )

            for stat_id in range(1, stat_count):
                var sum_left = histograms.unsafe_load(
                    leaf_base + stat_id * bin_feature_count + bin_feature_id
                )
                var part_stat = part_stats.unsafe_load(
                    leaf_id * stat_count + stat_id
                )
                var sum_right = part_stat - sum_left

                # `TScoreCalcer::AddLeaf` for the L2 calcer: the Newton
                # objective, summed over both sides of every leaf.
                # `leafScore = (weight > 1e-20f ? (-sum*sum)/(weight+Lambda)
                #             : 0)` -- `score_calcers.cuh:54`. Every calcer of
                # theirs carries this guard. Ours divided unconditionally.
                # The weights are `max(.., 0)` clamped but the SUMS are not,
                # so a side whose weight cancels to zero with a non-zero
                # residual contributed `sum^2 / lambda` here and 0 in theirs,
                # and divides by zero outright if lambda is ever 0.
                if weight_left > Float32(1e-20):
                    score += (sum_left * sum_left) / (
                        weight_left + lambda_l2
                    )
                if weight_right > Float32(1e-20):
                    score += (sum_right * sum_right) / (
                        weight_right + lambda_l2
                    )

        if score > best_score:
            best_score = score
            best_bin = UInt32(bin_feature_id)

        offset += SCORE_BLOCK_SIZE * Int(grid_dim.x)

    # Block argmax by shared tree reduction. `__syncthreads` only, no shuffle
    # (`compute_scores.cu:20-51`), so this one ports without deviation.
    var s_score = stack_allocation[
        SCORE_BLOCK_SIZE,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var s_bin = stack_allocation[
        SCORE_BLOCK_SIZE,
        Scalar[DType.uint32],
        address_space = AddressSpace.SHARED,
    ]()
    s_score[tid] = best_score
    s_bin[tid] = best_bin
    barrier()

    var stride = SCORE_BLOCK_SIZE // 2
    while stride > 0:
        if tid < stride:
            # `gains[tid] > gains[tid+s] || (gains[tid] == gains[tid+s] &&
            #  indices[tid] > indices[tid+s])` -- `compute_scores.cu:30`.
            # On an exact tie the SMALLER bin-feature index wins. Ours kept
            # the lower `tid`, and a lower `tid` does not mean a lower bin:
            # thread 0 covers bins 0, 128, 256..., thread 5 covers 5, 133...
            # Ties are the common case, not the rare one -- constant
            # features, duplicated columns and small-integer gradient sums
            # all produce exact float equality -- so this made the split
            # depend on block geometry.
            var take = s_score[tid + stride] > s_score[tid]
            if s_score[tid + stride] == s_score[tid]:
                if s_bin[tid + stride] < s_bin[tid]:
                    take = True
            if take:
                s_score[tid] = s_score[tid + stride]
                s_bin[tid] = s_bin[tid + stride]
        barrier()
        stride //= 2

    if tid == 0:
        out_score.unsafe_store(Int(block_idx.x), s_score[0])
        out_bin.unsafe_store(Int(block_idx.x), s_bin[0])
