"""`random_strength`'s magnitude: how big the score noise is.

PORT OF `catboost/cuda/methods/random_score_helper.h` at CatBoost
`54a8143a`. Transliterated. Do not improve.

Three functions, and between them they decide the ONE scalar every noisy
score kernel multiplies its normal draw by:

    ComputeStdDev(target)                     -- sqrt(sum wt^2/w / count)
    CalcScoreModelLengthMult(n, modelSize)    -- the decay with tree count
    ComputeScoreStdDev(mult, strength, target)-- mult * stdDev * strength

`CalcScoreModelLengthMult` is the only reason `random_strength` fades: the
noise is scaled by `L / (1 + L)` where `L = exp(log(n) - iteration * step)`,
so at iteration 0 it is ~1 and it decays toward 0 as the ensemble's total
step approaches `log(n)`. Their `modelSize` argument is
`iteration * learningRate` (`doc_parallel_boosting.h:358-359`), NOT a byte
count and NOT `model_size_reg`.

=================== THE TWO ARMS DO NOT AGREE, MEASURED ==================
The brief this port was written to said the greedy arm and the doc-parallel
arm "compute the SAME product by different routes". THAT IS HALF TRUE and
the half that is false matters.

Same: the multiplier. The doc-parallel arm forms
`mult * stdDev * randomStrength` here; the greedy arm has already folded
`mult` into its option (`greedy_subsets_searcher.h:76`,
`options.RandomStrength *= randomStrengthMult`) and then forms
`Options.RandomStrength * ComputeTargetStdDev(target)`
(`greedy_search_helper.cpp:384-385`). Those two products are the same three
factors.

DIFFERENT: the standard deviation itself, in TWO ways.

  1. THE DENOMINATOR. `ComputeStdDev` below divides `sum2` by the OBJECT
     COUNT (`random_score_helper.h:14-15`); `ComputeTargetStdDev` divides
     the same numerator by the TOTAL WEIGHT
     (`greedy_search_helper.cpp:376-377`). They coincide only when every
     weight is exactly 1. Under Newton -- CatBoost's default leaf
     estimation -- the weight plane is the HESSIAN, so they do not
     coincide, and for Logloss (hessian <= 0.25) the greedy arm's std dev
     is at least twice the doc-parallel arm's on the same target.
  2. THE ZERO GUARD. `ComputeTargetStdDev`'s kernel skips any row with
     `w <= 1e-15` (`compute_scores.cu:239`); `ComputeStdDev` has no such
     test and divides straight through `DivideVector`, so a zero-weight row
     with a non-zero gradient contributes an inf and poisons the whole
     reduction to NaN.

Neither is corrected here. Both are transliterated where their file puts
them: this one here, `compute_target_std_dev` in `greedy_search_helper`.
=========================================================================

DEVIATION 137 (also stated in PORTING.md): their `ComputeStdDev` is built
out of two generic device ops, `DivideVector` then
`DotProduct(tmp, tmp, &weights)` (`cuda_util/transform.h`,
`cuda_util/dot_product.h`), NEITHER OF WHICH IS PORTED here. This file
fuses them into one kernel that never materializes their `tmp`. The
arithmetic per row is theirs exactly -- `w * (wt/w)^2`, including the
missing zero guard -- but the accumulation is Float32 with a deterministic
lane fold where theirs is a double block reduce ending in a float atomic.
"""

from std.gpu import block_idx, grid_dim, thread_idx
from std.math import exp, log, sqrt
from max.gpu.host import DeviceBuffer, DeviceContext

from mojo_only.kernel_matrix import partition_chunks_sm_for
from mojo_only.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_mul_add,
)
from gbdt.targets.kernel.pointwise_targets import (
    deterministic_sum_lanes_kernel,
    pinned_block_sum,
)


#: `ComputeTargetVariance`'s block (`compute_scores.cu:290`). `ComputeStdDev`
#: has no kernel of its own upstream -- it is two library calls -- so the one
#: written here borrows the sibling's geometry rather than inventing one.
comptime STD_DEV_BLOCK = 512


def std_dev_partials_kernel(
    stats: MutPointer[Float32, MutAnyOrigin],
    size_in: Int32,
    stat_line_size_in: Int32,
    partials: MutPointer[Float32, MutAnyOrigin],
):
    """`DivideVector(tmp, Weights)` then `DotProduct(tmp, tmp, &Weights)`,
    fused (`random_score_helper.h:11-13`). See DEVIATION 137 above.

    Per row: `tmp = wt / w`, contribution `w * tmp * tmp`. NO ZERO GUARD --
    theirs has none on this path (`DivideVector`'s `skipZeroes` defaults to
    false, `cuda_util/kernel/transform.cu:135-144`) and adding one would
    make this arm agree with the greedy arm on a case where CatBoost's two
    arms disagree.

    THE TARGET ARRIVES AS ONE TWO-PLANE BUFFER, plane 0 the weight and
    plane 1 the gradient, which is this repository's `stats` convention and
    not `TL2Target`'s two buffers -- the same bridge `split_stat_planes`
    documents, taken here without its host round trip because a kernel may
    read two planes of one buffer through ONE pointer.

    One slot per block; `deterministic_sum_lanes_kernel[1]` folds them.
    """
    var size = Int(size_in)
    var stat_line_size = Int(stat_line_size_in)
    var i = STD_DEV_BLOCK * Int(block_idx.x) + Int(thread_idx.x)
    var acc = Float32(0.0)
    # ======================== DEVIATION 256 ==========================
    # IDENTITY_PATHS row 9 names this chain by name -- the
    # `std_dev_partials_kernel` `w*tmp*tmp` leftover. `acc += w*tmp*tmp`
    # is a multiply feeding an add, so which backend fuses it was the
    # codegen's whim; `identical_mul_add` pins it to one `fma` under
    # IDENTICAL and is the naive chain under FAST, same association
    # (`(w*tmp)*tmp + acc`), same bits. Row 10 rides along -- `tmp` is a
    # quotient and `w*tmp` squares toward zero, so both intermediates
    # and the accumulator are stored through `ftz` (the gradient can be
    # denormal-small while the noise scale it feeds is not). Every
    # `ftz` is a comptime no-op under FAST.
    # =================================================================
    while i < size:
        var w = stats.unsafe_load(i)
        var tmp = ftz(stats.unsafe_load(stat_line_size + i) / w)
        acc = ftz(identical_mul_add(ftz(w * tmp), tmp, acc))
        i += Int(grid_dim.x) * STD_DEV_BLOCK
    # IDENTITY_PATHS row 8 (DEVIATION 251): the pinned-shape fold, so the
    # within-block sum does not follow AMD's 64-wide wavefront.
    var total = pinned_block_sum[block_size=STD_DEV_BLOCK](acc)
    if thread_idx.x == 0:
        partials.unsafe_store(Int(block_idx.x), total)


def std_dev_blocks(size: Int, sm_count: Int) -> Int:
    """`min(4 * SMCount(), CeilDivide(size, blockSize))`
    (`compute_scores.cu:291`), the same machine-sized strided grid the
    sibling reduce uses.

    **AND IT IS A NUMERIC ROW, PINNED HERE FOR THE SAME REASON
    `partition_stats_chunks` IS** (IDENTITY_PATHS row 7, DEVIATION 252):
    the block count partitions a FLOAT sum -- each block's partial is
    reduced in float and `deterministic_sum_lanes_kernel` adds the
    partials -- so the machine's core count decides the last bits of the
    score-noise std dev, and through it every noised score of the fit.
    Inert at this port's default `random_strength = 0`, but CatBoost's
    default is 1.0, so the row must hold BEFORE anyone wires that default.

    Under `IDENTICAL` the `sm_count` fed to the formula therefore comes
    from `kernel_matrix.partition_chunks_sm_for`, the SAME pin row 7 uses
    (32 on every vendor, deliberately no real device's own number); under
    `FAST` it is the device's count, CatBoost's behavior unchanged. The
    pin is applied INSIDE this function -- the only place the formula
    lives -- so the launch, the `partials` buffer sizing and the fold
    count in `compute_std_dev` cannot disagree, which is row 7's exact
    argument. `size`'s arm (`by_data`) is pure f(size) and needs no pin.
    """
    comptime _identical = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var sm = partition_chunks_sm_for[_identical](sm_count)
    var by_data = (size + STD_DEV_BLOCK - 1) // STD_DEV_BLOCK
    var by_machine = 4 * sm
    if by_machine < 1:
        by_machine = 1
    if by_data < by_machine:
        return by_data if by_data > 0 else 1
    return by_machine


def compute_std_dev(
    ctx: DeviceContext,
    mut stats: DeviceBuffer[DType.float32],
    count: Int,
    stat_line_size: Int,
    sm_count: Int,
) raises -> Float64:
    """`ComputeStdDev` (`random_score_helper.h:9-16`).

        DivideVector(tmp, target.Weights);
        const double sum2 = DotProduct(tmp, tmp, &target.Weights);
        const double count = target.WeightedTarget.GetObjectsSlice().Size();
        return sqrt(sum2 / (count + 1e-100));

    `count` is the OBJECT COUNT, not the summed weight. See the file
    docstring for why that is the difference between the two arms.
    """
    if count <= 0:
        return 0.0
    var n_blocks = std_dev_blocks(count, sm_count)
    var partials = ctx.enqueue_create_buffer[DType.float32](n_blocks)
    var out = ctx.enqueue_create_buffer[DType.float32](1)
    ctx.enqueue_function[std_dev_partials_kernel](
        stats.unsafe_ptr(),
        Int32(count),
        Int32(stat_line_size),
        partials.unsafe_ptr(),
        grid_dim=(n_blocks, 1, 1),
        block_dim=(STD_DEV_BLOCK, 1, 1),
    )
    ctx.enqueue_function[deterministic_sum_lanes_kernel[1]](
        partials.unsafe_ptr(), Int32(n_blocks), out.unsafe_ptr(),
        grid_dim=1, block_dim=256,
    )
    var h = ctx.enqueue_create_host_buffer[DType.float32](1)
    ctx.enqueue_copy(dst_buf=h, src_buf=out)
    ctx.synchronize()
    # the staging buffer must outlive the synchronize that guarantees the
    # copy ran ([[mojo-buffer-freed-at-last-use]]); reading it here IS that
    # use, and the two device buffers are held to the same point.
    var sum2 = Float64(h[0])
    _ = partials^
    _ = out^
    return sqrt(sum2 / (Float64(count) + 1e-100))


def calc_score_model_length_mult(
    sample_count: Float64, model_size: Float64
) -> Float64:
    """`CalcScoreModelLengthMult` (`random_score_helper.h:18-22`).

        double modelExpLength = log(sampleCount);
        double modelLeft = exp(modelExpLength - modelSize);
        return modelLeft / (1 + modelLeft);

    `modelSize` is `iteration * learningRate`
    (`doc_parallel_boosting.h:358-359`).

    `std.math.log` carries this repository's known ~5e-8 absolute error
    ([[mojo-log-breaks-ties]]). It is tolerated HERE and only here: the
    result scales a random normal, so an error five parts in 1e8 in the
    scale is not a decision. It would NOT be tolerable inside the score
    comparison, and no caller may reuse this value as a tie-break.
    """
    # DEVIATION 256, justified UNPINNED: HOST-side Float64 exp/log, so
    # IDENTITY_PATHS row 12 (device transcendentals) does not reach it,
    # and the docstring above already prices the libm error as a scale
    # on a random normal, never a decision.
    var model_exp_length = log(sample_count)
    var model_left = exp(model_exp_length - model_size)
    return model_left / (1.0 + model_left)


def compute_score_std_dev(
    ctx: DeviceContext,
    model_length_mult: Float64,
    random_strength: Float64,
    mut stats: DeviceBuffer[DType.float32],
    count: Int,
    stat_line_size: Int,
    sm_count: Int,
) raises -> Float64:
    """`ComputeScoreStdDev` (`random_score_helper.h:24-32`).

        if (modelLengthMult * randomStrength) {
            double stdDev = ComputeStdDev(target);
            return modelLengthMult * stdDev * randomStrength;
        } else {
            return 0;
        }

    THE GUARD IS ON THE PRODUCT, not on either factor, and it is what keeps
    the reduce off the critical path of every fit that does not ask for
    noise. The multiplication order below is theirs
    (`mult * stdDev * strength`) and is kept because float multiplication
    does not associate.
    """
    if model_length_mult * random_strength != 0.0:
        var std_dev = compute_std_dev(
            ctx, stats, count, stat_line_size, sm_count
        )
        return model_length_mult * std_dev * random_strength
    return 0.0
