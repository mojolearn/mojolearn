"""The row-sampling draws: Bayesian, Bernoulli and Poisson.

PORT OF `BayesianBootstrapImpl` (`catboost/cuda/cuda_util/kernel/
bootstrap.cu:35-49`), `UniformBootstrapImpl` (`:51-62`) and
`PoissonBootstrapImpl` (`:7-19`), and the weight application around them
(`gpu_data/bootstrap.h`: `BootstrappedWeights` fills ones, `Bootstrap`
draws, `BootstrapAndFilter` multiplies BOTH the der plane and the weight
plane by the same draw; Bayesian never produces zero weights, so their
zero-filter/gather branch is never taken -- `AreZeroWeightsAfterBootstrap`
is false for it and this port carries none of that machinery).

Their GPU oblivious searcher takes Bayesian by default and ASSERTS MVS
away (`greedy_subsets_searcher/weak_objective_impl.h:30`), so Bayesian is
the parity target; Bernoulli and Poisson are the two other arms of the
same `Bootstrap` dispatch (`gpu_data/bootstrap.h:41-92`), and
`GammaBootstrapImpl` (`:21-33`) is NOT ported because their own
`BayesianBootstrap` has the only call to it COMMENTED OUT (`:82`).

WHAT BERNOULLI IS, since the name differs from their kernel's:
`EBootstrapType::Bernoulli` dispatches to `UniformBootstrap`
(`bootstrap.h:60-66`), whose kernel draws `w * (u < sampleRate ? 1 : 0)`.
It is the `subsample` knob every GBDT user knows: each tree sees a
different random `subsample` fraction of the rows. It is REGULARIZATION,
not speed -- see DEVIATION 69 for why it is not speed HERE in particular. CatBoost's own CPU and GPU
produce DIFFERENT models under bootstrap (different RNG streams), so a
split-level oracle against CPU CatBoost cannot exist for this feature
even in principle; the gates are instead: the device stream is their
kernel's stream (same arithmetic, same per-thread seed walk), a seeded
fit is bit-reproducible, the draw distribution is Exp(1) at temperature
1, and `temperature = 0` collapses the whole path to an EXACT identity
(their `powf(tmp, 0) == 1`), which pins the wiring against the
bootstrap-off fit bit for bit.

================= DEVIATION BLOCK =================
DEVIATION 69: THE ZERO-WEIGHT ROWS ARE NOT FILTERED OUT.

For Bernoulli and Poisson their `BootstrapAndFilter` does not stop at the
multiply. `AreZeroWeightsAfterBootstrap` is true for exactly those two
(`enum_helpers.cpp:849-856`), so they run `FilterZeroEntries`, gather the
der/weight/index columns down to the surviving rows, and return
`isContinuousIndices = false` (`gpu_data/bootstrap.h:126-153`,
`weak_objective_impl.h:30-45`). The searcher then works on a SMALLER row
set addressed through a non-contiguous index list.

This port multiplies and stops. THE MODEL IS THE SAME MODEL: a row whose
bootstrap weight is zero contributes `0` to its weight plane and `0` to
its gradient plane, so it adds nothing to any histogram cell, nothing to
any leaf sum, and nothing to either fixed-point magnitude (`|0| == 0`, so
the scale is unchanged too). Filtering is a way of not paying for those
rows, not a way of getting a different answer.

WHAT IT COSTS: at `subsample = 0.66` we stream 100% of the rows through
the histogram where they stream 66%, so the sampling buys accuracy here
and buys accuracy AND about a third of the histogram time there. That is
the whole of the difference and it is a real one to close.

THE ONE PLACE THE ANSWER COULD DIVERGE, and it cannot today: a score-side
test that counts ROWS rather than summing WEIGHTS would see the filtered
count on their side and the full count on ours. `min_data_in_leaf` is
that test, and it is NOT WIRED in this port's searcher -- grep
`greedy_search_helper.mojo` for it and there is nothing. **If
`min_data_in_leaf` is ever wired, this deviation stops being
output-identical and the filter becomes required.** That sentence is the
whole reason this paragraph exists.

ONE kernel where they launch three. Theirs: `BayesianBootstrapImpl` over
a ones buffer, then `MultiplyVector` twice (der, weights). Ours folds the
two multiplies into the draw loop -- the draw stream is UNCHANGED (same
grid shape, same per-thread seed sequence, one draw per row) -- and, when
the build quantizes (`fixed_point.mojo`), accumulates the two plane
magnitudes their design never needs: the bootstrapped plane is what the
histogram accumulates, a Bayesian weight reaches ~46 at the tail of
`-log(u + 1e-20)`, and a scale bounded by the UN-bootstrapped magnitudes
would overflow Int32 silently. Same block-reduce-plus-one-atomicAdd shape
as `mse_kernel`'s magnitude reduce.

The SEED FILL is host-side splitmix64 from the caller's seed where theirs
is 65536 draws of their host `TRandom::NextUniformL`
(`gpu_random.cpp:258-267`): their host RNG is not ported, every
per-thread DEVICE walk from those seeds is. An exact replay of a specific
CatBoost-GPU run was never available on this machine to compare against;
reproducibility of OUR seeded runs is what the fill must provide.
===================================================
"""

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.math import log
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from gbdt.gpu_util.kernel.random_gen import (
    next_poisson_f,
    next_uniform_f,
)

#: the three arms of their `Bootstrap` dispatch this file draws
#: (`gpu_data/bootstrap.h:41-92`). The names are their `EBootstrapType`
#: spellings, not their kernel names: `Bernoulli` dispatches to
#: `UniformBootstrap`.
comptime BOOTSTRAP_KERNEL_BAYESIAN = 0
comptime BOOTSTRAP_KERNEL_BERNOULLI = 1
comptime BOOTSTRAP_KERNEL_POISSON = 2

#: their launch shape (`bootstrap.cu:79-84`): block 256, grid
#: `min(ceil(seeds/256), ceil(rows/256))`.
comptime BOOTSTRAP_BLOCK_SIZE = 256

#: `TGpuAwareRandom::CreateSeeds` default `maxCountPerDevice = 256 * 256`.
comptime BOOTSTRAP_SEED_COUNT = 65536


def bootstrap_kernel[bootstrap_type: Int](
    seeds: MutPointer[UInt64, MutAnyOrigin],
    stats: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    param: Float32,
    mags: MutPointer[Float32, MutAnyOrigin],
    compute_mags_in: Int32,
    stat_count_in: Int32,
):
    """Their three draw kernels, with the two `MultiplyVector`s folded in.

    Their three loops, term for term:

      BAYESIAN (`bootstrap.cu:41-47`), `param` is `bagging_temperature`
        const float tmp = (-log(NextUniform(&s) + 1e-20f));
        weights[i] = w * (temperature != 1.0f
                          ? powf(tmp, temperature) : tmp);

      BERNOULLI (`UniformBootstrapImpl`, `:56-60`), `param` is `subsample`
        const float flag = (NextUniform(&s) < sampleRate) ? 1.0f : 0.0f;
        weights[i] = w * flag;

      POISSON (`PoissonBootstrapImpl`, `:12-16`), `param` is `lambda`
        weights[i] = w * NextPoisson(&s, lambda);

    One draw per ROW in one grid-stride walk, per-thread seed advanced in
    registers and written back at the end -- the write-back is the RNG
    state between trees. `stats` is the two-plane layout `mse_kernel`
    (now `pointwise_target_kernel`) writes: plane 0 (weights) and plane 1
    (der) are both multiplied by the same draw, their
    `BootstrapAndFilter`'s two `MultiplyVector`s.

    NOTE THE SEED BASE DIFFERS BETWEEN THEIR OWN ARMS and is copied:
    Bayesian and Poisson do `seeds += blockIdx.x * blockDim.x +
    threadIdx.x` (`:9`, `:37`), Uniform does `seeds += i` where `i` is the
    same expression (`:53-54`). Same address, written twice.
    """
    var n = Int(n_rows_in)
    var tid = Int(thread_idx.x)
    var gid = Int(block_idx.x) * BOOTSTRAP_BLOCK_SIZE + tid
    var s = seeds.unsafe_load(gid)

    var mag_w = Float32(0.0)
    var mag_g = Float32(0.0)

    var i = gid
    var stride = Int(grid_dim.x) * BOOTSTRAP_BLOCK_SIZE
    while i < n:
        var bw: Float32

        @parameter
        if bootstrap_type == BOOTSTRAP_KERNEL_BAYESIAN:
            var draw = next_uniform_f(s)
            s = draw[1]
            var tmp = -log(draw[0] + Float32(1e-20))
            bw = tmp
            if param != Float32(1.0):
                bw = tmp**param
        elif bootstrap_type == BOOTSTRAP_KERNEL_BERNOULLI:
            var draw = next_uniform_f(s)
            s = draw[1]
            bw = Float32(1.0) if draw[0] < param else Float32(0.0)
        else:
            var draw = next_poisson_f(s, param)
            s = draw[1]
            bw = draw[0]

        # EVERY DERIVATIVE PLANE, NOT JUST THE FIRST. This loop used to be
        # two hard-coded loads, `stats[i]` and `stats[n + i]`, and
        # `launch_bootstrap` had no plane count at all. That is correct for
        # a single-target loss, where `stat_count == 2`, and SILENTLY WRONG
        # for MultiClass / MultiClassOneVsAll, where it is
        # `1 + approx_dim`: planes 2..N kept their un-bootstrapped values,
        # so a row was sampled OUT of one class and fully present in the
        # others. `train()` accepts `objective="MultiClass"` together with
        # `bootstrap_type` and refused neither.
        #
        # Their draw is a per-row WEIGHT, and it scales every column:
        # `MultiplyVector(der, tmp); MultiplyVector(weights, tmp)`
        # (`gpu_data/bootstrap.h:96-98`), and on the greedy arm it is folded
        # into the weight BEFORE `Approximate` writes any der column
        # (`pointwise_target_impl.h:181-183`).
        var w = stats.unsafe_load(i) * bw
        stats.unsafe_store(i, w)
        mag_w += abs(w)

        # THE GRADIENT MAGNITUDE FOLLOWS DEVIATION 79's CONVENTION:
        # `sum over rows of max over classes |der_k|`, one bound valid for
        # every plane at once, because `choose_scale` takes ONE scale for
        # the whole histogram. The old two-plane code also bounded the
        # fixed-point scale from 2 of N planes, which under-bounded it for
        # multiclass. At `stat_count == 2` this reduces to `abs(g)` exactly,
        # so nothing about a single-target fit changes.
        var gmax = Float32(0.0)
        for k in range(1, Int(stat_count_in)):
            var g = stats.unsafe_load(k * n + i) * bw
            stats.unsafe_store(k * n + i, g)
            var a = abs(g)
            if a > gmax:
                gmax = a
        mag_g += gmax
        i += stride
    seeds.unsafe_store(gid, s)

    if compute_mags_in != Int32(0):
        # the same reduce shape as `mse_kernel`'s magnitude tail; the
        # block total is STORED at this block's slot and folded in one
        # fixed order by `deterministic_sum_lanes_kernel`, because these
        # magnitudes feed `fixed_scale` and a float-atomic combine made
        # same-seed fits differ (caught by bootstrap_check 2026-08-21).
        var red = stack_allocation[
            2 * BOOTSTRAP_BLOCK_SIZE,
            Scalar[DType.float32],
            address_space = AddressSpace.SHARED,
        ]()
        red[tid] = mag_w
        red[BOOTSTRAP_BLOCK_SIZE + tid] = mag_g
        barrier()
        var step = BOOTSTRAP_BLOCK_SIZE // 2
        while step > 0:
            if tid < step:
                red[tid] = red[tid] + red[tid + step]
                red[BOOTSTRAP_BLOCK_SIZE + tid] = (
                    red[BOOTSTRAP_BLOCK_SIZE + tid]
                    + red[BOOTSTRAP_BLOCK_SIZE + tid + step]
                )
            barrier()
            step //= 2
        if tid == 0:
            mags.unsafe_store(2 * Int(block_idx.x), red[0])
            mags.unsafe_store(
                2 * Int(block_idx.x) + 1, red[BOOTSTRAP_BLOCK_SIZE]
            )


def create_bootstrap_seeds(
    ctx: DeviceContext, base_seed: UInt64
) raises -> DeviceBuffer[DType.uint64]:
    """The 65536-seed buffer of `TGpuAwareRandom::CreateSeeds`, filled by
    splitmix64 from `base_seed` (see the DEVIATION BLOCK)."""
    var seeds = ctx.enqueue_create_buffer[DType.uint64](
        BOOTSTRAP_SEED_COUNT
    )
    var h = ctx.enqueue_create_host_buffer[DType.uint64](
        BOOTSTRAP_SEED_COUNT
    )
    var x = base_seed
    for i in range(BOOTSTRAP_SEED_COUNT):
        # splitmix64: the standard seed-expansion mix
        x += UInt64(0x9E3779B97F4A7C15)
        var z = x
        z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
        z = z ^ (z >> 31)
        h.unsafe_ptr().unsafe_store(i, z)
    ctx.enqueue_copy(dst_buf=seeds, src_ptr=h.unsafe_ptr())
    ctx.synchronize()
    return seeds^


def bootstrap_grid_blocks(n_rows: Int) -> Int:
    """Their grid formula, exported so the caller sizes the partials
    buffer the reduce reads."""
    var by_seeds = BOOTSTRAP_SEED_COUNT // BOOTSTRAP_BLOCK_SIZE
    var by_rows = (n_rows + BOOTSTRAP_BLOCK_SIZE - 1) // BOOTSTRAP_BLOCK_SIZE
    var blocks = by_seeds
    if by_rows < blocks:
        blocks = by_rows
    if blocks < 1:
        blocks = 1
    return blocks


def launch_bootstrap(
    ctx: DeviceContext,
    bootstrap_type: Int,
    mut seeds: DeviceBuffer[DType.uint64],
    mut stats: DeviceBuffer[DType.float32],
    n_rows: Int,
    param: Float32,
    mut mags: DeviceBuffer[DType.float32],
    compute_mags: Bool,
    stat_count: Int,
) raises:
    """`TBootstrap::Bootstrap`'s switch (`gpu_data/bootstrap.h:41-92`)
    over their three launches (`bootstrap.cu:64-84`), each grid
    `min(ceil(seeds/256), ceil(rows/256))` blocks of 256.

    `param` is whichever of their three scalars the arm takes:
    `bagging_temperature`, `subsample`, or `lambda`. Their config reads
    them from three different option fields and hands each kernel one
    float, which is the shape kept here.
    """
    var blocks = bootstrap_grid_blocks(n_rows)
    if bootstrap_type == BOOTSTRAP_KERNEL_BAYESIAN:
        ctx.enqueue_function[
            bootstrap_kernel[BOOTSTRAP_KERNEL_BAYESIAN]
        ](
            seeds.unsafe_ptr(), stats.unsafe_ptr(), Int32(n_rows), param,
            mags.unsafe_ptr(), Int32(1) if compute_mags else Int32(0),
            Int32(stat_count),
            grid_dim=(blocks, 1, 1),
            block_dim=(BOOTSTRAP_BLOCK_SIZE, 1, 1),
        )
    elif bootstrap_type == BOOTSTRAP_KERNEL_BERNOULLI:
        ctx.enqueue_function[
            bootstrap_kernel[BOOTSTRAP_KERNEL_BERNOULLI]
        ](
            seeds.unsafe_ptr(), stats.unsafe_ptr(), Int32(n_rows), param,
            mags.unsafe_ptr(), Int32(1) if compute_mags else Int32(0),
            Int32(stat_count),
            grid_dim=(blocks, 1, 1),
            block_dim=(BOOTSTRAP_BLOCK_SIZE, 1, 1),
        )
    elif bootstrap_type == BOOTSTRAP_KERNEL_POISSON:
        ctx.enqueue_function[
            bootstrap_kernel[BOOTSTRAP_KERNEL_POISSON]
        ](
            seeds.unsafe_ptr(), stats.unsafe_ptr(), Int32(n_rows), param,
            mags.unsafe_ptr(), Int32(1) if compute_mags else Int32(0),
            Int32(stat_count),
            grid_dim=(blocks, 1, 1),
            block_dim=(BOOTSTRAP_BLOCK_SIZE, 1, 1),
        )
    else:
        raise Error(
            "unknown bootstrap kernel type " + String(bootstrap_type)
        )
