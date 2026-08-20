"""Bayesian bootstrap: CatBoost's GPU default sampling, ported.

PORT OF `BayesianBootstrapImpl` (`catboost/cuda/cuda_util/kernel/
bootstrap.cu:35-49`) and the weight application around it
(`gpu_data/bootstrap.h`: `BootstrappedWeights` fills ones, `Bootstrap`
draws, `BootstrapAndFilter` multiplies BOTH the der plane and the weight
plane by the same draw; Bayesian never produces zero weights, so their
zero-filter/gather branch is never taken -- `AreZeroWeightsAfterBootstrap`
is false for it and this port carries none of that machinery).

Their GPU oblivious searcher takes Bayesian by default and ASSERTS MVS
away (`greedy_subsets_searcher/weak_objective_impl.h:30`), so Bayesian is
the parity target, not one option of many. CatBoost's own CPU and GPU
produce DIFFERENT models under bootstrap (different RNG streams), so a
split-level oracle against CPU CatBoost cannot exist for this feature
even in principle; the gates are instead: the device stream is their
kernel's stream (same arithmetic, same per-thread seed walk), a seeded
fit is bit-reproducible, the draw distribution is Exp(1) at temperature
1, and `temperature = 0` collapses the whole path to an EXACT identity
(their `powf(tmp, 0) == 1`), which pins the wiring against the
bootstrap-off fit bit for bit.

================= DEVIATION BLOCK =================
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

from gbdt.gpu_util.kernel.random_gen import next_uniform_f

#: their launch shape (`bootstrap.cu:79-84`): block 256, grid
#: `min(ceil(seeds/256), ceil(rows/256))`.
comptime BOOTSTRAP_BLOCK_SIZE = 256

#: `TGpuAwareRandom::CreateSeeds` default `maxCountPerDevice = 256 * 256`.
comptime BOOTSTRAP_SEED_COUNT = 65536


def bayesian_bootstrap_kernel(
    seeds: MutPointer[UInt64, MutAnyOrigin],
    stats: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    temperature: Float32,
    mags: MutPointer[Float32, MutAnyOrigin],
    compute_mags_in: Int32,
):
    """`BayesianBootstrapImpl`, with the two `MultiplyVector`s folded in.

    Their loop (`bootstrap.cu:41-47`):

        const float tmp = (-log(NextUniform(&s) + 1e-20f));
        weights[i] = w * (temperature != 1.0f ? powf(tmp, temperature) : tmp);

    One draw per ROW in one grid-stride walk, per-thread seed advanced in
    registers and written back at the end -- the write-back is the RNG
    state between trees. `stats` is the two-plane layout `mse_kernel`
    writes: plane 0 (weights) and plane 1 (der) are both multiplied by
    the same draw, their `BootstrapAndFilter`'s two `MultiplyVector`s.
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
        var draw = next_uniform_f(s)
        var u = draw[0]
        s = draw[1]
        var tmp = -log(u + Float32(1e-20))
        var bw = tmp
        if temperature != Float32(1.0):
            bw = tmp**temperature
        var w = stats.unsafe_load(i) * bw
        var g = stats.unsafe_load(n + i) * bw
        stats.unsafe_store(i, w)
        stats.unsafe_store(n + i, g)
        mag_w += abs(w)
        mag_g += abs(g)
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


def launch_bayesian_bootstrap(
    ctx: DeviceContext,
    mut seeds: DeviceBuffer[DType.uint64],
    mut stats: DeviceBuffer[DType.float32],
    n_rows: Int,
    temperature: Float32,
    mut mags: DeviceBuffer[DType.float32],
    compute_mags: Bool,
) raises:
    """Their launch (`bootstrap.cu:79-84`): grid
    `min(ceil(seeds/256), ceil(rows/256))` blocks of 256."""
    var blocks = bootstrap_grid_blocks(n_rows)
    ctx.enqueue_function[bayesian_bootstrap_kernel](
        seeds.unsafe_ptr(), stats.unsafe_ptr(), Int32(n_rows), temperature,
        mags.unsafe_ptr(), Int32(1) if compute_mags else Int32(0),
        grid_dim=(blocks, 1, 1),
        block_dim=(BOOTSTRAP_BLOCK_SIZE, 1, 1),
    )
