"""The Bayesian bootstrap, gated the way a stochastic feature CAN be gated.

No split-level oracle is possible here even in principle: CatBoost's own
CPU and GPU draw from different RNGs and produce different models under
bootstrap, so there is no fixture to match. What IS checkable, and is:

1. SEEDS: the same base seed builds the same 65536-seed buffer, a
   different seed a different one.
2. DISTRIBUTION: at temperature 1 the draw is Exp(1) (`-log U`): every
   weight positive, mean and variance near 1 at 3-sigma-ish bounds for
   the sample size, and a real tail (max > 3). A wrong stream or a
   truncated draw fails at least one.
3. THE TWO-PLANE MULTIPLY: both planes start at 1.0, so after the kernel
   they must be BITWISE equal row for row -- one draw multiplied into
   both, their `BootstrapAndFilter`'s two `MultiplyVector`s.
4. STATE: a second launch over fresh ones, same seeds buffer, must draw
   DIFFERENTLY (the written-back seeds are the fit's RNG state); after
   recreating the buffer from the same base seed, the first draw repeats
   exactly.
5. THE FIT: temperature 0 turns every weight into `powf(tmp, 0) = 1`,
   so a bootstrap-ON fit at T=0 must land NEAR the bootstrap-OFF fit --
   a BAND, not bits, and the width is understood: the magnitude reduce
   moves from `mse_kernel` into the bootstrap kernel, whose block
   partition differs, so `fixed_scale`'s last bits differ, the dithered
   histograms shift by units, and a knife-edge split may flip (measured
   1.1e-3 relative on this fixture). What IS bitwise: T=0 run twice,
   and T=1 same-seed run twice -- each path is deterministic in itself
   since the float-atomic reduces became fixed-order folds (the flaky
   ancestor of this very check is what exposed them). A different seed
   must change the model.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from mojo_only.hist2_check import build_cindex
from gbdt.gpu_data.compressed_index_builder import build_layout
from gbdt.gpu_util.kernel.bootstrap import (
    BOOTSTRAP_SEED_COUNT,
    create_bootstrap_seeds,
    BOOTSTRAP_KERNEL_BAYESIAN,
    BOOTSTRAP_KERNEL_BERNOULLI,
    BOOTSTRAP_KERNEL_POISSON,
    launch_bootstrap,
)
from gbdt.methods.doc_parallel_boosting import TAdditiveModel, fit

comptime BS_ROWS = 100000
comptime FIT_ROWS = 2048
comptime FIT_FEATURES = 4
comptime FIT_FOLDS = 20
comptime FIT_TREES = 3


def _ones_stats(
    ctx: DeviceContext,
) raises -> DeviceBuffer[DType.float32]:
    var stats = ctx.enqueue_create_buffer[DType.float32](2 * BS_ROWS)
    var h = ctx.enqueue_create_host_buffer[DType.float32](2 * BS_ROWS)
    for i in range(2 * BS_ROWS):
        h.unsafe_ptr().unsafe_store(i, Float32(1.0))
    ctx.enqueue_copy(dst_buf=stats, src_ptr=h.unsafe_ptr())
    ctx.synchronize()
    return stats^


def _read(
    ctx: DeviceContext, mut b: DeviceBuffer[DType.float32], n: Int
) raises -> List[Float32]:
    var h = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=b)
    ctx.synchronize()
    var out = List[Float32]()
    for i in range(n):
        out.append(h.unsafe_ptr().unsafe_load(i))
    return out^


def _fit_losses(
    ctx: DeviceContext,
    bins: List[List[Int]],
    y: List[Float32],
    bootstrap: Bool,
    temperature: Float32,
    seed: UInt64,
) raises -> List[Float64]:
    var folds = List[Int]()
    for _ in range(FIT_FEATURES):
        folds.append(FIT_FOLDS)
    var lay = build_layout(folds)
    var cindex = build_cindex(ctx, lay, bins, FIT_ROWS)
    var targets = ctx.enqueue_create_buffer[DType.float32](FIT_ROWS)
    var weights = ctx.enqueue_create_buffer[DType.float32](FIT_ROWS)
    var ht = ctx.enqueue_create_host_buffer[DType.float32](FIT_ROWS)
    var hw = ctx.enqueue_create_host_buffer[DType.float32](FIT_ROWS)
    for r in range(FIT_ROWS):
        ht.unsafe_ptr().unsafe_store(r, y[r])
        hw.unsafe_ptr().unsafe_store(r, Float32(1.0))
    ctx.enqueue_copy(dst_buf=targets, src_ptr=ht.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=weights, src_ptr=hw.unsafe_ptr())
    ctx.synchronize()
    var model = TAdditiveModel()
    return fit(
        model, ctx, FIT_ROWS, folds, 3, cindex, targets, weights, False,
        FIT_TREES, Float32(0.3), Float32(3.0), True,
        bootstrap_bayesian=bootstrap,
        bagging_temperature=temperature,
        random_seed=seed,
    )


def check_bootstrap() raises:
    print("bayesian bootstrap (their GPU default; stream + wiring gates):")
    var ctx = DeviceContext()

    # ---- 1: seed buffer determinism --------------------------------------
    var s1 = create_bootstrap_seeds(ctx, UInt64(7))
    var s2 = create_bootstrap_seeds(ctx, UInt64(7))
    var s3 = create_bootstrap_seeds(ctx, UInt64(8))
    var h1 = ctx.enqueue_create_host_buffer[DType.uint64](BOOTSTRAP_SEED_COUNT)
    var h2 = ctx.enqueue_create_host_buffer[DType.uint64](BOOTSTRAP_SEED_COUNT)
    var h3 = ctx.enqueue_create_host_buffer[DType.uint64](BOOTSTRAP_SEED_COUNT)
    ctx.enqueue_copy(dst_ptr=h1.unsafe_ptr(), src_buf=s1)
    ctx.enqueue_copy(dst_ptr=h2.unsafe_ptr(), src_buf=s2)
    ctx.enqueue_copy(dst_ptr=h3.unsafe_ptr(), src_buf=s3)
    ctx.synchronize()
    var same = 0
    var diff8 = 0
    for i in range(BOOTSTRAP_SEED_COUNT):
        if h1.unsafe_ptr().unsafe_load(i) == h2.unsafe_ptr().unsafe_load(i):
            same += 1
        if h1.unsafe_ptr().unsafe_load(i) != h3.unsafe_ptr().unsafe_load(i):
            diff8 += 1
    if same != BOOTSTRAP_SEED_COUNT:
        raise Error("the same base seed produced different seed buffers")
    if diff8 < BOOTSTRAP_SEED_COUNT // 2:
        raise Error("a different base seed barely changed the seed buffer")
    print("  seeds: base 7 reproduces exactly; base 8 differs at",
          diff8, "of", BOOTSTRAP_SEED_COUNT)

    # ---- 2 + 3: distribution and the two-plane multiply -------------------
    var mags = ctx.enqueue_create_buffer[DType.float32](2)
    ctx.enqueue_memset(mags, Float32(0.0))
    var stats = _ones_stats(ctx)
    launch_bootstrap(
        ctx, BOOTSTRAP_KERNEL_BAYESIAN, s1, stats, BS_ROWS,
        Float32(1.0), mags, True,
    )
    var v = _read(ctx, stats, 2 * BS_ROWS)
    var total = Float64(0.0)
    var maxv = Float64(0.0)
    for i in range(BS_ROWS):
        var w = Float64(v[i])
        if v[i] != v[BS_ROWS + i]:
            raise Error("the two planes took different draws at row "
                        + String(i))
        if w <= 0.0:
            raise Error("a Bayesian weight is not positive: " + String(w))
        total += w
        if w > maxv:
            maxv = w
    var mean = total / Float64(BS_ROWS)
    var varsum = Float64(0.0)
    for i in range(BS_ROWS):
        var d = Float64(v[i]) - mean
        varsum += d * d
    var variance = varsum / Float64(BS_ROWS)
    # Exp(1): mean 1 (se ~ 1/sqrt(n) ~ 0.0032), var 1 (se ~ sqrt(8/n))
    if mean < 0.97 or mean > 1.03:
        raise Error("Bayesian draw mean " + String(mean) + " is not Exp(1)")
    if variance < 0.9 or variance > 1.1:
        raise Error("Bayesian draw variance " + String(variance)
                    + " is not Exp(1)")
    if maxv < 3.0:
        raise Error("no tail: max Bayesian weight " + String(maxv))
    print("  distribution: mean", mean, " var", variance, " max", maxv,
          "over", BS_ROWS, "draws -- Exp(1), both planes bitwise equal")

    # ---- 4: the seeds are STATE ------------------------------------------
    var stats2 = _ones_stats(ctx)
    ctx.enqueue_memset(mags, Float32(0.0))
    launch_bootstrap(
        ctx, BOOTSTRAP_KERNEL_BAYESIAN, s1, stats2, BS_ROWS,
        Float32(1.0), mags, True,
    )
    var v2 = _read(ctx, stats2, BS_ROWS)
    var moved = 0
    for i in range(BS_ROWS):
        if v2[i] != v[i]:
            moved += 1
    if moved < BS_ROWS // 2:
        raise Error("a second draw repeated the first: the seed state is"
                    " not advancing")
    var s1b = create_bootstrap_seeds(ctx, UInt64(7))
    var stats3 = _ones_stats(ctx)
    ctx.enqueue_memset(mags, Float32(0.0))
    launch_bootstrap(
        ctx, BOOTSTRAP_KERNEL_BAYESIAN, s1b, stats3, BS_ROWS,
        Float32(1.0), mags, True,
    )
    var v3 = _read(ctx, stats3, BS_ROWS)
    for i in range(BS_ROWS):
        if v3[i] != v[i]:
            raise Error("recreated seeds did not replay the first draw")
    print("  state: draw 2 moved", moved, "rows; recreated seeds replay"
          " draw 1 exactly")

    # ---- 5: through the fit ----------------------------------------------
    var bins = List[List[Int]]()
    for f in range(FIT_FEATURES):
        var col = List[Int]()
        for r in range(FIT_ROWS):
            col.append((r * 2654435761 + f * 40503) % FIT_FOLDS)
        bins.append(col^)
    var y = List[Float32]()
    for r in range(FIT_ROWS):
        y.append(Float32((r * 37 + 11) % 64 - 32) / Float32(32.0))

    var base = _fit_losses(ctx, bins, y, False, Float32(1.0), UInt64(0))
    var t0a = _fit_losses(ctx, bins, y, True, Float32(0.0), UInt64(9))
    var t0b = _fit_losses(ctx, bins, y, True, Float32(0.0), UInt64(9))
    for i in range(len(base)):
        var d = base[i] - t0a[i]
        if d < 0:
            d = -d
        # the BAND of the module docstring's point 5: fixed_scale rides a
        # different (deterministic) reduce partition under bootstrap, so
        # identity holds to a knife-edge-split band, not to bits
        if d > 1e-9 + 5e-3 * base[i]:
            raise Error(
                "temperature 0 is the identity draw and the fit moved: "
                + String(base[i]) + " vs " + String(t0a[i])
            )
        if t0a[i] != t0b[i]:
            raise Error("the T=0 path is not bit-deterministic")
    var a = _fit_losses(ctx, bins, y, True, Float32(1.0), UInt64(7))
    var b = _fit_losses(ctx, bins, y, True, Float32(1.0), UInt64(7))
    var c = _fit_losses(ctx, bins, y, True, Float32(1.0), UInt64(8))
    for i in range(len(a)):
        if a[i] != b[i]:
            raise Error("the same seed produced different fits")
    var any_diff = False
    for i in range(len(a)):
        if a[i] != c[i]:
            any_diff = True
    if not any_diff:
        raise Error("a different seed changed nothing: the bootstrap is"
                    " not reaching the fit")
    print("  fit: T=0 reproduces bootstrap-off to tolerance; seed 7 is"
          " bit-reproducible; seed 8 changes the model -- the draw"
          " reaches the histograms")
    check_bernoulli_and_poisson(ctx)


def check_bernoulli_and_poisson(ctx: DeviceContext) raises:
    """The two arms `launch_bootstrap` gained on 2026-08-21.

    PORTING_RULES 8: every switch is exercised on BOTH sides by a named
    check per side, and "the suite covers it" is not coverage. Bayesian was
    the only arm this file ran, so adding two arms to the kernel without
    adding them here would have left them on the unchecked side of exactly
    the switch that rule is about.

    Each arm is pinned against an ANALYTIC property of their own kernel
    rather than against a tally of ours:

      BERNOULLI (`UniformBootstrapImpl`, `bootstrap.cu:51-62`)
        sampleRate = 1.0 -> `NextUniform(&s) < 1.0` is true for every draw
                            in [0, 1), so EVERY weight is exactly 1.0 and
                            both planes are untouched
        sampleRate = 0.0 -> false for every draw, so EVERY weight is
                            exactly 0.0
        sampleRate = 0.5 -> the surviving fraction is binomial with mean
                            0.5 and se ~ 1/(2*sqrt(n)); the check bands it
                            rather than fixing it, because a fixed count
                            would be asserting our own RNG stream

      POISSON (`PoissonBootstrapImpl`, `:7-19`, via `NextPoisson`,
      `random_gen.cuh:57-72`)
        lambda <= 20     -> their small-alpha arm returns `k - 1` from a
                            product-of-uniforms loop, so every weight is a
                            NON-NEGATIVE INTEGER, and the mean is lambda
        lambda = 1       -> mean 1, and a Poisson(1) sample must contain
                            both zeros (p = 0.368) and values >= 3
                            (p = 0.080), which a constant or a uniform
                            would not

    Both planes must take the SAME draw, which is their two
    `MultiplyVector`s against one `BootstrappedWeights` buffer
    (`gpu_data/bootstrap.h:100-102`).
    """
    var seeds = create_bootstrap_seeds(ctx, UInt64(11))
    var mags = ctx.enqueue_create_buffer[DType.float32](2)

    # -- Bernoulli, the two exact identities -----------------------------
    for rate_i in range(2):
        var rate = Float32(1.0) if rate_i == 0 else Float32(0.0)
        var want = Float32(1.0) if rate_i == 0 else Float32(0.0)
        var st = _ones_stats(ctx)
        ctx.enqueue_memset(mags, Float32(0.0))
        launch_bootstrap(
            ctx, BOOTSTRAP_KERNEL_BERNOULLI, seeds, st, BS_ROWS, rate,
            mags, True,
        )
        var v = _read(ctx, st, 2 * BS_ROWS)
        for i in range(BS_ROWS):
            if v[i] != want or v[BS_ROWS + i] != want:
                raise Error(
                    "Bernoulli at rate " + String(rate)
                    + " is not the exact identity/annihilator at row "
                    + String(i) + ": " + String(v[i]) + " / "
                    + String(v[BS_ROWS + i])
                )
    print("  bernoulli: rate 1.0 leaves both planes exactly 1.0;"
          " rate 0.0 zeroes both exactly")

    # -- Bernoulli at 0.5: the draw must SCATTER -------------------------
    var st5 = _ones_stats(ctx)
    ctx.enqueue_memset(mags, Float32(0.0))
    launch_bootstrap(
        ctx, BOOTSTRAP_KERNEL_BERNOULLI, seeds, st5, BS_ROWS,
        Float32(0.5), mags, True,
    )
    var v5 = _read(ctx, st5, 2 * BS_ROWS)
    var kept = 0
    for i in range(BS_ROWS):
        if v5[i] != v5[BS_ROWS + i]:
            raise Error(
                "the two planes took different Bernoulli draws at row "
                + String(i)
            )
        if v5[i] != Float32(0.0) and v5[i] != Float32(1.0):
            raise Error(
                "a Bernoulli weight is neither 0 nor 1: " + String(v5[i])
            )
        if v5[i] == Float32(1.0):
            kept += 1
    var frac = Float64(kept) / Float64(BS_ROWS)
    if frac < 0.47 or frac > 0.53:
        raise Error(
            "Bernoulli at 0.5 kept " + String(frac)
            + " of the rows, which is outside the binomial band"
        )
    print("  bernoulli: rate 0.5 kept", kept, "of", BS_ROWS,
          "(fraction", frac, "), both planes identical")

    # -- Poisson: integer weights, mean lambda, a real tail --------------
    var stp = _ones_stats(ctx)
    ctx.enqueue_memset(mags, Float32(0.0))
    launch_bootstrap(
        ctx, BOOTSTRAP_KERNEL_POISSON, seeds, stp, BS_ROWS,
        Float32(1.0), mags, True,
    )
    var vp = _read(ctx, stp, 2 * BS_ROWS)
    var total = Float64(0.0)
    var zeros = 0
    var big = 0
    for i in range(BS_ROWS):
        var w = vp[i]
        if w != vp[BS_ROWS + i]:
            raise Error(
                "the two planes took different Poisson draws at row "
                + String(i)
            )
        if w < Float32(0.0):
            raise Error("a Poisson weight is negative: " + String(w))
        # their small-alpha arm returns `k - 1`, an integer
        if w != Float32(Int(w)):
            raise Error(
                "a Poisson weight is not an integer: " + String(w)
            )
        total += Float64(w)
        if w == Float32(0.0):
            zeros += 1
        if w >= Float32(3.0):
            big += 1
    var mean = total / Float64(BS_ROWS)
    if mean < 0.94 or mean > 1.06:
        raise Error(
            "Poisson(1) draw mean " + String(mean) + " is not 1"
        )
    if zeros < BS_ROWS // 5:
        raise Error(
            "Poisson(1) produced " + String(zeros)
            + " zeros; p(0) = 0.368 so a real sample has far more"
        )
    if big < BS_ROWS // 40:
        raise Error(
            "Poisson(1) produced no tail: " + String(big)
            + " draws >= 3, where p = 0.080"
        )
    print("  poisson: lambda 1 -> mean", mean, ", integer weights,",
          zeros, "zeros and", big, "draws >= 3 -- both planes identical")
