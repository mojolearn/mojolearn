"""Which searcher should `fit` default to? A measurement, not an argument.

`PORTING.md` 106 shipped `use_pointwise_searcher` defaulting to False and
said the flip was a measurement's job. This is that measurement.

The two arms are BIT-IDENTICAL over twenty iterations
(`pixi run check-fit-pointwise`), so this is purely a performance question
and nothing about the answer can change a user's numbers -- which is what
makes flipping the default a free action if the measurement says to
([[mojotrees-switches-must-flip]]).

METHOD, and it matters on this box:

* the two arms are INTERLEAVED inside one loop and the MINIMUM of each is
  reported, which is `density_probe`'s convention. Another session is
  running GPU work on this machine; a sequential A-then-B measurement on a
  contended box has already drifted a result 4.5x invisibly in this
  repository's history.
* the ratio is reported beside both absolute numbers, because a ratio
  between two contended runs is more trustworthy than either alone.
* the fixture spans all three feature policies, so both arms dispatch
  across every kernel family they have.
"""

from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from gbdt.gpu_data.compressed_index_builder import build_layout
from gbdt.gpu_data.kernel.binarize import (
    WRITE_BLOCK_SIZE,
    write_compressed_index_kernel,
)
from gbdt.methods.doc_parallel_boosting import TAdditiveModel, fit

comptime N_ROWS = 60000
comptime N_ITERS = 25
comptime MAX_DEPTH = 6
comptime REPS = 3


def main() raises:
    var ctx = DeviceContext()
    var folds: List[Int] = [
        1, 1, 1, 12, 9, 15, 20, 32, 48, 64, 100, 127, 40, 55, 80, 110,
    ]
    var n_features = len(folds)
    var lay = build_layout(folds)

    var cindex = ctx.enqueue_create_buffer[DType.uint32](
        N_ROWS * lay.columns
    )
    ctx.enqueue_memset(cindex, UInt32(0))
    var hb = ctx.enqueue_create_host_buffer[DType.uint8](N_ROWS)
    var bins = ctx.enqueue_create_buffer[DType.uint8](N_ROWS)
    var col14 = List[Int]()
    var col3 = List[Int]()
    for f in range(n_features):
        ref cf = lay.features[f]
        for r in range(N_ROWS):
            var x = UInt32(r * 2654435761 + f * 40503 + 0x2545F491)
            x ^= x << 13
            x ^= x >> 17
            x ^= x << 5
            var v = Int(x % UInt32(folds[f]))
            if f == 14:
                col14.append(v)
            if f == 3:
                col3.append(v)
            hb.unsafe_ptr().unsafe_store(r, UInt8(v))
        ctx.enqueue_copy(dst_buf=bins, src_ptr=hb.unsafe_ptr())
        ctx.enqueue_function[write_compressed_index_kernel](
            Int32(Int(cf.offset) * N_ROWS), cf.mask, cf.shift,
            bins.unsafe_ptr(), Int32(N_ROWS), cindex.unsafe_ptr(),
            grid_dim=(N_ROWS + WRITE_BLOCK_SIZE - 1) // WRITE_BLOCK_SIZE,
            block_dim=WRITE_BLOCK_SIZE,
        )
        ctx.synchronize()

    var targets = ctx.enqueue_create_buffer[DType.float32](N_ROWS)
    var weights = ctx.enqueue_create_buffer[DType.float32](N_ROWS)
    var ht = ctx.enqueue_create_host_buffer[DType.float32](N_ROWS)
    var hw = ctx.enqueue_create_host_buffer[DType.float32](N_ROWS)
    for r in range(N_ROWS):
        var y = Float32(col14[r]) * 0.4 - Float32(col3[r]) * 1.5
        if col14[r] > 40 and col3[r] > 6:
            y += 5.0
        ht.unsafe_ptr().unsafe_store(r, y)
        hw.unsafe_ptr().unsafe_store(r, Float32(1.0))
    ctx.enqueue_copy(dst_buf=targets, src_ptr=ht.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=weights, src_ptr=hw.unsafe_ptr())
    ctx.synchronize()

    print(
        "fixture:", N_ROWS, "rows x", n_features, "features,",
        lay.columns, "cindex columns, depth", MAX_DEPTH, ",", N_ITERS,
        "iterations,", REPS, "interleaved reps",
    )

    var t_greedy = Float64(1e18)
    var t_pw = Float64(1e18)
    var last_g = Float64(0.0)
    var last_p = Float64(0.0)

    for rep in range(REPS):
        var mg = TAdditiveModel()
        var t0 = perf_counter_ns()
        var lg = fit(
            mg, ctx, N_ROWS, folds, MAX_DEPTH, cindex, targets, weights,
            False, N_ITERS, Float32(0.3), Float32(3.0), True,
        )
        ctx.synchronize()
        var dg = Float64(perf_counter_ns() - t0) / 1e6
        if dg < t_greedy:
            t_greedy = dg
        last_g = lg[len(lg) - 1]

        var mp = TAdditiveModel()
        t0 = perf_counter_ns()
        var lp = fit(
            mp, ctx, N_ROWS, folds, MAX_DEPTH, cindex, targets, weights,
            False, N_ITERS, Float32(0.3), Float32(3.0), True,
            use_pointwise_searcher=True,
        )
        ctx.synchronize()
        var dp = Float64(perf_counter_ns() - t0) / 1e6
        if dp < t_pw:
            t_pw = dp
        last_p = lp[len(lp) - 1]

        print(
            "  rep", rep, ": greedy", dg, "ms   pointwise", dp, "ms",
        )

    print()
    print("greedy-subsets (default) :", t_greedy, "ms  (best of", REPS, ")")
    print("pointwise                :", t_pw, "ms")
    print("ratio pointwise/greedy   :", t_pw / t_greedy)
    print("final loss, greedy       :", last_g)
    print("final loss, pointwise    :", last_p)
    if last_g != last_p:
        print(
            "  !! the two arms' final losses DIFFER -- the timing is not"
            " comparing the same computation"
        )
    else:
        print("  (identical final loss, so this is a like-for-like timing)")
