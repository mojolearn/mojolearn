"""What the NeedEstimation arm COSTS per tree, priced by same-arm subtraction.

gbm-bench runs Logloss on five of its six datasets, and `fraud` (285k rows)
sits below our crossover, where FIXED per-tree cost decides the row
(`bench/results/PERF_2026-08-20_fixed-cost.md`: ours 9.43 ms + 18.5 us/1k
rows on the RMSE arm). The Logloss path adds a whole stage RMSE never runs:
three gathers, the p_off export sync, the oracle's buffers, and a Newton
walker whose EVERY iteration is a MoveTo launch plus a fused-eval launch
plus a partition-stats reduce plus a BLOCKING host readback
(`doc_parallel_boosting.mojo` estimation block, `pointwise_oracle.mojo`).
On Metal a blocking readback is the expensive syllable, so ten iterations
of their `catboost_options.cpp:157-164` default could plausibly DOUBLE the
fixed cost. Or cost nothing that matters. This file prices it; nobody
reasons about it.

Method: one process, one context, alternated arms (this box drifts 1.7x in
twenty minutes -- `PERF_2026-08-20_fixed-cost.md` -- so nothing here is an
A/B across time). Per arm, per rep: T(2 trees) and T(2+N trees) back to
back; (T2N - T2)/N is ms/tree with quantization, borders, pool setup and
kernel compilation SUBTRACTED OUT by the same-arm difference. Arms:

    rmse       objective RMSE       -- no estimation stage at all
    ll10       Logloss, iters 10    -- their default
    ll1        Logloss, iters 1     -- the Iterations==1 shortcut

Both at depth 6 (the fixed-cost fixture's depth) AND depth 8 (the depth
gbm-bench PINS -- `algorithms.py:172` -- so the row the predictions need).

    ll1  - rmse          = the stage's entry fee (gathers, offsets export,
                           oracle setup, ONE Newton iteration, apply)
    (ll10 - ll1) / 9     = one Newton iteration (eval + readback + move)

This is a SELF-diagnosis on our own arm only; no CatBoost process runs and
no scoreboard row comes from here. Comparison windows stay the user's.
"""

from std.time import perf_counter_ns

from std.math import exp
from max.gpu.host import DeviceContext

from gbdt.train import train


def splitmix(x: UInt64) -> UInt64:
    var z = x + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D268D24F605EB8)
    return z ^ (z >> 31)


def frac(i: Int, salt: UInt64) -> Float64:
    var h = splitmix(UInt64(i) * UInt64(2654435761) + salt * UInt64(0x9E37))
    return Float64(h >> 11) * (1.0 / 9007199254740992.0)


comptime N_ROWS = 50000
comptime N_FEATURES = 100
comptime WARM_TREES = 2
comptime BENCH_TREES = 20
comptime REPS = 3


def run_arm(
    ctx: DeviceContext,
    x: List[Float32],
    y: List[Float32],
    loss: String,
    iters: Int,
    n_estimators: Int,
    depth: Int,
) raises -> Float64:
    var t0 = perf_counter_ns()
    var tm = train(
        ctx, x, y, N_ROWS, N_FEATURES,
        border_count=254, n_estimators=n_estimators, max_depth=depth,
        learning_rate=Float32(0.1), l2_leaf_reg=Float32(1.0),
        loss=loss, leaf_estimation_iterations=iters,
    )
    var t1 = perf_counter_ns()
    # Touch the result so nothing is dead-code eliminated.
    if len(tm.losses) != n_estimators:
        raise Error("arm trained a wrong tree count")
    return Float64(t1 - t0) / 1e6


def main() raises:
    var ctx = DeviceContext()

    var x = List[Float32]()
    for feat in range(N_FEATURES):
        for r in range(N_ROWS):
            x.append(Float32((frac(r * N_FEATURES + feat, UInt64(5)) - 0.5) * 2.0))
    var y = List[Float32]()
    for r in range(N_ROWS):
        var x0 = Float64(x[0 * N_ROWS + r])
        var x3 = Float64(x[3 * N_ROWS + r])
        var p = 1.0 / (1.0 + exp(-(3.0 * x0 - 2.5 * x3)))
        y.append(Float32(1.0) if frac(r, UInt64(77)) < p else Float32(0.0))

    var depths: List[Int] = [6, 8]
    for d in range(len(depths)):
        var depth = depths[d]
        # Compile + first-touch warmup for every arm before any timing.
        _ = run_arm(ctx, x, y, "RMSE", -1, WARM_TREES, depth)
        _ = run_arm(ctx, x, y, "Logloss", 10, WARM_TREES, depth)
        _ = run_arm(ctx, x, y, "Logloss", 1, WARM_TREES, depth)

        print("depth", depth, ": rep arm ms/tree, (T", WARM_TREES + BENCH_TREES, "- T", WARM_TREES, ")/", BENCH_TREES)
        var per_tree = List[List[Float64]]()
        for _ in range(3):
            per_tree.append(List[Float64]())
        for rep in range(REPS):
            # Alternate arms inside the rep so drift hits all three alike.
            for arm in range(3):
                var loss: String
                var iters: Int
                var name: String
                if arm == 0:
                    loss = "RMSE"; iters = -1; name = "rmse"
                elif arm == 1:
                    loss = "Logloss"; iters = 10; name = "ll10"
                else:
                    loss = "Logloss"; iters = 1; name = "ll1 "
                var t_short = run_arm(ctx, x, y, loss, iters, WARM_TREES, depth)
                var t_long = run_arm(ctx, x, y, loss, iters, WARM_TREES + BENCH_TREES, depth)
                var ms = (t_long - t_short) / Float64(BENCH_TREES)
                per_tree[arm].append(ms)
                print(rep, " ", name, " ", ms)

        # Median of reps per arm.
        var med = List[Float64]()
        for arm in range(3):
            var a = per_tree[arm][0]
            var b = per_tree[arm][1]
            var c = per_tree[arm][2]
            var lo = min(a, min(b, c))
            var hi = max(a, max(b, c))
            med.append(a + b + c - lo - hi)
        print("")
        print("depth", depth, "median ms/tree   rmse", med[0], "  ll10", med[1], "  ll1", med[2])
        print("  stage entry fee (ll1 - rmse):     ", med[2] - med[0], "ms/tree")
        print("  per Newton iteration ((ll10-ll1)/9):", (med[1] - med[2]) / 9.0, "ms")
        print("  full default bill (ll10 - rmse):  ", med[1] - med[0], "ms/tree")
        print("")
