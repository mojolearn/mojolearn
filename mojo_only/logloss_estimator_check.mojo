"""The device estimation stage against a float64 host simulation.

    pixi run check-logloss-estimator

`BinOptimizedOracle` + `newton_like_walker_estimate` run the full Logloss
leaves estimation on device -- shift-add MoveTo, fused der/der2 kernel,
per-bin partition reduces, Newton with AnyImprovement backtracking, ten
iterations, their defaults (lambda from l2, MinLeafWeight 1e-20). The
reference is an INDEPENDENT reimplementation in this file: float64
derivatives and reduces, the same control flow, the same Float32 point
and cursor storage the device uses (so storage rounding is shared and
the comparison isolates the device path's arithmetic).

Seven leaves with hashed uneven sizes, ONE DELIBERATELY EMPTY (leaf 3):
its weight sum is 0 < MinLeafWeight, so RegularizeImpl must pin it to
exactly 0.0 through every iteration -- the empty-leaf arm of the depth-8
regime where empty leaves are the common case. Both weight modes run;
the weighted arm also exercises the WeightsCpu device reduce.

Proven to have teeth, measured 2026-08-20: flipping the simulation's
class assignment fails all 12 non-empty leaf comparisons; truncating the
simulation to ONE Newton iteration also fails all 12, which is the proof
that the device side really runs all ten. A lambda perturbation (1.0 ->
1.05) does NOT trip it, and that is mathematics rather than a gap: their
lambda DAMPS THE NEWTON STEP (H + lambda in the direction), it is not
ridge on the objective -- that is the separate AddRidgeToTargetFunction,
off by default -- so ten converged iterations end at the same zero-
gradient point under either value. All probes restored, green before
this landed.
"""
from max.gpu.host import DeviceContext
from std.math import exp, log

from gbdt.methods.leaves_estimation.descent_helpers import (
    newton_like_walker_estimate,
)
from gbdt.methods.leaves_estimation.pointwise_oracle import (
    make_bin_optimized_oracle,
)
from gbdt.options.catboost_options import (
    LEAF_ESTIMATION_NEWTON,
)
from gbdt.targets.kernel.pointwise_targets import (
    OBJECTIVE_LOGLOSS,
)
from gbdt.methods.leaves_estimation.step_estimator import (
    BACKTRACKING_ANY_IMPROVEMENT,
)


def splitmix(x: UInt64) -> UInt64:
    var z = x + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def frac(i: Int, salt: UInt64) -> Float64:
    return Float64(splitmix(UInt64(i) * UInt64(2654435761) + salt) >> 11) * (
        1.0 / 9007199254740992.0
    )


comptime LEAVES = 7
comptime LAMBDA = 1.0
comptime ITERATIONS = 10


def _sim_eval(
    targets: List[Float32],
    weights: List[Float32],
    cursor: List[Float32],
    leaf_of: List[Int],
    has_weights: Bool,
    mut value: Float64,
    mut grad: List[Float64],
    mut hess: List[Float64],
):
    value = 0.0
    grad.clear()
    hess.clear()
    for _ in range(LEAVES):
        grad.append(0.0)
        hess.append(LAMBDA)
    for i in range(len(targets)):
        var v = Float64(cursor[i])
        var w = Float64(weights[i]) if has_weights else 1.0
        var ev = exp(v)
        var p = ev / (1.0 + ev)
        if p > 1.0 - 1e-40:
            p = 1.0 - 1e-40
        if p < 1e-40:
            p = 1e-40
        var c = 1.0 if Float64(targets[i]) > 0.5 else 0.0
        grad[leaf_of[i]] += w * (c - p)
        hess[leaf_of[i]] += w * p * (1.0 - p)
        value += w * (c * v - log(1.0 + ev))


def _sim_direction(
    grad: List[Float64], hess: List[Float64]
) -> List[Float32]:
    var d = List[Float32]()
    for i in range(LEAVES):
        if hess[i] > 0:
            d.append(Float32(grad[i] / (hess[i] + 1e-20)))
        else:
            d.append(Float32(0.0))
    return d^


def sim_estimate(
    targets: List[Float32],
    weights: List[Float32],
    cursor0: List[Float32],
    leaf_of: List[Int],
    leaf_sizes: List[Int],
    has_weights: Bool,
) -> List[Float32]:
    """Float64 reimplementation of oracle + walker, same control flow."""
    var n = len(targets)
    var cursor = cursor0.copy()
    var wsum = List[Float64]()
    for _ in range(LEAVES):
        wsum.append(0.0)
    for i in range(n):
        wsum[leaf_of[i]] += Float64(weights[i]) if has_weights else 1.0

    var cur_value = Float64(0.0)
    var cur_grad = List[Float64]()
    var cur_hess = List[Float64]()
    _sim_eval(
        targets, weights, cursor, leaf_of, has_weights,
        cur_value, cur_grad, cur_hess,
    )
    var direction = _sim_direction(cur_grad, cur_hess)

    var cur_point = List[Float32]()
    for _ in range(LEAVES):
        cur_point.append(Float32(0.0))

    var updated = False
    var iteration = 0
    while iteration < ITERATIONS:
        var round_value = cur_value
        var step = Float64(1.0)
        var accepted = False
        while iteration < ITERATIONS or ((not updated) and iteration < 100):
            var next_point = List[Float32]()
            for i in range(LEAVES):
                next_point.append(
                    Float32(
                        Float64(cur_point[i]) + step * Float64(direction[i])
                    )
                )
            for leaf in range(LEAVES):
                if wsum[leaf] < 1e-20:
                    next_point[leaf] = Float32(0.0)
            # move the cursor by the DELTA, like MoveTo does
            var shift = List[Float32]()
            for i in range(LEAVES):
                shift.append(next_point[i] - cur_point[i])
            for i in range(n):
                cursor[i] = cursor[i] + shift[leaf_of[i]]
            var next_value = Float64(0.0)
            var next_grad = List[Float64]()
            var next_hess = List[Float64]()
            _sim_eval(
                targets, weights, cursor, leaf_of, has_weights,
                next_value, next_grad, next_hess,
            )
            if round_value <= next_value:
                cur_point = next_point.copy()
                cur_value = next_value
                cur_grad = next_grad.copy()
                cur_hess = next_hess.copy()
                direction = _sim_direction(cur_grad, cur_hess)
                iteration += 1
                updated = True
                accepted = True
                break
            # rejected: walk the cursor BACK, like the next MoveTo's
            # delta from an unchanged current point does on device
            for i in range(n):
                cursor[i] = cursor[i] - shift[leaf_of[i]]
            iteration += 1
            step /= 2
        if not accepted:
            break
    return cur_point^


def main() raises:
    var ctx = DeviceContext()
    var failures = 0

    # hashed uneven leaf sizes, leaf 3 EMPTY
    var leaf_sizes = List[Int]()
    var n = 0
    for leaf in range(LEAVES):
        var sz = 0
        if leaf != 3:
            sz = 200 + Int(frac(leaf, UInt64(7)) * 600.0)
        leaf_sizes.append(sz)
        n += sz

    var leaf_of = List[Int]()
    var targets = List[Float32]()
    var weights = List[Float32]()
    var cursor0 = List[Float32]()
    for leaf in range(LEAVES):
        for _ in range(leaf_sizes[leaf]):
            var i = len(leaf_of)
            leaf_of.append(leaf)
            # leaf-dependent class balance so leaves want DIFFERENT values
            targets.append(
                Float32(1.0) if frac(i, UInt64(13))
                > (0.15 + 0.1 * Float64(leaf)) else Float32(0.0)
            )
            weights.append(Float32(0.5 + 2.0 * frac(i, UInt64(29))))
            cursor0.append(Float32((frac(i, UInt64(31)) - 0.5) * 4.0))

    for hw in range(2):
        var has_weights = hw == 1

        var d_target = ctx.enqueue_create_buffer[DType.float32](n)
        var d_weights = ctx.enqueue_create_buffer[DType.float32](n)
        var d_cursor = ctx.enqueue_create_buffer[DType.float32](n)
        var d_p_off = ctx.enqueue_create_buffer[DType.uint32](LEAVES)
        var d_p_sz = ctx.enqueue_create_buffer[DType.uint32](LEAVES)
        var h_f = ctx.enqueue_create_host_buffer[DType.float32](n)
        var h_u = ctx.enqueue_create_host_buffer[DType.uint32](LEAVES)
        ctx.synchronize()

        for i in range(n):
            h_f.unsafe_ptr().unsafe_store(i, targets[i])
        ctx.enqueue_copy(dst_buf=d_target, src_ptr=h_f.unsafe_ptr())
        ctx.synchronize()
        for i in range(n):
            h_f.unsafe_ptr().unsafe_store(i, weights[i])
        ctx.enqueue_copy(dst_buf=d_weights, src_ptr=h_f.unsafe_ptr())
        ctx.synchronize()
        for i in range(n):
            h_f.unsafe_ptr().unsafe_store(i, cursor0[i])
        ctx.enqueue_copy(dst_buf=d_cursor, src_ptr=h_f.unsafe_ptr())
        var off = 0
        for leaf in range(LEAVES):
            h_u.unsafe_ptr().unsafe_store(leaf, UInt32(off))
            off += leaf_sizes[leaf]
        ctx.enqueue_copy(dst_buf=d_p_off, src_ptr=h_u.unsafe_ptr())
        ctx.synchronize()
        for leaf in range(LEAVES):
            h_u.unsafe_ptr().unsafe_store(leaf, UInt32(leaf_sizes[leaf]))
        ctx.enqueue_copy(dst_buf=d_p_sz, src_ptr=h_u.unsafe_ptr())
        ctx.synchronize()

        var oracle = make_bin_optimized_oracle(
            ctx, n, LEAVES, leaf_sizes,
            d_target^, d_weights^, d_cursor^, d_p_off^, d_p_sz^,
            # `has_border=True` became `objective=OBJECTIVE_LOGLOSS`
            # plus the kernel's one alpha slot on 2026-08-21, when the
            # oracle stopped being cross-entropy-only. Logloss takes no
            # alpha, so it is their declared 0 (`pointwise_target_impl.h
            # :364`), and the estimation method is Newton, which is what
            # this check's `newton_like_walker_estimate` exercises.
            has_weights, OBJECTIVE_LOGLOSS, Float32(0.0), Float32(0.5),
            Float32(0.5),
            LAMBDA, -1, LEAF_ESTIMATION_NEWTON,
        )
        var _not_pd = 0
        var got = newton_like_walker_estimate(
            oracle, ITERATIONS, BACKTRACKING_ANY_IMPROVEMENT,
            List[Float32](), _not_pd,
        )
        var want = sim_estimate(
            targets, weights, cursor0, leaf_of, leaf_sizes, has_weights
        )

        var max_diff = Float64(0.0)
        for leaf in range(LEAVES):
            var diff = abs(Float64(got[leaf]) - Float64(want[leaf]))
            var scale = abs(Float64(want[leaf])) + 1.0
            if diff / scale > max_diff:
                max_diff = diff / scale
            if diff / scale > 5e-4:
                print("mode", hw, "leaf", leaf, "device", got[leaf],
                      "sim", want[leaf])
                failures += 1
        if got[3] != Float32(0.0):
            print("mode", hw, "EMPTY leaf 3 not regularized to zero:",
                  got[3])
            failures += 1
        print("mode", hw, "max rel diff", max_diff)

    if failures != 0:
        raise Error(
            "logloss estimator check FAILED with " + String(failures)
        )
    print("logloss estimator check: device oracle+walker matches the "
          "float64 simulation in both weight modes; empty leaf pinned at 0")
