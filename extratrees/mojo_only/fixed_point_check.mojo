"""Fixed point: the bound, the order-independence, and the comparator's width.

DEVIATION 135 is closed in favour of fixed point, so the properties it was
chosen FOR are the properties this file has to establish. Three of them, and
none is a spot check:

1. **Order independence.** The whole reason for integers. Sum a node's
   quantized labels in many different orders — forward, reverse, strided,
   pairwise-tree — and require BIT-IDENTICAL totals every time. A float
   accumulator fails this, and the check proves that too, so that the property
   is shown to be doing work rather than holding vacuously.
2. **Overflow is impossible, not unlikely.** The scale is derived from the
   whole dataset's magnitude sum, so the worst case is every row in one node
   with every label at its extreme. Build exactly that adversarial fixture and
   assert the accumulated total stays inside the slot.
3. **The comparator fits `Int128`.** This is the constraint root's fixed-point
   file does not have. The worst-case product is computed in `Int128` and
   compared against the bound at row counts spanning eight orders of
   magnitude, INCLUDING the counts where `accumulator_bits_for` starts giving
   resolution back.

Rule 8 throughout: hashed and scattered labels, per-cell comparison, and a
sabotage per mechanism.
"""

from std.testing import assert_equal, assert_true

from extratrees.mojo_only.fixed_point import (
    MIN_ACCUMULATOR_BITS,
    SLOT_BITS,
    accumulator_bits_for,
    ceil_log2,
    choose_scale,
    comparator_product_fits,
    compare_mse_proxy_exact,
    dequantize,
    max_representable,
    mse_proxy_exact,
    quantize,
)


def mix64(x: UInt64) -> UInt64:
    var h = x
    h ^= h >> 33
    h *= 0xFF51AFD7ED558CCD
    h ^= h >> 33
    h *= 0xC4CEB9FE1A85EC53
    h ^= h >> 33
    return h


def main() raises:
    var cells = 0

    # --- ceil_log2, exactly, at and around every power of two --------------
    assert_equal(ceil_log2(1), 0)
    assert_equal(ceil_log2(2), 1)
    assert_equal(ceil_log2(3), 2)
    assert_equal(ceil_log2(4), 2)
    assert_equal(ceil_log2(5), 3)
    # From k = 2 up: `p - 1` is not itself a power of two, so its ceiling is
    # still k. At k = 1 it is, and ceil_log2(1) == 0 -- the one place the
    # pattern does not hold, asserted separately above rather than papered
    # over with a loop bound that hides it.
    for k in range(2, 40):
        var p = 1 << k
        assert_equal(ceil_log2(p), k, "exact power of two")
        assert_equal(ceil_log2(p - 1), k, "just below")
        assert_equal(ceil_log2(p + 1), k + 1, "just above")
        cells += 3
    assert_equal(ceil_log2(1), 0, "and 2^1 - 1 == 1 is the exception")
    var raised = False
    try:
        _ = ceil_log2(0)
    except:
        raised = True
    assert_true(raised, "ceil_log2(0) must raise, not return a bound")
    cells += 6

    # --- the resolution schedule, worked out loud --------------------------
    print("[bits] accumulator bits by row count")
    for k in [10, 16, 20, 22, 23, 24, 25, 26]:
        var n = 1 << k
        var bits = accumulator_bits_for(n)
        print("    n = 2^" + String(k) + "  ->  b =", bits)
        assert_true(bits <= SLOT_BITS, "never above the slot cap")
        assert_true(bits >= MIN_ACCUMULATOR_BITS, "never below the floor")
        # The derived rule, restated independently: 2b + 3*ceil_log2(n) <= 126.
        assert_true(
            2 * bits + 3 * ceil_log2(n) <= 126,
            "the derived inequality must hold at n = 2^" + String(k),
        )
        cells += 3
    assert_equal(
        accumulator_bits_for(1 << 22), SLOT_BITS, "4.2M rows must cost nothing"
    )
    assert_true(
        accumulator_bits_for(1 << 26) < SLOT_BITS,
        "67M rows must cost resolution -- if this passes at the full slot the"
        " comparator has silently been allowed to overflow",
    )
    cells += 2

    # --- the comparator's product really fits, computed not asserted -------
    print("[width] worst-case comparator product against Int128")
    for k in range(4, 27):
        var n = 1 << k
        assert_true(
            comparator_product_fits(n),
            "the worst-case num*den must fit Int128 at n = 2^" + String(k),
        )
        cells += 1

    # --- order independence, the property fixed point exists for ------------
    # Hashed labels spanning both signs and several magnitudes.
    var n_rows = 4096
    var y = List[Float64]()
    var mag_sum = Float64(0.0)
    for r in range(n_rows):
        var h = mix64(UInt64(r) ^ 0xC0FFEE)
        var v = (Float64(Int(h % 2000001)) / 1000.0 - 1000.0) * (
            1.0 if (h >> 40) % 2 == 0 else 0.001
        )
        y.append(v)
        mag_sum += v if v >= 0.0 else -v
    var scale = choose_scale(mag_sum, n_rows)
    print("[scale] magnitude sum", mag_sum, " scale", scale)

    var q = List[Int64]()
    for r in range(n_rows):
        q.append(quantize(y[r], scale))

    # forward
    var s_fwd = Int64(0)
    for r in range(n_rows):
        s_fwd += q[r]
    # reverse
    var s_rev = Int64(0)
    for r in range(n_rows - 1, -1, -1):
        s_rev += q[r]
    # strided, several coprime strides
    var strides = [3, 5, 7, 11, 4095]
    for stride in strides:
        var s = Int64(0)
        var i = 0
        for _ in range(n_rows):
            s += q[i]
            i = (i + stride) % n_rows
        assert_equal(s, s_fwd, "a strided sum must give identical bits")
        cells += 1
    # pairwise tree, the shape a device reduction actually has
    var level = q.copy()
    while len(level) > 1:
        var nxt = List[Int64]()
        var i = 0
        while i + 1 < len(level):
            nxt.append(level[i] + level[i + 1])
            i += 2
        if i < len(level):
            nxt.append(level[i])
        level = nxt^
    assert_equal(s_fwd, s_rev, "forward and reverse must agree")
    assert_equal(level[0], s_fwd, "a pairwise tree reduction must agree")
    cells += 2

    # ... AND THE SAME TEST FAILS FOR FLOATS, so the property is not vacuous.
    var f_fwd = Float32(0.0)
    for r in range(n_rows):
        f_fwd += Float32(y[r])
    var f_rev = Float32(0.0)
    for r in range(n_rows - 1, -1, -1):
        f_rev += Float32(y[r])
    var f_tree = List[Float32]()
    for r in range(n_rows):
        f_tree.append(Float32(y[r]))
    while len(f_tree) > 1:
        var nxt = List[Float32]()
        var i = 0
        while i + 1 < len(f_tree):
            nxt.append(f_tree[i] + f_tree[i + 1])
            i += 2
        if i < len(f_tree):
            nxt.append(f_tree[i])
        f_tree = nxt^
    var float_orders_disagree = (
        f_fwd.to_bits() != f_rev.to_bits()
        or f_fwd.to_bits() != f_tree[0].to_bits()
    )
    print(
        "[control] float32 fwd/rev/tree:",
        f_fwd,
        f_rev,
        f_tree[0],
        " disagree:",
        float_orders_disagree,
    )
    assert_true(
        float_orders_disagree,
        "the float control MUST disagree across orders -- if it agrees, this"
        " fixture is too easy and the integer result proves nothing",
    )
    cells += 1

    # --- the bound holds for the WORST case, not the average ---------------
    # Every row at the extreme, all of them in one node: exactly the case the
    # whole-dataset bound is derived to survive.
    var extreme = Float64(1234.5)
    var worst_mag = extreme * Float64(n_rows)
    var worst_scale = choose_scale(worst_mag, n_rows)
    var worst_total = Int64(0)
    for _ in range(n_rows):
        worst_total += quantize(extreme, worst_scale)
    var bits = accumulator_bits_for(n_rows)
    var slot_limit = Int64(1) << Int64(bits)
    print(
        "[bound] worst-case total",
        worst_total,
        "against slot limit 2^" + String(bits),
        "=",
        slot_limit,
    )
    assert_true(
        worst_total < slot_limit,
        "the worst case must stay inside the slot -- an overflow here is"
        " garbage, not imprecision",
    )
    # and it must be USING most of the slot, or the scale is wastefully small
    assert_true(
        worst_total > slot_limit // 4,
        "the scale must actually spend the slot; a tiny scale passes the"
        " bound and throws away every small label",
    )
    # negative extreme too: truncation toward zero is asymmetric-looking and
    # a scale that only works for one sign is a real bug.
    var worst_neg = Int64(0)
    for _ in range(n_rows):
        worst_neg += quantize(-extreme, worst_scale)
    assert_equal(worst_neg, -worst_total, "the bound must be sign-symmetric")
    cells += 3

    # --- truncation toward zero, both signs, per cell -----------------------
    assert_equal(quantize(1.7, 1.0), Int64(1))
    assert_equal(quantize(-1.7, 1.0), Int64(-1), "toward zero, NOT floor")
    assert_equal(quantize(-0.9, 1.0), Int64(0))
    assert_equal(quantize(0.9, 1.0), Int64(0))
    cells += 4

    # --- the scale is a power of two, and a STEP function -------------------
    # Two magnitudes inside the same 2x band must give the SAME scale: that is
    # the property that pins the realization across runs and platforms.
    for k in range(-20, 21):
        var m = 1000.0 * (2.0 ** Float64(k))
        var s = choose_scale(m, n_rows)
        # exact power of two
        var t = s
        while t > 1.0:
            t /= 2.0
        while t < 1.0:
            t *= 2.0
        assert_equal(t, 1.0, "the scale must be an exact power of two")
        # and a 1e-9 wobble must not move it
        assert_equal(
            choose_scale(m * (1.0 + 1e-9), n_rows),
            s,
            "a tiny wobble in the magnitude must NOT re-roll the scale",
        )
        cells += 2
    assert_equal(choose_scale(0.0, n_rows), 1.0, "an all-zero target")
    cells += 1

    # --- round trip ---------------------------------------------------------
    var back = dequantize(s_fwd, scale)
    var direct = Float64(0.0)
    for r in range(n_rows):
        direct += y[r]
    var err = back - direct
    if err < 0.0:
        err = -err
    # Truncation loses at most one unit per row, by construction.
    var allowed = Float64(n_rows) / scale
    assert_true(
        err <= allowed,
        "the round trip must be inside the truncation budget: err "
        + String(err)
        + " vs "
        + String(allowed),
    )
    print("[roundtrip] error", err, "within budget", allowed)
    cells += 1
    _ = max_representable(scale, n_rows)

    # --- the exact MSE comparator ------------------------------------------
    # Analytic: two candidates whose proxies are hand-computable.
    #   a: sum_L=3, n_L=1, sum_R=-3, n_R=5  ->  9/1 + 9/5 = 54/5
    #   b: sum_L=-3, n_L=1, sum_R=3, n_R=5  ->  the same, EXACTLY
    var a = mse_proxy_exact(3, 1, -3, 5)
    var b = mse_proxy_exact(-3, 1, 3, 5)
    assert_equal(
        compare_mse_proxy_exact(a[0], a[1], b[0], b[1]),
        0,
        "mirrored sums must tie EXACTLY",
    )
    #   c: sum_L=4, n_L=1, sum_R=-3, n_R=5  ->  16 + 9/5 > 54/5
    var c = mse_proxy_exact(4, 1, -3, 5)
    assert_equal(compare_mse_proxy_exact(c[0], c[1], a[0], a[1]), 1)
    assert_equal(compare_mse_proxy_exact(a[0], a[1], c[0], c[1]), -1)
    cells += 3
    # an empty child is invalid and loses to anything valid
    var empty = mse_proxy_exact(0, 0, 7, 6)
    assert_equal(empty[1], Int128(0), "an empty child gives den == 0")
    assert_equal(compare_mse_proxy_exact(empty[0], empty[1], a[0], a[1]), -1)
    assert_equal(compare_mse_proxy_exact(a[0], a[1], empty[0], empty[1]), 1)
    assert_equal(
        compare_mse_proxy_exact(empty[0], empty[1], empty[0], empty[1]), 0
    )
    cells += 4

    # THE CASE THAT SEPARATES A REAL COMPARATOR FROM A NUMERATOR COMPARISON.
    # Two candidates in the SAME node necessarily have different n_left, hence
    # different denominators -- so a comparator that ignored `den` would be
    # wrong in general and right on every case above, all of which happened to
    # fix n_left = 1. Constructed so the numerators are EQUAL and the values
    # are not:
    #     d: n_L=3, n_R=3, sum_L=3, sum_R=-3  ->  9/3 + 9/3   =  6
    #        num = 9*3 + 9*3 = 54, den = 9
    #     e: n_L=1, n_R=5, sum_L=3, sum_R=-3  ->  9/1 + 9/5   = 10.8
    #        num = 9*5 + 9*1 = 54, den = 5
    # A numerator-only comparator calls these a tie; e beats d.
    var d = mse_proxy_exact(3, 3, -3, 3)
    var e = mse_proxy_exact(3, 1, -3, 5)
    assert_equal(d[0], e[0], "the fixture requires EQUAL numerators")
    assert_true(d[1] != e[1], "and different denominators")
    assert_equal(
        compare_mse_proxy_exact(e[0], e[1], d[0], d[1]),
        1,
        "e (10.8) must beat d (6) -- a comparator that ignores the"
        " denominator calls this a tie",
    )
    assert_equal(compare_mse_proxy_exact(d[0], d[1], e[0], e[1]), -1)
    cells += 4

    # And the order must agree with an INDEPENDENT Float64 evaluation on many
    # random same-node pairs, restricted to pairs whose float gap is wide
    # enough that float64 is itself trustworthy. This is the general form of
    # the case above: both n_left values vary, so both denominators do.
    var order_checked = 0
    for i in range(4000):
        var h = mix64(UInt64(i) ^ 0x0DDE71)
        var nl_a = 1 + Int(h % 4095)
        var nl_b = 1 + Int((h >> 16) % 4095)
        var nr_a = 4096 - nl_a
        var nr_b = 4096 - nl_b
        var sl_a = Int64(Int((h >> 32) % 200001)) - 100000
        var sl_b = Int64(Int((h >> 48) % 200001)) - 100000
        var sr_a = Int64(50000) - sl_a
        var sr_b = Int64(50000) - sl_b
        var pa = mse_proxy_exact(sl_a, nl_a, sr_a, nr_a)
        var pb = mse_proxy_exact(sl_b, nl_b, sr_b, nr_b)
        var va = Float64(sl_a) * Float64(sl_a) / Float64(nl_a) + Float64(
            sr_a
        ) * Float64(sr_a) / Float64(nr_a)
        var vb = Float64(sl_b) * Float64(sl_b) / Float64(nl_b) + Float64(
            sr_b
        ) * Float64(sr_b) / Float64(nr_b)
        var gap = va - vb
        if gap < 0.0:
            gap = -gap
        var mag = va if va > vb else vb
        if gap <= mag * 1e-9:
            continue  # too close for float64 to be an authority
        var want = 1 if va > vb else -1
        assert_equal(
            compare_mse_proxy_exact(pa[0], pa[1], pb[0], pb[1]),
            want,
            "the exact comparator disagreed with a float64 evaluation whose"
            " gap is far outside rounding",
        )
        assert_equal(
            compare_mse_proxy_exact(pb[0], pb[1], pa[0], pa[1]),
            -want,
            "the comparator must be antisymmetric",
        )
        order_checked += 2
        cells += 2
    print("[order] ", order_checked, "comparisons agree with a float64 tally")
    assert_true(
        order_checked > 6000,
        "too few pairs survived the float64-trustworthy filter for this to"
        " be evidence",
    )

    # THE ONE THAT MATTERS: pairs whose exact proxies differ but whose Float32
    # proxies collide. If any exist, a builder comparing on the float score
    # alone would pick by feature index instead of by score -- DEVIATION 145.
    var collisions = 0
    var checked = 0
    var total = 4096
    for i in range(3000):
        var h = mix64(UInt64(i) ^ 0xD15EA5E)
        var nl = 1 + Int(h % UInt64(total - 1))
        var nr = total - nl
        var sl = Int64(Int((h >> 20) % 1000000)) - 500000
        var sr = Int64(Int((h >> 42) % 1000000)) - 500000
        var p = mse_proxy_exact(sl, nl, sr, nr)
        var q2 = mse_proxy_exact(sl + 1, nl, sr, nr)
        var order = compare_mse_proxy_exact(p[0], p[1], q2[0], q2[1])
        if order == 0:
            continue
        var fp = Float32(sl) * Float32(sl) / Float32(nl) + Float32(sr) * Float32(
            sr
        ) / Float32(nr)
        var fq = Float32(sl + 1) * Float32(sl + 1) / Float32(nl) + Float32(
            sr
        ) * Float32(sr) / Float32(nr)
        if fp.to_bits() == fq.to_bits():
            collisions += 1
        checked += 1
    print(
        "[float collision] of",
        checked,
        "pairs the exact comparator separates,",
        collisions,
        "collide in Float32",
    )
    assert_true(
        collisions > 0,
        "if NO pair collides in Float32 then DEVIATION 145's premise is"
        " unsupported at this scale and the entry must be re-argued, not"
        " quietly kept",
    )
    cells += 1

    print("fixed_point: ", cells, "cells")
    print("fixed_point_check: PASS")
