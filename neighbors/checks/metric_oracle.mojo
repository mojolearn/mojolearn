# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Host oracles for the six implemented distances, and for the distance weights.

NO UPSTREAM. Two SPELLINGS, not two imports, which is this tree's whole
oracle contract (`kde/checks/kde_oracle.mojo` says the same about itself):
if `metric_distance_kernel` and `oracle_metric_distance` ever disagree, one
of them is wrong and the gate says which cell. An oracle that CALLED the
kernel's helpers would agree with a broken kernel.

Three levels, deliberately:

  1. `oracle_metric_distance` -- FLOAT32, the same arithmetic in the same
     order through the same pinned primitives. Under IDENTICAL this must
     agree with the device BIT FOR BIT. It catches a wrong loop order, a
     wrong epilogue, a missed `ftz`, a norm taken on the wrong axis.
  2. `reference_metric_distance_f64` -- FLOAT64, the metric as
     MATHEMATICS, through the host libm. It agrees with neither of the
     above by construction: cosine here is a direct dot and two direct
     norms, not an expanded identity; Minkowski here is the host `pow`,
     not `exp(p log)`. It catches a wrong FORMULA, which level 1 cannot,
     because level 1 would reproduce a wrong formula faithfully.
  3. `oracle_distance_weights` -- the weight rule, spelled from
     scikit-learn's `_base.py:108-113` rather than from
     `host_distance_weights`.

`neighbors/checks/metric_check.mojo` runs all three against the device.
"""

from std.math import sqrt

from std.math import log

from checks.numerics import (
    ftz,
    identical_div,
    identical_log,
    identical_mul,
    identical_mul_add,
    identical_pow,
    identical_sqrt,
)
from core.row_norms import NORM_TPB
from neighbors.impl.distance.detail.distance_ops import (
    DIST_BRAY_CURTIS,
    DIST_CANBERRA,
    DIST_CORRELATION_EXPANDED,
    DIST_HAMMING_UNEXPANDED,
    DIST_INNER_PRODUCT,
    DIST_JENSEN_SHANNON,
    DIST_RUSSEL_RAO_EXPANDED,
    DIST_COSINE_EXPANDED,
    DIST_L1,
    DIST_L2_EXPANDED,
    DIST_L2_SQRT_EXPANDED,
    DIST_L2_SQRT_UNEXPANDED,
    DIST_L2_UNEXPANDED,
    DIST_LINF,
    DIST_LP_UNEXPANDED,
)


def oracle_row_norm(
    a: List[Float32], row: Int, d: Int, take_sqrt: Bool
) -> Float32:
    """`row_norm_kernel` / `cosine_row_norm_kernel` replayed on the host:
    `NORM_TPB` strided lane partials through `identical_mul_add`, then the
    halving tree `pinned_block_sum` performs, then `ftz`, then the optional
    clamp-and-root.

    The STRIDE AND THE TREE ARE THE POINT. A plain ascending sum would be a
    different float and the gate would report a difference that is the
    oracle's fault. This mirrors `kde/checks/kde_oracle.mojo::
    _host_row_norm_halving`, which replays the same kernel for the same
    reason.
    """
    var red = List[Float32]()
    for t in range(NORM_TPB):
        var acc = Float32(0.0)
        var col = t
        while col < d:
            var v = ftz(a[row * d + col])
            acc = ftz(identical_mul_add(v, v, acc))
            col += NORM_TPB
        red.append(acc)
    var step = NORM_TPB // 2
    while step > 0:
        for t in range(step):
            red[t] = red[t] + red[t + step]
        step //= 2
    var total = ftz(red[0])
    if not take_sqrt:
        return total
    if total <= Float32(0.0):
        total = Float32(0.0)
    return ftz(identical_sqrt(total))


def oracle_metric_distance(
    x: List[Float32],
    y: List[Float32],
    i: Int,
    j: Int,
    d: Int,
    metric: Int,
    xn: Float32,
    yn: Float32,
    metric_arg: Float32 = Float32(2.0),
) raises -> Float32:
    """One cell of `metric_distance_kernel`, spelled again.

    `xn` / `yn` are read only for the three expanded metrics and must be
    the SQUARED norms for the L2 pair and the TRUE L2 norms for cosine
    (`metric_norm_takes_sqrt`). Passing them the other way round is the
    sabotage `check_metric_cosine_norm_flag` performs on purpose.
    """
    var acc = Float32(0.0)

    if metric == DIST_L1:
        for f in range(d):
            acc = ftz(acc + abs(ftz(ftz(x[i * d + f]) - ftz(y[j * d + f]))))
        return acc

    if metric == DIST_LINF:
        for f in range(d):
            var diff = abs(ftz(ftz(x[i * d + f]) - ftz(y[j * d + f])))
            if diff > acc:  # row 39: strict `>`, operands non-negative
                acc = diff
        return acc

    if metric == DIST_L2_UNEXPANDED or metric == DIST_L2_SQRT_UNEXPANDED:
        for f in range(d):
            var diff = ftz(ftz(x[i * d + f]) - ftz(y[j * d + f]))
            acc = ftz(identical_mul_add(diff, diff, acc))
        if metric == DIST_L2_SQRT_UNEXPANDED:
            acc = ftz(identical_sqrt(acc))
        return acc

    if metric == DIST_LP_UNEXPANDED:
        for f in range(d):
            var diff = abs(ftz(ftz(x[i * d + f]) - ftz(y[j * d + f])))
            acc = ftz(acc + ftz(identical_pow(diff, metric_arg)))
        var one_over_p = ftz(identical_div(Float32(1.0), metric_arg))
        return ftz(identical_pow(acc, one_over_p))

    if (
        metric == DIST_COSINE_EXPANDED
        or metric == DIST_L2_EXPANDED
        or metric == DIST_L2_SQRT_EXPANDED
    ):
        for f in range(d):
            acc = ftz(
                identical_mul_add(ftz(x[i * d + f]), ftz(y[j * d + f]), acc)
            )
        if metric == DIST_COSINE_EXPANDED:
            var denom = ftz(identical_mul(ftz(xn), ftz(yn)))
            return ftz(Float32(1.0) - ftz(identical_div(acc, denom)))
        var dist = ftz(
            identical_mul_add(Float32(-2.0), acc, ftz(ftz(xn) + ftz(yn)))
        )
        if dist <= Float32(0.0):
            dist = Float32(0.0)
        if metric == DIST_L2_SQRT_EXPANDED:
            dist = ftz(identical_sqrt(dist))
        return dist

    # lane/neighbors-rest (2026-09-15): the seven metrics, spelled again
    # from the reference op files and DEVIATIONS 2898 to 2901, not from
    # `neighbors_rest_cell`.
    if metric == DIST_CORRELATION_EXPANDED:
        var sx = Float32(0.0)
        var sy = Float32(0.0)
        var sxx = Float32(0.0)
        var syy = Float32(0.0)
        for f in range(d):
            var a = ftz(x[i * d + f])
            var b = ftz(y[j * d + f])
            sx = ftz(sx + a)
            sy = ftz(sy + b)
            sxx = ftz(identical_mul_add(a, a, sxx))
            syy = ftz(identical_mul_add(b, b, syy))
            acc = ftz(identical_mul_add(a, b, acc))
        var kf = Float32(d)
        var numer = ftz(ftz(identical_mul(kf, acc)) - ftz(identical_mul(sx, sy)))
        var qd = ftz(ftz(identical_mul(kf, sxx)) - ftz(identical_mul(sx, sx)))
        var rd = ftz(ftz(identical_mul(kf, syy)) - ftz(identical_mul(sy, sy)))
        var root = ftz(identical_sqrt(ftz(identical_mul(qd, rd))))
        return ftz(Float32(1.0) - ftz(identical_div(numer, root)))
    if metric == DIST_BRAY_CURTIS:
        var den = Float32(0.0)
        for f in range(d):
            var a = ftz(x[i * d + f])
            var b = ftz(y[j * d + f])
            acc = ftz(acc + abs(ftz(a - b)))
            den = ftz(den + abs(ftz(a + b)))
        if den == Float32(0.0):
            if acc == Float32(0.0):
                return Float32(0.0)
            return Float32(1.0) / Float32(0.0)
        return ftz(identical_div(acc, den))
    var one_over_k = ftz(identical_div(Float32(1.0), Float32(d)))
    for f in range(d):
        var a = ftz(x[i * d + f])
        var b = ftz(y[j * d + f])
        if metric == DIST_CANBERRA:
            var add = ftz(abs(a) + abs(b))
            if add != Float32(0.0):
                acc = ftz(acc + ftz(identical_div(abs(ftz(a - b)), add)))
        elif metric == DIST_JENSEN_SHANNON:
            var m = ftz(identical_mul(Float32(0.5), ftz(a + b)))
            var lm = ftz(identical_log(m)) if m != Float32(0.0) else Float32(0.0)
            var la = ftz(identical_log(a)) if a != Float32(0.0) else Float32(0.0)
            var lb = ftz(identical_log(b)) if b != Float32(0.0) else Float32(0.0)
            var ta = ftz(identical_mul(-a, ftz(lm - la)))
            var tb = ftz(identical_mul(-b, ftz(lm - lb)))
            acc = ftz(acc + ftz(ta + tb))
        elif metric == DIST_HAMMING_UNEXPANDED:
            if a != b:
                acc = ftz(acc + Float32(1.0))
        elif metric == DIST_RUSSEL_RAO_EXPANDED:
            if a != Float32(0.0) and b != Float32(0.0):
                acc = ftz(acc + Float32(1.0))
        elif metric == DIST_INNER_PRODUCT:
            acc = ftz(identical_mul_add(a, b, acc))
        else:
            raise Error(
                "oracle_metric_distance: unknown DistanceType " + String(metric)
            )
    if metric == DIST_CANBERRA:
        return acc
    if metric == DIST_JENSEN_SHANNON:
        var half = ftz(identical_mul(Float32(0.5), acc))
        if half <= Float32(0.0):
            return Float32(0.0)
        return ftz(identical_sqrt(half))
    if metric == DIST_HAMMING_UNEXPANDED:
        return ftz(identical_mul(acc, one_over_k))
    if metric == DIST_RUSSEL_RAO_EXPANDED:
        return ftz(identical_mul(ftz(Float32(d) - acc), one_over_k))
    if metric == DIST_INNER_PRODUCT:
        # the STORED value, DEVIATION 2901
        return ftz(Float32(0.0) - acc)

    raise Error(
        "oracle_metric_distance: unknown DistanceType " + String(metric)
    )


def reference_metric_distance_f64(
    x: List[Float32],
    y: List[Float32],
    i: Int,
    j: Int,
    d: Int,
    metric: Int,
    metric_arg: Float64 = 2.0,
) raises -> Float64:
    """The metric as MATHEMATICS in float64, computed a third way.

    Deliberately NOT the expanded identity and NOT `exp(p log z)`:

      - cosine is `1 - dot/(|x| |y|)` with the norms taken directly;
      - the L2 pair is the plain sum of squared DIFFERENCES, so it
        cross-checks the expansion rather than repeating it;
      - Minkowski is the host libm `pow`.

    So a bug that lives in the expansion, or in the `exp(p log)`
    composition, shows up HERE and cannot show up in
    `oracle_metric_distance`, which would reproduce it.
    """
    if metric == DIST_COSINE_EXPANDED:
        var dot = 0.0
        var xn = 0.0
        var yn = 0.0
        for f in range(d):
            var a = Float64(x[i * d + f])
            var b = Float64(y[j * d + f])
            dot += a * b
            xn += a * a
            yn += b * b
        return 1.0 - dot / (sqrt(xn) * sqrt(yn))
    # lane/neighbors-rest: the seven as mathematics. Correlation CENTERS
    # first (scipy's form), not the expanded identity; jensen-shannon uses
    # the host libm log; inner product is the stored negation.
    var kd = Float64(d)
    if metric == DIST_CORRELATION_EXPANDED:
        var mx = 0.0
        var my = 0.0
        for f in range(d):
            mx += Float64(x[i * d + f])
            my += Float64(y[j * d + f])
        mx /= kd
        my /= kd
        var num = 0.0
        var vx = 0.0
        var vy = 0.0
        for f in range(d):
            var a = Float64(x[i * d + f]) - mx
            var b = Float64(y[j * d + f]) - my
            num += a * b
            vx += a * a
            vy += b * b
        return 1.0 - num / sqrt(vx * vy)
    if (
        metric == DIST_CANBERRA or metric == DIST_BRAY_CURTIS
        or metric == DIST_JENSEN_SHANNON or metric == DIST_HAMMING_UNEXPANDED
        or metric == DIST_RUSSEL_RAO_EXPANDED or metric == DIST_INNER_PRODUCT
    ):
        var s = 0.0
        var den = 0.0
        for f in range(d):
            var a = Float64(x[i * d + f])
            var b = Float64(y[j * d + f])
            if metric == DIST_CANBERRA:
                if abs(a) + abs(b) > 0.0:
                    s += abs(a - b) / (abs(a) + abs(b))
            elif metric == DIST_BRAY_CURTIS:
                s += abs(a - b)
                den += abs(a + b)
            elif metric == DIST_JENSEN_SHANNON:
                var m = 0.5 * (a + b)
                if a > 0.0:
                    s += a * log(a / m)
                if b > 0.0:
                    s += b * log(b / m)
            elif metric == DIST_HAMMING_UNEXPANDED:
                if a != b:
                    s += 1.0
            elif metric == DIST_RUSSEL_RAO_EXPANDED:
                if a != 0.0 and b != 0.0:
                    s += 1.0
            else:
                s += a * b
        if metric == DIST_BRAY_CURTIS:
            return s / den
        if metric == DIST_JENSEN_SHANNON:
            return sqrt(0.5 * s) if s > 0.0 else 0.0
        if metric == DIST_HAMMING_UNEXPANDED:
            return s / kd
        if metric == DIST_RUSSEL_RAO_EXPANDED:
            return (kd - s) / kd
        if metric == DIST_INNER_PRODUCT:
            return -s
        return s
    var acc = 0.0
    if metric == DIST_LP_UNEXPANDED:
        for f in range(d):
            var df = abs(
                Float64(x[i * d + f]) - Float64(y[j * d + f])
            )
            acc += df**metric_arg
        return acc ** (1.0 / metric_arg)
    for f in range(d):
        var diff = Float64(x[i * d + f]) - Float64(y[j * d + f])
        if metric == DIST_L1:
            acc += abs(diff)
        elif metric == DIST_LINF:
            if abs(diff) > acc:
                acc = abs(diff)
        else:
            acc += diff * diff
    if metric == DIST_L2_SQRT_UNEXPANDED or metric == DIST_L2_SQRT_EXPANDED:
        return sqrt(acc)
    return acc


def oracle_distance_weights(
    dist: List[Float32], n_queries: Int, k: Int
) -> List[Float32]:
    """`sklearn/neighbors/_base.py:108-113`, spelled from THEIR file rather
    than from `host_distance_weights`.

        dist = 1.0 / dist
        inf_mask = np.isinf(dist)
        inf_row  = np.any(inf_mask, axis=1)
        dist[inf_row] = inf_mask[inf_row]

    Written with a plain `/` and a plain `==` so that it is a second
    arithmetic and not a copy: `host_distance_weights` goes through
    `identical_div` and `ftz`, and if the two disagree on a normal input
    the gate has found a real difference between the pinned division and
    the plain one, which is worth knowing.
    """
    var out = List[Float32](capacity=n_queries * k)
    var pos_inf = Float32(1.0) / Float32(0.0)
    for i in range(n_queries):
        var row = List[Float32]()
        var any_inf = False
        for j in range(k):
            var v = Float32(1.0) / dist[i * k + j]
            if v == pos_inf or v == -pos_inf:
                any_inf = True
            row.append(v)
        for j in range(k):
            if any_inf:
                out.append(
                    Float32(1.0) if row[j] == pos_inf else Float32(0.0)
                )
            else:
                out.append(row[j])
    return out^
