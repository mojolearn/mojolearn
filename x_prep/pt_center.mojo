# SPDX-License-Identifier: Apache-2.0
"""Affine-equivalent power coordinates for homogeneous-sign columns.

v(x) = (g_lambda(x)-g_lambda(anchor))/t(anchor)**a. Standardizing v
is identical in exact arithmetic to standardizing g; the likelihood score
uses log(t(x)/t(anchor)), so its Jacobian correction cancels the scale.
"""
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, identical_log1p, ftz
from x_prep.prims import expf, logf

# OPEN repair of bc112b172/gap26-pt-score-quality: nearconstant score
# inputs rounded before compensation, fixed bracket missed lambda52/-63,
# and standardization materialized a saturated original transform. Use
# affine-equivalent centered log/power coordinates throughout; full quality
# and timing still owed. See docs/apple-fast/PT_SCORE_STABLE.md.
comptime PT_SCORE_STABLE = (GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator() and is_defined["MOJOLEARN_PT_SCORE_STABLE"]())


@always_inline
def center_log(x: Float32, anchor: Float32, kind: Float32, method: Int) -> Float32:
    var norm = anchor if method == 1 else Float32(1) + abs(anchor)
    var delta = (x - anchor) if kind > Float32(0) else (anchor - x)
    return ftz(identical_log1p(delta / norm))


@always_inline
def center_expm1(z: Float32) -> Float32:
    if abs(z) >= Float32(0.5):
        return expf(z) - Float32(1)
    var term = z
    var value = z
    for k in range(2, 14):
        term *= z / Float32(k)
        value += term
    return value


@always_inline
def center_from_log(lg: Float32, kind: Float32, lam: Float32) -> Float32:
    var a = lam if kind > Float32(0) else Float32(2) - lam
    var value = lg if a == Float32(0) else center_expm1(a * lg) / a
    return value if kind > Float32(0) else -value


@always_inline
def center_original(x: Float32, lam: Float32, method: Int) -> Float32:
    var positive = method == 1 or x >= Float32(0)
    var lg = logf(x) if method == 1 else ftz(identical_log1p(abs(x)))
    return center_from_log(lg, Float32(1) if positive else Float32(-1), lam)


@always_inline
def center_apply(x: Float32, anchor: Float32, kind: Float32, lam: Float32, method: Int) -> Float32:
    if method == 1 or (x >= Float32(0)) == (kind > Float32(0)):
        return center_from_log(center_log(x, anchor, kind, method), kind, lam)
    # An inference query may cross the training column's sign. Keep the
    # same affine map, using the other YJ branch for this query only.
    var a = lam if kind > Float32(0) else Float32(2) - lam
    var lg0 = ftz(identical_log1p(abs(anchor)))
    return (center_original(x, lam, method) - center_original(anchor, lam, method)) / expf(a * lg0)


@always_inline
def center_inverse(v: Float32, anchor: Float32, kind: Float32, lam: Float32, method: Int) -> Float32:
    var a = lam if kind > Float32(0) else Float32(2) - lam
    var signed_v = v if kind > Float32(0) else -v
    var lg = signed_v
    if a != Float32(0):
        lg = ftz(identical_log1p(a * signed_v)) / a
    var norm = anchor if method == 1 else Float32(1) + abs(anchor)
    var delta = norm * center_expm1(lg)
    var x = anchor + delta if kind > Float32(0) else anchor - delta
    if method == 1 or (x == x and (x >= Float32(0)) == (kind > Float32(0))):
        return x
    # Crossing-sign inverse uses the original YJ branch after undoing the
    # affine coordinate map. Same-sign training roundtrips stay centered.
    var lg0 = ftz(identical_log1p(abs(anchor)))
    var original = v * expf(a * lg0) + center_original(anchor, lam, method)
    var positive = original >= Float32(0)
    var b = lam if positive else Float32(2) - lam
    var sv = original if positive else -original
    var ilog = sv if b == Float32(0) else ftz(identical_log1p(b * sv)) / b
    var out = center_expm1(ilog)
    return out if positive else -out
