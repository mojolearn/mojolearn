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
def same_side(x: Float32, kind: Float32, method: Int) -> Bool:
    """x lies on the training column's YJ branch. Zero belongs to both:
    psi(0) = 0 on either branch, so a nonpositive column keeps it centered."""
    return method == 1 or (x >= Float32(0) if kind > Float32(0) else x <= Float32(0))


@always_inline
def cross_scaled(x: Float32, anchor: Float32, kind: Float32, lam: Float32) -> Float32:
    """(psi(x) - psi(anchor)) / t(anchor)**a for a YJ query on the other
    branch (x and anchor of opposite signs; never Box-Cox). With
    L0 = log t(anchor) and s = exp(-a L0):
      psi(anchor) * s = kind * (1 - s) / a = -kind * expm1(-a L0) / a (kind * L0 at a = 0),
      psi(x) * s      = sx * (exp(b L - a L0) - s) / b,  b the query branch,
    each exponent formed before exp so inf * 0 never occurs."""
    var a = lam if kind > Float32(0) else Float32(2) - lam
    var l0 = ftz(identical_log1p(abs(anchor)))
    var positive = x >= Float32(0)
    var b = lam if positive else Float32(2) - lam
    var lx = ftz(identical_log1p(abs(x)))
    var s = expf(-a * l0)
    var px: Float32
    if b == Float32(0):
        px = lx * s
    else:
        px = (expf(b * lx - a * l0) - s) / b
    if not positive:
        px = -px
    var pa = kind * l0 if a == Float32(0) else -kind * center_expm1(-a * l0) / a
    return px - pa


@always_inline
def center_apply(x: Float32, anchor: Float32, kind: Float32, lam: Float32, method: Int) -> Float32:
    if same_side(x, kind, method):
        return center_from_log(center_log(x, anchor, kind, method), kind, lam)
    # An inference query may cross the training column's sign. Keep the
    # same affine map, using the other YJ branch for this query only.
    return cross_scaled(x, anchor, kind, lam)


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
    if x == x and same_side(x, kind, method):
        return x
    if method == 1:
        return x  # Box-Cox: outside the domain stays NaN, as scipy inv_boxcox
    # Crossing-sign inverse: undo the affine map to psi, then the original YJ
    # branch of psi's sign (sklearn's _yeo_johnson_inverse_transform).
    var l0 = ftz(identical_log1p(abs(anchor)))
    var pa = kind * l0 if a == Float32(0) else -kind * center_expm1(-a * l0) / a
    var original = (v + pa) * expf(a * l0)
    var positive = original >= Float32(0)
    var b = lam if positive else Float32(2) - lam
    var sv = original if positive else -original
    var ilog = sv if b == Float32(0) else ftz(identical_log1p(b * sv)) / b
    var out = center_expm1(ilog)
    return out if positive else -out
