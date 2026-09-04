# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Deterministic fit of UMAP's differentiable distance curve."""

from std.math import exp, log, pow
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL


struct UMAPCurve(Copyable, Movable):
    var a: Float32
    var b: Float32

    def __init__(out self, a: Float32, b: Float32):
        self.a = a
        self.b = b


def _target(x: Float64, min_dist: Float64, spread: Float64) -> Float64:
    if x < min_dist:
        return Float64(1.0)
    return exp(-(x - min_dist) / spread)


def _loss(a: Float64, b: Float64, min_dist: Float64, spread: Float64) -> Float64:
    var loss = Float64(0.0)
    var i = 0
    while i < 300:
        var x = spread * Float64(3.0) * Float64(i) / Float64(299.0)
        var p = Float64(0.0) if i == 0 else pow(x, Float64(2.0) * b)
        var residual = Float64(1.0) / (Float64(1.0) + a * p) - (
            _target(x, min_dist, spread)
        )
        loss += residual * residual
        i += 1
    return loss


@no_inline
def fit_umap_curve(
    min_dist_in: Float32, spread_in: Float32
) raises -> UMAPCurve:
    """Fit `1/(1+a*x^(2b))` to UMAP's standard 300-point target.

    Levenberg–Marquardt starts from `(1,1)` and visits samples in ascending
    order using Float64. IDENTICAL performs exactly 64 proposals. FAST may
    return after an accepted update has max magnitude below 1e-10.
    """
    if min_dist_in < Float32(0.0) or not (spread_in > Float32(0.0)) or (
        min_dist_in > spread_in
    ):
        raise Error("UMAP curve requires 0 <= min_dist <= spread")
    var min_dist = Float64(min_dist_in)
    var spread = Float64(spread_in)
    var a = Float64(1.0)
    var b = Float64(1.0)
    var damping = Float64(1.0e-3)
    var current_loss = _loss(a, b, min_dist, spread)
    var iteration = 0
    while iteration < 64:
        var haa = Float64(0.0)
        var hab = Float64(0.0)
        var hbb = Float64(0.0)
        var ga = Float64(0.0)
        var gb = Float64(0.0)
        var i = 1
        while i < 300:
            var x = spread * Float64(3.0) * Float64(i) / Float64(299.0)
            var p = pow(x, Float64(2.0) * b)
            var denominator = Float64(1.0) + a * p
            var fitted = Float64(1.0) / denominator
            var residual = fitted - _target(x, min_dist, spread)
            var ja = -p / (denominator * denominator)
            var jb = -a * p * Float64(2.0) * log(x) / (
                denominator * denominator
            )
            haa += ja * ja
            hab += ja * jb
            hbb += jb * jb
            ga += ja * residual
            gb += jb * residual
            i += 1
        haa += damping
        hbb += damping
        var determinant = haa * hbb - hab * hab
        if not (determinant > Float64(0.0)):
            raise Error("UMAP curve fit normal equations are singular")
        var da = (-hbb * ga + hab * gb) / determinant
        var db = (hab * ga - haa * gb) / determinant
        var next_a = a + da
        var next_b = b + db
        var accepted = False
        if next_a > Float64(1.0e-12) and next_b > Float64(1.0e-12):
            var next_loss = _loss(next_a, next_b, min_dist, spread)
            if next_loss < current_loss:
                a = next_a
                b = next_b
                current_loss = next_loss
                damping *= Float64(0.5)
                accepted = True
            else:
                damping *= Float64(10.0)
        else:
            damping *= Float64(10.0)
        if damping > Float64(1.0e30):
            damping = Float64(1.0e30)
        comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
            if accepted:
                var magnitude = da if da >= Float64(0.0) else -da
                var mb = db if db >= Float64(0.0) else -db
                if mb > magnitude:
                    magnitude = mb
                if magnitude <= Float64(1.0e-10):
                    return UMAPCurve(Float32(a), Float32(b))
        iteration += 1
    if not (a > Float64(0.0)) or not (b > Float64(0.0)):
        raise Error("UMAP curve fit did not produce positive parameters")
    return UMAPCurve(Float32(a), Float32(b))
