# SPDX-License-Identifier: Apache-2.0
"""Wide full-SVD: spectrum, null-space orthogonality and optimal residuals."""
from max.gpu.host import DeviceContext
from std.memory import bitcast
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from decomposition.checks.svd_full_check import _fit_full, _make_x, _centered_f64, _ref
from decomposition.impl.linalg.detail.svd_full import pca_full_validate


def check(m: Int, n: Int, kind: Int) raises:
    var xs = _make_x(m, n, 9183, 1.0, -1, 1.0)
    if kind != 0:
        for i in range(m):
            for j in range(n):
                xs[i*n+j] = Float32(7.5) if kind == 1 else Float32((i % 3 - 1) * (j % 5 - 2))
    var xc = _centered_f64(xs, m, n)
    var ctx = DeviceContext()
    var full = _fit_full(ctx, xs, m, n, m)
    var trunc = _fit_full(ctx, xs, m, n, 1)
    if len(full.components) != m*n:
        raise Error("wide full component shape")
    var orth = Float64(0.0)
    for a in range(m):
        for b in range(m):
            var dot = Float64(0.0)
            for j in range(n):
                dot += full.components[a*n+j] * full.components[b*n+j]
            orth = max(orth, abs(dot - (1.0 if a == b else 0.0)))
    if orth > 3e-5:
        raise Error("wide basis is not orthonormal: " + String(orth))
    var energy = Float64(0.0)
    var residual = Float64(0.0)
    var truncated_residual = Float64(0.0)
    for i in range(m):
        var projected = List[Float64]()
        for c in range(m):
            var v = Float64(0.0)
            for j in range(n):
                v += xc[i*n+j] * full.components[c*n+j]
            projected.append(v)
        for j in range(n):
            var rec = Float64(0.0)
            for c in range(m):
                rec += projected[c] * full.components[c*n+j]
            var d = xc[i*n+j] - rec
            residual += d*d
            d = xc[i*n+j] - projected[0] * full.components[j]
            truncated_residual += d*d
            energy += xc[i*n+j] * xc[i*n+j]
    var spectral_energy = Float64(0.0)
    var tail = Float64(0.0)
    for c in range(m):
        var s = full.singular_vals[c]
        if c > 0 and s > full.singular_vals[c-1]:
            raise Error("wide spectrum not sorted")
        spectral_energy += s*s
        if c > 0:
            tail += s*s
    var scale = max(1.0, energy)
    if residual > 1e-8 * scale or abs(energy-spectral_energy) > 3e-5 * scale:
        raise Error("wide reconstruction/energy failed")
    if abs(truncated_residual-tail) > 5e-5 * scale:
        raise Error("wide truncated basis is not optimal")
    var expected_noise = tail / Float64((m-1)*(m-1))
    if abs(trunc.noise_var-expected_noise) > 3e-5 * max(1.0, expected_noise):
        raise Error("wide noise variance uses feature count rather than spectrum size")
    if n <= 33 and kind == 0:
        var reference = _ref(xs, m, n)
        for c in range(m-1):
            if abs(full.singular_vals[c]-reference[c]) > 2e-4 * max(1.0, reference[c]):
                raise Error("wide singular value disagrees with Float64 reference")
    var hash = UInt64(14695981039346656037)
    for j in range(len(full.components)):
        hash = (hash ^ UInt64(bitcast[DType.uint32](Float32(full.components[j])))) * UInt64(1099511628211)
    for j in range(m):
        hash = (hash ^ UInt64(bitcast[DType.uint32](Float32(full.singular_vals[j])))) * UInt64(1099511628211)
    print("WIDE", m, n, kind, "hash", hash, "orth", orth, "residual", residual/scale)


def main() raises:
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("wide gate requires IDENTICAL")
    check(4, 8, 0)
    check(8, 17, 0)
    check(17, 33, 0)
    check(4, 129, 0)
    check(3, 7, 1)
    check(8, 13, 2)
    var refused = False
    try:
        pca_full_validate(4, 8, 5)
    except:
        refused = True
    if not refused:
        raise Error("wide n_components above min shape accepted")
    print("wide full SVD PASS")
