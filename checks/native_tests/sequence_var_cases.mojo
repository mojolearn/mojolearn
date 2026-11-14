# SPDX-License-Identifier: Apache-2.0
"""VAR(p) against a float64 least-squares restatement of statsmodels'
estimator (ported from python/mojolearn/tests/test_x_sequence_var.py
`test_var_matches_least_squares`).

The series: 300 rows of two variables, y[t] = 0.5 + A y[t-1] + e[t],
A = [[0.5, 0.1], [0.2, 0.3]], e hashed normals (the original drew e from
NumPy; the reference does not depend on the stream), rounded to float32.
One fixture per (p, trend) of (1, 'c'), (2, 'c'), (2, 'n'): the params of
`sequence/ops_impl.mojo::var_fit_impl` ((trend + 2 p) x 2: the intercept
row, then lag 1, lag 2) must equal B = lstsq(Z, y[p:]) within atol 2e-4 and
sigma_u must equal R^T R / (rows - columns of Z), R = y[p:] - Z B, within
rtol 1e-3, all in float64 (normal equations by Cholesky; Z is small and
well conditioned). The rank-deficient refusal stays in the Python file (its
public LinAlgError).
"""
from std.math import sqrt
from core.host_parallel import host_parallelize
from sequence.exec import Exec
from sequence.ops_impl import var_fit_impl
from checks.native_tests.sequence_port_common import require, allclose, normal

comptime VAR_FIXTURES = 3
comptime VAR_N = 300
comptime VAR_K = 2


def var_series() -> List[Float32]:
    var y = List[Float64](length=VAR_N * VAR_K, fill=0.0)
    for t in range(1, VAR_N):  # an autoregression: serial by nature
        var a = y[(t - 1) * VAR_K]
        var b = y[(t - 1) * VAR_K + 1]
        y[t * VAR_K] = 0.5 + 0.5 * a + 0.1 * b + normal(0x7A40, 2 * t)
        y[t * VAR_K + 1] = 0.5 + 0.2 * a + 0.3 * b + normal(0x7A40, 2 * t + 1)
    var rounded = List[Float32](length=VAR_N * VAR_K, fill=0.0)
    var op = rounded.unsafe_ptr()
    var yp = y.unsafe_ptr()
    def cell(i: Int) {imm op, imm yp}:
        op.unsafe_store(i, Float32(yp[i]))
    host_parallelize(cell, VAR_N * VAR_K)
    _ = y^
    return rounded^


def run_var_fixture[E: Exec](mut ex: E, unit: Int) raises:
    var p = 1 if unit == 0 else 2
    var kt = 0 if unit == 2 else 1
    var y = var_series()
    var rows = VAR_N - p
    var m = kt + VAR_K * p
    var params = List[Float32](length=m * VAR_K, fill=0.0)
    var sigma = List[Float32](length=VAR_K * VAR_K, fill=0.0)
    var resid = List[Float32](length=rows * VAR_K, fill=0.0)
    var code = var_fit_impl(ex, [Int(y.unsafe_ptr()), Int(params.unsafe_ptr()), Int(sigma.unsafe_ptr()),
                                 Int(resid.unsafe_ptr())], [VAR_N, VAR_K, p, kt])
    require(code == 0, "VAR fit refused a full-rank design (code " + String(code) + ")")
    # Z (rows x m) and the targets in float64
    var Z = List[Float64](length=rows * m, fill=0.0)
    var zp = Z.unsafe_ptr()
    var yp = y.unsafe_ptr()
    def design(index: Int) {imm zp, imm yp, imm m, imm kt, imm p}:
        var r = index // m
        var c = index % m
        var t = r + p
        if c < kt:
            zp.unsafe_store(index, 1.0)
            return
        var lag = 1 + (c - kt) // VAR_K
        var variable = (c - kt) % VAR_K
        zp.unsafe_store(index, Float64(yp[(t - lag) * VAR_K + variable]))
    host_parallelize(design, rows * m)
    # G = Z^T Z (m x m) and C = Z^T Y (m x K), one cell per task, rows ascending
    var G = List[Float64](length=m * m, fill=0.0)
    var C = List[Float64](length=m * VAR_K, fill=0.0)
    var gp = G.unsafe_ptr()
    var cp = C.unsafe_ptr()
    def gram(index: Int) {imm zp, imm yp, imm gp, imm cp, imm m, imm p, imm rows}:
        var acc = 0.0
        if index < m * m:
            var i = index // m
            var j = index % m
            for r in range(rows):
                acc += zp[r * m + i] * zp[r * m + j]
            gp.unsafe_store(index, acc)
            return
        var slot = index - m * m
        var zi = slot // VAR_K
        var yk = slot % VAR_K
        for r in range(rows):
            acc += zp[r * m + zi] * Float64(yp[(r + p) * VAR_K + yk])
        cp.unsafe_store(slot, acc)
    host_parallelize(gram, m * m + m * VAR_K)
    # Cholesky G = L L^T (m <= 5), then two triangular solves per column of C
    var L = List[Float64](length=m * m, fill=0.0)
    for j in range(m):  # a factorisation: serial by nature
        var d = G[j * m + j]
        for k in range(j):
            d -= L[j * m + k] * L[j * m + k]
        require(d > 0.0, "VAR reference: the design Gram is not positive definite")
        L[j * m + j] = sqrt(d)
        for i in range(j + 1, m):
            var s = G[i * m + j]
            for k in range(j):
                s -= L[i * m + k] * L[j * m + k]
            L[i * m + j] = s / L[j * m + j]
    var B = List[Float64](length=m * VAR_K, fill=0.0)
    for k in range(VAR_K):  # two right-hand sides, each a serial substitution
        var w = List[Float64](length=m, fill=0.0)
        for i in range(m):
            var s = C[i * VAR_K + k]
            for j in range(i):
                s -= L[i * m + j] * w[j]
            w[i] = s / L[i * m + i]
        for ii in range(m):
            var i = m - 1 - ii
            var s = w[i]
            for j in range(i + 1, m):
                s -= L[j * m + i] * B[j * VAR_K + k]
            B[i * VAR_K + k] = s / L[i * m + i]
    for i in range(m * VAR_K):  # at most ten coefficients
        require(allclose(Float64(params[i]), B[i], 1e-7, 2e-4), "VAR p=" + String(p) + " kt=" + String(kt) + " param "
                + String(i) + " got " + String(params[i]) + " want " + String(B[i]) + " (atol 2e-4)")
    # sigma_u = R^T R / (rows - m), one cell per task
    var S = List[Float64](length=VAR_K * VAR_K, fill=0.0)
    var sp = S.unsafe_ptr()
    var bp = B.unsafe_ptr()
    def cov(cell: Int) {imm zp, imm yp, imm sp, imm bp, imm m, imm p, imm rows}:
        var a = cell // VAR_K
        var b = cell % VAR_K
        var acc = 0.0
        for r in range(rows):
            var fa = 0.0
            var fb = 0.0
            for c in range(m):
                fa += zp[r * m + c] * bp[c * VAR_K + a]
                fb += zp[r * m + c] * bp[c * VAR_K + b]
            acc += (Float64(yp[(r + p) * VAR_K + a]) - fa) * (Float64(yp[(r + p) * VAR_K + b]) - fb)
        sp.unsafe_store(cell, acc / Float64(rows - m))
    host_parallelize(cov, VAR_K * VAR_K)
    for i in range(VAR_K * VAR_K):  # four covariance cells
        require(allclose(Float64(sigma[i]), S[i], 1e-3, 0.0), "VAR p=" + String(p) + " kt=" + String(kt)
                + " sigma_u cell " + String(i) + " got " + String(sigma[i]) + " want " + String(S[i]) + " (rtol 1e-3)")
    _ = y^
    _ = params^
    _ = sigma^
    _ = resid^
    _ = Z^
    _ = G^
    _ = C^
    _ = B^
    _ = S^
