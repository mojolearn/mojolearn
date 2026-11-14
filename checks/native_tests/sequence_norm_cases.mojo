# SPDX-License-Identifier: Apache-2.0
"""LayerNorm forward and backward against a float64 restatement of
F.layer_norm and its gradients, and the module's normalised rows (ported
from python/mojolearn/tests/test_x_sequence_norm.py `test_forward_backward`
and the numerical half of `test_module`).

Fixture 0: x (32 x 6) = 2 z + 1, dy, weight and bias hashed normals (the
original drew them from NumPy; its reference never depended on the stream),
eps 1e-5, through `sequence/ops_impl.mojo::layer_norm_impl`: y within atol
2e-5 of xh w + b; dx within 2e-5 of rstd (g - mean(g) - xh mean(g xh)),
g = dy w; dweight within 2e-4 of sum(dy xh); dbias within 2e-5 of sum(dy).
Fixture 1: LayerNorm(4) (weight ones, bias zeros) on arange(8) as 2 x 4:
every row of y has mean 0 within 1e-6. The module's gradient shapes stay a
Python boundary matter.
"""
from std.math import sqrt
from core.host_parallel import host_parallelize
from sequence.exec import Exec
from sequence.ops_impl import layer_norm_impl
from checks.native_tests.sequence_port_common import require, allclose, normal

comptime NORM_FIXTURES = 2
comptime LN_M = 32
comptime LN_D = 6
comptime LN_EPS = 1e-5


def normals(count: Int, salt: Int, scale: Float64, shift: Float64) -> List[Float32]:
    var values = List[Float32](length=count, fill=0.0)
    var vp = values.unsafe_ptr()
    def cell(i: Int) {imm vp, imm salt, imm scale, imm shift}:
        vp.unsafe_store(i, Float32(normal(salt, i) * scale + shift))
    host_parallelize(cell, count)
    return values^


def module_fixture[E: Exec](mut ex: E) raises:
    var x: List[Float32] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    var w: List[Float32] = [1.0, 1.0, 1.0, 1.0]
    var b: List[Float32] = [0.0, 0.0, 0.0, 0.0]
    var y = List[Float32](length=8, fill=7.0)
    _ = layer_norm_impl(ex, [Int(x.unsafe_ptr()), Int(w.unsafe_ptr()), Int(b.unsafe_ptr()), Int(y.unsafe_ptr()),
                             0, 0, 0, 0], [2, 4, 1, 1, 0], List[Float64](length=1, fill=LN_EPS))
    for row in range(2):  # two rows, each a four-term mean
        var mean = (Float64(y[4 * row]) + Float64(y[4 * row + 1]) + Float64(y[4 * row + 2])
                    + Float64(y[4 * row + 3])) / 4.0
        require(abs(mean) <= 1e-6, "LayerNorm module: row " + String(row) + " mean " + String(mean))
    _ = x^
    _ = w^
    _ = b^
    _ = y^


def run_norm_fixture[E: Exec](mut ex: E, unit: Int) raises:
    if unit == 1:
        module_fixture(ex)
        return
    var x = normals(LN_M * LN_D, 0x1A01, 2.0, 1.0)
    var dy = normals(LN_M * LN_D, 0x1A02, 1.0, 0.0)
    var w = normals(LN_D, 0x1A03, 1.0, 0.0)
    var b = normals(LN_D, 0x1A04, 1.0, 0.0)
    var y = List[Float32](length=LN_M * LN_D, fill=0.0)
    var dx = List[Float32](length=LN_M * LN_D, fill=0.0)
    var dw = List[Float32](length=LN_D, fill=0.0)
    var db = List[Float32](length=LN_D, fill=0.0)
    _ = layer_norm_impl(ex, [Int(x.unsafe_ptr()), Int(w.unsafe_ptr()), Int(b.unsafe_ptr()), Int(y.unsafe_ptr()),
                             0, 0, 0, 0], [LN_M, LN_D, 1, 1, 0], List[Float64](length=1, fill=LN_EPS))
    _ = layer_norm_impl(ex, [Int(x.unsafe_ptr()), Int(w.unsafe_ptr()), Int(b.unsafe_ptr()), 0,
                             Int(dy.unsafe_ptr()), Int(dx.unsafe_ptr()), Int(dw.unsafe_ptr()), Int(db.unsafe_ptr())],
                        [LN_M, LN_D, 1, 1, 1], List[Float64](length=1, fill=LN_EPS))
    # float64 statistics and xh per row, then the row and column relations
    var xh = List[Float64](length=LN_M * LN_D, fill=0.0)
    var rstd = List[Float64](length=LN_M, fill=0.0)
    var xp = x.unsafe_ptr()
    var hp = xh.unsafe_ptr()
    var sp = rstd.unsafe_ptr()
    def stats(r: Int) {imm xp, imm hp, imm sp}:
        var mu = 0.0
        for c in range(LN_D):
            mu += Float64(xp[r * LN_D + c])
        mu /= Float64(LN_D)
        var var_ = 0.0
        for c in range(LN_D):
            var d = Float64(xp[r * LN_D + c]) - mu
            var_ += d * d
        var s = 1.0 / sqrt(var_ / Float64(LN_D) + LN_EPS)
        sp.unsafe_store(r, s)
        for c in range(LN_D):
            hp.unsafe_store(r * LN_D + c, (Float64(xp[r * LN_D + c]) - mu) * s)
    host_parallelize(stats, LN_M)
    var failures = List[Int](length=LN_M + LN_D, fill=0)
    var fp = failures.unsafe_ptr()
    var yp = y.unsafe_ptr()
    var gp = dy.unsafe_ptr()
    var wp = w.unsafe_ptr()
    var bp = b.unsafe_ptr()
    var dxp = dx.unsafe_ptr()
    var dwp = dw.unsafe_ptr()
    var dbp = db.unsafe_ptr()
    def relation(unit_: Int) {imm fp, imm hp, imm sp, imm yp, imm gp, imm wp, imm bp, imm dxp, imm dwp, imm dbp}:
        if unit_ < LN_M:
            var r = unit_
            var gmean = 0.0
            var gxmean = 0.0
            for c in range(LN_D):
                var g = Float64(gp[r * LN_D + c]) * Float64(wp[c])
                gmean += g
                gxmean += g * hp[r * LN_D + c]
            gmean /= Float64(LN_D)
            gxmean /= Float64(LN_D)
            for c in range(LN_D):
                var h = hp[r * LN_D + c]
                if not allclose(Float64(yp[r * LN_D + c]), h * Float64(wp[c]) + Float64(bp[c]), 1e-7, 2e-5):
                    fp.unsafe_store(unit_, 1)
                var g = Float64(gp[r * LN_D + c]) * Float64(wp[c])
                if not allclose(Float64(dxp[r * LN_D + c]), sp[r] * (g - gmean - h * gxmean), 1e-7, 2e-5):
                    fp.unsafe_store(unit_, 2)
            return
        var c = unit_ - LN_M
        var sw = 0.0
        var sb = 0.0
        for r in range(LN_M):
            sw += Float64(gp[r * LN_D + c]) * hp[r * LN_D + c]
            sb += Float64(gp[r * LN_D + c])
        if not allclose(Float64(dwp[c]), sw, 1e-7, 2e-4):
            fp.unsafe_store(unit_, 3)
        if not allclose(Float64(dbp[c]), sb, 1e-7, 2e-5):
            fp.unsafe_store(unit_, 4)
    host_parallelize(relation, LN_M + LN_D)
    for i in range(LN_M + LN_D):
        require(failures[i] != 1, "LayerNorm forward row " + String(i) + " off xh w + b (atol 2e-5)")
        require(failures[i] != 2, "LayerNorm dx row " + String(i) + " off the float64 gradient (atol 2e-5)")
        require(failures[i] != 3, "LayerNorm dweight column " + String(i - LN_M) + " off sum(dy xh) (atol 2e-4)")
        require(failures[i] != 4, "LayerNorm dbias column " + String(i - LN_M) + " off sum(dy) (atol 2e-5)")
    _ = x^
    _ = dy^
    _ = w^
    _ = b^
    _ = y^
    _ = dx^
    _ = dw^
    _ = db^
    _ = xh^
    _ = rstd^
