# SPDX-License-Identifier: Apache-2.0
"""LayerNorm on pending transpose views, forward and gradients, moved from
test_ops_layernorm_view.py (2026-11-14; the whole file).

Three fixtures: the input a view (x stored transposed, then transposed back),
the affine weight and bias views, and all three. x (4, 3), w = [.7, -1.2, .4],
b = [.2, .1, -.3], eps = 0.03, the backward seed linspace(-0.6, 0.9, 12)
reshaped (4, 3); loss = sum(layer_norm(x, w, b) * seed). The independent
float64 oracle: normalized = (x - mean) / sqrt(var + eps), output =
normalized * w + b (rtol 4e-6, atol 2e-7), and
dx = inv * (sw - mean(sw) - normalized * mean(sw * normalized)) with sw = seed * w,
dw = sum_rows(seed * normalized), db = sum_rows(seed) (rtol 5e-6, atol 3e-7),
each gradient in its stored (transposed when a view) layout and shape.
"""
from std.math import sqrt
from autograd.ops_api import OpsSession
from sequence.exec_trait import Exec
from checks.native_tests.ops_higher import tensor, op1, op2, total, values, gradient
from checks.native_tests.core_ops_common import allclose, dims2, same_shape, opi
from core.host_parallel import host_parallelize
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


comptime CORE_OPS_LAYERNORM_VIEW_FIXTURES = 3

comptime _MUL = 4
comptime _TRANSPOSE = 13
comptime _LAYER_NORM = 20


def _transposed(v: List[Float32], rows: Int, cols: Int) -> List[Float32]:
    var out = List[Float32](length=rows * cols, fill=0)
    for i in range(rows):
        for j in range(cols):
            out[j * rows + i] = v[i * cols + j]
    return out^


def _transposed64(v: List[Float64], rows: Int, cols: Int) -> List[Float64]:
    var out = List[Float64](length=rows * cols, fill=0)
    for i in range(rows):
        for j in range(cols):
            out[j * rows + i] = v[i * cols + j]
    return out^


def _stored[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], v: List[Float32], rows: Int, cols: Int, view: Bool
) raises -> Tuple[Int, Int]:
    """(the uploaded handle, the operand): a view uploads v transposed and
    transposes it back."""
    if view:
        var h = tensor(s, _transposed(v, rows, cols), dims2(cols, rows), True)
        return (h, op1(s, _TRANSPOSE, h))
    var h = tensor(s, v, dims2(rows, cols), True)
    return (h, h)


def run_core_ops_layernorm_view_fixture[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], mut again: OpsSession[E], fixture_id: Int
) raises:
    require(fixture_id >= 0 and fixture_id < CORE_OPS_LAYERNORM_VIEW_FIXTURES, "unknown layernorm view fixture")
    var input_view = fixture_id == 0 or fixture_id == 2
    var affine_view = fixture_id == 1 or fixture_id == 2
    var x: List[Float32] = [0.2, -0.7, 1.1, 2.0, 0.3, -0.4, -0.1, 0.8, 1.7, 0.6, -0.2, -0.9]
    var w: List[Float32] = [0.7, -1.2, 0.4]
    var b: List[Float32] = [0.2, 0.1, -0.3]
    var seed = List[Float32](length=12, fill=0)
    for i in range(12):
        seed[i] = Float32(-0.6 + Float64(i) * (1.5 / 11.0))
    var eps = Float32(0.03)
    var hx = _stored(s, x, 4, 3, input_view)
    var hw = _stored(s, w, 1, 3, affine_view)
    var hb = _stored(s, b, 1, 3, affine_view)
    var hs: List[Int] = [hx[1], hw[1], hb[1]]
    var no_ints = List[Int]()
    var eps_words: List[Float32] = [eps]
    var output = opi(s, _LAYER_NORM, hs, no_ints, eps_words)
    s.backward(total(s, op2(s, _MUL, output, tensor(s, seed, dims2(4, 3)))), False)
    # the float64 oracle
    var normalized = List[Float64](length=12, fill=0)
    var inv = List[Float64](length=4, fill=0)
    var expected_out = List[Float64](length=12, fill=0)
    var dx = List[Float64](length=12, fill=0)
    var dw = List[Float64](length=3, fill=0)
    var db = List[Float64](length=3, fill=0)
    for r in range(4):
        var mean = (Float64(x[3 * r]) + Float64(x[3 * r + 1]) + Float64(x[3 * r + 2])) / 3.0
        var var_ = Float64(0)
        for c in range(3):
            var d = Float64(x[3 * r + c]) - mean
            var_ += d * d
        inv[r] = 1.0 / sqrt(var_ / 3.0 + Float64(eps))
        var sw_mean = Float64(0)
        var swn_mean = Float64(0)
        for c in range(3):
            normalized[3 * r + c] = (Float64(x[3 * r + c]) - mean) * inv[r]
            expected_out[3 * r + c] = normalized[3 * r + c] * Float64(w[c]) + Float64(b[c])
            var sw = Float64(seed[3 * r + c]) * Float64(w[c])
            sw_mean += sw / 3.0
            swn_mean += sw * normalized[3 * r + c] / 3.0
        for c in range(3):
            var sw = Float64(seed[3 * r + c]) * Float64(w[c])
            dx[3 * r + c] = inv[r] * (sw - sw_mean - normalized[3 * r + c] * swn_mean)
            dw[c] += Float64(seed[3 * r + c]) * normalized[3 * r + c]
            db[c] += Float64(seed[3 * r + c])
    allclose(values(s, output, dims2(4, 3)), expected_out, 4e-6, 2e-7, "layer_norm forward")
    var x_dims = dims2(3, 4) if input_view else dims2(4, 3)
    var a_dims = dims2(3, 1) if affine_view else dims2(1, 3)
    same_shape(s, hx[0], x_dims, "the input's gradient")
    same_shape(s, hw[0], a_dims, "the weight's gradient")
    same_shape(s, hb[0], a_dims, "the bias's gradient")
    allclose(gradient(s, hx[0], x_dims), _transposed64(dx, 4, 3) if input_view else dx.copy(), 5e-6, 3e-7, "d input")
    allclose(gradient(s, hw[0], a_dims), _transposed64(dw, 1, 3) if affine_view else dw.copy(), 5e-6, 3e-7, "d weight")
    allclose(gradient(s, hb[0], a_dims), _transposed64(db, 1, 3) if affine_view else db.copy(), 5e-6, 3e-7, "d bias")


# ---- The host column (was core_ops_layernorm_view_cpu_suite.mojo) ----
# LayerNorm on pending transpose views: each fixture on its own two host sessions, the fixtures run
# across host cores with a private failure slot each.
def _run_cpu(mut report: Report) raises:
    var cases = report.cases(CORE_OPS_LAYERNORM_VIEW_FIXTURES)
    def task(fixture_id: Int) {imm cases}:
        try:
            var record = True
            var s = OpsSession[HostExec](HostExec(), record)
            var again = OpsSession[HostExec](HostExec(), record)
            try:
                run_core_ops_layernorm_view_fixture(s, again, fixture_id)
            except error:
                s.close()
                again.close()
                raise error
            s.close()
            again.close()
        except error:
            cases.fail(fixture_id, error)
    report.parallel(task, CORE_OPS_LAYERNORM_VIEW_FIXTURES)


# ---- The device column (was core_ops_layernorm_view_gpu_suite.mojo) ----
# LayerNorm on pending transpose views on the device, in order.
#
# DeviceExec shares sequence_ctx() and its pool, so independent concurrent
# sessions would race that context's ownership (as in ops_higher_gpu); the
# fixtures run one after another and each kernel keeps its own parallelism.
def _run_gpu(mut report: Report) raises:
    var cases = report.cases(CORE_OPS_LAYERNORM_VIEW_FIXTURES)
    for fixture_id in range(CORE_OPS_LAYERNORM_VIEW_FIXTURES):
        try:
            var record = True
            var s = OpsSession[DeviceExec](DeviceExec(), record)
            var again = OpsSession[DeviceExec](DeviceExec(), record)
            try:
                run_core_ops_layernorm_view_fixture(s, again, fixture_id)
            except error:
                s.close()
                again.close()
                raise error
            s.close()
            again.close()
        except error:
            cases.fail(fixture_id, error)


def run_cases(column: Column, mut report: Report):
    try:
        comptime if CPU_COLUMN_BUILD:
            _run_cpu(report)
        else:
            _run_gpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("core_ops_layernorm_view", cpu=True, gpu=True, serial="", run=run_cases)
