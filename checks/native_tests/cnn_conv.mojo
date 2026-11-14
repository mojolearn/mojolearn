# SPDX-License-Identifier: Apache-2.0
"""Conv2d / Conv1d forward and backward against the float64 reference, on
the host column (lane tmig-sequence-1, 2026-11-14; ported from
`test_x_cnn_conv.py` and `test_cnn_padding_gradient_ownership.py`).

Units 0-3: `test_conv2d_forward_backward` (CASES2D: stride, padding,
dilation, with and without bias), out, dx, dW and db within 1e-5.
Unit 4: `test_conv1d_forward_backward` (Conv2d over (N, C, 1, L)).
Units 5-10: `test_conv2d_options` (OPTION_CASES: reflect, replicate and
circular padding, 'same' and 'valid', groups), within 1e-5.
Units 11-18: `test_explicit_padding_backward_matches_independent_scatter`
(two shapes x reflect, replicate, circular, 'same'): the layer equals a
zero-padding convolution of the independently padded input bit for bit, and
its input gradient equals the pad's adjoint applied to that convolution's
padded gradient bit for bit.

The torch comparisons of the references (`test_reference_matches_torch`,
`test_option_reference_matches_torch`) stay in Python as ORACLE tests.
The units are independent and run across the native host worker pool.
"""
from std.math import sqrt
from core.host_parallel import host_parallelize
from x_cnn.host.entry_runs_host import publish_x_cnn_host_entries
from checks.native_tests.cnn_cases import (
    ConvCase,
    ref_conv,
    run_conv,
    normals,
    check_close,
    check_equal,
    pad_host,
    pad_adjoint_host,
    zeros32,
    PAD_ZEROS,
    PAD_REFLECT,
    PAD_REPLICATE,
    PAD_CIRCULAR,
)
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


# ---- The host column (was cnn_conv_cpu_suite.mojo) ----
comptime CNN_CONV_UNITS = 19


def _conv2d_case(k: Int) -> ConvCase:
    if k == 0:
        return ConvCase(3, 2, 7, 6, 4, 3, 3, 1, 1, [1, 1, 1, 1], 1, 1, 1, PAD_ZEROS, True)
    if k == 1:
        return ConvCase(2, 3, 9, 8, 5, 3, 2, 2, 1, [2, 2, 1, 1], 2, 1, 1, PAD_ZEROS, True)
    if k == 2:
        return ConvCase(4, 1, 5, 5, 2, 2, 2, 2, 2, [0, 0, 0, 0], 1, 1, 1, PAD_ZEROS, False)
    return ConvCase(1, 4, 6, 10, 3, 1, 3, 1, 3, [0, 0, 2, 2], 1, 2, 1, PAD_ZEROS, True)


def _option_case(k: Int) -> ConvCase:
    """x (2, c, 7, 8); the explicit pads of `_pads(cs)`."""
    if k == 0:
        return ConvCase(2, 2, 7, 8, 3, 3, 3, 1, 1, [2, 2, 1, 1], 1, 1, 1, PAD_REFLECT, True)
    if k == 1:
        return ConvCase(2, 4, 7, 8, 6, 3, 2, 2, 1, [1, 1, 2, 2], 1, 1, 2, PAD_REPLICATE, True)
    if k == 2:
        return ConvCase(2, 2, 7, 8, 2, 3, 3, 1, 1, [1, 1, 1, 1], 2, 1, 1, PAD_CIRCULAR, True)
    if k == 3:
        # 'same', kernel (4, 3), dilation (1, 2): totals 3 and 4
        return ConvCase(2, 4, 7, 8, 8, 4, 3, 1, 1, [1, 2, 2, 2], 1, 2, 4, PAD_ZEROS, True)
    if k == 4:
        # 'same', kernel (2, 2): totals 1 and 1
        return ConvCase(2, 3, 7, 8, 2, 2, 2, 1, 1, [0, 1, 0, 1], 1, 1, 1, PAD_REFLECT, True)
    return ConvCase(2, 3, 7, 8, 3, 2, 2, 1, 1, [0, 0, 0, 0], 1, 1, 3, PAD_ZEROS, True)


def _ownership_case(k: Int) -> ConvCase:
    """shape (2, 2, 3, 5) for k < 4, (3, 2, 4, 6) after; reflect, replicate,
    circular, 'same' (the Python test's `options` and `pads`)."""
    var big = k >= 4
    var n = 3 if big else 2
    var h = 4 if big else 3
    var w = 6 if big else 5
    var which = k % 4
    if which == 0:
        return ConvCase(n, 2, h, w, 3, 2, 3, 1, 1, [1, 1, 2, 2], 1, 1, 1, PAD_REFLECT, True)
    if which == 1:
        return ConvCase(n, 2, h, w, 4, 3, 3, 1, 1, [2, 2, 2, 2], 1, 1, 2, PAD_REPLICATE, True)
    if which == 2:
        return ConvCase(n, 2, h, w, 2, 3, 2, 1, 2, [1, 1, 1, 1], 1, 1, 1, PAD_CIRCULAR, True)
    return ConvCase(n, 2, h, w, 2, 2, 4, 1, 1, [0, 1, 1, 2], 1, 1, 2, PAD_ZEROS, True)


def _against_reference(cs: ConvCase, var x: List[Float32], var w: List[Float32], var b: List[Float32],
                       var g: List[Float32], what: String) raises:
    var want = ref_conv(cs, x, w, b, g)
    var got = run_conv(cs, x, w, b, g)
    check_close(got.y, want.y, 1e-5, 1e-5, what + " out")
    check_close(got.dx, want.dx, 1e-5, 1e-5, what + " dx")
    check_close(got.dw, want.dw, 1e-5, 1e-5, what + " grad_weight_")
    if cs.bias:
        check_close(got.db, want.db, 1e-5, 1e-5, what + " grad_bias_")


def _conv2d(k: Int) raises:
    var cs = _conv2d_case(k)
    var x = normals(cs.x_size(), 100 + k)
    var w = normals(cs.w_size(), 200 + k, 0.3)
    var b = normals(cs.oc, 300 + k) if cs.bias else zeros32(cs.oc)
    var g = normals(cs.y_size(), 400 + k)
    _against_reference(cs, x^, w^, b^, g^, "conv2d case " + String(k))


def _conv1d() raises:
    # Conv1d(2, 4, 3, stride=2, padding=2, dilation=2) over (3, 2, 11)
    var cs = ConvCase(3, 2, 1, 11, 4, 1, 3, 1, 2, [0, 0, 2, 2], 1, 2, 1, PAD_ZEROS, True)
    var x = normals(cs.x_size(), 2)
    var w = normals(cs.w_size(), 3)
    var b = normals(cs.oc, 4)
    var g = normals(cs.y_size(), 5)
    _against_reference(cs, x^, w^, b^, g^, "conv1d")


def _options(k: Int) raises:
    var cs = _option_case(k)
    var bound = 1.0 / sqrt(Float64((cs.c // cs.groups) * cs.kh * cs.kw))
    var x = normals(cs.x_size(), 700 + k)
    var w = normals(cs.w_size(), 710 + k, bound)
    var b = normals(cs.oc, 720 + k, bound)
    var g = normals(cs.y_size(), 730 + k)
    _against_reference(cs, x^, w^, b^, g^, "conv2d options case " + String(k))


def _ownership(k: Int) raises:
    var cs = _ownership_case(k)
    var x = zeros32(cs.x_size())
    for i in range(len(x)):  # small-loop(fixture input)
        x[i] = Float32(i) / Float32(97) - Float32(0.4)
    var bound = 1.0 / sqrt(Float64((cs.c // cs.groups) * cs.kh * cs.kw))
    var w = normals(cs.w_size(), 51 + k, bound)
    var b = normals(cs.oc, 61 + k, bound)
    var ny = cs.y_size()
    var g = zeros32(ny)
    for i in range(ny):  # small-loop(np.linspace(-0.5, 0.75, y.size))
        g[i] = Float32(-0.5 + 1.25 * Float64(i) / Float64(ny - 1)) if ny > 1 else Float32(-0.5)
    if ny > 1:
        g[ny - 1] = Float32(0.75)
    var got = run_conv(cs, x, w, b, g)
    # A separate zero-padding convolution of the independently padded input.
    var peer = ConvCase(cs.n, cs.c, cs.h + cs.pads[0] + cs.pads[1], cs.w + cs.pads[2] + cs.pads[3], cs.oc,
                        cs.kh, cs.kw, cs.sh, cs.sw, [0, 0, 0, 0], cs.dh, cs.dw, cs.groups, PAD_ZEROS, True)
    var xp = pad_host(cs, x)
    var padded = run_conv(peer, xp, w, b, g)
    var what = "padding adjoint case " + String(k) + " (" + cs.mode_name() + ")"
    check_equal(padded.y, got.y, what + ": the zero-padding peer's forward")
    check_equal(got.dx, pad_adjoint_host(cs, padded.dx), what + ": dx")
    _ = x^
    _ = xp^
    _ = w^
    _ = b^
    _ = g^


def case(unit: Int) raises:
    if unit < 4:
        _conv2d(unit)
    elif unit == 4:
        _conv1d()
    elif unit < 11:
        _options(unit - 5)
    else:
        _ownership(unit - 11)


def _run_cpu(mut report: Report) raises:
    # The publication _mojolearn_x_cnn_host makes at module init.
    publish_x_cnn_host_entries()
    var cases = report.cases(CNN_CONV_UNITS)
    def task(unit: Int) {imm cases}:
        try:
            case(unit)
        except problem:
            cases.fail(unit, problem)
    report.parallel(task, CNN_CONV_UNITS)


def run_cases(column: Column, mut report: Report):
    try:
        comptime if CPU_COLUMN_BUILD:
            _run_cpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("cnn_conv", cpu=True, gpu=False, serial="", run=run_cases)
