# SPDX-License-Identifier: Apache-2.0
"""BatchNorm2d / BatchNorm1d against the float64 reference of PyTorch's
batch_norm, on the host column (lane tmig-sequence-1, 2026-11-14; ported
from `test_x_cnn_batchnorm.py`).

Units 0-2: `test_batchnorm` over (4, 3, 5, 6), (7, 5) and (6, 4, 9), momentum
0.2: two training steps (y within 1e-4 / 1e-5; dx, grad_weight_, grad_bias_
within 1e-4; the running mean and variance within 1e-5 / 1e-6 of the float64
running update), then an eval forward and backward from the running
statistics. Unit 3: `test_batchnorm_options` momentum=None (the cumulative
average, factor 1 / batches seen) over three batches. Unit 4: the same
test's track_running_stats=False layer in eval mode (batch statistics).

The torch comparisons of the reference stay in Python as ORACLE tests.
The units are independent and run across the native host worker pool.
"""
from core.host_parallel import host_parallelize
from x_cnn.host.entry_runs_host import publish_x_cnn_host_entries
from checks.native_tests.cnn_cases import require, BnLayer, ref_bn, normals, widen, check_close

comptime CNN_BN_UNITS = 5


def _data(seed: Int, n: Int) -> List[Float32]:
    """`_data(seed, shape)`: N(0, 1) * 2 + 0.5."""
    return normals(n, seed, 2.0, 0.5)


def _shape(k: Int) -> List[Int]:
    """(N, C, HW) of (4, 3, 5, 6), (7, 5) and (6, 4, 9)."""
    if k == 0:
        return [4, 3, 30]
    if k == 1:
        return [7, 5, 1]
    return [6, 4, 9]


def _batchnorm(k: Int) raises:
    var s = _shape(k)
    var n = s[0]
    var c = s[1]
    var hw = s[2]
    var total = n * c * hw
    var what = "batchnorm shape " + String(k)
    var bn = BnLayer(c)
    bn.gamma = _data(900 + k, c)
    bn.beta = _data(800 + k, c)
    var gamma = widen(bn.gamma)
    var beta = widen(bn.beta)
    var rm = widen(bn.running_mean())
    var rv = widen(bn.running_var())
    for step in range(2):  # driver-loop: over training steps (the running statistics carry)
        var x = _data(10 * k + step, total)
        var g = _data(10 * k + 5 + step, total)
        var y = bn.forward(x, n, hw, True, True, True, 0.2)
        var dx = bn.backward(x, g, n, hw)
        var want = ref_bn(n, c, hw, widen(x), gamma, beta, rm, rv, True, 0.2, 1e-5, widen(g))
        var at = what + " step " + String(step)
        check_close(y, want.y, 1e-4, 1e-5, at + " y")
        check_close(dx, want.dx, 1e-4, 1e-4, at + " dx")
        check_close(bn.grad_weight(), want.dgamma, 1e-4, 1e-4, at + " grad_weight_")
        check_close(bn.grad_bias(), want.dbeta, 1e-4, 1e-4, at + " grad_bias_")
        check_close(bn.running_mean(), want.rm, 1e-5, 1e-6, at + " running_mean_")
        check_close(bn.running_var(), want.rv, 1e-5, 1e-6, at + " running_var_")
        rm = want.rm.copy()
        rv = want.rv.copy()
        _ = x^
        _ = g^
    var x = _data(10 * k + 3, total)
    var g = _data(10 * k + 4, total)
    var y = bn.forward(x, n, hw, False, True, True, 0.2)
    var dx = bn.backward(x, g, n, hw)
    var want = ref_bn(n, c, hw, widen(x), gamma, beta, widen(bn.running_mean()), widen(bn.running_var()),
                      False, 0.2, 1e-5, widen(g))
    check_close(y, want.y, 1e-4, 1e-5, what + " eval y")
    check_close(dx, want.dx, 1e-4, 1e-5, what + " eval dx")
    _ = x^
    _ = g^


def _cumulative() raises:
    """momentum=None over (5, 3, 4, 4): the factor is 1 / batches seen."""
    var n = 5
    var c = 3
    var hw = 16
    var bn = BnLayer(c)
    var ones = List[Float64](length=c, fill=1.0)
    var nulls = List[Float64](length=c, fill=0.0)
    var rm = nulls.copy()
    var rv = ones.copy()
    for step in range(3):  # driver-loop: over batches (the running statistics carry)
        var x = _data(20 + step, n * c * hw)
        var g = _data(30 + step, n * c * hw)
        _ = bn.forward(x, n, hw, True, True, False, 0.0)
        var want = ref_bn(n, c, hw, widen(x), ones, nulls, rm, rv, True, 1.0 / Float64(step + 1), 1e-5, widen(g))
        check_close(bn.running_mean(), want.rm, 1e-5, 1e-6, "momentum None step " + String(step) + " running_mean_")
        check_close(bn.running_var(), want.rv, 1e-5, 1e-6, "momentum None step " + String(step) + " running_var_")
        rm = want.rm.copy()
        rv = want.rv.copy()
        _ = x^
        _ = g^


def _untracked() raises:
    """track_running_stats=False in eval mode: batch statistics."""
    var n = 5
    var c = 3
    var hw = 16
    var bn = BnLayer(c)
    var ones = List[Float64](length=c, fill=1.0)
    var nulls = List[Float64](length=c, fill=0.0)
    var x = _data(40, n * c * hw)
    var g = _data(41, n * c * hw)
    var y = bn.forward(x, n, hw, False, False, True, 0.1)
    var dx = bn.backward(x, g, n, hw)
    var want = ref_bn(n, c, hw, widen(x), ones, nulls, nulls, ones, True, 0.1, 1e-5, widen(g))
    check_close(y, want.y, 1e-4, 1e-5, "untracked eval y")
    check_close(dx, want.dx, 1e-4, 1e-4, "untracked eval dx")
    _ = x^
    _ = g^


def case(unit: Int) raises:
    if unit < 3:
        _batchnorm(unit)
    elif unit == 3:
        _cumulative()
    else:
        _untracked()


def run_checks() raises:
    # The publication _mojolearn_x_cnn_host makes at module init.
    publish_x_cnn_host_entries()
    var failures = List[Int](length=CNN_BN_UNITS, fill=0)
    var flags = failures.unsafe_ptr()
    def task(unit: Int) {imm flags}:
        try:
            case(unit)
        except problem:
            flags[unit] = 1
            print("FAIL cnn batchnorm unit", unit, String(problem))
    host_parallelize(task, CNN_BN_UNITS)
    var failed = 0
    for failure in failures:  # collect independent case status words
        failed += failure
    _ = failures
    require(failed == 0, "cnn batchnorm units failed: " + String(failed))
    print("PASS cnn batchnorm", CNN_BN_UNITS)
