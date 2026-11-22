# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""MLPRegressor (Adam) and MLPClassifier (SGD) loss curves against scikit-learn's from the same initial weights
(lane py2mojo-A, 2026-11-22; ported from python/mojolearn/tests/test_x_sequence_mlp.py `test_against_sklearn`).

X (200 x 5) and the two targets are the original `_data()` draws; the initial weights (Glorot-uniform bounds, w then
b of each layer, the packed order the fit trains) and scikit-learn's `loss_curve_` from those same weights
(`shuffle=False`, `batch_size=50`, `_init_coef` replaced) are stored in `fixtures/xseq_mlp_sklearn/` by
`tools/py2mojo_native_fixtures_b8.py` (scikit-learn 1.9.1). The original handed scikit-learn our seeded init draws;
the init stream is not what the test asserts (the curves compare the optimisation alone), so here both fits start
from NumPy draws instead.

The fit is `sequence/ops_impl.mojo` `mlp_fit_impl`, the MLP_FIT body of the sequence image's neural entry
(`sequence/neural_model_dispatch.mojo`), on the column's executor, with the words the door sends
(bindings/xseq_neural_native_door.mojo `_mlp_fit`, `xseq_mlp`; sequence/xseq_neural_run.mojo `xseq_run_mlp_fit`):

    unit 0  MLPRegressor(hidden_layer_sizes=(8, 4), activation="tanh", max_iter=15): act 2, out_act 0 identity,
            loss 0, solver 0 adam, the (200, 1) float target
    unit 1  MLPClassifier(hidden_layer_sizes=(6,), solver="sgd", max_iter=15): act 3 relu, out_act 4 softmax, loss 2,
            solver 1 sgd, nesterov momentum 0.9, three classes as int32 codes (target-codes word 1)

with the estimators' defaults learning_rate "constant", learning_rate_init 1e-3, beta 0.9 / 0.999, epsilon 1e-8,
power_t 0.5, alpha 1e-4, tol 1e-4, n_iter_no_change 10. The shuffle seed word is 0 (shuffle is off). Checks: the
same epoch count as scikit-learn's curve and every loss to rtol 1e-4, atol 1e-5.

The public estimators, their seeded init stream, labels and shapes stay in the Python file.
"""
from std.math import isnan
from std.sys.compile import is_defined
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases
from checks.native_tests.tsa_metrics_fixture_words import fixture_f32, fixture_i32
from checks.native_tests.py2mojo_fixture_words import fixture_f64
from sequence.exec import Exec, HostExec
from sequence.exec_device import DeviceExec
from sequence.ops_impl import mlp_fit_impl

comptime CPU_COLUMN_BUILD = is_defined["MOJOLEARN_COLUMN_CPU"]()
comptime MLP_DIR = "checks/native_tests/fixtures/xseq_mlp_sklearn/"
comptime MLP_N = 200
comptime MLP_D = 5
comptime MLP_MAX_ITER = 15
comptime MLP_UNITS = 2


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def run_mlp[E: Exec](mut ex: E, unit: Int) raises:
    var label = String("MLPRegressor adam") if unit == 0 else String("MLPClassifier sgd")
    var x = fixture_f32(MLP_DIR + "x.f32", MLP_N * MLP_D)
    var yr = fixture_f32(MLP_DIR + "yr.f32", MLP_N)
    var y3 = fixture_i32(MLP_DIR + "y3.i32", MLP_N)
    var count = 89 if unit == 0 else 57  # 5*8+8 + 8*4+4 + 4*1+1; 5*6+6 + 6*3+3
    var params = fixture_f32(MLP_DIR + "init" + String(unit) + ".f32", count)
    var curve = List[Float32](length=MLP_MAX_ITER, fill=0)
    var ip: List[Int]
    if unit == 0:
        ip = [MLP_N, MLP_D, 1, 2, 0, 0, 0, 0, 1, 50, MLP_MAX_ITER, 0, 0, 10, 2, 8, 4]
    else:
        ip = [MLP_N, MLP_D, 3, 3, 4, 2, 1, 0, 1, 50, MLP_MAX_ITER, 0, 0, 10, 1, 6, 1]
    var fp: List[Float64] = [1e-3, 0.9, 0.999, 1e-8, 0.9, 0.5, 1e-4, 1e-4]
    var target = Int(yr.unsafe_ptr()) if unit == 0 else Int(y3.unsafe_ptr())
    var addrs: List[Int] = [Int(x.unsafe_ptr()), target, Int(params.unsafe_ptr()), Int(curve.unsafe_ptr())]
    var n_iter = mlp_fit_impl(ex, addrs, ip, fp)
    var want = fixture_f64(MLP_DIR + "curve" + String(unit) + ".f64", MLP_MAX_ITER)
    require(n_iter == MLP_MAX_ITER, label + ": " + String(n_iter) + " epochs, scikit-learn ran " + String(MLP_MAX_ITER))
    for i in range(MLP_MAX_ITER):  # small-loop(oracle: one loss curve)
        var got = Float64(curve[i])
        require(not isnan(got) and abs(got - want[i]) <= 1e-5 + 1e-4 * abs(want[i]),
                label + ": loss of epoch " + String(i) + " is " + String(curve[i]) + ", scikit-learn " + String(want[i]))
    _ = x^
    _ = yr^
    _ = y3^
    _ = params^
    _ = curve^


def _host_unit(unit: Int) raises:
    var ex = HostExec()
    run_mlp(ex, unit)


def test_host_against_sklearn() raises:
    comptime if CPU_COLUMN_BUILD:
        check_cases("xseq_mlp_fit host", MLP_UNITS, _host_unit)


# The device column: the two fits in order on one DeviceExec (one device, one queue).
def test_device_against_sklearn() raises:
    comptime if not CPU_COLUMN_BUILD:
        var ex = DeviceExec()
        var failed = String("")
        for unit in range(MLP_UNITS):  # driver-loop: over the fits on the one device
            try:
                run_mlp(ex, unit)
            except problem:
                failed += "\n      xseq_mlp_fit device row " + String(unit) + ": " + String(problem)
        if failed.byte_length() > 0:
            raise Error("xseq_mlp_fit device: cases failed" + failed)


def main() raises:
    # A two-column file names its tests (discovery elaborates the device helpers in the host build).
    var suite = TestSuite()
    comptime if CPU_COLUMN_BUILD:
        suite.test[test_host_against_sklearn]()
    else:
        suite.test[test_device_against_sklearn]()
    suite^.run()
