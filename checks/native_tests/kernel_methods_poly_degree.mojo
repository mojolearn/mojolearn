# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""KernelRidge's polynomial degree reaches the fit and the prediction (ported
from python/mojolearn/tests/test_cpu_kernel_variants.py
`test_polynomial_degree_affects_host_fit_and_inference`).

The original literal 4 x 2 workload, kernel POLYNOMIAL with gamma 0.5,
coef0 0.25 and alpha 4, fitted once per degree 0, 1, 2 and 3 and predicted on
its own rows: the four predictions must be four different word sets, so a fit
or a prediction that ignored the degree fails. Cases 0 to 3 are the four
independent fits; case 4 compares their predictions.
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, CPU_COLUMN_BUILD, require
from std.memory import bitcast
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.hostptr import f32_ptr, f64_ptr
from kernel_methods.host.kernel_methods_host_run import kernel_ridge_fit_host_run, kernel_ridge_predict_host_run
from core.identity_trace import IdentityTrace
from kernel_methods.estimator import kernel_ridge_fit_host, kernel_ridge_predict_host
from svm.impl.svm_parameter import KernelParams, KERNEL_POLYNOMIAL
from std.sys.compile import is_defined



comptime POLY_DEGREES = 4
comptime POLY_CASES = POLY_DEGREES + 1
comptime POLY_ROWS = 4
comptime POLY_FEATURES = 2
comptime POLY_GAMMA = Float64(0.5)
comptime POLY_COEF0 = Float64(0.25)
comptime POLY_ALPHA = Float32(4.0)


def poly_x() -> List[Float32]:
    var x: List[Float32] = [0.25, -0.5, -0.25, 0.125, 0.5, 0.25, -0.125, -0.25]
    return x^


def poly_y() -> List[Float32]:
    var y: List[Float32] = [0.25, -0.5, 0.75, 0.125]
    return y^


trait PolyDegreeCalls:
    @staticmethod
    def fit_predict(x: List[Float32], y: List[Float32], degree: Int) raises -> List[Float32]:
        """The fit at `degree`, then the prediction of X: POLY_ROWS words."""
        ...


def distinct_predictions(words: List[Float32]) raises:
    """Every pair of degrees differs in at least one prediction word."""
    for a in range(POLY_DEGREES):
        for b in range(a + 1, POLY_DEGREES):
            var differ = False
            for i in range(POLY_ROWS):
                if bitcast[DType.uint32](words[a * POLY_ROWS + i]) != bitcast[DType.uint32](words[b * POLY_ROWS + i]):
                    differ = True
            require(differ, "degree was ignored in fit or prediction: degrees " + String(a) + " and "
                    + String(b) + " predict the same words")


def fit_unit[Calls: PolyDegreeCalls](degree: Int, words_address: Int) raises:
    var words = f32_ptr(words_address)
    var prediction = Calls.fit_predict(poly_x(), poly_y(), degree)
    require(len(prediction) == POLY_ROWS, "KernelRidge poly prediction extent")
    for i in range(POLY_ROWS):
        words.unsafe_store(degree * POLY_ROWS + i, prediction[i])


# ---- The host column ----
# The host entries the KernelRidge host binding calls; the four fits run
# across host cores, the comparison after them.
struct HostPolyDegree(PolyDegreeCalls):
    @staticmethod
    def fit_predict(x: List[Float32], y: List[Float32], degree: Int) raises -> List[Float32]:
        var dual = List[Float32](length=POLY_ROWS, fill=0)
        var scalars = List[Float64](length=1, fill=0)
        var info = kernel_ridge_fit_host_run(Int(x.unsafe_ptr()), Int(y.unsafe_ptr()), False, 0,
            POLY_ROWS, POLY_FEATURES, 1, KERNEL_POLYNOMIAL, degree, POLY_GAMMA, POLY_COEF0, POLY_ALPHA,
            f32_ptr(Int(dual.unsafe_ptr())), f64_ptr(Int(scalars.unsafe_ptr())))
        require(info == 0, "KernelRidge poly host fit info")
        var prediction = List[Float32](length=POLY_ROWS, fill=0)
        kernel_ridge_predict_host_run(Int(x.unsafe_ptr()), Int(dual.unsafe_ptr()), Int(x.unsafe_ptr()),
            POLY_ROWS, POLY_FEATURES, 1, KERNEL_POLYNOMIAL, degree, POLY_GAMMA, POLY_COEF0, POLY_ROWS,
            f32_ptr(Int(prediction.unsafe_ptr())))
        _ = x
        _ = y
        _ = dual^
        _ = scalars^
        return prediction^


# ---- The device column ----
# The device estimator entries. The fits run in order: each submits to the
# shared family DeviceContext (see kernel_methods_krr_options.mojo); the
# device kernels keep their own parallelism.
struct DevicePolyDegree(PolyDegreeCalls):
    @staticmethod
    def fit_predict(x: List[Float32], y: List[Float32], degree: Int) raises -> List[Float32]:
        var trace = IdentityTrace()
        var model = kernel_ridge_fit_host(x, y, POLY_ROWS, POLY_FEATURES, 1,
            KernelParams(KERNEL_POLYNOMIAL, degree, POLY_GAMMA, POLY_COEF0), POLY_ALPHA, trace)
        require(model.info == 0, "KernelRidge poly device fit info")
        var prediction = kernel_ridge_predict_host(model, x, POLY_ROWS, trace)
        _ = model^
        return prediction^


# The fits fill one prediction row per degree; the distinct-predictions case
# reads them all afterwards.
def test_host_fixtures() raises:
    comptime if CPU_COLUMN_BUILD:
        comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
        var words = List[Float32](length=POLY_DEGREES * POLY_ROWS, fill=0)
        var wp = Int(words.unsafe_ptr())
        def fit(unit: Int) raises {imm wp}:
            fit_unit[HostPolyDegree](unit, wp)
        check_cases("kernel_methods_poly_degree host fits", POLY_DEGREES, fit)
        distinct_predictions(words)
        _ = words^


def test_device_fixtures() raises:
    comptime if not CPU_COLUMN_BUILD:
        comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
        var words = List[Float32](length=POLY_DEGREES * POLY_ROWS, fill=0)
        var wp = Int(words.unsafe_ptr())
        var failed = String("")
        for unit in range(POLY_DEGREES):  # driver-loop: fits in order on the shared family device context
            try:
                fit_unit[DevicePolyDegree](unit, wp)
            except problem:
                failed += "\n      kernel_methods_poly_degree device fits row " + String(unit) + ": " + String(problem)
        if failed.byte_length() > 0:
            raise Error("kernel_methods_poly_degree device fits: cases failed" + failed)
        distinct_predictions(words)
        _ = words^


def main() raises:
    # Each column registers its own tests by hand. Discovery (`__functions_in_module()`) reflects over every
    # function of the module, which elaborates the device helpers in the host column's build ("target 'unknown'
    # is not supported"), so a two-column file names its tests.
    var suite = TestSuite()
    comptime if CPU_COLUMN_BUILD:
        suite.test[test_host_fixtures]()
    else:
        suite.test[test_device_fixtures]()
    suite^.run()
