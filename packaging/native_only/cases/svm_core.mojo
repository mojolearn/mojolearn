# SPDX-License-Identifier: Apache-2.0
# Native binary C-SVC / epsilon-SVR fitting and consumed prediction.
# The oracle solves the symmetric two-point dual analytically; it never
# invokes the product SMO solver or prediction kernel to obtain expected
# coefficients/margins. Linear fixtures use exactly representable optima.
from std.math import exp
from std.memory import bitcast
from std.sys.compile import is_defined
from svm.estimator import svc_fit_host, svc_predict_host, svr_fit_host, svr_predict_host
from svm.impl.svm_parameter import KERNEL_LINEAR, KERNEL_RBF


def _exact(actual: Float32, expected: Float32, name: String) raises:
    # Signed zero is mathematically the same analytic optimum. Actual words,
    # including its sign, still enter the cross-vendor result digest.
    if actual != expected:
        print("FAIL svm", name, "actual", actual, "expected", expected)
        raise Error("native SVM differs from the analytic two-point optimum")


def _digest(mut value: UInt64, word: Float32):
    value = (value ^ UInt64(bitcast[DType.uint32](word))) * UInt64(1099511628211)


def _model(
    x: List[Float32], support: List[Float32], indices: List[Int32], dual: List[Float32],
    count: Int, iterations: Int, bias: Float32, alpha: Float32, mut digest: UInt64,
) raises:
    if count != 2 or len(support) != 2 or len(indices) != 2 or len(dual) != 2:
        raise Error("SVM fixture needs both nonzero support coefficients")
    if iterations < 1 or iterations > 1000:
        raise Error("SVM fit must make a nonzero update within its declared iteration limit")
    digest = (digest ^ UInt64(iterations)) * UInt64(1099511628211)
    digest = (digest ^ UInt64(count)) * UInt64(1099511628211)
    _exact(bias, Float32(0.0), "intercept")
    for i in range(2):
        if indices[i] != Int32(i):
            raise Error("SVM support selection must retain ascending original row indices")
        _exact(support[i], x[i], "support vector")
        _exact(dual[i], alpha * x[i], "dual coefficient")
        _digest(digest, support[i])
        _digest(digest, dual[i])
        _digest(digest, bitcast[DType.float32](indices[i]))
    _digest(digest, bias)


def _rbf_expected(query: Float32, alpha: Float32) -> Float64:
    # Independent binary64 RBF expression, gamma=0.5. The product evaluates
    # FP32 kernels, so this check has a declared 2e-6 absolute error budget;
    # its complete actual output words are additionally retained by digest.
    var q = Float64(query)
    return Float64(alpha) * (exp(-0.5 * (q - 1.0) ** 2) - exp(-0.5 * (q + 1.0) ** 2))


def _prediction(actual: Float32, query: Float32, alpha: Float32, kernel: Int) raises:
    if kernel == KERNEL_LINEAR:
        _exact(actual, Float32(2.0) * alpha * query, "linear prediction")
    else:
        var expected = _rbf_expected(query, alpha)
        if not (abs(Float64(actual) - expected) <= 2.0e-6):
            print("FAIL svm RBF prediction", actual, "oracle", expected)
            raise Error("native SVM RBF prediction differs from independent binary64 oracle")


def check_svc() raises:
    var queries: List[Float32] = [-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0]
    var digest = UInt64(1469598103934665603)
    for fixture in range(3):
        # Reversed training order also tests sorted noncontiguous label
        # mapping and support-row collection independently of label order.
        var x: List[Float32] = [1.0, -1.0]
        var labels: List[Float32] = [7.0, -3.0]
        var kernel = KERNEL_RBF if fixture == 2 else KERNEL_LINEAR
        var C = 4.0 if fixture == 0 else (0.125 if fixture == 1 else 0.25)
        # Linear two-point optimum: a=min(C,1/2), w=2a, b=0.
        # RBF optimum: a=min(C,1/(1-exp(-2))); C=1/4 is strictly
        # below that unconstrained optimum, hence both dual bounds are active.
        var alpha = Float32(0.5 if fixture == 0 else C)
        var model = svc_fit_host(x, labels, 2, 1, kernel, 0.5, C, 1e-6, 1000, 20)
        _exact(model.label0, Float32(-3.0), "lower class")
        _exact(model.label1, Float32(7.0), "upper class")
        _digest(digest, model.label0)
        _digest(digest, model.label1)
        _model(x, model.support_matrix, model.support_idx, model.dual_coefs,
               model.n_support, model.n_iter, model.b, alpha, digest)
        # Consume both margin and label prediction, including the documented
        # nonnegative-margin -> upper-label tie. Use two real batch budgets.
        for budget in [0.000016, 1.0]:
            var margins = svc_predict_host(queries, len(queries), 1, model.support_matrix,
                model.dual_coefs, model.n_support, model.b, model.label0, model.label1,
                kernel, 0.5, False, budget)
            var classes = svc_predict_host(queries, len(queries), 1, model.support_matrix,
                model.dual_coefs, model.n_support, model.b, model.label0, model.label1,
                kernel, 0.5, True, budget)
            if len(margins) != len(queries) or len(classes) != len(queries):
                raise Error("SVC prediction shape differs from complete query batch")
            comptime if is_defined["MOJOLEARN_NATIVE_SVM_BAD_PREDICTION"]():
                margins[0] = Float32(123.0)
            for i in range(len(queries)):
                _prediction(margins[i], queries[i], alpha, kernel)
                _exact(classes[i], Float32(-3.0 if queries[i] < Float32(0.0) else 7.0), "class prediction")
                _digest(digest, margins[i])
                _digest(digest, classes[i])
    print("PASS native_svc", "fits", 3, "query_batches", 6, "digest", digest)


def check_svr() raises:
    var queries: List[Float32] = [-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0]
    var digest = UInt64(1469598103934665603)
    for fixture in range(3):
        var x: List[Float32] = [-1.0, 1.0]
        var targets: List[Float32] = [-1.0, 1.0]
        var kernel = KERNEL_RBF if fixture == 2 else KERNEL_LINEAR
        var C = 4.0 if fixture == 0 else (0.125 if fixture == 1 else 0.25)
        # Epsilon=1/4: linear optimum a=min(C,(1-epsilon)/2).
        # In the bounded-C RBF case, a=C because
        # C < (1-epsilon)/(1-exp(-2)). These optima have b=0.
        var alpha = Float32(0.375 if fixture == 0 else C)
        var model = svr_fit_host(x, targets, 2, 1, kernel, 0.5, C, 0.25, 1e-6, 1000, 20,
                                 List[Float32]())
        _model(x, model.support_matrix, model.support_idx, model.dual_coefs,
               model.n_support, model.n_iter, model.b, alpha, digest)
        for budget in [0.000016, 1.0]:
            var predictions = svr_predict_host(queries, len(queries), 1, model.support_matrix,
                model.dual_coefs, model.n_support, model.b, kernel, 0.5, budget)
            if len(predictions) != len(queries):
                raise Error("SVR prediction shape differs from complete query batch")
            comptime if is_defined["MOJOLEARN_NATIVE_SVM_BAD_PREDICTION"]():
                predictions[0] = Float32(123.0)
            for i in range(len(queries)):
                _prediction(predictions[i], queries[i], alpha, kernel)
                _digest(digest, predictions[i])
    print("PASS native_svr", "fits", 3, "query_batches", 6, "digest", digest)
