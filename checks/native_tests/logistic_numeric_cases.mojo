# SPDX-License-Identifier: Apache-2.0
"""Original LogisticRegression numerical workloads through typed native requests.

Fit inputs are recorded source fixtures, not fitted reference outputs. The two
fit settings and Float64 probability bound are unchanged. Separate exact states
cover first-max ties and signed-zero binary decisions. All input/output storage
is owned by native Lists across the synchronous descriptor calls.
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.logistic_native_api import (
    QnFitRequest, QnQueryRequest, QnNativeDriver, QN_NATIVE_DECISION,
    QN_NATIVE_BINARY, QN_NATIVE_PROBA,
)
from checks.native_tests.logistic_numeric_fixtures import (
    LOGISTIC_GUARD, require, original_inputs, independent_links, planted_accuracy,
)

comptime LOGISTIC_NUMERIC_FIXTURES = 4
comptime CODE_GUARD = Int64(-192837)


trait LogisticPrediction:
    @staticmethod
    def codes(driver: QnNativeDriver, request: QnQueryRequest, scores: Int) raises: ...


def same_f32(actual: List[Float32], expected: List[Float32], label: String) raises:
    require(len(actual) == len(expected), label + " extent")
    var ap = Pointer[UInt32, ImmUntrackedOrigin](unsafe_from_address=Int(actual.unsafe_ptr()))
    var ep = Pointer[UInt32, ImmUntrackedOrigin](unsafe_from_address=Int(expected.unsafe_ptr()))
    var failures = List[Int](length=len(actual), fill=0)
    var fp = failures.unsafe_ptr()
    def cell(index: Int) {imm ap, imm ep, imm fp}:
        if ap[index] != ep[index]:
            fp.unsafe_store(index, 1)
    host_parallelize(cell, len(actual))
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, label + " was mutated")
    _ = actual
    _ = expected


def fixture[P: LogisticPrediction](driver: QnNativeDriver, fixture_id: Int) raises:
    var rows = 600 if fixture_id == 0 else (200 if fixture_id == 1 else (64 if fixture_id == 2 else 6))
    var classes = 3 if fixture_id == 0 or fixture_id == 2 else 2
    var features = 2 if fixture_id == 3 else 4
    var intercept = fixture_id != 3
    var columns = classes if classes > 2 else 1
    var x = List[Float32]()
    var y = List[Float32]()
    if fixture_id < 3:
        var recorded = original_inputs(rows)
        x = recorded[0].copy()
        y = recorded[1].copy()
        if fixture_id == 1:
            var yp = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(y.unsafe_ptr()) + 4)
            def binary_target(index: Int) {imm yp}:
                yp[index] = Float32(1) if yp[index] == 1 else Float32(0)
            host_parallelize(binary_target, rows)
            _ = y
    else:
        # Literal source of test_logistic_native_predict's strict-score seam.
        x = [LOGISTIC_GUARD, -2, 0, bitcast[DType.float32](UInt32(0x80000000)), 0,
             0, 0, 2, 0, 1, -1, -1, 1, LOGISTIC_GUARD]
        y = [LOGISTIC_GUARD, 0, 0, 0, 1, 1, 0, LOGISTIC_GUARD]
    var source_x = x.copy()
    var source_y = y.copy()
    var coefficients = List[Float32](length=(features + Int(intercept)) * columns + 2, fill=0)
    coefficients[0] = LOGISTIC_GUARD
    coefficients[len(coefficients) - 1] = LOGISTIC_GUARD
    var info = List[Float32](length=4, fill=LOGISTIC_GUARD)
    var status: List[Int32] = [765432, 0, 765432]
    var status_address = Int(status.unsafe_ptr()) + 4
    var xp = Int(x.unsafe_ptr()) + 4
    var yp = Int(y.unsafe_ptr()) + 4
    var wp = Int(coefficients.unsafe_ptr()) + 4
    if fixture_id < 2:
        var max_iter = 100 if fixture_id == 0 else 50
        var request = QnFitRequest(xp, yp, wp, Int(info.unsafe_ptr()) + 4, 0, status_address,
            rows, features, classes, max_iter, 50, 5, True, True,
            2 if classes > 2 else 0, Float64(0), Float64(1), Float64(1e-4), Float64(1e-4) * Float64(0.01), Float64(0))
        _ = driver.fit(request)
        require(status[1] == 0, "finite original fit input was refused")
        if fixture_id == 0:
            require(info[2] == 0, "original multiclass fit retcode is not zero")
    elif fixture_id == 3:
        coefficients[1] = 1
        coefficients[2] = -1
    var source_coefficients = coefficients.copy()
    var scores = List[Float32](length=rows * columns + 2, fill=LOGISTIC_GUARD)
    var probabilities = List[Float64](length=rows * classes + 2, fill=Float64(LOGISTIC_GUARD))
    var codes = List[Int64](length=rows + 2, fill=CODE_GUARD)
    var sp = Int(scores.unsafe_ptr()) + 4
    var pp = Int(probabilities.unsafe_ptr()) + 8
    var cp = Int(codes.unsafe_ptr()) + 8
    driver.query(QN_NATIVE_DECISION, QnQueryRequest(xp, wp, sp, status_address, rows, features, classes, intercept, False))
    require(status[1] == 0, "finite decision input was refused")
    driver.query(QN_NATIVE_PROBA, QnQueryRequest(xp, wp, pp, status_address, rows, features, classes, intercept, False))
    require(status[1] == 0, "finite probability input was refused")
    var prediction = QnQueryRequest(xp, wp, cp, status_address, rows, features, classes, intercept, False)
    if classes == 2:
        driver.query(QN_NATIVE_BINARY, prediction)
    else:
        P.codes(driver, prediction, sp)
    require(status[1] == 0, "finite prediction input was refused")
    independent_links(scores, probabilities, codes, rows, classes)
    if fixture_id == 0:
        planted_accuracy(codes, y, rows)
    if fixture_id >= 2:
        var expected_scores: List[Float32] = [-2, 0, 0, 2, 2, -2]
        var ep = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=Int(expected_scores.unsafe_ptr()))
        var sb = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=sp)
        var pb = Pointer[Float64, ImmUntrackedOrigin](unsafe_from_address=pp)
        var cb = Pointer[Int64, ImmUntrackedOrigin](unsafe_from_address=cp)
        var failures = List[Int](length=rows, fill=0)
        var fp = failures.unsafe_ptr()
        def exact_row(row: Int) {imm sb, imm pb, imm cb, imm ep, imm fp, imm fixture_id}:
            if fixture_id == 2:
                if cb[row] != 0:
                    fp.unsafe_store(row, 1)
                for column in range(3):
                    if sb[row * 3 + column] != 0 or pb[row * 3 + column] != Float64(1) / 3:
                        fp.unsafe_store(row, 1)
            elif sb[row] != ep[row] or cb[row] != Int64(1 if row == 3 or row == 4 else 0):
                fp.unsafe_store(row, 1)
        host_parallelize(exact_row, rows)
        var failed = 0
        for value in failures:
            failed += value
        require(failed == 0, "exact logistic state/decision differs")
        _ = expected_scores
    require(coefficients[0] == LOGISTIC_GUARD and coefficients[len(coefficients) - 1] == LOGISTIC_GUARD,
        "fit coefficient output guard")
    require(info[0] == LOGISTIC_GUARD and info[3] == LOGISTIC_GUARD, "fit info output guard")
    require(status[0] == 765432 and status[2] == 765432, "finite status output guard")
    require(scores[0] == LOGISTIC_GUARD and scores[len(scores) - 1] == LOGISTIC_GUARD, "decision output guard")
    require(probabilities[0] == Float64(LOGISTIC_GUARD) and probabilities[len(probabilities) - 1] == Float64(LOGISTIC_GUARD),
        "probability output guard")
    require(codes[0] == CODE_GUARD and codes[len(codes) - 1] == CODE_GUARD, "prediction output guard")
    same_f32(x, source_x, "feature source including signed zero")
    same_f32(y, source_y, "target source")
    same_f32(coefficients, source_coefficients, "fitted state during inference")
    _ = info
    _ = scores
    _ = probabilities
    _ = codes
