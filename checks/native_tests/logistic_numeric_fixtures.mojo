# SPDX-License-Identifier: Apache-2.0
"""Recorded original logistic inputs and independent numerical expectations.

The binary files contain only the original Python fixture's input X/class codes,
not fitted outputs or reference hashes. Their source and encoding are recorded
in fixtures/logistic-inputs.json. No Python is used by the native checks.
"""
from std.math import exp
from std.memory import bitcast
from core.host_parallel import host_parallelize

comptime LOGISTIC_GUARD = Float32(27182.5)


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def original_inputs(rows: Int) raises -> Tuple[List[Float32], List[Float32]]:
    var features = List[Float32](length=rows * 4 + 2, fill=LOGISTIC_GUARD)
    var labels = List[Float32](length=rows + 2, fill=LOGISTIC_GUARD)
    with open("checks/native_tests/fixtures/logistic-" + String(rows) + "x4-seed0.bin", "r") as file:
        var bytes = file.read_bytes()
        require(len(bytes) == rows * 5 * 4, "recorded logistic input extent differs")
        var bp = Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=Int(bytes.unsafe_ptr()))
        var xp = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(features.unsafe_ptr()) + 4)
        var yp = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(labels.unsafe_ptr()) + 4)
        def cell(index: Int) {imm bp, imm xp, imm yp, imm rows}:
            var offset = index * 4
            var word = UInt32(bp[offset]) | (UInt32(bp[offset + 1]) << 8) | (UInt32(bp[offset + 2]) << 16) | (UInt32(bp[offset + 3]) << 24)
            if index < rows * 4:
                xp[index] = bitcast[DType.float32](word)
            else:
                yp[index - rows * 4] = bitcast[DType.float32](word)
        host_parallelize(cell, rows * 5)
        _ = bytes
        _ = features
        _ = labels
    return (features^, labels^)


def independent_links(scores: List[Float32], probabilities: List[Float64], codes: List[Int64],
    rows: Int, classes: Int) raises:
    """Independent Float64 sigmoid/softmax and lowest-index maximum equations.

Keep the original multiclass absolute probability/row-sum bound1e-12. Kernel
scores are inputs to this independent link equation, never re-fit references.
"""
    var columns = 1 if classes == 2 else classes
    require(len(scores) == rows * columns + 2 and len(probabilities) == rows * classes + 2
        and len(codes) == rows + 2, "logistic result storage extent")
    var sp = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=Int(scores.unsafe_ptr()) + 4)
    var pp = Pointer[Float64, ImmUntrackedOrigin](unsafe_from_address=Int(probabilities.unsafe_ptr()) + 8)
    var cp = Pointer[Int64, ImmUntrackedOrigin](unsafe_from_address=Int(codes.unsafe_ptr()) + 8)
    var failures = List[Int](length=rows, fill=0)
    var fp = failures.unsafe_ptr()
    def row(index: Int) {imm sp, imm pp, imm cp, imm fp, imm classes, imm columns}:
        var expected_code = 0
        var sum = Float64(0)
        var maximum = Float64(sp[index * columns])
        if classes == 2:
            expected_code = 1 if maximum > 0 else 0
            var positive = Float64(0)
            if maximum >= 0:
                positive = 1 / (1 + exp(-maximum))
            else:
                var e = exp(maximum)
                positive = e / (1 + e)
            if not (abs(pp[index * 2 + 1] - positive) < 1e-12
                and abs(pp[index * 2] - (1 - positive)) < 1e-12):
                fp.unsafe_store(index, 1)
        else:
            for column in range(1, classes):  # Dependent first-maximum scan over one class row.
                var score = Float64(sp[index * columns + column])
                if score > maximum:
                    maximum = score
                    expected_code = column
            var denominator = Float64(0)
            for column in range(classes):
                denominator += exp(Float64(sp[index * columns + column]) - maximum)
            for column in range(classes):
                var expected = exp(Float64(sp[index * columns + column]) - maximum) / denominator
                if not (abs(pp[index * classes + column] - expected) < 1e-12):
                    fp.unsafe_store(index, 1)
        var probability_argmax = 0
        for column in range(classes):  # Dependent normalization and first-max over one probability row.
            var probability = pp[index * classes + column]
            if not (probability >= 0 and probability <= 1):
                fp.unsafe_store(index, 1)
            sum += probability
            if probability > pp[index * classes + probability_argmax]:
                probability_argmax = column
        if not (abs(sum - 1) < 1e-12) or cp[index] != Int64(expected_code):
            fp.unsafe_store(index, 1)
        # Binary uses strict score>0 even when extreme rounding saturates both
        # links; the multiclass contract compares both first-maximum routes.
        if classes > 2 and cp[index] != Int64(probability_argmax):
            fp.unsafe_store(index, 1)
    host_parallelize(row, rows)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "independent logistic link/argmax differs in " + String(failed) + " rows")
    _ = scores
    _ = probabilities
    _ = codes


def planted_accuracy(codes: List[Int64], labels: List[Float32], rows: Int) raises:
    var cp = Pointer[Int64, ImmUntrackedOrigin](unsafe_from_address=Int(codes.unsafe_ptr()) + 8)
    var yp = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=Int(labels.unsafe_ptr()) + 4)
    var correct = List[Int](length=rows, fill=0)
    var rp = correct.unsafe_ptr()
    def row(index: Int) {imm cp, imm yp, imm rp}:
        if cp[index] == Int64(yp[index]):
            rp.unsafe_store(index, 1)
    host_parallelize(row, rows)
    var count = 0
    for value in correct:
        count += value
    require(Float64(count) / Float64(rows) > 0.9, "original planted logistic training accuracy is not above0.9")
    _ = codes
    _ = labels
