# SPDX-License-Identifier: Apache-2.0
"""The sequence-expansion score RUN half (`sequence/xseq_score_run.mojo`),
reached without Python (lane doorsplit-cnn-xseq, 2026-11-11). One case per
RUN check: the metrics refusal texts on literal shapes and float32 words.
`xseq_run_r2` reads the metrics image's published R^2 entry and is covered
with that table (`core/metrics_entry_table.mojo`), not here.

The cases are independent units and run across the native host worker pool.
"""
from core.host_parallel import host_parallelize
from sequence.xseq_score_run import (
    xseq_run_vector_rows, xseq_run_vector_finite, xseq_run_lengths_check, xseq_run_weights_check,
    xseq_run_metrics_mode,
)

comptime XSEQ_SCORE_RUN_CASES = 5


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _refuses(problem: Error, text: String) raises:
    require(String(problem) == text, "refusal text: got '" + String(problem) + "', expected '" + text + "'")


def _weights_refusal(shape: List[Int], n: Int, values: List[Float32], text: String) raises:
    try:
        xseq_run_weights_check(shape, n, Int(values.unsafe_ptr()))
        require(False, "weights accepted: " + text)
    except problem:
        _refuses(problem, text)
    _ = values


def run_case(unit: Int) raises:
    if unit == 0:
        require(xseq_run_vector_rows([4], "y_true") == 4 and xseq_run_vector_rows([3, 1], "y_pred") == 3, "vector rows")
        try:
            _ = xseq_run_vector_rows([3, 2], "y_true")
            require(False, "wide target accepted")
        except problem:
            _refuses(problem, "ValueError: mojolearn metrics: y_true must be 1-D, got shape (3, 2)")
        try:
            _ = xseq_run_vector_rows([0], "y_pred")
            require(False, "empty target accepted")
        except problem:
            _refuses(problem, "ValueError: mojolearn metrics: y_pred is empty")
    elif unit == 1:
        var values: List[Float32] = [1.0, 2.0, 3.0]
        xseq_run_vector_finite(Int(values.unsafe_ptr()), 3, "y_true")
        values[1] = Float32(1.0) / Float32(0.0)
        try:
            xseq_run_vector_finite(Int(values.unsafe_ptr()), 3, "y_true")
            require(False, "infinite target accepted")
        except problem:
            _refuses(problem, "ValueError: mojolearn metrics: y_true contains NaN or infinity; refused here "
                     + "(metrics/README.md HAND-OFF ask 3) rather than letting it reach a kernel whose answer "
                     + "for it is a washed sentinel")
        _ = values
    elif unit == 2:
        xseq_run_lengths_check(3, 3)
        try:
            xseq_run_lengths_check(3, 2)
            require(False, "length mismatch accepted")
        except problem:
            _refuses(problem, "ValueError: mojolearn metrics: y_true has 3 entries and y_pred has 2; they must be "
                     + "the same length")
    elif unit == 3:
        var good: List[Float32] = [0.0, 1.0, 2.0]
        xseq_run_weights_check([3], 3, Int(good.unsafe_ptr()))
        _ = good
        _weights_refusal([3, 1], 3, [1.0, 1.0, 1.0],
                         "ValueError: mojolearn r2_score: sample_weight must be 1-D, got shape (3, 1)")
        _weights_refusal([2], 3, [1.0, 1.0],
                         "ValueError: mojolearn r2_score: sample_weight has 2 entries for 3 samples")
        _weights_refusal([2], 2, [1.0, Float32(1.0) / Float32(0.0)],
                         "ValueError: mojolearn r2_score: sample_weight must have finite nonnegative entries")
        _weights_refusal([2], 2, [1.0, -1.0],
                         "ValueError: mojolearn r2_score: sample_weight must have finite nonnegative entries")
        _weights_refusal([2], 2, [0.0, 0.0],
                         "ValueError: mojolearn r2_score: sample_weight must have positive total weight")
    else:
        xseq_run_metrics_mode("identical", 1)
        xseq_run_metrics_mode("fast", 0)
        try:
            xseq_run_metrics_mode("identical", 0)
            require(False, "mode mismatch accepted")
        except problem:
            _refuses(problem, "RuntimeError: mojolearn metrics: requested identical, binary reports fast; rebuild it")


def run_checks() raises:
    var failures = List[Int](length=XSEQ_SCORE_RUN_CASES, fill=0)
    var flags = failures.unsafe_ptr()
    def task(unit: Int) {imm flags}:
        try:
            run_case(unit)
        except problem:
            flags[unit] = 1
            print("FAIL xseq score run case", unit, String(problem))
    host_parallelize(task, XSEQ_SCORE_RUN_CASES)
    var failed = 0
    for failure in failures:  # collect independent case status words
        failed += failure
    _ = failures
    require(failed == 0, "xseq score run cases failed: " + String(failed))
    print("PASS xseq score run", XSEQ_SCORE_RUN_CASES)
