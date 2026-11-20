# SPDX-License-Identifier: Apache-2.0
"""The CNN DeviceTensor RUN half (`x_cnn/tensor_run.mojo`), reached without
Python (lane doorsplit-cnn-xseq, 2026-11-11). One case per pure RUN
function: literal words and the tensor's own refusal texts. The handle
release (`xcnn_tensor_run_free`) needs a published entry table and is
covered by `xcnn_entry_table_suite.mojo`.

The cases are independent units and run across the native host worker pool.
"""
from core.host_parallel import host_parallelize
from x_cnn.tensor_run import (
    xcnn_tensor_run_check_dim, xcnn_tensor_run_check_handle, xcnn_tensor_run_handle_live,
    xcnn_tensor_run_reshape_dims,
)

comptime XCNN_TENSOR_RUN_CASES = 4


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _refuses(problem: Error, text: String) raises:
    require(String(problem) == text, "refusal text: got '" + String(problem) + "', expected '" + text + "'")


def run_case(unit: Int) raises:
    if unit == 0:
        xcnn_tensor_run_check_dim(0, False)
        xcnn_tensor_run_check_dim(-1, True)
        try:
            xcnn_tensor_run_check_dim(-1, False)
            require(False, "negative dimension accepted")
        except problem:
            _refuses(problem, "ValueError: CNN tensor dimensions must be nonnegative")
        try:
            xcnn_tensor_run_check_dim(-2, True)
            require(False, "dimension below -1 accepted in a reshape")
        except problem:
            _refuses(problem, "ValueError: CNN tensor dimensions must be nonnegative")
    elif unit == 1:
        xcnn_tensor_run_check_handle(1)
        try:
            xcnn_tensor_run_check_handle(0)
            require(False, "zero handle accepted")
        except problem:
            _refuses(problem, "TypeError: CNN tensor handle requires a positive native integer")
    elif unit == 2:
        require(xcnn_tensor_run_handle_live(3) and not xcnn_tensor_run_handle_live(0)
                and not xcnn_tensor_run_handle_live(-4), "live handle words")
    else:
        var dims = xcnn_tensor_run_reshape_dims([4, -1], [2, 3, 4])
        require(len(dims) == 2 and dims[0] == 4 and dims[1] == 6, "inferred reshape")
        var empty = xcnn_tensor_run_reshape_dims([0, -1], [0, 5])
        require(empty[1] == 0, "inferred dimension of an empty tensor")
        try:
            _ = xcnn_tensor_run_reshape_dims([-1, -1], [4])
            require(False, "two inferred dimensions accepted")
        except problem:
            _refuses(problem, "ValueError: CNN reshape accepts only one inferred dimension")


def run_checks() raises:
    var failures = List[Int](length=XCNN_TENSOR_RUN_CASES, fill=0)
    var flags = failures.unsafe_ptr()
    def task(unit: Int) {imm flags}:
        try:
            run_case(unit)
        except problem:
            flags[unit] = 1
            print("FAIL x_cnn tensor run case", unit, String(problem))
    host_parallelize(task, XCNN_TENSOR_RUN_CASES)
    var failed = 0
    for failure in failures:  # collect independent case status words
        failed += failure
    _ = failures
    require(failed == 0, "x_cnn tensor run cases failed: " + String(failed))
    print("PASS x_cnn tensor run", XCNN_TENSOR_RUN_CASES)
