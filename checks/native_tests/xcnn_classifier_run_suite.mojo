# SPDX-License-Identifier: Apache-2.0
"""The CNNClassifier RUN half (`x_cnn/classifier_run.mojo`), reached without
Python (lane doorsplit-cnn-xseq, 2026-11-11). One case per RUN function:
literal words and the classifier's own refusal texts. The whole-model door
the descriptor words feed is covered by the sequence program suites.

The cases are independent units and run across the native host worker pool.
"""
from core.host_parallel import host_parallelize
from x_cnn.classifier_run import (
    CnnClfPlan, CnnClfFitSettings, CNNCLF_PREDICT_ROWS, cnnclf_run_check_input_rank, cnnclf_run_check_input_dim,
    cnnclf_run_check_channels, cnnclf_run_check_setting, cnnclf_run_check_optimizer, cnnclf_run_optimizer_word,
    cnnclf_run_check_nesterov, cnnclf_run_block_plane, cnnclf_run_conv_seed, cnnclf_run_head_seed,
    cnnclf_run_conv_fan_in, cnnclf_run_kaiming_bound, cnnclf_run_check_rows, cnnclf_run_images,
    cnnclf_run_features, cnnclf_run_check_labels, cnnclf_run_fit_steps, cnnclf_run_fit_addresses,
    cnnclf_run_fit_ints, cnnclf_run_fit_floats, cnnclf_run_open_addresses, cnnclf_run_open_ints,
    cnnclf_run_check_session, cnnclf_run_predict_ints,
)

comptime XCNN_CLASSIFIER_RUN_CASES = 22


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _refuses(problem: Error, text: String) raises:
    require(String(problem) == text, "refusal text: got '" + String(problem) + "', expected '" + text + "'")


def _same(got: List[Int], expected: List[Int]) -> Bool:
    if len(got) != len(expected):
        return False
    for i in range(len(got)):  # driver-loop: over a handful of literal descriptor words
        if got[i] != expected[i]:
            return False
    return True


def _plan(n: Int) -> CnnClfPlan:
    return CnnClfPlan(n, 1, 8, 8, 3, 3, 2, [4, 6], [1, 0])


def _settings() -> CnnClfFitSettings:
    var seed = (UInt64(5) << 32) | UInt64(7)
    return CnnClfFitSettings(16, 3, True, seed, 1, False, 0.01, 0.9, 0.8, 0.95, 1e-8, 0.001, 0.0)


def run_case(unit: Int) raises:
    if unit == 0:
        cnnclf_run_check_input_rank(3)
        try:
            cnnclf_run_check_input_rank(2)
            require(False, "rank-two input_shape accepted")
        except problem:
            _refuses(problem, "ValueError: input_shape must contain C, H and W")
    elif unit == 1:
        cnnclf_run_check_input_dim(1)
        try:
            cnnclf_run_check_input_dim(0)
            require(False, "zero image dimension accepted")
        except problem:
            _refuses(problem, "ValueError: input_shape dimensions must be positive")
    elif unit == 2:
        cnnclf_run_check_channels(1)
        try:
            cnnclf_run_check_channels(0)
            require(False, "zero channels accepted")
        except problem:
            _refuses(problem, "ValueError: convolution channels must be positive")
    elif unit == 3:
        cnnclf_run_check_setting(3, 0)
        cnnclf_run_check_setting(0, 1)
        try:
            cnnclf_run_check_setting(2, 0)
            require(False, "zero batch size accepted")
        except problem:
            _refuses(problem, "ValueError: invalid classifier size or iteration setting")
        try:
            cnnclf_run_check_setting(3, -1)
            require(False, "negative max_iter accepted")
        except problem:
            _refuses(problem, "ValueError: invalid classifier size or iteration setting")
    elif unit == 4:
        cnnclf_run_check_optimizer("adamw")
        try:
            cnnclf_run_check_optimizer("rmsprop")
            require(False, "unknown optimizer accepted")
        except problem:
            _refuses(problem, "ValueError: optimizer must be sgd, adam or adamw")
    elif unit == 5:
        require(cnnclf_run_optimizer_word("sgd") == 0 and cnnclf_run_optimizer_word("adam") == 1
                and cnnclf_run_optimizer_word("adamw") == 2, "optimizer words")
    elif unit == 6:
        cnnclf_run_check_nesterov(False, 0.0, 0.5)
        cnnclf_run_check_nesterov(True, 0.9, 0.0)
        try:
            cnnclf_run_check_nesterov(True, 0.9, 0.1)
            require(False, "Nesterov with dampening accepted")
        except problem:
            _refuses(problem, "ValueError: Nesterov momentum requires a momentum and zero dampening")
    elif unit == 7:
        var pooled = cnnclf_run_block_plane(28, 28, 3, 2)
        require(pooled.pools and pooled.h == 14 and pooled.w == 14, "pooled 'same' block plane")
        var even = cnnclf_run_block_plane(5, 5, 4, 2)
        require(even.pools and even.h == 3 and even.w == 3, "even kernel plane")
        var small = cnnclf_run_block_plane(1, 1, 3, 2)
        require(not small.pools and small.h == 1 and small.w == 1, "plane below one pool window")
        var unit_pool = cnnclf_run_block_plane(6, 6, 3, 1)
        require(not unit_pool.pools and unit_pool.h == 6, "pool size one keeps the plane")
    elif unit == 8:
        require(cnnclf_run_conv_seed(5, 2) == 207 and cnnclf_run_head_seed(5) == 1002, "layer seeds")
    elif unit == 9:
        require(cnnclf_run_conv_fan_in(3, 5) == 75, "conv fan-in")
    elif unit == 10:
        require(cnnclf_run_kaiming_bound(4) == 0.5 and cnnclf_run_kaiming_bound(1) == 1.0, "Kaiming bound")
    elif unit == 11:
        cnnclf_run_check_rows([0])
        try:
            cnnclf_run_check_rows(List[Int]())
            require(False, "rank-zero images accepted")
        except problem:
            _refuses(problem, "ValueError: classifier images require a row dimension")
    elif unit == 12:
        var flat = cnnclf_run_images([4, 12], [3, 2, 2])
        require(flat.reshape and _same(flat.full, [4, 3, 2, 2]), "flat rows reshaped")
        var nchw = cnnclf_run_images([4, 3, 2, 2], [3, 2, 2])
        require(not nchw.reshape and _same(nchw.full, [4, 3, 2, 2]), "NCHW images kept")
        try:
            _ = cnnclf_run_images([4, 3, 2], [3, 2, 2])
            require(False, "rank-three images accepted")
        except problem:
            _refuses(problem, "ValueError: classifier images must have rank two or four")
        try:
            _ = cnnclf_run_images([4, 3, 2, 3], [3, 2, 2])
            require(False, "mismatched images accepted")
        except problem:
            _refuses(problem, "ValueError: images do not match the model input_shape")
    elif unit == 13:
        require(cnnclf_run_features([3, 4, 5]) == 60, "n_features_in_")
    elif unit == 14:
        cnnclf_run_check_labels(7, 7)
        try:
            cnnclf_run_check_labels(6, 7)
            require(False, "row mismatch accepted")
        except problem:
            _refuses(problem, "ValueError: X and y must contain the same number of rows")
    elif unit == 15:
        require(cnnclf_run_fit_steps(10, 4) == 3 and cnnclf_run_fit_steps(8, 4) == 2, "steps per epoch")
    elif unit == 16:
        require(_same(cnnclf_run_fit_addresses(1, 2, 3, 4, [5, 6]), [1, 2, 3, 4, 5, 6]), "fit addresses")
    elif unit == 17:
        var ints = cnnclf_run_fit_ints(_plan(10), _settings())
        require(_same(ints, [10, 1, 8, 8, 3, 2, 3, 2, 16, 3, 1, 7, 5, 1, 0, 4, 6, 1, 0]), "fit descriptor words")
    elif unit == 18:
        var floats = cnnclf_run_fit_floats(_settings())
        require(len(floats) == 7 and floats[0] == 0.01 and floats[2] == 0.8 and floats[4] == 1e-8
                and floats[6] == 0.0, "fit descriptor scalars")
    elif unit == 19:
        require(_same(cnnclf_run_open_addresses(9, [5, 6]), [9, 5, 6]), "open addresses")
    elif unit == 20:
        require(_same(cnnclf_run_open_ints(_plan(1)), [1, 1, 8, 8, 3, 2, 3, 2, 4, 6, 1, 0]), "open descriptor words")
    else:
        cnnclf_run_check_session(3, 2)
        try:
            cnnclf_run_check_session(0, 2)
            require(False, "closed session accepted")
        except problem:
            _refuses(problem, "RuntimeError: CNN resident session is closed")
        require(_same(cnnclf_run_predict_ints(1, 2, 3, 40, True), [1, 2, 3, 40, 1, CNNCLF_PREDICT_ROWS]),
                "predict descriptor words")


def run_checks() raises:
    var failures = List[Int](length=XCNN_CLASSIFIER_RUN_CASES, fill=0)
    var flags = failures.unsafe_ptr()
    def task(unit: Int) {imm flags}:
        try:
            run_case(unit)
        except problem:
            flags[unit] = 1
            print("FAIL x_cnn classifier run case", unit, String(problem))
    host_parallelize(task, XCNN_CLASSIFIER_RUN_CASES)
    var failed = 0
    for failure in failures:  # collect independent case status words
        failed += failure
    _ = failures
    require(failed == 0, "x_cnn classifier run cases failed: " + String(failed))
    print("PASS x_cnn classifier run", XCNN_CLASSIFIER_RUN_CASES)
