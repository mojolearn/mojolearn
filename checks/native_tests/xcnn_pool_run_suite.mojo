# SPDX-License-Identifier: Apache-2.0
"""The pooling and channel-dropout RUN half (`x_cnn/pool_run.mojo`),
reached without Python (lane doorsplit-cnn-xseq, 2026-11-11). One case per
group of RUN functions: literal words and the layers' own refusal texts. The
native entries the RUN calls are covered by `xcnn_entry_table_suite.mojo`;
the index cast's conversions by the hot-path cast suites (the case here
checks only the dtype refusal, so no case nests a parallel cast inside the
case pool).

The cases are independent units and run across the native host worker pool.
"""
from core.host_parallel import host_parallelize
from x_cnn.pool_run import (
    PoolSettings, pool_run_words, pool_run_m, pool_run_out_words, pool_run_flat_rest, pool_run_row_dims,
    pool_run_check_nchw, pool_run_check_ncl, pool_run_check_backward_ncl, pool_run_check_saved,
    pool_run_check_adaptive_saved, pool_run_check_shape, pool_run_check_dim, pool_run_check_divisor,
    pool_run_check_probability, pool_run_check_count, pool_run_check_output, pool_run_cache_matches,
    pool_run_dropout_shape, pool_run_dropout_threshold, pool_run_dropout_words, pool_run_adaptive_divides,
    pool_run_adaptive_kernel, pool_run_adaptive_words, pool_run_cast_indices,
)

comptime XCNN_POOL_RUN_CASES = 12


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _refuses(problem: Error, text: String) raises:
    require(String(problem) == text, "refusal text: got '" + String(problem) + "', expected '" + text + "'")


def _same(got: List[Int], expected: List[Int]) -> Bool:
    if len(got) != len(expected):
        return False
    for i in range(len(expected)):  # driver-loop: over a few literal words
        if got[i] != expected[i]:
            return False
    return True


def run_case(unit: Int) raises:
    if unit == 0:
        var settings = PoolSettings(2, 3, 2, 1, 1, 0, 1, 1, True, False, 4)
        var words = pool_run_words(settings, [1, 2, 6, 5])
        require(_same(words, [1, 2, 6, 5, 2, 3, 2, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 4]), "pool launch words")
        require(_same(pool_run_m(7, [3, 4]), [7, 3, 4]), "_m words")
    elif unit == 1:
        var settings = PoolSettings(2, 2, 2, 2, 0, 0, 1, 1, True, False, 0)
        var words = pool_run_words(settings, [1, 3, 4, 5])
        require(_same(pool_run_out_words(words, [1, 3, 4, 5]), [1, 3, 2, 2]), "floor-mode output shape")
        var ceil = PoolSettings(2, 2, 2, 2, 0, 0, 1, 1, True, True, 0)
        require(_same(pool_run_out_words(pool_run_words(ceil, [1, 3, 4, 5]), [1, 3, 4, 5]), [1, 3, 2, 3]),
                "ceil-mode output shape")
        try:
            var wide = PoolSettings(2, 2, 2, 2, 2, 0, 1, 1, True, False, 0)
            _ = pool_run_out_words(pool_run_words(wide, [1, 3, 4, 5]), [1, 3, 4, 5])
            require(False, "padding wider than half the kernel accepted")
        except problem:
            _refuses(problem, "x_cnn: pad should be at most half of the kernel size (PyTorch's rule)")
    elif unit == 2:
        require(pool_run_flat_rest([4, 3, 2, 5]) == 30 and pool_run_flat_rest([7]) == 1, "sample words")
        require(_same(pool_run_row_dims(8, [3, 2, 2]), [8, 3, 2, 2]), "row input dims")
    elif unit == 3:
        pool_run_check_nchw([1, 2, 3, 4], "MaxPool2d")
        try:
            pool_run_check_nchw([1, 2, 3], "AvgPool2d")
            require(False, "rank-three 2-D pool input accepted")
        except problem:
            _refuses(problem, "mojolearn: AvgPool2d.forward takes (N, C, H, W)")
        try:
            pool_run_check_ncl([1, 2], "MaxPool1d")
            require(False, "rank-two 1-D pool input accepted")
        except problem:
            _refuses(problem, "mojolearn: MaxPool1d.forward takes (N, C, L)")
        try:
            pool_run_check_backward_ncl([1, 2, 3, 4])
            require(False, "rank-four 1-D pool gradient accepted")
        except problem:
            _refuses(problem, "mojolearn: one-dimensional pool backward takes (N, C, L)")
    elif unit == 4:
        try:
            pool_run_check_saved([1, 2, 3])
            require(False, "rank-three saved pool input accepted")
        except problem:
            _refuses(problem, "mojolearn: saved pool input must have four dimensions")
        try:
            pool_run_check_adaptive_saved([1, 2, 3])
            require(False, "rank-three saved adaptive input accepted")
        except problem:
            _refuses(problem, "mojolearn: saved adaptive pool input must have four dimensions")
    elif unit == 5:
        pool_run_check_shape([1, 2, 3], [1, 2, 3])
        try:
            pool_run_check_shape([1, 2], [1, 2, 3])
            require(False, "gradient rank mismatch accepted")
        except problem:
            _refuses(problem, "mojolearn: grad_out rank differs from the pool output")
        try:
            pool_run_check_shape([1, 2, 4], [1, 2, 3])
            require(False, "gradient shape mismatch accepted")
        except problem:
            _refuses(problem, "mojolearn: grad_out shape differs from the pool output")
    elif unit == 6:
        pool_run_check_dim(0)
        pool_run_check_divisor(1)
        try:
            pool_run_check_dim(-1)
            require(False, "negative pool shape accepted")
        except problem:
            _refuses(problem, "mojolearn: negative pool shape")
        try:
            pool_run_check_divisor(0)
            require(False, "zero divisor accepted")
        except problem:
            _refuses(problem, "mojolearn: divisor must be not zero")
    elif unit == 7:
        pool_run_check_probability(0.0)
        pool_run_check_probability(1.0)
        try:
            pool_run_check_probability(1.5)
            require(False, "probability above one accepted")
        except problem:
            _refuses(problem, "mojolearn: dropout probability has to be between 0 and 1")
        try:
            pool_run_check_count(5, 6)
            require(False, "mask size mismatch accepted")
        except problem:
            _refuses(problem, "mojolearn: grad_out does not match the forward's input")
    elif unit == 8:
        require(_same(pool_run_dropout_shape([3, 4, 5]), [1, 3, 4, 5]), "one-sample dropout shape")
        require(_same(pool_run_dropout_shape([2, 3, 4, 5]), [2, 3, 4, 5]), "batch dropout shape")
        try:
            _ = pool_run_dropout_shape([3, 4])
            require(False, "rank-two dropout input accepted")
        except problem:
            _refuses(problem, "mojolearn: Dropout2d.forward takes (N, C, H, W) or (C, H, W)")
    elif unit == 9:
        # round(p * 2**32) with round-to-even at the half words
        require(pool_run_dropout_threshold(1.5 / 4294967296.0) == 2, "half rounds up to even")
        require(pool_run_dropout_threshold(2.5 / 4294967296.0) == 2, "half rounds down to even")
        require(pool_run_dropout_threshold(1.0) == 4294967296, "p = 1")
        var words = pool_run_dropout_words(0.5, UInt64(5) | (UInt64(1) << 31), 2, 2, 3, 20)
        require(_same(words, [2, 3, 20, 5, 1000005, 32768, 0]), "dropout mask words")
    elif unit == 10:
        require(pool_run_adaptive_divides([1, 2, 6, 4], [3, 2]) and not pool_run_adaptive_divides([1, 2, 7, 4], [3, 2]),
                "adaptive divides")
        require(_same(pool_run_adaptive_kernel([1, 2, 6, 4], [3, 2]), [2, 2]), "adaptive kernel")
        require(_same(pool_run_adaptive_words([1, 2, 7, 4], [3, 2], True, False), [1, 2, 7, 4, 3, 2, 2]), "max fwd")
        require(_same(pool_run_adaptive_words([1, 2, 7, 4], [3, 2], False, True), [1, 2, 7, 4, 3, 2, 1]), "avg bwd")
        require(pool_run_adaptive_words([1, 2, 7, 4], [3, 2], True, True)[6] == 3, "max bwd")
        try:
            pool_run_check_output(0, 2)
            require(False, "zero adaptive output accepted")
        except problem:
            _refuses(problem, "mojolearn: adaptive output_size must be positive")
    else:
        require(pool_run_cache_matches([2, 3], [2, 3]) and not pool_run_cache_matches([2, 3], [3, 2]), "cache key")
        try:
            _ = pool_run_cache_matches([2], [2, 3])
            require(False, "one-dimension cache key accepted")
        except problem:
            _refuses(problem, "TypeError: adaptive pool cache must contain two dimensions")
        try:
            pool_run_cast_indices(99, 4, 0, 0, 0)
            require(False, "non-numeric index dtype accepted")
        except problem:
            _refuses(problem, "TypeError: adaptive pool indices require a native numeric dtype")


def run_checks() raises:
    var failures = List[Int](length=XCNN_POOL_RUN_CASES, fill=0)
    var flags = failures.unsafe_ptr()
    def task(unit: Int) {imm flags}:
        try:
            run_case(unit)
        except problem:
            flags[unit] = 1
            print("FAIL x_cnn pool run case", unit, String(problem))
    host_parallelize(task, XCNN_POOL_RUN_CASES)
    var failed = 0
    for failure in failures:  # collect independent case status words
        failed += failure
    _ = failures
    require(failed == 0, "x_cnn pool run cases failed: " + String(failed))
    print("PASS x_cnn pool run", XCNN_POOL_RUN_CASES)
