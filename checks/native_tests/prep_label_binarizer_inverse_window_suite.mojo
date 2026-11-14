# SPDX-License-Identifier: Apache-2.0
"""LabelBinarizer inverse programs never read or write beyond their window.

Ported from python/mojolearn/tests/test_label_binarizer_inverse_window.py
(`test_binary_one_and_two_columns`, `test_single_class_two_columns`,
`test_multiclass_inverse_unchanged`). Each fixture stages the program that
`bindings/prep_labels_native_door.mojo::_lb_inverse` stages for integer
classes (c2_bin_code 242 or block_argmax 76, then f2_code_gather 241 and
count_neg 8) on an exact-size arena with one guard word past its end, runs it
through the x_prep host program runner and checks the int64 class words, the
untouched scores and the guard. Scores, thresholds and classes are the
original literals; the threshold is the binding's default (neg_label +
pos_label) / 2 unless the original passed one.

The fixtures are independent units and run across host cores.
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.preprocessing_program import prep_program_stage
from x_prep.host.program import run_program_host

comptime LB_INVERSE_FIXTURES = 5
comptime GUARD = Float32(-918273.5)
comptime OP_COUNT_NEG = 8
comptime OP_BLOCK_ARGMAX = 76
comptime OP_F2_CODE_GATHER = 241
comptime OP_C2_BIN_CODE = 242


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def run_inverse(scores: List[Float32], n: Int, W: Int, multiclass: Bool, threshold: Float32,
                table: List[Float32], expected: List[Int64], label: String) raises:
    """One inverse program over an exact window: X, THR, START, WIDTH, codes,
    class table, int64 out, bad flags, the bad count, then one guard word."""
    require(len(scores) == n * W and len(expected) == n, label + ": fixture extent")
    var K = len(table)
    var thr = n * W
    var start = thr + 1
    var width = thr + 2
    var codes = thr + 3
    var cats = codes + n
    var out = cats + K
    var bad = out + 2 * n
    var neg = bad + n
    var guard = neg + 1
    var arena = List[Float32](length=guard + 1, fill=0)
    for i in range(n * W):  # small-loop(fixture: literal scores of at most nine words)
        arena[i] = scores[i]
    arena[thr] = threshold
    arena[start] = 0
    arena[width] = Float32(W)
    for k in range(K):  # small-loop(fixture: at most three class words)
        arena[cats + k] = table[k]
    arena[guard] = GUARD
    var program = List[Int32]()
    if multiclass:
        prep_program_stage(program, OP_BLOCK_ARGMAX, n, 0, n, W, 1, start, width, -1, 0, codes)
    else:
        prep_program_stage(program, OP_C2_BIN_CODE, n, 0, n, W, thr, codes)
    prep_program_stage(program, OP_F2_CODE_GATHER, n, codes, K, cats, 0, out, bad)
    prep_program_stage(program, OP_COUNT_NEG, 1, bad, n, 1, neg)
    run_program_host(Int(arena.unsafe_ptr()), len(arena), Int(program.unsafe_ptr()), 3)
    var words = arena.unsafe_ptr().bitcast[UInt32]()
    for i in range(n):  # small-loop(fixture: at most four rows)
        var lo = UInt64(words[out + 2 * i])
        var hi = UInt64(words[out + 2 * i + 1])
        var got = bitcast[DType.int64](lo | (hi << UInt64(32)))
        require(got == expected[i], label + ": row " + String(i) + " class " + String(got)
            + " expected " + String(expected[i]))
    require(arena[neg] == 0, label + ": a code was outside the class table")
    for i in range(n * W):  # small-loop(fixture: the scores are inputs only)
        require(bitcast[DType.uint32](arena[i]) == bitcast[DType.uint32](scores[i]), label + ": scores were written")
    require(bitcast[DType.uint32](arena[guard]) == bitcast[DType.uint32](GUARD),
        label + ": the program wrote past its window")
    _ = arena^
    _ = program^


def fixture(unit: Int) raises:
    # LabelBinarizer(neg_label=-1, pos_label=2).fit([3, 7, 3, 7]): threshold 0.5.
    var classes: List[Float32] = [3, 7]
    if unit == 0:
        # One column of scores.
        run_inverse([-1, 0.5, 0.75, 2], 4, 1, False, 0.5, classes, [3, 3, 7, 7], "binary one column")
    elif unit == 1:
        # Two columns: only the last column is read.
        run_inverse([99, -1, -99, 0.5, 0, 0.75, 1, 2], 4, 2, False, 0.5, classes, [3, 3, 7, 7],
            "binary two columns")
    elif unit == 2:
        run_inverse([99, -1, -99, 0.5, 0, 0.75, 1, 2], 4, 2, False, 1, classes, [3, 3, 3, 7],
            "binary two columns, threshold 1")
    elif unit == 3:
        # LabelBinarizer().fit([11, 11]): one class; both codes name it (the
        # binding doubles the table), default threshold (0 + 1) / 2.
        run_inverse([0, 1, 1, 0], 2, 2, False, 0.5, [11, 11], [11, 11], "single class two columns")
    else:
        # LabelBinarizer().fit([3, 7, 11]): first maximum per row.
        run_inverse([0, 2, 1, 4, 0, -1, 0, 1, 2], 3, 3, True, 0.5, [3, 7, 11], [7, 3, 11],
            "multiclass inverse")


def run_checks() raises:
    var failures = List[Int](length=LB_INVERSE_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL prep_label_binarizer_inverse_window fixture", unit, String(error))
    host_parallelize(task, LB_INVERSE_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    _ = failures
    require(failed == 0, "prep_label_binarizer_inverse_window failed " + String(failed) + " fixtures")
    print("PASS prep_label_binarizer_inverse_window:", LB_INVERSE_FIXTURES, "inverse programs")
