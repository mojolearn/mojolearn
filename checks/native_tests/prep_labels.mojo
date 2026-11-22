# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""LabelEncoder, LabelBinarizer and MultiLabelBinarizer programs against scikit-learn's values (lane py2mojo-B,
2026-11-22; ported from python/mojolearn/tests/test_x_prep_labels.py `test_label_encoder`, `test_label_binarizer`,
`test_multilabel_binarizer` and test_x_prep_parity.py `test_multilabel_binarizer`).

The inputs are the original tests' NumPy draws and literals and the references the scikit-learn calls they made,
stored as words in `fixtures/prep_lab/` by `tools/py2mojo_native_fixtures_b2.py`. The programs are the typed label
builders (bindings/prep_labels_prog_run.mojo) with the requests the label door makes
(bindings/prep_labels_native_door.mojo): the fit of a numeric label buffer by presence flags first (the identical
image exports `x_prep_label_present`) and by the sort when the BAD flag says the labels are not small exact
integers (`_label_fit_device`); LabelEncoder.transform and inverse_transform of the label and code buffers;
LabelBinarizer.transform of the label buffer (two classes or more) or of the looked-up labels (one class),
inverse_transform by the threshold or the block argmax with the classes gathered (one class: its word twice);
the multilabel indicator's transform and inverse; MultiLabelBinarizer's flat int64 label sets (`_mlb_flat`) fitted
by the same label fit and transformed by row_ones. Everything is exact (codes, indicators, classes).

Stays in Python: the str and object labels (their table and codes are built at UNPACK), `classes_` of a multilabel
indicator (a range), the dtypes and shapes of the results (the public boundary). Every case is an independent unit;
the case tables run across host cores.
"""
from std.math import isnan
from std.memory import bitcast
from bindings.prep_encode_prog_run import EncProgram
from bindings.prep_labels_prog_run import (
    LabelWords, LabelCats, LabelCodes, LabelUnbinarize, LAB_CODES_LOOKUP, LAB_Y_MULTILABEL, LAB_Y_MULTICLASS,
    LAB_Y_BINARY, lab_fit_stage, lab_transform_stage, lab_inverse_stage, lab_binarize_buffer_stage,
    lab_binarize_codes_stage, lab_unbinarize_stage, lab_indicator_stage, lab_row_ones_stage,
)
from x_prep.label_fast import IDN_LABEL
from checks.native_tests.prep_enc_host import enc_host_run
from checks.native_tests.prep_cls_host import arena_words, arena_codes
from checks.native_tests.py2mojo_fixture_words import fixture_f64
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require, fixture_words, fixture_f32, fixture_i32

comptime LAB_DIR = "checks/native_tests/fixtures/prep_lab/"
comptime KIND_F32 = 0   # `_lab_spec`: <f4
comptime KIND_I32 = 1   # <i4
comptime KIND_I64 = 3   # <i8 (two words a label)


def raw_words(path: String, count: Int) raises -> List[Float32]:
    """A fixture's raw 32-bit words as the arena words of an input record."""
    var words = fixture_words(path, count)
    var out = List[Float32](length=count, fill=0)
    for i in range(count):  # small-loop(fixture: the raw label words)
        out[i] = bitcast[DType.float32](words[i])
    return out^


def word64(arena: List[Float32], at: Int) -> UInt64:
    return UInt64(bitcast[DType.uint32](arena[at])) | (UInt64(bitcast[DType.uint32](arena[at + 1])) << UInt64(32))


def label_fit(words: List[Float32], kind: Int, n: Int, nw: Int) raises -> List[Float32]:
    """`_label_fit_device`: the presence route, else the sort; the sorted distinct float32 class words."""
    var owners = List[List[Float32]]()
    owners.append(words.copy())
    comptime if IDN_LABEL:
        var prog = EncProgram(False)
        var plan = lab_fit_stage(prog, LabelWords(kind, n, nw, 0), True, False, False)
        var arena = enc_host_run(prog, owners)
        if bitcast[DType.int32](arena[plan.bad]) == 0:
            var K = Int(arena[plan.k])
            require(K > 0 and not isnan(arena[plan.u + K - 1]), "the presence fit found no class")
            return arena_words(arena, plan.u, K)
    var prog = EncProgram(False)
    var plan = lab_fit_stage(prog, LabelWords(kind, n, nw, 0), False, False, False)
    var arena = enc_host_run(prog, owners)
    var K = Int(arena[plan.k])
    require(K > 0 and not isnan(arena[plan.u + K - 1]), "a label has no exact float32 word")
    return arena_words(arena, plan.u, K)


def check_classes(got: List[Float32], tag: String, label: String) raises:
    var want = fixture_f64(LAB_DIR + tag + "-classes.f64", len(got))
    for i in range(len(got)):  # small-loop(oracle: the classes, exact)
        require(Float64(got[i]) == want[i], label + ": classes_[" + String(i) + "] " + String(got[i]) + " expected "
            + String(want[i]))


def check_i32(got: List[Int32], path: String, label: String) raises:
    var want = fixture_i32(path, len(got))
    for i in range(len(got)):  # small-loop(oracle: exact codes or indicators)
        require(got[i] == want[i], label + " word " + String(i) + ": " + String(got[i]) + " expected "
            + String(want[i]))


def check_class_words(arena: List[Float32], at: Int, n: Int, ints: Bool, want: List[Float64], label: String) raises:
    """The 64-bit class words of a gather against the reference's labels."""
    for i in range(n):  # small-loop(oracle: one class word per row, exact)
        var w = word64(arena, at + 2 * i)
        var got = Float64(bitcast[DType.int64](w)) if ints else bitcast[DType.float64](w)
        require(got == want[i], label + " row " + String(i) + ": " + String(got) + " expected " + String(want[i]))


# ---- test_label_encoder: int64 labels (multiples of three, negative too) and float32 labels ----
def encoder_unit(unit: Int) raises:
    var tag = "le" + String(unit)
    var ints = unit == 0
    var n = 200 if ints else 100
    var kind = KIND_I64 if ints else KIND_F32
    var nw = 2 * n if ints else n
    var y = raw_words(LAB_DIR + tag + ("-y.i64" if ints else "-y.f32"), nw)
    var classes = label_fit(y, kind, n, nw)
    var K = Int(fixture_i32(LAB_DIR + tag + "-k.i32", 1)[0])
    require(len(classes) == K, tag + ": " + String(len(classes)) + " classes, scikit-learn " + String(K))
    check_classes(classes, tag, tag)
    # transform: each label's index among the classes, and no chunk with an unknown label
    var prog = EncProgram(False)
    var plan = lab_transform_stage(prog, LabelWords(kind, n, nw, 0), LabelCats(1, K, K), False)
    var owners = List[List[Float32]]()
    owners.append(y.copy())
    owners.append(classes.copy())
    var arena = enc_host_run(prog, owners)
    for c in range(plan.chunks):  # small-loop(oracle: the per-chunk unknown flags)
        require(arena[plan.neg + c] == 0, tag + ": transform found an unknown label")
    var codes = arena_codes(arena, plan.codes, n)
    check_i32(codes, LAB_DIR + tag + "-transform.i32", tag + ": transform")
    # inverse_transform of the int32 codes: the labels themselves
    var inv = EncProgram(False)
    var gather = lab_inverse_stage(inv, LabelWords(KIND_I32, n, n, 0), LabelCats(1, K, K), ints)
    var owners2 = List[List[Float32]]()
    var code_words = List[Float32](length=n, fill=0)
    for i in range(n):  # small-loop(fixture: the codes as int32 words)
        code_words[i] = bitcast[DType.float32](codes[i])
    owners2.append(code_words^)
    owners2.append(classes.copy())
    var back = enc_host_run(inv, owners2)
    require(back[gather.bad] == 0, tag + ": inverse_transform found a code outside the classes")
    var want = List[Float64](length=n, fill=0)
    for i in range(n):  # small-loop(oracle: the original labels)
        if ints:
            want[i] = Float64(bitcast[DType.int64](word64(y, 2 * i)))
        else:
            want[i] = Float64(y[i])
    check_class_words(back, gather.words, n, ints, want, tag + ": inverse_transform")


def test_label_encoder() raises:
    check_cases("label_encoder", 2, encoder_unit)


# ---- test_label_binarizer: the three int64 label vectors (four classes, two, one) x (neg 0, pos 1), (neg -1, pos 3) ----
def binarizer_unit(unit: Int) raises:
    var which: List[Int] = [0, 1, 3]
    var i = which[unit // 2]
    var k = unit % 2
    var tag = "lb" + String(i) + String(k)
    var neg = 0 if k == 0 else -1
    var pos = 1 if k == 0 else 3
    var shape = fixture_i32(LAB_DIR + tag + "-shape.i32", 3)
    var n = Int(shape[0])
    var W = Int(shape[1])
    var y = raw_words(LAB_DIR + tag + "-y.i64", 2 * n)
    var classes = label_fit(y, KIND_I64, n, 2 * n)
    var K = len(classes)
    require(K == Int(shape[2]), tag + ": " + String(K) + " classes, scikit-learn " + String(shape[2]))
    check_classes(classes, tag, tag)
    require(W == (1 if K <= 2 else K), tag + ": the indicator width")
    var prog = EncProgram(False)
    var owners = List[List[Float32]]()
    owners.append(y.copy())
    owners.append(classes.copy())
    var out: Int
    if K > 1:
        out = lab_binarize_buffer_stage(prog, LabelWords(KIND_I64, n, 2 * n, 0), LabelCats(1, K, K), K, W, neg, pos,
            False)
    else:
        # one class: the labels' float values are looked up (`_list_codes`), then every code is absent
        var values = List[Float64](length=n, fill=0)
        for r in range(n):  # small-loop(fixture: the labels' float values)
            values[r] = Float64(bitcast[DType.int64](word64(y, 2 * r)))
        out = lab_binarize_codes_stage(prog, LabelCodes(LAB_CODES_LOOKUP, n, -1, values^, LabelCats(1, K, K)), K, W,
            neg, pos, False)
    var arena = enc_host_run(prog, owners)
    check_i32(arena_codes(arena, out, n * W), LAB_DIR + tag + "-transform.i32", tag + ": transform")
    # inverse_transform of Y, of Y plus noise, of the noisy Y at threshold 0.4
    var table = classes.copy()
    if K <= 1:
        table.append(classes[0])   # one class: both codes name it
    var Y = List[Float32](length=n * W, fill=0)
    var T = fixture_i32(LAB_DIR + tag + "-transform.i32", n * W)
    for r in range(n * W):  # small-loop(fixture: the reference's indicator as float words)
        Y[r] = Float32(T[r])
    var y2 = fixture_f32(LAB_DIR + tag + "-y2.f32", n * W)
    var halfway = Float64(pos + neg) / 2.0
    for call in range(3):  # small-loop(fixture: three inverse calls)
        var z = Y.copy() if call == 0 else y2.copy()
        var names: List[String] = ["inv-y", "inv-y2", "inv-y2t"]
        var t = 0.4 if call == 2 else halfway
        var target = LAB_Y_BINARY if K <= 2 else LAB_Y_MULTICLASS
        var p2 = EncProgram(False)
        var plan = lab_unbinarize_stage(p2, LabelUnbinarize(target, n, W, K, 0, t, True,
            LabelCats(1, len(table), len(table)), True))
        var o2 = List[List[Float32]]()
        o2.append(z^)
        o2.append(table.copy())
        var back = enc_host_run(p2, o2)
        require(back[plan.bad] == 0, tag + ": inverse_transform found a code outside the classes")
        check_class_words(back, plan.words, n, True, fixture_f64(LAB_DIR + tag + "-" + names[call] + ".f64", n),
            tag + ": " + names[call])


def test_label_binarizer() raises:
    check_cases("label_binarizer", 6, binarizer_unit)


# ---- test_multilabel_binarizer: [[1, 3], [2], [], [3, 1, 7], [7, 7]] as the flat int64 sets and row offsets ----
def test_multilabel_binarizer() raises:
    var labels: List[Int64] = [1, 3, 2, 3, 1, 7, 7, 7]
    var offsets: List[Int32] = [0, 2, 3, 3, 6, 8]
    var t = len(labels)
    var rows = len(offsets) - 1
    var words = List[Float32](length=2 * t, fill=0)
    for i in range(t):  # small-loop(fixture: eight literal labels as int64 words)
        var w = bitcast[DType.uint64](labels[i])
        words[2 * i] = bitcast[DType.float32](UInt32(w & UInt64(0xFFFFFFFF)))
        words[2 * i + 1] = bitcast[DType.float32](UInt32(w >> UInt64(32)))
    var classes = label_fit(words, KIND_I64, t, 2 * t)
    var K = len(classes)
    require(K == 4, "multilabel: " + String(K) + " classes, scikit-learn 4")
    check_classes(classes, "mlb", "multilabel")
    var offs = List[Float32](length=rows + 1, fill=0)
    for i in range(rows + 1):  # small-loop(fixture: six literal row offsets)
        offs[i] = bitcast[DType.float32](offsets[i])
    var prog = EncProgram(False)
    var out = lab_row_ones_stage(prog, LabelWords(KIND_I64, t, 2 * t, 0), LabelCats(1, K, K), rows, 2, K, False)
    var owners = List[List[Float32]]()
    owners.append(words^)
    owners.append(classes.copy())
    owners.append(offs^)
    var arena = enc_host_run(prog, owners)
    check_i32(arena_codes(arena, out, rows * K), LAB_DIR + "mlb-transform.i32", "multilabel: transform")


# ---- test_x_prep_parity.py::test_multilabel_binarizer: a 50 x 4 multilabel indicator x three label pairs ----
def indicator_unit(unit: Int) raises:
    var negs: List[Int] = [0, -1, -2]
    var poss: List[Int] = [1, 2, 0]
    var n = 50
    var K = 4
    var tag = "ind" + String(unit)
    var Y = fixture_f32(LAB_DIR + "ind-y.f32", n * K)
    var prog = EncProgram(False)
    var out = lab_indicator_stage(prog, 0, n * K, negs[unit], poss[unit], False)
    var owners = List[List[Float32]]()
    owners.append(Y^)
    var arena = enc_host_run(prog, owners)
    check_i32(arena_codes(arena, out, n * K), LAB_DIR + tag + "-transform.i32", tag + ": transform")
    var S = fixture_f32(LAB_DIR + tag + "-s.f32", n * K)
    var halfway = Float64(poss[unit] + negs[unit]) / 2.0
    for call in range(2):  # small-loop(fixture: the default threshold and 0.3)
        var p2 = EncProgram(False)
        var plan = lab_unbinarize_stage(p2, LabelUnbinarize(LAB_Y_MULTILABEL, n, K, K, 0,
            halfway if call == 0 else 0.3, False, LabelCats(-1, 0, 0), False))
        var o2 = List[List[Float32]]()
        o2.append(S.copy())
        var back = enc_host_run(p2, o2)
        check_i32(arena_codes(back, plan.words, n * K), LAB_DIR + tag + ("-inv.i32" if call == 0 else "-inv3.i32"),
            tag + ": inverse_transform threshold call " + String(call))


def test_multilabel_indicator() raises:
    check_cases("multilabel_indicator", 3, indicator_unit)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
