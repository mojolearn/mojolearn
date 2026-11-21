# SPDX-License-Identifier: Apache-2.0
"""RUN of the label owner (LabelEncoder, LabelBinarizer, MultiLabelBinarizer):
the typed requests and the stage builders of every x_prep program the label
door runs (lane dw-prep-encode, 2026-11-21).

The door (bindings/prep_labels_native_door.mojo) is UNPACK (the estimator's
fields and the caller's labels read once into the values below; every caller
buffer becomes an owner index of the door's `EncProg`), then one stage
builder here on the typed `EncProgram` (bindings/prep_encode_prog_run.mojo),
then the typed run, then PACK (the arena words read back into the result).
No builder here reads a field, a mode, an entry table or a Python object:
what a builder needs to decide (the presence route, an output region on the
image, the class kind) is a plain value of its request.

Each builder makes the allocations and the stages its door function made, in
the same order with the same operands, so the bits do not move. No Python
import (tools/python_free_dirs.sh)."""
from bindings.prep_encode_prog_run import EncProgram, enc_label_load, enc_label_chunk, enc_input
from core.preprocessing_program import prep_program_stage

comptime LAB_NONE = -1
# x_prep stage codes (_expansion_prep._OPS).
comptime LAB_OP_SORT_COLS = 0
comptime LAB_OP_LOOKUP = 7
comptime LAB_OP_COUNT_NEG = 8
comptime LAB_OP_F2I = 11
comptime LAB_OP_LABEL_BINARIZE = 51
comptime LAB_OP_SCATTER_ONES = 52
comptime LAB_OP_BLOCK_ARGMAX = 76
comptime LAB_OP_INDICATOR = 87
comptime LAB_OP_UNIQ_COUNT = 120
comptime LAB_OP_UNIQ_SCAN = 121
comptime LAB_OP_UNIQ_WRITE = 122
comptime LAB_OP_CHUNK_NEG = 123
comptime LAB_OP_ROW_ONES = 135
comptime LAB_OP_CAT_ZERO = 157
comptime LAB_OP_CAT_PRESENT = 158
comptime LAB_OP_PRES_COUNT = 159
comptime LAB_OP_PRES_WRITE = 160
comptime LAB_OP_F2_CODE_GATHER = 241
comptime LAB_OP_C2_BIN_CODE = 242
# The presence route's table: exact integer labels in [0, LAB_CAT_R).
comptime LAB_CAT_R = 4096
comptime LAB_CAT_CH = 64

# `LabelCodes.kind`
comptime LAB_CODES_BUFFER = 0
comptime LAB_CODES_WORDS = 1
comptime LAB_CODES_LOOKUP = 2
# `LabelUnbinarize.target`
comptime LAB_Y_MULTILABEL = 0
comptime LAB_Y_MULTICLASS = 1
comptime LAB_Y_BINARY = 2


@fieldwise_init
struct LabelWords(Copyable, Movable):
    """A numeric label vector as the program reads it: its lab_load kind,
    its label count, its raw int32 word count and the door's owner index of
    the buffer that holds the words."""
    var kind: Int
    var n: Int
    var nw: Int
    var owner: Int


@fieldwise_init
struct LabelCats(Copyable, Movable):
    """The fitted numeric classes (float32 words): the door's owner index of
    the admitted vector, the class count and the vector's word count."""
    var owner: Int
    var size: Int
    var words: Int


@fieldwise_init
struct LabelFitPlan(Copyable, Movable):
    """Where a fit program leaves its results: the distinct words, their
    count, the presence route's BAD flag (LAB_NONE on the sort route) and
    the int32 codes (LAB_NONE without codes)."""
    var u: Int
    var k: Int
    var bad: Int
    var codes: Int


@fieldwise_init
struct LabelCodesPlan(Copyable, Movable):
    """LabelEncoder.transform of a label buffer: the int32 codes, the
    per-chunk unknown flags and their count."""
    var codes: Int
    var neg: Int
    var chunks: Int


@fieldwise_init
struct LabelGatherPlan(Copyable, Movable):
    """A class gather: the 64-bit class words (two arena words a label), the
    count of codes outside the classes, and the float codes it read."""
    var words: Int
    var bad: Int
    var codes: Int


@fieldwise_init
struct LabelCodes(Movable):
    """Where a program's float codes come from. LAB_CODES_BUFFER: `n` float32
    code words of owner `owner`. LAB_CODES_WORDS: `values` are the codes.
    LAB_CODES_LOOKUP: `values` are the labels' float values, looked up among
    `cats` (-1 when absent)."""
    var kind: Int
    var n: Int
    var owner: Int
    var values: List[Float64]
    var cats: LabelCats


@fieldwise_init
struct LabelUnbinarize(Copyable, Movable):
    """LabelBinarizer.inverse_transform of an (n, W) block (owner `x`): the
    fitted target kind, the class count, the threshold, and whether the
    codes' classes are gathered by the program (numeric classes: `cats` and
    `ints`; for one class the door hands its words twice, since both codes
    name it)."""
    var target: Int
    var n: Int
    var W: Int
    var K: Int
    var x: Int
    var threshold: Float64
    var gather: Bool
    var cats: LabelCats
    var ints: Bool


def _one(value: Float64) -> List[Float64]:
    var out = List[Float64]()
    out.append(value)
    return out^


def lab_load(mut prog: EncProgram, w: LabelWords) -> Int:
    # `_label_load`: the float32 word of each label (NaN when float32 cannot hold it).
    var x = prog.work(w.n)
    var wo = enc_input(prog, w.nw, w.owner)
    enc_label_load(prog, w.n, wo, w.kind, x)
    return x


def lab_unique(mut prog: EncProgram, x: Int, n: Int) -> Tuple[Int, Int]:
    """`_label_unique`: the sorted distinct words of x (sort_cols and a
    chunked run scan). Returns (U offset, count offset)."""
    var s = prog.work(n)
    prep_program_stage(prog.program, LAB_OP_SORT_COLS, 1, x, n, 1, s, 1)
    var ch = enc_label_chunk(n)
    var nch = (n + ch - 1) // ch
    var cnt = prog.work(nch)
    var off = prog.work(nch)
    var u = prog.alloc(n)
    var k = prog.alloc(1)
    prep_program_stage(prog.program, LAB_OP_UNIQ_COUNT, nch, s, n, ch, cnt)
    prep_program_stage(prog.program, LAB_OP_UNIQ_SCAN, 1, cnt, nch, off, k)
    prep_program_stage(prog.program, LAB_OP_UNIQ_WRITE, nch, s, n, ch, off, u)
    return (u, k)


def lab_fit_stage(mut prog: EncProgram, w: LabelWords, present: Bool, codes: Bool,
                  region: Bool) raises -> LabelFitPlan:
    """The fit of a numeric label buffer, one program: the distinct labels
    by presence flags (`present`: exact small integers; the BAD flag says
    when that does not hold) or by a sort and a run scan, and with `codes`
    each label's int32 index among them."""
    var n = w.n
    var x = lab_load(prog, w)
    var u = LAB_NONE
    var k = LAB_NONE
    var bad = LAB_NONE
    var stride = n
    if present:
        var fl = prog.work(LAB_CAT_R)
        bad = prog.alloc(1)
        u = prog.alloc(LAB_CAT_R)
        k = prog.alloc(1)
        var nch = (LAB_CAT_R + LAB_CAT_CH - 1) // LAB_CAT_CH
        var cnt = prog.work(nch)
        var off = prog.work(nch)
        prep_program_stage(prog.program, LAB_OP_CAT_ZERO, LAB_CAT_R, fl)
        prep_program_stage(prog.program, LAB_OP_CAT_PRESENT, n, x, n, 1, LAB_CAT_R, fl, bad)
        prep_program_stage(prog.program, LAB_OP_PRES_COUNT, nch, fl, LAB_CAT_R, LAB_CAT_CH, cnt)
        prep_program_stage(prog.program, LAB_OP_UNIQ_SCAN, 1, cnt, nch, off, k)
        prep_program_stage(prog.program, LAB_OP_PRES_WRITE, nch, fl, LAB_CAT_R, LAB_CAT_CH, off, u)
        stride = LAB_CAT_R
    else:
        var uk = lab_unique(prog, x, n)
        u = uk[0]
        k = uk[1]
    var out = LAB_NONE
    if codes:
        var c = prog.work(n)
        prep_program_stage(prog.program, LAB_OP_LOOKUP, n, x, n, 1, u, stride, k, c)
        out = prog.output(n, region)
        prep_program_stage(prog.program, LAB_OP_F2I, n, c, out)
    return LabelFitPlan(u, k, bad, out)


def lab_buffer_codes(mut prog: EncProgram, w: LabelWords, cats: LabelCats) -> Int:
    """`_label_codes_device`: each label's index among the classes (-1 when
    absent). Returns the float codes offset."""
    var x = lab_load(prog, w)
    var c = prog.work(w.n)
    var uo = enc_input(prog, cats.words, cats.owner)
    var ko = prog.put_floats(_one(Float64(cats.size)))
    prep_program_stage(prog.program, LAB_OP_LOOKUP, w.n, x, w.n, 1, uo, cats.size, ko, c)
    return c


def lab_transform_stage(mut prog: EncProgram, w: LabelWords, cats: LabelCats, region: Bool) raises -> LabelCodesPlan:
    """LabelEncoder.transform of a numeric label buffer: the int32 codes and
    the per-chunk unknown flags, one program."""
    var n = w.n
    var codes = lab_buffer_codes(prog, w, cats)
    var out = prog.output(n, region)
    prep_program_stage(prog.program, LAB_OP_F2I, n, codes, out)
    var ch = enc_label_chunk(n)
    var nch = (n + ch - 1) // ch
    var neg = prog.alloc(nch)
    prep_program_stage(prog.program, LAB_OP_CHUNK_NEG, nch, codes, n, ch, neg)
    return LabelCodesPlan(out, neg, nch)


def lab_lookup_codes(mut prog: EncProgram, values: List[Float64], cats: LabelCats) -> Tuple[Int, Int]:
    """`_label_codes`: each label's index among the classes (or -1) from its
    float value rounded to float32, and the unknown count. Returns (codes,
    count). The label words are program-owned (was a float32 Array built for
    the call) and the classes are read from the fitted vector itself (was a
    one-column copy of it); the same words reach the same lookup."""
    var n = len(values)
    var xo = prog.put_floats(values)
    var block = enc_input(prog, cats.words, cats.owner)
    var co = prog.put_floats(_one(Float64(cats.size)))
    var codes = prog.alloc(n)
    prep_program_stage(prog.program, LAB_OP_LOOKUP, n, xo, n, 1, block, cats.size, co, codes)
    var neg = prog.alloc(1)
    prep_program_stage(prog.program, LAB_OP_COUNT_NEG, 1, codes, n, 1, neg)
    return (codes, neg)


def lab_list_transform_stage(mut prog: EncProgram, values: List[Float64], cats: LabelCats) -> LabelCodesPlan:
    """LabelEncoder.transform of numeric labels read from a Python list: the
    int32 codes (arena words) and the unknown count (`neg`, one float word)."""
    var n = len(values)
    var lc = lab_lookup_codes(prog, values, cats)
    var out = prog.alloc(n)
    prep_program_stage(prog.program, LAB_OP_F2I, n, lc[0], out)
    return LabelCodesPlan(out, lc[1], 1)


def lab_class_gather(mut prog: EncProgram, codes: Int, n: Int, cats: LabelCats, ints: Bool) -> LabelGatherPlan:
    """`_stage_class_gather`: the n float codes to their classes as 64-bit
    words (f2_code_gather) and the count of codes that are not an integer in
    [0, cats.size)."""
    var out = prog.alloc(2 * n)
    var bad = prog.work(n)
    var neg = prog.alloc(1)
    var co = enc_input(prog, cats.words, cats.owner)
    prep_program_stage(prog.program, LAB_OP_F2_CODE_GATHER, n, codes, cats.size, co, 0 if ints else 1, out, bad)
    prep_program_stage(prog.program, LAB_OP_COUNT_NEG, 1, bad, n, 1, neg)
    return LabelGatherPlan(out, neg, codes)


def lab_inverse_stage(mut prog: EncProgram, w: LabelWords, cats: LabelCats, ints: Bool) -> LabelGatherPlan:
    """LabelEncoder.inverse_transform of a numeric code buffer: each code
    checked (an integer in [0, K)) and its class written as an int64 or
    float64 word, one program."""
    var x = lab_load(prog, w)
    return lab_class_gather(prog, x, w.n, cats, ints)


def lab_codes(mut prog: EncProgram, source: LabelCodes) -> Int:
    # The float codes of `source` as arena words; returns their offset.
    if source.kind == LAB_CODES_BUFFER:
        return enc_input(prog, source.n, source.owner)
    if source.kind == LAB_CODES_WORDS:
        return prog.put_floats(source.values)
    return lab_lookup_codes(prog, source.values, source.cats)[0]


def lab_binarize(mut prog: EncProgram, var codes: Int, n: Int, K: Int, W: Int, neg: Int, pos: Int,
                 region: Bool) raises -> Int:
    """`_lb_binarize`: the dense label_binarize output (n, W) of n float
    codes (one class: every code absent). Returns the int32 output offset."""
    if K == 1:
        codes = prog.put_floats(List[Float64](length=n, fill=-1.0))
    var out = prog.output(n * W, region)
    prep_program_stage(prog.program, LAB_OP_LABEL_BINARIZE, n * W, codes, n, K, 1 if K <= 2 else 0, neg, pos, W, out)
    return out


def lab_binarize_buffer_stage(mut prog: EncProgram, w: LabelWords, cats: LabelCats, K: Int, W: Int, neg: Int, pos: Int,
                              region: Bool) raises -> Int:
    # LabelBinarizer.transform of a numeric label buffer (two classes or
    # more): the codes and the dense output in one program.
    var codes = lab_buffer_codes(prog, w, cats)
    return lab_binarize(prog, codes, w.n, K, W, neg, pos, region)


def lab_binarize_codes_stage(mut prog: EncProgram, source: LabelCodes, K: Int, W: Int, neg: Int, pos: Int,
                             region: Bool) raises -> Int:
    # LabelBinarizer.transform of str, object or list labels: their float
    # codes (read at the boundary, or looked up here) and the dense output.
    var codes = lab_codes(prog, source)
    return lab_binarize(prog, codes, source.n, K, W, neg, pos, region)


def lab_indicator_stage(mut prog: EncProgram, owner: Int, words: Int, neg: Int, pos: Int, region: Bool) raises -> Int:
    # LabelBinarizer.transform of a multilabel indicator (owner `owner`,
    # `words` int32 words): each nonzero entry pos_label, each zero neg_label.
    var out = prog.output(words, region)
    var io = enc_input(prog, words, owner)
    prep_program_stage(prog.program, LAB_OP_INDICATOR, words, io, words, 0, LAB_NONE, neg, pos, out)
    return out


def lab_unbinarize_stage(mut prog: EncProgram, r: LabelUnbinarize) -> LabelGatherPlan:
    """LabelBinarizer.inverse_transform, one program. Multilabel: the (n, W)
    int32 indicator at `words` (nothing else). Multiclass: one code per row
    by the block argmax. Binary: the last column against the threshold. With
    `gather`, each code's class as a 64-bit word at `words`; without it the
    float codes at `codes` are the result (str or object classes)."""
    var n = r.n
    var W = r.W
    if r.target == LAB_Y_MULTILABEL:
        var out = prog.alloc(n * W)
        var xo = prog.input(n * W, r.x)
        var to = prog.put_floats(_one(r.threshold))
        prep_program_stage(prog.program, LAB_OP_INDICATOR, n * W, xo, n * W, 1, to, 0, 1, out)
        return LabelGatherPlan(out, LAB_NONE, LAB_NONE)
    var codes = LAB_NONE
    if r.target == LAB_Y_MULTICLASS:
        # `_block_argmax(pr, arr, [K], None, False)`
        var xo = prog.input(n * W, r.x)
        var so = prog.put_floats(_one(0.0))
        var wo = prog.put_floats(_one(Float64(r.K)))
        codes = prog.alloc(n)
        prep_program_stage(prog.program, LAB_OP_BLOCK_ARGMAX, n, xo, n, W, 1, so, wo, LAB_NONE, 0, codes)
    else:
        codes = prog.alloc(n)
        var xo = prog.input(n * W, r.x)
        var to = prog.put_floats(_one(r.threshold))
        prep_program_stage(prog.program, LAB_OP_C2_BIN_CODE, n, xo, n, W, to, codes)
    if not r.gather:
        return LabelGatherPlan(LAB_NONE, LAB_NONE, codes)
    return lab_class_gather(prog, codes, n, r.cats, r.ints)


def lab_row_ones_stage(mut prog: EncProgram, w: LabelWords, cats: LabelCats, rows: Int, offsets: Int, K: Int,
                       region: Bool) raises -> Int:
    """MultiLabelBinarizer.transform of the flat label sets (`w`: every
    label; owner `offsets`: the rows + 1 int32 row offsets): the (rows, K)
    int32 indicator, one program."""
    var codes = lab_buffer_codes(prog, w, cats)
    var offs = enc_input(prog, rows + 1, offsets)
    var out = prog.output(rows * max(K, 1), region)
    prep_program_stage(prog.program, LAB_OP_ROW_ONES, rows, codes, offs, K, out)
    return out


def lab_scatter_ones_stage(mut prog: EncProgram, source: LabelCodes, row_of: List[Float64], rows: Int, K: Int,
                           region: Bool) raises -> Int:
    """MultiLabelBinarizer.transform of ragged label sets: label t (its
    float code from `source`) sets column code of row `row_of[t]` in the
    (rows, K) int32 indicator. No label at all: the zero indicator, no stage."""
    if source.n == 0:
        return prog.alloc(rows * max(K, 1))
    var codes = lab_codes(prog, source)
    var ro = prog.put_floats(row_of)
    var out = prog.output(rows * max(K, 1), region)
    prep_program_stage(prog.program, LAB_OP_SCATTER_ONES, source.n, codes, ro, K, out)
    return out
