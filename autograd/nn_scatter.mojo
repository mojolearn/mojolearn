# SPDX-License-Identifier: Apache-2.0
# SCATTER_ADD AND THE SEGMENT REDUCTIONS ON THE TAPE (lane nn-blocks-scatter, 2026-11-25).
#
# Session codes 113..116 (autograd/ops_api.mojo, the `# nn-blocks-scatter` group), kernel codes 6800..6819
# (autograd/nn_scatter_ops.mojo), custom tags 6810..6813 (NnScatterRules below, reached through ImportRules, so every
# backward that differentiates a session or a captured program, Program.fit and Sequential's ModelRules included,
# knows them). 117 is reserved for this family and unused.
#
#   scatter_add(x, dim, index, src)                 torch.Tensor.scatter_add: out = x, then for every element of
#                                                   src, out[..., index[e], ...] += src[e] along dim
#   segment_sum / segment_mean / segment_max(data, segment_ids, num_segments)
#                                                   tf.math.unsorted_segment_* / torch_scatter on rows: row s of the
#                                                   result reduces the rows r of data with segment_ids[r] == s
#
# THE ORDER (the same bits on the host and on every vendor; no float atomics). The source positions of each target
# row are put in ASCENDING position order by the embedding backward's runs (autograd/embedding_ops.mojo
# embedding_runs_into: a counting scan, or a bitonic sort of (id, position) pairs, chosen by work, exact integers),
# then ONE thread per output cell folds its row in that order:
#   sum      acc = +0.0; acc = ftz(ftz(acc) + ftz(data[pos])) for each pos ascending (OP_EMB_FOLD, the embedding's
#            own fold body: one body for the embedding backward and the segment sum)
#   mean     the sum, then ftz(sum / count) by identical_div (OP_SCT_MEAN); an empty segment keeps +0.0:
#            mean = sum / max(count, 1)
#   max      the largest value over the row ascending, the FIRST position holding it kept (OP_SCT_MAX); a NaN
#            propagates (the first NaN is kept); an empty segment gives +0.0
#   scatter  out = x (OP_COPY), then the sum fold CONTINUED from x's value (the embedding fold's carry): for an
#            output cell the sources are taken in ascending source position, which along dim is ascending index
#            position k: out = ((x + src[k0]) + src[k1]) + ..., k0 < k1 < ...; a cell no index names keeps x.
# The flat target of a scatter source (o, k, i) is the output cell (o, index, i) (OP_SCT_KEYS), so scatter_add is
# the same width-1 fold over flat cells.
#
# EMPTY SEGMENTS. sum 0, mean 0 (sum / max(count, 1)), max 0. The max of an empty segment is 0, not the lowest
# float32 (tf.math.unsorted_segment_max's choice) nor -inf: torch_scatter.scatter_max's choice, a finite value, so a
# model that pools a graph with an isolated node or a bag with no item stays finite; an empty segment passes no
# gradient to any row.
#
# GRADIENTS (first order; a VJP each, every destination word written once, no accumulation inside a rule):
#   scatter_add  d x = the upstream gradient (a copy); d src[e] = upstream[key[e]] (OP_EMB_GATHER, width 1)
#   segment_sum  d data[r] = upstream[segment_ids[r]] (OP_EMB_GATHER, the embedding forward's gather body)
#   segment_mean d data[r] = upstream[segment_ids[r]] / count(segment_ids[r]) (OP_SCT_MEAN_VJP)
#   segment_max  d data[r, j] = upstream[s, j] when r is cell (s, j)'s argmax, else 0 (OP_SCT_MAX_VJP): ties go to
#                the lowest source position
# The index and segment ids take no gradient. A second-order backward (create_graph) through these ops is refused
# by the tape's custom-node rule (autograd/higher_ops.mojo names the tag).
#
# LAUNCHES. Forward: the id check (one level per 256x ids, then one two-word read-back), then the runs (scan: 2
# launches; sort: 1 + log2(P)(log2(P) + 1) / 2 + 1), then sum 1 fold, mean 1 fold + 1, max 1, scatter 1 keys + 1
# copy + 1 fold. Backward: 1 launch per input (scatter: 2). Nothing is launched per row, per segment or per id.
from sequence.exec_trait import Exec
from sequence.ops import FP, Args, OP_COPY, dummy_ptr
from autograd.tape import Tape, Tensor, BackwardRules
from autograd.rankn import ShapeN
from autograd.rankn_api import rn_shape, rn_dims
from autograd.embedding_ops import (
    OP_EMB_GATHER, AG_EMB_PLAN_AUTO, AG_EMB_PLAN_SORT, embedding_plan, embedding_perm_words, embedding_check_words,
    embedding_refuse_ids, embedding_backward_into, embedding_runs_into,
)
from autograd.nn_scatter_ops import OP_SCT_KEYS, OP_SCT_MEAN, OP_SCT_MAX, OP_SCT_MAX_VJP, OP_SCT_MEAN_VJP
from embedding.checks.embedding_oracle import EMB_MAX_POSITIONS

#: Session codes (autograd/ops_api.mojo, python/mojolearn/ops.py); 117 is reserved for this family.
comptime OPS_SCATTER_ADD = 113
comptime OPS_SEGMENT_SUM = 114
comptime OPS_SEGMENT_MEAN = 115
comptime OPS_SEGMENT_MAX = 116

#: The segment reduction kinds (the session code minus OPS_SEGMENT_SUM).
comptime SEGMENT_SUM = 0
comptime SEGMENT_MEAN = 1
comptime SEGMENT_MAX = 2

#: Custom tags of the four nodes (NnScatterRules).
comptime NN_SCATTER_ADD_TAG = 6810
comptime NN_SEGMENT_SUM_TAG = 6811
comptime NN_SEGMENT_MEAN_TAG = 6812
comptime NN_SEGMENT_MAX_TAG = 6813
comptime NN_SCT_TAG_FIRST = 6810
comptime NN_SCT_TAG_LAST = 6813

# Every launch count and Args integer stays inside the device executor's Int32 range, and a flat key or a sort key
# (the id bound itself marks padding) is an int32 word.
comptime _SCT_I32_MAX = 2147483647


def segment_op_name(kind: Int) -> String:
    """The public name of segment kind `kind`, for the refusals."""
    if kind == SEGMENT_MEAN:
        return "segment_mean"
    if kind == SEGMENT_MAX:
        return "segment_max"
    return "segment_sum"


def _sct_refuse_ids[E: Exec](
    mut tape: Tape, mut ex: E, ids: FP, n: Int, bound: Int, what: String, word: String,
) raises:
    """Refuse an id outside [0, bound), by position and value, before any kernel reads it: the embedding's
    first-bad search (one two-word read-back), its words said for this operation."""
    var cw = embedding_check_words(n)
    var scratch_a = tape.reserve(ex, cw)
    var scratch_b = tape.reserve(ex, cw)
    try:
        embedding_refuse_ids(ex, ids, n, bound, scratch_a, scratch_b)
    except e:
        var message = String(e)
        if "embedding: id " in message:
            raise Error("ValueError: " + what + ": " + message.replace("embedding: id", word))
        raise Error(message)


def _sct_runs_scratch[E: Exec](mut tape: Tape, mut ex: E, n: Int, rows: Int) raises -> Tuple[Int, FP, FP, FP]:
    """The resolved run plan and its scratch (run_begin rows + 1 words, perm, the sort's keys), tape storage until
    the reset (run_begin stays for the mean's backward)."""
    var plan = embedding_plan(AG_EMB_PLAN_AUTO, rows, n)
    var pw = embedding_perm_words(plan, n)
    var run_begin = tape.reserve(ex, rows + 1)
    var perm = tape.reserve(ex, pw)
    var keys = tape.reserve(ex, pw) if plan == AG_EMB_PLAN_SORT else dummy_ptr()
    return (plan, run_begin, perm, keys)


def nn_scatter_add[E: Exec](
    mut tape: Tape, mut ex: E, x: Tensor, index: FP, index_shape: ShapeN, src: Tensor, dim: Int,
) raises -> Tensor:
    """torch.Tensor.scatter_add(dim, index, src) on the tape: a new tensor of x's shape. `index` holds int32 words
    (an ids tensor of the session) of `index_shape`, which must be src's shape; src matches x on every dimension
    but `dim`. Refuses an index outside [0, x.shape[dim]) by position and value."""
    var what = String("scatter_add")
    var xs = tape.shape_of(x)
    var ss = tape.shape_of(src)
    var nd = xs.ndim
    if nd < 1 or ss.ndim != nd or index_shape.ndim != nd:
        raise Error("ValueError: scatter_add: x, index and src must have the same number of dimensions (x has "
                    + String(nd) + ", index " + String(index_shape.ndim) + ", src " + String(ss.ndim) + ")")
    if dim < -nd or dim >= nd:
        raise Error("ValueError: scatter_add: dim " + String(dim) + " is out of range for a tensor of "
                    + String(nd) + " dimensions")
    var d = dim + nd if dim < 0 else dim
    if not index_shape.same(ss):
        raise Error("ValueError: scatter_add: index must have the shape of src")
    var inner = 1
    for k in range(nd):  # small-loop(nd: tensor rank, at most 8): shape metadata, no data
        if k != d and ss.d[k] != xs.d[k]:
            raise Error("ValueError: scatter_add: src must have x's size on every dimension except dim (dimension "
                        + String(k) + ": x " + String(xs.d[k]) + ", src " + String(ss.d[k]) + ")")
        if k > d:
            inner *= xs.d[k]
    var K = ss.d[d]
    var V = xs.d[d]
    var P = ss.count()
    var C = xs.count()
    if P > EMB_MAX_POSITIONS or C > EMB_MAX_POSITIONS:
        raise Error("ValueError: scatter_add: src and x must each hold fewer than 2^30 values")
    _sct_refuse_ids(tape, ex, index, P, V, what, "index")
    var keys = tape.reserve(ex, P)
    var ka = Args()
    ka.p0 = index
    ka.p1 = keys
    ka.i0 = K
    ka.i1 = inner
    ka.i2 = V
    ex.launch[OP_SCT_KEYS](ka, P)
    var out = tape.reserve(ex, C)
    var ca = Args()
    ca.p0 = tape.value_of(ex, x)
    ca.p1 = out
    ex.launch[OP_COPY](ca, C)
    var sp = tape.value_of(ex, src)
    var runs = _sct_runs_scratch(tape, ex, P, C)
    # the width-1 fold over the flat output cells, continued from x's words (the carry)
    embedding_backward_into(ex, keys, sp, out, P, C, 1, -1, True, runs[0], runs[1], runs[2], runs[3])
    if not (tape.is_recording() and (tape.requires_grad(x) or tape.requires_grad(src))):
        return tape.variable_nd(ex, out, xs, False)
    var saved = Args()
    saved.p0 = keys
    saved.i0 = P
    saved.i1 = C
    var inputs = List[Tensor]()
    inputs.append(x)
    inputs.append(src)
    return tape.custom_nd(ex, inputs, out, xs, NN_SCATTER_ADD_TAG, saved)


def nn_segment[E: Exec](
    mut tape: Tape, mut ex: E, data: Tensor, ids: FP, n_ids: Int, num_segments: Int, kind: Int,
) raises -> Tensor:
    """segment_sum / segment_mean / segment_max (kind SEGMENT_*) of data's rows (its first dimension) by `ids`
    (n_ids int32 words, one per row, each in [0, num_segments)): a (num_segments, *data.shape[1:]) tensor."""
    var what = segment_op_name(kind)
    if kind != SEGMENT_SUM and kind != SEGMENT_MEAN and kind != SEGMENT_MAX:
        raise Error("ValueError: segment reduction kind must be 0 (sum), 1 (mean) or 2 (max)")
    var ds = tape.shape_of(data)
    if ds.ndim < 1:
        raise Error("ValueError: " + what + ": data must have at least one dimension")
    var N = ds.d[0]
    if n_ids != N:
        raise Error("ValueError: " + what + ": segment_ids must hold one id per row of data (" + String(N)
                    + " rows, " + String(n_ids) + " ids)")
    if num_segments < 1:
        raise Error("ValueError: " + what + ": num_segments must be a positive integer, got " + String(num_segments))
    var width = ds.count() // N
    if N > EMB_MAX_POSITIONS or num_segments >= _SCT_I32_MAX // width or N > _SCT_I32_MAX // width:
        raise Error("ValueError: " + what + ": data and the result must each hold fewer than 2^31 values")
    _sct_refuse_ids(tape, ex, ids, N, num_segments, what, "segment id")
    var dims = rn_dims(ds)
    dims[0] = num_segments
    var shape = rn_shape(dims, what)
    var S = num_segments
    var cells = S * width
    var dp = tape.value_of(ex, data)
    var out = tape.reserve(ex, cells)
    var runs = _sct_runs_scratch(tape, ex, N, S)
    var arg = dummy_ptr()
    if kind == SEGMENT_MAX:
        embedding_runs_into(ex, ids, N, S, -1, runs[0], runs[1], runs[2], runs[3])
        arg = tape.reserve(ex, cells)
        var m = Args()
        m.p0 = dp
        m.p1 = runs[2]
        m.p2 = runs[1]
        m.p3 = out
        m.p4 = arg
        m.i0 = width
        ex.launch[OP_SCT_MAX](m, cells)
    else:
        embedding_backward_into(ex, ids, dp, out, N, S, width, -1, False, runs[0], runs[1], runs[2], runs[3])
        if kind == SEGMENT_MEAN:
            var a = Args()
            a.p0 = out
            a.p1 = runs[1]
            a.i0 = width
            ex.launch[OP_SCT_MEAN](a, cells)
    if not (tape.is_recording() and tape.requires_grad(data)):
        return tape.variable_nd(ex, out, shape, False)
    var saved = Args()
    saved.p0 = ids
    saved.p1 = runs[1]
    saved.p2 = arg
    saved.i0 = N
    saved.i1 = S
    saved.i2 = width
    var inputs = List[Tensor]()
    inputs.append(data)
    var tag = NN_SEGMENT_SUM_TAG
    if kind == SEGMENT_MEAN:
        tag = NN_SEGMENT_MEAN_TAG
    elif kind == SEGMENT_MAX:
        tag = NN_SEGMENT_MAX_TAG
    return tape.custom_nd(ex, inputs, out, shape, tag, saved)


@fieldwise_init
struct NnScatterRules(BackwardRules, ImplicitlyCopyable, Movable):
    """The VJPs of scatter_add and the segment reductions (tags 6810..6813). Each rule writes every word of every
    destination once (zeroed_destination False); the tape adds the destination into the input's gradient, so an
    input used along several paths gets every contribution."""

    @staticmethod
    def owns(tag: Int) -> Bool:
        return tag >= NN_SCT_TAG_FIRST and tag <= NN_SCT_TAG_LAST

    def supports(self, tag: Int) -> Bool:
        return tag >= NN_SCT_TAG_FIRST and tag <= NN_SCT_TAG_LAST

    def zeroed_destination(self, tag: Int, saved: Args, edge: Int) -> Bool:
        return False

    def writes_total(self, tag: Int, saved: Args, edge: Int) -> Bool:
        return False

    def apply[E: Exec](
        self, mut ex: E, tag: Int, saved: Args, output: FP, upstream: FP,
        inputs: List[FP], gradients: List[FP], rows: List[Int], cols: List[Int],
    ) raises:
        if tag == NN_SCATTER_ADD_TAG:
            # saved: p0 the flat keys (P words); i0 P, i1 C (x's element count)
            var c = Args()
            c.p0 = upstream
            c.p1 = gradients[0]
            ex.launch[OP_COPY](c, saved.i1)
            var g = Args()
            g.p0 = upstream
            g.p1 = saved.p0
            g.p2 = gradients[1]
            g.i0 = 1
            ex.launch[OP_EMB_GATHER](g, saved.i0)
            return
        # the segment reductions: saved p0 the ids (N words), p1 run_begin (S + 1), p2 the max's argmax words;
        # i0 N, i1 S, i2 the width
        var a = Args()
        a.i0 = saved.i2
        if tag == NN_SEGMENT_SUM_TAG:
            a.p0 = upstream
            a.p1 = saved.p0
            a.p2 = gradients[0]
            ex.launch[OP_EMB_GATHER](a, saved.i0 * saved.i2)
        elif tag == NN_SEGMENT_MEAN_TAG:
            a.p0 = saved.p0
            a.p1 = saved.p1
            a.p2 = upstream
            a.p3 = gradients[0]
            ex.launch[OP_SCT_MEAN_VJP](a, saved.i0 * saved.i2)
        elif tag == NN_SEGMENT_MAX_TAG:
            a.p0 = saved.p0
            a.p1 = saved.p2
            a.p2 = upstream
            a.p3 = gradients[0]
            ex.launch[OP_SCT_MAX_VJP](a, saved.i0 * saved.i2)
        else:
            raise Error("nn_scatter rules: unknown tag")
