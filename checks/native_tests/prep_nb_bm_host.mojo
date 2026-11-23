# SPDX-License-Identifier: Apache-2.0
"""Host-column runs of the typed CategoricalNB fit and naive Bayes partial_fit programs, for prep_naive_bayes.mojo
(lane bm-rest-nb, 2026-11-23). Importable; no `main()`, no `test_*`.

The programs are the typed builders the door builds through (bindings/prep_classify_prog_run.mojo
`cls_gnb_partial_program`, `cls_dnb_partial_program` with `cls_dnb_params`, `cls_cat_count_program`,
`cls_cat_categories`, `cls_cat_bounds`, `cls_cat_table_program`, `cls_partial_codes`), with the buffers in the record
order the door owns them, run by `prep_cls_host.cls_host_run` (the plain host route of `cls_prog_execute`). The route
words are the identical host image's (prep_cls_host's module docstring).

Labels: CategoricalNB.fit hands the programs the encoded class codes (the walk's codes, `ClsLabels(False, ...)`).
partial_fit with numeric classes [0, 1, 2] and an int64 label vector takes the device lookup
(`ClsLabels(True, ...)`, lab_load kind 3: two int32 words per label) against the class words [0.0, 1.0, 2.0].
"""
from std.memory import bitcast
from bindings.prep_classify_prog_run import (
    ClsProgram, ClsLabels, CLS_NONE, CLS_KIND_GNB, CLS_KIND_MNB, CLS_KIND_BNB, CLS_KIND_CNB, CLS_KIND_CAT,
    CLS_PRIOR_FIT, CLS_W_PROBA, CLS_W_PREDICT, CLS_W_LOG,
    cls_gnb_partial_program, cls_dnb_partial_program, cls_dnb_params, cls_cat_count_program, cls_cat_categories,
    cls_cat_bounds, cls_cat_table_program, cls_score_program,
)
from x_prep.blocked import IDN_CLASS_ONEPASS, IDN_NB_ONEPASS
from checks.native_tests.prep_cls_host import cls_host_run, codes_words, arena_words, arena_codes
from checks.native_tests.testsuite_cases import require

# lab_load's kind of an int64 label vector (bindings/prep_classify_common_door.mojo `_label_kind`)
comptime LABEL_KIND_I64 = 3


def walk_labels(n: Int) -> ClsLabels:
    """The encoded class codes of a fit (n int32 words)."""
    return ClsLabels(False, n, n, 0, 0, 0)


def dev_labels(n: Int, K: Int) -> ClsLabels:
    """An int64 label vector of n labels looked up among K class words."""
    return ClsLabels(True, n, 0, 2 * n, LABEL_KIND_I64, K)


def i64_label_words(codes: List[Int32]) -> List[Float32]:
    """The labels 0..K-1 as the little-endian int64 words of the label buffer (low word, then a zero high word)."""
    var words = List[Float32](length=max(2 * len(codes), 1), fill=0)
    for i in range(len(codes)):  # small-loop(fixture: the labels into an int64 record)
        words[2 * i] = bitcast[DType.float32](codes[i])
    return words^


def class_words(K: Int) -> List[Float32]:
    """The exact ascending float32 class words [0.0, 1.0, ...] of the classes 0..K-1."""
    var words = List[Float32](length=K, fill=0)
    for k in range(K):  # small-loop(fixture: the class words)
        words[k] = Float32(k)
    return words^


def add_dev_labels(mut buffers: List[List[Float32]], codes: List[Int32], K: Int):
    """`own_partial_labels`' order for the device lookup: the label words, then the class words."""
    buffers.append(i64_label_words(codes))
    buffers.append(class_words(K))


def take_rows(x: List[Float32], d: Int, start: Int, stop: Int) -> List[Float32]:
    var out = List[Float32](capacity=max((stop - start) * d, 1))
    for i in range(start * d, stop * d):  # small-loop(fixture: one batch's rows)
        out.append(x[i])
    return out^


def take_codes(y: List[Int32], start: Int, stop: Int) -> List[Int32]:
    var out = List[Int32](capacity=max(stop - start, 1))
    for i in range(start, stop):  # small-loop(fixture: one batch's labels)
        out.append(y[i])
    return out^


def pick_rows(x: List[Float32], d: Int, rows: List[Int32]) -> List[Float32]:
    var out = List[Float32](capacity=max(len(rows) * d, 1))
    for r in range(len(rows)):  # small-loop(fixture: the selected rows)
        for j in range(d):  # small-loop(fixture: one row's features)
            out.append(x[Int(rows[r]) * d + j])
    return out^


def pick_codes(y: List[Int32], rows: List[Int32]) -> List[Int32]:
    var out = List[Int32](capacity=max(len(rows), 1))
    for r in range(len(rows)):  # small-loop(fixture: the selected labels)
        out.append(y[Int(rows[r])])
    return out^


def no_unknown_label(arena: List[Float32], neg: Int, label: String) raises:
    """`partial_refuse`: no label of the batch outside the classes."""
    if neg != CLS_NONE:
        require(arena[neg] == 0, label + ": " + String(arena[neg]) + " labels outside the classes")


# ---------------------------------------------------------------- GaussianNB partial_fit


@fieldwise_init
struct GnbState(Movable):
    """The fitted blocks a GaussianNB partial_fit publishes (empty `cnt`: before the first call)."""
    var theta: List[Float32]
    var var_: List[Float32]
    var cst: List[Float32]
    var cnt: List[Float32]
    var raw: List[Float32]


def gnb_empty() -> GnbState:
    var e = List[Float32]()
    return GnbState(e.copy(), e.copy(), e.copy(), e.copy(), e.copy())


def gnb_partial(state: GnbState, x: List[Float32], codes: List[Int32], weights: List[Float32], d: Int, K: Int,
                blocked: Bool) raises -> GnbState:
    """GaussianNB.partial_fit as `_gnb_partial_fit` builds it (var_smoothing 1e-9, no priors; `weights` empty:
    unweighted). Records: X, the label words, the class words, the weights, then the previous class_count_, theta_
    and `_raw_var`."""
    var n = len(codes)
    var weighted = len(weights) > 0
    var first = len(state.cnt) == 0
    var prev = List[Int]()
    if not first:
        prev = [K, K * d, K * d]
    var program = ClsProgram(False)
    var o = cls_gnb_partial_program(program, n, d, K, dev_labels(n, K), 1e-9, weighted, prev, False,
        List[Float64](), blocked, blocked and IDN_CLASS_ONEPASS)
    var buffers = List[List[Float32]]()
    buffers.append(x.copy())
    add_dev_labels(buffers, codes, K)
    if weighted:
        buffers.append(weights.copy())
    if not first:
        buffers.append(state.cnt.copy())
        buffers.append(state.theta.copy())
        buffers.append(state.raw.copy())
    var arena = cls_host_run(program, buffers)
    no_unknown_label(arena, o.neg, "GaussianNB partial_fit")
    var f = o.fit
    return GnbState(arena_words(arena, f.theta, K * d), arena_words(arena, f.var_, K * d),
        arena_words(arena, f.cst, K), arena_words(arena, f.cnt, K), arena_words(arena, f.raw, K * d))


# ---------------------------------------------------------------- discrete partial_fit


@fieldwise_init
struct DnbState(Movable):
    """The fitted blocks a discrete partial_fit publishes (empty `cnt`: before the first call)."""
    var cnt: List[Float32]
    var fc: List[Float32]
    var clp: List[Float32]
    var flp: List[Float32]
    var w: List[Float32]
    var bias: List[Float32]


def dnb_empty() -> DnbState:
    var e = List[Float32]()
    return DnbState(e.copy(), e.copy(), e.copy(), e.copy(), e.copy(), e.copy())


def dnb_partial(kind: Int, state: DnbState, x: List[Float32], codes: List[Int32], weights: List[Float32], d: Int,
                K: Int, blocked: Bool) raises -> DnbState:
    """MultinomialNB / ComplementNB / BernoulliNB.partial_fit with the default parameters (alpha 1.0, fit_prior,
    BernoulliNB's binarize 0.0, ComplementNB's norm False) as `_dnb_partial_fit` builds it. Records: X, the weights,
    the label words, the class words, then the previous class_count_ and feature_count_."""
    var n = len(codes)
    var weighted = len(weights) > 0
    var first = len(state.cnt) == 0
    var prev = List[Int]()
    if not first:
        prev = [K, K * d]
    var has_threshold = kind == CLS_KIND_BNB
    var program = ClsProgram(False)
    var o = cls_dnb_partial_program(program, kind, n, d, K, dev_labels(n, K), weighted, has_threshold, 0.0,
        blocked and IDN_NB_ONEPASS, blocked, prev, CLS_PRIOR_FIT, List[Float64]())
    var c = o.counts
    var p = cls_dnb_params(program, kind, c, 1.0, False)
    var buffers = List[List[Float32]]()
    buffers.append(x.copy())
    if weighted:
        buffers.append(weights.copy())
    add_dev_labels(buffers, codes, K)
    if not first:
        buffers.append(state.cnt.copy())
        buffers.append(state.fc.copy())
    var arena = cls_host_run(program, buffers)
    no_unknown_label(arena, o.neg, "discrete partial_fit")
    if kind == CLS_KIND_MNB or kind == CLS_KIND_CNB:
        for j in range(d):  # small-loop(oracle: the door's negative-word refusal, feature count)
            require(arena[c.st + 3 * d + j] >= 0, "a negative word reached a count partial_fit")
    var e = List[Float32]()
    var w = arena_words(arena, p.w, K * d) if kind == CLS_KIND_BNB else e.copy()
    var bias = arena_words(arena, p.bias, K) if kind == CLS_KIND_BNB else e.copy()
    return DnbState(arena_words(arena, c.cnt, K), arena_words(arena, c.fc, K * d), arena_words(arena, c.clp, K),
        arena_words(arena, p.flp, K * d), w^, bias^)


# ---------------------------------------------------------------- CategoricalNB fit and partial_fit


@fieldwise_init
struct CatState(Movable):
    """The fitted state of a CategoricalNB fit (empty `cnt`: before the first call): the (d, K, cmax) `_cc` and
    `_flp`, class_count_, class_log_prior_, n_categories_, the bound `_cmax`."""
    var cc: List[Float32]
    var flp: List[Float32]
    var cnt: List[Float32]
    var clp: List[Float32]
    var ncat: List[Int]
    var cmax: Int


def cat_empty() -> CatState:
    var e = List[Float32]()
    return CatState(e.copy(), e.copy(), e.copy(), e.copy(), List[Int](), 0)


def cat_fit(state: CatState, x: List[Float32], codes: List[Int32], weights: List[Float32], d: Int, K: Int,
            alpha: Float64, has_min: Bool, mins: List[Int], dev: Bool, blocked: Bool) raises -> CatState:
    """CategoricalNB's two programs as `_cat_fit_programs` builds them (`state` not empty: partial_fit after the first
    call, which merges; `dev`: the partial_fit labels, else the fit's codes). Records of the first program: X, the
    labels, the weights, then the previous class_count_; of the second: X, the labels, the first program's
    class_count_, the weights, then the previous `_cc` (when its bound is above zero)."""
    var n = len(codes)
    var weighted = len(weights) > 0
    var merge = len(state.cnt) > 0
    var labels = dev_labels(n, K) if dev else walk_labels(n)
    var first_program = ClsProgram(False)
    var o = cls_cat_count_program(first_program, n, d, K, labels, weighted, K if merge else 0, merge, CLS_PRIOR_FIT,
        List[Float64](), blocked)
    var buffers = List[List[Float32]]()
    buffers.append(x.copy())
    if dev:
        add_dev_labels(buffers, codes, K)
    else:
        buffers.append(codes_words(codes))
    if weighted:
        buffers.append(weights.copy())
    if merge:
        buffers.append(state.cnt.copy())
    var arena = cls_host_run(first_program, buffers)
    no_unknown_label(arena, o.neg, "CategoricalNB counts")
    var ncat = cls_cat_categories(Int(arena.unsafe_ptr()), o.st, d)
    var cmax = cls_cat_bounds(ncat, has_min, mins, state.ncat)
    var cnt = arena_words(arena, o.cnt, K)
    var clp = arena_words(arena, o.clp, K)
    _ = arena^
    var second = ClsProgram(False)
    var t = cls_cat_table_program(second, n, d, K, labels, ncat, cmax, alpha, weighted, merge, state.cmax,
        len(state.cc), blocked)
    var tables = List[List[Float32]]()
    tables.append(x.copy())
    if dev:
        add_dev_labels(tables, codes, K)
    else:
        tables.append(codes_words(codes))
    tables.append(cnt.copy())
    if weighted:
        tables.append(weights.copy())
    if merge and state.cmax > 0:
        tables.append(state.cc.copy())
    var table = cls_host_run(second, tables)
    no_unknown_label(table, t.neg, "CategoricalNB table")
    var words = d * K * cmax
    return CatState(arena_words(table, t.cc, words), arena_words(table, t.flp, words), cnt^, clp^, ncat^, cmax)


def cat_blocks(table: List[Float32], d: Int, K: Int, cmax: Int, ncat: List[Int]) -> List[Float32]:
    """`_cat_blocks`: the per-feature (K, ncat[j]) blocks of the (d, K, cmax) table, concatenated in feature order."""
    var out = List[Float32]()
    for j in range(d):  # small-loop(oracle: the feature blocks)
        for k in range(K):  # small-loop(oracle: one block's classes)
            for c in range(ncat[j]):  # small-loop(oracle: one class's categories)
                out.append(table[(j * K + k) * cmax + c])
    return out^


@fieldwise_init
struct NbProba(Movable):
    var proba: List[Float32]
    var codes: List[Int32]


def nb_proba(kind: Int, x: List[Float32], n: Int, d: Int, K: Int, var records: List[List[Float32]], cmax: Int,
             blocked: Bool) raises -> NbProba:
    """`_score`'s program (`cls_score_program`) asked for the probabilities and the codes, over X and the fitted
    records in `cls_jll_stages` order (BernoulliNB with binarize 0.0; CategoricalNB with its bound `cmax`)."""
    var words = List[Int](length=3, fill=0)
    for i in range(len(records)):  # small-loop(params: at most three fitted records)
        words[i] = len(records[i])
    var program = ClsProgram(False)
    var o = cls_score_program(program, kind, CLS_NONE, n, d, K, words[0], words[1], words[2],
        CLS_W_PROBA | CLS_W_PREDICT, False, False, kind == CLS_KIND_CAT and blocked, kind == CLS_KIND_BNB, 0.0, cmax)
    var buffers = List[List[Float32]]()
    buffers.append(x.copy())
    for i in range(len(records)):  # small-loop(params: at most three fitted records)
        if len(records[i]) > 0:
            buffers.append(records[i].copy())
    var arena = cls_host_run(program, buffers)
    return NbProba(arena_words(arena, o.pp, n * K), arena_codes(arena, o.am, n))


def gnb_proba(m: GnbState, x: List[Float32], n: Int, d: Int, K: Int, blocked: Bool) raises -> NbProba:
    var records = List[List[Float32]]()
    records.append(m.theta.copy())
    records.append(m.var_.copy())
    records.append(m.cst.copy())
    return nb_proba(CLS_KIND_GNB, x, n, d, K, records^, 0, blocked)


def dnb_proba(kind: Int, m: DnbState, x: List[Float32], n: Int, d: Int, K: Int, blocked: Bool) raises -> NbProba:
    var records = List[List[Float32]]()
    if kind == CLS_KIND_BNB:
        records.append(m.w.copy())
        records.append(m.bias.copy())
    else:
        records.append(m.flp.copy())
        # ComplementNB adds the class log prior only with one class
        records.append(m.clp.copy() if kind == CLS_KIND_MNB else List[Float32]())
    return nb_proba(kind, x, n, d, K, records^, 0, blocked)


def cat_proba(m: CatState, x: List[Float32], n: Int, d: Int, K: Int, blocked: Bool) raises -> NbProba:
    var records = List[List[Float32]]()
    records.append(m.flp.copy())
    records.append(m.clp.copy())
    return nb_proba(CLS_KIND_CAT, x, n, d, K, records^, m.cmax, blocked)
