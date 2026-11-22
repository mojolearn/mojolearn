# SPDX-License-Identifier: Apache-2.0
"""The host-column run of the typed classifier x_prep programs, for the batch 2 checks (lane py2mojo-B, 2026-11-22;
docs/plans/PY2MOJO_CONVERSION.md). Importable; no `main()`, no `test_*`.

`cls_host_run` is the plain host route of bindings/prep_classify_prog_run.mojo `cls_prog_execute` (no device
scratch, no ranges): the arena of `program.size + program.scratch_size` zero words, every buffer record filled from
the caller's words in record order and every constant record from the program's own words, then one run of
`program.stage_words()` through the x_prep host program runner. The programs themselves are the typed builders the
doors and the members build through (`cls_gnb_fit_program`, `cls_dnb_counts`, `cls_dnb_params`,
`cls_score_program`, the discriminant builders), so the stages, their order and their words are the product's.

The route words are the identical host image's: MOJOLEARN_XPREP_BLOCKED on (its default), the class and naive Bayes
one-pass bits (`IDN_CLASS_ONEPASS`, `IDN_NB_ONEPASS`), the family word of `x_prep_idn_fam2` composed from the same
switches the host binding composes it from (bindings/_mojolearn_x_prep_host.mojo `idn_fam2_binding`), no float64
probabilities (`PROBA64` is the fast mode's).
"""
from std.math import isnan
from std.memory import bitcast, unsafe_memcpy
from bindings.prep_classify_prog_run import (
    ClsProgram, ClsDaMode, ClsScore, CLS_IN_BUFFER, CLS_STAGE_WORDS, CLS_F32P, CLS_NONE,
    CLS_W_PREDICT, CLS_W_PROBA, CLS_W_LOG, CLS_KIND_GNB, CLS_KIND_MNB, CLS_KIND_BNB, CLS_KIND_CNB,
    cls_gnb_fit_program, cls_dnb_counts, cls_dnb_params, cls_score_program,
)
from x_prep.host.program import run_program_host
from x_prep.blocked import IDN_STATS_BLOCKED, IDN_CLASS_ONEPASS, IDN_NB_ONEPASS
from x_prep.fam2 import IDN_WDRAW, IDN_PERM_DRAW, IDN_WPICK, IDN_PARTIAL_CODES, IDN_LABEL_INV
from x_prep.gram_blocked import IDN_GRAM_BLOCKED, IDN_GRAM_ROWS
from checks.native_tests.py2mojo_fixture_words import fixture_f64
from checks.native_tests.testsuite_cases import require, fixture_f32, fixture_i32

comptime PREP_NB_DIR = "checks/native_tests/fixtures/prep_nb/"
comptime PREP_DA_DIR = "checks/native_tests/fixtures/prep_da/"


def all_close(actual: Float32, expected: Float64, rtol: Float64, atol: Float64, label: String) raises:
    """numpy.testing.assert_allclose for one value: NaN only against NaN, an infinity only against itself."""
    var a = Float64(actual)
    if a == expected:
        return
    if isnan(expected) or isnan(a):
        require(isnan(expected) and isnan(a), label + ": " + String(actual) + " expected " + String(expected))
        return
    require(abs(a - expected) <= atol + rtol * abs(expected),
        label + ": " + String(actual) + " expected " + String(expected))


def check_close(got: List[Float32], want: List[Float64], rtol: Float64, atol: Float64, label: String) raises:
    """assert_allclose over two arrays of the same extent; the first differing word is named."""
    require(len(got) == len(want), label + ": " + String(len(got)) + " values, reference " + String(len(want)))
    for i in range(len(want)):  # small-loop(oracle: one fitted block or one answer)
        all_close(got[i], want[i], rtol, atol, label + " word " + String(i))


def check_signed_columns(got: List[Float32], gw: Int, want: List[Float64], ww: Int, n: Int, rtol: Float64,
                         atol: Float64, label: String) raises:
    """The original's transform comparison: each reference column against ours times the sign of their dot."""
    require(gw >= ww and len(got) == n * gw and len(want) == n * ww, label + ": transform extent")
    for j in range(ww):  # small-loop(oracle: at most two components)
        var dot = 0.0
        for i in range(n):  # small-loop(oracle: the rows of one component)
            dot += Float64(got[i * gw + j]) * want[i * ww + j]
        var s = Float32(1.0) if dot >= 0.0 else Float32(-1.0)
        for i in range(n):  # small-loop(oracle: the rows of one component)
            all_close(s * got[i * gw + j], want[i * ww + j], rtol, atol,
                label + " component " + String(j) + " row " + String(i))


def agreement(got: List[Int32], want: List[Int32]) -> Float64:
    var same = 0
    for i in range(len(want)):  # small-loop(oracle: the predicted codes)
        if got[i] == want[i]:
            same += 1
    return Float64(same) / Float64(max(len(want), 1))


def codes_words(codes: List[Int32]) -> List[Float32]:
    """Class codes as the int32 words of a buffer record."""
    var words = List[Float32](length=max(len(codes), 1), fill=0)
    for i in range(len(codes)):  # small-loop(fixture: the class codes into a record)
        words[i] = bitcast[DType.float32](codes[i])
    return words^


def abs_words(x: List[Float32]) -> List[Float32]:
    """np.abs of the original tests (exact)."""
    var out = List[Float32](length=len(x), fill=0)
    for i in range(len(x)):  # small-loop(fixture: the original's np.abs)
        out[i] = abs(x[i])
    return out^


def arena_words(arena: List[Float32], offset: Int, count: Int) -> List[Float32]:
    var out = List[Float32](length=count, fill=0)
    for i in range(count):  # small-loop(oracle: one block out of the arena)
        out[i] = arena[offset + i]
    return out^


def arena_codes(arena: List[Float32], offset: Int, count: Int) -> List[Int32]:
    var out = List[Int32](length=count, fill=0)
    for i in range(count):  # small-loop(oracle: the int32 codes out of the arena)
        out[i] = bitcast[DType.int32](arena[offset + i])
    return out^


def cls_host_run(program: ClsProgram, buffers: List[List[Float32]]) raises -> List[Float32]:
    """`cls_prog_execute`'s host route: inputs into a zero arena, one program run (see the module docstring)."""
    var arena = List[Float32](length=max(program.size + program.scratch_size, 1), fill=0)
    var b = 0
    for k in range(len(program.inputs)):  # driver-loop: over the program's input records
        var r = program.inputs[k]
        if r.words == 0:
            continue
        if r.kind == CLS_IN_BUFFER:
            require(b < len(buffers), "cls_host_run: more buffer records than buffers")
            require(len(buffers[b]) >= r.words, "cls_host_run: buffer " + String(b) + " holds "
                + String(len(buffers[b])) + " words, its record " + String(r.words))
            unsafe_memcpy(dest=arena.unsafe_ptr() + r.offset, src=buffers[b].unsafe_ptr(), count=r.words)
            b += 1
        else:
            unsafe_memcpy(dest=arena.unsafe_ptr() + r.offset,
                src=CLS_F32P(unsafe_from_address=program.const_address(r)), count=r.words)
    require(b == len(buffers), "cls_host_run: " + String(len(buffers)) + " buffers for " + String(b) + " records")
    var stages = program.stage_words()
    run_program_host(Int(arena.unsafe_ptr()), len(arena), Int(stages.unsafe_ptr()),
        len(program.stages) // CLS_STAGE_WORDS)
    _ = stages^
    return arena^


def da_host_mode(blocked: Bool) -> ClsDaMode:
    """bindings/prep_classify_common_door.mojo `cls_da_mode` on the identical host image."""
    if not blocked:
        return ClsDaMode(False, False, 0)
    var fam2 = ((1 if IDN_WDRAW else 0) | (2 if IDN_PERM_DRAW else 0) | (4 if IDN_WPICK else 0)
        | (8 if IDN_PARTIAL_CODES else 0) | (16 if IDN_GRAM_BLOCKED else 0) | (64 if IDN_LABEL_INV else 0)
        | (IDN_GRAM_ROWS << 16))
    return ClsDaMode(IDN_STATS_BLOCKED, IDN_CLASS_ONEPASS, fam2)


@fieldwise_init
struct NbModel(Movable):
    """The fitted blocks of a naive Bayes fit (an empty list: not this kind's)."""
    var theta: List[Float32]
    var var_: List[Float32]
    var cst: List[Float32]
    var class_count: List[Float32]
    var class_prior: List[Float32]
    var feature_count: List[Float32]
    var log_prior: List[Float32]
    var log_prob: List[Float32]
    var w: List[Float32]
    var bias: List[Float32]


def gnb_fit(x: List[Float32], codes: List[Int32], weights: List[Float32], n: Int, d: Int, K: Int,
            priors: List[Float64], blocked: Bool) raises -> NbModel:
    """GaussianNB.fit as core/naive_bayes_member.mojo `_fit_gaussian` builds it (var_smoothing 1e-9; `weights`
    empty: unweighted; records X, the codes, the weights)."""
    var weighted = len(weights) > 0
    var program = ClsProgram(False)
    var o = cls_gnb_fit_program(program, n, d, K, weighted, 1e-9, len(priors) > 0, priors, blocked,
        blocked and IDN_CLASS_ONEPASS)
    var buffers = List[List[Float32]]()
    buffers.append(x.copy())
    buffers.append(codes_words(codes))
    if weighted:
        buffers.append(weights.copy())
    var arena = cls_host_run(program, buffers)
    var e = List[Float32]()
    return NbModel(arena_words(arena, o.theta, K * d), arena_words(arena, o.var_, K * d),
        arena_words(arena, o.cst, K), arena_words(arena, o.cnt, K), arena_words(arena, o.prior, K),
        e.copy(), e.copy(), e.copy(), e.copy(), e.copy())


def dnb_fit(kind: Int, x: List[Float32], codes: List[Int32], weights: List[Float32], n: Int, d: Int, K: Int,
            alpha: Float64, prior: Int, priors: List[Float64], has_threshold: Bool, threshold: Float64, norm: Bool,
            blocked: Bool) raises -> NbModel:
    """MultinomialNB / BernoulliNB / ComplementNB fit as `_fit_discrete` builds it (records X, the weights, the
    codes)."""
    var weighted = len(weights) > 0
    var program = ClsProgram(False)
    var c = cls_dnb_counts(program, kind, CLS_NONE, n, d, K, weighted, has_threshold, threshold,
        blocked and IDN_NB_ONEPASS, blocked, prior, priors)
    var o = cls_dnb_params(program, kind, c, alpha, norm)
    var buffers = List[List[Float32]]()
    buffers.append(x.copy())
    if weighted:
        buffers.append(weights.copy())
    buffers.append(codes_words(codes))
    var arena = cls_host_run(program, buffers)
    if kind == CLS_KIND_MNB or kind == CLS_KIND_CNB:
        for j in range(d):  # small-loop(oracle: the door's negative-word refusal, feature count)
            require(arena[c.st + 3 * d + j] >= 0, "a negative word reached a count fit")
    var e = List[Float32]()
    var w = arena_words(arena, o.w, K * d) if kind == CLS_KIND_BNB else e.copy()
    var bias = arena_words(arena, o.bias, K) if kind == CLS_KIND_BNB else e.copy()
    return NbModel(e.copy(), e.copy(), e.copy(), arena_words(arena, c.cnt, K), e.copy(),
        arena_words(arena, c.fc, K * d), arena_words(arena, c.clp, K), arena_words(arena, o.flp, K * d), w^, bias^)


@fieldwise_init
struct ClsAnswer(Movable):
    var proba: List[Float32]
    var log_proba: List[Float32]
    var codes: List[Int32]
    var scores: List[Float32]


def cls_scores(kind: Int, x: List[Float32], n: Int, d: Int, K: Int, var records: List[List[Float32]],
               has_threshold: Bool, threshold: Float64, pair: Bool) raises -> ClsAnswer:
    """`_score`'s program (`cls_score_program`) over X and the fitted records in `cls_jll_stages` order, asked for
    the probabilities, their logarithms and the codes in one run (the softmax and argmax stages each answer reads
    are the same stages). `pair`: the two-class difference of the scores (QDA's decision function) in `scores`."""
    var words = List[Int](length=3, fill=0)
    for i in range(len(records)):  # small-loop(params: at most three fitted records)
        words[i] = len(records[i])
    var program = ClsProgram(False)
    var o = cls_score_program(program, kind, CLS_NONE, n, d, K, words[0], words[1], words[2],
        CLS_W_PROBA | CLS_W_LOG | CLS_W_PREDICT, False, pair, False, has_threshold, threshold, 0)
    var buffers = List[List[Float32]]()
    buffers.append(x.copy())
    for i in range(len(records)):  # small-loop(params: at most three fitted records)
        if len(records[i]) > 0:
            buffers.append(records[i].copy())
    var arena = cls_host_run(program, buffers)
    var scores = arena_words(arena, o.am, n) if pair else List[Float32]()
    var codes = List[Int32]() if pair else arena_codes(arena, o.am, n)
    return ClsAnswer(arena_words(arena, o.pp, n * K), arena_words(arena, o.lp, n * K), codes^, scores^)
