# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""LinearDiscriminantAnalysis and QuadraticDiscriminantAnalysis programs against scikit-learn's values (lane
py2mojo-B, 2026-11-22; ported from python/mojolearn/tests/test_x_prep_discriminant.py `test_lda`, `test_qda`,
`test_priors` and `test_solvers_and_shrinkage`).

The inputs are the original tests' NumPy draws (500 x 5 float32, two and three classes) and the references are the
scikit-learn 1.9.1 calls they made on the float64 copies, stored as words in `fixtures/prep_da/` (and, for the naive
Bayes half of `test_priors`, the same set) by `tools/py2mojo_native_fixtures_b2.py`. The programs are the typed
builders core/discriminant_member.mojo `_fit_lda`, `_fit_qda`, `_score`, `_pair_decision` and `_transform` build
(bindings/prep_classify_prog_run.mojo), with the public attributes read where the door reads them
(bindings/prep_classify_native_door.mojo: coef_ / intercept_ are the two-class differences with two classes,
explained_variance_ratio_ the first min(n_classes - 1, n_features) words, transform the min(components, rank)
columns), run on the host column by `prep_cls_host.cls_host_run` on the default route (blocked class statistics,
the one-pass bit and the blocked Gram of the identical image) and with the blocks off.

Tolerances are the original's (assert_allclose's default rtol 1e-7 where it gave atol only): coef_, intercept_,
decision_function rtol = atol = 2e-3 (5e-3 for the solver table), QDA scalings_ rtol 2e-3 atol 1e-4, covariance_
rtol 2e-3 atol 2e-4, predict_proba atol 2e-3 (3e-3 for the solver table), explained_variance_ratio_ atol 1e-3,
transform up to the component sign rtol = atol = 2e-3 (5e-3), predict more than 99% equal; test_priors holds 99% of
the probabilities to 3e-3. The solver table compares explained_variance_ratio_ and transform for 'eigen' only where
the reference's ratios are at most one (its own condition, stored with the references).

Stays in Python: the label encoding, the priors' admission and the rank and count refusals' messages (the
public boundary). Every case is an independent unit; the case tables run across host cores.
"""
from bindings.prep_classify_prog_run import (
    ClsProgram, ClsDaMode, CLS_KIND_GNB, CLS_KIND_MNB, CLS_KIND_LDA, CLS_KIND_QDA, CLS_PRIOR_GIVEN, CLS_NONE,
    cls_lda_svd_program, cls_lda_cov_program, cls_qda_fit_program, cls_lda_transform_program, cls_lda_pair_program,
)
from checks.native_tests.prep_cls_host import (
    PREP_DA_DIR, NbModel, ClsAnswer, check_close, check_signed_columns, agreement, abs_words, codes_words, arena_words, cls_host_run, da_host_mode, gnb_fit, dnb_fit, cls_scores,
)
from checks.native_tests.py2mojo_fixture_words import fixture_f64
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, fixture_f32, fixture_i32, require

comptime DA_N = 500
comptime DA_D = 5
comptime DA_TOL = 1e-4  # tol, scikit-learn's default for both estimators
comptime SOLVER_SVD = 0
comptime SOLVER_LSQR = 1
comptime SOLVER_EIGEN = 2


def want_f64(tag: String, name: String, count: Int) raises -> List[Float64]:
    return fixture_f64(PREP_DA_DIR + tag + "-" + name + ".f64", count)


def want_i32(tag: String, name: String, count: Int) raises -> List[Int32]:
    return fixture_i32(PREP_DA_DIR + tag + "-" + name + ".i32", count)


@fieldwise_init
struct Lda(Movable):
    """A fitted LDA: the blocks the member keeps and the public reading of them."""
    var K: Int
    var coef: List[Float32]       # _coef, K x d
    var inter: List[Float32]      # _inter, K
    var pub_coef: List[Float32]   # coef_
    var pub_inter: List[Float32]  # intercept_
    var xbar: List[Float32]
    var scal: List[Float32]       # _scal_full (not 'lsqr')
    var evr: List[Float32]        # the first max-components words (not 'lsqr')
    var cov: List[Float32]        # covariance_ (empty: not stored)
    var width: Int                # transform's columns


def lda_fit(x: List[Float32], y: List[Int32], n: Int, d: Int, K: Int, solver: Int, has_shrinkage: Bool,
            shrinkage: Float64, store_covariance: Bool, priors: List[Float64], blocked: Bool) raises -> Lda:
    var program = ClsProgram(False)
    var mode = da_host_mode(blocked)
    var eigen = solver == SOLVER_EIGEN
    var o = cls_lda_svd_program(program, mode, n, d, K, DA_TOL, priors, store_covariance) if solver == SOLVER_SVD else (
        cls_lda_cov_program(program, mode, n, d, K, eigen, has_shrinkage, shrinkage, priors))
    var buffers = List[List[Float32]]()
    buffers.append(x.copy())
    buffers.append(codes_words(y))
    if eigen:
        buffers.append(List[Float32](length=n, fill=0))   # the one-class code vector
    var arena = cls_host_run(program, buffers)
    var components = min(K - 1, d)
    var rank = 0
    if solver == SOLVER_SVD:
        rank = Int(arena[o.meta + 2])
    elif eigen:
        for j in range(d):  # small-loop(oracle: the door's positive-definite refusal, feature count)
            require(arena[o.e + j] > 0, "the within-class covariance is not positive definite")
        rank = d
    var e = List[Float32]()
    var pub_coef = arena_words(arena, o.cd, d) if K == 2 else arena_words(arena, o.coef, K * d)
    var pub_inter = arena_words(arena, o.ci, 1) if K == 2 else arena_words(arena, o.inter, K)
    var scal = arena_words(arena, o.scal, d * d) if o.scal != CLS_NONE else e.copy()
    var evr = arena_words(arena, o.evr, components) if o.evr != CLS_NONE else e.copy()
    var cov = arena_words(arena, o.sw, d * d) if o.sw != CLS_NONE else e.copy()
    return Lda(K, arena_words(arena, o.coef, K * d), arena_words(arena, o.inter, K), pub_coef^, pub_inter^,
        arena_words(arena, o.xbar, d), scal^, evr^, cov^, min(components, rank))


def lda_transform(m: Lda, xh: List[Float32], n: Int, d: Int, eigen: Bool) raises -> List[Float32]:
    var program = ClsProgram(False)
    var dst = cls_lda_transform_program(program, n, d, m.width, eigen, False)
    var buffers = List[List[Float32]]()
    buffers.append(xh.copy())
    buffers.append(m.xbar.copy())
    buffers.append(m.scal.copy())
    var arena = cls_host_run(program, buffers)
    return arena_words(arena, dst, n * m.width)


def lda_decision(m: Lda, xh: List[Float32], n: Int, d: Int) raises -> List[Float32]:
    var program = ClsProgram(False)
    var dst = cls_lda_pair_program(program, n, d)
    var buffers = List[List[Float32]]()
    buffers.append(xh.copy())
    buffers.append(m.pub_coef.copy())
    buffers.append(m.pub_inter.copy())
    var arena = cls_host_run(program, buffers)
    return arena_words(arena, dst, n)


def lda_scores(m: Lda, xh: List[Float32], n: Int, d: Int) raises -> ClsAnswer:
    var records = List[List[Float32]]()
    records.append(m.coef.copy())
    records.append(m.inter.copy())
    return cls_scores(CLS_KIND_LDA, xh, n, d, m.K, records^, False, 0.0, False)


@fieldwise_init
struct Qda(Movable):
    var K: Int
    var mean: List[Float32]
    var rot: List[Float32]
    var logc: List[Float32]
    var scalings: List[Float32]
    var cov: List[Float32]


def qda_fit(x: List[Float32], y: List[Int32], n: Int, d: Int, K: Int, eigen: Bool, reg_param: Float64,
            has_shrinkage: Bool, shrinkage: Float64, store_covariance: Bool, priors: List[Float64],
            blocked: Bool) raises -> Qda:
    var program = ClsProgram(False)
    var o = cls_qda_fit_program(program, da_host_mode(blocked), n, d, K, eigen, reg_param, has_shrinkage, shrinkage,
        priors, store_covariance)
    var buffers = List[List[Float32]]()
    buffers.append(x.copy())
    buffers.append(codes_words(y))
    var arena = cls_host_run(program, buffers)
    for k in range(K):  # small-loop(oracle: the door's count and rank refusals, class count)
        require(arena[o.cnt + k] >= 2, "a class has one sample")
        for j in range(d):  # small-loop(oracle: one class's scalings)
            require(Float64(arena[o.s2 + k * d + j]) > DA_TOL, "class code " + String(k) + " is not full rank")
    var cov = arena_words(arena, o.keep, K * d * d) if o.keep != CLS_NONE else List[Float32]()
    return Qda(K, arena_words(arena, o.mean, K * d), arena_words(arena, o.rot, K * d * d),
        arena_words(arena, o.logc, K), arena_words(arena, o.s2, K * d), cov^)


def qda_scores(m: Qda, xh: List[Float32], n: Int, d: Int, pair: Bool) raises -> ClsAnswer:
    var records = List[List[Float32]]()
    records.append(m.mean.copy())
    records.append(m.rot.copy())
    records.append(m.logc.copy())
    return cls_scores(CLS_KIND_QDA, xh, n, d, m.K, records^, False, 0.0, pair)


def inputs(tag: String) raises -> Tuple[List[Float32], List[Int32], List[Float32]]:
    return (fixture_f32(PREP_DA_DIR + tag + "-x.f32", DA_N * DA_D), fixture_i32(PREP_DA_DIR + tag + "-y.i32", DA_N),
        fixture_f32(PREP_DA_DIR + tag + "-xh.f32", DA_N * DA_D))


def check_predict(got: List[Int32], tag: String, label: String) raises:
    var share = agreement(got, want_i32(tag, "predict", DA_N))
    require(share > 0.99, label + ": predict agrees with scikit-learn on " + String(share) + " of the rows")


def check_lda(m: Lda, tag: String, xh: List[Float32], eigen: Bool, transform: Bool, cov: Bool, tol: Float64,
              proba_atol: Float64, label: String) raises:
    var n = DA_N
    var d = DA_D
    var K = m.K
    var rows = 1 if K == 2 else K
    if cov:
        check_close(m.cov, want_f64(tag, "cov", d * d), 2e-3, 2e-4, label + ": covariance_")
    check_close(m.pub_coef, want_f64(tag, "coef", rows * d), tol, tol, label + ": coef_")
    check_close(m.pub_inter, want_f64(tag, "intercept", rows), tol, tol, label + ": intercept_")
    var s = lda_scores(m, xh, n, d)
    check_close(s.proba, want_f64(tag, "proba", n * K), 1e-7, proba_atol, label + ": predict_proba")
    if transform:
        var components = min(K - 1, d)
        check_close(m.evr, want_f64(tag, "evr", components), 1e-7, 1e-3, label + ": explained_variance_ratio_")
        var tw = Int(want_i32(tag, "tw", 1)[0])
        check_signed_columns(lda_transform(m, xh, n, d, eigen), m.width, want_f64(tag, "transform", n * tw), tw, n, tol,
            tol, label + ": transform")


# ---- test_lda: two and three classes, solver 'svd', x the two routes ----
def lda_unit(unit: Int) raises:
    var K = 2 + unit // 2
    var blocked = unit % 2 == 0
    var tag = "lda" + String(K)
    var label = tag + " blocked " + String(blocked)
    var data = inputs(tag)
    var m = lda_fit(data[0], data[1], DA_N, DA_D, K, SOLVER_SVD, False, 0.0, False, List[Float64](), blocked)
    check_lda(m, tag, data[2], False, True, False, 2e-3, 2e-3, label)
    check_predict(lda_scores(m, data[2], DA_N, DA_D).codes, tag, label)
    if K == 2:
        check_close(lda_decision(m, data[2], DA_N, DA_D), want_f64(tag, "decision", DA_N), 2e-3, 2e-3,
            label + ": decision_function")


def test_lda() raises:
    check_cases("lda", 4, lda_unit)


# ---- test_qda: (two classes, reg_param 0), (three classes, reg_param 0.1), x the two routes ----
def qda_unit(unit: Int) raises:
    var K = 2 + unit // 2
    var blocked = unit % 2 == 0
    var tag = "qda" + String(K)
    var label = tag + " blocked " + String(blocked)
    var data = inputs(tag)
    var m = qda_fit(data[0], data[1], DA_N, DA_D, K, False, 0.0 if K == 2 else 0.1, False, 0.0, False,
        List[Float64](), blocked)
    check_close(m.scalings, want_f64(tag, "scalings", K * DA_D), 2e-3, 1e-4, label + ": scalings_")
    var s = qda_scores(m, data[2], DA_N, DA_D, False)
    check_close(s.proba, want_f64(tag, "proba", DA_N * K), 1e-7, 2e-3, label + ": predict_proba")
    check_predict(s.codes, tag, label)
    if K == 2:
        check_close(qda_scores(m, data[2], DA_N, DA_D, True).scores, want_f64(tag, "decision", DA_N), 2e-3, 2e-3,
            label + ": decision_function")


def test_qda() raises:
    check_cases("qda", 4, qda_unit)


# ---- test_priors: GaussianNB, MultinomialNB, LDA (priors [1, 2, 1], renormalized), QDA, x the two routes ----
def priors_unit(unit: Int) raises:
    var which = unit // 2
    var blocked = unit % 2 == 0
    var n = DA_N
    var d = DA_D
    var K = 3
    var data = inputs("pri")
    var names: List[String] = ["gnb", "mnb", "lda", "qda"]
    var label = "priors " + names[which] + " blocked " + String(blocked)
    var proba: List[Float32]
    if which == 0:
        var priors: List[Float64] = [0.2, 0.5, 0.3]
        var m = gnb_fit(data[0], data[1], List[Float32](), n, d, K, priors, blocked)
        var records = List[List[Float32]]()
        records.append(m.theta.copy())
        records.append(m.var_.copy())
        records.append(m.cst.copy())
        proba = cls_scores(CLS_KIND_GNB, data[2], n, d, K, records^, False, 0.0, False).proba.copy()
    elif which == 1:
        var priors: List[Float64] = [0.1, 0.6, 0.3]
        var m = dnb_fit(CLS_KIND_MNB, abs_words(data[0]), data[1], List[Float32](), n, d, K, 1.0, CLS_PRIOR_GIVEN,
            priors, False, 0.0, False, blocked)
        var records = List[List[Float32]]()
        records.append(m.log_prob.copy())
        records.append(m.log_prior.copy())
        proba = cls_scores(CLS_KIND_MNB, abs_words(data[2]), n, d, K, records^, False, 0.0, False).proba.copy()
    elif which == 2:
        var priors: List[Float64] = [1.0, 2.0, 1.0]
        var m = lda_fit(data[0], data[1], n, d, K, SOLVER_SVD, False, 0.0, False, priors, blocked)
        proba = lda_scores(m, data[2], n, d).proba.copy()
    else:
        var priors: List[Float64] = [0.25, 0.25, 0.5]
        var m = qda_fit(data[0], data[1], n, d, K, False, 0.0, False, 0.0, False, priors, blocked)
        proba = qda_scores(m, data[2], n, d, False).proba.copy()
    var want = want_f64("pri-" + names[which], "proba", n * K)
    var close = 0
    for i in range(n * K):  # small-loop(oracle: the probabilities)
        if abs(Float64(proba[i]) - want[i]) <= 3e-3:
            close += 1
    var share = Float64(close) / Float64(n * K)
    require(share > 0.99, label + ": " + String(share) + " of the probabilities within 3e-3 of scikit-learn")


def test_priors() raises:
    check_cases("discriminant_priors", 8, priors_unit)


# ---- test_solvers_and_shrinkage: the seven LDA keyword sets for two and three classes, the four QDA sets, x the two
# routes (shrinkage 'auto' is -1.0, the door's word for it) ----
def solver_unit(unit: Int) raises:
    var blocked = unit % 2 == 0
    var row = unit // 2
    var n = DA_N
    var d = DA_D
    if row < 14:
        var K = 2 + row // 7
        var j = row % 7
        var solvers: List[Int] = [SOLVER_LSQR, SOLVER_LSQR, SOLVER_LSQR, SOLVER_EIGEN, SOLVER_EIGEN, SOLVER_EIGEN,
                                  SOLVER_SVD]
        var shrink: List[Float64] = [0.0, -1.0, 0.3, 0.0, -1.0, 0.6, 0.0]
        var has: List[Bool] = [False, True, True, False, True, True, False]
        var tag = "sol" + String(K) + "-lda" + String(j)
        var label = tag + " blocked " + String(blocked)
        var data = inputs("sol" + String(K))
        var m = lda_fit(data[0], data[1], n, d, K, solvers[j], has[j], shrink[j], j == 6, List[Float64](), blocked)
        var view = want_i32(tag, "view", 1)[0] != 0
        check_lda(m, tag, data[2], solvers[j] == SOLVER_EIGEN, view, True, 5e-3, 3e-3, label)
        return
    var j = row - 14
    var K = 3
    var eigen: List[Bool] = [True, True, True, False]
    var shrink: List[Float64] = [0.0, -1.0, 0.2, 0.0]
    var has: List[Bool] = [False, True, True, False]
    var store: List[Bool] = [False, True, False, True]
    var tag = "solq-qda" + String(j)
    var label = tag + " blocked " + String(blocked)
    var data = inputs("solq")
    var m = qda_fit(data[0], data[1], n, d, K, eigen[j], 0.1 if j == 3 else 0.0, has[j], shrink[j], store[j],
        List[Float64](), blocked)
    check_close(m.scalings, want_f64(tag, "scalings", K * d), 2e-3, 1e-4, label + ": scalings_")
    if store[j]:
        check_close(m.cov, want_f64(tag, "cov", K * d * d), 2e-3, 2e-4, label + ": covariance_")
    check_close(qda_scores(m, data[2], n, d, False).proba, want_f64(tag, "proba", n * K), 1e-7, 3e-3,
        label + ": predict_proba")


def test_solvers_and_shrinkage() raises:
    check_cases("discriminant_solvers", 36, solver_unit)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
