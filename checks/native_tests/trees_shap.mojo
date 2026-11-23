# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""TreeExplainer on a host random forest: additivity and the exact path-dependent Shapley values (lane py2mojo-C,
2026-11-22; ported from python/mojolearn/tests/test_x_trees_sanity.py `test_tree_explainer_matches_shap_recursion`,
its RandomForestClassifier arm).

The forest is the original's RandomForestClassifier(n_estimators=5, max_depth=5, random_state=0) fitted on the
`_clf()` training split (`fixtures/trees_sk/`, `py2mojo_forest_fits.mojo`); the explainer is what
`TreeExplainer(m, data=Xa)` builds (bindings/trees_explainer_native_door.mojo `_construct`: tree scale 1 / n_trees,
initial value 0, the cover of the 1050 background rows) and `shap_values(Xb[:40])` runs: `shap_prepare` and
`tree_shap_values` of xtrees/shap_host.mojo. Assertions: the values plus the expected value equal the forest's
probabilities (atol 1e-5), and they equal the tree_path_dependent Shapley values of the same trees and cover
(rtol 1e-4, atol 1e-5). The original took that reference from the shap package's recursion over our exported trees;
here it is an independent oracle in this file, the Shapley sum over every feature subset of the cover-weighted
conditional expectation (the definition the shap recursion computes exactly), so no stored reference depends on
our fitted bits. The DART arm stays in Python: the host DART fit needs the forest image published from the binding
module bindings/_mojolearn_rf_host.mojo. The 40 explained rows are independent oracle rows across the host cores.

KernelExplainer and PermutationExplainer (lane bm-rest-shap, 2026-11-23; ported from test_x_trees_sanity.py
`test_kernel_explainer`, `test_permutation_explainer`, `test_kernel_explainer_logit`): the typed RUNs of
xtrees/agnostic_values_run.mojo over the x_trees glue entry of this image, the model a host forest fitted here (the
`_ForestModel` evaluator: the strict host predict of py2mojo_forest_fits.mojo). The originals compared against the
shap package; here the reference is the exact Shapley value of the same game (KernelExplainer with every coalition
enumerated, M <= 6, is exact): v(S) = the background mean of the model on x_S over the background's other
features (the logit link applied to that mean for link 'logit'). Each RUN also runs with its chunks on the host
workers and must give the serial words exactly. The logit case runs the original's DARTClassifier (fitted through
bm_rest_boost_fits.mojo over the product's DART round RUN, predict_proba as the door's `dart_predict_proba`) and a
binary RandomForestClassifier beside it.

A FLOAT64 MEMBER through the native model handle (lane mf-shaprfe, 2026-11-23; `test_agnostic_logistic_member`):
a binary LogisticRegression fitted through its member entry (core/logistic_member.mojo on the CPU image's QN
entry, as checks/native_tests/meta_runs_gil_free.mojo fits it), explained through `AgnMember` with its predict_proba
written as float64 and narrowed by the RUN (`wide`). LogisticRegression went through the door's `_DoorModel`
before. The reference evaluator `_WideRef` calls the same member into a float64 block and narrows each word with
`cast[DType.float32]` (the door's `_evaluate` cast of the public float64 output): Kernel SHAP (identity and logit)
and Permutation SHAP through `AgnMember` give the reference evaluator's words exactly, serially and with the chunks
on the host workers, and Kernel SHAP equals the exact Shapley values of the reference game (rtol 1e-6, atol 1e-6)
and is additive (atol 1e-5).
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, fixture_f32, fixture_i32, require
from checks.native_tests.py2mojo_forest_fits import RfSpec, Forest, rf_fit, predict
from xtrees.shap import shap_path_width
from xtrees.shap_host import shap_prepare, tree_shap_values
from std.math import log
from core.entry_table import EntryRef
from core.host_predict_threads import host_predict_workers
from core.trees_entry_table import TreesEntryKey
from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.native_tests.testsuite_cases import assert_same_words
from xtrees.glue_runs import publish_x_trees_glue
from xtrees.glue_calls import xt_ref, xt_stack_w64, xt_softmax_rows
from checks.native_tests.bm_rest_boost_fits import DartSpec, DartModel, dart_fit, dart_raw, publish_host_images, trees_key
from xtrees.agnostic_values_run import (
    AgnModel, AgnMember, AgnRequest, agn_expected_run, agn_kernel_values_run, agn_permutation_values_run,
    AGN_SW_IDN_SYN_POOL,
)
from core.device_route import DEVICE_CPU
from core.logistic_member import logistic_member_v2
from core.logistic_native_api import QnNativeDriver
from core.logistic_route import LogisticRoute, logistic_route_pack
from core.member_abi import (
    MemberCall, MemberMatrix, MemberPlan, MemberRef, member_call, member_call_of, member_new, member_free,
    member_kind_capabilities, MEMBER_OP_CAPABILITIES, MEMBER_OP_FIT, MEMBER_OP_OUTPUTS, MEMBER_OP_PREDICT_PROBA,
    MEMBER_F64, MEMBER_I32, MEMBER_GIL_FREE,
)
from glm.host.logistic_native_host import qn_host_native_v1

comptime SK_DIR = "checks/native_tests/fixtures/trees_sk/"
comptime NA = 1050
comptime D = 10
comptime K = 3
comptime ROWS = 40
comptime TREES = 5

comptime SHAP_DIR = "checks/native_tests/fixtures/trees_shap/"
# test_kernel_explainer_logit's fits (check-trim-2, 2026-11-23; the file went from about 12 s to 166 s when that test
# arrived): the two models are fitted on the first rows of the training split, not all 1,050, and the DART model
# boosts DART_TREES rounds, not 20. Its assertions (exact Shapley values of the linked game, additivity in log-odds,
# workers == serial words) hold for any fitted model, so neither size is needed to prove them; DART still drops
# trees in its later rounds (drop_rate 0.1, skip_drop 0.5), and the RF keeps 20 trees of min_samples_leaf 30, so
# every probability the logit link meets stays inside (0, 1).
comptime LOGIT_FIT_ROWS = 400  # RandomForestClassifier(20, max_depth=4, min_samples_leaf=30)
comptime DART_FIT_ROWS = 160  # DARTClassifier: min_child_samples 20, so up to 8 leaves a tree
comptime DART_TREES = 6
comptime _FOREST = MutPointer[Forest, MutUntrackedOrigin]
comptime _DART = MutPointer[DartModel, MutUntrackedOrigin]
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]


def _expect(node: Int, base: Int, c: Int, subset: Int, x: _F32P, colid: _I32P, quesval: _F32P, left: _I32P,
            leaves: _F32P, cover: _I32P) -> Float64:
    """E[f(x) | x_S] of one tree on the path-dependent law: a split on a feature in S follows x, any other split
    averages its children by cover."""
    var g = base + node
    var l = Int(left[g])
    if l == -1:
        return Float64(leaves[g * K + c])
    var f = Int(colid[g])
    if (subset >> f) & 1 == 1:
        if x[f] <= quesval[g]:
            return _expect(l, base, c, subset, x, colid, quesval, left, leaves, cover)
        return _expect(l + 1, base, c, subset, x, colid, quesval, left, leaves, cover)
    var cl = Float64(cover[base + l])
    var cr = Float64(cover[base + l + 1])
    var total = cl + cr
    if total == 0.0:
        return 0.0
    return (cl * _expect(l, base, c, subset, x, colid, quesval, left, leaves, cover)
            + cr * _expect(l + 1, base, c, subset, x, colid, quesval, left, leaves, cover)) / total


def test_tree_explainer_rf() raises:
    var xa = fixture_f32(SK_DIR + "clf-xa.f32", NA * D)
    var xb = fixture_f32(SK_DIR + "clf-xb.f32", 450 * D)
    var ya = fixture_i32(SK_DIR + "clf-ya.i32", NA)
    var spec = RfSpec(True, TREES, 0)
    spec.max_depth = 5
    var forest = rf_fit(spec, xa, NA, D, ya, List[Float32](), K, List[Float32]())
    require(forest.outputs == K and forest.trees == TREES, "forest shape")
    var nodes = forest.nodes()
    var model = forest.model()
    var tscale = List[Float32](length=TREES, fill=Float32(1.0 / Float64(TREES)))
    var cover = List[Int32](length=nodes, fill=0)
    var ev = List[Float32](length=K, fill=Float32(0))
    var meta = List[Int32](length=3, fill=0)
    shap_prepare(model, Int(tscale.unsafe_ptr()), Int(xa.unsafe_ptr()), Int(cover.unsafe_ptr()), Int(ev.unsafe_ptr()),
                 Int(meta.unsafe_ptr()), NA, D, TREES, K, nodes)
    var width = shap_path_width(Int(meta[1]), Int(meta[0]))
    require(width > 0, "shap path width")
    var x = List[Float32](length=ROWS * D, fill=Float32(0))
    for i in range(ROWS * D):  # small-loop(fixture: Xb[:40])
        x[i] = xb[i]
    var phi = List[Float32](length=ROWS * D * K, fill=Float32(0))
    tree_shap_values(model, Int(tscale.unsafe_ptr()), Int(cover.unsafe_ptr()), Int(x.unsafe_ptr()),
                     Int(phi.unsafe_ptr()), ROWS, D, TREES, K, nodes, Int(meta[0]), width)
    var proba = predict(forest, x, ROWS)
    # the Shapley weights |S|! (M - |S| - 1)! / M!
    var weights = List[Float64](length=D, fill=0.0)
    for s in range(D):  # small-loop(oracle: one weight per subset size)
        var w = 1.0
        for a in range(1, s + 1):  # small-loop(s!)
            w *= Float64(a)
        for a in range(1, D - s):  # small-loop((M - s - 1)!)
            w *= Float64(a)
        for a in range(1, D + 1):  # small-loop(/ M!)
            w /= Float64(a)
        weights[s] = w
    var xp = _F32P(unsafe_from_address=Int(x.unsafe_ptr()))
    var colid = _I32P(unsafe_from_address=Int(forest.colid.unsafe_ptr()))
    var quesval = _F32P(unsafe_from_address=Int(forest.quesval.unsafe_ptr()))
    var left = _I32P(unsafe_from_address=Int(forest.left.unsafe_ptr()))
    var leaves = _F32P(unsafe_from_address=Int(forest.leaves.unsafe_ptr()))
    var cp = _I32P(unsafe_from_address=Int(cover.unsafe_ptr()))
    var offsets = forest.offsets.copy()
    var op = offsets.unsafe_ptr()
    var wp = weights.unsafe_ptr()
    var pp = phi.unsafe_ptr()
    var qp = proba.unsafe_ptr()
    var evp = ev.unsafe_ptr()

    def row(r: Int) raises {imm xp, imm colid, imm quesval, imm left, imm leaves, imm cp, imm op, imm wp, imm pp, imm qp,
                            imm evp}:
        var xr = xp + r * D
        for c in range(K):  # small-loop(oracle: the classes of one row)
            var total = Float64(evp[c])
            var exact = List[Float64](length=D, fill=0.0)
            for t in range(TREES):  # small-loop(oracle: the trees)
                var base = Int(op[t])
                var v = List[Float64](length=1 << D, fill=0.0)
                for subset in range(1 << D):  # small-loop(oracle: every feature subset)
                    v[subset] = _expect(0, base, c, subset, xr, colid, quesval, left, leaves, cp) / Float64(TREES)
                for subset in range(1 << D):  # small-loop(oracle: the Shapley sum)
                    var size = 0
                    for j in range(D):  # small-loop(|S|)
                        size += (subset >> j) & 1
                    for j in range(D):  # small-loop(the features outside S)
                        if (subset >> j) & 1 == 0:
                            exact[j] += wp[size] * (v[subset | (1 << j)] - v[subset])
            for j in range(D):  # small-loop(oracle: compare one row's values)
                var got = Float64(pp[(r * D + j) * K + c])
                total += got
                require(abs(got - exact[j]) <= 1e-5 + 1e-4 * abs(exact[j]), "row " + String(r) + " feature "
                    + String(j) + " class " + String(c) + ": phi " + String(got) + ", exact " + String(exact[j]))
            var want = Float64(qp[r * K + c])
            require(abs(total - want) <= 1e-5, "row " + String(r) + " class " + String(c) + ": phi sum + expected "
                + String(total) + ", predict_proba " + String(want))

    check_cases("tree_explainer_rf", ROWS, row)
    _ = xa^
    _ = xb^
    _ = x^
    _ = phi^
    _ = proba^
    _ = cover^
    _ = tscale^
    _ = ev^
    _ = meta^
    _ = model^
    _ = offsets^
    _ = weights^
    _ = forest^


# ---------------------------------------------------------------- KernelExplainer / PermutationExplainer RUNs


def _glue() raises -> EntryRef:
    """This image's x_trees glue entry (the CPU column), published as the x_trees host binding does at init."""
    publish_x_trees_glue("cpu")
    return xt_ref(TreesEntryKey(Int(GLOBAL_NUMERIC_MODE), 0))


def _cols(src: List[Float32], rows: Int, width: Int, take: Int) -> List[Float32]:
    """The first `rows` rows and the first `take` columns of a row-major rows x width matrix (`X[:rows, :take]`)."""
    var out = List[Float32](length=rows * take, fill=Float32(0))
    for i in range(rows):  # small-loop(fixture slice)
        for j in range(take):  # small-loop(fixture slice)
            out[i * take + j] = src[i * width + j]
    return out^


struct _ForestModel(AgnModel):
    """The explained model of these checks: a host forest fitted here (`forest`: the address of the check's
    `Forest`), on the strict host predict route; `x` the explained rows."""
    var forest: Int
    var x: List[Float32]
    var n: Int
    var d: Int
    var k: Int
    var parallel: Bool
    var fx: List[Float32]
    var syn: List[Float32]
    var out: List[Float32]

    def __init__(out self, forest: Int, x: List[Float32], n: Int, d: Int, k: Int, parallel: Bool):
        self.forest = forest
        self.x = x.copy()
        self.n = n
        self.d = d
        self.k = k
        self.parallel = parallel
        self.fx = List[Float32]()
        self.syn = List[Float32]()
        self.out = List[Float32]()

    def explained(mut self) raises -> Int:
        self.fx = predict(_FOREST(unsafe_from_address=self.forest)[], self.x, self.n)
        return Int(self.fx.unsafe_ptr())

    def synthetic(mut self, count: Int, filled: Bool) raises -> Int:
        if len(self.syn) != count:
            self.syn = List[Float32](length=count, fill=Float32(0))
        return Int(self.syn.unsafe_ptr())

    def rows(mut self, n: Int) raises -> Int:
        if n * self.d == len(self.syn):
            self.out = predict(_FOREST(unsafe_from_address=self.forest)[], self.syn, n)
        else:
            var front = List[Float32](length=n * self.d, fill=Float32(0))
            for i in range(n * self.d):  # small-loop(the delta prefix; never on the identical CPU column)
                front[i] = self.syn[i]
            self.out = predict(_FOREST(unsafe_from_address=self.forest)[], front, n)
        return Int(self.out.unsafe_ptr())

    def workers(self) -> Int:
        return host_predict_workers() if self.parallel else 1

    def worker(self) raises -> Self:
        return Self(self.forest, self.x, self.n, self.d, self.k, self.parallel)


def _dart_proba32(m: Int, x: List[Float32], n: Int) raises -> List[Float32]:
    """Binary DARTClassifier.predict_proba as the explainer reads it: the door's `dart_predict_proba` (softmax
    over the rows [0, raw], `xt_stack_w64` + `xt_softmax_rows`) on the helper's native raw score, narrowed to
    float32 (the door's `_evaluate` cast)."""
    var raw = dart_raw(_DART(unsafe_from_address=m)[], x, n)
    var acc = List[Float64](length=max(2 * n, 1), fill=0.0)
    var zero = List[Float64](length=max(n, 1), fill=0.0)
    var glue = xt_ref(trees_key())
    var addrs: List[Int] = [Int(acc.unsafe_ptr()), Int(zero.unsafe_ptr()), Int(raw.unsafe_ptr())]
    _ = xt_stack_w64(glue, addrs, n)
    _ = xt_softmax_rows(glue, Int(acc.unsafe_ptr()), n, 2)
    _ = len(raw)  # read through its address by the stack above
    _ = len(zero)
    var out = List[Float32](length=max(2 * n, 1), fill=Float32(0))
    for i in range(2 * n):  # small-loop(one prediction block, narrowed)
        out[i] = Float32(acc[i])
    return out^


struct _DartEval(AgnModel):
    """The explained model of the DART arm: a binary DART classifier fitted here (`model`: the address of the
    check's `DartModel`), predict_proba through `_dart_proba32`; `x` the explained rows."""
    var model: Int
    var x: List[Float32]
    var n: Int
    var d: Int
    var parallel: Bool
    var fx: List[Float32]
    var syn: List[Float32]
    var out: List[Float32]

    def __init__(out self, model: Int, x: List[Float32], n: Int, d: Int, parallel: Bool):
        self.model = model
        self.x = x.copy()
        self.n = n
        self.d = d
        self.parallel = parallel
        self.fx = List[Float32]()
        self.syn = List[Float32]()
        self.out = List[Float32]()

    def explained(mut self) raises -> Int:
        self.fx = _dart_proba32(self.model, self.x, self.n)
        return Int(self.fx.unsafe_ptr())

    def synthetic(mut self, count: Int, filled: Bool) raises -> Int:
        if len(self.syn) != count:
            self.syn = List[Float32](length=count, fill=Float32(0))
        return Int(self.syn.unsafe_ptr())

    def rows(mut self, n: Int) raises -> Int:
        if n * self.d == len(self.syn):
            self.out = _dart_proba32(self.model, self.syn, n)
        else:
            var front = List[Float32](length=n * self.d, fill=Float32(0))
            for i in range(n * self.d):  # small-loop(the delta prefix; never on the identical CPU column)
                front[i] = self.syn[i]
            self.out = _dart_proba32(self.model, front, n)
        return Int(self.out.unsafe_ptr())

    def workers(self) -> Int:
        return host_predict_workers() if self.parallel else 1

    def worker(self) raises -> Self:
        return Self(self.model, self.x, self.n, self.d, self.parallel)


struct _Shap(Movable):
    var phi: List[Float64]
    var ev: List[Float64]

    def __init__(out self, var phi: List[Float64], var ev: List[Float64]):
        self.phi = phi^
        self.ev = ev^


def _expected(glue: EntryRef, fp: Int, bg: List[Float32], nb: Int, d: Int, k: Int, logit: Bool) raises -> List[Float64]:
    """The explainer's construction: the background mean of the model (then the link)."""
    var on_bg = _ForestModel(fp, bg, nb, d, k, False)
    var y = on_bg.explained()
    var ev = List[Float64](length=k, fill=0.0)
    agn_expected_run(glue, y, nb, k, logit, Int(ev.unsafe_ptr()))
    _ = on_bg^
    return ev^


def _kernel_of[M: AgnModel](glue: EntryRef, mut on_bg: M, mut model: M, x: List[Float32], bg: List[Float32], n: Int,
                            nb: Int, d: Int, k: Int, logit: Bool) raises -> _Shap:
    """`KernelExplainer(m, bg, link).shap_values(x)` with nsamples 'auto' (2 M + 2048, capped at 2^M - 2): `on_bg`
    evaluates the model on the background (the construction), `model` on the explained rows."""
    var ev = List[Float64](length=k, fill=0.0)
    agn_expected_run(glue, on_bg.explained(), nb, k, logit, Int(ev.unsafe_ptr()))
    var phi = List[Float64](length=n * d * k, fill=0.0)
    var req = AgnRequest(Int(x.unsafe_ptr()), Int(bg.unsafe_ptr()), Int(phi.unsafe_ptr()), n, nb, d, k, 0,
                         AGN_SW_IDN_SYN_POOL, False, True)
    agn_kernel_values_run(glue, req, 2 * d + 2048, Int(ev.unsafe_ptr()), logit, model)
    _ = req^
    return _Shap(phi^, ev^)


def _kernel(glue: EntryRef, fp: Int, x: List[Float32], bg: List[Float32], n: Int, nb: Int, d: Int, k: Int,
            logit: Bool, parallel: Bool) raises -> _Shap:
    """`_kernel_of` over the host forest at `fp`."""
    var on_bg = _ForestModel(fp, bg, nb, d, k, False)
    var model = _ForestModel(fp, x, n, d, k, parallel)
    var r = _kernel_of(glue, on_bg, model, x, bg, n, nb, d, k, logit)
    _ = on_bg^
    _ = model^
    return r^


def _permutation(glue: EntryRef, fp: Int, x: List[Float32], bg: List[Float32], n: Int, nb: Int, d: Int, k: Int,
                 npm: Int, seed: Int, parallel: Bool) raises -> _Shap:
    """`PermutationExplainer(m, bg, random_state=seed).shap_values(x, npermutations=npm)`."""
    var ev = _expected(glue, fp, bg, nb, d, k, False)
    var phi = List[Float64](length=n * d * k, fill=0.0)
    var req = AgnRequest(Int(x.unsafe_ptr()), Int(bg.unsafe_ptr()), Int(phi.unsafe_ptr()), n, nb, d, k, seed,
                         AGN_SW_IDN_SYN_POOL, False, True)
    var model = _ForestModel(fp, x, n, d, k, parallel)
    agn_permutation_values_run(glue, req, npm, 0, 0, model)
    _ = model^
    _ = req^
    return _Shap(phi^, ev^)


def _shapley_weights(d: Int) -> List[Float64]:
    """|S|! (M - |S| - 1)! / M! for |S| = 0 .. M - 1."""
    var weights = List[Float64](length=d, fill=0.0)
    for s in range(d):  # small-loop(oracle: one weight per subset size)
        var w = 1.0
        for a in range(1, s + 1):  # small-loop(s!)
            w *= Float64(a)
        for a in range(1, d - s):  # small-loop((M - s - 1)!)
            w *= Float64(a)
        for a in range(1, d + 1):  # small-loop(/ M!)
            w /= Float64(a)
        weights[s] = w
    return weights^


def _exact(fp: Int, x: List[Float32], bg: List[Float32], n: Int, nb: Int, d: Int, k: Int, logit: Bool) raises -> List[Float64]:
    """`_exact_of` over the host forest at `fp`."""
    var model = _ForestModel(fp, x, n, d, k, True)
    var out = _exact_of(model, x, bg, n, nb, d, k, logit)
    _ = model^
    return out^


def _exact_of[M: AgnModel](model: M, x: List[Float32], bg: List[Float32], n: Int, nb: Int, d: Int, k: Int,
                           logit: Bool) raises -> List[Float64]:
    """The exact Shapley values (n x d x k) of v(S) = mean_b f(x_S, bg_b on the other features), linked when
    `logit`: every subset of the d features, the model (a worker evaluator of `model` per row) on all of them at
    once per row; rows across the cores."""
    var out = List[Float64](length=n * d * k, fill=0.0)
    var weights = _shapley_weights(d)
    var op = out.unsafe_ptr()
    var wp = weights.unsafe_ptr()
    var xp = x.unsafe_ptr()
    var bp = bg.unsafe_ptr()

    def row(r: Int) raises {imm model, imm op, imm wp, imm xp, imm bp, imm nb, imm d, imm k, imm logit}:
        var S = 1 << d
        var mine = model.worker()
        var z = _F32P(unsafe_from_address=mine.synthetic(S * nb * d, True))
        for s in range(S):  # small-loop(oracle: every feature subset)
            for b in range(nb):  # small-loop(oracle: the background rows)
                for j in range(d):  # small-loop(oracle: one synthetic row)
                    z[(s * nb + b) * d + j] = xp[r * d + j] if ((s >> j) & 1) == 1 else bp[b * d + j]
        var y = _F32P(unsafe_from_address=mine.rows(S * nb))
        var v = List[Float64](length=S * k, fill=0.0)
        for s in range(S):  # small-loop(oracle: the game's values)
            for c in range(k):  # small-loop(oracle: the outputs)
                var acc = 0.0
                for b in range(nb):  # small-loop(oracle: the background mean)
                    acc += Float64(y[(s * nb + b) * k + c])
                var mean = acc / Float64(nb)
                if logit:
                    require(mean > 0.0 and mean < 1.0, "row " + String(r) + ": a coalition's mean probability "
                        + String(mean) + " is not inside (0, 1)")
                    mean = log(mean / (1.0 - mean))
                v[s * k + c] = mean
        for c in range(k):  # small-loop(oracle: the outputs)
            for s in range(S):  # small-loop(oracle: the Shapley sum)
                var size = 0
                for j in range(d):  # small-loop(|S|)
                    size += (s >> j) & 1
                for j in range(d):  # small-loop(the features outside S)
                    if ((s >> j) & 1) == 0:
                        op[(r * d + j) * k + c] += wp[size] * (v[(s | (1 << j)) * k + c] - v[s * k + c])
        _ = mine^  # z and y are its buffers

    check_cases("shapley oracle", n, row)
    _ = weights^
    return out^


def _close(got: List[Float64], want: List[Float64], rtol: Float64, atol: Float64, what: String) raises:
    require(len(got) == len(want), what + ": extent " + String(len(got)) + " vs " + String(len(want)))
    for i in range(len(got)):  # small-loop(compare)
        require(abs(got[i] - want[i]) <= atol + rtol * abs(want[i]), what + " cell " + String(i) + ": "
            + String(got[i]) + ", reference " + String(want[i]))


def _additive(r: _Shap, f: List[Float32], n: Int, d: Int, k: Int, logit: Bool, rtol: Float64, atol: Float64,
              what: String) raises:
    """sum_j phi[i, j, c] + expected[c] equals the model's output (its log-odds for link 'logit')."""
    for i in range(n):  # small-loop(rows)
        for c in range(k):  # small-loop(outputs)
            var total = r.ev[c]
            for j in range(d):  # small-loop(features)
                total += r.phi[(i * d + j) * k + c]
            var want = Float64(f[i * k + c])
            if logit:
                require(want > 0.0 and want < 1.0, what + " row " + String(i) + ": probability " + String(want)
                    + " is not inside (0, 1)")
                want = log(want / (1.0 - want))
            require(abs(total - want) <= atol + rtol * abs(want), what + " row " + String(i) + " output "
                + String(c) + ": phi sum + expected " + String(total) + ", model " + String(want))


def test_kernel_explainer() raises:
    """test_kernel_explainer: DecisionTreeRegressor(max_depth=5) on `_reg()[:, :6]`, background Xa[:20],
    explained Xb[:5]: phi equals the exact Shapley values (rtol 1e-6, atol 1e-6) and is additive (rtol 1e-6,
    atol 1e-5); RandomForestClassifier(4, max_depth=4, random_state=0) on `_clf()[:, :5]`, background Xc[:15],
    explained Xcb[:3]: three outputs, additive in the probabilities (atol 1e-5)."""
    var glue = _glue()
    var ra = fixture_f32(SK_DIR + "reg-xa.f32", NA * D)
    var rb = fixture_f32(SK_DIR + "reg-xb.f32", 450 * D)
    var rya = fixture_f32(SK_DIR + "reg-ya.f32", NA)
    var xa6 = _cols(ra, NA, D, 6)
    var tree = rf_fit(RfSpec.decision_tree(False, 5), xa6, NA, 6, List[Int32](), rya, 0, List[Float32]())
    require(tree.outputs == 1, "regressor tree outputs")
    var bg = _cols(ra, 20, D, 6)
    var x = _cols(rb, 5, D, 6)
    var fp = Int(MutPointer(to=tree))
    var serial = _kernel(glue, fp, x, bg, 5, 20, 6, 1, False, False)
    var spread = _kernel(glue, fp, x, bg, 5, 20, 6, 1, False, True)
    assert_same_words(spread.phi, serial.phi, "kernel shap: chunks on the workers vs in order")
    var exact = _exact(fp, x, bg, 5, 20, 6, 1, False)
    _close(serial.phi, exact, 1e-6, 1e-6, "kernel shap vs exact Shapley")
    var f = predict(tree, x, 5)
    _additive(serial, f, 5, 6, 1, False, 1e-6, 1e-5, "kernel shap additivity")

    var ca = fixture_f32(SK_DIR + "clf-xa.f32", NA * D)
    var cb = fixture_f32(SK_DIR + "clf-xb.f32", 450 * D)
    var cya = fixture_i32(SK_DIR + "clf-ya.i32", NA)
    var spec = RfSpec(True, 4, 0)
    spec.max_depth = 4
    var forest = rf_fit(spec, _cols(ca, NA, D, 5), NA, 5, cya, List[Float32](), K, List[Float32]())
    require(forest.outputs == K, "classifier forest outputs")
    var bgc = _cols(ca, 15, D, 5)
    var xq = _cols(cb, 3, D, 5)
    var cp = Int(MutPointer(to=forest))
    var pc = _kernel(glue, cp, xq, bgc, 3, 15, 5, K, False, True)
    require(len(pc.phi) == 3 * 5 * K, "kernel shap classifier shape (3, 5, 3)")
    var proba = predict(forest, xq, 3)
    _additive(pc, proba, 3, 5, K, False, 0.0, 1e-5, "kernel shap classifier additivity")
    _ = tree^
    _ = forest^
    _ = ra^
    _ = rb^
    _ = ca^
    _ = cb^


def test_permutation_explainer() raises:
    """test_permutation_explainer: the same tree, background and rows; PermutationExplainer(random_state=0) with
    300 permutations is additive (rtol 1e-6, atol 1e-5) and within 0.15 max|exact| + 1e-6 of the exact values
    (the full-enumeration KernelExplainer)."""
    var glue = _glue()
    var ra = fixture_f32(SK_DIR + "reg-xa.f32", NA * D)
    var rb = fixture_f32(SK_DIR + "reg-xb.f32", 450 * D)
    var rya = fixture_f32(SK_DIR + "reg-ya.f32", NA)
    var tree = rf_fit(RfSpec.decision_tree(False, 5), _cols(ra, NA, D, 6), NA, 6, List[Int32](), rya, 0,
                      List[Float32]())
    var bg = _cols(ra, 20, D, 6)
    var x = _cols(rb, 5, D, 6)
    var fp = Int(MutPointer(to=tree))
    var exact = _kernel(glue, fp, x, bg, 5, 20, 6, 1, False, False)
    var serial = _permutation(glue, fp, x, bg, 5, 20, 6, 1, 300, 0, False)
    var spread = _permutation(glue, fp, x, bg, 5, 20, 6, 1, 300, 0, True)
    assert_same_words(spread.phi, serial.phi, "permutation shap: chunks on the workers vs in order")
    var f = predict(tree, x, 5)
    _additive(serial, f, 5, 6, 1, False, 1e-6, 1e-5, "permutation shap additivity")
    var top = 0.0
    var gap = 0.0
    for i in range(len(exact.phi)):  # small-loop(compare)
        top = max(top, abs(exact.phi[i]))
        gap = max(gap, abs(serial.phi[i] - exact.phi[i]))
    require(gap <= 0.15 * top + 1e-6, "permutation shap: max |pp - exact| " + String(gap) + " > 0.15 * "
        + String(top) + " + 1e-6")
    _ = tree^
    _ = ra^
    _ = rb^


def test_kernel_explainer_logit() raises:
    """test_kernel_explainer_logit: DARTClassifier(n_estimators=DART_TREES, random_state=0) (the original's model,
    with fewer rounds, through bm_rest_boost_fits.mojo `dart_fit` over the product's DART round RUN) on the first
    DART_FIT_ROWS rows and a binary RandomForestClassifier(20, max_depth=4, min_samples_leaf=30, random_state=0) on
    the first LOGIT_FIT_ROWS rows, each of `_clf(n_classes=2)[:, :6]`,
    background Xa[:20], explained Xb[:4], link 'logit': phi equals the exact Shapley values of the linked game
    (rtol 1e-6, atol 1e-6) and is additive in log-odds (atol 1e-5)."""
    var glue = _glue()
    var xa = fixture_f32(SHAP_DIR + "clf2-xa.f32", NA * D)
    var xb = fixture_f32(SHAP_DIR + "clf2-xb.f32", 450 * D)
    var ya = fixture_i32(SHAP_DIR + "clf2-ya.i32", NA)
    var spec = RfSpec(True, 20, 0)
    spec.max_depth = 4
    spec.min_samples_leaf = 30
    var ya_rf = List[Int32](length=LOGIT_FIT_ROWS, fill=Int32(0))
    for i in range(LOGIT_FIT_ROWS):  # small-loop(fixture: the fit rows' class codes)
        ya_rf[i] = ya[i]
    var forest = rf_fit(spec, _cols(xa, LOGIT_FIT_ROWS, D, 6), LOGIT_FIT_ROWS, 6, ya_rf, List[Float32](), 2,
                        List[Float32]())
    require(forest.outputs == 2, "binary forest outputs")
    var bg = _cols(xa, 20, D, 6)
    var x = _cols(xb, 4, D, 6)
    var fp = Int(MutPointer(to=forest))
    var serial = _kernel(glue, fp, x, bg, 4, 20, 6, 2, True, False)
    var spread = _kernel(glue, fp, x, bg, 4, 20, 6, 2, True, True)
    assert_same_words(spread.phi, serial.phi, "logit kernel shap: chunks on the workers vs in order")
    var exact = _exact(fp, x, bg, 4, 20, 6, 2, True)
    _close(serial.phi, exact, 1e-6, 1e-6, "logit kernel shap vs exact Shapley")
    var p = predict(forest, x, 4)
    _additive(serial, p, 4, 6, 2, True, 0.0, 1e-5, "logit kernel shap additivity")

    # The DART arm: the original test's model.
    publish_host_images()
    var xa6 = _cols(xa, DART_FIT_ROWS, D, 6)
    var codes = List[Float32](length=DART_FIT_ROWS, fill=Float32(0))
    var ones = 0
    for i in range(DART_FIT_ROWS):  # small-loop(fixture: the class codes as the fit's float32 targets)
        codes[i] = Float32(ya[i])
        ones += Int(ya[i])
    require(ones > 0 and ones < DART_FIT_ROWS, "DART fit rows: both classes present")
    var dart = dart_fit(DartSpec(DART_TREES, 0), xa6, DART_FIT_ROWS, 6, codes, 1, 1)
    var dp = Int(MutPointer(to=dart))
    var dbg = _DartEval(dp, bg, 20, 6, False)
    var dserial_m = _DartEval(dp, x, 4, 6, False)
    var dserial = _kernel_of(glue, dbg, dserial_m, x, bg, 4, 20, 6, 2, True)
    var dbg2 = _DartEval(dp, bg, 20, 6, False)
    var dspread_m = _DartEval(dp, x, 4, 6, True)
    var dspread = _kernel_of(glue, dbg2, dspread_m, x, bg, 4, 20, 6, 2, True)
    assert_same_words(dspread.phi, dserial.phi, "DART logit kernel shap: chunks on the workers vs in order")
    var dexact = _exact_of(dspread_m, x, bg, 4, 20, 6, 2, True)
    _close(dserial.phi, dexact, 1e-6, 1e-6, "DART logit kernel shap vs exact Shapley")
    var pd = _dart_proba32(dp, x, 4)
    _additive(dserial, pd, 4, 6, 2, True, 0.0, 1e-5, "DART logit kernel shap additivity")
    _ = dbg^
    _ = dbg2^
    _ = dserial_m^
    _ = dspread_m^
    _ = dart^
    _ = xa6^
    _ = codes^
    _ = ya_rf^
    _ = forest^
    _ = xa^
    _ = xb^


# ---- a float64 member through the native model handle (lane mf-shaprfe) ----
comptime LR_FIT_ROWS = 200
comptime LR_D = 4


struct _Lr(Movable):
    """One fitted binary LogisticRegression member: its plan (AgnMember reads it by address: keep this value
    unmoved while a handle points at it) and its state."""
    var plan: MemberPlan
    var state: Int

    def __init__(out self, var plan: MemberPlan, state: Int):
        self.plan = plan^
        self.state = state


def _lr_fit(x: List[Float32], codes: List[Int32], n: Int, d: Int) raises -> _Lr:
    """LogisticRegression() (l2 1.0, intercept, lbfgs) fitted on n rows of d features and binary codes through
    the member ABI on the CPU image's QN entry (meta_runs_gil_free.mojo `_fit`)."""
    var plan = MemberPlan()
    plan.route = logistic_route_pack(LogisticRoute(DEVICE_CPU, Int(GLOBAL_NUMERIC_MODE), 0, False, False,
                                                   QnNativeDriver(qn_host_native_v1)))
    plan.member = MemberRef(EntryRef(logistic_member_v2, 0), 0)
    var probe = MemberCall(plan.route)
    plan.member.capabilities = member_call(plan.member, MEMBER_OP_CAPABILITIES, probe)
    plan.params = [0.0, 1.0, 0.0, 1.0, 100.0, 20.0, 10.0, 0.0, 1e-4, 1e-6]
    plan.classes = 2
    require((member_kind_capabilities(plan) & MEMBER_GIL_FREE) != 0,
            "the logistic kind lost MEMBER_GIL_FREE: SHAP would run its chunks serially")
    var call = member_new(plan)
    call.x = MemberMatrix(Int(x.unsafe_ptr()), n, d)
    call.y = MemberMatrix(Int(codes.unsafe_ptr()), n, 1, MEMBER_I32)
    call.classes = 2
    try:
        _ = member_call(plan.member, MEMBER_OP_FIT, call)
    except e:
        member_free(plan, call)
        raise e^
    _ = len(x)  # read through the fit above
    _ = len(codes)
    var state = call.state
    return _Lr(plan^, state)


def _lr_free(mut f: _Lr):
    var call = MemberCall(f.plan.route, f.state)
    member_free(f.plan, call)
    f.state = 0


struct _WideRef(AgnModel):
    """The reference evaluator: the member's predict_proba into a float64 block, each word narrowed with
    `cast[DType.float32]` (the door's cast of the public float64 output). `plan` the address of the `_Lr`'s plan."""
    var plan: Int
    var state: Int
    var x: List[Float32]
    var n: Int
    var d: Int
    var k: Int
    var parallel: Bool
    var fx: List[Float32]
    var syn: List[Float32]
    var out: List[Float32]

    def __init__(out self, plan: Int, state: Int, x: List[Float32], n: Int, d: Int, k: Int, parallel: Bool):
        self.plan = plan
        self.state = state
        self.x = x.copy()
        self.n = n
        self.d = d
        self.k = k
        self.parallel = parallel
        self.fx = List[Float32]()
        self.syn = List[Float32]()
        self.out = List[Float32]()

    def _proba(self, src: Int, n: Int) raises -> List[Float32]:
        var plan = MutPointer[MemberPlan, MutUntrackedOrigin](unsafe_from_address=self.plan)
        var wide = List[Float64](length=max(n * self.k, 1), fill=0.0)
        var call = member_call_of(plan[], self.state)
        call.x = MemberMatrix(src, n, self.d)
        call.dst = MemberMatrix(Int(wide.unsafe_ptr()), n, self.k, MEMBER_F64)
        _ = member_call(plan[].member, MEMBER_OP_PREDICT_PROBA, call)
        var words = List[Float32](length=max(n * self.k, 1), fill=Float32(0))
        for i in range(n * self.k):  # small-loop(oracle: the door's float32 cast, word by word)
            words[i] = wide[i].cast[DType.float32]()
        _ = wide^
        return words^

    def explained(mut self) raises -> Int:
        self.fx = self._proba(Int(self.x.unsafe_ptr()), self.n)
        return Int(self.fx.unsafe_ptr())

    def synthetic(mut self, count: Int, filled: Bool) raises -> Int:
        if len(self.syn) != count:
            self.syn = List[Float32](length=count, fill=Float32(0))
        return Int(self.syn.unsafe_ptr())

    def rows(mut self, n: Int) raises -> Int:
        self.out = self._proba(Int(self.syn.unsafe_ptr()), n)
        return Int(self.out.unsafe_ptr())

    def workers(self) -> Int:
        return host_predict_workers() if self.parallel else 1

    def worker(self) raises -> Self:
        return Self(self.plan, self.state, self.x, self.n, self.d, self.k, self.parallel)


def _handle(f: _Lr, plan_at: Int, x: List[Float32], n: Int, parallel: Bool, refused: List[Int]) raises -> AgnMember:
    """The door's native model handle over the fitted member (`_member_open`: predict_proba, float64); `plan_at`
    the address of `f.plan`."""
    return AgnMember(f.plan.member, plan_at, f.state, MEMBER_OP_PREDICT_PROBA,
                     Int(x.unsafe_ptr()), n, LR_D, 2, parallel, Int(refused.unsafe_ptr()), True, True)


def _lr_permutation[M: AgnModel](glue: EntryRef, mut on_bg: M, mut model: M, x: List[Float32], bg: List[Float32],
                                 n: Int, nb: Int, seed: Int) raises -> _Shap:
    """`PermutationExplainer(m, bg, random_state=seed).shap_values(x, npermutations=40)` over `model`."""
    var ev = List[Float64](length=2, fill=0.0)
    agn_expected_run(glue, on_bg.explained(), nb, 2, False, Int(ev.unsafe_ptr()))
    var phi = List[Float64](length=n * LR_D * 2, fill=0.0)
    var req = AgnRequest(Int(x.unsafe_ptr()), Int(bg.unsafe_ptr()), Int(phi.unsafe_ptr()), n, nb, LR_D, 2, seed,
                         AGN_SW_IDN_SYN_POOL, False, True)
    agn_permutation_values_run(glue, req, 40, 0, 0, model)
    _ = req^
    return _Shap(phi^, ev^)


def test_agnostic_logistic_member() raises:
    """A binary LogisticRegression member (`_clf(n_classes=2)[:LR_FIT_ROWS, :4]`), background Xa[:20], explained
    Xb[:4], through `AgnMember` (float64 predict_proba, narrowed in the RUN): Kernel SHAP identity and logit and
    Permutation SHAP equal the reference evaluator's words exactly, in order and on the workers; Kernel SHAP
    equals the exact Shapley values (rtol 1e-6, atol 1e-6) and is additive (atol 1e-5)."""
    var glue = _glue()
    var xa = fixture_f32(SHAP_DIR + "clf2-xa.f32", NA * D)
    var xb = fixture_f32(SHAP_DIR + "clf2-xb.f32", 450 * D)
    var ya = fixture_i32(SHAP_DIR + "clf2-ya.i32", NA)
    var codes = List[Int32](length=LR_FIT_ROWS, fill=Int32(0))
    for i in range(LR_FIT_ROWS):  # small-loop(fixture: the fit rows' class codes)
        codes[i] = ya[i]
    var fit_x = _cols(xa, LR_FIT_ROWS, D, LR_D)
    var lr = _lr_fit(fit_x, codes, LR_FIT_ROWS, LR_D)
    var asked = member_call_of(lr.plan, lr.state)
    asked.flags = MEMBER_OP_PREDICT_PROBA
    _ = member_call(lr.plan.member, MEMBER_OP_OUTPUTS, asked)
    require(asked.out_dtype == MEMBER_F64 and asked.out_cols == 2,
            "LogisticRegression's member predict_proba is no longer float64 x 2: this check no longer covers the "
            + "float64 handle")
    var bg = _cols(xa, 20, D, LR_D)
    var x = _cols(xb, 4, D, LR_D)
    var refused = List[Int](length=1, fill=0)
    var plan_at = Int(UnsafePointer(to=lr.plan))
    for logit in range(2):  # small-loop(the two links)
        var linked = logit == 1
        var label = String("logistic member kernel shap, link ") + ("logit" if linked else "identity")
        var ref_bg = _WideRef(plan_at, lr.state, bg, 20, LR_D, 2, False)
        var ref_m = _WideRef(plan_at, lr.state, x, 4, LR_D, 2, False)
        var want = _kernel_of(glue, ref_bg, ref_m, x, bg, 4, 20, LR_D, 2, linked)
        var h_bg = _handle(lr, plan_at, bg, 20, False, refused)
        var h_serial = _handle(lr, plan_at, x, 4, False, refused)
        var serial = _kernel_of(glue, h_bg, h_serial, x, bg, 4, 20, LR_D, 2, linked)
        var h_bg2 = _handle(lr, plan_at, bg, 20, True, refused)
        var h_spread = _handle(lr, plan_at, x, 4, True, refused)
        var spread = _kernel_of(glue, h_bg2, h_spread, x, bg, 4, 20, LR_D, 2, linked)
        assert_same_words(serial.ev, want.ev, label + ": expected value, handle vs reference")
        assert_same_words(serial.phi, want.phi, label + ": handle vs reference evaluator")
        assert_same_words(spread.phi, serial.phi, label + ": chunks on the workers vs in order")
        var exact = _exact_of(ref_m, x, bg, 4, 20, LR_D, 2, linked)
        _close(serial.phi, exact, 1e-6, 1e-6, label + " vs exact Shapley")
        var p = ref_m._proba(Int(x.unsafe_ptr()), 4)
        _additive(serial, p, 4, LR_D, 2, linked, 0.0, 1e-5, label + " additivity")
        _ = h_bg^
        _ = h_serial^
        _ = h_bg2^
        _ = h_spread^
        _ = ref_bg^
        _ = ref_m^
    var pref_bg = _WideRef(plan_at, lr.state, bg, 20, LR_D, 2, False)
    var pref_m = _WideRef(plan_at, lr.state, x, 4, LR_D, 2, False)
    var pwant = _lr_permutation(glue, pref_bg, pref_m, x, bg, 4, 20, 0)
    var p_bg = _handle(lr, plan_at, bg, 20, False, refused)
    var p_serial = _handle(lr, plan_at, x, 4, False, refused)
    var pserial = _lr_permutation(glue, p_bg, p_serial, x, bg, 4, 20, 0)
    var p_bg2 = _handle(lr, plan_at, bg, 20, True, refused)
    var p_spread = _handle(lr, plan_at, x, 4, True, refused)
    var pspread = _lr_permutation(glue, p_bg2, p_spread, x, bg, 4, 20, 0)
    assert_same_words(pserial.phi, pwant.phi, "logistic member permutation shap: handle vs reference evaluator")
    assert_same_words(pspread.phi, pserial.phi, "logistic member permutation shap: chunks on the workers vs in order")
    var pp = pref_m._proba(Int(x.unsafe_ptr()), 4)
    _additive(pserial, pp, 4, LR_D, 2, False, 0.0, 1e-5, "logistic member permutation shap additivity")
    require(refused[0] == 0, "the logistic member refused a chunk")
    _ = p_bg^
    _ = p_serial^
    _ = p_bg2^
    _ = p_spread^
    _ = pref_bg^
    _ = pref_m^
    _lr_free(lr)
    _ = refused^
    _ = fit_x^
    _ = codes^
    _ = bg^
    _ = x^
    _ = xa^
    _ = xb^


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
