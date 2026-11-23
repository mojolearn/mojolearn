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
from std.math import log, exp
from core.entry_table import EntryRef
from core.host_predict_threads import host_predict_workers
from core.trees_entry_table import TreesEntryKey
from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.native_tests.testsuite_cases import assert_same_words
from xtrees.glue_runs import publish_x_trees_glue
from xtrees.glue_calls import xt_ref, xt_stack_w64, xt_softmax_rows
from checks.native_tests.bm_rest_boost_fits import DartSpec, DartModel, dart_fit, dart_raw, publish_host_images, trees_key
from xtrees.agnostic_values_run import (
    AgnModel, AgnMember, AgnMeta, AgnMetaPlan, AgnRequest, agn_expected_run, agn_kernel_values_run, agn_permutation_values_run,
    agn_labels_run, AGN_SW_IDN_SYN_POOL, AGN_META_SOFT, AGN_META_REGRESS, AGN_META_HARD, AGN_META_BAG,
    AGN_META_STACK, AGN_META_OVR, AGN_META_CALIBRATED, AGN_META_ADA, AGN_META_COLUMNS, AgnLabelProp,
)
from xtrees.trees_meta_vote_run import (
    VoteMember, CalPair, CalFit, STACK_PROBA, STACK_PREDICT, ada_classifier_run, ADA_PROBA,
)
from core.neighbors_native_driver import XnDriver
from x_neighbors.native_descriptor_host import xn_native_v1
from neighbors.label_propagation_run import LpQueryRequest
from core.entry_table import entry_ref_word
from core.device_route import DeviceRoute
from core.member_families import MEMBER_FAMILY_NAIVE_BAYES
from core.naive_bayes_member import naive_bayes_member_v2, NB_KIND_GAUSSIAN, NB_KIND_CATEGORICAL
from x_prep.transport_host import x_prep_transport_host_v1
from x_prep.transport_abi import XP_OP_COUNT
from core.device_route import DEVICE_CPU
from core.logistic_member import logistic_member_v2
from core.logistic_native_api import QnNativeDriver
from core.logistic_route import LogisticRoute, logistic_route_pack
from core.member_abi import (
    MemberCall, MemberMatrix, MemberPlan, MemberRef, member_call, member_call_of, member_new, member_free,
    member_kind_capabilities, MEMBER_OP_CAPABILITIES, MEMBER_OP_FIT, MEMBER_OP_OUTPUTS, MEMBER_OP_PREDICT_PROBA,
    MEMBER_F64, MEMBER_F32, MEMBER_I32, MEMBER_GIL_FREE, MEMBER_OP_PREDICT, MEMBER_OP_DECISION, MEMBER_I64,
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


def _handle(glue: EntryRef, f: _Lr, plan_at: Int, x: List[Float32], n: Int, parallel: Bool) raises -> AgnMember:
    """The door's native model handle over the fitted member (`_member_open`: predict_proba, float64); `plan_at`
    the address of `f.plan`."""
    return AgnMember(f.plan.member, plan_at, f.state, MEMBER_OP_PREDICT_PROBA,
                     Int(x.unsafe_ptr()), n, LR_D, 2, parallel, True, True, glue, 0, 0)


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
    var plan_at = Int(UnsafePointer(to=lr.plan))
    for logit in range(2):  # small-loop(the two links)
        var linked = logit == 1
        var label = String("logistic member kernel shap, link ") + (String("logit") if linked else String("identity"))
        var ref_bg = _WideRef(plan_at, lr.state, bg, 20, LR_D, 2, False)
        var ref_m = _WideRef(plan_at, lr.state, x, 4, LR_D, 2, False)
        var want = _kernel_of(glue, ref_bg, ref_m, x, bg, 4, 20, LR_D, 2, linked)
        var h_bg = _handle(glue, lr, plan_at, bg, 20, False)
        var h_serial = _handle(glue, lr, plan_at, x, 4, False)
        var serial = _kernel_of(glue, h_bg, h_serial, x, bg, 4, 20, LR_D, 2, linked)
        var h_bg2 = _handle(glue, lr, plan_at, bg, 20, True)
        var h_spread = _handle(glue, lr, plan_at, x, 4, True)
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
    var p_bg = _handle(glue, lr, plan_at, bg, 20, False)
    var p_serial = _handle(glue, lr, plan_at, x, 4, False)
    var pserial = _lr_permutation(glue, p_bg, p_serial, x, bg, 4, 20, 0)
    var p_bg2 = _handle(glue, lr, plan_at, bg, 20, True)
    var p_spread = _handle(glue, lr, plan_at, x, 4, True)
    var pspread = _lr_permutation(glue, p_bg2, p_spread, x, bg, 4, 20, 0)
    assert_same_words(pserial.phi, pwant.phi, "logistic member permutation shap: handle vs reference evaluator")
    assert_same_words(pspread.phi, pserial.phi, "logistic member permutation shap: chunks on the workers vs in order")
    var pp = pref_m._proba(Int(x.unsafe_ptr()), 4)
    _additive(pserial, pp, 4, LR_D, 2, False, 0.0, 1e-5, "logistic member permutation shap additivity")
    _ = p_bg^
    _ = p_serial^
    _ = p_bg2^
    _ = p_spread^
    _ = pref_bg^
    _ = pref_m^
    _lr_free(lr)
    _ = fit_x^
    _ = codes^
    _ = bg^
    _ = x^
    _ = xa^
    _ = xb^


# ---------------------------------------------------------------- lane mf-shap2 (2026-11-23): the new served kinds
# THE LABEL MAP (`test_agn_labels_run`): `agn_labels_run` against an exact host oracle (dst[i] = labels[codes[i]]),
# split across the host's tasks and in one gather alike, and its refusal of a code outside the table. Fail-proof
# edit: `codes + 4 * r0` to `codes + 4 * r0 + 4` in its task (the split differs from the oracle).
# A CLASSIFIER EXPLAINED THROUGH PREDICT (`test_agnostic_label_member`): a binary GaussianNB member (its predict
# writes int32 codes) explained through `AgnMember`'s label route with labels [-1.5, 4.0]; the reference evaluator
# maps the member's codes on the host. Kernel and Permutation SHAP equal the reference words exactly; Kernel SHAP
# equals the exact Shapley values (M = 4: every coalition) and is additive. Fail-proof edit: pass
# `self.n_labels - 1` as the table length in `AgnMember._run` (the gather refuses code 1).
# CategoricalNB (`test_agnostic_categorical_member`): the kind-4 member fitted here, its float32 predict_proba
# through `AgnMember`: Kernel and Permutation SHAP equal the reference evaluator's words, exact Shapley, additive.
# Fail-proof edit: `NB_KIND_CATEGORICAL` to `NB_KIND_COMPLEMENT` in `naive_bayes_kind_published` (the member
# refuses kind 4: "is not published"); its fitted state is proven word for word in prep_naive_bayes.mojo
# `test_categorical_member`.
# A SOFT VOTE (`test_agnostic_vote`): two LogisticRegression members (weights 1 and 2) folded by `AgnMeta` over
# `vote_soft_run`: its outputs equal the hand formula (rtol 1e-6), Kernel SHAP equals the exact Shapley values of
# the hand formula's game (rtol 1e-5, atol 1e-6), the chunks on the workers give the serial words, and Permutation
# SHAP is additive. Fail-proof edit: `n * width` to `n` in `AgnMeta._run`'s soft narrowing (the second class column is
# garbage).


def test_agn_labels_run() raises:
    var glue = _glue()
    var n = 1 << 20
    var table: List[Float32] = [-1.5, 4.0, 7.25]
    var codes = List[Int32](length=n, fill=Int32(0))
    for i in range(n):  # small-loop(fixture: codes cycling through the table)
        codes[i] = Int32((i * 7 + i // 3) % 3)
    var split = List[Float32](length=n, fill=Float32(0))
    var whole = List[Float32](length=n, fill=Float32(0))
    agn_labels_run(glue, Int(table.unsafe_ptr()), 3, Int(codes.unsafe_ptr()), n, Int(split.unsafe_ptr()), True)
    agn_labels_run(glue, Int(table.unsafe_ptr()), 3, Int(codes.unsafe_ptr()), n, Int(whole.unsafe_ptr()), False)
    for i in range(n):  # small-loop(oracle: every row)
        var want = table[Int(codes[i])]
        require(split[i] == want and whole[i] == want, "label map row " + String(i) + ": split " + String(split[i])
            + ", one gather " + String(whole[i]) + ", oracle " + String(want))
    codes[n - 1] = 3
    var refused = False
    try:
        agn_labels_run(glue, Int(table.unsafe_ptr()), 3, Int(codes.unsafe_ptr()), n, Int(split.unsafe_ptr()), True)
    except e:
        refused = "out of range" in String(e)
    require(refused, "the label map took a code outside its table")
    _ = table^
    _ = codes^
    _ = split^
    _ = whole^


def _nb_route() -> DeviceRoute:
    """The host x_prep image's route for the naive Bayes members (no route flags: float32 probabilities)."""
    var mode = Int(GLOBAL_NUMERIC_MODE)
    var route = DeviceRoute(DEVICE_CPU)
    route.mode = mode
    route.vendor = 0
    route.entry = EntryRef(x_prep_transport_host_v1, entry_ref_word(mode, 0, XP_OP_COUNT))
    route.flags = 0
    return route


def _nb_fit(var params: List[Float64], x: List[Float32], codes: List[Int32], n: Int, d: Int, k: Int) raises -> _Lr:
    """A naive Bayes member (`params`: its NEW words) fitted on n rows of d features and k class codes."""
    var plan = MemberPlan()
    plan.route = _nb_route()
    plan.family = MEMBER_FAMILY_NAIVE_BAYES
    plan.kind = Int(params[0])
    plan.member = MemberRef(EntryRef(naive_bayes_member_v2, 0), 0)
    var probe = MemberCall(plan.route)
    plan.member.capabilities = member_call(plan.member, MEMBER_OP_CAPABILITIES, probe)
    plan.params = params^
    plan.classes = k
    var call = member_new(plan)
    call.x = MemberMatrix(Int(x.unsafe_ptr()), n, d)
    call.y = MemberMatrix(Int(codes.unsafe_ptr()), n, 1, MEMBER_I32)
    call.classes = k
    try:
        _ = member_call(plan.member, MEMBER_OP_FIT, call)
    except e:
        member_free(plan, call)
        raise e^
    _ = len(x)
    _ = len(codes)
    var state = call.state
    return _Lr(plan^, state)


struct _OpRef(AgnModel):
    """A reference evaluator: the member's `operation` on the rows. `labels` empty: its float32 output (k columns);
    else its int32 codes mapped to `labels` on the host (k is 1)."""
    var plan: Int
    var state: Int
    var operation: Int
    var labels: List[Float32]
    var x: List[Float32]
    var n: Int
    var d: Int
    var k: Int
    var fx: List[Float32]
    var syn: List[Float32]
    var out: List[Float32]

    def __init__(out self, plan: Int, state: Int, operation: Int, labels: List[Float32], x: List[Float32], n: Int,
                 d: Int, k: Int):
        self.plan = plan
        self.state = state
        self.operation = operation
        self.labels = labels.copy()
        self.x = x.copy()
        self.n = n
        self.d = d
        self.k = k
        self.fx = List[Float32]()
        self.syn = List[Float32]()
        self.out = List[Float32]()

    def answer(self, src: Int, n: Int) raises -> List[Float32]:
        var plan = MutPointer[MemberPlan, MutUntrackedOrigin](unsafe_from_address=self.plan)
        var call = member_call_of(plan[], self.state)
        call.x = MemberMatrix(src, n, self.d)
        var words = List[Float32](length=max(n * self.k, 1), fill=Float32(0))
        if len(self.labels) == 0:
            call.dst = MemberMatrix(Int(words.unsafe_ptr()), n, self.k, MEMBER_F32)
            _ = member_call(plan[].member, self.operation, call)
            return words^
        var codes = List[Int32](length=max(n, 1), fill=Int32(0))
        call.dst = MemberMatrix(Int(codes.unsafe_ptr()), n, 1, MEMBER_I32)
        _ = member_call(plan[].member, self.operation, call)
        for i in range(n):  # small-loop(oracle: the host label map)
            words[i] = self.labels[Int(codes[i])]
        return words^

    def explained(mut self) raises -> Int:
        self.fx = self.answer(Int(self.x.unsafe_ptr()), self.n)
        return Int(self.fx.unsafe_ptr())

    def synthetic(mut self, count: Int, filled: Bool) raises -> Int:
        if len(self.syn) != count:
            self.syn = List[Float32](length=count, fill=Float32(0))
        return Int(self.syn.unsafe_ptr())

    def rows(mut self, n: Int) raises -> Int:
        self.out = self.answer(Int(self.syn.unsafe_ptr()), n)
        return Int(self.out.unsafe_ptr())

    def workers(self) -> Int:
        return 1

    def worker(self) raises -> Self:
        return Self(self.plan, self.state, self.operation, self.labels, self.x, self.n, self.d, self.k)


def _exact_serial[M: AgnModel](mut model: M, x: List[Float32], bg: List[Float32], n: Int, nb: Int, d: Int,
                               k: Int) raises -> List[Float64]:
    """`_exact_of` (identity link) with the rows in order on this thread: for a member kind that is not traced for
    parallel workers (the naive Bayes members are `MEMBER_GIL_DOOR`)."""
    var out = List[Float64](length=n * d * k, fill=0.0)
    var weights = _shapley_weights(d)
    var S = 1 << d
    for r in range(n):  # small-loop(oracle: the explained rows, in order)
        var z = _F32P(unsafe_from_address=model.synthetic(S * nb * d, True))
        for s in range(S):  # small-loop(oracle: every feature subset)
            for b in range(nb):  # small-loop(oracle: the background rows)
                for j in range(d):  # small-loop(oracle: one synthetic row)
                    z[(s * nb + b) * d + j] = x[r * d + j] if ((s >> j) & 1) == 1 else bg[b * d + j]
        var y = _F32P(unsafe_from_address=model.rows(S * nb))
        var v = List[Float64](length=S * k, fill=0.0)
        for s in range(S):  # small-loop(oracle: the game's values)
            for c in range(k):  # small-loop(oracle: the outputs)
                var acc = 0.0
                for b in range(nb):  # small-loop(oracle: the background mean)
                    acc += Float64(y[(s * nb + b) * k + c])
                v[s * k + c] = acc / Float64(nb)
        for c in range(k):  # small-loop(oracle: the outputs)
            for s in range(S):  # small-loop(oracle: the Shapley sum)
                var size = 0
                for j in range(d):  # small-loop(|S|)
                    size += (s >> j) & 1
                for j in range(d):  # small-loop(the features outside S)
                    if ((s >> j) & 1) == 0:
                        out[(r * d + j) * k + c] += weights[size] * (v[(s | (1 << j)) * k + c] - v[s * k + c])
    return out^


def _perm_of[M: AgnModel](glue: EntryRef, mut on_bg: M, mut model: M, x: List[Float32], bg: List[Float32], n: Int,
                          nb: Int, d: Int, k: Int, seed: Int) raises -> _Shap:
    """`PermutationExplainer(m, bg, random_state=seed).shap_values(x, npermutations=40)` over `model`."""
    var ev = List[Float64](length=k, fill=0.0)
    agn_expected_run(glue, on_bg.explained(), nb, k, False, Int(ev.unsafe_ptr()))
    var phi = List[Float64](length=n * d * k, fill=0.0)
    var req = AgnRequest(Int(x.unsafe_ptr()), Int(bg.unsafe_ptr()), Int(phi.unsafe_ptr()), n, nb, d, k, seed,
                         AGN_SW_IDN_SYN_POOL, False, True)
    agn_permutation_values_run(glue, req, 40, 0, 0, model)
    _ = req^
    return _Shap(phi^, ev^)


def test_agnostic_label_member() raises:
    """A binary GaussianNB member on `_clf(n_classes=2)[:LR_FIT_ROWS, :4]`, background Xa[:20], explained Xb[:4],
    explained through predict: its int32 codes mapped to labels [-1.5, 4.0] by the RUN's glue gather."""
    var glue = _glue()
    var xa = fixture_f32(SHAP_DIR + "clf2-xa.f32", NA * D)
    var xb = fixture_f32(SHAP_DIR + "clf2-xb.f32", 450 * D)
    var ya = fixture_i32(SHAP_DIR + "clf2-ya.i32", NA)
    var codes = List[Int32](length=LR_FIT_ROWS, fill=Int32(0))
    for i in range(LR_FIT_ROWS):  # small-loop(fixture: the fit rows' class codes)
        codes[i] = ya[i]
    var fit_x = _cols(xa, LR_FIT_ROWS, D, LR_D)
    var gnb_words: List[Float64] = [Float64(NB_KIND_GAUSSIAN), 1e-9, 0.0, 0.0, 0.0]
    var gnb = _nb_fit(gnb_words^, fit_x, codes, LR_FIT_ROWS, LR_D, 2)
    var asked = member_call_of(gnb.plan, gnb.state)
    asked.flags = MEMBER_OP_PREDICT
    _ = member_call(gnb.plan.member, MEMBER_OP_OUTPUTS, asked)
    require(asked.out_dtype == MEMBER_I32 and asked.out_cols == 1,
            "GaussianNB's member predict no longer writes one int32 code per row: this check no longer covers the "
            + "label route")
    var labels: List[Float32] = [-1.5, 4.0]
    var bg = _cols(xa, 20, D, LR_D)
    var x = _cols(xb, 4, D, LR_D)
    var plan_at = Int(UnsafePointer(to=gnb.plan))
    var lab_at = Int(labels.unsafe_ptr())
    var ref_bg = _OpRef(plan_at, gnb.state, MEMBER_OP_PREDICT, labels, bg, 20, LR_D, 1)
    var ref_m = _OpRef(plan_at, gnb.state, MEMBER_OP_PREDICT, labels, x, 4, LR_D, 1)
    var want = _kernel_of(glue, ref_bg, ref_m, x, bg, 4, 20, LR_D, 1, False)
    var h_bg = AgnMember(gnb.plan.member, plan_at, gnb.state, MEMBER_OP_PREDICT, Int(bg.unsafe_ptr()), 20, LR_D, 1,
                         False, False, True, glue, lab_at, 2)
    var h = AgnMember(gnb.plan.member, plan_at, gnb.state, MEMBER_OP_PREDICT, Int(x.unsafe_ptr()), 4, LR_D, 1,
                      False, False, True, glue, lab_at, 2)
    var got = _kernel_of(glue, h_bg, h, x, bg, 4, 20, LR_D, 1, False)
    assert_same_words(got.ev, want.ev, "label member kernel shap: expected value, handle vs reference")
    assert_same_words(got.phi, want.phi, "label member kernel shap: handle vs reference evaluator")
    var exact = _exact_serial(ref_m, x, bg, 4, 20, LR_D, 1)
    _close(got.phi, exact, 1e-6, 1e-6, "label member kernel shap vs exact Shapley")
    var f = ref_m.answer(Int(x.unsafe_ptr()), 4)
    _additive(got, f, 4, LR_D, 1, False, 0.0, 1e-5, "label member kernel shap additivity")
    var p_ref_bg = _OpRef(plan_at, gnb.state, MEMBER_OP_PREDICT, labels, bg, 20, LR_D, 1)
    var p_ref_m = _OpRef(plan_at, gnb.state, MEMBER_OP_PREDICT, labels, x, 4, LR_D, 1)
    var pwant = _perm_of(glue, p_ref_bg, p_ref_m, x, bg, 4, 20, LR_D, 1, 0)
    var p_bg = AgnMember(gnb.plan.member, plan_at, gnb.state, MEMBER_OP_PREDICT, Int(bg.unsafe_ptr()), 20, LR_D, 1,
                         False, False, True, glue, lab_at, 2)
    var p_h = AgnMember(gnb.plan.member, plan_at, gnb.state, MEMBER_OP_PREDICT, Int(x.unsafe_ptr()), 4, LR_D, 1,
                        False, False, True, glue, lab_at, 2)
    var pgot = _perm_of(glue, p_bg, p_h, x, bg, 4, 20, LR_D, 1, 0)
    assert_same_words(pgot.phi, pwant.phi, "label member permutation shap: handle vs reference evaluator")
    _additive(pgot, f, 4, LR_D, 1, False, 0.0, 1e-5, "label member permutation shap additivity")
    _ = h_bg^
    _ = h^
    _ = p_bg^
    _ = p_h^
    _ = ref_bg^
    _ = ref_m^
    _ = p_ref_bg^
    _ = p_ref_m^
    _lr_free(gnb)
    _ = labels^
    _ = fit_x^
    _ = codes^
    _ = bg^
    _ = x^
    _ = xa^
    _ = xb^


comptime NBX_DIR = "checks/native_tests/fixtures/prep_nbx/"


def test_agnostic_categorical_member() raises:
    """CategoricalNB(alpha=0.5) as the kind-4 member on the `cat` fixture (500 x 4 categories, three classes),
    background X[:20], explained X[20:24] (training rows: every category is inside the fitted range), its
    float32 predict_proba through `AgnMember`."""
    var glue = _glue()
    var n = 500
    var d = 4
    var k = 3
    var xs = fixture_f32(NBX_DIR + "cat-x.f32", n * d)
    var ys = fixture_i32(NBX_DIR + "cat-y.i32", n)
    var cat_words: List[Float64] = [Float64(NB_KIND_CATEGORICAL), 0.5, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    var cat = _nb_fit(cat_words^, xs, ys, n, d, k)
    var bg = _cols(xs, 20, d, d)
    var x = List[Float32](length=4 * d, fill=Float32(0))
    for i in range(4 * d):  # small-loop(fixture: X[20:24])
        x[i] = xs[20 * d + i]
    var none = List[Float32]()
    var plan_at = Int(UnsafePointer(to=cat.plan))
    var ref_bg = _OpRef(plan_at, cat.state, MEMBER_OP_PREDICT_PROBA, none, bg, 20, d, k)
    var ref_m = _OpRef(plan_at, cat.state, MEMBER_OP_PREDICT_PROBA, none, x, 4, d, k)
    var want = _kernel_of(glue, ref_bg, ref_m, x, bg, 4, 20, d, k, False)
    var h_bg = AgnMember(cat.plan.member, plan_at, cat.state, MEMBER_OP_PREDICT_PROBA, Int(bg.unsafe_ptr()), 20, d, k,
                         False, False, True, glue, 0, 0)
    var h = AgnMember(cat.plan.member, plan_at, cat.state, MEMBER_OP_PREDICT_PROBA, Int(x.unsafe_ptr()), 4, d, k,
                      False, False, True, glue, 0, 0)
    var got = _kernel_of(glue, h_bg, h, x, bg, 4, 20, d, k, False)
    assert_same_words(got.ev, want.ev, "categorical member kernel shap: expected value, handle vs reference")
    assert_same_words(got.phi, want.phi, "categorical member kernel shap: handle vs reference evaluator")
    var exact = _exact_serial(ref_m, x, bg, 4, 20, d, k)
    _close(got.phi, exact, 1e-6, 1e-6, "categorical member kernel shap vs exact Shapley")
    var f = ref_m.answer(Int(x.unsafe_ptr()), 4)
    _additive(got, f, 4, d, k, False, 0.0, 1e-5, "categorical member kernel shap additivity")
    var p_ref_bg = _OpRef(plan_at, cat.state, MEMBER_OP_PREDICT_PROBA, none, bg, 20, d, k)
    var p_ref_m = _OpRef(plan_at, cat.state, MEMBER_OP_PREDICT_PROBA, none, x, 4, d, k)
    var pwant = _perm_of(glue, p_ref_bg, p_ref_m, x, bg, 4, 20, d, k, 0)
    var p_bg = AgnMember(cat.plan.member, plan_at, cat.state, MEMBER_OP_PREDICT_PROBA, Int(bg.unsafe_ptr()), 20, d, k,
                         False, False, True, glue, 0, 0)
    var p_h = AgnMember(cat.plan.member, plan_at, cat.state, MEMBER_OP_PREDICT_PROBA, Int(x.unsafe_ptr()), 4, d, k,
                        False, False, True, glue, 0, 0)
    var pgot = _perm_of(glue, p_bg, p_h, x, bg, 4, 20, d, k, 0)
    assert_same_words(pgot.phi, pwant.phi, "categorical member permutation shap: handle vs reference evaluator")
    _additive(pgot, f, 4, d, k, False, 0.0, 1e-5, "categorical member permutation shap additivity")
    _ = h_bg^
    _ = h^
    _ = p_bg^
    _ = p_h^
    _ = ref_bg^
    _ = ref_m^
    _ = p_ref_bg^
    _ = p_ref_m^
    _lr_free(cat)
    _ = none^
    _ = bg^
    _ = x^
    _ = xs^
    _ = ys^


struct _VoteRef(AgnModel):
    """The hand formula of VotingClassifier(voting='soft', weights=[w0, w1]).predict_proba over two members:
    (w0 f32(p0) + w1 f32(p1)) / (w0 + w1) in float64, narrowed to float32. `plans` the members' plan addresses."""
    var p0: Int
    var s0: Int
    var p1: Int
    var s1: Int
    var w0: Float64
    var w1: Float64
    var x: List[Float32]
    var n: Int
    var d: Int
    var fx: List[Float32]
    var syn: List[Float32]
    var out: List[Float32]

    def __init__(out self, p0: Int, s0: Int, p1: Int, s1: Int, w0: Float64, w1: Float64, x: List[Float32], n: Int,
                 d: Int):
        self.p0 = p0
        self.s0 = s0
        self.p1 = p1
        self.s1 = s1
        self.w0 = w0
        self.w1 = w1
        self.x = x.copy()
        self.n = n
        self.d = d
        self.fx = List[Float32]()
        self.syn = List[Float32]()
        self.out = List[Float32]()

    def _member(self, plan_at: Int, state: Int, src: Int, n: Int) raises -> List[Float64]:
        var plan = MutPointer[MemberPlan, MutUntrackedOrigin](unsafe_from_address=plan_at)
        var wide = List[Float64](length=max(n * 2, 1), fill=0.0)
        var call = member_call_of(plan[], state)
        call.x = MemberMatrix(src, n, self.d)
        call.dst = MemberMatrix(Int(wide.unsafe_ptr()), n, 2, MEMBER_F64)
        _ = member_call(plan[].member, MEMBER_OP_PREDICT_PROBA, call)
        return wide^

    def answer(self, src: Int, n: Int) raises -> List[Float32]:
        var a = self._member(self.p0, self.s0, src, n)
        var b = self._member(self.p1, self.s1, src, n)
        var words = List[Float32](length=max(n * 2, 1), fill=Float32(0))
        for i in range(n * 2):  # small-loop(oracle: the soft vote, word by word)
            var acc = self.w0 * Float64(a[i].cast[DType.float32]()) + self.w1 * Float64(b[i].cast[DType.float32]())
            words[i] = (acc / (self.w0 + self.w1)).cast[DType.float32]()
        return words^

    def explained(mut self) raises -> Int:
        self.fx = self.answer(Int(self.x.unsafe_ptr()), self.n)
        return Int(self.fx.unsafe_ptr())

    def synthetic(mut self, count: Int, filled: Bool) raises -> Int:
        if len(self.syn) != count:
            self.syn = List[Float32](length=count, fill=Float32(0))
        return Int(self.syn.unsafe_ptr())

    def rows(mut self, n: Int) raises -> Int:
        self.out = self.answer(Int(self.syn.unsafe_ptr()), n)
        return Int(self.out.unsafe_ptr())

    def workers(self) -> Int:
        return 1

    def worker(self) raises -> Self:
        return Self(self.p0, self.s0, self.p1, self.s1, self.w0, self.w1, self.x, self.n, self.d)


def test_agnostic_vote() raises:
    """VotingClassifier(voting='soft', weights=[1, 2]) over two LogisticRegression members (fitted on rows
    [0, 200) and [200, 400) of `_clf(n_classes=2)[:, :4]`), background Xa[:20], explained Xb[:4], through
    `AgnMeta`."""
    var glue = _glue()
    var xa = fixture_f32(SHAP_DIR + "clf2-xa.f32", NA * D)
    var xb = fixture_f32(SHAP_DIR + "clf2-xb.f32", 450 * D)
    var ya = fixture_i32(SHAP_DIR + "clf2-ya.i32", NA)
    var all_x = _cols(xa, 2 * LR_FIT_ROWS, D, LR_D)
    var first_x = List[Float32](length=LR_FIT_ROWS * LR_D, fill=Float32(0))
    var second_x = List[Float32](length=LR_FIT_ROWS * LR_D, fill=Float32(0))
    for i in range(LR_FIT_ROWS * LR_D):  # small-loop(fixture: the two members' rows)
        first_x[i] = all_x[i]
        second_x[i] = all_x[LR_FIT_ROWS * LR_D + i]
    var first_y = List[Int32](length=LR_FIT_ROWS, fill=Int32(0))
    var second_y = List[Int32](length=LR_FIT_ROWS, fill=Int32(0))
    for i in range(LR_FIT_ROWS):  # small-loop(fixture: the two members' codes)
        first_y[i] = ya[i]
        second_y[i] = ya[LR_FIT_ROWS + i]
    var m0 = _lr_fit(first_x, first_y, LR_FIT_ROWS, LR_D)
    var m1 = _lr_fit(second_x, second_y, LR_FIT_ROWS, LR_D)
    var p0 = Int(UnsafePointer(to=m0.plan))
    var p1 = Int(UnsafePointer(to=m1.plan))
    var class_cols: List[Int32] = [0, 1]
    var cc = Int(class_cols.unsafe_ptr())
    var members = List[VoteMember]()
    members.append(VoteMember(p0, m0.state, MEMBER_OP_PREDICT_PROBA, 2, MEMBER_F64, cc, 1.0))
    members.append(VoteMember(p1, m1.state, MEMBER_OP_PREDICT_PROBA, 2, MEMBER_F64, cc, 2.0))
    var bg = _cols(xa, 20, D, LR_D)
    var x = _cols(xb, 4, D, LR_D)
    var plan = AgnMetaPlan(AGN_META_SOFT)
    plan.members = members.copy()
    plan.k = 2
    plan.classes = 2
    var ref_m = _VoteRef(p0, m0.state, p1, m1.state, 1.0, 2.0, x, 4, LR_D)
    var v_bg = AgnMeta(plan, Int(bg.unsafe_ptr()), 20, LR_D, False, True, glue)
    var v = AgnMeta(plan, Int(x.unsafe_ptr()), 4, LR_D, False, True, glue)
    var hand = ref_m.answer(Int(x.unsafe_ptr()), 4)
    var folded = _F32P(unsafe_from_address=v.explained())
    for i in range(8):  # small-loop(oracle: the explained rows' vote)
        require(abs(Float64(folded[i]) - Float64(hand[i])) <= 1e-6 * abs(Float64(hand[i])) + 1e-7,
                "vote output " + String(i) + ": fold " + String(folded[i]) + ", hand formula " + String(hand[i]))
    var serial = _kernel_of(glue, v_bg, v, x, bg, 4, 20, LR_D, 2, False)
    var w_bg = AgnMeta(plan, Int(bg.unsafe_ptr()), 20, LR_D, True, True, glue)
    var w = AgnMeta(plan, Int(x.unsafe_ptr()), 4, LR_D, True, True, glue)
    var spread = _kernel_of(glue, w_bg, w, x, bg, 4, 20, LR_D, 2, False)
    assert_same_words(spread.phi, serial.phi, "vote kernel shap: chunks on the workers vs in order")
    var exact = _exact_serial(ref_m, x, bg, 4, 20, LR_D, 2)
    _close(serial.phi, exact, 1e-5, 1e-6, "vote kernel shap vs the exact Shapley values of the hand formula")
    _additive(serial, hand, 4, LR_D, 2, False, 0.0, 1e-5, "vote kernel shap additivity")
    var q_bg = AgnMeta(plan, Int(bg.unsafe_ptr()), 20, LR_D, True, True, glue)
    var q = AgnMeta(plan, Int(x.unsafe_ptr()), 4, LR_D, True, True, glue)
    var perm = _perm_of(glue, q_bg, q, x, bg, 4, 20, LR_D, 2, 0)
    _additive(perm, hand, 4, LR_D, 2, False, 0.0, 1e-5, "vote permutation shap additivity")
    _ = v_bg^
    _ = v^
    _ = w_bg^
    _ = w^
    _ = q_bg^
    _ = q^
    _ = ref_m^
    _ = plan^
    _ = members^
    _ = class_cols^
    _lr_free(m0)
    _lr_free(m1)
    _ = all_x^
    _ = first_x^
    _ = second_x^
    _ = first_y^
    _ = second_y^
    _ = bg^
    _ = x^
    _ = xa^
    _ = xb^


# ---------------------------------------------------------------- lane mf-shap3 (2026-11-23): every meta-estimator kind
# `AgnMeta` (xtrees/agnostic_values_run.mojo) over fitted binary LogisticRegression members (`_meta_fixture`: two
# members on rows [0, 200) and [200, 400) of `_clf(n_classes=2)[:, :4]`, one on columns [0, 2] of the first rows,
# and a final estimator on the first rows' stacked probabilities). Each test: the handle's outputs on Xb[:4] equal
# an oracle written out here from the members' own outputs (direct member calls; `rtol` 0 = word for word), Kernel
# SHAP with the chunks on the workers gives the serial words, Kernel SHAP equals the exact Shapley values of the
# oracle's game (rtol 1e-5, atol 1e-6) and is additive, and Permutation SHAP is additive (`_check_meta`). The
# regressor-shaped kinds (REGRESS, COLUMNS) take the members' decision_function as their prediction: the RUN folds
# a member's float32 column whatever its operation. Fail-proof edits (each makes its test fail):
#   HARD          `vote_codes_run(..., False, ...)` -> `True` in AgnMeta._run (a soft fold over codes).
#   BAG           `self.plan.features[j] == 0` -> `!= 0` in AgnMeta._bag (member 0 reads the wrong columns).
#   STACK         `n * width` -> `n` in AgnMeta._stack's meta narrowing (half the meta features unwritten).
#   OVR           `at >= 0` -> `at > 0` in AgnMeta._ovr (the positive column is never gathered: p = 0).
#   CALIBRATED    `n * classes` -> `n` in the CALIBRATED narrowing of AgnMeta._run (the second column unwritten).
#   ADA           `ADA_PROBA` -> `ADA_DECISION` in AgnMeta._ada.
#   REGRESS       `agn_narrow(at, ...)` -> `agn_narrow(at + 8, ...)` in the vote fold of AgnMeta._run.
#   COLUMNS       `n * k` -> `n` in the COLUMNS narrowing of AgnMeta._run.
# Lane mf-shap3b (2026-11-23), the last `_DoorModel` users made native:
#   STACK_LABELS  (a classifier member a Stacking asks for predict, its codes mapped to its labels) drop the line
#                 `members[j] = VoteMember(Int(block.unsafe_ptr()), 1, 0, 1.0)` in AgnMeta._stack (the member's
#                 int codes are read as a float32 block: its call is refused and the test raises).
#   SELECT        (an RFE: the estimator's member over the support columns) `self.call.x =
#                 MemberMatrix(self._select(x, n), n, self.width)` -> `MemberMatrix(x, n, d)` in AgnMember._run.
#   INTS          (an outlier detector's integer predict widened to float32) `HP_I32` -> `HP_I64` (and the reverse)
#                 in AgnMember._run's ints branch.
#   LABEL_PROP    drop `admitted_finite_refuse(status, self.name)` in AgnLabelProp._run (the NaN row is answered
#                 instead of refused in the public method's words).
comptime META_ROWS = 4
comptime META_BG = 20


struct _Meta3(Movable):
    """The fitted members and rows of the mf-shap3 meta tests."""
    var m0: _Lr
    var m1: _Lr
    var sub: _Lr
    var last: _Lr
    var x: List[Float32]
    var bg: List[Float32]
    var cc: List[Int32]
    var cols02: List[Int32]

    def __init__(out self, var m0: _Lr, var m1: _Lr, var sub: _Lr, var last: _Lr, var x: List[Float32],
                 var bg: List[Float32]):
        self.m0 = m0^
        self.m1 = m1^
        self.sub = sub^
        self.last = last^
        self.x = x^
        self.bg = bg^
        self.cc = [0, 1]
        self.cols02 = [0, 2]


def _outs(f: _Lr, operation: Int) raises -> Tuple[Int, Int]:
    """(columns, element type) the fitted member writes for `operation`."""
    var asked = member_call_of(f.plan, f.state)
    asked.flags = operation
    _ = member_call(f.plan.member, MEMBER_OP_OUTPUTS, asked)
    return (asked.out_cols, asked.out_dtype)


def _call_f32(f: _Lr, operation: Int, src: Int, n: Int, d: Int) raises -> List[Float32]:
    """The member's `operation` on `n` rows of `d` features, as float32 (a float64 answer narrowed word by word,
    the door's admission of it)."""
    var o = _outs(f, operation)
    var cols = o[0]
    var call = member_call_of(f.plan, f.state)
    call.x = MemberMatrix(src, n, d)
    var words = List[Float32](length=max(1, n * cols), fill=Float32(0))
    if o[1] == MEMBER_F64:
        var wide = List[Float64](length=max(1, n * cols), fill=0.0)
        call.dst = MemberMatrix(Int(wide.unsafe_ptr()), n, cols, MEMBER_F64)
        _ = member_call(f.plan.member, operation, call)
        for i in range(n * cols):  # small-loop(oracle: the float32 narrowing, word by word)
            words[i] = wide[i].cast[DType.float32]()
        _ = wide^
        return words^
    call.dst = MemberMatrix(Int(words.unsafe_ptr()), n, cols, MEMBER_F32)
    _ = member_call(f.plan.member, operation, call)
    return words^


def _call_codes(f: _Lr, src: Int, n: Int, d: Int) raises -> List[Int32]:
    """The member's predict codes (int64 narrowed)."""
    var o = _outs(f, MEMBER_OP_PREDICT)
    var call = member_call_of(f.plan, f.state)
    call.x = MemberMatrix(src, n, d)
    var codes = List[Int32](length=max(1, n), fill=Int32(0))
    if o[1] == MEMBER_I64:
        var wide = List[Int64](length=max(1, n), fill=Int64(0))
        call.dst = MemberMatrix(Int(wide.unsafe_ptr()), n, 1, MEMBER_I64)
        _ = member_call(f.plan.member, MEMBER_OP_PREDICT, call)
        for i in range(n):  # small-loop(oracle: one narrowing pass)
            codes[i] = Int32(wide[i])
        _ = wide^
        return codes^
    call.dst = MemberMatrix(Int(codes.unsafe_ptr()), n, 1, MEMBER_I32)
    _ = member_call(f.plan.member, MEMBER_OP_PREDICT, call)
    return codes^


def _vm(f: _Lr, operation: Int, class_cols: Int, weight: Float64) raises -> VoteMember:
    """The fitted member as the door's `_meta_take` hands it to the RUN (`f` held unmoved while it is used)."""
    var o = _outs(f, operation)
    return VoteMember(Int(UnsafePointer(to=f.plan)), f.state, operation, o[0], o[1], class_cols, weight)


def _meta_fixture() raises -> _Meta3:
    var xa = fixture_f32(SHAP_DIR + "clf2-xa.f32", NA * D)
    var xb = fixture_f32(SHAP_DIR + "clf2-xb.f32", 450 * D)
    var ya = fixture_i32(SHAP_DIR + "clf2-ya.i32", NA)
    var all_x = _cols(xa, 2 * LR_FIT_ROWS, D, LR_D)
    var first_x = List[Float32](length=LR_FIT_ROWS * LR_D, fill=Float32(0))
    var second_x = List[Float32](length=LR_FIT_ROWS * LR_D, fill=Float32(0))
    var sub_x = List[Float32](length=LR_FIT_ROWS * 2, fill=Float32(0))
    for i in range(LR_FIT_ROWS * LR_D):  # small-loop(fixture: the two members' rows)
        first_x[i] = all_x[i]
        second_x[i] = all_x[LR_FIT_ROWS * LR_D + i]
    for i in range(LR_FIT_ROWS):  # small-loop(fixture: columns 0 and 2 of the first rows)
        sub_x[2 * i] = first_x[i * LR_D]
        sub_x[2 * i + 1] = first_x[i * LR_D + 2]
    var first_y = List[Int32](length=LR_FIT_ROWS, fill=Int32(0))
    var second_y = List[Int32](length=LR_FIT_ROWS, fill=Int32(0))
    for i in range(LR_FIT_ROWS):  # small-loop(fixture: the members' codes)
        first_y[i] = ya[i]
        second_y[i] = ya[LR_FIT_ROWS + i]
    var m0 = _lr_fit(first_x, first_y, LR_FIT_ROWS, LR_D)
    var m1 = _lr_fit(second_x, second_y, LR_FIT_ROWS, LR_D)
    var sub = _lr_fit(sub_x, first_y, LR_FIT_ROWS, 2)
    # The final estimator's rows: the first rows' stacked probabilities of code 1 (`_StackingBase` with
    # stack_method predict_proba and two classes keeps the second column).
    var p0 = _call_f32(m0, MEMBER_OP_PREDICT_PROBA, Int(first_x.unsafe_ptr()), LR_FIT_ROWS, LR_D)
    var p1 = _call_f32(m1, MEMBER_OP_PREDICT_PROBA, Int(first_x.unsafe_ptr()), LR_FIT_ROWS, LR_D)
    var meta_x = List[Float32](length=LR_FIT_ROWS * 2, fill=Float32(0))
    for i in range(LR_FIT_ROWS):  # small-loop(fixture: the stacked meta features)
        meta_x[2 * i] = p0[2 * i + 1]
        meta_x[2 * i + 1] = p1[2 * i + 1]
    var last = _lr_fit(meta_x, first_y, LR_FIT_ROWS, 2)
    var bg = _cols(xa, META_BG, D, LR_D)
    var x = _cols(xb, META_ROWS, D, LR_D)
    _ = all_x^
    _ = first_x^
    _ = second_x^
    _ = sub_x^
    _ = first_y^
    _ = second_y^
    _ = p0^
    _ = p1^
    _ = meta_x^
    _ = xa^
    _ = xb^
    _ = ya^
    return _Meta3(m0^, m1^, sub^, last^, x^, bg^)


def _meta_free(mut f: _Meta3):
    _lr_free(f.m0)
    _lr_free(f.m1)
    _lr_free(f.sub)
    _lr_free(f.last)


comptime _REF_HARD = 0
comptime _REF_BAG = 1
comptime _REF_STACK = 2
comptime _REF_OVR = 3
comptime _REF_CAL = 4
comptime _REF_ADA = 5
comptime _REF_REGRESS = 6
comptime _REF_COLUMNS = 7
comptime _REF_STACK_LABELS = 8
comptime _REF_SELECT = 9
comptime _REF_INTS = 10
comptime _CAL_A = -1.3
comptime _CAL_B = 0.2


struct _MetaRef(AgnModel):
    """The oracle of one meta kind (`_REF_*`) over the fixture's members (`fp`: the address of the `_Meta3`, held
    unmoved): their outputs by direct member calls, then the public operation's formula written out here (ADA: the
    RUN `ada_classifier_run` over the members' own codes, as meta_runs_gil_free.mojo proves the meta RUNs)."""
    var kind: Int
    var fp: Int
    var x: List[Float32]
    var n: Int
    var k: Int
    var glue: EntryRef
    var fx: List[Float32]
    var syn: List[Float32]
    var out: List[Float32]

    def __init__(out self, kind: Int, fp: Int, x: List[Float32], n: Int, k: Int, glue: EntryRef):
        self.kind = kind
        self.fp = fp
        self.x = x.copy()
        self.n = n
        self.k = k
        self.glue = glue
        self.fx = List[Float32]()
        self.syn = List[Float32]()
        self.out = List[Float32]()

    def answer(self, src: Int, n: Int) raises -> List[Float32]:
        var f = MutPointer[_Meta3, MutUntrackedOrigin](unsafe_from_address=self.fp)
        var d = LR_D
        var words = List[Float32](length=max(1, n * self.k), fill=Float32(0))
        if self.kind == _REF_HARD:
            var labels: List[Float32] = [-1.5, 4.0]
            var c0 = _call_codes(f[].m0, src, n, d)
            var c1 = _call_codes(f[].m1, src, n, d)
            for i in range(n):  # small-loop(oracle: the weighted hard vote, weights 1 and 2)
                var zero = (1.0 if c0[i] == 0 else 0.0) + (2.0 if c1[i] == 0 else 0.0)
                var one = (1.0 if c0[i] == 1 else 0.0) + (2.0 if c1[i] == 1 else 0.0)
                words[i] = labels[0] if zero >= one else labels[1]
            return words^
        if self.kind == _REF_BAG:
            var rows = List[Float32](length=max(1, n * 2), fill=Float32(0))
            var xs = _F32P(unsafe_from_address=src)
            for i in range(n):  # small-loop(oracle: member 0's columns 0 and 2)
                rows[2 * i] = xs[i * d]
                rows[2 * i + 1] = xs[i * d + 2]
            var p0 = _call_f32(f[].sub, MEMBER_OP_PREDICT_PROBA, Int(rows.unsafe_ptr()), n, 2)
            var p1 = _call_f32(f[].m1, MEMBER_OP_PREDICT_PROBA, src, n, d)
            for i in range(n * 2):  # small-loop(oracle: the mean of the two members' probabilities)
                words[i] = ((Float64(p0[i]) + Float64(p1[i])) / 2.0).cast[DType.float32]()
            _ = rows^
            return words^
        if self.kind == _REF_STACK:
            var p0 = _call_f32(f[].m0, MEMBER_OP_PREDICT_PROBA, src, n, d)
            var p1 = _call_f32(f[].m1, MEMBER_OP_PREDICT_PROBA, src, n, d)
            var meta = List[Float32](length=max(1, n * 2), fill=Float32(0))
            for i in range(n):  # small-loop(oracle: the stacked probabilities of code 1)
                meta[2 * i] = p0[2 * i + 1]
                meta[2 * i + 1] = p1[2 * i + 1]
            var got = _call_f32(f[].last, MEMBER_OP_PREDICT_PROBA, Int(meta.unsafe_ptr()), n, 2)
            _ = meta^
            return got^
        if self.kind == _REF_OVR:
            var p = _call_f32(f[].m0, MEMBER_OP_PREDICT_PROBA, src, n, d)
            for i in range(n):  # small-loop(oracle: the rows (1 - p, p) of the positive column)
                words[2 * i] = (1.0 - Float64(p[2 * i + 1])).cast[DType.float32]()
                words[2 * i + 1] = p[2 * i + 1]
            return words^
        if self.kind == _REF_CAL:
            var s = _call_f32(f[].m0, MEMBER_OP_DECISION, src, n, d)
            for i in range(n):  # small-loop(oracle: Platt's sigmoid expit(-(a s + b)) and its complement)
                var c = 1.0 / (1.0 + exp(_CAL_A * Float64(s[i]) + _CAL_B))
                words[2 * i] = (1.0 - c).cast[DType.float32]()
                words[2 * i + 1] = c.cast[DType.float32]()
            return words^
        if self.kind == _REF_ADA:
            var c0 = _call_codes(f[].m0, src, n, d)
            var c1 = _call_codes(f[].m1, src, n, d)
            var codes: List[Int] = [Int(c0.unsafe_ptr()), Int(c1.unsafe_ptr())]
            var alphas: List[Float64] = [0.7, 0.4]
            var acc = List[Float64](length=max(1, n * 2), fill=0.0)
            var pairs = List[Float64](length=max(1, n * 2), fill=0.0)
            ada_classifier_run(self.glue, codes, alphas, n, 2, Int(acc.unsafe_ptr()), ADA_PROBA, Int(pairs.unsafe_ptr()))
            for i in range(n * 2):  # small-loop(oracle: the float32 narrowing)
                words[i] = pairs[i].cast[DType.float32]()
            _ = c0^
            _ = c1^
            _ = acc^
            return words^
        if self.kind == _REF_STACK_LABELS:
            # member 0 asked for predict: its labels [-1.5, 4.0] as the first meta feature; member 1's probability
            # of code 1 as the second; the final estimator's predict_proba on them.
            var labels: List[Float32] = [-1.5, 4.0]
            var c0 = _call_codes(f[].m0, src, n, d)
            var p1 = _call_f32(f[].m1, MEMBER_OP_PREDICT_PROBA, src, n, d)
            var meta = List[Float32](length=max(1, n * 2), fill=Float32(0))
            for i in range(n):  # small-loop(oracle: the stacked label and probability)
                meta[2 * i] = labels[Int(c0[i])]
                meta[2 * i + 1] = p1[2 * i + 1]
            var got = _call_f32(f[].last, MEMBER_OP_PREDICT_PROBA, Int(meta.unsafe_ptr()), n, 2)
            _ = meta^
            return got^
        if self.kind == _REF_SELECT:
            # RFE(support [0, 2]).predict_proba = estimator_.predict_proba(X[:, [0, 2]]).
            var rows = List[Float32](length=max(1, n * 2), fill=Float32(0))
            var xs = _F32P(unsafe_from_address=src)
            for i in range(n):  # small-loop(oracle: the support columns 0 and 2)
                rows[2 * i] = xs[i * d]
                rows[2 * i + 1] = xs[i * d + 2]
            var got = _call_f32(f[].sub, MEMBER_OP_PREDICT_PROBA, Int(rows.unsafe_ptr()), n, 2)
            _ = rows^
            return got^
        if self.kind == _REF_INTS:
            var c0 = _call_codes(f[].m0, src, n, d)
            for i in range(n):  # small-loop(oracle: each integer value as float32)
                words[i] = Float32(c0[i])
            return words^
        var d0 = _call_f32(f[].m0, MEMBER_OP_DECISION, src, n, d)
        var d1 = _call_f32(f[].m1, MEMBER_OP_DECISION, src, n, d)
        if self.kind == _REF_REGRESS:
            for i in range(n):  # small-loop(oracle: the weighted mean, weights 1 and 2)
                words[i] = ((Float64(d0[i]) + 2.0 * Float64(d1[i])) / 3.0).cast[DType.float32]()
            return words^
        for i in range(n):  # small-loop(oracle: each member's prediction at its column)
            words[2 * i] = d0[i]
            words[2 * i + 1] = d1[i]
        return words^

    def explained(mut self) raises -> Int:
        self.fx = self.answer(Int(self.x.unsafe_ptr()), self.n)
        return Int(self.fx.unsafe_ptr())

    def synthetic(mut self, count: Int, filled: Bool) raises -> Int:
        if len(self.syn) != count:
            self.syn = List[Float32](length=count, fill=Float32(0))
        return Int(self.syn.unsafe_ptr())

    def rows(mut self, n: Int) raises -> Int:
        self.out = self.answer(Int(self.syn.unsafe_ptr()), n)
        return Int(self.out.unsafe_ptr())

    def workers(self) -> Int:
        return 1

    def worker(self) raises -> Self:
        return Self(self.kind, self.fp, self.x, self.n, self.k, self.glue)


def _check_meta(what: String, glue: EntryRef, plan: AgnMetaPlan, kind: Int, fp: Int, rtol: Float64) raises:
    """`AgnMeta` over `plan` against the oracle `_MetaRef(kind)` (see the section comment)."""
    var f = MutPointer[_Meta3, MutUntrackedOrigin](unsafe_from_address=fp)
    var x = f[].x.copy()
    var bg = f[].bg.copy()
    var k = plan.k
    var ref_m = _MetaRef(kind, fp, x, META_ROWS, k, glue)
    var want = ref_m.answer(Int(x.unsafe_ptr()), META_ROWS)
    var m_bg = AgnMeta(plan, Int(bg.unsafe_ptr()), META_BG, LR_D, False, True, glue)
    var m = AgnMeta(plan, Int(x.unsafe_ptr()), META_ROWS, LR_D, False, True, glue)
    var got = _F32P(unsafe_from_address=m.explained())
    for i in range(META_ROWS * k):  # small-loop(oracle: every output word)
        var ok = got[i] == want[i] if rtol == 0.0 else (
            abs(Float64(got[i]) - Float64(want[i])) <= rtol * abs(Float64(want[i])) + 1e-7)
        require(ok, what + " output " + String(i) + ": handle " + String(got[i]) + ", oracle " + String(want[i]))
    var serial = _kernel_of(glue, m_bg, m, x, bg, META_ROWS, META_BG, LR_D, k, False)
    var w_bg = AgnMeta(plan, Int(bg.unsafe_ptr()), META_BG, LR_D, True, True, glue)
    var w = AgnMeta(plan, Int(x.unsafe_ptr()), META_ROWS, LR_D, True, True, glue)
    var spread = _kernel_of(glue, w_bg, w, x, bg, META_ROWS, META_BG, LR_D, k, False)
    assert_same_words(spread.phi, serial.phi, what + " kernel shap: chunks on the workers vs in order")
    var exact = _exact_serial(ref_m, x, bg, META_ROWS, META_BG, LR_D, k)
    _close(serial.phi, exact, 1e-5, 1e-6, what + " kernel shap vs the exact Shapley values of the oracle")
    _additive(serial, want, META_ROWS, LR_D, k, False, 0.0, 1e-5, what + " kernel shap additivity")
    var p_bg = AgnMeta(plan, Int(bg.unsafe_ptr()), META_BG, LR_D, True, True, glue)
    var p = AgnMeta(plan, Int(x.unsafe_ptr()), META_ROWS, LR_D, True, True, glue)
    var perm = _perm_of(glue, p_bg, p, x, bg, META_ROWS, META_BG, LR_D, k, 0)
    _additive(perm, want, META_ROWS, LR_D, k, False, 0.0, 1e-5, what + " permutation shap additivity")
    _ = m_bg^
    _ = m^
    _ = w_bg^
    _ = w^
    _ = p_bg^
    _ = p^
    _ = ref_m^
    _ = want^
    _ = x^
    _ = bg^


def test_agnostic_meta_hard_vote() raises:
    """VotingClassifier(voting='hard', weights=[1, 2]).predict, classes_ [-1.5, 4.0] as float32 labels."""
    var glue = _glue()
    var f = _meta_fixture()
    var labels: List[Float32] = [-1.5, 4.0]
    var plan = AgnMetaPlan(AGN_META_HARD)
    plan.members.append(_vm(f.m0, MEMBER_OP_PREDICT, 0, 1.0))
    plan.members.append(_vm(f.m1, MEMBER_OP_PREDICT, 0, 2.0))
    plan.classes = 2
    plan.k = 1
    plan.labels = Int(labels.unsafe_ptr())
    plan.n_labels = 2
    _check_meta("hard vote", glue, plan, _REF_HARD, Int(UnsafePointer(to=f)), 0.0)
    _ = plan^
    _ = labels^
    _meta_free(f)


def test_agnostic_meta_bagging() raises:
    """BaggingClassifier.predict_proba: member 0 on features [0, 2] (gathered in the RUN), member 1 on every
    feature (the rows themselves)."""
    var glue = _glue()
    var f = _meta_fixture()
    var plan = AgnMetaPlan(AGN_META_BAG)
    plan.members.append(_vm(f.sub, MEMBER_OP_PREDICT_PROBA, Int(f.cc.unsafe_ptr()), 1.0))
    plan.members.append(_vm(f.m1, MEMBER_OP_PREDICT_PROBA, Int(f.cc.unsafe_ptr()), 1.0))
    plan.features = [Int(f.cols02.unsafe_ptr()), 0]
    plan.widths = [2, LR_D]
    plan.classifier = True
    plan.classes = 2
    plan.k = 2
    _check_meta("bagging", glue, plan, _REF_BAG, Int(UnsafePointer(to=f)), 1e-7)
    _ = plan^
    _meta_free(f)


def test_agnostic_meta_stacking() raises:
    """StackingClassifier(stack_method='predict_proba').predict_proba: the two members' probabilities of code 1
    as the meta features, the final LogisticRegression's predict_proba on them (word for word)."""
    var glue = _glue()
    var f = _meta_fixture()
    var plan = AgnMetaPlan(AGN_META_STACK)
    plan.members.append(_vm(f.m0, MEMBER_OP_PREDICT_PROBA, 0, 1.0))
    plan.members.append(_vm(f.m1, MEMBER_OP_PREDICT_PROBA, 0, 1.0))
    plan.stack_kinds = [STACK_PROBA, STACK_PROBA]
    plan.classes = 2
    plan.binary = True
    plan.meta_width = 2
    plan.final_est.append(_vm(f.last, MEMBER_OP_PREDICT_PROBA, Int(f.cc.unsafe_ptr()), 1.0))
    plan.classifier = True
    plan.k = 2
    _check_meta("stacking", glue, plan, _REF_STACK, Int(UnsafePointer(to=f)), 0.0)
    _ = plan^
    _meta_free(f)


def test_agnostic_meta_ovr() raises:
    """OneVsRestClassifier.predict_proba with two classes: (1 - p, p) of its first member's column of code 1."""
    var glue = _glue()
    var f = _meta_fixture()
    var plan = AgnMetaPlan(AGN_META_OVR)
    plan.members.append(_vm(f.m0, MEMBER_OP_PREDICT_PROBA, 0, 1.0))
    plan.at = [1]
    plan.classes = 2
    plan.k = 2
    _check_meta("one-vs-rest", glue, plan, _REF_OVR, Int(UnsafePointer(to=f)), 1e-6)
    _ = plan^
    _meta_free(f)


def test_agnostic_meta_calibrated() raises:
    """CalibratedClassifierCV(method='sigmoid').predict_proba with one pair: its member's decision_function
    through Platt's sigmoid (a, b fixed here), the rows (1 - c, c)."""
    var glue = _glue()
    var f = _meta_fixture()
    var plan = AgnMetaPlan(AGN_META_CALIBRATED)
    plan.members.append(_vm(f.m0, MEMBER_OP_DECISION, 0, 1.0))
    require(plan.members[0].cols == 1, "LogisticRegression's binary decision_function is no longer one column")
    plan.pairs.append(CalPair(0, 0, 0, 0, 1))
    plan.cals.append(CalFit(False, _CAL_A, _CAL_B, 0, 0, 0))
    plan.classes = 2
    plan.k = 2
    _check_meta("calibrated", glue, plan, _REF_CAL, Int(UnsafePointer(to=f)), 1e-6)
    _ = plan^
    _meta_free(f)


def test_agnostic_meta_adaboost() raises:
    """AdaBoostClassifier.predict_proba (two classes, weights 0.7 and 0.4): the members' codes through
    `ada_classifier_run`."""
    var glue = _glue()
    var f = _meta_fixture()
    var plan = AgnMetaPlan(AGN_META_ADA)
    plan.members.append(_vm(f.m0, MEMBER_OP_PREDICT, 0, 1.0))
    plan.members.append(_vm(f.m1, MEMBER_OP_PREDICT, 0, 1.0))
    plan.alphas = [0.7, 0.4]
    plan.classifier = True
    plan.classes = 2
    plan.k = 2
    _check_meta("adaboost", glue, plan, _REF_ADA, Int(UnsafePointer(to=f)), 0.0)
    _ = plan^
    _meta_free(f)


def test_agnostic_meta_regressor_shapes() raises:
    """The regressor-shaped kinds over the members' decision_function columns: VotingRegressor (weights 1 and
    2) and MultiOutputRegressor (member j at column j, word for word)."""
    var glue = _glue()
    var f = _meta_fixture()
    var plan = AgnMetaPlan(AGN_META_REGRESS)
    plan.members.append(_vm(f.m0, MEMBER_OP_DECISION, 0, 1.0))
    plan.members.append(_vm(f.m1, MEMBER_OP_DECISION, 0, 2.0))
    plan.k = 1
    _check_meta("vote regressor", glue, plan, _REF_REGRESS, Int(UnsafePointer(to=f)), 1e-6)
    var cols = AgnMetaPlan(AGN_META_COLUMNS)
    cols.members.append(_vm(f.m0, MEMBER_OP_DECISION, 0, 1.0))
    cols.members.append(_vm(f.m1, MEMBER_OP_DECISION, 0, 1.0))
    cols.k = 2
    _check_meta("multi-output columns", glue, cols, _REF_COLUMNS, Int(UnsafePointer(to=f)), 0.0)
    _ = plan^
    _ = cols^
    _meta_free(f)


def test_agnostic_meta_stacking_labels() raises:
    """StackingClassifier with stack_method_ ['predict', 'predict_proba'] (lane mf-shap3b): member 0, a classifier
    asked for predict, gives its labels [-1.5, 4.0] (its int codes mapped in the RUN), member 1 its probability of
    code 1; the final LogisticRegression's predict_proba on them, word for word."""
    var glue = _glue()
    var f = _meta_fixture()
    var labels: List[Float32] = [-1.5, 4.0]
    var plan = AgnMetaPlan(AGN_META_STACK)
    plan.members.append(_vm(f.m0, MEMBER_OP_PREDICT, 0, 1.0))
    plan.members.append(_vm(f.m1, MEMBER_OP_PREDICT_PROBA, 0, 1.0))
    plan.stack_kinds = [STACK_PREDICT, STACK_PROBA]
    plan.stack_labels = [Int(labels.unsafe_ptr()), 0]
    plan.stack_n_labels = [2, 0]
    plan.classes = 2
    plan.binary = True
    plan.meta_width = 2
    plan.final_est.append(_vm(f.last, MEMBER_OP_PREDICT_PROBA, Int(f.cc.unsafe_ptr()), 1.0))
    plan.classifier = True
    plan.k = 2
    _check_meta("stacking with a predict member", glue, plan, _REF_STACK_LABELS, Int(UnsafePointer(to=f)), 0.0)
    _ = plan^
    _ = labels^
    _meta_free(f)


def _check_handle(what: String, glue: EntryRef, f: _Lr, operation: Int, k: Int, wide: Bool, ints: Bool, cols: Int,
                  width: Int, kind: Int, fp: Int) raises:
    """`AgnMember` over `f` against the oracle `_MetaRef(kind)`: its explained words, Kernel SHAP serial vs the
    chunks on the host workers (the same words), the exact Shapley values of the oracle, additivity (Kernel and
    Permutation SHAP)."""
    var m = MutPointer[_Meta3, MutUntrackedOrigin](unsafe_from_address=fp)
    var x = m[].x.copy()
    var bg = m[].bg.copy()
    var plan_at = Int(UnsafePointer(to=f.plan))
    var ref_m = _MetaRef(kind, fp, x, META_ROWS, k, glue)
    var want = ref_m.answer(Int(x.unsafe_ptr()), META_ROWS)
    var h_bg = AgnMember(f.plan.member, plan_at, f.state, operation, Int(bg.unsafe_ptr()), META_BG, LR_D, k, False,
                         wide, True, glue, 0, 0, ints, cols, width)
    var h = AgnMember(f.plan.member, plan_at, f.state, operation, Int(x.unsafe_ptr()), META_ROWS, LR_D, k, False,
                      wide, True, glue, 0, 0, ints, cols, width)
    var got = _F32P(unsafe_from_address=h.explained())
    for i in range(META_ROWS * k):  # small-loop(oracle: every output word)
        require(got[i] == want[i], what + " output " + String(i) + ": handle " + String(got[i]) + ", oracle "
                + String(want[i]))
    var serial = _kernel_of(glue, h_bg, h, x, bg, META_ROWS, META_BG, LR_D, k, False)
    var w_bg = AgnMember(f.plan.member, plan_at, f.state, operation, Int(bg.unsafe_ptr()), META_BG, LR_D, k, True,
                         wide, True, glue, 0, 0, ints, cols, width)
    var w = AgnMember(f.plan.member, plan_at, f.state, operation, Int(x.unsafe_ptr()), META_ROWS, LR_D, k, True,
                      wide, True, glue, 0, 0, ints, cols, width)
    var spread = _kernel_of(glue, w_bg, w, x, bg, META_ROWS, META_BG, LR_D, k, False)
    assert_same_words(spread.phi, serial.phi, what + " kernel shap: chunks on the workers vs in order")
    var exact = _exact_serial(ref_m, x, bg, META_ROWS, META_BG, LR_D, k)
    _close(serial.phi, exact, 1e-5, 1e-6, what + " kernel shap vs the exact Shapley values of the oracle")
    _additive(serial, want, META_ROWS, LR_D, k, False, 0.0, 1e-5, what + " kernel shap additivity")
    var p_bg = AgnMember(f.plan.member, plan_at, f.state, operation, Int(bg.unsafe_ptr()), META_BG, LR_D, k, True,
                         wide, True, glue, 0, 0, ints, cols, width)
    var p = AgnMember(f.plan.member, plan_at, f.state, operation, Int(x.unsafe_ptr()), META_ROWS, LR_D, k, True,
                      wide, True, glue, 0, 0, ints, cols, width)
    var perm = _perm_of(glue, p_bg, p, x, bg, META_ROWS, META_BG, LR_D, k, 0)
    _additive(perm, want, META_ROWS, LR_D, k, False, 0.0, 1e-5, what + " permutation shap additivity")
    _ = h_bg^
    _ = h^
    _ = w_bg^
    _ = w^
    _ = p_bg^
    _ = p^
    _ = ref_m^
    _ = want^
    _ = x^
    _ = bg^


def test_agnostic_select_member() raises:
    """An RFE explained through its fitted estimator (lane mf-shap3b): a binary LogisticRegression fitted on columns
    0 and 2, the handle gathering those columns of every chunk (`cols` / `width`), predict_proba as float64
    narrowed: exactly `estimator_.predict_proba(X[:, [0, 2]])`."""
    var glue = _glue()
    var f = _meta_fixture()
    var o = _outs(f.sub, MEMBER_OP_PREDICT_PROBA)
    require(o[0] == 2, "the binary LogisticRegression's predict_proba is no longer two columns")
    _check_handle("rfe support", glue, f.sub, MEMBER_OP_PREDICT_PROBA, 2, o[1] == MEMBER_F64, False,
                  Int(f.cols02.unsafe_ptr()), 2, _REF_SELECT, Int(UnsafePointer(to=f)))
    _meta_free(f)


def test_agnostic_int_values() raises:
    """A model that is not a classifier whose member writes integer values (lane mf-shap3b: IsolationForest's
    predict, int32 1 / -1; LocalOutlierFactor's and OneClassSVM's, int64): the RUN widens them to float32 (`ints`).
    The integer member here is a LogisticRegression's predict, whose codes are explained as plain values."""
    var glue = _glue()
    var f = _meta_fixture()
    var o = _outs(f.m0, MEMBER_OP_PREDICT)
    require(o[0] == 1 and (o[1] == MEMBER_I32 or o[1] == MEMBER_I64),
            "LogisticRegression's member predict no longer writes one integer per row: this check no longer "
            + "covers the integer route")
    _check_handle("integer values", glue, f.m0, MEMBER_OP_PREDICT, 1, o[1] == MEMBER_I64, True, 0, 0, _REF_INTS,
                  Int(UnsafePointer(to=f)))
    _meta_free(f)


comptime _LP_FIT = 20
comptime _LP_GAMMA = 0.5
comptime _LP_RBF = 2  # python/mojolearn/_expansion_neighbors.py `_KERNELS["rbf"]`


struct _LpRef(AgnModel):
    """LabelPropagation(kernel='rbf', gamma=0.5).predict_proba written out: w_ij = exp(-gamma |q_i - x_j|^2) over
    the fitted rows, the product with `label_distributions_`, each row divided by its sum (float64 here)."""
    var fit: List[Float32]
    var dist: List[Float32]
    var x: List[Float32]
    var n: Int
    var fx: List[Float32]
    var syn: List[Float32]
    var out: List[Float32]

    def __init__(out self, fit: List[Float32], dist: List[Float32], x: List[Float32], n: Int):
        self.fit = fit.copy()
        self.dist = dist.copy()
        self.x = x.copy()
        self.n = n
        self.fx = List[Float32]()
        self.syn = List[Float32]()
        self.out = List[Float32]()

    def answer(self, src: Int, n: Int) -> List[Float32]:
        var q = _F32P(unsafe_from_address=src)
        var words = List[Float32](length=max(1, n * 2), fill=Float32(0))
        for i in range(n):  # small-loop(oracle: the queries)
            var p0 = 0.0
            var p1 = 0.0
            for j in range(_LP_FIT):  # small-loop(oracle: the fitted rows)
                var dd = 0.0
                for c in range(LR_D):  # small-loop(oracle: one squared distance)
                    var t = Float64(q[i * LR_D + c]) - Float64(self.fit[j * LR_D + c])
                    dd += t * t
                var w = exp(-_LP_GAMMA * dd)
                p0 += w * Float64(self.dist[2 * j])
                p1 += w * Float64(self.dist[2 * j + 1])
            words[2 * i] = (p0 / (p0 + p1)).cast[DType.float32]()
            words[2 * i + 1] = (p1 / (p0 + p1)).cast[DType.float32]()
        return words^

    def explained(mut self) raises -> Int:
        self.fx = self.answer(Int(self.x.unsafe_ptr()), self.n)
        return Int(self.fx.unsafe_ptr())

    def synthetic(mut self, count: Int, filled: Bool) raises -> Int:
        if len(self.syn) != count:
            self.syn = List[Float32](length=count, fill=Float32(0))
        return Int(self.syn.unsafe_ptr())

    def rows(mut self, n: Int) raises -> Int:
        self.out = self.answer(Int(self.syn.unsafe_ptr()), n)
        return Int(self.out.unsafe_ptr())

    def workers(self) -> Int:
        return 1

    def worker(self) raises -> Self:
        return Self(self.fit, self.dist, self.x, self.n)


def test_agnostic_label_propagation() raises:
    """LabelPropagation / LabelSpreading through `AgnLabelProp` (lane mf-shap3b): the typed query of the public
    predict_proba (`lp_query_run` on the CPU image's x_neighbors entry, the RBF operation chain) over a fitted
    state of 20 rows and two classes; its words against `_LpRef` (rtol 1e-5), Kernel SHAP against the exact
    Shapley values of `_LpRef`, additivity; a NaN row refused in the public method's words. One query at a time
    (the handle has no parallel workers)."""
    var glue = _glue()
    var f = _meta_fixture()
    var fit = f.bg.copy()
    var dist = List[Float32](length=_LP_FIT * 2, fill=Float32(0))
    for j in range(_LP_FIT):  # small-loop(fixture: a label distribution per fitted row)
        var a = Float32(0.2) if j % 3 == 0 else (Float32(0.7) if j % 3 == 1 else Float32(0.5))
        dist[2 * j] = a
        dist[2 * j + 1] = Float32(1.0) - a
    var request = LpQueryRequest(0, Int(fit.unsafe_ptr()), Int(dist.unsafe_ptr()), 0, 0, 0, _LP_FIT, LR_D, 2, False,
                                 0, _LP_RBF, _LP_GAMMA, False, False, False, False)
    var driver = XnDriver(xn_native_v1)
    var x = f.x.copy()
    var bg = f.bg.copy()
    var ref_m = _LpRef(fit, dist, x, META_ROWS)
    var want = ref_m.answer(Int(x.unsafe_ptr()), META_ROWS)
    var h_bg = AgnLabelProp(driver, request, "LabelPropagation", Int(bg.unsafe_ptr()), META_BG, LR_D)
    var h = AgnLabelProp(driver, request, "LabelPropagation", Int(x.unsafe_ptr()), META_ROWS, LR_D)
    var got = _F32P(unsafe_from_address=h.explained())
    for i in range(META_ROWS * 2):  # small-loop(oracle: every output word)
        require(abs(Float64(got[i]) - Float64(want[i])) <= 1e-5 * abs(Float64(want[i])) + 1e-6,
                "label propagation output " + String(i) + ": handle " + String(got[i]) + ", oracle " + String(want[i]))
    var serial = _kernel_of(glue, h_bg, h, x, bg, META_ROWS, META_BG, LR_D, 2, False)
    var exact = _exact_serial(ref_m, x, bg, META_ROWS, META_BG, LR_D, 2)
    _close(serial.phi, exact, 1e-4, 1e-5, "label propagation kernel shap vs the exact Shapley values of the oracle")
    var fx = List[Float32](length=META_ROWS * 2, fill=Float32(0))
    for i in range(META_ROWS * 2):  # small-loop(the handle's explained words)
        fx[i] = got[i]
    _additive(serial, fx, META_ROWS, LR_D, 2, False, 0.0, 1e-5, "label propagation kernel shap additivity")
    var bad = x.copy()
    bad[1] = Float32.MAX_FINITE * Float32(2.0)
    var nan_h = AgnLabelProp(driver, request, "LabelPropagation", Int(bad.unsafe_ptr()), META_ROWS, LR_D)
    var words = String("")
    try:
        _ = nan_h.explained()
    except e:
        words = String(e)
    require(words == "Input X contains NaN or infinity (LabelPropagation).",
            "label propagation: a non-finite row is not refused in the public method's words: '" + words + "'")
    _ = nan_h^
    _ = bad^
    _ = h_bg^
    _ = h^
    _ = ref_m^
    _ = want^
    _ = fx^
    _ = request^
    _ = fit^
    _ = dist^
    _ = x^
    _ = bg^
    _meta_free(f)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
