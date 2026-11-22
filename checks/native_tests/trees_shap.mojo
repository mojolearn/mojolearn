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
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, fixture_f32, fixture_i32, require
from checks.native_tests.py2mojo_forest_fits import RfSpec, Forest, rf_fit, predict
from xtrees.shap import shap_path_width
from xtrees.shap_host import shap_prepare, tree_shap_values

comptime SK_DIR = "checks/native_tests/fixtures/trees_sk/"
comptime NA = 1050
comptime D = 10
comptime K = 3
comptime ROWS = 40
comptime TREES = 5

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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
