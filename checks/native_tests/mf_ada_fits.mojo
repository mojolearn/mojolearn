# SPDX-License-Identifier: Apache-2.0
"""AdaBoost host fits through the product's typed RUN, for the Mojo checks of lane mf-ada (2026-11-23).
Importable; no `main()`, no `test_*`.

`ada_fit` builds the `AdaFitPlan` the boost door's UNPACK (bindings/trees_public_boost_door.mojo `_ada_native`)
builds for a default base, AdaBoostClassifier's DecisionTreeClassifier(max_depth=1) or AdaBoostRegressor's
DecisionTreeRegressor(max_depth=3): the base's configuration (checks/native_tests/py2mojo_forest_fits.mojo
`RfSpec.decision_tree` + `rf_config`, the door's `_rf_config` slots), the RF host image's forest entry and caps,
the uniform starting weights 1 / n (the door's `_normalized_weights` of sample_weight=None), the class codes or
targets, and NO data session (a process with no Python publishes the host image without sessions,
core/mojolearn_rf_host_run.mojo `RfHostNoSessions`): the classifier takes RUN's column-major fit, the regressor its
column-major row gather, the bodies the host sessions run. Then `ada_fit_run`, and each kept round's export copied
out of the image (what the door's PACK `forest_export_handle` copies). `ada_predict_codes` / `ada_predict_values`
are the public predicts' RUNs (`ada_classifier_run` ADA_CODES, `ada_regressor_run`) over the members' host
predictions (`predict`, the STRICT route the members' doors take on the host column).

Call `publish_host_images()` (checks/native_tests/bm_rest_boost_fits.mojo) first. Each fit is serial by nature
(each round's weights come from the last); callers run independent fits as case-table rows across the host cores.
"""
from core.entry_table import EntryRef
from core.forest_calls import xf_ref, xf_caps, xf_export_sizes, xf_export_into, xf_export_release
from core.forest_entry_table import ForestEntryKey, FOREST_TABLE_RF
from xtrees.glue_calls import xt_ref
from xtrees.trees_forest_run import forest_export_counts
from xtrees.trees_ada_fit_run import AdaFitPlan, AdaFitResult, ada_fit_run, ada_release_handles
from xtrees.trees_meta_vote_run import ada_classifier_run, ada_regressor_run, ADA_CODES
from checks.native_tests.bm_rest_boost_fits import trees_key
from checks.native_tests.py2mojo_forest_fits import (
    RfSpec, rf_config, Forest, column_major, predict, argmax_rows, RF_SQUARED,
)


struct AdaSpec(Copyable, Movable):
    """The constructor arguments a fit reads (python/mojolearn/_expansion_trees.py defaults; n_estimators as
    given): learning_rate 1.0, loss 'linear' (code 0; 'square' 1, 'exponential' 2), the default base's max_depth
    (1 for the classifier, 3 for the regressor)."""
    var classifier: Bool
    var n_estimators: Int
    var seed: Int
    var rate: Float64
    var loss: Int
    var max_depth: Int

    def __init__(out self, classifier: Bool, n_estimators: Int, seed: Int):
        self.classifier = classifier
        self.n_estimators = n_estimators
        self.seed = seed
        self.rate = 1.0
        self.loss = 0
        self.max_depth = 1 if classifier else 3


struct AdaModel(Movable):
    """A fitted ensemble: the kept members' exports, `estimator_weights_`, `estimator_errors_`."""
    var members: List[Forest]
    var alphas: List[Float64]
    var errors: List[Float64]
    var classifier: Bool
    var k: Int

    def __init__(out self, var members: List[Forest], var alphas: List[Float64], var errors: List[Float64],
                 classifier: Bool, k: Int):
        self.members = members^
        self.alphas = alphas^
        self.errors = errors^
        self.classifier = classifier
        self.k = k


def _export(forest: EntryRef, handle: Int, d: Int) raises -> Forest:
    """One kept round's export (the five arrays), its handle released."""
    var offsets = List[Int32]()
    var colid = List[Int32]()
    var quesval = List[Float32]()
    var left = List[Int32]()
    var leaves = List[Float32]()
    var trees = 0
    var outputs = 0
    try:
        var sizes = xf_export_sizes(forest, handle)
        var counts = forest_export_counts(sizes)
        offsets = List[Int32](length=counts[0], fill=Int32(0))
        colid = List[Int32](length=counts[1], fill=Int32(0))
        quesval = List[Float32](length=counts[2], fill=Float32(0))
        left = List[Int32](length=counts[3], fill=Int32(0))
        leaves = List[Float32](length=counts[4], fill=Float32(0))
        var dst: List[Int] = [Int(offsets.unsafe_ptr()), Int(colid.unsafe_ptr()), Int(quesval.unsafe_ptr()),
                              Int(left.unsafe_ptr()), Int(leaves.unsafe_ptr())]
        xf_export_into(forest, sizes, dst)
        trees = sizes.trees
        outputs = sizes.outputs
        _ = len(dst)
    finally:
        xf_export_release(forest, handle)
    return Forest(offsets^, colid^, quesval^, left^, leaves^, trees, outputs, d)


def ada_fit(s: AdaSpec, x: List[Float32], n: Int, d: Int, codes: List[Int32], k: Int,
            y: List[Float32]) raises -> AdaModel:
    """`AdaBoost*(n_estimators, random_state=seed[, loss]).fit(X, y)` with sample_weight=None on the host
    images: `codes` the int32 class codes 0..k-1 (classifier) or `y` the float32 targets (regressor)."""
    var key = trees_key()
    var glue = xt_ref(key)
    var forest = xf_ref(ForestEntryKey(FOREST_TABLE_RF, key.copy()))
    var caps = xf_caps(forest)
    var plan = AdaFitPlan(s.classifier, n, d, k if s.classifier else 0, s.n_estimators, s.rate, s.seed,
                          rf_config(RfSpec.decision_tree(s.classifier, s.max_depth)), glue, forest, caps, 0)
    plan.squared = RF_SQUARED
    plan.loss = s.loss
    plan.x = Int(x.unsafe_ptr())
    var w = List[Float64](length=n, fill=1.0 / Float64(n))
    plan.weights = Int(w.unsafe_ptr())
    var xcm = List[Float32]()
    if s.classifier:
        xcm = column_major(x, n, d)
        plan.xcm = Int(xcm.unsafe_ptr())
        plan.codes = Int(codes.unsafe_ptr())
    else:
        plan.y = Int(y.unsafe_ptr())
    var result = ada_fit_run(plan)
    _ = len(x)  # RUN read these words through their addresses
    _ = len(w)
    _ = len(xcm)
    _ = len(codes)
    _ = len(y)
    var members = List[Forest]()
    var j = 0
    try:
        while j < len(result.handles):  # small-loop(the kept rounds: at most n_estimators exports)
            var handle = result.handles[j]
            j += 1
            members.append(_export(forest, handle, d))
    except problem:
        var rest = List[Int]()
        for i in range(j, len(result.handles)):  # small-loop(the kept rounds not exported, error path only)
            rest.append(result.handles[i])
        ada_release_handles(forest, rest)
        raise problem^
    return AdaModel(members^, result.alphas.copy(), result.errors.copy(), s.classifier, k)


def ada_predict_codes(m: AdaModel, x: List[Float32], n: Int) raises -> List[Int32]:
    """`AdaBoostClassifier.predict`'s int32 codes: each member's host codes, then `ada_classifier_run`."""
    var blocks = List[List[Int32]]()
    var addrs = List[Int]()
    for j in range(len(m.members)):  # small-loop(the members: their host predictions)
        var votes = predict(m.members[j], x, n)
        var c = argmax_rows(votes, n, m.k)
        var words = List[Int32](length=n, fill=Int32(0))
        for i in range(n):  # small-loop(one member's codes as int32)
            words[i] = Int32(c[i])
        blocks.append(words^)
    for j in range(len(blocks)):  # small-loop(the members' block addresses)
        addrs.append(Int(blocks[j].unsafe_ptr()))
    var acc = List[Float64](length=n * m.k, fill=0.0)
    var out = List[Int32](length=n, fill=Int32(0))
    ada_classifier_run(xt_ref(trees_key()), addrs, m.alphas, n, m.k, Int(acc.unsafe_ptr()), ADA_CODES,
                       Int(out.unsafe_ptr()))
    _ = len(acc)
    _ = len(blocks)  # the RUN above read the member blocks through `addrs`
    return out^


def ada_predict_values(m: AdaModel, x: List[Float32], n: Int) raises -> List[Float32]:
    """`AdaBoostRegressor.predict`: the weighted median of the members' host predictions (`ada_regressor_run`)."""
    var blocks = List[List[Float32]]()
    var addrs = List[Int]()
    for j in range(len(m.members)):  # small-loop(the members: their host predictions)
        blocks.append(predict(m.members[j], x, n))
    for j in range(len(blocks)):  # small-loop(the members' block addresses)
        addrs.append(Int(blocks[j].unsafe_ptr()))
    var out = List[Float32](length=n, fill=Float32(0))
    ada_regressor_run(xt_ref(trees_key()), addrs, m.alphas, n, Int(out.unsafe_ptr()))
    _ = len(blocks)  # the RUN above read the member blocks through `addrs`
    return out^
