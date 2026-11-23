# SPDX-License-Identifier: Apache-2.0
"""AdaBoost host fits through the product's typed RUN, for the Mojo checks of lane mf-ada (2026-11-23).
Importable; no `main()`, no `test_*`.

`ada_fit` builds the `AdaFitPlan` the boost door's UNPACK (bindings/trees_public_boost_door.mojo `_ada_native`)
builds for a default base, AdaBoostClassifier's DecisionTreeClassifier(max_depth=1) or AdaBoostRegressor's
DecisionTreeRegressor(max_depth=3): the base's configuration (checks/native_tests/py2mojo_forest_fits.mojo
`RfSpec.decision_tree` + `rf_config`, the door's `_rf_config` slots), the RF host image's forest entry and caps,
the uniform starting weights 1 / n (the door's `_normalized_weights` of sample_weight=None), the class codes or
targets, and either no data session (the classifier takes RUN's column-major fit, the regressor its column-major row
gather, the bodies the host sessions run) or, with `session` (lane mf-ada2), a data session opened on the host image
through the GIL-free marked entry as the door's `_member_session` opens it (the classifier's column-major X, the
regressor's row-major X): the session registry has its own lock, so a process with no Python serves it. Then `ada_fit_run`, and each kept round's export copied
out of the image (what the door's PACK `forest_export_handle` copies). `ada_predict_codes` / `ada_predict_values`
are the public predicts' RUNs (`ada_classifier_run` ADA_CODES, `ada_regressor_run`) over the members' host
predictions (`predict`, the STRICT route the members' doors take on the host column).

`ada_member_fit` (lane mf-ada2) is the MEMBER route the door's `_ada_members` takes for every other tree base: the
base's forest member plan (`member_plan_of`: the parameter words `forest_member_admit_common` writes for a
RandomForest*, ExtraTrees* or DecisionTree* configuration, a random-splitter tree's inner forest included; the host
forest entry of its family as the route), the class_weight rows the caller passes (the door's
`trees_class_rows_only`) multiplied in by the x_trees label weights kernels (`label_weights_native_v1`, what the
door's driver calls), then `ada_fit_run` and the ensemble's predict over the kept states' own member predicts.
With `oob` (lane mf-oob3: a RandomForest* base with oob_score=True) the plan's out-of-bag word is set, and each
kept state's out-of-bag blocks (`FM_BLOCK_OOB`, `FM_BLOCK_OOB_SCORE`: what the door's PACK writes into its member's
`oob_decision_function_` / `oob_prediction_` and `oob_score_`) are copied out before the states are freed.

Call `publish_host_images()` (checks/native_tests/bm_rest_boost_fits.mojo) and `publish_host_members()` first. Each fit is serial by nature
(each round's weights come from the last); callers run independent fits as case-table rows across the host cores.
"""
from std.memory import bitcast
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.entry_table import EntryRef, entry_ref_gil_free
from core.device_route import DeviceRoute, DEVICE_CPU
from core.member_abi import (
    MemberPlan, MemberMatrix, member_resolve_family, member_call, member_call_of, MEMBER_OP_PREDICT, MEMBER_OP_EXPORT,
    MEMBER_F32, MEMBER_I32,
)
from core.forest_member_run import FM_HEAD, FM_WORD_OOB, FM_BLOCK_OOB, FM_BLOCK_OOB_SCORE
from core.member_families import MEMBER_FAMILY_TREE, MEMBER_FAMILY_FOREST, MEMBER_FAMILY_EXTRA_TREES
from core.tree_member import tree_member_publish_image
from core.forest_member import forest_member_publish_image
from core.extra_trees_member import extra_trees_member_publish_image
from core.label_weights_native_api import LabelWeightsDriver
from xtrees.label_weights_native import label_weights_native_v1
from bindings.forest_config_run import ForestConfig, forest_target_need
from core.forest_calls import (
    xf_ref, xf_caps, xf_export_sizes, xf_export_into, xf_export_release, xf_session_open, xf_session_close,
)
from core.forest_entry_table import ForestEntryKey, FOREST_TABLE_RF, FOREST_TABLE_ET
from xtrees.glue_calls import xt_ref
from xtrees.trees_forest_run import forest_export_counts
from xtrees.trees_ada_fit_run import AdaFitPlan, AdaFitResult, ada_fit_run, ada_release_handles, ada_release_members
from xtrees.trees_meta_vote_run import ada_classifier_run, ada_regressor_run, ADA_CODES
from checks.native_tests.bm_rest_boost_fits import trees_key
from checks.native_tests.testsuite_cases import require
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
            y: List[Float32], session: Bool = False) raises -> AdaModel:
    """`AdaBoost*(n_estimators, random_state=seed[, loss]).fit(X, y)` with sample_weight=None on the host
    images: `codes` the int32 class codes 0..k-1 (classifier) or `y` the float32 targets (regressor)."""
    var key = trees_key()
    var glue = xt_ref(key)
    var forest = entry_ref_gil_free(xf_ref(ForestEntryKey(FOREST_TABLE_RF, key.copy())))
    var caps = xf_caps(forest)
    var xcm = List[Float32]()
    if s.classifier:
        xcm = column_major(x, n, d)
    var handle = 0
    if session:
        if s.classifier:
            handle = xf_session_open(forest, Int(xcm.unsafe_ptr()), n, d, False, False)
        else:
            handle = xf_session_open(forest, Int(x.unsafe_ptr()), n, d, True, False)
    var plan = AdaFitPlan(s.classifier, n, d, k if s.classifier else 0, s.n_estimators, s.rate, s.seed,
                          rf_config(RfSpec.decision_tree(s.classifier, s.max_depth)), glue, forest, caps, handle)
    plan.squared = RF_SQUARED
    plan.loss = s.loss
    plan.x = Int(x.unsafe_ptr())
    var w = List[Float64](length=n, fill=1.0 / Float64(n))
    plan.weights = Int(w.unsafe_ptr())
    if s.classifier:
        plan.xcm = Int(xcm.unsafe_ptr())
        plan.codes = Int(codes.unsafe_ptr())
    else:
        plan.y = Int(y.unsafe_ptr())
    var result = AdaFitResult()
    try:
        result = ada_fit_run(plan)
    finally:
        if handle > 0:
            xf_session_close(forest, handle)
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


# ------------------------------------------------------------ the member route (lane mf-ada2)


def publish_host_members() raises:
    """The member entries the host forest images publish at module init (Tree and Forest beside the RF entry,
    ExtraTrees beside the ET entry)."""
    tree_member_publish_image("cpu")
    forest_member_publish_image("cpu")
    extra_trees_member_publish_image("cpu")


def member_plan_of(config: ForestConfig, family: Int, classifier: Bool, oob: Bool = False) raises -> MemberPlan:
    """The plan `forest_member_admit_common` fills for a base of this configuration on the host column: words 0-12
    (kind, layout, criterion, the max_features form / count / fraction, the image bin flag, the ET rows fraction,
    tier fast 0, engine groves 0, no importance fit, the out-of-bag word `oob` (the estimator's oob_score), the slot
    count), then three words per slot from `FM_HEAD`; the route the CPU forest entry of the family's table (RF for
    Tree and Forest, ET for ExtraTrees)."""
    var words = List[Float64](capacity=FM_HEAD + 3 * len(config.ints))
    words.append(0.0 if classifier else 1.0)
    words.append(1.0 if config.et else 0.0)
    words.append(Float64(config.criterion))
    words.append(Float64(config.features.form))
    words.append(Float64(config.features.count))
    words.append(config.features.fraction)
    words.append(1.0 if config.bins_image else 0.0)
    words.append(config.rows_fraction)
    words.append(0.0)
    words.append(0.0)
    words.append(0.0)
    words.append(1.0 if oob else 0.0)
    var slots = len(config.ints)
    words.append(Float64(slots))
    for k in range(slots):  # small-loop(the configuration's parameter slots, once per plan)
        words.append(bitcast[DType.float64](Int64(config.ints[k])))
        words.append(config.floats[k])
        words.append(1.0 if config.bad[k] != 0 else 0.0)
    var route = DeviceRoute(DEVICE_CPU)
    route.mode = Int(GLOBAL_NUMERIC_MODE)
    route.vendor = 0
    var table = FOREST_TABLE_ET if family == MEMBER_FAMILY_EXTRA_TREES else FOREST_TABLE_RF
    route.entry = xf_ref(ForestEntryKey(table, trees_key()))
    var plan = MemberPlan()
    plan.family = family
    plan.kind = 0 if classifier else 1
    plan.route = route
    require(len(words) == FM_HEAD + 3 * slots and words[FM_WORD_OOB] == (1.0 if oob else 0.0),
            "member_plan_of: the words disagree with the forest member's layout")
    plan.params = words^
    plan.member = member_resolve_family(family, route)
    return plan^


struct AdaMemberModel(Movable):
    """A member-route fit: `estimator_weights_`, `estimator_errors_` and the ensemble's predict of the training X
    (int32 codes or float32 values), from the kept states before they are freed."""
    var alphas: List[Float64]
    var errors: List[Float64]
    var codes: List[Int32]
    var values: List[Float32]
    #: with `oob`: every kept member's out-of-bag output words, concatenated in round order, and its score
    var oob: List[Float64]
    var oob_scores: List[Float64]

    def __init__(out self, var alphas: List[Float64], var errors: List[Float64], var codes: List[Int32],
                 var values: List[Float32]):
        self.alphas = alphas^
        self.errors = errors^
        self.codes = codes^
        self.values = values^
        self.oob = List[Float64]()
        self.oob_scores = List[Float64]()


def ada_member_fit(s: AdaSpec, var config: ForestConfig, family: Int, tree_base: Bool, x: List[Float32], n: Int,
                   d: Int, codes: List[Int32], k: Int, y: List[Float32],
                   class_rows: List[Float32], oob: Bool = False) raises -> AdaMemberModel:
    """`AdaBoost*(estimator=<base of `config`>, n_estimators, random_state=seed[, loss]).fit(X, y)` with
    sample_weight=None on the member route: `codes` 0..k-1 (classifier) or `y` (regressor); `class_rows` the
    base's class_weight row weights (empty: none); `tree_base` the base is a DecisionTree* (the steps' word; a
    random-splitter tree's `family` is its inner forest's, ExtraTrees). The deviance criteria check each round's
    drawn targets."""
    var member_plan = member_plan_of(config, family, s.classifier, oob)
    var key = trees_key()
    var glue = xt_ref(key)
    var need = forest_target_need(config)
    var plan = AdaFitPlan(s.classifier, n, d, k if s.classifier else 0, s.n_estimators, s.rate, s.seed, config^,
                          glue, xf_ref(ForestEntryKey(FOREST_TABLE_RF, key.copy())), 0, 0)
    plan.member_route = True
    plan.member_plan = member_plan.copy()
    plan.tree_base = 1 if tree_base else 0
    plan.loss = s.loss
    plan.x = Int(x.unsafe_ptr())
    var w = List[Float64](length=n, fill=1.0 / Float64(n))
    plan.weights = Int(w.unsafe_ptr())
    if s.classifier:
        plan.codes = Int(codes.unsafe_ptr())
        if len(class_rows) > 0:
            plan.class_rows = Int(class_rows.unsafe_ptr())
            plan.multiply = LabelWeightsDriver(label_weights_native_v1)
    else:
        plan.y = Int(y.unsafe_ptr())
        plan.target_need = need
    var result = ada_fit_run(plan)
    _ = len(x)  # RUN read these words through their addresses
    _ = len(w)
    _ = len(codes)
    _ = len(y)
    _ = len(class_rows)
    var out_codes = List[Int32]()
    var out_values = List[Float32]()
    var oob_words = List[Float64]()
    var oob_scores = List[Float64]()
    try:
        if oob:
            for j in range(len(result.handles)):  # small-loop(the kept states: their out-of-bag blocks)
                var call = member_call_of(member_plan, result.handles[j])
                call.block = FM_BLOCK_OOB
                _ = member_call(member_plan.member, MEMBER_OP_EXPORT, call)
                var words = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call.dst.data)
                for i in range(call.dst.rows * call.dst.cols):  # small-loop(one member's out-of-bag words)
                    oob_words.append(words[i])
                call.block = FM_BLOCK_OOB_SCORE
                _ = member_call(member_plan.member, MEMBER_OP_EXPORT, call)
                oob_scores.append(MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call.dst.data)[0])
        var blocks_i = List[List[Int32]]()
        var blocks_f = List[List[Float32]]()
        var addrs = List[Int]()
        for j in range(len(result.handles)):  # small-loop(the kept states: their member predicts of X)
            var call = member_call_of(member_plan, result.handles[j])
            call.x = MemberMatrix(Int(x.unsafe_ptr()), n, d, MEMBER_F32)
            if s.classifier:
                blocks_i.append(List[Int32](length=n, fill=Int32(0)))
                call.dst = MemberMatrix(Int(blocks_i[j].unsafe_ptr()), n, 1, MEMBER_I32)
            else:
                blocks_f.append(List[Float32](length=n, fill=Float32(0)))
                call.dst = MemberMatrix(Int(blocks_f[j].unsafe_ptr()), n, 1, MEMBER_F32)
            _ = member_call(member_plan.member, MEMBER_OP_PREDICT, call)
        for j in range(len(result.handles)):  # small-loop(the members' block addresses)
            addrs.append(Int(blocks_i[j].unsafe_ptr()) if s.classifier else Int(blocks_f[j].unsafe_ptr()))
        if s.classifier:
            var acc = List[Float64](length=n * k, fill=0.0)
            out_codes = List[Int32](length=n, fill=Int32(0))
            ada_classifier_run(glue, addrs, result.alphas, n, k, Int(acc.unsafe_ptr()), ADA_CODES,
                               Int(out_codes.unsafe_ptr()))
            _ = len(acc)
        else:
            out_values = List[Float32](length=n, fill=Float32(0))
            ada_regressor_run(glue, addrs, result.alphas, n, Int(out_values.unsafe_ptr()))
        _ = len(blocks_i)  # the RUNs above read the member blocks through `addrs`
        _ = len(blocks_f)
        _ = len(x)
    finally:
        ada_release_members(member_plan, result.handles)
    var model = AdaMemberModel(result.alphas.copy(), result.errors.copy(), out_codes^, out_values^)
    model.oob = oob_words^
    model.oob_scores = oob_scores^
    return model^
