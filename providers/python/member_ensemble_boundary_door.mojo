# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the Ensemble family: the META member kind (lane
mf-meta3, 2026-11-23; RUN: core/ensemble_member.mojo). PUBLISHED for
VotingClassifier, VotingRegressor, StackingClassifier and StackingRegressor,
and (lane mf-meta3b, THE UNIT KINDS) OneVsRestClassifier,
CalibratedClassifierCV, BaggingClassifier and BaggingRegressor
(python/mojolearn/_expansion_trees.py `_mojolearn_member = (39, kind)`), and
(lane mf-ada4, THE ADABOOST KIND) AdaBoostClassifier and AdaBoostRegressor
(`_admit_ada` / `_ada_pack` / `_ada_state`; a base's class_weight is the
kind's class-weight tail since lane mf-split4).
THE LABELS (lane mf-split4): a FIT caller's labels of the codes it fits on
(`MemberOwner.labels`, bindings/member_boundary_door.mojo
`member_admit_labeled`) walk down the admission: each nested estimator gets
the labels its own fit sees (`_child_labels`), and a nested classifier's
class_weight table is read over them (`_admit_nested`), as
`member_set_admit` reads a leaf's.
Since lane mf-search4 also GridSearchCV / RandomizedSearchCV (THE SEARCH
KIND, mark (39, 10): `_admit_search` / `_search_pack` / `_search_state`;
the candidates and their clones are read once per admission and kept by
the search's identity, `_search_prepare`) and MultiOutputClassifier /
MultiOutputRegressor (THE MULTI-OUTPUT KIND, (39, 12) / (39, 13):
`_admit_multi` / `_multi_pack` / `_multi_state`). RandomTreesEmbedding
carries no mark and keeps its door. A unit kind answers False, and its door keeps the estimator, for:
a OneVsRest base whose kind lacks the scores its predict reads; a
Calibrated cv that is a splitter of ours other than KFold / StratifiedKFold
(since lane mf-split4b an index list and our KFold / StratifiedKFold are
the kind's, as for a stack) or a base that takes the resident DeviceRows
route (a measured winner); a
Bagging base that is not a tree, forest or extra-trees member, or one that
names another device. A Bagging configuration that may take the batched
forest route (a measured winner, decided per fit by the weights and X's
shape) is admitted with that forest as its second plan (lane mf-meta3c,
`_batched_forest`); the kind takes the route per fit as the door does.

UNPACK (`ensemble_member_admit`, GIL held, once per admission): the
meta-estimator's parameters, read as its own fit reads them
(bindings/trees_public_meta_door.mojo `_voting_fit` / `_stacking_fit`);
every nested estimator admitted through the LEAF dispatch
(bindings/member_leaf_boundary_door.mojo), a nested meta-estimator at any
depth through the admission WORK-LIST (lane mf-meta3c: deepest first, no
recursion; PACK and IMPORT walk the same way);
the nested plans minted once into the plan store and released by the
capsule kept in the owner's pins. Answers False (the caller keeps its own
path, whose doors raise the estimator's own words) for anything the native
kind does not take exactly as the door does: a nested estimator with no
serving member, a cv SPLITTER of ours other than KFold / StratifiedKFold
(those are read here as typed words and split inside each fit since lane
mf-split4, `_split_words`; an index list is read here into typed folds
since lane mf-meta3b), a non-boolean passthrough, weights of the wrong
length, an unknown `voting`. A classifier stacked with
`stack_method='predict'` is served since lane mf-meta3b (its predicted
codes are the meta features). A user's OWN splitter object (Stacking,
Calibrated) is REFUSED here in the meta door's words (`_refuse_cv_callback`,
lane mf-split4b): its `split` is a Python callback, never called.

PACK (`ensemble_member_pack`): the attributes the meta-estimator's own fit
writes: one clone per nested estimator with its state packed into it
(`member_leaf_pack` / `member_leaf_pack_labeled` over the codes 0 .. k-1,
the classes the door's member sets give them), `estimators_`,
`named_estimators_`, `n_features_in_`, `_fitted`, and per kind `classes_` /
`n_classes_` (vote) or `stack_method_`, `_widths_`, `final_estimator_`,
`classes_`, `_binary` (stack); then the nested fit notices with each nested
member's own context, in fit order. The META state keeps its nested states.

IMPORT (`ensemble_member_state`): a fitted meta-estimator's nested fitted
estimators imported through their leaf families (`member_leaf_state`) and
handed to a new META state (`META_BLOCK_NESTED`, `META_BLOCK_WIDTHS`).

SERIAL BY NATURE: one meta-estimator's admission, publication or import.
"""
from std.python import Python, PythonObject
from bindings.member_fit_notices_pack import member_fit_notices_raise
from bindings.member_leaf_boundary_door import (
    member_leaf_admit_serving, member_leaf_pack, member_leaf_pack_labeled, member_leaf_state,
)
from bindings.member_owner_unpack import MemberOwner, member_labels_take, member_labels_hold, member_mark
from bindings.member_class_weight_unpack import member_cw_fill
from bindings.class_weight_boundary_unpack import class_weight_dictionary_values, CLASS_WEIGHT_TREES
from bindings.meta_plan_capsule_pack import meta_plan_capsule
from bindings.prep_native_estimator_door import prep_estimator_plan, prep_estimator_clone, prep_estimator_has
from bindings.py_balanced import py_call, py_get, py_item, py_setattr, py_setitem, py_type, py_contains, py_fields
from core.device_route import DeviceRoute, DEVICE_CPU
from bindings.admitted_input_unpack import admitted_unfilled
from bindings.native_unit_threads_unpack import native_unit_device_workers, native_unit_jobs
from core.ensemble_member import (
    meta_plan_mint_with, meta_state_plans, meta_state_notices,
    META_VOTE_CLASSIFIER, META_VOTE_REGRESSOR, META_STACK_CLASSIFIER, META_STACK_REGRESSOR, META_BLOCK_NESTED,
    META_BLOCK_WIDTHS, META_OVR, META_CALIBRATED, META_BAG_CLASSIFIER, META_BAG_REGRESSOR, META_KIND_LAST,
    META_BLOCK_CAL, META_BLOCK_DRAWS, META_BLOCK_OOB, META_BLOCK_BATCHED,
    META_ADA_CLASSIFIER, META_ADA_REGRESSOR, META_BLOCK_ADA,
)
from core.member_families import (
    MEMBER_FAMILY_TREE, MEMBER_FAMILY_FOREST, MEMBER_FAMILY_EXTRA_TREES, MEMBER_FAMILY_ENSEMBLE,
)
from xtrees.trees_boost_run import _sub_seed, boost_ada_session_choice, boost_check_n_estimators, boost_check_learning_rate
from bindings.trees_public_forest_door import (
    forest_kind, forest_refresh, forest_config, _report_engine, forest_dt_random_inner, forest_dt_random_adopt,
)
from bindings.trees_route_boundary_door import trees_route_context
from providers.python.forest_member_common_boundary_door import (
    forest_member_admit_common, forest_member_state_common, forest_member_pack_common,
)
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_call_of, member_free, member_kind_capabilities,
    member_resolve_family,
    MEMBER_OP_NEW, MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_I32, MEMBER_I64, MEMBER_F64, MEMBER_IMPORT_DONE,
    MEMBER_CAN_FIT, MEMBER_CAN_PREDICT, MEMBER_CAN_PREDICT_PROBA, MEMBER_CAN_DECISION, MEMBER_CAN_CLASSIFIER,
    MEMBER_CAN_IMPORT, MEMBER_CAN_EXPORT, MEMBER_CAN_SAMPLE_WEIGHT, MEMBER_BLOCK_SCALARS,
)
from xtrees.trees_meta_vote_run import STACK_PREDICT, STACK_PROBA, STACK_DECISION
from core.ensemble_member import (
    meta_plan_metric, META_SEARCH_CLASSIFIER, META_SEARCH_REGRESSOR, META_MULTI_CLASSIFIER, META_MULTI_REGRESSOR,
    META_BLOCK_SEARCH,
)
from core.member_abi import MEMBER_CAN_SCORE, MEMBER_CAN_MULTI_OUTPUT, MEMBER_OP_PREDICT, member_output_dtype
from core.member_abi import MEMBER_OP_PREDICT_PROBA, MEMBER_OP_DECISION
from core.msel_scorer_run import msel_scorer_code, msel_scorer_row, msel_scorer_regression
from bindings.msel_scorer_native_door import msel_scorer_response
from bindings.native_loaded_module_door import native_loaded_module
from bindings.metrics_keys_unpack import metrics_keys
from bindings.arena_prog_native_door import metric_route_of, metric_route_entries
from bindings.msel_candidates_door import msel_grid_list, msel_sampler_check, msel_sampler_list
from bindings.msel_estimator_native_door import msel_native_clone, msel_native_set_params, msel_clone_parameter
from bindings.msel_common_door import msel_get_scorer, msel_f64_list, msel_i32_list, msel_is_integral
from bindings.msel_search_state_door import msel_search_update
from bindings.msel_split_door import (
    msel_cls, MselSplitter, SPLITTER_KFOLD, SPLITTER_STRATIFIED_KFOLD, msel_seed, msel_truth, msel_sabotage,
)
from bindings.msel_common_door import msel_int
from bindings.msel_split_unpack import msel_split_route, MselRoutePin
from bindings.msel_split_plan_run import MselSeed
from core.ensemble_member import meta_plan_split, META_SPLIT_NONE, META_SPLIT_KFOLD, META_SPLIT_STRATIFIED

comptime _I64P = MutPointer[Int64, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]


def _stack(kind: Int) -> Bool:
    return kind == META_STACK_CLASSIFIER or kind == META_STACK_REGRESSOR


def _classifier(kind: Int) -> Bool:
    return (kind == META_VOTE_CLASSIFIER or kind == META_STACK_CLASSIFIER or kind == META_OVR
            or kind == META_CALIBRATED or kind == META_BAG_CLASSIFIER or kind == META_ADA_CLASSIFIER
            or kind == META_SEARCH_CLASSIFIER or kind == META_MULTI_CLASSIFIER)


def _search(kind: Int) -> Bool:
    return kind == META_SEARCH_CLASSIFIER or kind == META_SEARCH_REGRESSOR


def _multi(kind: Int) -> Bool:
    return kind == META_MULTI_CLASSIFIER or kind == META_MULTI_REGRESSOR


def _ada(kind: Int) -> Bool:
    return kind == META_ADA_CLASSIFIER or kind == META_ADA_REGRESSOR


def _unit(kind: Int) -> Bool:
    return kind == META_OVR or kind == META_CALIBRATED or kind == META_BAG_CLASSIFIER or kind == META_BAG_REGRESSOR


def _bag(kind: Int) -> Bool:
    return kind == META_BAG_CLASSIFIER or kind == META_BAG_REGRESSOR


def _kind_name(kind: Int) -> String:
    if kind == META_OVR:
        return String("OneVsRestClassifier")
    if kind == META_CALIBRATED:
        return String("CalibratedClassifierCV")
    if kind == META_BAG_CLASSIFIER:
        return String("BaggingClassifier")
    return String("BaggingRegressor")


def _addr(array: PythonObject) raises -> Int:
    """The words of a mojolearn Array (its `_addr` slot)."""
    return Int(py=array._addr)


struct _Active(Movable):
    """The active (name, estimator) entries of `estimators` ('drop' left out),
    as the door's `_active` reads them. `ok` False: the door raises."""
    var names: PythonObject
    var ests: PythonObject
    var ok: Bool

    def __init__(out self) raises:
        self.names = Python.list()
        self.ests = Python.list()
        self.ok = False


def _active(fields: PythonObject, b: PythonObject) raises -> _Active:
    var got = _Active()
    var estimators = py_get(fields, "estimators")
    if estimators is None or (py_type(estimators) is not b.list and py_type(estimators) is not b.tuple):
        return got^
    var seen = Python.list()
    for i in range(len(estimators)):  # driver-loop: over the (name, estimator) entries (admission, once each)
        var pair = estimators[i]
        if (py_type(pair) is not b.tuple and py_type(pair) is not b.list) or len(pair) != 2:
            return got^
        if py_contains(seen, pair[0]):
            return got^
        _ = py_call(seen.append, pair[0])
        if py_type(pair[1]) is b.str:
            if String(pair[1]) == "drop":
                continue
            return got^
        _ = py_call(got.names.append, pair[0])
        _ = py_call(got.ests.append, pair[1])
    got.ok = len(got.ests) > 0
    return got^


def _stack_method(fields: PythonObject, est: PythonObject, classifier: Bool, context: PythonObject) raises -> Int:
    """`_StackingBase._method` (the door's `_stack_method`): -1 for a value
    the door refuses."""
    var method = py_get(fields, "stack_method")
    var name = String("auto") if method is None else String(method)
    if name == "auto":
        if classifier:
            if prep_estimator_has(est, "predict_proba", context):
                return STACK_PROBA
            if prep_estimator_has(est, "decision_function", context):
                return STACK_DECISION
        return STACK_PREDICT
    if name == "predict_proba":
        return STACK_PROBA
    if name == "decision_function":
        return STACK_DECISION
    if name == "predict":
        return STACK_PREDICT
    return -1


def _stack_need(kind: Int) -> Int:
    if kind == STACK_PROBA:
        return MEMBER_CAN_PREDICT_PROBA
    if kind == STACK_DECISION:
        return MEMBER_CAN_DECISION
    return MEMBER_CAN_PREDICT


def _stack_name(kind: Int) -> String:
    if kind == STACK_PROBA:
        return String("predict_proba")
    if kind == STACK_DECISION:
        return String("decision_function")
    return String("predict")


struct _Nested(Movable):
    """The admitted nested members: plans, kind masks, pins."""
    var plans: List[MemberPlan]
    var masks: List[Int]
    var pins: PythonObject

    def __init__(out self) raises:
        self.plans = List[MemberPlan]()
        self.masks = List[Int]()
        self.pins = Python.list()


def _child_labels(kind: Int, labels: PythonObject, b: PythonObject) raises -> PythonObject:
    """THE LABELS A NESTED ESTIMATOR'S FIT SEES (lane mf-split4), from the
    labels `labels` of the codes the meta-estimator itself is fitted on
    (None or empty: none known), as scikit-learn's fits hand them on: a
    Voting, Stacking or Bagging classifier fits its members (and a stack its
    final estimator) on the label encoder's codes 0 .. k-1; OneVsRest fits
    each base on its indicator column, the labels 0 and 1; AdaBoost's door
    reads its base's class_weight over its own label encoder's codes
    (`ada_fit`: `trees_class_rows_only` over `trees_encode_labels(codes)`),
    so its labels are the codes too; Calibrated and a search fit theirs on
    y as given, the same labels. A
    MultiOutput base sees one column per output (its labels differ per
    output: none here), a regressor none. A nested class_weight dict is read
    over these (`_admit_nested`), as the member's own fit door reads it."""
    if not _classifier(kind) or _multi(kind):
        return PythonObject(None)
    if kind == META_OVR:
        return py_call(b.list, py_call(b.range, PythonObject(2)))
    if labels is None or len(labels) == 0:
        return PythonObject(None)
    if kind == META_CALIBRATED or _search(kind):
        return labels
    return py_call(b.list, py_call(b.range, PythonObject(len(labels))))


def _admit_nested(done: _Done, mut nested: _Nested, est: PythonObject, classifier: Bool, needs: Int,
                  context: PythonObject, b: PythonObject, labels: PythonObject) raises -> Bool:
    """One nested estimator through the LEAF dispatch, with the kind-exact
    mask (one state minted, asked, released); a nested META estimator from
    the work-list's admitted table `done` (lane mf-meta3c: admitted before
    this one, no recursion). False: not served. `labels`: the labels the
    nested estimator's fit sees (`_child_labels`; None: none known): a
    leaf classifier's class_weight dict table is read over them here, as
    `member_set_admit` reads a leaf's (lane mf-split4); a value or a dict
    its estimator refuses is not served (its door raises the words)."""
    var plan = prep_estimator_plan(est, context)
    if plan is None or py_get(plan, "member") is None:
        return False
    # A nested META estimator answers by its kind-exact mask (a search's class
    # says no estimator type of its own: its base decides, lane mf-search4).
    if member_mark(plan, b).family != MEMBER_FAMILY_ENSEMBLE and Bool(py=plan["classifier"]) != classifier:
        return False
    if member_mark(plan, b).family == MEMBER_FAMILY_ENSEMBLE:
        var at = done.find(est)
        if at < 0:
            return False
        var meta_mask = done.masks[at]
        if (meta_mask & needs) != needs or ((meta_mask & MEMBER_CAN_CLASSIFIER) != 0) != classifier:
            return False
        nested.plans.append(done.plans[at].copy())
        nested.masks.append(meta_mask)
        _ = py_call(nested.pins.append, done.pins[at])
        return True
    var fields = py_call(plan["fields_get"], est, py_type(est))
    var owner = member_leaf_admit_serving(est, fields, plan, plan["context"], b, needs)
    if not owner.published:
        return False
    if classifier and labels is not None and (needs & MEMBER_CAN_FIT) != 0:
        try:
            if not member_cw_fill(owner, est, fields, labels, plan["context"], b):
                return False
        except:
            return False  # a class_weight its estimator refuses: the door raises its words
    var mask = 0
    try:
        mask = member_kind_capabilities(owner.plan)
    except:
        return False
    if (mask & needs) != needs or ((mask & MEMBER_CAN_CLASSIFIER) != 0) != classifier:
        return False
    nested.plans.append(owner.plan.copy())
    nested.masks.append(mask)
    _ = py_call(nested.pins.append, owner.pins)
    return True


def _int_word(value: PythonObject, b: PythonObject) raises -> Int:
    """An exact int (not a bool), else -1."""
    if py_type(value) is not b.int:
        return -1
    return Int(py=value)


def _trees_key(fields: PythonObject, context: PythonObject, b: PythonObject) raises -> List[Int]:
    """The (mode, vendor) of the x_trees image the meta-estimator's own fit
    connects (`_xt`: `numeric_mode`, else the process default); empty when
    it is not connected (the door raises the table's words)."""
    var none = List[Int]()
    var mode = py_get(fields, "numeric_mode")
    if mode is None:
        var backend = py_get(context, "backend_fields")
        if backend is None:
            return none^
        mode = py_get(backend, "_DEFAULT_MODE")
    if mode is None or py_type(mode) is not b.str:
        return none^
    var word = String(mode)
    if word != "fast" and word != "identical":
        return none^
    var bindings = py_get(context, "trees_bindings")
    if bindings is None:
        return none^
    var tables = py_get(bindings, mode)
    if tables is None:
        return none^
    var table = py_get(tables, "_mojolearn_x_trees")
    if table is None or py_type(table) is b.str:
        return none^
    var key = py_get(table, "x_trees_native_key")
    if key is None:
        return none^
    var out: List[Int] = [Int(py=key[0]), Int(py=key[1])]
    return out^


struct _Done(Movable):
    """THE ADMISSION WORK-LIST's table (lane mf-meta3c): the nested META
    estimators admitted so far (by identity), each one's plan, kind-exact
    mask and pins."""
    var objs: PythonObject
    var plans: List[MemberPlan]
    var masks: List[Int]
    var pins: PythonObject

    def __init__(out self) raises:
        self.objs = Python.list()
        self.plans = List[MemberPlan]()
        self.masks = List[Int]()
        self.pins = Python.list()

    def find(self, est: PythonObject) raises -> Int:
        for i in range(len(self.plans)):  # driver-loop: over the admitted nested meta-estimators (identity)
            if self.objs[i] is est:
                return i
        return -1


#: A meta-estimator nests at most this many meta-estimators (all depths):
#: a guard against an estimator graph that contains itself.
comptime _NESTED_META_MAX = 256


def _meta_children(est: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
                   b: PythonObject, prepared: PythonObject) raises -> PythonObject:
    """The nested estimators of an unfitted meta-estimator that its
    admission reads (and so may be meta-estimators themselves): the active
    voting / stacked members and the final estimator, a OneVsRest or
    Calibrated base. A Bagging base is a tree, forest or extra-trees member
    (`_require_tree`): none."""
    var out = Python.list()
    if _search(kind):
        # THE SEARCH KIND (lane mf-search4): its candidate clones, made once
        # here and kept in `prepared` for its admission.
        var got = _search_prepare(est, fields, context, b, prepared)
        if got is not None:
            var clones = got[1]
            for c in range(len(clones)):  # driver-loop: over the candidates' clones
                _ = py_call(out.append, clones[c])
        return out
    if kind == META_OVR or kind == META_CALIBRATED or _multi(kind):
        var base = py_get(fields, "estimator")
        if base is not None:
            _ = py_call(out.append, base)
        return out
    if _bag(kind) or _ada(kind):
        return out  # a Bagging or AdaBoost base is a tree, forest or extra-trees member
    var active = _active(fields, b)
    if active.ok:
        for i in range(len(active.ests)):  # driver-loop: over the active members
            _ = py_call(out.append, active.ests[i])
    if _stack(kind):
        var final = py_get(fields, "final_estimator")
        if final is not None:
            _ = py_call(out.append, final)
    return out


def ensemble_member_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """UNPACK: fill `owner.plan.route`, `owner.plan.params` and `owner.pins`
    from the unfitted meta-estimator (module note). False: the caller keeps
    its own path.

    META ESTIMATORS NESTED AT ANY DEPTH (lane mf-meta3c; AGENTS.md part 6:
    no recursion on the admission path): an explicit WORK-LIST. A pre-order
    walk over a stack of pending meta-estimators collects every nested
    meta-estimator (`_meta_children`); they are then admitted deepest first
    (reverse pre-order), each through `_admit_one` with the table of those
    already admitted (`_Done`), which `_admit_nested` reads for a nested
    meta-estimator; the outermost last, into `owner`. A nested one that is
    not served leaves its parents unserved (their doors keep them).

    THE LABELS (lane mf-split4): `owner.labels`, the caller's labels of the
    codes this meta-estimator is fitted on, walk down with the work-list:
    each nested meta-estimator gets the labels its parent's fit hands it
    (`_child_labels`), and each admission hands its leaves theirs."""
    if kind < META_VOTE_CLASSIFIER or kind > META_KIND_LAST:
        return False
    var done = _Done()
    var prepared = Python.dict()  # a search's candidates and clones, once per admission (lane mf-search4)
    var order = Python.list()
    var order_fields = Python.list()
    var order_context = Python.list()
    var order_labels = Python.list()
    var order_kinds = List[Int]()
    var pending = Python.list(model)
    var pending_fields = Python.list(fields)
    var pending_context = Python.list(context)
    var pending_labels = Python.list(owner.labels)
    var pending_kinds: List[Int] = [kind]
    while len(pending_kinds) > 0:  # driver-loop: over the pending meta-estimators (the work-list, no recursion)
        var at = len(pending_kinds) - 1
        var est = pending[at]
        var est_fields = pending_fields[at]
        var est_context = pending_context[at]
        var est_labels = pending_labels[at]
        var est_kind = pending_kinds.pop()
        _ = py_call(pending.pop)
        _ = py_call(pending_fields.pop)
        _ = py_call(pending_context.pop)
        _ = py_call(pending_labels.pop)
        _ = py_call(order.append, est)
        _ = py_call(order_fields.append, est_fields)
        _ = py_call(order_context.append, est_context)
        _ = py_call(order_labels.append, est_labels)
        order_kinds.append(est_kind)
        var child_labels = _child_labels(est_kind, est_labels, b)
        if len(order_kinds) > _NESTED_META_MAX:
            return False
        var children = _meta_children(est, est_fields, est_kind, est_context, b, prepared)
        for c in range(len(children)):  # driver-loop: over one meta-estimator's nested estimators
            var child = children[c]
            var child_plan = prep_estimator_plan(child, est_context)
            if child_plan is None or py_get(child_plan, "member") is None:
                continue
            var mark = member_mark(child_plan, b)
            if mark.family != MEMBER_FAMILY_ENSEMBLE or mark.kind < META_VOTE_CLASSIFIER or mark.kind > META_KIND_LAST:
                continue
            var seen = False
            for q in range(len(order_kinds)):  # driver-loop: over the meta-estimators walked (identity)
                if order[q] is child:
                    seen = True
            for q in range(len(pending_kinds)):  # driver-loop: over the meta-estimators pending (identity)
                if pending[q] is child:
                    seen = True
            if seen:
                continue
            _ = py_call(pending.append, child)
            _ = py_call(pending_fields.append, py_call(child_plan["fields_get"], child, py_type(child)))
            _ = py_call(pending_context.append, child_plan["context"])
            _ = py_call(pending_labels.append, child_labels)
            pending_kinds.append(mark.kind)
    var i = len(order_kinds) - 1
    while i >= 1:  # driver-loop: over the nested meta-estimators, deepest first (admission, once each)
        var sub = MemberOwner()
        sub.plan.family = MEMBER_FAMILY_ENSEMBLE
        sub.plan.kind = order_kinds[i]
        var served = False
        try:
            served = _admit_one(order[i], order_fields[i], order_kinds[i], order_context[i], b, sub, done, prepared,
                                order_labels[i])
            if served:
                sub.plan.member = member_resolve_family(MEMBER_FAMILY_ENSEMBLE, sub.plan.route)
                var mask = member_kind_capabilities(sub.plan)
                _ = py_call(done.objs.append, order[i])
                done.plans.append(sub.plan.copy())
                done.masks.append(mask)
                _ = py_call(done.pins.append, sub.pins)
        except:
            served = False  # not served: its parents look it up and keep their doors
        i -= 1
    return _admit_one(model, fields, kind, context, b, owner, done, prepared, owner.labels)


def _admit_one(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner, done: _Done, prepared: PythonObject,
    labels: PythonObject) raises -> Bool:
    """One meta-estimator's admission (the body `ensemble_member_admit`'s
    work-list runs per meta-estimator): its nested meta-estimators are in
    `done`, a search's candidates in `prepared`; `labels` the labels of the
    codes it is fitted on (THE LABELS, lane mf-split4)."""
    var child = _child_labels(kind, labels, b)
    if _search(kind):
        return _admit_search(model, fields, context, b, owner, done, prepared, child)
    if _multi(kind):
        return _admit_multi(fields, kind, context, b, owner, done)
    if _ada(kind):
        return _admit_ada(fields, kind, context, b, child, owner)
    if _unit(kind):
        return _admit_unit(model, fields, kind, context, b, owner, done, child)
    var classifier = _classifier(kind)
    var active = _active(fields, b)
    if not active.ok:
        return False
    var m = len(active.ests)
    var flag = 0.0
    var folds = 0
    var weights = List[Float64]()
    var stack_kinds = List[Int]()
    var fold_train = List[List[Int32]]()
    var fold_test = List[List[Int32]]()
    var split = _Split()
    var nested = _Nested()
    var base_needs = MEMBER_CAN_FIT | MEMBER_CAN_PREDICT
    if classifier:
        base_needs |= MEMBER_CAN_CLASSIFIER
    if not _stack(kind):
        if kind == META_VOTE_CLASSIFIER:
            var voting = py_get(fields, "voting")
            if voting is None or py_type(voting) is not b.str:
                return False
            var name = String(voting)
            if name == "soft":
                flag = 1.0
            elif name != "hard":
                return False
        var given = py_get(fields, "weights")
        if given is None:
            for _ in range(m):  # driver-loop: over the voting members
                weights.append(1.0)
        else:
            var items = py_call(b.list, given)
            if len(items) != m:
                return False
            for i in range(m):  # driver-loop: over the user weights (one per member)
                weights.append(Float64(py=py_call(b.float, items[i])))
        var needs = base_needs | (MEMBER_CAN_PREDICT_PROBA if flag != 0.0 else 0)
        for i in range(m):  # driver-loop: over the voting members (admission, once each)
            if not _admit_nested(done, nested, active.ests[i], classifier, needs, context, b, child):
                return False
    else:
        var cv = py_get(fields, "cv")
        _refuse_cv_callback(cv, context, b, _CV_CALLBACK_WORDS, False)
        if cv is not None and (py_type(cv) is b.list or py_type(cv) is b.tuple):
            # A user's index list (lane mf-meta3b): its folds read ONCE into
            # typed int32 rows, checked against the fit's rows by the RUN as
            # the door's `_splits` checks them.
            if not _user_folds(cv, b, fold_train, fold_test):
                return False
            folds = len(fold_test)
        elif cv is not None and py_type(cv) is not b.int:
            # THE SPLITTER CV (lane mf-split4): one of our splitters, read
            # here as typed words; each fit computes its folds over its own
            # rows (core/ensemble_member.mojo `_split_folds`).
            if not _split_words(cv, classifier, context, b, split):
                return False
            folds = split.k
        else:
            folds = 5 if cv is None else _int_word(cv, b)
            if folds < 2:
                return False
        var passthrough = py_get(fields, "passthrough")
        if passthrough is not None and py_type(passthrough) is not b.bool:
            return False
        if passthrough is not None and Bool(py=passthrough):
            flag = 1.0
        for i in range(m):  # driver-loop: over the stacked members (admission, once each)
            # A classifier stacked with stack_method='predict' writes its
            # predicted codes (core/ensemble_member.mojo `_label_block`).
            var method = _stack_method(fields, active.ests[i], classifier, context)
            if method < 0:
                return False
            stack_kinds.append(method)
            if not _admit_nested(done, nested, active.ests[i], classifier, base_needs | _stack_need(method), context, b,
                                 child):
                return False
        var final = py_get(fields, "final_estimator")
        if final is None:
            var defaults = py_get(context, "native_estimator_defaults")
            if defaults is None:
                return False
            final = py_get(defaults, "LogisticRegression" if classifier else "Ridge")
            if final is None:
                return False
        if not _admit_nested(done, nested, final, classifier, base_needs, context, b, child):
            return False
    if not _finish(fields, context, b, owner, nested, kind, flag, folds, weights^, stack_kinds^, List[Float64](),
                   fold_train^, fold_test^):
        return False
    _finish_split(owner, split)
    return True


def _finish_split(mut owner: MemberOwner, split: _Split) raises:
    """THE SPLITTER CV's words into the minted plan entry (a stack's or a
    calibration's), its entries' connections pinned for every fit."""
    if split.kind != META_SPLIT_NONE:
        meta_plan_split(Int(owner.plan.params[1]), split.kind, split.k, split.shuffle, split.seed, split.pin.route,
                        split.sabotage)
        _ = py_call(owner.pins.append, split.pin.provider)  # the program runner's connection, for every fit
        _ = py_call(owner.pins.append, split.pin.entries)


comptime _CV_CALLBACK_WORDS = "TypeError: cv requires an integer, native splitter or list of index pairs; Python callbacks cannot execute"
#: the search door's words for the same cv (bindings/msel_cv_door.mojo `msel_fold_source`)
comptime _SEARCH_CV_CALLBACK_WORDS = "TypeError: Python fold iterator callbacks are not accepted; provide native splitters or explicit list/tuple index pairs"


def _refuse_cv_callback(cv: PythonObject, context: PythonObject, b: PythonObject,
                        words: String, integral_ok: Bool) raises:
    """A cv that is none of None, an int, an index list or one of OUR
    splitters (`family_msel_splitter_layouts`): a user's own splitter object,
    refused at UNPACK in the meta door's words (bindings/
    trees_public_meta_door.mojo `_cv_value`), before anything is fitted, as
    the same meta-estimator fitted at the top level refuses it. Its `split`
    is never called, at UNPACK or in RUN: a user's splitter is a Python
    callback (AGENTS.md "No calls back into Python": callbacks are refused
    before they are invoked; the 2026-11-25 boundary note admits only the
    INPUT's data-interchange protocols). Lane mf-split4b. `integral_ok`: the
    search door's rule, which takes any native integer scalar (a NumPy one
    too, `msel_is_integral`: exact type identities, no ABC hook)."""
    if cv is None or py_type(cv) is b.int or py_type(cv) is b.bool or py_type(cv) is b.list or py_type(cv) is b.tuple:
        return
    if integral_ok:
        try:
            if msel_is_integral(cv, context):
                return
        except:
            return  # no integer table connected: the caller's door decides, in the same words
    var layouts = py_get(context, "family_msel_splitter_layouts")
    if layouts is None:
        return  # no splitter table connected: the caller's door decides, in the same words
    if py_contains(layouts, py_type(cv)):
        return
    raise Error(words)


struct _Split(Movable):
    """UNPACK of THE SPLITTER CV (lane mf-split4): META_SPLIT_*, n_splits,
    shuffle, the seed and the splitters' entries with their pins."""
    var kind: Int
    var k: Int
    var shuffle: Bool
    var seed: MselSeed
    var pin: MselRoutePin
    var sabotage: Bool

    def __init__(out self):
        self.kind = META_SPLIT_NONE
        self.k = 0
        self.shuffle = False
        self.seed = MselSeed(0, False)
        self.pin = MselRoutePin()
        self.sabotage = False


def _split_words(cv: PythonObject, classifier: Bool, context: PythonObject, b: PythonObject,
                 mut split: _Split) -> Bool:
    """A cv that is one of OUR KFold / StratifiedKFold splitters, read as the
    splitter door reads it for `split` (bindings/msel_split_door.mojo
    `msel_split`: `MselSplitter`, its n_splits, shuffle and random_state
    through `field`, the seed by `msel_seed`, the entries by
    `msel_split_route`, the sabotage word). False (the door keeps the
    meta-estimator and splits natively with X and y) for any other of our
    splitter classes, a StratifiedKFold on a regressor (its door refuses
    continuous y in the splitter's words), fewer than two splits, or a value
    the door refuses. A user's own splitter object never reaches here:
    `_refuse_cv_callback` refuses it first."""
    try:
        var cls = py_type(cv)
        var stratified = cls is msel_cls("StratifiedKFold", context)
        if not stratified and cls is not msel_cls("KFold", context):
            return False
        if stratified and not classifier:
            return False
        var sp = MselSplitter(cv, context)
        if sp.kind != (SPLITTER_STRATIFIED_KFOLD if stratified else SPLITTER_KFOLD):
            return False
        split.k = msel_int(sp.field("n_splits", context))
        if split.k < 2:
            return False
        split.shuffle = msel_truth(sp.field("shuffle", context))
        split.seed = msel_seed(sp.field("random_state", context), split.shuffle, context)
        # Every entry either splitter may call (the base table past the
        # stratified program bounds; the program runner for a draw).
        split.pin = msel_split_route(context, True, True)
        split.sabotage = msel_sabotage()
        split.kind = META_SPLIT_STRATIFIED if stratified else META_SPLIT_KFOLD
        return True
    except:
        return False


def _cap_of(fields: PythonObject, b: PythonObject) raises -> Int:
    """The worker bound of the nested sets: an exact positive `n_jobs`, else none (0)."""
    var jobs = py_get(fields, "n_jobs")
    if jobs is not None:
        var word = _int_word(jobs, b)
        if word > 0:
            return word
    return 0


def _finish(fields: PythonObject, context: PythonObject, b: PythonObject, mut owner: MemberOwner, nested: _Nested,
            kind: Int, flag: Float64, folds: Int, var weights: List[Float64], var stack_kinds: List[Int],
            var words: List[Float64], var fold_train: List[List[Int32]], var fold_test: List[List[Int32]]) raises -> Bool:
    """The admission's tail for every kind: the x_trees image's route, the
    nested plans minted once into the plan store (released by the capsule
    kept in the owner's pins), the parameter words."""
    var cap = _cap_of(fields, b)
    var key = _trees_key(fields, context, b)
    if len(key) != 2:
        return False
    var device = DEVICE_CPU
    for u in range(len(nested.plans)):  # driver-loop: over the nested members (their routes)
        if nested.plans[u].route.device != DEVICE_CPU:
            device = nested.plans[u].route.device
            break
    var route = DeviceRoute(device)
    route.mode = key[0]
    route.vendor = key[1]
    var pins = nested.pins
    # Copies, not field moves: `nested` stays whole (AGENTS.md part 6).
    var handle = meta_plan_mint_with(nested.plans.copy(), weights^, stack_kinds^, nested.masks.copy(), words^,
                                     fold_train^, fold_test^)
    _ = py_call(pins.append, meta_plan_capsule(handle))
    owner.plan.route = route
    owner.plan.params = [Float64(kind), Float64(handle), flag, Float64(folds), Float64(cap), 0.0]
    owner.pins = pins
    return True


def _index_run(value: PythonObject, b: PythonObject, mut dst: List[Int32]) raises -> Bool:
    """One side of a user's cv fold as int32 row numbers (exact ints; an
    integer array's items are read as ints). False: a value the door's
    int64 cast takes and these words cannot hold (the caller keeps the
    door)."""
    var items = py_call(b.list, value)
    dst = List[Int32](capacity=len(items))
    for i in range(len(items)):  # driver-loop: over one fold side's row numbers (UNPACK, once per admission)
        var item = items[i]
        if py_type(item) is b.bool:
            return False
        var word = Int(py=py_call(b.int, item))
        if word < -2147483648 or word > 2147483647:
            return False
        dst.append(Int32(word))
    return True


def _user_folds(cv: PythonObject, b: PythonObject, mut fold_train: List[List[Int32]],
                mut fold_test: List[List[Int32]]) raises -> Bool:
    """UNPACK of a cv given as a list of (train, test) pairs (the door's
    `_splits` over an iterable): each pair's two sides as typed rows. False
    (the door keeps it, and raises its own words) for anything not a
    sequence of two-item pairs of integer rows."""
    try:
        for i in range(len(cv)):  # driver-loop: over the user's folds (UNPACK, once per admission)
            var both = py_call(b.tuple, cv[i])
            if len(both) != 2:
                return False
            var tr = List[Int32]()
            var te = List[Int32]()
            if not _index_run(both[0], b, tr) or not _index_run(both[1], b, te):
                return False
            fold_train.append(tr^)
            fold_test.append(te^)
    except:
        return False
    if len(fold_test) == 0:
        raise Error("cv produced no splits")  # the door's `_splits` words, before any fit
    return True


def _export_i64(plan: MemberPlan, state: Int, block: Int) raises -> List[Int]:
    var call = MemberCall(plan.route, state)
    call.block = block
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var out = List[Int]()
    var count = call.dst.rows * call.dst.cols
    if count > 0:
        var words = _I64P(unsafe_from_address=call.dst.data)
        for i in range(count):  # small-loop(a META state's handles or widths: one word per member)
            out.append(Int(words[i]))
    return out^


struct _PackJobs(Movable):
    """THE PACK WORK-LIST (lane mf-meta3c): the nested meta-estimator clones
    still to be packed, each with its fields, context, plan and state (a
    classifier's labels already held in its fields)."""
    var objs: PythonObject
    var fields: PythonObject
    var contexts: PythonObject
    var plans: List[MemberPlan]
    var states: List[Int]

    def __init__(out self) raises:
        self.objs = Python.list()
        self.fields = Python.list()
        self.contexts = Python.list()
        self.plans = List[MemberPlan]()
        self.states = List[Int]()


def _pack_into(clone: PythonObject, clone_fields: PythonObject, clone_context: PythonObject, b: PythonObject,
               plan: MemberPlan, state: Int, labels: PythonObject, mut jobs: _PackJobs) raises:
    """One nested member's PACK: a leaf family's at once; a nested META
    state queued on the work-list (its labels held for its pack), packed
    after its parent (no recursion)."""
    if plan.family == MEMBER_FAMILY_ENSEMBLE:
        if labels is not None:
            member_labels_hold(clone_fields, labels)
        _ = py_call(jobs.objs.append, clone)
        _ = py_call(jobs.fields.append, clone_fields)
        _ = py_call(jobs.contexts.append, clone_context)
        jobs.plans.append(plan.copy())
        jobs.states.append(state)
        return
    if labels is None:
        member_leaf_pack(clone, clone_fields, clone_context, b, plan, state)
    else:
        member_leaf_pack_labeled(clone, clone_fields, clone_context, b, plan, state, labels)


def ensemble_member_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, state: Int) raises:
    """PACK: write the fitted attributes of META state `state` into the
    meta-estimator `model`, as its own fit writes them (module note). A
    nested META state (lane mf-meta3c) is packed into its clone through the
    work-list `_PackJobs`, after its parent, never by recursion."""
    var jobs = _PackJobs()
    _pack_one(model, fields, context, b, plan, state, jobs)
    var next = 0
    while next < len(jobs.states):  # driver-loop: over the nested META states queued (the work-list)
        # Locals first: `jobs` is passed mutably below (it may grow).
        var job_plan = jobs.plans[next].copy()
        var job_state = jobs.states[next]
        var job_model = jobs.objs[next]
        var job_fields = jobs.fields[next]
        var job_context = jobs.contexts[next]
        _pack_one(job_model, job_fields, job_context, b, job_plan, job_state, jobs)
        next += 1


def _pack_one(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, state: Int, mut jobs: _PackJobs) raises:
    """One META state's PACK (the body of `ensemble_member_pack`'s work-list)."""
    var kind = Int(plan.params[0])
    var classifier = _classifier(kind)
    var labels = member_labels_take(fields, b) if classifier else PythonObject(None)
    var nested = meta_state_plans(state)
    var handles = _export_i64(plan, state, META_BLOCK_NESTED)
    var scalars = MemberCall(plan.route, state)
    scalars.block = MEMBER_BLOCK_SCALARS
    _ = member_call(plan.member, MEMBER_OP_EXPORT, scalars)
    var words = _F64P(unsafe_from_address=scalars.dst.data)
    var features = Int(words[0])
    var k = Int(words[1])
    if classifier and labels is None:
        raise Error("RuntimeError: mojolearn: a meta classifier member is packed without its labels")
    if _ada(kind):
        _ada_pack(fields, context, b, plan, state, labels, features, k, jobs)
        return
    if _search(kind):
        _search_pack(model, fields, context, b, plan, state, labels, jobs)
        return
    if _multi(kind):
        _multi_pack(model, fields, context, b, plan, state, labels, features, jobs)
        return
    if _unit(kind):
        # THE UNIT KINDS (lane mf-meta3b): the attributes the door's fit writes.
        var unit_contexts = _unit_pack(model, fields, context, b, plan, state, labels, features, k, jobs)
        if _bag(kind):
            py_setitem(fields, PythonObject("n_features_in_"), PythonObject(features))
        else:
            py_setattr(model, "n_features_in_", PythonObject(features))
            py_setattr(model, "_fitted", PythonObject(True))
        var unit_owners = List[Int]()
        var unit_notices = List[List[Float64]]()
        meta_state_notices(state, unit_owners, unit_notices)
        for i in range(len(unit_owners)):  # driver-loop: over the nested notices (a few warnings each)
            if unit_owners[i] < len(unit_contexts):
                member_fit_notices_raise(unit_notices[i], unit_contexts[unit_owners[i]])
        return
    var active = _active(fields, b)
    if not active.ok:
        raise Error("RuntimeError: mojolearn: the meta-estimator's estimators changed since its admission")
    var bases = len(active.ests)
    if len(handles) != len(nested) or bases + (1 if _stack(kind) else 0) != len(nested):
        raise Error("RuntimeError: mojolearn: the meta member state does not match its estimators")
    var codes = py_call(b.list, py_call(b.range, PythonObject(k)))
    var contexts = Python.list()
    var fitted = Python.list()
    for u in range(bases):  # driver-loop: over the nested members (one clone and one pack each)
        var clone = prep_estimator_clone(active.ests[u], context)
        var clone_plan = prep_estimator_plan(clone, context)
        var clone_fields = py_call(clone_plan["fields_get"], clone, py_type(clone))
        var clone_context = clone_plan["context"]
        _pack_into(clone, clone_fields, clone_context, b, nested[u], handles[u],
                   codes if classifier else PythonObject(None), jobs)
        _ = py_call(fitted.append, clone)
        _ = py_call(contexts.append, clone_context)
    var named = Python.dict()
    for u in range(bases):  # driver-loop: over the nested members
        py_setitem(named, active.names[u], fitted[u])
    py_setattr(model, "estimators_", fitted)
    py_setattr(model, "named_estimators_", named)
    if _stack(kind):
        var methods = Python.list()
        var widths = Python.list()
        var width_words = _export_i64(plan, state, META_BLOCK_WIDTHS)
        for u in range(bases):  # driver-loop: over the stacked members
            var method = _stack_method(fields, active.ests[u], classifier, context)
            _ = py_call(methods.append, PythonObject(_stack_name(method)))
            _ = py_call(widths.append, PythonObject(width_words[u]))
        py_setattr(model, "stack_method_", methods)
        py_setattr(model, "_widths_", widths)
        var final = py_get(fields, "final_estimator")
        if final is None:
            final = py_get(py_get(context, "native_estimator_defaults"), "LogisticRegression" if classifier else "Ridge")
        var final_ = prep_estimator_clone(final, context)
        var final_plan = prep_estimator_plan(final_, context)
        var final_fields = py_call(final_plan["fields_get"], final_, py_type(final_))
        var final_context = final_plan["context"]
        _pack_into(final_, final_fields, final_context, b, nested[bases], handles[bases],
                   codes if classifier else PythonObject(None), jobs)
        _ = py_call(contexts.append, final_context)
        py_setattr(model, "final_estimator_", final_)
        if classifier:
            py_setattr(model, "classes_", labels)
            py_setattr(model, "_binary", PythonObject(k == 2))
    elif classifier:
        py_setattr(model, "classes_", labels)
        py_setattr(model, "n_classes_", PythonObject(len(labels)))
    py_setattr(model, "n_features_in_", PythonObject(features))
    py_setattr(model, "_fitted", PythonObject(True))
    # The nested fit notices, each with its nested member's own context, in
    # fit order (the door's member sets raised them after each publication).
    var owners = List[Int]()
    var notices = List[List[Float64]]()
    meta_state_notices(state, owners, notices)
    for i in range(len(owners)):  # driver-loop: over the nested notices (a few warnings each)
        if owners[i] < len(contexts):
            member_fit_notices_raise(notices[i], contexts[owners[i]])


struct _Imported(Movable):
    """THE IMPORT WORK-LIST's table (lane mf-meta3c): the nested fitted
    META estimators imported so far (by identity), each one's plan, its
    state (0 once handed to its parent) and its admission's pins."""
    var objs: PythonObject
    var plans: List[MemberPlan]
    var states: List[Int]
    var pins: PythonObject

    def __init__(out self) raises:
        self.objs = Python.list()
        self.plans = List[MemberPlan]()
        self.states = List[Int]()
        self.pins = Python.list()

    def find(self, est: PythonObject) raises -> Int:
        for i in range(len(self.plans)):  # driver-loop: over the imported nested meta-estimators (identity)
            if self.objs[i] is est and self.states[i] != 0:
                return i
        return -1

    def free_left(mut self):
        """Free every imported state no parent took. Never raises."""
        for i in range(len(self.states)):  # driver-loop: over the imported nested meta-estimators
            if self.states[i] != 0:
                var call = MemberCall(self.plans[i].route, self.states[i])
                member_free(self.plans[i], call)
                self.states[i] = 0


def _fitted_children(est: PythonObject, fields: PythonObject, kind: Int, b: PythonObject) raises -> PythonObject:
    """The fitted nested estimators an import of a fitted meta-estimator
    reads: `estimators_` (and a stack's `final_estimator_`), the calibrated
    pairs' members. A Bagging's are trees and forests: none."""
    var out = Python.list()
    if _bag(kind) or _ada(kind):
        return out
    if _search(kind):
        var best = py_get(fields, "best_estimator_")
        if best is not None:
            _ = py_call(out.append, best)
        return out
    if kind == META_CALIBRATED:
        var pairs = py_get(fields, "calibrated_classifiers_")
        if pairs is not None:
            for i in range(len(pairs)):  # driver-loop: over the pairs
                _ = py_call(out.append, pairs[i][0])
        return out
    var members = py_get(fields, "estimators_")
    if members is not None:
        var items = py_call(b.list, members)
        for i in range(len(items)):  # driver-loop: over the fitted members
            _ = py_call(out.append, items[i])
    if _stack(kind):
        var final = py_get(fields, "final_estimator_")
        if final is not None:
            _ = py_call(out.append, final)
    return out


def _nested_state(e: PythonObject, context: PythonObject, b: PythonObject, classes: Int, mut imported: _Imported,
                  mut nested: List[MemberPlan]) raises -> Int:
    """One fitted nested estimator's state for its parent's import: a leaf
    family's import, or a nested META state the work-list imported before
    (taken from `imported`). Its plan is appended to `nested`."""
    var e_plan = prep_estimator_plan(e, context)
    if e_plan is None or py_get(e_plan, "member") is None:
        raise Error("TypeError: mojolearn: a nested estimator has no member entry")
    if member_mark(e_plan, b).family == MEMBER_FAMILY_ENSEMBLE:
        var at = imported.find(e)
        if at < 0:
            raise Error("TypeError: mojolearn: a nested estimator cannot be imported")
        var handle = imported.states[at]
        imported.states[at] = 0
        nested.append(imported.plans[at].copy())
        return handle
    var e_fields = py_call(e_plan["fields_get"], e, py_type(e))
    var e_context = e_plan["context"]
    var e_owner = member_leaf_admit_serving(e, e_fields, e_plan, e_context, b, MEMBER_CAN_IMPORT)
    if not e_owner.published:
        raise Error("TypeError: mojolearn: a nested estimator cannot be imported")
    e_owner.plan.classes = classes
    var s = member_leaf_state(e, e_fields, e_context, b, e_owner.plan)
    nested.append(e_owner.plan.copy())
    return s


def ensemble_member_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan) raises -> Int:
    """UNPACK: a new META state holding the fitted meta-estimator `model`:
    its fitted nested estimators imported through their leaf families. The
    caller frees it.

    META ESTIMATORS NESTED AT ANY DEPTH (lane mf-meta3c): an explicit
    WORK-LIST, no recursion. A pre-order walk collects the fitted nested
    meta-estimators (`_fitted_children`); deepest first, each is admitted
    (`ensemble_member_admit`, its own work-list), given its class count (the
    length of its `classes_`) and imported (`_state_one`) into the table
    `_Imported`, from which its parent's import takes its state."""
    var imported = _Imported()
    var order = Python.list()
    var order_fields = Python.list()
    var order_context = Python.list()
    var order_kinds = List[Int]()
    var pending = Python.list(model)
    var pending_fields = Python.list(fields)
    var pending_context = Python.list(context)
    var pending_kinds: List[Int] = [Int(plan.params[0])]
    while len(pending_kinds) > 0:  # driver-loop: over the pending fitted meta-estimators (the work-list)
        var at = len(pending_kinds) - 1
        var est = pending[at]
        var est_fields = pending_fields[at]
        var est_context = pending_context[at]
        var est_kind = pending_kinds.pop()
        _ = py_call(pending.pop)
        _ = py_call(pending_fields.pop)
        _ = py_call(pending_context.pop)
        _ = py_call(order.append, est)
        _ = py_call(order_fields.append, est_fields)
        _ = py_call(order_context.append, est_context)
        order_kinds.append(est_kind)
        if len(order_kinds) > _NESTED_META_MAX:
            raise Error("TypeError: mojolearn: a nested estimator cannot be imported")
        var children = _fitted_children(est, est_fields, est_kind, b)
        for c in range(len(children)):  # driver-loop: over one fitted meta-estimator's nested estimators
            var child = children[c]
            var child_plan = prep_estimator_plan(child, est_context)
            if child_plan is None or py_get(child_plan, "member") is None:
                continue
            var mark = member_mark(child_plan, b)
            if mark.family != MEMBER_FAMILY_ENSEMBLE:
                continue
            _ = py_call(pending.append, child)
            _ = py_call(pending_fields.append, py_call(child_plan["fields_get"], child, py_type(child)))
            _ = py_call(pending_context.append, child_plan["context"])
            pending_kinds.append(mark.kind)
    var top = 0
    try:
        var i = len(order_kinds) - 1
        while i >= 1:  # driver-loop: over the fitted nested meta-estimators, deepest first (one import each)
            var owner = MemberOwner()
            owner.plan.family = MEMBER_FAMILY_ENSEMBLE
            owner.plan.kind = order_kinds[i]
            if not ensemble_member_admit(order[i], order_fields[i], order_kinds[i], order_context[i], b, owner):
                raise Error("TypeError: mojolearn: a nested estimator cannot be imported")
            owner.plan.member = member_resolve_family(MEMBER_FAMILY_ENSEMBLE, owner.plan.route)
            var labels = py_get(order_fields[i], "classes_")
            var word = Int(owner.plan.params[0])
            if _search(word):
                # A search's classes are its best estimator's (lane mf-search4).
                var best = py_get(order_fields[i], "best_estimator_")
                labels = py_get(py_fields(best), "classes_") if best is not None else PythonObject(None)
            owner.plan.classes = len(labels) if _classifier(word) and not _multi(word) and labels is not None else 0
            var handle = _state_one(order[i], order_fields[i], order_context[i], b, owner.plan, imported)
            _ = py_call(imported.objs.append, order[i])
            imported.plans.append(owner.plan.copy())
            imported.states.append(handle)
            _ = py_call(imported.pins.append, owner.pins)
            i -= 1
        top = _state_one(model, fields, context, b, plan, imported)
    finally:
        imported.free_left()  # states no parent took (an error, or a fitted estimator held twice)
        _ = len(imported.pins)  # the nested admissions' plan entries lived through every NEW above
    return top


def _state_one(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, mut imported: _Imported) raises -> Int:
    """One fitted meta-estimator's import (the body of
    `ensemble_member_state`'s work-list): its nested META states from
    `imported`."""
    var kind = Int(plan.params[0])
    if _ada(kind):
        return _ada_state(fields, context, b, plan)
    if _search(kind):
        return _search_state(fields, context, b, plan, imported)
    if _multi(kind):
        return _multi_state(fields, context, b, plan, imported)
    if _unit(kind):
        return _unit_state(model, fields, context, b, plan, imported)
    var classifier = _classifier(kind)
    var done = py_get(fields, "_fitted")
    var members = py_get(fields, "estimators_")
    if done is None or not Bool(py=done) or members is None:
        raise Error("NotFittedError: this meta-estimator is not fitted yet")
    var ests = py_call(b.list, members)
    if _stack(kind):
        _ = py_call(ests.append, py_get(fields, "final_estimator_"))
    var call = member_call_of(plan)
    _ = member_call(plan.member, MEMBER_OP_NEW, call)
    var handles = List[Int64]()
    var nested = List[MemberPlan]()
    var ok = False
    try:
        for i in range(len(ests)):  # driver-loop: over the fitted nested estimators (one import each)
            var s = _nested_state(ests[i], context, b, plan.classes if classifier else 0, imported, nested)
            handles.append(Int64(s))
        var block = MemberCall(plan.route, call.state)
        block.block = META_BLOCK_NESTED
        block.x = MemberMatrix(Int(handles.unsafe_ptr()), 1, len(handles), MEMBER_I64)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, block)
        handles = List[Int64]()  # the META state owns them now
        if _stack(kind):
            var given = py_get(fields, "_widths_")
            var widths = List[Int64]()
            if given is not None:
                for j in range(len(given)):  # driver-loop: over the stacked members
                    widths.append(Int64(Int(py=given[j])))
            var w = MemberCall(plan.route, call.state)
            w.block = META_BLOCK_WIDTHS
            w.x = MemberMatrix(Int(widths.unsafe_ptr()), 1 if len(widths) > 0 else 0, len(widths), MEMBER_I64)
            _ = member_call(plan.member, MEMBER_OP_IMPORT, w)
            _ = len(widths)  # copied in by the call above
        var last = MemberCall(plan.route, call.state)
        last.block = 0
        last.flags = MEMBER_IMPORT_DONE
        last.classes = plan.classes
        _ = member_call(plan.member, MEMBER_OP_IMPORT, last)
        ok = True
    finally:
        if not ok:
            for i in range(len(handles)):  # driver-loop: over the nested states not handed over
                var free = MemberCall(nested[i].route, Int(handles[i]))
                member_free(nested[i], free)
            member_free(plan, call)
    return call.state


# ------------------------------------------------------------ the unit kinds (lane mf-meta3b)


def _truthy(value: PythonObject, b: PythonObject) raises -> Bool:
    """Python's truth of a parameter, as the door's `_truthy` reads it."""
    return Bool(py=py_call(b.bool, value))


def _count_words(value: PythonObject, b: PythonObject, mut words: List[Float64]) raises -> Bool:
    """`_BaggingBase._count`'s two readings of max_samples / max_features:
    an exact int (1.0, its value) or a fraction in (0, 1] (0.0, its value).
    False for anything else (the door raises its words)."""
    if py_type(value) is b.int:
        words.append(1.0)
        words.append(Float64(Int(py=value)))
        return True
    if py_type(value) is b.float:
        var f = Float64(py=value)
        if not (f > 0.0 and f <= 1.0):
            return False
        words.append(0.0)
        words.append(f)
        return True
    return False


def _may_be_batched(fields: PythonObject, base: PythonObject, context: PythonObject, b: PythonObject,
                    words: List[Float64]) raises -> Bool:
    """True when a fit may take the door's batched route (`_bagging_batched_ok`:
    a bag of best-splitter decision trees over every feature, with no weights
    and no out-of-bag score, is one forest fit; a measured winner, TOMBSTONE
    MOJOLEARN_TREES_FAST_HR2_BAGGING_MEMBERS). Whether it does depends on the
    fit's weights and X's shape: such a configuration carries the batched
    forest as a second plan and the kind decides per fit (lane mf-meta3c)."""
    var dt_c = py_get(context, "trees_dt_classifier")
    var dt_r = py_get(context, "trees_dt_regressor")
    if not (py_type(base) is dt_c) and not (py_type(base) is dt_r):
        return False
    var bf = py_fields(base)
    var splitter = py_get(bf, "splitter")
    if splitter is None or py_type(splitter) is not b.str or String(splitter) != "best":
        return False
    if py_get(bf, "max_leaf_nodes") is not None:
        return False
    if words[8] != 0.0 or words[7] != 0.0:  # oob_score, bootstrap_features
        return False
    # max_features: a fraction below 1 never resolves to every column
    if words[4] == 0.0 and words[5] < 1.0:
        return False
    # max_samples without bootstrap: a fraction below 1 never resolves to every row
    if words[6] == 0.0 and words[2] == 0.0 and words[3] < 1.0:
        return False
    return True


def _new_tree(context: PythonObject, key: String, b: PythonObject) raises -> PythonObject:
    """The door's default Bagging member `_new_member(context[key], {})`:
    the connected class's constructor state, assembled without running
    Python."""
    var cls = py_item(context, PythonObject(key))
    var member = py_call(b.object.__new__, cls)
    var fields = py_fields(member)
    _ = py_call(fields.update, py_item(py_item(context, PythonObject("trees_defaults")), cls))
    py_setitem(fields, PythonObject("_has_constructor_parameters"), PythonObject(True))
    return member


def _bag_template(fields: PythonObject, base: PythonObject, classifier: Bool, context: PythonObject,
                  b: PythonObject) raises -> PythonObject:
    """The Bagging members' template: the base (the default tree when None)
    cloned with the ensemble's `device` (`boost_member_device`), or None
    when the base names another device (the door raises its words)."""
    var est = base
    if est is None:
        est = _new_tree(context, "trees_dt_classifier" if classifier else "trees_dt_regressor", b)
    var device = py_get(fields, "device")
    var own = py_get(py_fields(est), "device")
    if own is not None and own is not device:
        if device is None or String(own) != String(device):
            return PythonObject(None)
    var clone = prep_estimator_clone(est, context)
    py_setitem(py_fields(clone), PythonObject("device"), device)
    return clone


def _batched_forest(fields: PythonObject, template: PythonObject, classifier: Bool, count: Int, seed: Int,
                    bootstrap: Bool, fraction: PythonObject, image_bins: Bool, bins: Int, context: PythonObject,
                    b: PythonObject) raises -> PythonObject:
    """THE BATCHED ROUTE's forest, built from the decision-tree base as the
    door's `_bagging_fit_batched` builds it (`_new_member(rf class, common)`):
    `fraction` its max_samples when bootstrap (None otherwise; the
    admission's placeholder 1.0, PACK's the fit's n_rows / n), and a
    regressor's n_bins the base's or, left at None (`image_bins`), `bins`
    (the image's count the fit read; the admission's placeholder 0 never
    reaches a fit: the kind writes the image's count into its slot words)."""
    var bf = py_fields(template)
    var common = Python.dict()
    py_setitem(common, PythonObject("n_estimators"), PythonObject(count))
    py_setitem(common, PythonObject("criterion"), py_get(bf, "criterion"))
    py_setitem(common, PythonObject("max_depth"), py_get(bf, "max_depth"))
    py_setitem(common, PythonObject("min_samples_split"), py_get(bf, "min_samples_split"))
    py_setitem(common, PythonObject("min_samples_leaf"), py_get(bf, "min_samples_leaf"))
    var mf = py_get(bf, "max_features")
    py_setitem(common, PythonObject("max_features"), PythonObject(1.0) if mf is None else mf)
    py_setitem(common, PythonObject("min_impurity_decrease"), py_get(bf, "min_impurity_decrease"))
    py_setitem(common, PythonObject("bootstrap"), PythonObject(bootstrap))
    py_setitem(common, PythonObject("max_samples"), fraction if bootstrap else PythonObject(None))
    py_setitem(common, PythonObject("random_state"), PythonObject(seed))
    if classifier or not image_bins:
        py_setitem(common, PythonObject("n_bins"), py_get(bf, "n_bins"))
    else:
        py_setitem(common, PythonObject("n_bins"), PythonObject(bins))
    py_setitem(common, PythonObject("numeric_mode"), py_get(bf, "numeric_mode"))
    py_setitem(common, PythonObject("device"), py_get(bf, "device"))
    var key = "trees_rf_classifier" if classifier else "trees_rf_regressor"
    if classifier:
        py_setitem(common, PythonObject("class_weight"), py_get(bf, "class_weight"))
    var cls = py_item(context, PythonObject(key))
    var forest = py_call(b.object.__new__, cls)
    var forest_fields = py_fields(forest)
    _ = py_call(forest_fields.update, py_item(py_item(context, PythonObject("trees_defaults")), cls))
    _ = py_call(forest_fields.update, common)
    py_setitem(forest_fields, PythonObject("_has_constructor_parameters"), PythonObject(True))
    return forest


def _admit_caller_rows(mut nested: _Nested, est: PythonObject, classifier: Bool, needs: Int, context: PythonObject,
                       b: PythonObject) raises -> Bool:
    """A tree, forest or extra-trees estimator whose class_weight row weights
    its META caller computes (lane mf-split4; as `_admit_ada` admits its
    base): `forest_member_admit_common(..., caller_rows=True)`, the
    kind-exact mask checked as `_admit_nested` checks it. False: not served
    (a random-splitter tree, another family, a refused configuration)."""
    var plan = prep_estimator_plan(est, context)
    if plan is None or py_get(plan, "member") is None:
        return False
    var family = member_mark(plan, b).family
    var prefix = String("dt_")
    if family == MEMBER_FAMILY_FOREST:
        prefix = String("rf_")
    elif family == MEMBER_FAMILY_EXTRA_TREES:
        prefix = String("et_")
    elif family != MEMBER_FAMILY_TREE:
        return False
    var sub = MemberOwner()
    sub.plan.family = family
    sub.plan.kind = 0 if classifier else 1
    var mask = 0
    try:
        if not forest_member_admit_common(est, sub.plan.kind, prefix, plan["context"], sub, True):
            return False
        sub.plan.member = member_resolve_family(family, sub.plan.route)
        if not sub.plan.member.has(needs):
            return False
        mask = member_kind_capabilities(sub.plan)
    except:
        return False  # its own refusal: the door raises the estimator's words
    if (mask & needs) != needs or ((mask & MEMBER_CAN_CLASSIFIER) != 0) != classifier:
        return False
    nested.plans.append(sub.plan.copy())
    nested.masks.append(mask)
    _ = py_call(nested.pins.append, sub.pins)
    return True


def _tree_cw_words(class_weight: PythonObject, labels: PythonObject, context: PythonObject, b: PythonObject,
                   mut words: List[Float64]) raises -> Bool:
    """UNPACK of a tree base's class_weight (lane mf-split4), appended as the
    kind's class-weight tail (core/ensemble_member.mojo META_CW_*): [mode,
    K, all-one, the K dict weights, the K named flags]. None: mode 0. 'balanced': mode 1 (the
    FIT computes the ratios from its codes). A dict: mode 2, its weights
    over `labels` (the labels of the codes the trees are fitted on, in code
    order) read with the forest door's own admission
    (`class_weight_dictionary_values[CLASS_WEIGHT_TREES]`, as
    `trees_class_weight_rows` reads it); K = -1 when no labels are known (a
    state for import or predict only: its FIT refuses). False (the door
    keeps the estimator and raises its words) for a value or a dict the
    forest door refuses."""
    if class_weight is None:
        words.append(0.0)
        words.append(0.0)
        words.append(0.0)
        return True
    if py_type(class_weight) is b.str:
        if String(class_weight) != "balanced":
            return False
        words.append(1.0)
        words.append(0.0)
        words.append(0.0)
        return True
    if labels is None or len(labels) == 0:
        words.append(2.0)
        words.append(-1.0)
        words.append(0.0)
        return True
    try:
        var inputs = class_weight_dictionary_values[CLASS_WEIGHT_TREES](class_weight, labels, context)
        words.append(2.0)
        words.append(Float64(len(inputs.values)))
        words.append(1.0 if inputs.all_one else 0.0)
        for c in range(len(inputs.values)):  # driver-loop: over the classes (one weight each, UNPACK once)
            words.append(inputs.values[c])
        # The K named flags (lane mf-split4b): 1.0 where the dict names the
        # class (a Bagging unit whose draw lacks a named class is refused in
        # the tree door's words, core/ensemble_member.mojo `_bag_class_rows`).
        for c in range(len(inputs.values)):  # driver-loop: over the classes (one membership test each, UNPACK once)
            words.append(1.0 if py_contains(class_weight, labels[c]) else 0.0)
    except:
        return False
    return True


def _cal_base(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    """CalibratedClassifierCV's base: its estimator, else a clone of the
    default LinearSVC (`_calibrated_fit`)."""
    var base = py_get(fields, "estimator")
    if base is None:
        var defaults = py_get(context, "native_estimator_defaults")
        if defaults is None or py_get(defaults, "LinearSVC") is None:
            return PythonObject(None)
        base = prep_estimator_clone(py_get(defaults, "LinearSVC"), context)
    return base


def _admit_unit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject, b: PythonObject,
                mut owner: MemberOwner, done: _Done, labels: PythonObject) raises -> Bool:
    """UNPACK of a unit kind (core/ensemble_member.mojo, THE UNIT KINDS):
    the base admitted once through the LEAF dispatch with the bits the
    door's operations need, the settings words read as the door's fit reads
    them. False (the door keeps it and raises its own words) for anything
    the kind does not take exactly as the door does."""
    var classifier = kind != META_BAG_REGRESSOR
    var name = _kind_name(kind)
    var nested = _Nested()
    var words = List[Float64]()
    var folds = 0
    var fold_train = List[List[Int32]]()
    var fold_test = List[List[Int32]]()
    var split = _Split()
    var base_needs = MEMBER_CAN_FIT | MEMBER_CAN_PREDICT
    if classifier:
        base_needs |= MEMBER_CAN_CLASSIFIER
    var jobs = 0
    try:
        jobs = native_unit_jobs(py_get(fields, "n_jobs"), name)
    except:
        return False
    if kind == META_OVR:
        var base = py_get(fields, "estimator")
        if base is None:
            return False
        var verbose = py_get(fields, "verbose")
        if verbose is not None and _truthy(verbose, b):
            return False
        var decision = prep_estimator_has(base, "decision_function", context)
        var needs = base_needs | (MEMBER_CAN_DECISION if decision else MEMBER_CAN_PREDICT_PROBA)
        if not _admit_nested(done, nested, base, True, needs, context, b, labels):
            return False
        words.append(1.0 if decision else 0.0)
    elif kind == META_CALIBRATED:
        var base = _cal_base(fields, context)
        if base is None:
            return False
        var method = py_get(fields, "method")
        var sigmoid = True
        if method is not None:
            if py_type(method) is not b.str:
                return False
            var m = String(method)
            if m == "isotonic":
                sigmoid = False
            elif m != "sigmoid":
                return False
        var ensemble = py_get(fields, "ensemble")
        var whole = True
        if ensemble is not None:
            if py_type(ensemble) is b.bool:
                whole = Bool(py=ensemble)
            elif py_type(ensemble) is b.int and (Int(py=ensemble) == 0 or Int(py=ensemble) == 1):
                whole = Int(py=ensemble) == 1
            else:
                return False
        # The cv as the door's `_splits` takes it (lane mf-split4b): an int;
        # a user's index list read ONCE into typed folds; one of OUR KFold /
        # StratifiedKFold splitters as typed words, its folds computed inside
        # each fit over that fit's rows (core/ensemble_member.mojo
        # `_split_folds`); a user's own splitter object refused here.
        var cv = py_get(fields, "cv")
        _refuse_cv_callback(cv, context, b, _CV_CALLBACK_WORDS, False)
        if cv is not None and (py_type(cv) is b.list or py_type(cv) is b.tuple):
            if not _user_folds(cv, b, fold_train, fold_test):
                return False
            folds = len(fold_test)
        elif cv is not None and py_type(cv) is not b.int:
            if not _split_words(cv, True, context, b, split):
                return False  # another of our splitters: the door's `_splits`
            folds = split.k
        else:
            folds = 5 if cv is None else _int_word(cv, b)
            if folds < 2:
                return False  # a bool or an int below 2: the door raises its words
        # The resident DeviceRows route (a measured winner) stays the door's.
        var plan = prep_estimator_plan(base, context)
        if plan is not None:
            var rows = py_get(plan, "device_rows")
            if rows is not None and _truthy(rows, b) and native_unit_device_workers(min(jobs, folds - 1), context) <= 1:
                return False
        var decision = prep_estimator_has(base, "decision_function", context)
        var needs = base_needs | (MEMBER_CAN_DECISION if decision else MEMBER_CAN_PREDICT_PROBA)
        if not _admit_nested(done, nested, base, True, needs, context, b, labels):
            return False
        words.append(1.0 if sigmoid else 0.0)
        words.append(1.0 if whole else 0.0)
        words.append(1.0 if decision else 0.0)
    else:
        var oob = py_get(fields, "oob_score")
        var bootstrap = py_get(fields, "bootstrap")
        var bootstrap_features = py_get(fields, "bootstrap_features")
        var warm = py_get(fields, "warm_start")
        var verbose = py_get(fields, "verbose")
        if oob is None or bootstrap is None or bootstrap_features is None:
            return False
        var oob_on = _truthy(oob, b)
        if oob_on and not _truthy(bootstrap, b):
            return False
        if (warm is not None and _truthy(warm, b)) or (verbose is not None and _truthy(verbose, b)):
            return False
        var count = _int_word(py_get(fields, "n_estimators"), b)
        if count < 1:
            return False
        var state = py_get(fields, "random_state")
        var seed = 0
        if state is not None:
            seed = _int_word(state, b)
            if seed < 0:
                return False
        words.append(Float64(count))
        words.append(Float64(seed))
        if not _count_words(py_get(fields, "max_samples"), b, words):
            return False
        if not _count_words(py_get(fields, "max_features"), b, words):
            return False
        words.append(1.0 if _truthy(bootstrap, b) else 0.0)
        words.append(1.0 if _truthy(bootstrap_features, b) else 0.0)
        words.append(1.0 if oob_on else 0.0)
        var template = _bag_template(fields, py_get(fields, "estimator"), classifier, context, b)
        if template is None:
            return False
        var needs = base_needs | (MEMBER_CAN_PREDICT_PROBA if classifier else 0)
        # A CLASS_WEIGHT ON THE TREE (lane mf-split4): the batched route's
        # forest carries it (`_bagging_fit_batched` copies it) and the kind
        # hands that forest its class_weight row weights (core/
        # ensemble_member.mojo `_tree_class_rows`, the forest door's own
        # rows), so both plans are admitted with the caller's rows
        # (`forest_member_admit_common(..., caller_rows=True)`). The table
        # is read over the codes the forest is fitted on (`labels`: the
        # Bagging's codes 0 .. k-1, `_child_labels`). Since lane mf-split4b
        # the per-member route takes it too: each unit's row weights over its
        # draw's own classes (core/ensemble_member.mojo `_bag_class_rows`).
        var weighted = classifier and py_get(py_fields(template), "class_weight") is not None
        var batched = _may_be_batched(fields, template, context, b, words)
        if weighted:
            if not _admit_caller_rows(nested, template, classifier, needs, context, b):
                return False
        elif not _admit_nested(done, nested, template, classifier, needs, context, b, labels):
            return False
        var family = nested.plans[0].family
        if family != MEMBER_FAMILY_TREE and family != MEMBER_FAMILY_FOREST and family != MEMBER_FAMILY_EXTRA_TREES:
            return False  # `_require_tree`: a DecisionTree*, RandomForest* or ExtraTrees* member only
        # THE BATCHED ROUTE (lane mf-meta3c): a configuration whose fit may
        # take it carries the batched forest as the entry's second plan; the
        # kind decides per fit, from the weights and X's shape, as the door.
        var forest = PythonObject(None)
        if batched:
            var image_bins = not classifier and py_get(py_fields(template), "n_bins") is None
            forest = _batched_forest(fields, template, classifier, count, seed, _truthy(bootstrap, b),
                                     PythonObject(1.0), image_bins, 0, context, b)
            if weighted:
                if not _admit_caller_rows(nested, forest, classifier, needs, context, b):
                    return False
            elif not _admit_nested(done, nested, forest, classifier, needs, context, b, labels):
                return False
            words.append(1.0)
            words.append(1.0 if image_bins else 0.0)
        else:
            words.append(0.0)
            words.append(0.0)
        # META_BAG_CW: the batched forest's class_weight tail.
        var cw = PythonObject(None)
        var cw_context = context
        if weighted:
            var owner_of = forest if batched else template  # the estimator whose fit door reads it
            cw = py_get(py_fields(owner_of), "class_weight")
            cw_context = prep_estimator_plan(owner_of, context)["context"]  # its fit door's context
        if not _tree_cw_words(cw, labels, cw_context, b, words):
            return False
    if not _finish(fields, context, b, owner, nested, kind, 0.0, folds, List[Float64](), List[Int](), words^,
                   fold_train^, fold_test^):
        return False
    _finish_split(owner, split)
    return True


def _export_view(plan: MemberPlan, state: Int, block: Int) raises -> MemberCall:
    var call = MemberCall(plan.route, state)
    call.block = block
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    return call^


def _codes_list(held: List[Int], k: Int, b: PythonObject) raises -> PythonObject:
    """A unit's labels for its PACK: its held ensemble codes, else 0 .. k-1."""
    if len(held) == 0:
        return py_call(b.list, py_call(b.range, PythonObject(k)))
    var out = Python.list()
    for c in range(len(held)):  # small-loop(a unit's held classes)
        _ = py_call(out.append, PythonObject(held[c]))
    return out


def _f64_array(words: _F64P, start: Int, count: Int, context: PythonObject) raises -> PythonObject:
    var buf = admitted_unfilled(List[Int](length=1, fill=count), "<f8", context)
    var dst = _F64P(unsafe_from_address=buf.addr)
    for i in range(count):  # small-loop(one copy of a fitted vector into its public Array)
        dst[i] = words[start + i]
    return buf.keep


def _i32_array(words: MutPointer[Int32, MutUntrackedOrigin], start: Int, count: Int,
               context: PythonObject) raises -> PythonObject:
    var buf = admitted_unfilled(List[Int](length=1, fill=count), "<i4", context)
    var dst = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=buf.addr)
    for i in range(count):  # small-loop(one copy of a fitted index vector into its public Array)
        dst[i] = words[start + i]
    return buf.keep


def _pack_member(est: PythonObject, context: PythonObject, b: PythonObject, plan: MemberPlan, state: Int,
                 labels: PythonObject, seed: Int, mut jobs: _PackJobs) raises -> PythonObject:
    """One unit's public estimator: a clone of `est` (its `random_state` the
    unit's seed when `seed` >= 0) with the unit's state packed into it."""
    var clone = prep_estimator_clone(est, context)
    if seed >= 0:
        py_setitem(py_fields(clone), PythonObject("random_state"), PythonObject(seed))
    var clone_plan = prep_estimator_plan(clone, context)
    var clone_fields = py_call(clone_plan["fields_get"], clone, py_type(clone))
    var clone_context = clone_plan["context"]
    _pack_into(clone, clone_fields, clone_context, b, plan, state, labels, jobs)
    return clone


def _unit_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
               plan: MemberPlan, state: Int, labels: PythonObject, features: Int, k: Int,
               mut jobs: _PackJobs) raises -> PythonObject:
    """PACK of a unit kind: the attributes the door's fit writes (module
    note). Answers the nested contexts, in unit order, for the notices."""
    var kind = Int(plan.params[0])
    var nested = meta_state_plans(state)
    var handles = _export_i64(plan, state, META_BLOCK_NESTED)
    if len(handles) != len(nested):
        raise Error("RuntimeError: mojolearn: the meta member state does not match its units")
    var contexts = Python.list()
    if kind == META_OVR:
        var base = py_get(fields, "estimator")
        var two = py_call(b.list, py_call(b.range, PythonObject(2)))
        var fitted = Python.list()
        for u in range(len(handles)):  # driver-loop: over the units (one clone and one pack each)
            var clone = _pack_member(base, context, b, nested[u], handles[u], two, -1, jobs)
            _ = py_call(fitted.append, clone)
            _ = py_call(contexts.append, prep_estimator_plan(clone, context)["context"])
        py_setattr(model, "classes_", labels)
        py_setattr(model, "estimators_", fitted)
    elif kind == META_CALIBRATED:
        var base = _cal_base(fields, context)
        var view = _export_view(plan, state, META_BLOCK_CAL)
        var w = _F64P(unsafe_from_address=view.dst.data)
        var count = view.dst.rows * view.dst.cols
        var pairs = Python.list()
        var at = 1
        for u in range(len(handles)):  # driver-loop: over the pairs (one clone, one pack and its calibrators each)
            if at >= count:
                raise Error("RuntimeError: mojolearn: the calibrated member's pair words are short")
            var h = Int(w[at])
            at += 1
            var held = List[Int]()
            for c in range(h):  # small-loop(a pair's held classes)
                held.append(Int(w[at + c]))
            at += h
            var clone = _pack_member(base, context, b, nested[u], handles[u], _codes_list(held, k, b), -1, jobs)
            _ = py_call(contexts.append, prep_estimator_plan(clone, context)["context"])
            var cals = Python.list()
            var n_cals = Int(w[at])
            at += 1
            for _ in range(n_cals):  # driver-loop: over the pair's calibrators
                var iso = w[at] != 0.0
                var a = w[at + 1]
                var bb = w[at + 2]
                var m = Int(w[at + 3])
                at += 4
                if iso:
                    _ = py_call(cals.append, Python.tuple(PythonObject("isotonic"), Python.tuple(
                        _f64_array(w, at, m, context), _f64_array(w, at + m, m, context), PythonObject(m))))
                else:
                    _ = py_call(cals.append, Python.tuple(PythonObject("sigmoid"),
                                                          Python.tuple(PythonObject(a), PythonObject(bb))))
                at += 2 * m
            _ = py_call(pairs.append, Python.tuple(clone, cals))
        py_setattr(model, "classes_", labels)
        py_setattr(model, "_cal_fast", PythonObject(None))
        py_setattr(model, "calibrated_classifiers_", pairs)
    else:
        var classifier = kind == META_BAG_CLASSIFIER
        var template = _bag_template(fields, py_get(fields, "estimator"), classifier, context, b)
        if template is None:
            raise Error("RuntimeError: mojolearn: the Bagging member's estimator changed since its admission")
        var state_seed = py_get(fields, "random_state")
        var seed = 0 if state_seed is None else Int(py=state_seed)
        var scalars = _export_view(plan, state, MEMBER_BLOCK_SCALARS)
        var sw = _F64P(unsafe_from_address=scalars.dst.data)
        if scalars.dst.rows * scalars.dst.cols >= 5 and sw[2] != 0.0:
            # THE BATCHED ROUTE (`_bagging_fit_batched`'s attributes): the
            # one forest, packed into the forest the door builds, beside the
            # unfitted base; every tree over every feature; no out-of-bag rows.
            var bootstrap = _truthy(py_get(fields, "bootstrap"), b)
            var count = Int(py=py_get(fields, "n_estimators"))
            var image_bins = not classifier and py_get(py_fields(template), "n_bins") is None
            var forest = _batched_forest(fields, template, classifier, count, seed, bootstrap, PythonObject(sw[3]),
                                         image_bins, Int(sw[4]), context, b)
            var forest_plan = prep_estimator_plan(forest, context)
            var forest_fields = py_call(forest_plan["fields_get"], forest, py_type(forest))
            var forest_context = forest_plan["context"]
            if classifier:
                member_leaf_pack_labeled(forest, forest_fields, forest_context, b, nested[0], handles[0],
                                         py_call(b.list, py_call(b.range, PythonObject(k))))
            else:
                member_leaf_pack(forest, forest_fields, forest_context, b, nested[0], handles[0])
            _ = py_call(contexts.append, forest_context)
            var base = py_get(fields, "estimator")
            if base is None:
                base = _new_tree(context, "trees_dt_classifier" if classifier else "trees_dt_regressor", b)
            var every = admitted_unfilled(List[Int](length=1, fill=features), "<i4", context)
            var ep = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=every.addr)
            for j in range(features):  # small-loop(the column numbers 0 .. d-1, one shared array)
                ep[j] = Int32(j)
            var trees = Int(py=py_get(py_fields(forest), "_n_trees"))
            var shared = Python.list()
            for _ in range(trees):  # driver-loop: over the member trees (one shared all-columns array)
                _ = py_call(shared.append, every.keep)
            py_setitem(fields, PythonObject("_batched"), Python.tuple(forest, base))
            py_setitem(fields, PythonObject("_estimators"), PythonObject(None))
            py_setitem(fields, PythonObject("estimators_features_"), shared)
            py_setitem(fields, PythonObject("_oob_rows"), Python.list())
            if classifier:
                py_setitem(fields, PythonObject("classes_"), labels)
                py_setitem(fields, PythonObject("n_classes_"), PythonObject(k))
            return contexts
        var oob_on = _truthy(py_get(fields, "oob_score"), b)
        var view = _export_view(plan, state, META_BLOCK_DRAWS)
        var w = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=view.dst.data)
        var count = view.dst.rows * view.dst.cols
        var members = Python.list()
        var features_ = Python.list()
        var oob_rows = Python.list()
        var at = 1
        for u in range(len(handles)):  # driver-loop: over the member estimators (one clone and one pack each)
            if at >= count:
                raise Error("RuntimeError: mojolearn: the Bagging member's draw words are short")
            var c = Int(w[at])
            at += 1
            if c == 0:
                var every = admitted_unfilled(List[Int](length=1, fill=features), "<i4", context)
                var dst = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=every.addr)
                for j in range(features):  # small-loop(the column numbers 0 .. d-1 of a member over every feature)
                    dst[j] = Int32(j)
                _ = py_call(features_.append, every.keep)
            else:
                _ = py_call(features_.append, _i32_array(w, at, c, context))
            at += c
            var h = Int(w[at])
            at += 1
            var held = List[Int]()
            for q in range(h):  # small-loop(a member's held classes)
                held.append(Int(w[at + q]))
            at += h
            var o = Int(w[at])
            at += 1
            if oob_on:
                _ = py_call(oob_rows.append, _i32_array(w, at, o, context))
            at += o
            var labels_u = _codes_list(held, k, b) if classifier else PythonObject(None)
            var clone = _pack_member(template, context, b, nested[u], handles[u], labels_u, _sub_seed(seed, u), jobs)
            _ = py_call(members.append, clone)
            _ = py_call(contexts.append, prep_estimator_plan(clone, context)["context"])
        py_setitem(fields, PythonObject("_estimators"), members)
        py_setitem(fields, PythonObject("_batched"), PythonObject(None))
        py_setitem(fields, PythonObject("estimators_features_"), features_)
        py_setitem(fields, PythonObject("_oob_rows"), oob_rows)
        if classifier:
            py_setitem(fields, PythonObject("classes_"), labels)
            py_setitem(fields, PythonObject("n_classes_"), PythonObject(k))
        if oob_on:
            var o = _export_view(plan, state, META_BLOCK_OOB)
            var ow = _F64P(unsafe_from_address=o.dst.data)
            var total = o.dst.rows * o.dst.cols
            var width = k if classifier else 1
            var n = (total - 1) // width
            if classifier:
                var acc = admitted_unfilled([n, k], "<f8", context)
                var dst = _F64P(unsafe_from_address=acc.addr)
                for i in range(n * k):  # small-loop(one copy of the out-of-bag decision function into its Array)
                    dst[i] = ow[i]
                py_setitem(fields, PythonObject("oob_decision_function_"), acc.keep)
            else:
                py_setitem(fields, PythonObject("oob_prediction_"), _f64_array(ow, 0, n, context))
            py_setitem(fields, PythonObject("oob_score_"), PythonObject(ow[total - 1]))
    return contexts


def _held_of(e: PythonObject, k: Int, b: PythonObject) raises -> List[Int]:
    """IMPORT: a fitted unit's classes as ensemble codes when it lacks some
    (its `classes_` are ints of 0 .. k-1), else empty."""
    var held = List[Int]()
    var classes = py_get(py_fields(e), "classes_")
    if classes is None:
        return held^
    var items = py_call(b.list, classes)
    if len(items) == k:
        return held^
    for c in range(len(items)):  # small-loop(a unit's classes)
        held.append(Int(py=py_call(b.int, items[c])))
    return held^


def _unit_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
                plan: MemberPlan, mut imported: _Imported) raises -> Int:
    """UNPACK of a fitted unit-kind estimator into a new META state: its
    fitted units imported through their leaf families, then its pairs'
    calibrators (Calibrated) or its members' feature subsets and classes
    (Bagging). The caller frees it."""
    var kind = Int(plan.params[0])
    var k = plan.classes
    var ests = PythonObject(None)
    var pairs = PythonObject(None)
    if kind == META_OVR:
        var done = py_get(fields, "_fitted")
        if done is None or not Bool(py=done):
            raise Error("NotFittedError: this meta-estimator is not fitted yet")
        ests = py_call(b.list, py_get(fields, "estimators_"))
    elif kind == META_CALIBRATED:
        var done = py_get(fields, "_fitted")
        pairs = py_get(fields, "calibrated_classifiers_")
        if done is None or not Bool(py=done) or pairs is None:
            raise Error("NotFittedError: this meta-estimator is not fitted yet")
        ests = Python.list()
        for i in range(len(pairs)):  # driver-loop: over the pairs
            _ = py_call(ests.append, pairs[i][0])
    else:
        var batched = py_get(fields, "_batched")
        if batched is not None:
            # THE BATCHED ROUTE: its one forest is the state's one unit.
            var forest = batched[0]
            ests = Python.list(forest)
        else:
            var members = py_get(fields, "_estimators")
            if members is None:
                raise Error("NotFittedError: this meta-estimator is not fitted yet")
            ests = py_call(b.list, members)
    var call = member_call_of(plan)
    _ = member_call(plan.member, MEMBER_OP_NEW, call)
    var handles = List[Int64]()
    var nested = List[MemberPlan]()
    var helds = List[List[Int]]()
    var ok = False
    try:
        for i in range(len(ests)):  # driver-loop: over the fitted units (one import each)
            var e = ests[i]
            var held = List[Int]()
            var classes = 0
            if kind == META_OVR:
                classes = 2
            elif kind != META_BAG_REGRESSOR:
                held = _held_of(e, k, b)
                classes = len(held) if len(held) > 0 else k
            var s = _nested_state(e, context, b, classes, imported, nested)
            handles.append(Int64(s))
            helds.append(held^)
        var block = MemberCall(plan.route, call.state)
        block.block = META_BLOCK_NESTED
        block.x = MemberMatrix(Int(handles.unsafe_ptr()), 1, len(handles), MEMBER_I64)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, block)
        handles = List[Int64]()  # the META state owns them now
        if kind == META_CALIBRATED:
            var w = List[Float64]()
            w.append(Float64(len(pairs)))
            for i in range(len(pairs)):  # driver-loop: over the pairs (their held classes and calibrators)
                w.append(Float64(len(helds[i])))
                for c in range(len(helds[i])):  # small-loop(a pair's held classes)
                    w.append(Float64(helds[i][c]))
                var fitted = pairs[i][1]
                w.append(Float64(len(fitted)))
                for j in range(len(fitted)):  # driver-loop: over the pair's calibrators
                    var par = fitted[j][1]
                    if String(fitted[j][0]) == "sigmoid":
                        w.append(0.0)
                        w.append(Float64(py=par[0]))
                        w.append(Float64(py=par[1]))
                        w.append(0.0)
                    else:
                        var m = Int(py=par[2])
                        w.append(1.0)
                        w.append(0.0)
                        w.append(0.0)
                        w.append(Float64(m))
                        var kx = _F64P(unsafe_from_address=_addr(par[0]))
                        var ky = _F64P(unsafe_from_address=_addr(par[1]))
                        for q in range(m):  # small-loop(one calibrator's knots)
                            w.append(kx[q])
                        for q in range(m):  # small-loop(one calibrator's knots)
                            w.append(ky[q])
                        _ = par  # the knots are read through the copies above
            var cw = MemberCall(plan.route, call.state)
            cw.block = META_BLOCK_CAL
            cw.x = MemberMatrix(Int(w.unsafe_ptr()), 1, len(w), MEMBER_F64)
            _ = member_call(plan.member, MEMBER_OP_IMPORT, cw)
            _ = len(w)  # copied in by the call above
        elif _bag(kind) and py_get(fields, "_batched") is not None:
            var mark = MemberCall(plan.route, call.state)
            mark.block = META_BLOCK_BATCHED
            var one: List[Int32] = [Int32(1)]
            mark.x = MemberMatrix(Int(one.unsafe_ptr()), 1, 1, MEMBER_I32)
            _ = member_call(plan.member, MEMBER_OP_IMPORT, mark)
            _ = len(one)  # read by the call above
        elif _bag(kind):
            var feats = py_get(fields, "estimators_features_")
            var w = List[Int32]()
            w.append(Int32(len(ests)))
            for i in range(len(ests)):  # driver-loop: over the member estimators (their feature subsets and classes)
                var cols = feats[i]
                var c = Int(py=cols.shape[0])
                w.append(Int32(c))
                var words = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=_addr(cols))
                for j in range(c):  # small-loop(a member's feature subset)
                    w.append(words[j])
                _ = cols  # read through the copy above
                w.append(Int32(len(helds[i])))
                for q in range(len(helds[i])):  # small-loop(a member's held classes)
                    w.append(Int32(helds[i][q]))
                w.append(Int32(0))  # no out-of-bag rows: a predict reads none
            var dw = MemberCall(plan.route, call.state)
            dw.block = META_BLOCK_DRAWS
            dw.x = MemberMatrix(Int(w.unsafe_ptr()), 1, len(w), MEMBER_I32)
            _ = member_call(plan.member, MEMBER_OP_IMPORT, dw)
            _ = len(w)  # copied in by the call above
        var last = MemberCall(plan.route, call.state)
        last.block = 0
        last.flags = MEMBER_IMPORT_DONE
        last.classes = plan.classes
        _ = member_call(plan.member, MEMBER_OP_IMPORT, last)
        ok = True
    finally:
        if not ok:
            for i in range(len(handles)):  # driver-loop: over the nested states not handed over
                var free = MemberCall(nested[i].route, Int(handles[i]))
                member_free(nested[i], free)
            member_free(plan, call)
    return call.state


# ------------------------------------------------------------ the AdaBoost kind (lane mf-ada4)


def _mode_of(fields: PythonObject, context: PythonObject, b: PythonObject) raises -> String:
    """The tier an estimator's operations bind (`_mode` / `_member_mode`):
    its `numeric_mode`, else the process default; "" when neither is a
    known tier (the door raises its words)."""
    var mode = py_get(fields, "numeric_mode")
    if mode is None:
        var backend = py_get(context, "backend_fields")
        if backend is None:
            return String("")
        mode = py_get(backend, "_DEFAULT_MODE")
    if mode is None or py_type(mode) is not b.str:
        return String("")
    var word = String(mode)
    if word != "fast" and word != "identical":
        return String("")
    return word


def _ada_switches(mode: String, context: PythonObject, b: PythonObject) raises -> Int:
    """The x_trees image's fast switches (`_switches(xt)`: `x_trees_fast_switches`, 0 when the binary has none)."""
    var bindings = py_get(context, "trees_bindings")
    if bindings is None:
        return 0
    var tables = py_get(bindings, PythonObject(mode))
    if tables is None:
        return 0
    var table = py_get(tables, "_mojolearn_x_trees")
    if table is None or py_type(table) is b.str:
        return 0
    var query = py_get(table, "x_trees_fast_switches")
    if query is None:
        return 0
    return Int(py=py_call(query))


def _ada_base(fields: PythonObject, classifier: Bool, context: PythonObject, b: PythonObject) raises -> PythonObject:
    """AdaBoost's base: its estimator, else the door's default member
    `DecisionTreeClassifier(max_depth=1)` / `DecisionTreeRegressor(max_depth=3)`
    (`_new_member(cls, {"max_depth": ...})`)."""
    var base = py_get(fields, "estimator")
    if base is not None:
        return base
    base = _new_tree(context, "trees_dt_classifier" if classifier else "trees_dt_regressor", b)
    py_setitem(py_fields(base), PythonObject("max_depth"), PythonObject(1 if classifier else 3))
    return base


def _ada_template(fields: PythonObject, classifier: Bool, context: PythonObject, b: PythonObject) raises -> PythonObject:
    """The rounds' template (`template = forest_get_params(base)` with the
    ensemble's `device`, `boost_member_device`): None when the base names
    another device (the door raises its words)."""
    return _bag_template(fields, _ada_base(fields, classifier, context, b), classifier, context, b)


def _admit_ada(fields: PythonObject, kind: Int, context: PythonObject, b: PythonObject, labels: PythonObject,
               mut owner: MemberOwner) raises -> Bool:
    """UNPACK of THE ADABOOST KIND (core/ensemble_member.mojo): the door's
    `ada_fit` UNPACK minus X and y, read once per admission. `_ada_validate`
    as a gate (a value it refuses keeps the door, which raises its words);
    the base's first round clone (`_sub_seed(seed, 0)`) admitted ONCE as
    its native forest member with the caller's target checks
    (`forest_member_admit_common(..., caller_rows=True)`, the door's
    `_ada_member_admit`), whose words carry the configuration every round
    fits (they differ only in the seed); the session route words of a
    best-splitter DecisionTree base (`_ada_native_base`: the forest entry is
    the member's route, the share word `_ada_session_choice` /
    `_member_session`). False (the door keeps the estimator) for: an
    algorithm, loss, n_estimators, learning_rate or random_state the door
    refuses; a base that is not a tree, forest or extra-trees member or
    names another device; a DecisionTree splitter='random' base (its inner
    forest is the door's); a class_weight value the forest door refuses.
    A class_weight on the base (lane mf-split4) is the kind's class-weight
    tail (`_tree_cw_words` over `labels`, the codes the door's rows read):
    its FIT hands `ada_fit_run` the rows, as the door does."""
    var classifier = kind == META_ADA_CLASSIFIER
    var loss_code = 0
    if classifier:
        var algorithm = py_get(fields, "algorithm")
        if algorithm is None or py_type(algorithm) is not b.str:
            return False
        var name = String(algorithm)
        if name != "SAMME" and name != "deprecated":
            return False
    else:
        var losses = py_get(context, "trees_ada_losses")
        var loss = py_get(fields, "loss")
        if losses is None or loss is None or py_type(loss) is not b.str or not py_contains(losses, loss):
            return False
        loss_code = Int(py=py_item(losses, loss))
    var count = _int_word(py_get(fields, "n_estimators"), b)
    var given_rate = py_get(fields, "learning_rate")
    if given_rate is None or (py_type(given_rate) is not b.float and py_type(given_rate) is not b.int):
        return False
    var rate = Float64(py=py_call(b.float, given_rate))
    try:
        boost_check_n_estimators(count)
        boost_check_learning_rate(rate)
    except:
        return False
    var state = py_get(fields, "random_state")
    var seed = 0
    if state is not None:
        seed = _int_word(state, b)
        if seed < 0:
            return False
    var template = _ada_template(fields, classifier, context, b)
    if template is None:
        return False
    var template_plan = prep_estimator_plan(template, context)
    if template_plan is None or py_get(template_plan, "member") is None:
        return False
    var family = member_mark(template_plan, b).family
    var prefix = String("dt_")
    if family == MEMBER_FAMILY_FOREST:
        prefix = String("rf_")
    elif family == MEMBER_FAMILY_EXTRA_TREES:
        prefix = String("et_")
    elif family != MEMBER_FAMILY_TREE:
        return False  # `_require_tree`: a DecisionTree*, RandomForest* or ExtraTrees* base only
    # The first round's member (`_validated_clone(base, template, _sub_seed(seed, 0))`).
    var probe = prep_estimator_clone(template, context)
    var probe_fields = py_fields(probe)
    if py_contains(probe_fields, "random_state"):
        py_setitem(probe_fields, PythonObject("random_state"), PythonObject(_sub_seed(seed, 0)))
    var probe_context = template_plan["context"]
    # A DecisionTree splitter='random' base (lane mf-split4; the door's
    # `_ada_member_admit`): its rounds fit the tree's inner one-tree
    # ExtraTrees forest (`forest_dt_random_inner`), admitted as an
    # extra-trees member; PACK packs each round into a clone's inner forest
    # and the clone adopts it (`_ada_pack`), IMPORT reads `_random_inner`.
    var admitted = probe
    var dt_random = False
    if family == MEMBER_FAMILY_TREE:
        var splitter = py_get(probe_fields, "splitter")
        if splitter is not None and py_type(splitter) is b.str and String(splitter) == "random":
            dt_random = True
            admitted = forest_dt_random_inner(probe, forest_kind(probe, probe_context), probe_context)
            family = MEMBER_FAMILY_EXTRA_TREES
            prefix = String("et_")
    var sub = MemberOwner()
    sub.plan.family = family
    sub.plan.kind = 0 if classifier else 1
    if not forest_member_admit_common(admitted, sub.plan.kind, prefix, probe_context, sub, True):
        return False  # another kind
    sub.plan.member = member_resolve_family(family, sub.plan.route)
    sub.published = True
    # The configuration the rounds' checks read (the door's `forest_config`
    # of the refreshed first member, on this scratch clone; a random-splitter
    # tree's: its inner forest's, as the door's `_ada_members`).
    var run = trees_route_context(admitted, probe_context)
    var fk = forest_kind(admitted, run)
    var refreshed = forest_refresh(admitted, fk, run)
    var config = forest_config(refreshed, py_item(refreshed, PythonObject("_cfg")), fk, not classifier, run)
    # The class_weight the base's own fit door reads (the door's `_ada_native`
    # / `_ada_members`): ExtraTrees' from its derived configuration,
    # RandomForest's / DecisionTree's field.
    var class_weight = PythonObject(None)
    if classifier:
        var cfg = py_get(refreshed, "_cfg")
        if fk.startswith("et_"):
            if cfg is not None and py_type(cfg) is b.dict:
                class_weight = py_get(cfg, "class_weight")
        else:
            class_weight = py_get(refreshed, "class_weight")
    var mask = 0
    try:
        mask = member_kind_capabilities(sub.plan)
    except:
        return False
    var needs = MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT
    if classifier:
        needs |= MEMBER_CAN_CLASSIFIER | MEMBER_CAN_SAMPLE_WEIGHT
    if (mask & needs) != needs or ((mask & MEMBER_CAN_CLASSIFIER) != 0) != classifier:
        return False
    # `_ada_native_base`: an exact best-splitter DecisionTree (squared error
    # for the regressor) fits through the data session rounds.
    var tf = py_fields(template)
    var splitter = py_get(tf, "splitter")
    var best = splitter is not None and py_type(splitter) is b.str and String(splitter) == "best"
    var native = False
    if classifier:
        native = py_type(template) is py_get(context, "trees_dt_classifier") and best
    elif py_type(template) is py_get(context, "trees_dt_regressor") and best:
        var crit = py_get(tf, "criterion")
        native = crit is not None and py_type(crit) is b.str and (String(crit) == "squared_error"
                                                                   or String(crit) == "mse")
    var squared = Int(py=py_item(py_item(py_item(context, PythonObject("trees_rf_tables")), PythonObject("reg_criteria")),
                                 PythonObject("squared_error")))
    var ada_mode = _mode_of(fields, context, b)
    var member_mode = _mode_of(tf, context, b)
    if ada_mode == "" or member_mode == "":
        return False
    var share = boost_ada_session_choice(ada_mode, _ada_switches(ada_mode, context, b)) == "share" and member_mode == "fast"
    var words = List[Float64](capacity=11)
    words.append(Float64(count))                 # META_ADA_COUNT
    words.append(Float64(seed))                  # META_ADA_SEED
    words.append(rate)                           # META_ADA_RATE
    words.append(Float64(loss_code))             # META_ADA_LOSS
    words.append(1.0 if native else 0.0)         # META_ADA_NATIVE
    words.append(1.0 if share else 0.0)          # META_ADA_SHARE
    words.append(Float64(squared))               # META_ADA_SQUARED
    words.append(Float64(config.target_code))    # META_ADA_TARGET
    words.append(Float64(config.poisson))        # META_ADA_POISSON
    words.append(Float64(config.gamma))          # META_ADA_GAMMA
    words.append(Float64(config.inverse_gaussian))  # META_ADA_INVERSE_GAUSSIAN
    words.append(1.0 if dt_random else 0.0)      # META_ADA_DT_RANDOM
    if not _tree_cw_words(class_weight, labels, probe_context, b, words):  # META_CW_* after META_ADA_WORDS
        return False
    var nested = _Nested()
    nested.plans.append(sub.plan.copy())
    nested.masks.append(mask)
    _ = py_call(nested.pins.append, sub.pins)
    return _finish(fields, context, b, owner, nested, kind, 0.0, 0, List[Float64](), List[Int](), words^,
                   List[List[Int32]](), List[List[Int32]]())


def _ada_pack(fields: PythonObject, context: PythonObject, b: PythonObject, plan: MemberPlan, state: Int,
              labels: PythonObject, features: Int, k: Int, mut jobs: _PackJobs) raises:
    """PACK of THE ADABOOST KIND: the attributes the door's `ada_fit` writes
    (`_ada_pack` / `_ada_member_pack`): the rounds' fit notices in the
    estimator's context, then one member per kept round, in round order
    (member j a clone of the template with `_sub_seed(seed, j)`, its state
    packed into it through its family's PACK, labelled with the ensemble's
    codes 0 .. k-1 as its fit on the codes stores them, its
    `inference_engine_`), `estimators_`, `estimator_weights_`,
    `estimator_errors_`, `classes_` / `n_classes_` (classifier) and
    `n_features_in_`."""
    var kind = Int(plan.params[0])
    var classifier = kind == META_ADA_CLASSIFIER
    var nested = meta_state_plans(state)
    var handles = _export_i64(plan, state, META_BLOCK_NESTED)
    if len(handles) != len(nested):
        raise Error("RuntimeError: mojolearn: the AdaBoost member state does not match its rounds")
    var view = _export_view(plan, state, META_BLOCK_ADA)
    var w = _F64P(unsafe_from_address=view.dst.data)
    var count = view.dst.rows * view.dst.cols
    if count < 2:
        raise Error("RuntimeError: mojolearn: the AdaBoost member's weight words are short")
    var na = Int(w[0])
    var ne = Int(w[1 + na])
    var alphas = Python.list()
    for i in range(na):  # small-loop(the kept rounds' weights)
        _ = py_call(alphas.append, PythonObject(w[1 + i]))
    var errors = Python.list()
    for i in range(ne):  # small-loop(the kept rounds' errors)
        _ = py_call(errors.append, PythonObject(w[2 + na + i]))
    var owners = List[Int]()
    var notices = List[List[Float64]]()
    meta_state_notices(state, owners, notices)
    for i in range(len(owners)):  # driver-loop: over the rounds' notices (a few warnings)
        member_fit_notices_raise(notices[i], context)
    var template = _ada_template(fields, classifier, context, b)
    if template is None:
        raise Error("RuntimeError: mojolearn: the AdaBoost member's estimator changed since its admission")
    var state_seed = py_get(fields, "random_state")
    var seed = 0 if state_seed is None else Int(py=state_seed)
    var codes = py_call(b.list, py_call(b.range, PythonObject(k))) if classifier else PythonObject(None)
    var members = Python.list()
    var tf = py_fields(template)
    var splitter = py_get(tf, "splitter")
    var dt_key = PythonObject("trees_dt_classifier") if classifier else PythonObject("trees_dt_regressor")
    var dt_random = (py_type(template) is py_get(context, dt_key)
                     and splitter is not None and py_type(splitter) is b.str and String(splitter) == "random")
    for j in range(len(handles)):  # driver-loop: over the kept boosting rounds (one clone and one pack each)
        var clone: PythonObject
        if dt_random:
            # The door's `_ada_member_pack` for a random-splitter tree: the
            # round packed into the clone's inner forest, which it adopts.
            clone = prep_estimator_clone(template, context)
            py_setitem(py_fields(clone), PythonObject("random_state"), PythonObject(_sub_seed(seed, j)))
            var clone_context = prep_estimator_plan(clone, context)["context"]
            var inner = forest_dt_random_inner(clone, forest_kind(clone, clone_context), clone_context)
            if classifier:
                member_labels_hold(py_fields(inner), codes)
            forest_member_pack_common(inner, clone_context, b, nested[j], handles[j])
            forest_dt_random_adopt(clone, inner)
        else:
            clone = _pack_member(template, context, b, nested[j], handles[j], codes, _sub_seed(seed, j), jobs)
        _report_engine(clone, context)
        _ = py_call(members.append, clone)
    py_setitem(fields, PythonObject("estimators_"), members)
    py_setitem(fields, PythonObject("estimator_weights_"), alphas)
    py_setitem(fields, PythonObject("estimator_errors_"), errors)
    if classifier:
        py_setitem(fields, PythonObject("classes_"), labels)
        py_setitem(fields, PythonObject("n_classes_"), PythonObject(k))
    py_setitem(fields, PythonObject("n_features_in_"), PythonObject(features))


def _ada_state(fields: PythonObject, context: PythonObject, b: PythonObject, plan: MemberPlan) raises -> Int:
    """UNPACK of a fitted AdaBoost into a new META state: each member's five
    model arrays imported into a state of the admitted base's member
    (`forest_member_state_common`), then its `estimator_weights_` and
    `estimator_errors_` (META_BLOCK_ADA). The caller frees it."""
    var kind = Int(plan.params[0])
    var classifier = kind == META_ADA_CLASSIFIER
    var members = py_get(fields, "estimators_")
    var alphas = py_get(fields, "estimator_weights_")
    var errors = py_get(fields, "estimator_errors_")
    if members is None or alphas is None or errors is None:
        raise Error("NotFittedError: this meta-estimator is not fitted yet")
    var ests = py_call(b.list, members)
    var call = member_call_of(plan)
    _ = member_call(plan.member, MEMBER_OP_NEW, call)
    var base = meta_state_plans(call.state)[0].copy()
    base.classes = plan.classes if classifier else 0
    var handles = List[Int64]()
    var ok = False
    try:
        for i in range(len(ests)):  # driver-loop: over the fitted round members (one import each)
            # A random-splitter tree's model arrays are its inner forest's
            # (`forest_dt_random_adopt` kept it as `_random_inner`).
            var inner = py_get(py_fields(ests[i]), "_random_inner")
            handles.append(Int64(forest_member_state_common(ests[i] if inner is None else inner, context, base)))
        var block = MemberCall(plan.route, call.state)
        block.block = META_BLOCK_NESTED
        block.x = MemberMatrix(Int(handles.unsafe_ptr()), 1, len(handles), MEMBER_I64)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, block)
        handles = List[Int64]()  # the META state owns them now
        var a = py_call(b.list, alphas)
        var e = py_call(b.list, errors)
        var w = List[Float64](capacity=2 + len(a) + len(e))
        w.append(Float64(len(a)))
        for i in range(len(a)):  # driver-loop: over the rounds' weights (one float each)
            w.append(Float64(py=py_call(b.float, a[i])))
        w.append(Float64(len(e)))
        for i in range(len(e)):  # driver-loop: over the rounds' errors (one float each)
            w.append(Float64(py=py_call(b.float, e[i])))
        var aw = MemberCall(plan.route, call.state)
        aw.block = META_BLOCK_ADA
        aw.x = MemberMatrix(Int(w.unsafe_ptr()), 1, len(w), MEMBER_F64)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, aw)
        _ = len(w)  # copied in by the call above
        var last = MemberCall(plan.route, call.state)
        last.block = 0
        last.flags = MEMBER_IMPORT_DONE
        last.classes = plan.classes
        _ = member_call(plan.member, MEMBER_OP_IMPORT, last)
        ok = True
    finally:
        if not ok:
            for i in range(len(handles)):  # driver-loop: over the round states not handed over
                var free = MemberCall(base.route, Int(handles[i]))
                member_free(base, free)
            member_free(plan, call)
    return call.state


# ------------------------------------------------------------ the search kind (lane mf-search4)


def _search_seed_word(fields: PythonObject, b: PythonObject) raises -> Int:
    """A RandomizedSearchCV's sampler seed for its admission and its PACK:
    its own exact non-negative int below 2^52 (the words are float64), or,
    for random_state=None, one fresh draw per admission (scikit-learn's
    sampler draws afresh per fit too). -1: a seed the words cannot hold (the
    door keeps the search)."""
    var state = py_get(fields, "random_state")
    if state is None:
        var raw = py_call(b.int.from_bytes, py_call(native_loaded_module("os").urandom, PythonObject(7)),
                          PythonObject("little"))
        return Int(py=raw) >> 4  # 52 random bits
    var word = _int_word(state, b)
    if word < 0 or word >= 4503599627370496:
        return -1
    return word


def _search_candidates(model: PythonObject, fields: PythonObject, seed: Int, warn: Bool,
                       context: PythonObject) raises -> PythonObject:
    """UNPACK (and the PACK's re-draw): the search's candidate parameter
    dicts, as its fit reads them (bindings/msel_candidates_door.mojo
    `msel_candidates`), a RandomizedSearchCV's drawn with `seed`."""
    var cls = py_type(model)
    if cls is msel_cls("GridSearchCV", context):
        return msel_grid_list(fields["param_grid"], context)
    var dists = msel_sampler_check(fields["param_distributions"], context, True)
    return msel_sampler_list(dists, fields["n_iter"], PythonObject(seed), context, True, warn)


def _search_prepare(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
                    prepared: PythonObject) raises -> PythonObject:
    """UNPACK, once per admission: (candidates, their clones, the seed word),
    kept in `prepared` by the search's identity so the work-list admits a
    META candidate clone before the search reads it; None: not served."""
    var key = py_call(b.id, model)
    var got = py_get(prepared, key)
    if got is not None:
        return got
    var base = py_get(fields, "estimator")
    if base is None:
        return PythonObject(None)
    var seed = -1
    if py_type(model) is not msel_cls("GridSearchCV", context):
        if py_type(model) is not msel_cls("RandomizedSearchCV", context):
            return PythonObject(None)
        seed = _search_seed_word(fields, b)
        if seed < 0:
            return PythonObject(None)
    var candidates = _search_candidates(model, fields, seed, True, context)
    var clones = Python.list()
    for c in range(len(candidates)):  # driver-loop: over the candidates (one native clone with its parameters each)
        var clone = msel_native_clone(base, context)
        _ = msel_native_set_params(clone, msel_clone_parameter(candidates[c], context), context)
        _ = py_call(clones.append, clone)
    var out = Python.tuple(candidates, clones, PythonObject(seed))
    py_setitem(prepared, key, out)
    return out


def _scoring_many(scoring: PythonObject, context: PythonObject, b: PythonObject) raises -> Bool:
    """True for a multi-metric `scoring` (a list, tuple or set of names, or
    one of the native mapping types), as bindings/msel_cv_door.mojo
    `msel_scorers` tells them apart. Lane mf-search4b item 5: these never
    reach `msel_get_scorer` (which refuses them as scorer objects)."""
    var kind = py_type(scoring)
    if kind is b.list or kind is b.tuple or kind is b.set:
        return True
    var mappings = context["mapping_types"]
    for i in range(len(mappings)):  # driver-loop: fixed native dict/OrderedDict/defaultdict types
        if kind is mappings[i]:
            return True
    return False


def _scorer_name(scorer: PythonObject, context: PythonObject, b: PythonObject) raises -> String:
    """UNPACK: the table name of ONE scorer value: a str is itself (an
    unknown name raises `msel_get_scorer`'s words); a scorer object is
    admitted by `msel_get_scorer` (a Python callback raises the door's
    words) and read back to the name the scorer table holds it under (the
    object `get_scorer(name)` answers). "": a native scorer no table name
    holds (make_scorer's own; the door scores it)."""
    var got = msel_get_scorer(scorer, context)
    if py_type(scorer) is b.str:
        return String(py=scorer)
    var table = context["family_msel_scorers"]
    var items = py_call(b.list, py_call(table.items))
    for i in range(len(items)):  # driver-loop: over the scorer table (a fixed table, once per admission)
        if items[i][1] is got:
            return String(py=items[i][0])
    return String("")


def _admit_search(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
                  mut owner: MemberOwner, done: _Done, prepared: PythonObject, labels: PythonObject) raises -> Bool:
    """UNPACK of THE SEARCH KIND (core/ensemble_member.mojo): every
    candidate (the base's clone with the candidate's parameters set) admitted
    as a member with those words, read as `msel_search_fit` reads the
    search's settings. Answers False (the door's fit keeps the search) for an
    int cv it cannot take (a splitter, an index list), a named scorer the
    typed RUN does not serve (core/msel_scorer_run.mojo) or several scorers
    (the kind's words carry one), refit other than True, the search's
    own `device`, and any candidate with no serving member. A Python scorer
    callback is refused in the door's words."""
    var scoring_in = py_get(fields, "scoring")
    var scoring = 0
    var scoring_name = String("")
    if scoring_in is not None:
        if _scoring_many(scoring_in, context, b):
            return False  # several scorers: the kind's words carry one
        # A named scorer, or a native scorer object of the table (read back
        # to its name): its code is read below, with the base's response
        # method (lane mf-metric4, core/msel_scorer_run.mojo). A Python
        # scorer callback raises the door's admission words here.
        scoring_name = _scorer_name(scoring_in, context, b)
        if len(scoring_name) == 0:
            return False  # a native scorer the table does not name (make_scorer's)
        scoring = -1
    var refit = py_get(fields, "refit")
    if refit is None or py_type(refit) is not b.bool or not Bool(py=refit):
        return False
    var device = py_get(fields, "device")
    if device is not None:
        return False
    var cv = py_get(fields, "cv")
    # A user's own splitter object: refused here in the search door's words
    # (lane mf-split4b; never called, `_refuse_cv_callback`).
    _refuse_cv_callback(cv, context, b, _SEARCH_CV_CALLBACK_WORDS, True)
    var folds = 5 if cv is None else _int_word(cv, b)
    if folds < 2:
        return False
    var error_score = py_get(fields, "error_score")
    var tolerate = 0.0
    var error_value = 0.0
    if error_score is not None and py_type(error_score) is b.str:
        if String(py=error_score) != "raise":
            return False
    elif error_score is not None and (py_type(error_score) is b.float or py_type(error_score) is b.int):
        tolerate = 1.0
        error_value = Float64(py=py_call(b.float, error_score))
    else:
        return False
    var train = py_get(fields, "return_train_score")
    var return_train = train is not None and _truthy(train, b)
    var got = _search_prepare(model, fields, context, b, prepared)
    if got is None:
        return False
    var clones = got[1]
    var seed = Int(py=got[2])
    if len(clones) == 0:
        return False  # the door raises "No fits were performed. ..."
    var base = py_get(fields, "estimator")
    var base_plan = prep_estimator_plan(base, context)
    if base_plan is None or py_get(base_plan, "member") is None:
        return False
    var classifier = False
    if member_mark(base_plan, b).family == MEMBER_FAMILY_ENSEMBLE:
        var at = done.find(clones[0])
        if at < 0:
            return False
        classifier = (done.masks[at] & MEMBER_CAN_CLASSIFIER) != 0
    else:
        classifier = Bool(py=base_plan["classifier"])
    # THE SCORER (lane mf-metric4): a name the typed search RUN serves
    # (core/msel_scorer_run.mojo `msel_scorer_code`), with the response its
    # scorer reads off the base's class (`msel_scorer_response`, the
    # scorer's own rule: 'roc_auc' takes decision_function when the class
    # has it); any other name keeps the door.
    var response = MEMBER_OP_PREDICT
    if scoring < 0:
        var scorer = msel_get_scorer(scoring_in, context)
        var method = String(py=msel_scorer_response(scorer, base_plan, context))
        scoring = msel_scorer_code(scoring_name, method == "decision_function")
        if scoring < 0:
            return False
        var row = msel_scorer_row(scoring)
        if msel_scorer_regression(row) == classifier:
            return False
        response = row.op
    var needs = MEMBER_CAN_FIT | MEMBER_CAN_PREDICT
    if classifier:
        needs |= MEMBER_CAN_CLASSIFIER
    if scoring == 0:
        needs |= MEMBER_CAN_SCORE
    elif response == MEMBER_OP_PREDICT_PROBA:
        needs |= MEMBER_CAN_PREDICT_PROBA
    elif response == MEMBER_OP_DECISION:
        needs |= MEMBER_CAN_DECISION
    var nested = _Nested()
    for c in range(len(clones)):  # driver-loop: over the candidates (admission, once each)
        if not _admit_nested(done, nested, clones[c], classifier, needs, context, b, labels):
            return False
    for c in range(len(nested.plans)):  # driver-loop: over the candidates (their predict's element type)
        var multi = (nested.masks[c] & MEMBER_CAN_MULTI_OUTPUT) != 0
        if multi and scoring != 0:
            return False  # a multi-output target is scored by the member itself
        if scoring != 0:
            var probe = nested.plans[c].copy()
            probe.classes = 2 if classifier else 0
            var labels_out = response == MEMBER_OP_PREDICT and classifier
            var dtype = member_output_dtype(probe, response, 1, 1, MEMBER_I32 if labels_out else MEMBER_F32)
            if labels_out and dtype != MEMBER_I32 and dtype != MEMBER_I64:
                return False  # a member whose predict is not class codes
            if not labels_out and dtype != MEMBER_F32 and dtype != MEMBER_F64:
                return False  # a response the scorer's metric does not read
    var words: List[Float64] = [Float64(scoring), tolerate, error_value, 1.0 if return_train else 0.0, Float64(seed)]
    var kind = META_SEARCH_CLASSIFIER if classifier else META_SEARCH_REGRESSOR
    # The META entry and its glue run on an x_trees image: the search's own
    # context is model selection's, so the image is the one the meta classes'
    # context names (`_finish` reads its bindings; not loaded: the door).
    var trees = py_call(native_loaded_module("sys").modules.get, PythonObject("mojolearn._expansion_trees"))
    if trees is None:
        return False
    var trees_context = py_get(py_call(b.vars, trees), "_TREES_CONTEXT")
    if trees_context is None:
        return False
    if not _finish(fields, trees_context, b, owner, nested, kind, 0.0, folds, List[Float64](), List[Int](), words^,
                   List[List[Int32]](), List[List[Int32]]()):
        return False
    if scoring != 0:
        # THE NATIVE METRIC RUN's route (bindings/arena_prog_run.mojo
        # `metric_inputs_run`): the default-mode x_metrics runner, its
        # connection pinned with the plan entry.
        var nk = metrics_keys()
        var prog_context = py_get(context, nk[].k_family_metrics_prog_context)
        if prog_context is None:
            return False
        var pin = metric_route_of(metric_route_entries(PythonObject(None), prog_context))
        meta_plan_metric(Int(owner.plan.params[1]), pin.route)
        _ = py_call(owner.pins.append, pin.provider)
        _ = py_call(owner.pins.append, pin.entries)
    return True


def _search_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
                 plan: MemberPlan, state: Int, labels: PythonObject, mut jobs: _PackJobs) raises:
    """PACK of THE SEARCH KIND: the attributes `msel_search_fit` writes
    (`cv_results_` with `params`, `param_<name>`, `split<i>_test_score`,
    `mean_test_score`, `std_test_score`, `rank_test_score` and
    `mean_train_score`; `n_splits_`, `multimetric_`, `best_index_`,
    `best_score_`, `best_params_`, `best_estimator_` (the best candidate's
    clone holding the refit state), `scorer_`), then the units' fit notices
    with the best estimator's context."""
    var nested = meta_state_plans(state)
    var handles = _export_i64(plan, state, META_BLOCK_NESTED)
    if len(nested) != 1 or len(handles) != 1:
        raise Error("RuntimeError: mojolearn: the search member state does not match its best estimator")
    var view = _export_view(plan, state, META_BLOCK_SEARCH)
    var words = _F64P(unsafe_from_address=view.dst.data)
    var count = view.dst.rows * view.dst.cols
    if count < 6:
        raise Error("RuntimeError: mojolearn: the search member state has no tables")
    var best = Int(words[0])
    var folds = Int(words[1])
    var best_score = words[2]
    var m = Int(words[3])
    var train = words[4] != 0.0
    var seed = Int(words[5])
    var candidates = _search_candidates(model, fields, seed, False, context)
    if len(candidates) != m or best < 0 or best >= m:
        raise Error("RuntimeError: mojolearn: the search's candidates changed since its admission")
    var at = 6
    var test = List[List[Float64]](capacity=m)
    for c in range(m):  # driver-loop: over the candidates (their fold scores)
        var row = List[Float64](capacity=folds)
        for f in range(folds):  # small-loop(one candidate's folds)
            row.append(words[at + c * folds + f])
        test.append(row^)
    at += m * folds
    var means = List[Float64](capacity=m)
    var stds = List[Float64](capacity=m)
    var ranks = List[Int](capacity=m)
    for c in range(m):  # driver-loop: over the candidates (their tables)
        means.append(words[at + c])
        stds.append(words[at + m + c])
        ranks.append(Int(words[at + 2 * m + c]))
    at += 3 * m
    var results = Python.dict()
    py_setitem(results, "params", candidates)
    var itertools = native_loaded_module("itertools")
    var keys = py_call(b.sorted, py_call(b.set, py_call(itertools.chain.from_iterable, candidates)))
    for i in range(len(keys)):  # driver-loop: over the candidates' parameter names
        var getter = py_call(native_loaded_module("operator").methodcaller, PythonObject("get"), keys[i])
        py_setitem(results, py_call(b.str.__add__, PythonObject("param_"), py_call(b.str, keys[i])),
                   py_call(b.list, py_call(b.map, getter, candidates)))
    for f in range(folds):  # driver-loop: over the folds (one column each)
        var column = List[Float64](capacity=m)
        for c in range(m):  # small-loop(one fold's candidates)
            column.append(test[c][f])
        py_setitem(results, PythonObject("split" + String(f) + "_test_score"), msel_f64_list(column, context))
    py_setitem(results, PythonObject("mean_test_score"), msel_f64_list(means, context))
    py_setitem(results, PythonObject("std_test_score"), msel_f64_list(stds, context))
    py_setitem(results, PythonObject("rank_test_score"), msel_i32_list(ranks, context))
    if train:
        var train_means = List[Float64](capacity=m)
        for c in range(m):  # driver-loop: over the candidates (their train means)
            train_means.append(words[at + m * folds + c])
        py_setitem(results, PythonObject("mean_train_score"), msel_f64_list(train_means, context))
    # The best estimator: the best candidate's clone holding the refit state.
    var base = py_get(fields, "estimator")
    var clone = msel_native_clone(base, context)
    _ = msel_native_set_params(clone, msel_clone_parameter(candidates[best], context), context)
    var clone_plan = prep_estimator_plan(clone, context)
    var clone_fields = py_call(clone_plan["fields_get"], clone, py_type(clone))
    var clone_context = clone_plan["context"]
    _pack_into(clone, clone_fields, clone_context, b, nested[0], handles[0], labels, jobs)
    var updates = Python.dict()
    py_setitem(updates, "cv_results_", results)
    py_setitem(updates, "n_splits_", PythonObject(folds))
    py_setitem(updates, "multimetric_", PythonObject(False))
    py_setitem(updates, "best_index_", PythonObject(best))
    py_setitem(updates, "best_score_", PythonObject(best_score))
    py_setitem(updates, "best_params_", candidates[best])
    py_setitem(updates, "best_estimator_", clone)
    py_setitem(updates, "scorer_", msel_get_scorer(py_get(fields, "scoring"), context))
    msel_search_update(fields, updates, context)
    var owners = List[Int]()
    var notices = List[List[Float64]]()
    meta_state_notices(state, owners, notices)
    for i in range(len(owners)):  # driver-loop: over the units' notices (a few warnings each)
        member_fit_notices_raise(notices[i], clone_context)


def _search_state(fields: PythonObject, context: PythonObject, b: PythonObject, plan: MemberPlan,
                  mut imported: _Imported) raises -> Int:
    """UNPACK of a FITTED search: a new search state holding its
    `best_estimator_` (imported through its family) and the best index."""
    var best = py_get(fields, "best_estimator_")
    var index = py_get(fields, "best_index_")
    if best is None or index is None:
        raise Error("NotFittedError: this search is not fitted yet")
    var call = member_call_of(plan)
    _ = member_call(plan.member, MEMBER_OP_NEW, call)
    var nested = List[MemberPlan]()
    var handles = List[Int64]()
    var ok = False
    try:
        handles.append(Int64(_nested_state(best, context, b, plan.classes, imported, nested)))
        var block = MemberCall(plan.route, call.state)
        block.block = META_BLOCK_NESTED
        block.x = MemberMatrix(Int(handles.unsafe_ptr()), 1, 1, MEMBER_I64)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, block)
        handles = List[Int64]()  # the search state owns it now
        var words: List[Float64] = [Float64(Int(py=index)), 0.0, 0.0, 0.0, 0.0, -1.0]
        var search = MemberCall(plan.route, call.state)
        search.block = META_BLOCK_SEARCH
        search.x = MemberMatrix(Int(words.unsafe_ptr()), 1, len(words), MEMBER_F64)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, search)
        _ = len(words)  # copied in by the call above
        var last = MemberCall(plan.route, call.state)
        last.flags = MEMBER_IMPORT_DONE
        last.classes = plan.classes
        _ = member_call(plan.member, MEMBER_OP_IMPORT, last)
        ok = True
    finally:
        if not ok:
            for i in range(len(handles)):  # driver-loop: over the nested state not handed over
                var free = MemberCall(nested[i].route, Int(handles[i]))
                member_free(nested[i], free)
            member_free(plan, call)
    return call.state


# ------------------------------------------------------------ the multi-output kind (lane mf-search4)


def _admit_multi(fields: PythonObject, kind: Int, context: PythonObject, b: PythonObject,
                 mut owner: MemberOwner, done: _Done) raises -> Bool:
    """UNPACK of THE MULTI-OUTPUT KIND: the base admitted once through the
    LEAF dispatch. False (the door keeps it) for a plain Ridge base of a
    MultiOutputRegressor (the door's one-program ridge route over every
    output, a measured winner kept on the door) and a base with no serving
    member."""
    var classifier = kind == META_MULTI_CLASSIFIER
    var base = py_get(fields, "estimator")
    if base is None:
        return False
    if not classifier:
        var defaults = py_get(context, "native_estimator_defaults")
        var ridge = py_get(defaults, "Ridge") if defaults is not None else PythonObject(None)
        if ridge is not None and py_type(base) is py_type(ridge):
            return False
    var needs = MEMBER_CAN_FIT | MEMBER_CAN_PREDICT
    if classifier:
        needs |= MEMBER_CAN_CLASSIFIER
    var nested = _Nested()
    # One base for every output, whose labels differ per output: a
    # class_weight dict stays pending (its FIT refuses; the door fits it).
    if not _admit_nested(done, nested, base, classifier, needs, context, b, PythonObject(None)):
        return False
    var words: List[Float64] = [0.0]
    return _finish(fields, context, b, owner, nested, kind, 0.0, 0, List[Float64](), List[Int](), words^,
                   List[List[Int32]](), List[List[Int32]]())


def _multi_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
                plan: MemberPlan, state: Int, labels: PythonObject, features: Int, mut jobs: _PackJobs) raises:
    """PACK of THE MULTI-OUTPUT KIND: the attributes the door's fit writes
    (`estimators_`, one clone per output holding its unit's state; a
    classifier's `classes_`, the list of each output's labels, the labels the
    caller held; `n_features_in_`, `_fitted`, and `_multi_ridge` None for
    the regressor), then the units' fit notices."""
    var kind = Int(plan.params[0])
    var classifier = kind == META_MULTI_CLASSIFIER
    var nested = meta_state_plans(state)
    var handles = _export_i64(plan, state, META_BLOCK_NESTED)
    if len(handles) != len(nested):
        raise Error("RuntimeError: mojolearn: the meta member state does not match its outputs")
    if classifier and len(labels) != len(nested):
        raise Error("RuntimeError: mojolearn: a MultiOutputClassifier member is packed without each output's labels")
    var base = py_get(fields, "estimator")
    var fitted = Python.list()
    var contexts = Python.list()
    for j in range(len(nested)):  # driver-loop: over the outputs (one clone and one pack each)
        var clone = prep_estimator_clone(base, context)
        var clone_plan = prep_estimator_plan(clone, context)
        var clone_fields = py_call(clone_plan["fields_get"], clone, py_type(clone))
        var clone_context = clone_plan["context"]
        _pack_into(clone, clone_fields, clone_context, b, nested[j], handles[j],
                   labels[j] if classifier else PythonObject(None), jobs)
        _ = py_call(fitted.append, clone)
        _ = py_call(contexts.append, clone_context)
    py_setattr(model, "estimators_", fitted)
    if classifier:
        py_setattr(model, "classes_", py_call(b.list, labels))
    else:
        py_setattr(model, "_multi_ridge", PythonObject(None))
    py_setattr(model, "n_features_in_", PythonObject(features))
    py_setattr(model, "_fitted", PythonObject(True))
    var owners = List[Int]()
    var notices = List[List[Float64]]()
    meta_state_notices(state, owners, notices)
    for i in range(len(owners)):  # driver-loop: over the nested notices (a few warnings each)
        if owners[i] < len(contexts):
            member_fit_notices_raise(notices[i], contexts[owners[i]])


def _multi_state(fields: PythonObject, context: PythonObject, b: PythonObject, plan: MemberPlan,
                 mut imported: _Imported) raises -> Int:
    """UNPACK of a FITTED MultiOutput estimator: its `estimators_` imported
    through their family, each output's class count from `classes_`."""
    var kind = Int(plan.params[0])
    var classifier = kind == META_MULTI_CLASSIFIER
    var done = py_get(fields, "_fitted")
    var members = py_get(fields, "estimators_")
    if done is None or not Bool(py=done) or members is None or py_get(fields, "_multi_ridge") is not None:
        raise Error("NotFittedError: this MultiOutput estimator is not fitted as one member per output")
    var ests = py_call(b.list, members)
    var counts = List[Int]()
    if classifier:
        var classes = py_get(fields, "classes_")
        if classes is None or len(classes) != len(ests):
            raise Error("NotFittedError: this MultiOutputClassifier is not fitted yet")
        for j in range(len(ests)):  # driver-loop: over the outputs (their class counts)
            counts.append(len(classes[j]))
    var call = member_call_of(plan)
    _ = member_call(plan.member, MEMBER_OP_NEW, call)
    var handles = List[Int64]()
    var nested = List[MemberPlan]()
    var ok = False
    try:
        for j in range(len(ests)):  # driver-loop: over the fitted outputs (one import each)
            handles.append(Int64(_nested_state(ests[j], context, b, counts[j] if classifier else 0, imported, nested)))
        var block = MemberCall(plan.route, call.state)
        block.block = META_BLOCK_NESTED
        block.x = MemberMatrix(Int(handles.unsafe_ptr()), 1, len(handles), MEMBER_I64)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, block)
        handles = List[Int64]()
        var last = MemberCall(plan.route, call.state)
        last.flags = MEMBER_IMPORT_DONE
        last.class_counts = Int(counts.unsafe_ptr()) if classifier else 0
        _ = member_call(plan.member, MEMBER_OP_IMPORT, last)
        _ = len(counts)  # read through the call above
        ok = True
    finally:
        if not ok:
            for i in range(len(handles)):  # driver-loop: over the nested states not handed over
                var free = MemberCall(nested[i].route, Int(handles[i]))
                member_free(nested[i], free)
            member_free(plan, call)
    return call.state
