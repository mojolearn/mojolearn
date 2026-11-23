# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the Ensemble family: the META member kind (lane
mf-meta3, 2026-11-23; RUN: core/ensemble_member.mojo). PUBLISHED for
VotingClassifier, VotingRegressor, StackingClassifier and StackingRegressor,
and (lane mf-meta3b, THE UNIT KINDS) OneVsRestClassifier,
CalibratedClassifierCV, BaggingClassifier and BaggingRegressor
(python/mojolearn/_expansion_trees.py `_mojolearn_member = (39, kind)`).
AdaBoost, MultiOutput and RandomTreesEmbedding carry no mark and keep their
doors. A unit kind answers False, and its door keeps the estimator, for:
a OneVsRest base whose kind lacks the scores its predict reads; a
Calibrated cv that is not an int (a splitter, an index list, 'prefit') or a
base that takes the resident DeviceRows route (a measured winner); a
Bagging configuration that may take the batched forest route (a measured
winner, decided per fit by the weights and X's shape), a base that is not a
tree, forest or extra-trees member, or one that names another device.

UNPACK (`ensemble_member_admit`, GIL held, once per admission): the
meta-estimator's parameters, read as its own fit reads them
(bindings/trees_public_meta_door.mojo `_voting_fit` / `_stacking_fit`);
every nested estimator admitted through the LEAF dispatch
(bindings/member_leaf_boundary_door.mojo: a meta of metas keeps its doors);
the nested plans minted once into the plan store and released by the
capsule kept in the owner's pins. Answers False (the caller keeps its own
path, whose doors raise the estimator's own words) for anything the native
kind does not take exactly as the door does: a nested estimator with no
serving member, a user cv SPLITTER (its folds need the fit's X and y and
the splitter's own warnings; an index list is read here into typed folds
since lane mf-meta3b), a non-boolean passthrough, weights of the wrong
length, an unknown `voting`. A classifier stacked with
`stack_method='predict'` is served since lane mf-meta3b (its predicted
codes are the meta features).

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
from bindings.member_owner_unpack import MemberOwner, member_labels_take
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
    META_BLOCK_CAL, META_BLOCK_DRAWS, META_BLOCK_OOB,
)
from core.member_families import MEMBER_FAMILY_TREE, MEMBER_FAMILY_FOREST, MEMBER_FAMILY_EXTRA_TREES
from xtrees.trees_boost_run import _sub_seed
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_call_of, member_free, member_kind_capabilities,
    MEMBER_OP_NEW, MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_I64, MEMBER_F64, MEMBER_IMPORT_DONE,
    MEMBER_CAN_FIT, MEMBER_CAN_PREDICT, MEMBER_CAN_PREDICT_PROBA, MEMBER_CAN_DECISION, MEMBER_CAN_CLASSIFIER,
    MEMBER_CAN_IMPORT, MEMBER_BLOCK_SCALARS,
)
from xtrees.trees_meta_vote_run import STACK_PREDICT, STACK_PROBA, STACK_DECISION

comptime _I64P = MutPointer[Int64, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]


def _stack(kind: Int) -> Bool:
    return kind == META_STACK_CLASSIFIER or kind == META_STACK_REGRESSOR


def _classifier(kind: Int) -> Bool:
    return (kind == META_VOTE_CLASSIFIER or kind == META_STACK_CLASSIFIER or kind == META_OVR
            or kind == META_CALIBRATED or kind == META_BAG_CLASSIFIER)


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


def _admit_nested(mut nested: _Nested, est: PythonObject, classifier: Bool, needs: Int, context: PythonObject,
                  b: PythonObject) raises -> Bool:
    """One nested estimator through the LEAF dispatch, with the kind-exact
    mask (one state minted, asked, released). False: not served."""
    var plan = prep_estimator_plan(est, context)
    if plan is None or py_get(plan, "member") is None:
        return False
    if Bool(py=plan["classifier"]) != classifier:
        return False
    var fields = py_call(plan["fields_get"], est, py_type(est))
    var owner = member_leaf_admit_serving(est, fields, plan, plan["context"], b, needs)
    if not owner.published:
        return False
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


def ensemble_member_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """UNPACK: fill `owner.plan.route`, `owner.plan.params` and `owner.pins`
    from the unfitted meta-estimator (module note). False: the caller keeps
    its own path."""
    if kind < META_VOTE_CLASSIFIER or kind > META_KIND_LAST:
        return False
    if _unit(kind):
        return _admit_unit(model, fields, kind, context, b, owner)
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
            if not _admit_nested(nested, active.ests[i], classifier, needs, context, b):
                return False
    else:
        var cv = py_get(fields, "cv")
        if cv is not None and (py_type(cv) is b.list or py_type(cv) is b.tuple):
            # A user's index list (lane mf-meta3b): its folds read ONCE into
            # typed int32 rows, checked against the fit's rows by the RUN as
            # the door's `_splits` checks them.
            if not _user_folds(cv, b, fold_train, fold_test):
                return False
            folds = len(fold_test)
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
            if not _admit_nested(nested, active.ests[i], classifier, base_needs | _stack_need(method), context, b):
                return False
        var final = py_get(fields, "final_estimator")
        if final is None:
            var defaults = py_get(context, "native_estimator_defaults")
            if defaults is None:
                return False
            final = py_get(defaults, "LogisticRegression" if classifier else "Ridge")
            if final is None:
                return False
        if not _admit_nested(nested, final, classifier, base_needs, context, b):
            return False
    return _finish(fields, context, b, owner, nested, kind, flag, folds, weights^, stack_kinds^, List[Float64](),
                   fold_train^, fold_test^)


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
                mut fold_test: List[List[Int32]]) -> Bool:
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
        return True
    except:
        return False


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


def ensemble_member_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, state: Int) raises:
    """PACK: write the fitted attributes of META state `state` into the
    meta-estimator `model`, as its own fit writes them (module note)."""
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
    if _unit(kind):
        # THE UNIT KINDS (lane mf-meta3b): the attributes the door's fit writes.
        var unit_contexts = _unit_pack(model, fields, context, b, plan, state, labels, features, k)
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
        if classifier:
            member_leaf_pack_labeled(clone, clone_fields, clone_context, b, nested[u], handles[u], codes)
        else:
            member_leaf_pack(clone, clone_fields, clone_context, b, nested[u], handles[u])
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
        if classifier:
            member_leaf_pack_labeled(final_, final_fields, final_context, b, nested[bases], handles[bases], codes)
        else:
            member_leaf_pack(final_, final_fields, final_context, b, nested[bases], handles[bases])
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


def ensemble_member_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan) raises -> Int:
    """UNPACK: a new META state holding the fitted meta-estimator `model`:
    its fitted nested estimators imported through their leaf families. The
    caller frees it."""
    var kind = Int(plan.params[0])
    if _unit(kind):
        return _unit_state(model, fields, context, b, plan)
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
            var e = ests[i]
            var e_plan = prep_estimator_plan(e, context)
            if e_plan is None or py_get(e_plan, "member") is None:
                raise Error("TypeError: mojolearn: a nested estimator has no member entry")
            var e_fields = py_call(e_plan["fields_get"], e, py_type(e))
            var e_context = e_plan["context"]
            var e_owner = member_leaf_admit_serving(e, e_fields, e_plan, e_context, b, MEMBER_CAN_IMPORT)
            if not e_owner.published:
                raise Error("TypeError: mojolearn: a nested estimator cannot be imported")
            e_owner.plan.classes = plan.classes if classifier else 0
            var s = member_leaf_state(e, e_fields, e_context, b, e_owner.plan)
            handles.append(Int64(s))
            nested.append(e_owner.plan.copy())
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
    fit's weights and X's shape, so such a configuration stays the door's."""
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
                mut owner: MemberOwner) raises -> Bool:
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
        if not _admit_nested(nested, base, True, needs, context, b):
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
        var cv = py_get(fields, "cv")
        folds = 5 if cv is None else _int_word(cv, b)
        if folds < 2:
            return False  # a splitter, an index list or 'prefit': the door's units
        # The resident DeviceRows route (a measured winner) stays the door's.
        var plan = prep_estimator_plan(base, context)
        if plan is not None:
            var rows = py_get(plan, "device_rows")
            if rows is not None and _truthy(rows, b) and native_unit_device_workers(min(jobs, folds - 1), context) <= 1:
                return False
        var decision = prep_estimator_has(base, "decision_function", context)
        var needs = base_needs | (MEMBER_CAN_DECISION if decision else MEMBER_CAN_PREDICT_PROBA)
        if not _admit_nested(nested, base, True, needs, context, b):
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
        if template is None or _may_be_batched(fields, template, context, b, words):
            return False
        var needs = base_needs | (MEMBER_CAN_PREDICT_PROBA if classifier else 0)
        if not _admit_nested(nested, template, classifier, needs, context, b):
            return False
        var family = nested.plans[0].family
        if family != MEMBER_FAMILY_TREE and family != MEMBER_FAMILY_FOREST and family != MEMBER_FAMILY_EXTRA_TREES:
            return False  # `_require_tree`: a DecisionTree*, RandomForest* or ExtraTrees* member only
    return _finish(fields, context, b, owner, nested, kind, 0.0, folds, List[Float64](), List[Int](), words^,
                   List[List[Int32]](), List[List[Int32]]())


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
                 labels: PythonObject, seed: Int) raises -> PythonObject:
    """One unit's public estimator: a clone of `est` (its `random_state` the
    unit's seed when `seed` >= 0) with the unit's state packed into it."""
    var clone = prep_estimator_clone(est, context)
    if seed >= 0:
        py_setitem(py_fields(clone), PythonObject("random_state"), PythonObject(seed))
    var clone_plan = prep_estimator_plan(clone, context)
    var clone_fields = py_call(clone_plan["fields_get"], clone, py_type(clone))
    var clone_context = clone_plan["context"]
    if labels is None:
        member_leaf_pack(clone, clone_fields, clone_context, b, plan, state)
    else:
        member_leaf_pack_labeled(clone, clone_fields, clone_context, b, plan, state, labels)
    return clone


def _unit_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
               plan: MemberPlan, state: Int, labels: PythonObject, features: Int, k: Int) raises -> PythonObject:
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
            var clone = _pack_member(base, context, b, nested[u], handles[u], two, -1)
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
            var clone = _pack_member(base, context, b, nested[u], handles[u], _codes_list(held, k, b), -1)
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
            var clone = _pack_member(template, context, b, nested[u], handles[u], labels_u, _sub_seed(seed, u))
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
                plan: MemberPlan) raises -> Int:
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
        if py_get(fields, "_batched") is not None:
            raise Error("TypeError: mojolearn: a batched Bagging fit has no member units to import")
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
            var e_plan = prep_estimator_plan(e, context)
            if e_plan is None or py_get(e_plan, "member") is None:
                raise Error("TypeError: mojolearn: a nested estimator has no member entry")
            var e_fields = py_call(e_plan["fields_get"], e, py_type(e))
            var e_context = e_plan["context"]
            var e_owner = member_leaf_admit_serving(e, e_fields, e_plan, e_context, b, MEMBER_CAN_IMPORT)
            if not e_owner.published:
                raise Error("TypeError: mojolearn: a nested estimator cannot be imported")
            var held = List[Int]()
            if kind == META_OVR:
                e_owner.plan.classes = 2
            elif kind == META_BAG_REGRESSOR:
                e_owner.plan.classes = 0
            else:
                held = _held_of(e, k, b)
                e_owner.plan.classes = len(held) if len(held) > 0 else k
            var s = member_leaf_state(e, e_fields, e_context, b, e_owner.plan)
            handles.append(Int64(s))
            nested.append(e_owner.plan.copy())
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
