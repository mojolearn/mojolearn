# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the Ensemble family: the META member kind (lane
mf-meta3, 2026-11-23; RUN: core/ensemble_member.mojo). PUBLISHED for
VotingClassifier, VotingRegressor, StackingClassifier and StackingRegressor
(python/mojolearn/_expansion_trees.py `_mojolearn_member = (39, kind)`).
The other classes of the family (Bagging, AdaBoost, MultiOutput, OneVsRest,
CalibratedClassifierCV, RandomTreesEmbedding) carry no mark yet and keep
their doors.

UNPACK (`ensemble_member_admit`, GIL held, once per admission): the
meta-estimator's parameters, read as its own fit reads them
(bindings/trees_public_meta_door.mojo `_voting_fit` / `_stacking_fit`);
every nested estimator admitted through the LEAF dispatch
(bindings/member_leaf_boundary_door.mojo: a meta of metas keeps its doors);
the nested plans minted once into the plan store and released by the
capsule kept in the owner's pins. Answers False (the caller keeps its own
path, whose doors raise the estimator's own words) for anything the native
kind does not take exactly as the door does: a nested estimator with no
serving member, a user cv splitter or index list, a classifier stacked with
`stack_method='predict'`, a non-boolean passthrough, weights of the wrong
length, an unknown `voting`.

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
from bindings.py_balanced import py_call, py_get, py_item, py_setattr, py_setitem, py_type, py_contains
from core.device_route import DeviceRoute, DEVICE_CPU
from core.ensemble_member import (
    meta_plan_mint, meta_state_plans, meta_state_notices,
    META_VOTE_CLASSIFIER, META_VOTE_REGRESSOR, META_STACK_CLASSIFIER, META_STACK_REGRESSOR, META_BLOCK_NESTED,
    META_BLOCK_WIDTHS,
)
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
    return kind == META_VOTE_CLASSIFIER or kind == META_STACK_CLASSIFIER


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
    if kind < META_VOTE_CLASSIFIER or kind > META_STACK_REGRESSOR:
        return False
    var classifier = _classifier(kind)
    var active = _active(fields, b)
    if not active.ok:
        return False
    var m = len(active.ests)
    var flag = 0.0
    var folds = 0
    var weights = List[Float64]()
    var stack_kinds = List[Int]()
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
        folds = 5 if cv is None else _int_word(cv, b)
        if folds < 2:
            return False
        var passthrough = py_get(fields, "passthrough")
        if passthrough is not None and py_type(passthrough) is not b.bool:
            return False
        if passthrough is not None and Bool(py=passthrough):
            flag = 1.0
        for i in range(m):  # driver-loop: over the stacked members (admission, once each)
            var method = _stack_method(fields, active.ests[i], classifier, context)
            if method < 0 or (classifier and method == STACK_PREDICT):
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
    var cap = 0
    var jobs = py_get(fields, "n_jobs")
    if jobs is not None:
        var word = _int_word(jobs, b)
        if word > 0:
            cap = word
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
    var handle = meta_plan_mint(nested.plans.copy(), weights^, stack_kinds^, nested.masks.copy())
    _ = py_call(pins.append, meta_plan_capsule(handle))
    owner.plan.route = route
    owner.plan.params = [Float64(kind), Float64(handle), flag, Float64(folds), Float64(cap), 0.0]
    owner.pins = pins
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
