# SPDX-License-Identifier: Apache-2.0
# The native owner of RFE and of mutual_info_classif / mutual_info_regression
# (dispatched by bindings/prep_owner_native.mojo). Each public operation is
# one call: the parameter refusals, the input conversion, the x_prep
# programs the Python built (the same stages, operands and arena order, so
# the same bits) and the fitted attributes or the returned scores.
#
# UNPACK / RUN / PACK (lane bm-rest-sel, 2026-11-23): every program here is
# staged by the typed builders of bindings/prep_select_run.mojo (no Python
# object in them); this door reads the arguments, owns the programs' caller
# buffers and packs results. RFE over a native member (lane mf-meta5) is the
# typed `rfe_fit_run` there WHOLE (every round and the final fit: THE ONE BODY,
# which THE RFE KIND of core/ensemble_member.mojo runs too); its UNPACK and
# PACK pieces are providers/python/member_rfe_boundary_door.mojo's.
#
# RFE runs its elimination rounds here (one driver loop over the rounds); its
# estimator must be a mojolearn estimator with a connected native plan. Its
# clone, fit, predict, predict_proba, decision_function and score run as the
# plan's compiled operations (bindings/prep_native_estimator.mojo), so a round
# never enters a Python method: neither the estimator's fit / predict nor its
# get_params or constructor.
# Its programs run on the shared arena runner (bindings/prep_classify_prog.mojo
# `ClsProg`, the classifier owner's context `select_context`) because the
# support mask and the ranking are in-out inputs, as `_Prog.put(inout=True)`
# made them. The mutual-information programs and the selector transform run
# on the encode owner's transport (bindings/prep_encode_native_door.mojo `EncProg`).
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr, GILReleased
from bindings.classical_validate_unpack import classical_check_y, classical_estimator_name
from bindings.member_leaf_boundary_door import member_leaf_pack, member_leaf_pack_labeled
from bindings.member_boundary_door import member_pack, member_pack_labeled, member_admit_serving_labeled
from core.member_abi import (
    member_kind_capabilities, MEMBER_CAN_FIT, MEMBER_CAN_EXPORT, MEMBER_CAN_CLASSIFIER, MEMBER_GIL_FREE,
    MEMBER_GIL_DOOR,
)
from core.member_families import MEMBER_FAMILY_ENSEMBLE
from core.ensemble_member import META_BLOCK_FORWARD
from bindings.member_fit_notices_pack import member_fit_notices_raise
from core.member_abi import MemberCall, member_free, MEMBER_F32, MEMBER_I32
from core.member_families import MEMBER_FAMILY_GBDT
from providers.python.member_rfe_boundary_door import (
    RfeBase, rfe_base_admit_labeled, rfe_count_words, rfe_image, rfe_publish, rfe_plan_labels, rfe_fit_keywords,
    rfe_member_fit_params, rfe_dotted_path,
)
from bindings.arena_owner_door import arena_pin, arena_unpin
from bindings.x_prep_native_owner_door import xprep_resident
from core.member_set_run import MemberFitExtras
from providers.python.member_gbdt_boundary_door import gbdt_member_fit_params, gbdt_member_label_check
from bindings.native_loaded_module_door import native_loaded_module
from bindings.optimizer_public_words_unpack import stash_python_error
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.classical_arrays_door import classical_address, classical_typed, classical_input
from bindings.admitted_input_unpack import admit_x
from bindings.py_admitted_registry import admitted_scope_begin, admitted_scope_end
from bindings.admitted_meta_scope_unpack import admitted_hold_array_f32
from bindings.classical_arrays_door import classical_is_array
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.classical_labels_door import classical_encode_labels
from bindings.admitted_input_unpack import admit_f32, admitted_vector
from bindings.scaler_native_door import _ptr
from bindings.scaler_lifecycle_native_door import _entries
from bindings.x_prep_native_owner_door import xprep_values
from bindings.prep_stateless_native_door import _select_transform, _select_gather, _support_columns, _words, _fam
from bindings.prep_encode_native_door import (
    EncProg, _fit_categories, _take1, _take2, _default_mode, _seed64, _score_words, _category_build, _cat_block,
    _record_sizes, IDN_STATS_BLOCKED)
from bindings.prep_encode_cats_run import CatBlock
from bindings.prep_select_run import (
    mi_check_neighbors, mi_check_shape, mi_mask_of_bools, mi_mask_of_indices, mi_columns, mi_gather_stage,
    mi_class_count_stage, mi_int_words, mi_class_check, mi_discrete_check, mi_stage, mi_scatter, MiProgram,
    rfe_select_count, rfe_step_count, RfeKeys, rfe_first_stage, rfe_round_stage,
    RfeFitPlan, RfeFitResult, rfe_fit_run,
)
from bindings.prep_classify_prog_door import ClsProg, f32_array
from bindings.prep_classify_common_door import cls_x2d, cls_mode, XIn
from bindings.prep_classify_native_door import resident_enter_min, _resident_exit
from bindings.py_optional import py_or_none, py_present
from bindings.prep_native_estimator_door import (
    prep_estimator_admit, prep_estimator_has, prep_estimator_clone, prep_estimator_call, prep_estimator_attribute,
    prep_estimator_plan)
from bindings.prep_keys_unpack import (
    prep_keys,
    PREP_OPN_DECISION_FUNCTION, PREP_OPN_FIT, PREP_OPN_FIT_TRANSFORM, PREP_OPN_GET_SUPPORT,
    PREP_OPN_MUTUAL_INFO_CLASSIF, PREP_OPN_PREDICT, PREP_OPN_PREDICT_PROBA, PREP_OPN_SCORE, PREP_OPN_TRANSFORM,
)
from bindings.py_balanced import py_item

comptime F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime I64P = MutPointer[Int64, MutUntrackedOrigin]


# ---------------------------------------------------------------- calls into estimators
def _call(callable: PythonObject, args: PythonObject, kwargs: PythonObject) raises -> PythonObject:
    """`callable(*args, **kwargs)` (args a tuple, kwargs a dict or None); an
    exception it raises is restored at the door unchanged."""
    ref cpy = Python().cpython()
    var keywords = PyObjectPtr()
    if kwargs is not None:
        keywords = kwargs._obj_ptr
    var result = cpy.PyObject_Call(callable._obj_ptr, args._obj_ptr, keywords)
    if not result:
        var failure = stash_python_error()
        raise failure^
    return PythonObject(from_owned=result)


def _attr(owner: PythonObject, var name: String) raises -> PythonObject:
    """`owner.name`; its AttributeError (or any other) reaches the caller unchanged."""
    ref cpy = Python().cpython()
    var result = cpy.PyObject_GetAttrString(owner._obj_ptr, name^)
    if not result:
        var failure = stash_python_error()
        raise failure^
    return PythonObject(from_owned=result)


def _class_name(value: PythonObject) raises -> String:
    return String(py_type(value).__name__)


def _fit_clone(estimator: PythonObject, x: PythonObject, y: PythonObject, fit_params: PythonObject,
               context: PythonObject) raises -> PythonObject:
    """RFE's `_clone(estimator).fit(x, y)`: the clone built from the
    constructor schema and fitted by the plan's compiled fit (no Python
    get_params, constructor or fit). Returns the fitted clone.

    WHO STILL COMES HERE (lane mf-del5b audit of mf-meta5b's list; each keeps the estimator's own compiled
    fit door because only that door has the words or the route): a class with no serving member (since lane
    mf-hostent5 OrderedRMSE is Gbdt kind 4, its `permutation` carried from RFE's fit parameters, so no class of
    ours is left here for want of a mark); a member refusal with no fit parameters (THE FALLBACK
    RULE: the door re-fits and raises the estimator's own words, or answers); a one-class or encoder-refused
    target (each family's fit door has its own one-class words); a class_weight table the member cannot
    complete; sample weights for a kind without MEMBER_CAN_SAMPLE_WEIGHT; a non-bool feature_importances; a
    cat_features index outside X; a tree or forest leaf's feature_importances_ under a META step (its
    importance word is the nested plan's: `rfe_dotted_path`); X held as DeviceRows (a fold's rows: the device
    gather route). Each is a refusal or a route the member path does not carry, not a duplicate body."""
    var est = prep_estimator_clone(estimator, context)
    _ = prep_estimator_call(est, "fit", Python.tuple(x, y), context, fit_params)
    return est


# ---------------------------------------------------------------- RFE programs
# TOMBSTONE: _gather_cls (one gather program per round plus one for the final fit, each putting X) deleted
# 2026-11-09 by d6f2d5ed09 (lane w1-readmission-rfe); code recoverable at d6f2d5ed09^.
# Restore: git apply experiments/removed/W1_RFE_GATHER_PROGRAM_PER_ROUND.patch
# The round programs are typed (lane bm-rest-sel, 2026-11-23): bindings/prep_select_run.mojo
# `rfe_first_stage`, `rfe_round_stage`, `rfe_keys_stage`, `rfe_block_stage` stage what `_stage_block` and
# `_keys_of` staged here over the Python-holding program, in the same order; this door gives each caller
# buffer record its owner (`ClsProg.own`, record order).
def _importance_keys(mut pr: ClsProg, est: PythonObject, m: Int, getter: PythonObject, context: PythonObject,
                     cc: PythonObject) raises -> RfeKeys:
    """UNPACK of the fitted estimator's m column importances for the typed
    key stages (`_stage_importance_keys(pr, est, m, getter)`): coef_
    (squared, summed over rows when 2-D, on the device), else
    feature_importances_ as given; a dotted attribute path picks the
    importances instead (squared). The words become the round program's
    first caller buffer."""
    var b = native_loaded_module("builtins")
    var auto = py_type(getter) is b.str and String(getter) == "auto"
    # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
    var coef = Optional[PythonObject]()
    if not auto:
        coef = py_present(prep_estimator_attribute(est, getter, context))
    else:
        coef = py_present(prep_estimator_attribute(est, PythonObject("coef_"), context))
    if not coef and auto:
        var imp = prep_estimator_attribute(est, PythonObject("feature_importances_"), context)
        if imp is None:
            raise Error("mojolearn: RFE needs an estimator with coef_ or feature_importances_")
        return _keys_of(pr, imp, False, m, context, cc)
    return _keys_of(pr, py_or_none(coef), True, m, context, cc)


def _keys_of(mut pr: ClsProg, values: PythonObject, squared: Bool, m: Int, context: PythonObject,
             cc: PythonObject) raises -> RfeKeys:
    """UNPACK of m column importances: `values` as given (their own words
    and c2_key64 kind), or (`squared`: a coefficient block, 1-D or rows x m)
    admitted as float32 rows. The words are owned by `pr` as the next
    caller buffer; returns what `rfe_keys_stage` stages."""
    if not squared:
        var got = _score_words(values, m, context)
        pr.own(got[0])
        return RfeKeys(m, 1, False, got[1])
    var coef = py_present(values)
    # `as_f32_c(coef, ndim=None)` then `rows, d = (1, len) if 1-D else shape`
    var raw = classical_input(py_or_none(coef), cc, "coef_")
    var nd = Int(py=raw.ndim)
    var rows = 1
    var d = 0
    if nd == 1:
        d = Int(py=raw.shape[0])
    elif nd == 2:
        rows = Int(py=raw.shape[0])
        d = Int(py=raw.shape[1])
    elif nd == 0:
        raise Error("not enough values to unpack (expected 2, got 0)")
    else:
        raise Error("too many values to unpack (expected 2)")
    if d != m:
        raise Error("mojolearn: RFE importances have " + String(d) + " columns, expected " + String(m))
    pr.own(admit_x(py_or_none(coef), nd, "coef_", cc))
    return RfeKeys(m, rows, True, 0)


# ---------------------------------------------------------------- a dotted importance getter (lane mf-meta5b)
def _rfe_dotted_base(estimator: PythonObject, g: PythonObject, context: PythonObject, b: PythonObject) -> RfeBase:
    """UNPACK, once per RFE fit, of a dotted importance_getter: the path walked ONCE
    (providers/python/member_rfe_boundary_door.mojo `rfe_dotted_path`, the walk THE RFE KIND's admission runs
    too, lane mf-del5b) to THE FORWARDED EXPORT's hops and the leaf's attrgetter words, then the META
    estimator admitted as a member through the member dispatch (no labels: a nested class_weight dict keeps
    the door). Not served (the door keeps its attribute reads): whatever `rfe_dotted_path` does not serve,
    a META estimator with no serving member. Never raises."""
    var base = RfeBase()
    try:
        var path = rfe_dotted_path(estimator, g, context, b)
        if not path.served:
            return base^
        var source_refusal = path.refusal.copy()
        var hops = path.hops.copy()
        var plan = prep_estimator_plan(estimator, context)
        if plan is None:
            return base^
        var template = prep_estimator_clone(estimator, context)
        var member_context = plan["context"]
        var fields = py_call(plan["fields_get"], template, py_type(template))
        var owner = member_admit_serving_labeled(template, fields, plan, member_context, b,
                                                 MEMBER_CAN_FIT | MEMBER_CAN_EXPORT, PythonObject(None))
        if not owner.published or owner.plan.family != MEMBER_FAMILY_ENSEMBLE:
            return base^
        var mask = member_kind_capabilities(owner.plan)
        # The estimator type is the META kind's (lane mf-del5b): a search's class says none of its own (its
        # base decides, lane mf-search4), so its plan's word would have refused a search over a classifier;
        # THE RFE KIND's admission reads the same mask.
        var classifier = (mask & MEMBER_CAN_CLASSIFIER) != 0
        if (mask & MEMBER_CAN_FIT) == 0 or (mask & MEMBER_CAN_EXPORT) == 0:
            return base^
        base.refusal = source_refusal^
        base.block = META_BLOCK_FORWARD
        base.squared = True
        base.hops = hops^
        base.meta = True
        base.gil_free = (mask & (MEMBER_GIL_FREE | MEMBER_GIL_DOOR)) != 0
        base.mask = mask
        base.classifier = classifier
        base.owner = owner^
        base.template = template
        base.fields = fields
        base.member_context = member_context
        base.native = True
    except:
        return RfeBase()
    return base^


# ---------------------------------------------------------------- RFE over a native member
struct RfeMember(Movable):
    """RFE's hold on its estimator as a NATIVE MEMBER (core/member_abi.mojo), built once per fit
    (`_rfe_member`): the estimator admitted with no fit target (providers/python/member_rfe_boundary_door.mojo
    `rfe_base_admit`, which THE RFE KIND's admission runs too), then the fit's target and parameters.
    `base.native` False: the estimator's own doors (`_fit_clone`, `prep_estimator_attribute`) keep every
    round. `eval_rows` / `eval_keys` (lane mf-meta5b): a Gbdt fit with cat_features's held-out rows and
    keys; `final_importances`: a forest's feature_importances=True (the final fit keeps its importance word). `y` is the admitted target the RUN reads by address (float32 values for a regressor kind, the
    int32 class codes of `labels` for a classifier kind: encoded once per fit). `extras` (lane mf-gbdt5):
    RFE's fit parameters, admitted once with the estimator's fit door's checks and words (a Gbdt member:
    providers/python/member_gbdt_boundary_door.mojo `gbdt_member_fit_params`; `pins` holds every buffer
    they address). `strict`: fit parameters were admitted, so a member refusal is raised (the estimator's
    own doors would read the held-out X whole)."""
    var base: RfeBase
    var y: PythonObject
    var y_address: Int
    var y_dtype: Int
    var labels: PythonObject
    var extras: MemberFitExtras
    var pins: PythonObject
    var ids: List[Int32]
    var codes: List[Int32]
    var eval_rows: List[Float32]
    var eval_keys: List[UInt64]
    var final_importances: Bool
    var strict: Bool

    def __init__(out self):
        self.base = RfeBase()
        self.y = PythonObject(None)
        self.y_address = 0
        self.y_dtype = MEMBER_F32
        self.labels = PythonObject(None)
        self.extras = MemberFitExtras()
        self.pins = PythonObject(None)
        self.ids = List[Int32]()
        self.codes = List[Int32]()
        self.eval_rows = List[Float32]()
        self.eval_keys = List[UInt64]()
        self.final_importances = False
        self.strict = False


def _rfe_member(estimator: PythonObject, y: PythonObject, n: Int, d: Int, g: PythonObject, fit_params: PythonObject,
                context: PythonObject, b: PythonObject) raises -> RfeMember:
    """UNPACK, once per RFE fit: the fit parameters' names and values in the estimator's compiled fit's
    words (`rfe_fit_keywords`: its refusals raise here, before any round, as `_fit_clone`'s first fit
    raised them), the estimator as a native member (`rfe_base_admit_labeled`, its labels deferred: lane
    mf-meta5b), then its fit target: a classifier kind's labels encoded here, once, and the plan completed
    over them (`rfe_plan_labels`: LabelPropagation / LabelSpreading's -1 marker word, a class_weight dict's
    table; `gbdt_member_label_check`: the classifier adapter's multiclass refusal, in its fit door's place),
    a regressor kind's target checked as its member's fit door admits it. Then the fit parameters: a Gbdt
    member's (`gbdt_member_fit_params`: sample_weight, eval_set (with cat_features too), group_id, pairs,
    pairs_weight), any other family's (`rfe_member_fit_params`: sample_weight through the shared weight
    admission, a forest's feature_importances); their refusals raise from here. What stays on the
    estimator's own doors (`_fit_clone`, the fallback rule): a class with no serving member, a target the
    encoder or the regressor admission refuses (one class among them: each family's fit door has its own
    one-class words), a class_weight table it cannot complete, a key or value the member does not carry
    (weights for a kind that reads none, a non-bool feature_importances, a cat_features index outside X)."""
    var member = RfeMember()
    var has_params = len(fit_params) != 0
    if has_params:
        rfe_fit_keywords(estimator, fit_params, context, b)
    var base = rfe_base_admit_labeled(estimator, g, context, b, Python.list())
    if not base.native:
        base = _rfe_dotted_base(estimator, g, context, b)
    if not base.native:
        return member^
    try:
        if base.classifier:
            # The labels once, as a classifier's fit door encodes them: the class list it would publish
            # and one int32 code per row.
            var encoded = classical_encode_labels(y, base.member_context)
            var count = len(encoded.classes)
            if encoded.n != n or count < 2:
                return member^
            if not rfe_plan_labels(base, encoded.classes, b):
                return member^
            base.owner.plan.classes = count
            member.y = encoded.codes
            member.y_dtype = MEMBER_I32
            member.labels = encoded.classes
        else:
            # The target once, as a regressor member's fit door admits it (present, real, one column,
            # finite, as long as X).
            member.y = classical_check_y(y, n, classical_estimator_name(estimator), base.member_context)
    except:
        return member^
    # The classifier adapter over more than two classes: its fit door's refusal, raised by its first fit
    # (no elimination round fits a Gbdt kind first: `rfe_member_source`).
    if member.labels is not None:
        gbdt_member_label_check(base.owner.plan, len(member.labels))
    if has_params:
        if base.owner.plan.family == MEMBER_FAMILY_GBDT:
            # UNPACK of the fit parameters (their refusals raise here, in the fit door's words, as the
            # door's first fit would raise them).
            var params = gbdt_member_fit_params(base.template, base.owner.plan, fit_params, y, member.labels, n, d,
                                                base.member_context)
            if not params.admitted:
                return member^
            member.extras = params.extras.copy()
            member.pins = params.pins
            # By swap: the addresses in `extras` point into these lists' storage.
            swap(member.ids, params.ids)
            swap(member.codes, params.codes)
            swap(member.eval_rows, params.eval_rows)
            swap(member.eval_keys, params.eval_keys)
        else:
            var params = rfe_member_fit_params(base, fit_params, n, b)
            if not params.admitted:
                return member^
            member.extras = params.extras.copy()
            member.pins = params.pins
            member.final_importances = params.final_importances
        member.strict = True
    member.y_address = Int(py=classical_address(member.y))
    member.base = base^
    return member^


def _rfe_native(member: RfeMember, arr: XIn, n: Int, d: Int, counts: List[Float64], cc: PythonObject,
                b: PythonObject, mut support: List[Float32], mut ranking: List[Float32],
                mut selected: Int) raises -> PythonObject:
    """RFE.fit over the admitted member: UNPACK of the image, the thread's resident scope and X's pin;
    RUN `rfe_fit_run` (THE ONE BODY, the GIL released for a kind with MEMBER_GIL_FREE or MEMBER_GIL_DOOR,
    as the door's rounds released it); PACK the final state into the one clone (`member_leaf_pack`, a
    classifier with its labels: the attributes its own fit writes) and the rounds' and the final fit's
    notices, in fit order. Answers the packed clone; `support`, `ranking` and `selected` are written."""
    var image = rfe_image(cc)
    var plan = RfeFitPlan(member.base.owner.plan.copy(), image.run, image.work_on,
                          Int(py=classical_address(arr.value)), n, d, member.y_address, member.y_dtype)
    plan.nsel = rfe_select_count(d, Int(counts[0]), counts[1], Int(counts[2]))
    plan.step = rfe_step_count(d, counts[3] != 0.0, counts[4], Int(counts[5]))
    plan.block = member.base.block
    plan.squared = member.base.squared
    # A forest's feature_importances=True keeps the final fit's importance word (estimator_ publishes them, as
    # the estimator's own fit with that parameter does); the rounds set it whatever the parameter says.
    plan.importance_word = member.base.importance_word and not member.final_importances
    plan.refusal = member.base.refusal.copy()
    plan.extras = member.extras.copy()
    plan.hops = member.base.hops.copy()
    # The thread's resident scope (the door's `resident()` scope over the rounds, or an enclosing one):
    # X keeps one device copy there, pinned by one reference the scope's cache may take (THE PIN RULE).
    var resident = xprep_resident(cc["prog_context"], image.run.key, image.run.dev_store)
    var pin = 0
    var kept = 0
    if resident.scope >= 0:
        pin = arena_pin(arr.value)
        plan.scope = resident.scope
        plan.pin = pin
        plan.kept_at = Int(UnsafePointer(to=kept))
    var res = RfeFitResult()
    try:
        if member.base.gil_free:
            var py = Python()
            with GILReleased(py):
                res = rfe_fit_run(plan)
        else:
            res = rfe_fit_run(plan)
    except error:
        if pin != 0 and kept == 0:
            arena_unpin(pin)
        raise error^
    if pin != 0 and kept == 0:
        arena_unpin(pin)
    _ = arr.value  # read by address through the run above
    _ = member.y
    _ = member.pins
    _ = len(member.ids)  # read by address through the run above (group ids, held-out codes and rows, keys)
    _ = len(member.codes)
    _ = len(member.eval_rows)
    _ = len(member.eval_keys)
    _ = image.entries  # the selection pins the image through the run
    var call = MemberCall(res.final_plan.route, res.final_state)
    try:
        if member.base.meta:
            # A dotted getter's META estimator (`_rfe_dotted_base`): its own family's PACK.
            if member.labels is None:
                member_pack(member.base.template, member.base.fields, member.base.member_context, b,
                            res.final_plan, res.final_state)
            else:
                member_pack_labeled(member.base.template, member.base.fields, member.base.member_context, b,
                                    res.final_plan, res.final_state, member.labels)
        elif member.labels is None:
            member_leaf_pack(member.base.template, member.base.fields, member.base.member_context, b,
                             res.final_plan, res.final_state)
        else:
            member_leaf_pack_labeled(member.base.template, member.base.fields, member.base.member_context, b,
                                     res.final_plan, res.final_state, member.labels)
    finally:
        member_free(res.final_plan, call)
    for i in range(len(res.notices)):  # driver-loop: over the member fits' notices, in fit order
        member_fit_notices_raise(res.notices[i], member.base.member_context)
    support = res.support.copy()
    ranking = res.ranking.copy()
    selected = res.selected
    return member.base.template


def _fit_params(args: PythonObject) raises -> PythonObject:
    var params = args[2] if len(args) > 2 else PythonObject(None)
    if params is None:
        return Python.dict()
    return params


def _rfe_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    """RFE.fit(X, y, **fit_params): the elimination rounds, then the final
    estimator on the selected columns."""
    var nk = prep_keys()
    var b = native_loaded_module("builtins")
    var cc = py_item(context, nk[].k_select_context)
    # RFE uses the classifier arena's independent input context. The outer
    # prep door connects only its encode context, so a first RFE call could
    # reject the ndarray that composition_before just prepared. Connect this
    # existing native NumPy metadata here; a prior GaussianNB/LDA call must
    # not be required, and NumPy may be loaded after package import.
    classical_connect_numpy(cc)
    var fields = py_fields(model)
    var estimator = py_item(fields, nk[].k_estimator)
    prep_estimator_admit(estimator, "RFE", context)
    var g = py_item(fields, nk[].k_importance_getter)
    if not Bool(py=py_call(b.isinstance, g, b.str)):
        if Bool(py=py_call(b.callable, g)):
            raise Error("TypeError: mojolearn: RFE importance_getter must be 'auto' or an attribute path; a callable is not supported")
        raise Error("mojolearn: RFE importance_getter must be 'auto', a str or a callable")
    var y = args[1]
    var fit_params = _fit_params(args)
    if py_type(fit_params) is not b.dict:
        raise Error("TypeError: mojolearn: RFE.fit parameters require an exact dict")
    var arr = cls_x2d(args[0], cc)
    # X stays on the device for every elimination round and the final fit
    # (lane ne2-b-rounds, N059; lane w1-readmission-rfe). One resident scope
    # (the classifier owner's native `with resident():`) spans the rounds
    # with min_words = n * d, so X (the same admitted value every round
    # program puts) goes up once and every later gather reads its device
    # copy; the d-word support, ranking and column ids are in-out words of
    # the round's program, not held. An enclosing user scope, when one is
    # active, is the one used. The scope is required: the arena rows owner
    # connects when the package imports model_selection, and entering it only
    # records the scope, so the silent "no scope, put X every round" fallback
    # is deleted with it.
    # S4 (lane s1): X came through admit_x above (refused unless finite), so
    # every round's block, columns gathered out of it, is finite: it is held
    # in the admitted registry while that round's estimator fits, and the
    # member door (in any binding: one registry per process) reads it with
    # no view, check or scan. A fold's DeviceRows X was not tested here, so
    # its blocks are not held.
    var hold = py_type(args[0]) is not py_item(cc, nk[].k_maxabs_rows_class)
    var mark = -1
    var scope = resident_enter_min(cc, PythonObject(arr.n * arr.d))
    try:
        _rfe_rounds(model, fields, estimator, g, y, fit_params, arr, context, cc, hold, mark)
    except error:
        try:
            _resident_exit(scope, cc)
        except ignored:
            _ = ignored^
        if mark >= 0:
            admitted_scope_end(mark)
        raise error^
    try:
        _resident_exit(scope, cc)
    finally:
        if mark >= 0:
            admitted_scope_end(mark)


def _rfe_hold_block(block: PythonObject, hold: Bool, mut mark: Int, cc: PythonObject) raises:
    """Hold this round's block for its fit in the one RFE hold scope: the
    previous round's scope (`mark`, -1 for none) is closed first, so one
    block is held at a time; `_rfe_fit` closes the last one on any exit."""
    if mark >= 0:
        admitted_scope_end(mark)
        mark = -1
    if not hold or not classical_is_array(block, cc):
        return
    mark = admitted_scope_begin()
    _ = admitted_hold_array_f32(block, False)


def _rfe_rounds(model: PythonObject, fields: PythonObject, estimator: PythonObject, g: PythonObject,
                y: PythonObject, fit_params: PythonObject, arr: XIn, context: PythonObject,
                cc: PythonObject, hold: Bool, mut mark: Int) raises:
    """The elimination rounds and the final fit of `_rfe_fit`, inside its resident scope: over a native
    member, the typed RUN whole (`_rfe_native`); else the estimator's own doors per round."""
    var b = native_loaded_module("builtins")
    var n = arr.n
    var d = arr.d
    var mode = cls_mode(cc)
    # UNPACK of n_features_to_select and step (providers/python/member_rfe_boundary_door.mojo
    # `rfe_count_words`); their counts are typed (bindings/prep_select_run.mojo `rfe_select_count`,
    # `rfe_step_count`).
    var counts = rfe_count_words(fields, b)
    var nsel = rfe_select_count(d, Int(counts[0]), counts[1], Int(counts[2]))
    var step = rfe_step_count(d, counts[3] != 0.0, counts[4], Int(counts[5]))
    # The estimator as a native member (admitted once for the whole fit) when its family has published
    # what RFE reads: THE ONE BODY (bindings/prep_select_run.mojo `rfe_fit_run`, which THE RFE KIND also
    # runs). A member that refuses (a view or dtype its family does not read, a fit failure) leaves the
    # whole fit to the estimator's own doors below, which raise the estimator's own words if the request is
    # wrong (the fallback rule); with admitted fit parameters its words are raised. X held as DeviceRows
    # (a fold's rows) is the own doors' (its blocks are gathered on the device).
    var member = _rfe_member(estimator, y, n, d, g, fit_params, context, b)
    if member.base.native and classical_is_array(arr.value, cc):
        var support = List[Float32]()
        var ranking = List[Float32]()
        var selected = 0
        var final = PythonObject(None)
        try:
            final = _rfe_native(member, arr, n, d, counts, cc, b, support, ranking, selected)
        except error:
            if member.strict or member.base.refusal.byte_length() > 0:
                raise error^
        if final is not None:
            rfe_publish(fields, final, support, ranking, selected, context)
            return
    # The estimator's own doors, one program per elimination round (lane w1-readmission-rfe): the fitted
    # estimator's importance keys, c2_rfe_step on the support mask, c2_rfe_rank, then the next block staged
    # from the updated mask (c2_rfe_keep + gather_cols). Program 0 stages the first block from the all-ones
    # mask. The support, ranking and column ids ride each program as its in-out and output words (d words
    # each). Each round drops exactly min(step, nsup - nsel) columns (c2_rfe_step's stable rank is a total
    # order), so the round count and every block width are known before the program runs. The programs are
    # the typed builders of bindings/prep_select_run.mojo; this door owns their caller buffers in record
    # order. The rounds are serial by nature: a round's block is chosen from the previous round's
    # importances.
    var x_words = Int(py=arr.value.size)
    var support = f32_array(d, cc)
    var sp = _ptr(support)
    var ranking = f32_array(d, cc)
    var rp = _ptr(ranking)
    for j in range(d):  # driver-loop: over the initial support and ranking (feature count)
        sp[j] = 1.0
        rp[j] = 1.0
        _ = support  # keep the Python object alive through the native call above
        _ = ranking  # keep the Python object alive through the native call above
    var pr0 = ClsProg()
    var first = rfe_first_stage(pr0.program, x_words, n, d)
    pr0.own(support)
    pr0.own(arr.value)
    pr0.run(mode, cc)
    var feat = pr0.get_i32(first.ids, d, cc)
    var block = pr0.get(first.block, n, d, cc)
    var nsup = d
    while nsup > nsel:  # driver-loop: over elimination rounds
        if member.base.native and member.base.refusal.byte_length() > 0:
            # A kind with no importance block: refused before the round fits.
            raise Error(member.base.refusal.copy())
        var drop = min(step, nsup - nsel)
        var k = nsup - drop
        var pr = ClsProg()
        _rfe_hold_block(block, hold, mark, cc)
        var est = _fit_clone(estimator, block, y, fit_params, context)
        var keys = _importance_keys(pr, est, nsup, g, context, cc)
        var staged = rfe_round_stage(pr.program, keys, List[Float32](), x_words, n, d, nsup, drop, k)
        pr.own(support)
        pr.own(ranking)
        pr.own(feat)
        pr.own(arr.value)
        pr.run(mode, cc)
        support = pr.get(staged.support, d, 0, cc)
        ranking = pr.get(staged.ranking, d, 0, cc)
        feat = pr.get_i32(staged.next.ids, k, cc)
        block = pr.get(staged.next.block, n, k, cc)
        nsup = k
    _rfe_hold_block(block, hold, mark, cc)
    var final = _fit_clone(estimator, block, y, fit_params, context)
    _ = member.base.native  # the member's pins outlive the admission above
    var support_words = List[Float32](length=d, fill=0.0)
    var ranking_words = List[Float32](length=d, fill=0.0)
    var fs = _ptr(support)
    var fr = _ptr(ranking)
    for j in range(d):  # driver-loop: over the fitted support and ranking (feature count)
        support_words[j] = fs[j]
        ranking_words[j] = fr[j]
        _ = support  # keep the Python object alive through the native call above
        _ = ranking  # keep the Python object alive through the native call above
    _ = mode
    rfe_publish(fields, final, support_words, ranking_words, nsup, context)


def _rfe_transform(model: PythonObject, x: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = prep_keys()
    # `_SelectorMixin.transform`, as the stateless owner runs it.
    var fields = py_fields(model)
    if not py_contains(fields, nk[].k_n_features_in_):
        raise Error("RuntimeError: mojolearn: this " + _class_name(model) + " instance is not fitted yet")
    var values = xprep_values(x, _entries(py_item(fields, nk[].k_numeric_mode_), context), context)
    var out: PythonObject
    if rfe_support_cached(fields):
        var d = Int(py=values[4])
        var fitted = py_item(fields, nk[].k_n_features_in_)
        if Int(py=fitted) != d:
            raise Error("mojolearn: X has " + String(d) + " features, " + _class_name(model) + " was fitted with " + String(Int(py=fitted)))
        out = _select_gather(fields, values, py_item(fields, nk[].k__support_words_), Int(py=py_item(fields, nk[].k__support_k_)), context)
    else:
        out = _select_transform(model, values, context)
    _ = values^
    return out


def rfe_support_cached(fields: PythonObject) raises -> Bool:
    """Whether fit's stored support words still describe `_mask` (the mask
    object fit stored, not one put there since) and select any column."""
    var nk = prep_keys()
    var words = py_get(fields, nk[].k__support_words_)
    if words is None:
        return False
    var mask = py_get(fields, nk[].k__mask)
    var k = py_get(fields, nk[].k__support_k_)
    return mask is not None and mask is py_get(fields, nk[].k__support_mask_) and k is not None and Int(py=k) > 0


def prep_rfe_call(model: PythonObject, op: Int, operation: String, args: PythonObject,
                  context: PythonObject) raises -> PythonObject:
    var nk = prep_keys()
    if op == PREP_OPN_GET_SUPPORT:
        var fields = py_fields(model)
        var mask = py_item(fields, nk[].k__mask)
        var indices = Bool(py=args[0])
        if not indices:
            # list(mask): one call (p3-prep)
            return py_call(native_loaded_module("builtins").list, mask)
        # The kept indices by a pointer scan of the fitted bool list.
        var keep = _support_columns(mask)
        var result = Python.list()
        for i in range(len(keep)):  # driver-loop: completed feature support metadata
            _ = py_call(result.append, PythonObject(keep[i]))
        return result
    if op == PREP_OPN_TRANSFORM:
        return _rfe_transform(model, args[0], context)
    if op == PREP_OPN_FIT:
        _rfe_fit(model, args, context)
        return model
    if op == PREP_OPN_FIT_TRANSFORM:
        _rfe_fit(model, args, context)
        return _rfe_transform(model, args[0], context)
    if op == PREP_OPN_PREDICT or op == PREP_OPN_PREDICT_PROBA or op == PREP_OPN_DECISION_FUNCTION or op == PREP_OPN_SCORE:
        # `self.estimator_.<op>(self.transform(X)[, y])`: the method is looked
        # up first, in the fitted estimator's native plan
        var fitted = py_fields(model)
        if not py_contains(fitted, nk[].k_estimator_):
            raise Error("AttributeError: '" + _class_name(model) + "' object has no attribute 'estimator_'")
        var est = py_item(fitted, nk[].k_estimator_)
        if est is not py_get(fitted, nk[].k__estimator_admitted_):
            prep_estimator_admit(est, "RFE", context)
        if not prep_estimator_has(est, operation, context):
            raise Error("AttributeError: '" + _class_name(est) + "' object has no attribute '" + operation + "'")
        var xs = _rfe_transform(model, args[0], context)
        if op == PREP_OPN_SCORE:
            return prep_estimator_call(est, operation, Python.tuple(xs, args[1]), context)
        return prep_estimator_call(est, operation, Python.tuple(xs), context)
    raise Error("TypeError: the native RFE owner does not serve " + operation)


# ---------------------------------------------------------------- mutual information
def _discrete_mask(df: PythonObject, d: Int, context: PythonObject) raises -> List[Bool]:
    """UNPACK of `discrete_features` (`_mi_discrete_mask`): 'auto' (none), a
    bool for every column, a bool mask of length d, or column indices, read
    into the typed mask (bindings/prep_select_run.mojo `mi_mask_of_bools`,
    `mi_mask_of_indices`)."""
    var nk = prep_keys()
    var b = native_loaded_module("builtins")
    if Bool(py=py_call(b.isinstance, df, b.str)):
        if String(df) != "auto":
            raise Error("mojolearn: Invalid string value for discrete_features.")
        return List[Bool](length=d, fill=False)
    if Bool(py=py_call(b.isinstance, df, b.bool)) or String(py_type(df).__name__) == "bool_":
        return List[Bool](length=d, fill=Bool(py=py_call(b.bool, df)))
    var vals = PythonObject(None)
    if Bool(py=py_call(b.hasattr, df, "tolist")):
        vals = _call(_attr(df, "tolist"), Python.tuple(), PythonObject(None))
    else:
        vals = py_call_typed(b.list, df)
    if Bool(py=py_call(b.isinstance, vals, b.bool)):
        return List[Bool](length=d, fill=Bool(py=vals))
    if py_type(vals) is not b.list:
        vals = py_call_typed(b.list, vals)
    var count = len(vals)
    var all_bool = count > 0
    for i in range(count):  # driver-loop: over the discrete_features argument
        if not Bool(py=py_call(b.isinstance, vals[i], b.bool)):
            all_bool = False
            break
    if all_bool:
        var given = List[Bool](capacity=count)
        for i in range(count):  # driver-loop: over the discrete_features mask
            given.append(Bool(py=vals[i]))
        return mi_mask_of_bools(given, d)
    # (int, numbers.Integral), connected once (python/mojolearn/_native_prep_encode_bridge.py);
    # the per-call numbers import fallback is gone.
    var integral = py_item(context, nk[].k_integral_type)
    var indices = List[Int](capacity=count)
    for i in range(count):  # driver-loop: over the discrete_features indices
        var v = vals[i]
        if Bool(py=py_call(b.isinstance, v, b.bool)) or not Bool(py=py_call(b.isinstance, v, integral)):
            # An earlier index out of bounds is refused first, as the one
            # admission loop refused it before reaching this entry.
            _ = mi_mask_of_indices(indices, d)
            raise Error("mojolearn: discrete_features must be 'auto', a bool, a bool mask or indices")
        indices.append(Int(py=py_call_typed(b.int, v)))
    return mi_mask_of_indices(indices, d)


def _mi_gather(values: PythonObject, cols: List[Int], entries: PythonObject, context: PythonObject) raises -> PythonObject:
    """The columns `cols` of X (`xprep_values` record) as an `xprep_values`
    record (0, Array, None, n, k): UNPACK of X, the typed gather
    (bindings/prep_select_run.mojo `mi_gather_stage`), the run, PACK."""
    var n = Int(py=values[3])
    var k = len(cols)
    var p = EncProg()
    var x = p.own_x(values)
    var dst = mi_gather_stage(p.prog, x, n, Int(py=values[4]), cols)
    var ran = p.run(entries, context)
    return Python.tuple(0, _take2(ran, dst, n, k, context), PythonObject(None), n, k)


def prep_mutual_info_call(op: Int, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_mutual_info(X, y, discrete_target, discrete_features, n_neighbors,
    random_state)`. UNPACK: the arguments, X, the targets and the discrete
    block's categories as owners of the programs and typed words; RUN: the
    typed builders of bindings/prep_select_run.mojo (`mi_class_count_stage`,
    `mi_gather_stage`, `mi_stage`) on the encode owner's transport; PACK:
    the float32 scores Array."""
    var nk = prep_keys()
    var b = native_loaded_module("builtins")
    var discrete_target = op == PREP_OPN_MUTUAL_INFO_CLASSIF
    var x = args[0]
    var y = args[1]
    var k = Int(py=py_call_typed(b.int, args[3]))
    mi_check_neighbors(k)
    var mode = _default_mode(context)
    var entries = _entries(mode, context)
    var values = xprep_values(x, entries, context)
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    mi_check_shape(n, d)
    var columns = mi_columns(_discrete_mask(args[2], d, context))
    var cont = columns[0].copy()
    var disc = columns[1].copy()
    var seed = _seed64(args[4])[0] & 0x3FFFFFFF
    var scratch = Bool(py=py_item(context, nk[].k_select_mi_work)) and String(mode) == "fast"
    # The image's IDN_STATS_BLOCKED word, read once, where the column
    # statistics run (a continuous column or a continuous target).
    var stats_blocked = False
    if len(cont) > 0 or not discrete_target:
        stats_blocked = (_fam(mode, entries) & IDN_STATS_BLOCKED) != 0
    var p = EncProg()
    var y_owner = -1
    var y_words = 0
    var counts = List[Int]()
    var target = PythonObject(None)
    if discrete_target:
        var labels = classical_encode_labels(y, context)
        if labels.n != n:
            raise Error("mojolearn: X and y have different numbers of rows")
        var K = len(labels.classes)
        y_words = Int(py=labels.codes.size)
        var pc = EncProg()
        var held = pc.own(labels.codes, y_words)
        var tot = mi_class_count_stage(pc.prog, held, y_words, n, K)
        var counted = pc.run(entries, context)
        counts = mi_int_words(Int(py=counted[0][1]), tot, K)
        _ = counted^  # the arena store lives through the reads above
        mi_class_check(counts, len(cont) > 0)
        y_owner = p.own(labels.codes, y_words)
    else:
        var t = admit_f32(y, 1, "y", context, finite=False)
        if t.rows != n:
            raise Error("mojolearn: X and y have different numbers of rows")
        target = admitted_vector(n, "<f4", context)
        unsafe_memcpy(dest=_ptr(target), src=F32P(unsafe_from_address=t.address), count=n)
        _ = target  # keep the Python object alive through the native call above
        _ = t^
        y_owner = p.own(target, n)
    var x_owner = -1
    if len(cont) > 0:
        x_owner = p.own_x(_mi_gather(values, cont, entries, context) if len(disc) > 0 else values)
    var xd_owner = -1
    var cats = CatBlock(-1, 0, 0, -1, 0)
    var kx = List[Int]()
    if len(disc) > 0:
        var xd = _mi_gather(values, disc, entries, context)
        var record = _category_build(_fit_categories(xd, mode, entries, context), context)
        kx = _record_sizes(record)
        if not discrete_target:
            mi_discrete_check(kx, disc, n)
        xd_owner = p.own_x(xd)
        cats = _cat_block(p, record, True)
    var staged = mi_stage(p.prog, MiProgram(n, k, seed, discrete_target, scratch, stats_blocked, cont.copy(),
                                            disc.copy(), x_owner, y_owner, y_words, counts^, xd_owner, cats^, kx^))
    var ran = p.run(entries, context)
    _ = target^
    if len(disc) == 0:
        var res = _take1(ran, staged.outc, d, context)
        _ = values^
        return res
    # PACK: the scores scattered to their columns, as a float32 Array
    var out = classical_typed(Python.tuple(d), "<f4", py_item(context, nk[].k_output_context))
    mi_scatter(Int(py=ran[0][1]), staged, cont, disc, Int(py=classical_address(out)))
    _ = out  # keep the Python object alive through the native call above
    _ = ran^  # the arena store lives through the scatter above
    _ = values^
    return out
