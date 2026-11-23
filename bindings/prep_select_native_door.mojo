# SPDX-License-Identifier: Apache-2.0
# The native owner of RFE and of mutual_info_classif / mutual_info_regression
# (dispatched by bindings/prep_owner_native.mojo). Each public operation is
# one call: the parameter refusals, the input conversion, the x_prep
# programs the Python built (the same stages, operands and arena order, so
# the same bits) and the fitted attributes or the returned scores.
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
from bindings.member_boundary_door import member_admit_serving, member_pack, member_pack_labeled
from bindings.member_fit_notices_pack import member_fit_notices_raise
from core.member_notices import member_notices_take
from bindings.member_owner_unpack import MemberOwner
from bindings.member_set_door import MEMBER_SET_NATIVE
from core.member_abi import (
    MemberCall, MemberMatrix, MemberPlan, member_call, member_new, member_free, member_kind_capabilities,
    MEMBER_OP_FIT, MEMBER_OP_EXPORT, MEMBER_CAN_FIT, MEMBER_CAN_EXPORT, MEMBER_CAN_COEF, MEMBER_CAN_IMPORTANCES,
    MEMBER_CAN_CLASSIFIER, MEMBER_BLOCK_COEF, MEMBER_BLOCK_FEATURE_IMPORTANCES, MEMBER_F32, MEMBER_I32,
    MEMBER_GIL_FREE, MEMBER_GIL_DOOR,
)
from core.member_families import MEMBER_FAMILY_TREE, MEMBER_FAMILY_FOREST
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

# x_prep stage codes (_expansion_prep._OPS).
comptime OP_GATHER_COLS = 61
comptime OP_SQSUM_COLS = 71
comptime OP_C2_KEY64 = 253
comptime OP_C2_RFE_STEP = 256
comptime OP_C2_RFE_RANK = 257
comptime OP_C2_RFE_KEEP = 271


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
    get_params, constructor or fit). Returns the fitted clone."""
    var est = prep_estimator_clone(estimator, context)
    _ = prep_estimator_call(est, "fit", Python.tuple(x, y), context, fit_params)
    return est


# ---------------------------------------------------------------- RFE programs
# TOMBSTONE: _gather_cls (one gather program per round plus one for the final fit, each putting X) deleted
# 2026-11-09 by d6f2d5ed09 (lane w1-readmission-rfe); code recoverable at d6f2d5ed09^.
# Restore: git apply experiments/removed/W1_RFE_GATHER_PROGRAM_PER_ROUND.patch
def _stage_block(mut pr: ClsProg, x_value: PythonObject, n: Int, d: Int, sup: Int,
                 k: Int) raises -> Tuple[Int, Int]:
    """Stage the next round's block: the supported columns of the support
    mask at `sup` compacted on the device (c2_rfe_keep) and those k columns
    of X gathered into an (n, k) block (gather_cols, bit for bit). X is the
    one admitted value every round puts: the rounds' resident scope keeps
    its single device copy. Returns the offsets of the int32 column ids and
    of the block."""
    var xo = pr.put(x_value)
    var kf = pr.alloc(d)
    var ki = pr.alloc(d)
    pr.stage(OP_C2_RFE_KEEP, d, sup, d, kf, ki)
    var out = pr.alloc(n * k)
    pr.stage(OP_GATHER_COLS, n * k, xo, n, d, kf, k, out)
    return (ki, out)


def _importance_keys(mut pr: ClsProg, est: PythonObject, m: Int, getter: PythonObject, context: PythonObject,
                     cc: PythonObject) raises -> Int:
    """`_stage_importance_keys(pr, est, m, getter)`: the ordered keys
    (c2_key64) of the fitted estimator's m column importances: coef_ squared
    (summed over rows when 2-D) on the device, else feature_importances_ as
    given; a dotted attribute path picks the importances instead (squared).
    Returns the keys' offset."""
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
             cc: PythonObject) raises -> Int:
    """The ordered keys (c2_key64) of m column importances: `values` as
    given, or (`squared`: a coefficient block, 1-D or rows x m) squared and
    summed over its rows on the device. Returns the keys' offset."""
    var key = pr.work(2 * m)
    if not squared:
        var got = _score_words(values, m, context)
        var wo = pr.put_words(got[0])
        pr.stage(OP_C2_KEY64, m, wo, got[1], key)
        return key
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
    var c = admit_x(py_or_none(coef), nd, "coef_", cc)
    var out = pr.work(d)
    var co = pr.put(c)
    pr.stage(OP_SQSUM_COLS, d, co, rows, d, out)
    pr.stage(OP_C2_KEY64, d, out, 0, key)
    return key


# ---------------------------------------------------------------- RFE over a native member
struct RfeMember(Movable):
    """RFE's hold on its estimator as a NATIVE MEMBER (core/member_abi.mojo),
    built once per fit (`_rfe_member`). `native` False: the estimator's
    family has not published a member entry RFE can use, and the rounds keep
    the estimator's own doors (`_fit_clone`, `prep_estimator_attribute`).
    True: `owner.plan` makes one member state per round; `template` is the
    one clone PACK writes the final fit into; `y` is the admitted target
    every round reads by address (float32 values for a regressor kind, the
    int32 class codes of `labels` for a classifier kind: encoded once per
    fit); `block` names the fitted block the importances come from and
    `squared` whether it is squared and summed over its rows.
    `importance_word`: the rounds' fits compute feature importances because
    this door set the family's parameter word for it (the tree and forest
    families, core/forest_member_run.mojo word [10]); the final fit, whose
    importances nobody reads, clears it."""
    var native: Bool
    var owner: MemberOwner
    var template: PythonObject
    var fields: PythonObject
    var member_context: PythonObject
    var y: PythonObject
    var y_address: Int
    var y_dtype: Int
    var labels: PythonObject
    var block: Int
    var squared: Bool
    var importance_word: Bool
    # THE GIL RULE (core/member_abi.mojo): the member's kind has MEMBER_GIL_FREE.
    var gil_free: Bool

    def __init__(out self):
        self.native = False
        self.gil_free = False
        self.owner = MemberOwner()
        self.template = PythonObject(None)
        self.fields = PythonObject(None)
        self.member_context = PythonObject(None)
        self.y = PythonObject(None)
        self.y_address = 0
        self.y_dtype = MEMBER_F32
        self.labels = PythonObject(None)
        self.block = 0
        self.squared = False
        self.importance_word = False


def _rfe_member(estimator: PythonObject, y: PythonObject, n: Int, g: PythonObject, fit_params: PythonObject,
                context: PythonObject, b: PythonObject) raises -> RfeMember:
    """UNPACK, once per RFE fit: admit the estimator as a native member when
    its class is a published member whose KIND fits and exports the block
    the importance getter names ('auto': coef, else feature importances as
    given; 'coef_' / 'feature_importances_': that block, squared, as the
    attribute path squares it), with no fit parameters. A classifier kind
    gets its labels encoded here, once (codes and class count to every
    round's FIT, the labels to the final PACK). Dotted getter paths, an
    admission the family refuses and a target the encoder or the regressor
    admission refuses keep the estimator's own doors (the fallback rule):
    they answer or refuse in their own words."""
    var member = RfeMember()
    comptime if not MEMBER_SET_NATIVE:
        return member^
    if len(fit_params) != 0:
        return member^
    var getter = String(g)
    var auto = getter == "auto"
    if not auto and getter != "coef_" and getter != "feature_importances_":
        return member^
    var plan = prep_estimator_plan(estimator, context)
    if plan is None:
        return member^
    if py_get(plan, "member") is None:
        return member^
    var classifier = Bool(py=plan["classifier"])
    var template = prep_estimator_clone(estimator, context)
    var member_context = plan["context"]
    var fields = py_call(plan["fields_get"], template, py_type(template))
    var owner = member_admit_serving(template, fields, plan, member_context, b, MEMBER_CAN_FIT | MEMBER_CAN_EXPORT)
    if not owner.published:
        return member^
    try:
        if classifier:
            # The labels once, as a classifier's fit door encodes them: the
            # class list it would publish and one int32 code per row.
            var encoded = classical_encode_labels(y, member_context)
            var count = len(encoded.classes)
            if encoded.n != n or count < 2:
                return member^
            owner.plan.classes = count
            member.y = encoded.codes
            member.y_dtype = MEMBER_I32
            member.labels = encoded.classes
        else:
            # The target once, as a regressor member's fit door admits it
            # (present, real, one column, finite, as long as X).
            member.y = classical_check_y(y, n, classical_estimator_name(estimator), member_context)
        # A DecisionTree or RandomForest member computes its feature
        # importances only when its parameter word [10] asks for them
        # (core/forest_member_run.mojo: "a caller that reads importances sets
        # it to 1 in `plan.params` before `member_new`"; its admission writes
        # 0, and the kind's mask has MEMBER_CAN_IMPORTANCES only then). Set
        # before the mask is asked, unless the getter names `coef_`.
        if ((owner.plan.family == MEMBER_FAMILY_TREE or owner.plan.family == MEMBER_FAMILY_FOREST)
                and getter != "coef_" and len(owner.plan.params) > 10):
            owner.plan.params[10] = 1.0
            member.importance_word = True
        # The mask of THIS kind (the family's union also holds its other
        # kinds' bits): one state minted, asked and released, once per fit.
        var mask = member_kind_capabilities(owner.plan)
        if ((mask & MEMBER_CAN_CLASSIFIER) != 0) != classifier:
            return member^
        if (mask & MEMBER_CAN_FIT) == 0 or (mask & MEMBER_CAN_EXPORT) == 0:
            return member^
        if getter != "feature_importances_" and (mask & MEMBER_CAN_COEF) != 0:
            member.block = MEMBER_BLOCK_COEF
            member.squared = True
        elif getter != "coef_" and (mask & MEMBER_CAN_IMPORTANCES) != 0:
            member.block = MEMBER_BLOCK_FEATURE_IMPORTANCES
            member.squared = not auto
        else:
            return member^
        # THE GIL RULE (core/member_abi.mojo): the same answer says whether
        # this kind's fit may run with the GIL released by this door.
        # One fit on this thread either way: released too for a kind whose
        # own door releases around the same work (MEMBER_GIL_DOOR).
        member.gil_free = (mask & (MEMBER_GIL_FREE | MEMBER_GIL_DOOR)) != 0
    except:
        return member^
    member.y_address = Int(py=classical_address(member.y))
    member.owner = owner^
    member.template = template
    member.fields = fields
    member.member_context = member_context
    member.native = True
    return member^


def _rfe_member_fit(member: RfeMember, plan: MemberPlan, block: PythonObject, n: Int, k: Int) raises -> MemberCall:
    """One member state of `plan` (the member's own plan, or the final fit's
    copy of it) fitted on the round's (n, k) float32 host block. THE GIL RULE:
    the GIL is released around the fit only for a kind with `MEMBER_GIL_FREE`;
    any other kind is fitted with the GIL held (its route's entries release
    inside, as for the estimator's own door). The caller frees it (`member_free`) and keeps
    `plan` alive until then (the call holds the address of its words)."""
    var call = member_new(plan)
    var done = False
    try:
        call.x = MemberMatrix(Int(py=classical_address(block)), n, k)
        call.y = MemberMatrix(member.y_address, n, 1, member.y_dtype)
        if member.gil_free:
            var py = Python()
            with GILReleased(py):
                _ = member_call(plan.member, MEMBER_OP_FIT, call)
        else:
            _ = member_call(plan.member, MEMBER_OP_FIT, call)
        done = True
    finally:
        if not done:
            member_free(plan, call)
        _ = block  # keep the Python objects alive through the native call above
        _ = member.y
    return call^


def _rfe_member_words(member: RfeMember, block: PythonObject, n: Int, m: Int,
                      cc: PythonObject) raises -> PythonObject:
    """A round over a native member: fit a new state on the block, read its
    importance block by EXPORT (a view of words the state owns; no Python
    attribute) and release the state. Returns the block as the float32 Array
    `_keys_of` stages: m words, or (rows, m) for a coefficient block with one
    row per class or output. Raises when the member refuses the round; the
    caller then runs the estimator's own doors (nothing was staged)."""
    var call = _rfe_member_fit(member, member.owner.plan, block, n, m)
    var words = PythonObject(None)
    var notices = List[Float64]()
    try:
        member_notices_take(member.owner.plan, call.state, notices)
        call.block = member.block
        call.flags = 0
        _ = member_call(member.owner.plan.member, MEMBER_OP_EXPORT, call)
        var view = call.dst
        var rows = view.rows
        if not view.dense_host(MEMBER_F32) or view.cols != m or rows < 1 or (rows > 1 and not member.squared):
            raise Error("mojolearn: RFE importances are " + String(view.rows) + " x " + String(view.cols)
                        + " words, expected " + String(m) + " columns")
        # rows * m words, copied once into the Array the round's program
        # puts; the view dies with the state below.
        if rows == 1:
            words = f32_array(m, cc)
        else:
            words = classical_typed(Python.tuple(rows, m), "<f4", cc["output_context"])
        unsafe_memcpy(dest=_ptr(words), src=F32P(unsafe_from_address=view.data), count=rows * m)
    finally:
        member_free(member.owner.plan, call)
    # The round's fit notices, as the estimator's own fit of the round raised them.
    member_fit_notices_raise(notices, member.member_context)
    return words


def _rfe_member_final(member: RfeMember, block: PythonObject, n: Int, k: Int, b: PythonObject) raises -> PythonObject:
    """The final estimator over a native member: fitted on the selected
    block, its state written into the clone (`member_pack`, a classifier
    with its labels: the attributes its own fit writes, so RFE's predict /
    score / transform read it as before) and released."""
    # The final fit is the estimator's plain fit: the importance word this
    # door set for the rounds is cleared (nobody reads the final fit's
    # importances, and computing them costs a column-major copy of the block).
    var plan = member.owner.plan.copy()
    if member.importance_word:
        plan.params[10] = 0.0
    var call = _rfe_member_fit(member, plan, block, n, k)
    var notices = List[Float64]()
    try:
        member_notices_take(plan, call.state, notices)
        if member.labels is None:
            member_pack(member.template, member.fields, member.member_context, b, plan, call.state)
        else:
            member_pack_labeled(member.template, member.fields, member.member_context, b, plan,
                                call.state, member.labels)
    finally:
        member_free(plan, call)
    member_fit_notices_raise(notices, member.member_context)
    return member.template


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
    """The elimination rounds and the final fit of `_rfe_fit`, inside its
    resident scope."""
    var nk = prep_keys()
    var b = native_loaded_module("builtins")
    var n = arr.n
    var d = arr.d
    var mode = cls_mode(cc)
    # n_features_to_select: None -> d // 2; a float in (0, 1) -> int(nsel * d); then max(1, int(nsel))
    var nsel_arg = py_item(fields, nk[].k_n_features_to_select)
    var nsel = 0
    if nsel_arg is None:
        nsel = d // 2
    elif Bool(py=py_call(b.isinstance, nsel_arg, b.float)) and Float64(py=nsel_arg) > 0.0 and Float64(py=nsel_arg) < 1.0:
        nsel = Int(Float64(py=nsel_arg) * Float64(d))
    else:
        nsel = Int(py=py_call_typed(b.int, nsel_arg))
    nsel = max(1, nsel)
    # step: a float below 1 -> int(max(1, step * d)); else int(step)
    var step_arg = py_item(fields, nk[].k_step)
    var step = 0
    if Bool(py=py_call(b.isinstance, step_arg, b.float)) and Float64(py=step_arg) < 1.0:
        var v = Float64(py=step_arg) * Float64(d)
        step = 1 if v <= 1.0 else Int(v)
    else:
        step = Int(py=py_call_typed(b.int, step_arg))
    if step <= 0:
        raise Error("mojolearn: step must be > 0")
    # One program per elimination round (lane w1-readmission-rfe): the
    # fitted estimator's importance keys, c2_rfe_step on the support mask,
    # c2_rfe_rank, then the next block staged from the updated mask
    # (c2_rfe_keep + gather_cols). Program 0 stages the first block from the
    # all-ones mask. The support, ranking and column ids ride each program as
    # its in-out and output words (d words each); the host reads the support
    # once, after the last round. Each round drops exactly min(step, nsup -
    # nsel) columns (c2_rfe_step's stable rank is a total order), so the
    # round count and every block width are known before the program runs.
    # Same stages on the same words as the two programs per round they
    # replace: no bit moves.
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
    var sup0 = pr0.put(support)
    var first = _stage_block(pr0, arr.value, n, d, sup0, d)
    pr0.run(mode, cc)
    var feat = pr0.get_i32(first[0], d, cc)
    var block = pr0.get(first[1], n, d, cc)
    var nsup = d
    # The estimator as a native member (admitted once for the whole fit) when
    # its family has published what RFE reads; else its own doors per round.
    # The rounds are serial by nature: a round's block is chosen from the
    # previous round's importances.
    var member = _rfe_member(estimator, y, n, g, fit_params, context, b)
    while nsup > nsel:  # driver-loop: over elimination rounds
        var drop = min(step, nsup - nsel)
        var k = nsup - drop
        var pr = ClsProg()
        var key = 0
        # A member that refuses a round (a view or dtype its family does not
        # read, a fit failure) is dropped for the rest of the fit: this round
        # and the later ones run the estimator's own doors, which raise the
        # estimator's own words if the request is wrong (the fallback rule).
        var words = PythonObject(None)
        if member.native and classical_is_array(block, cc):
            try:
                words = _rfe_member_words(member, block, n, nsup, cc)
            except:
                member.native = False
        if words is not None:
            key = _keys_of(pr, words, member.squared, nsup, context, cc)
        else:
            _rfe_hold_block(block, hold, mark, cc)
            var est = _fit_clone(estimator, block, y, fit_params, context)
            key = _importance_keys(pr, est, nsup, g, context, cc)
        var sup = pr.put(support, True)
        var rk = pr.put(ranking, True)
        var fo = pr.put_words(feat)
        pr.stage(OP_C2_RFE_STEP, nsup, key, fo, nsup, drop, sup)
        pr.stage(OP_C2_RFE_RANK, d, sup, rk)
        var nxt = _stage_block(pr, arr.value, n, d, sup, k)
        pr.run(mode, cc)
        support = pr.get(sup, d, 0, cc)
        ranking = pr.get(rk, d, 0, cc)
        feat = pr.get_i32(nxt[0], k, cc)
        block = pr.get(nxt[1], n, k, cc)
        nsup = k
    var final = PythonObject(None)
    if member.native and classical_is_array(block, cc):
        try:
            final = _rfe_member_final(member, block, n, nsup, b)
        except:
            member.native = False
    if final is None:
        _rfe_hold_block(block, hold, mark, cc)
        final = _fit_clone(estimator, block, y, fit_params, context)
    _ = member.native  # the member's pins outlive every member call above
    var mask_list = Python.list()
    var kept = List[Int]()
    var fs = _ptr(support)
    for j in range(d):  # driver-loop: over the fitted support (feature count)
        _ = py_call(mask_list.append, PythonObject(fs[j] != Float32(0)))
        if fs[j] != Float32(0):
            kept.append(j)
        _ = support  # keep the Python object alive through the native call above
    py_setitem(fields, nk[].k_estimator_, final)
    py_setitem(fields, nk[].k__mask, mask_list)
    # Lane p3-msel-parallel-and-process-pool (audit B: RFE redid its support
    # and member admission on every predict): the kept columns as the gather's
    # index words and the admitted member, stored once here. A predict reads
    # them while `_mask` and `estimator_` are still the objects stored here
    # (identity), else it falls back to reading the mask and admitting.
    py_setitem(fields, nk[].k__support_words_, _words(kept, context))
    py_setitem(fields, nk[].k__support_k_, PythonObject(len(kept)))
    py_setitem(fields, nk[].k__support_mask_, mask_list)
    py_setitem(fields, nk[].k__estimator_admitted_, final)
    py_setitem(fields, nk[].k_support_, py_call(b.list, mask_list))
    # ranking.astype("<i8")
    var ranks = classical_typed(Python.tuple(d), "<i8", py_item(cc, nk[].k_output_context))
    var dst = I64P(unsafe_from_address=Int(py=classical_address(ranks)))
    var src = _ptr(ranking)
    for j in range(d):  # driver-loop: over the fitted ranking (feature count)
        dst[j] = Int64(Int(src[j]))
        _ = ranks  # keep the Python object alive through the native call above
        _ = ranking  # keep the Python object alive through the native call above
    py_setitem(fields, nk[].k_ranking_, ranks)
    py_setitem(fields, nk[].k_n_features_, nsup)
    py_setitem(fields, nk[].k_numeric_mode_, mode)
    py_setitem(fields, nk[].k_n_features_in_, d)


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
