# SPDX-License-Identifier: Apache-2.0
# The native owner of IterativeImputer (dispatched by bindings/prep_owner_native.mojo).
# fit, fit_transform and transform are each one call here: UNPACK (the
# parameter refusals, the selection, X, the initial SimpleImputer's words and,
# for transform, the fitted fields and the addresses of the fitted vectors),
# the typed RUN (x_prep/iterimp_run.mojo `iterimp_fit` / `iterimp_transform`:
# the initial fill, the statistics and orders, the n_nearest_features draws,
# every round of the built-in Bayesian ridge in one program, the indicator),
# then PACK (the fitted attributes, read from the arenas the runner holds).
# A user estimator whose class is a native member (core/member_abi.mojo; lane
# mf-iterimp, 2026-11-23) is admitted ONCE per fit (`_member`: its family's
# member entry, the kind's mask) and its rounds are the typed RUN
# x_prep/iterimp_run.mojo `iterimp_member_rounds` (per step: the gather
# program, the member's FIT on the gathered rows and PREDICT, or PREDICT_STD
# and the mean for sample_posterior, then the scatter program), the member
# calls with the GIL released for a kind that allows it; PACK writes each
# step's state into a clone of the estimator (`member_pack`: the attributes
# its own fit writes) for `imputation_sequence_`, HANDS the state to that
# clone (`member_held_adopt`, lane mf-iterimp2) and raises the fits'
# notices. transform with such an estimator (lane mf-iterimp2) reads each
# fitted step's held state once (`_held_steps`: a hit is a record read, no
# import) and runs `iterimp_member_transform` (per step the gather program,
# the member's PREDICT / PREDICT_STD on the held state, the scatter
# program). sample_posterior goes through the member route for the
# families whose PREDICT_STD also answers the same query's mean
# (`_posterior_family`: GaussianProcessRegressor, BayesianRidge,
# ARDRegression; these are the only plans that serve predict_std, so
# `_refuse` refuses sample_posterior for every other estimator first).
# Lane mf-iterimp3: a CLASSIFIER member is served too (each step's classes
# and codes made natively, the labels published as the classifier's own fit
# publishes them, kept as `_step_classes` for transform); fewer than two
# kept features is scikit-learn's edge case (n_iter_ = 0, the initial
# imputation, no estimator fitted: `iterimp_initial_only`); a kept column
# with no observed value is marked not missing as scikit-learn marks it
# (`_empty_kept`; every route). Lane mf-iterimp3b: every explicit Host*
# class whose public class has a member mark carries its own
# (python/mojolearn/_classical_host.py), so HostLinearRegression / HostRidge
# run as linear members on their CPU context.
# Lane mf-iterimp7 (2026-11-23): the member rounds are the ONLY rounds of a
# supplied estimator. The door rounds (a clone, a fit and a predict of the
# estimator's compiled plan per step, `_ii_take` / `_ii_put` / `_ii_stop`,
# and the same over the fitted steps in transform) are deleted. `_member`
# admits with the family's RAISING admission (the words its own fit door
# raises, in its order, since the round-6 admissions: lanes mf-*6), so a
# setting the family refuses raises those words here, where the door
# rounds' first step fit raised them. LabelPropagation / LabelSpreading are
# admitted with deferred labels and each step writes the code of its -1
# class into the marker word (`iterimp_marker_code`), as model selection's
# units do. A class with no fit or predict (its member kind serves neither,
# or a connected class with no mark and no predict) is refused in
# scikit-learn's own words (its parameter check: `estimator` must implement
# fit and predict). transform reads every step's held member state and
# raises when one is gone (`_held_steps`).
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.member_boundary_door import member_admit_labeled, member_pack, member_pack_labeled, member_state
from bindings.member_class_weight_unpack import member_cw_fill_keyed
from bindings.member_fit_notices_pack import member_fit_notices_raise
from bindings.member_owner_unpack import MemberOwner, member_mark
from bindings.classifier_public_lifecycle_door import _public_labels
from providers.python.member_xneighbors_boundary_door import xneighbors_kind_reads_marker
from core.xneighbors_member import LP_MARKER_WORD
from core.member_abi import (
    MemberPlan, MemberCall, member_call, member_kind_capabilities, member_call_of, member_free,
    member_output_dtype,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_PREDICT, MEMBER_I32,
    MEMBER_CAN_FIT, MEMBER_CAN_PREDICT, MEMBER_CAN_PREDICT_STD, MEMBER_CAN_CLASSIFIER, MEMBER_BORROWS_FIT,
    MEMBER_GIL_FREE, MEMBER_GIL_DOOR, MEMBER_CAN_IMPORT,
)
from core.member_families import MEMBER_FAMILY_GP, MEMBER_FAMILY_XLINEAR
from bindings.member_held_door import MemberHeld, member_held, member_held_adopt, member_held_release
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem, py_contains
from bindings.classical_arrays_door import classical_typed, classical_address
from bindings.scaler_lifecycle_native_door import scaler_entries
from bindings.x_prep_native_owner_door import xprep_values, xprep_storage, xprep_run_records
from bindings.x_prep_native_owner_run import XprepInput, XprepCaps, XPREP_OWNER_NATIVE, xprep_r3
from bindings.user_objects_native_door import user_objects_call
from bindings.prep_stateless_native_door import _int_like, _repr, _name, _fam
from bindings.prep_native_estimator_door import (
    prep_estimator_admit, prep_estimator_clone, prep_estimator_has, prep_estimator_plan)
from bindings.prep_encode_prog_run import EncProgram
from bindings.prep_encode_imputer_run import (
    ImputerFit, ImputerApply, IMP_MEAN, IMP_MEDIAN, IMP_MOST_FREQUENT, IMP_CONSTANT)
from bindings.prep_encode_native_door import (
    enc_caps, IDN_STATS_BLOCKED, _precheck, _missing_mark, _qselect, _real_like, _fitted_flat,
    _default_mode, _take1, _take2, _seed64, _truth, _float, _is_str,
    _check_width, _int_list, _index_list)
from bindings.prep_owner_route_unpack import prep_route_context
from bindings.prep_keys_unpack import (
    prep_keys,
    PREP_OPN_FIT, PREP_OPN_FIT_TRANSFORM, PREP_OPN_TRANSFORM,
)
from bindings.py_balanced import py_item
from x_prep.iterimp_run import (
    IterImpRunner, IterImpFit, IterImpFitted, IterImpApply, IterImpStep, iterimp_fit, iterimp_prepare,
    iterimp_transform, II_NONE, II_ASCENDING, II_DESCENDING, II_ROMAN, II_ARABIC, II_RANDOM,
    IterImpMember, IterImpMemberFit, iterimp_member_rounds, iterimp_member_free, IterImpHeldStep,
    IterImpApplied, iterimp_member_transform, iterimp_initial_only, iterimp_marker_code)

comptime F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime F64P = MutPointer[Float64, MutUntrackedOrigin]


# ---------------------------------------------------------------- small helpers
def _b() raises -> PythonObject:
    return native_loaded_module("builtins")


def _u64_of(value: PythonObject) raises -> UInt64:
    # A Python int (already reduced mod 2**64) as its 64-bit word.
    var words = _seed64(value)
    return (UInt64(words[1]) << 32) | UInt64(words[0])


def _u64_obj(value: UInt64) raises -> PythonObject:
    # The 64-bit word as a nonnegative Python int.
    return py_call(_b().int, PythonObject(String(value)))


def _floats_from(value: PythonObject) raises -> List[Float64]:
    """`_user_objects.call("floats", v)`: float(x) of every entry."""
    var words = user_objects_call(Python.tuple("floats", value))
    var n = len(words)
    var src = F64P(unsafe_from_address=Int(py=py_call(words.buffer_info)[0]))
    # One native copy of the float64 words (no per-entry append).
    var out = List[Float64](length=n, fill=0.0)
    if n > 0:
        unsafe_memcpy(dest=F64P(unsafe_from_address=Int(out.unsafe_ptr())), src=src, count=n)
    _ = words  # keep the Python object alive through the native copy above
    return out^


# ---------------------------------------------------------------- parameters
# `_refuse`'s sample_posterior words; the member admission raises the same (lane mf-ens6b).
comptime _POSTERIOR_WORDS = "mojolearn: If 'sample_posterior' is True, the estimator must support 'return_std' in its 'predict' method."


def _has_methods_words(estimator: PythonObject) raises -> String:
    """scikit-learn's parameter check of `estimator` (sklearn/impute/
    _iterative.py `_parameter_constraints`: `[None, HasMethods(["fit",
    "predict"])]`, raised by `_validate_params` as InvalidParameterError, a
    ValueError), for a class that fits or predicts nothing as a member (lane
    mf-iterimp7). The value is named by its class, not its repr: a repr would
    call the estimator's Python `__repr__`."""
    return ("ValueError: The 'estimator' parameter of IterativeImputer must be None or an object implementing "
            + "'fit' and 'predict'. Got a " + String(py_type(estimator).__name__) + " instance instead.")


def _refuse(model: PythonObject, context: PythonObject) raises:
    var nk = prep_keys()
    # `_refuse`: the parameter refusals, before X is read.
    var b = _b()
    var fields = py_fields(model)
    var estimator = py_item(fields, nk[].k_estimator)
    if estimator is not None:
        # A user estimator is a mojolearn estimator with a connected native
        # plan (its clone, fit and predict are compiled operations); any other
        # class is refused by name first.
        prep_estimator_admit(estimator, "IterativeImputer", context)
        # A connected class with no member mark and no predict (Cholesky,
        # NearestNeighbors, OPTICS, the manifold embeddings): scikit-learn's
        # parameter check refuses it first, in its words (lane mf-iterimp7).
        var plan = prep_estimator_plan(estimator, context)
        if py_get(plan, "member") is None and not prep_estimator_has(estimator, "predict", context):
            raise Error(_has_methods_words(estimator))
    var mi = py_item(fields, nk[].k_max_iter)
    if Bool(py=py_call(b.isinstance, mi, b.bool)) or not _int_like(mi, context) or Int(py=py_call(b.int, mi)) < 0:
        raise Error("The 'max_iter' parameter of IterativeImputer must be an int in the range [0, inf). Got "
                    + _repr(mi) + " instead.")
    var order = py_item(fields, nk[].k_imputation_order)
    var text = String(order) if _is_str(order) else String("")
    if text != "ascending" and text != "descending" and text != "roman" and text != "arabic" and text != "random":
        raise Error("mojolearn: invalid imputation_order " + _repr(order))
    var nnf = py_item(fields, nk[].k_n_nearest_features)
    if nnf is not None and (not _int_like(nnf, context) or Int(py=py_call(b.int, nnf)) < 1):
        raise Error("mojolearn: n_nearest_features must be an int >= 1; got " + _repr(nnf))
    if (_truth(py_item(fields, nk[].k_sample_posterior)) and estimator is not None
            and not prep_estimator_has(estimator, "predict_std", context)):
        # The native plan declares the supported return_std operation;
        # inspecting/calling the estimator's Python predict is unnecessary.
        raise Error(_POSTERIOR_WORDS)


def _per_bound(v: PythonObject, d: Int) raises -> List[Float64]:
    # `_bounds.per`: a list, tuple or array-like as given, else v for every column.
    var b = _b()
    var vals = PythonObject(None)
    if Bool(py=py_call(b.isinstance, v, Python.tuple(b.list, b.tuple))) or Bool(py=py_call(b.hasattr, v, "tolist")):
        vals = py_call(b.list, v)
    else:
        vals = Python.list()
        for _ in range(d):  # driver-loop: over the columns (one bound argument each)
            _ = py_call(vals.append, v)
    var out = List[Float64]()
    for i in range(len(vals)):  # driver-loop: over the columns' min_value / max_value parameters (one per feature)
        out.append(_float(vals[i]))
    return out^


def _float_list(values: List[Float64]) raises -> PythonObject:
    var out = Python.list()
    for i in range(len(values)):  # driver-loop: over fitted per-column bounds
        _ = py_call(out.append, PythonObject(values[i]))
    return out


def _bounds_of(model: PythonObject) raises -> List[Float64]:
    var nk = prep_keys()
    # The fitted `_bounds_k` list.
    var items = py_call(_b().list, py_item(py_fields(model), nk[].k__bounds_k))
    var out = List[Float64]()
    for i in range(len(items)):  # driver-loop: over fitted per-column bounds
        out.append(_float(items[i]))
    return out^


def _nbl_of(value: PythonObject) raises -> List[Int]:
    return _index_list(value)


# ---------------------------------------------------------------- the runner
struct _DoorRunner(IterImpRunner):
    """The RUN's programs over the x_prep transport of the operation's image
    (`xprep_run_records`: UNPACK of X's record, the typed run with the GIL
    released, cleanup). Owner 0 is X as `xprep_values` converted it; a
    RUN-held or door-held native buffer (owner k >= 1) is an
    XPREP_OWNER_NATIVE record at `seconds[k - 1]`. Each arena (and output
    store) is held until `release` or the runner's end; PACK reads them
    with `_take1` / `_take2` (`ran`). `release_members`: a supplied
    estimator's member operations run with the GIL released (its kind has
    MEMBER_GIL_FREE or MEMBER_GIL_DOOR; set by the door after `_member`).
    It runs every typed RUN of this owner (the built-in ridge, the typed
    prologue and the member rounds); it never held a door-round step, so the
    door rounds' deletion (lane mf-iterimp7) leaves it whole."""
    var owners: List[PythonObject]
    var entries: PythonObject
    var context: PythonObject
    var caps: XprepCaps
    var held: List[PythonObject]
    var addresses: List[Int]
    var release_members: Bool

    def __init__(out self, values: PythonObject, entries: PythonObject, context: PythonObject, caps: XprepCaps):
        self.owners = List[PythonObject]()
        self.owners.append(values)
        self.entries = entries
        self.context = context
        self.caps = caps
        self.held = List[PythonObject]()
        self.addresses = List[Int]()
        self.release_members = False

    def member(mut self, plan: MemberPlan, operation: Int, mut call: MemberCall) raises -> Int:
        # THE GIL RULE (core/member_abi.mojo): released around the one call
        # for a kind that allows it (every argument is a typed word, read
        # before the region), else the call runs with the GIL held.
        if not self.release_members:
            return member_call(plan.member, operation, call)
        var result = 0
        var py = Python()
        with GILReleased(py):
            result = member_call(plan.member, operation, call)
        return result

    def run(mut self, mut prog: EncProgram, seconds: List[Int]) raises -> Int:
        prog.check_bound()
        var host = prog.host
        var scratch = prog.scratch_size
        var out_words = prog.out_words
        var device_scratch = scratch > 0 and (self.caps.has_out or self.caps.has_scratch)
        var arena = xprep_storage(host + (0 if device_scratch else scratch), self.context)
        var store = xprep_storage(out_words, self.context) if out_words > 0 else PythonObject(None)
        var records = List[XprepInput](capacity=len(prog.inputs))
        for i in range(len(prog.inputs)):  # driver-loop: over the program's input records
            var record = prog.inputs[i]
            if record.owner >= 1:
                records.append(XprepInput(record.offset, record.words, seconds[record.owner - 1], XPREP_OWNER_NATIVE))
            else:
                records.append(record)
        xprep_run_records(arena, host, scratch, store, out_words, prog.stage_words(), records, self.owners,
                          self.entries, self.context, out_words, prog.inout_ranges, self.caps)
        self.held.append(Python.tuple(arena, store))
        self.addresses.append(Int(py=arena[1]))
        return len(self.held) - 1

    def words(self, handle: Int) -> Int:
        return self.addresses[handle]

    def release(mut self, handle: Int):
        self.held[handle] = PythonObject(None)
        self.addresses[handle] = 0


def _f32_array(values: List[Float32], context: PythonObject) raises -> PythonObject:
    # A fitted float32 vector (`_Prog.get(off, count)`) from the RUN's copy.
    var nk = prep_keys()
    var out = classical_typed(Python.tuple(len(values)), "<f4", py_item(context, nk[].k_output_context))
    if len(values) > 0:
        unsafe_memcpy(dest=F32P(unsafe_from_address=Int(py=classical_address(out))),
                      src=F32P(unsafe_from_address=Int(values.unsafe_ptr())), count=len(values))
    _ = values  # the RUN's copy stays alive through the native copy above
    return out


# ---------------------------------------------------------------- fit: UNPACK
def _initial(model: PythonObject, context: PythonObject) raises -> PythonObject:
    """The initial `SimpleImputer(missing_values, strategy=initial_strategy,
    fill_value, keep_empty_features)`: a fresh instance holding
    SimpleImputer.__init__'s six parameters and this estimator's `device`
    (None for an instance saved before the parameter existed), its route
    resolved for this operation's context, its parameter refusals raised.
    The RUN fits it; PACK stores its fitted fields."""
    var nk = prep_keys()
    var fields = py_fields(model)
    var cls = py_item(py_item(context, nk[].k_prep_owner_classes), nk[].k_SimpleImputer)
    var imputer = py_call(_b().object.__new__, cls)
    var own = py_fields(imputer)
    py_setitem(own, nk[].k_missing_values, py_item(fields, nk[].k_missing_values))
    py_setitem(own, nk[].k_strategy, py_item(fields, nk[].k_initial_strategy))
    py_setitem(own, nk[].k_fill_value, py_item(fields, nk[].k_fill_value))
    py_setitem(own, nk[].k_copy, PythonObject(True))
    py_setitem(own, nk[].k_add_indicator, PythonObject(False))
    py_setitem(own, nk[].k_keep_empty_features, py_item(fields, nk[].k_keep_empty_features))
    py_setitem(own, nk[].k_device, py_get(fields, nk[].k_device))
    _ = prep_route_context(imputer, context)
    _precheck(imputer, "simple_imputer", context)
    return imputer


def _initial_request(imputer: PythonObject, n: Int, d: Int, mode: PythonObject, entries: PythonObject,
                     context: PythonObject) raises -> ImputerFit:
    # `_imputer_fit`'s UNPACK of the initial imputer's parameters (X is owner 0).
    var fields = py_fields(imputer)
    var strategy_field = fields["strategy"]
    var name = String(strategy_field) if _is_str(strategy_field) else String("")
    var strategy = IMP_MEAN
    if name == "median":
        strategy = IMP_MEDIAN
    elif name == "most_frequent":
        strategy = IMP_MOST_FREQUENT
    elif name == "constant":
        strategy = IMP_CONSTANT
    elif name != "mean":
        raise Error("mojolearn: SimpleImputer strategy " + _repr(strategy_field) + " is not valid")
    var konst = strategy == IMP_CONSTANT
    var fill_value = fields["fill_value"]
    if konst and fill_value is not None and not _real_like(fill_value, context):
        raise Error("TypeError: mojolearn: SimpleImputer fill_value must be numeric")
    var mark = _missing_mark(fields["missing_values"])
    var fill = 0.0
    if konst and fill_value is not None:
        fill = _float(fill_value)
    return ImputerFit(n, d, 0, mark[0], mark[1], strategy,
                      strategy == IMP_MEDIAN and _qselect(mode, entries, context),
                      (_fam(mode, entries) & IDN_STATS_BLOCKED) != 0, fill, _truth(fields["keep_empty_features"]))


def _order_code(fields: PythonObject) raises -> Int:
    var nk = prep_keys()
    var name = String(py_item(fields, nk[].k_imputation_order))  # admitted by `_refuse`
    if name == "descending":
        return II_DESCENDING
    if name == "roman":
        return II_ROMAN
    if name == "arabic":
        return II_ARABIC
    if name == "random":
        return II_RANDOM
    return II_ASCENDING


def _fit_request(model: PythonObject, values: PythonObject, init: ImputerFit, mode: PythonObject,
                 entries: PythonObject, caps: XprepCaps) raises -> IterImpFit:
    # UNPACK: this estimator's parameters, read once into the typed request.
    var nk = prep_keys()
    var b = _b()
    var fields = py_fields(model)
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    var mark = _missing_mark(py_item(fields, nk[].k_missing_values))
    var nnf_field = py_item(fields, nk[].k_n_nearest_features)
    var nnf = II_NONE if nnf_field is None else Int(py=py_call(b.int, nnf_field))
    return IterImpFit(init.copy(), n, d, mark[0], mark[1], (_fam(mode, entries) & IDN_STATS_BLOCKED) != 0,
                      xprep_r3("work"), caps.has_out, _order_code(fields),
                      _truth(py_item(fields, nk[].k_skip_complete)), _truth(py_item(fields, nk[].k_sample_posterior)),
                      _truth(py_item(fields, nk[].k_add_indicator)), Int(py=py_call(b.int, py_item(fields, nk[].k_max_iter))),
                      nnf, _float(py_item(fields, nk[].k_tol)), _u64_of(py_item(fields, nk[].k_random_state)),
                      _per_bound(py_item(fields, nk[].k_min_value), d), _per_bound(py_item(fields, nk[].k_max_value), d))


# ---------------------------------------------------------------- fit: PACK
def _pack_initial(model: PythonObject, imputer: PythonObject, fitted: IterImpFitted, mode: PythonObject, d: Int,
                  context: PythonObject) raises:
    """The initial imputer's fitted fields (`_imputer_fit`'s PACK: one object
    for statistics_ and _fill when they are the same values), then this
    estimator's fields that the prologue decided."""
    var nk = prep_keys()
    var own = py_fields(imputer)
    var statistics = _f32_array(fitted.statistics, context)
    py_setitem(own, "statistics_", statistics)
    py_setitem(own, "_fill", statistics if fitted.shared else _f32_array(fitted.fill, context))
    py_setitem(own, "_keep", _int_list(fitted.keep))
    py_setitem(own, "_indicator", _int_list(List[Int]()))
    py_setitem(own, "numeric_mode_", mode)
    py_setitem(own, "n_features_in_", d)
    var fields = py_fields(model)
    py_setitem(fields, nk[].k_initial_imputer_, imputer)
    py_setitem(fields, nk[].k__keep, _int_list(fitted.keep))
    # the kept positions with no observed value at fit, which transform marks
    # not missing as scikit-learn's `_is_empty_feature` does
    py_setitem(fields, "_empty_kept", _int_list(fitted.empty))
    py_setitem(fields, nk[].k__bounds_k, _float_list(fitted.bounds))
    py_setitem(fields, nk[].k__indicator, _int_list(fitted.indicator))
    py_setitem(fields, nk[].k_n_features_with_missing_, PythonObject(fitted.n_missing))
    py_setitem(fields, nk[].k_numeric_mode_, mode)
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(d))


def _nbl_obj(has: Bool, nbl: List[Int]) raises -> PythonObject:
    if not has:
        return PythonObject(None)
    return _int_list(nbl)


# ---------------------------------------------------------------- a supplied estimator as a member
struct _Member(Movable):
    """The supplied estimator's hold as a native member, built once per fit
    (`_member`): `owner` (its plan and the pins that keep its images loaded,
    alive through PACK), `template` the first step's clone (the admission
    read its fields) and `fields` its instance dictionary, `member_context`
    the family's context, `fields_get` the plan's field reader (for the
    other steps' clones), `plan_key` the class's import-time plan (the
    held-state record's key), `borrows` / `release` the kind's
    MEMBER_BORROWS_FIT and GIL bits, `classifier` a classifier kind whose
    PREDICT writes codes of `code_dtype` (lane mf-iterimp3); `cw_refusal`
    the estimator's own words for a class_weight it refuses (lane mf-left4:
    raised by the RUN at its first classifier fit, where the estimator's own
    first fit raised them); `marker_word` the plan word of the -1 marker's
    code for LabelPropagation / LabelSpreading (-1: a kind without one; lane
    mf-iterimp7)."""
    var owner: MemberOwner
    var plan_key: PythonObject
    var template: PythonObject
    var fields: PythonObject
    var member_context: PythonObject
    var fields_get: PythonObject
    var borrows: Bool
    var release: Bool
    var classifier: Bool
    var code_dtype: Int
    var cw_refusal: String
    var marker_word: Int

    def __init__(out self):
        self.cw_refusal = String("")
        self.owner = MemberOwner()
        self.plan_key = PythonObject(None)
        self.template = PythonObject(None)
        self.fields = PythonObject(None)
        self.member_context = PythonObject(None)
        self.fields_get = PythonObject(None)
        self.borrows = False
        self.release = False
        self.classifier = False
        self.code_dtype = MEMBER_I32
        self.marker_word = -1


def _posterior_family(family: Int) -> Bool:
    """The families whose PREDICT_STD writes the same query's mean to
    `call.y` when given (core/gp_member.mojo, core/xlinear_member.mojo): the
    pair a sample_posterior step draws from, in one member call."""
    return family == MEMBER_FAMILY_GP or family == MEMBER_FAMILY_XLINEAR


def _member(estimator: PythonObject, posterior: Bool, dk: Int, context: PythonObject) raises -> _Member:
    """UNPACK, once per fit: the estimator as a native member, the only
    route of a supplied estimator (lane mf-iterimp7: the door rounds are
    deleted). Its KIND fits and predicts one target (a regressor, or a
    classifier: lane mf-iterimp3, fitted on each step's codes as
    scikit-learn fits it on the float target; the kind's own capability bit
    says which, so a meta kind over a classifier reads codes whatever its
    class's `_estimator_type`), and for sample_posterior one whose kind
    publishes PREDICT_STD answering the same query's mean too
    (`_posterior_family`: GaussianProcessRegressor, BayesianRidge,
    ARDRegression).

    Raises, in this order: the family's admission words (its own fit door's
    refusal of a setting, raised natively since the round-6 admissions,
    where the door rounds' first step fit raised them); the pre-mark words
    for a class whose family admits nothing for it (a predict-only saved
    model such as HostGBDT, whose clone holds no model); `_refuse`'s
    posterior words for a kind with no posterior pair; scikit-learn's
    parameter words for a kind that does not fit or predict."""
    var out = _Member()
    # TOMBSTONE: MOJOLEARN_MEMBER_SET_OFF (plumbing) deleted 2026-11-23 by lane/ab-prep-classical; code recoverable at 4d76a9e16e.
    var b = _b()
    var name = String(py_type(estimator).__name__)
    var plan = prep_estimator_plan(estimator, context)
    if plan is None or py_get(plan, "member") is None:
        # `_refuse` admitted the class and refused one without predict;
        # every connected class with a predict carries a member mark.
        raise Error("TypeError: mojolearn: IterativeImputer requires a mojolearn estimator; got " + name)
    var template = prep_estimator_clone(estimator, context)
    var member_context = plan["context"]
    var fields_get = plan["fields_get"]
    var fields = py_call(fields_get, template, py_type(template))
    # LabelPropagation / LabelSpreading read -1 in y as unlabeled: admitted
    # with deferred labels (an empty list: the codes 0 .. classes-1), each
    # step writes its -1 class's code into the marker word (RUN), as model
    # selection's units do after their encode (lane mf-meta5).
    var mark = member_mark(plan, b)
    var marker = xneighbors_kind_reads_marker(mark.family, mark.kind)
    var labels = Python.list() if marker else PythonObject(None)
    # The family's admission, raising its own fit door's words.
    var owner = member_admit_labeled(template, fields, plan, member_context, b, labels)
    if not owner.published:
        raise Error("TypeError: mojolearn: IterativeImputer requires a mojolearn estimator; got " + name)
    # The mask of THIS kind (the family's union holds its other kinds'
    # bits): one state minted, asked and released, once per fit.
    var mask = member_kind_capabilities(owner.plan)
    var classifier = (mask & MEMBER_CAN_CLASSIFIER) != 0
    if posterior and (classifier or not _posterior_family(owner.plan.family)
                      or (mask & MEMBER_CAN_PREDICT_STD) == 0):
        raise Error(_POSTERIOR_WORDS)  # `_refuse`'s words (lane mf-ens6b)
    if (mask & (MEMBER_CAN_FIT | MEMBER_CAN_PREDICT)) != (MEMBER_CAN_FIT | MEMBER_CAN_PREDICT):
        raise Error(_has_methods_words(estimator))
    # A classifier's class_weight dict is keyed by the step's values, read
    # into the KEYED tail here (lane mf-adm3); each step makes its table over
    # its classes (x_prep/iterimp_run.mojo `_member_step`). A value or a key
    # the estimator refuses leaves its words for the RUN's first classifier
    # fit (lane mf-left4). Every family whose tail can be pending reads a
    # dict (bindings/member_class_weight_unpack.mojo `_dict_values`), so the
    # False answer is unreachable; it is refused rather than served wrong.
    var refusal = String("")
    var complete = True
    try:
        complete = member_cw_fill_keyed(owner, template, fields, member_context, b)
    except error:
        refusal = String(error)
    if not complete:
        raise Error("TypeError: mojolearn: IterativeImputer cannot read the class_weight of " + name)
    if classifier:
        # The element type of its predicted codes, asked once (two classes
        # stand in for the steps' counts: the type is the kind's).
        var two = owner.plan.copy()
        two.classes = 2
        out.code_dtype = member_output_dtype(two, MEMBER_OP_PREDICT, max(dk - 1, 1), 1, MEMBER_I32)
        out.classifier = True
        if marker and len(owner.plan.params) > LP_MARKER_WORD:
            out.marker_word = LP_MARKER_WORD
    out.cw_refusal = refusal
    out.borrows = (mask & MEMBER_BORROWS_FIT) != 0
    out.release = (mask & (MEMBER_GIL_FREE | MEMBER_GIL_DOOR)) != 0
    out.owner = owner^
    out.plan_key = plan
    out.template = template
    out.fields = fields
    out.member_context = member_context
    out.fields_get = fields_get
    return out^


def _member_pack(estimator: PythonObject, member: _Member, mut fit: IterImpMemberFit, labels: PythonObject,
                 context: PythonObject) raises -> PythonObject:
    """PACK of the member rounds: `imputation_sequence_`, one (feat_idx,
    neighbor_feat_idx, estimator) triple per finished step, each estimator a
    clone of the supplied one (the first is the admission's) with the step's
    member state written into it (`member_pack`: the attributes its own fit
    writes). One clone per step, as scikit-learn clones per step. Each step's state
    is then HANDED to its estimator (`member_held_adopt`, lane mf-iterimp2):
    transform predicts through it with no import; its word in `fit.states`
    becomes 0 so `iterimp_member_free` leaves it to the estimator. A
    classifier step's estimator gets its labels as its own fit publishes
    them for the float target (`_public_labels` over the classes as Python
    floats, `member_pack_labeled`), its plan the step's class count; the
    labels are also kept in `labels` (one entry per step, None for a
    regressor), which the door stores as `_step_classes` for transform."""
    var b = _b()
    var seq = Python.list()
    for s in range(len(fit.states)):  # driver-loop: over the fitted imputation steps (one clone and one pack each)
        var est = member.template
        var fields = member.fields
        if s > 0:
            est = prep_estimator_clone(estimator, context)
            fields = py_call(member.fields_get, est, py_type(est))
        if len(fit.classes[s]) == 0:
            member_pack(est, fields, member.member_context, b, member.owner.plan, fit.states[s])
            if member_held_adopt(est, member.plan_key, member.member_context, member.owner.pins, member.owner.plan,
                                 fit.states[s]):
                fit.states[s] = 0  # the estimator's now
            _ = py_call(labels.append, PythonObject(None))
        else:
            var plan = member.owner.plan.copy()
            plan.classes = len(fit.classes[s])
            if member.marker_word >= 0:
                # the -1 class's code, as the step's fit read it (`iterimp_marker_code`):
                # the pack leaves it out of classes_, the held state predicts around it
                var code = iterimp_marker_code(fit.classes[s])
                if code >= 0:
                    plan.params[member.marker_word] = Float64(code)
            var values = Python.list()
            for c in range(len(fit.classes[s])):  # driver-loop: over one step's classes
                _ = py_call(values.append, PythonObject(Float64(fit.classes[s][c])))
            var published = _public_labels(values, member.member_context)
            member_pack_labeled(est, fields, member.member_context, b, plan, fit.states[s], published)
            if member_held_adopt(est, member.plan_key, member.member_context, member.owner.pins, plan,
                                 fit.states[s]):
                fit.states[s] = 0  # the estimator's now
            _ = py_call(labels.append, published)
        _ = py_call(seq.append, Python.tuple(fit.seq_j[s], _int_list(fit.seq_nbl[s]), est))
    return seq


# ---------------------------------------------------------------- fit
def _fit_transform(model: PythonObject, x: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = prep_keys()
    # UNPACK: the refusals, the selection (ONCE per operation: every program
    # of the rounds runs on it), X, the initial imputer and the request.
    _refuse(model, context)
    var fields = py_fields(model)
    var mode = _default_mode(context)
    var entries = scaler_entries(model, mode, context)
    var caps = enc_caps(model, entries)
    var values = xprep_values(x, entries, context)
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    var imputer = _initial(model, context)
    var request = _fit_request(model, values, _initial_request(imputer, n, d, mode, entries, context), mode,
                               entries, caps)
    var runner = _DoorRunner(values, entries, context, caps)
    var estimator = py_item(fields, nk[].k_estimator)
    if estimator is None:
        # RUN: the built-in Bayesian ridge, every program typed (x_prep/iterimp_run.mojo)
        var fitted = iterimp_fit(runner, request)
        # PACK
        _pack_initial(model, imputer, fitted, mode, d, context)
        var dk = len(fitted.keep)
        var got = runner.held[fitted.rounds]
        py_setitem(fields, nk[].k_n_iter_, PythonObject(fitted.n_iter))
        var sequence = Python.list()
        for s in range(len(fitted.seq_j)):  # driver-loop: fitted imputation sequence views (finished imputation steps)
            _ = py_call(sequence.append, Python.tuple(fitted.seq_j[s], _take1(got, fitted.seq_coef[s], dk, context),
                                                      _take1(got, fitted.seq_inter[s], 1, context)))
        py_setitem(fields, nk[].k_imputation_sequence_, sequence)
        var post = Python.list()
        for s in range(len(fitted.post_sig)):  # driver-loop: fitted posterior views (finished imputation steps)
            var side = fitted.post_side[s]
            _ = py_call(post.append, Python.tuple(_nbl_obj(fitted.post_has_nbl[s], fitted.post_nbl[s]),
                                                  _take1(got, fitted.post_sig[s], side * side, context),
                                                  _take1(got, fitted.post_al[s], 2, context),
                                                  _take1(got, fitted.post_means[s], dk, context)))
        py_setitem(fields, nk[].k__posterior, post)
        py_setitem(fields, nk[].k__rng, _u64_obj(fitted.rng))
        py_setitem(fields, nk[].k__post_step, PythonObject(fitted.post_step))
        return _take2(runner.held[fitted.out_handle], fitted.out_offset, n, fitted.cols, context)
    # A supplied estimator: the typed prologue (programs 1 to 4), then the
    # rounds over its native member (the only route; lane mf-iterimp7)
    var fitted = IterImpFitted()
    var plan = iterimp_prepare(runner, request, fitted)
    _pack_initial(model, imputer, fitted, mode, d, context)
    var dk = len(fitted.keep)
    if dk < 2:
        # scikit-learn's edge case (fewer than two kept features): n_iter_ = 0,
        # no step, the initial imputation; the estimator is never fitted
        iterimp_initial_only(runner, request, plan, fitted)
        py_setitem(fields, nk[].k__rng, _u64_obj(fitted.rng))
        py_setitem(fields, nk[].k__post_step, PythonObject(0))
        py_setitem(fields, nk[].k_n_iter_, PythonObject(0))
        py_setitem(fields, nk[].k_imputation_sequence_, Python.list())
        py_setitem(fields, "_step_classes", Python.list())
        return _take2(runner.held[fitted.out_handle], fitted.out_offset, n, fitted.cols, context)
    # UNPACK of the member (raises the family's, scikit-learn's or the posterior words)
    var member = _member(estimator, request.posterior, dk, context)
    # RUN: the rounds over the native member (x_prep/iterimp_run.mojo)
    runner.release_members = member.release
    var run = IterImpMember(member.owner.plan.copy(), member.borrows, request.posterior)
    run.classifier = member.classifier
    run.code_dtype = member.code_dtype
    run.cw_refusal = member.cw_refusal
    run.marker_word = member.marker_word
    var fit = IterImpMemberFit()
    iterimp_member_rounds(runner, request, plan, fitted, run, fit)
    # PACK (every state is released after it, on every path)
    try:
        var labels = Python.list()
        var seq = _member_pack(estimator, member, fit, labels, context)
        py_setitem(fields, nk[].k__rng, _u64_obj(fit.rng))
        py_setitem(fields, nk[].k__post_step, PythonObject(0))
        py_setitem(fields, nk[].k_n_iter_, PythonObject(fit.n_iter))
        py_setitem(fields, nk[].k_imputation_sequence_, seq)
        py_setitem(fields, "_step_classes", labels)
    finally:
        iterimp_member_free(run, fit)
    # The steps' fit notices, in step order, as the estimator's own fits raised them.
    member_fit_notices_raise(fit.notices, member.member_context)
    _ = member.owner.published  # the member's images stay loaded past its last call
    return _take2(runner.held[fit.out_handle], fit.out_offset, n, fit.cols, context)


# ---------------------------------------------------------------- transform
def _steps(sequence: PythonObject, posts: PythonObject, posterior: Bool) raises -> List[IterImpStep]:
    """UNPACK of the fitted steps: each step's feature and the addresses of its
    coef and intercept vectors (and, with sample_posterior, its neighbour list
    and the addresses and sizes of its sigma, alpha and means). The door
    keeps `sequence` and `posts` alive through the RUN."""
    var out_steps = List[IterImpStep]()
    for s in range(len(sequence)):  # driver-loop: over the fitted imputation steps (one record read each)
        var step = sequence[s]
        var has = False
        var nbl = List[Int]()
        var sig = 0
        var sig_words = 0
        var al = 0
        var al_words = 0
        var means = 0
        var means_words = 0
        if posterior:
            var post = posts[s]
            if post[0] is not None:
                has = True
                nbl = _nbl_of(post[0])
            sig_words = Int(py=post[1].size)
            al_words = Int(py=post[2].size)
            means_words = Int(py=post[3].size)
            sig = Int(py=classical_address(post[1])) if sig_words > 0 else 0
            al = Int(py=classical_address(post[2])) if al_words > 0 else 0
            means = Int(py=classical_address(post[3])) if means_words > 0 else 0
        out_steps.append(IterImpStep(Int(py=step[0]), Int(py=classical_address(step[1])),
                                     Int(py=classical_address(step[2])), has, nbl^, sig, sig_words, al, al_words,
                                     means, means_words))
    return out_steps^


struct _HeldSteps(Movable):
    """transform's hold on the fitted steps' member states (`_held_steps`):
    `steps` (in order, each step's held plan and state), `keepers` the
    estimators' owner objects (kept named until the RUN has joined), `loose`
    the steps whose state no owner keeps (freed by `_held_free`), `release`
    every step's kind allows its member calls with the GIL released."""
    var steps: List[IterImpHeldStep]
    var keepers: PythonObject
    var loose: List[Bool]
    var release: Bool

    def __init__(out self):
        self.steps = List[IterImpHeldStep]()
        self.keepers = PythonObject(None)
        self.loose = List[Bool]()
        self.release = False


def _held_free(mut held: _HeldSteps):
    """Free the states no estimator keeps (never raises; after the RUN, or
    on a refusal)."""
    for s in range(len(held.steps)):  # driver-loop: over the fitted imputation steps (one state each)
        if held.loose[s] and held.steps[s].state != 0:
            try:
                var call = member_call_of(held.steps[s].plan, held.steps[s].state)
                member_free(held.steps[s].plan, call)
            except:
                pass  # member_call_of builds words only; a state that cannot be freed is the store's
            held.steps[s].state = 0


def _unheld(s: Int, estimator: PythonObject, why: String) raises -> String:
    """transform's refusal of a fitted step whose estimator holds no member
    state it can serve (lane mf-iterimp7: no door rounds remain to answer
    it). A step fitted by these rounds always holds one; this is a step
    whose estimator was replaced or changed after fit."""
    return ("RuntimeError: mojolearn: IterativeImputer's imputation_sequence_ step " + String(s) + " holds a "
            + String(py_type(estimator).__name__) + " " + why + "; fit the imputer again")


def _held_marker_import(est: PythonObject, fields: PythonObject, plan: PythonObject, member_context: PythonObject,
                        b: PythonObject, step_classes: PythonObject, s: Int, needs: Int) raises -> MemberHeld:
    """The held-state IMPORT of a LabelPropagation / LabelSpreading step whose
    estimator holds no state of this process (a deep copy, an unpickled
    imputer): `member_held` admits with no labels, which `_lp_admit` leaves
    unpublished (providers/python/member_xneighbors_boundary_door.mojo
    `_lp_admit`: `owner.labels is None`). Here it is admitted with deferred
    labels as the fit admitted it, the -1 class's code read from the
    imputer's fitted `_step_classes[s]` (the step's classes in code order,
    the marker included; its fitted `classes_` leaves the marker out) into
    LP_MARKER_WORD, and the state imported from its fitted fields
    (`member_state`: X_, label_distributions_, n_iter_). Its predicted codes
    are then the step's codes, as at fit (lane mf-iterimp7). Any other kind:
    unpublished, as before. The state is the operation's (`loose`)."""
    var held = MemberHeld()
    var mark = member_mark(plan, b)
    if not xneighbors_kind_reads_marker(mark.family, mark.kind):
        return held^
    var labels = PythonObject(None)
    if step_classes is not None and s < len(step_classes):
        labels = step_classes[s]
    if labels is None:
        return held^
    var values = _floats_from(labels)
    var owner = member_admit_labeled(est, fields, plan, member_context, b, Python.list())
    if not owner.published or not owner.plan.member.has(needs | MEMBER_CAN_IMPORT):
        return held^
    var classes = List[Float32](capacity=len(values))
    for c in range(len(values)):  # driver-loop: over one step's classes
        classes.append(Float32(values[c]))
    var code = iterimp_marker_code(classes)
    if code >= 0 and len(owner.plan.params) > LP_MARKER_WORD:
        owner.plan.params[LP_MARKER_WORD] = Float64(code)
    owner.plan.classes = len(classes)
    var state = member_state(est, fields, member_context, b, owner.plan)
    if state == 0:
        return held^
    held.plan = owner.plan.copy()
    held.state = state
    held.loose = True
    held.keeper = owner.pins  # the member's images stay loaded through the RUN
    held.published = True
    return held^


def _held_steps(sequence: PythonObject, step_classes: PythonObject, posterior: Bool,
                context: PythonObject) raises -> _HeldSteps:
    """UNPACK, once per transform: every fitted step's estimator as its HELD
    member state (bindings/member_held_door.mojo `member_held`: the state its
    fit handed it, a hit; else one import, kept), each a kind serving
    PREDICT (and, with sample_posterior, PREDICT_STD with the mean:
    `_posterior_family`). A classifier step (its kind's capability bit, as
    the fit read it; lane mf-iterimp3) also needs the labels its fit stored
    (`step_classes`, the fitted `_step_classes`) and the element type of its
    codes (asked once per transform: the kind's). An empty sequence (fewer
    than two kept features: n_iter_ = 0) is served with no step. A step the
    member route cannot serve raises (`_unheld`), after every state taken so
    far is released (lane mf-iterimp7: the door rounds that served it are
    deleted)."""
    var out = _HeldSteps()
    # TOMBSTONE: MOJOLEARN_MEMBER_SET_OFF (plumbing) deleted 2026-11-23 by lane/ab-prep-classical; code recoverable at 4d76a9e16e.
    try:
        var b = _b()
        out.keepers = Python.list()
        var needs = MEMBER_CAN_PREDICT
        if posterior:
            needs |= MEMBER_CAN_PREDICT_STD
        var release = True
        var code_dtype = -1
        for s in range(len(sequence)):  # driver-loop: over the fitted imputation steps (one held-state read each)
            var step = sequence[s]
            var est = step[2]
            var plan = prep_estimator_plan(est, context)
            if plan is None or py_get(plan, "member") is None:
                raise Error(_unheld(s, est, "that is not a native member"))
            var member_context = plan["context"]
            var fields = py_call(plan["fields_get"], est, py_type(est))
            var held = member_held(est, fields, plan, member_context, b, needs)
            if not held.published:
                held = _held_marker_import(est, fields, plan, member_context, b, step_classes, s, needs)
            if not held.published:
                raise Error(_unheld(s, est, "with no fitted member state"))
            # held first, so a refusal below releases it with the others
            out.steps.append(IterImpHeldStep(Int(py=step[0]), _nbl_of(step[1]), held.plan.copy(), held.state))
            out.loose.append(held.loose)
            _ = py_call(out.keepers.append, held.keeper)
            var ask = MemberCall(held.plan.route, held.state)
            var mask = member_call(held.plan.member, MEMBER_OP_CAPABILITIES, ask)
            var classifier = (mask & MEMBER_CAN_CLASSIFIER) != 0
            if posterior and (classifier or not _posterior_family(held.plan.family)):
                raise Error(_POSTERIOR_WORDS)
            if (mask & needs) != needs:
                raise Error(_unheld(s, est, "whose member does not predict"))
            if classifier:
                var labels = PythonObject(None)
                if step_classes is not None and s < len(step_classes):
                    labels = step_classes[s]
                if labels is None:
                    raise Error(_unheld(s, est, "without the classes its fit stored"))
                var last = len(out.steps) - 1
                var values = _floats_from(labels)
                if len(values) == 0:
                    raise Error(_unheld(s, est, "without the classes its fit stored"))
                for c in range(len(values)):  # driver-loop: over one step's classes
                    out.steps[last].classes.append(Float32(values[c]))
                if code_dtype < 0:
                    code_dtype = member_output_dtype(held.plan, MEMBER_OP_PREDICT, max(len(out.steps[last].nbl), 1),
                                                     1, MEMBER_I32)
                out.steps[last].code_dtype = code_dtype
            release = release and (mask & (MEMBER_GIL_FREE | MEMBER_GIL_DOOR)) != 0
        out.release = release
    except error:
        _held_free(out)
        raise error^
    return out^


def _apply_request(model: PythonObject, fields: PythonObject, imputer: PythonObject, n: Int, d: Int,
                   mark: Tuple[Bool, Float64], var keep: List[Int], var bounds: List[Float64],
                   var indicator: List[Int], posterior: Bool, rng: UInt64, caps: XprepCaps,
                   var steps: List[IterImpStep], mut fill: PythonObject, context: PythonObject) raises -> IterImpApply:
    """UNPACK of transform's typed request: the initial imputer's transform
    words (`_imputer_transform`'s UNPACK; its fitted fill vector is returned
    in `fill`, which the caller keeps alive through the RUN) and the fitted
    fields."""
    var nk = prep_keys()
    var own = py_fields(imputer)
    _check_width(imputer, d)
    var init_mark = _missing_mark(own["missing_values"])
    fill = _fitted_flat(imputer, "_fill", own["_fill"], context)
    var fill_words = Int(py=fill[2])
    var init = ImputerApply(n, d, 0, init_mark[0], init_mark[1], 1, fill_words, _index_list(own["_keep"]),
                            _index_list(own["_indicator"]))
    return IterImpApply(init^, Int(py=classical_address(fill[1])), n, d, mark[0], mark[1], xprep_r3("work"),
                        caps.has_out, keep^, bounds^, indicator^, posterior, Int(rng & 0x7FFFFFFF),
                        Int(py=py_call(_b().int, py_item(fields, nk[].k__post_step))), steps^)


def _transform(model: PythonObject, x: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = prep_keys()
    # UNPACK
    var fields = py_fields(model)
    if not py_contains(fields, nk[].k_imputation_sequence_):
        raise Error("RuntimeError: mojolearn: this IterativeImputer instance is not fitted yet")
    var mode = py_item(fields, nk[].k_numeric_mode_)
    var entries = scaler_entries(model, mode, context)  # once per operation; passed to every round's programs
    var caps = enc_caps(model, entries)
    var values = xprep_values(x, entries, context)
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    _check_width(model, d)
    var keep = _index_list(py_item(fields, nk[].k__keep))
    var bounds = _bounds_of(model)
    var imputer = py_item(fields, nk[].k_initial_imputer_)
    if not py_contains(py_fields(imputer), nk[].k_n_features_in_):
        raise Error("RuntimeError: mojolearn: this " + _name(imputer) + " instance is not fitted yet")
    var posterior = _truth(py_item(fields, nk[].k_sample_posterior))
    var rng = _u64_of(py_item(fields, nk[].k__rng))
    var sequence = py_item(fields, nk[].k_imputation_sequence_)
    var indicator = _index_list(py_item(fields, nk[].k__indicator))
    var mark = _missing_mark(py_item(fields, nk[].k_missing_values))
    # the kept positions marked not missing (scikit-learn's `_is_empty_feature`)
    var empty_field = py_get(fields, "_empty_kept")
    var empty = _index_list(empty_field) if empty_field is not None else List[Int]()
    var runner = _DoorRunner(values, entries, context, caps)
    if py_item(fields, nk[].k_estimator) is not None:
        # The fitted steps' held member states (lane mf-iterimp2), the only
        # route of a supplied estimator since the door rounds' deletion (lane
        # mf-iterimp7): the step programs, the members' PREDICT /
        # PREDICT_STD on their fitted states (no import on a hit, no Python
        # estimator call). `_held_steps` raises when a step has none.
        var held = _held_steps(sequence, py_get(fields, "_step_classes"), posterior, context)
        var fill = PythonObject(None)
        var request = _apply_request(model, fields, imputer, n, d, mark, keep.copy(), bounds.copy(),
                                     indicator.copy(), posterior, rng, caps, List[IterImpStep](), fill, context)
        request.empty = empty.copy()
        runner.release_members = held.release
        var applied = IterImpApplied(II_NONE, 0, 0, 0)
        try:
            # RUN (x_prep/iterimp_run.mojo)
            applied = iterimp_member_transform(runner, request, held.steps, rng)
        except error:
            _held_free(held)
            raise error^
        _held_free(held)
        _ = held.keepers  # the estimators' owner objects stay alive past the member calls
        _ = fill  # the fitted fill vector the RUN read stays alive past it
        # PACK
        py_setitem(fields, nk[].k__rng, _u64_obj(rng))
        return _take2(runner.held[applied.out_handle], applied.out_offset, n, applied.cols, context)
    # The initial imputer's transform words (`_imputer_transform`'s UNPACK)
    var posts = py_item(fields, nk[].k__posterior) if posterior else PythonObject(None)
    var fill = PythonObject(None)
    var request = _apply_request(model, fields, imputer, n, d, mark, keep^, bounds^, indicator^, posterior, rng, caps,
                                 _steps(sequence, posts, posterior), fill, context)
    request.empty = empty^
    # RUN (x_prep/iterimp_run.mojo)
    var applied = iterimp_transform(runner, request)
    _ = fill  # the fitted fill vector the RUN read stays alive past it
    _ = sequence  # and the fitted steps' vectors
    _ = posts
    # PACK
    py_setitem(fields, nk[].k__post_step, PythonObject(applied.post_step))
    return _take2(runner.held[applied.out_handle], applied.out_offset, n, applied.cols, context)


# ---------------------------------------------------------------- the door
def prep_iterimp_call(model: PythonObject, op: Int, operation: String, args: PythonObject,
                      context: PythonObject) raises -> PythonObject:
    """IterativeImputer.fit(X, y=None) (returns the model), fit_transform(X,
    y=None) and transform(X)."""
    if op == PREP_OPN_FIT:
        _ = _fit_transform(model, args[0], context)
        return model
    if op == PREP_OPN_FIT_TRANSFORM:
        return _fit_transform(model, args[0], context)
    if op == PREP_OPN_TRANSFORM:
        return _transform(model, args[0], context)
    raise Error("TypeError: the native IterativeImputer owner does not serve " + operation)
