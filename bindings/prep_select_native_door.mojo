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
# RFE's estimator must be a mojolearn estimator with a connected native plan and a native member (lane
# mf-rfe7: every fit runs over the member; the door's own rounds through the plan's compiled fit and the
# fitted estimator's attribute reads are deleted). Its fitted estimator_'s predict, predict_proba,
# decision_function and score run as the plan's compiled operations (bindings/prep_native_estimator.mojo),
# so nothing enters a Python method. The rounds' programs run on the classifier owner's arena runner
# (`cls_borrowed_run` inside `rfe_fit_run`, the context `select_context`) because the support mask and the
# ranking are in-out inputs. The mutual-information programs and the selector transform run on the encode
# owner's transport (bindings/prep_encode_native_door.mojo `EncProg`).
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr, GILReleased
from bindings.classical_validate_unpack import classical_check_y, classical_estimator_name
from bindings.member_leaf_boundary_door import member_leaf_pack, member_leaf_pack_labeled
from bindings.member_boundary_door import member_pack, member_pack_labeled, member_admit_labeled
from core.member_abi import (
    member_kind_capabilities, MEMBER_CAN_FIT, MEMBER_CAN_EXPORT, MEMBER_CAN_CLASSIFIER, MEMBER_GIL_FREE,
    MEMBER_GIL_DOOR,
)
from core.member_families import MEMBER_FAMILY_ENSEMBLE
from core.ensemble_member import meta_plan_leaf_importances
from bindings.member_fit_notices_pack import member_fit_notices_raise
from core.member_abi import MemberCall, member_free, MEMBER_F32, MEMBER_I32, MEMBER_READS_Y_GIVEN
from core.member_abi import MEMBER_IGNORES_Y
from bindings.member_given_y_unpack import member_given_y
from core.member_families import MEMBER_FAMILY_GBDT, MEMBER_FAMILY_TREE, MEMBER_FAMILY_FOREST, MEMBER_FAMILY_EXTRA_TREES
from bindings.trees_public_inputs_run import trees_finite_f32
from providers.python.member_rfe_boundary_door import (
    RfeBase, RfeDottedPath, rfe_base_admit_door, rfe_count_words, rfe_image, rfe_publish, rfe_plan_labels,
    rfe_fit_keywords, rfe_member_fit_params, rfe_dotted_path, rfe_member_context_connect,
)
from bindings.arena_owner_door import arena_pin, arena_unpin
from bindings.x_prep_native_owner_door import xprep_resident
from core.member_set_run import MemberFitExtras
from providers.python.member_gbdt_boundary_door import gbdt_member_fit_params, gbdt_member_label_check
from bindings.native_loaded_module_door import native_loaded_module
from bindings.optimizer_public_words_unpack import stash_python_error
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.classical_arrays_door import classical_address, classical_typed
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.classical_labels_door import classical_encode_labels, ClassicalLabels
from bindings.admitted_input_unpack import admit_f32, admitted_vector, classical_view
from bindings.scaler_native_door import _ptr
from bindings.scaler_lifecycle_native_door import _entries
from bindings.x_prep_native_owner_door import xprep_values
from bindings.prep_stateless_native_door import _select_transform, _select_gather, _support_columns, _fam
from bindings.prep_encode_native_door import (
    EncProg, _fit_categories, _take1, _take2, _default_mode, _seed64, _category_build, _cat_block,
    _record_sizes, IDN_STATS_BLOCKED)
from bindings.prep_encode_cats_run import CatBlock
from bindings.prep_select_run import (
    mi_check_neighbors, mi_check_shape, mi_mask_of_bools, mi_mask_of_indices, mi_columns, mi_gather_stage,
    mi_class_count_stage, mi_int_words, mi_class_check, mi_discrete_check, mi_stage, mi_scatter, MiProgram,
    rfe_select_count, rfe_step_count, RfeFitPlan, RfeFitResult, rfe_fit_run, rfe_rows_gather, RFE_NO_IMPORTANCES,
)
from bindings.x_prep_native_inputs_door import _rows_fields, _dense_values, _values
from bindings.prep_classify_common_door import cls_x2d, XIn
from bindings.prep_classify_native_door import resident_enter_min, _resident_exit
from bindings.prep_native_estimator_door import (
    prep_estimator_admit, prep_estimator_has, prep_estimator_clone, prep_estimator_call, prep_estimator_plan)
from bindings.prep_keys_unpack import (
    prep_keys,
    PREP_OPN_DECISION_FUNCTION, PREP_OPN_FIT, PREP_OPN_FIT_TRANSFORM, PREP_OPN_GET_SUPPORT,
    PREP_OPN_MUTUAL_INFO_CLASSIF, PREP_OPN_PREDICT, PREP_OPN_PREDICT_PROBA, PREP_OPN_SCORE, PREP_OPN_TRANSFORM,
)
from bindings.py_balanced import py_item

comptime F32P = MutPointer[Float32, MutUntrackedOrigin]


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


# ---------------------------------------------------------------- RFE programs
# TOMBSTONE: _gather_cls (one gather program per round plus one for the final fit, each putting X) deleted
# 2026-11-09 by d6f2d5ed09 (lane w1-readmission-rfe); code recoverable at d6f2d5ed09^.
# Restore: git apply experiments/removed/W1_RFE_GATHER_PROGRAM_PER_ROUND.patch
# REMOVED (lane mf-rfe7, 2026-11-23): `_fit_clone` (RFE's `_clone(estimator).fit(x, y)` through the plan's
# compiled fit door), the door half of `_rfe_rounds` (one ClsProg program per elimination round over the
# estimator's own fit and attribute reads), `_importance_keys` / `_keys_of` (the fitted estimator's coef_ or
# feature_importances_ read through `prep_estimator_attribute`), `_rfe_hold_block` and the admitted hold scope
# of each round's block. Every RFE fit is THE ONE BODY over a native member (`_rfe_native`); each case the
# door half served is a member route or a native refusal in the words that door raised (rows:
# ~/mojolearn-evidence/briefs-20261122/mf-rfe7-work/rows.tsv). Plain door code, not an experiment define:
# no tombstone patch; recoverable at the parent of the commit that removed it.


# ---------------------------------------------------------------- a META estimator (lanes mf-meta5b, mf-rfe7)
def _rfe_meta_base(estimator: PythonObject, g: PythonObject, y: PythonObject, context: PythonObject,
                   b: PythonObject, mut encoded: Optional[ClassicalLabels]) raises -> RfeBase:
    """UNPACK, once per RFE fit, of an estimator the leaf admission does not hold (`rfe_base_admit_door`
    answered not served with no words): a dotted importance_getter walked ONCE (providers/python/
    member_rfe_boundary_door.mojo `rfe_dotted_path`, the walk THE RFE KIND's admission runs too) to THE
    FORWARDED EXPORT's hops and the leaf's words, then the META estimator admitted as a member through the
    member dispatch (`member_admit_labeled`: its family's refusals raise in their words, as the estimator's
    own fit door raised them). A plain getter over a META estimator (a search, a stack, a calibration, a
    Pipeline: none of them has coef_ or feature_importances_ in scikit-learn): admitted with the refusal its
    first round raises (RFE_NO_IMPORTANCES for 'auto', attrgetter's words for a name), so with no round the
    final fit runs, as in scikit-learn (lane mf-rfe7; before, the door's own rounds fitted it and raised the
    same words after their first fit). The labels (lane mf-rfe7): y encoded once, as the classifier fit
    doors encode it, and handed to the admission so a nested class_weight dict is read over them
    (`_child_labels`); a target the encoder refuses is admitted with none (a regressor META). The encode is
    handed back in `encoded` (the caller's classifier encode reuses it: one encode per fit). `native` False
    never returns: every case not served raises its words here."""
    var base = RfeBase()
    var getter = String(py=g)
    var dotted = getter.find(".") >= 0
    var path = RfeDottedPath()
    if dotted:
        path = rfe_dotted_path(estimator, g, context, b)
        if not path.served:
            raise Error(path.why.copy())
    var cls = String(py_type(estimator).__name__)
    var plan = prep_estimator_plan(estimator, context)
    if plan is None:
        raise Error("TypeError: mojolearn: RFE requires a mojolearn estimator; got " + cls)
    # The META estimator's own context, connected before its labels are encoded there (lane w3-fix-a, row 3:
    # unconnected, the encode below was refused and swallowed, so a classifier META read as a regressor).
    rfe_member_context_connect(plan["context"])
    var labels = PythonObject(None)
    try:
        var got = classical_encode_labels(y, plan["context"])
        labels = got.classes
        encoded = got^
    except:
        labels = PythonObject(None)
    var template = prep_estimator_clone(estimator, context)
    var member_context = plan["context"]
    var fields = py_call(plan["fields_get"], template, py_type(template))
    var owner = member_admit_labeled(template, fields, plan, member_context, b, labels)
    var words = String("TypeError: mojolearn: RFE requires an estimator whose native member fits and exports its "
                       + "fitted state; " + cls + " has none")
    if not owner.published or (dotted and owner.plan.family != MEMBER_FAMILY_ENSEMBLE):
        raise Error(words)
    var mask = member_kind_capabilities(owner.plan)
    if (mask & MEMBER_CAN_FIT) == 0 or (mask & MEMBER_CAN_EXPORT) == 0:
        raise Error(words)
    # The estimator type is the META kind's (lane mf-del5b): a search's class says none of its own (its
    # base decides, lane mf-search4), so its plan's word would have refused a search over a classifier;
    # THE RFE KIND's admission reads the same mask.
    var classifier = (mask & MEMBER_CAN_CLASSIFIER) != 0
    if dotted:
        # A tree or forest leaf read for feature_importances_ (lane mf-rfe6): its importance word set in the
        # nested plan this admission minted, before any state is made.
        if path.leaf_word and not meta_plan_leaf_importances(owner.plan, path.hops):
            raise Error(words)
        base.refusal = path.refusal.copy()
        base.block = path.forward_block
        base.squared = True
        base.hops = path.hops.copy()
    elif getter == "auto":
        base.refusal = String(RFE_NO_IMPORTANCES)
    else:
        base.refusal = "AttributeError: '" + classical_estimator_name(estimator) + "' object has no attribute '" + getter + "'"
    base.meta = True
    base.gil_free = (mask & (MEMBER_GIL_FREE | MEMBER_GIL_DOOR)) != 0
    base.mask = mask
    base.classifier = classifier
    base.owner = owner^
    base.template = template
    base.fields = fields
    base.member_context = member_context
    base.native = True
    return base^


# ---------------------------------------------------------------- RFE over a native member
#: GradientBoostingRegressor's Gbdt member kind (python/mojolearn/_gbdt_adapters.py `_mojolearn_member = (16, 2)`;
#: core/gbdt_member.mojo `_KIND_REGRESSOR`): its fit door admits y in its own words.
comptime _GBDT_KIND_REGRESSOR_ADAPTER = 2


struct RfeMember(Movable):
    """RFE's hold on its estimator as a NATIVE MEMBER (core/member_abi.mojo), built once per fit
    (`_rfe_member`): the estimator admitted with no fit target (providers/python/member_rfe_boundary_door.mojo
    `rfe_base_admit`, which THE RFE KIND's admission runs too), then the fit's target and parameters.
    Lane mf-rfe7: the admission always serves or raises (the estimator's own doors, which kept every round
    of a base not served, are deleted). `eval_rows` / `eval_keys` (lane mf-meta5b): a Gbdt fit with cat_features's held-out rows and
    keys; `final_importances`: a forest's feature_importances=True (the final fit keeps its importance word). `y` is the admitted target the RUN reads by address (float32 values for a regressor kind, the
    int32 class codes of `labels` for a classifier kind: encoded once per fit). `extras` (lane mf-gbdt5):
    RFE's fit parameters, admitted once with the estimator's fit door's checks and words (a Gbdt member:
    providers/python/member_gbdt_boundary_door.mojo `gbdt_member_fit_params`; `pins` holds every buffer
    they address). Lane mf-rfe6: every member refusal is raised (REMOVED: `strict`, which raised it only
    with admitted fit parameters, and the fallback rule's second fit)."""
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


def _rfe_member(estimator: PythonObject, y: PythonObject, n: Int, d: Int, g: PythonObject, fit_params: PythonObject,
                context: PythonObject, b: PythonObject) raises -> RfeMember:
    """UNPACK, once per RFE fit: the fit parameters' names and values in the estimator's compiled fit's
    words (`rfe_fit_keywords`: its refusals raise here, before any round, as `_fit_clone`'s first fit
    raised them), the estimator as a native member (`rfe_base_admit_door`, its labels deferred: lane
    mf-meta5b), then its fit target: a classifier kind's labels encoded here, once, and the plan completed
    over them (`rfe_plan_labels`: LabelPropagation / LabelSpreading's -1 marker word, a class_weight dict's
    table; `gbdt_member_label_check`: the classifier adapter's multiclass refusal, in its fit door's place),
    a regressor kind's target checked as its member's fit door admits it. Then the fit parameters: a Gbdt
    member's (`gbdt_member_fit_params`: sample_weight, eval_set (with cat_features too), group_id, pairs,
    pairs_weight), any other family's (`rfe_member_fit_params`: sample_weight through the shared weight
    admission, a forest's feature_importances); their refusals raise from here. A target the encoder or the
    regressor admission refuses raises its words here (lane mf-rfe6: in the family's fit door's words, the
    tree, forest and Gbdt regressors' own). One class is the member's (its family's fit decides, lane
    mf-cal5); a feature_importances value the forest door refuses raises its words here
    (`rfe_member_fit_params`). Lane mf-rfe7: nothing stays on the estimator's own doors (`_fit_clone` is
    deleted): the leaf admission (`rfe_base_admit_door`) raises its family's words or says why a class is not
    served; a dotted getter or a META estimator is admitted through the member dispatch (`_rfe_meta_base`);
    weights for a kind that reads none and a key the member does not carry raise (`rfe_member_fit_params`,
    `gbdt_member_fit_params`)."""
    var member = RfeMember()
    var unread = False
    var has_params = len(fit_params) != 0
    if has_params:
        rfe_fit_keywords(estimator, fit_params, context, b)
    var base = rfe_base_admit_door(estimator, g, context, b)
    var pre = Optional[ClassicalLabels]()  # a META admission's one label encode
    if not base.native:
        if base.why.byte_length() > 0:
            raise Error(base.why.copy())
        base = _rfe_meta_base(estimator, g, y, context, b, pre)
    # THE MEMBER'S REFUSALS ARE RAISED (lane mf-rfe6; THE FALLBACK RULE's last users went): the target is
    # admitted here in the words the estimator's own fit door admits it with, and a refusal raises from here,
    # where that door's first fit raised it (X is admitted already, so its target checks come first).
    if base.classifier:
        # The labels once, as a classifier's fit door encodes them (THE one label encoder,
        # bindings/classical_labels_door.mojo, which every classifier fit door of a served family calls): the
        # class list it would publish and one int32 code per row. ONE class (lane mf-cal5): the member is
        # fitted on it, as scikit-learn's RFE fits the estimator on whatever y holds; a family that refuses
        # one class raises its fit door's words from its member fit.
        var encoded = classical_encode_labels(y, base.member_context) if not pre else pre.value().copy()
        var count = len(encoded.classes)
        if encoded.n != n:
            # The member's fit reads n codes: a shorter y is refused here, in scikit-learn's words.
            raise Error("ValueError: Found input variables with inconsistent numbers of samples: [" + String(n)
                        + ", " + String(encoded.n) + "]")
        rfe_plan_labels(base, encoded.classes, b)
        base.owner.plan.classes = count
        member.y = encoded.codes
        member.y_dtype = MEMBER_I32
        member.labels = encoded.classes
    elif (base.mask & MEMBER_IGNORES_Y) != 0:
        # Round b (lane mf-member8): a kind whose fit reads no target (NearestNeighbors, RadiusNeighbors, OPTICS,
        # MDS, SpectralEmbedding): y is admitted only as scikit-learn's RFE validates it (`check_X_y`,
        # multi_output=True: present, as long as X, finite when numeric; a 2-D or str y is accepted), and the
        # member is handed no y view.
        member.y = _rfe_unread_y(y, n, base.member_context)
        unread = True
    elif (base.owner.plan.family == MEMBER_FAMILY_TREE or base.owner.plan.family == MEMBER_FAMILY_FOREST
            or base.owner.plan.family == MEMBER_FAMILY_EXTRA_TREES):
        # A tree or forest regressor's fit door (bindings/trees_public_forest_door.mojo `_rf_fit_regressor`,
        # `_et_fit_arrays`): float32 words with no finite test (its criterion decides; a deviance criterion
        # is not a member, so it never reaches here), as long as X in its words.
        var ya = admit_f32(y, 1, "y", base.member_context, finite=False)
        if ya.rows != n:
            # xtrees/trees_forest_run.mojo `_y_rows_check`'s words
            raise Error("y has " + String(ya.rows) + " rows, X has " + String(n))
        member.y = ya.array
    elif base.owner.plan.family == MEMBER_FAMILY_GBDT and base.owner.plan.kind == _GBDT_KIND_REGRESSOR_ADAPTER:
        # GradientBoostingRegressor's own fit door (bindings/trees_public_gbdt_fit_door.mojo
        # `adapter_regressor_fit` -> `_regression_target`): one nonempty dimension, then finite, in ITS words
        # (lane w3-fix-a, row 13: the booster's words below named GradientBoosting for the adapter).
        var view = classical_view(y, base.member_context, "y")
        if Int(py=view.ndim) != 1 or Int(py=view.size) == 0:
            raise Error("y must be nonempty one-dimensional targets")
        var ya = admit_f32(view, 1, "y", base.member_context, finite=False, admitted=True)
        _ = view^
        if not trees_finite_f32(ya.address, ya.rows):
            raise Error("y must be finite")
        if ya.rows != n:
            raise Error("mojolearn: y has " + String(ya.rows) + " values for " + String(n) + " rows")
        member.y = ya.array
    elif base.owner.plan.family == MEMBER_FAMILY_GBDT:
        # A Gbdt regressor's fit door (bindings/trees_public_gbdt_fit_door.mojo, one output): float32
        # words, as long as X, then finite, in its words.
        var ya = admit_f32(y, 1, "y", base.member_context, finite=False)
        if ya.rows != n:
            raise Error("mojolearn: y has " + String(ya.rows) + " values for " + String(n) + " rows")
        if not trees_finite_f32(ya.address, ya.rows * ya.cols):
            raise Error("Input y contains NaN or infinity (GradientBoosting).")
        member.y = ya.array
    else:
        # The target once, as a regressor member's fit door admits it (present, real, one column, finite, as
        # long as X: bindings/admitted_input_unpack.mojo `admit_y`, the regressor fit doors' one check).
        member.y = classical_check_y(y, n, classical_estimator_name(estimator), base.member_context)
    # The classifier adapter over more than two classes: its fit door's refusal, raised before the first
    # round fits (since lane mf-imp6 the elimination rounds fit Gbdt kinds and read their importances).
    if member.labels is not None:
        gbdt_member_label_check(base.owner.plan, len(member.labels))
    if has_params:
        if base.owner.plan.family == MEMBER_FAMILY_GBDT:
            # UNPACK of the fit parameters (their refusals raise here, in the fit door's words, as the
            # door's first fit would raise them).
            var params = gbdt_member_fit_params(base.template, base.owner.plan, fit_params, y, member.labels, n, d,
                                                base.member_context)
            if not params.admitted:
                # Lane mf-rfe7: a key `rfe_fit_keywords` let through (the plan's fit keyword table names it) that
                # the Gbdt kind's member fit does not carry; the deleted rounds handed it to the fit door.
                raise Error("TypeError: mojolearn: RFE: " + classical_estimator_name(estimator)
                            + "'s native member fit does not take these fit parameters: "
                            + String(py_call(b.list, fit_params)))
            member.extras = params.extras.copy()
            member.pins = params.pins
            # By swap: the addresses in `extras` point into these lists' storage.
            swap(member.ids, params.ids)
            swap(member.codes, params.codes)
            swap(member.eval_rows, params.eval_rows)
            swap(member.eval_keys, params.eval_keys)
        else:
            var params = rfe_member_fit_params(base, fit_params, n, b)
            member.extras = params.extras.copy()
            member.pins = params.pins
            member.final_importances = params.final_importances
    member.y_address = 0 if unread else Int(py=classical_address(member.y))
    if not base.classifier and base.owner.plan.member.can(MEMBER_READS_Y_GIVEN):
        # THE GIVEN y (lane mf-tail5; core/member_abi.mojo `MemberCall.y_given`): y's own words, a view (the
        # door's `y` lives through the RUN), for a kind whose stratified splitter codes them; asked once.
        try:
            if (member_kind_capabilities(base.owner.plan) & MEMBER_READS_Y_GIVEN) != 0:
                var given = member_given_y(y, n, 1, context["array_class"])
                if given.given():
                    member.extras.y_given = given.data
                    member.extras.y_given_dtype = given.dtype
        except:
            pass
    member.base = base^
    return member^


def _rfe_unread_y(y: PythonObject, n: Int, context: PythonObject) raises -> PythonObject:
    """Round b (lane mf-member8): RFE's own target validation for a member that reads no target
    (scikit-learn's `validate_data(X, y, multi_output=True)`, sklearn/utils/validation.py `check_X_y` /
    `_check_y`): y present, one or two dimensions, as long as X, finite when numeric (its words); a non-numeric
    y (str labels) is admitted as the one label encoder admits it (no finite test, as `_check_y` with
    dtype=None makes none). Answers the admitted object, held through the fit (nothing reads its words)."""
    if y is None:
        raise Error("ValueError: RFE requires y to be passed, but the target y is None.")
    var numeric = True
    var held = PythonObject(None)
    var rows = 0
    var words = 0
    var address = 0
    try:
        var ya = admit_f32(y, -1, "y", context, finite=False)
        held = ya.array
        rows = ya.rows
        words = ya.rows * ya.cols
        address = ya.address
    except:
        numeric = False
    if numeric:
        if rows != n:
            raise Error("ValueError: Found input variables with inconsistent numbers of samples: [" + String(n)
                        + ", " + String(rows) + "]")
        if not trees_finite_f32(address, words):
            var values = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=address)
            for i in range(words):  # small-loop(the refusal path only: is a non-finite word a NaN)
                if values[i] != values[i]:
                    raise Error("ValueError: Input y contains NaN.")
            raise Error("ValueError: Input y contains infinity or a value too large for dtype('float64').")
        return held
    var labels = classical_encode_labels(y, context)
    if labels.n != n:
        raise Error("ValueError: Found input variables with inconsistent numbers of samples: [" + String(n) + ", "
                    + String(labels.n) + "]")
    return labels.codes


def _rfe_native(member: RfeMember, arr: XIn, n: Int, d: Int, counts: List[Float64], cc: PythonObject,
                b: PythonObject, mut support: List[Float32], mut ranking: List[Float32],
                mut selected: Int) raises -> PythonObject:
    """RFE.fit over the admitted member: UNPACK of the image, the thread's resident scope and X's pin;
    RUN `rfe_fit_run` (THE ONE BODY, the GIL released for a kind with MEMBER_GIL_FREE or MEMBER_GIL_DOOR,
    as the door's rounds released it); PACK the final state into the one clone (`member_leaf_pack`, a
    classifier with its labels: the attributes its own fit writes) and the rounds' and the final fit's
    notices, in fit order. Answers the packed clone; `support`, `ranking` and `selected` are written."""
    var image = rfe_image(cc)
    # THE DEVICE ROWS ROUTE (lane mf-rfe6): a fold's DeviceRows X (`cls_x2d` keeps one only when its base is
    # dense float32 and proved finite) is handed to the RUN as its base and row numbers, which every round's
    # program gathers on the device, as the door's programs did (bindings/prep_classify_prog_door.mojo
    # `ClsProg.run`); an image that gathers no rows gets the fold's host rows, as the door's programs did.
    # The resident scope keeps the base, as the door's run kept it. Who gathers moves; the programs do not.
    var x_value = arr.value
    var x_base = PythonObject(None)
    var x_index = PythonObject(None)
    var x_base_words = 0
    # Lane w3f2-rfe (failures row A): only a fold's DeviceRows takes this branch, tested by its class (one
    # identity compare). It was `not classical_is_array(X)`, so every plain NumPy X (`cls_x2d` hands the
    # admitted float32 ndarray itself) was admitted a second time in the program context below, which no
    # public call had connected in a fresh process: every RFE fit raised the shared admission's "Sequential
    # requires native input types", after X had passed its own admission. REMOVED: that second admission
    # (and its copy) of an already admitted dense X, per fit.
    if py_type(arr.value) is py_item(cc, prep_keys()[].k_maxabs_rows_class):
        var pc = cc["prog_context"]
        # The fold's base and row numbers are admitted in the program context: connect NumPy there first
        # (one dict probe once connected), as `_rfe_fit` connects `cc`.
        classical_connect_numpy(pc)
        if rfe_rows_gather(image.run):
            var fold = _rows_fields(arr.value, pc)
            x_base = _dense_values(fold.base, pc)
            x_index = fold.indices
            x_value = x_base
            x_base_words = Int(py=x_base.size)
        else:
            x_value = _values(arr.value, pc, True)
    var plan = RfeFitPlan(member.base.owner.plan.copy(), image.run, image.work_on,
                          Int(py=classical_address(x_value)), n, d, member.y_address, member.y_dtype)
    if x_index is not None:
        plan.x_index = Int(py=classical_address(x_index))
        plan.x_base_words = x_base_words
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
        pin = arena_pin(x_value)
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
    _ = x_value  # the base or host rows the run read by address
    _ = x_base
    _ = x_index  # the fold's row numbers, read by address through the run above
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
            # A META estimator (`_rfe_meta_base`: a dotted getter, or a plain one): its own family's PACK.
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
    # copy (`_rfe_native` pins X in it: THE PIN RULE). An enclosing user
    # scope, when one is active, is the one used.
    # REMOVED (lane mf-rfe7): the admitted hold scope of each round's block (S4, lane s1), which the deleted
    # door rounds held while the estimator's own fit door read it; the member rounds hand the block by
    # address (core/member_abi.mojo), and X itself is admitted finite by `cls_x2d`.
    var scope = resident_enter_min(cc, PythonObject(arr.n * arr.d))
    try:
        _rfe_rounds(model, fields, estimator, g, y, fit_params, arr, context, cc)
    except error:
        try:
            _resident_exit(scope, cc)
        except ignored:
            _ = ignored^
        raise error^
    _resident_exit(scope, cc)


def _rfe_rounds(model: PythonObject, fields: PythonObject, estimator: PythonObject, g: PythonObject,
                y: PythonObject, fit_params: PythonObject, arr: XIn, context: PythonObject,
                cc: PythonObject) raises:
    """The elimination rounds and the final fit of `_rfe_fit`, inside its resident scope: the estimator
    admitted as a native member once for the whole fit (`_rfe_member`, which serves it or raises), then the
    typed RUN whole (`_rfe_native`: THE ONE BODY, bindings/prep_select_run.mojo `rfe_fit_run`, which THE RFE
    KIND also runs) and the PACK of the fitted attributes. X held as DeviceRows (a fold's rows) is the RUN's
    too since lane mf-rfe6 (THE DEVICE ROWS ROUTE). REMOVED (lane mf-rfe7): the door half, one ClsProg
    program per round over the estimator's own fit door (`_fit_clone`) and attribute reads."""
    var b = native_loaded_module("builtins")
    var n = arr.n
    var d = arr.d
    # UNPACK of n_features_to_select and step (providers/python/member_rfe_boundary_door.mojo
    # `rfe_count_words`); their counts are typed (bindings/prep_select_run.mojo `rfe_select_count`,
    # `rfe_step_count`, resolved in `_rfe_native`).
    var counts = rfe_count_words(fields, b)
    # step's refusal first, in the door's order (before the estimator's admission).
    _ = rfe_step_count(d, counts[3] != 0.0, counts[4], Int(counts[5]))
    var member = _rfe_member(estimator, y, n, d, g, fit_params, context, b)
    var support = List[Float32]()
    var ranking = List[Float32]()
    var selected = 0
    var final = _rfe_native(member, arr, n, d, counts, cc, b, support, ranking, selected)
    rfe_publish(fields, final, support, ranking, selected, context)


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
