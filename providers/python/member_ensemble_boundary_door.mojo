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
carries no mark (lane mf-door7: it is a transformer; no META kind holds
one, since every kind here needs a predictor and scikit-learn's own fit of
Voting, Stacking, Bagging, AdaBoost, OneVsRest, Calibrated, MultiOutput
and a search with no `score` fails on it, and the one meta that holds a
transformer, Pipeline, has no member entry:
providers/python/member_pipeline_boundary_door.mojo). A unit kind answers
False, and its door keeps the estimator, for: a OneVsRest base whose kind
lacks the scores its predict reads; a Bagging base that is not a tree,
forest or extra-trees member, or one that names another device. A
Calibrated base that takes the resident DeviceRows route is the kind's
since lane mf-door7 (the route, a measured winner, is
CalibratedClassifierCV's own public fit's and never reaches this
admission).

A SERVING ADMISSION (lane mf-door7, `MemberOwner.serving`: a caller that
only predicts on fitted state, `member_admit_serving` without
MEMBER_CAN_FIT, and every nested admission of a META import) reads no
setting only a fit reads: a Stacking or Calibrated cv (a user's splitter
object too), a search's scorers, cv and candidates (`_admit_search_serving`:
its fitted best), and a MultiOutputRegressor over Ridge is the kind. A Bagging configuration that may take the batched
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
serving member, a cv SPLITTER of ours whose stored settings the
door reads in another form (every native splitter is read here as typed
words and split inside each fit: KFold / StratifiedKFold since lane
mf-split4, the others since lane mf-split5, `_split_words`; an index list,
and an `_IterableCV`'s stored pairs, are read here into typed folds), a non-boolean passthrough, weights of the wrong
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
from bindings.member_fit_notices_pack import member_fit_notices_raise, member_fit_notices_raise_with
from providers.python.member_xneighbors_boundary_door import xneighbors_kind_reads_marker, xneighbors_marker_index
from core.xneighbors_member import LP_MARKER_WORD
from bindings.member_leaf_boundary_door import member_leaf_admit_serving_labeled
from bindings.member_leaf_boundary_door import (
    member_leaf_admit_serving, member_leaf_pack, member_leaf_pack_labeled, member_leaf_state,
)
from bindings.member_owner_unpack import MemberOwner, member_labels_take, member_labels_hold, member_mark
from bindings.member_class_weight_unpack import member_cw_fill
from bindings.class_weight_boundary_unpack import class_weight_dictionary_values, CLASS_WEIGHT_TREES
from bindings.meta_plan_capsule_pack import meta_plan_capsule
from bindings.prep_native_estimator_door import prep_estimator_plan, prep_estimator_clone, prep_estimator_has
from bindings.prep_native_estimator_door import prep_estimator_admit
# Lane mf-refuse5, THE DOOR'S REFUSAL: the meta doors' constructor checks, one body with the doors.
from bindings.trees_meta_validate_unpack import (
    meta_active, meta_active_entries, meta_owner, meta_voting_validate, meta_voting_weights, meta_stacking_validate,
    meta_calibrated_validate, meta_cv_value, meta_jobs, meta_bagging_validate, meta_ada_validate, meta_seed,
    MetaCount, meta_count_reading, meta_require_tree, boost_member_device,
)
from bindings.py_balanced import py_call, py_get, py_item, py_setattr, py_setitem, py_type, py_contains, py_fields
from core.device_route import DeviceRoute, DEVICE_CPU
from bindings.admitted_input_unpack import admitted_unfilled
from bindings.native_unit_threads_unpack import native_unit_jobs
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
from bindings.trees_public_inputs_door import trees_table
from bindings.trees_public_forest_door import (
    forest_kind, forest_refresh, forest_config, _report_engine, forest_dt_random_inner, forest_dt_random_adopt,
    _int, _float,
)
from bindings.trees_route_boundary_door import trees_route_context
from providers.python.forest_member_common_boundary_door import (
    forest_member_admit_common, forest_member_state_common, forest_member_pack_common,
)
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_call_of, member_free, member_kind_capabilities,
    member_resolve_family,
    MEMBER_OP_NEW, MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_F32, MEMBER_I32, MEMBER_I64, MEMBER_F64, MEMBER_IMPORT_DONE,
    MEMBER_CAN_FIT, MEMBER_CAN_PREDICT, MEMBER_CAN_PREDICT_PROBA, MEMBER_CAN_DECISION, MEMBER_CAN_CLASSIFIER,
    MEMBER_CAN_IMPORT, MEMBER_CAN_EXPORT, MEMBER_CAN_SAMPLE_WEIGHT, MEMBER_BLOCK_SCALARS,
)
from xtrees.trees_meta_vote_run import STACK_PREDICT, STACK_PROBA, STACK_DECISION
from core.ensemble_member import (
    meta_plan_metric, meta_plan_search_labels, META_SEARCH_CLASSIFIER, META_SEARCH_REGRESSOR, META_MULTI_CLASSIFIER, META_MULTI_REGRESSOR,
    META_BLOCK_SEARCH, META_BLOCK_FOLDS,
)
from core.ensemble_member import meta_plan_leaf_importances, META_PARAM_SERVING
from core.member_abi import MEMBER_CAN_SCORE, MEMBER_CAN_MULTI_OUTPUT, MEMBER_OP_PREDICT, member_output_dtype
from bindings.msel_common_door import msel_f64_block, msel_i64_list, msel_index_i64, msel_p64, msel_addr
from bindings.classical_labels_door import labels_decode_labels
from bindings.family_common_pack import family_array
from core.member_abi import MEMBER_OP_PREDICT_PROBA, MEMBER_OP_DECISION
from core.msel_scorer_run import (
    msel_scorer_code, msel_scorer_row, msel_scorer_regression, msel_scorer_positive, msel_scorer_positive_kind,
    msel_scorer_metric_code, SCORER_POSITIVE_LABEL_ONE, SCORER_POSITIVE_MARKS, SCORER_POSITIVE_ABSENT, SCORER_POSITIVE_KIND,
    SCORER_POSITIVE_MIXED,
    msel_scorer_cluster, msel_scorer_cluster_entry, msel_scorer_top_k, msel_scorer_ap, msel_scorer_brier,
    msel_scorer_label_row, msel_scorer_f_row, msel_scorer_group_words, MselScorerOptions,
    msel_scorer_weighted, msel_scorer_caller, SCORER_WEIGHTS_EMPTY,
    msel_scorer_log_loss, SCORER_LABEL_NUMBER, SCORER_LABEL_TEXT, SCORER_LABEL_UNREAD,
    msel_scorer_roc_problem, msel_scorer_mi, msel_scorer_v_measure, msel_scorer_d2_proba, msel_scorer_decision_only,
    msel_scorer_float_labels,
    MselScorer,
)
from bindings.mcls_ranking_run import ROC_BINARY, ROC_OVR, ROC_OVR_WEIGHTED
# lane mf-score6: a make_scorer's sample_weight read as float32 words
from std.memory import bitcast
from bindings.family_common_unpack import family_shape_repr
from bindings.classical_validate_unpack import classical_view
from bindings.admitted_family_door import admit_family
from bindings.classical_arrays_door import classical_address
from core.entry_table import EntryRef, entry_ref_unresolved
from core.metrics_cluster_table import cluster_ops_entry
from providers.python.estimator_binding_pyapi_door import estimator_binding_mode
from bindings.metrics_exp_rank_run import BRIER_MARK_ZERO, BRIER_MARK_ONE, BRIER_MARK_MINUS_ONE
from bindings.msel_scorer_native_door import msel_scorer_response
from bindings.native_loaded_module_door import native_loaded_module
from bindings.metrics_keys_unpack import metrics_keys
from bindings.arena_prog_native_door import metric_route_of, metric_route_entries
from bindings.msel_candidates_door import msel_grid_list, msel_sampler_check, msel_sampler_list
from bindings.msel_estimator_native_door import msel_native_clone, msel_native_set_params, msel_clone_parameter
from bindings.msel_common_door import msel_get_scorer, msel_f64_list, msel_i32_list, msel_is_integral, msel_is_int_arg
from bindings.msel_search_state_door import msel_search_update
from bindings.msel_member_device_unpack import msel_member_host_folds, msel_params_name_device
from bindings.msel_split_door import msel_cls, MselSplitter, SPLITTER_ITERABLE
from bindings.msel_split_unpack import MselRoutePin
from bindings.msel_split_spec_run import MselSplitSpec, MSEL_SPEC_NONE, msel_spec_stratified
from bindings.msel_split_spec_unpack import msel_split_spec_read
from core.ensemble_member import (
    meta_plan_split, meta_plan_rfe, meta_plan_refuse, META_CAL_BINARY_MARKER_WORDS,
    META_RFE_CLASSIFIER, META_RFE_REGRESSOR, META_BLOCK_RFE, META_BLOCK_FORWARD,
)
from providers.python.member_rfe_boundary_door import (
    RfeBase, rfe_base_admit, rfe_count_words, rfe_image, rfe_publish, rfe_dotted_path,
)
from bindings.prep_classify_prog_run import ClsRunPlan
from core.msel_refusal_words import (
    msel_empty_scoring_words, msel_cvp_method_words, msel_kfold_words, MSEL_REFIT_CALLBACK_WORDS, MSEL_NO_FITS_WORDS,
)
from bindings.prep_keys_unpack import prep_keys

comptime _I64P = MutPointer[Int64, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]


def _stack(kind: Int) -> Bool:
    return kind == META_STACK_CLASSIFIER or kind == META_STACK_REGRESSOR


def _classifier(kind: Int) -> Bool:
    return (kind == META_VOTE_CLASSIFIER or kind == META_STACK_CLASSIFIER or kind == META_OVR
            or kind == META_CALIBRATED or kind == META_BAG_CLASSIFIER or kind == META_ADA_CLASSIFIER
            or kind == META_SEARCH_CLASSIFIER or kind == META_MULTI_CLASSIFIER or kind == META_RFE_CLASSIFIER)


def _rfe(kind: Int) -> Bool:
    return kind == META_RFE_CLASSIFIER or kind == META_RFE_REGRESSOR


def _rfe_dotted(getter: String) -> Bool:
    """An RFE importance_getter that is an attribute path over its estimator (lane mf-del5b), not one of
    the leaf getters `rfe_base_admit` reads (lane mf-rfe7: any plain attribute name is a leaf getter)."""
    return getter.find(".") >= 0


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
    as the door reads them (bindings/trees_meta_validate_unpack.mojo
    `meta_active_entries`, one body). `ok` False: the door refuses them."""
    var names: PythonObject
    var ests: PythonObject
    var ok: Bool

    def __init__(out self) raises:
        self.names = Python.list()
        self.ests = Python.list()
        self.ok = False


def _active(fields: PythonObject, b: PythonObject) raises -> _Active:
    var got = _Active()
    try:
        var entries = meta_active_entries(py_get(fields, "estimators"))
        got.names = entries.names
        got.ests = entries.ests
        got.ok = True
    except:
        got.ok = False  # the door's words: `_admit_one` raises them through `meta_active`
    return got^


def _stack_method(fields: PythonObject, est: PythonObject, classifier: Bool, context: PythonObject) raises -> Int:
    """`_StackingBase._method` (the door's `_stack_method`). A value the
    door refuses raises its words (lane mf-fitadm7; bindings/
    trees_meta_validate_unpack.mojo `meta_stacking_validate`, one wording):
    before, -1 and the caller kept the door, which raised them."""
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
    raise Error("stack_method must be auto, predict_proba, decision_function or predict")


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
    so its labels are the codes too; Calibrated, a search and an RFE
    (lane mf-del5b) fit theirs on y as given, the same labels. A
    MultiOutput base sees one column per output (its labels differ per
    output: none here), a regressor none. A nested class_weight dict is read
    over these (`_admit_nested`), as the member's own fit door reads it."""
    if not _classifier(kind) or _multi(kind):
        return PythonObject(None)
    if kind == META_OVR:
        return py_call(b.list, py_call(b.range, PythonObject(2)))
    if labels is None or len(labels) == 0:
        return PythonObject(None)
    if kind == META_CALIBRATED or _search(kind) or _rfe(kind):
        return labels
    return py_call(b.list, py_call(b.range, PythonObject(len(labels))))


def _admit_cal_base(done: _Done, mut nested: _Nested, base: PythonObject, needs: Int, context: PythonObject,
                    b: PythonObject, labels: PythonObject, whole: Bool) raises -> Bool:
    """A CalibratedClassifierCV base (`_admit_nested`), and THE UNLABELED MARKER (lane mf-meta5b): a
    LabelPropagation / LabelSpreading base whose labels (`_child_labels`: the calibration's own, y as given)
    hold -1 is admitted over the codes with the marker's code in its marker word (scikit-learn fits it on y
    as given: its classes_ exclude -1); the kind's fit leaves the marker out of each unit's held classes
    (xtrees/trees_calibrated_fit_run.mojo) and clears it on cross_val_predict's folds (ensemble=False). Two
    classes with ensemble=False: scikit-learn's own fit fails (META_CAL_BINARY_MARKER_WORDS, its words, raised
    here natively; lane mf-ens6)."""
    var plan = prep_estimator_plan(base, context)
    if plan is not None and py_get(plan, "member") is not None and labels is not None:
        var mark = member_mark(plan, b)
        if xneighbors_kind_reads_marker(mark.family, mark.kind):
            var marker = xneighbors_marker_index(labels)
            if marker >= 0:
                if not whole and len(labels) == 2:
                    raise Error(META_CAL_BINARY_MARKER_WORDS)
                var fields = py_call(plan["fields_get"], base, py_type(base))
                var owner = member_leaf_admit_serving_labeled(base, fields, plan, plan["context"], b, needs,
                                                              Python.list())
                if not owner.published or len(owner.plan.params) <= LP_MARKER_WORD:
                    return False
                owner.plan.params[LP_MARKER_WORD] = Float64(marker)
                var mask = 0
                try:
                    mask = member_kind_capabilities(owner.plan)
                except:
                    return False
                if (mask & needs) != needs or (mask & MEMBER_CAN_CLASSIFIER) == 0:
                    return False
                nested.plans.append(owner.plan.copy())
                nested.masks.append(mask)
                _ = py_call(nested.pins.append, owner.pins)
                return True
    return _admit_nested(done, nested, base, True, needs, context, b, labels)


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
        # A class_weight its estimator refuses (lane mf-ens6b): its own words
        # (`member_cw_fill` raises the estimator's, one body with its fit
        # door), raised here where the door raised them at its fit.
        if not member_cw_fill(owner, est, fields, labels, plan["context"], b):
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
    connects (`_xt`: `numeric_mode`, else the process default). An image
    that is not connected raises the door's words (`trees_table`: the
    refusal recorded at import), as the door's fit does (lane mf-ens6: it
    answered empty and the caller kept the door)."""
    var mode = py_get(fields, "numeric_mode")
    if mode is None:
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    if mode is None or py_type(mode) is not b.str or (String(mode) != "fast" and String(mode) != "identical"):
        raise Error("numeric_mode must be fast, identical or None")
    var table = trees_table(context, String(mode), "_mojolearn_x_trees")
    var key = py_get(table, "x_trees_native_key")
    if key is None:
        raise Error("ImportError: mojolearn: _mojolearn_x_trees is not connected to the tree owner")
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


# Lane mf-ens6: no nesting bound. Both work-lists skip an estimator already
# walked or pending (identity), so an estimator graph that contains itself is
# walked once per object and the walk ends; the plan store has no fixed
# table (core/ensemble_member.mojo `meta_plan_mint_with`), so any depth is a
# member. REMOVED: `_NESTED_META_MAX` and its two refusals.


def _meta_children(est: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
                   b: PythonObject, prepared: PythonObject, serving: Bool) raises -> PythonObject:
    """The nested estimators of an unfitted meta-estimator that its
    admission reads (and so may be meta-estimators themselves): the active
    voting / stacked members and the final estimator, a OneVsRest or
    Calibrated base. A Bagging base is a tree, forest or extra-trees member
    (`_require_tree`): none. A SERVING search (lane mf-door7,
    `_admit_search_serving`): its one nested estimator, `_search_served`."""
    var out = Python.list()
    if _search(kind) and serving:
        var served = _search_served(fields)
        if served is not None:
            _ = py_call(out.append, served)
        return out
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
    if _rfe(kind):
        # THE RFE KIND (lane mf-del5b): with a dotted importance_getter its estimator may be a META
        # estimator (the path's first step: `rfe_dotted_path`), admitted by this work-list before the RFE.
        # With 'auto' / 'coef_' / 'feature_importances_' it is a leaf (`rfe_base_admit`): none.
        var g = py_get(fields, "importance_getter")
        var base = py_get(fields, "estimator")
        if base is not None and g is not None and py_type(g) is b.str and _rfe_dotted(String(g)):
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
        var children = _meta_children(est, est_fields, est_kind, est_context, b, prepared, owner.serving)
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
        sub.serving = owner.serving  # a serving admission's nested meta-estimators serve too (lane mf-door7)
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
    var owner_labels = owner.labels  # a second reference: owner is passed mutably beside it
    return _admit_one(model, fields, kind, context, b, owner, done, prepared, owner_labels)


def _admit_one(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner, done: _Done, prepared: PythonObject,
    labels: PythonObject) raises -> Bool:
    """One meta-estimator's admission (the body `ensemble_member_admit`'s
    work-list runs per meta-estimator): its nested meta-estimators are in
    `done`, a search's candidates in `prepared`; `labels` the labels of the
    codes it is fitted on (THE LABELS, lane mf-split4)."""
    var child = _child_labels(kind, labels, b)
    if _rfe(kind):
        return _admit_rfe(model, fields, context, b, owner, done, child)
    if _search(kind):
        if owner.serving:
            return _admit_search_serving(fields, context, b, owner, done, child)
        return _admit_search(model, fields, context, b, owner, done, prepared, child)
    if _multi(kind):
        return _admit_multi(fields, kind, context, b, owner, done)
    if _ada(kind):
        return _admit_ada(fields, kind, context, b, child, owner)
    if _unit(kind):
        return _admit_unit(model, fields, kind, context, b, owner, done, child)
    var classifier = _classifier(kind)
    # THE DOOR'S REFUSAL (lane mf-refuse5): the door's constructor checks, in its order and words, one
    # body with the door (bindings/trees_meta_validate_unpack.mojo): the voting value and n_jobs / verbose
    # (Voting), n_jobs / verbose, stack_method and the cv value (Stacking), then the estimators (and a
    # Stacking's final estimator) as its `_active` / `_require_member` admit them. A value it refuses is
    # a plan whose FIT raises its words first (`_refuse_first`), where its caller fell back to the door.
    var active = _Active()
    try:
        if _stack(kind):
            meta_stacking_validate(model, context)
        else:
            _ = meta_voting_validate(model, classifier)
        var entries = meta_active(py_get(fields, "estimators"), meta_owner(model), context)
        active.names = entries.names
        active.ests = entries.ests
        active.ok = True
        if _stack(kind):
            var given_final = py_get(fields, "final_estimator")
            if given_final is not None:
                prep_estimator_admit(given_final, meta_owner(model), context)
    except e:
        return _refuse_first(fields, context, b, owner, kind, String(e))
    var m = len(active.ests)
    var late = String("")
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
            # 'hard' or 'soft' by the door's equality (`meta_voting_validate` refused anything else).
            if Bool(py=py_get(fields, "voting") == PythonObject("soft")):
                flag = 1.0
        # The weights as the door reads them after its fits (`meta_voting_weights`, one body): a value
        # it refuses (an item `float` refuses, a count that is not one per member) is THE DOOR'S REFUSAL
        # raised after the nested fits (`late`); the members fit with unit weights until then.
        try:
            var given = meta_voting_weights(py_get(fields, "weights"), m)
            for i in range(m):  # driver-loop: over the user weights (one per member)
                weights.append(Float64(py=given[i]))
        except e:
            late = String(e)
            weights = List[Float64](length=m, fill=1.0)
        var needs = base_needs | (MEMBER_CAN_PREDICT_PROBA if flag != 0.0 else 0)
        for i in range(m):  # driver-loop: over the voting members (admission, once each)
            if not _admit_nested(done, nested, active.ests[i], classifier, needs, context, b, child):
                return False
    else:
        var cv = py_get(fields, "cv")
        if not owner.serving:
            _refuse_cv_callback(cv, context, b, _CV_CALLBACK_WORDS, False)
        if owner.serving:
            # A SERVING admission (lane mf-door7): the fitted stack predicts with no folds; its cv (a user's
            # splitter object too) is read only by a fit. The folds word is cv=None's 5, which NEW checks and
            # nothing here reads.
            folds = 5
        elif cv is not None and (py_type(cv) is b.list or py_type(cv) is b.tuple):
            # A user's index list (lane mf-meta3b): its folds read ONCE into
            # typed int32 rows, checked against the fit's rows by the RUN as
            # the door's `_splits` checks them.
            _ = _user_folds(cv, b, fold_train, fold_test)
            folds = len(fold_test)
        elif cv is not None and py_type(cv) is not b.int:
            # THE SPLITTER CV (lanes mf-split4, mf-split5): one of our
            # splitters, read here as typed words; each fit computes its
            # folds over its own rows (core/ensemble_member.mojo
            # `_split_folds`). An `_IterableCV` holds a user's pairs.
            _ = _split_cv(cv, context, b, split, fold_train, fold_test, _CV_CALLBACK_WORDS)
            folds = len(fold_test)
        else:
            folds = 5 if cv is None else _int_word(cv, b)
            if folds < 2:
                _ = meta_cv_value(cv, context)  # the door's words (`meta_stacking_validate` raised them first)
        # passthrough by the door's truth (`_truthy`; lane mf-refuse5: a non-bool was refused here).
        var passthrough = py_get(fields, "passthrough")
        if passthrough is not None and _truthy(passthrough, b):
            flag = 1.0
        for i in range(m):  # driver-loop: over the stacked members (admission, once each)
            # A classifier stacked with stack_method='predict' writes its
            # predicted codes (core/ensemble_member.mojo `_label_block`).
            var method = _stack_method(fields, active.ests[i], classifier, context)
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
    if len(late) != 0:
        meta_plan_refuse(Int(owner.plan.params[1]), late^, True)
    return True


def _refuse_first(fields: PythonObject, context: PythonObject, b: PythonObject, mut owner: MemberOwner, kind: Int,
                  var words: String) raises -> Bool:
    """THE DOOR'S REFUSAL first (lane mf-refuse5, core/ensemble_member.mojo `meta_plan_refuse`): a plan
    entry with no members whose FIT raises `words`, the door's own (caught from its check, one body),
    before it reads anything, as the door raises them before it reads X. False only when the meta
    estimator's x_trees image is not connected (`_finish`; the door raises the table's words)."""
    var nested = _Nested()
    if not _finish(fields, context, b, owner, nested, kind, 0.0, 0, List[Float64](), List[Int](), List[Float64](),
                   List[List[Int32]](), List[List[Int32]]()):
        return False
    meta_plan_refuse(Int(owner.plan.params[1]), words^, False)
    return True


def _finish_split(mut owner: MemberOwner, split: _Split) raises:
    """THE SPLITTER CV's words into the minted plan entry (a stack's or a
    calibration's), its entries' connections pinned for every fit."""
    if split.spec.kind != MSEL_SPEC_NONE:
        meta_plan_split(Int(owner.plan.params[1]), split.spec)
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
    """UNPACK of THE SPLITTER CV (lanes mf-split4, mf-split5): the splitter's
    typed words (bindings/msel_split_spec_run.mojo `MselSplitSpec`; kind
    MSEL_SPEC_NONE: none) and the splitters' entries with their pins."""
    var spec: MselSplitSpec
    var pin: MselRoutePin

    def __init__(out self):
        self.spec = MselSplitSpec()
        self.pin = MselRoutePin()


def _split_cv(cv: PythonObject, context: PythonObject, b: PythonObject, mut split: _Split,
              mut fold_train: List[List[Int32]], mut fold_test: List[List[Int32]], words: String) raises -> Bool:
    """A cv that is one of OUR splitter objects (lanes mf-split5, mf-ens6):
    every splitter the public split door splits, read as its typed words by
    the strict reader (bindings/msel_split_spec_unpack.mojo
    `msel_split_spec_read`, lane mf-lin6): every refusal is the split door's,
    in its words (a field it cannot read raises here; a KFold-family
    n_splits below 2 and an unreadable TimeSeriesSplit max_train_size raise
    at fit, in the door's order); each fit computes its folds over its own
    rows (core/ensemble_member.mojo `_split_folds`). An `_IterableCV`'s
    stored (train, test) pairs (DATA) are read once as an index list
    (`_user_folds`). A class the splitter table does not hold is a user's
    splitter, whose `split` would be a Python callback: refused natively in
    `words` (`_refuse_cv_callback` refuses it first when the table is
    connected). Never False (the meta door no longer keeps any splitter)."""
    if msel_split_spec_read(cv, context, split.spec, split.pin, True):
        return True
    var layouts = py_get(context, "family_msel_splitter_layouts")
    if layouts is None or not py_contains(layouts, py_type(cv)):
        raise Error(words)
    # The one class the strict reader leaves to its caller: an `_IterableCV`.
    var sp = MselSplitter(cv, context)
    return _user_folds(sp.field("_pairs", context), b, fold_train, fold_test)


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
    kept in the owner's pins), the parameter words. Never answers False
    (lane mf-fitadm7, read: an image that is not connected raises
    `_trees_key`'s words), so no caller's door is reached through it."""
    var cap = _cap_of(fields, b)
    var key = _trees_key(fields, context, b)
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
    integer array's items are read as ints; a bool is the int the door's
    int64 store reads). Lane mf-tail5b: a row number outside int32 is
    outside [0, n) for every fit, so it is written -1 and the kind's fold
    check raises the door's "a cv index is outside [0, n)" words with the
    fit's n (core/ensemble_member.mojo `_user_folds`). False only for an
    item `int()` refuses (the door's array cast raises Python's own words)."""
    var items = py_call(b.list, value)
    dst = List[Int32](capacity=len(items))
    for i in range(len(items)):  # driver-loop: over one fold side's row numbers (UNPACK, once per admission)
        var word = Int(py=py_call(b.int, items[i]))
        if word < -2147483648 or word > 2147483647:
            word = -1
        dst.append(Int32(word))
    return True


def _user_folds(cv: PythonObject, b: PythonObject, mut fold_train: List[List[Int32]],
                mut fold_test: List[List[Int32]]) raises -> Bool:
    """UNPACK of a cv given as a list of (train, test) pairs (the door's
    `_splits` over an iterable; DATA, read once): each pair's two sides as
    typed rows. Never False (lane mf-ens6): a pair that is not a two-item
    sequence raises the door's words; an item Python's tuple() or int()
    refuses raises Python's own words, as the door's tuple() and index
    reads do, before any fit."""
    for i in range(len(cv)):  # driver-loop: over the user's folds (UNPACK, once per admission)
        var both = py_call(b.tuple, cv[i])
        if len(both) != 2:
            # the door's `_splits` words (lane mf-tail5b: natively, before any fit)
            raise Error("each cv split must be a (train, test) pair")
        var tr = List[Int32]()
        var te = List[Int32]()
        _ = _index_run(both[0], b, tr)
        _ = _index_run(both[1], b, te)
        fold_train.append(tr^)
        fold_test.append(te^)
    if len(fold_test) == 0:
        raise Error("cv produced no splits")  # the door's `_splits` words, before any fit
    return True


def _tool_folds(pairs: PythonObject, context: PythonObject, mut fold_train: List[List[Int32]],
                mut fold_test: List[List[Int32]]) raises -> Bool:
    """UNPACK of THE CV TOOLS' folds (lane mf-msel5): the door's own checked
    (train, test) integer index Arrays (bindings/msel_cv_door.mojo
    `msel_cv_folds`), each side read through its int64 storage once (no
    Python object per row, as the door's units read them). False: a row
    number int32 words cannot hold (the door's units)."""
    for f in range(len(pairs)):  # driver-loop: over the door's folds (UNPACK, once per admission)
        var pair = pairs[f]
        for side in range(2):  # small-loop(train, then test)
            var rows = msel_index_i64(pair[side], context)
            var count = Int(py=rows.size)
            var dst = List[Int32](capacity=count)
            if count > 0:
                var words = msel_p64(rows)
                for i in range(count):  # driver-loop: over one side's row numbers
                    var word = Int(words[i])
                    if word < 0 or word > 2147483647:
                        return False
                    dst.append(Int32(word))
            _ = rows  # the int64 storage read above
            if side == 0:
                fold_train.append(dst^)
            else:
                fold_test.append(dst^)
    return len(fold_test) > 0


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
    if _rfe(kind):
        _rfe_pack(fields, context, b, plan, state, labels, jobs)
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
    if _rfe(kind):
        # An RFE's estimator_ (lane mf-del5b): a META one (a dotted importance_getter's) is imported by
        # this work-list first; a leaf is skipped by the caller's family test.
        var final = py_get(fields, "estimator_")
        if final is not None:
            _ = py_call(out.append, final)
        return out
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
        var children = _fitted_children(est, est_fields, est_kind, b)
        for c in range(len(children)):  # driver-loop: over one fitted meta-estimator's nested estimators
            var child = children[c]
            var child_plan = prep_estimator_plan(child, est_context)
            if child_plan is None or py_get(child_plan, "member") is None:
                continue
            var mark = member_mark(child_plan, b)
            if mark.family != MEMBER_FAMILY_ENSEMBLE:
                continue
            # Identity: an object already walked or pending is imported once.
            var walked = False
            for q in range(len(order_kinds)):  # driver-loop: over the fitted meta-estimators walked (identity)
                if order[q] is child:
                    walked = True
            for q in range(len(pending_kinds)):  # driver-loop: over the fitted meta-estimators pending (identity)
                if pending[q] is child:
                    walked = True
            if walked:
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
            owner.serving = True  # an import of fitted state: nothing here fits (lane mf-door7)
            if not ensemble_member_admit(order[i], order_fields[i], order_kinds[i], order_context[i], b, owner):
                raise Error("TypeError: mojolearn: a nested estimator cannot be imported")
            owner.plan.member = member_resolve_family(MEMBER_FAMILY_ENSEMBLE, owner.plan.route)
            var labels = py_get(order_fields[i], "classes_")
            var word = Int(owner.plan.params[0])
            if _search(word) or _rfe(word):
                # A search's classes are its best estimator's (lane mf-search4); an RFE's its estimator_'s.
                var best = py_get(order_fields[i], "best_estimator_" if _search(word) else "estimator_")
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
    if _rfe(kind):
        return _rfe_state(fields, context, b, plan, imported)
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


def _count_words(reading: MetaCount, mut words: List[Float64]):
    """`_BaggingBase._count`'s two readings of max_samples / max_features
    (bindings/trees_meta_validate_unpack.mojo `meta_count_reading`, one
    body with the door): an integral count (1.0, its value) or a fraction
    in (0, 1] (0.0, its value)."""
    if reading.integral:
        words.append(1.0)
        words.append(Float64(reading.count))
    else:
        words.append(0.0)
        words.append(reading.fraction)


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
    # Its own refusal (lane mf-ens6b): the estimator's words, raised by the
    # forest admission (one body with the door's `_rf_base` and target
    # checks), where the door raised them at its fit.
    if not forest_member_admit_common(est, sub.plan.kind, prefix, plan["context"], sub, True):
        return False
    try:
        sub.plan.member = member_resolve_family(family, sub.plan.route)
        if not sub.plan.member.has(needs):
            return False
        mask = member_kind_capabilities(sub.plan)
    except:
        return False  # the family's image is not built: the door
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
    state for import or predict only: its FIT refuses). A dict the forest
    door refuses raises that door's words (`class_weight_dictionary_values`;
    the callers make them THE DOOR'S REFUSAL, lane mf-refuse5). False only
    for a str the base's own admission already refused (unreachable)."""
    if class_weight is None:
        words.append(0.0)
        words.append(0.0)
        words.append(0.0)
        return True
    if py_type(class_weight) is b.str:
        if String(class_weight) != "balanced":
            return False  # unreachable: the base's own admission refused it (its `_rf_base`)
        words.append(1.0)
        words.append(0.0)
        words.append(0.0)
        return True
    if labels is None or len(labels) == 0:
        words.append(2.0)
        words.append(-1.0)
        words.append(0.0)
        return True
    # A dict the forest door refuses raises ITS words (`class_weight_dictionary_values`, the door's own
    # reading): the caller makes them THE DOOR'S REFUSAL (lane mf-refuse5; it answered False).
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
    them. A constructor value the door refuses is THE DOOR'S REFUSAL (lane
    mf-refuse5: its words raised by the FIT); False (the door keeps it) for
    a base or a cv the kind does not serve."""
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
    # THE DOOR'S REFUSAL (lane mf-refuse5): each unit door's constructor checks, in its order and words,
    # one body with the door (bindings/trees_meta_validate_unpack.mojo): OneVsRest's n_jobs / verbose and
    # its estimator's admission (`_require_member`); Calibrated's method, n_jobs, ensemble and cv value, then
    # its base's admission; Bagging's `_validate_params`, its max_samples / max_features readings, its
    # base's `_require_tree` and device. A value it refuses is a plan whose FIT raises the words first
    # (`_refuse_first`), where the caller fell back to the door to raise them.
    var bag_count = 0
    var bag_seed = 0
    var bag_samples = MetaCount(True, 0, 0.0)
    var bag_features = MetaCount(True, 0, 0.0)
    try:
        if kind == META_OVR:
            _ = meta_jobs(model, True)  # its words only (the kind's fit runs its own workers)
            prep_estimator_admit(py_get(fields, "estimator"), meta_owner(model), context)
        elif kind == META_CALIBRATED:
            meta_calibrated_validate(model, context)
            _ = native_unit_jobs(py_get(fields, "n_jobs"), name)  # its words only
            var cal_base = _cal_base(fields, context)
            if cal_base is not None:
                prep_estimator_admit(cal_base, meta_owner(model), context)
        else:
            meta_bagging_validate(fields, context)
            _ = native_unit_jobs(py_get(fields, "n_jobs"), name)  # its words only
            bag_count = Int(py=_int(py_get(fields, "n_estimators")))
            bag_seed = meta_seed(py_get(fields, "random_state"), context)
            bag_samples = meta_count_reading(py_get(fields, "max_samples"), "max_samples", context)
            bag_features = meta_count_reading(py_get(fields, "max_features"), "max_features", context)
            var bag_base = py_get(fields, "estimator")
            if bag_base is None:
                bag_base = _new_tree(context, "trees_dt_classifier" if classifier else "trees_dt_regressor", b)
            var bag_owner = String("BaggingClassifier") if classifier else String("BaggingRegressor")
            meta_require_tree(bag_base, bag_owner, context)
            _ = boost_member_device(fields, bag_base, bag_owner)
    except e:
        return _refuse_first(fields, context, b, owner, kind, String(e))
    if kind == META_OVR:
        var base = py_get(fields, "estimator")
        var decision = prep_estimator_has(base, "decision_function", context)
        var needs = base_needs | (MEMBER_CAN_DECISION if decision else MEMBER_CAN_PREDICT_PROBA)
        if not _admit_nested(done, nested, base, True, needs, context, b, labels):
            return False
        words.append(1.0 if decision else 0.0)
    elif kind == META_CALIBRATED:
        var base = _cal_base(fields, context)
        if base is None:
            return False
        # method and ensemble by the door's equality (`meta_calibrated_validate` refused anything else).
        var method = py_get(fields, "method")
        var sigmoid = method is None or not Bool(py=method == PythonObject("isotonic"))
        var ensemble = py_get(fields, "ensemble")
        var whole = ensemble is None or _truthy(ensemble, b)
        # The cv as the door's `_splits` takes it (lanes mf-split4b,
        # mf-split5): an int; a user's index list read ONCE into typed folds;
        # one of OUR splitters as typed words, its folds computed inside each
        # fit over that fit's rows (core/ensemble_member.mojo `_split_folds`);
        # a user's own splitter object refused here.
        var cv = py_get(fields, "cv")
        if not owner.serving:
            _refuse_cv_callback(cv, context, b, _CV_CALLBACK_WORDS, False)
        # scikit-learn's fold-count check reads an exact int cv (not None's 5): META_CAL_INT_CV.
        var int_cv = False
        if owner.serving:
            # A SERVING admission (lane mf-door7): the fitted calibration predicts through its pairs; its cv
            # (a user's splitter object too) is read only by a fit. The folds word is cv=None's 5, which NEW
            # checks and nothing here reads.
            folds = 5
        elif cv is not None and (py_type(cv) is b.list or py_type(cv) is b.tuple):
            _ = _user_folds(cv, b, fold_train, fold_test)
            folds = len(fold_test)
        elif cv is not None and py_type(cv) is not b.int:
            _ = _split_cv(cv, context, b, split, fold_train, fold_test, _CV_CALLBACK_WORDS)
            folds = len(fold_test)
        else:
            folds = 5 if cv is None else _int_word(cv, b)
            int_cv = cv is not None
            if folds < 2:
                _ = meta_cv_value(cv, context)  # the door's words (`meta_calibrated_validate` raised them first)
        # A base that takes the resident DeviceRows route (lane mf-door7): admitted as the kind's base. That
        # route (a measured winner) is CalibratedClassifierCV's own public fit's (bindings/trees_public_meta_door.mojo
        # `_cal_native`, THE RESIDENT ROWS), which never reaches this admission; a NESTED Calibrated (a member of a meta-estimator, a
        # search candidate) and a fitted one held for predict run its member calls on host rows, as every other
        # base. Before, this answered False and the caller kept its door (SHAP's `_DoorModel`, the meta doors).
        var decision = prep_estimator_has(base, "decision_function", context)
        var needs = base_needs | (MEMBER_CAN_DECISION if decision else MEMBER_CAN_PREDICT_PROBA)
        if not _admit_cal_base(done, nested, base, needs, context, b, labels, whole):
            return False
        words.append(1.0 if sigmoid else 0.0)
        words.append(1.0 if whole else 0.0)
        words.append(1.0 if decision else 0.0)
        words.append(1.0 if int_cv else 0.0)  # META_CAL_INT_CV
    else:
        # The settings as the door reads them (`meta_bagging_validate` and the readings above refused
        # anything else; lane mf-refuse5: numpy integers and floats are read as the door reads them).
        var bootstrap = py_get(fields, "bootstrap")
        var bootstrap_features = py_get(fields, "bootstrap_features")
        var oob_on = _truthy(py_get(fields, "oob_score"), b)
        var count = bag_count
        var seed = bag_seed
        words.append(Float64(count))
        words.append(Float64(seed))
        _count_words(bag_samples, words)
        _count_words(bag_features, words)
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
        try:
            if not _tree_cw_words(cw, labels, cw_context, b, words):
                return False
        except e:
            return _refuse_first(fields, context, b, owner, kind, String(e))
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
    `ada_fit` UNPACK minus X and y, read once per admission. `_ada_validate`,
    `_require_tree` and the device check first (one body with the door; a
    value they refuse is THE DOOR'S REFUSAL, lane mf-refuse5);
    the base's first round clone (`_sub_seed(seed, 0)`) admitted ONCE as
    its native forest member with the caller's target checks
    (`forest_member_admit_common(..., caller_rows=True)`, the door's
    `_ada_member_admit`), whose words carry the configuration every round
    fits (they differ only in the seed); the session route words of a
    best-splitter DecisionTree base (`_ada_native_base`: the forest entry is
    the member's route, the share word `_ada_session_choice` /
    `_member_session`). False (the door keeps the estimator) for: a base
    whose admission this kind does not serve; a class_weight value the
    base's own admission refused.
    A class_weight on the base (lane mf-split4) is the kind's class-weight
    tail (`_tree_cw_words` over `labels`, the codes the door's rows read):
    its FIT hands `ada_fit_run` the rows, as the door does."""
    var classifier = kind == META_ADA_CLASSIFIER
    # THE DOOR'S REFUSAL (lane mf-refuse5): the door's `_ada_validate` (algorithm, loss, n_estimators,
    # learning_rate, random_state; one body, bindings/trees_meta_validate_unpack.mojo) first; a value it
    # refuses is a plan whose FIT raises its words first (`_refuse_first`), where the caller fell back to
    # the door to raise them. The values are then read as the door reads them (numpy scalars too).
    var loss_code = 0
    var count = 0
    var rate = 0.0
    var seed = 0
    try:
        meta_ada_validate(fields, classifier, context)
        if not classifier:
            loss_code = Int(py=py_item(py_get(context, "trees_ada_losses"), py_get(fields, "loss")))
        count = Int(py=_int(py_get(fields, "n_estimators")))
        rate = Float64(py=_float(py_get(fields, "learning_rate")))
        seed = meta_seed(py_get(fields, "random_state"), context)
        # The door's base checks (`_require_tree`, then the device), one body with it.
        var ada_owner = String("AdaBoostClassifier") if classifier else String("AdaBoostRegressor")
        var ada_base = _ada_base(fields, classifier, context, b)
        meta_require_tree(ada_base, ada_owner, context)
        _ = boost_member_device(fields, ada_base, ada_owner)
    except e:
        return _refuse_first(fields, context, b, owner, kind, String(e))
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
    try:
        if not _tree_cw_words(class_weight, labels, probe_context, b, words):  # META_CW_* after META_ADA_WORDS
            return False
    except e:
        return _refuse_first(fields, context, b, owner, kind, String(e))
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


def _effective_kwargs(params: PythonObject, kwargs: PythonObject, b: PythonObject) raises -> PythonObject:
    """A scorer's metric options as its metric call reads them: the metric
    function's keyword defaults (`params`, its signature's parameters; None:
    unknown) updated with the scorer's own kwargs."""
    var out = Python.dict()
    if params is not None:
        var names = py_call(b.list, params)
        for i in range(len(names)):  # small-loop(the metric's parameters, once per admission)
            var default = py_call(b.getattr, params[names[i]], PythonObject("default"))
            if default is not py_call(b.getattr, params[names[i]], PythonObject("empty")):
                py_setitem(out, names[i], default)
    var given = py_call(b.list, py_call(kwargs.items))
    for i in range(len(given)):  # small-loop(the scorer's kwargs)
        py_setitem(out, given[i][0], given[i][1])
    return out


struct _ScorerKw(Movable):
    """UNPACK (lane mf-tail5b): a make_scorer's kwargs over its table row,
    read ONCE per admission: the sign flip, top_k's k, a given pos_label,
    fbeta's beta (-1: none), zero_division (-1 'warn', else the cls_epi
    code: 0, 1, 2 for nan) and a `labels` value (None: none)."""
    var flip: Bool
    var top_k: Int
    var given_pos: Bool
    var pos_label: PythonObject
    var beta: Float64
    var zd: Int
    var labels: PythonObject
    # lane mf-score6: average=None / 'samples' on an averaged label row, a
    # sample_weight kwarg (None: none) and whether the metric takes none.
    var avg_none: Bool
    var samples: Bool
    var weights: PythonObject
    var w_refused: Bool
    # lane mf-score8 (core/msel_scorer_run.mojo module note): the regression
    # options (multioutput rule, force_finite -1 / 0 / 1, alpha or power or
    # kappa's replace_undefined_by with its printed words), the words of a
    # kwarg value the metric's door refuses ("": none), the normalize /
    # adjusted flag (-1 the row's) and kappa's weights word.
    var mo: Int
    var ff: Int
    var param: Float64
    var has_param: Bool
    var param_text: String
    var refuse_text: String
    var flag: Int
    var wmode: Int
    # lane mf-score8 round b: another row's name the kwargs select (roc_auc's
    # multiclass averages; "": the matched row)
    var rename: String

    def __init__(out self):
        self.flip = False
        self.top_k = 0
        self.given_pos = False
        self.pos_label = PythonObject(None)
        self.beta = -1.0
        self.zd = -1
        self.labels = PythonObject(None)
        self.avg_none = False
        self.samples = False
        self.weights = PythonObject(None)
        self.w_refused = False
        self.mo = 0
        self.ff = -1
        self.param = 0.0
        self.has_param = False
        self.param_text = String("")
        self.refuse_text = String("")
        self.flag = -1
        self.wmode = 0
        self.rename = String("")


def _sorted_labels(value: PythonObject, b: PythonObject) raises -> PythonObject:
    """UNPACK (lane mf-score8): a `labels` value in sorted order, as the
    doors that sort it read it (`mx_sorted(mx_selected(...))`); a list that
    does not sort (mixed kinds) is kept as given (the kinds' refusal then
    reads it)."""
    var listed = py_call(b.list, value)
    try:
        return py_call(b.sorted, listed)
    except:
        return listed


def _is_sorted(value: PythonObject, b: PythonObject) raises -> Bool:
    """UNPACK (lane mf-score8): `labels == sorted(labels)` (False for one
    that does not sort)."""
    var listed = py_call(b.list, value)
    try:
        return Bool(py=py_call(b.bool, py_call(native_loaded_module("operator").eq, listed,
                                               py_call(b.sorted, listed))))
    except:
        return False


def _refuse(mut kw: _ScorerKw, words: String):
    """The first refused kwarg's words (raised per side)."""
    if kw.refuse_text.byte_length() == 0:
        kw.refuse_text = words


def _table_row_kwarg(typed: MselScorer, word: String, theirs: PythonObject, mut kw: _ScorerKw,
                     b: PythonObject) raises -> Bool:
    """UNPACK of a get_scorer row's kwarg the row reads as a word (lane
    mf-score8 round b), as its metric's door reads it: True when consumed.
    roc_auc: `max_fpr` (a real in (0, 1], else the door's words), `labels`
    (the multiclass rows: one-vs-rest checks they are ordered), `average`
    (the one-vs-rest / one-vs-one averages select their rows; None names the
    scorer; the binary row validates and ignores it); average_precision:
    `average`; log_loss: `labels` (given order); brier / the D^2 probability
    scores: `labels` (sorted); top_k: `labels` (ordered, else the door's
    words); NMI / AMI: `average_method`; v_measure: `beta`; fowlkes_mallows:
    `sparse` (it changes no value)."""
    var roc = msel_scorer_roc_problem(typed)
    if roc >= 0:
        if word == "max_fpr":
            if theirs is None:
                return True
            var v = _real_word(theirs, b)
            if not (v > 0.0 and v <= 1.0):
                _refuse(kw, "Expected max_fpr in range (0, 1], got: " + String(py=py_call(b.repr, theirs)))
                return True
            kw.param = v
            kw.has_param = True
            kw.param_text = String(py=py_call(b.str, theirs))
            return True
        if word == "labels":
            if theirs is None or roc == ROC_BINARY:
                return True  # the binary arm reads no labels
            if (roc == ROC_OVR or roc == ROC_OVR_WEIGHTED) and not _is_sorted(theirs, b):
                _refuse(kw, "Parameter 'labels' must be ordered")
                return True
            kw.labels = theirs
            return True
        if word == "average":
            if roc == ROC_BINARY:
                var ok = theirs is None
                if not ok and py_type(theirs) is b.str:
                    var a = String(py=theirs)
                    ok = a == "micro" or a == "macro" or a == "samples" or a == "weighted"
                if not ok:
                    _refuse(kw, "average has to be one of (None, 'micro', 'macro', 'weighted', 'samples')")
                return True
            var ovr = roc == ROC_OVR or roc == ROC_OVR_WEIGHTED
            var text = String("") if theirs is None or py_type(theirs) is not b.str else String(py=theirs)
            if theirs is None:
                if ovr:
                    kw.rename = String("roc_auc(multi_class='ovr', average=None)")
                    kw.avg_none = True
                else:
                    _refuse(kw, "NotImplementedError: average=None is not implemented for multi_class='ovo'.")
            elif text == "macro":
                kw.rename = String("roc_auc_ovr") if ovr else String("roc_auc_ovo")
            elif text == "weighted":
                kw.rename = String("roc_auc_ovr_weighted") if ovr else String("roc_auc_ovo_weighted")
            elif text == "micro" and ovr:
                kw.rename = String("roc_auc(multi_class='ovr', average='micro')")
            elif ovr:
                _refuse(kw, "average must be one of ('micro', 'macro', 'weighted', None) for multiclass problems")
            else:
                _refuse(kw, "average must be one of ('macro', 'weighted', None) for multiclass problems")
            return True
        return False
    if msel_scorer_ap(typed) and word == "average":
        if theirs is None:
            kw.flag = 3
            kw.avg_none = True
            return True
        var a = String(py=theirs) if py_type(theirs) is b.str else String("")
        if a == "macro":
            return True
        if a == "weighted":
            kw.flag = 1
        elif a == "micro":
            kw.flag = 2
        elif a == "samples":
            kw.flag = 4
        else:
            _refuse(kw, "average has to be one of (None, 'micro', 'macro', 'weighted', 'samples')")
        return True
    if word == "labels" and (msel_scorer_log_loss(typed) or msel_scorer_brier(typed) or msel_scorer_top_k(typed)):
        if theirs is None:
            return True
        if msel_scorer_log_loss(typed):
            kw.labels = theirs
        elif msel_scorer_top_k(typed):
            if not _is_sorted(theirs, b):
                _refuse(kw, "Parameter 'labels' must be ordered.")
                return True
            kw.labels = theirs
        else:
            kw.labels = _sorted_labels(theirs, b)
        return True
    if word == "average_method" and msel_scorer_mi(typed):
        var m = String(py=theirs) if py_type(theirs) is b.str else String("")
        if m == "min":
            kw.flag = 0
        elif m == "geometric":
            kw.flag = 1
        elif m == "arithmetic":
            kw.flag = 2
        elif m == "max":
            kw.flag = 3
        else:
            _refuse(kw, "'average_method' must be 'min', 'geometric', 'arithmetic', or 'max'")
        return True
    if word == "beta" and msel_scorer_v_measure(typed):
        var beta = _real_word(theirs, b)
        if beta != beta:
            return False  # not a real number: no typed word reads it
        kw.param = beta
        kw.has_param = True
        kw.param_text = String(py=py_call(b.str, theirs))
        return True
    if word == "sparse" and String(typed.name) == "fowlkes_mallows_score":
        return True  # deprecated, reads no value
    return False


def _bool_word(value: PythonObject, b: PythonObject) raises -> Int:
    """UNPACK (lane mf-score8): a kwarg the metric door reads as a bool
    (`mx_is_bool` / `mc_is_bool`: Python's bool or NumPy's bool_) as 0 / 1;
    -1 for any other value (the door's "must be a bool" refusal)."""
    if not Bool(py=py_call(b.isinstance, value, b.bool)) and String(py=py_type(value).__name__) != "bool_":
        return -1
    return 1 if Bool(py=py_call(b.bool, value)) else 0


def _metric_row_kwarg(metric: String, word: String, value: PythonObject, mut kw: _ScorerKw, mut normalize: Bool,
                      context: PythonObject, b: PythonObject) raises -> Bool:
    """UNPACK of one kwarg of a METRIC ROW (lane mf-score8; core/
    msel_scorer_run.mojo `msel_scorer_metric_name`): read ONCE into `kw` as
    the metric's door reads it (bindings/metrics_reg_door.mojo,
    metrics_exp_cls_door.mojo, mcls_classify_door.mojo). A value that door
    refuses becomes `kw.refuse_text`, the door's words, raised per side.
    False: a kwarg no typed word carries (the caller answers "")."""
    var reg_mo = (metric == "r2_score" or metric == "explained_variance_score" or metric == "mean_squared_error"
                  or metric == "mean_absolute_error" or metric == "root_mean_squared_error"
                  or metric == "mean_squared_log_error" or metric == "root_mean_squared_log_error"
                  or metric == "mean_absolute_percentage_error" or metric == "median_absolute_error"
                  or metric == "d2_absolute_error_score" or metric == "mean_pinball_loss"
                  or metric == "d2_pinball_score")
    if word == "multioutput" and reg_mo:
        # `_reg_new`'s rule on a 1-D target: a string among the allowed ones
        # (variance only for r2 / explained variance), None the default,
        # custom weights refused ("useful only in multi-output cases")
        if value is None:
            return True
        if Bool(py=py_call(b.isinstance, value, b.str)):
            var text = String(py=value)
            var variance = metric == "r2_score" or metric == "explained_variance_score"
            if text == "uniform_average":
                kw.mo = 0
            elif text == "raw_values":
                kw.mo = 1
            elif text == "variance_weighted" and variance:
                kw.mo = 2
            elif kw.refuse_text.byte_length() == 0:
                var allowed = String("('raw_values', 'uniform_average', 'variance_weighted')") if variance else String(
                    "('raw_values', 'uniform_average')")
                kw.refuse_text = ("Allowed 'multioutput' string values are " + allowed + ". You provided multioutput="
                                  + String(py=py_call(b.repr, value)))
            return True
        if kw.refuse_text.byte_length() == 0:
            try:
                _ = classical_view(value, context, "multioutput")
                kw.refuse_text = String("Custom weights are useful only in multi-output cases.")
            except error:
                kw.refuse_text = String(error)
        return True
    if word == "force_finite" and (metric == "r2_score" or metric == "explained_variance_score"):
        kw.ff = 1 if Bool(py=py_call(b.bool, value)) else 0  # the door's `bool(force_finite)`
        return True
    if word == "alpha" and (metric == "mean_pinball_loss" or metric == "d2_pinball_score"):
        var alpha = _real_word(value, b)
        if not (alpha >= 0.0 and alpha <= 1.0):
            if kw.refuse_text.byte_length() == 0:
                kw.refuse_text = String("alpha must be a real number in [0, 1]")  # `_check_alpha`
            return True
        kw.param = alpha
        kw.has_param = True
        kw.param_text = String(py=py_call(b.str, value))
        return True
    if word == "power" and (metric == "mean_tweedie_deviance" or metric == "d2_tweedie_score"):
        if py_type(value) is b.bool or not Bool(py=py_call(b.isinstance, value, native_loaded_module("numbers").Real)):
            if kw.refuse_text.byte_length() == 0:
                kw.refuse_text = String("power must be a real number")  # `_tweedie_power`
            return True
        kw.param = Float64(py=py_call(b.float, value))
        kw.has_param = True
        kw.param_text = String(py=py_call(b.str, value))  # `family_str(power)`: the domain words print it
        return True
    if word == "normalize" and (metric == "zero_one_loss" or metric == "accuracy_score"):
        var flag = _bool_word(value, b)
        if flag < 0:
            if kw.refuse_text.byte_length() == 0:
                kw.refuse_text = String("normalize must be a bool")
            return True
        normalize = flag == 1
        return True
    if word == "adjusted" and metric == "balanced_accuracy_score":
        kw.flag = 1 if Bool(py=py_call(b.bool, value)) else 0  # the door's `mx_truthy(adjusted)`
        return True
    if word == "labels" and (metric == "hinge_loss" or metric == "d2_log_loss_score" or metric == "d2_brier_score"):
        # lane mf-score8 round b: sorted, as `_hinge` / `_proba` sort them
        if value is not None:
            kw.labels = _sorted_labels(value, b)
        return True
    if word == "pos_label" and metric == "d2_brier_score":
        if value is not None:
            kw.given_pos = True
            kw.pos_label = value
        return True
    if metric == "cohen_kappa_score":
        if word == "labels":
            if value is not None:
                kw.labels = value
            return True
        if word == "weights":
            if value is None:
                kw.wmode = 0
            elif Bool(py=py_call(b.isinstance, value, b.str)) and String(py=value) == "linear":
                kw.wmode = 1
            elif Bool(py=py_call(b.isinstance, value, b.str)) and String(py=value) == "quadratic":
                kw.wmode = 2
            elif kw.refuse_text.byte_length() == 0:
                kw.refuse_text = String("weights must be None, 'linear' or 'quadratic'")
            return True
        if word == "replace_undefined_by":
            # the door answers the given object itself when kappa is
            # undefined: a real number is that score (NaN included)
            if py_type(value) is b.bool or not Bool(py=py_call(b.isinstance, value,
                                                               native_loaded_module("numbers").Real)):
                return False
            kw.param = Float64(py=py_call(b.float, value))
            kw.has_param = True
            kw.param_text = String(py=py_call(b.str, value))
            return True
    return False


def _real_word(value: PythonObject, b: PythonObject) raises -> Float64:
    """A kwarg's real value (a Python or NumPy int / float, not a bool), or
    NaN for any other value (the caller's door then keeps the scorer)."""
    if py_type(value) is b.bool:
        return _nan()
    var numbers = native_loaded_module("numbers")
    if not Bool(py=py_call(b.isinstance, value, numbers.Real)):
        return _nan()
    return Float64(py=py_call(b.float, value))


def _nan() -> Float64:
    var zero = Float64(0.0)
    return zero / zero


def _scorer_name(scorer: PythonObject, context: PythonObject, b: PythonObject, mut kw: _ScorerKw) raises -> String:
    """UNPACK: the table name of ONE scorer value: a str is itself (an
    unknown name raises `msel_get_scorer`'s words); a scorer object is
    admitted by `msel_get_scorer` (a Python callback raises the door's
    words) and read back to the name the scorer table holds it under (the
    object `get_scorer(name)` answers). "": a native scorer no table row
    holds (the door scores it).

    A NATIVE MAKE_SCORER SCORER (lane mf-cluster5): its typed parts (the
    metric it wraps, its kwargs, its response method) read ONCE here and
    matched against the table's rows; the row whose parts are the same is
    its name, and `kw.flip` says its sign (greater_is_better) is the row's
    negated. Both sides' options are compared as the metric reads them (its
    keyword defaults updated with the kwargs, so an explicit default is the
    row's; lane mf-msel5c). THE KWARGS A ROW TAKES AS WORDS (lanes
    mf-msel5c, mf-tail5b; core/msel_scorer_run.mojo `MselScorerOptions`):
    a binary label row's, 'average_precision''s or 'neg_brier_score''s other
    `pos_label`; top_k_accuracy's `k`; an averaged label row's `labels`
    (read before any comparison: an array kwarg compares no other way);
    `zero_division` 0 / 1 on a label row (NaN on 'fbeta' only, whose door
    takes it); and 'fbeta_score' itself, which is the F row of its
    `average` with its `beta` (a real >= 0). A pos_label of None or 1 on an
    averaged row is the row's (scikit-learn ignores it there). Any other
    difference: "" (the door)."""
    kw = _ScorerKw()
    var got = msel_get_scorer(scorer, context)
    if py_type(scorer) is b.str:
        return String(py=scorer)
    var table = context["family_msel_scorers"]
    var items = py_call(b.list, py_call(table.items))
    for i in range(len(items)):  # driver-loop: over the scorer table (a fixed table, once per admission)
        if items[i][1] is got:
            return String(py=items[i][0])
    var func = py_call(b.getattr, got, PythonObject("_score_func"), PythonObject(None))
    if func is None:
        return String("")
    # 'fbeta_score' is the F row of its average (its module's f1_score's rows)
    var fbeta = String(py=py_call(b.getattr, func, PythonObject("__name__"), PythonObject(""))) == "fbeta_score"
    var module = py_call(b.getattr, func, PythonObject("__module__"), PythonObject(None))
    var eq = native_loaded_module("operator").eq
    var params = PythonObject(None)
    try:
        params = py_call(native_loaded_module("inspect").signature, func).parameters
    except:
        params = PythonObject(None)  # no signature: the kwargs are compared as given
    var missing = py_call(b.object)
    try:
        var got_kwargs = _effective_kwargs(params, py_call(b.getattr, got, PythonObject("_kwargs")), b)
        var got_method = py_call(b.getattr, got, PythonObject("_response_method"))
        for i in range(len(items)):  # driver-loop: over the scorer table (its rows' typed parts, once per admission)
            var row = items[i][1]
            var row_func = py_call(b.getattr, row, PythonObject("_score_func"))
            if fbeta:
                if String(py=py_call(b.getattr, row_func, PythonObject("__name__"), PythonObject(""))) != "f1_score":
                    continue
                if not Bool(py=py_call(b.bool, py_call(eq, py_call(b.getattr, row_func, PythonObject("__module__"),
                                                                   PythonObject(None)), module))):
                    continue
            elif row_func is not func:
                continue
            if not Bool(py=py_call(b.bool, py_call(eq, py_call(b.getattr, row, PythonObject("_response_method")),
                                                   got_method))):
                continue
            var name = String(py=items[i][0])
            var code = msel_scorer_code(name, False)
            if code < 0:
                continue
            var typed = msel_scorer_row(code)
            var binary = msel_scorer_positive(typed)
            var label_row = msel_scorer_label_row(typed)
            var pos_row = binary or msel_scorer_ap(typed) or msel_scorer_brier(typed)
            var ranked = msel_scorer_top_k(typed)
            var row_kwargs = _effective_kwargs(params, py_call(b.getattr, row, PythonObject("_kwargs")), b)
            var keyset = py_call(b.set, py_call(b.list, row_kwargs))
            var keys = py_call(b.list, py_call(keyset.union, py_call(b.list, got_kwargs)))
            var same = True
            var mine_kw = _ScorerKw()
            for q in range(len(keys)):  # small-loop(the two scorers' options)
                var key = keys[q]
                var mine = py_call(row_kwargs.get, key, missing)
                var theirs = py_call(got_kwargs.get, key, missing)
                var word = String(py=key) if py_type(key) is b.str else String("")
                if theirs is not missing and _table_row_kwarg(typed, word, theirs, mine_kw, b):
                    continue  # lane mf-score8 round b: a typed word of the row
                if word == "labels" and theirs is not missing and label_row > 0:
                    # read first: an array compares no other way (labels None is the row's)
                    if theirs is not None and label_row == 2:
                        mine_kw.labels = theirs
                    continue
                if word == "beta" and fbeta and theirs is not missing and msel_scorer_f_row(typed):
                    var beta = _real_word(theirs, b)
                    if not (beta >= 0.0):
                        same = False  # the door raises its F-beta words
                        break
                    mine_kw.beta = beta
                    continue
                if word == "sample_weight" and theirs is not missing and theirs is not None:
                    # A MAKE_SCORER's sample_weight (lane mf-score6): data read
                    # once at the admission below; a metric without the
                    # parameter (`mine` missing) is refused per side in the
                    # door's words (the RUN, `w_n` -1). Read before any
                    # comparison: an array compares no other way.
                    if mine is missing:
                        mine_kw.w_refused = True
                    else:
                        mine_kw.weights = theirs
                    continue
                if word == "normalize" and theirs is not missing and (msel_scorer_log_loss(typed) or ranked):
                    # lane mf-score8: log_loss's normalize (`mc_is_bool`, else
                    # the door's words per side) and top_k_accuracy_score's
                    # (`mx_truthy`), as the RUN's flag word
                    if msel_scorer_log_loss(typed):
                        var nf = _bool_word(theirs, b)
                        if nf < 0:
                            if mine_kw.refuse_text.byte_length() == 0:
                                mine_kw.refuse_text = String("normalize must be a bool")
                        else:
                            mine_kw.flag = nf
                    else:
                        mine_kw.flag = 1 if Bool(py=py_call(b.bool, theirs)) else 0
                    continue
                if word == "scale_by_half" and theirs is not missing and msel_scorer_brier(typed):
                    # lane mf-score8: brier_score_loss's scale_by_half ('auto' the
                    # row's; else the door's `mx_truthy`)
                    if not (py_type(theirs) is b.str and String(py=theirs) == "auto"):
                        mine_kw.flag = 1 if Bool(py=py_call(b.bool, theirs)) else 0
                    continue
                if word == "average" and theirs is not missing and label_row == 2:
                    # average=None / 'samples' on an averaged label row (lane
                    # mf-score6): the per-label values, or the door's refusal
                    if theirs is None:
                        mine_kw.avg_none = True
                        continue
                    if py_type(theirs) is b.str and String(py=theirs) == "samples":
                        mine_kw.samples = True
                        continue
                if mine is not missing and theirs is not missing and Bool(py=py_call(b.bool, py_call(eq, mine,
                                                                                                    theirs))):
                    continue
                if word == "pos_label" and theirs is not missing:
                    if pos_row:
                        mine_kw.given_pos = True
                        mine_kw.pos_label = theirs
                        continue
                    if theirs is None or (py_type(theirs) is b.int and Int(py=theirs) == 1):
                        continue  # scikit-learn ignores pos_label on an averaged row
                if word == "k" and ranked and py_type(theirs) is b.int:
                    mine_kw.top_k = Int(py=theirs)
                    if mine_kw.top_k < 1:
                        same = False  # the metric refuses it in its words: the door
                        break
                    continue
                if word == "zero_division" and label_row > 0 and theirs is not missing:
                    if py_type(theirs) is b.str and String(py=theirs) == "warn":
                        continue
                    var zv = _real_word(theirs, b)
                    if zv == 0.0 or zv == 1.0:
                        mine_kw.zd = Int(zv)
                        continue
                    if zv != zv and fbeta and py_type(theirs) is not b.bool and Bool(
                            py=py_call(b.isinstance, theirs, b.float)):
                        mine_kw.zd = 2  # NaN: 'fbeta''s door takes it (`_epi_zd`)
                        continue
                same = False
                break
            if not same:
                continue
            if fbeta and mine_kw.beta < 0.0:
                continue  # fbeta_score without its beta: the door raises its words
            mine_kw.flip = not Bool(py=py_call(b.bool, py_call(eq, py_call(b.getattr, row, PythonObject("_sign")),
                                                               py_call(b.getattr, got, PythonObject("_sign")))))
            if mine_kw.rename.byte_length() > 0:
                name = mine_kw.rename.copy()  # lane mf-score8: the row the kwargs select
            kw = mine_kw^
            return name
        # THE METRIC ROWS (lane mf-fitadm7; core/msel_scorer_run.mojo
        # `msel_scorer_metric_code`): a make_scorer over a connected metric
        # no get_scorer name holds, response 'predict', whose kwargs are
        # only the ones its typed row reads: zero_one_loss's `normalize` (an
        # exact bool), `sample_weight` (data, read once as the other rows'),
        # `numeric_mode` None. Its row's sign is the metric's value: the
        # flip is greater_is_better=False.
        var plan = py_get(context["family_msel_metric_plans"], func)
        if plan is None or py_type(got_method) is not b.str:
            return String("")
        var metric = String(py=plan["name"])
        var normalize = True
        var metric_kw = _ScorerKw()
        # only the kwargs the scorer gave (the metric's defaults are its row's)
        var given = py_call(b.list, py_call(py_call(b.getattr, got, PythonObject("_kwargs")).items))
        for q in range(len(given)):  # small-loop(the scorer's options, once per admission)
            var word = String(py=given[q][0]) if py_type(given[q][0]) is b.str else String("")
            var value = given[q][1]
            if word == "sample_weight":
                if value is not None:
                    metric_kw.weights = value
            elif word == "numeric_mode":
                if value is not None:
                    return String("")
            elif not _metric_row_kwarg(metric, word, value, metric_kw, normalize, context, b):
                return String("")
        var code = msel_scorer_metric_code(metric, normalize)
        if code < 0:
            return String("")
        # lane mf-score8 round b: the make_scorer's response is the row's
        # (predict; hinge_loss's decision_function; the D^2 probability
        # scores' predict_proba)
        var op = msel_scorer_row(code).op
        var wanted = String("decision_function") if op == MEMBER_OP_DECISION else (
            String("predict_proba") if op == MEMBER_OP_PREDICT_PROBA else String("predict"))
        if String(py=got_method) != wanted:
            return String("")
        # lane mf-score8: the row's sign is its get_scorer sign (or +1 for a
        # metric row); the flip is a make_scorer sign that differs from it
        var row_sign = Int(msel_scorer_row(code).sign)
        metric_kw.flip = Int(py=py_call(b.int, py_call(b.getattr, got, PythonObject("_sign")))) != row_sign
        kw = metric_kw^
        return String(msel_scorer_row(code).name)
    except:
        return String("")  # options no exact comparison reads: the door
    return String("")


def _search_served(fields: PythonObject) raises -> PythonObject:
    """The one nested estimator of a SERVING search admission: its fitted `best_estimator_`, else (an unfitted
    search inside a fitted parent, whose fitted clones the parent's import reads) its `estimator`, the class
    every candidate and the best are clones of."""
    var best = py_get(fields, "best_estimator_")
    if best is not None:
        return best
    var base = py_get(fields, "estimator")
    return base


def _admit_search_serving(fields: PythonObject, context: PythonObject, b: PythonObject, mut owner: MemberOwner,
                          done: _Done, labels: PythonObject) raises -> Bool:
    """UNPACK of THE SEARCH KIND for a caller that only predicts on fitted state (lane mf-door7,
    `MemberOwner.serving`: SHAP's members, a meta-estimator's fitted member held for predict, a META import).
    A fitted search predicts through its `best_estimator_` (scikit-learn's `BaseSearchCV.predict` /
    `predict_proba` / `decision_function` delegate to it), so its scorers, cv, splitter object, candidates and
    device, which only its fit reads, are not read: one nested plan, `_search_served`'s estimator, admitted
    for predict, the head words of scoring=None with cv=None's 5 folds (which NEW checks; nothing fits), and
    META_PARAM_SERVING set (`_search_state` imports the best as candidate 0). Before, every
    scorer, splitter or candidate list the FIT admission does not take answered False here and the caller kept
    its door (SHAP's `_DoorModel`). False only for a nested estimator with no serving member, or an x_trees
    image that is not built."""
    var base = _search_served(fields)
    if base is None:
        return False
    var base_plan = prep_estimator_plan(base, context)
    if base_plan is None or py_get(base_plan, "member") is None:
        return False
    var classifier = False
    if member_mark(base_plan, b).family == MEMBER_FAMILY_ENSEMBLE:
        var at = done.find(base)
        if at < 0:
            return False
        classifier = (done.masks[at] & MEMBER_CAN_CLASSIFIER) != 0
    else:
        classifier = Bool(py=base_plan["classifier"])
    var needs = MEMBER_CAN_PREDICT | (MEMBER_CAN_CLASSIFIER if classifier else 0)
    var nested = _Nested()
    if not _admit_nested(done, nested, base, classifier, needs, context, b, labels):
        return False
    var trees = py_call(native_loaded_module("sys").modules.get, PythonObject("mojolearn._expansion_trees"))
    if trees is None:
        try:
            trees = py_call(native_loaded_module("importlib").import_module,
                            PythonObject("mojolearn._expansion_trees"))
        except:
            return False  # the x_trees image is not built: no META image to run on
    var trees_context = py_get(py_call(b.vars, trees), "_TREES_CONTEXT")
    if trees_context is None:
        return False
    # The head words of a search that is never fitted here (`_admit_search`'s layout for scoring=None: scorer
    # 0, no tolerance, no train scores, a grid's seed -1, refit True, refit scorer 0, count 0) and its one
    # scorer group.
    var words: List[Float64] = [0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0]
    var sels = List[Int32]()
    msel_scorer_group_words(words, 0, -1, MselScorerOptions(), sels)
    var kind = META_SEARCH_CLASSIFIER if classifier else META_SEARCH_REGRESSOR
    if not _finish(fields, trees_context, b, owner, nested, kind, 0.0, 5, List[Float64](), List[Int](), words^,
                   List[List[Int32]](), List[List[Int32]]()):
        return False
    owner.plan.params[META_PARAM_SERVING] = 1.0
    return True


def _admit_search(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
                  mut owner: MemberOwner, done: _Done, prepared: PythonObject, labels: PythonObject) raises -> Bool:
    """UNPACK of THE SEARCH KIND (core/ensemble_member.mojo): every
    candidate (the base's clone with the candidate's parameters set) admitted
    as a member with those words, read as `msel_search_fit` reads the
    search's settings: the scorer LIST (one scorer, or a multi-metric list /
    dict in its name order: codes, responses, positive class codes) and the
    refit word (True, False or a scorer name; lane mf-search4b items 1, 3),
    an int cv, one of our KFold / StratifiedKFold splitters or a user's
    index list (item 6). Answers False (the door's fit keeps the search) for
    a scorer the typed RUN does not serve (core/msel_scorer_run.mojo), a
    binary scorer whose pos_label is not one of the labels, another of our
    splitters, and any candidate with no serving member. A Python scorer,
    refit or cv callback is refused in the door's words.

    THE SEARCH'S OWN `device` (lane mf-search5) says where its fold rows
    live, as the door reads it (bindings/msel_search_door.mojo
    `_search_host_folds`): 'cpu' beside a GPU set, or a candidate that names
    a `device`, keeps them on the host; otherwise the kind's `flag` word lets
    a caller in a resident scope hand the RUN its scope and X's pin
    (core/ensemble_member.mojo `_fit_search`; a member keeps its own
    `device`). SCORING None over a classifier whose `score` is the prep
    classifiers' mean accuracy (mojolearn._expansion_prep `_Classifier.score`:
    GaussianNB and the other naive Bayes, LDA, QDA; their member has no score
    operation) is the 'accuracy' scorer, which is what that `score` is."""
    var scoring_in = py_get(fields, "scoring")
    # THE SCORERS: (name, scorer value as given) in the caller's order; an
    # empty list: scoring=None (the member's own score).
    var many = scoring_in is not None and _scoring_many(scoring_in, context, b)
    var keys = List[String]()
    var givens = Python.list()
    if many:
        # the door's admission (bindings/msel_cv_door.mojo `msel_scorers`:
        # exact str names / keys, in their order) and its words
        var mapping = not (py_type(scoring_in) is b.list or py_type(scoring_in) is b.tuple
                           or py_type(scoring_in) is b.set)
        var names = py_call(b.list, scoring_in)
        for i in range(len(names)):  # driver-loop: over the scorer names (once per admission)
            if py_type(names[i]) is not b.str:
                if mapping:
                    raise Error("TypeError: multiple scoring keys must be exact strings")
                raise Error("TypeError: multiple scoring names must be exact strings")
            var repeated = False
            for j in range(len(keys)):  # small-loop(the names so far: the door's dict keeps one of each)
                repeated = repeated or keys[j] == String(py=names[i])
            if repeated:
                continue
            keys.append(String(py=names[i]))
            _ = py_call(givens.append, scoring_in[names[i]] if mapping else names[i])
        if len(keys) == 0:
            # AN EMPTY MULTI-METRIC SCORING (lane mf-ens6b): scikit-learn's
            # own fit fails (sklearn 1.9 metrics/_scorer.py
            # `_check_multimetric_scoring`, lines 634 and 644, from the
            # search's `_get_scorers` before any fit); its words, natively.
            raise Error(msel_empty_scoring_words(mapping, String(py=py_call(b.repr, scoring_in))))
    elif scoring_in is not None:
        keys.append(String(""))
        _ = py_call(givens.append, scoring_in)
    var refit = py_get(fields, "refit")
    if refit is None or (py_type(refit) is not b.bool and py_type(refit) is not b.str):
        # THE DOOR'S REFUSAL of a refit callback (lane mf-ens6b; bindings/
        # msel_search_door.mojo `msel_search_fit`), in its order: no
        # candidates, then its scorers (`msel_scorers`: each name's
        # `msel_get_scorer`), then the refit words.
        var drawn = _search_prepare(model, fields, context, b, prepared)
        if drawn is not None and len(drawn[1]) == 0:
            raise Error(MSEL_NO_FITS_WORDS)
        if many:
            for j in range(len(givens)):  # driver-loop: over the scorer values (the door's admission, once)
                _ = msel_get_scorer(givens[j], context)
        raise Error(MSEL_REFIT_CALLBACK_WORDS)
    var refit_false = py_type(refit) is b.bool and not Bool(py=refit)
    var refit_scorer = 0
    if many and not refit_false:
        refit_scorer = -1
        if py_type(refit) is b.str:
            for j in range(len(keys)):  # small-loop(the scorer names)
                if keys[j] == String(py=refit):
                    refit_scorer = j
        if refit_scorer < 0:
            raise Error("For multi-metric scoring, refit must be False or a scorer key.")
    # THE CV TOOLS (lane mf-msel5): the door's carrier search of
    # cross_validate / cross_val_score / learning_curve / cross_val_predict
    # (bindings/msel_cv_door.mojo `msel_cv_by_search`) holds
    # `_msel_cv_tool` = (return_estimator, method or None, return_indices,
    # the door's folds, a one-slot holder list, the target's outputs (0: a
    # 1-D y), the rows), set by that door, never by a user: the RUN fits
    # those folds, keeps the units' states, answers the folds' rows, or
    # writes the test-side `method` instead of a score straight into the
    # result Array this UNPACK places in the holder (lane mf-msel5b: one
    # buffer, written once). A public search has no such field.
    var tool = py_get(fields, "_msel_cv_tool")
    var tool_keep = False
    var tool_respond = -1
    var tool_indices = False
    var tool_log = False
    if tool is not None:
        tool_keep = Bool(py=py_call(b.bool, tool[0]))
        tool_indices = Bool(py=py_call(b.bool, tool[2]))
        if tool[1] is not None:
            var method = String(py=tool[1])
            if method == "predict":
                tool_respond = MEMBER_OP_PREDICT
            elif method == "predict_proba":
                tool_respond = MEMBER_OP_PREDICT_PROBA
            elif method == "decision_function":
                tool_respond = MEMBER_OP_DECISION
            elif method == "predict_log_proba":
                # `np.log(predict_proba(X))` in the member's precision (lane
                # mf-msel5b); a class without the method (lane mf-ens6b): the
                # door's units' `getattr` words, raised here by the same
                # attribute read before any fit
                if py_call(b.getattr, py_get(fields, "estimator"), PythonObject("predict_log_proba"),
                           PythonObject(None)) is None:
                    _ = py_call(b.getattr, py_get(fields, "estimator"), PythonObject("predict_log_proba"))
                tool_respond = MEMBER_OP_PREDICT_PROBA
                tool_log = True
            else:
                # ANOTHER METHOD (lane mf-ens6b): scikit-learn's own call fails
                # (sklearn 1.9 model_selection/_validation.py line 1018,
                # cross_val_predict's `method` StrOptions); its words, the
                # options in sorted order (a set prints in hash order).
                raise Error(msel_cvp_method_words(String(py=py_call(b.repr, tool[1]))))
            if len(keys) > 0:
                return False
    # Where the fold rows live (the docstring): read once per admission.
    var resident = not msel_member_host_folds(fields, b, context)
    var cv = py_get(fields, "cv")
    # A user's own splitter object: refused here in the search door's words
    # (lane mf-split4b; never called, `_refuse_cv_callback`).
    _refuse_cv_callback(cv, context, b, _SEARCH_CV_CALLBACK_WORDS, True)
    var fold_train = List[List[Int32]]()
    var fold_test = List[List[Int32]]()
    var split = _Split()
    var folds = 0
    if tool is not None:
        # THE CV TOOLS: the door's own folds (already split and checked).
        if not _tool_folds(tool[3], context, fold_train, fold_test):
            return False
        folds = len(fold_test)
    elif cv is not None and (py_type(cv) is b.list or py_type(cv) is b.tuple):
        # A user's index list: its folds read ONCE into typed int32 rows,
        # checked against the fit's rows by the kind (as Stacking's).
        _ = _user_folds(cv, b, fold_train, fold_test)
        folds = len(fold_test)
    elif cv is not None and py_type(cv) is not b.int and not msel_is_integral(cv, context):
        # One of our splitters, typed words; each fit splits its own rows
        # (the search's groups are its call's, `MemberCall.groups`).
        _ = _split_cv(cv, context, b, split, fold_train, fold_test, _SEARCH_CV_CALLBACK_WORDS)
        folds = len(fold_test)
    else:
        # None, an int or any integral value (a NumPy integer; lane mf-ens6b:
        # read as `int(cv)`, as the door's split reads it, where it kept the
        # door before)
        folds = 5 if cv is None else Int(py=py_call(b.int, cv))
        if folds < 2:
            # KFold's own refusal (lane mf-tail5b item 4, natively in its
            # words; the RUN raises the same for a fit's n_splits)
            raise Error(msel_kfold_words(folds))
    var error_score = py_get(fields, "error_score")
    var tolerate = 0.0
    var error_value = 0.0
    if error_score is not None and py_type(error_score) is b.str:
        if String(py=error_score) != "raise":
            # scikit-learn's parameter validation (lane mf-msel5c; the doors'
            # `msel_unit_jobs` raise the same words before any fold)
            raise Error("ValueError: The 'error_score' parameter of " + String(py=py_type(model).__name__)
                        + " must be a str among {'raise'} or an instance of 'float'. Got "
                        + String(py=py_call(b.repr, error_score)) + " instead.")
    elif Bool(py=py_call(b.isinstance, error_score, native_loaded_module("numbers").Real)):
        # an instance of 'float' as scikit-learn's `Real` constraint reads it
        # (lane mf-tail5b: NumPy reals and bools too)
        tolerate = 1.0
        error_value = Float64(py=py_call(b.float, error_score))
    else:
        # scikit-learn's parameter validation, natively (lane mf-tail5b item 4)
        raise Error("ValueError: The 'error_score' parameter of " + String(py=py_type(model).__name__)
                    + " must be a str among {'raise'} or an instance of 'float'. Got "
                    + String(py=py_call(b.repr, error_score)) + " instead.")
    var train = py_get(fields, "return_train_score")
    var return_train = train is not None and _truthy(train, b)
    var got = _search_prepare(model, fields, context, b, prepared)
    if got is None:
        return False
    var clones = got[1]
    var seed = Int(py=got[2])
    if len(clones) == 0:
        # the RUN's and the door's words (lane mf-tail5b item 4, natively)
        raise Error(MSEL_NO_FITS_WORDS)
    if resident:
        var candidates = got[0]
        for c in range(len(candidates)):  # driver-loop: over the candidates' parameter dicts, names only
            if msel_params_name_device(candidates[c], b):
                resident = False
                break
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
    # A stratified splitter over a regressor (lane mf-tail5b item 4): the
    # kind's split stratifies the fit's given targets and raises the
    # splitter's own words for a continuous one (core/ensemble_member.mojo
    # `_split_folds`, lane mf-del5), so it is admitted.
    # THE SCORERS (lanes mf-metric4, mf-search4b): each a name the typed
    # search RUN serves (core/msel_scorer_run.mojo `msel_scorer_code`), with
    # the response its scorer reads off the base's class
    # (`msel_scorer_response`, the scorer's own rule: 'roc_auc' takes
    # decision_function when the class has it) and, for a binary label
    # scorer, its pos_label=1 as the class code among the fit's labels; any
    # other scorer keeps the door.
    var codes = List[Int]()
    var positives = List[Int]()
    # Each scorer's make_scorer options and their `labels` codes (lane
    # mf-tail5b; one word group per scorer below), and the label words a
    # given pos_label that is not one of the fit's labels adds.
    var options = List[MselScorerOptions]()
    var sels = List[Int32]()
    var extra_words = List[String]()
    var responses = List[Int]()
    var needs = MEMBER_CAN_FIT | MEMBER_CAN_PREDICT
    if classifier:
        needs |= MEMBER_CAN_CLASSIFIER
    if tool_keep:
        needs |= MEMBER_CAN_EXPORT  # each kept unit state is packed into its estimator
    if tool_respond >= 0:
        # THE CV TOOLS' response: no scorer, the method's capability.
        if tool_respond == MEMBER_OP_PREDICT_PROBA:
            needs |= MEMBER_CAN_PREDICT_PROBA
        elif tool_respond == MEMBER_OP_DECISION:
            needs |= MEMBER_CAN_DECISION
        positives.append(-1)
        responses.append(tool_respond)
    elif len(keys) == 0 and classifier and _mean_accuracy_score(base, b):
        # scoring None = the class's `score` = accuracy (the docstring)
        var accuracy = msel_scorer_code("accuracy", False)
        codes.append(accuracy)
        positives.append(-1)
        responses.append(msel_scorer_row(accuracy).op)
    elif len(keys) == 0:
        needs |= MEMBER_CAN_SCORE
        positives.append(-1)
    var clusterer = False
    var cluster_entry = False
    var others = False
    for j in range(len(keys)):  # driver-loop: over the scorers (once per admission)
        var kw = _ScorerKw()
        var name = _scorer_name(givens[j], context, b, kw)
        if name.byte_length() == 0:
            return False  # a native scorer no table row holds (another metric or kwargs)
        var scorer = msel_get_scorer(givens[j], context)
        var no_method = _response_refusal(scorer, base, b)
        if no_method.byte_length() > 0:
            # A CLASS WITH NONE OF THE SCORER'S RESPONSE METHODS (lane mf-score8
            # round b): scikit-learn's scorer raises `_check_response_method`'s
            # AttributeError on every score, inside the score's try (so a
            # numeric error_score replaces it). The scorer's slot is a stand-in
            # row over `predict` (accuracy, or r2 for a regressor: no metric
            # runs) whose refusal word raises those words per side.
            var stand = msel_scorer_code("accuracy", False) if classifier else msel_scorer_code("r2", False)
            var ropts = MselScorerOptions()
            ropts.refuse_word = _next_word(labels, classifier, extra_words, b)
            extra_words.append(no_method)
            codes.append(stand)
            positives.append(-1)
            options.append(ropts)
            others = True
            var seen_p = False
            for o in range(len(responses)):  # small-loop(the distinct responses, at most three)
                seen_p = seen_p or responses[o] == MEMBER_OP_PREDICT
            if not seen_p:
                responses.append(MEMBER_OP_PREDICT)
            continue
        var method = String(py=msel_scorer_response(scorer, base_plan, context))
        var code = msel_scorer_code(name, method == "decision_function")
        if code < 0:
            return False
        var opts = MselScorerOptions()
        opts.flip = kw.flip
        opts.k = kw.top_k
        opts.beta = kw.beta
        opts.zd = kw.zd
        var row = msel_scorer_row(code)
        # LANE mf-score6: average=None (the words scikit-learn's `_score`
        # names the scorer by, `str(scorer)` or a multi-metric key, as one
        # more label word: a per-label array is no number), average='samples'
        # (refused per side in the door's words), a sample_weight kwarg read
        # ONCE as float32 words (their bits appended to the selection words;
        # the RUN checks them per side in the door's order and words; a
        # refusal the door raises on reading them is one more label word), a
        # sample_weight the metric does not take (refused per side).
        if kw.avg_none:
            opts.none_word = _next_word(labels, classifier, extra_words, b)
            if many:
                extra_words.append(keys[j])
            else:
                extra_words.append(String(py=py_call(b.str, givens[j])))
        opts.samples = kw.samples
        if kw.w_refused or (kw.weights is not None and not msel_scorer_weighted(row)):
            opts.w_n = -1
        elif kw.weights is not None:
            _weight_words(kw.weights, msel_scorer_caller(row), _next_word(labels, classifier, extra_words, b),
                          context, sels, opts, extra_words)
        # LANE mf-score8: the typed kwargs of a metric row (core/
        # msel_scorer_run.mojo module note), read once in `_scorer_name`: the
        # multioutput rule, force_finite, the normalize / adjusted flag,
        # kappa's weights; a given alpha / power / replace_undefined_by with
        # its printed words, and a refused kwarg's door words, each one more
        # label word.
        opts.mo = kw.mo
        if kw.mo == 1 and opts.none_word < 0:
            # 'raw_values' (lane mf-score8 round b): over several outputs no
            # number; the scorer's words name it in scikit-learn's refusal
            opts.none_word = _next_word(labels, classifier, extra_words, b)
            if many:
                extra_words.append(keys[j])
            else:
                extra_words.append(String(py=py_call(b.str, givens[j])))
        opts.ff = kw.ff
        opts.flag = kw.flag
        opts.wmode = kw.wmode
        if kw.has_param:
            opts.param = kw.param
            opts.param_word = _next_word(labels, classifier, extra_words, b)
            extra_words.append(kw.param_text)
        if kw.refuse_text.byte_length() > 0:
            opts.refuse_word = _next_word(labels, classifier, extra_words, b)
            extra_words.append(kw.refuse_text)
        # A CLUSTERER base (lane mf-cluster5): a non-classifier scored by the
        # clustering scorers only (its predict is cluster ids).
        if not classifier and msel_scorer_cluster(row):
            clusterer = True
        elif classifier and msel_scorer_regression(row):
            # A REGRESSION SCORER ON A CLASSIFIER (lane mf-score8): scikit-learn
            # scores the predicted labels as numbers; each fit label's value
            # read once (`_label_values`), each side maps its codes onto them.
            if labels is None:
                return False  # a classifier admitted with no fit labels known: nothing to map (guard)
            _label_values(labels, sels, opts, b)
            others = True
        elif not classifier and msel_scorer_float_labels(row):
            # A LABEL SCORER ON A NON-CLASSIFIER (lane mf-score8 round b): each
            # side checks its targets and predictions as `_check_targets` does
            # (continuous refuses in its words, integral values score as labels)
            others = True
        elif msel_scorer_regression(row) == classifier:
            # the binary label rows on a non-classifier: their pos_label=1 among
            # float labels and the refusal words that print them (NumPy's
            # float reprs) have no typed words yet
            return False
        else:
            others = True
        if msel_scorer_cluster_entry(row):
            cluster_entry = True
        var positive = -1
        var positive_kind = msel_scorer_positive_kind(row)
        if positive_kind == SCORER_POSITIVE_LABEL_ONE:
            # the binary label scorers' pos_label=1 and 'average_precision''s
            # (its default; a binary member's column, lane mf-score5)
            if labels is None:
                return False
            positive = _positive_code(labels, b)
            if kw.given_pos:
                # A MAKE_SCORER's pos_label (lanes mf-msel5c, mf-tail5b): its
                # class code, the refusal words naming it (`pos_word`). Mixed
                # labels: the metric doors' refusal first, as for pos_label=1;
                # one that is not among the labels: as pos_label=1 absent, its
                # own word appended to the label words.
                var n_labels = len(py_call(b.list, labels))
                if msel_scorer_ap(row) and n_labels > 2:
                    # lane mf-score8 round b: per side, as the AP door decides
                    # it from the side's classes: more than two refuse a
                    # pos_label other than 1 ("fixed to 1"); two or fewer
                    # take it as the binary curve's (`wmode` 1 marks it)
                    if not Bool(py=py_call(b.bool, py_call(native_loaded_module("operator").eq, kw.pos_label,
                                                           PythonObject(1)))):
                        opts.wmode = 1
                opts.pos_given = True
                positive = _label_code(labels, kw.pos_label, b)
                # A FLOAT pos_label (lane mf-score6). The binary label rows'
                # door reads it through `_classification_labels`, which
                # refuses a floating label: each side raises those words
                # after its multiclass check (`opts.pos_float`,
                # core/msel_scorer_run.mojo SCORER_FLOAT_POS_WORDS).
                # average_precision's door compares it by value (`_ap_binary`:
                # `pos_label in present`): one equal to an integer label is
                # that label's code, any other is absent. Its word is
                # `str(pos_label)`, as the door's f-string prints it.
                var real = _real_label(kw.pos_label, b)
                opts.pos_float = real and not msel_scorer_ap(row)
                if real and positive >= 0 and not _labels_mixed(labels, b):
                    opts.pos_word = n_labels + len(extra_words)
                    extra_words.append(String(py=py_call(b.str, kw.pos_label)))
                if _labels_mixed(labels, b):
                    positive = SCORER_POSITIVE_MIXED if msel_scorer_positive(row) else -1
                    if positive == -1:
                        return False
                elif positive < 0:
                    var kind = 1 if real else _label_kind(kw.pos_label, b)
                    if kind == 0:
                        return False  # a pos_label of no kind the metrics read: the door's words
                    var fit_kind = _absent_positive(labels, b)
                    if fit_kind == -1:
                        return False
                    var same_kind = (kind == 1) == (fit_kind == SCORER_POSITIVE_ABSENT)
                    if not msel_scorer_positive(row) and not msel_scorer_ap(row):
                        return False  # guard: only the binary label rows and AP read a pos_label here
                    positive = SCORER_POSITIVE_ABSENT if (same_kind or msel_scorer_ap(row)) else SCORER_POSITIVE_KIND
                    opts.pos_word = n_labels + len(extra_words)
                    if real:
                        extra_words.append(String(py=py_call(b.str, kw.pos_label)))
                    else:
                        var v = py_call(b.int, kw.pos_label) if kind == 1 else py_call(b.str, kw.pos_label)
                        extra_words.append(String(py=py_call(b.repr, v)))
            elif _labels_mixed(labels, b):
                # MIXED-KIND LABELS (lane mf-msel5b): strings and integers; the
                # RUN raises the metric doors' refusal per side.
                positive = SCORER_POSITIVE_MIXED
            elif positive < 0 and msel_scorer_positive(row):
                # pos_label=1 is not one of the labels (lane mf-cluster5): the
                # RUN raises the door's words per side (or scores it absent)
                positive = _absent_positive(labels, b)
                if positive == -1:
                    return False  # labels of no kind the door's metrics read
            elif positive < 0 and len(py_call(b.list, labels)) == 2:
                # 'average_precision' of two labels without 1 (lane mf-msel5):
                # scikit-learn's scorer refuses pos_label=1 in
                # `_get_response_values` before any metric; the RUN raises its
                # words per side (msel_search_run `_scorer_side`, the labels'
                # reprs). Labels of mixed or other kinds: the door.
                positive = _absent_positive(labels, b)
                if positive == -1:
                    return False
        elif positive_kind == SCORER_POSITIVE_MARKS:
            # 'neg_brier_score': `_pos_label(None, ...)`'s label marks (lane mf-score5)
            if labels is None:
                return False
            positive = _brier_marks(labels, context, b)
            if kw.given_pos:
                # a given pos_label (lane mf-tail5b): its class code
                # (`mx_pos_label`: y_true == it). Not one of the labels: the
                # door (`_get_response_values`' words for a binary member).
                positive = _label_code(labels, kw.pos_label, b)
                if positive < 0 or _labels_mixed(labels, b):
                    return False
                opts.pos_given = True
                # a FLOAT pos_label equal to a label (lane mf-score6): its
                # column is picked as `_get_response_values` picks it, then
                # each side raises `mx_pos_label`'s refusal of a floating label
                opts.pos_float = _real_label(kw.pos_label, b)
        if kw.labels is not None:
            # AN AVERAGED ROW's `labels` (lane mf-tail5b): each label's class
            # code; a label of the targets' kind that no fit label equals is
            # one more code no row holds (`_selected_labels` keeps it). Mixed
            # kinds or a repeated label: the door's words.
            if labels is None or _labels_mixed(labels, b):
                return False
            var fit_kind = _absent_positive(labels, b)
            var n_labels = len(py_call(b.list, labels))
            var listed = py_call(b.list, kw.labels)
            opts.sel_from = len(sels)
            opts.sel_n = len(listed)
            var absent = 0
            var refused = String("")
            for q in range(len(listed)):  # small-loop(the scorer's labels, once per admission)
                var c = _label_code(labels, listed[q], b)
                if c < 0:
                    var kind = _label_kind(listed[q], b)
                    if kind == 0 or fit_kind == -1:
                        return False  # a label of no kind the doors' labels read (`mx_list_labels`' own words)
                    if (kind == 1) != (fit_kind == SCORER_POSITIVE_ABSENT):
                        # lane mf-score8: `mx_selected`'s words, per side
                        refused = String("TypeError: labels must use the same type as the targets")
                        break
                    c = n_labels + absent
                    absent += 1
                var repeated = False
                for r in range(opts.sel_from, len(sels)):  # small-loop(the labels so far)
                    repeated = repeated or Int(sels[r]) == c
                if repeated:
                    refused = String("labels must be unique")  # lane mf-score8: the door's words, per side
                    break
                sels.append(Int32(c))
            if refused.byte_length() > 0:
                # the selection is dropped; the refusal is raised per side
                # before the metric reads any label (the doors' order)
                for _ in range(len(sels) - opts.sel_from):  # small-loop(the labels read so far)
                    _ = sels.pop()
                opts.sel_n = 0
                if opts.refuse_word < 0:
                    opts.refuse_word = _next_word(labels, classifier, extra_words, b)
                    extra_words.append(refused)
            elif opts.sel_n == 0:
                return False
        codes.append(code)
        positives.append(positive)
        options.append(opts)
        var seen = False
        for o in range(len(responses)):  # small-loop(the distinct responses, at most three)
            seen = seen or responses[o] == row.op
        if not seen:
            responses.append(row.op)
        if row.op == MEMBER_OP_PREDICT_PROBA:
            needs |= MEMBER_CAN_PREDICT_PROBA
        elif row.op == MEMBER_OP_DECISION:
            needs |= MEMBER_CAN_DECISION
    # lane mf-score8 round b: a clusterer scored by clustering scorers and
    # others together: its ids are read as numbers or labels per side (the
    # RUN's `MselScorer.ids`).
    var tool_ids = False
    # A multi-output classifier's predicted labels (lane mf-msel5c): 0 none,
    # 1 integers (an int64 block), 2 reals (a float64 block).
    var multi_labels = 0
    var multi_proba = False
    var proba_addresses = PythonObject(None)
    var nested = _Nested()
    for c in range(len(clones)):  # driver-loop: over the candidates (admission, once each)
        if not _admit_nested(done, nested, clones[c], classifier, needs, context, b, labels):
            return False
    for c in range(len(nested.plans)):  # driver-loop: over the candidates (their responses' element types)
        var multi = (nested.masks[c] & MEMBER_CAN_MULTI_OUTPUT) != 0
        if multi and len(codes) > 0 and not classifier and not msel_spec_stratified(split.spec.kind):
            # lane mf-score8 round b: a multi-output regressor scored by
            # regression rows (each side's rows x outputs)
            for q in range(len(codes)):  # small-loop(the scorers)
                if not msel_scorer_regression(msel_scorer_row(codes[q])):
                    return False  # a label scorer on multi-output targets: no typed rows (see rows.tsv)
        elif multi and (len(codes) > 0 or msel_spec_stratified(split.spec.kind)):
            return False  # a multi-output target is scored by the member itself, folded by KFold
        if multi and tool_respond >= 0 and tool_respond != MEMBER_OP_PREDICT and not (
                classifier and tool_respond == MEMBER_OP_PREDICT_PROBA and not tool_log
                and (nested.masks[c] & MEMBER_CAN_PREDICT_PROBA) != 0):
            # decision_function / predict_log_proba: MultiOutputClassifier has
            # neither (scikit-learn's AttributeError, the door's words: lane
            # mf-ens6b raises them by the units' own attribute read, before
            # any fit; a class that has it keeps the door)
            var asked = String("predict_log_proba") if tool_log else String(py=tool[1])
            if py_call(b.getattr, clones[c], PythonObject(asked), PythonObject(None)) is None:
                _ = py_call(b.getattr, clones[c], PythonObject(asked))
            return False
        if multi and tool_respond == MEMBER_OP_PREDICT_PROBA:
            multi_proba = True  # (lane mf-tail5) one result block per output
        if multi and tool_respond >= 0 and classifier:
            # A MULTI-OUTPUT CLASSIFIER's predict (lane mf-msel5c): each output's
            # codes decoded at PACK into one rows x outputs block of its labels,
            # int64 for integer labels, float64 for real ones; labels of
            # another kind keep the door (no block holds them).
            multi_labels = _numeric_outputs(labels, b)
            if multi_labels == 0:
                return False
        for o in range(len(responses)):  # small-loop(the distinct responses)
            var response = responses[o]
            var probe = nested.plans[c].copy()
            probe.classes = 2 if classifier else 0
            var labels_out = response == MEMBER_OP_PREDICT and (classifier or clusterer)
            var dtype = member_output_dtype(probe, response, 1, 1, MEMBER_I32 if labels_out else MEMBER_F32)
            # cross_val_predict of a non-classifier's predict (lane mf-msel5b):
            # float values, or a clusterer's integer ids
            # integer ids of a non-classifier's predict (a clusterer): a cv
            # tool's answer, or (lane mf-score8 round b) the side words a label
            # or regression scorer reads as numbers
            var ids = (not labels_out and response == MEMBER_OP_PREDICT and not classifier
                       and (dtype == MEMBER_I32 or dtype == MEMBER_I64))
            if ids and tool_respond == MEMBER_OP_PREDICT:
                tool_ids = True
            if labels_out and dtype != MEMBER_I32 and dtype != MEMBER_I64:
                return False  # a member whose predict is not class codes
            if not labels_out and not ids and dtype != MEMBER_F32 and dtype != MEMBER_F64:
                return False  # a response the scorer's metric does not read
    # The settings words (core/ensemble_member.mojo META_SEARCH_*): the head,
    # then one scorer word group per scorer (lane mf-tail5b; its code, its
    # positive word, its options and labels codes:
    # core/msel_scorer_run.mojo `msel_scorer_group_words`).
    var one = codes[0] if len(codes) == 1 and not many else 0
    var count = len(codes) if many else 0
    var words: List[Float64] = [Float64(one), tolerate, error_value, 1.0 if return_train else 0.0, Float64(seed),
                                0.0 if refit_false else 1.0, Float64(refit_scorer), Float64(count)]
    for j in range(len(positives)):  # small-loop(one word group per scorer)
        var code_j = codes[j] if j < len(codes) else one
        var opts_j = options[j] if j < len(options) else MselScorerOptions()
        msel_scorer_group_words(words, code_j, positives[j], opts_j, sels)
    if tool is not None:
        # THE CV TOOLS' tail (core/ensemble_member.mojo META_SEARCH_TOOL_WORDS)
        # and, with a response, THE RESULT BUFFER (lane mf-msel5b): the
        # cross_val_predict result itself, (rows,) or (rows, columns), int64
        # for class codes (decoded at PACK) and integer ids, float64 values
        # otherwise; columns: one per output of a predict, one per class of
        # predict_proba, a decision_function's 1 (two classes) or one per class.
        var dst = 0
        var width = 1
        var dst_dtype = MEMBER_F64
        if tool_respond >= 0 and multi_proba:
            # THE PER-OUTPUT BUFFERS (lane mf-tail5; core/msel_search_run.mojo
            # `_place_multi_proba`): scikit-learn's cross_val_predict of a
            # multi-output classifier's predict_proba, a list of n_outputs
            # float64 (n, n_classes_j) arrays, allocated here; the RUN reads
            # their addresses (an int64 vector, pinned) and writes each
            # output's block at the units' rows.
            var n_rows = Int(py=tool[6])
            var outputs = Int(py=tool[5])
            if labels is None or py_type(labels) is not b.list or len(labels) != outputs or outputs < 2:
                return False
            var blocks = Python.list()
            var addresses = family_array(Python.tuple(outputs), "<i8", context)
            var words_at = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=msel_addr(addresses))
            for j in range(outputs):  # driver-loop: over the outputs (one result block each)
                var k = len(py_call(b.list, labels[j]))
                var block = family_array(Python.tuple(n_rows, k), "<f8", context)
                _ = py_call(blocks.append, block)
                words_at[j] = Int64(msel_addr(block) if n_rows > 0 else 0)
            _ = addresses  # written through its address above
            py_setitem(tool[4], PythonObject(0), blocks)
            proba_addresses = addresses  # pinned with the plan below
            dst = msel_addr(addresses)
            width = -outputs
            dst_dtype = MEMBER_F64
        elif tool_respond >= 0:
            var n_rows = Int(py=tool[6])
            var outputs = Int(py=tool[5])
            var two_d = False
            if tool_respond == MEMBER_OP_PREDICT:
                width = max(outputs, 1)
                two_d = outputs > 0
            else:
                if not classifier or labels is None:
                    return False
                var k = len(py_call(b.list, labels))
                width = k if (tool_respond == MEMBER_OP_PREDICT_PROBA or k > 2) else 1
                two_d = width > 1 or tool_respond == MEMBER_OP_PREDICT_PROBA
            var integral = tool_respond == MEMBER_OP_PREDICT and (classifier or tool_ids) and multi_labels != 2
            dst_dtype = MEMBER_I64 if integral else MEMBER_F64
            var shape = Python.tuple(n_rows, width) if two_d else Python.tuple(n_rows)
            var result = family_array(shape, "<i8" if integral else "<f8", context)
            py_setitem(tool[4], PythonObject(0), result)
            dst = msel_addr(result) if n_rows > 0 else 0
        words.append(1.0 if tool_keep else 0.0)
        words.append(Float64(tool_respond))
        words.append(1.0 if tool_indices else 0.0)
        words.append(Float64(dst))
        words.append(Float64(width))
        words.append(Float64(dst_dtype))
        words.append(1.0 if tool_log else 0.0)
    var kind = META_SEARCH_CLASSIFIER if classifier else META_SEARCH_REGRESSOR
    # The META entry and its glue run on an x_trees image: the search's own
    # context is model selection's, so the image is the one the meta classes'
    # context names (`_finish` reads its bindings).
    var trees = py_call(native_loaded_module("sys").modules.get, PythonObject("mojolearn._expansion_trees"))
    if trees is None:
        # not imported yet (lane mf-ens6b): imported once, as the metrics
        # module is below. AN IMAGE THAT IS NOT BUILT (lane mf-fitadm7): its
        # loader's ImportError is raised as it stands (no image can run the
        # search kind; before, the door's units fitted it).
        trees = py_call(native_loaded_module("importlib").import_module, PythonObject("mojolearn._expansion_trees"))
    var trees_context = py_get(py_call(b.vars, trees), "_TREES_CONTEXT")
    if trees_context is None:
        # the x_trees image is not connected (lane mf-fitadm7): `_trees_key`'s
        # words, the refusal every meta estimator's own fit raises
        raise Error("ImportError: mojolearn: _mojolearn_x_trees is not connected to the tree owner")
    # THE CLUSTER LABEL-PAIR ENTRY (lane mf-cluster5): the default-mode
    # metrics image the public doors resolve (`estimator_binding_mode`), its
    # key read and its entry resolved ONCE; the image is pinned below. Not
    # loaded or not published: the door's units.
    var cluster = entry_ref_unresolved()
    var metrics_binding = PythonObject(None)
    if cluster_entry:
        # A MISSING METRICS IMAGE (lane mf-msel5b): a native refusal in the
        # loader's words, before any fit (the door's scorers read the same
        # image, so its units could not score either).
        metrics_binding = estimator_binding_mode(PythonObject(None), "_mojolearn_metrics", context)
        var key = py_call(b.getattr, metrics_binding, PythonObject("metrics_cluster_key"), PythonObject(None))
        if key is None:
            raise Error("ImportError: mojolearn: the metrics image publishes no cluster label-pair entry "
                        + "(metrics_cluster_key); rebuild the metrics binding")
        cluster = cluster_ops_entry(Int(py=key[0]), Int(py=key[1]))
    var label_words = _label_words(labels, b) if classifier and labels is not None else List[String]()
    for j in range(len(extra_words)):  # small-loop(a given pos_label's word, after the fit's labels)
        label_words.append(extra_words[j])
    # `flag`: the fold rows may stay resident (the docstring's `device`).
    if not _finish(fields, trees_context, b, owner, nested, kind, 1.0 if resident else 0.0, folds, List[Float64](),
                   List[Int](), words^, fold_train^, fold_test^):
        return False
    _finish_split(owner, split)
    if proba_addresses is not None:
        _ = py_call(owner.pins.append, proba_addresses)  # the per-output result addresses, read by the RUN
    if cluster_entry or len(label_words) > 0:
        meta_plan_search_labels(Int(owner.plan.params[1]), cluster, label_words^)
        if metrics_binding is not None:
            _ = py_call(owner.pins.append, metrics_binding)
    # A classifier scored by its members (scoring None) needs the route too
    # (lane mf-kmeans5): a fold that lacks a class is scored as
    # `ClassifierMixin.score` scores it, by the accuracy program
    # (core/msel_search_run.mojo); resolved once per admission.
    if len(codes) > 0 or (classifier and len(keys) == 0 and tool_respond < 0):
        # THE NATIVE METRIC RUN's route (bindings/arena_prog_run.mojo
        # `metric_inputs_run`): the default-mode x_metrics runner, its
        # connection pinned with the plan entry.
        var nk = metrics_keys()
        var prog_context = py_get(context, nk[].k_family_metrics_prog_context)
        if prog_context is None:
            # A MISSING METRICS IMAGE (lane mf-msel5b): the metrics module is
            # imported once (its import connects the runner, as the door's own
            # scorer calls do), else the refusal in the loader's words.
            _ = py_call(native_loaded_module("importlib").import_module, PythonObject("mojolearn._expansion_metrics"))
            prog_context = py_get(context, nk[].k_family_metrics_prog_context)
            if prog_context is None:
                raise Error("ImportError: mojolearn: the metrics runner is not connected to model selection "
                            + "(family_metrics_prog_context); import mojolearn.metrics first")
        var pin = metric_route_of(metric_route_entries(PythonObject(None), prog_context))
        meta_plan_metric(Int(owner.plan.params[1]), pin.route)
        _ = py_call(owner.pins.append, pin.provider)
        _ = py_call(owner.pins.append, pin.entries)
    return True


def _mean_accuracy_score(base: PythonObject, b: PythonObject) raises -> Bool:
    """UNPACK (lane mf-search5): the base's class takes its `score` from the
    prep classifiers (mojolearn._expansion_prep `_Classifier.score`, the mean
    accuracy of its predicted labels). One attribute read and one identity
    test per admission; False when that module is not loaded."""
    var prep = py_call(native_loaded_module("sys").modules.get, PythonObject("mojolearn._expansion_prep"))
    if prep is None:
        return False
    var classifiers = py_get(py_call(b.vars, prep), "_Classifier")
    if classifiers is None:
        return False
    var own = py_get(py_call(b.vars, classifiers), "score")
    if own is None:
        return False
    return py_call(b.getattr, py_type(base), PythonObject("score"), PythonObject(None)) is own


def _positive_code(labels: PythonObject, b: PythonObject) raises -> Int:
    """UNPACK: the class code of pos_label=1 among the fit's labels (the
    sorted labels the codes index; equality as scikit-learn's
    `_check_set_wise_labels` compares). -1: no labels known, or 1 is not
    one of them."""
    if labels is None:
        return -1
    var items = py_call(b.list, labels)
    var eq = native_loaded_module("operator").eq
    for i in range(len(items)):  # driver-loop: over the fit's labels (once per admission)
        if Bool(py=py_call(b.bool, py_call(eq, items[i], PythonObject(1)))):
            return i
    return -1


def _label_kind(item: PythonObject, b: PythonObject) raises -> Int:
    """1: an integer label (a Python int or a NumPy integer, not a bool);
    2: a string; 0: any other."""
    if py_type(item) is b.bool:
        return 0
    if Bool(py=py_call(b.isinstance, item, b.str)):
        return 2
    if py_type(item) is b.int:
        return 1
    var numbers = native_loaded_module("numbers")
    if Bool(py=py_call(b.isinstance, item, numbers.Integral)):
        return 1
    return 0


def _real_label(item: PythonObject, b: PythonObject) raises -> Bool:
    """UNPACK (lane mf-score6): a pos_label that is a real number of no
    integer type (a Python or NumPy float; not a bool, an int or a string)."""
    if py_type(item) is b.bool or _label_kind(item, b) != 0:
        return False
    return Bool(py=py_call(b.isinstance, item, native_loaded_module("numbers").Real))


def _next_word(labels: PythonObject, classifier: Bool, extra_words: List[String], b: PythonObject) raises -> Int:
    """UNPACK (lane mf-score6): the index the next extra label word takes
    (after a classifier's fit labels' words, then the extra words so far:
    the order `_admit_search` builds `label_words` in)."""
    var n_labels = len(py_call(b.list, labels)) if classifier and labels is not None else 0
    return n_labels + len(extra_words)


def _weight_words(source: PythonObject, caller: String, word: Int, context: PythonObject, mut sels: List[Int32],
                  mut opts: MselScorerOptions, mut extra_words: List[String]) raises:
    """UNPACK (lane mf-score6): a make_scorer's sample_weight read ONCE as
    float32 words (the door's admission, `classical_view` then
    `admit_family`, as bindings/metrics_reg_door.mojo `_reg_weights` reads
    it), each word's bits appended to `sels` at `opts.w_from`. Their checks
    (length, finite, nonnegative, positive) run per side in the RUN, in the
    door's order. A refusal `_reg_weights` raises before those checks (an
    unreadable sequence, not 1-D) is kept as label word `word` in its words
    (`opts.w_word`); an empty array is SCORER_WEIGHTS_EMPTY (each side's
    length refusal)."""
    var view: PythonObject
    try:
        view = classical_view(source, context, "sample_weight")
    except error:
        var text = String(error)
        if not text.startswith("TypeError: "):
            text = "mojolearn " + caller + ": sample_weight: " + text
        opts.w_word = word
        extra_words.append(text)
        return
    if Int(py=view.ndim) != 1:
        opts.w_word = word
        extra_words.append("mojolearn " + caller + ": sample_weight must be 1-D, got shape " + family_shape_repr(view))
        return
    var count = Int(py=view.shape[0])
    if count < 1:
        opts.w_word = SCORER_WEIGHTS_EMPTY
        return
    var w = admit_family(view, -1, "sample_weight", context, admitted=True, empty=True).array
    var words = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(w)))
    opts.w_from = len(sels)
    opts.w_n = count
    for i in range(count):  # driver-loop: over the weights (their float32 bits, once per admission)
        sels.append(bitcast[DType.int32](words[i]))
    _ = w  # read through its address above
    _ = view^


def _response_refusal(scorer: PythonObject, base: PythonObject, b: PythonObject) raises -> String:
    """UNPACK (lane mf-score8 round b): scikit-learn's `_check_response_method`
    (sklearn 1.9 utils/validation.py): an estimator class with none of the
    scorer's response methods fails every score in these words (an
    AttributeError, raised inside each score's try). "": the class has one.
    One attribute read per method, once per admission."""
    var methods = py_call(b.getattr, scorer, PythonObject("_response_method"), PythonObject("predict"))
    var listed = Python.list()
    if py_type(methods) is b.str:
        _ = py_call(listed.append, methods)
    else:
        listed = py_call(b.list, methods)
    for i in range(len(listed)):  # small-loop(the scorer's response methods)
        if py_call(b.getattr, base, listed[i], PythonObject(None)) is not None:
            return String("")
    var shown = String("")
    for i in range(len(listed)):  # small-loop(the scorer's response methods: their names)
        shown += (", " if i > 0 else "") + String(py=py_call(b.str, listed[i]))
    return ("AttributeError: " + String(py=py_type(base).__name__) + " has none of the following attributes: "
            + shown + ".")


def _label_values(labels: PythonObject, mut sels: List[Int32], mut opts: MselScorerOptions, b: PythonObject) raises:
    """UNPACK (lane mf-score8): a regression scorer on a classifier. Each fit
    label (by class code) read ONCE as the number scikit-learn's regression
    metric converts it to (`check_array` / `np.asarray(..., dtype=float)`:
    an integer, a bool or a real as its value; a string as `float(label)`
    when that converts), as float32 words: two per label at `opts.lv_from`,
    the value's bits and its kind (core/msel_scorer_run.mojo SCORER_LABEL_*;
    an unconverted string's side refusal is NumPy's words over its repr).
    Read once per admission; nothing per side."""
    var items = py_call(b.list, labels)
    opts.lv_from = len(sels)
    opts.lv_n = len(items)
    for i in range(len(items)):  # driver-loop: over the fit's labels (their values, once per admission)
        var item = items[i]
        var value = Float32(0.0)
        var kind = SCORER_LABEL_NUMBER
        var text = Bool(py=py_call(b.isinstance, item, b.str))
        try:
            value = Float32(Float64(py=py_call(b.float, item)))
            if text:
                kind = SCORER_LABEL_TEXT
        except:
            kind = SCORER_LABEL_UNREAD
        sels.append(bitcast[DType.int32](value))
        sels.append(Int32(kind))


def _numeric_outputs(labels: PythonObject, b: PythonObject) raises -> Int:
    """UNPACK (lane mf-msel5c): the kind of a multi-output classifier's
    per-output labels (`labels`: one label list per output): 1 every label
    an integer (not a bool), 2 every label a real number with at least one
    not an integer, 3 (lane mf-tail5b) every label a string, 0 otherwise or
    unknown. Read once per admission."""
    if labels is None or py_type(labels) is not b.list:
        return 0
    var numbers = native_loaded_module("numbers")
    var kind = 1
    var strings = 0
    var total = 0
    for j in range(len(labels)):  # driver-loop: over the outputs (their labels, once per admission)
        var items = py_call(b.list, labels[j])
        for i in range(len(items)):  # driver-loop: over one output's labels
            var item = items[i]
            total += 1
            if Bool(py=py_call(b.isinstance, item, b.str)):
                strings += 1
                continue
            if strings > 0:
                return 0  # strings and other labels
            if py_type(item) is b.bool:
                return 0
            if _label_kind(item, b) == 1:
                continue
            if not Bool(py=py_call(b.isinstance, item, numbers.Real)):
                return 0
            kind = 2
    if strings > 0:
        return 3 if strings == total else 0
    return kind


def _decode_outputs(result: PythonObject, labels: PythonObject, b: PythonObject) raises:
    """PACK (lane mf-msel5c): a multi-output classifier's predicted codes
    (the rows x outputs int64 or float64 block the RUN wrote) replaced IN
    PLACE by each output's labels (code c of output j is `labels[j][c]`),
    as the estimator's own predict decodes each output. The labels are read
    once per class, the rows natively."""
    var rows = Int(py=result.shape[0])
    var outputs = Int(py=result.shape[1])
    var wide = String(py=result.dtype) == "<i8"
    var ints = List[List[Int64]]()
    var reals = List[List[Float64]]()
    for j in range(outputs):  # driver-loop: over the outputs (their labels, once per class)
        var items = py_call(b.list, labels[j])
        var iv = List[Int64](capacity=len(items))
        var fv = List[Float64](capacity=len(items))
        for c in range(len(items)):  # small-loop(one output's classes)
            if wide:
                iv.append(Int64(Int(py=py_call(b.int, items[c]))))
            else:
                fv.append(Float64(py=py_call(b.float, items[c])))
        ints.append(iv^)
        reals.append(fv^)
    if rows == 0:
        return
    var at = msel_addr(result)
    var iw = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=at)
    var fw = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=at)
    for i in range(rows):  # driver-loop: over the rows (each output's code to its label, in place)
        for j in range(outputs):  # small-loop(the outputs)
            if wide:
                var c = Int(iw[i * outputs + j])
                if c < 0 or c >= len(ints[j]):
                    raise Error("RuntimeError: mojolearn: a multi-output prediction code outside its output's classes")
                iw[i * outputs + j] = ints[j][c]
            else:
                var c = Int(fw[i * outputs + j])
                if c < 0 or c >= len(reals[j]):
                    raise Error("RuntimeError: mojolearn: a multi-output prediction code outside its output's classes")
                fw[i * outputs + j] = reals[j][c]
    _ = result  # written through its address above


def _decode_text_outputs(result: PythonObject, labels: PythonObject, context: PythonObject,
                         b: PythonObject) raises -> PythonObject:
    """PACK (lane mf-tail5b): a text multi-output classifier's predicted
    codes (the rows x outputs int64 block the RUN wrote) as the rows the
    door's placement returns: per output its codes copied natively into one
    int64 vector and decoded by `labels_decode_labels` (the 1-D text route's
    decoder, one list), then one list per row of its outputs' labels."""
    var rows = Int(py=result.shape[0])
    var outputs = Int(py=result.shape[1])
    var src = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=msel_addr(result)) if rows > 0 else MutPointer[
        Int64, MutUntrackedOrigin](unsafe_from_address=0)
    var decoded = Python.list()
    for j in range(outputs):  # driver-loop: over the outputs (one native column copy and one decode each)
        var column = family_array(Python.tuple(rows), "<i8", context)
        if rows > 0:
            var dst = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=msel_addr(column))
            for i in range(rows):  # driver-loop: over the rows (output j's codes)
                dst[i] = src[i * outputs + j]
        _ = column  # written through its address above
        _ = py_call(decoded.append, py_call(b.list, labels_decode_labels(labels[j], column, context)))
    _ = result  # read through its address above
    var out = Python.list()
    for i in range(rows):  # driver-loop: over the rows (each row's labels, the returned objects)
        var row = Python.list()
        for j in range(outputs):  # small-loop(the outputs)
            _ = py_call(row.append, decoded[j][i])
        _ = py_call(out.append, row)
    return out


def _label_code(labels: PythonObject, value: PythonObject, b: PythonObject) raises -> Int:
    """UNPACK (lane mf-msel5c): the class code of the label `value` among
    the fit's labels (equality as scikit-learn compares a pos_label), -1:
    not one of them. Read once per admission."""
    var items = py_call(b.list, labels)
    var eq = native_loaded_module("operator").eq
    for i in range(len(items)):  # driver-loop: over the fit's labels (once per admission)
        if Bool(py=py_call(b.bool, py_call(eq, items[i], value))):
            return i
    return -1


def _absent_positive(labels: PythonObject, b: PythonObject) raises -> Int:
    """UNPACK (lane mf-cluster5): the positive word of a binary label scorer
    whose pos_label=1 is not one of the fit's labels: SCORER_POSITIVE_ABSENT
    (integer labels), SCORER_POSITIVE_KIND (string labels: the door's "same
    type" refusal), -1 for labels of mixed or other kinds (the door). Read
    once per admission."""
    var items = py_call(b.list, labels)
    var kind = -1
    for i in range(len(items)):  # driver-loop: over the fit's labels (once per admission)
        var k = _label_kind(items[i], b)
        if k == 0 or (kind >= 0 and k != kind):
            return -1
        kind = k
    if kind == 1:
        return SCORER_POSITIVE_ABSENT
    if kind == 2:
        return SCORER_POSITIVE_KIND
    return -1


def _labels_mixed(labels: PythonObject, b: PythonObject) raises -> Bool:
    """UNPACK (lane mf-msel5b): the fit's labels hold strings AND integers
    (and nothing else: another kind keeps the door). Read once per
    admission."""
    var items = py_call(b.list, labels)
    var ints = False
    var strs = False
    for i in range(len(items)):  # driver-loop: over the fit's labels (once per admission)
        var k = _label_kind(items[i], b)
        if k == 0:
            return False
        ints = ints or k == 1
        strs = strs or k == 2
    return ints and strs


def _label_words(labels: PythonObject, b: PythonObject) raises -> List[String]:
    """UNPACK (lane mf-cluster5): each fit label's repr as `str(present)`
    prints it in the door's words (an int's `int(v)`, a string's `str(v)`),
    by class code. Read once per admission."""
    var items = py_call(b.list, labels)
    var out = List[String](capacity=len(items))
    for i in range(len(items)):  # driver-loop: over the fit's labels (once per admission)
        var k = _label_kind(items[i], b)
        var v = items[i]
        if k == 1:
            v = py_call(b.int, items[i])
        elif k == 2:
            v = py_call(b.str, items[i])
        out.append(String(py=py_call(b.repr, v)))
    return out^


def _brier_marks(labels: PythonObject, context: PythonObject, b: PythonObject) raises -> Int:
    """UNPACK (lane mf-score5): the fit's labels' marks for 'neg_brier_score'
    (bindings/metrics_exp_rank_run.mojo `brier_marks_positive`): two bits per
    class code, an exact integer label 0, 1 or -1 marked (`mx_pos_label_ok`
    reads integer labels only); other labels 0. Read once per admission."""
    var items = py_call(b.list, labels)
    var marks = 0
    var eq = native_loaded_module("operator").eq
    for i in range(min(len(items), 26)):  # driver-loop: over the fit's labels (once per admission; 2 bits each, exact in a float64 word)
        if not msel_is_int_arg(items[i], context):
            continue
        var mark = 0
        if Bool(py=py_call(b.bool, py_call(eq, items[i], PythonObject(0)))):
            mark = BRIER_MARK_ZERO
        elif Bool(py=py_call(b.bool, py_call(eq, items[i], PythonObject(1)))):
            mark = BRIER_MARK_ONE
        elif Bool(py=py_call(b.bool, py_call(eq, items[i], PythonObject(-1)))):
            mark = BRIER_MARK_MINUS_ONE
        marks |= mark << (2 * i)
    return marks


def _search_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
                 plan: MemberPlan, state: Int, labels: PythonObject, mut jobs: _PackJobs) raises:
    """PACK of THE SEARCH KIND: the attributes `msel_search_fit` writes
    (`cv_results_` with `params`, `param_<name>`, and per scorer
    `split<i>_test_score`, `mean_test_score`, `std_test_score`,
    `rank_test_score` and `mean_train_score`, suffixed `_<name>` for a
    multi-metric scoring (lane mf-search4b); `n_splits_`, `multimetric_`;
    `best_index_`, `best_score_`, `best_params_` unless a multi-metric
    search has refit=False; `best_estimator_` (the best candidate's clone
    holding the refit state) and `scorer_` unless refit=False), then the
    units' fit notices with the best estimator's (or the base's) context."""
    var nested = meta_state_plans(state)
    var handles = _export_i64(plan, state, META_BLOCK_NESTED)
    if len(nested) != 1 or len(handles) != 1:
        raise Error("RuntimeError: mojolearn: the search member state does not match its best estimator")
    var view = _export_view(plan, state, META_BLOCK_SEARCH)
    var words = _F64P(unsafe_from_address=view.dst.data)
    var count = view.dst.rows * view.dst.cols
    if count < 8:
        raise Error("RuntimeError: mojolearn: the search member state has no tables")
    var best = Int(words[0])
    var folds = Int(words[1])
    var best_score = words[2]
    var m = Int(words[3])
    var train = words[4] != 0.0
    var seed = Int(words[5])
    var tables = Int(words[6])
    var refitted = words[7] != 0.0
    var candidates = _search_candidates(model, fields, seed, False, context)
    if len(candidates) != m or best < 0 or best >= m:
        raise Error("RuntimeError: mojolearn: the search's candidates changed since its admission")
    var scoring_in = py_get(fields, "scoring")
    var many = scoring_in is not None and _scoring_many(scoring_in, context, b)
    # The scorer names in the admission's order (its dict keeps one of each).
    var names = List[String]()
    if many:
        var given = py_call(b.list, scoring_in)
        for i in range(len(given)):  # driver-loop: over the scorer names (once per pack)
            var name = String(py=given[i])
            var repeated = False
            for j in range(len(names)):  # small-loop(the names so far)
                repeated = repeated or names[j] == name
            if not repeated:
                names.append(name)
    if (len(names) if many else 1) != tables:
        raise Error("RuntimeError: mojolearn: the search's scorers changed since its admission")
    var per = m * (2 * folds + 4) if train else m * (folds + 3)
    var tool = py_get(fields, "_msel_cv_tool")
    if (count != 8 + tables * per and tool is None) or count < 8 + tables * per:
        raise Error("RuntimeError: mojolearn: the search member state's tables are malformed")
    var results = Python.dict()
    py_setitem(results, "params", candidates)
    var itertools = native_loaded_module("itertools")
    var keys = py_call(b.sorted, py_call(b.set, py_call(itertools.chain.from_iterable, candidates)))
    for i in range(len(keys)):  # driver-loop: over the candidates' parameter names
        var getter = py_call(native_loaded_module("operator").methodcaller, PythonObject("get"), keys[i])
        py_setitem(results, PythonObject("param_") + py_call(b.str, keys[i]),
                   py_call(b.list, py_call(b.map, getter, candidates)))
    for j in range(tables):  # driver-loop: over the scorers (one table each, in their order)
        var suffix = String("_") + names[j] if many else String("")
        var at = 8 + j * per
        for f in range(folds):  # driver-loop: over the folds (one column each)
            var column = List[Float64](capacity=m)
            for c in range(m):  # small-loop(one fold's candidates)
                column.append(words[at + c * folds + f])
            py_setitem(results, PythonObject("split" + String(f) + "_test_score" + suffix),
                       msel_f64_list(column, context))
        at += m * folds
        var means = List[Float64](capacity=m)
        var stds = List[Float64](capacity=m)
        var ranks = List[Int](capacity=m)
        for c in range(m):  # driver-loop: over the candidates (their tables)
            means.append(words[at + c])
            stds.append(words[at + m + c])
            ranks.append(Int(words[at + 2 * m + c]))
        at += 3 * m
        py_setitem(results, PythonObject("mean_test_score" + suffix), msel_f64_list(means, context))
        py_setitem(results, PythonObject("std_test_score" + suffix), msel_f64_list(stds, context))
        py_setitem(results, PythonObject("rank_test_score" + suffix), msel_i32_list(ranks, context))
        if train:
            # scikit-learn's split<i>_train_score columns (lane mf-msel5: the
            # validation curve and the cv tools read them)
            for f in range(folds):  # driver-loop: over the folds (one train column each)
                var column = List[Float64](capacity=m)
                for c in range(m):  # small-loop(one fold's candidates)
                    column.append(words[at + c * folds + f])
                py_setitem(results, PythonObject("split" + String(f) + "_train_score" + suffix),
                           msel_f64_list(column, context))
            var train_means = List[Float64](capacity=m)
            for c in range(m):  # driver-loop: over the candidates (their train means)
                train_means.append(words[at + m * folds + c])
            py_setitem(results, PythonObject("mean_train_score" + suffix), msel_f64_list(train_means, context))
    var updates = Python.dict()
    py_setitem(updates, "cv_results_", results)
    py_setitem(updates, "n_splits_", PythonObject(folds))
    py_setitem(updates, "multimetric_", PythonObject(many))
    if refitted or not many:
        py_setitem(updates, "best_index_", PythonObject(best))
        py_setitem(updates, "best_score_", PythonObject(best_score))
        py_setitem(updates, "best_params_", candidates[best])
    var base = py_get(fields, "estimator")
    var notice_context = prep_estimator_plan(base, context)["context"]
    if refitted:
        if handles[0] == 0:
            raise Error("RuntimeError: mojolearn: the search member state holds no refitted best")
        # The best estimator: the best candidate's clone holding the refit state.
        var clone = msel_native_clone(base, context)
        _ = msel_native_set_params(clone, msel_clone_parameter(candidates[best], context), context)
        var clone_plan = prep_estimator_plan(clone, context)
        var clone_fields = py_call(clone_plan["fields_get"], clone, py_type(clone))
        notice_context = clone_plan["context"]
        _pack_into(clone, clone_fields, notice_context, b, nested[0], handles[0], labels, jobs)
        py_setitem(updates, "best_estimator_", clone)
        if many:
            # the door's `scorer_`: the dict of the native scorer records
            var scorers = Python.dict()
            var mapping = not (py_type(scoring_in) is b.list or py_type(scoring_in) is b.tuple
                               or py_type(scoring_in) is b.set)
            for j in range(len(names)):  # small-loop(the scorer names)
                var key = PythonObject(names[j])
                py_setitem(scorers, key, msel_get_scorer(scoring_in[key] if mapping else key, context))
            py_setitem(updates, "scorer_", scorers)
        else:
            py_setitem(updates, "scorer_", msel_get_scorer(scoring_in, context))
    if tool is not None:
        py_setitem(updates, "_msel_cv_result",
                   _search_tool_pack(fields, tool, words, 8 + tables * per, count, m * folds, folds, plan, state,
                                     nested[0], labels, notice_context, context, b, jobs))
    msel_search_update(fields, updates, context)
    var owners = List[Int]()
    var notices = List[List[Float64]]()
    meta_state_notices(state, owners, notices)
    for i in range(len(owners)):  # driver-loop: over the units' notices (a few warnings each)
        # the scorers' metric warnings with the search's own context (lane
        # mf-score5: the context its scorers' metric calls carry)
        member_fit_notices_raise_with(notices[i], notice_context, context)


def _search_tool_pack(fields: PythonObject, tool: PythonObject, words: _F64P, at_in: Int, count: Int, units: Int,
                      folds: Int, plan: MemberPlan, state: Int, unit_plan: MemberPlan, labels: PythonObject,
                      unit_context: PythonObject, context: PythonObject, b: PythonObject,
                      mut jobs: _PackJobs) raises -> PythonObject:
    """PACK of THE CV TOOLS (lane mf-msel5; core/ensemble_member.mojo
    `_search_tool_words`): a dict with `fit_time` and `score_time` (one
    float64 per unit), and as the tool asked `estimator` (each kept unit
    state packed into a clone of the base, as `best_estimator_` is),
    `indices` ({'train': [...], 'test': [...]}, int64 per fold) and
    `predictions` (lane mf-msel5b: the result Array the RUN wrote at every
    row, or a classifier's codes decoded to its labels, with `placed` False:
    the door places those labels), read by the carrier's door
    (bindings/msel_cv_door.mojo)."""
    var at = at_in
    if at >= count or Int(words[at]) != units:
        raise Error("RuntimeError: mojolearn: the search member state's cv tool words are malformed")
    at += 1
    var fit = List[Float64](capacity=units)
    var score = List[Float64](capacity=units)
    for u in range(units):  # driver-loop: over the units (their seconds)
        fit.append(words[at + u])
        score.append(words[at + units + u])
    at += 2 * units
    var out = Python.dict()
    py_setitem(out, "fit_time", msel_f64_list(fit, context))
    py_setitem(out, "score_time", msel_f64_list(score, context))
    if Bool(py=py_call(b.bool, tool[2])):
        var trains = Python.list()
        var tests = Python.list()
        for f in range(folds):  # driver-loop: over the folds (their rows)
            var ntr = Int(words[at])
            var nte = Int(words[at + 1])
            at += 2
            var tr = List[Int](capacity=ntr)
            for i in range(ntr):  # driver-loop: over the fold's train rows
                tr.append(Int(words[at + i]))
            at += ntr
            var te = List[Int](capacity=nte)
            for i in range(nte):  # driver-loop: over the fold's test rows
                te.append(Int(words[at + i]))
            at += nte
            _ = py_call(trains.append, msel_i64_list(tr, context))
            _ = py_call(tests.append, msel_i64_list(te, context))
        var idx = Python.dict()
        py_setitem(idx, "train", trains)
        py_setitem(idx, "test", tests)
        py_setitem(out, "indices", idx)
    if tool[1] is not None:
        # THE RESULT BUFFER (lane mf-msel5b): the RUN wrote every unit's
        # test-side response at its rows; a classifier's predicted codes are
        # decoded to its labels here, once for every row (the estimator's own
        # predict decodes its codes the same way), and placed by the door.
        var result = tool[4][0]
        if String(py=tool[1]) == "predict" and labels is not None and Int(py=tool[5]) > 0:
            if _numeric_outputs(labels, b) == 3:
                # TEXT LABELS (lane mf-tail5b): each output's codes decoded by
                # the one label decoder, the rows built as the door's own
                # placement leaves text multi-output rows (one list per row)
                py_setitem(out, "predictions", _decode_text_outputs(result, labels, unit_context, b))
            else:
                # a multi-output classifier's codes, decoded per output in place (lane mf-msel5c)
                _decode_outputs(result, labels, b)
                py_setitem(out, "predictions", result)
            py_setitem(out, "placed", PythonObject(True))
        elif String(py=tool[1]) == "predict" and labels is not None:
            py_setitem(out, "predictions", labels_decode_labels(labels, result, unit_context))
            py_setitem(out, "placed", PythonObject(False))
        else:
            py_setitem(out, "predictions", result)
            py_setitem(out, "placed", PythonObject(True))
    if Bool(py=py_call(b.bool, tool[0])):
        var handles = _export_i64(plan, state, META_BLOCK_FOLDS)
        if len(handles) != units or units != folds:
            raise Error("RuntimeError: mojolearn: the search member state does not hold its units' estimators")
        var base = py_get(fields, "estimator")
        var estimators = Python.list()
        for u in range(units):  # driver-loop: over the kept unit states (one estimator each, GIL held)
            var clone = msel_native_clone(base, context)
            var clone_plan = prep_estimator_plan(clone, context)
            var clone_fields = py_call(clone_plan["fields_get"], clone, py_type(clone))
            if handles[u] != 0:
                _pack_into(clone, clone_fields, clone_plan["context"], b, unit_plan, handles[u], labels, jobs)
            _ = py_call(estimators.append, clone)
        py_setitem(out, "estimator", estimators)
    if at != count:
        raise Error("RuntimeError: mojolearn: the search member state's cv tool words are malformed")
    return out


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
        # the META_BLOCK_SEARCH head (no tables: an imported search is not refitted here). A SERVING plan
        # (`_admit_search_serving`, META_PARAM_SERVING) holds one candidate, the fitted best: index 0.
        var best_at = 0 if plan.params[META_PARAM_SERVING] != 0.0 else Int(py=index)
        var words: List[Float64] = [Float64(best_at), 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0]
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
    MultiOutputRegressor that may be FITTED here (the door's one-program
    ridge route over every output, a measured winner kept on the door) and
    a base with no serving member. A SERVING admission of one (lane
    mf-door7: a fitted MultiOutputRegressor over Ridge as a member that
    only predicts) is the kind: its import reads every fitted output Ridge
    (`_multi_state`) and its predict is each output's Ridge member, the
    plumbing around the same fitted coefficients."""
    var classifier = kind == META_MULTI_CLASSIFIER
    var base = py_get(fields, "estimator")
    if base is None:
        # NO BASE (lane mf-fitadm7): the door's `_require_member` refusal, in
        # its words (its fit refuses None before any column is fitted).
        prep_estimator_admit(base, String("MultiOutputClassifier") if classifier else String("MultiOutputRegressor"),
                             context)
        return False
    if not classifier and not owner.serving:
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


# ------------------------------------------------------------ THE RFE KIND (lane mf-meta5)


def _admit_rfe(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
               mut owner: MemberOwner, done: _Done, labels: PythonObject) raises -> Bool:
    """UNPACK of THE RFE KIND (core/ensemble_member.mojo): RFE's estimator admitted once as a leaf member
    with the block its importance getter names (providers/python/member_rfe_boundary_door.mojo
    `rfe_base_admit`, the admission RFE's own door runs), n_features_to_select and step as words
    (`rfe_count_words`), the x_prep image of the rounds' programs (`rfe_image`, pinned here). The kind
    (classifier or regressor) is its estimator's.

    A DOTTED importance_getter (lane mf-del5b; 'best_estimator_.coef_', 'estimator_.coef_',
    'final_estimator_.feature_importances_', ...): the path walked once (`rfe_dotted_path`, the walk RFE's
    door runs) to THE FORWARDED EXPORT's hops, handed to every state (`meta_plan_rfe`); the META estimator
    is the one this work-list admitted before the RFE (`_meta_children`, `_admit_nested` with the labels
    RFE's fit hands it: y as given); the kind is the META estimator's (a search's base decides).

    Answers False (RFE's door keeps it) for anything the door's member path does not take: a callable
    importance getter, a dotted path `rfe_dotted_path` does not serve, an estimator with no serving member
    or no importance block, a value the door's reads refuse (the door raises its words). A base kind that
    exports no importance block is admitted with its refusal, which the first round raises (with no round
    the final fit runs, as in scikit-learn)."""
    var estimator = py_get(fields, "estimator")
    var g = py_get(fields, "importance_getter")
    # RFE's door's first reads, in its order and words (bindings/prep_select_native_door.mojo `_rfe_fit`,
    # lane mf-ens6): the estimator's plan (a None or foreign estimator refused), then the getter: a callable
    # is a Python callback (refused natively; the native alternatives are 'auto' or an attribute path).
    prep_estimator_admit(estimator, "RFE", context)
    if g is None or py_type(g) is not b.str:
        if g is not None and Bool(py=py_call(b.callable, g)):
            raise Error("TypeError: mojolearn: RFE importance_getter must be 'auto' or an attribute path; a callable is not supported")
        raise Error("mojolearn: RFE importance_getter must be 'auto', a str or a callable")
    var nested = _Nested()
    var base = RfeBase()
    if _rfe_dotted(String(g)):
        var path = rfe_dotted_path(estimator, g, context, b)
        if not path.served:
            return False
        var at = done.find(estimator)
        if at < 0:
            return False
        var classifier = (done.masks[at] & MEMBER_CAN_CLASSIFIER) != 0
        if not _admit_nested(done, nested, estimator, classifier, MEMBER_CAN_FIT | MEMBER_CAN_EXPORT, context, b,
                             labels):
            return False
        # A tree or forest leaf read for feature_importances_ (lane mf-rfe6): its importance word set in the
        # nested plan the work-list minted (one admission per RFE: the work-list walked this estimator for it).
        if path.leaf_word and not meta_plan_leaf_importances(nested.plans[0], path.hops):
            return False
        base.block = path.forward_block  # lane mf-rfe7b: META_BLOCK_FORWARD and a named attribute's row
        base.squared = True
        base.refusal = path.refusal.copy()
        base.hops = path.hops.copy()
        base.classifier = classifier
        base.meta = True
        base.native = True
    else:
        base = rfe_base_admit(estimator, g, context, b)
        if not base.native:
            return False
        nested.plans.append(base.owner.plan.copy())
        nested.masks.append(base.mask)
        _ = py_call(nested.pins.append, base.owner.pins)
    # The door reads the same values first and raises the same words (lane mf-ens6: no catch).
    var words = rfe_count_words(fields, b)
    var image = rfe_image(py_item(context, prep_keys()[].k_select_context))
    var run = image.run
    var work_on = image.work_on
    var image_entries = image.entries
    words.append(Float64(base.block))
    words.append(1.0 if base.squared else 0.0)
    words.append(1.0 if base.importance_word else 0.0)
    words.append(1.0 if work_on else 0.0)
    words.append(1.0 if (run.dev_store and run.has_ranges) else 0.0)
    var kind = META_RFE_CLASSIFIER if base.classifier else META_RFE_REGRESSOR
    # The META entry and its glue run on an x_trees image: RFE's own context is the prep owner's, so the
    # image is the one the meta classes' context names (as the search kind's admission reads it).
    var trees = py_call(native_loaded_module("sys").modules.get, PythonObject("mojolearn._expansion_trees"))
    var trees_context = PythonObject(None) if trees is None else py_get(py_call(b.vars, trees), "_TREES_CONTEXT")
    if trees_context is None:
        raise Error("ImportError: mojolearn: _mojolearn_x_trees is not connected to the tree owner")
    _ = _finish(fields, trees_context, b, owner, nested, kind, 0.0, 0, List[Float64](), List[Int](), words^,
                List[List[Int32]](), List[List[Int32]]())
    meta_plan_rfe(Int(owner.plan.params[1]), run, base.refusal.copy(), base.hops.copy())
    _ = py_call(owner.pins.append, image_entries)  # the rounds' image, for every fit
    if not base.meta:
        _ = py_call(owner.pins.append, base.template)
    return True


def _rfe_words(plan: MemberPlan, state: Int) raises -> List[Float64]:
    """The fitted RFE state's META_BLOCK_RFE words, copied out of its view."""
    var call = MemberCall(plan.route, state)
    call.block = META_BLOCK_RFE
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var out = List[Float64]()
    var count = call.dst.rows * call.dst.cols
    var words = _F64P(unsafe_from_address=call.dst.data)
    for i in range(count):  # small-loop(an RFE state's words: 2 + 2 x features + kept)
        out.append(words[i])
    return out^


def _rfe_pack(fields: PythonObject, context: PythonObject, b: PythonObject, plan: MemberPlan, state: Int,
              labels: PythonObject, mut jobs: _PackJobs) raises:
    """PACK of THE RFE KIND: the attributes RFE's door writes (providers/python/member_rfe_boundary_door.mojo
    `rfe_publish`: estimator_, the support and its cached gather words, ranking_, n_features_,
    numeric_mode_, n_features_in_), estimator_ a clone of the estimator holding the final state (a
    classifier's with the caller's labels), then the member fits' notices with its context, in fit order.
    A META estimator_ (a dotted importance_getter's, lane mf-del5b) is queued on the pack work-list
    (`_pack_into`, its labels held: RFE fits it on y as given), packed after this RFE."""
    var nested = meta_state_plans(state)
    var handles = _export_i64(plan, state, META_BLOCK_NESTED)
    if len(nested) != 1 or len(handles) != 1:
        raise Error("RuntimeError: mojolearn: an RFE member state holds one final estimator")
    var words = _rfe_words(plan, state)
    var d = Int(words[0])
    var k = Int(words[1])
    var support = List[Float32](length=d, fill=0.0)
    var ranking = List[Float32](length=d, fill=0.0)
    for j in range(d):  # driver-loop: over the fitted support and ranking (feature count)
        support[j] = Float32(words[2 + j])
        ranking[j] = Float32(words[2 + d + j])
    var estimator = py_get(fields, "estimator")
    var clone = prep_estimator_clone(estimator, context)
    var clone_plan = prep_estimator_plan(clone, context)
    var clone_fields = py_call(clone_plan["fields_get"], clone, py_type(clone))
    var clone_context = clone_plan["context"]
    _pack_into(clone, clone_fields, clone_context, b, nested[0], handles[0], labels, jobs)
    rfe_publish(fields, clone, support, ranking, k, context)
    var owners = List[Int]()
    var notices = List[List[Float64]]()
    meta_state_notices(state, owners, notices)
    for i in range(len(owners)):  # driver-loop: over the member fits' notices (a few warnings each)
        member_fit_notices_raise(notices[i], clone_context)


def _rfe_state(fields: PythonObject, context: PythonObject, b: PythonObject, plan: MemberPlan,
               mut imported: _Imported) raises -> Int:
    """UNPACK of a FITTED RFE: a new RFE state holding its `estimator_` (imported through its family) and
    its support, ranking and kept columns (META_BLOCK_RFE)."""
    var final = py_get(fields, "estimator_")
    var support = py_get(fields, "support_")
    var ranking = py_get(fields, "ranking_")
    if final is None or support is None or ranking is None:
        raise Error("NotFittedError: this RFE is not fitted yet")
    var mask = py_call(b.list, support)
    var ranks = py_call(b.list, ranking)
    var d = len(mask)
    if len(ranks) != d:
        raise Error("NotFittedError: this RFE is not fitted yet")
    var kept = List[Int]()
    for j in range(d):  # driver-loop: over the fitted support (feature count)
        if Bool(py=mask[j]):
            kept.append(j)
    var words: List[Float64] = [Float64(d), Float64(len(kept))]
    for j in range(d):  # driver-loop: over the fitted support (feature count)
        words.append(1.0 if Bool(py=mask[j]) else 0.0)
    for j in range(d):  # driver-loop: over the fitted ranking (feature count)
        words.append(Float64(Int(py=ranks[j])))
    for j in range(len(kept)):  # driver-loop: over the kept columns
        words.append(Float64(kept[j]))
    var call = member_call_of(plan)
    _ = member_call(plan.member, MEMBER_OP_NEW, call)
    var nested = List[MemberPlan]()
    var handles = List[Int64]()
    var ok = False
    try:
        handles.append(Int64(_nested_state(final, context, b, plan.classes, imported, nested)))
        var block = MemberCall(plan.route, call.state)
        block.block = META_BLOCK_NESTED
        block.x = MemberMatrix(Int(handles.unsafe_ptr()), 1, 1, MEMBER_I64)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, block)
        handles = List[Int64]()  # the RFE state owns it now
        var fitted = MemberCall(plan.route, call.state)
        fitted.block = META_BLOCK_RFE
        fitted.x = MemberMatrix(Int(words.unsafe_ptr()), 1, len(words), MEMBER_F64)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, fitted)
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
