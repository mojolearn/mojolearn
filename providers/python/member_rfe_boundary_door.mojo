# SPDX-License-Identifier: Apache-2.0
"""UNPACK and PACK of RFE over a native member (lane mf-meta5, 2026-11-23). RUN:
bindings/prep_select_run.mojo `rfe_fit_run`, THE ONE BODY of RFE.fit over a member, run by RFE's own door
(bindings/prep_select_native_door.mojo `_rfe_rounds`) and by THE RFE KIND (core/ensemble_member.mojo
`_fit_rfe`: RFE as a member of another meta-estimator, a search, a calibration or an imputer; mark
(39, 14), admitted by providers/python/member_ensemble_boundary_door.mojo `_admit_rfe`).

What both read once per admission, in the door's words and order:
  * `rfe_base_admit`: the estimator as a LEAF member (no fit target yet): its published class, its kind's
    capability mask, the block the importance getter names in it (`rfe_member_source`) or the refusal a
    first round raises, the tree / forest importance word, the GIL rule word, the one clone PACK writes
    the final fit into. Moved here from the door (lane mf-gbdt5's `_rfe_member`, its first half) so the
    kind admits the same way; the door adds the fit's target and parameters (`_rfe_member`).
  * `rfe_count_words`: n_features_to_select and step as typed words (`rfe_select_count`,
    `rfe_step_count` resolve them per fit over X's columns).
  * `rfe_image`: the x_prep image the rounds' programs run on (the selection the door's programs take:
    the process mode's x_prep entries, every capability by name), its `work` switch, its pin.
  * `rfe_publish`: the PACK tail the door's fit wrote (estimator_, the support mask and its cached gather
    words, ranking_, n_features_, numeric_mode_, n_features_in_), from typed words.

No import of the member dispatch (bindings/member_boundary_door.mojo): this file is imported by the
ensemble boundary, which that dispatch imports. A META estimator is never an RFE base here (it exports
no importance block; the door's own path keeps it).
"""
from std.python import Python, PythonObject
from bindings.member_owner_unpack import MemberOwner
from bindings.member_leaf_boundary_door import member_leaf_admit_serving
from bindings.prep_native_estimator_door import prep_estimator_plan, prep_estimator_clone
from bindings.py_balanced import py_call, py_get, py_item, py_setitem, py_type
from bindings.py_typed_call import py_call_typed
from bindings.classical_validate_unpack import classical_estimator_name
from bindings.classical_arrays_door import classical_typed, classical_address
from bindings.native_loaded_module_door import native_loaded_module
from bindings.prep_keys_unpack import prep_keys
from bindings.prep_stateless_native_door import _words
from bindings.prep_classify_common_door import cls_mode
from bindings.arena_prog_native_door import _family_entries
from bindings.x_prep_native_owner_door import xprep_key
from bindings.x_prep_native_owner_run import xprep_r3
from bindings.prep_classify_prog_run import ClsRunPlan
from bindings.prep_select_run import RFE_SELECT_HALF, RFE_SELECT_FRACTION, RFE_SELECT_COUNT, rfe_member_source
from core.member_abi import (
    MemberPlan, member_kind_capabilities,
    MEMBER_CAN_FIT, MEMBER_CAN_EXPORT, MEMBER_CAN_CLASSIFIER, MEMBER_GIL_FREE, MEMBER_GIL_DOOR,
)
from core.member_families import (
    MEMBER_FAMILY_TREE, MEMBER_FAMILY_FOREST, MEMBER_FAMILY_NAIVE_BAYES, MEMBER_FAMILY_DISCRIMINANT,
    MEMBER_FAMILY_GBDT, MEMBER_FAMILY_ENSEMBLE,
)

comptime _I64P = MutPointer[Int64, MutUntrackedOrigin]


struct RfeBase(Movable):
    """RFE's estimator admitted as a native LEAF member (`rfe_base_admit`). `native` False: not served (the
    door's own estimator doors keep it; the kind's admission answers False). `owner.plan` makes one state
    per round (its importance word set when `importance_word`); `mask` its kind-exact capabilities;
    `classifier` the class's estimator type; `block` / `squared` the importance source; `refusal` the
    words a first round raises (a kind that exports no importance block; empty: none); `gil_free`: the
    kind has MEMBER_GIL_FREE or MEMBER_GIL_DOOR; `template` the one clone PACK writes the final fit into,
    with its `fields` and `member_context`."""
    var native: Bool
    var owner: MemberOwner
    var mask: Int
    var classifier: Bool
    var block: Int
    var squared: Bool
    var importance_word: Bool
    var refusal: String
    var gil_free: Bool
    var template: PythonObject
    var fields: PythonObject
    var member_context: PythonObject

    def __init__(out self):
        self.native = False
        self.owner = MemberOwner()
        self.mask = 0
        self.classifier = False
        self.block = 0
        self.squared = False
        self.importance_word = False
        self.refusal = String("")
        self.gil_free = False
        self.template = PythonObject(None)
        self.fields = PythonObject(None)
        self.member_context = PythonObject(None)


def rfe_base_admit(estimator: PythonObject, g: PythonObject, context: PythonObject, b: PythonObject) -> RfeBase:
    """UNPACK, once per RFE fit (the door) or admission (the kind): admit the estimator as a native leaf
    member when its class is a published member whose KIND fits and exports the block the importance
    getter names ('auto': coef, else feature importances as given; 'coef_' / 'feature_importances_': that
    block, squared, as the attribute path squares it). A published naive Bayes, discriminant or Gbdt kind
    that exports neither block (QDA, GaussianNB, MultinomialNB, BernoulliNB, ComplementNB,
    GradientBoosting and its adapters) is admitted with its refusal, which the first elimination round
    raises before it fits (with no round the final fit runs, as in scikit-learn). A dotted getter path, a
    META estimator, an admission the family refuses: not served. Never raises (the door's own path then
    answers or refuses in its own words)."""
    var base = RfeBase()
    try:
        var getter = String(g)
        var auto = getter == "auto"
        if not auto and getter != "coef_" and getter != "feature_importances_":
            return base^
        var plan = prep_estimator_plan(estimator, context)
        if plan is None or py_get(plan, "member") is None:
            return base^
        var classifier = Bool(py=plan["classifier"])
        var template = prep_estimator_clone(estimator, context)
        var member_context = plan["context"]
        var fields = py_call(plan["fields_get"], template, py_type(template))
        var owner = member_leaf_admit_serving(template, fields, plan, member_context, b,
                                              MEMBER_CAN_FIT | MEMBER_CAN_EXPORT)
        if not owner.published or owner.plan.family == MEMBER_FAMILY_ENSEMBLE:
            return base^
        # A DecisionTree or RandomForest member computes its feature importances only when its parameter
        # word [10] asks for them (core/forest_member_run.mojo); set before the mask is asked, unless the
        # getter names `coef_`.
        if ((owner.plan.family == MEMBER_FAMILY_TREE or owner.plan.family == MEMBER_FAMILY_FOREST)
                and getter != "coef_" and len(owner.plan.params) > 10):
            owner.plan.params[10] = 1.0
            base.importance_word = True
        # The mask of THIS kind (one state minted, asked and released).
        var mask = member_kind_capabilities(owner.plan)
        if ((mask & MEMBER_CAN_CLASSIFIER) != 0) != classifier:
            return base^
        if (mask & MEMBER_CAN_FIT) == 0 or (mask & MEMBER_CAN_EXPORT) == 0:
            return base^
        var family = owner.plan.family
        var name = (classical_estimator_name(estimator)
                    if (family == MEMBER_FAMILY_NAIVE_BAYES or family == MEMBER_FAMILY_DISCRIMINANT
                        or family == MEMBER_FAMILY_GBDT) else String(""))
        var source = rfe_member_source(family, mask, getter, name)
        if source.refusal.byte_length() == 0 and source.block == 0:
            return base^
        base.refusal = source.refusal.copy()
        base.block = source.block
        base.squared = source.squared
        # THE GIL RULE (core/member_abi.mojo): the same answer says whether this kind's fit may run with the
        # GIL released (one fit on the calling thread either way).
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


def rfe_count_words(fields: PythonObject, b: PythonObject) raises -> List[Float64]:
    """UNPACK of n_features_to_select and step, as the door reads them: [select kind (RFE_SELECT_*),
    fraction, count, step is a fraction (1.0), step value, step count]. Their refusals (a value `int`
    refuses) raise here, in the door's order."""
    var nk = prep_keys()
    var nsel_arg = py_item(fields, nk[].k_n_features_to_select)
    var nsel_kind = RFE_SELECT_COUNT
    var nsel_fraction = 0.0
    var nsel_count = 0
    if nsel_arg is None:
        nsel_kind = RFE_SELECT_HALF
    elif (Bool(py=py_call(b.isinstance, nsel_arg, b.float)) and Float64(py=nsel_arg) > 0.0
            and Float64(py=nsel_arg) < 1.0):
        nsel_kind = RFE_SELECT_FRACTION
        nsel_fraction = Float64(py=nsel_arg)
    else:
        nsel_count = Int(py=py_call_typed(b.int, nsel_arg))
    var step_arg = py_item(fields, nk[].k_step)
    var step_fraction = Bool(py=py_call(b.isinstance, step_arg, b.float)) and Float64(py=step_arg) < 1.0
    var step_value = 0.0
    var step_count = 0
    if step_fraction:
        step_value = Float64(py=step_arg)
    else:
        step_count = Int(py=py_call_typed(b.int, step_arg))
    var words: List[Float64] = [Float64(nsel_kind), nsel_fraction, Float64(nsel_count),
                                1.0 if step_fraction else 0.0, step_value, Float64(step_count)]
    return words^


struct RfeImage(Movable):
    """The x_prep image of RFE's round programs as the typed run reads it: the run plan (every capability
    by name, as the door's programs read them: `cls_run_route` decides the route per program), the image's
    `work` switch, and `entries`, the selection that pins the image for as long as a plan uses it."""
    var run: ClsRunPlan
    var work_on: Bool
    var entries: PythonObject

    def __init__(out self, run: ClsRunPlan, work_on: Bool, entries: PythonObject):
        self.run = run
        self.work_on = work_on
        self.entries = entries


def rfe_image(cc: PythonObject) raises -> RfeImage:
    """UNPACK of the image RFE's programs run on (the classifier arena context `cc`, the door's
    `k_select_context`): the process mode's x_prep selection (the door's `ClsProg` with no estimator:
    `cls_selection(None, ...)`), its transport entry and capabilities read once."""
    var entries = _family_entries(cls_mode(cc), "x_prep", cc["prog_context"])
    var has_out = py_get(entries, "x_prep_run_out") is not None
    var dev_store = py_get(entries, "x_prep_dev_put") is not None and py_get(entries, "x_prep_dev_free") is not None
    var run = ClsRunPlan(xprep_key(entries), has_out, py_get(entries, "x_prep_run_scratch") is not None,
                         py_get(entries, "x_prep_run_ranges") is not None, dev_store,
                         dev_store and py_get(entries, "x_prep_dev_take_rows") is not None)
    return RfeImage(run, xprep_r3("work"), entries)


def rfe_publish(fields: PythonObject, final: PythonObject, support: List[Float32], ranking: List[Float32],
                selected: Int, context: PythonObject) raises:
    """PACK: RFE's fitted attributes from typed words, as the door's fit writes them: estimator_ (the
    packed final clone), `_mask` / support_ (a bool list), the kept columns as the gather's index words
    and the admitted estimator (lane p3-msel-parallel-and-process-pool: stored once here, read by
    predict while `_mask` and `estimator_` are still these objects), ranking_ (int64), n_features_,
    numeric_mode_, n_features_in_. `context` is RFE's own (the door's)."""
    var nk = prep_keys()
    var b = native_loaded_module("builtins")
    var cc = py_item(context, nk[].k_select_context)
    var d = len(support)
    var mask_list = Python.list()
    var kept = List[Int]()
    for j in range(d):  # driver-loop: over the fitted support (feature count)
        _ = py_call(mask_list.append, PythonObject(support[j] != Float32(0)))
        if support[j] != Float32(0):
            kept.append(j)
    py_setitem(fields, nk[].k_estimator_, final)
    py_setitem(fields, nk[].k__mask, mask_list)
    py_setitem(fields, nk[].k__support_words_, _words(kept, context))
    py_setitem(fields, nk[].k__support_k_, PythonObject(len(kept)))
    py_setitem(fields, nk[].k__support_mask_, mask_list)
    py_setitem(fields, nk[].k__estimator_admitted_, final)
    py_setitem(fields, nk[].k_support_, py_call(b.list, mask_list))
    # ranking.astype("<i8")
    var ranks = classical_typed(Python.tuple(d), "<i8", py_item(cc, nk[].k_output_context))
    var dst = _I64P(unsafe_from_address=Int(py=classical_address(ranks)))
    for j in range(d):  # driver-loop: over the fitted ranking (feature count)
        dst[j] = Int64(Int(ranking[j]))
    _ = ranks  # written through the pointer above
    py_setitem(fields, nk[].k_ranking_, ranks)
    py_setitem(fields, nk[].k_n_features_, PythonObject(selected))
    py_setitem(fields, nk[].k_numeric_mode_, cls_mode(cc))
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(d))
