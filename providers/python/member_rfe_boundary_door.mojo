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

  * `rfe_dotted_path` (lane mf-del5b; moved here from the door's `_rfe_dotted_base`): a dotted
    importance_getter ('best_estimator_.coef_', ...) walked ONCE over the META kinds it steps through
    (`rfe_hop`), to the hops of THE FORWARDED EXPORT and the leaf's refusal words. The door then admits
    the META estimator through the member dispatch; THE RFE KIND through the ensemble boundary's
    work-list (`_admit_nested`: the META estimator is one of RFE's `_meta_children`).

No import of the member dispatch (bindings/member_boundary_door.mojo): this file is imported by the
ensemble boundary, which that dispatch imports. A META estimator is an RFE base through a dotted getter
(`rfe_dotted_path`); with a plain getter it exports no importance block, so RFE's door admits it through the
dispatch with the 'auto' / attrgetter refusal its first round raises (bindings/prep_select_native_door.mojo
`_rfe_meta_base`, lane mf-rfe7: the door's own rounds over the estimator's doors are deleted).
"""
from std.python import Python, PythonObject
from bindings.member_owner_unpack import MemberOwner, member_mark
from bindings.member_leaf_boundary_door import (
    member_leaf_admit_serving_labeled, member_leaf_admit_serving, member_leaf_admit_labeled,
)
from core.ensemble_member import (
    META_RFE_CLASSIFIER, META_RFE_REGRESSOR, META_SEARCH_CLASSIFIER, META_SEARCH_REGRESSOR, META_STACK_CLASSIFIER,
    META_STACK_REGRESSOR, META_BLOCK_FORWARD,
)
from bindings.member_class_weight_unpack import member_cw_fill_plan
from bindings.prep_native_estimator_door import prep_native_context
from bindings.classical_pandas_door import classical_admit_value
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.admitted_input_unpack import admit_weight_words
from providers.python.member_xneighbors_boundary_door import xneighbors_kind_reads_marker, xneighbors_marker_index
from core.xneighbors_member import LP_MARKER_WORD
from core.member_set_run import MemberFitExtras
from bindings.prep_native_estimator_door import prep_estimator_plan, prep_estimator_clone
from bindings.py_balanced import py_call, py_get, py_item, py_setitem, py_type, py_contains
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
from bindings.prep_select_run import (
    RFE_SELECT_HALF, RFE_SELECT_FRACTION, RFE_SELECT_COUNT, rfe_member_source_named, rfe_block_of, RfeSource,
    RFE_COMPUTED_BAYES_SIGMA,
)
from core.member_abi import (
    MemberPlan, member_kind_capabilities,
    MEMBER_CAN_FIT, MEMBER_CAN_EXPORT, MEMBER_CAN_CLASSIFIER, MEMBER_GIL_FREE, MEMBER_GIL_DOOR,
    MEMBER_CAN_SAMPLE_WEIGHT,
)
from core.member_families import (
    MEMBER_FAMILY_TREE, MEMBER_FAMILY_FOREST, MEMBER_FAMILY_EXTRA_TREES, MEMBER_FAMILY_ENSEMBLE,
    MEMBER_FAMILY_XCLUSTER, MEMBER_FAMILY_DISCRIMINANT, MEMBER_FAMILY_XDECOMP,
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
    # Lane mf-meta5b: a dotted importance_getter over a META estimator (RFE's door, `_rfe_meta_base`):
    # `meta` True, `block` the META kind's forwarded export and `hops` its words (each nested index along
    # the path, then the leaf's block); PACK packs a META member.
    var hops: List[Float64]
    var meta: Bool
    # Lane mf-rfe7: `native` False, the words RFE's door raises (its own rounds through the estimator's
    # doors are deleted); empty: the caller tries its next admission (a dotted path, a META estimator).
    var why: String

    def __init__(out self):
        self.why = String("")
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
        self.hops = List[Float64]()
        self.meta = False


def rfe_base_admit(estimator: PythonObject, g: PythonObject, context: PythonObject, b: PythonObject) -> RfeBase:
    """`rfe_base_admit_labeled` with no labels (THE RFE KIND's admission: its caller hands codes)."""
    return rfe_base_admit_labeled(estimator, g, context, b, PythonObject(None))


def rfe_base_admit_labeled(estimator: PythonObject, g: PythonObject, context: PythonObject, b: PythonObject,
                           labels: PythonObject) -> RfeBase:
    """UNPACK, once per RFE fit (the door) or admission (the kind): admit the estimator as a native leaf
    member when its class is a published member whose KIND fits and exports the block the importance
    getter names ('auto': coef, else feature importances as given; 'coef_' / 'feature_importances_': that
    block, squared, as the attribute path squares it). A published naive Bayes, discriminant or Gbdt kind
    that exports neither block (QDA, GaussianNB, MultinomialNB, BernoulliNB, ComplementNB,
    GradientBoosting and its adapters) is admitted with its refusal, which the first elimination round
    raises before it fits (with no round the final fit runs, as in scikit-learn); since lane mf-rfe7 every
    kind with neither block is. A dotted getter path, a META estimator, an admission the family refuses:
    not served. Never raises.

    `labels` (lane mf-meta5b; `member_leaf_admit_labeled`): None, or an empty list from RFE's own door,
    which encodes y after this admission and completes the plan over its classes itself (a class_weight
    dict's table, LabelPropagation / LabelSpreading's unlabeled marker word: `rfe_plan_labels`).

    Lane mf-rfe7: any attribute name is read here (`rfe_member_source_named`: a named feature-wide block,
    else attrgetter's words at the first round); a dotted getter or a META estimator is the caller's next
    admission. THE RFE KIND keeps this never-raising form; RFE's door calls `rfe_base_admit_door`, which
    raises the family's admission words and says, in `why`, why a class is not served."""
    try:
        return _rfe_base_admit(estimator, g, context, b, labels, False)
    except:
        return RfeBase()


def rfe_base_admit_door(estimator: PythonObject, g: PythonObject, context: PythonObject,
                        b: PythonObject) raises -> RfeBase:
    """RFE's door's admission (lane mf-rfe7): `rfe_base_admit_labeled` with deferred labels, raising the
    family's admission refusals in their words (they raised from the estimator's own fit door's first fit,
    which this admission replaces). `native` False: `why` holds the door's refusal, or is empty for a dotted
    getter or a META estimator (the door's next admission, `_rfe_meta_base`)."""
    return _rfe_base_admit(estimator, g, context, b, Python.list(), True)


def _rfe_base_admit(estimator: PythonObject, g: PythonObject, context: PythonObject, b: PythonObject,
                    labels: PythonObject, raising: Bool) raises -> RfeBase:
    var base = RfeBase()
    var getter = String(g)
    if getter.find(".") >= 0:
        return base^
    var cls = String(py_type(estimator).__name__)
    var plan = prep_estimator_plan(estimator, context)
    if plan is None:
        base.why = "TypeError: mojolearn: RFE requires a mojolearn estimator; got " + cls
        return base^
    if py_get(plan, "member") is None:
        # Lane mf-member8: NearestNeighbors and RadiusNeighbors (neighbors kinds 2, 3), OPTICS (x_cluster kind 3),
        # MDS (x_decomp kind 18) and SpectralEmbedding (spectral kind 1) are fit + export members now, so RFE
        # with no elimination round (n_features_to_select >= n_features: scikit-learn's `_fit` fits the
        # estimator once on every column and reads no importances) runs their fit natively; a round raises
        # RFE_NO_IMPORTANCES or attrgetter's words before it fits, as scikit-learn fails there. The one
        # registered class left unmarked is Cholesky (python/mojolearn/_cholesky_impl.py:87): its `fit(A)`
        # takes no y, and scikit-learn's RFE calls `estimator.fit(X[:, features], y)` in every round and in
        # the final fit (sklearn/feature_selection/_rfe.py:365), so scikit-learn itself fails there with
        # Python's TypeError: refused in those words. Any other unmarked class keeps the door's words.
        if cls == "Cholesky" or cls == "HostCholesky":
            base.why = "TypeError: Cholesky.fit() takes 2 positional arguments but 3 were given"
            return base^
        base.why =("TypeError: mojolearn: RFE requires an estimator with a native member; " + cls
                    + " has none")
        return base^
    var classifier = Bool(py=plan["classifier"])
    var template = prep_estimator_clone(estimator, context)
    var member_context = plan["context"]
    var fields = py_call(plan["fields_get"], template, py_type(template))
    var owner = MemberOwner()
    if raising:
        owner = member_leaf_admit_labeled(template, fields, plan, member_context, b, labels)
    else:
        owner = member_leaf_admit_serving_labeled(template, fields, plan, member_context, b,
                                                  MEMBER_CAN_FIT | MEMBER_CAN_EXPORT, labels)
    if owner.published and owner.plan.family == MEMBER_FAMILY_ENSEMBLE:
        return base^  # a META estimator: the caller's next admission
    if not owner.published:
        # The leaf admission's mark is a META one, or no leaf family fills it: the caller's next admission
        # (the member dispatch) answers, or refuses in `_rfe_meta_base`'s words.
        return base^
    # A DecisionTree or RandomForest member computes its feature importances only when its parameter
    # word [10] asks for them (core/forest_member_run.mojo); set before the mask is asked, when the getter
    # reads them.
    if ((owner.plan.family == MEMBER_FAMILY_TREE or owner.plan.family == MEMBER_FAMILY_FOREST)
            and (getter == "auto" or getter == "feature_importances_") and len(owner.plan.params) > 10):
        owner.plan.params[10] = 1.0
        base.importance_word = True
    # The mask of THIS kind (one state minted, asked and released).
    var mask = member_kind_capabilities(owner.plan)
    if ((mask & MEMBER_CAN_CLASSIFIER) != 0) != classifier:
        base.why = ("TypeError: mojolearn: RFE: " + cls + "'s native member kind and its estimator type "
                    + "disagree")
        return base^
    if (mask & MEMBER_CAN_FIT) == 0 or (mask & MEMBER_CAN_EXPORT) == 0:
        base.why = ("TypeError: mojolearn: RFE requires an estimator whose native member fits and exports its "
                    + "fitted state; " + cls + "'s does not")
        return base^
    var family = owner.plan.family
    var source = rfe_member_source_named(family, owner.plan.kind, mask, getter, classical_estimator_name(estimator))
    if (family == MEMBER_FAMILY_XCLUSTER and owner.plan.kind == 4 and getter == "cluster_centers_"
            and len(owner.plan.params) > 7 and owner.plan.params[7] != 0.0):
        # Lane mf-member8: an affinity='precomputed' AffinityPropagation publishes no cluster_centers_
        # (its PACK, as its fit), so scikit-learn's attrgetter fails at the first round: its words.
        source = RfeSource(0, False, "AttributeError: '" + classical_estimator_name(estimator)
                           + "' object has no attribute 'cluster_centers_'")
    # Round b (lane mf-member8): attributes PACK publishes only under a setting, as scikit-learn does. LDA's
    # covariance_ (NEW words [1] solver 0 'svd', [5] store_covariance): absent for 'svd' without
    # store_covariance (attrgetter's words). MinCovDet / EllipticEnvelope precision_ (word [5]
    # store_precision): None without it, so scikit-learn's `importances.ndim` fails.
    if (family == MEMBER_FAMILY_DISCRIMINANT and owner.plan.kind == 0 and getter == "covariance_"
            and len(owner.plan.params) > 5 and owner.plan.params[1] == 0.0 and owner.plan.params[5] == 0.0):
        source = RfeSource(0, False, "AttributeError: '" + classical_estimator_name(estimator)
                           + "' object has no attribute 'covariance_'")
    if (family == MEMBER_FAMILY_XDECOMP and owner.plan.kind == 14 and getter == "precision_"
            and len(owner.plan.params) > 5 and owner.plan.params[5] == 0.0):
        source = RfeSource(0, False, "AttributeError: 'NoneType' object has no attribute 'ndim'")
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
    return base^


# ---------------------------------------------------------------- a dotted importance getter (lanes mf-meta5b, mf-del5b)
def rfe_hop(kind: Int, name: String, mut field: String) -> Int:
    """One step of a dotted importance_getter over a META kind: the nested member the fitted attribute
    `name` is (its index among the state's nested members; -1 the last) and, in `field`, the constructor
    field that holds its unfitted estimator (whose kind the next step reads). -2: a step no META state
    holds as one nested member (a list attribute, an unfitted parameter): not served."""
    if (kind == META_RFE_CLASSIFIER or kind == META_RFE_REGRESSOR) and name == "estimator_":
        field = String("estimator")
        return 0
    if (kind == META_SEARCH_CLASSIFIER or kind == META_SEARCH_REGRESSOR) and name == "best_estimator_":
        field = String("estimator")
        return 0
    if (kind == META_STACK_CLASSIFIER or kind == META_STACK_REGRESSOR) and name == "final_estimator_":
        field = String("final_estimator")
        return -1
    return -2


struct RfeDottedPath(Movable):
    """A dotted importance_getter resolved ONCE (`rfe_dotted_path`). `served` False: not served (the door
    keeps its attribute reads; the kind's admission answers False). `hops`: THE FORWARDED EXPORT's words
    (core/ensemble_member.mojo META_BLOCK_FORWARD: each step's nested index, then the leaf's block);
    `refusal`: scikit-learn's attrgetter words a first round raises for a leaf whose kind exports neither
    block (empty: none)."""
    var served: Bool
    var hops: List[Float64]
    var refusal: String
    # Lane mf-rfe6: the leaf is a tree or forest read for feature_importances_, so the caller sets the leaf's
    # importance word in the nested plan it admits (core/ensemble_member.mojo `meta_plan_leaf_importances`).
    var leaf_word: Bool
    # Lane mf-rfe7: `served` False, the words RFE's door raises (its rounds through the estimator's own
    # attribute reads are deleted).
    var why: String
    # Lane mf-rfe7b: the RFE block code of THE FORWARDED EXPORT (META_BLOCK_FORWARD, plus the row of the leaf's
    # block a named attribute is: bindings/prep_select_run.mojo `rfe_block_code`); `hops` ends with the plain block.
    var forward_block: Int

    def __init__(out self):
        self.forward_block = META_BLOCK_FORWARD
        self.served = False
        self.hops = List[Float64]()
        self.refusal = String("")
        self.leaf_word = False
        self.why = String("")


def rfe_dotted_path(estimator: PythonObject, g: PythonObject, context: PythonObject, b: PythonObject) -> RfeDottedPath:
    """UNPACK, once per RFE fit (the door) or admission (THE RFE KIND), of a dotted importance_getter
    ('best_estimator_.coef_', 'estimator_.coef_', 'final_estimator_.feature_importances_', or a longer
    chain of such steps): each step a nested member of the META kind it walks (`rfe_hop`), the leaf's
    block from its kind's mask (`rfe_member_source`; squared, as the attribute path squares it). A leaf of
    a family whose kinds export neither block (naive Bayes, discriminant, Gbdt, XNeighbors) is served with
    scikit-learn's attrgetter words ("'<leaf class>' object has no attribute '<name>'"), which the first
    round raises before it fits. Not served: a getter with no dot, a step no META state holds as one
    member, a leaf with no serving member, a leaf path other than coef_ / feature_importances_. A tree or
    forest leaf's feature_importances_ is served with `leaf_word` (lane mf-rfe6): the caller sets the leaf's
    importance word in the nested plan after it admits the META estimator. The META estimator itself is
    admitted by the caller. Never raises.

    Lane mf-rfe7: the leaf may be any attribute name (`rfe_member_source_named`: a named feature-wide block,
    else attrgetter's words at the first round); a path not served says why in `why` (RFE's door raises it:
    its own walk over the fitted estimator's attributes is deleted). A step that is not one fitted member of
    a META kind (a list such as estimators_, a Pipeline's steps, an unfitted parameter) is refused there: the
    deleted walk also found no connected native estimator at such a step and refused it."""
    var path = RfeDottedPath()
    var last_why = String("")
    try:
        var getter = String(py=g)
        var parts = py_call(g.split, ".")
        var count = len(parts)
        if count < 2:
            return path^
        var leaf_name = String(parts[count - 1])
        var hops = List[Float64]()
        var current = estimator
        for k in range(count - 1):  # driver-loop: over the attribute path's steps
            var step = String(parts[k])
            path.why = ("TypeError: mojolearn: RFE importance_getter '" + getter + "': '" + step
                        + "' of " + String(py_type(current).__name__) + " is not one fitted estimator RFE can "
                        + "follow natively (best_estimator_ of a search, estimator_ of an RFE, final_estimator_ "
                        + "of a stacking estimator)")
            last_why = path.why.copy()
            var step_plan = prep_estimator_plan(current, context)
            if step_plan is None or py_get(step_plan, "member") is None:
                return path^
            var mark = member_mark(step_plan, b)
            if mark.family != MEMBER_FAMILY_ENSEMBLE:
                return path^
            var field = String("")
            var at = rfe_hop(mark.kind, step, field)
            if at < -1:
                return path^
            var step_fields = py_call(step_plan["fields_get"], current, py_type(current))
            var nested = py_get(step_fields, PythonObject(field))
            if nested is None:
                return path^
            hops.append(Float64(at))
            current = nested
        var leaf_class = String(py_type(current).__name__)
        path.why = ("TypeError: mojolearn: RFE importance_getter '" + getter + "': " + leaf_class
                    + " has no native member that fits and exports its fitted state")
        last_why = path.why.copy()
        var leaf_plan = prep_estimator_plan(current, context)
        if leaf_plan is None or py_get(leaf_plan, "member") is None:
            return path^
        var leaf_fields = py_call(leaf_plan["fields_get"], current, py_type(current))
        var leaf = member_leaf_admit_serving(current, leaf_fields, leaf_plan, leaf_plan["context"], b,
                                             MEMBER_CAN_FIT | MEMBER_CAN_EXPORT)
        if not leaf.published:
            return path^
        var family = leaf.plan.family
        # A tree or forest leaf's feature_importances_ (lane mf-rfe6): its kind computes them when its
        # importance word [10] asks, as `rfe_base_admit` sets it on a leaf base; the mask is asked with the
        # word set, and the caller sets it on the nested plan its META admission minted
        # (core/ensemble_member.mojo `meta_plan_leaf_importances`).
        if ((family == MEMBER_FAMILY_TREE or family == MEMBER_FAMILY_FOREST) and leaf_name == "feature_importances_"
                and len(leaf.plan.params) > 10):
            leaf.plan.params[10] = 1.0
            path.leaf_word = True
        var source = rfe_member_source_named(family, leaf.plan.kind, member_kind_capabilities(leaf.plan), leaf_name,
                                             leaf_class)
        if source.block >= RFE_COMPUTED_BAYES_SIGMA:
            # Round b (lane mf-member8): a computed attribute (BayesianRidge sigma_, X_scale_) is served on a leaf
            # base only; through a META path it has no forwarded block.
            path.why = ("TypeError: mojolearn: RFE importance_getter '" + String(g) + "' is not one fitted "
                        + "estimator RFE can follow natively (a computed attribute of a nested estimator)")
            return path^
        hops.append(Float64(rfe_block_of(source.block)))
        path.forward_block = META_BLOCK_FORWARD + (source.block - rfe_block_of(source.block))
        path.hops = hops^
        path.refusal = source.refusal.copy()
        path.why = String("")
        path.served = True
    except:
        var failed = RfeDottedPath()
        failed.why = last_why^
        return failed^
    return path^


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


# ---------------------------------------------------------------- the fit's labels and parameters (lane mf-meta5b)
def rfe_plan_labels(mut base: RfeBase, classes: PythonObject, b: PythonObject) raises:
    """UNPACK, after the door's one label encode (`classes`: the class list in code order), for a base
    admitted with deferred labels (`rfe_base_admit_labeled` with an empty list), the way model selection's
    units complete theirs (bindings/msel_member_units_door.mojo): LabelPropagation / LabelSpreading get the
    code of the -1 marker in their word LP_MARKER_WORD (its rows take part in the graph and not in classes_,
    as the estimator's own fit reads them; PACK leaves it out of classes_), and a class_weight dict's table
    is read over the classes (`member_cw_fill_plan`: its refusals raise here in the estimator's words, where
    its first fit raised them).
    Lane mf-rfe7: `member_cw_fill_plan` answers False only for a pending table without labels, or for a family
    with a class-weight tail that reads no dict (none: core/member_class_weight.mojo `member_cw_start` gives a
    tail to the five families bindings/member_class_weight_unpack.mojo `_dict_values` reads). RFE hands the
    labels, so False cannot happen; it raises (REMOVED: the False answer that sent the fit to `_fit_clone`)."""
    if xneighbors_kind_reads_marker(base.owner.plan.family, base.owner.plan.kind):
        var marker = xneighbors_marker_index(classes)
        if marker >= 0 and len(base.owner.plan.params) > LP_MARKER_WORD:
            base.owner.plan.params[LP_MARKER_WORD] = Float64(marker)
    if not member_cw_fill_plan(base.owner.plan, base.template, base.fields, classes, base.member_context, b):
        raise Error("RuntimeError: mojolearn: RFE: " + classical_estimator_name(base.template)
                    + "'s class_weight table was left incomplete over the fit's classes")


def rfe_fit_keywords(estimator: PythonObject, fit_params: PythonObject, context: PythonObject,
                     b: PythonObject) raises:
    """UNPACK of RFE.fit's `**fit_params` names and values, in the words and order of the estimator's own
    compiled fit (bindings/msel_estimator_native_door.mojo `_msel_admitted_dispatch`, which `_fit_clone`'s
    every round ran): an exact str name, one the plan's `fit` keyword table names, each value admitted
    (`classical_admit_value`). Raises the first refusal; once per RFE fit, before any round."""
    var plan = prep_estimator_plan(estimator, context)
    if plan is None:
        return
    var table = py_get(plan, "keywords")
    var allowed = PythonObject(None) if table is None else py_get(table, PythonObject("fit"))
    if allowed is None:
        allowed = Python.tuple()
    var native = prep_native_context(context)
    classical_connect_numpy(native)
    var keys = py_call(b.list, fit_params)
    for k in range(len(keys)):  # driver-loop: over the caller's fit parameter names
        if py_type(keys[k]) is not b.str:
            raise Error("TypeError: native estimator keyword names must be exact strings")
        if not py_contains(allowed, keys[k]):
            raise Error("TypeError: " + String(py_type(estimator).__name__) + ".fit got an unexpected keyword argument '"
                        + String(keys[k]) + "'")
        classical_admit_value(py_get(fit_params, keys[k]), native)


struct RfeFitParams(Movable):
    """RFE.fit's parameters for a member of any family but Gbdt (`rfe_member_fit_params`): `extras` the
    typed addresses (the weights), `pins` their holder, `final_importances` a forest's
    feature_importances=True (the final fit keeps its importance word, so estimator_ publishes them as that
    fit does). REMOVED (lane mf-rfe7): `admitted`, whose False sent the fit to the estimator's own doors."""
    var extras: MemberFitExtras
    var pins: PythonObject
    var final_importances: Bool

    def __init__(out self):
        self.extras = MemberFitExtras()
        self.pins = PythonObject(None)
        self.final_importances = False


def rfe_member_fit_params(base: RfeBase, fit_params: PythonObject, n: Int, b: PythonObject) raises -> RfeFitParams:
    """UNPACK, once per RFE fit (lane mf-meta5b), of the fit parameters a non-Gbdt member reads, after
    `rfe_fit_keywords`: `sample_weight` through the one weight admission the families' fit doors share
    (bindings/admitted_input_unpack.mojo `admit_weights`, via `admit_weight_words`: scikit-learn's shape,
    finiteness, sign and all-zero refusals in its words; None: no weights), handed to every member fit
    (the member re-checks what it reads, as for Bagging's and Voting's members); a forest's
    `feature_importances` (an exact bool: the rounds compute importances whatever it says, as before; True
    keeps them on the final fit). A feature_importances value the forest's fit door refuses raises its
    words here (lane mf-cal5). Lane mf-rfe7: weights for a kind without MEMBER_CAN_SAMPLE_WEIGHT and any
    other key raise here too (the estimator's own fit door, which answered them, is no longer called)."""
    var got = RfeFitParams()
    var extras = MemberFitExtras()
    extras.rows = n
    var weights = PythonObject(None)
    var keys = py_call(b.list, fit_params)
    for k in range(len(keys)):  # driver-loop: over the caller's fit parameter names
        var name = String(py=keys[k])
        var value = py_get(fit_params, keys[k])
        if name == "sample_weight":
            if value is None:
                continue
            if (base.mask & MEMBER_CAN_SAMPLE_WEIGHT) == 0:
                # Lane mf-rfe7 (the fit door no longer answers it): a kind whose fit reads no weights. The
                # RandomForest / DecisionTree regressor with oob_score is such a kind: its fit keyword
                # table takes sample_weight: its fit door refuses the pair before any work
                # (bindings/trees_public_forest_door.mojo `_rf_fit_regressor`), so the deleted rounds raised
                # these words at their first fit; scikit-learn fits it (a weighted out-of-bag fit is kernel
                # work the forest family does not have, for its public fit too).
                if ((base.owner.plan.family == MEMBER_FAMILY_FOREST or base.owner.plan.family == MEMBER_FAMILY_TREE)
                        and not base.classifier):
                    raise Error("NotImplementedError: sample_weight is not implemented together with oob_score "
                                + "or a tree-ID shard")
                raise Error("NotImplementedError: mojolearn: RFE: " + classical_estimator_name(base.template)
                            + "'s native member fit reads no sample_weight")
            var w = admit_weight_words(value, n, base.member_context)
            weights = w.array
            extras.weights = w.address
        elif name == "feature_importances":
            # THE FALLBACK RULE (lane mf-cal5), as each forest fit door reads it
            # (bindings/trees_public_forest_door.mojo): RandomForest / DecisionTree (`rf_fit`) take Python's
            # bool or NumPy's bool_ and refuse anything else in their words before any work; ExtraTrees
            # (`et_fit`) never reads it. Before, any value but an exact bool fell back to those doors.
            var t = py_type(value)
            var boolean = t is b.bool or String(py=t.__name__) == "bool_"
            if base.owner.plan.family == MEMBER_FAMILY_EXTRA_TREES:
                got.final_importances = boolean and Bool(py=value)
            else:
                if not boolean:
                    raise Error("feature_importances must be a boolean fit-output request")
                got.final_importances = Bool(py=value)
        else:
            # Lane mf-rfe7: a key the estimator's fit keyword table names (`rfe_fit_keywords` refused every
            # other) that its member fit does not carry; the deleted rounds handed it to the fit door.
            raise Error("TypeError: mojolearn: RFE: " + classical_estimator_name(base.template)
                        + "'s native member fit takes no '" + name + "' parameter")
    got.pins = weights
    got.extras = extras^
    return got^
