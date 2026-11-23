# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the Gbdt family (16): UNPACK / PACK of
core/gbdt_member.mojo, called by `bindings/member_boundary_door.mojo`.

Published classes: `GradientBoostingClassifier` (kind 0),
`GradientBoosting` (kind 1, the booster: its own predict returns raw scores),
`GradientBoostingRegressor` (kind 2) and, since lane mf-meta5,
`ExperimentalTwoLevelFeatureFreq` (kind 3: its own fit, `xg_feature_freq_fit`
after the door's checks; its settings and source columns read here as its fit
door reads them; PACK `_feature_freq_pack`, the door's `_fitted_text`). The adapters (kinds 0 and 2) are
their inner learner: the admission builds the learner their fit builds
(`adapter_member_learner`, a new object; the adapter is only read) and
takes ITS config, table and route, so the member fits exactly the learner's
request; the pack writes one learner (`_learner_`, packed from the member
state) and the adapter's copies of its attributes, as the adapter's fit
door writes them. Not published: `OrderedRMSE` (no mark; its own door):
its fit is a second algorithm, not a config word (XG_ORDERED_FIT, five
settings, an ordered RMSE booster of its own), and it needs a per-fit
`permutation` argument that no member call carries.

  * `gbdt_member_admit` (UNPACK, once per member): the fit's own option
    checks (`gbdt_check_option_tables`) and the typed config
    (`gbdt_config`, with the Ordered checks' texts) read once from the
    unfitted estimator, as the fit door reads them; the route of the gbdt
    image its fit would run in (`gbdt_table` of its run context: mode,
    vendor, entry), and the cat_features / one_hot_features lists as NEW's
    feature-list tail (lane mf-img3: the member fits the category map, the
    one-hot flags, the multi-target losses and, with the caller's group ids
    in `call.groups`, the query-wise losses). Returns False only for a class
    but GradientBoosting itself. An eval set, PairLogit's pairs and a text
    X's categorical keys travel in the member call (lane mf-left4:
    `call.eval_x` / `eval_y`, `pairs` / `pairs_weight`, `cat_keys` /
    `eval_cat_keys`, a text X's keys read by the caller's UNPACK with the fit
    door's own encoding, bindings/gbdt_category_unpack.mojo
    `gbdt_category_admit`). One caller passes them (lane mf-gbdt5): RFE's
    fit parameters, admitted by `gbdt_member_fit_params` below with the fit
    doors' checks and words; model selection, the ensembles and
    IterativeImputer hand a float32 X and sample weights only
    (cross_val_score documents fit parameters and eval_set as not
    supported).
  * `gbdt_member_fit_params` (UNPACK, once per caller fit): a caller's
    `fit(X, y, **fit_params)` keys as typed `MemberFitExtras`
    (core/member_set_run.mojo), see its docstring.
  * `gbdt_member_pack` (PACK): the fitted attributes the fit door's PACK
    writes (`model_`, `bias_`, `best_iteration_`, `stopped_early_`,
    `loss_curve_`, `test_loss_curve_` (the held-out curve of a fit given an
    eval set, else None: lane mf-left4), `n_features_in_`,
    `category_map_` (the member's map block, None without cat_features),
    `approx_dim_`, `n_classes_`, `learning_rate_`,
    `boosting_type_`), the old resident copy dropped as the fit drops it.
  * `gbdt_member_state` (UNPACK): a member state from a fitted estimator's
    model text, loss curve, scalars, (a CTR model) its category map and (a
    fit with an eval set) its held-out curve.

  * The adapters (kinds 0, 2): admission as above over the learner, plus
    the adapter's own refusals (`_new_learner`: numeric_mode, the
    regressor's class_weights; the image's compiled tier against the mode).
    In `fast` the classifier member's codes take the learner's class
    kernels, as its door does (core/gbdt_member.mojo `_fast_codes`). PACK: `_learner_`, `numeric_mode_`, `n_features_in_`,
    `model_`, the five learner attributes; the classifier's `classes_`
    (the caller's labels, normalized as the adapter's fit normalizes them),
    `n_classes_` 2 and `_label_kind_`; `_mojolearn_fitted`. The caller's
    PACK writes `feature_names_in_`. STATE: from the adapter's `_learner_`.

SERIAL BY NATURE: one member's admission or publication (a few record reads
and three block copies); a caller runs its members in parallel.
"""
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from bindings.member_owner_unpack import MemberOwner
from bindings.py_balanced import py_call, py_get, py_item, py_setitem, py_fields, py_type
from bindings.classical_arrays_door import classical_address
from bindings.admitted_input_unpack import classical_view, admitted_vector
from bindings.trees_keys_unpack import trees_keys
from bindings.trees_route_boundary_door import trees_route_context
from bindings.trees_public_gbdt_door import (
    gbdt_table, gbdt_entry_of, _field, _repr, _real_setting, _int, _source_columns, _fitted_text,
)
from bindings.py_typed_call import py_call_typed
from bindings.native_loaded_module_door import native_loaded_module
from bindings.trees_public_gbdt_fit_door import (
    gbdt_config, gbdt_check_option_tables, _tables, _truthy, _in, _eq, _f64_curve,
    adapter_member_learner, adapter_label_kind,
    _group_sizes, _pairs_arrays, _adapter_eval, _label_kind, _regression_target,
)
from bindings.labels_descriptor_boundary_door import labels_scope
from bindings.trees_public_inputs_door import trees_encode_labels, trees_address
from bindings.admitted_input_unpack import AdmittedX, admit_f32, admit_weight_words
from core.member_set_run import MemberFitExtras
from bindings.native_typed_context_pack import native_typed_context
from bindings.classifier_public_lifecycle_door import _public_labels
from bindings.member_owner_unpack import member_labels_take
from gbdt.fit_calls import xg_numeric_mode
from bindings.gbdt_config_run import gbdt_config_words
from gbdt.fit_request import gbdt_string_at
from core.entry_table import entry_ref_mode, entry_ref_vendor
from core.device_route import DeviceRoute, DEVICE_CPU, DEVICE_GPU
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_new, member_free,
    MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_SCALARS, MEMBER_IMPORT_DONE,
    MEMBER_F32, MEMBER_F64, MEMBER_I32, MEMBER_I64,
)


comptime _ADAPTER_COPIED = 5


def _adapter_name(kind: Int) -> String:
    return String("GradientBoostingClassifier") if kind == 0 else String("GradientBoostingRegressor")


def gbdt_member_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """UNPACK: fill `owner.plan.route`, `owner.plan.params` and `owner.pins`
    from the unfitted estimator (see the module docstring)."""
    var name = String(py=py_type(model).__name__)
    if kind == 1:
        if name != "GradientBoosting":
            return False
        return _learner_admit(model, 1.0, trees_route_context(model, context), owner, List[Float64]())
    if kind == 3:
        # ExperimentalTwoLevelFeatureFreq (lane mf-meta5; core/gbdt_member.mojo kind 3): the booster's
        # admission over its own fields, plus the FeatureFreq fit's settings as its fit door reads them
        # (`feature_freq_unpack`: float(learning_rate), float(l2_leaf_reg), int(random_state), the source
        # columns), before the feature-list tail.
        if name != "ExperimentalTwoLevelFeatureFreq":
            return False
        var own = py_fields(model)
        var nk = trees_keys()
        var extra: List[Float64] = [
            _real_setting(own, model, nk[].k_learning_rate, "learning_rate"),
            _real_setting(own, model, nk[].k_l2_leaf_reg, "l2_leaf_reg"),
            Float64(_int(_field(own, model, nk[].k_random_state, "random_state"))),
        ]
        var sources = _source_columns(_field(own, model, nk[].k_sources, "sources"))
        for i in range(len(sources)):  # small-loop(the source columns)
            extra.append(Float64(sources[i]))
        extra.append(Float64(len(sources)))
        return _learner_admit(model, 3.0, trees_route_context(model, context), owner, extra^)
    if (kind != 0 and kind != 2) or name != _adapter_name(kind):
        return False
    # The adapter's typed run context, as its lifecycle door hands its fit.
    var native = native_typed_context(trees_route_context(model, context), False, True)
    var learner = adapter_member_learner(model, native)
    var nk = trees_keys()
    var mode = String(py=py_item(py_fields(learner), nk[].k_numeric_mode))
    var code = xg_numeric_mode(gbdt_entry_of(gbdt_table(py_fields(learner), native)))
    if code != (0 if mode == "fast" else 1):
        raise Error("RuntimeError: GBDT adapter native numeric mode disagrees with requested mode")
    return _learner_admit(learner, Float64(kind), native, owner, List[Float64]())


def _learner_admit(model: PythonObject, kind: Float64, run: PythonObject, mut owner: MemberOwner,
                   var extra: List[Float64]) raises -> Bool:
    """The booster's admission over `model` (GradientBoosting itself, or an
    adapter's learner) in its run context `run`; `kind` is NEW's word 0;
    `extra` the kind's words before the feature-list tail (kind 3's)."""
    var own = py_fields(model)
    var nk = trees_keys()
    var cat = _field(own, model, nk[].k_cat_features, "cat_features")
    var hot = _field(own, model, nk[].k_one_hot_features, "one_hot_features")
    var t = _tables(run)
    var loss = _field(own, model, nk[].k_loss, "loss")
    if _in(loss, py_item(t, nk[].k_multi_regression_losses)):
        # The fit door's refusals of a multi-target loss that read no rows
        # (`gbdt_fit_unpack`, in its order and words); the rows' own (y's
        # width, group ids) are the member fit's.
        if _field(own, model, nk[].k_class_weights, "class_weights") is not None:
            raise Error("mojolearn: class_weights do not apply to loss=" + _repr(loss))
        if _truthy(cat) or _truthy(hot):
            raise Error("NotImplementedError: mojolearn: loss=" + _repr(loss) + " with cat_features or "
                        + "one_hot_features is not carried: the CTR target "
                        + "binarization reads one target per row")
        if _truthy(_field(own, model, nk[].k_boost_from_average, "boost_from_average")):
            raise Error("NotImplementedError: mojolearn: boost_from_average=True with loss="
                        + _repr(loss) + " is not carried: its StartingPoint is "
                        + "per-dimension (optimal_const_for_loss.h:230-239) and "
                        + "this model's bias is one value (DEVIATION 5951)")
    # The fit's own checks and settings, read once (the fit door reads the
    # same at every fit): refused now, in its words.
    gbdt_check_option_tables(own, model, run)
    var config = gbdt_config(own, model, True, run)
    var table = gbdt_table(own, run)
    var entry = gbdt_entry_of(table)
    var vendor = entry_ref_vendor(entry)
    var route = DeviceRoute(DEVICE_CPU if vendor == 0 else DEVICE_GPU)
    route.mode = entry_ref_mode(entry)
    route.vendor = vendor
    route.entry = entry
    owner.plan.route = route
    var words: List[Float64] = [kind]
    words.extend(gbdt_config_words(config))
    words.extend(extra^)
    # The feature-list tail (core/gbdt_member.mojo NEW): cat_features, then
    # one_hot_features, as the estimator holds them (`int` of each, the fit
    # door's `gbdt_category_columns` / `_flags` reads), then their counts.
    var counts = List[Int](length=2, fill=0)
    var b = native_loaded_module("builtins")
    for which in range(2):  # driver-loop: over fixed call structure, small-loop(lists): the two feature-list arguments
        var values = cat if which == 0 else hot
        if not _truthy(values):
            continue
        var listed = py_call(b.list, values)
        for k in range(len(listed)):  # driver-loop: over the feature-list argument indices
            words.append(Float64(Int(py=py_call_typed(b.int, listed[k]))))
        counts[which] = len(listed)
    words.append(Float64(counts[0]))
    words.append(Float64(counts[1]))
    owner.plan.params = words^
    owner.pins = Python.tuple(table, run)
    return True


# ---------------------------------------------------------------- a caller's fit parameters (lane mf-gbdt5)
struct GbdtMemberFitParams(Movable):
    """UNPACK of a caller's fit parameters for a Gbdt member's FIT
    (`gbdt_member_fit_params`). `admitted` False: a key or a case the member
    does not carry (the caller keeps the estimator's own fit door, which
    answers in its own words). True: `extras` holds the typed addresses
    (core/member_set_run.mojo `MemberFitExtras`); `pins` the Python holders
    of the eval set, the weights, the group sizes and the pairs, and `ids` /
    `codes` the native words built here (group ids, a classifier's held-out
    codes), all alive while this value lives (THE LIFETIME RULE)."""
    var admitted: Bool
    var extras: MemberFitExtras
    var pins: PythonObject
    var ids: List[Int32]
    var codes: List[Int32]

    def __init__(out self):
        self.admitted = False
        self.extras = MemberFitExtras()
        self.pins = PythonObject(None)
        self.ids = List[Int32]()
        self.codes = List[Int32]()


def _eval_items(eval_set: PythonObject, b: PythonObject) raises -> PythonObject:
    """The booster's `_eval_set` shape checks, in its words: one (X, y)
    pair, or a list of exactly one."""
    var pair = eval_set
    if Bool(py=py_call(b.isinstance, eval_set, b.list)):
        if len(eval_set) != 1:
            raise Error("mojolearn: eval_set takes ONE (X, y) pair; the "
                        + "boosting carries a single test cursor. CatBoost "
                        + "accepts a list, this takes " + String(len(eval_set)) + " as an "
                        + "error rather than scoring the first and dropping "
                        + "the rest")
        pair = eval_set[0]
    var items = PythonObject(None)
    try:
        items = py_call(b.tuple, pair)
    except:
        raise Error("mojolearn: eval_set must be (X_eval, y_eval)")
    if len(items) != 2:
        raise Error("mojolearn: eval_set must be (X_eval, y_eval)")
    return items


def _eval_x(source: PythonObject, d: Int, run: PythonObject) raises -> AdmittedX:
    """The booster's `_eval_set` admission of the held-out X (its words),
    C-order: the member reads dense row-major rows and makes its own
    column-major copy (core/gbdt_member.mojo `_fit`), so the door's F-order
    words are the same values in the other order."""
    var xe = admit_f32(source, 2, "eval_set X", run, finite=False)
    if xe.cols != d:
        raise Error("mojolearn: eval_set X has " + String(xe.cols) + " features "
                    + "for a fit on " + String(d))
    if xe.rows == 0:
        raise Error("mojolearn: eval_set X has no rows")
    return xe^


def _carried_keys(fit_params: PythonObject, kind: Int, b: PythonObject) raises -> Bool:
    """Every key is an exact str the kind's public fit takes: the booster's
    `fit(X, y, sample_weight, eval_set, group_id, subgroup_id, pairs,
    pairs_weight)`, an adapter's `fit(X, y, sample_weight, eval_set)`. Any
    other key keeps the fit door (its TypeError, in its words)."""
    var keys = py_call(b.list, fit_params)
    for i in range(len(keys)):  # driver-loop: over the caller's fit parameter names
        var key = keys[i]
        if py_type(key) is not b.str:
            return False
        var name = String(py=key)
        if name == "sample_weight" or name == "eval_set":
            continue
        if kind == 1 and (name == "group_id" or name == "subgroup_id" or name == "pairs" or name == "pairs_weight"):
            continue
        return False
    return True


def gbdt_member_fit_params(model: PythonObject, plan: MemberPlan, fit_params: PythonObject, y: PythonObject,
                           labels: PythonObject, n: Int, d: Int, context: PythonObject) raises -> GbdtMemberFitParams:
    """UNPACK, once per caller fit (lane mf-gbdt5; RFE.fit(X, y,
    **fit_params) over a Gbdt member, bindings/prep_select_native_door.mojo
    `_rfe_member`): the caller's fit parameters admitted with the estimator's
    own fit door's checks, in its order and words, into typed addresses for
    every member FIT of that caller (`MemberFitExtras`; the member re-checks
    what reads its views, core/gbdt_member.mojo `_fit`, in the same words).

    `model` the unfitted estimator (GradientBoosting, or an adapter), `plan`
    its admitted member plan (kind word), `y` the caller's target, `labels`
    the classes of the caller's one label encode (a classifier adapter: the
    held-out labels are coded over them), `n` x `d` the caller's X.
      * booster (kind 1), as `gbdt_fit_unpack`: sample_weight
        (`admit_weight_words`), eval_set (shape, X, y: `_eval_set`), then
        subgroup_id, pairs_weight without pairs, pairs' loss and group_id,
        group_id (`_group_sizes`: its runs become one int32 id per row, so the
        member's runs are the door's), pairs (`_pairs_arrays`: uint32 words,
        every one below n, read as int32).
      * adapters (kinds 0, 2), as `adapter_classifier_fit` /
        `adapter_regressor_fit` then the learner's `gbdt_fit_unpack`: the
        held-out shape (`_adapter_eval`) and target (the classifier: its
        labels' kind and vocabulary against the training labels, coded over
        `labels`; the regressor: `_regression_target`), then sample_weight,
        then the held-out X.
    Not admitted (the fit door keeps them): a key the kind's fit does not
    take, an eval_set with cat_features (a text held-out X is encoded through
    the fit's map by the door), a classifier adapter over other than two
    classes (the adapter's own binary refusal)."""
    var b = native_loaded_module("builtins")
    var nk = trees_keys()
    var got = GbdtMemberFitParams()
    var kind = plan.kind
    if not _carried_keys(fit_params, kind, b):
        return got^
    var sample_weight = py_get(fit_params, PythonObject("sample_weight"))
    var eval_set = py_get(fit_params, PythonObject("eval_set"))
    var group_id = py_get(fit_params, PythonObject("group_id"))
    var subgroup_id = py_get(fit_params, PythonObject("subgroup_id"))
    var pairs = py_get(fit_params, PythonObject("pairs"))
    var pairs_weight = py_get(fit_params, PythonObject("pairs_weight"))
    # The cat_features count is NEW's word len - 2 (the feature-list tail).
    var cats = 0
    if len(plan.params) >= 2:
        cats = Int(plan.params[len(plan.params) - 2])
    if eval_set is not None and cats > 0:
        return got^
    var run: PythonObject
    if kind == 1:
        run = trees_route_context(model, context)
    else:
        run = native_typed_context(trees_route_context(model, context), False, True)
    var extras = MemberFitExtras()
    extras.rows = n
    var weights = PythonObject(None)
    var eval_x = PythonObject(None)
    var eval_y = PythonObject(None)
    var group_holder = PythonObject(None)
    var pair_holders = PythonObject(None)
    if kind == 1:
        if sample_weight is not None:
            var w = admit_weight_words(sample_weight, n, run)
            weights = w.array
            extras.weights = w.address
        if eval_set is not None:
            var items = _eval_items(eval_set, b)
            var xe = _eval_x(items[0], d, run)
            var ye = admit_f32(items[1], 1, "eval_set y", run, finite=False)
            if ye.rows != xe.rows:
                raise Error("mojolearn: eval_set y has " + String(ye.rows) + " values for "
                            + String(xe.rows) + " rows")
            eval_x = xe.array
            eval_y = ye.array
            extras.eval_x = xe.address
            extras.eval_rows = xe.rows
            extras.eval_cols = xe.cols
            extras.eval_y = ye.address
            extras.eval_y_dtype = MEMBER_F32
        var loss = _field(py_fields(model), model, nk[].k_loss, "loss")
        if subgroup_id is not None:
            raise Error("NotImplementedError: mojolearn: subgroup_id is read by no loss this implementation "
                        + "trains; the ranking losses that read CatBoost's subgroup ids "
                        + "are not implemented")
        if pairs_weight is not None and pairs is None:
            raise Error("mojolearn: pairs_weight needs pairs")
        if pairs is not None:
            if not _eq(loss, "PairLogit"):
                raise Error("NotImplementedError: mojolearn: pairs are read by loss='PairLogit' only; "
                            + "loss=" + _repr(loss) + " does not use them")
            if group_id is None:
                raise Error("NotImplementedError: mojolearn: pairs without group_id are not implemented: "
                            + "the CatBoost reference then regroups and reorders the "
                            + "whole pool by the pairs' connected components; pass "
                            + "group_id")
        if group_id is not None:
            group_holder = _group_sizes(group_id, n, run)
            # One id per row from the door's runs (SERIAL BY NATURE: a prefix
            # over the runs; n words): run r is id r, so the member's
            # `_group_sizes` reads back the same runs.
            var runs = Int(py=group_holder.size)
            var sizes = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=trees_address(group_holder))
            got.ids = List[Int32](length=n, fill=Int32(0))
            var at = 0
            for r in range(runs):  # driver-loop: over the door's group runs (each run's rows filled natively)
                var size = Int(sizes[r])
                for i in range(at, min(at + size, n)):
                    got.ids[i] = Int32(r)
                at += size
            _ = group_holder  # keep the Python object alive through the native reads above
            extras.groups = Int(got.ids.unsafe_ptr())
            if pairs is not None:
                pair_holders = _pairs_arrays(pairs, pairs_weight, n, run)
                # uint32 words each below n (the door's range check): the same
                # values as int32 words.
                extras.pairs = trees_address(pair_holders[0])
                extras.pair_count = Int(py=pair_holders[1].size)
                extras.pairs_weight = trees_address(pair_holders[1])
    else:
        var pair = PythonObject(None)
        if eval_set is not None:
            pair = _adapter_eval(eval_set)
            if kind == 0:
                if labels is None or len(labels) != 2:
                    return got^
                var scope = labels_scope(model, run)
                var train = trees_encode_labels(y, scope)
                var train_kind = _label_kind(train, y, "y", run)
                var classes = train_kind[1]
                var held = trees_encode_labels(pair[1], scope)
                var held_kind = _label_kind(held, pair[1], "eval_set y", run)
                if not Bool(py=held_kind[0] == train_kind[0]):
                    raise Error("eval_set contains labels outside the training vocabulary")
                var table = List[Int32](length=len(held_kind[1]), fill=Int32(0))
                for i in range(len(held_kind[1])):  # driver-loop: over the eval set's distinct labels
                    var found = -1
                    for j in range(len(classes)):  # driver-loop: over fixed call structure, small-loop(classes): the two training classes
                        if Bool(py=classes[j] == held_kind[1][i]):
                            found = j
                    if found < 0:
                        raise Error("eval_set contains labels outside the training vocabulary")
                    # The training class's code in the caller's encode (the
                    # same sorted order: one encoder, bindings/trees_public_inputs_door.mojo).
                    table[i] = Int32(found)
                var source = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=trees_address(held.codes))
                got.codes = List[Int32](length=held.n, fill=Int32(0))
                for r in range(held.n):  # native data loop: the held-out codes over the training classes (gather)
                    got.codes[r] = table[Int(source[r])]
                eval_y = held.codes
                extras.eval_y = Int(got.codes.unsafe_ptr())
                extras.eval_y_dtype = MEMBER_I32
                extras.eval_rows = held.n
            else:
                var target = _regression_target(pair[1], "eval_set y", run)
                eval_y = target.array
                extras.eval_y = target.address
                extras.eval_y_dtype = MEMBER_F32
                extras.eval_rows = target.rows
        if sample_weight is not None:
            var w = admit_weight_words(sample_weight, n, run)
            weights = w.array
            extras.weights = w.address
        if eval_set is not None:
            var xe = _eval_x(pair[0], d, run)
            if extras.eval_rows != xe.rows:
                raise Error("mojolearn: eval_set y has " + String(extras.eval_rows) + " values for "
                            + String(xe.rows) + " rows")
            eval_x = xe.array
            extras.eval_x = xe.address
            extras.eval_rows = xe.rows
            extras.eval_cols = xe.cols
    got.pins = Python.tuple(weights, eval_x, eval_y, group_holder, pair_holders)
    got.extras = extras^
    got.admitted = True
    return got^


def gbdt_member_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, state: Int) raises:
    """PACK: write the fitted attributes of state `state` into `model`, as
    the fit door's PACK writes them."""
    if plan.kind == 0 or plan.kind == 2:
        _adapter_pack(model, context, b, plan, state)
        return
    if plan.kind == 3:
        _feature_freq_pack(model, plan, state)
        return
    _learner_pack(model, context, plan, state)


def _feature_freq_pack(model: PythonObject, plan: MemberPlan, state: Int) raises:
    """PACK of an ExperimentalTwoLevelFeatureFreq state: what its fit door's PACK writes (`model_`, then
    `_fitted_text`: n_features_in_, approx_dim_ 1, n_classes_ None, bias_ 0, best_iteration_ 0,
    stopped_early_ False, no curves), the same body."""
    var nk = trees_keys()
    var own = py_fields(model)
    var call = MemberCall(plan.route, state)
    call.block = MEMBER_BLOCK_SCALARS
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var s = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call.dst.data)
    var features = Int(s[0])
    var nbytes = Int(s[7])
    call.block = MEMBER_BLOCK_FAMILY
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var text = gbdt_string_at(call.dst.data, nbytes)
    py_setitem(own, nk[].k_model_, PythonObject(text))
    _fitted_text(own, model, features, PythonObject(0))


def _adapter_pack(model: PythonObject, context: PythonObject, b: PythonObject, plan: MemberPlan, state: Int) raises:
    """PACK of an adapter: ONE learner packed from the member state, then the
    adapter's fields as `_fit_native` and `adapter_classifier_fit` write them."""
    var own = py_fields(model)
    var nk = trees_keys()
    var labels = PythonObject(None)
    if plan.kind == 0:
        labels = member_labels_take(own, b)
        if labels is None or len(labels) != 2:
            raise Error("member ABI: a GradientBoostingClassifier member's pack needs its two class labels")
    var native = native_typed_context(trees_route_context(model, context), False, True)
    var learner = adapter_member_learner(model, native)
    _learner_pack(learner, context, plan, state)
    var learner_fields = py_fields(learner)
    py_setitem(own, nk[].k__learner_, learner)
    py_setitem(own, nk[].k_numeric_mode_, py_item(learner_fields, nk[].k_numeric_mode))
    py_setitem(own, nk[].k_n_features_in_, py_item(learner_fields, nk[].k_n_features_in_))
    py_setitem(own, nk[].k_model_, py_item(learner_fields, nk[].k_model_))
    var copied = Python.tuple("loss_curve_", "test_loss_curve_", "best_iteration_", "stopped_early_", "learning_rate_")
    for i in range(_ADAPTER_COPIED):  # driver-loop: over fixed call structure, small-loop(copied): the five learner attributes
        py_setitem(own, copied[i], learner_fields[copied[i]])
    if plan.kind == 0:
        var kind = adapter_label_kind(labels, "y", native)
        py_setitem(own, nk[].k_classes_, _public_labels(kind[1], native))
        py_setitem(own, nk[].k_n_classes_, PythonObject(2))
        py_setitem(own, nk[].k__label_kind_, kind[0])
    py_setitem(own, PythonObject("_mojolearn_fitted"), PythonObject(True))


def _learner_pack(model: PythonObject, context: PythonObject, plan: MemberPlan, state: Int) raises:
    """The booster's fitted attributes of state `state` into `model`."""
    var run = trees_route_context(model, context)
    var nk = trees_keys()
    var own = py_fields(model)
    # A refit drops the previous model's device copy (its holder releases it).
    _ = py_call(own.pop, "_resident", PythonObject(None))
    var call = MemberCall(plan.route, state)
    call.block = MEMBER_BLOCK_SCALARS
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var s = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call.dst.data)
    var features = Int(s[0])
    var dim = Int(s[1])
    var classes = Int(s[2])
    var bias = s[3]
    var best = Int(s[4])
    var stopped = s[5] != 0.0
    var rate = s[6]
    var nbytes = Int(s[7])
    var ordered = s[8] != 0.0
    call.block = MEMBER_BLOCK_FAMILY
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var text = gbdt_string_at(call.dst.data, nbytes)
    call.block = MEMBER_BLOCK_FAMILY + 1
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var count = call.dst.rows * call.dst.cols
    var learn = List[Float64](length=count, fill=0.0)
    if count > 0:
        unsafe_memcpy(dest=learn.unsafe_ptr(), src=MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call.dst.data),
                      count=count)
    # The category map (a model fitted with cat_features): one int64 Array of
    # its stored words, as the fit door's `gbdt_category_fit` returns it.
    call.block = MEMBER_BLOCK_FAMILY + 2
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var map_words = call.dst.rows * call.dst.cols
    var category_map = PythonObject(None)
    if map_words > 0:
        category_map = admitted_vector(map_words, "<i8", run)
        unsafe_memcpy(dest=MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(category_map))),
                      src=MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=call.dst.data), count=map_words)
    # The held-out loss curve (lane mf-left4): None for a fit without an
    # eval set, else one float64 Array, as the fit door's PACK writes it.
    call.block = MEMBER_BLOCK_FAMILY + 3
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var test_curve = PythonObject(None)
    if call.dst.rows > 0:
        var test = List[Float64](length=call.dst.cols, fill=0.0)
        if call.dst.cols > 0:
            unsafe_memcpy(dest=test.unsafe_ptr(),
                          src=MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call.dst.data),
                          count=call.dst.cols)
        test_curve = _f64_curve(test, run)
    py_setitem(own, nk[].k_model_, PythonObject(text))
    py_setitem(own, nk[].k_bias_, PythonObject(bias))
    py_setitem(own, nk[].k_best_iteration_, PythonObject(best))
    py_setitem(own, nk[].k_stopped_early_, PythonObject(stopped))
    py_setitem(own, nk[].k_loss_curve_, _f64_curve(learn, run))
    py_setitem(own, nk[].k_test_loss_curve_, test_curve)
    py_setitem(own, nk[].k_n_features_in_, PythonObject(features))
    py_setitem(own, nk[].k_category_map_, category_map)
    py_setitem(own, nk[].k_approx_dim_, PythonObject(dim))
    py_setitem(own, nk[].k_n_classes_, PythonObject(classes) if classes > 0 else PythonObject(None))
    py_setitem(own, nk[].k_learning_rate_, PythonObject(rate))
    py_setitem(own, nk[].k_boosting_type_, PythonObject("Ordered" if ordered else "Plain"))


def gbdt_member_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan) raises -> Int:
    """UNPACK: a new member state holding the fitted attributes of `model`
    (an adapter's: of its `_learner_`). The caller frees it (`member_free`)."""
    var nk = trees_keys()
    var own = py_fields(model)
    if plan.kind == 0 or plan.kind == 2:
        var learner = py_get(own, nk[].k__learner_)
        if learner is None:
            raise Error("NotFittedError: this member is not fitted yet")
        own = py_fields(learner)
    var source = py_get(own, nk[].k_model_)
    if source is None:
        raise Error("NotFittedError: this member is not fitted yet")
    var category_map = py_get(own, nk[].k_category_map_)
    var map_view = PythonObject(None)
    if category_map is not None:
        map_view = classical_view(category_map, context, "category_map_")
    var text = String(py=source)
    var nbytes = text.byte_length()
    var packed = List[Int32](length=max((nbytes + 3) // 4, 1), fill=0)
    unsafe_memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(packed.unsafe_ptr())),
                  src=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(text.unsafe_ptr())), count=nbytes)
    var curve = py_get(own, nk[].k_loss_curve_)
    # An ExperimentalTwoLevelFeatureFreq model has no curves and no learning_rate_ (its fit door writes
    # neither): an empty learn curve and its learning_rate setting (lane mf-meta5).
    var feature_freq = plan.kind == 3
    var no_curve = List[Float64](length=1, fill=0.0)
    var curve_view = PythonObject(None)
    if not feature_freq or curve is not None:
        curve_view = classical_view(curve, context, "gbdt loss curve")
    var test_curve = py_get(own, nk[].k_test_loss_curve_)
    var test_view = PythonObject(None)
    if test_curve is not None:
        test_view = classical_view(test_curve, context, "gbdt test loss curve")
    var classes_source = py_get(own, nk[].k_n_classes_)
    var boosting = py_get(own, nk[].k_boosting_type_)
    var scalars: List[Float64] = [
        Float64(py=py_item(own, nk[].k_n_features_in_)), Float64(py=py_item(own, nk[].k_approx_dim_)),
        0.0 if classes_source is None else Float64(py=classes_source), Float64(py=py_item(own, nk[].k_bias_)),
        Float64(py=py_item(own, nk[].k_best_iteration_)), 1.0 if Bool(py=py_item(own, nk[].k_stopped_early_)) else 0.0,
        Float64(py=py_item(own, nk[].k_learning_rate if feature_freq else nk[].k_learning_rate_)), Float64(nbytes),
        1.0 if boosting is not None and Bool(py=boosting == PythonObject("Ordered")) else 0.0]
    var call = member_new(plan)
    try:
        call.block = MEMBER_BLOCK_FAMILY
        call.x = MemberMatrix(Int(packed.unsafe_ptr()), 1, len(packed), MEMBER_I32)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = MEMBER_BLOCK_FAMILY + 1
        if curve_view is None:
            call.x = MemberMatrix(Int(no_curve.unsafe_ptr()), 1, 0, MEMBER_F64)
        else:
            call.x = MemberMatrix(Int(py=classical_address(curve_view)), 1, Int(py=curve_view.size), MEMBER_F64)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        if map_view is not None:
            call.block = MEMBER_BLOCK_FAMILY + 2
            call.x = MemberMatrix(Int(py=classical_address(map_view)), 1, Int(py=map_view.size), MEMBER_I64)
            _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        if test_view is not None:
            call.block = MEMBER_BLOCK_FAMILY + 3
            call.x = MemberMatrix(Int(py=classical_address(test_view)), 1, Int(py=test_view.size), MEMBER_F64)
            _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, len(scalars), MEMBER_F64)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = curve_view
    _ = len(no_curve)
    _ = map_view
    _ = test_view
    _ = len(packed)
    _ = len(scalars)
    _ = text
    return call.state
