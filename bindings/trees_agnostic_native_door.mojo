# SPDX-License-Identifier: Apache-2.0
# Whole KernelExplainer and PermutationExplainer operations: UNPACK, RUN, PACK
# (lane bm-rest-shap, 2026-11-23). This owner admits the model, background,
# settings and rows and opens the explained estimator's held native member
# (UNPACK); xtrees/agnostic_values_run.mojo runs the schedule, the synthetic
# rows, the model through the native model handle, the solve or the
# permutation fold and the link through the x_trees glue entry (RUN, no Python
# object); this owner shapes the float64 values (PACK). A meta-estimator is
# opened over its members' held native states through the meta doors' X-free
# member UNPACK and evaluated by its public operation's RUN (`AgnMeta`; lane
# mf-shap3, 2026-11-23); a fitted search by its best estimator; an RFE by its
# fitted estimator's member over its support columns, a label propagation model by
# its typed query body, an outlier detector by its member's integer predict (lane
# mf-shap3b). A class (or a member's class) with no member entry whose plan serves
# the method is evaluated through its compiled model-selection plan
# (bindings/prep_native_estimator.mojo; never its Python predict), `_DoorModel`;
# one with neither is refused in words, and so is a forecaster. A Python callable
# model is refused: native code never calls back into it.
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem, py_fields
from bindings.mode_metadata_native_door import _fields, _class_fields
from bindings.native_loaded_module_door import native_loaded_module
from bindings.classical_attributes_door import classical_check_layout
from bindings.prep_native_estimator_door import (
    prep_estimator_admit, prep_estimator_has, prep_estimator_call, prep_estimator_predict_internal,
    prep_estimator_plan, prep_native_context,
)
from bindings.member_held_door import MemberHeld, member_held, member_held_release
from core.member_abi import (
    member_call, member_call_of, member_outputs_of,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_PREDICT, MEMBER_OP_PREDICT_PROBA, MEMBER_OP_DECISION, MEMBER_CAN_PREDICT,
    MEMBER_CAN_PREDICT_PROBA, MEMBER_CAN_TEST_ROWS, MEMBER_GIL_FREE, MEMBER_GIL_DOOR, MEMBER_F32, MEMBER_F64,
    MEMBER_I32, MEMBER_I64, MEMBER_DTYPE_UNSAID,
)
from bindings.trees_public_inputs_run import trees_finite_f32
from bindings.trees_public_inputs_door import trees_table, trees_address, trees_reshape, trees_cast
from bindings.admitted_input_unpack import admitted_vector, admitted_unfilled, classical_view
# TOMBSTONE: trees_view / trees_vector uses (the trees admission aliases) deleted 2026-11-09 by lane a1b-doors-opt-in-trees-linear; code recoverable at f59d913b0b.
# Restore: experiments/removed/A1B_LABELS_ADMISSION_COPY.patch.
from bindings.classical_pandas_door import classical_admit_value
from bindings.trees_public_forest_door import forest_effective_mode, _engine, _ordered_export, _resident
from xtrees.trees_forest_run import _family
from bindings.trees_public_boost_door import _seed, _xk, _xt, _xr
from bindings.trees_explainer_native_door import _get, _int, _entry, _snapshot, _replace, _tree_mode
from xtrees.trees_agnostic_native_run import _name, _SWITCH_FACTS
from xtrees.glue_calls import xt_ref
from xtrees.agnostic_values_run import (
    AgnModel, AgnMember, AgnMeta, AgnMetaPlan, AgnRequest, AGN_META_SOFT, AGN_META_REGRESS, AGN_META_HARD,
    AGN_META_BAG, AGN_META_STACK, AGN_META_OVR, AGN_META_CALIBRATED, AGN_META_ADA, AGN_META_COLUMNS,
    AGN_META_RIDGE, AgnLabelProp, agn_expected_run, agn_kernel_values_run, agn_permutation_values_run,
)
from xtrees.trees_meta_vote_run import VoteMember, CalPair, CalFit, STACK_PREDICT, STACK_PROBA
from bindings.member_set_door import MemberHeldCall, member_held_close, member_held_open
from bindings.neighbors_lp_query_unpack import lp_query_unpack
from bindings.lof_provider_unpack import LofConnection
from providers.python.x_neighbors_descriptor_pyapi_door import xn_admit
from providers.python.estimator_binding_pyapi_door import estimator_binding
from providers.python.member_xneighbors_boundary_door import _family_context as xn_family_context
from core.neighbors_native_driver import XnDriver
from neighbors.label_propagation_run import LpQueryRequest
from bindings.trees_public_meta_door import (
    meta_kind, meta_member_open, _sub_cols, _voting_weights, _attr as meta_attr, calibrated_pair_codes,
    calibrated_pair_words, ovr_member_words, stacking_layout, multioutput_ridge_ref,
)
from bindings.trees_public_boost_door import boost_kind, bag_member_open, bag_member_columns, _ada_weights
from bindings.msel_search_state_door import msel_search_fields
from core.entry_table import EntryRef
from core.trees_entry_table import TreesEntryKey
from bindings.trees_keys_unpack import trees_keys
from bindings.operation_names_trees import TREES_OPN_TREES_AGNOSTIC_KERNEL_INIT, TREES_OPN_TREES_AGNOSTIC_PERMUTATION_INIT, TREES_OPN_TREES_AGNOSTIC_KERNEL_VALUES, TREES_OPN_TREES_AGNOSTIC_PERMUTATION_VALUES
from bindings.py_balanced import py_item

comptime _F32 = MutPointer[Float32, MutUntrackedOrigin]

#: x_trees_fast_switches bit 64 (providers/python/xtrees_api_pyapi_door.mojo
#: XTREES_FAST_SWITCHES): MOJOLEARN_IDN_SHAP_DEVICE_MODEL (an IDENTICAL GPU
#: build: Kernel/Permutation SHAP over a built-in flat forest evaluate the
#: model on the device, xtrees/agnostic_device.mojo `model_kernel`; moves no
#: bit). The RUN reads bits 8, 16 and 32 (xtrees/agnostic_values_run.mojo).
comptime _AGN_IDN_DEVICE_MODEL = 64
# TOMBSTONE: MOJOLEARN_KSHAP_FAST_DEVICE_MODEL (never measured, deleted) deleted 2026-11-09 by e2adeea367; code
# recoverable at f59d913b0b. Same commit: its twin MOJOLEARN_PSHAP_FAST_DEVICE_MODEL (bits 1024 / 2048).
# Restore: git apply experiments/removed/MOJOLEARN_KSHAP_FAST_DEVICE_MODEL.patch


def _b() raises -> PythonObject:
    return native_loaded_module("builtins")



def _switches(table: PythonObject) raises -> Int:
    var query = _entry(table, "x_trees_fast_switches", False)
    if query is None:
        return 0
    var key = Int(query._obj_ptr)
    var facts = _SWITCH_FACTS.get_or_create_ptr()
    if key in facts[].answers:
        return facts[].answers[key]
    var bits = py_call(query)
    if py_type(bits) is not _b().int:
        raise Error("TypeError: x_trees_fast_switches must return an exact integer")
    var answer = Int(py=bits)
    facts[].answers[key] = answer
    return answer


def _class_value(model: PythonObject, name: String) raises -> PythonObject:
    """A class attribute found on the model's MRO namespaces (no descriptor
    or metaclass protocol), or None."""
    var nk = trees_keys()
    var mro = py_call(py_item(_b().type.__dict__, nk[].k___mro__).__get__, py_type(model))
    for i in range(len(mro)):  # driver-loop: the model class ancestry
        var namespace = _class_fields(mro[i])
        if py_contains(namespace, name):
            return namespace[name]
    return PythonObject(None)


#: The refusals the explainers raise natively (lane mf-shap3, 2026-11-23; before, the model's door was asked the
#: whole call again only to raise its own words).
comptime _NONFINITE = "ValueError: Input X contains NaN or infinity."
comptime _WIDTH_CHANGED = "ValueError: the explained model output width changed after construction"
#: scikit-learn's MultiOutputClassifier.predict_proba answers one array per output; shap's KernelExplainer reads
#: one model output array (its `fnull` is the weighted sum of that array over the background), so it does not
#: explain such a model, and neither does this one.
comptime _MULTI_REFUSAL = ("ValueError: the explained model output must be 1-D or 2-D: "
    + "MultiOutputClassifier.predict_proba answers one array per output; explain each fitted output estimator "
    + "(estimators_) instead")
#: A classifier explained through predict answers its labels; shap needs a numeric model output (its background
#: mean of non-numeric labels is refused by NumPy).
comptime _LABELS_REFUSAL = ("TypeError: the explained classifier's classes_ are not numeric, so its predict "
    + "output (its labels) cannot be explained; encode the labels as numbers, or explain a classifier with "
    + "predict_proba")


#: A forecaster's predict takes a forecast horizon (or time points), not rows of features. shap's explainers call
#: the model on the background rows (`match_model_to_data`: "Provided model function fails when applied to the
#: provided data set.") and fail; these refuse it by name instead (lane mf-shap3b, 2026-11-23).
def _forecaster_refusal(name: String, model: PythonObject) raises -> String:
    return ("TypeError: " + name + " explains a model of the feature rows X; " + String(py_type(model).__name__)
            + " is a forecaster: its predict takes a forecast horizon, not rows of features, so it has no output "
            + "over X to explain (shap: Provided model function fails when applied to the provided data set.)")


def _explained_kind(model: PythonObject) raises -> String:
    """The class's own explained kind (lane mf-shap3b, 2026-11-23; `_mojolearn_explained`): "label_propagation"
    and "rfe" from the exact class's own namespace (a subclass is not admitted), "forecaster" from its MRO (every
    forecaster refuses), "" for any other class."""
    var own = _class_fields(py_type(model))
    if py_contains(own, "_mojolearn_explained"):
        var word = own["_mojolearn_explained"]
        if py_type(word) is _b().str:
            return String(word)
    var inherited = _class_value(model, "_mojolearn_explained")
    if inherited is not None and py_type(inherited) is _b().str and String(inherited) == "forecaster":
        return String("forecaster")
    return String("")


def _rfe_estimator(model: PythonObject) raises -> PythonObject:
    """An RFE's fitted `estimator_`, with `RFE.predict`'s own words when it has none."""
    var est = py_get(py_fields(model), PythonObject("estimator_"))
    if est is None:
        raise Error("AttributeError: '" + String(py_type(model).__name__) + "' object has no attribute 'estimator_'")
    return est


def _admit_explained(model: PythonObject, name: String, context: PythonObject) raises:
    """Only connected mojolearn estimators run here. A forecaster is refused (`_forecaster_refusal`); a label
    propagation model is its exact class (its query is typed: `lp_query_unpack`); an RFE admits its fitted
    estimator as `RFE.predict` does (`prep_estimator_admit(estimator_, "RFE")`); any other class needs its
    connected plan (`prep_estimator_admit`)."""
    var kind = _explained_kind(model)
    if kind == "forecaster":
        raise Error(_forecaster_refusal(name, model))
    if kind == "label_propagation":
        return
    if kind == "rfe":
        prep_estimator_admit(_rfe_estimator(model), "RFE", context)
        return
    prep_estimator_admit(model, name, context)


def _member_only(model: PythonObject, context: PythonObject) -> Bool:
    """The class's connected plan is a member-only plan (python/mojolearn/_native_msel_more.py `connect_members`:
    a marked class with no method recipes, IsolationForest among them): its methods are its member's. Never
    raises."""
    try:
        var plan = prep_estimator_plan(model, context)
        if plan is None or py_get(plan, "member") is None:
            return False
        var word = py_get(plan, "member_only", PythonObject(False))
        var answer = Bool(py=word)
        return answer
    except:
        return False


def _meta_kind(model: PythonObject, context: PythonObject) -> String:
    """The connected meta schema of the model's class (bindings/trees_public_meta_door.mojo `meta_kind`, then
    bindings/trees_public_boost_door.mojo `boost_kind`), or "" for any other class. Never raises."""
    try:
        return meta_kind(model, context)
    except:
        pass
    try:
        return boost_kind(model, context)
    except:
        return String("")


def _explained(model: PythonObject, context: PythonObject) raises -> PythonObject:
    """The estimator the explainer evaluates (lane mf-shap3, 2026-11-23): `model` itself or, for a fitted
    GridSearchCV / RandomizedSearchCV, its `best_estimator_` (the search's predict and predict_proba are its best
    estimator's own, `family_msel_delegate`), with the search's words when it has none (refit=False or not
    fitted). Read at construction and again at every values call, as the search's own predict reads it."""
    var m = model
    var native = prep_native_context(context)
    var layouts = py_get(native, PythonObject("family_msel_search_layouts"))
    if layouts is None:
        return m
    for _ in range(8):  # driver-loop: over nested searches (model structure)
        if py_get(layouts, py_type(m)) is None:
            return m
        var fields = msel_search_fields(m, native)
        if not py_contains(fields, "best_estimator_"):
            raise Error("AttributeError: This " + String(py_type(m).__name__)
                        + " instance was initialized with refit=False or is not fitted.")
        m = fields["best_estimator_"]
    return m


def _method(model: PythonObject, context: PythonObject) raises -> String:
    """The explained method: a classifier's predict_proba, else predict. A hard VotingClassifier has no
    predict_proba (scikit-learn's `available_if`): its predict (lane mf-shap3). A MultiOutputClassifier is
    refused (`_MULTI_REFUSAL`). An RFE: its fitted estimator's (`RFE.predict` / `predict_proba` are that
    estimator's over the support columns); a label propagation model: predict_proba (lane mf-shap3b)."""
    var kind = _explained_kind(model)
    if kind == "rfe":
        return _method_of(_rfe_estimator(model), context)
    if kind == "label_propagation":
        return "predict_proba"
    return _method_of(model, context)


def _method_of(model: PythonObject, context: PythonObject) raises -> String:
    """`_method` of a class with a connected plan. A member-only plan (`_member_only`) serves its member's
    methods (a classifier's predict_proba, else predict: IsolationForest's predict, lane mf-shap3b)."""
    var member_only = _member_only(model, context)
    var kind = _class_value(model, "_estimator_type")
    if (kind is not None and py_type(kind) is _b().str and String(kind) == "classifier"):
        var meta = _meta_kind(model, context)
        if meta == "multioutput_classifier":
            raise Error(_MULTI_REFUSAL)
        var nk = trees_keys()
        var hard = meta == "voting_classifier" and String(meta_attr(model, nk[].k_voting, PythonObject(""))) == "hard"
        if not hard and (member_only or prep_estimator_has(model, "predict_proba", context)):
            return "predict_proba"
    if not member_only and not prep_estimator_has(model, "predict", context):
        raise Error("TypeError: the explained mojolearn estimator has no native predict")
    return "predict"


def _evaluate(model: PythonObject, method: String, x: PythonObject, n: Int,
              context: PythonObject) raises -> PythonObject:
    """The model on rows x (an (n, d) Array) as a float32 (n, k) Array."""
    # Lane fx-rows-rest (2026-11-18): predict through the plan's internal
    # float32 operation when it names one (prep_estimator_predict_internal, as
    # the stacking and voting doors): this door reads float32 words, and the
    # public predict of a linear model returned a float64 ndarray that was then
    # narrowed back through a zero-filled snapshot, per chunk. Same words (the
    # widening was exact). A plan without one runs its predict as before.
    var out = (prep_estimator_predict_internal(model, Python.tuple(x), context) if method == "predict"
               else prep_estimator_call(model, method, Python.tuple(x), context))
    classical_admit_value(out, context)
    var view = classical_view(out, context, "output", True)
    var rank = Int(py=view.ndim)
    if rank != 1 and rank != 2:
        raise Error("ValueError: the explained model output must be 1-D or 2-D")
    if Int(py=view.shape[0]) != n:
        raise Error("ValueError: the explained model returned a different row count")
    var k = Int(py=view.shape[1]) if rank == 2 else 1
    _ = view
    # Lane fb3-named (2026-11-17; kernel-shap +247 ms taxi, +1143 ms istella vs 0.8.55): the model output of
    # every chunk went through _snapshot, a zero-filled Array of the output's size plus a serial memcpy of the
    # whole output (and a second zero-filled Array plus a cast when it was not float32): nsamples * background
    # rows per explained row. The output of a compiled estimator call is a fresh owned Array nobody else holds,
    # so a C-order float32 one is read in place (0.8.55's as_f32_c borrowed it the same way) and any other
    # numeric dtype is cast once into an unfilled float32 Array. The reshaped Array shares the store, which
    # keeps the words alive; every caller also holds its result through the kernels that read it.
    var nk = trees_keys()
    if (py_type(out) is py_item(context, nk[].k_array_class) and out._base is None and out._pin is None
            and (rank == 1 or String(out.order) == "C")):
        var source_type = String(out.dtype)
        if source_type == "<f4":
            return trees_reshape(out, Python.tuple(n, k), context)
        var narrow = admitted_unfilled([n * k], "<f4", context)
        _ = _entry(py_item(context, nk[].k_trees_helpers), "cast_elements")
        trees_cast(trees_address(out), source_type, narrow.addr, "<f4", n * k, context)
        _ = out  # the source words are read through the whole cast
        return trees_reshape(narrow.keep, Python.tuple(n, k), context)
    # Any other output object (a view over another owner): the snapshot copy, with a fresh snapshot table
    # per output (a freed output's identity may be reused).
    var flat = _snapshot(out, "<f4", rank, "output", context, Python.dict())
    _ = out
    return trees_reshape(flat, Python.tuple(n, k), context)


def _shape(phi: PythonObject, n: Int, d: Int, k: Int, context: PythonObject) raises -> PythonObject:
    if n == 0:
        var empty = admitted_vector(0, "<f8", context)
        return trees_reshape(empty, Python.tuple(0, d) if k == 1 else Python.tuple(0, d, k), context)
    return trees_reshape(phi, Python.tuple(n, d) if k == 1 else Python.tuple(n, d, k), context)


def _device_forest(model: PythonObject, mode: String, permutation: Bool, table: PythonObject,
                   context: PythonObject) raises -> PythonObject:
    """MOJOLEARN_IDN_SHAP_DEVICE_MODEL: (arrays, k,
    n_trees, rf_input) when the model is a built-in flat forest whose
    predict is the strict increasing-tree device kernel; else None (the
    model runs through its native predict per chunk). rf_input: the random
    forest's predict flushes the compared feature, ExtraTrees' does not."""
    var bits = _switches(table)
    # The FAST tier has no device-model route (MOJOLEARN_KSHAP_FAST_DEVICE_MODEL deleted 2026-11-09).
    if mode == "fast" or (bits & _AGN_IDN_DEVICE_MODEL) == 0:
        return PythonObject(None)
    var m = model
    var outer = _fields(model)
    var inner = _get(outer, "_random_inner")
    if inner is not None:
        m = inner
    var cls = py_type(m)
    var nk = trees_keys()
    var keys = py_call(_b().list, py_item(context, nk[].k_trees_classes))
    var kind = String("")
    for i in range(len(keys)):  # driver-loop: connected exact tree classes
        if cls is keys[i]:
            kind = String(py_item(context, nk[].k_trees_classes)[keys[i]])
    var rf_input = 0
    if kind == "rf_classifier" or kind == "rf_regressor":
        rf_input = 1
    elif kind != "et_classifier" and kind != "et_regressor":
        return PythonObject(None)
    var fields = _fields(m)
    if not py_contains(fields, nk[].k__offsets):
        return PythonObject(None)
    try:
        var forest_mode = forest_effective_mode(fields, context)
        var native = trees_table(context, forest_mode, "_mojolearn_trees" if _family(kind) == "et" else "_mojolearn_rf")
        # identical only: the FAST tier left before this point (K1 deleted its route)
        if forest_mode != "identical" or not (_engine(fields, forest_mode) == "sequential" and _ordered_export(native)):
            return PythonObject(None)
        _ = _resident(m, fields, native, forest_mode, context)
    except:
        # The forest has no resident route here: its native predict serves.
        return PythonObject(None)
    var names = Python.tuple("_offsets", "_colid", "_quesval", "_left_child", "_leaves")
    var arrays = Python.list()
    for i in range(5):  # driver-loop: the five flat model arrays
        _ = py_call(arrays.append, fields[names[i]])
    return Python.tuple(py_call(_b().tuple, arrays), _int(py_item(fields, nk[].k__num_outputs), "output count", True),
                        _int(py_item(fields, nk[].k__n_trees), "tree count", True), rf_input)


def _forest_words(dev: PythonObject, bg: PythonObject, mut req: AgnRequest) raises:
    """UNPACK of MOJOLEARN_IDN_SHAP_DEVICE_MODEL: the flat forest's five array addresses and their words into the
    request (`x_trees_agn_model_load`'s, which the RUN loads once and releases); `dev` (the model's own arrays)
    and `bg` stay alive through the RUN with the explainer and the model."""
    if dev is None:
        return
    var arrays = dev[0]
    req.forest = List[Int]()
    for i in range(5):  # driver-loop: one address per model array
        req.forest.append(trees_address(arrays[i]))
    var words: List[Int] = [Int(py=dev[2]), Int(py=arrays[1].size), Int(py=dev[1]), Int(py=bg.shape[1]),
                            Int(py=bg.shape[0]), Int(py=dev[3])]
    req.forest_words = words^
    _ = arrays


def _construct(fields: PythonObject, args: PythonObject, permutation: Bool,
               context: PythonObject) raises:
    var model = args[0]
    var data = args[1]
    var random_state = args[2]
    var link = args[3]
    var requested = args[4]
    var name = _name(permutation)
    if not permutation:
        if py_type(link) is not _b().str or (String(link) != "identity" and String(link) != "logit"):
            raise Error("NotImplementedError: " + name + " supports the identity and logit links, by name")
    # Only connected mojolearn estimators run here; their predict is a
    # compiled operation. A Python callable or foreign model is refused.
    # A fitted search is explained through its best estimator (`_explained`).
    var explained = _explained(model, context)
    _admit_explained(explained, name, context)
    var seed = _seed(random_state, context)
    var model_fields = _fields(explained)
    if requested is None:
        requested = _get(model_fields, "numeric_mode")
    var nk = trees_keys()
    var mode = _tree_mode(requested if requested is not None else py_item(py_item(context, nk[].k_backend_fields), nk[].k__DEFAULT_MODE))
    var table = trees_table(context, mode, "_mojolearn_x_trees")
    var method = _method(explained, context)
    var bg = _snapshot(data, "<f4", 2, "data", context, Python.dict())
    var nb = Int(py=bg.shape[0])
    var d = Int(py=bg.shape[1])
    if nb < 1:
        raise Error("ValueError: " + name + " requires at least one background row")
    var logit = not permutation and String(link) == "logit"
    var glue_key = _xk(table)
    var glue = xt_ref(glue_key)
    # The model on the background through the same native handle the values calls use (lane mf-shap3,
    # 2026-11-23: it went through the model's door, `_evaluate`, before); a class with no member entry keeps
    # its door. RUN (xtrees/agnostic_values_run.mojo): the background mean, then (shap KernelExplainer:
    # expected_value = link(fnull)) the link on each output, through the glue entry of the x_trees image this
    # tier connected.
    var k = 0
    var ev = PythonObject(None)
    var em = _Member()
    try:
        var s = _State(explained, method, bg, table, mode, seed, nb, d, -1, PythonObject(None))
        _open(em, s, bg, nb, context)
        if em.meta or em.native or em.lp:
            k = em.k
            ev = admitted_vector(k, "<f8", context)
            var ev_at = trees_address(ev)
            var bg_at = trees_address(bg)
            if em.lp:
                var lp = AgnLabelProp(em.lp_drivers[0], em.lp_requests[0], em.lp_name, bg_at, nb, d)
                _expected_with(lp, True, glue, nb, k, logit, ev_at)
                _ = lp^
            elif em.meta:
                var meta = AgnMeta(em.meta_plan, bg_at, nb, d, False, glue_key.vendor == 0, _xr(em.meta_xt))
                _expected_with(meta, em.meta_gil_free, glue, nb, k, logit, ev_at)
                _ = meta^
            else:
                var handle = _handle_of(em, bg_at, nb, d, k, glue_key.vendor == 0, glue)
                _expected_with(handle, em.gil_free, glue, nb, k, logit, ev_at)
                _ = handle^
        else:
            var out = _evaluate(explained, method, bg, nb, context)
            k = Int(py=out.shape[1])
            ev = admitted_vector(k, "<f8", context)
            agn_expected_run(glue, trees_address(out), nb, k, logit, trees_address(ev))
            _ = out
        _ = s^
    finally:
        _member_release(em)
    _ = bg  # the background is read through the RUN above
    var expected = PythonObject(Float64(MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=trees_address(ev))[0])) if k == 1 else ev
    var dev = _device_forest(explained, mode, permutation, table, context)
    if dev is not None:
        var fitted = _get(model_fields, "n_features_in_")
        if Int(py=dev[1]) != k or fitted is None or py_type(fitted) is not _b().int or Int(py=fitted) != d:
            dev = PythonObject(None)
    var changes = Python.dict()
    var keys = Python.tuple("model", "data", "random_state", "numeric_mode", "_bg", "_fnull", "_method",
                            "_dev", "n_outputs_", "expected_value", "n_features_in_")
    var values = Python.tuple(model, data, random_state, requested, bg, ev,
                              PythonObject(method), dev, PythonObject(k), expected, PythonObject(d))
    for i in range(len(keys)):  # driver-loop: fixed explainer state fields
        py_setitem(changes, keys[i], values[i])
    if not permutation:
        py_setitem(changes, nk[].k_link, link)
    _replace(fields, changes, context)


struct _State(Copyable, Movable):
    var model: PythonObject
    var method: String
    var bg: PythonObject
    var table: PythonObject
    var mode: String
    var seed: Int
    var nb: Int
    var d: Int
    var k: Int
    var dev: PythonObject

    def __init__(out self, model: PythonObject, method: String, bg: PythonObject, table: PythonObject,
                 mode: String, seed: Int, nb: Int, d: Int, k: Int, dev: PythonObject):
        self.model = model
        self.method = method
        self.bg = bg
        self.table = table
        self.mode = mode
        self.seed = seed
        self.nb = nb
        self.d = d
        self.k = k
        self.dev = dev


def _state(fields: PythonObject, name: String, context: PythonObject) raises -> _State:
    var bg = _get(fields, "_bg")
    var method = _get(fields, "_method")
    if bg is None or method is None:
        raise Error("RuntimeError: " + name + " has not been initialized by the native owner")
    var requested = _get(fields, "numeric_mode")
    var nk = trees_keys()
    var mode = _tree_mode(requested if requested is not None else py_item(py_item(context, nk[].k_backend_fields), nk[].k__DEFAULT_MODE))
    var table = trees_table(context, mode, "_mojolearn_x_trees")
    # A fitted search: its best estimator now (`_explained`), as the search's own predict reads it.
    var model = _explained(_get(fields, "model"), context)
    _admit_explained(model, name, context)
    return _State(model, String(method), bg, table, mode, _seed(_get(fields, "random_state"), context),
                  Int(py=bg.shape[0]), Int(py=bg.shape[1]), _int(_get(fields, "n_outputs_"), "output count", True),
                  _get(fields, "_dev"))


def _rows(x_source: PythonObject, s: _State, name: String, context: PythonObject) raises -> PythonObject:
    var x = _snapshot(x_source, "<f4", 2, "X", context, Python.dict())
    if Int(py=x.shape[1]) != s.d:
        raise Error("ValueError: X has " + String(Int(py=x.shape[1])) + " features, data has " + String(s.d))
    return x


# ---------------------------------------------------------------- the explained model as native members
struct _Member(Movable):
    """UNPACK of the explained estimator for one values call (or the construction's background call): its
    NATIVE MEMBER (core/member_abi.mojo) or, for a meta-estimator, its members and the typed plan of its public
    operation. `native`: `held` is the fitted member state the estimator keeps (bindings/member_held_door.mojo:
    imported at its first use, a hit on every later call until the estimator changes) and `operation` the member
    operation of the explained method; the RUN calls it through `AgnMember`. `meta` (lane mf-shap3, 2026-11-23;
    the soft and regressor votes since lane mf-shap2): `meta_plan` (`AgnMetaPlan`, its members' plans by address
    into `meta_opened`, each opened on its held state and closed by `_member_release`), the owners of its tables
    (`meta_keep`) and its own x_trees table (`meta_xt`); the RUN folds it through `AgnMeta`. Neither: the class
    (or a member's class) has no member entry that serves the call, and the RUN evaluates the model through
    `_DoorModel`. `k`: the output columns the open found. `refusal`: words an open found the call must be refused
    with (raised by `_open`). Released on every exit of the values call (`_member_release`)."""
    var native: Bool
    var held: MemberHeld
    var operation: Int
    var k: Int
    # THE GIL RULE (core/member_abi.mojo): the held state's kind has `MEMBER_GIL_FREE` or `MEMBER_GIL_DOOR`: the
    # RUN runs with the GIL released by this door (one release per values call; it was one per chunk). False: it
    # runs with the GIL held. `parallel`: `MEMBER_GIL_FREE` itself, the chunks may run on parallel workers.
    var gil_free: Bool
    var parallel: Bool
    # The member writes the explained operation's output as float64 (a classifier's predict_proba on most
    # families; on the label route, its codes as int64): the RUN narrows each chunk's words.
    var wide: Bool
    # A classifier explained through predict (lane mf-shap2, 2026-11-23): `labels` is its `classes_` narrowed to
    # a float32 Array once here (None: not this route), `labels_at` its address read here with the GIL held,
    # `n_labels` its length. The member writes codes and the RUN maps them through the glue gather. A hard
    # vote's labels are the same three words (lane mf-shap3).
    var labels: PythonObject
    var labels_at: Int
    var n_labels: Int
    var meta: Bool
    var meta_plan: AgnMetaPlan
    var meta_opened: List[MemberHeldCall]
    var meta_keep: PythonObject
    var meta_xt: PythonObject
    # Every opened member: tests its rows (no scan of X and the background), may run with the GIL released
    # (`MEMBER_GIL_FREE` or `MEMBER_GIL_DOOR`), is `MEMBER_GIL_FREE` (the chunks may run on parallel workers when
    # the meta's glue is the CPU column, `meta_parallel`).
    var meta_tests: Bool
    var meta_gil_free: Bool
    var meta_free: Bool
    var meta_parallel: Bool
    # A model that is not a classifier whose member writes integer values (lane mf-shap3b: an outlier detector's
    # predict): widened to float32 by the RUN.
    var ints: Bool
    # An RFE (lane mf-shap3b): its support column numbers (int32, the member reads these columns of each row);
    # empty for every other model.
    var select: List[Int32]
    # A label propagation model (lane mf-shap3b): its typed query (`lp_requests[0]` over `lp_drivers[0]`), its
    # class name for the refusal words, and the owners of the addresses the request holds (`lp_keep`: X_,
    # label_distributions_, the connection's provider, the binding). The RUN runs with the GIL released.
    var lp: Bool
    var lp_drivers: List[XnDriver]
    var lp_requests: List[LpQueryRequest]
    var lp_name: String
    var lp_keep: PythonObject
    var refusal: String

    def __init__(out self):
        self.gil_free = False
        self.parallel = False
        self.wide = False
        self.native = False
        self.held = MemberHeld()
        self.operation = MEMBER_OP_PREDICT
        self.k = 0
        self.labels = PythonObject(None)
        self.labels_at = 0
        self.n_labels = 0
        self.meta = False
        self.meta_plan = AgnMetaPlan(AGN_META_SOFT)
        self.meta_opened = List[MemberHeldCall]()
        self.meta_keep = PythonObject(None)
        self.meta_xt = PythonObject(None)
        self.meta_tests = True
        self.meta_gil_free = True
        self.meta_free = True
        self.meta_parallel = False
        self.ints = False
        self.select = List[Int32]()
        self.lp = False
        self.lp_drivers = List[XnDriver]()
        self.lp_requests = List[LpQueryRequest]()
        self.lp_name = String("")
        self.lp_keep = PythonObject(None)
        self.refusal = String("")


def _member_release(mut em: _Member):
    """End of the values call (or of the member's use in it): a state the
    estimator keeps stays with it, a loose one is freed. Never raises."""
    em.native = False
    member_held_release(em.held)
    em.labels = PythonObject(None)
    em.labels_at = 0
    em.ints = False
    em.select = List[Int32]()
    em.lp = False
    em.lp_drivers = List[XnDriver]()
    em.lp_requests = List[LpQueryRequest]()
    em.lp_keep = PythonObject(None)
    _meta_release(em)


def _meta_release(mut em: _Member):
    """Close every member a meta-estimator's open opened (each kept state stays with its estimator). Never
    raises."""
    em.meta = False
    for i in range(len(em.meta_opened)):  # driver-loop: over the opened members
        member_held_close(em.meta_opened[i])
    em.meta_opened = List[MemberHeldCall]()
    em.meta_plan = AgnMetaPlan(AGN_META_SOFT)
    em.meta_keep = PythonObject(None)
    em.meta_xt = PythonObject(None)
    em.meta_tests = True
    em.meta_gil_free = True
    em.meta_free = True
    em.meta_parallel = False


def _label_snapshot(fields: PythonObject, classes: Int, context: PythonObject, mut em: _Member) raises -> PythonObject:
    """An estimator's `classes_` as a float32 Array (`_label_table`'s read), or None when there is none or its
    length is not `classes` (when known). Non-numeric labels: `em.refusal` (`_LABELS_REFUSAL`), None."""
    var given = _get(fields, "classes_")
    if given is None:
        return PythonObject(None)
    var table = PythonObject(None)
    try:
        table = _snapshot(given, "<f4", 1, "classes_", context, Python.dict())
    except:
        em.refusal = String(_LABELS_REFUSAL)
        return PythonObject(None)
    var count = Int(py=table.size)
    if count < 1 or (classes > 0 and count != classes):
        return PythonObject(None)
    return table


def _label_table(fields: PythonObject, classes: Int, context: PythonObject, mut em: _Member) raises -> Bool:
    """UNPACK of the label route: the estimator's `classes_` as a float32 Array (the cast the door's `_evaluate`
    made of its public predict's labels, element by element), once per values call. False (not served: the
    door then answers, or raises its own words) when there is no `classes_` or its length is not the member's
    class count. Non-numeric labels: `em.refusal` (`_LABELS_REFUSAL`)."""
    var table = _label_snapshot(fields, classes, context, em)
    if table is None:
        return False
    em.labels_at = trees_address(table)
    em.n_labels = Int(py=table.size)
    em.labels = table
    return True


def _rows_finite(s: _State, x: PythonObject, n: Int) raises -> Bool:
    """The finite verdict of the explained rows and the background (every synthetic chunk mixes their cells),
    one host scan each, for a model with a member that does not test its rows."""
    var finite = trees_finite_f32(trees_address(x), n * s.d) and trees_finite_f32(trees_address(s.bg), s.nb * s.d)
    _ = x  # keep the Python objects alive through the scans above
    _ = s.bg
    return finite


def _member_open(mut em: _Member, s: _State, model: PythonObject, x: PythonObject, n: Int,
                 context: PythonObject):
    """UNPACK, once per values call (the RUN then calls the member through `AgnMember`, the native model
    handle): `model`'s held member state (the explained model, or a Bagging's one batched forest) when its
    family serves the explained method (predict of a regressor kind, predict_proba of a classifier, and
    predict of a classifier, whose member writes int32 or int64 codes that the RUN maps to its `classes_` read
    here once as float32: `_label_table`). Anything else (no mark, an admission or import the family refuses,
    an element type other than float32 or float64, a classifier member that does not say its element type)
    leaves `em` closed: the model then goes to `_meta_open`, else to its door. Never raises; a refusal is left
    in `em.refusal`: an output width other than the one construction recorded (`s.k`, -1 at construction),
    non-numeric `classes_`, non-finite rows for a member that does not test them.

    WHO REFUSES A NON-FINITE ROW. A member whose kind tests its rows (MEMBER_CAN_TEST_ROWS) refuses a non-finite
    chunk in its own kernel pass, and the RUN raises its words: no host scan here. Any other member reads
    admitted words, and every chunk is a mix of cells of X and of the background, so both are scanned once per
    values call (`_rows_finite`); not finite = `_NONFINITE`."""
    # TOMBSTONE: MOJOLEARN_MEMBER_SET_OFF (plumbing) deleted 2026-11-23 by lane/ab-prep-classical; code recoverable at 4d76a9e16e.
    try:
        var plan = prep_estimator_plan(model, context)
        if plan is None or py_get(plan, "member") is None:
            return
        var proba = s.method == "predict_proba"
        # A classifier explained through predict: its member writes CODES, mapped to the labels by the RUN.
        var labelled = not proba and Bool(py=plan["classifier"])
        var b = _b()
        var member_context = plan["context"]
        var fields = py_call(plan["fields_get"], model, py_type(model))
        var operation = MEMBER_OP_PREDICT_PROBA if proba else MEMBER_OP_PREDICT
        var held = member_held(model, fields, plan, member_context, b,
                               MEMBER_CAN_PREDICT_PROBA if proba else MEMBER_CAN_PREDICT)
        if not held.published:
            return
        var cols = -1
        var dtype = MEMBER_DTYPE_UNSAID
        var classes = 0
        var tests = False
        var gil_free = False
        var parallel = False
        try:
            var probe = member_call_of(held.plan, held.state)
            cols = member_outputs_of(held.plan, probe, operation)
            dtype = probe.out_dtype
            classes = probe.classes
            if dtype == MEMBER_DTYPE_UNSAID and not proba and not labelled:
                dtype = MEMBER_F32   # a regressor writes float32 (every published regressor documents it)
            # Kind-exact, asked of the held state; a family whose union mask
            # lacks the bit is not asked.
            # THE GIL RULE: the same answer says whether a chunk's call may
            # run released.
            if held.plan.member.can(MEMBER_CAN_TEST_ROWS | MEMBER_GIL_FREE | MEMBER_GIL_DOOR):
                var kind_mask = member_call(held.plan.member, MEMBER_OP_CAPABILITIES, probe)
                tests = (kind_mask & MEMBER_CAN_TEST_ROWS) != 0
                # one call per chunk on this thread either way (MEMBER_GIL_DOOR:
                # released as the family's own door releases)
                gil_free = (kind_mask & (MEMBER_GIL_FREE | MEMBER_GIL_DOOR)) != 0
                # MEMBER_GIL_FREE alone: safe on parallel workers (the RUN's chunks on the CPU column).
                parallel = (kind_mask & MEMBER_GIL_FREE) != 0
        except:
            cols = -1
        # float64 too (lane mf-shaprfe, 2026-11-23): LogisticRegression, SVC, the DART, GP and (on some routes)
        # GBDT classifiers and the naive Bayes / discriminant members on a float64-proba route write their
        # predict_proba as float64.
        var served = cols >= 1 and (dtype == MEMBER_F32 or dtype == MEMBER_F64)
        # Integer values of a model that is not a classifier (lane mf-shap3b, 2026-11-23): IsolationForest's
        # predict writes int32 1 / -1, the other outlier detectors' int64; the RUN widens them to float32.
        var ints = not labelled and not proba and cols >= 1 and (dtype == MEMBER_I32 or dtype == MEMBER_I64)
        if ints:
            served = True
        if labelled:
            # codes, one column, into a table of the member's classes (int64 codes narrowed in the RUN).
            served = (cols == 1 and (dtype == MEMBER_I32 or dtype == MEMBER_I64)
                      and _label_table(fields, classes, context, em))
        if served and s.k >= 0 and cols != s.k:
            em.refusal = String(_WIDTH_CHANGED)
            served = False
        if served and not tests and not _rows_finite(s, x, n):
            em.refusal = String(_NONFINITE)
            served = False
        if not served:
            member_held_release(held)
            em.labels = PythonObject(None)
            em.labels_at = 0
            return
        em.held = held^
        em.operation = operation
        em.k = cols
        em.gil_free = gil_free
        em.parallel = parallel
        em.wide = dtype == MEMBER_F64 or dtype == MEMBER_I64
        em.ints = ints
        em.native = True
    except:
        return


def _meta_take(mut em: _Member, var opened: MemberHeldCall, class_cols: Int, weight: Float64,
               last: Bool) raises -> Bool:
    """One opened member of the explained meta-estimator into `em.meta_plan` (its plan's address is set once
    every member is open: `_meta_plans`), its GIL words folded into `em`. False (closed) when it is not served.
    `last`: the final estimator of a Stacking (its rows are the meta features, not X)."""
    if not opened.served:
        member_held_close(opened)
        return False
    if not last:
        em.meta_tests = em.meta_tests and opened.tests
    else:
        em.meta_plan.final_tests = opened.tests
    em.meta_gil_free = em.meta_gil_free and opened.gil_free
    var free = False
    if opened.held.plan.member.can(MEMBER_GIL_FREE):
        var probe = member_call_of(opened.held.plan, opened.held.state)
        free = (member_call(opened.held.plan.member, MEMBER_OP_CAPABILITIES, probe) & MEMBER_GIL_FREE) != 0
    em.meta_free = em.meta_free and free
    var vm = VoteMember(len(em.meta_opened), opened.held.state, opened.operation, opened.cols, opened.dtype,
                        class_cols, weight)
    if last:
        em.meta_plan.final_est.append(vm)
    else:
        em.meta_plan.members.append(vm)
    em.meta_opened.append(opened^)
    return True


def _meta_member(mut em: _Member, e: PythonObject, operation: Int, classifier: Bool, width: Int,
                 class_cols: Int, weight: Float64, last: Bool, context: PythonObject) raises -> Bool:
    """`_meta_take` of member `e` opened by the meta door's X-FREE MEMBER UNPACK
    (bindings/trees_public_meta_door.mojo `meta_member_open`): every family a public predict serves, the trees
    too (the explainers stage no device copy of X: every member is called on the synthetic rows)."""
    return _meta_take(em, meta_member_open(e, operation, classifier, context, width, False), class_cols, weight,
                      last)


def _meta_cols(mut em: _Member, e: PythonObject, context: PythonObject) raises -> Int:
    """A member's class codes into the ensemble (`_sub_cols`, int32), kept alive in `em.meta_keep`: its
    address."""
    var cols = _sub_cols(e, context)
    _ = py_call(em.meta_keep.append, cols)
    return trees_address(cols)


def _vote_plan(mut em: _Member, s: _State, kind: String, context: PythonObject) raises -> Bool:
    """VotingClassifier soft (predict_proba: `vote_soft_run`) and hard (predict: `vote_codes_run`, the codes
    mapped to its `classes_` as float32), VotingRegressor (predict: `vote_regress_run`): every member with its
    weight (`_voting_weights`)."""
    var nk = trees_keys()
    var model = s.model
    var classifier = kind == "voting_classifier"
    var soft = classifier and String(meta_attr(model, nk[].k_voting, PythonObject(""))) == "soft"
    if (soft and s.method != "predict_proba") or (not soft and s.method != "predict"):
        return False
    var members = meta_attr(model, nk[].k_estimators_, Python.list())
    var m = len(members)
    if m < 1:
        return False
    var weights = _voting_weights(model)
    var classes = Int(py=meta_attr(model, nk[].k_n_classes_, PythonObject(0))) if classifier else 0
    em.meta_plan = AgnMetaPlan(AGN_META_SOFT if soft else (AGN_META_HARD if classifier else AGN_META_REGRESS))
    for i in range(m):  # driver-loop: over members (admission, once each)
        var e = members[i]
        var weight = Float64(py=weights[i])
        var ok = False
        if soft:
            var class_cols = _meta_cols(em, e, context)
            ok = _meta_member(em, e, MEMBER_OP_PREDICT_PROBA, True, 0, class_cols, weight, False, context)
        elif classifier:
            ok = _meta_member(em, e, MEMBER_OP_PREDICT, True, 0, 0, weight, False, context)
        else:
            ok = _meta_member(em, e, MEMBER_OP_PREDICT, False, 1, 0, weight, False, context)
        if not ok:
            return False
    em.meta_plan.classes = classes
    em.meta_plan.k = classes if soft else 1
    if classifier and not soft:
        if not _label_table(_fields(model), classes, context, em):
            return False
        em.meta_plan.labels = em.labels_at
        em.meta_plan.n_labels = em.n_labels
    return True


def _bag_plan(mut em: _Member, s: _State, classifier: Bool, context: PythonObject) raises -> Bool:
    """Bagging*: every member (`_estimators`) on its feature subset (`bag_member_columns`: the rows themselves,
    or its int32 columns gathered in the RUN), `bag_predict_run`. The batched route is `_batched_member`'s."""
    if (classifier and s.method != "predict_proba") or (not classifier and s.method != "predict"):
        return False
    var fields = py_fields(s.model)
    var members = fields["_estimators"]
    var features = fields["estimators_features_"]
    _ = py_call(em.meta_keep.append, features)
    var m = len(members)
    if m < 1:
        return False
    var classes = len(fields["classes_"]) if classifier else 0
    em.meta_plan = AgnMetaPlan(AGN_META_BAG)
    var operation = MEMBER_OP_PREDICT_PROBA if classifier else MEMBER_OP_PREDICT
    for i in range(m):  # driver-loop: over the member estimators (admission, once each)
        var e = members[i]
        var sub = PythonObject(None)
        var class_cols = 0
        if classifier:
            sub = _sub_cols(e, context)
            _ = py_call(em.meta_keep.append, sub)
            class_cols = trees_address(sub)
        if not _meta_take(em, bag_member_open(e, sub, operation, classifier, context), class_cols, 1.0, False):
            return False
        var cols = features[i]
        em.meta_plan.features.append(bag_member_columns(cols, s.d))
        em.meta_plan.widths.append(len(cols))
    em.meta_plan.classifier = classifier
    em.meta_plan.classes = classes
    em.meta_plan.k = classes if classifier else 1
    return True


def _plan_classifier(e: PythonObject, context: PythonObject) raises -> Bool:
    """`e`'s connected plan is a classifier's."""
    var plan = prep_estimator_plan(e, context)
    if plan is None:
        return False
    var word = plan["classifier"]
    return Bool(py=word)


def _stack_label_member(mut em: _Member, e: PythonObject, context: PythonObject) raises -> Bool:
    """A CLASSIFIER member a Stacking asks for predict (lane mf-shap3b, 2026-11-23): its held state's predict (the
    member writes int32 or int64 class codes; a member's code c is its c-th class), and its `classes_` as a
    float32 table kept in `em.meta_keep` (the RUN maps the codes to these labels: the column
    `_StackingBase._out` places for that member). False (closed) when it is not served or its labels do not
    match its class count; non-numeric labels leave `em.refusal`."""
    var opened = member_held_open(e, MEMBER_OP_PREDICT, True, 1, False, context, _b())
    if not opened.served or (opened.dtype != MEMBER_I32 and opened.dtype != MEMBER_I64):
        member_held_close(opened)
        return False
    var probe = member_call_of(opened.held.plan, opened.held.state)
    _ = member_outputs_of(opened.held.plan, probe, MEMBER_OP_PREDICT)
    var table = _label_snapshot(_fields(e), probe.classes, context, em)
    if table is None:
        member_held_close(opened)
        return False
    _ = py_call(em.meta_keep.append, table)
    em.meta_plan.stack_labels.append(trees_address(table))
    em.meta_plan.stack_n_labels.append(Int(py=table.size))
    return _meta_take(em, opened^, 0, 1.0, False)


def _stack_plan(mut em: _Member, s: _State, classifier: Bool, context: PythonObject) raises -> Bool:
    """Stacking*: the members of its meta features (`stacking_layout`: each asked as `stack_method_` says),
    then its final estimator (`final_estimator_`) over them: a classifier's predict_proba at its class
    columns, a regressor's predict."""
    if (classifier and s.method != "predict_proba") or (not classifier and s.method != "predict"):
        return False
    var nk = trees_keys()
    var layout = stacking_layout(s.model, s.d, context)
    var members = layout.members
    if len(members) < 1:
        return False
    em.meta_plan = AgnMetaPlan(AGN_META_STACK)
    for i in range(len(members)):  # driver-loop: over members (admission, once each)
        var kind = layout.kinds[i]
        if kind == STACK_PREDICT and _plan_classifier(members[i], context):
            if not _stack_label_member(em, members[i], context):
                return False
            continue
        em.meta_plan.stack_labels.append(0)
        em.meta_plan.stack_n_labels.append(0)
        var operation = MEMBER_OP_PREDICT if kind == STACK_PREDICT else (
            MEMBER_OP_PREDICT_PROBA if kind == STACK_PROBA else MEMBER_OP_DECISION)
        if not _meta_member(em, members[i], operation, kind != STACK_PREDICT, 0, 0, 1.0, False, context):
            return False
    var final_est = meta_attr(s.model, nk[].k_final_estimator_, PythonObject(None))
    if final_est is None:
        return False
    var ok = False
    if classifier:
        var class_cols = _meta_cols(em, final_est, context)
        ok = _meta_member(em, final_est, MEMBER_OP_PREDICT_PROBA, True, 0, class_cols, 1.0, True, context)
    else:
        ok = _meta_member(em, final_est, MEMBER_OP_PREDICT, False, 1, 0, 1.0, True, context)
    if not ok:
        return False
    em.meta_plan.stack_kinds = layout.kinds.copy()
    em.meta_plan.classes = layout.k
    em.meta_plan.binary = layout.binary
    em.meta_plan.passthrough = layout.passthrough
    em.meta_plan.meta_width = layout.width
    em.meta_plan.classifier = classifier
    em.meta_plan.k = layout.k if classifier else 1
    return True


def _ovr_plan(mut em: _Member, s: _State, context: PythonObject) raises -> Bool:
    """OneVsRestClassifier.predict_proba: two classes, its first member's probability column of code 1; more,
    one member per class (`ovr_member_words`: by predict_proba, the positive column) folded by
    `ovr_assemble_run`."""
    if s.method != "predict_proba":
        return False
    var nk = trees_keys()
    var classes = len(meta_attr(s.model, nk[].k_classes_, Python.list()))
    var members = meta_attr(s.model, nk[].k_estimators_, Python.list())
    if classes < 2 or len(members) < 1 or (classes > 2 and len(members) != classes):
        return False
    em.meta_plan = AgnMetaPlan(AGN_META_OVR)
    var count = 1 if classes == 2 else len(members)
    for j in range(count):  # driver-loop: over members (admission, once each)
        var words = ovr_member_words(members[j], False, context)
        em.meta_plan.at.append(words[1])
        if not _meta_member(em, members[j], MEMBER_OP_PREDICT_PROBA, True, 0, 0, 1.0, False, context):
            return False
    em.meta_plan.classes = classes
    em.meta_plan.k = classes
    return True


def _cal_plan(mut em: _Member, s: _State, context: PythonObject) raises -> Bool:
    """CalibratedClassifierCV.predict_proba (two classes or more): each (member, calibrators) pair's member
    scores (decision_function when it has one, else predict_proba) and its words (`calibrated_pair_codes`,
    `calibrated_pair_words`), `calibrated_proba_run`."""
    if s.method != "predict_proba":
        return False
    var nk = trees_keys()
    var classes = len(meta_attr(s.model, nk[].k_classes_, Python.list()))
    var pairs = meta_attr(s.model, nk[].k_calibrated_classifiers_, Python.list())
    if classes < 2 or len(pairs) < 1:
        return False
    em.meta_plan = AgnMetaPlan(AGN_META_CALIBRATED)
    var cal_pairs = List[CalPair]()
    var cals = List[CalFit]()
    var keep = em.meta_keep
    for i in range(len(pairs)):  # driver-loop: over the (member, calibrators) pairs (admission, once each)
        var e = pairs[i][0]
        var decision = prep_estimator_has(e, "decision_function", context)
        var codes = calibrated_pair_codes(e, classes, keep, context)
        var operation = MEMBER_OP_DECISION if decision else MEMBER_OP_PREDICT_PROBA
        if not _meta_member(em, e, operation, True, 0, codes[0], 1.0, False, context):
            return False
        var cols = em.meta_plan.members[len(em.meta_plan.members) - 1].cols
        calibrated_pair_words(e, pairs[i][1], classes, cols, codes[0], codes[1], keep, cal_pairs, cals, context)
    em.meta_plan.pairs = cal_pairs^
    em.meta_plan.cals = cals^
    em.meta_plan.classes = classes
    em.meta_plan.k = classes
    return True


def _ada_plan(mut em: _Member, s: _State, classifier: Bool, context: PythonObject) raises -> Bool:
    """AdaBoostClassifier.predict_proba (each member's int32 codes, `ada_classifier_run`) and
    AdaBoostRegressor.predict (each member's float32 predictions, the weighted median), the weights read once
    as float64 (`_ada_weights`)."""
    if (classifier and s.method != "predict_proba") or (not classifier and s.method != "predict"):
        return False
    var fields = py_fields(s.model)
    var members = fields["estimators_"]
    var alphas = _ada_weights(fields["estimator_weights_"], context)
    var m = len(members)
    if m < 1 or len(alphas) < m:
        return False
    var classes = len(fields["classes_"]) if classifier else 0
    if classifier and classes < 2:
        return False
    em.meta_plan = AgnMetaPlan(AGN_META_ADA)
    for j in range(m):  # driver-loop: over the member estimators (admission, once each)
        if not _meta_member(em, members[j], MEMBER_OP_PREDICT, classifier, 0 if classifier else 1, 0, 1.0, False,
                            context):
            return False
    em.meta_plan.alphas = alphas^
    em.meta_plan.classifier = classifier
    em.meta_plan.classes = classes
    em.meta_plan.k = classes if classifier else 1
    return True


def _multi_plan(mut em: _Member, s: _State, context: PythonObject) raises -> Bool:
    """MultiOutputRegressor.predict: its one multi-output ridge (`_multi_ridge`: `rm_predict`, float32, as its
    public predict), else every member's float32 prediction at its column (`vote_transform_run`)."""
    if s.method != "predict":
        return False
    var nk = trees_keys()
    var members = meta_attr(s.model, nk[].k_estimators_, Python.list())
    var m = len(members)
    if m < 1:
        return False
    var mr = meta_attr(s.model, nk[].k__multi_ridge, PythonObject(None))
    if mr is not None:
        em.meta_plan = AgnMetaPlan(AGN_META_RIDGE)
        em.meta_plan.ridge.append(multioutput_ridge_ref(members, context))
        _ = py_call(em.meta_keep.append, mr)
        em.meta_plan.ridge_coef = trees_address(mr[0])
        em.meta_plan.ridge_icpt = trees_address(mr[1])
        em.meta_plan.k = Int(py=mr[2])
        # The ridge kernel reads admitted words: X and the background are proven finite; its call is released
        # as the public predict releases it; no parallel-worker trace, so the chunks run in order (each call
        # splits its own rows).
        em.meta_tests = False
        em.meta_free = False
        return True
    em.meta_plan = AgnMetaPlan(AGN_META_COLUMNS)
    for j in range(m):  # driver-loop: over members (outputs; admission, once each)
        if not _meta_member(em, members[j], MEMBER_OP_PREDICT, False, 1, 0, 1.0, False, context):
            return False
    em.meta_plan.k = m
    return True


def _meta_plans(mut em: _Member):
    """The members' plans by address into `em.meta_opened`, now that it is final: the RUN reads them in place."""
    for i in range(len(em.meta_plan.members)):  # driver-loop: over members
        var j = em.meta_plan.members[i].plan
        var at = Int(UnsafePointer(to=em.meta_opened[j].held.plan))
        em.meta_plan.members[i].plan = at
    for i in range(len(em.meta_plan.final_est)):  # driver-loop: the final estimator
        var j = em.meta_plan.final_est[i].plan
        var at = Int(UnsafePointer(to=em.meta_opened[j].held.plan))
        em.meta_plan.final_est[i].plan = at


def _meta_open(mut em: _Member, s: _State, x: PythonObject, n: Int, context: PythonObject):
    """UNPACK of a meta-estimator as the model, once per values call (the RUN then evaluates it through
    `AgnMeta`, by the RUN its public operation runs; lane mf-shap3, 2026-11-23, the votes since lane mf-shap2):
    Voting soft, hard and regressor, Bagging, Stacking, OneVsRest, CalibratedClassifierCV, AdaBoost,
    MultiOutputRegressor. Every member is opened on its held native state through the meta doors' X-free member
    UNPACK (a tree, forest or extra-trees member too: the public predicts stage one device copy of X for those,
    the synthetic rows have none), its tables read here once. A member any family does not serve (no member
    entry: a class without a mark, a meta-estimator as a member) leaves `em.meta` False and the model's door
    answers. Never raises; `em.refusal` as `_member_open`. The rows are scanned once when a member does not
    test them (every chunk mixes cells of X and of the background)."""
    # TOMBSTONE: MOJOLEARN_MEMBER_SET_OFF (plumbing) deleted 2026-11-23 by lane/ab-prep-classical; code recoverable at 4d76a9e16e.
    try:
        var kind = _meta_kind(s.model, context)
        if kind == "":
            return
        em.meta_keep = Python.list()
        var ok = False
        if kind == "voting_classifier" or kind == "voting_regressor":
            ok = _vote_plan(em, s, kind, context)
        elif kind == "bagging_classifier" or kind == "bagging_regressor":
            ok = _bag_plan(em, s, kind == "bagging_classifier", context)
        elif kind == "stacking_classifier" or kind == "stacking_regressor":
            ok = _stack_plan(em, s, kind == "stacking_classifier", context)
        elif kind == "ovr":
            ok = _ovr_plan(em, s, context)
        elif kind == "calibrated":
            ok = _cal_plan(em, s, context)
        elif kind == "ada_classifier" or kind == "ada_regressor":
            ok = _ada_plan(em, s, kind == "ada_classifier", context)
        elif kind == "multioutput_regressor":
            ok = _multi_plan(em, s, context)
        if ok and s.k >= 0 and em.meta_plan.k != s.k:
            em.refusal = String(_WIDTH_CHANGED)
            ok = False
        if ok and not em.meta_tests and not _rows_finite(s, x, n):
            em.refusal = String(_NONFINITE)
            ok = False
        if not ok:
            _meta_release(em)
            return
        _meta_plans(em)
        em.meta_xt = _xt(py_fields(s.model), context)
        em.meta_parallel = em.meta_free and _xk(em.meta_xt).vendor == 0
        em.k = em.meta_plan.k
        em.meta = True
    except:
        _meta_release(em)


def _batched_member(model: PythonObject, context: PythonObject) -> PythonObject:
    """A Bagging estimator fitted on the batched route (`_batched`): its one forest, whose predict_proba (or
    predict) is the ensemble's (float32 widened to float64: narrowed back to the same words); else None. Never
    raises."""
    try:
        var kind = _meta_kind(model, context)
        if kind != "bagging_classifier" and kind != "bagging_regressor":
            return PythonObject(None)
        var batched = py_get(py_fields(model), "_batched")
        if batched is None:
            return PythonObject(None)
        # Named before the return (AGENTS.md part 6: no subscript in tail position beside a borrowed argument).
        var inner = batched[0]
        return inner
    except:
        return PythonObject(None)


def _rfe_open(mut em: _Member, s: _State, x: PythonObject, n: Int, context: PythonObject) raises:
    """UNPACK of an RFE (lane mf-shap3b, 2026-11-23): `RFE.predict` / `predict_proba` are its fitted estimator's
    over the support columns (`self.estimator_.<op>(self.transform(X))`, bindings/prep_select_native_door.mojo
    `prep_rfe_call`). Its fitted width is checked in `_rfe_transform`'s words, the support columns (`_mask`, the
    mask `transform` selects by) are read once into `em.select`, and the estimator's own member is opened
    (`_member_open`): the RUN gathers those columns of every chunk and calls it. Not served: `em.refusal`."""
    var name = String(py_type(s.model).__name__)
    var fields = py_fields(s.model)
    var fitted = py_get(fields, PythonObject("n_features_in_"))
    if fitted is None:
        raise Error("RuntimeError: mojolearn: this " + name + " instance is not fitted yet")
    if Int(py=fitted) != s.d:
        raise Error("mojolearn: X has " + String(s.d) + " features, " + name + " was fitted with "
                    + String(Int(py=fitted)))
    var mask = py_get(fields, PythonObject("_mask"))
    if mask is None:
        raise Error("RuntimeError: mojolearn: this " + name + " instance is not fitted yet")
    em.select = List[Int32]()
    for i in range(len(mask)):  # driver-loop: over the fitted support mask (model structure, once per call)
        if Bool(py=mask[i]):
            em.select.append(Int32(i))
    var est = _rfe_estimator(s.model)
    if len(em.select) > 0 and len(em.select) <= s.d:
        _member_open(em, s, est, x, n, context)
    if not em.native and em.refusal.byte_length() == 0:
        em.refusal = ("TypeError: the explained " + name + "'s fitted estimator (" + String(py_type(est).__name__)
                      + ") has no native member that serves its " + s.method + " over the selected columns")


def _lp_open(mut em: _Member, s: _State, context: PythonObject) raises:
    """UNPACK of a LabelPropagation / LabelSpreading model (lane mf-shap3b, 2026-11-23): its predict_proba's
    query words through the one body its public query reads them with (bindings/neighbors_lp_query_unpack.mojo
    `lp_query_unpack`), on the x_neighbors image its own query calls (`estimator_binding`, the shared family
    context the x_neighbors doors receive), and that image's native operations entry. The query refuses a
    non-finite chunk itself (its deferred finite word): no host scan."""
    var family = xn_family_context()
    var binding = xn_admit(estimator_binding(s.model, "_mojolearn_x_neighbors", family))
    var unpacked = lp_query_unpack(s.model, binding, False, family)
    var name = String(py_type(s.model).__name__)
    if unpacked.request.features != s.d:
        raise Error("ValueError: X has " + String(s.d) + " features, but " + name + " is expecting "
                    + String(unpacked.request.features) + " features as input.")
    var k = unpacked.request.classes
    if s.k >= 0 and k != s.k:
        raise Error(_WIDTH_CHANGED)
    var connection = LofConnection(binding)
    em.lp_drivers = List[XnDriver]()
    em.lp_drivers.append(connection.driver.copy())
    em.lp_requests = List[LpQueryRequest]()
    em.lp_requests.append(unpacked.request.copy())
    em.lp_keep = Python.list(unpacked.fit_x, unpacked.distributions, connection.provider, binding)
    em.lp_name = name^
    em.k = k
    em.lp = True
    _ = unpacked^


def _open(mut em: _Member, s: _State, x: PythonObject, n: Int, context: PythonObject) raises:
    """UNPACK of the explained model over the rows `x` (`n` rows; the background at construction): a label
    propagation model's typed query (`_lp_open`), an RFE's estimator over its support (`_rfe_open`), its own
    member (`_member_open`), a batched Bagging's forest, else a meta-estimator over members (`_meta_open`). An
    open that found the call must be refused raises its words here. Nothing served: the door answers when the
    class's plan serves the method; else the call is refused (lane mf-shap3b: a member-only class whose member
    does not serve it)."""
    var kind = _explained_kind(s.model)
    if kind == "label_propagation":
        _lp_open(em, s, context)
        return
    if kind == "rfe":
        _rfe_open(em, s, x, n, context)
    else:
        _member_open(em, s, s.model, x, n, context)
        if not em.native and em.refusal.byte_length() == 0:
            var inner = _batched_member(s.model, context)
            if inner is not None:
                _member_open(em, s, inner, x, n, context)
            else:
                _meta_open(em, s, x, n, context)
        if (not em.native and not em.meta and em.refusal.byte_length() == 0
                and not prep_estimator_has(s.model, s.method, context)):
            em.refusal = "TypeError: the explained mojolearn estimator has no native " + s.method
    if em.refusal.byte_length() > 0:
        raise Error(em.refusal)


def _handle_of(mut em: _Member, x: Int, n: Int, d: Int, k: Int, host: Bool, glue: EntryRef) raises -> AgnMember:
    """The native model handle over the opened member (`em.native`); the plan it reads stays in `em`."""
    var plan_addr = Int(UnsafePointer(to=em.held.plan))
    var cols = Int(em.select.unsafe_ptr()) if len(em.select) > 0 else 0
    return AgnMember(em.held.plan.member, plan_addr, em.held.state, em.operation, x, n, d, k, em.parallel, em.wide,
                     host, glue, em.labels_at, em.n_labels, em.ints, cols, len(em.select))


def _expected_with[M: AgnModel](mut model: M, gil_free: Bool, glue: EntryRef, nb: Int, k: Int, logit: Bool,
                                ev: Int) raises:
    """The construction's RUN: the model on the background (`model.explained()`, the GIL released when every
    member may run without it), its mean and link into `ev`."""
    if gil_free:
        with GILReleased(Python()):
            agn_expected_run(glue, model.explained(), nb, k, logit, ev)
    else:
        agn_expected_run(glue, model.explained(), nb, k, logit, ev)


struct _DoorModel(AgnModel):
    """The explained estimator through its own compiled door (`_evaluate`, per chunk), serial, with the GIL held:
    the one place the RUN's model reaches Python objects.

    WHO STILL COMES HERE (lanes mf-shap3b, mf-cal3b, mf-iterimp3b and mf-meta3b, 2026-11-23): ONLY an explained
    estimator whose class has a plan that serves the method by its door but no member entry that serves the call, or
    whose meta-estimator has a member of that kind. The Host* subclasses carry their public class's mark since
    mf-iterimp3b. Since mf-meta3b a VotingClassifier / VotingRegressor / StackingClassifier / StackingRegressor /
    OneVsRestClassifier / CalibratedClassifierCV / BaggingClassifier / BaggingRegressor used as a member opens as its
    META state. What is left: a member of a meta-estimator that is an AdaBoost*, a MultiOutput* or a search CV (no
    META kind: AdaBoost's rounds over device sessions and its one-tree door members, MultiOutput's per-output label
    vectors and the search's Python scorers and candidate clones are not typed yet); a META class whose admit keeps
    its door (a Calibrated non-int cv or DeviceRows base, a Stacking splitter cv). Since lane mf-meta3c the batched
    Bagging route and metas nested at any depth (the boundary's admission work-list) open as META states too. When
    those left are typed, this door and `_evaluate` are deleted. Everything else is native: every member
    family (an outlier detector's or clusterer's integer predict too), the votes (soft and hard), Bagging, Stacking
    (a classifier member asked for predict too), OneVsRest, CalibratedClassifierCV, AdaBoost, MultiOutputRegressor,
    the search CVs' best estimator, RFE (its estimator over its support), LabelPropagation and LabelSpreading (their
    typed query), the device forest's one call; a class with neither a serving member nor a door plan is refused, and
    so is a forecaster; a refusal (a member's non-finite chunk, a changed width, non-numeric labels, a
    MultiOutputClassifier) is raised natively in its own words."""
    var model: PythonObject
    var method: String
    var context: PythonObject
    var x: PythonObject
    var n: Int
    var d: Int
    var k: Int
    # Optional, not a PythonObject read and reassigned in a loop (modular/modular#7546 pattern).
    var fx: Optional[PythonObject]
    var syn: Optional[PythonObject]
    var out: Optional[PythonObject]

    def __init__(out self, s: _State, x: PythonObject, n: Int, context: PythonObject):
        self.model = s.model
        self.method = s.method.copy()
        self.context = context
        self.x = x
        self.n = n
        self.d = s.d
        self.k = s.k
        self.fx = Optional[PythonObject]()
        self.syn = Optional[PythonObject]()
        self.out = Optional[PythonObject]()

    def _width(self, y_out: PythonObject) raises:
        if Int(py=y_out.shape[1]) != self.k:
            raise Error("ValueError: the explained model output width changed after construction")

    def explained(mut self) raises -> Int:
        var fx = _evaluate(self.model, self.method, self.x, self.n, self.context)
        self._width(fx)
        var address = trees_address(fx)
        self.fx = Optional[PythonObject](fx)
        return address

    def synthetic(mut self, count: Int, filled: Bool) raises -> Int:
        if not self.syn or Int(py=self.syn.value().size) != count:
            self.syn = Optional[PythonObject]()
            if filled:
                self.syn = Optional[PythonObject](admitted_vector(count, "<f4", self.context))
            else:
                # Unfilled (fb-infer/C, 2026-11-17): the synthetic writer writes every cell.
                self.syn = Optional[PythonObject](admitted_unfilled([count], "<f4", self.context).keep)
        return trees_address(self.syn.value())

    def rows(mut self, n: Int) raises -> Int:
        var syn = self.syn.value()
        var view = PythonObject(None)
        if n * self.d == Int(py=syn.size):
            view = trees_reshape(syn, Python.tuple(n, self.d), self.context)
        else:
            # MOJOLEARN_PSHAP_DELTA: the first n rows, copied once into the model's input.
            var front = admitted_vector(n * self.d, "<f4", self.context)
            if n * self.d > 0:
                unsafe_memcpy(dest=_F32(unsafe_from_address=trees_address(front)),
                              src=_F32(unsafe_from_address=trees_address(syn)), count=n * self.d)
            view = trees_reshape(front, Python.tuple(n, self.d), self.context)
        _ = syn  # the source words are read through the copy above
        self.out = Optional[PythonObject]()
        var out = _evaluate(self.model, self.method, view, n, self.context)
        _ = view
        self._width(out)
        var address = trees_address(out)
        self.out = Optional[PythonObject](out)
        return address

    def workers(self) -> Int:
        return 1

    def worker(self) raises -> Self:
        raise Error("RuntimeError: the explained model's door runs on the calling thread only")


def _request(s: _State, x: PythonObject, phi: PythonObject, n: Int, glue_key: TreesEntryKey) raises -> AgnRequest:
    """UNPACK: the typed request of one values call (the buffers stay alive with the door's locals)."""
    return AgnRequest(trees_address(x), trees_address(s.bg), trees_address(phi), n, s.nb, s.d, s.k, s.seed,
                      _switches(s.table), s.mode == "fast", glue_key.vendor == 0)


def _kernel_values(fields: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """KernelExplainer values; the member state opened inside is released on
    every exit."""
    var em = _Member()
    var result = PythonObject(None)
    try:
        result = _kernel_values_over(fields, args, context, em)
    finally:
        _member_release(em)
    return result


def _kernel_with[M: AgnModel](mut model: M, gil_free: Bool, glue: EntryRef, req: AgnRequest, ns: Int, nl: Int,
                              logit: Bool) raises:
    """RUN `agn_kernel_values_run` over `model`, the GIL released once around it when every member may run
    without it. A member refusal raises its own words."""
    if gil_free:
        with GILReleased(Python()):
            agn_kernel_values_run(glue, req, ns, nl, logit, model)
    else:
        agn_kernel_values_run(glue, req, ns, nl, logit, model)


def _kernel_values_over(fields: PythonObject, args: PythonObject, context: PythonObject,
                        mut em: _Member) raises -> PythonObject:
    """UNPACK (settings, rows, the native member or meta-estimator), RUN (xtrees/agnostic_values_run.mojo
    `agn_kernel_values_run`), PACK (the shaped float64 values)."""
    var name = String("KernelExplainer")
    var nsamples = args[1]
    var l1 = args[2]
    var b = _b()
    var l1_ok = ((py_type(l1) is b.str and String(l1) == "auto") or (py_type(l1) is b.bool and not Bool(py=l1))
                 or (py_type(l1) is b.int and Int(py=l1) == 0))
    if not l1_ok:
        raise Error("NotImplementedError: KernelExplainer carries no l1 feature selection; l1_reg must be 'auto', False or 0")
    var s = _state(fields, name, context)
    var link = _get(fields, "link")
    var logit = py_type(link) is b.str and String(link) == "logit"
    var x = _rows(args[0], s, name, context)
    var n = Int(py=x.shape[0])
    var d = s.d
    var k = s.k
    var ns = 0
    if py_type(nsamples) is b.str and String(nsamples) == "auto":
        ns = 2 * d + 2048
    elif py_type(nsamples) is b.int:
        ns = Int(py=nsamples)
    else:
        raise Error("TypeError: nsamples must be an int or 'auto'")
    var phi = admitted_vector(max(n * d * k, 1), "<f8", context)
    if n == 0:
        return _shape(phi, 0, d, k, context)
    var glue_key = _xk(s.table)
    var glue = xt_ref(glue_key)
    var req = _request(s, x, phi, n, glue_key)
    var fnull = _get(fields, "_fnull")
    var nl = trees_address(fnull)
    # The device forest route needs the model once (fx), through the same handle as every other route.
    _forest_words(s.dev, s.bg, req)
    _open(em, s, x, n, context)
    if em.meta:
        # The meta's glue, resolved here with the GIL held; the members' plans stay in `em` through the RUN.
        var meta = AgnMeta(em.meta_plan, req.x, n, d, em.meta_parallel, req.host, _xr(em.meta_xt))
        _kernel_with(meta, em.meta_gil_free, glue, req, ns, nl, logit)
        _ = meta^
    elif em.native:
        var handle = _handle_of(em, req.x, n, d, k, req.host, glue)
        _kernel_with(handle, em.gil_free, glue, req, ns, nl, logit)
        _ = handle^
    elif em.lp:
        # The typed query, released as the public predict_proba releases it; the request's buffers stay in `em`.
        var lp = AgnLabelProp(em.lp_drivers[0], em.lp_requests[0], em.lp_name, req.x, n, d)
        _kernel_with(lp, True, glue, req, ns, nl, logit)
        _ = lp^
    else:
        var door = _DoorModel(s, x, n, context)
        agn_kernel_values_run(glue, req, ns, nl, logit, door)
        _ = door^
    _ = x
    _ = fnull
    _ = s.bg  # the background the RUN read through req.bg
    _ = req^
    return _shape(phi, n, d, k, context)


def _linear_block(s: _State, context: PythonObject) raises -> PythonObject:
    """MOJOLEARN_IDN_PSHAP_RESIDENT: ([w | b] float32 block, link) when the
    binding carries the switch and the explained model's predict is this
    library's linear `link(X W^T + b)`; else None. Byte copies only."""
    var ok = _entry(s.table, "x_trees_pshap_linear_ok", False)
    if ok is None:
        return PythonObject(None)
    var flag = py_call(ok)
    if py_type(flag) is not _b().int or Int(py=flag) != 1:
        return PythonObject(None)
    var nk = trees_keys()
    var linear_predict = py_get(context, nk[].k_trees_linear_predict)
    if linear_predict is None or _class_value(s.model, "predict") is not linear_predict:
        return PythonObject(None)
    var fields = _fields(s.model)
    var fitted = _get(fields, "n_features_in_")
    if fitted is None or py_type(fitted) is not _b().int or Int(py=fitted) != s.d:
        return PythonObject(None)
    var link = _get(fields, "_link")
    if link is None:
        link = _class_value(s.model, "_link")
    if link is None or py_type(link) is not _b().int:
        return PythonObject(None)
    var coef = _get(fields, "coef_")
    var intercept = _get(fields, "intercept_")
    if coef is None or intercept is None:
        return PythonObject(None)
    var snaps = Python.dict()
    var c_view = classical_view(coef, context, "coef_")
    var c_rank = Int(py=c_view.ndim)
    _ = c_view
    var c = _snapshot(coef, "<f4", c_rank, "coef_", context, snaps)
    var bias = PythonObject(None)
    if py_type(intercept) is _b().float or py_type(intercept) is _b().int:
        bias = _snapshot(Python.list(intercept), "<f4", 1, "intercept_", context, snaps)
    else:
        var i_view = classical_view(intercept, context, "intercept_")
        var i_rank = Int(py=i_view.ndim)
        _ = i_view
        bias = _snapshot(intercept, "<f4", i_rank, "intercept_", context, snaps)
    var kk = Int(py=bias.size)
    var d = Int(py=c.size) // max(kk, 1)
    if kk != s.k or d != s.d or Int(py=c.size) != kk * d:
        return PythonObject(None)
    var wb = admitted_vector(kk * (d + 1), "<f4", context)
    var o = _F32(unsafe_from_address=trees_address(wb))
    var cp = _F32(unsafe_from_address=trees_address(c))
    var bp = _F32(unsafe_from_address=trees_address(bias))
    for j in range(kk):  # driver-loop: over the fitted output rows (model structure)
        for f in range(d):  # driver-loop: one coefficient row copy
            o[j * (d + 1) + f] = cp[j * d + f]
        o[j * (d + 1) + d] = bp[j]
    _ = c
    _ = bias
    return Python.tuple(wb, link)


struct _Unevaluated(AgnModel):
    """The model of a Permutation SHAP route that never evaluates it (the device forest, the resident linear
    block: both evaluate on the device per chunk). Any call raises."""

    def __init__(out self):
        pass

    def explained(mut self) raises -> Int:
        raise Error("RuntimeError: the explained model is not evaluated on this route")

    def synthetic(mut self, count: Int, filled: Bool) raises -> Int:
        raise Error("RuntimeError: the explained model is not evaluated on this route")

    def rows(mut self, n: Int) raises -> Int:
        raise Error("RuntimeError: the explained model is not evaluated on this route")

    def workers(self) -> Int:
        return 1

    def worker(self) raises -> Self:
        return Self()


def _permutation_values(fields: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """PermutationExplainer values; the member state opened inside is
    released on every exit."""
    var em = _Member()
    var result = PythonObject(None)
    try:
        result = _permutation_values_over(fields, args, context, em)
    finally:
        _member_release(em)
    return result


def _permutation_with[M: AgnModel](mut model: M, gil_free: Bool, glue: EntryRef, req: AgnRequest, npm: Int,
                                   wb: Int, wb_link: Int) raises:
    """RUN `agn_permutation_values_run` over `model`, the GIL released once around it when every member may run
    without it. A member refusal raises its own words."""
    if gil_free:
        with GILReleased(Python()):
            agn_permutation_values_run(glue, req, npm, wb, wb_link, model)
    else:
        agn_permutation_values_run(glue, req, npm, wb, wb_link, model)


def _permutation_values_over(fields: PythonObject, args: PythonObject, context: PythonObject,
                             mut em: _Member) raises -> PythonObject:
    """UNPACK (settings, rows, the device forest, the resident linear block, the native member or
    meta-estimator), RUN (xtrees/agnostic_values_run.mojo `agn_permutation_values_run`), PACK (the shaped float64
    values)."""
    var name = String("PermutationExplainer")
    var b = _b()
    var s = _state(fields, name, context)
    var x = _rows(args[0], s, name, context)
    var n = Int(py=x.shape[0])
    var d = s.d
    var k = s.k
    if py_type(args[1]) is not b.int:
        raise Error("TypeError: npermutations must be an int")
    var npm = Int(py=args[1])
    if npm < 1:
        raise Error("ValueError: npermutations must be >= 1")
    var phi = admitted_vector(max(n * d * k, 1), "<f8", context)
    if n == 0:
        return _shape(phi, 0, d, k, context)
    var glue_key = _xk(s.table)
    var glue = xt_ref(glue_key)
    var req = _request(s, x, phi, n, glue_key)
    _forest_words(s.dev, s.bg, req)
    var wb = 0
    var wb_link = 0
    var lin = PythonObject(None)
    if not req.device_forest():
        lin = _linear_block(s, context)
        if lin is not None:
            # IDN_PSHAP_RESIDENT: synthetic rows, the linear predict, means and marginals on the device per
            # chunk; only x rows and phi cross.
            wb = trees_address(lin[0])
            wb_link = Int(py=lin[1])
    if req.device_forest() or lin is not None:
        # Neither route calls the model (the forest and the linear block are evaluated on the device per chunk):
        # nothing is opened (lane mf-shap3: the door's evaluator stood here, never called).
        var unused = _Unevaluated()
        _permutation_with(unused, True, glue, req, npm, wb, wb_link)
        _ = unused^
    else:
        # The explained model as a native member, else a meta-estimator over native members.
        _open(em, s, x, n, context)
        if em.meta:
            var meta = AgnMeta(em.meta_plan, req.x, n, d, em.meta_parallel, req.host, _xr(em.meta_xt))
            _permutation_with(meta, em.meta_gil_free, glue, req, npm, wb, wb_link)
            _ = meta^
        elif em.native:
            var handle = _handle_of(em, req.x, n, d, k, req.host, glue)
            _permutation_with(handle, em.gil_free, glue, req, npm, wb, wb_link)
            _ = handle^
        elif em.lp:
            var lp = AgnLabelProp(em.lp_drivers[0], em.lp_requests[0], em.lp_name, req.x, n, d)
            _permutation_with(lp, True, glue, req, npm, wb, wb_link)
            _ = lp^
        else:
            var door = _DoorModel(s, x, n, context)
            agn_permutation_values_run(glue, req, npm, wb, wb_link, door)
            _ = door^
    _ = lin
    _ = x
    _ = s.bg  # the background the RUN read through req.bg
    _ = req^
    return _shape(phi, n, d, k, context)


def trees_agnostic_operation(model: PythonObject, operation: Int, opname: String, args: PythonObject,
                             context: PythonObject) raises -> PythonObject:
    """One `trees_agnostic_*` method by its leaf code
    (bindings/operation_names_trees.mojo; lane bw3-trees); `opname` (the
    table's String, borrowed) is for the unknown-operation text only."""
    classical_check_layout(model, context)
    var fields = _fields(model)
    if operation == TREES_OPN_TREES_AGNOSTIC_KERNEL_INIT:
        _construct(fields, args, False, context)
        return PythonObject(None)
    if operation == TREES_OPN_TREES_AGNOSTIC_PERMUTATION_INIT:
        _construct(fields, args, True, context)
        return PythonObject(None)
    if operation == TREES_OPN_TREES_AGNOSTIC_KERNEL_VALUES:
        return _kernel_values(fields, args, context)
    if operation == TREES_OPN_TREES_AGNOSTIC_PERMUTATION_VALUES:
        return _permutation_values(fields, args, context)
    raise Error("unknown model-agnostic explainer operation " + opname)
