# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the NaiveBayes family: its place in
`bindings/member_boundary_door.mojo` (recipe: MEMBER-ABI-V2.md; worked form
providers/python/member_linear_boundary_door.mojo). Kind 0 is GaussianNB
(`_mojolearn_member = (MEMBER_FAMILY_NAIVE_BAYES, 0)` on the class), kinds
1 to 3 MultinomialNB, BernoulliNB and ComplementNB, kind 4 CategoricalNB
(lane mf-shap2, 2026-11-23); a kind the member entry does not serve
(core/naive_bayes_member.mojo `naive_bayes_kind_published`) is not
published (admit answers False and the caller keeps its door).

  * `naive_bayes_member_admit` (UNPACK, once per member): the estimator's
    own parameter checks (`var_smoothing` and the values of `priors`;
    `alpha` and the values of `class_prior` for the discrete kinds), its
    numeric mode, the x_prep image its own fit selects
    (bindings/prep_classify_prog_door.mojo `cls_selection`: the device
    route's held selection, else resolved and kept), that image's transport
    entry in the route's entry slot, its capabilities and mode bits in
    `route.flags`, and the parameter words of core/naive_bayes_member.mojo.
    The switches the door reads per fit (MOJOLEARN_ARENA_RANGES,
    MOJOLEARN_XPREP_BLOCKED, the R3 `work` choice) are read once here.
  * `naive_bayes_member_pack` (PACK): the fitted attributes the estimator's
    own fit writes (bindings/prep_classify_native_door.mojo `_gnb_publish`,
    `_params`),
    from the member state's blocks. The class labels are the caller's
    (`member_pack_labeled` writes `classes_`).
  * `naive_bayes_member_state` (UNPACK): a member state from a fitted
    estimator.

SERIAL BY NATURE: one member's admission or publication (field reads and
one copy per fitted block); a caller runs its members in parallel.
"""
from std.os import getenv
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from bindings.member_owner_unpack import MemberOwner
from bindings.py_balanced import py_call, py_contains, py_get, py_item, py_setitem, py_type
from bindings.classical_arrays_door import classical_address, classical_typed
from bindings.native_loaded_module_door import native_loaded_module
from bindings.native_array_storage_door import native_array_empty
from bindings.classical_arrays_door import classical_wrap_array
from bindings.prep_keys_unpack import prep_keys
from bindings.prep_classify_prog_door import cls_selection, cls_image_plan, f32_array, f32_input
from bindings.prep_classify_common_door import (
    cls_requested_mode, cls_mode, cls_entry, cls_float, idn_fam, idn_int, blocked, float_list, check_priors,
    gnb_refuse, is_real, IDN_CLASS_ONEPASS, IDN_NB_ONEPASS,
)
from bindings.x_prep_native_owner_run import xprep_r3
from core.device_route import DeviceRoute, DEVICE_CPU, DEVICE_GPU
from core.entry_table import entry_ref_mode, entry_ref_vendor
from core.hostptr import f32_ptr
from core.naive_bayes_member import (
    NB_KIND_GAUSSIAN, NB_KIND_MULTINOMIAL, NB_KIND_BERNOULLI, NB_KIND_COMPLEMENT, NB_KIND_CATEGORICAL,
    NB_BLOCK_CATEGORIES, NB_GAUSSIAN_PARAM_HEAD,
    NB_DISCRETE_PARAM_HEAD, NB_BLOCK_THETA, NB_BLOCK_VAR, NB_BLOCK_CONST,
    NB_BLOCK_CLASS_COUNT, NB_BLOCK_CLASS_PRIOR, NB_BLOCK_RAW_VAR, NB_BLOCK_LOG_PROB, NB_BLOCK_LOG_PRIOR,
    NB_BLOCK_FEATURE_COUNT, NB_BLOCK_W, NB_BLOCK_BIAS,
    NB_ROUTE_OUT, NB_ROUTE_SCRATCH, NB_ROUTE_RANGES, NB_ROUTE_DEV_STORE, NB_ROUTE_TAKE_ROWS, NB_ROUTE_PROBA64,
    NB_ROUTE_CLASS_ONEPASS, NB_ROUTE_NB_ONEPASS, naive_bayes_kind_published,
)
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_new, member_free,
    MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_BLOCK_SCALARS, MEMBER_IMPORT_DONE, MEMBER_F64,
)


def _route(model: PythonObject, mode: PythonObject, is_blocked: Bool, context: PythonObject,
    mut owner: MemberOwner) raises:
    """The x_prep image the estimator's own fit selects for `mode`, as the
    member's route: its transport entry, its capabilities and mode bits."""
    var entries = cls_selection(model, mode, context)
    var image = cls_image_plan(model, entries)
    var flags = 0
    if image.has_out:
        flags |= NB_ROUTE_OUT
    if image.has_scratch:
        flags |= NB_ROUTE_SCRATCH
    if image.has_ranges and String(getenv("MOJOLEARN_ARENA_RANGES", "1").strip()) != "0":
        flags |= NB_ROUTE_RANGES
    if image.dev_store:
        flags |= NB_ROUTE_DEV_STORE
    if image.take_rows:
        flags |= NB_ROUTE_TAKE_ROWS
    if cls_entry(mode, "x_prep_proba64", context) is not None:
        flags |= NB_ROUTE_PROBA64
    if is_blocked and (idn_fam(mode, context) & IDN_CLASS_ONEPASS) != 0:
        flags |= NB_ROUTE_CLASS_ONEPASS
    if is_blocked and (idn_int(mode, context) & IDN_NB_ONEPASS) != 0:
        flags |= NB_ROUTE_NB_ONEPASS
    # The image words of the transport entry: the member table key the
    # x_prep images publish this family under.
    var vendor = entry_ref_vendor(image.key)
    var route = DeviceRoute(DEVICE_CPU if vendor == 0 else DEVICE_GPU)
    route.mode = entry_ref_mode(image.key)
    route.vendor = vendor
    route.entry = image.key
    route.flags = flags
    owner.plan.route = route
    owner.pins = entries


def _kind_name(kind: Int) -> String:
    if kind == NB_KIND_MULTINOMIAL:
        return String("MultinomialNB")
    if kind == NB_KIND_BERNOULLI:
        return String("BernoulliNB")
    if kind == NB_KIND_COMPLEMENT:
        return String("ComplementNB")
    if kind == NB_KIND_CATEGORICAL:
        return String("CategoricalNB")
    return String("GaussianNB")


def _admit_discrete(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    mut owner: MemberOwner) raises:
    """The discrete kinds: the door's alpha and class prior checks
    (`_check_alpha`, `given_priors` without its count check), the process
    mode its fit runs in, the parameter words."""
    var nk = prep_keys()
    var name = _kind_name(kind)
    var alpha = py_item(fields, nk[].k_alpha)
    if not is_real(alpha, context) or not (cls_float(alpha) > 0):
        raise Error("NotImplementedError: mojolearn: " + name + " needs alpha > 0 (alpha = 0 makes log(0) terms)")
    var priors = List[Float64]()
    var given = py_get(fields, nk[].k_class_prior)
    if given is not None:
        priors = float_list(given, context)
        if len(priors) == 0:
            raise Error("mojolearn: " + name + ": number of priors must match number of classes")
        check_priors(priors, name, False)
    var is_blocked = blocked()
    _route(model, cls_mode(context), is_blocked, context, owner)
    # CategoricalNB's min_categories (the door's `_cat_fit_programs` reads it per fit): None, one integer for
    # every feature, or a list whose length its fit checks.
    var min_mode = 0
    var min_value = 0.0
    var mins = List[Float64]()
    if kind == NB_KIND_CATEGORICAL:
        var mc = py_item(fields, nk[].k_min_categories)
        if mc is not None:
            if Bool(py=py_call(native_loaded_module("builtins").isinstance, mc, py_item(context, nk[].k_integral_type))):
                min_mode = 1
                min_value = Float64(Int(py=py_call(native_loaded_module("builtins").int, mc)))
            else:
                min_mode = 2
                var listed = float_list(mc, context)
                for j in range(len(listed)):  # driver-loop: over a user parameter list (feature-count sized)
                    mins.append(Float64(Int(listed[j])))
    var words = List[Float64](length=NB_DISCRETE_PARAM_HEAD + len(priors) + len(mins), fill=0.0)
    words[0] = Float64(kind)
    words[1] = cls_float(alpha)
    words[2] = 1.0 if is_blocked else 0.0
    words[3] = 1.0 if xprep_r3("work") else 0.0
    words[4] = Float64(len(priors))
    words[5] = 1.0 if Bool(py=py_item(fields, nk[].k_fit_prior)) else 0.0
    if kind == NB_KIND_CATEGORICAL:
        words[6] = Float64(min_mode)
        words[7] = min_value
        words[8] = Float64(len(mins))
        for j in range(len(mins)):  # driver-loop: over a user parameter list (feature-count sized)
            words[NB_DISCRETE_PARAM_HEAD + len(priors) + j] = mins[j]
    if kind == NB_KIND_BERNOULLI and py_item(fields, nk[].k_binarize) is not None:
        words[6] = 1.0
        words[7] = cls_float(py_item(fields, nk[].k_binarize))
    if kind == NB_KIND_COMPLEMENT and Bool(py=py_item(fields, nk[].k_norm)):
        words[8] = 1.0
    for i in range(len(priors)):  # driver-loop: over the user prior list (class-count sized)
        words[NB_DISCRETE_PARAM_HEAD + i] = priors[i]
    owner.plan.params = words^


def naive_bayes_member_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """UNPACK: fill `owner.plan.route`, `owner.plan.params` and `owner.pins`
    from the unfitted estimator. False: this class has no member entry yet."""
    if not naive_bayes_kind_published(kind):
        return False
    if kind != NB_KIND_GAUSSIAN:
        _admit_discrete(model, fields, kind, context, owner)
        return True
    var nk = prep_keys()
    gnb_refuse(fields, context)
    var is_blocked = blocked()
    _route(model, cls_requested_mode(fields, context), is_blocked, context, owner)
    var priors = List[Float64]()
    if py_item(fields, nk[].k_priors) is not None:
        priors = float_list(py_item(fields, nk[].k_priors), context)
        if len(priors) == 0:
            raise Error("mojolearn: GaussianNB: number of priors must match number of classes")
        check_priors(priors, "GaussianNB", True)
    var words = List[Float64](length=NB_GAUSSIAN_PARAM_HEAD + len(priors), fill=0.0)
    words[0] = Float64(NB_KIND_GAUSSIAN)
    words[1] = cls_float(py_item(fields, nk[].k_var_smoothing))
    words[2] = 1.0 if is_blocked else 0.0
    words[3] = 1.0 if xprep_r3("work") else 0.0
    words[4] = Float64(len(priors))
    for i in range(len(priors)):  # driver-loop: over the user prior list (class-count sized)
        words[NB_GAUSSIAN_PARAM_HEAD + i] = priors[i]
    owner.plan.params = words^
    return True


def _export(plan: MemberPlan, mut call: MemberCall, block: Int) raises -> MemberMatrix:
    call.block = block
    call.flags = 0
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    return call.dst


def _cat_blocks(view: MemberMatrix, d: Int, K: Int, cmax: Int, ncat: List[Int], context: PythonObject) raises -> PythonObject:
    """The door's `_cat_blocks` from a member view: per feature j, the (K, ncat[j]) float32 block of the
    (d, K, cmax) table."""
    var out = Python.list()
    for j in range(d):  # driver-loop: over the feature columns (model structure)
        var block = (classical_typed(Python.tuple(K, ncat[j]), "<f4", context["output_context"]) if ncat[j] > 0
                     else classical_wrap_array(native_array_empty("f"), Python.tuple(K, 0), "<f4", "C",
                                               context["output_context"]))
        if ncat[j] > 0:
            var dest = f32_ptr(Int(py=classical_address(block)))
            for k in range(K):  # driver-loop: over the classes (model structure)
                unsafe_memcpy(dest=dest + k * ncat[j], src=f32_ptr(view.data) + (j * K + k) * cmax, count=ncat[j])
            _ = block  # keep the Python object alive through the copies above
        _ = py_call(out.append, block)
    return out


def _cat_pack(plan: MemberPlan, mut call: MemberCall, fields: PythonObject, d: Int, K: Int, cmax: Int,
              mode_word: String, context: PythonObject) raises:
    """PACK of CategoricalNB: the fitted attributes the door's `_cat_fit_programs` writes."""
    var categories = _export(plan, call, NB_BLOCK_CATEGORIES)
    var counts = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=categories.data)
    var ncat = List[Int](capacity=d)
    for j in range(d):  # driver-loop: over the feature columns (model structure)
        ncat.append(Int(counts[j]))
    var cc = _export(plan, call, NB_BLOCK_FEATURE_COUNT)
    py_setitem(fields, "_cc", _array(cc, False, context))
    py_setitem(fields, "category_count_", _cat_blocks(cc, d, K, cmax, ncat, context))
    var n_categories = classical_typed(Python.tuple(d), "<i8", context["output_context"])
    var cp = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(n_categories)))
    for j in range(d):  # driver-loop: over the feature columns (model structure)
        cp[j] = Int64(ncat[j])
    _ = n_categories  # keep the Python object alive through the writes above
    py_setitem(fields, "n_categories_", n_categories)
    var flp = _export(plan, call, NB_BLOCK_LOG_PROB)
    py_setitem(fields, "_flp", _array(flp, False, context))
    py_setitem(fields, "_cmax", PythonObject(cmax))
    py_setitem(fields, "feature_log_prob_", _cat_blocks(flp, d, K, cmax, ncat, context))
    py_setitem(fields, "class_count_", _array(_export(plan, call, NB_BLOCK_CLASS_COUNT), False, context))
    py_setitem(fields, "class_log_prior_", _array(_export(plan, call, NB_BLOCK_LOG_PRIOR), False, context))
    py_setitem(fields, "numeric_mode_", PythonObject(mode_word))
    py_setitem(fields, "n_features_in_", PythonObject(d))
    py_setitem(fields, "_mojolearn_fitted", True)


def _array(view: MemberMatrix, matrix: Bool, context: PythonObject) raises -> PythonObject:
    # The fitted attribute type of the door's `ClsProg.get`: a float32
    # C-ordered Array, (rows, cols) or (cols,).
    var count = view.rows * view.cols
    var value = (classical_typed(Python.tuple(view.rows, view.cols), "<f4", context["output_context"]) if matrix
                 else f32_array(count, context))
    unsafe_memcpy(dest=f32_ptr(Int(py=classical_address(value))), src=f32_ptr(view.data), count=count)
    return value


def naive_bayes_member_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, state: Int) raises:
    """PACK: write the fitted attributes of state `state` into `model`, as
    the door's fit does (`_gnb_publish`, `_params`)."""
    if not naive_bayes_kind_published(plan.kind):
        raise Error("ValueError: mojolearn: NaiveBayes member kind " + String(plan.kind) + " is not published")
    var call = MemberCall(plan.route, state)
    var scalars = _export(plan, call, MEMBER_BLOCK_SCALARS)
    var words = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=scalars.data)
    var d = Int(words[0])
    # The mode word the fit ran in: the route's image word.
    var mode_word = String("identical") if plan.route.mode == 1 else String("fast")
    if plan.kind == NB_KIND_CATEGORICAL:
        _cat_pack(plan, call, fields, d, Int(words[1]), Int(words[2]), mode_word, context)
        return
    if plan.kind != NB_KIND_GAUSSIAN:
        # the door's `_params` (`_finish_counts`)
        py_setitem(fields, "class_count_", _array(_export(plan, call, NB_BLOCK_CLASS_COUNT), False, context))
        py_setitem(fields, "feature_count_", _array(_export(plan, call, NB_BLOCK_FEATURE_COUNT), True, context))
        py_setitem(fields, "class_log_prior_", _array(_export(plan, call, NB_BLOCK_LOG_PRIOR), False, context))
        py_setitem(fields, "numeric_mode_", PythonObject(mode_word))
        py_setitem(fields, "n_features_in_", PythonObject(d))
        py_setitem(fields, "feature_log_prob_", _array(_export(plan, call, NB_BLOCK_LOG_PROB), True, context))
        if plan.kind == NB_KIND_BERNOULLI:
            py_setitem(fields, "_w", _array(_export(plan, call, NB_BLOCK_W), True, context))
            py_setitem(fields, "_bias", _array(_export(plan, call, NB_BLOCK_BIAS), False, context))
        py_setitem(fields, "_mojolearn_fitted", True)
        return
    var epsilon = words[2]
    py_setitem(fields, "theta_", _array(_export(plan, call, NB_BLOCK_THETA), True, context))
    py_setitem(fields, "var_", _array(_export(plan, call, NB_BLOCK_VAR), True, context))
    py_setitem(fields, "class_count_", _array(_export(plan, call, NB_BLOCK_CLASS_COUNT), False, context))
    py_setitem(fields, "class_prior_", _array(_export(plan, call, NB_BLOCK_CLASS_PRIOR), False, context))
    py_setitem(fields, "epsilon_", PythonObject(epsilon))
    py_setitem(fields, "_const", _array(_export(plan, call, NB_BLOCK_CONST), False, context))
    py_setitem(fields, "_raw_var", _array(_export(plan, call, NB_BLOCK_RAW_VAR), True, context))
    py_setitem(fields, "numeric_mode_", PythonObject(mode_word))
    py_setitem(fields, "n_features_in_", PythonObject(d))
    py_setitem(fields, "_mojolearn_fitted", True)


def _import(plan: MemberPlan, mut call: MemberCall, fields: PythonObject, name: String, block: Int, rows: Int,
    cols: Int, context: PythonObject) raises:
    var who = _kind_name(plan.kind)
    if not py_contains(fields, PythonObject(name)):
        raise Error("NotFittedError: this " + who + " member needs its fitted " + name)
    var value = f32_input(fields[PythonObject(name)], context)
    if Int(py=value.size) != rows * cols:
        raise Error("ValueError: " + who + " " + name + " does not hold its fitted size")
    call.block = block
    call.flags = 0
    call.x = MemberMatrix(Int(py=classical_address(value)), rows, cols)
    _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    _ = value  # the member copied the words in before the buffer is released


def naive_bayes_member_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan) raises -> Int:
    """UNPACK: a new member state holding the fitted attributes of `model`.
    The caller frees it (`member_free`)."""
    if not naive_bayes_kind_published(plan.kind):
        raise Error("ValueError: mojolearn: NaiveBayes member kind " + String(plan.kind) + " is not published")
    var who = _kind_name(plan.kind)
    if plan.kind == NB_KIND_CATEGORICAL:
        return _cat_state(plan, fields, context)
    var gaussian = plan.kind == NB_KIND_GAUSSIAN
    # the block whose size is the class count
    var sized = String("_const") if gaussian else String("class_log_prior_")
    if (not py_contains(fields, "n_features_in_") or not py_contains(fields, PythonObject(sized))
            or (gaussian and not py_contains(fields, "epsilon_"))):
        raise Error("NotFittedError: This " + who + " instance is not fitted yet. Call 'fit' with appropriate "
                    + "arguments before using this estimator.")
    var bi = native_loaded_module("builtins")
    if py_type(fields["n_features_in_"]) is not bi.int:
        raise Error("TypeError: " + who + " n_features_in_ must be a native integer")
    var d = Int(py=fields["n_features_in_"])
    var K = Int(py=f32_input(fields[PythonObject(sized)], context).size)
    if d < 1 or K < 1:
        raise Error("ValueError: " + who + " does not hold a fitted state")
    var scalars: List[Float64] = [Float64(d), Float64(K)]
    if gaussian:
        scalars.append(cls_float(fields["epsilon_"]))
    var call = member_new(plan)
    try:
        if gaussian:
            _import(plan, call, fields, "theta_", NB_BLOCK_THETA, K, d, context)
            _import(plan, call, fields, "var_", NB_BLOCK_VAR, K, d, context)
            _import(plan, call, fields, "_const", NB_BLOCK_CONST, 1, K, context)
            _import(plan, call, fields, "class_count_", NB_BLOCK_CLASS_COUNT, 1, K, context)
            _import(plan, call, fields, "class_prior_", NB_BLOCK_CLASS_PRIOR, 1, K, context)
            _import(plan, call, fields, "_raw_var", NB_BLOCK_RAW_VAR, K, d, context)
        else:
            _import(plan, call, fields, "feature_log_prob_", NB_BLOCK_LOG_PROB, K, d, context)
            _import(plan, call, fields, "class_log_prior_", NB_BLOCK_LOG_PRIOR, 1, K, context)
            _import(plan, call, fields, "feature_count_", NB_BLOCK_FEATURE_COUNT, K, d, context)
            _import(plan, call, fields, "class_count_", NB_BLOCK_CLASS_COUNT, 1, K, context)
            if plan.kind == NB_KIND_BERNOULLI:
                _import(plan, call, fields, "_w", NB_BLOCK_W, K, d, context)
                _import(plan, call, fields, "_bias", NB_BLOCK_BIAS, 1, K, context)
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, len(scalars), MEMBER_F64)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = len(scalars)
    return call.state


def _cat_state(plan: MemberPlan, fields: PythonObject, context: PythonObject) raises -> Int:
    """UNPACK of a fitted CategoricalNB as a member state: `_flp`, class_log_prior_, `_cc`, class_count_ and
    n_categories_ copied in, with the scalars [n_features_in_, classes, `_cmax`]."""
    var who = String("CategoricalNB")
    var names: List[String] = ["n_features_in_", "class_log_prior_", "_flp", "_cc", "class_count_", "n_categories_",
                               "_cmax"]
    for i in range(len(names)):  # driver-loop: over the fitted attribute names
        if not py_contains(fields, PythonObject(names[i])):
            raise Error("NotFittedError: This " + who + " instance is not fitted yet. Call 'fit' with appropriate "
                        + "arguments before using this estimator.")
    var bi = native_loaded_module("builtins")
    if py_type(fields["n_features_in_"]) is not bi.int or py_type(fields["_cmax"]) is not bi.int:
        raise Error("TypeError: " + who + " n_features_in_ and _cmax must be native integers")
    var d = Int(py=fields["n_features_in_"])
    var cmax = Int(py=fields["_cmax"])
    var K = Int(py=f32_input(fields["class_log_prior_"], context).size)
    if d < 1 or K < 1 or cmax < 1:
        raise Error("ValueError: " + who + " does not hold a fitted state")
    var given = fields["n_categories_"]
    if Int(py=given.size) != d or String(py=given.dtype) != "<i8":
        raise Error("ValueError: " + who + " n_categories_ does not hold one int64 count per feature")
    var source = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(given)))
    var categories = List[Float64](capacity=d)
    for j in range(d):  # driver-loop: over the feature columns (model structure)
        categories.append(Float64(source[j]))
    _ = given  # the counts are read above
    var scalars: List[Float64] = [Float64(d), Float64(K), Float64(cmax)]
    var call = member_new(plan)
    try:
        _import(plan, call, fields, "_flp", NB_BLOCK_LOG_PROB, d * K, cmax, context)
        _import(plan, call, fields, "class_log_prior_", NB_BLOCK_LOG_PRIOR, 1, K, context)
        _import(plan, call, fields, "_cc", NB_BLOCK_FEATURE_COUNT, d * K, cmax, context)
        _import(plan, call, fields, "class_count_", NB_BLOCK_CLASS_COUNT, 1, K, context)
        call.block = NB_BLOCK_CATEGORIES
        call.flags = 0
        call.x = MemberMatrix(Int(categories.unsafe_ptr()), 1, d, MEMBER_F64)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, len(scalars), MEMBER_F64)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = len(scalars)
    _ = len(categories)
    return call.state
