# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the Gp family: its place in
`bindings/member_boundary_door.mojo` (recipe: MEMBER-ABI-V2.md; worked form
providers/python/member_linear_boundary_door.mojo). Kind 0 is
GaussianProcessRegressor (`_mojolearn_member = (MEMBER_FAMILY_GP, 0)`), kind 1
GaussianProcessClassifier (`(MEMBER_FAMILY_GP, 1)`).

Admit never changes the caller's template estimator: the estimator's own
checks run on a scratch copy of its fields; PACK runs them again on the clone
it packs, which is where the private fit settings land.

Classifier: admit = `gpc_validate_fields` (kernel, optimizer None,
n_restarts_optimizer 0, max_iter_predict, warm_start, copy_X_train,
multi_class, n_jobs) + `_gpc_ext` + `GPConnection` + `gp_kernel_arrays`; PACK =
`_gpc_set_fitted` over one `_BinaryLaplace` per fit (`_gpc_binary_result`)
with the caller's labels (`member_labels_take`; the codes 0 .. k-1 when the
caller gave none); state = import of X_train_ and every fit's y_train_, L_,
pi_, W_sr_ (back to back, one copy each) and the fitted kernel.

  * `gp_member_admit` (UNPACK, once per member): the estimator's own
    parameter checks (`_gpr_validate`, bindings/gp_regressor_native_door.mojo:
    kernel, alpha, optimizer, n_restarts_optimizer, normalize_y,
    copy_X_train, n_targets, random_state), its binding and mode check
    (`_gpr_ext`), the kernel's postfix table and free hyperparameters
    (`gp_kernel_arrays`, `gp_free`: the same admission and refusals), the GP
    provider entry in the route's entry slot with its capability word in
    `route.flags`, and, for normalize_y, the scaler and program provider
    entries (`GPNormalizationConnection`, as `_gpr_normalize` builds it) in
    route slots two and three. The parameter words are those of
    core/gp_member.mojo.
  * `gp_member_pack` (PACK): the fitted attributes `_gpr_fit` writes, from
    the member state's blocks, including the RuntimeWarning of each tenfold
    ridge retry and `optimizer_runs_`.
  * `gp_member_state` (UNPACK): a member state from a fitted
    GaussianProcessRegressor.

SERIAL BY NATURE: one member's admission or publication (record reads and
one copy per fitted block); a caller runs its members in parallel.
"""
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from bindings.classical_arrays_door import classical_address
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.family_common_pack import family_array, family_call
from bindings.gp_common_native_pack import gp_numeric_mode_used, gp_with_free_values, gp_f32, gp_model_kernel
from bindings.gp_common_native_unpack import gp_kernel_arrays, gp_free
from bindings.gp_kernel_storage_boundary import GPPackedKernel
from bindings.gp_keys_unpack import gp_keys
from bindings.gp_native_boundary_door import GPConnection, gp_seed_word, gp_entry_key
from bindings.gp_normalization_boundary_unpack import GPNormalizationConnection
from bindings.gp_regressor_native_door import _gpr_validate, _gpr_validate_fields, _gpr_ext
from bindings.gp_classifier_native_door import (
    gpc_validate, gpc_validate_fields, _gpc_ext, _gpc_binary_result, _gpc_set_fitted, GPCFitPublication,
)
from bindings.member_owner_unpack import MemberOwner, member_labels_take, member_host_only
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_item, py_setitem, py_fields
from core.device_route import DeviceRoute, DEVICE_CPU, DEVICE_GPU
from core.entry_table import EntryRef
from core.gp_native_storage import GPFreeParameters
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_new, member_free,
    MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_IMPORT_DONE,
    MEMBER_F64,
)
from providers.python.estimator_binding_pyapi_door import estimator_binding
from providers.python.gp_public_lifecycle_pyapi_door import gp_host_class_binding
from providers.python.host_estimator_binding_pyapi_door import estimator_member_binding
from providers.python.estimator_lifecycle_pyapi_unpack import estimator_fit_complete

comptime _B_X = MEMBER_BLOCK_FAMILY
comptime _B_Y = MEMBER_BLOCK_FAMILY + 1
comptime _B_L = MEMBER_BLOCK_FAMILY + 2
comptime _B_DUAL = MEMBER_BLOCK_FAMILY + 3
comptime _B_KPARAMS = MEMBER_BLOCK_FAMILY + 4
comptime _B_KSCALES = MEMBER_BLOCK_FAMILY + 5
comptime _B_THETA = MEMBER_BLOCK_FAMILY + 6
comptime _B_VALUES = MEMBER_BLOCK_FAMILY + 7
comptime _B_RUNS = MEMBER_BLOCK_FAMILY + 8
comptime _B_PI = MEMBER_BLOCK_FAMILY + 9
comptime _B_WSR = MEMBER_BLOCK_FAMILY + 10
comptime _B_MEMBERS = MEMBER_BLOCK_FAMILY + 11
comptime _SCALARS = 16
comptime _GPR = "mojolearn GaussianProcessRegressor"
comptime _GPC = "mojolearn GaussianProcessClassifier"


def gp_member_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """UNPACK: fill `owner.plan.route`, `owner.plan.params` and `owner.pins`
    from the unfitted estimator. False: this kind or kernel has no member
    entry (a kernel table the member cannot address)."""
    if kind == 1:
        return _gpc_admit(model, context, owner)
    if kind != 0:
        return False
    classical_connect_numpy(context)
    # The estimator's own checks, on a SCRATCH copy of its fields: the private
    # fit settings (`_kernel`, `_alpha`, ...) they write land in the scratch
    # record, never in the caller's template estimator. PACK writes them into
    # the clone it packs (`_gpr_validate` on the clone).
    var b_mod = native_loaded_module("builtins")
    var settings = py_call(b_mod.dict, py_fields(model))
    _gpr_validate_fields(settings, context)
    var nk = gp_keys()
    var kernel = py_item(settings, nk[].k__kernel)
    # An explicit HostGaussianProcessRegressor (its own host-only word, lane mf-iterimp3b): the CPU identical
    # GP image (`estimator_member_binding`), never the device one; a wheel's CPU set without that image
    # refuses at admission and the caller keeps its own path. Any other class: its binding route.
    var binding = estimator_member_binding(model, "_mojolearn_gp", context)
    var ext = _gpr_ext(model, binding, context)
    var connection = GPConnection(ext)
    var packed = gp_kernel_arrays(kernel, context)
    var free = GPFreeParameters()
    gp_free(kernel, free, context)
    ref storage = packed.downcast_value_ptr[GPPackedKernel]()[].storage
    var nodes = len(storage.kinds)
    var scale_words = storage.scale_count
    # Node targets: 0 fixed, 1 a free parameter, 2 free length scales
    # (`gp_replace`'s walk); a table whose count disagrees is not served.
    var targets = List[Int]()
    var taken = 0
    for i in range(nodes):  # driver-loop: over the kernel's postfix nodes
        var target = 0
        if free.flags[i] != 0:
            target = 2 if storage.lengths[i] > 0 else 1
            taken += Int(storage.lengths[i]) if target == 2 else 1
        targets.append(target)
    if len(free.flags) != nodes or taken != len(free.values):
        return False
    var optimizer = py_item(settings, nk[].k__optimizer)
    var optimize = optimizer is not None and len(free.values) > 0
    var seed_source = py_item(settings, nk[].k__random_state)
    var bits = gp_seed_word(family_call(b_mod.int, PythonObject(0) if seed_source is None else seed_source))
    var normalize = Bool(py=py_item(settings, nk[].k__normalize_y))
    var words = List[Float64](length=12, fill=0.0)
    words[0] = 0.0
    words[1] = Float64(py=py_item(settings, nk[].k__alpha))
    words[2] = 1.0 if normalize else 0.0
    words[3] = 1.0 if optimize else 0.0
    words[4] = Float64(Int(py=py_item(settings, nk[].k__n_restarts_optimizer)))
    words[5] = Float64(Int(bits & UInt64(0xFFFFFFFF)))
    words[6] = Float64(Int(bits >> 32))
    words[8] = Float64(nodes)
    words[9] = Float64(scale_words)
    words[10] = Float64(len(free.values))
    words[11] = Float64(len(free.bounds))
    for i in range(nodes):  # driver-loop: kernel node kinds
        words.append(Float64(Int(storage.kinds[i])))
    for i in range(nodes):  # driver-loop: kernel node parameters
        words.append(Float64(storage.parameters[i]))
    for i in range(nodes):  # driver-loop: kernel node length-scale counts
        words.append(Float64(Int(storage.lengths[i])))
    for i in range(scale_words):  # driver-loop: kernel length scales
        words.append(Float64(storage.scales[i]))
    for i in range(nodes):  # driver-loop: kernel node targets
        words.append(Float64(targets[i]))
    for i in range(len(free.bounds)):  # driver-loop: free-value bounds
        words.append(Float64(free.bounds[i]))
    var native = connection.native.entry
    var vendor = gp_entry_key(binding).vendor
    var route = DeviceRoute(DEVICE_CPU if vendor == 0 else DEVICE_GPU)
    route.mode = connection.native.mode
    route.vendor = vendor
    # The entry `GPConnection` resolved (its table read and key check ran there).
    route.entry = native
    route.flags = connection.native.capabilities
    var pins = Python.list(binding, ext)
    if normalize:
        # `_gpr_normalize`'s connection, built once here.
        var scaler_context = py_item(context, nk[].k_family_gp_scaler)[1]
        classical_connect_numpy(scaler_context)
        var mode = gp_numeric_mode_used(binding, context)
        var norm = GPNormalizationConnection(py_item(scaler_context, nk[].k_scaler_bindings)[mode],
            1 if String(mode) == "identical" else 0)
        words[7] = 1.0 if norm.direct else 0.0
        route.second = EntryRef(norm.scaler.entry, 0)
        route.third = EntryRef(norm.program.entry, 0)
        route.has_second = True
        route.has_third = True
        _ = py_call(pins.append, norm.scaler_provider)
        _ = py_call(pins.append, norm.program_provider)
    owner.plan.route = route
    owner.plan.params = words^
    owner.pins = py_call(b_mod.tuple, pins)
    _ = packed^
    _ = free^
    return True


def _gpc_admit(model: PythonObject, context: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """UNPACK of GaussianProcessClassifier: its own checks on a scratch copy
    of its fields, its binding and mode check, the kernel's postfix table
    (fixed: optimizer=None is the only one admitted) and the GP provider
    entry, in the words of core/gp_member.mojo kind 1."""
    classical_connect_numpy(context)
    var b_mod = native_loaded_module("builtins")
    var settings = py_call(b_mod.dict, py_fields(model))
    gpc_validate_fields(settings, context)
    var nk = gp_keys()
    # An explicit HostGaussianProcessClassifier (its own host-only word, lane mf-cal3b): the image its own
    # doors bind (`gp_host_class_binding`). On the inference-only image a wheel ships, the member's fit reaches
    # the same unavailable Laplace entry its own fit reaches first (`gpc_fit_all`, else `gpc_fit`: the same
    # GP_CAP_OVR test, bindings/gp_classifier_native_door.mojo `_gpc_fit` and core/gp_member.mojo `_gpc_fit`)
    # and the native set raises those words; on the reference image it fits as its own fit does.
    var binding = (gp_host_class_binding(model, context) if member_host_only(model)
                   else estimator_binding(model, "_mojolearn_gp", context))
    var ext = _gpc_ext(model, binding, context)
    var connection = GPConnection(ext)
    var packed = gp_kernel_arrays(py_item(settings, nk[].k__kernel), context)
    ref storage = packed.downcast_value_ptr[GPPackedKernel]()[].storage
    var nodes = len(storage.kinds)
    var scale_words = storage.scale_count
    var words = List[Float64](length=12, fill=0.0)
    words[0] = 1.0
    words[1] = Float64(Int(py=py_item(settings, nk[].k__max_iter_predict)))
    words[8] = Float64(nodes)
    words[9] = Float64(scale_words)
    for i in range(nodes):  # driver-loop: kernel node kinds
        words.append(Float64(Int(storage.kinds[i])))
    for i in range(nodes):  # driver-loop: kernel node parameters
        words.append(Float64(storage.parameters[i]))
    for i in range(nodes):  # driver-loop: kernel node length-scale counts
        words.append(Float64(Int(storage.lengths[i])))
    for i in range(scale_words):  # driver-loop: kernel length scales
        words.append(Float64(storage.scales[i]))
    for _ in range(nodes):  # driver-loop: kernel node targets (all fixed)
        words.append(0.0)
    var vendor = gp_entry_key(binding).vendor
    var route = DeviceRoute(DEVICE_CPU if vendor == 0 else DEVICE_GPU)
    route.mode = connection.native.mode
    route.vendor = vendor
    route.entry = connection.native.entry
    route.flags = connection.native.capabilities
    owner.plan.route = route
    owner.plan.params = words^
    owner.pins = Python.tuple(binding, ext)
    _ = packed^
    return True


def _export(plan: MemberPlan, mut call: MemberCall, block: Int) raises -> MemberMatrix:
    call.block = block
    call.flags = 0
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    return call.dst


def _f32_array(plan: MemberPlan, mut call: MemberCall, block: Int, shape: PythonObject, count: Int,
               context: PythonObject) raises -> PythonObject:
    """A new public float32 array holding one exported block."""
    var view = _export(plan, call, block)
    if view.rows * view.cols != count:
        raise Error("RuntimeError: mojolearn: a Gp member block has the wrong size")
    return _f32_copy(view.data, shape, count, context)


def _f32_copy(address: Int, shape: PythonObject, count: Int, context: PythonObject) raises -> PythonObject:
    """A new public float32 array holding `count` words at `address`."""
    var array = family_array(shape, "<f4", context)
    if count > 0:
        unsafe_memcpy(dest=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(array))),
            src=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=address), count=count)
    return array


def _f64_words(plan: MemberPlan, mut call: MemberCall, block: Int) raises -> List[Float64]:
    var view = _export(plan, call, block)
    var count = view.rows * view.cols
    var words = List[Float64](length=count, fill=0.0)
    var source = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=view.data)
    for i in range(count):  # driver-loop: over one small float64 block (theta, values, runs)
        words[i] = source[i]
    return words^


def gp_member_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, state: Int) raises:
    """PACK: write the fitted attributes of state `state` into `model`, as
    `_gpr_fit` does (the same names its predict, score, sample_y and
    log_marginal_likelihood read); the classifier's as `_gpc_fit` does."""
    if plan.kind == 1:
        _gpc_pack(model, fields, context, b, plan, state)
        return
    classical_connect_numpy(context)
    var b_mod = native_loaded_module("builtins")
    var nk = gp_keys()
    _gpr_validate(model, context)
    var call = MemberCall(plan.route, state)
    var scalars_view = _export(plan, call, MEMBER_BLOCK_SCALARS)
    var s = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=scalars_view.data)
    var n = Int(s[1])
    var d = Int(s[2])
    var info = Int(s[3])
    var nb = Int(s[4])
    var logdet = s[5]
    var ydotalpha = s[6]
    var lml = s[7]
    var y_mean = s[8]
    var y_std = s[9]
    var normalize = s[10] != 0.0
    var alpha_used = s[11]
    var optimized = s[12] != 0.0
    # `_gpr_fit`'s ridge retries: the private `_alpha` left at the ridge used
    # (the same tenfold chain the member fit ran). Their RuntimeWarnings are
    # the state's fit notices, raised by the caller (core/member_notices.mojo).
    if alpha_used > Float64(py=py_item(fields, nk[].k__alpha)):
        py_setitem(fields, nk[].k__alpha, PythonObject(alpha_used))
    var x = _f32_array(plan, call, _B_X, Python.tuple(n, d), n * d, context)
    var y = _f32_array(plan, call, _B_Y, Python.tuple(n), n, context)
    var lf = _f32_array(plan, call, _B_L, Python.tuple(n, n), n * n, context)
    var dual = _f32_array(plan, call, _B_DUAL, Python.tuple(n), n, context)
    var kernel_fit = py_item(fields, nk[].k__kernel)
    var runs_out = Python.list()
    if optimized:
        var values = _f64_words(plan, call, _B_VALUES)
        var theta = _f64_words(plan, call, _B_THETA)
        var runs = _f64_words(plan, call, _B_RUNS)
        var stops = py_item(context, nk[].k_family_gp_opt_stops)
        if py_type(stops) is not b_mod.tuple and py_type(stops) is not b_mod.list:
            raise Error("TypeError: native GP optimizer stop metadata requires an exact sequence")
        for run in range(len(runs) // 4):  # driver-loop: pack public optimizer restart records
            _ = py_call(runs_out.append, Python.tuple(PythonObject(Int(runs[run * 4])),
                PythonObject(Int(runs[run * 4 + 1])), stops[Int(runs[run * 4 + 2])],
                PythonObject(-runs[run * 4 + 3])))
        kernel_fit = gp_with_free_values(kernel_fit, values, theta, context)
    py_setitem(fields, nk[].k_input_copied_, PythonObject(False))
    py_setitem(fields, nk[].k_optimizer_runs_, runs_out)
    py_setitem(fields, nk[].k_X_train_, x)
    py_setitem(fields, nk[].k_y_train_, y)
    py_setitem(fields, nk[].k__y_train_mean, PythonObject(y_mean))
    py_setitem(fields, nk[].k__y_train_std, PythonObject(y_std))
    py_setitem(fields, nk[].k_normalize_y_, PythonObject(normalize))
    py_setitem(fields, nk[].k_kernel_, kernel_fit)
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(d))
    py_setitem(fields, nk[].k_L_, lf)
    py_setitem(fields, nk[].k_alpha_, dual)
    py_setitem(fields, nk[].k_info_, PythonObject(info))
    py_setitem(fields, nk[].k_nb_, PythonObject(nb))
    py_setitem(fields, nk[].k_log_marginal_likelihood_value_, PythonObject(lml))
    py_setitem(fields, nk[].k__logdet, PythonObject(logdet))
    py_setitem(fields, nk[].k__ydotalpha, PythonObject(ydotalpha))
    estimator_fit_complete(model, PythonObject(None), False, context)


def _gpc_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, state: Int) raises:
    """PACK of GaussianProcessClassifier: `_gpc_set_fitted` over one
    `_BinaryLaplace` per fit, each fit's arrays copied out of the member's
    back-to-back words (the estimator's fit owns its arrays)."""
    classical_connect_numpy(context)
    var labels = member_labels_take(fields, b)
    # The private fit settings (`_kernel`, `_max_iter_predict`) land in the
    # clone here, never at admit.
    gpc_validate(model, context)
    var nk = gp_keys()
    var call = MemberCall(plan.route, state)
    var scalars_view = _export(plan, call, MEMBER_BLOCK_SCALARS)
    var s = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=scalars_view.data)
    var n = Int(s[1])
    var d = Int(s[2])
    var k = Int(s[3])
    var m = Int(s[4])
    var x = _f32_array(plan, call, _B_X, Python.tuple(n, d), n * d, context)
    var targets = _export(plan, call, _B_Y)
    var factors = _export(plan, call, _B_L)
    var pis = _export(plan, call, _B_PI)
    var wsrs = _export(plan, call, _B_WSR)
    var fit_words = _export(plan, call, _B_MEMBERS)
    if (targets.rows * targets.cols != m * n or factors.rows * factors.cols != m * n * n
            or pis.rows * pis.cols != m * n or wsrs.rows * wsrs.cols != m * n
            or fit_words.rows * fit_words.cols != 3 * m):
        raise Error("RuntimeError: mojolearn: a Gp classifier member block has the wrong size")
    var w = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=fit_words.data)
    var fits = GPCFitPublication(m)
    for i in range(m):  # driver-loop: pack each binary fit's public arrays
        var y01 = _f32_copy(targets.data + 4 * i * n, Python.tuple(n), n, context)
        var lf = _f32_copy(factors.data + 4 * i * n * n, Python.tuple(n, n), n * n, context)
        var pi = _f32_copy(pis.data + 4 * i * n, Python.tuple(n), n, context)
        var wsr = _f32_copy(wsrs.data + 4 * i * n, Python.tuple(n), n, context)
        fits.append(_gpc_binary_result(context, y01, lf, pi, wsr, w[3 * i], Int(w[3 * i + 1]), Int(w[3 * i + 2])))
    if labels is None:
        # No labels from the caller: the codes are the classes.
        labels = Python.list()
        for c in range(k):  # driver-loop: over the class codes
            _ = py_call(labels.append, PythonObject(c))
    py_setitem(fields, nk[].k_input_copied_, PythonObject(False))
    _gpc_set_fitted(fields, x, labels, fits, context)
    estimator_fit_complete(model, PythonObject(None), False, context)


def _gpc_concat(fits: PythonObject, key: PythonObject, name: String, per: Int,
                context: PythonObject) raises -> List[Float32]:
    """One fitted array of every binary fit, back to back (the member's
    import layout)."""
    var m = len(fits)
    var words = List[Float32](length=max(1, m * per), fill=Float32(0))
    for i in range(m):  # driver-loop: over the fitted binary fits
        var array = gp_f32(py_fields(fits[i]), key, name, -1, context)
        if Int(py=array.size) != per:
            raise Error("ValueError: " + String(_GPC) + ": the fitted " + name + " does not match X_train_")
        if per > 0:
            unsafe_memcpy(dest=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(words.unsafe_ptr()) + 4 * i * per),
                src=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(array))), count=per)
        _ = array^
    return words^


def _gpc_state(fields: PythonObject, context: PythonObject, plan: MemberPlan) raises -> Int:
    """UNPACK: a classifier member state from a fitted
    GaussianProcessClassifier. The caller frees it (`member_free`)."""
    classical_connect_numpy(context)
    var nk = gp_keys()
    var xt = gp_f32(fields, nk[].k_X_train_, "X_train_", 2, context)
    var n = Int(py=xt.shape[0])
    var d = Int(py=xt.shape[1])
    var fits = py_item(fields, nk[].k_estimators_)
    var m = len(fits)
    var k = Int(py=py_item(fields, nk[].k_n_classes_))
    if k < 2 or m != (1 if k == 2 else k):
        raise Error("ValueError: " + String(_GPC) + ": the fitted estimators_ do not match n_classes_")
    var targets = _gpc_concat(fits, nk[].k_y_train_, "y_train_", n, context)
    var factors = _gpc_concat(fits, nk[].k_L_, "L_", n * n, context)
    var pis = _gpc_concat(fits, nk[].k_pi_, "pi_", n, context)
    var wsrs = _gpc_concat(fits, nk[].k_W_sr_, "W_sr_", n, context)
    var fit_words = List[Float64](length=3 * m, fill=0.0)
    for i in range(m):  # driver-loop: over the fitted binary fits (their scalars)
        var own = py_fields(fits[i])
        fit_words[3 * i] = Float64(py=py_item(own, nk[].k_log_marginal_likelihood_value_))
        fit_words[3 * i + 1] = Float64(Int(py=py_item(own, nk[].k_n_iter_)))
        fit_words[3 * i + 2] = Float64(Int(py=py_get(own, nk[].k_nb_, PythonObject(0))))
    var packed = gp_kernel_arrays(gp_model_kernel(fields), context)
    ref storage = packed.downcast_value_ptr[GPPackedKernel]()[].storage
    var scalars = List[Float64](length=_SCALARS, fill=0.0)
    scalars[0] = 1.0
    scalars[1] = Float64(n)
    scalars[2] = Float64(d)
    scalars[3] = Float64(k)
    scalars[4] = Float64(m)
    scalars[5] = plan.params[1] if len(plan.params) > 1 else 0.0
    scalars[15] = Float64(len(storage.kinds))
    var call = member_new(plan)
    try:
        call.flags = 0
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, _SCALARS, MEMBER_F64)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = _B_X
        call.x = MemberMatrix(Int(py=classical_address(xt)), n, d)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = _B_Y
        call.x = MemberMatrix(Int(targets.unsafe_ptr()), m, n)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = _B_L
        call.x = MemberMatrix(Int(factors.unsafe_ptr()), m * n, n)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = _B_PI
        call.x = MemberMatrix(Int(pis.unsafe_ptr()), m, n)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = _B_WSR
        call.x = MemberMatrix(Int(wsrs.unsafe_ptr()), m, n)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = _B_MEMBERS
        call.x = MemberMatrix(Int(fit_words.unsafe_ptr()), m, 3, MEMBER_F64)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = _B_KPARAMS
        call.x = MemberMatrix(Int(storage.parameters.unsafe_ptr()), 1, len(storage.parameters))
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = _B_KSCALES
        call.x = MemberMatrix(Int(storage.scales.unsafe_ptr()), 1, len(storage.scales))
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = xt^
    _ = packed^
    _ = len(scalars)
    _ = len(targets)
    _ = len(factors)
    _ = len(pis)
    _ = len(wsrs)
    _ = len(fit_words)
    return call.state


def gp_member_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan) raises -> Int:
    """UNPACK: a new member state holding the fitted attributes of `model`.
    The caller frees it (`member_free`)."""
    if plan.kind == 1:
        return _gpc_state(fields, context, plan)
    classical_connect_numpy(context)
    var nk = gp_keys()
    var xt = gp_f32(fields, nk[].k_X_train_, "X_train_", 2, context)
    var lf = gp_f32(fields, nk[].k_L_, "L_", -1, context)
    var dual = gp_f32(fields, nk[].k_alpha_, "alpha_", -1, context)
    var n = Int(py=xt.shape[0])
    var d = Int(py=xt.shape[1])
    if Int(py=lf.size) != n * n or Int(py=dual.size) != n:
        raise Error("ValueError: " + String(_GPR) + ": the fitted L_ and alpha_ do not match X_train_")
    var packed = gp_kernel_arrays(gp_model_kernel(fields), context)
    ref storage = packed.downcast_value_ptr[GPPackedKernel]()[].storage
    var b_mod = native_loaded_module("builtins")
    var normalize = Bool(py=py_call(b_mod.bool, py_get(fields, nk[].k_normalize_y_, PythonObject(False))))
    var scalars = List[Float64](length=_SCALARS, fill=0.0)
    scalars[1] = Float64(n)
    scalars[2] = Float64(d)
    scalars[3] = Float64(Int(py=py_item(fields, nk[].k_info_)))
    scalars[4] = Float64(Int(py=py_get(fields, nk[].k_nb_, PythonObject(0))))
    scalars[8] = Float64(py=py_item(fields, nk[].k__y_train_mean)) if normalize else 0.0
    scalars[9] = Float64(py=py_item(fields, nk[].k__y_train_std)) if normalize else 1.0
    scalars[10] = 1.0 if normalize else 0.0
    scalars[15] = Float64(len(storage.kinds))
    var call = member_new(plan)
    try:
        call.flags = 0
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, _SCALARS, MEMBER_F64)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = _B_X
        call.x = MemberMatrix(Int(py=classical_address(xt)), n, d)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = _B_L
        call.x = MemberMatrix(Int(py=classical_address(lf)), n, n)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = _B_DUAL
        call.x = MemberMatrix(Int(py=classical_address(dual)), 1, n)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = _B_KPARAMS
        call.x = MemberMatrix(Int(storage.parameters.unsafe_ptr()), 1, len(storage.parameters))
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = _B_KSCALES
        call.x = MemberMatrix(Int(storage.scales.unsafe_ptr()), 1, len(storage.scales))
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = xt^
    _ = lf^
    _ = dual^
    _ = packed^
    _ = len(scalars)
    return call.state
