# SPDX-License-Identifier: Apache-2.0
# The public parallel boundary owns admission, binding selection, preparation
# and consumed-output conversion. Fixed ABI plans are connected at import;
# no Python estimator methods, _bind hooks or public-result helpers execute.
# UNPACK and PACK only: every refusal that is a comparison on a word or a
# number, and the recipe / route codes, are typed in
# bindings/parallel_public_native_run.mojo; the device indices are admitted by
# bindings/parallel_plan_run.mojo. The member's compiled entry is reached
# through `family_call` (a boundary call) until member families publish entry
# tables.
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.family_common_pack import family_call
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.classical_pandas_door import classical_admit_value
from bindings.admitted_input_unpack import admit_x
from bindings.parallel_public_result_door import parallel_public_result
from bindings.msel_member_device_unpack import member_says_cpu, member_multi_gpu_refusal
from bindings.native_buffer_owner_pack import native_storage_allocate
from bindings.trees_public_forest_door import forest_effective_mode, _engine
from bindings.parallel_plan_door import plan_indices
from bindings.parallel_plan_run import PLAN_DEVICES_REFUSAL, plan_indices_admit
from bindings.parallel_public_native_run import (
    PUBLIC_CHECK_GRAM, PUBLIC_CHECK_KMEANS, PUBLIC_CHECK_GMM_PREDICT, PUBLIC_CHECK_POSTERIOR, PUBLIC_CHECK_IFOREST,
    PUBLIC_CHECK_UMAP_TRANSFORM, PUBLIC_CHECK_RESAMPLE,
    PUBLIC_ITEM_ARG, PUBLIC_ITEM_CONSTANT, PUBLIC_ITEM_MODEL, PUBLIC_ITEM_BINDING,
    PUBLIC_MODE_DEFAULT, PUBLIC_MODE_EFFECTIVE, PUBLIC_ROUTE_DIRECT, PUBLIC_ROUTE_SHARD,
    public_check_code, public_check_admits_x, public_item_code, public_policy_code, public_binding_mode_code,
    public_route_code, public_identical_admit, public_gram_columns_admit, public_gram_pca_admit,
    public_gram_weight_admit, public_kmeans_metric_admit, public_kmeans_oversampling_admit,
    public_module_refusal_is_optional, public_predictor_route_admit, public_predictor_devices_admit,
    public_predictor_open_admit, public_predictor_proba_admit,
)
from bindings.parallel_keys_unpack import (
    parallel_keys,
    PARALLEL_PREDICTOR_OPN_PREDICTOR_CLOSE,
    PARALLEL_PREDICTOR_OPN_PREDICTOR_ENTER,
    PARALLEL_PREDICTOR_OPN_PREDICTOR_EXIT,
    PARALLEL_PREDICTOR_OPN_PREDICTOR_INIT,
    PARALLEL_PREDICTOR_OPN_PREDICTOR_PREDICT,
    PARALLEL_PREDICTOR_OPN_PREDICTOR_PREDICT_PROBA,
)
from bindings.py_keys import py_op_code
from bindings.py_balanced import py_item


def _fields(model: PythonObject, plan: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var cls = py_type(model)
    if py_type(cls) is not b.type:
        raise Error("TypeError: native parallel drivers refuse Python metaclass hooks")
    var nk = parallel_keys()
    var ancestry = py_call(py_item(b.type.__dict__, nk[].k___mro__).__get__, cls)
    if len(ancestry) != len(py_item(plan, nk[].k_mro)):
        raise Error("TypeError: parallel estimator class ancestry changed")
    for i in range(len(ancestry)):  # driver-loop: fixed class metadata
        if ancestry[i] is not py_item(plan, nk[].k_mro)[i]:
            raise Error("TypeError: parallel estimator class ancestry changed")
    var hooks = py_item(plan, nk[].k_hooks)
    for i in range(len(hooks)):  # driver-loop: fixed native storage identities
        var row = py_call(py_item(b.type.__dict__, nk[].k___dict__).__get__, hooks[i][0])
        if py_get(row, hooks[i][1]) is not hooks[i][2]:
            raise Error("TypeError: parallel estimator methods or storage hooks changed")
    var fields = py_call(py_item(plan, nk[].k_fields_get), model, cls)
    var keys = py_call(b.list, fields)
    for i in range(len(keys)):  # driver-loop: keys before dictionary lookup
        if py_type(keys[i]) is not b.str:
            raise Error("TypeError: parallel estimator state requires exact string keys")
    return fields


def _entry(entry: PythonObject) raises -> PythonObject:
    if py_type(entry) is not py_type(native_loaded_module("builtins").len):
        raise Error("ImportError: parallel driver requires a connected compiled entry")
    return entry


def _devices(value: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if py_type(value) is not b.tuple and py_type(value) is not b.list and py_type(value) is not b.range:
        raise Error("TypeError: devices requires a native tuple, list or range; Python iteration hooks are unsupported")
    var devices = py_call(b.tuple, value)
    if len(devices) == 0:
        raise Error(PLAN_DEVICES_REFUSAL)
    plan_indices_admit(plan_indices(devices, b))
    return devices


def _module(meta: PythonObject, name: PythonObject, mode: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if py_type(mode) is not b.str:
        raise Error("TypeError: numeric_mode requires an exact string or None")
    var normalized = py_call(py_call(mode.strip).lower)
    var nk = parallel_keys()
    var found = py_get(py_item(meta, nk[].k_modules)[name], normalized)
    if found is None:
        raise Error("ValueError: numeric_mode must be identical or fast")
    if py_type(found) is b.str:
        raise Error(String(found))
    return found


def _modules(meta: PythonObject, names: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var result = Python.list()
    var nk = parallel_keys()
    if not Bool(py=py_item(meta, nk[].k_cpu_only)):
        for i in range(len(names)):  # driver-loop: the owner's fixed binding list
            var module = py_item(py_item(meta, nk[].k_modules)[names[i]], nk[].k_identical)
            if py_type(module) is b.str:
                # Match plan_bindings' optional-module refusals. A readable
                # mode mismatch or invalid native export must still stop the
                # operation even if another listed module is available.
                var error = String(module)
                if not public_module_refusal_is_optional(error):
                    raise Error(error)
            else:
                _ = py_call(result.append, module)
    return py_call(b.tuple, result)


def _run(model: PythonObject, driver: PythonObject, operation: PythonObject, args: PythonObject,
          entry: PythonObject, owner: PythonObject, names: PythonObject, devices: PythonObject,
          context: PythonObject) raises -> PythonObject:
    var nk = parallel_keys()
    var meta = py_item(context, nk[].k_parallel_public)
    return family_call(_entry(py_item(meta, nk[].k_family)), model, "family_par_run",
                        Python.tuple(driver, devices, operation, args, entry, owner, _modules(meta, names)), context)


def _predictor(model: PythonObject, operation: Int, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`operation` is a `PARALLEL_PREDICTOR_OPN_*` code; any other
    `predictor_*` name runs as `predictor_predict` (the former chain's last
    branch)."""
    var b = native_loaded_module("builtins")
    var nk = parallel_keys()
    var meta = py_item(context, nk[].k_parallel_public)
    if py_type(model) is not py_item(meta, nk[].k_predictor_class):
        raise Error("TypeError: requires the native ParallelForestPredictor class")
    var fields = _fields(model, py_item(meta, nk[].k_predictor_layout))
    if operation == PARALLEL_PREDICTOR_OPN_PREDICTOR_INIT:
        var source = args[0]
        var cls = py_type(source)
        if py_type(cls) is not b.type or not py_contains(py_item(meta, nk[].k_forests), cls):
            raise Error("TypeError: ParallelForestPredictor requires a mojolearn RandomForest or ExtraTrees estimator")
        var original = _fields(source, py_item(meta, nk[].k_classes)[cls])
        var required = Python.tuple("_offsets", "_colid", "_quesval", "_left_child", "_leaves", "_n_trees", "_num_outputs", "n_features_in_")
        for i in range(len(required)):  # driver-loop: fixed fitted metadata
            if not py_contains(original, required[i]):
                raise Error("RuntimeError: ParallelForestPredictor requires a fitted estimator")
        classical_admit_value(Python.tuple(py_get(original, nk[].k_numeric_mode), py_get(original, nk[].k__fit_numeric_mode),
                              py_get(original, nk[].k_inference_engine)), context)
        var mode = forest_effective_mode(original, py_item(meta, nk[].k_trees_context))
        # The engine is read only for an identical forest, as before.
        public_predictor_route_admit(mode, _engine(original, mode) if mode == "identical" else String(""))
        # Constructor admission historically allows the CPU case; the query's
        # cooperative driver supplies its route refusal before GPU execution.
        var devices = _devices(args[1])
        public_predictor_devices_admit(len(devices))
        # Admit the stored values consumed below before their scalar/string
        # conversion; an injected Python value must never execute its hooks.
        classical_admit_value(Python.tuple(py_item(original, nk[].k_n_features_in_), py_item(original, nk[].k__n_trees), py_item(original, nk[].k__num_outputs)), context)
        var snapshot = native_storage_allocate(cls)
        var saved = py_call(py_item(py_item(meta, nk[].k_classes)[cls], nk[].k_fields_get), snapshot, cls)
        _ = py_call(saved.update, original)
        _ = py_call(saved.pop, "_resident_forest", PythonObject(None))
        py_setitem(saved, nk[].k__resident_holder, Python.dict())
        py_setitem(fields, nk[].k__model, snapshot)
        py_setitem(fields, nk[].k__lock, py_call(py_item(meta, nk[].k_lock_type)))
        py_setitem(fields, nk[].k__closed, PythonObject(False))
        var classifier = py_contains(py_item(meta, nk[].k_forest_classifier), cls)
        py_setitem(fields, nk[].k__classifier, PythonObject(classifier))
        py_setitem(fields, nk[].k__public_kind, cls)
        py_setitem(fields, nk[].k_devices, devices)
        py_setitem(fields, nk[].k_n_features_in_, PythonObject(Int(py=py_item(original, nk[].k_n_features_in_))))
        py_setitem(fields, nk[].k_n_estimators_, PythonObject(Int(py=py_item(original, nk[].k__n_trees))))
        if classifier:
            py_setitem(fields, nk[].k_classes_, py_item(original, nk[].k_classes_))
            py_setitem(fields, nk[].k_n_classes_, PythonObject(Int(py=py_item(original, nk[].k__num_outputs))))
        return PythonObject(None)
    var lock = py_get(fields, nk[].k__lock)
    if lock is None and (operation == PARALLEL_PREDICTOR_OPN_PREDICTOR_CLOSE or operation == PARALLEL_PREDICTOR_OPN_PREDICTOR_EXIT):
        # A CONSTRUCTION THAT FAILED before its lock (every refusal of
        # predictor_init above runs before `_model` and `_lock` are stored):
        # nothing resident was built, since the pool is built on the first
        # predict. close is safe and idempotent there; drop whatever exists
        # (cpu-surface-defects, 2026-11-16).
        py_setitem(fields, nk[].k__model, PythonObject(None))
        py_setitem(fields, nk[].k__closed, PythonObject(True))
        return PythonObject(None)
    if py_type(lock) is not py_item(meta, nk[].k_lock_type):
        raise Error("TypeError: ParallelForestPredictor requires its native lock")
    if py_type(py_get(fields, nk[].k__closed)) is not b.bool or py_type(py_get(fields, nk[].k__classifier)) is not b.bool:
        raise Error("TypeError: ParallelForestPredictor state requires native booleans")
    _ = py_call(lock.acquire)
    try:
        if operation == PARALLEL_PREDICTOR_OPN_PREDICTOR_CLOSE or operation == PARALLEL_PREDICTOR_OPN_PREDICTOR_EXIT:
            if not Bool(py=py_item(fields, nk[].k__closed)):
                py_setitem(fields, nk[].k__closed, PythonObject(True))
                var snapshot = py_item(fields, nk[].k__model)
                var saved = _fields(snapshot, py_item(meta, nk[].k_classes)[py_type(snapshot)])
                # Dropping the compiled TreesResident owner releases its GPU
                # handle once; no Python weakref.finalize callback is invoked.
                py_setitem(saved, nk[].k__resident_holder, Python.dict())
                py_setitem(fields, nk[].k__model, PythonObject(None))
            return PythonObject(None)
        public_predictor_open_admit(Bool(py=py_item(fields, nk[].k__closed)))
        if operation == PARALLEL_PREDICTOR_OPN_PREDICTOR_ENTER:
            return model
        var proba = operation == PARALLEL_PREDICTOR_OPN_PREDICTOR_PREDICT_PROBA
        public_predictor_proba_admit(proba, Bool(py=py_item(fields, nk[].k__classifier)))
        classical_admit_value(args, context)
        var snapshot = py_item(fields, nk[].k__model)
        var result = _run(snapshot, "forest_predict", "trees_public_forest_proba" if proba else "trees_public_forest_predict",
                          Python.tuple(args[0]), _entry(py_item(meta, nk[].k_trees_entry)), py_item(meta, nk[].k_trees_context),
                          Python.tuple("_mojolearn_rf", "_mojolearn_trees"), py_item(fields, nk[].k_devices), context)
        return parallel_public_result(result, "predict_proba" if proba else "predict", py_item(py_item(meta, nk[].k_classes)[py_type(snapshot)], nk[].k_kind), context)
    finally:
        _ = py_call(lock.release)


def parallel_public_operation(model: PythonObject, packed: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var nk = parallel_keys()
    var meta = py_item(context, nk[].k_parallel_public)
    if py_type(packed[0]) is not b.str:
        raise Error("TypeError: parallel operation requires an exact name")
    var args = py_call(b.list, packed[1])
    classical_connect_numpy(context)
    # One lookup of the caller's own str: a predictor operation by its code;
    # a plan by its name. Only a name that is neither builds its String (an
    # other `predictor_*` name runs as predict, as the prefix test ran it;
    # any other raises the plans' KeyError, as before).
    var predictor = py_op_code(nk[].ops_predictor, packed[0])
    if predictor >= 0:
        return _predictor(model, predictor, py_call(b.tuple, args), context)
    var plans = py_item(meta, nk[].k_plans)
    var options = py_get(plans, packed[0])
    if options is None:
        if String(packed[0]).startswith("predictor_"):
            return _predictor(model, PARALLEL_PREDICTOR_OPN_PREDICTOR_PREDICT, py_call(b.tuple, args), context)
        options = plans[packed[0]]
    var cls = py_type(model) if model is not None else PythonObject(None)
    if model is not None and py_type(cls) is not b.type:
        raise Error("TypeError: parallel drivers refuse Python metaclass hooks")
    var spec = py_get(options, cls)
    if spec is None:
        var first = py_call(b.list, options.values())[0]
        raise Error("TypeError: " + String(py_item(first, nk[].k_error)))
    var device_at = Int(py=py_item(spec, nk[].k_device_arg))
    py_setitem(args, PythonObject(device_at), _devices(args[device_at]))
    classical_admit_value(args, context)
    var fields = _fields(model, py_item(meta, nk[].k_classes)[cls]) if model is not None else Python.dict()
    # The member's own device (device wave, 2026-11-21): every driver here
    # runs the estimator on the GPUs in `devices=` and has no host form. A
    # member that says device='cpu' in a process with a GPU set is refused in
    # words, before the mode checks, the input admission and any worker; it
    # used to be run on the GPUs whatever it said. None, a class without the
    # parameter and 'gpu' go on as before. The shared one read (`device_request`).
    if model is not None and member_says_cpu(fields, b):
        if not Bool(py=py_item(meta, nk[].k_cpu_only)):
            member_multi_gpu_refusal(String(packed[0]))
    var mode = py_get(fields, nk[].k_numeric_mode)
    var process_mode = py_item(py_item(meta, nk[].k_backend), nk[].k__DEFAULT_MODE)
    var effective = process_mode if mode is None else mode
    classical_admit_value(Python.tuple(mode, process_mode), context)
    if py_item(spec, nk[].k_mode) is not None:
        var policy = public_policy_code(String(py_item(spec, nk[].k_mode)))
        var selected = process_mode if policy == PUBLIC_MODE_DEFAULT else effective if policy == PUBLIC_MODE_EFFECTIVE else mode
        if selected is not None:
            var exact = py_type(selected) is b.str
            public_identical_admit(True, exact, String(selected) if exact else String(""),
                                   "parallel " + String(packed[0]) + " requires IDENTICAL numeric mode")
    var check = public_check_code(String(py_item(spec, nk[].k_check))) if py_item(spec, nk[].k_check) is not None else public_check_code("")
    if public_check_admits_x(check):
        var data = admit_x(args[0], 2, "X", context)
        py_setitem(args, PythonObject(0), data)
        if check == PUBLIC_CHECK_GRAM:
            public_gram_columns_admit(Int(py=data.shape[1]))
            var solver = py_get(fields, nk[].k_svd_solver)
            classical_admit_value(solver, context)
            public_gram_pca_admit(cls is py_item(meta, nk[].k_pca) and py_contains(py_item(meta, nk[].k_dense_solvers), solver),
                                  Int(py=data.shape[0]), Int(py=data.shape[1]))
            public_gram_weight_admit(args[3] is not None, args[3] is not None and py_contains(py_item(meta, nk[].k_decompositions), cls))
        elif check == PUBLIC_CHECK_KMEANS:
            var metric = py_get(fields, nk[].k_metric, PythonObject("euclidean"))
            var oversample = py_get(fields, nk[].k_oversampling_factor, PythonObject(2.0))
            classical_admit_value(Python.tuple(metric, oversample), context)
            var exact_metric = py_type(metric) is b.str
            public_kmeans_metric_admit(exact_metric, String(metric) if exact_metric else String(""))
            public_kmeans_oversampling_admit(Float64(py=oversample))
            py_setitem(fields, nk[].k_numeric_mode, PythonObject("identical"))
            effective = PythonObject("identical")
        elif check == PUBLIC_CHECK_GMM_PREDICT and not py_contains(fields, nk[].k_weights_):
            raise Error("ValueError: GaussianMixture is not fitted")
    elif check == PUBLIC_CHECK_POSTERIOR and py_type(args[2]) is not b.bool:
        raise Error("ValueError: return_std must be bool")
    elif check == PUBLIC_CHECK_IFOREST and args[2] is not None:
        raise Error("NotImplementedError: IsolationForest does not support sample_weight")
    elif check == PUBLIC_CHECK_UMAP_TRANSFORM:
        var transform_mode = py_get(fields, nk[].k__transform_mode)
        var exact = py_type(transform_mode) is b.str
        public_identical_admit(True, exact, String(transform_mode) if exact else String(""), "requires an IDENTICAL fitted UMAP")
    elif check == PUBLIC_CHECK_RESAMPLE:
        var index = Int(py=py_item(spec, nk[].k_numeric_mode_arg))
        var requested = args[index]
        if requested is not None:
            var exact = py_type(requested) is b.str
            public_identical_admit(True, exact, String(requested) if exact else String(""),
                                   "parallel resampling requires IDENTICAL numeric mode")
    var owner_op = py_item(spec, nk[].k_operation)
    var method = py_item(spec, nk[].k_output)
    if py_item(spec, nk[].k_method_arg) is not None:
        var requested = args[Int(py=py_item(spec, nk[].k_method_arg))]
        if py_type(requested) is not b.str:
            raise Error("ValueError: parallel method must be an exact supported name")
        owner_op = py_get(owner_op, requested)
        if owner_op is None:
            raise Error("ValueError: unsupported parallel prediction method")
        method = requested
    var values = Python.list()
    var recipe = py_item(spec, nk[].k_recipe)
    for i in range(len(recipe)):  # driver-loop: the fixed native owner ABI
        var item = recipe[i]
        var kind = public_item_code(String(item[0]))
        var value = PythonObject(None)
        if kind == PUBLIC_ITEM_ARG:
            value = args[Int(py=item[1])]
        elif kind == PUBLIC_ITEM_CONSTANT:
            value = item[1]
        elif kind == PUBLIC_ITEM_MODEL:
            value = model
        elif kind == PUBLIC_ITEM_BINDING:
            var route = public_binding_mode_code(String(item[2]))
            var selected = effective if route == PUBLIC_MODE_EFFECTIVE else process_mode if route == PUBLIC_MODE_DEFAULT else item[2]
            value = _module(meta, item[1], selected)
        _ = py_call(values.append, value)
    var owner_args = py_call(b.tuple, values)
    var result = PythonObject(None)
    var route = public_route_code(String(py_item(spec, nk[].k_route)))
    if route == PUBLIC_ROUTE_DIRECT:
        result = family_call(_entry(py_item(meta, nk[].k_family)), model, owner_op, owner_args, context)
    elif route == PUBLIC_ROUTE_SHARD:
        result = family_call(_entry(py_item(meta, nk[].k_shard)), owner_op, owner_args)
    else:
        # All cooperative wrappers put their device tuple in the connected
        # argument position; no Python loop or method constructs this envelope.
        result = _run(model, py_item(spec, nk[].k_driver), owner_op, owner_args, py_item(spec, nk[].k_entry), py_item(spec, nk[].k_owner), py_item(spec, nk[].k_modules), args[device_at], context)
    if method is not None:
        result = parallel_public_result(result, String(method), py_item(py_item(meta, nk[].k_classes)[cls], nk[].k_kind), context)
    return result
