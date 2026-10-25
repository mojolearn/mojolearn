# SPDX-License-Identifier: Apache-2.0
# The public parallel boundary owns admission, binding selection, preparation
# and consumed-output conversion. Fixed ABI plans are connected at import;
# no Python estimator methods, _bind hooks or public-result helpers execute.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.family_common import family_call
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_pandas import classical_admit_value
from bindings.classical_arrays import classical_host_f32
from bindings.parallel_public_result import parallel_public_result
from bindings.native_buffer_owner import native_storage_allocate
from bindings.trees_public_forest import forest_effective_mode, _engine


def _fields(model: PythonObject, plan: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var cls = py_type(model)
    if py_type(cls) is not b.type:
        raise Error("TypeError: native parallel drivers refuse Python metaclass hooks")
    var ancestry = py_call(b.type.__dict__["__mro__"].__get__, cls)
    if len(ancestry) != len(plan["mro"]):
        raise Error("TypeError: parallel estimator class ancestry changed")
    for i in range(len(ancestry)):  # driver-loop: fixed class metadata
        if ancestry[i] is not plan["mro"][i]:
            raise Error("TypeError: parallel estimator class ancestry changed")
    var hooks = plan["hooks"]
    for i in range(len(hooks)):  # driver-loop: fixed native storage identities
        var row = py_call(b.type.__dict__["__dict__"].__get__, hooks[i][0])
        if py_get(row, hooks[i][1]) is not hooks[i][2]:
            raise Error("TypeError: parallel estimator methods or storage hooks changed")
    var fields = py_call(plan["fields_get"], model, cls)
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
        raise Error("ValueError: devices must be distinct nonnegative integer indices")
    for i in range(len(devices)):  # driver-loop: admitted native device indices
        if py_type(devices[i]) is not b.int or Int(py=devices[i]) < 0:
            raise Error("ValueError: devices must be distinct nonnegative integer indices")
        for j in range(i):  # driver-loop: distinct selected indices
            if Int(py=devices[i]) == Int(py=devices[j]):
                raise Error("ValueError: devices must be distinct nonnegative integer indices")
    return devices


def _module(meta: PythonObject, name: PythonObject, mode: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if py_type(mode) is not b.str:
        raise Error("TypeError: numeric_mode requires an exact string or None")
    var normalized = py_call(py_call(mode.strip).lower)
    var found = py_get(meta["modules"][name], normalized)
    if found is None:
        raise Error("ValueError: numeric_mode must be identical or fast")
    if py_type(found) is b.str:
        raise Error(String(found))
    return found


def _modules(meta: PythonObject, names: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var result = Python.list()
    if not Bool(py=meta["cpu_only"]):
        for i in range(len(names)):  # driver-loop: the owner's fixed binding list
            var module = meta["modules"][names[i]]["identical"]
            if py_type(module) is b.str:
                # Match plan_bindings' optional-module refusals. A readable
                # mode mismatch or invalid native export must still stop the
                # operation even if another listed module is available.
                var error = String(module)
                if not error.startswith("ImportError:") and not error.startswith("AttributeError:") and not error.startswith("ValueError:"):
                    raise Error(error)
            else:
                _ = py_call(result.append, module)
    return py_call(b.tuple, result)


def _run(model: PythonObject, driver: PythonObject, operation: PythonObject, args: PythonObject,
          entry: PythonObject, owner: PythonObject, names: PythonObject, devices: PythonObject,
          context: PythonObject) raises -> PythonObject:
    var meta = context["parallel_public"]
    return family_call(_entry(meta["family"]), model, "family_par_run",
                        Python.tuple(driver, devices, operation, args, entry, owner, _modules(meta, names)), context)


def _predictor(model: PythonObject, operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var meta = context["parallel_public"]
    if py_type(model) is not meta["predictor_class"]:
        raise Error("TypeError: requires the native ParallelForestPredictor class")
    var fields = _fields(model, meta["predictor_layout"])
    if operation == "predictor_init":
        var source = args[0]
        var cls = py_type(source)
        if py_type(cls) is not b.type or not py_contains(meta["forests"], cls):
            raise Error("TypeError: ParallelForestPredictor requires a mojolearn RandomForest or ExtraTrees estimator")
        var original = _fields(source, meta["classes"][cls])
        var required = Python.tuple("_offsets", "_colid", "_quesval", "_left_child", "_leaves", "_n_trees", "_num_outputs", "n_features_in_")
        for i in range(len(required)):  # driver-loop: fixed fitted metadata
            if not py_contains(original, required[i]):
                raise Error("RuntimeError: ParallelForestPredictor requires a fitted estimator")
        classical_admit_value(Python.tuple(py_get(original, "numeric_mode"), py_get(original, "_fit_numeric_mode"),
                              py_get(original, "inference_engine")), context)
        var mode = forest_effective_mode(original, meta["trees_context"])
        if mode != "identical":
            raise Error("ValueError: ParallelForestPredictor requires IDENTICAL numeric mode")
        if _engine(original, mode) != "parallel_groves":
            raise Error("ValueError: ParallelForestPredictor requires inference_engine='parallel_groves'")
        # Constructor admission historically allows the CPU case; the query's
        # cooperative driver supplies its route refusal before GPU execution.
        var devices = _devices(args[1])
        if len(devices) > 64:
            raise Error("ValueError: ParallelForestPredictor supports at most 64 devices")
        # Admit the stored values consumed below before their scalar/string
        # conversion; an injected Python value must never execute its hooks.
        classical_admit_value(Python.tuple(original["n_features_in_"], original["_n_trees"], original["_num_outputs"]), context)
        var snapshot = native_storage_allocate(cls)
        var saved = py_call(meta["classes"][cls]["fields_get"], snapshot, cls)
        _ = py_call(saved.update, original)
        _ = py_call(saved.pop, "_resident_forest", PythonObject(None))
        py_setitem(saved, "_resident_holder", Python.dict())
        py_setitem(fields, "_model", snapshot)
        py_setitem(fields, "_lock", py_call(meta["lock_type"]))
        py_setitem(fields, "_closed", PythonObject(False))
        var classifier = py_contains(meta["forest_classifier"], cls)
        py_setitem(fields, "_classifier", PythonObject(classifier))
        py_setitem(fields, "_public_kind", cls)
        py_setitem(fields, "devices", devices)
        py_setitem(fields, "n_features_in_", PythonObject(Int(py=original["n_features_in_"])))
        py_setitem(fields, "n_estimators_", PythonObject(Int(py=original["_n_trees"])))
        if classifier:
            py_setitem(fields, "classes_", original["classes_"])
            py_setitem(fields, "n_classes_", PythonObject(Int(py=original["_num_outputs"])))
        return PythonObject(None)
    var lock = py_get(fields, "_lock")
    if py_type(lock) is not meta["lock_type"]:
        raise Error("TypeError: ParallelForestPredictor requires its native lock")
    if py_type(py_get(fields, "_closed")) is not b.bool or py_type(py_get(fields, "_classifier")) is not b.bool:
        raise Error("TypeError: ParallelForestPredictor state requires native booleans")
    _ = py_call(lock.acquire)
    try:
        if operation == "predictor_close" or operation == "predictor_exit":
            if not Bool(py=fields["_closed"]):
                py_setitem(fields, "_closed", PythonObject(True))
                var snapshot = fields["_model"]
                var saved = _fields(snapshot, meta["classes"][py_type(snapshot)])
                # Dropping the compiled TreesResident owner releases its GPU
                # handle once; no Python weakref.finalize callback is invoked.
                py_setitem(saved, "_resident_holder", Python.dict())
                py_setitem(fields, "_model", PythonObject(None))
            return PythonObject(None)
        if Bool(py=fields["_closed"]):
            raise Error("RuntimeError: pooled forest predictor is closed")
        if operation == "predictor_enter":
            return model
        var proba = operation == "predictor_predict_proba"
        if proba and not Bool(py=fields["_classifier"]):
            raise Error("ValueError: predict_proba requires a classifier")
        classical_admit_value(args, context)
        var snapshot = fields["_model"]
        var result = _run(snapshot, "forest_predict", "trees_public_forest_proba" if proba else "trees_public_forest_predict",
                          Python.tuple(args[0]), _entry(meta["trees_entry"]), meta["trees_context"],
                          Python.tuple("_mojolearn_rf", "_mojolearn_trees"), fields["devices"], context)
        return parallel_public_result(result, "predict_proba" if proba else "predict", meta["classes"][py_type(snapshot)]["kind"], context)
    finally:
        _ = py_call(lock.release)


def parallel_public_operation(model: PythonObject, packed: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var meta = context["parallel_public"]
    if py_type(packed[0]) is not b.str:
        raise Error("TypeError: parallel operation requires an exact name")
    var operation = String(packed[0])
    var args = py_call(b.list, packed[1])
    classical_connect_numpy(context)
    if operation.startswith("predictor_"):
        return _predictor(model, operation, py_call(b.tuple, args), context)
    var options = meta["plans"][packed[0]]
    var cls = py_type(model) if model is not None else PythonObject(None)
    if model is not None and py_type(cls) is not b.type:
        raise Error("TypeError: parallel drivers refuse Python metaclass hooks")
    var spec = py_get(options, cls)
    if spec is None:
        var first = py_call(b.list, options.values())[0]
        raise Error("TypeError: " + String(first["error"]))
    var device_at = Int(py=spec["device_arg"])
    py_setitem(args, PythonObject(device_at), _devices(args[device_at]))
    classical_admit_value(args, context)
    var fields = _fields(model, meta["classes"][cls]) if model is not None else Python.dict()
    var mode = py_get(fields, "numeric_mode")
    var process_mode = meta["backend"]["_DEFAULT_MODE"]
    var effective = process_mode if mode is None else mode
    classical_admit_value(Python.tuple(mode, process_mode), context)
    if spec["mode"] is not None:
        var policy = String(spec["mode"])
        var selected = process_mode if policy == "default" else effective if policy == "effective" else mode
        if selected is not None and (py_type(selected) is not b.str or String(selected) != "identical"):
            raise Error("ValueError: parallel " + operation + " requires IDENTICAL numeric mode")
    var check = String(spec["check"]) if spec["check"] is not None else String("")
    if check == "gram" or check == "matrix" or check == "kmeans" or check == "gmm_predict":
        var data = classical_host_f32(args[0], 2, "X", context)
        py_setitem(args, PythonObject(0), data)
        if check == "gram":
            if Int(py=data.shape[1]) < 1:
                raise Error("ValueError: parallel Gram requires at least one feature")
            var solver = py_get(fields, "svd_solver")
            classical_admit_value(solver, context)
            if cls is meta["pca"] and py_contains(meta["dense_solvers"], solver) and Int(py=data.shape[0]) < Int(py=data.shape[1]):
                raise Error("ValueError: parallel full PCA currently requires the tall TSQR route")
            if args[3] is not None and py_contains(meta["decompositions"], cls):
                raise Error("TypeError: fit() got an unexpected keyword argument 'sample_weight'")
        elif check == "kmeans":
            var metric = py_get(fields, "metric", PythonObject("euclidean"))
            var oversample = py_get(fields, "oversampling_factor", PythonObject(2.0))
            classical_admit_value(Python.tuple(metric, oversample), context)
            if py_type(metric) is not b.str or String(metric) != "euclidean":
                raise Error("ValueError: parallel KMeans requires metric='euclidean'")
            if Float64(py=oversample) != 2.0:
                raise Error("ValueError: parallel KMeans requires the default oversampling_factor")
            py_setitem(fields, "numeric_mode", PythonObject("identical"))
            effective = PythonObject("identical")
        elif check == "gmm_predict" and not py_contains(fields, "weights_"):
            raise Error("ValueError: GaussianMixture is not fitted")
    elif check == "posterior" and py_type(args[2]) is not b.bool:
        raise Error("ValueError: return_std must be bool")
    elif check == "iforest" and args[2] is not None:
        raise Error("NotImplementedError: IsolationForest does not support sample_weight")
    elif check == "umap_transform":
        var transform_mode = py_get(fields, "_transform_mode")
        if py_type(transform_mode) is not b.str or String(transform_mode) != "identical":
            raise Error("ValueError: requires an IDENTICAL fitted UMAP")
    elif check == "resample":
        var index = Int(py=spec["numeric_mode_arg"])
        var requested = args[index]
        if requested is not None and (py_type(requested) is not b.str or String(requested) != "identical"):
            raise Error("ValueError: parallel resampling requires IDENTICAL numeric mode")
    var owner_op = spec["operation"]
    var method = spec["output"]
    if spec["method_arg"] is not None:
        var requested = args[Int(py=spec["method_arg"])]
        if py_type(requested) is not b.str:
            raise Error("ValueError: parallel method must be an exact supported name")
        owner_op = py_get(owner_op, requested)
        if owner_op is None:
            raise Error("ValueError: unsupported parallel prediction method")
        method = requested
    var values = Python.list()
    var recipe = spec["recipe"]
    for i in range(len(recipe)):  # driver-loop: the fixed native owner ABI
        var item = recipe[i]
        var kind = String(item[0])
        var value = PythonObject(None)
        if kind == "arg":
            value = args[Int(py=item[1])]
        elif kind == "constant":
            value = item[1]
        elif kind == "model":
            value = model
        elif kind == "binding":
            var route = String(item[2])
            var selected = effective if route == "model" else process_mode if route == "default" else item[2]
            value = _module(meta, item[1], selected)
        _ = py_call(values.append, value)
    var owner_args = py_call(b.tuple, values)
    var result = PythonObject(None)
    var route = String(spec["route"])
    if route == "direct":
        result = family_call(_entry(meta["family"]), model, owner_op, owner_args, context)
    elif route == "shard":
        result = family_call(_entry(meta["shard"]), owner_op, owner_args)
    else:
        # All cooperative wrappers put their device tuple in the connected
        # argument position; no Python loop or method constructs this envelope.
        result = _run(model, spec["driver"], owner_op, owner_args, spec["entry"], spec["owner"], spec["modules"], args[device_at], context)
    if method is not None:
        result = parallel_public_result(result, String(method), meta["classes"][cls]["kind"], context)
    return result
