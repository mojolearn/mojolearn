# SPDX-License-Identifier: Apache-2.0
# CPU IsolationForest rebuilds and scores through its existing numerical ABI.
# Preparation, resolved controls, state readback and output publication stay native.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.sequential_admission import _one_of
from bindings.classical_arrays import classical_input, classical_address, classical_is_array, classical_is_f32, classical_c_order, classical_kind, classical_f64
from bindings.host_archive_native import host_archive_typed
from bindings.host_kernel_inference_native import _kernel_entry
from bindings.host_inference_scalars import host_inference_scalar
from bindings.host_inference_features import host_inference_feature_names
from bindings.host_svc_native import _svc_input
from bindings.host_sparse_native import host_sparse_format
from bindings.host_tree_native import host_tree_wide
from bindings.native_numpy_metadata import native_numpy_dtype
from bindings.classical_pandas import classical_admit_value


def host_iforest_admit_option(value: PythonObject, context: PythonObject) raises:
    # Fit retains the original options by identity, including native NumPy
    # Unicode and numeric aliases that have distinct types from int64/float64.
    # Scoring uses the separately admitted resolved controls; do not reject a
    # successful fit merely because its unused original option is an alias.
    if py_get(context, "ndarray") is not None:
        var constructor = native_numpy_dtype(context)
        var aliases = Python.tuple("q", "Q", "g", "U", "S")
        for i in range(len(aliases)):  # driver-loop: exact original native option aliases
            if py_type(value) is py_call(constructor, aliases[i]).type:
                return
    classical_admit_value(value, context)


def _iforest_resolve(fields: PythonObject, context: PythonObject) raises:
    # A fitted instance assembled without a resolved seed historically resolves
    # its constructor knobs on first score. Resolve exactly those stored options.
    var b = Python.import_module("builtins")
    var native_int = py_type(PythonObject(0))
    var native_float = py_type(PythonObject(Float64(0.0)))
    var names = Python.tuple("n_estimators", "max_samples", "max_depth", "max_features", "bootstrap", "contamination", "random_state", "warm_start", "verbose")
    for i in range(len(names)):  # driver-loop: fixed constructor scalar fields
        host_inference_scalar(fields[names[i]], context, String(names[i]) == "bootstrap" or String(names[i]) == "warm_start" or String(names[i]) == "verbose")
    var trees = Int(py=py_call(native_int, fields["n_estimators"]))
    if trees < 1:
        raise Error("IsolationForest n_estimators must be at least one")
    var samples = fields["max_samples"]
    if py_type(samples) is b.str:
        if String(py_call(samples.lower)) != "auto":
            raise Error("IsolationForest max_samples must be auto, a positive integer or a fraction")
        py_setitem(fields, "_max_samples_mode", 0)
        py_setitem(fields, "_max_samples_int", 256)
        py_setitem(fields, "_max_samples_frac", 1.0)
    elif _one_of(py_type(samples), py_get(context, "bool_types", Python.tuple(b.bool))):
        raise Error("IsolationForest max_samples cannot be a bool")
    elif _one_of(py_type(samples), context["integer_types"]):
        var count = Int(py=py_call(native_int, samples))
        if count < 1:
            raise Error("IsolationForest max_samples must be positive")
        py_setitem(fields, "_max_samples_mode", 1)
        py_setitem(fields, "_max_samples_int", count)
        py_setitem(fields, "_max_samples_frac", 1.0)
    else:
        var fraction = Float64(py=py_call(native_float, samples))
        if not (fraction > 0.0 and fraction <= 1.0):
            raise Error("IsolationForest fractional max_samples must be in (0, 1]")
        py_setitem(fields, "_max_samples_mode", 2)
        py_setitem(fields, "_max_samples_int", 256)
        py_setitem(fields, "_max_samples_frac", fraction)
    var depth = -1 if fields["max_depth"] is None else Int(py=py_call(native_int, fields["max_depth"]))
    py_setitem(fields, "_max_depth", depth)
    if fields["max_depth"] is not None and depth < 1:
        raise Error("IsolationForest max_depth must be None or positive")
    var features = fields["max_features"]
    if _one_of(py_type(features), py_get(context, "bool_types", Python.tuple(b.bool))):
        raise Error("IsolationForest max_features cannot be a bool")
    if _one_of(py_type(features), context["integer_types"]):
        var count = Int(py=py_call(native_int, features))
        if count < 1:
            raise Error("IsolationForest integer max_features must be positive")
        py_setitem(fields, "_max_features_mode", 1)
        py_setitem(fields, "_max_features_int", count)
        py_setitem(fields, "_max_features_frac", 1.0)
    else:
        var fraction = Float64(py=py_call(native_float, features))
        if not (fraction > 0.0 and fraction <= 1.0):
            raise Error("IsolationForest fractional max_features must be in (0, 1]")
        py_setitem(fields, "_max_features_mode", 0)
        py_setitem(fields, "_max_features_int", 0)
        py_setitem(fields, "_max_features_frac", fraction)
    var contamination = fields["contamination"]
    if py_type(contamination) is b.str:
        if String(py_call(contamination.lower)) != "auto":
            raise Error("IsolationForest contamination must be auto or in (0, .5]")
        py_setitem(fields, "_contamination_auto", True)
        py_setitem(fields, "_contamination", 0.0)
    else:
        var fraction = Float64(py=py_call(native_float, contamination))
        if not (fraction > 0.0 and fraction <= 0.5):
            raise Error("IsolationForest contamination must be auto or in (0, .5]")
        py_setitem(fields, "_contamination_auto", False)
        py_setitem(fields, "_contamination", fraction)
    var seed = 0 if fields["random_state"] is None else Int(py=py_call(native_int, fields["random_state"]))
    if seed < 0 or seed > 4294967295:
        raise Error("IsolationForest random_state must fit uint32")
    if Bool(py=fields["warm_start"]):
        raise Error("NotImplementedError: IsolationForest warm_start is not supported")
    if fields["n_jobs"] is not None:
        raise Error("NotImplementedError: IsolationForest n_jobs is not supported")
    if Bool(py=fields["verbose"]):
        raise Error("NotImplementedError: IsolationForest verbose is not supported")
    if fields["output_type"] is not None:
        raise Error("NotImplementedError: IsolationForest output_type is not supported")
    py_setitem(fields, "_n_estimators", trees)
    py_setitem(fields, "_bootstrap", Bool(py=fields["bootstrap"]))
    py_setitem(fields, "_seed", seed)


def host_iforest_admit_state(fields: PythonObject, context: PythonObject) raises:
    var names = Python.tuple("_n_estimators", "_max_samples_mode", "_max_samples_int", "_max_samples_frac", "_max_depth", "_max_features_mode", "_max_features_int", "_max_features_frac", "_seed", "_contamination")
    for i in range(len(names)):  # driver-loop: fixed resolved native ABI controls
        if py_contains(fields, names[i]):
            host_inference_scalar(fields[names[i]], context)
    var flags = Python.tuple("_bootstrap", "_contamination_auto")
    for i in range(len(flags)):  # driver-loop: fixed native truth controls
        if py_contains(fields, flags[i]):
            host_inference_scalar(fields[flags[i]], context, True)
    var train = py_get(fields, "_x")
    if py_type(train) is py_get(context, "ndarray") and classical_kind(train, context) == "O":
        raise Error("TypeError: IsolationForest training state requires native numeric storage")


def host_iforest_inference(model: PythonObject, fields: PythonObject, source: PythonObject, want: Int,
                           context: PythonObject, public: Bool) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var name = py_call(b.type.__dict__["__name__"].__get__, py_type(model))
    if host_sparse_format(source).byte_length() > 0:
        raise Error("TypeError: Sparse data was passed for X, but dense data is required. Use '.toarray()' to convert to a dense numpy array. (" + String(name) + " takes dense X only.)")
    if not py_contains(fields, "_x"):
        raise Error("IsolationForest requires fit before scoring")
    var query = _svc_input(source, context)
    var rows = Int(py=query.shape[0])
    var features = Int(py=fields["n_features_in_"])
    if Int(py=query.shape[1]) != features:
        raise Error("IsolationForest query width differs from fitted dimensions")
    if not Bool(py=py_get(fields, "_mojolearn_fitting", False)):
        host_inference_feature_names(model, fields, context)
    var train = classical_input(fields["_x"], context, "IsolationForest fitted samples")
    if Int(py=train.ndim) != 2 or Int(py=train.shape[1]) != features:
        raise Error("IsolationForest fitted sample shape differs from fitted dimensions")
    if not classical_is_f32(train, context) or (not classical_is_array(train, context) and not Bool(py=train.dtype.isnative)):
        raise Error("IsolationForest fitted samples require native float32 storage")
    var f_order = String(train.order) == "F" if classical_is_array(train, context) else Bool(py=train.flags.f_contiguous)
    if not classical_c_order(train, context) and not f_order:
        raise Error("IsolationForest fitted samples require contiguous storage")
    var count = Int(py=train.shape[0])
    if features < 1 or count < 1 or count > 2147483647 // features:
        raise Error("IsolationForest fitted dimensions exceed native indexing")
    var out = host_archive_typed(Python.tuple(rows), "<i4" if want == 2 else "<f4", context["output_context"])
    var info = host_archive_typed(Python.tuple(3), "<f8", context["output_context"])
    if not py_contains(fields, "_seed"):
        _iforest_resolve(fields, context)
    var params = Python.list(count, features, rows, fields["_n_estimators"], fields["_max_samples_mode"], fields["_max_samples_int"], fields["_max_samples_frac"], fields["_max_depth"], fields["_max_features_mode"], fields["_max_features_int"], fields["_max_features_frac"], Int(Bool(py=fields["_bootstrap"])), fields["_seed"], Int(Bool(py=fields["_contamination_auto"])), fields["_contamination"], want)
    var entry = _kernel_entry("iforest_run", context)
    try:
        # The explicit CPU entry has no resident registry; preserve its existing
        # seeded reconstruction on each call and consume the matching buffer only.
        _ = py_call(entry, classical_address(train), classical_address(query), classical_address(out) if want != 2 else PythonObject(0), classical_address(out) if want == 2 else PythonObject(0), classical_address(info), params)
    except error:
        var text = String(error)
        if ("NaN" in text or "Infinit" in text or "infinit" in text
                or "nonfinite" in text or "non-finite" in text
                or "not finite" in text or "requires finite" in text or "must be finite" in text):
            raise Error("Input X contains NaN or infinity (" + String(name) + ": " + text + ")")
        raise Error("HostIForestKernelError: " + text)
    _ = train^
    _ = query^
    py_setitem(fields, "offset_", Float64(Float32(classical_f64(info, 0))))
    py_setitem(fields, "max_samples_", Int(classical_f64(info, 1)))
    _ = info^
    if want != 2 and public:
        out = host_tree_wide(out, context)
    return classical_input(out, context, "isolation forest result") if public and py_get(context, "ndarray") is not None else out
