# SPDX-License-Identifier: Apache-2.0
# CPU IsolationForest rebuilds and scores through its existing numerical ABI.
# Preparation, resolved controls, state readback and output publication stay native.
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem, py_fields
from bindings.sequential_admission_door import _one_of
from bindings.classical_arrays_door import classical_input, classical_address, classical_is_array, classical_is_f32, classical_c_order, classical_kind, classical_f64
from bindings.host_array_native_pack import host_archive_typed
from bindings.host_kernel_inference_native_unpack import _kernel_entry
from bindings.host_inference_scalars_door import host_inference_scalar
from bindings.host_inference_features_unpack import host_inference_feature_names
from bindings.host_svc_native_door import _svc_input
from bindings.host_sparse_native_door import host_sparse_format
from bindings.host_tree_native_door import host_tree_wide
from bindings.native_numpy_metadata_unpack import native_numpy_dtype
from bindings.classical_pandas_door import classical_admit_value
from bindings.trees_lifetime_door import trees_resident_new
from bindings.trees_side_records_unpack import trees_side_get, trees_side_put
from bindings.p3nc_fitted_freeze_unpack import fitted_is_freezable_any
from bindings.py_optional import py_present
from bindings.host_tree_archive_keys_unpack import host_tree_archive_keys
from bindings.py_balanced import py_item
from std.python._cpython import GILReleased
from core.entry_table import EntryRef
from core.iforest_entry import IFRunRequest, IF_CAP_RESIDENT, if_entry_caps, if_entry_ref, if_entry_run

comptime _HOST_TOKEN_SLOT = "trees_iforest_host_resident_v1"


def host_iforest_admit_option(value: PythonObject, context: PythonObject) raises:
    var nk = host_tree_archive_keys()
    # Fit retains the original options by identity, including native NumPy
    # Unicode and numeric aliases that have distinct types from int64/float64.
    # Scoring uses the separately admitted resolved controls; do not reject a
    # successful fit merely because its unused original option is an alias.
    if py_get(context, nk[].k_ndarray) is not None:
        var kind = py_type(value)
        var types = host_iforest_alias_types(context)
        for i in range(1, len(types)):  # driver-loop: exact original native option aliases
            if kind is types[i]:
                return
    classical_admit_value(value, context)


def host_iforest_alias_types(context: PythonObject) raises -> PythonObject:
    """`(dtype constructor, type of 'q', 'Q', 'g', 'U', 'S')`, built once per
    connected NumPy dtype constructor and kept in the context (p3-trees: the
    five np.dtype() calls ran for each option on every scoring call)."""
    var nk = host_tree_archive_keys()
    var constructor = native_numpy_dtype(context)
    var held = py_get(context, nk[].k__host_iforest_alias_types)
    if held is not None and held[0] is constructor:
        return held
    var aliases = Python.tuple("q", "Q", "g", "U", "S")
    var types = Python.list(constructor)
    for i in range(len(aliases)):  # driver-loop: exact original native option aliases
        _ = py_call(types.append, py_call(constructor, aliases[i]).type)
    var record = py_call(native_loaded_module("builtins").tuple, types)
    py_setitem(context, nk[].k__host_iforest_alias_types, record)
    return record


def _iforest_resolve(fields: PythonObject, context: PythonObject) raises:
    var nk = host_tree_archive_keys()
    # A fitted instance assembled without a resolved seed historically resolves
    # its constructor knobs on first score. Resolve exactly those stored options.
    var b = native_loaded_module("builtins")
    var native_int = py_type(PythonObject(0))
    var native_float = py_type(PythonObject(Float64(0.0)))
    var names = Python.tuple("n_estimators", "max_samples", "max_depth", "max_features", "bootstrap", "contamination", "random_state", "warm_start", "verbose")
    for i in range(len(names)):  # driver-loop: fixed constructor scalar fields
        host_inference_scalar(fields[names[i]], context, String(names[i]) == "bootstrap" or String(names[i]) == "warm_start" or String(names[i]) == "verbose")
    var trees = Int(py=py_call(native_int, py_item(fields, nk[].k_n_estimators)))
    if trees < 1:
        raise Error("IsolationForest n_estimators must be at least one")
    var samples = py_item(fields, nk[].k_max_samples)
    if py_type(samples) is b.str:
        if String(py_call(samples.lower)) != "auto":
            raise Error("IsolationForest max_samples must be auto, a positive integer or a fraction")
        py_setitem(fields, nk[].k__max_samples_mode, 0)
        py_setitem(fields, nk[].k__max_samples_int, 256)
        py_setitem(fields, nk[].k__max_samples_frac, 1.0)
    elif _one_of(py_type(samples), py_get(context, nk[].k_bool_types, Python.tuple(b.bool))):
        raise Error("IsolationForest max_samples cannot be a bool")
    elif _one_of(py_type(samples), py_item(context, nk[].k_integer_types)):
        var count = Int(py=py_call(native_int, samples))
        if count < 1:
            raise Error("IsolationForest max_samples must be positive")
        py_setitem(fields, nk[].k__max_samples_mode, 1)
        py_setitem(fields, nk[].k__max_samples_int, count)
        py_setitem(fields, nk[].k__max_samples_frac, 1.0)
    else:
        var fraction = Float64(py=py_call(native_float, samples))
        if not (fraction > 0.0 and fraction <= 1.0):
            raise Error("IsolationForest fractional max_samples must be in (0, 1]")
        py_setitem(fields, nk[].k__max_samples_mode, 2)
        py_setitem(fields, nk[].k__max_samples_int, 256)
        py_setitem(fields, nk[].k__max_samples_frac, fraction)
    var depth = -1 if py_item(fields, nk[].k_max_depth) is None else Int(py=py_call(native_int, py_item(fields, nk[].k_max_depth)))
    py_setitem(fields, nk[].k__max_depth, depth)
    if py_item(fields, nk[].k_max_depth) is not None and depth < 1:
        raise Error("IsolationForest max_depth must be None or positive")
    var features = py_item(fields, nk[].k_max_features)
    if _one_of(py_type(features), py_get(context, nk[].k_bool_types, Python.tuple(b.bool))):
        raise Error("IsolationForest max_features cannot be a bool")
    if _one_of(py_type(features), py_item(context, nk[].k_integer_types)):
        var count = Int(py=py_call(native_int, features))
        if count < 1:
            raise Error("IsolationForest integer max_features must be positive")
        py_setitem(fields, nk[].k__max_features_mode, 1)
        py_setitem(fields, nk[].k__max_features_int, count)
        py_setitem(fields, nk[].k__max_features_frac, 1.0)
    else:
        var fraction = Float64(py=py_call(native_float, features))
        if not (fraction > 0.0 and fraction <= 1.0):
            raise Error("IsolationForest fractional max_features must be in (0, 1]")
        py_setitem(fields, nk[].k__max_features_mode, 0)
        py_setitem(fields, nk[].k__max_features_int, 0)
        py_setitem(fields, nk[].k__max_features_frac, fraction)
    var contamination = py_item(fields, nk[].k_contamination)
    if py_type(contamination) is b.str:
        if String(py_call(contamination.lower)) != "auto":
            raise Error("IsolationForest contamination must be auto or in (0, .5]")
        py_setitem(fields, nk[].k__contamination_auto, True)
        py_setitem(fields, nk[].k__contamination, 0.0)
    else:
        var fraction = Float64(py=py_call(native_float, contamination))
        if not (fraction > 0.0 and fraction <= 0.5):
            raise Error("IsolationForest contamination must be auto or in (0, .5]")
        py_setitem(fields, nk[].k__contamination_auto, False)
        py_setitem(fields, nk[].k__contamination, fraction)
    var seed = 0 if py_item(fields, nk[].k_random_state) is None else Int(py=py_call(native_int, py_item(fields, nk[].k_random_state)))
    if seed < 0 or seed > 4294967295:
        raise Error("IsolationForest random_state must fit uint32")
    if Bool(py=py_item(fields, nk[].k_warm_start)):
        raise Error("NotImplementedError: IsolationForest warm_start is not supported")
    if py_item(fields, nk[].k_n_jobs) is not None:
        raise Error("NotImplementedError: IsolationForest n_jobs is not supported")
    if Bool(py=py_item(fields, nk[].k_verbose)):
        raise Error("NotImplementedError: IsolationForest verbose is not supported")
    if py_item(fields, nk[].k_output_type) is not None:
        raise Error("NotImplementedError: IsolationForest output_type is not supported")
    py_setitem(fields, nk[].k__n_estimators, trees)
    py_setitem(fields, nk[].k__bootstrap, Bool(py=py_item(fields, nk[].k_bootstrap)))
    py_setitem(fields, nk[].k__seed, seed)


def host_iforest_admit_state(fields: PythonObject, context: PythonObject) raises:
    var nk = host_tree_archive_keys()
    var names = Python.tuple("_n_estimators", "_max_samples_mode", "_max_samples_int", "_max_samples_frac", "_max_depth", "_max_features_mode", "_max_features_int", "_max_features_frac", "_seed", "_contamination")
    for i in range(len(names)):  # driver-loop: fixed resolved native ABI controls
        if py_contains(fields, names[i]):
            host_inference_scalar(fields[names[i]], context)
    var flags = Python.tuple("_bootstrap", "_contamination_auto")
    for i in range(len(flags)):  # driver-loop: fixed native truth controls
        if py_contains(fields, flags[i]):
            host_inference_scalar(fields[flags[i]], context, True)
    var train = py_get(fields, nk[].k__x)
    if py_type(train) is py_get(context, nk[].k_ndarray) and classical_kind(train, context) == "O":
        raise Error("TypeError: IsolationForest training state requires native numeric storage")


def host_iforest_owned_x(fields: PythonObject) raises -> Bool:
    """The fitted `_x` is the fit's own read-only float32 copy
    (`input_copied_` True and its read-only flag set by the fit): the only
    training storage a resident forest may mirror (lesson 10)."""
    var nk = host_tree_archive_keys()
    var copied = py_get(fields, nk[].k_input_copied_)
    if copied is None or not Bool(py=copied):
        return False
    var x = py_get(fields, nk[].k__x)
    if x is None or not fitted_is_freezable_any(x):
        return False
    return Bool(py=x._readonly)


def host_iforest_entry() raises -> EntryRef:
    """The CPU image's native iforest entry (core/iforest_entry.mojo;
    `_mojolearn_svm_host` publishes it under identical / CPU, the only tier
    that image builds), resolved once per operation: the table's slot read,
    no CPython call."""
    return if_entry_ref(1, 0)


def host_iforest_request(fields: PythonObject, n_train: Int, n_features: Int, n_query: Int,
                         want: Int) raises -> IFRunRequest:
    """UNPACK: the fit's knobs from the resolved fields into the typed
    request (the former sixteen `iforest_run` slots); addresses and the
    resident form are the caller's."""
    var nk = host_tree_archive_keys()
    var r = IFRunRequest()
    r.n_train = n_train
    r.n_features = n_features
    r.n_query = n_query
    r.n_estimators = Int(py=py_item(fields, nk[].k__n_estimators))
    r.max_samples_mode = Int(py=py_item(fields, nk[].k__max_samples_mode))
    r.max_samples_int = Int(py=py_item(fields, nk[].k__max_samples_int))
    r.max_samples_frac = Float64(py=py_item(fields, nk[].k__max_samples_frac))
    r.max_depth = Int(py=py_item(fields, nk[].k__max_depth))
    r.max_features_mode = Int(py=py_item(fields, nk[].k__max_features_mode))
    r.max_features_int = Int(py=py_item(fields, nk[].k__max_features_int))
    r.max_features_frac = Float64(py=py_item(fields, nk[].k__max_features_frac))
    r.bootstrap = Bool(py=py_item(fields, nk[].k__bootstrap))
    r.random_state = Int(py=py_item(fields, nk[].k__seed))
    r.contamination_auto = Bool(py=py_item(fields, nk[].k__contamination_auto))
    r.contamination = Float64(py=py_item(fields, nk[].k__contamination))
    r.want = want
    return r^


def host_iforest_run(entry: EntryRef, request: IFRunRequest) raises:
    """One fit-and-score through the native entry with the GIL released (as
    inside the former Python binding `iforest_run`); the caller keeps every
    lent buffer alive."""
    with GILReleased(Python()):
        _ = if_entry_run(entry, request)


def _host_resident_entries(entry: PythonObject, context: PythonObject) raises -> PythonObject:
    """The `(iforest_resident_release,)` entry of the binary that serves
    `entry` (any fixed entry of the CPU image, `iforest_forest_run`) when
    that binary keeps the fitted forest resident between calls
    (IF_CAP_RESIDENT of its native iforest entry); None otherwise. Read from
    the minting module's dictionary (no module __getattr__), and refused
    unless it is a fixed native entry."""
    var nk = host_tree_archive_keys()
    if entry is None:
        return PythonObject(None)
    var owner = entry.__self__
    if owner is None or py_type(owner) is not py_item(context, nk[].k_host_iforest_module_type):
        return PythonObject(None)
    var table = py_fields(owner)
    var release = py_get(table, nk[].k_iforest_resident_release)
    if release is None:
        return PythonObject(None)
    if py_type(release) is not py_item(context, nk[].k_host_tree_entry_type):
        raise Error("TypeError: explicit CPU inference requires a fixed native entry")
    if (if_entry_caps(host_iforest_entry()) & IF_CAP_RESIDENT) == 0:
        return PythonObject(None)
    return Python.tuple(release)


def _host_forest_words(forest: PythonObject, context: PythonObject) raises -> PythonObject:
    """UNPACK: `(words, views)` of a held saved forest: five flat native
    arrays (offsets int32 of n_trees + 1, feature int32, threshold float32,
    left int32, right int32, equal node counts), their addresses, the tree
    and node counts; the views keep the buffers alive through the call. The
    node-level checks run in the binding (`if_forest_check`)."""
    var b = native_loaded_module("builtins")
    if py_type(forest) is not b.tuple or len(forest) != 5:
        raise Error("IsolationForest saved forest must be a tuple of five arrays")
    var kinds = Python.tuple("i", "i", "f", "i", "i")
    var words = Python.list()
    var views = Python.list()
    var sizes = List[Int]()
    for i in range(5):  # driver-loop: the five fixed forest arrays
        var view = classical_input(forest[i], context, "IsolationForest saved forest")
        if (Int(py=view.ndim) != 1 or classical_kind(view, context) != String(kinds[i])
                or Int(py=view.itemsize) != 4 or not classical_c_order(view, context)
                or (not classical_is_array(view, context) and not Bool(py=view.dtype.isnative))):
            raise Error("IsolationForest saved forest requires flat native 32-bit storage")
        _ = py_call(words.append, classical_address(view))
        _ = py_call(views.append, view)
        sizes.append(Int(py=view.size))
    var total = sizes[1]
    if sizes[0] < 2 or sizes[2] != total or sizes[3] != total or sizes[4] != total:
        raise Error("IsolationForest saved forest arrays do not have matching lengths")
    _ = py_call(words.append, PythonObject(sizes[0] - 1))
    _ = py_call(words.append, PythonObject(total))
    return Python.tuple(words, views)


def _host_forest_inference(model: PythonObject, fields: PythonObject, query: PythonObject, rows: Int, features: Int,
                           want: Int, name: PythonObject, context: PythonObject, public: Bool) raises -> PythonObject:
    """A saved forest (`mojolearn-iforest-2`, held as `_if_forest`, no
    training matrix) scored on the CPU (lane fix-iforest-save-device-forest):
    `iforest_forest_run` walks the same four node arrays the host export
    writes, so the scores are those of the fit that saved them, on every
    column. A resident binary keeps the imported forest under a token, the
    record anchored on the held forest tuple."""
    var nk = host_tree_archive_keys()
    var forest = py_item(fields, nk[].k__if_forest)
    var unpacked = _host_forest_words(forest, context)
    var entry = _kernel_entry("iforest_forest_run", context)
    if not py_contains(fields, nk[].k_max_samples_) or not py_contains(fields, nk[].k_offset_):
        raise Error("IsolationForest saved forest lacks its fitted offset_ or max_samples_")
    var out = host_archive_typed(Python.tuple(rows), "<i4" if want == 2 else "<f4", py_item(context, nk[].k_output_context))
    var info = host_archive_typed(Python.tuple(4), "<f8", py_item(context, nk[].k_output_context))
    var resident = _host_resident_entries(entry, context)
    var record = Optional[PythonObject]()
    var held = 0
    var same = False
    if resident is not None:
        record = py_present(trees_side_get(_HOST_TOKEN_SLOT, model))
        same = Bool(record) and record.value()[1] is forest
        if same:
            held = Int(py=record.value()[0])
    var b = native_loaded_module("builtins")
    var params = Python.list(PythonObject(features), PythonObject(rows), py_call(b.int, py_item(fields, nk[].k_max_samples_)),
                             py_call(b.float, py_item(fields, nk[].k_offset_)), PythonObject(want), PythonObject(held))
    try:
        _ = py_call(entry, unpacked[0], classical_address(query), classical_address(out) if want != 2 else PythonObject(0),
                    classical_address(out) if want == 2 else PythonObject(0), classical_address(info), params)
    except error:
        var text = String(error)
        if ("NaN" in text or "Infinit" in text or "infinit" in text
                or "nonfinite" in text or "non-finite" in text
                or "not finite" in text or "requires finite" in text or "must be finite" in text):
            raise Error("Input X contains NaN or infinity (" + String(name) + ": " + text + ")")
        raise Error("HostIForestKernelError: " + text)
    _ = unpacked  # keep the forest views alive through the native call above
    _ = query
    if resident is not None:
        var token = Int(MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(info)))[3])
        if token != 0 and (token != held or not same):
            var lifetime = record.value()[2] if Bool(record) and token == Int(py=record.value()[0]) else trees_resident_new(resident[0], token)
            _ = trees_side_put(_HOST_TOKEN_SLOT, model, Python.list(PythonObject(token), forest, lifetime))
    _ = info^
    if want != 2 and public:
        out = host_tree_wide(out, context)
    return classical_input(out, context, "isolation forest result") if public and py_get(context, nk[].k_ndarray) is not None else out


def host_iforest_inference(model: PythonObject, fields: PythonObject, source: PythonObject, want: Int,
                           context: PythonObject, public: Bool) raises -> PythonObject:
    var nk = host_tree_archive_keys()
    var b = native_loaded_module("builtins")
    var name = py_call(py_item(b.type.__dict__, nk[].k___name__).__get__, py_type(model))
    if host_sparse_format(source).byte_length() > 0:
        raise Error("TypeError: Sparse data was passed for X, but dense data is required. Use '.toarray()' to convert to a dense numpy array. (" + String(name) + " takes dense X only.)")
    var forest_only = not py_contains(fields, nk[].k__x) and py_contains(fields, nk[].k__if_forest)
    if not py_contains(fields, nk[].k__x) and not forest_only:
        raise Error("IsolationForest requires fit before scoring")
    var query = _svc_input(source, context)
    var rows = Int(py=query.shape[0])
    var features = Int(py=py_item(fields, nk[].k_n_features_in_))
    if Int(py=query.shape[1]) != features:
        raise Error("IsolationForest query width differs from fitted dimensions")
    if not Bool(py=py_get(fields, nk[].k__mojolearn_fitting, False)):
        host_inference_feature_names(model, fields, context)
    if forest_only:
        return _host_forest_inference(model, fields, query, rows, features, want, name, context, public)
    var train = classical_input(py_item(fields, nk[].k__x), context, "IsolationForest fitted samples")
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
    var out = host_archive_typed(Python.tuple(rows), "<i4" if want == 2 else "<f4", py_item(context, nk[].k_output_context))
    if not py_contains(fields, nk[].k__seed):
        _iforest_resolve(fields, context)
    var request = host_iforest_request(fields, count, features, rows, want)
    var resident = _host_resident_entries(_kernel_entry("iforest_forest_run", context), context)
    if resident is not None and not host_iforest_owned_x(fields):
        resident = PythonObject(None)
    var held = 0
    var record = Optional[PythonObject]()
    var same_train = False
    if resident is not None:
        record = py_present(trees_side_get(_HOST_TOKEN_SLOT, model))
        same_train = Bool(record) and record.value()[1] is py_item(fields, nk[].k__x)
        if same_train:
            held = Int(py=record.value()[0])
        request.resident = True
        request.token = held
    var info = host_archive_typed(Python.tuple(3 if resident is None else 4), "<f8", py_item(context, nk[].k_output_context))
    request.train = Int(py=classical_address(train))
    request.query = Int(py=classical_address(query))
    request.out_f32 = Int(py=classical_address(out)) if want != 2 else 0
    request.out_i32 = Int(py=classical_address(out)) if want == 2 else 0
    request.info = Int(py=classical_address(info))
    try:
        host_iforest_run(host_iforest_entry(), request)
    except error:
        var text = String(error)
        if ("NaN" in text or "Infinit" in text or "infinit" in text
                or "nonfinite" in text or "non-finite" in text
                or "not finite" in text or "requires finite" in text or "must be finite" in text):
            raise Error("Input X contains NaN or infinity (" + String(name) + ": " + text + ")")
        raise Error("HostIForestKernelError: " + text)
    _ = train^
    _ = query^
    py_setitem(fields, nk[].k_offset_, Float64(Float32(classical_f64(info, 0))))
    py_setitem(fields, nk[].k_max_samples_, Int(classical_f64(info, 1)))
    if resident is not None:
        var token = Int(MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(info)))[3])
        if token != held or not same_train:
            # A fresh forest replaced the held one (the store also evicts
            # past its cap). Replacing the record drops the old lifetime,
            # which releases the old token; a reissued equal token keeps its
            # lifetime. A model without weak references keeps no record, so
            # its token is released at once.
            var lifetime = record.value()[2] if Bool(record) and token == Int(py=record.value()[0]) else trees_resident_new(resident[0], token)
            _ = trees_side_put(_HOST_TOKEN_SLOT, model, Python.list(PythonObject(token), py_item(fields, nk[].k__x), lifetime))
    _ = info^
    if want != 2 and public:
        out = host_tree_wide(out, context)
    return classical_input(out, context, "isolation forest result") if public and py_get(context, nk[].k_ndarray) is not None else out
