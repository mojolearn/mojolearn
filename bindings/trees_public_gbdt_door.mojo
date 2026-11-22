# SPDX-License-Identifier: Apache-2.0
# The gradient-boosting public operations in one native call each:
# `GradientBoosting.predict` / `predict_proba` / `predict_classes`
# (python/mojolearn/ensemble.py, inherited by OrderedRMSE and
# ExperimentalTwoLevelFeatureFreq), the OrderedRMSE and FeatureFreq fits, and
# the scikit-learn adapters' `decision_function` / `predict` /
# `predict_proba` (python/mojolearn/_gbdt_adapters.py). This is the Python
# orchestration those methods ran, step for step: the fitted checks with
# their wording, the binding of the model's numeric mode, the device-resident
# model copy (prepared once, reused while the model text is the same bytes,
# released when its holder drops it), the input layout, the predict entry
# (resident, multi-output or single-output per-call parse, the CPU CTR
# reader) and the output shape. Same entries, same order, same bits.
#
# Round trips removed against the Python path: the resident handle lookup,
# the X staging and the launch run in one call (Python returned between the
# handle, the layout conversion and the predict entry); `predict_proba`'s
# binary fallback keeps the raw scores in the call and widens them for
# `gbdt_sigmoid_pair` without a Python `astype`; the adapters' margins go
# from the predict entry to the binary output kernel without a Python step.
from core.mode_answer import MODE_FAMILY_GBDT, answer_mode_word
from std.python import Python, PythonObject
from bindings.base_helpers_key_unpack import base_helpers_key
from bindings.base_helpers_released_unpack import base_reduce_stat, base_check_indices_i64, base_bincount_i64
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem, py_fields
from bindings.params_metadata_native_door import _resolved
from bindings.native_instance_item_pack import native_instance_item
from bindings.py_typed_call import py_call_typed
from bindings.classical_exceptions_door import classical_not_fitted_class
from bindings.host_tree_native_door import host_tree_operation
from bindings.trees_lifetime_door import trees_resident_new
from std.math import isfinite
from std.memory import unsafe_memcpy
from core.trees_gbdt_metadata import gbdt_tree_metadata
from bindings.trees_public_inputs_door import (
    trees_builtins,
    trees_table,
    trees_entry,
    trees_helper,
    trees_reshape,
    trees_address,
    trees_forest_x,
    trees_cast,
    trees_decode_labels,
    trees_call,
)
from bindings.trees_public_inputs_run import trees_finite_f32
from bindings.gbdt_category_unpack import gbdt_category_block_list, gbdt_category_admit, gbdt_category_apply
from bindings.admitted_input_unpack import AdmittedX, admit_f32, admit_ids, admitted_array, admitted_vector
from bindings.py_optional import py_present
from gbdt.trees_gbdt_run import gbdt_mode_checked, gbdt_text_has_ctr, _type_or_value
from bindings.py_cpython_table import cpython_table, cpython_utf8
from core.trees_entry_table import TreesEntryKey
from core.entry_table import EntryRef, entry_ref_vendor, entry_ref_mode
from gbdt.fit_calls import (
    xg_ref, xg_resident, xg_predict, xg_predict_multi, xg_binary, xg_sigmoid_pair, xg_resident_prepare,
    xg_resident_predict, xg_ordered_fit, xg_feature_freq_fit,
)
from bindings.trees_keys_unpack import trees_keys
from bindings.py_balanced import py_item
# TOMBSTONE: trees_f32 / trees_forest_layout / trees_all_finite / TreesMatrix / trees_view / trees_check_rank / trees_vector / trees_empty uses (the trees admission aliases) deleted 2026-11-09 by lane a1b-doors-opt-in-trees-linear; code recoverable at f59d913b0b.
# Restore: experiments/removed/A1B_LABELS_ADMISSION_COPY.patch.

comptime _PREDICT_RAW = 0
comptime _PREDICT_SOFTMAX = 1
comptime _PREDICT_SIGMOID = 2
comptime _PREDICT_SIGMOID_PAIR = 3
comptime _PREDICT_CLASSES_BINARY = 4
comptime _PREDICT_CLASSES_PINNED = 5
comptime _PREDICT_CLASSES_OVA = 6


# --------------------------------------------------------------- binding


def _repr(value: PythonObject) raises -> String:
    return String(py_call(trees_builtins().repr, value))


def gbdt_mode(fields: PythonObject, context: PythonObject) raises -> String:
    """`NumericModeMixin._bind`'s mode: the estimator's `numeric_mode`, or
    the process default, checked as `_backend.binding` checks it."""
    var b = trees_builtins()
    var nk = trees_keys()
    var mode = py_get(fields, nk[].k_numeric_mode)
    if mode is None:
        mode = py_item(py_item(context, nk[].k_backend_fields), nk[].k__DEFAULT_MODE)
    if py_type(mode) is b.str:
        # Already clean (the usual case): no isinstance/strip/lower calls
        # (p3-trees).
        var exact = String(py=mode)
        if exact == "fast" or exact == "identical":
            answer_mode_word(exact, MODE_FAMILY_GBDT)
            return exact
    if not Bool(py=py_call(b.isinstance, mode, b.str)):
        raise Error("numeric_mode must be fast, identical or None")
    var checked = gbdt_mode_checked(String(py_call(py_call(mode.strip).lower)))
    answer_mode_word(checked, MODE_FAMILY_GBDT)
    return checked


def gbdt_table(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    return trees_table(context, gbdt_mode(fields, context), "_mojolearn_gbdt")


def gbdt_entry_of(table: PythonObject) raises -> EntryRef:
    """The gbdt entry, resolved once per operation (lane
    entry-table-zero-alloc), of the image this table connected (the bridge
    records it at import as `gbdt_native_key`): the fits, the predicts, the
    probes and the model dimension run through that image's published
    native entry (gbdt/fit_calls.mojo), not through Python entries. The
    ref carries the image's vendor word (`entry_ref_vendor`, 0 = the host
    image) and its compiled tier (`entry_ref_mode`, 0 = fast), both checked
    against the publication when it is resolved: no door asks the binary
    for either again."""
    var nk = trees_keys()
    var key = py_get(table, nk[].k_gbdt_native_key)
    if key is None:
        raise Error("ImportError: mojolearn: the gbdt binding publishes no native entry table")
    return xg_ref(TreesEntryKey(Int(py=key[0]), Int(py=key[1])))


def _text_span(text: PythonObject, mut nbytes: Int) raises -> Int:
    """UNPACK of the model text: the address and byte count of the UTF-8
    bytes CPython holds in the str (no copy; the image copies them once, as
    its Python entry did). The span lives as long as `text`, which the
    caller keeps named past the native call."""
    # cpython_utf8 (bindings/py_cpython_table.mojo): the size read back
    # through a volatile slot; the former `Int(MutPointer(to=size))` form
    # handed the image a byte count of 0 (lane sr-gbdt).
    var table = cpython_table()
    var address = cpython_utf8(table, text._obj_ptr, nbytes)
    if address == 0:
        table[].PyErr_Clear()
        raise Error("TypeError: mojolearn: model_ must be a str")
    return address


def gbdt_has_ctr(text: PythonObject) raises -> Bool:
    """`ensemble._has_ctr_records`: a CTR table or tensor CTR registry."""
    var b = trees_builtins()
    var names: List[String] = ["ctr_columns", "ctr_table", "tensor_ctr_registry"]
    if py_type(text) is b.str:
        # Searched in place (lane perf-sweep-trees, 2026-11-05): the model
        # text can be megabytes, and every predict used to copy it whole.
        for i in range(3):  # driver-loop: over fixed call structure, small-loop(names): the three fixed record names
            if py_contains(text, PythonObject("\n" + names[i] + " ")):
                return True
            if Bool(py=py_call(text.startswith, PythonObject(names[i] + " "))):
                return True
        return False
    return gbdt_text_has_ctr(String(py_call(b.str, text)))


def gbdt_host_tree_context(context: PythonObject) raises -> PythonObject:
    """The CPU tree owner's context (`linear_model._CLASSICAL`'s
    `host_tree_context`, connected when the host surface loads), read from
    the loaded module's dictionary."""
    var nk = trees_keys()
    var module = py_get(py_item(context, nk[].k_modules), "mojolearn.linear_model")
    if module is None:
        raise Error("ImportError: mojolearn: the CPU tree owner is not connected")
    var classical = py_get(py_fields(module), nk[].k__CLASSICAL)
    var host = py_get(classical, nk[].k_host_tree_context) if classical is not None else PythonObject(None)
    if host is None:
        raise Error("ImportError: mojolearn: the CPU tree owner is not connected")
    return host


def _field(fields: PythonObject, model: PythonObject, key: PythonObject, name: String) raises -> PythonObject:
    """An attribute the model holds (instance first, then its class), or
    the AttributeError Python's attribute read raised. `key` is the held key
    of `name` (`nk[].k_<name>`; lane bw3-trees: up to three str were built
    from the name per call); `name` is for the error text."""
    var missing = Python.tuple()
    var cls = py_type(model)
    var member = _resolved(cls, key, missing)
    if member is not missing and _resolved(py_type(member), "__get__", missing) is not missing:
        raise Error("TypeError: native GBDT fields cannot invoke Python property/descriptor hooks")
    if py_contains(fields, key):
        return native_instance_item(fields, key)
    if member is not missing:
        return member
    raise Error("AttributeError: '" + String(cls.__name__) + "' object has no attribute '" + name + "'")


def _int(value: PythonObject) raises -> Int:
    return Int(py=py_call_typed(trees_builtins().int, value))


# -------------------------------------------------------- resident model


def gbdt_resident(fields: PythonObject, table: PythonObject, entry: EntryRef) raises -> Int:
    """`GradientBoosting._resident_handle`: the handle of the device copy of
    `model_`, prepared on the first call and reused while the binding and
    the text bytes are the same; -1 where the image holds no resident model
    (the host image: the per-call parse runs). The holder tuple
    (lifetime, binding, text, handle) releases the copy when it is
    dropped."""
    if not xg_resident(entry):
        return -1
    var nk = trees_keys()
    var text = py_item(fields, nk[].k_model_)
    var cached = py_get(fields, nk[].k__resident)
    if cached is not None and py_type(cached) is trees_builtins().tuple and len(cached) == 4:
        if cached[1] is table and (cached[2] is text or Bool(py=cached[2] == text)):
            return Int(py=cached[3])
    # A different text or binding: dropping the holder releases its copy.
    _ = py_call(fields.pop, "_resident", PythonObject(None))
    var nbytes = 0
    var address = _text_span(text, nbytes)
    var handle = xg_resident_prepare(entry, address, nbytes)
    _ = text  # the str owns the bytes the image read above
    # The lifetime owner (bindings/trees_lifetime_door.mojo, shared with the
    # forests) releases through the image's Python release entry.
    var release = trees_entry(table, "gbdt_resident_release")
    var lifetime = trees_resident_new(release, handle)
    py_setitem(fields, nk[].k__resident, Python.tuple(lifetime, table, text, PythonObject(handle)))
    return handle


struct GbdtApply(Movable):
    """One predict as UNPACK leaves it. RUN (`gbdt_apply_run`,
    `gbdt_apply_single`) reads the plain words only; `held_x` and
    `held_text` own what `x` and `text` address until PACK returns.

    Fields:
        entry: the native entry of the gbdt image of the estimator's run
            context.
        handle: the image's resident model, or -1: the per-call parse of
            `text` runs.
        text, text_bytes: the model text's UTF-8 span (per-call parse only).
        x, rows, row_major: the admitted float32 block (column-major, or
            C-order for a resident model that takes it as it is).
    """
    var entry: EntryRef
    var handle: Int
    var text: Int
    var text_bytes: Int
    var x: Int
    var rows: Int
    var row_major: Bool
    var held_x: AdmittedX
    var held_text: PythonObject

    def __init__(out self, entry: EntryRef, handle: Int, text: Int, text_bytes: Int, var held_x: AdmittedX,
                 held_text: PythonObject):
        self.entry = entry
        self.handle = handle
        self.text = text
        self.text_bytes = text_bytes
        self.x = held_x.address
        self.rows = held_x.rows
        self.row_major = held_x.c_order
        self.held_x = held_x^
        self.held_text = held_text


def _fitted_x(fields: PythonObject, X: PythonObject, layout: Bool, before: String, context: PythonObject) raises -> AdmittedX:
    """`_check_fitted_layout` (layout) / `_check_fitted`: the fitted check,
    X in the entry's layout, the fitted width."""
    var nk = trees_keys()
    if py_item(fields, nk[].k_model_) is None:
        raise Error("RuntimeError: mojolearn: " + before + " before fit()")
    # lane cat-ctr: a model fitted with cat_features carries its category
    # map; its categorical columns are admitted like the fit's (numbers,
    # text or None) and encoded through the map, searched in place, before
    # RUN. One dict read per call with the interned key; None for every
    # other model, whose admission below is unchanged.
    var category_map = py_get(fields, nk[].k_category_map_)
    if category_map is not None:
        var fitted_width = _int(py_item(fields, nk[].k_n_features_in_))
        var cat_columns = gbdt_category_block_list(category_map, fitted_width)
        var cat_keys = List[UInt64]()
        var xc = gbdt_category_admit(X, "X", cat_columns, cat_keys, context)
        if xc.cols != fitted_width:
            raise Error("mojolearn: model was fitted on " + String(fitted_width) + " features, got " + String(xc.cols))
        gbdt_category_apply(xc, cat_keys, category_map, fitted_width, context)
        return xc^
    var x = trees_forest_x(X, "X", context) if layout else admit_f32(X, 2, "X", context, order="F", finite=False)
    var fitted = _int(py_item(fields, nk[].k_n_features_in_))
    if x.cols != fitted:
        raise Error("mojolearn: model was fitted on " + String(fitted) + " features, got " + String(x.cols))
    return x^


def gbdt_apply_unpack(fields: PythonObject, X: PythonObject, entry: EntryRef, handle: Int,
                      context: PythonObject) raises -> GbdtApply:
    """UNPACK of a predict: the fitted check, X admitted in the layout the
    image takes (either order for a resident model, column-major for the
    per-call parse), the fitted width, and for the per-call parse the model
    text's span."""
    var x = _fitted_x(fields, X, handle >= 0, "predict()", context)
    if handle >= 0:
        return GbdtApply(entry, handle, 0, 0, x^, PythonObject(None))
    var nk = trees_keys()
    var text = py_item(fields, nk[].k_model_)
    var nbytes = 0
    var address = _text_span(text, nbytes)
    return GbdtApply(entry, handle, address, nbytes, x^, text)


def gbdt_apply_run(call: GbdtApply, mode: Int, dst: Int) raises -> Int:
    """RUN of a predict in `mode` into the caller's block at `dst`: the
    resident model, or the per-call parse of a multi-output model. Plain
    words through the image's native entry (the image releases the GIL
    around its work). Returns the width written per row."""
    if call.handle >= 0:
        return xg_resident_predict(call.entry, call.handle, call.x, call.rows, dst, mode, call.row_major)
    return xg_predict_multi(call.entry, call.text, call.text_bytes, call.x, call.rows, dst, mode)


def gbdt_apply_single(call: GbdtApply, dst: Int) raises -> Int:
    """RUN of the per-call parse of a one-dimensional model: raw scores,
    the rows written."""
    return xg_predict(call.entry, call.text, call.text_bytes, call.x, call.rows, dst)


# --------------------------------------------------------------- predict


def gbdt_predict(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    """`GradientBoosting.predict`: raw float32 scores. UNPACK
    (`gbdt_apply_unpack`), RUN (`gbdt_apply_run` / `gbdt_apply_single`),
    PACK (the output array's shape)."""
    var fields = py_fields(model)
    var nk = trees_keys()
    var text = _field(fields, model, nk[].k_model_, "model_")
    if text is None:
        raise Error("RuntimeError: mojolearn: predict() before fit()")
    var table = gbdt_table(fields, context)
    var entry = gbdt_entry_of(table)
    if entry_ref_vendor(entry) == 0 and gbdt_has_ctr(text):
        # The CPU restatement of a CTR model's apply (`HostGBDT`).
        var args = Python.tuple(X, py_item(fields, nk[].k_loss), text, py_item(fields, nk[].k_n_features_in_), py_item(fields, nk[].k_approx_dim_),
                                py_item(fields, nk[].k_n_classes_), py_type(model).__name__, py_get(fields, nk[].k_category_map_))
        return host_tree_operation(PythonObject(None), "host_tree_ctr_predict", args, gbdt_host_tree_context(context))
    var dim = _int(py_item(fields, nk[].k_approx_dim_))
    var call = gbdt_apply_unpack(fields, X, entry, gbdt_resident(fields, table, entry), context)
    var out = admitted_vector(call.rows * dim, "<f4", context)
    if call.handle < 0 and dim == 1:
        var wrote = gbdt_apply_single(call, trees_address(out))
        _ = out  # keep the Python object alive through the native call above
        if wrote != call.rows:
            raise Error("RuntimeError: mojolearn: predict wrote " + String(wrote) + " of " + String(call.rows) + " rows")
        return out
    var width = gbdt_apply_run(call, _PREDICT_RAW, trees_address(out))
    _ = out  # keep the Python object alive through the native call above
    if width != dim:
        raise Error("RuntimeError: mojolearn: predict wrote width " + String(width) + ", expected " + String(dim))
    if dim == 1:
        return out
    return trees_reshape(out, Python.tuple(call.rows, width), context)


def _is_multi_output(loss: PythonObject) raises -> Bool:
    return Bool(py=loss == PythonObject("MultiClass")) or Bool(py=loss == PythonObject("MultiClassOneVsAll"))


def _is_binary(loss: PythonObject) raises -> Bool:
    return Bool(py=loss == PythonObject("Logloss")) or Bool(py=loss == PythonObject("CrossEntropy"))


def gbdt_predict_proba(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    """`GradientBoosting.predict_proba`: softmax (MultiClass), elementwise
    sigmoid (MultiClassOneVsAll) or float64 `[1 - p, p]` (Logloss,
    CrossEntropy)."""
    var fields = py_fields(model)
    var nk = trees_keys()
    var loss = _field(fields, model, nk[].k_loss, "loss")
    if _is_multi_output(loss):
        if _field(fields, model, nk[].k_model_, "model_") is None:
            raise Error("RuntimeError: mojolearn: predict_proba() before fit()")
        var mode = _PREDICT_SIGMOID if Bool(py=loss == PythonObject("MultiClassOneVsAll")) else _PREDICT_SOFTMAX
        var table = gbdt_table(fields, context)
        var entry = gbdt_entry_of(table)
        var classes = _int(py_item(fields, nk[].k_n_classes_))
        var call = gbdt_apply_unpack(fields, X, entry, gbdt_resident(fields, table, entry), context)
        var out = admitted_vector(call.rows * classes, "<f4", context)
        var width = gbdt_apply_run(call, mode, trees_address(out))
        _ = out  # keep the Python object alive through the native call above
        if width != classes:
            raise Error("RuntimeError: mojolearn: predict_proba wrote width " + String(width) + ", expected " + String(classes))
        return trees_reshape(out, Python.tuple(call.rows, width), context)
    if not _is_binary(loss):
        raise Error("mojolearn: predict_proba is defined for Logloss, "
                    + "CrossEntropy, MultiClass and MultiClassOneVsAll; this model was fitted with "
                    + _repr(loss) + ". Use predict() and apply the link "
                    + "yourself.")
    if _field(fields, model, nk[].k_model_, "model_") is None:
        raise Error("RuntimeError: mojolearn: predict_proba() before fit()")
    var table = gbdt_table(fields, context)
    var entry = gbdt_entry_of(table)
    var handle = gbdt_resident(fields, table, entry)
    if handle >= 0:
        var call = gbdt_apply_unpack(fields, X, entry, handle, context)
        var out = admitted_array(Python.tuple(call.rows, 2), "<f8", context)
        var width = gbdt_apply_run(call, _PREDICT_SIGMOID_PAIR, trees_address(out))
        _ = out  # keep the Python object alive through the native call above
        if width != 2:
            raise Error("RuntimeError: mojolearn: predict_proba wrote width " + String(width) + ", expected 2")
        return out
    var raw32 = gbdt_predict(model, X, context)
    var n = Int(py=raw32.shape[0])
    var raw = admitted_vector(Int(py=raw32.size), "<f8", context)
    trees_cast(trees_address(raw32), "<f4", trees_address(raw), "<f8", Int(py=raw32.size), context)
    _ = raw  # keep the Python object alive through the native call above
    _ = raw32  # keep the Python object alive through the native call above
    var out = admitted_array(Python.tuple(n, 2), "<f8", context)
    var wrote = xg_sigmoid_pair(entry, trees_address(raw), trees_address(out), n)
    _ = out  # keep the Python object alive through the native call above
    _ = raw  # keep the Python object alive through the native call above
    if wrote != n:
        raise Error("RuntimeError: mojolearn: gbdt_sigmoid_pair wrote " + String(wrote) + " of " + String(n) + " rows")
    return out


def _argmax_rows(scores: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_labels.argmax_rows` of a C-order float32 or float64 block."""
    var rows = Int(py=scores.shape[0])
    var cols = Int(py=scores.shape[1])
    var out = admitted_vector(rows, "<i8", context)
    if rows == 0:
        return out
    var wide = String(scores.dtype) == "<f8"
    _ = py_call_typed(trees_helper(context, "argmax_rows_f64" if wide else "argmax_rows_f32"),
                      PythonObject(trees_address(scores)), PythonObject(rows), PythonObject(cols),
                      PythonObject(trees_address(out)))
    _ = out  # keep the Python object alive through the native call above
    return out


def gbdt_predict_classes(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    """`GradientBoosting.predict_classes`: int64 class codes; under FAST the
    resident direct-class kernels where they apply, else the argmax of
    `predict_proba`."""
    var fields = py_fields(model)
    var nk = trees_keys()
    var loss = _field(fields, model, nk[].k_loss, "loss")
    if not _is_multi_output(loss) and not _is_binary(loss):
        raise Error("mojolearn: predict_classes needs a classification loss; "
                    + "this model was fitted with " + _repr(loss) + ".")
    var table = gbdt_table(fields, context)
    var entry = gbdt_entry_of(table)
    if entry_ref_mode(entry) == 0:
        # A fast image (`numeric_mode_used()` == "fast": the tier word of
        # the resolved entry).
        if _is_binary(loss):
            var handle = gbdt_resident(fields, table, entry)
            if handle >= 0:
                var call = gbdt_apply_unpack(fields, X, entry, handle, context)
                var out = admitted_vector(call.rows, "<i8", context)
                if call.rows <= 0:
                    raise Error("invalid resident binary class prediction row count")
                var width = gbdt_apply_run(call, _PREDICT_CLASSES_BINARY, trees_address(out))
                _ = out  # keep the Python object alive through the native call above
                if width != 1:
                    raise Error("resident binary class prediction returned an invalid width")
                return out
        elif _int(py_item(fields, nk[].k_n_classes_)) <= 3 and Bool(py=_field(fields, model, nk[].k_grow_policy, "grow_policy") == PythonObject("SymmetricTree")):
            if py_item(fields, nk[].k_model_) is None:
                raise Error("RuntimeError: mojolearn: predict_proba() before fit()")
            var handle = gbdt_resident(fields, table, entry)
            if handle >= 0:
                var call = gbdt_apply_unpack(fields, X, entry, handle, context)
                var out = admitted_vector(call.rows, "<i8", context)
                var mode = _PREDICT_CLASSES_PINNED if Bool(py=loss == PythonObject("MultiClass")) else _PREDICT_CLASSES_OVA
                var width = gbdt_apply_run(call, mode, trees_address(out))
                _ = out  # keep the Python object alive through the native call above
                if width != 1:
                    raise Error("RuntimeError: mojolearn: predict_classes wrote width " + String(width) + ", expected 1")
                return out
    return _argmax_rows(gbdt_predict_proba(model, X, context), context)


# ------------------------------------------------------------- adapters


def _adapter_fitted(model: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_GBDTAdapter._check_fitted`: the fitted learner, or NotFittedError."""
    var fields = py_fields(model)
    var nk = trees_keys()
    var learner = py_get(fields, nk[].k__learner_)
    var fitted = learner is not None and py_get(py_fields(learner), nk[].k_model_) is not None
    if not fitted:
        _ = classical_not_fitted_class(context)
        raise Error("NotFittedError: " + String(py_type(model).__name__) + " is not fitted")
    return learner


def adapter_decision_function(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    """`GradientBoostingClassifier.decision_function` and
    `GradientBoostingRegressor.predict`: the learner's raw scores."""
    return gbdt_predict(_adapter_fitted(model, context), X, context)


def _adapter_binary(model: PythonObject, X: PythonObject, probabilities: Bool, context: PythonObject) raises -> PythonObject:
    """`_GBDTAdapter._binary_output`: FAST labels straight from the
    learner's class kernels; else the margins through the binding's binary
    probability or class kernel."""
    var fields = py_fields(model)
    var nk = trees_keys()
    var mode = py_get(fields, nk[].k_numeric_mode_)
    if not probabilities and mode is not None and Bool(py=mode == PythonObject("fast")):
        var learner = py_get(fields, nk[].k__learner_)
        if learner is None:
            raise Error("AttributeError: '" + String(py_type(model).__name__) + "' object has no attribute '_learner_'")
        return gbdt_predict_classes(learner, X, context)
    var margins = adapter_decision_function(model, X, context)
    if (Int(py=margins.ndim) != 1 or String(margins.dtype) != "<f4"
            or not trees_finite_f32(trees_address(margins), Int(py=margins.size))):
        raise Error("Classifier margins must be finite scalar Float32 values")
    _ = margins  # keep the Python object alive through the native calls above
    var n = Int(py=margins.size)
    var table = trees_table(context, String(mode) if mode is not None else gbdt_mode(fields, context), "_mojolearn_gbdt")
    var entry = gbdt_entry_of(table)
    var output = admitted_array(Python.tuple(n, 2), "<f4", context) if probabilities else admitted_vector(n, "<i4", context)
    var wrote = xg_binary(entry, probabilities, trees_address(margins), trees_address(output), n)
    _ = output  # keep the Python object alive through the native call above
    _ = margins  # keep the Python object alive through the native call above
    if wrote != (2 * n if probabilities else n):
        raise Error("RuntimeError: GBDT binary output returned an unexpected row count")
    return output


def adapter_predict_proba(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    return _adapter_binary(model, X, True, context)


def adapter_codes(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_binary_output(X, False)`: the class codes `score` compares."""
    return _adapter_binary(model, X, False, context)


def adapter_predict(model: PythonObject, X: PythonObject, context: PythonObject) raises -> PythonObject:
    """`GradientBoostingClassifier.predict`: the class codes decoded through
    `classes_`."""
    _ = _adapter_fitted(model, context)
    var codes = _adapter_binary(model, X, False, context)
    var nk = trees_keys()
    return trees_decode_labels(py_item(py_fields(model), nk[].k_classes_), codes, context)


# -------------------------------------------------------- tree metadata


def gbdt_tree_metadata_op(model: PythonObject, which: String, context: PythonObject) raises -> PythonObject:
    """`get_tree_leaf_counts` (uint32 per tree) / `get_leaf_values`
    (float64, tree-major then leaf-major) from the model text."""
    var fields = py_fields(model)
    var nk = trees_keys()
    var text = _field(fields, model, nk[].k_model_, "model_")
    if text is None:
        raise Error("RuntimeError: mojolearn: tree inspection before fit()")
    var parsed = gbdt_tree_metadata(String(py_call(trees_builtins().str, text)))
    if which == "counts":
        var n = len(parsed.counts)
        var out = admitted_vector(n, "<u4", context)
        if n:
            unsafe_memcpy(dest=MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=trees_address(out)),
                   src=parsed.counts.unsafe_ptr(), count=n)
            _ = out  # keep the Python object alive through the native call above
        return out
    var n = len(parsed.values)
    var out = admitted_vector(n, "<f8", context)
    if n:
        unsafe_memcpy(dest=MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=trees_address(out)),
               src=parsed.values.unsafe_ptr(), count=n)
        _ = out  # keep the Python object alive through the native call above
    return out


# ------------------------------------------------- OrderedRMSE / FeatureFreq


def _is_bool(value: PythonObject) raises -> Bool:
    var t = py_type(value)
    return t is trees_builtins().bool or String(t.__name__) == "bool_"


def _isinstance(value: PythonObject, cls: PythonObject) raises -> Bool:
    return Bool(py=py_call(trees_builtins().isinstance, value, cls))


def _ordered_options(fields: PythonObject, model: PythonObject, context: PythonObject) raises:
    """`OrderedRMSE._ordered_options` over the stored settings."""
    var b = trees_builtins()
    var names: List[String] = ["n_estimators", "max_depth", "border_count"]
    var upper: List[Int] = [2147483647, 8, 255]
    var nk = trees_keys()
    for i in range(3):  # driver-loop: over fixed call structure, small-loop(names): the three integer settings
        var value = _field(fields, model, PythonObject(names[i]), names[i])
        var ok = not _is_bool(value) and _isinstance(value, py_item(context, nk[].k_trees_integral_type))
        if ok:
            ok = Bool(py=value >= PythonObject(1)) and Bool(py=value <= PythonObject(upper[i]))
        if not ok:
            raise Error("mojolearn: " + names[i] + " must be an integer in 1.." + String(upper[i]))
    var reals: List[String] = ["learning_rate", "l2_leaf_reg"]
    for i in range(2):  # driver-loop: over fixed call structure, small-loop(reals): the two real settings
        var value = _field(fields, model, PythonObject(reals[i]), reals[i])
        var positive = i == 0
        var ok = not _is_bool(value) and _isinstance(value, py_item(context, nk[].k_trees_real_type))
        if ok:
            var f = Float64(py=py_call_typed(b.float, value))
            ok = isfinite(f) and abs(f) <= 3.4028234663852886e+38 and f >= 0 and not (positive and Float32(f) <= 0)
        if not ok:
            raise Error("mojolearn: " + reals[i] + " must be finite float32 and "
                        + ("positive" if positive else "non-negative"))


def ordered_option_values(settings: PythonObject, context: PythonObject) raises:
    """`OrderedRMSE._ordered_options(n_estimators, max_depth, border_count,
    learning_rate, l2_leaf_reg)` over the constructor's arguments, in that
    order: the same checks and wording as `_ordered_options` above.
    DEVIATION 2336: `numbers.Real` is every Python and NumPy int and float;
    the float32 bound is the literal `np.finfo(np.float32).max`; the float32
    cast sends 1e-100 to 0.0, which is refused as non-positive."""
    var b = trees_builtins()
    var names: List[String] = ["n_estimators", "max_depth", "border_count"]
    var upper: List[Int] = [2147483647, 8, 255]
    var nk = trees_keys()
    for i in range(3):  # driver-loop: over fixed call structure, small-loop(names): the three integer settings
        var value = settings[i]
        var ok = not _is_bool(value) and _isinstance(value, py_item(context, nk[].k_trees_integral_type))
        if ok:
            ok = Bool(py=value >= PythonObject(1)) and Bool(py=value <= PythonObject(upper[i]))
        if not ok:
            raise Error("mojolearn: " + names[i] + " must be an integer in 1.." + String(upper[i]))
    var reals: List[String] = ["learning_rate", "l2_leaf_reg"]
    for i in range(2):  # driver-loop: over fixed call structure, small-loop(reals): the two real settings
        var value = settings[3 + i]
        var positive = i == 0
        var ok = not _is_bool(value) and _isinstance(value, py_item(context, nk[].k_trees_real_type))
        if ok:
            var f = Float64(py=py_call_typed(b.float, value))
            ok = isfinite(f) and abs(f) <= 3.4028234663852886e+38 and f >= 0 and not (positive and Float32(f) <= 0)
        if not ok:
            raise Error("mojolearn: " + reals[i] + " must be finite float32 and "
                        + ("positive" if positive else "non-negative"))


def _checked_weights(source: PythonObject, n: Int, wrap: Bool, context: PythonObject) raises -> AdmittedX:
    """The OrderedRMSE / FeatureFreq weight checks: one value per row,
    finite, non-negative, a positive sum."""
    var w: AdmittedX
    try:
        w = admit_f32(source, 1, "sample_weight", context, finite=False)
    except error:
        if wrap and _type_or_value(error):
            raise Error("mojolearn: sample_weight must have one value per row")
        raise error^
    if w.rows != n:
        raise Error("mojolearn: sample_weight must have one value per row")
    var nk = trees_keys()
    var helpers = base_helpers_key(py_item(context, nk[].k_trees_helpers))
    if not trees_finite_f32(w.address, n) or base_reduce_stat(helpers, w.address, 0, n, 0).f < 0.0:
        raise Error("mojolearn: sample_weight must be finite and non-negative")
    if not base_reduce_stat(helpers, w.address, 0, n, 1).f > 0.0:
        raise Error("mojolearn: sample_weight must have positive sum")
    return w^


def _fitted_text(fields: PythonObject, model: PythonObject, n_features: Int, iteration: PythonObject) raises:
    var nk = trees_keys()
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(n_features))
    py_setitem(fields, nk[].k_approx_dim_, PythonObject(1))
    py_setitem(fields, nk[].k_n_classes_, PythonObject(None))
    py_setitem(fields, nk[].k_bias_, PythonObject(0.0))
    py_setitem(fields, nk[].k_best_iteration_, iteration)
    py_setitem(fields, nk[].k_stopped_early_, PythonObject(False))
    py_setitem(fields, nk[].k_loss_curve_, PythonObject(None))
    py_setitem(fields, nk[].k_test_loss_curve_, PythonObject(None))


struct GbdtTextFit(Movable):
    """An OrderedRMSE or two-level FeatureFreq fit as UNPACK leaves it. RUN
    (`xg_ordered_fit` / `xg_feature_freq_fit`, gbdt/fit_calls.mojo) reads
    the plain words only; the `held_*` fields own what the addresses point
    into until PACK returns.

    Fields:
        entry: the native entry of the gbdt image of the estimator's run
            context.
        x, y, weights: float32 blocks (X column-major); `weights` is y's
            address when `n_weights` is 0, as the entries always took it.
        extra, n_extra: the UInt32 permutation (one id per row) or the
            UInt32 source columns.
        n_estimators, max_depth, border_count: OrderedRMSE's settings.
        random_seed: FeatureFreq's `random_state`.
    """
    var entry: EntryRef
    var x: Int
    var y: Int
    var weights: Int
    var extra: Int
    var n_rows: Int
    var n_features: Int
    var n_weights: Int
    var n_extra: Int
    var n_estimators: Int
    var max_depth: Int
    var border_count: Int
    var learning_rate: Float64
    var l2_leaf_reg: Float64
    var random_seed: Int
    var held_x: AdmittedX
    var held_y: AdmittedX
    var held_weights: Optional[AdmittedX]
    var held_extra: PythonObject

    def __init__(out self, entry: EntryRef, weights: Int, n_weights: Int, extra: Int, n_extra: Int,
                 var held_x: AdmittedX, var held_y: AdmittedX, var held_weights: Optional[AdmittedX],
                 held_extra: PythonObject):
        self.entry = entry
        self.x = held_x.address
        self.y = held_y.address
        self.weights = weights
        self.extra = extra
        self.n_rows = held_x.rows
        self.n_features = held_x.cols
        self.n_weights = n_weights
        self.n_extra = n_extra
        self.n_estimators = 0
        self.max_depth = 0
        self.border_count = 0
        self.learning_rate = 0.0
        self.l2_leaf_reg = 0.0
        self.random_seed = 0
        self.held_x = held_x^
        self.held_y = held_y^
        self.held_weights = held_weights^
        self.held_extra = held_extra


def _real_setting(fields: PythonObject, model: PythonObject, key: PythonObject, name: String) raises -> Float64:
    """`float(self.<name>)` of a stored setting."""
    return Float64(py=py_call_typed(trees_builtins().float, _field(fields, model, key, name)))


def ordered_rmse_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`OrderedRMSE.fit(X, y, permutation=, sample_weight=)`: UNPACK
    (`ordered_rmse_unpack`: every check in the former order and words), RUN
    (the fit in the gbdt image of the estimator's run context, through its
    native entry: plain words, no Python object), PACK (the model text and
    the fitted attributes)."""
    var call = ordered_rmse_unpack(model, args, context)
    var text = xg_ordered_fit(call.entry, call.x, call.y, call.weights, call.extra, call.n_rows, call.n_features,
                              call.n_weights, call.n_estimators, call.max_depth, call.border_count,
                              call.learning_rate, call.l2_leaf_reg)
    var fields = py_fields(model)
    var nk = trees_keys()
    py_setitem(fields, nk[].k_model_, PythonObject(text))
    _fitted_text(fields, model, call.n_features, PythonObject(None))
    return model


def ordered_rmse_unpack(model: PythonObject, args: PythonObject, context: PythonObject) raises -> GbdtTextFit:
    """UNPACK of `OrderedRMSE.fit`: the settings, X, y, the permutation and
    the weights admitted and checked, the request's plain words."""
    var b = trees_builtins()
    var fields = py_fields(model)
    _ordered_options(fields, model, context)
    var x = admit_f32(args[0], 2, "X", context, order="F", finite=False)
    var n = x.rows
    if n < 4 or n > 4294967295 or x.cols < 1:
        raise Error("mojolearn: ordered RMSE requires >=4 rows and >=1 feature")
    if not trees_finite_f32(x.address, n * x.cols):
        raise Error("mojolearn: ordered RMSE X must be finite")
    _ = x  # keep the Python object alive through the native calls above
    var y: AdmittedX
    try:
        y = admit_f32(args[1], 1, "y", context, finite=False)
    except error:
        if _type_or_value(error):
            raise Error("mojolearn: y must be one-dimensional with one value per row")
        raise error^
    if y.rows != n:
        raise Error("mojolearn: y must be one-dimensional with one value per row")
    if not trees_finite_f32(y.address, n):
        raise Error("mojolearn: y must be finite")
    var permutation = args[2]
    var bijection = String("mojolearn: permutation must be an integer bijection of row ids")
    var nk = trees_keys()
    if py_type(permutation) is b.list or py_type(permutation) is b.tuple:
        # An exact int is integral and not a bool: only other items take the
        # isinstance checks (lane perf-trees-owed; the connected integral
        # type was also looked up in the context once per item).
        var int_t = b.int
        var integral = py_item(context, nk[].k_trees_integral_type)
        for i in range(len(permutation)):  # cpu-route: user permutation Python list type check [py-data-loop]
            var v = permutation[i]
            if py_type(v) is int_t:
                continue
            if _is_bool(v) or not _isinstance(v, integral):
                raise Error(bijection)
    elif py_type(permutation) is py_item(context, nk[].k_array_class):
        var letter = String(String(permutation.dtype)[byte=1:2])
        if letter != "i" and letter != "u":
            raise Error(bijection)
    else:
        # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
        var view = Optional[PythonObject]()
        try:
            view = py_present(py_call(b.memoryview, permutation))
        except:
            view = Optional[PythonObject]()
        if Bool(view):
            var fmt = String(view.value().format)
            var code = String(fmt[byte=fmt.byte_length() - 1:fmt.byte_length()])
            if code == "?" or code == "f" or code == "d" or code == "e" or code == "O":
                raise Error(bijection)
    var order64 = PythonObject(None)
    try:
        var view = trees_view_i64(permutation, context)
        order64 = view
    except error:
        if _type_or_value(error) or String(error).startswith("OverflowError:"):
            raise Error(bijection)
        raise error^
    var m = Int(py=order64.size)
    if m != n or n < 1 or base_check_indices_i64(base_helpers_key(py_item(context, nk[].k_trees_helpers)),
                                                 trees_address(order64), n, n) != 0:
        raise Error(bijection)
    _ = order64  # keep the Python object alive through the native calls above
    var order = admitted_vector(n, "<u4", context)
    trees_cast(trees_address(order64), "<i8", trees_address(order), "<u4", n, context)
    _ = order  # keep the Python object alive through the native call above
    _ = order64  # keep the Python object alive through the native call above
    var w_addr = y.address
    var n_weights = 0
    var weights: Optional[AdmittedX] = None
    if args[3] is not None:
        weights = _checked_weights(args[3], n, True, context)
        w_addr = weights.value().address
        n_weights = n
    var entry = gbdt_entry_of(gbdt_table(fields, context))
    var n_estimators = _int(_field(fields, model, nk[].k_n_estimators, "n_estimators"))
    var max_depth = _int(_field(fields, model, nk[].k_max_depth, "max_depth"))
    var border_count = _int(_field(fields, model, nk[].k_border_count, "border_count"))
    var learning_rate = _real_setting(fields, model, nk[].k_learning_rate, "learning_rate")
    var l2_leaf_reg = _real_setting(fields, model, nk[].k_l2_leaf_reg, "l2_leaf_reg")
    # x, y (the float32 target copy the fit reads), the weights and the
    # UInt32 order stay alive in the request until PACK returns.
    var call = GbdtTextFit(entry, w_addr, n_weights, trees_address(order), n, x^, y^, weights^, order)
    call.n_estimators = n_estimators
    call.max_depth = max_depth
    call.border_count = border_count
    call.learning_rate = learning_rate
    call.l2_leaf_reg = l2_leaf_reg
    return call^


def trees_view_i64(source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_buffer.as_i64_c(source, ndim=1, name="permutation")`: a C-order
    int64 vector (the input itself, or one converted copy)."""
    var b = trees_builtins()
    var prepared = source
    if py_type(source) is b.range:
        # `_materialize` lists a range's ints (a C iteration).
        prepared = py_call(b.list, source)
    # The one integer admission (admit_ids: a TypeError for a float or
    # text input, which the caller refuses as a non-bijection).
    return admit_ids(prepared, "permutation", context)


def _source_columns(sources: PythonObject) raises -> List[Int]:
    """UNPACK of `sources`: the named source columns as native ints, read
    once; every check of them runs on this list."""
    var cols = List[Int](capacity=len(sources))
    for i in range(len(sources)):  # driver-loop: over the named source columns, unpacked once
        cols.append(_int(sources[i]))
    return cols^


def _int_list_text(values: List[Int]) -> String:
    """`[a, b, ...]`, the refusal's spelling of a typed column list."""
    var text = String("[")
    for i in range(len(values)):  # small-loop(the named source columns of one refusal message)
        if i > 0:
            text += ", "
        text += String(values[i])
    return text + "]"


def feature_freq_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`ExperimentalTwoLevelFeatureFreq.fit(X, y, sample_weight=)`: UNPACK
    (`feature_freq_unpack`), RUN (the fit in the gbdt image of the
    estimator's run context, through its native entry), PACK (the model
    text and the fitted attributes)."""
    var call = feature_freq_unpack(model, args, context)
    var text = xg_feature_freq_fit(call.entry, call.x, call.y, call.weights, call.extra, call.n_rows,
                                   call.n_features, call.n_weights, call.n_extra, call.learning_rate,
                                   call.l2_leaf_reg, call.random_seed)
    var fields = py_fields(model)
    var nk = trees_keys()
    py_setitem(fields, nk[].k_model_, PythonObject(text))
    _fitted_text(fields, model, call.n_features, PythonObject(0))
    return model


def feature_freq_unpack(model: PythonObject, args: PythonObject, context: PythonObject) raises -> GbdtTextFit:
    """UNPACK of the FeatureFreq fit: reads X, `sources`, y and
    sample_weight into native values; every admission check after it runs
    on those typed values (column numbers, pointers and `HpStat` answers),
    never on Python objects."""
    var fields = py_fields(model)
    var x = admit_f32(args[0], 2, "X", context, order="F", finite=False)
    var n = x.rows
    var d = x.cols
    if n == 0 or d < 2:
        raise Error("mojolearn: experimental FeatureFreq needs rows and columns")
    var nk = trees_keys()
    var sources = _source_columns(_field(fields, model, nk[].k_sources, "sources"))
    var helpers = base_helpers_key(py_item(context, nk[].k_trees_helpers))
    for i in range(len(sources)):  # driver-loop: over the named source columns
        var f = sources[i]
        if f < 0 or f >= d:
            raise Error("mojolearn: sources " + _int_list_text(sources) + " outside " + String(d) + " features")
    if not trees_finite_f32(x.address, n * d):
        raise Error("mojolearn: experimental FeatureFreq X must be finite")
    _ = x  # keep the Python object alive through the native calls above
    var is_source = List[Bool](length=d, fill=False)
    for i in range(len(sources)):  # driver-loop: over the named source columns, one native stat set each
        var f = sources[i]
        is_source[f] = True
        var col = x.address + 4 * f * n
        _ = x  # keep the Python object alive through the native call above
        var hi = base_reduce_stat(helpers, col, 0, n, 1).f
        _ = x  # keep the Python object alive through the native call above
        if base_reduce_stat(helpers, col, 0, n, 4).i == 0 or base_reduce_stat(helpers, col, 0, n, 0).f < 0.0:
            raise Error("mojolearn: experimental FeatureFreq source columns must "
                        + "contain non-negative integer category codes")
        _ = x  # keep the Python object alive through the native calls above
        var k = Int(hi) + 1
        var dense = 2 <= k and k <= n
        if dense:
            var codes = admitted_vector(n, "<i8", context)
            trees_cast(col, "<f4", trees_address(codes), "<i8", n, context)
            _ = codes  # keep the Python object alive through the native call above
            _ = x  # keep the Python object alive through the native call above
            var counts = admitted_vector(k, "<i8", context)
            _ = base_bincount_i64(helpers, trees_address(codes), 3, n, k, trees_address(counts), False)
            _ = codes  # keep the Python object alive through the native call above
            _ = counts  # keep the Python object alive through the native call above
            dense = base_reduce_stat(helpers, trees_address(counts), 3, k, 0).i > 0
            _ = counts  # keep the Python object alive through the native call above
        if not dense:
            raise Error("mojolearn: every experimental FeatureFreq source must "
                        + "be densely coded 0..k-1; column " + String(f) + " has codes up to " + String(Int(hi))
                        + " with at least one missing (or fewer than two codes)")
    for f in range(d):  # driver-loop: over the feature columns, one native stat pair each
        if is_source[f]:
            continue
        var col = x.address + 4 * f * n
        _ = x  # keep the Python object alive through the native call above
        if base_reduce_stat(helpers, col, 0, n, 0).f == base_reduce_stat(helpers, col, 0, n, 1).f:
            raise Error("mojolearn: experimental FeatureFreq numeric columns "
                        + "must vary; column " + String(f) + " is constant")
        _ = x  # keep the Python object alive through the native calls above
    var y = admit_f32(args[1], 1, "y", context, finite=False)
    if y.rows != n or not trees_finite_f32(y.address, n):
        raise Error("mojolearn: y must be finite with one value per row")
    _ = y  # keep the Python object alive through the native calls above
    var w_addr = y.address
    _ = y  # keep the Python object alive through the native call above
    var n_weights = 0
    var weights: Optional[AdmittedX] = None
    if args[2] is not None:
        weights = _checked_weights(args[2], n, False, context)
        w_addr = weights.value().address
        _ = y  # keep the Python object alive through the native call above
        n_weights = n
    var k = len(sources)
    var source_array = admitted_vector(k, "<u4", context)
    var sp = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=trees_address(source_array))
    _ = source_array  # keep the Python object alive through the native call above
    for i in range(k):  # driver-loop: over the named source columns
        sp[i] = UInt32(sources[i])
        _ = source_array  # keep the Python object alive through the native call above
    var entry = gbdt_entry_of(gbdt_table(fields, context))
    var learning_rate = _real_setting(fields, model, nk[].k_learning_rate, "learning_rate")
    var l2_leaf_reg = _real_setting(fields, model, nk[].k_l2_leaf_reg, "l2_leaf_reg")
    var random_seed = _int(_field(fields, model, nk[].k_random_state, "random_state"))
    # x, y, the weights and the UInt32 source columns stay alive in the
    # request until PACK returns.
    var call = GbdtTextFit(entry, w_addr, n_weights, trees_address(source_array), k, x^, y^, weights^, source_array)
    call.learning_rate = learning_rate
    call.l2_leaf_reg = l2_leaf_reg
    call.random_seed = random_seed
    return call^
