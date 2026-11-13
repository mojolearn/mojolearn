# SPDX-License-Identifier: Apache-2.0
# IsolationForest's public operations in one native call each
# (python/mojolearn/_iforest_impl.py): `fit`, `score_samples`,
# `decision_function`, `predict` and `fit_predict`. This is the Python
# orchestration those methods ran, step for step: `_validate_params` with its
# wording, the float32 X, the device finite scan probe and its refusal, the
# resident forest token (IDN_IF_RESIDENT), the `iforest_run` launch with its
# sixteen (or seventeen) params, and `offset_` / `max_samples_` read back.
# The kernel and its arguments are the ones the Python path launched, so the
# forest, the scores and the labels keep their bits.
#
# The resident forest's release (formerly `weakref.finalize` plus a Python
# `__del__`) is a TreesResident lifetime: the binding's
# `iforest_resident_release` runs when the estimator drops it (a refit, a new
# token, the estimator's end). Lane perf-trees-owed (2026-11-06): the token,
# the training matrix it was minted for and that lifetime are one native side
# record of the estimator (bindings/trees_side_records.mojo), no longer the
# `_if_token` / `_if_lifetime` fields: scoring leaves `__dict__` unchanged
# (before, the first score after an unpickle or copy added the token), and a
# pickle or copy never carries a process-local handle; it mints its own on
# first use. The forest is a pure function of the seed and the X bits, so a
# re-minted forest scores bit for bit as the one it replaces.
#
# Round trips removed against the Python path: one native call replaces each
# method's Python sequence (validation, conversion, probes, buffers, launch,
# read-back); `fit_predict` is ONE `iforest_run` (fit a fresh forest and
# predict X with it) instead of the fit's one-row run and a second run for
# predict, and X is converted and handed over once.
from std.memory import unsafe_memcpy
from std.sys import is_defined
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem, py_setattr, py_fields
from bindings.py_typed_call import py_call_typed
from core.host_parallel_copy import host_parallel_copy_bytes
from bindings.native_buffer_owner_pack import native_storage_allocate
from bindings.trees_public_inputs_door import (
    trees_builtins, trees_table, trees_entry, trees_address, trees_call,
)
from bindings.admitted_input_unpack import AdmittedX, admit_f32, admitted_words, admitted_vector, classical_view, admitted_dtype
from bindings.classical_arrays_door import classical_address
from bindings.admitted_family_door import admitted_array as family_uninit_array
from bindings.classical_arrays_door import classical_is_f32, classical_c_order
from core.host_finite_scan import host_all_finite_f32
# TOMBSTONE: trees_f32 / trees_all_finite / TreesMatrix / trees_vector / trees_empty uses and _copied (the trees admission aliases) deleted 2026-11-09 by lane a1b-doors-opt-in-trees-linear; code recoverable at f59d913b0b.
# Restore: experiments/removed/A1B_LABELS_ADMISSION_COPY.patch.
from bindings.trees_lifetime_door import trees_resident_new
from bindings.trees_side_records_unpack import trees_side_get, trees_side_put, trees_side_drop, trees_finite_known
from bindings.native_buffer_owner_pack import native_buffer_view
from bindings.native_array_storage_door import native_scratch_bytes, native_scratch_address
from bindings.host_inference_scalars_door import host_inference_scalar
from bindings.metrics_native_support_door import metric_is_integral, metric_is_bool

comptime _WANT_SCORE_SAMPLES = 0
comptime _WANT_DECISION_FUNCTION = 1
comptime _WANT_PREDICT = 2
comptime _PREFIX = "mojolearn IsolationForest: "
comptime _NONFINITE = "mojolearn IsolationForest: X contains NaN or infinity"
comptime _TOKEN_SLOT = "trees_iforest_resident_v1"
# Lane restore-iforest-kde (2026-11-13, history audit row 3): the B arm of
# the owed fit A/B. Default (absent): a fit on a binary that keeps the forest
# resident borrows a float32 C-order X as 0.8.55 did (no copy). Defined: the
# fit owns a parallel copy of it (0292b73ab3, 2026-11-06). See `_fit`.
comptime _IF_FIT_OWN_X = is_defined["MOJOLEARN_IDN_IF_FIT_OWN_X"]()
# Lane restore-iforest-kde (history audit row 4): `want` + 3 asks a binary
# whose `iforest_scores_f64` is 1 for the float64 scores (bindings/
# _mojolearn_svm.mojo `_IF_WIDE_ROUTE`).
comptime _WANT_WIDE_BASE = 3
# Lane fix-iforest-save-device-forest (2026-11-13): the fitted forest itself,
# as the compact host form (isolation_forest/forest_io.mojo): a tuple of five
# Arrays (offsets <i4 n_trees + 1, feature <i4, threshold <f4, left <i4,
# right <i4). `save()` and pickle write it in place of the training matrix
# (the training matrix of a resident fit is the caller's borrowed buffer
# since restore-iforest-kde, so an in-place edit after fit reached the saved
# model); a loaded or unpickled estimator holds it instead of `_x` and
# scores it through `iforest_forest_run`, with no fit.
comptime _FOREST_FIELD = "_if_forest"


def _repr(value: PythonObject) raises -> String:
    return String(py_call(trees_builtins().repr, value))


def _field(fields: PythonObject, model: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """`self.<name>` of a stored constructor parameter."""
    var value = py_get(fields, PythonObject(name), context["trees_missing"])
    if value is context["trees_missing"]:
        raise Error("AttributeError: '" + String(py_type(model).__name__) + "' object has no attribute '" + name + "'")
    if name != "n_jobs" and name != "output_type":
        host_inference_scalar(value, context, True)
    return value


def iforest_validate(model: PythonObject, fields: PythonObject, context: PythonObject) raises:
    """`IsolationForest._validate_params()`: the parameter checks, in order,
    with their wording; each resolved knob is stored as it is checked."""
    var b = trees_builtins()
    var n_estimators = py_call_typed(b.int, _field(fields, model, "n_estimators", context))
    if Bool(py=n_estimators < PythonObject(1)):
        raise Error(_PREFIX + "n_estimators must be at least 1, got " + _repr(n_estimators))
    var max_samples = _field(fields, model, "max_samples", context)
    if Bool(py=py_call(b.isinstance, max_samples, b.str)):
        if String(py_call(max_samples.lower)) != "auto":
            raise Error(_PREFIX + "max_samples=" + _repr(max_samples) + " is not a name; it is 'auto', a positive int, "
                        + "or a float in (0.0, 1.0]")
        py_setitem(fields, "_max_samples_mode", PythonObject(0))
        py_setitem(fields, "_max_samples_int", PythonObject(256))
        py_setitem(fields, "_max_samples_frac", PythonObject(1.0))
    elif metric_is_bool(max_samples, context):
        raise Error(_PREFIX + "max_samples must be 'auto', an int, or a float, not a bool")
    elif metric_is_integral(max_samples, context):
        var count = py_call_typed(b.int, max_samples)
        if Bool(py=count <= PythonObject(0)):
            raise Error(_PREFIX + "max_samples must be a positive integer, got " + _repr(max_samples))
        py_setitem(fields, "_max_samples_mode", PythonObject(1))
        py_setitem(fields, "_max_samples_int", count)
        py_setitem(fields, "_max_samples_frac", PythonObject(1.0))
    else:
        var v = Float64(py=py_call_typed(b.float, max_samples))
        if not (v > 0.0 and v <= 1.0):
            raise Error(_PREFIX + "float max_samples must be in (0.0, 1.0], got " + _repr(max_samples))
        py_setitem(fields, "_max_samples_mode", PythonObject(2))
        py_setitem(fields, "_max_samples_int", PythonObject(256))
        py_setitem(fields, "_max_samples_frac", PythonObject(v))
    var max_depth = _field(fields, model, "max_depth", context)
    if max_depth is None:
        py_setitem(fields, "_max_depth", PythonObject(-1))
    else:
        var depth = py_call_typed(b.int, max_depth)
        py_setitem(fields, "_max_depth", depth)
        if Bool(py=depth < PythonObject(1)):
            raise Error(_PREFIX + "max_depth is None (their auto) or a positive int, got " + _repr(max_depth))
    var max_features = _field(fields, model, "max_features", context)
    if metric_is_bool(max_features, context):
        raise Error(_PREFIX + "max_features must be an int or a float, not a bool")
    var features_error = String(_PREFIX + "max_features must be an int in [1, n_features] or a float in (0.0, 1.0], got ")
    if metric_is_integral(max_features, context):
        var count = py_call_typed(b.int, max_features)
        if Bool(py=count < PythonObject(1)):
            raise Error(features_error + _repr(max_features))
        py_setitem(fields, "_max_features_mode", PythonObject(1))
        py_setitem(fields, "_max_features_int", count)
        py_setitem(fields, "_max_features_frac", PythonObject(1.0))
    else:
        var v = Float64(py=py_call_typed(b.float, max_features))
        if not (v > 0.0 and v <= 1.0):
            raise Error(features_error + _repr(max_features))
        py_setitem(fields, "_max_features_mode", PythonObject(0))
        py_setitem(fields, "_max_features_int", PythonObject(0))
        py_setitem(fields, "_max_features_frac", PythonObject(v))
    var contamination = _field(fields, model, "contamination", context)
    var contamination_error = String(_PREFIX + "contamination must be 'auto' or a float in the range (0.0, 0.5], got ")
    if Bool(py=py_call(b.isinstance, contamination, b.str)):
        if String(py_call(contamination.lower)) != "auto":
            raise Error(contamination_error + _repr(contamination))
        py_setitem(fields, "_contamination_auto", PythonObject(True))
        py_setitem(fields, "_contamination", PythonObject(0.0))
    else:
        var v = Float64(py=py_call_typed(b.float, contamination))
        if not (v > 0.0 and v <= 0.5):
            raise Error(contamination_error + _repr(contamination))
        py_setitem(fields, "_contamination_auto", PythonObject(False))
        py_setitem(fields, "_contamination", PythonObject(v))
    var random_state = _field(fields, model, "random_state", context)
    var seed = PythonObject(0)
    if random_state is not None:
        seed = py_call_typed(b.int, random_state)
        if Bool(py=seed < PythonObject(0)) or Bool(py=seed > PythonObject(4294967295)):
            raise Error(_PREFIX + "expected 0 <= random_state <= 2**32 - 1, got " + _repr(random_state))
    if Bool(py=py_call(b.bool, _field(fields, model, "warm_start", context))):
        raise Error("NotImplementedError: " + _PREFIX + "warm_start=True is not supported "
                    + "(cuML raises UnsupportedOnGPU for it too)")
    if _field(fields, model, "n_jobs", context) is not None:
        raise Error("NotImplementedError: " + _PREFIX + "n_jobs is refused; it is "
                    + "scikit-learn's CPU thread count and there is no CPU path in "
                    + "this library to spread")
    if Bool(py=py_call(b.bool, _field(fields, model, "verbose", context))):
        raise Error("NotImplementedError: " + _PREFIX + "verbose is refused; this implementation "
                    + "prints no log lines, so accepting it would be accepting-"
                    + "and-ignoring")
    if _field(fields, model, "output_type", context) is not None:
        raise Error("NotImplementedError: " + _PREFIX + "output_type is a cuML-internal "
                    + "array-type selector; this package returns mojolearn Arrays")
    py_setitem(fields, "_n_estimators", n_estimators)
    py_setitem(fields, "_bootstrap", PythonObject(Bool(py=py_call(b.bool, _field(fields, model, "bootstrap", context)))))
    py_setitem(fields, "_seed", seed)


def _mode(fields: PythonObject, context: PythonObject) raises -> String:
    """The tier `NumericModeMixin._bind` resolves: `numeric_mode`, or the
    process default."""
    var b = trees_builtins()
    var mode = py_get(fields, "numeric_mode")
    if mode is None:
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    host_inference_scalar(mode, context)
    if not Bool(py=py_call(b.isinstance, mode, b.str)):
        raise Error("numeric_mode must be fast, identical or None")
    var clean = String(py_call(py_call(mode.strip).lower))
    if clean != "fast" and clean != "identical":
        raise Error("numeric_mode must be fast, identical or None")
    return clean


def _params(model: PythonObject, fields: PythonObject, n_train: Int, n_features: Int, n_query: Int, want: Int,
            context: PythonObject) raises -> PythonObject:
    """`IsolationForest._params`: the sixteen `iforest_run` slots."""
    if not py_contains(fields, "_seed"):  # an instance assembled without fit or load
        iforest_validate(model, fields, context)
    var b = trees_builtins()
    return Python.list(
        PythonObject(n_train), PythonObject(n_features), PythonObject(n_query), fields["_n_estimators"],
        fields["_max_samples_mode"], fields["_max_samples_int"], fields["_max_samples_frac"], fields["_max_depth"],
        fields["_max_features_mode"], fields["_max_features_int"], fields["_max_features_frac"],
        PythonObject(1 if Bool(py=py_call(b.bool, fields["_bootstrap"])) else 0), fields["_seed"],
        PythonObject(1 if Bool(py=py_call(b.bool, fields["_contamination_auto"])) else 0), fields["_contamination"],
        PythonObject(want),
    )


def _release(model: PythonObject, fields: PythonObject) raises:
    """`_release_resident()`: drop the resident forest's record (its
    lifetime releases it in the binding). Fields an older build stored in
    the instance dictionary are dropped too."""
    trees_side_drop(_TOKEN_SLOT, model)
    if py_contains(fields, "_if_token"):
        _ = py_call(fields.pop, "_if_token", PythonObject(0))
    if py_contains(fields, "_if_lifetime"):
        _ = py_call(fields.pop, "_if_lifetime", PythonObject(None))


def _resident_table(svm: PythonObject) raises -> Bool:
    """True when the binding `svm` keeps the fitted forest resident
    between calls (`iforest_resident`)."""
    var probe = trees_entry(svm, "iforest_resident")
    return probe is not None and Int(py=py_call_typed(probe)) == 1


def _run(model: PythonObject, fields: PythonObject, x: AdmittedX, train: PythonObject, want: Int, fresh: Bool,
         wide: Bool, context: PythonObject) raises -> PythonObject:
    """`IsolationForest._run` on a converted query: one fit-and-score
    `iforest_run`, the resident token kept or renewed, `offset_` and
    `max_samples_` read back. Returns the float32 scores or int32 labels;
    with `wide` (a public caller's score_samples / decision_function) the
    float64 scores when the binary writes them (`iforest_scores_f64`),
    else the float32 scores the public door widens."""
    var n_query = x.rows
    var svm = trees_table(context, _mode(fields, context), "_mojolearn_svm")
    var resident = _resident_table(svm)
    # Lane restore-iforest-kde (history audit row 4, IDN_IF_SCORES_F64): the
    # device epilogue writes the public float64 scores into this buffer, so
    # the public door's cast pass and its float32 buffer are gone for this
    # binary. Bits unchanged (the float32 values widened exactly).
    var wide_out = False
    if wide and resident and want != _WANT_PREDICT:
        var f64_probe = trees_entry(svm, "iforest_scores_f64")
        wide_out = f64_probe is not None and Int(py=py_call_typed(f64_probe)) == 1
    # Lane perf-iforest (2026-11-13): the score vector is uninitialised
    # storage (admitted_family_door.admitted_array): `iforest_run` writes all
    # n_query words for want 0 and 1, so the zero fill (native_array_zeros)
    # was one more pass per scoring call. Record measurement of candidate
    # 4c2340386abc against 0.8.55, taxi score_samples (ms): NVIDIA L40S
    # 3.27 -> 5.32 (test) and 5.61 -> 8.63 (large), AMD MI325X 2.91 -> 3.56.
    # Bits unchanged (the kernel's words, read as before).
    var values = PythonObject(None) if want == _WANT_PREDICT else family_uninit_array(
        Python.tuple(n_query), "<f8" if wide_out else "<f4", context)
    var labels = admitted_vector(n_query, "<i4", context) if want == _WANT_PREDICT else PythonObject(None)
    var info = admitted_vector(4 if resident else 3, "<f8", context)
    var params = _params(model, fields, Int(py=train.shape[0]), Int(py=train.shape[1]), n_query,
                         want + _WANT_WIDE_BASE if wide_out else want, context)
    # The held record: (token, the training matrix it was minted for, its
    # lifetime). A record for another training matrix is not this forest's.
    var held = trees_side_get(_TOKEN_SLOT, model) if resident else PythonObject(None)
    # Hazard C: the token is minted by one binding (one numeric mode's svm
    # table). After a mode switch the other binding must never receive it,
    # so the record also holds the table that minted it, and a record from
    # another table is not this forest's.
    if held is not None and (len(held) < 4 or held[3] is not svm):
        held = PythonObject(None)
    var held_token = Int(py=held[0]) if held is not None else 0
    var same_train = held is not None and held[1] is train
    if resident:
        _ = py_call(params.append, PythonObject(0 if fresh or not same_train else held_token))
    var run = py_get(svm, "iforest_run")
    if run is None:
        raise Error("ImportError: mojolearn: the svm binding has no iforest_run")
    _ = trees_call(run, Python.list(PythonObject(trees_address(train)), PythonObject(x.address),
                                    PythonObject(0 if values is None else trees_address(values)),
                                    PythonObject(0 if labels is None else trees_address(labels)),
                                    PythonObject(trees_address(info)), params))
    _ = values  # keep the Python object alive through the native call above
    _ = labels  # keep the Python object alive through the native call above
    _ = info  # keep the Python object alive through the native call above
    var words = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=trees_address(info))
    _ = info  # keep the Python object alive through the native call above
    # `array.array("f", [float(info[0])])[0]`: the float32 offset as a float.
    py_setitem(fields, "offset_", PythonObject(Float64(Float32(words[0]))))
    _ = info  # keep the Python object alive through the native call above
    py_setitem(fields, "max_samples_", PythonObject(Int(words[1])))
    _ = info  # keep the Python object alive through the native call above
    if resident:
        var token = Int(words[3])
        _ = info  # keep the Python object alive through the native call above
        if token != held_token or not same_train:
            # The same token keeps its lifetime (no release of the live
            # forest); a new token gets its own, and replacing the record
            # releases the previous one.
            var lifetime = PythonObject(None)
            if token == held_token and held is not None:
                lifetime = held[2]
            else:
                var release = py_get(svm, "iforest_resident_release")
                if release is not None:
                    lifetime = trees_resident_new(release, token)
            # A model without weak references keeps no record: the record
            # (and its lifetime) is dropped here, releasing the token.
            _ = trees_side_put(_TOKEN_SLOT, model, Python.list(PythonObject(token), train, lifetime, svm))
    _ = x.array
    return labels if want == _WANT_PREDICT else values


def _as_array(x: AdmittedX, context: PythonObject) raises -> PythonObject:
    """The float32 C-order X as a mojolearn Array (what `as_f32_c`
    returned): the converted copy, the caller's Array, or a zero-copy Array
    view of the caller's buffer (`Array.from_buffer`)."""
    var b = trees_builtins()
    if py_type(x.array) is context["array_class"]:
        return x.array
    var view = x.array
    var array = native_storage_allocate(context["array_class"])
    py_setattr(array, "_store", view.obj)
    py_setattr(array, "_base", PythonObject(None))
    py_setattr(array, "_pin", view)
    py_setattr(array, "_mv", py_call(py_call(view.cast, "B").cast, "f"))
    py_setattr(array, "_addr", PythonObject(x.address))
    py_setattr(array, "_readonly", view.readonly)
    py_setattr(array, "shape", Python.tuple(x.rows, x.cols))
    py_setattr(array, "dtype", PythonObject("<f4"))
    py_setattr(array, "ndim", PythonObject(2))
    py_setattr(array, "size", PythonObject(x.rows * x.cols))
    py_setattr(array, "nbytes", PythonObject(x.rows * x.cols * 4))
    py_setattr(array, "itemsize", PythonObject(4))
    py_setattr(array, "strides", Python.tuple(x.cols * 4, 4))
    py_setattr(array, "order", PythonObject("C"))
    return array


def _device_scan(fields: PythonObject, context: PythonObject) raises -> Bool:
    """A binary that scans the uploaded X on the device
    (`iforest_device_scan` or `iforest_device_finite_scan`)."""
    var svm = trees_table(context, _mode(fields, context), "_mojolearn_svm")
    var names = Python.tuple("iforest_device_scan", "iforest_device_finite_scan")
    for i in range(2):  # driver-loop: over fixed call structure, small-loop(names): the two device scan probes
        var probe = trees_entry(svm, String(names[i]))
        if probe is not None and Int(py=py_call_typed(probe)) == 1:
            return True
    return False


def iforest_device_scan(model: PythonObject, context: PythonObject) raises -> Bool:
    """True when the model's binary tests every X it uploads on the device
    (the fit's training matrix and the query rows of score_samples /
    decision_function / predict: `_raise_first_nonfinite_device`), so a
    door may defer its finite test of a borrowed X to it (lane a7c). The
    CPU column's host query tests nothing: False there."""
    return _device_scan(py_fields(model), context)


def _fit(model: PythonObject, X: PythonObject, want: Int, context: PythonObject,
         copy_source: PythonObject) raises -> PythonObject:
    """`fit` (want < 0: the one-row fresh run) or `fit_predict` (one fresh
    run that predicts X)."""
    var fields = py_fields(model)
    iforest_validate(model, fields, context)
    # `input_copied_`: False when the conversion would borrow X in place (a
    # float32 C-order 2-D input). The admission borrows a float32 C-order
    # view, which the fit copies below, and casts anything else once into
    # storage the fit owns (the training matrix is the fit's own).
    var view = classical_view(X, context, "X")
    var copied = not ((view is X or view is copy_source) and Int(py=view.ndim) == 2
                      and classical_is_f32(view, context) and classical_c_order(view, context))
    py_setitem(fields, "input_copied_", PythonObject(copied))
    # Lane perf-iforest (2026-11-13): as_array=False. With as_array=True an
    # ndarray input (the board's float32 C-order X) was never borrowed: the
    # admission allocated a ZERO-FILLED Array store (native_array_zeros, one
    # extra serial memory pass over the destination) and then copied X into
    # it, so the snapshot cost a zero fill plus a copy instead of the one
    # parallel copy into uninitialised storage described below. Record
    # measurement of candidate 4c2340386abc against 0.8.55 (ms): istella fit
    # NVIDIA L40S 197 -> 1318, AMD MI325X 180 -> 888; taxi fit NVIDIA
    # 149 -> 320, AMD 189 -> 267 (measure-candidate/{nv,amd}-p1-4c2340386abc).
    # Now a float32 C-order view is borrowed (and copied once only on a
    # refit-per-call binary, the branch below); any other input is cast once into uninitialised storage this
    # fit owns, which is kept as it is (no second copy). The same float32
    # bytes either way: no bit change.
    var x = admit_f32(view, 2, "X", context, finite=False, admitted=True)
    _ = view^
    var device_scan = _device_scan(fields, context)
    # The public door's admission (admit_x) already refused NaN and
    # infinity in this X (trees_finite_known, as the forest fit): no second
    # scan of it (lane A1 admit-trees-labels).
    if not device_scan and not trees_finite_known(X) and not host_all_finite_f32(
            MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=x.address), x.rows * x.cols):
        raise Error(_NONFINITE)
    _ = x  # keep the Python object alive through the native calls above
    # Caller-aliased input (hazard A, p3-trees): a float32 C-order X arrives
    # as the caller's own buffer. The resident forest token is keyed on the
    # training matrix object, so keeping the caller's buffer as `_x` let an
    # in-place edit of X after fit diverge the device forest from `_x`,
    # save() and pickle. On a refit-per-call binary (or under
    # MOJOLEARN_IDN_IF_FIT_OWN_X) the fit owns its training matrix: one native
    # copy of the same float32 bytes (the forest and the scores keep their
    # bits); a resident binary borrows it (restore-iforest-kde, below).
    # `input_copied_` still reports whether the conversion copied.
    # The copy stays: every later score_samples / decision_function /
    # predict reads the training matrix again (`_score` -> `_run(...,
    # fields["_x"])` -> `iforest_run`, and the resident forest token is
    # keyed on `_x`), so X is used after fit returns. 0.8.66 regression C3
    # (iforest fit 39.6x on istella, with C1): the copy is now one parallel
    # copy across host cores instead of one serial memcpy.
    # Lane A1 admit-trees-labels (REGRESSIONS C3: the zero fill of the
    # 1.8 GB destination before the copy overwrote it): the copy lands in
    # one uninitialised byte store, which `_as_array` wraps as the fitted
    # Array (the shape `Array.from_buffer` gives a borrowed buffer). The
    # same float32 bytes: no bit change.
    # TOMBSTONE: iforest fit zero-filled trees_empty copy destination deleted 2026-11-08 by lane a1-admit-trees-labels; code recoverable at f7d10d2b09.
    # Restore: experiments/removed/A1_ADMIT_TREES_LABELS.patch.
    # Lane restore-iforest-kde (2026-11-13, history audit row 3): on a
    # binary that keeps the forest resident (`iforest_resident` 1) the fit
    # borrows a float32 C-order X again, as 0.8.55 did
    # (`git show 0e4de0f46e:python/mojolearn/_iforest_impl.py` 463-506:
    # `self._x = x` from as_f32_c, no copy). The resident forest is built
    # from X inside this fit and every later score_samples / decision_function
    # / predict scores that forest with no refit, so the caller's buffer is
    # read again only to refit after a token miss (a pickle or copy, a mode
    # switch) and by save(). `_x` holds the caller's buffer (its `_store` and
    # `_pin`), which keeps it alive for as long as the estimator holds `_x`.
    # Record measurement of candidate 4c2340386abc (ms), fit: istella NVIDIA
    # L40S 197 (0.8.55) -> 8,422 (0.8.66) -> 1,318, AMD MI325X 180 -> 888;
    # taxi NVIDIA 149 -> 320, AMD 189 -> 267. The copy (0292b73ab3, hazard A)
    # stays: on a refit-per-call binary (`iforest_resident` 0, the CPU column
    # with IDN_IF_RESIDENT off), whose every score refits from `_x`, and
    # behind `-D MOJOLEARN_IDN_IF_FIT_OWN_X` as the B arm of the owed A/B.
    # Same float32 bytes either way: no bit change. What the borrow gives up:
    # a caller editing X in place after fit changes what save() and pickle
    # write (as at 0.8.55); the scores of this estimator do not change.
    var keep_borrowed = False
    comptime if not _IF_FIT_OWN_X:
        keep_borrowed = x.borrowed and _resident_table(trees_table(context, _mode(fields, context), "_mojolearn_svm"))
    if keep_borrowed:
        # The caller's float32 C-order buffer itself: a mojolearn Array is
        # kept as it is (`_as_array`); an ndarray is kept as a one-dimensional
        # byte view of its buffer, whose `obj` is the caller's ndarray, so
        # `_as_array` wraps it as the zero-copy Array 0.8.55's
        # `Array.from_buffer` made.
        if py_type(x.array) is not context["array_class"]:
            var caller_view = py_call(native_buffer_view(x.array).cast, "B")
            x = admitted_words(caller_view, x.address, x.rows, x.cols)
    elif x.borrowed:
        var count = x.rows * x.cols
        var owned = native_scratch_bytes(count * 4)
        var dst = native_scratch_address(owned)
        if count > 0:
            host_parallel_copy_bytes(dst, x.address, count * 4)
        _ = x  # keep the caller's buffer alive through the copy above
        x = admitted_words(native_buffer_view(owned), dst, x.rows, x.cols)
    elif py_type(x.array) is not context["array_class"]:
        # The cast copy admit_f32 made into uninitialised storage (an ndarray
        # with NumPy): this fit owns it, so it is the training matrix as it
        # is. `_as_array` wraps its one-dimensional byte view as the fitted
        # Array (`_store` is the ndarray, which keeps the bytes alive).
        var owned_view = py_call(native_buffer_view(x.array).cast, "B")
        x = admitted_words(owned_view, x.address, x.rows, x.cols)
    var had = Python.dict()
    # `_if_forest` (a loaded or unpickled estimator's saved forest) is
    # replaced by this fit's training matrix, and restored on a refusal.
    var kept = Python.tuple("_x", "n_features_in_", _FOREST_FIELD)
    for i in range(3):  # driver-loop: over fixed call structure, small-loop(kept): the three fitted field names restored on a refusal
        if py_contains(fields, kept[i]):
            py_setitem(had, kept[i], fields[kept[i]])
    _release(model, fields)
    _ = py_call(fields.pop, _FOREST_FIELD, PythonObject(None))
    var train = _as_array(x, context)
    py_setitem(fields, "_x", train)
    py_setitem(fields, "n_features_in_", PythonObject(x.cols))
    # `fit` scores the training matrix's first row (the Python `x[:1]`
    # copy: the same bytes, read in place); `fit_predict` predicts X.
    var query = admitted_words(train, x.address, 1 if want < 0 else x.rows, x.cols)
    _ = x  # keep the Python object alive through the native call above
    var out = PythonObject(None)
    try:
        out = _run(model, fields, query, train, _WANT_SCORE_SAMPLES if want < 0 else want, True, False, context)
        _ = x  # keep the Python object alive through the native call above
    except error:
        var message = String(error)
        if device_scan and ("Input X contains" in message or "does not accept non-finite" in message):
            for i in range(3):  # driver-loop: over fixed call structure, small-loop(kept): the three fitted field names restored on a refusal
                _ = py_call(fields.pop, kept[i], PythonObject(None))
            _ = py_call(fields.update, had)
            raise Error(_NONFINITE)
        raise error^
    return model if want < 0 else out


def _score(model: PythonObject, X: PythonObject, want: Int, wide: Bool,
           context: PythonObject) raises -> PythonObject:
    """`score_samples` / `decision_function` / `predict`: `_run(X, want)`."""
    var fields = py_fields(model)
    var trained = py_contains(fields, "_x")
    if not trained and not py_contains(fields, _FOREST_FIELD):
        raise Error(_PREFIX + "call fit() first")
    var x = admit_f32(X, 2, "X", context, finite=False)
    var fitted = Int(py=py_call(trees_builtins().int, fields["n_features_in_"]))
    if x.cols != fitted:
        raise Error(_PREFIX + "X has " + String(x.cols) + " features, fit saw " + String(fitted))
    if not trained:
        return _run_forest(model, fields, x, want, wide, context)
    return _run(model, fields, x, fields["_x"], want, False, wide, context)


def _forest_words(forest: PythonObject, context: PythonObject) raises -> PythonObject:
    """UNPACK: `(words, views)` of a saved forest tuple: the seven words
    `iforest_forest_run` takes (five addresses, trees, nodes) after the
    dtype, layout and length checks, and the views that keep the five
    buffers alive through the native call. The node-level checks run in
    the binding (`if_forest_check`)."""
    var b = trees_builtins()
    if py_type(forest) is not b.tuple or len(forest) != 5:
        raise Error(_PREFIX + "the saved forest must be a tuple of five arrays")
    var names = Python.tuple("offsets", "feature", "threshold", "left", "right")
    var types = Python.tuple("<i4", "<i4", "<f4", "<i4", "<i4")
    var words = Python.list()
    var views = Python.list()
    var sizes = List[Int]()
    for i in range(5):  # driver-loop: over fixed call structure, small-loop(names): the five forest arrays
        # Admitted here (an unpickled or loaded state is caller data).
        var view = classical_view(forest[i], context, String(names[i]))
        if (Int(py=view.ndim) != 1 or admitted_dtype(view, context) != String(types[i])
                or not classical_c_order(view, context)):
            raise Error(_PREFIX + "the saved forest's " + String(names[i]) + " must be a flat " + String(types[i]) + " array")
        _ = py_call(words.append, classical_address(view))
        _ = py_call(views.append, view)
        sizes.append(Int(py=view.size))
    var n_trees = sizes[0] - 1
    var total = sizes[1]
    if n_trees < 1 or sizes[2] != total or sizes[3] != total or sizes[4] != total:
        raise Error(_PREFIX + "the saved forest's arrays do not have matching lengths")
    _ = py_call(words.append, PythonObject(n_trees))
    _ = py_call(words.append, PythonObject(total))
    return Python.tuple(words, views)


def _keep_token(model: PythonObject, svm: PythonObject, held: PythonObject, held_token: Int, anchor: PythonObject,
                same: Bool, token: Int) raises:
    """`_run`'s record rule for the token a call returned: the same token
    on the same anchor keeps its lifetime; a new one gets its own, and
    replacing the record releases the previous one."""
    if token == 0 or (token == held_token and same):
        return
    var lifetime = PythonObject(None)
    if token == held_token and held is not None:
        lifetime = held[2]
    else:
        var release = py_get(svm, "iforest_resident_release")
        if release is not None:
            lifetime = trees_resident_new(release, token)
    _ = trees_side_put(_TOKEN_SLOT, model, Python.list(PythonObject(token), anchor, lifetime, svm))


def _held(model: PythonObject, svm: PythonObject) raises -> PythonObject:
    """The resident record of this estimator for the binding `svm`, or None
    (hazard C: a record from another table is not this forest's)."""
    var held = trees_side_get(_TOKEN_SLOT, model)
    if held is not None and (len(held) < 4 or held[3] is not svm):
        return PythonObject(None)
    return held


def _run_forest(model: PythonObject, fields: PythonObject, x: AdmittedX, want: Int, wide: Bool,
                context: PythonObject) raises -> PythonObject:
    """`_run` for an estimator that holds its saved forest (`_if_forest`)
    and no training matrix: `iforest_forest_run` scores the forest (uploaded
    once, then kept under a token like a fitted one); no fit, and `offset_`
    / `max_samples_` are the saved ones."""
    var n_query = x.rows
    var svm = trees_table(context, _mode(fields, context), "_mojolearn_svm")
    var run = trees_entry(svm, "iforest_forest_run")
    if run is None:
        raise Error("ImportError: mojolearn: the svm binding has no iforest_forest_run")
    var forest = fields[_FOREST_FIELD]
    var unpacked = _forest_words(forest, context)
    var wide_out = False
    if wide and want != _WANT_PREDICT:
        var f64_probe = trees_entry(svm, "iforest_scores_f64")
        wide_out = f64_probe is not None and Int(py=py_call_typed(f64_probe)) == 1
    var values = PythonObject(None) if want == _WANT_PREDICT else family_uninit_array(
        Python.tuple(n_query), "<f8" if wide_out else "<f4", context)
    var labels = admitted_vector(n_query, "<i4", context) if want == _WANT_PREDICT else PythonObject(None)
    var info = admitted_vector(4, "<f8", context)
    var held = _held(model, svm)
    var held_token = Int(py=held[0]) if held is not None else 0
    var same = held is not None and held[1] is forest
    var b = trees_builtins()
    var params = Python.list(
        PythonObject(x.cols), PythonObject(n_query), py_call_typed(b.int, fields["max_samples_"]),
        py_call_typed(b.float, fields["offset_"]), PythonObject(want + _WANT_WIDE_BASE if wide_out else want),
        PythonObject(held_token if same else 0),
    )
    _ = trees_call(run, Python.list(unpacked[0], PythonObject(x.address),
                                    PythonObject(0 if values is None else trees_address(values)),
                                    PythonObject(0 if labels is None else trees_address(labels)),
                                    PythonObject(trees_address(info)), params))
    _ = unpacked  # keep the forest views alive through the native call above
    _ = values  # keep the Python object alive through the native call above
    _ = labels  # keep the Python object alive through the native call above
    var words = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=trees_address(info))
    var token = Int(words[3])
    _ = info  # keep the Python object alive through the native read above
    _keep_token(model, svm, held, held_token, forest, same, token)
    _ = x.array
    return labels if want == _WANT_PREDICT else values


def iforest_export_forest(model: PythonObject, context: PythonObject) raises -> PythonObject:
    """The fitted forest as a tuple of five Arrays (`_FOREST_FIELD`'s form),
    for `save()` and pickle. A loaded or unpickled estimator returns the
    forest it holds. A fitted one reads the resident forest its scores use
    from the binding (`iforest_forest_export`: the device's node arrays,
    or the CPU column's oracle trees); a binding with no live forest for
    this estimator fits one from `_x` first and keeps it. Raises when the
    estimator holds neither."""
    var fields = py_fields(model)
    if not py_contains(fields, "_x"):
        if py_contains(fields, _FOREST_FIELD):
            _ = _forest_words(fields[_FOREST_FIELD], context)  # refuse a malformed held forest here
            return fields[_FOREST_FIELD]
        raise Error("RuntimeError: " + _PREFIX + "this estimator is not fitted yet")
    var svm = trees_table(context, _mode(fields, context), "_mojolearn_svm")
    var export = trees_entry(svm, "iforest_forest_export")
    if export is None:
        raise Error("ImportError: mojolearn: the svm binding has no iforest_forest_export")
    var train = fields["_x"]
    var params = _params(model, fields, Int(py=train.shape[0]), Int(py=train.shape[1]), 1, _WANT_SCORE_SAMPLES,
                         context)
    var resident = _resident_table(svm)
    var held = _held(model, svm) if resident else PythonObject(None)
    var held_token = Int(py=held[0]) if held is not None else 0
    var same = held is not None and held[1] is train
    _ = py_call(params.append, PythonObject(held_token if same else 0))
    var info = admitted_vector(4, "<f8", context)
    var address = PythonObject(trees_address(train))
    # The sizes first (the binding keeps a fresh fit under its token), then
    # the forest into storage of exactly that size.
    var sized = trees_call(export, Python.list(address, PythonObject(trees_address(info)), params, Python.list()))
    var token = Int(py=sized[0])
    var n_trees = Int(py=sized[1])
    var total = Int(py=sized[2])
    py_setitem(params, PythonObject(16), PythonObject(token))
    var offsets = family_uninit_array(Python.tuple(n_trees + 1), "<i4", context)
    var feature = family_uninit_array(Python.tuple(total), "<i4", context)
    var threshold = family_uninit_array(Python.tuple(total), "<f4", context)
    var left = family_uninit_array(Python.tuple(total), "<i4", context)
    var right = family_uninit_array(Python.tuple(total), "<i4", context)
    var words = Python.list(
        PythonObject(trees_address(offsets)), PythonObject(trees_address(feature)),
        PythonObject(trees_address(threshold)), PythonObject(trees_address(left)),
        PythonObject(trees_address(right)), PythonObject(n_trees), PythonObject(total),
    )
    var written = trees_call(export, Python.list(address, PythonObject(trees_address(info)), params, words))
    _ = train  # keep the training matrix alive through the native calls above
    _ = info  # keep the Python object alive through the native calls above
    if resident:
        _keep_token(model, svm, held, held_token, train, same, Int(py=written[0]))
    return Python.tuple(offsets, feature, threshold, left, right)


def iforest_getstate(model: PythonObject, context: PythonObject) raises -> PythonObject:
    """`IsolationForest.__getstate__`: the instance dictionary without the
    process-local resident handles, and, for a fitted estimator, the fitted
    forest (`iforest_export_forest`) in place of the training matrix, so a
    pickle or copy scores the forest this estimator scores even after the
    caller edits the fitted X in place."""
    var fields = py_fields(model)
    var state = py_call(trees_builtins().dict, fields)
    _ = py_call(state.pop, "_if_token", PythonObject(None))
    _ = py_call(state.pop, "_if_lifetime", PythonObject(None))
    if py_contains(fields, "_x"):
        py_setitem(state, _FOREST_FIELD, iforest_export_forest(model, context))
        _ = py_call(state.pop, "_x", PythonObject(None))
    _ = fields
    return state


def iforest_operation_with_source(model: PythonObject, operation: String, args: PythonObject,
                                  context: PythonObject, copy_source: PythonObject,
                                  wide: Bool) raises -> PythonObject:
    """One IsolationForest operation. `wide`: the caller returns float64
    scores (the public door for a caller outside the package), so a
    binary that writes them directly is asked to."""
    if operation == "trees_iforest_fit":
        return _fit(model, args[0], -1, context, copy_source)
    if operation == "trees_iforest_fit_predict":
        return _fit(model, args[0], _WANT_PREDICT, context, copy_source)
    if operation == "trees_iforest_score_samples":
        return _score(model, args[0], _WANT_SCORE_SAMPLES, wide, context)
    if operation == "trees_iforest_decision_function":
        return _score(model, args[0], _WANT_DECISION_FUNCTION, wide, context)
    if operation == "trees_iforest_predict":
        return _score(model, args[0], _WANT_PREDICT, False, context)
    raise Error("unknown IsolationForest operation " + operation)


def iforest_operation(model: PythonObject, operation: String, args: PythonObject,
                      context: PythonObject) raises -> PythonObject:
    return iforest_operation_with_source(model, operation, args, context, args[0], False)
