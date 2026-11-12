# SPDX-License-Identifier: Apache-2.0
# The whole public lifecycle of the family estimators that formerly took the
# `_mode` class-creation wrappers (lane lifecycle-neighbors-cluster-20261028):
# OneClassSVM, KernelPCA, AdditiveChi2Sampler, SkewedChi2Sampler,
# LabelPropagation, LabelSpreading, PageRank, Louvain, SVGP, KMeans, LinearSVC
# and LinearSVR. One native public call runs, in order: model and fitted-state
# admission, the NotFittedError refusal, X validation (rank, emptiness, the
# finite scan scikit-learn's validate_data makes for these estimators), the
# width and feature-name checks, y validation at an outermost fit, fit depth
# bookkeeping, the algorithm owner, the fitted flag, feature_names_in_ and
# public class labels after a fit, and NumPy results (float64 where
# scikit-learn returns float64) for callers outside the package. The numerical
# owners are unchanged; this provider imports none of them.
from bindings.admitted_finite_wording import admitted_finite_wording
from std.math import isfinite
from std.python import Python, PythonObject
from bindings.admitted_family_door import admit_family_meta, admit_family_owned
from providers.python.array_metadata_pyapi_door import ArrayPin
from bindings.native_loaded_module_door import native_loaded_module
from bindings.admitted_call_door import AdmittedCall, admitted_input_dense, admitted_input_shape, ADMITTED_ITEM_F32, ADMITTED_SPARSE_OTHER, ADMITTED_GRAPH_TUPLE
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.mode_metadata_native_door import _fields, _fitted
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.classical_validate_unpack import classical_estimator_name
from bindings.classical_validate_run import classical_check_width
from bindings.classical_exceptions_door import classical_not_fitted_class
from bindings.host_inference_scalars_door import host_inference_scalar
from bindings.host_sparse_native_door import host_sparse_format, _sparse_fields
from bindings.family_common_unpack import FAMILY_F32P
from core.host_finite_scan import host_all_finite_f32
from bindings.cluster_side_records_unpack import cluster_refuse_fit_callbacks
from providers.python.estimator_lifecycle_pyapi_unpack import (
    estimator_outside, estimator_feature_check, estimator_fit_begin, estimator_fit_complete,
)
from providers.python.neighbors_public_lifecycle_pyapi_door import NeighborPublicCall, _target, _public_result
from bindings.device_rows_door import door_is_device_rows
from bindings.admitted_input import admit_x
from providers.python.family_public_lifecycle_run import FamilyLifecycle


def _precomputed(fields: PythonObject, b: PythonObject) raises -> Bool:
    # scikit-learn skips the width check for a precomputed kernel or metric:
    # the query's columns are the training rows. `b`: the operation's
    # admitted builtins connection.
    for key in ["kernel", "metric", "affinity"]:  # small-loop(three fixed precomputed-mode parameters)
        var value = py_get(fields, key)
        if py_type(value) is b.str and String(py=value) == "precomputed":
            return True
    return False


def family_lifecycle_begin(model: PythonObject, operation: String, spec: FamilyLifecycle,
                           args: PythonObject, x_index: Int, context: PythonObject) raises -> NeighborPublicCall:
    """Admission and validation before the owner runs; the returned call
    carries the owner's argument list (X replaced by its admitted snapshot
    where `spec.replace`, y by its validated form)."""
    # Owners that do not yet hold an admission record.
    var admitted = AdmittedCall()
    return family_lifecycle_begin(admitted, model, operation, spec, args, x_index, context)


def family_lifecycle_begin(mut admitted: AdmittedCall, model: PythonObject, operation: String,
                           spec: FamilyLifecycle, args: PythonObject, x_index: Int,
                           context: PythonObject) raises -> NeighborPublicCall:
    """As above, filling the owner's admission record: its modules, the model
    fields and X (shape, and for a dense X the admitted float32 snapshot's
    address and read-only flag at position `x_index`), so the owner reads
    them from `admitted` instead of admitting X again."""
    classical_connect_numpy(context)
    var b = admitted.builtins()
    if py_type(py_type(model)) is not b.type:
        raise Error("TypeError: native estimator operations refuse Python metaclass hooks")
    var fields = _fields(admitted.modules, model)
    admitted.admit_fields(fields)
    var method = spec.method
    var fitting = method == "fit" or method == "fit_predict" or method == "fit_transform"
    var depth = py_get(fields, "_mojolearn_fitting", PythonObject(0))
    if py_type(depth) is not b.int:
        raise Error("TypeError: native fit depth must be an exact integer")
    var fitted = py_get(fields, "_mojolearn_fitted", PythonObject(False))
    if py_type(fitted) is not b.bool and py_type(fitted) is not b.int:
        raise Error("TypeError: native fitted flags cannot invoke Python boolean hooks")
    # Perf (lane perf2-cluster): the ancestry walk runs once per class state
    # (bindings/cluster_side_records.mojo), not on every public call.
    cluster_refuse_fit_callbacks(py_type(model), "TypeError: native fitting cannot invoke Python completion callbacks")
    if py_contains(fields, "_mojolearn_after_fit"):
        raise Error("TypeError: native fitting cannot invoke Python completion callbacks")
    var name = classical_estimator_name(model)
    if spec.requires_fit and not fitting and not Bool(py=_fitted(model)):
        _ = classical_not_fitted_class(context)
        raise Error("NotFittedError: This " + name + " instance is not fitted yet. Call 'fit' with "
                    + "appropriate arguments before using " + method + ".")
    var source = args[x_index]
    var values = Python.list()
    for i in range(len(args)):  # small-loop(fixed operation argument envelope)
        _ = py_call(values.append, args[i])
    var rows = -1
    var columns = -1
    if door_is_device_rows(source, context):
        # A model-selection fold's X (`_arena_io.DeviceRows`) goes to the one
        # admission path unchanged (lane a2-admit-family; admit_x resolves
        # it, records it on this call and refuses NaN or infinity with the
        # same "(<name>)" wording); this door no longer downloads the fold or
        # scans it a second time. A fit keeps an owned copy, as below.
        # A door whose spec does not scan accepts NaN: the fold is not
        # refused either (lane a6; admit_x's no-check mode).
        var x = admit_x(admitted, source, name, context, x_index, finite=spec.scan, device_x=spec.device_x)
        rows = x.rows
        columns = x.cols
        if not x.store.active():
            source = x.array
            py_setitem(values, PythonObject(x_index), admit_family_owned(x.array, context) if spec.replace and fitting else x.array)
        # else (lane a7d, `spec.device_x`): the fold stays resident; the
        # owner receives the DeviceRows and its device entry gathers the
        # rows (finite and range words returned with its result).
    elif host_sparse_format(source).byte_length() > 0:
        # Sparse X keeps its own owner path (densified there, or a sparse
        # result); only its declared shape is read here.
        var shape = _sparse_fields(source)["_shape"]
        if py_type(shape) is not b.tuple or len(shape) != 2 or py_type(shape[0]) is not b.int or py_type(shape[1]) is not b.int:
            raise Error("TypeError: sparse input shape requires native dimensions")
        rows = Int(py=shape[0])
        columns = Int(py=shape[1])
        _ = admitted.add_input(admitted_input_shape(x_index, source, source, shape, ADMITTED_SPARSE_OTHER))
    elif not (spec.graph and py_type(source) is b.tuple):
        # The finiteness test rides in the admission (the cast's own verdict,
        # or one parallel scan of a borrowed X): no second pass over X.
        var pin = ArrayPin()
        var snapshot = admit_family_meta(source, 2, "X", context, pin, True, finite=spec.scan,
                                         refusal=admitted_finite_wording(name, "X"),
                                         defer_finite=spec.defer)
        rows = snapshot.shape[0]
        columns = snapshot.shape[1]
        # The snapshot's buffer and owner, recorded before any refusal below
        # so an owner never re-admits the X this door admitted.
        _ = admitted.add_input(admitted_input_dense(x_index, source, pin.keep, Int(snapshot.source.addr),
                                                    rows, columns, ADMITTED_ITEM_F32, snapshot.readonly))
        if rows == 0:
            raise Error("Found array with 0 sample(s) (shape=(0, " + String(columns) + ")) while a minimum of 1 is required by " + name + ".")
        if columns == 0 and not spec.graph:
            raise Error("Found array with 0 feature(s) (shape=(" + String(rows) + ", 0)) while a minimum of 1 is required by " + name + ".")
        if spec.replace:
            # Perf: a query only reads X (its owners write separate outputs),
            # so an admitted C-order float32 X that is not a readonly view is
            # handed over without a per-call copy; a fit retains X and keeps
            # the owned copy.
            var borrow = not fitting and not snapshot.readonly and snapshot.f32_c()
            py_setitem(values, PythonObject(x_index), pin.keep if borrow else admit_family_owned(snapshot, pin, context))
    if not fitting and Int(py=depth) == 0 and columns >= 0 and not spec.graph and not _precomputed(fields, b):
        var width = py_get(fields, "n_features_in_")
        if py_type(width) is b.int:
            host_inference_scalar(width, context)
            classical_check_width(columns, Int(py=width), name)
            # scikit-learn's feature-name check runs after a fit only: a
            # stateless transform before any fit warns about nothing.
            estimator_feature_check(model, fields, source, context)
    if fitting and Int(py=depth) == 0 and spec.target >= 0:
        if spec.target < len(args):
            var target = _target(args[spec.target], rows, name, spec.classifier, spec.required, False, context)
            py_setitem(values, PythonObject(spec.target), target)
        elif spec.required:
            raise Error(name + " requires y to be passed, but the target y is None.")
    var previous = estimator_fit_begin(fields) if fitting else Int(py=depth)
    # The owners index an exact argument tuple, as estimator_arguments made it.
    return NeighborPublicCall(py_call(b.tuple, values), fields, source, method, fitting, estimator_outside(),
                              previous, rows, columns)


def family_lifecycle_finish(model: PythonObject, spec: FamilyLifecycle, result: PythonObject,
                            call: NeighborPublicCall, context: PythonObject) raises -> PythonObject:
    """Fit bookkeeping after a completed owner call, then the public result."""
    if call.fitting:
        # A nested set_params may have replaced the dictionary's contents;
        # estimator_fit_complete reads the live dictionary.
        estimator_fit_complete(model, call.source, spec.classifier, context)
        var fields = _fields(model)
        if not py_contains(fields, "n_features_in_") and call.columns >= 0:
            py_setitem(fields, "n_features_in_", PythonObject(call.columns))
    if call.outside and result is not model and call.method != "score":
        # A regressor's predict is float64 alongside NumPy (scikit-learn's
        # dtype); an outlier detector's scores are float64 for every caller.
        var widen = spec.widen and (call.method != "predict" or py_get(context, "ndarray") is not None)
        return _public_result(result, widen, context)
    return result
