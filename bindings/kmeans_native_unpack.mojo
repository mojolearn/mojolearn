# SPDX-License-Identifier: Apache-2.0
# KMeans' public methods, one native call each (python/mojolearn/cluster.py:
# `fit`, `predict`, `transform`, `fit_predict`, `fit_transform`). The shell
# hands over the binding `_bind` resolved for the model's numeric mode; this
# owner reads the parameters, makes the refusals the Python method made (same
# wording, same order), allocates the outputs, executes typed native requests
# through the admitted compiled provider, and stores the fitted attributes.
# Numerical kernels, fused operations and their inputs remain unchanged.
# Lane migrate-classical-b2: `fit_transform` uploads X once
# (`kmeans_fit_transform`), and `predict` / `transform` read the fitted
# centroids from a device copy minted at the first query after a fit
# (`kmeans_centers_prepare`, cluster/resident_centers.mojo) instead of
# uploading them per call; a binding without those doors (the CPU host
# binding) takes the per-call entries as before.
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from bindings.admitted_family_door import (
    admit_family, admit_family_weights_array, admit_family_input, admit_family_cast, admitted_family_x,
)
from providers.python.array_metadata_pyapi_door import ArrayPin, array_python_shape
from bindings.admitted_input_unpack import AdmittedX, admit_x_unrecorded, admitted_finite_refuse
from bindings.device_rows_door import door_is_device_rows, door_rows_refuse
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_fields, py_setitem
from bindings.classical_arrays_door import classical_address
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.py_cpython_table import cpython_table
from bindings.host_inference_scalars_door import host_inference_scalar
from bindings.native_numpy_metadata_unpack import native_numpy_dtype
from bindings.family_common_unpack import (
    FAMILY_F32P,
    family_repr,
    family_str,
)
from bindings.family_common_pack import (
    family_call,
    family_array,
    family_field,
)
from bindings.classical_arrays_door import classical_is_f32, classical_c_order
from core.host_finite_scan import host_all_finite_f32
from bindings.p3nc_fitted_freeze_unpack import fitted_is_freezable, fitted_freeze
from providers.python.family_public_lifecycle_pyapi_door import family_lifecycle_begin, family_lifecycle_finish
from providers.python.family_public_lifecycle_run import family_lifecycle_spec
from providers.python.neighbors_public_lifecycle_pyapi_door import neighbors_public_end, neighbors_public_refusal
from bindings.py_optional import py_or_none, py_present
from core.kmeans_native_api import (
    KMeansFoldStore, KMeansFitRequest, KMeansQueryRequest,
    KMEANS_CAP_JOINT, KMEANS_CAP_RESIDENT, KMEANS_CAP_FOLD, KMEANS_CAP_HOST,
)
from bindings.kmeans_descriptor_boundary_door import (
    kmeans_connection,
    kmeans_run_fit,
    kmeans_run_query,
    kmeans_integer,
)
from x_cluster.kmeans_descriptor_boundary_run import kmeans_capabilities
from bindings.kmeans_owner_boundary_door import (
    kmeans_owner_matching, kmeans_owner_handle, kmeans_owner_prepare,
    kmeans_owner_getstate, kmeans_owner_setstate,
)

comptime _HOLDER = "_resident_holder"


def _kmeans_release(model: PythonObject) raises:
    # Removing the model's reference retires the owner. A query still using
    # it pins the native handle until its own execution has finished.
    py_setitem(py_fields(model), _HOLDER, PythonObject(None))


def _kmeans_centers_ready(centers: PythonObject, context: PythonObject) raises -> Bool:
    """`cluster_centers_` is an exact Array or ndarray, 2-D, non-empty,
    float32 in C order: `as_f32_c` would return its own storage, so its
    words can be read (scanned) in place."""
    var cls = py_type(centers)
    if cls is not context["array_class"] and cls is not py_get(context, "ndarray"):
        return False
    if Int(py=centers.ndim) != 2 or Int(py=centers.size) <= 0:
        return False
    return classical_is_f32(centers, context) and classical_c_order(centers, context)


# TOMBSTONE: _kmeans_nonfinite / _kmeans_f32 (KMeans' own "X contains a NaN or an infinity; ... refused by name" wording and its host scan of X) deleted 2026-11-09 by lane a7d-device-x-decomp-cluster; code recoverable at 5b5490960d.
# Restore: experiments/removed/A7D_KMEANS_HOST_FINITE.patch. The kernel entry
# returns X's finite status word over its own upload; the refusal is the one
# admission wording (admitted_finite_refuse, "Input X contains NaN or infinity (KMeans).").
comptime _KMEANS_EST = "KMeans"


def _kmeans_x(source: PythonObject, context: PythonObject) raises -> AdmittedX:
    """X of a KMeans operation on the one admission path, without a host
    scan of a borrowed X (lane a7d): `finite` False means the kernel entry
    tests the words it uploads and returns the status word (the door then
    refuses through `admitted_finite_refuse`). A cast X is tested in the
    cast pass. A model-selection fold (`_arena_io.DeviceRows`) keeps its
    resident store ids (`store.active()`) for the device-X entries; where
    they do not resolve, its rows are gathered on the host and tested in
    the gather."""
    if door_is_device_rows(source, context):
        return admit_x_unrecorded(source, _KMEANS_EST, context, device_x=True)
    var pin = ArrayPin()
    var snapshot = admit_family_input(source, 2, "X", context, pin)
    var borrowed = snapshot.f32_c()
    admit_family_cast(snapshot, pin, "X", context, not borrowed,
                      "Input X contains NaN or infinity (" + String(_KMEANS_EST) + ").")
    return admitted_family_x(snapshot, pin, source, not borrowed)


def _kmeans_host_x(source: PythonObject, x: AdmittedX, context: PythonObject) raises -> AdmittedX:
    """`x` with host words: a resident fold whose operation has no device-X
    entry (sample weights, the host binding) is gathered on
    the host out of model selection's host base, tested in the gather."""
    if x.store.active():
        return admit_x_unrecorded(source, _KMEANS_EST, context)
    return x.copy()


def _kmeans_is_name(value: PythonObject, context: PythonObject) raises -> Bool:
    # isinstance may consult a rebound object's __class__. Inspect the real
    # type's C MRO instead, admitting only exact builtin/NumPy text storage
    # before dictionary hashing or equality can execute.
    var cls = py_type(value)
    var string_type = py_type(PythonObject(""))
    if cls is string_type:
        return True
    if cpython_table()[].PyType_IsSubtype(cls._obj_ptr, string_type._obj_ptr) == 0:
        return False
    if py_get(context, "ndarray") is not None and cls is py_call(native_numpy_dtype(context), "U").type:
        return True
    raise Error("TypeError: scalar model state requires native values without Python conversion hooks")


def _kmeans_conversion_scalar(value: PythonObject, context: PythonObject) raises:
    # tol/seed were converted before the compiled integer admission. Keep
    # their exact native NumPy text/byte and extended-real conversion forms.
    # Common builtin/numeric fields return through the shared admission.
    try:
        host_inference_scalar(value, context)
    except failure:
        if py_get(context, "ndarray") is not None:
            var cls = py_type(value)
            var constructor = native_numpy_dtype(context)
            var aliases = Python.tuple("U", "S", "q", "Q", "g")
            for i in range(len(aliases)):  # driver-loop: exact native scalar ABI aliases
                if cls is py_call(constructor, aliases[i]).type:
                    return
        raise failure^


def _kmeans_metric(fields: PythonObject, context: PythonObject) raises -> Int:
    """`KMeans._metric_code`: a name from the table, or `int(metric)`."""
    var metric = family_field(fields, "metric")
    if _kmeans_is_name(metric, context):
        var table = context["family_kmeans_metrics"]
        if not py_contains(table, metric):
            raise Error("mojolearn: metric must be one of "
                        + family_repr(py_call(native_loaded_module("builtins").sorted, table))
                        + ", got " + family_repr(metric))
        return Int(py=table[metric])
    _kmeans_conversion_scalar(metric, context)
    return Int(py=family_call(native_loaded_module("builtins").int, metric))


# TOMBSTONE: P1_KMEANS_CALLABLE_INIT (removed: the user's callable init plus numpy.asarray / RandomState called inside KMeans.fit) deleted 2026-11-09 by lane p1-python-callbacks-worst; code recoverable at e68654a757.
# Restore: experiments/removed/P1_KMEANS_CALLABLE_INIT.patch.
# P1 (audit A #2, audit B #3): _kmeans_callable_init removed; it imported
# NumPy, built numpy.random.RandomState and called the user's init(X, k,
# random_state=...) inside fit. A callable init is now refused before it is
# called (as MiniBatchKMeans already does, bindings/cluster_fit_a_native.mojo
# _init_array); the seeds of 'k-means++' and 'random' are drawn natively.


def _kmeans_admitted(source: PythonObject, context: PythonObject) raises -> Bool:
    """`source` is the public lifecycle's admitted dense snapshot: an exact
    Array or ndarray (a sparse X passes through unreplaced and is neither),
    float32 in C order."""
    var cls = py_type(source)
    if cls is not context["array_class"] and cls is not py_get(context, "ndarray"):
        return False
    return classical_is_f32(source, context) and classical_c_order(source, context)
