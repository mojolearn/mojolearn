# SPDX-License-Identifier: Apache-2.0
# UNPACK of KernelDensity's public operations (`family_xn_kde_*`): the state
# admission and its per-model record, the dense float32 admission of X (with
# the conversion's copied flag), the constructor-setting refusals in the
# Python method's order and wording, and the diagnostic text of a name.
# Every refusal that is a comparison on a number or a name is the typed RUN
# of neighbors/neighbors_kde_run.mojo; the numerical work is the typed
# provider call of core/kde_native_api.mojo. PACK and the operators are
# bindings/kde_public_native_pack.mojo. (This file and that one were the
# single door bindings/neighbors_kde_native_door.mojo.)
from providers.python.estimator_binding_pyapi_door import estimator_arguments
from providers.python.neighbors_public_lifecycle_pyapi_door import neighbors_public_begin, neighbors_public_end, neighbors_public_finish, neighbors_public_refusal
from std.python import PythonObject
from bindings.admitted_family_door import admit_family, admit_family_weights
from providers.python.array_metadata_pyapi_door import ArrayPin
from bindings.native_loaded_module_door import native_loaded_module
from bindings.kde_admission_boundary_unpack import kde_admission_hit, kde_admission_remember, kde_admission_drop
from bindings.kde_native_boundary_unpack import KDEConnection, KDEResidentOwner, kde_owner_admit, kde_run_query, kde_run_resident
from bindings.kde_native_boundary_pack import kde_owner_prepare, kde_array
from bindings.public_lifecycle_owner_door import lifecycle_admit_fields
from bindings.decomp_scalar_compare_unpack import decomp_equal_admitted, decomp_name_admitted
from bindings.decomp_scalar_admission_unpack import decomp_scalar_repr
from bindings.decomp_truth_boundary_unpack import decomp_truth
from neighbors.neighbors_kde_run import (
    KDE_DEFAULT_LEAF_SIZE, kde_refuse_bandwidth_name, kde_bandwidth_positive, kde_refuse_bandwidth,
    kde_refuse_algorithm, kde_refuse_tree_knobs, kde_refuse_names, kde_check_metric_params_count,
    kde_minkowski_supported, kde_refuse_minkowski, kde_refuse_unfitted, kde_refuse_features,
    kde_resident_capable, kde_immutable_capable, kde_copy_f32, kde_copy_finite_f32, kde_query_finite_status,
    kde_fit_key, kde_prepare_request, kde_query_request, kde_resident_query,
)
from bindings.py_balanced import py_call, py_get, py_contains, py_setitem, py_fields, py_type
from bindings.classical_arrays_door import classical_address
from bindings.classical_attributes_door import classical_connect_numpy, classical_check_layout
from bindings.classical_pandas_door import classical_admit_value
from bindings.host_inference_scalars_door import host_inference_scalar
from bindings.family_common_unpack import (
    family_repr,
    family_str,
    family_is_str,
)
from bindings.family_common_pack import (
    family_call,
    family_field,
)
from bindings.host_sparse_native_door import host_sparse_format, host_sparse_dense, _sparse_fields
# TOMBSTONE: _any_contiguous_address use (the labels admission copy's buffer address read) deleted 2026-11-09 by lane a1b-doors-opt-in-trees-linear; code recoverable at f59d913b0b.
# Restore: experiments/removed/A1B_LABELS_ADMISSION_COPY.patch.
from bindings.admitted_input_unpack import admitted_finite_refuse
from bindings.admitted_finite_wording import admitted_finite_wording
from bindings.p3nc_fitted_freeze_unpack import fitted_freeze


comptime _KDE_STATE_SLOT = "neighbors_kde_state_v2"


def _kde_admit_state(model: PythonObject, context: PythonObject, fitting: Bool = False) raises:
    classical_check_layout(model, context)
    var fields = py_fields(model)
    lifecycle_admit_fields(fields)
    var names: List[String] = ["bandwidth", "algorithm", "kernel", "metric", "atol", "rtol", "breadth_first",
                               "leaf_size", "n_features_in_", "n_samples_fit_"]
    var arrays: List[String] = ["_x", "_w"]
    var hit = kde_admission_hit(model, PythonObject(None), False, _KDE_STATE_SLOT, context)
    if not hit:
        for i in range(len(names)):  # small-loop(fixed KDE settings and fitted metadata)
            var value = py_get(fields, names[i])
            if value is not None:
                host_inference_scalar(value, context, True)
        var params = py_get(fields, "metric_params")
        if params is not None:
            if py_type(params) is not native_loaded_module("builtins").dict:
                raise Error("TypeError: KernelDensity metric_params must be an exact native dictionary")
            classical_admit_value(params, context)
            var values = py_call(native_loaded_module("builtins").tuple, params.values())
            for i in range(len(values)):  # driver-loop: admitted metric-parameter scalar values
                host_inference_scalar(values[i], context)
        for i in range(len(arrays)):  # small-loop(fitted KDE buffer owners)
            var value = py_get(fields, arrays[i])
            if value is not None:
                classical_admit_value(value, context)
    if fitting:
        kde_admission_drop(model, _KDE_STATE_SLOT)
        return
    if not hit:
        kde_admission_remember(model, PythonObject(None), False, _KDE_STATE_SLOT, context)


def _kde_sparse_f32(source: PythonObject, format: String) raises -> Bool:
    """Whether `source.toarray()` is already float32 (the matrix's own
    dtype, read from its stored fields as `host_sparse_dense` reads it)."""
    var fields = _sparse_fields(source)
    var dtype = fields["dtype"] if (format == "dok" or format == "lil") else fields["data"].dtype
    return String(dtype.kind) == "f" and Int(py=dtype.itemsize) == 4


def _kde_borrowed(source: PythonObject, x: PythonObject, context: PythonObject) raises -> Bool:
    """`_materialize` borrowed `source` and the cast kept it: the converted
    array reads the source's own memory (an Array, or a contiguous buffer
    export); a list, a strided export or a cast is a copy."""
    var b = native_loaded_module("builtins")
    if py_type(source) is b.list or py_type(source) is b.tuple:
        return False
    var address = Int(py=classical_address(x))
    if py_type(source) is context["array_class"]:
        return address == Int(py=classical_address(source))
    try:
        # The exporter's own data address (classical_address: an ndarray's,
        # or a buffer export's first element).
        return Int(py=classical_address(source)) == address
    except:
        return False


@fieldwise_init
struct KDEDense(Movable):
    var array: PythonObject
    var copied: Bool


def _kde_dense(source: PythonObject, context: PythonObject) raises -> KDEDense:
    """`_buffer.as_f32_dense_c(source, ndim=2, name="X")`: typed array/copy metadata.
    A scipy.sparse matrix is densified exactly first (`toarray()`), and
    `copied` is the conversion's flag for the dense array it was handed, as
    `as_f32_c(obj.toarray())` reported it: a copy unless the matrix holds
    float32 (`toarray()` keeps its dtype)."""
    var sparse = host_sparse_format(source)
    if sparse.byte_length() > 0:
        if not py_contains(context, PythonObject("output_context")):
            py_setitem(context, "output_context", context)
        var dense = admit_family(host_sparse_dense(source, sparse, context), 2, "X", context, False, True,
                                 refusal=admitted_finite_wording("KernelDensity", "X"), defer_finite=True).array
        return KDEDense(dense, not _kde_sparse_f32(source, sparse))
    # Lane a7b-device-x-neighbors: a cast X is refused from the cast's own
    # verdict; a borrowed one by the fit's copy (`_kde_copy_finite`) or the
    # score entry's status word.
    var x = admit_family(source, 2, "X", context, False, True,
                         refusal=admitted_finite_wording("KernelDensity", "X"), defer_finite=True).array
    var copied = not _kde_borrowed(source, x, context)
    return KDEDense(x, copied)


def _kde_validate(fields: PythonObject, context: PythonObject) raises:
    """`KernelDensity._validate_params`: the constructor checks, in the
    Python method's order and wording."""
    var b = native_loaded_module("builtins")
    var bandwidth = family_field(fields, "bandwidth")
    var algorithm = family_field(fields, "algorithm")
    var kernel = family_field(fields, "kernel")
    var metric = family_field(fields, "metric")
    var atol = family_field(fields, "atol")
    var rtol = family_field(fields, "rtol")
    var breadth_first = family_field(fields, "breadth_first")
    var leaf_size = family_field(fields, "leaf_size")
    var metric_params = family_field(fields, "metric_params")
    if family_is_str(bandwidth):
        kde_refuse_bandwidth_name(family_str(bandwidth))
    var width = family_call(b.float, bandwidth)
    if not kde_bandwidth_positive(Float64(py=width)):
        kde_refuse_bandwidth(family_repr(width))
    if algorithm is not None and not decomp_name_admitted(algorithm, "auto"):
        kde_refuse_algorithm(decomp_scalar_repr(algorithm, context))
    var knobs = not decomp_equal_admitted(atol, PythonObject(0))
    if not knobs:
        knobs = not decomp_equal_admitted(rtol, PythonObject(0))
    if not knobs:
        knobs = not decomp_truth(breadth_first, context)
    if not knobs:
        knobs = not decomp_equal_admitted(leaf_size, PythonObject(KDE_DEFAULT_LEAF_SIZE))
    if knobs:
        kde_refuse_tree_knobs()
    if not family_is_str(kernel) or not family_is_str(metric):
        kde_refuse_names()
    if decomp_truth(metric_params, context):
        kde_check_metric_params_count(len(metric_params))
        var values = py_call(b.tuple, metric_params.values())
        var p = family_call(b.float, values[0])
        if not kde_minkowski_supported(Float64(py=p)):
            kde_refuse_minkowski(_kde_params_repr(metric_params, context), family_repr(p))


def _kde_params_repr(params: PythonObject, context: PythonObject) raises -> String:
    # Do not enter NumPy arrayprint through a dict's element repr. Exact
    # primitive scalar diagnostics retain their usual representation.
    var pairs = py_call(native_loaded_module("builtins").tuple, params.items())
    var result = String("{")
    for i in range(len(pairs)):  # driver-loop: diagnostic parameter metadata only
        if i:
            result += ", "
        result += decomp_scalar_repr(pairs[i][0], context) + ": " + decomp_scalar_repr(pairs[i][1], context)
    return result + "}"


def _kde_text(value: PythonObject, context: PythonObject) raises -> String:
    if py_type(value) is py_get(context, "ndarray"):
        # Such a value is not a supported kernel/metric name. Avoid the
        # process-wide Python NumPy formatter on its diagnostic route.
        return decomp_scalar_repr(value, context)
    return family_str(value)
