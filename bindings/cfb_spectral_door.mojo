# SPDX-License-Identifier: Apache-2.0
# Spectral public UNPACK/PACK and lifecycle metadata. Numerical execution,
# scratch, requests and resident handles use the typed native plans. Public
# Arrays and provider images remain boundary pins. Normalized scalar fields
# retain archive/attribute compatibility; execution consumes native config.
from providers.python.function_device_boundary_door import function_device_admit
from providers.python.estimator_binding_pyapi_door import estimator_binding_device
from core.function_device import function_device_default
from std.math import isinf
from core.spectral_native_api import (
    SPECTRAL_CAP_DENSE, SPECTRAL_CAP_RESIDENT, SPECTRAL_DATASET, SPECTRAL_GRAPH, SPECTRAL_DENSE,
    SPECTRAL_CAP_AFFINITY, SPECTRAL_RBF, SPECTRAL_KNN, SPECTRAL_KNN_REFUSED,
    SPECTRAL_CAP_DEV_STORE, SPECTRAL_DATASET_STORE, SPECTRAL_MODE,
    SpectralInput, SpectralFitOptions, SpectralFitRequest,
    SpectralEmbeddingOptions, SpectralEmbeddingRequest, SpectralQueryShape,
    SpectralQueryRequest, SpectralPrepareRequest, SpectralResidentQueryRequest,
)
from bindings.spectral_native_boundary_unpack import (
    SpectralConnection, SpectralStatePins, SpectralRoutePins, SpectralPredictionPins,
    spectral_fit_run, spectral_embedding_run, spectral_query_run,
)
from std.memory import unsafe_memcpy
from bindings.function_scope_door import FunctionScope
from core.metrics_store_ref import metrics_store_find, METRICS_STORE_PREFIX
from core.device_route import device_vendor_code
from std.python import Python, PythonObject
from providers.python.host_estimator_binding_pyapi_door import host_estimator_binding
from providers.python.estimator_binding_pyapi_door import estimator_binding_routed
from bindings.admitted_family_door import admit_family, admit_family_f64, admit_family_input
from providers.python.array_metadata_pyapi_door import ArrayPin
from bindings.admitted_input_unpack import admit_typed
from bindings.py_balanced import py_call, py_type, py_contains, py_setitem, py_fields, py_get
from bindings.classical_arrays_door import classical_address, classical_is_array
from core.host_finite_scan import host_all_finite_f32
from core.family_cast_impl import family_min_f32
from core.spectral_embedding_config import SpectralEmbeddingConfig
from core.spectral_clustering_config import SpectralClusteringConfig, spectral_assignment_method
from bindings.host_inference_scalars_door import host_inference_scalar
from bindings.admitted_family_door import admit_family_refuse_empty
from bindings.family_common_unpack import (
    FAMILY_F32P,
    family_repr,
    family_str,
    family_is_str,
    family_min,
)
from bindings.family_common_pack import (
    family_call,
    family_array,
    family_field,
)
from bindings.cluster_native_door import _reference_queries
from core.spectral_assignment_api import SpectralAssignmentRequest
from bindings.native_array_storage_door import native_array_zeros
from std.python._cpython import GILReleased
from core.spectral_assignment_api import (
    SPECTRAL_ASSIGN_ABI, SPECTRAL_ASSIGN_COUNT, SpectralAssignmentDriver,
    SpectralAssignmentRequest, SpectralAssignmentOutput,
)
from core.native_array_layout import NativeArrayLayout
from bindings.spectral_umap_routes_unpack import spectral_umap_entry
from bindings.native_array_storage_door import native_array_unfilled
from bindings.native_buffer_address_unpack import native_buffer_address
from bindings.classical_arrays_door import classical_wrap_layout
from providers.python.neural_descriptor_pyapi_door import (
    NeuralDescriptorProvider, neural_descriptor_connect_factory_once,
)
from bindings.spectral_affinity_boundary import SpectralAffinityRoom, spectral_rbf_run
from core.spectral_preparation_api import (
    SpectralCOOCountRequest, SpectralCOOFillRequest, SpectralKNNRequest, SpectralAffinityStatus, SpectralCOOStorage,
)
from core.spectral_preparation_api import (
    SPECTRAL_PREPARATION_ABI, SPECTRAL_PREPARATION_COUNT, SpectralPreparationDriver,
    SpectralCOOCountRequest, SpectralCOOFillRequest, SpectralKNNRequest,
)
from providers.python.neural_descriptor_pyapi_door import NeuralDescriptorProvider, neural_descriptor_connect_factory_once
from bindings.py_cluster_resident import CRES_SPECTRAL, cres_spectral_lookup, cres_spectral_store, cres_forget
from bindings.spectral_resident_boundary_unpack import (
    SpectralResidentLease, spectral_resident_prepare, spectral_resident_query, spectral_resident_retire,
)
from bindings.p3nc_fitted_freeze_unpack import fitted_is_freezable, fitted_freeze
from bindings.operation_names_cfb import (
    CFB_OPN_FAMILY_CFB_SC_FIT, CFB_OPN_FAMILY_CFB_SC_FIT_PREDICT, CFB_OPN_FAMILY_CFB_SC_PREDICT,
    CFB_OPN_FAMILY_CFB_SC_PREDICT_EMBEDDING, CFB_OPN_FAMILY_CFB_SE_FIT, CFB_OPN_FAMILY_CFB_SE_FIT_TRANSFORM,
    CFB_OPN_FAMILY_CFB_SE_VALIDATE, CFB_OPN_FAMILY_CFB_SE_KNN_AFFINITY, CFB_OPN_FAMILY_CFB_SPECTRAL_EMBEDDING,
    CFB_OPN_FAMILY_CFB_COO_TRIPLES)
from bindings.spectral_umap_routes_pack import (
    spectral_umap_route,
)
from bindings.cfb_sparse_read_unpack import cfb_sparse_triples
from bindings.host_sparse_native_door import host_sparse_format, host_sparse_dense
from bindings.native_loaded_module_door import native_loaded_module
from bindings.family_raw_attr_unpack import family_raw_has
from bindings.family_raw_attr_pack import family_raw_get, family_raw_get_or
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.classical_pandas_door import classical_admit_value
from bindings.classical_validate_unpack import classical_estimator_name
from bindings.classical_validate_run import classical_check_width
from bindings.classical_exceptions_door import classical_require_fitted
from bindings.native_array_view_pack import native_array_ndarray_view
from bindings.native_owner_lifetime_door import native_release_admit
from providers.python.spectral_lifecycle_pyapi_unpack import spectral_input
from providers.python.spectral_lifecycle_pyapi_pack import spectral_admit
from providers.python.estimator_lifecycle_pyapi_unpack import estimator_outside, estimator_feature_check, estimator_fit_begin, estimator_fit_end, estimator_fit_complete
from providers.python.estimator_tabular_boundary_pyapi_unpack import _target, _refusal
from bindings.py_optional import py_present
from spectral.cfb_spectral_run import _sc_check_shape, _sc_refusal, _se_check_shape, spectral_lanczos_fits
from bindings.cluster_keys_unpack import cluster_keys
from bindings.py_balanced import py_item


# ---- from bindings/spectral_preparation_boundary_unpack.mojo (inlined: its one importer is this file) ----
# SPDX-License-Identifier: Apache-2.0
# Spectral preparation boundary image pins around typed native requests.


struct SpectralPreparationConnection(Movable):
    var provider: PythonObject
    var driver: SpectralPreparationDriver

    def __init__(out self, binding: PythonObject, context: PythonObject) raises:
        var factory = spectral_umap_entry(binding, "spectral_preparation_native_api_v1", context)
        var provider = neural_descriptor_connect_factory_once(factory,
            String(SPECTRAL_PREPARATION_ABI), SPECTRAL_PREPARATION_COUNT)
        self.driver = SpectralPreparationDriver(provider.downcast_value_ptr[NeuralDescriptorProvider]()[].record[0].entry)
        self.provider = provider


def spectral_coo_count_run(connection: SpectralPreparationConnection, request: SpectralCOOCountRequest) raises -> Int:
    var count: Int
    try:
        with GILReleased(Python()):
            count = connection.driver.count(request)
    finally:
        _ = connection.provider
    return count


def spectral_coo_fill_run(connection: SpectralPreparationConnection, request: SpectralCOOFillRequest) raises -> Int:
    var count: Int
    try:
        with GILReleased(Python()):
            count = connection.driver.fill(request)
    finally:
        _ = connection.provider
    return count


def spectral_knn_run(connection: SpectralPreparationConnection, request: SpectralKNNRequest) raises:
    try:
        with GILReleased(Python()):
            connection.driver.knn(request)
    finally:
        _ = connection.provider

# ---- from bindings/spectral_assignment_boundary_unpack.mojo (inlined: its one importer is this file) ----
# SPDX-License-Identifier: Apache-2.0
# Alternative label assignment UNPACK/PACK; execution is one native call.


def spectral_assignment_run(binding: PythonObject, request: SpectralAssignmentRequest,
                            context: PythonObject) raises -> SpectralAssignmentOutput:
    var factory = spectral_umap_entry(binding, "spectral_assign_native_api_v1", context)
    var provider = neural_descriptor_connect_factory_once(factory, String(SPECTRAL_ASSIGN_ABI), SPECTRAL_ASSIGN_COUNT)
    var driver = SpectralAssignmentDriver(provider.downcast_value_ptr[NeuralDescriptorProvider]()[].record[0].entry)
    var output = SpectralAssignmentOutput()
    try:
        with GILReleased(Python()):
            driver.assign(request, output)
    finally:
        _ = provider
    return output^

# ---- from bindings/spectral_assignment_boundary_pack.mojo (inlined: its one importer is this file) ----
# SPDX-License-Identifier: Apache-2.0
# Alternative label assignment UNPACK/PACK; execution is one native call.
# The PACK half of the former `bindings/spectral_assignment_boundary.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/spectral_assignment_boundary_unpack.mojo`.


def spectral_assignment_pack(output: SpectralAssignmentOutput, rows: Int,
                             context: PythonObject) raises -> PythonObject:
    # The old ClusterOut serializer copied one labels list into one exporter.
    # Retain that copy and owning exporter without its nested Python lists.
    var count = len(output.storage[0].labels)
    # lane fix-output-zero-fill (2026-11-14): unfilled; the memcpy below writes all count labels.
    var store = native_array_unfilled("i", count)
    if count > 0:
        unsafe_memcpy(dest=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=native_buffer_address(store)),
            src=output.storage[0].labels.unsafe_ptr(), count=count)
    _ = output
    var layout = NativeArrayLayout(List[Int](length=1, fill=rows), 4)
    return classical_wrap_layout(store, layout, "<i4", context)

def _builtins() raises -> PythonObject:
    return native_loaded_module("builtins")


def cfb_borrow_copy(x: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`x if input_copied_ else x.copy()` as a mojolearn Array: an Array the
    cast made is kept, anything else (the caller's own buffer) is copied so
    a later write by the caller cannot move what predict reads."""
    if classical_is_array(x, context) and not (x is source):
        return x
    var b = _builtins()
    var copied = family_array(py_call(b.tuple, x.shape), "<f4", context)
    unsafe_memcpy(dest=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(copied))),
           src=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(x))), count=Int(py=x.size))
    _ = copied  # keep the Python object alive through the native call above
    return copied


def _routed(model: PythonObject, group: String, name: String, requested: PythonObject,
            context: PythonObject) raises -> PythonObject:
    """The compiled module of one spectral group for this operation. An
    estimator's operation takes it from the ESTIMATOR'S binding route (its
    device, read once on a miss; a hit is one record read and, per held
    name, a name compare and two pointer compares). A module-level function
    has no estimator and keeps the import-time process connection."""
    if model is None:
        return spectral_umap_route(context, group, requested)
    return estimator_binding_routed(model, requested, name, context, False)


def _routes(model: PythonObject, clustering: Bool, context: PythonObject) raises -> SpectralRoutePins:
    """The four modules of a fit. The base image runs `identical` only: the
    mode object is the one retained in the connection context, so the route
    compares it by identity (a new str per call would never hit)."""
    var identical = context["family_cfb_identical"]
    var cluster = PythonObject(None)
    if clustering:
        var nk = cluster_keys()
        cluster = _routed(model, "cluster", "_mojolearn_x_cluster", py_get(py_fields(model), nk[].k_numeric_mode), context)
    return SpectralRoutePins(
        _routed(model, "metrics", "_mojolearn_metrics", PythonObject(None), context),
        _routed(model, "decomp", "_mojolearn_x_decomp", PythonObject(None), context),
        _routed(model, "base", "_mojolearn", identical, context), cluster)


def _int(value: PythonObject) raises -> Int:
    """`int(value)`, the value's own conversion."""
    return Int(py=family_call(_builtins().int, value))


def _float(value: PythonObject) raises -> PythonObject:
    """`float(value)`, the value's own conversion."""
    return family_call(_builtins().float, value)


def _shape_of(obj: PythonObject) raises -> PythonObject:
    """`_buffer.shape_of`: `.shape` when the object has one, the nesting of
    a list or tuple otherwise."""
    var b = _builtins()
    var shape = py_call(b.getattr, obj, PythonObject("shape"), PythonObject(None))
    if shape is not None:
        return family_call(b.tuple, shape)
    if Bool(py=family_call(b.isinstance, obj, Python.tuple(b.list, b.tuple))):
        if Bool(py=family_call(b.bool, obj)) and Bool(py=family_call(b.isinstance, obj[0], Python.tuple(b.list, b.tuple))):
            return Python.tuple(family_call(b.len, obj), family_call(b.len, obj[0]))
        return Python.tuple(family_call(b.len, obj))
    if Bool(py=family_call(b.hasattr, obj, PythonObject("__len__"))):
        return Python.tuple(family_call(b.len, obj))
    return Python.tuple()


def _square(shape: PythonObject) raises -> Bool:
    """`len(shape) == 2 and shape[0] == shape[1]`."""
    if len(shape) != 2:
        return False
    return Bool(py=family_call(_builtins().bool, family_call(native_loaded_module("operator").eq, shape[0], shape[1])))


def _as_i32(source: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """`_buffer.as_i32_c(source, ndim=1, name=name)`: the rank and emptiness
    refusals, then the vector itself when it is already C-order int32, else
    one int32 conversion by Array.astype's rules (truncation toward zero, a
    value outside int32 refused), split across the host's cores."""
    # TOMBSTONE: the family_view + float64 widen + serial cfb_f64_to_i32 cast (core/cfb_impl.mojo) deleted 2026-11-09 by lane a6-device-x-family-classical; code recoverable at f59d913b0b.
    # Restore: experiments/removed/A6_FAMILY_VIEW.patch. The multi-dtype
    # admission (bindings/admitted_input_unpack.mojo admit_typed) casts once in parallel.
    return admit_typed(source, "<i4", 1, name, context).keep


# TOMBSTONE: P1_SPECTRAL_TOCOO (removed: the caller's own tocoo() / toarray() inside spectral fit and predict) deleted 2026-11-09 by lane p1-python-callbacks-worst; code recoverable at e68654a757.
# Restore: experiments/removed/P1_SPECTRAL_TOCOO.patch.
# P1 (audit A #4): _tocoo removed. It looked up and called the caller's
# own `tocoo()` (SciPy's Python conversion, or any object's method) inside
# fit. A sparse input is now an exact SciPy container read natively from its
# stored buffers (_is_sparse, _sparse_triples, _densified); any other object
# takes the dense admission, which refuses what it cannot read.


def _is_sparse(obj: PythonObject) raises -> Bool:
    """Whether `obj` is an exact SciPy sparse container of any format (a
    type identity check; no attribute of the object is read)."""
    return host_sparse_format(obj).byte_length() > 0


def _densified(source: PythonObject, context: PythonObject) raises -> PythonObject:
    """An exact SciPy container whose stored coordinates were not taken
    (BSR, DIA, DOK, LIL, or a COO/CSR/CSC whose storage was not read)
    densified natively to float32 from its stored buffers, duplicates
    summed (`host_sparse_dense`); any other input unchanged."""
    var format = host_sparse_format(source)
    if format.byte_length() == 0:
        return source
    return host_sparse_dense(source, format, context)


struct _Coo(Movable):
    # Array pins are needed only for admitted sparse inputs and the public
    # triples boundary. Dense fit intermediates use the native storage owner.
    var rows: PythonObject
    var cols: PythonObject
    var vals: PythonObject
    var n: Int
    var native: Optional[SpectralCOOStorage]

    def __init__(out self, rows: PythonObject, cols: PythonObject, vals: PythonObject, n: Int):
        self.rows = rows
        self.cols = cols
        self.vals = vals
        self.n = n
        self.native = None

    def __init__(out self, n: Int, count: Int):
        self.rows = PythonObject(None)
        self.cols = PythonObject(None)
        self.vals = PythonObject(None)
        self.n = n
        self.native = SpectralCOOStorage(count)

    def count(self) raises -> Int:
        return len(self.native.value().values) if self.native else Int(py=self.vals.size)

    def row_address(self) raises -> Int:
        return Int(self.native.value().rows.unsafe_ptr()) if self.native else Int(py=classical_address(self.rows))

    def column_address(self) raises -> Int:
        return Int(self.native.value().columns.unsafe_ptr()) if self.native else Int(py=classical_address(self.cols))

    def value_address(self) raises -> Int:
        return Int(self.native.value().values.unsafe_ptr()) if self.native else Int(py=classical_address(self.vals))


def _coo_room(n: Int, count: Int, public_storage: Bool, context: PythonObject) raises -> _Coo:
    if not public_storage:
        return _Coo(n, count)
    var rows = family_array(Python.tuple(count), "<i4", context)
    var columns = family_array(Python.tuple(count), "<i4", context)
    var values = family_array(Python.tuple(count), "<f4", context)
    return _Coo(rows, columns, values, n)


def _coo_of_triples(rows: PythonObject, cols: PythonObject, vals: PythonObject, shape: PythonObject,
                    who: String, context: PythonObject) raises -> _Coo:
    """`_coo_triples`' sparse arm on (row, col, data) and the shape."""
    if not _square(shape):
        raise Error("mojolearn " + who + ": a precomputed affinity matrix "
                    + "must be square, got shape " + family_repr(shape))
    var r = _as_i32(rows, "rows", context)
    var c = _as_i32(cols, "cols", context)
    var v = admit_family(vals, 1, "vals", context).array
    return _Coo(r, c, v, _int(shape[0]))


def _sparse_triples(source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`(row, col, data, shape)` of an exact SciPy COO/CSR/CSC input read
    from its stored buffers (bindings/cfb_sparse_read.mojo); None for any
    other input (the caller densifies another SciPy format natively with
    `_densified`). No method of the input is called."""
    return cfb_sparse_triples(source, context)


def _coo_triples(source: PythonObject, who: String, base: PythonObject, context: PythonObject, public_storage: Bool = False) raises -> _Coo:
    """`_coo_triples(A, who)`: an exact SciPy COO/CSR/CSC input's stored
    triples (read natively), another SciPy format densified natively, a
    dense one through the base
    binding's float64 nonzero scan (exact zeros dropped, as
    `sp.coo_matrix(dense)` drops them)."""
    var triples = _sparse_triples(source, context)
    if triples is not None:
        return _coo_of_triples(triples[0], triples[1], triples[2], triples[3], who, context)
    var dense_source = _densified(source, context)
    var shape = _shape_of(dense_source)
    if not _square(shape):
        raise Error("mojolearn " + who + ": with affinity='precomputed', X "
                    + "must be a square affinity matrix or a SciPy sparse matrix, "
                    + "got shape " + family_repr(shape))
    # The metadata step of the one admission path (rank and emptiness
    # refusals), then the float64 widen of the same view (family_view deleted).
    var pin = ArrayPin()
    _ = admit_family_input(dense_source, 2, "X", context, pin)
    var dense = admit_family_f64(pin.keep, "X", context)
    var n = _int(shape[0])
    var size = Int(py=dense.size)
    var nnz = 0
    var connection: Optional[SpectralPreparationConnection] = None
    if size:
        connection = SpectralPreparationConnection(base, context)
        nnz = spectral_coo_count_run(connection.value(), SpectralCOOCountRequest(Int(py=classical_address(dense)), size, True))
        _ = dense  # keep the Python object alive through the native call above
    var room = _coo_room(n, nnz, public_storage, context)
    if nnz == 0:
        return room^
    var request = SpectralCOOFillRequest(Int(py=classical_address(dense)),
        room.row_address(), room.column_address(), room.value_address(), n, n, nnz, True)
    var wrote = spectral_coo_fill_run(connection.value(), request)
    _ = dense
    _ = room
    if wrote != nnz:
        raise Error("RuntimeError: mojolearn " + who + ": nonzero_f64_fill wrote "
                    + String(wrote) + " of " + String(nnz) + " entries")
    return room^


def _dense_coo(m: PythonObject, who: String, base: PythonObject, context: PythonObject) raises -> _Coo:
    var n_r = Int(py=m.shape[0])
    var n_c = Int(py=m.shape[1])
    var address = Int(py=classical_address(m)) if n_r * n_c else 0
    var result = _dense_coo_span(address, n_r, n_c, who, base, context)
    _ = m
    return result^


def _dense_coo_span(address: Int, n_r: Int, n_c: Int, who: String,
                    base: PythonObject, context: PythonObject) raises -> _Coo:
    """The existing row-major count/fill path over an owned native span."""
    var nnz = 0
    var connection: Optional[SpectralPreparationConnection] = None
    if n_r * n_c:
        connection = SpectralPreparationConnection(base, context)
        nnz = spectral_coo_count_run(connection.value(), SpectralCOOCountRequest(address, n_r * n_c, False))
    var room = _Coo(n_r, nnz)
    if nnz:
        var request = SpectralCOOFillRequest(address,
            room.row_address(), room.column_address(), room.value_address(), n_r, n_c, nnz, False)
        var wrote = spectral_coo_fill_run(connection.value(), request)
        _ = room
        if wrote != nnz:
            raise Error("RuntimeError: mojolearn: nonzero_f32_fill wrote " + String(wrote) + " of "
                        + String(nnz) + " entries")
    # The old _coo_of_triples checked squareness, then rejected the first
    # empty rows vector. Known native output dtypes need no second admission.
    if n_r != n_c:
        raise Error("mojolearn " + who + ": a precomputed affinity matrix "
                    + "must be square, got shape " + family_repr(Python.tuple(n_r, n_c)))
    if nnz == 0:
        admit_family_refuse_empty("rows", List[Int](length=1, fill=0))
    return room^


def _x_decomp_input(source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_M.from_input(X)`: a scipy.sparse matrix densified, `as_f32_c`, and
    the finiteness refusal. An exact SciPy container is densified natively
    from its stored buffers straight to float32 (duplicates summed, as
    `toarray()` sums them; `host_sparse_dense`, which refuses an empty
    matrix). No method of the input is called (P1, audit A #4: the
    `toarray()` call on other sparse-like inputs is removed; such an object
    takes the dense admission)."""
    var x = _densified(source, context)
    # TOMBSTONE: cfb_spectral.mojo second host scans after admission (four sites) deleted 2026-11-09 by lane a8d-device-x-decomp-manifold; code recoverable at 0ce88411d3.
    # Restore: experiments/removed/A8D_DECOMP_MANIFOLD_HOST_FINITE.patch.
    # Lane a8d: the finiteness refusal is the admission's own (tested in the
    # cast pass, or one parallel scan of a borrowed X, or in a fold's
    # gather), not a second scan after it; the same sentence.
    var a = admit_family(x, 2, "X", context, dense=True, finite=True,
                         refusal="X: input must be finite; NaN/inf are unsupported").array
    return a


def _rbf_affinity(x: PythonObject, gamma: Float64, xdecomp: PythonObject, context: PythonObject,
                  public_storage: Bool = True) raises -> SpectralAffinityRoom:
    """The existing sqdist → scale → exp cells, with native scratch/handles."""
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var affinity = SpectralAffinityRoom(n, public_storage, context)
    spectral_rbf_run(xdecomp, x, affinity, n, d, gamma)
    return affinity^


struct _KnnAdmitted(Movable):
    """The admitted precomputed distances of a kNN affinity: dense (n x n) or
    COO triples, and the resolved n_neighbors. Owns the Python arrays the
    native call borrows."""
    var n: Int
    var k: Int
    var nnz: Int
    var dense: Optional[PythonObject]
    var rows: PythonObject
    var cols: PythonObject
    var vals: PythonObject

    def __init__(out self, n: Int, k: Int, nnz: Int, var dense: Optional[PythonObject],
                 rows: PythonObject, cols: PythonObject, vals: PythonObject):
        self.n = n
        self.k = k
        self.nnz = nnz
        self.dense = dense^
        self.rows = rows
        self.cols = cols
        self.vals = vals

    def sparse(self) -> Bool:
        return not self.dense

    def dense_address(self) raises -> Int:
        return 0 if not self.dense else Int(py=classical_address(self.dense.value()))

    def row_address(self) raises -> Int:
        return Int(py=classical_address(self.rows)) if self.sparse() and self.nnz else 0

    def column_address(self) raises -> Int:
        return Int(py=classical_address(self.cols)) if self.sparse() and self.nnz else 0

    def value_address(self) raises -> Int:
        return Int(py=classical_address(self.vals)) if self.sparse() and self.nnz else 0


def _knn_admit(fields: PythonObject, source: PythonObject, clustering: Bool, context: PythonObject,
               native_options: Bool, neighbors: Int, automatic_neighbors: Bool) raises -> _KnnAdmitted:
    """`_knn_affinity`'s admission: the precomputed distances (an exact SciPy
    container's stored triples, else the dense square matrix) and k."""
    var b = _builtins()
    var triples = _sparse_triples(source, context)
    var n: Int
    var nnz = 0
    # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
    var dense = Optional[PythonObject]()
    var rows = PythonObject(None)
    var cols = PythonObject(None)
    var vals = PythonObject(None)
    if triples is not None:
        var shape = triples[3]
        n = _int(shape[0])
        if not Bool(py=family_call(b.bool, family_call(native_loaded_module("operator").eq,
                                                        shape, Python.tuple(n, n)))):
            raise Error("mojolearn SpectralEmbedding: the precomputed distance graph must be square")
        rows = _as_i32(triples[0], "rows", context)
        cols = _as_i32(triples[1], "cols", context)
        vals = admit_family(triples[2], 1, "data", context).array
        if not (Int(py=rows.shape[0]) == Int(py=cols.shape[0]) and Int(py=cols.shape[0]) == Int(py=vals.shape[0])):
            raise Error("mojolearn SpectralEmbedding: the precomputed graph's row, col and data differ in length")
        nnz = Int(py=vals.shape[0])
    else:
        dense = admit_family(_densified(source, context), 2, "X", context).array
        n = Int(py=dense.value().shape[0])
        if Int(py=dense.value().shape[1]) != n:
            raise Error("mojolearn SpectralEmbedding: the precomputed distance matrix must be square")
    var k = (max(n // 10, 1) if automatic_neighbors else neighbors) if native_options else _resolved_neighbors(fields, n, clustering)
    return _KnnAdmitted(n, k, nnz, dense^, rows, cols, vals)


def _knn_refusal(status: SpectralAffinityStatus, k: Int) raises:
    """The kNN affinity's refusal words from its 3 status words."""
    var code = Int(status.words[0])
    var bad_row = Int(status.words[1])
    var stored = Int(status.words[2])
    if code == 1:
        raise Error("mojolearn SpectralEmbedding: a precomputed distance is negative or NaN")
    if code == 2:
        raise Error("mojolearn SpectralEmbedding: row " + String(bad_row) + " of the precomputed graph has "
                    + String(stored) + " stored distances, fewer than n_neighbors=" + String(k))


def _knn_affinity(fields: PythonObject, source: PythonObject, clustering: Bool, base: PythonObject,
                   context: PythonObject, native_options: Bool = False, neighbors: Int = 0,
                   automatic_neighbors: Bool = False, public_storage: Bool = True) raises -> SpectralAffinityRoom:
    """`SpectralEmbedding._precomputed_knn_affinity`: scikit-learn's
    `kneighbors_graph(mode='connectivity')` of a precomputed distance matrix,
    symmetrized 0.5 (C + C^T), by the base binding's `knn_affinity_f32`.
    Sets and returns `affinity_matrix_`."""
    var admitted = _knn_admit(fields, source, clustering, context, native_options, neighbors, automatic_neighbors)
    var n = admitted.n
    var k = admitted.k
    var a = SpectralAffinityRoom(n, public_storage, context)
    var nk = cluster_keys()
    if n == 0:
        if fields is not None:
            py_setitem(fields, nk[].k_affinity_matrix_, a.public_array)
        return a^
    var status = SpectralAffinityStatus()
    var connection = SpectralPreparationConnection(base, context)
    var request = SpectralKNNRequest(admitted.dense_address(), admitted.row_address(),
        admitted.column_address(), admitted.value_address(),
        a.address(), status.address(), n, k, admitted.nnz, admitted.sparse())
    spectral_knn_run(connection, request)
    _ = admitted  # keep the admitted Python arrays alive through the native call above
    _ = status  # retain native status storage through the call
    _ = a  # retain the public or native affinity owner through the call
    _knn_refusal(status, k)
    if fields is not None:
        py_setitem(fields, nk[].k_affinity_matrix_, a.public_array)
    return a^


def _resolved_neighbors(fields: PythonObject, n: Int, clustering: Bool) raises -> Int:
    """SpectralClustering: `_n_neighbors`; SpectralEmbedding: `_n_neighbors`,
    or `max(n // 10, 1)` for None."""
    var value = family_field(fields, "_n_neighbors")
    if clustering or value is not None:
        return Int(py=value)
    return max(n // 10, 1)


def _scalar_repr(value: PythonObject, context: PythonObject) raises -> String:
    """Diagnostic for already-admitted scalar state, without NumPy printing.

    Spectral admission also permits exact size-one numeric arrays at any rank.
    Their repr enters NumPy's Python override/formatter hooks; identify that
    native storage directly. Primitive scalar representations stay unchanged.
    """
    var nk = cluster_keys()
    if py_type(value) is py_get(context, nk[].k_ndarray):
        return "<native scalar array>"
    return family_repr(value)


def _seed(random_state: PythonObject, who: String, clustering: Bool, context: PythonObject) raises -> Int:
    """`random_state` None is seed 0 (DEVIATION 891); else 0 <= seed < 2**32."""
    if random_state is None:
        return 0
    var seed = _int(random_state)
    if seed < 0 or seed >= 4294967296:
        var tail = String(" (cuML's check_random_seed refuses the same range)") if clustering else String("")
        raise Error("mojolearn " + who + ": random_state must satisfy "
                    + "0 <= random_state < 2**32, got " + _scalar_repr(random_state, context) + tail)
    return seed


def _sc_validate(fields: PythonObject, context: PythonObject) raises -> SpectralClusteringConfig:
    """`SpectralClustering._validate_params`."""
    var b = _builtins()
    var n_clusters = family_field(fields, "n_clusters")
    var n_components = family_field(fields, "n_components")
    var random_state = family_field(fields, "random_state")
    var n_neighbors = family_field(fields, "n_neighbors")
    var n_init = family_field(fields, "n_init")
    var eigen_tol = family_field(fields, "eigen_tol")
    var affinity = family_field(fields, "affinity")
    var assign_labels = family_field(fields, "assign_labels")
    var eigen_solver = family_field(fields, "eigen_solver")
    var gamma = family_field(fields, "gamma")
    var degree = family_field(fields, "degree")
    var coef0 = family_field(fields, "coef0")
    var kernel_params = family_field(fields, "kernel_params")
    var n_jobs = family_field(fields, "n_jobs")
    var verbose = family_field(fields, "verbose")
    var prediction_data = family_field(fields, "prediction_data")
    var nk = cluster_keys()
    var affinities = py_item(context, nk[].k_family_cfb_sc_affinities)
    if not py_contains(affinities, affinity):
        raise Error("mojolearn SpectralClustering: affinity=" + _scalar_repr(affinity, context) + " is "
                    + "refused; it must be one of " + family_repr(py_call(b.list, affinities)) + ". A "
                    + "callable or another kernel name builds the affinity outside "
                    + "every identity column.")
    var assigns = py_item(context, nk[].k_family_cfb_assign_labels)
    if not py_contains(assigns, assign_labels):
        raise Error("mojolearn SpectralClustering: assign_labels="
                    + _scalar_repr(assign_labels, context) + " must be one of " + family_repr(py_call(b.list, assigns)))
    var predicting = Bool(py=family_call(b.bool, prediction_data))
    var operator = native_loaded_module("operator")
    if predicting and Bool(py=family_call(b.bool, family_call(operator.ne, assign_labels, PythonObject("kmeans")))):
        raise Error("NotImplementedError: mojolearn SpectralClustering: prediction_data with "
                    + "assign_labels=" + _scalar_repr(assign_labels, context) + " is refused; the Nystrom "
                    + "predict (DEVIATION 2860) assigns by the k-means centroids")
    if eigen_solver is not None:
        raise Error("NotImplementedError: mojolearn SpectralClustering: eigen_solver=" + _scalar_repr(eigen_solver, context) + " "
                    + "is refused; there is one solver here, RAFT's thick-restart "
                    + "Lanczos, and it is the one the identity profile pins")
    var rbf = Bool(py=family_call(b.bool, family_call(operator.eq, affinity, PythonObject("rbf"))))
    if gamma is not None and not rbf:
        raise Error("NotImplementedError: mojolearn SpectralClustering: gamma is refused; it "
                    + "parameterizes the RBF affinity, and affinity is not 'rbf'")
    if predicting and py_contains(Python.tuple("rbf", "precomputed_nearest_neighbors"), affinity):
        raise Error("NotImplementedError: mojolearn SpectralClustering: prediction_data with affinity="
                    + _scalar_repr(affinity, context) + " "
                    + "is refused; the Nystrom predict (DEVIATION 2860) carries the "
                    + "'nearest_neighbors' and 'precomputed' affinities only")
    var names: List[String] = ["degree", "coef0", "kernel_params"]
    var values: List[PythonObject] = [degree, coef0, kernel_params]
    for i in range(3):  # small-loop(parameters: degree, coef0, kernel_params): 3
        if values[i] is not None:
            raise Error("NotImplementedError: mojolearn SpectralClustering: " + names[i] + " is refused; it "
                        + "parameterizes an RBF or polynomial affinity, and "
                        + "affinity is restricted to 'nearest_neighbors' and "
                        + "'precomputed'")
    if n_jobs is not None:
        raise Error("NotImplementedError: mojolearn SpectralClustering: n_jobs is refused; the work "
                    + "is on the GPU and there is no host thread pool to size")
    if Bool(py=family_call(b.bool, verbose)):
        raise Error("NotImplementedError: mojolearn SpectralClustering: verbose is refused; nothing "
                    + "in this path prints. Set MOJOLEARN_IDENTITY_TRACE=<path> "
                    + "for the identity card instead")
    var default_tol = family_str(py_item(context, nk[].k_family_cfb_default_eigen_tol))
    if family_is_str(eigen_tol):
        raise Error("NotImplementedError: mojolearn SpectralClustering: eigen_tol=" + _scalar_repr(eigen_tol, context) + " is "
                    + "refused (DEVIATION 890). cuML 26.08's Python maps 'auto' to "
                    + "0.0, which is passed straight through as the eigensolver "
                    + "tolerance and disables the convergence test (the loop "
                    + "condition is res > tol), so the solve runs to "
                    + "max_iterations = 10 * n_samples and an exactly converged "
                    + "problem hits the restart-breakdown refusal. Pass a positive "
                    + "float; " + default_tol + " is cuVS's own struct default "
                    + "and is this class's default.")
    var tol = _float(eigen_tol)
    var t = Float64(py=tol)
    if not (t > 0.0) or t != t:
        raise Error("mojolearn SpectralClustering: eigen_tol must be positive, "
                    + "got " + family_repr(tol) + " (DEVIATION 890; see the 'auto' note)")
    var seed = _seed(random_state, "SpectralClustering", True, context)
    var clusters = _int(n_clusters)
    if clusters < 1:
        raise Error("mojolearn SpectralClustering: n_clusters must be at least 1")
    var inits = _int(n_init)
    if inits < 1:
        raise Error("mojolearn SpectralClustering: n_init must be at least 1")
    var neighbors = _int(n_neighbors)
    if neighbors < 1:
        raise Error("mojolearn SpectralClustering: n_neighbors must be at least 1")
    var components = clusters if n_components is None else _int(n_components)
    if n_components is not None and components < 1:
        raise Error("mojolearn SpectralClustering: n_components must be at least "
                    + "1, or None to follow n_clusters")
    py_setitem(fields, nk[].k__n_clusters, PythonObject(clusters))
    py_setitem(fields, nk[].k__n_neighbors, PythonObject(neighbors))
    py_setitem(fields, nk[].k__n_init, PythonObject(inits))
    py_setitem(fields, nk[].k__eigen_tol, tol)
    # scikit-learn's SpectralClustering default gamma is 1.0.
    var g = PythonObject(None)
    if rbf:
        g = PythonObject(1.0) if gamma is None else _float(gamma)
    py_setitem(fields, nk[].k__gamma, g)
    py_setitem(fields, nk[].k__seed, PythonObject(seed))
    return SpectralClusteringConfig(
        SpectralFitOptions(clusters, components, inits, neighbors, t, seed, predicting),
        String(py=affinity), spectral_assignment_method(String(py=assign_labels)),
        Float64(py=g) if rbf else 0.0)

def _sc_fit(model: PythonObject, routes: SpectralRoutePins, source: PythonObject, context: PythonObject) raises:
    """SpectralClustering fit over admitted source and retained provider pins."""
    var b = _builtins()
    var fields = py_fields(model)
    var metrics = routes.metrics
    var config = _sc_validate(fields, context)
    var prediction_data = family_field(fields, "prediction_data")
    if py_type(prediction_data) is not b.bool:
        raise Error("TypeError: mojolearn SpectralClustering: prediction_data must be a bool, "
                    + "got " + String(py_type(prediction_data).__name__))
    var predicting = config.fit.keep_state
    var dropped: List[String] = ["_pd_eigenvalues", "_pd_eigenvectors", "_pd_diag", "_pd_centroids", "_fit_X"]
    for i in range(5):  # small-loop(prediction data: the five cached attribute names): 5
        _ = py_call(fields.pop, PythonObject(dropped[i]), PythonObject(None))
    var clusters = config.fit.clusters
    var k = config.fit.components
    _ = py_call(fields.pop, PythonObject("affinity_matrix_"), PythonObject(None))
    var affinity = config.affinity.copy()
    var graph_affinity = affinity == "precomputed" or affinity == "rbf" or affinity == "precomputed_nearest_neighbors"
    var connection: Optional[SpectralConnection] = None
    var use_dense = False
    if graph_affinity:
        connection = SpectralConnection(metrics, context)
        use_dense = (connection.value().capabilities & SPECTRAL_CAP_DENSE) != 0
    # `dense` is the matrix the dense entry reads (an Array); `coo_source`
    # the input of the COO route.
    # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
    var dense = Optional[PythonObject]()
    var from_matrix = False
    var coo_source = source
    var nk = cluster_keys()
    var labels = PythonObject(None)
    var embedding = PythonObject(None)
    var state = SpectralStatePins()
    var n_out = 0
    var n = 0
    var produced = False
    if (affinity == "rbf" or affinity == "precomputed_nearest_neighbors") and use_dense and (
        connection.value().capabilities & SPECTRAL_CAP_AFFINITY
    ) != 0:
        # lane vdf-misc4: the affinity is built on the metrics image's own
        # device and the dense fit reads it there; the n x n crosses once,
        # into the public `affinity_matrix_`. Was: built by the x_decomp (rbf)
        # or base (knn) image into host storage, uploaded again by this image.
        if affinity == "rbf":
            var x = _x_decomp_input(source, context)
            n = Int(py=x.shape[0])
            var d = Int(py=x.shape[1])
            if clusters <= n and spectral_lanczos_fits(k, n):
                var public = family_array(Python.tuple(n, n), "<f4", context)
                py_setitem(fields, nk[].k_affinity_matrix_, public)
                py_setitem(fields, nk[].k_n_features_in_, PythonObject(d))
                labels = family_array(Python.tuple(n), "<i4", context)
                embedding = family_array(Python.tuple(n, k), "<f4", context)
                if predicting:
                    state.allocate(n, k, clusters, context)
                var request = SpectralFitRequest(
                    SpectralInput(Int(py=classical_address(x)), 0, 0, 0, SPECTRAL_RBF, n, d, 0,
                                  Int(py=classical_address(public)), 0, config.gamma),
                    Int(py=classical_address(labels)), Int(py=classical_address(embedding)), state.view(),
                    config.fit)
                n_out = spectral_fit_run(connection.value(), request)
                _ = x  # the admitted X stays alive through the native call above
                _ = public  # the published affinity stays alive through the native call above
                _ = state
                _ = labels
                _ = embedding
                _sc_refusal(n_out, "SpectralClustering")
                produced = True
        else:
            var admitted = _knn_admit(fields, source, True, context, True, config.fit.neighbors, False)
            n = admitted.n
            if n > 0 and clusters <= n and spectral_lanczos_fits(k, n):
                var status = SpectralAffinityStatus()
                var public = family_array(Python.tuple(n, n), "<f4", context)
                labels = family_array(Python.tuple(n), "<i4", context)
                embedding = family_array(Python.tuple(n, k), "<f4", context)
                if predicting:
                    state.allocate(n, k, clusters, context)
                var request = SpectralFitRequest(
                    SpectralInput(admitted.dense_address(), admitted.row_address(), admitted.column_address(),
                                  admitted.value_address(), SPECTRAL_KNN, n, admitted.k, admitted.nnz,
                                  Int(py=classical_address(public)), status.address(), 0.0),
                    Int(py=classical_address(labels)), Int(py=classical_address(embedding)), state.view(),
                    config.fit)
                n_out = spectral_fit_run(connection.value(), request)
                _ = admitted  # the admitted arrays stay alive through the native call above
                _ = status  # the status words stay alive through the native call above
                _ = state
                _ = labels
                _ = embedding
                if n_out == SPECTRAL_KNN_REFUSED:
                    _knn_refusal(status, admitted.k)
                py_setitem(fields, nk[].k_affinity_matrix_, public)
                _sc_refusal(n_out, "SpectralClustering")
                produced = True
        # Not produced: n is 0 or the shape refusal will fire; the former
        # route below raises in the former order.
    if produced:
        pass
    elif affinity == "precomputed":
        py_setitem(fields, nk[].k_affinity_matrix_, source)
        if use_dense and not _is_sparse(source):
            var shape = _shape_of(source)
            if not _square(shape):
                raise Error("mojolearn SpectralClustering: with affinity='precomputed', X "
                            + "must be a square affinity matrix or a SciPy sparse matrix, "
                            + "got shape " + family_repr(shape))
            dense = admit_family(source, 2, "X", context).array
    elif affinity == "rbf":
        var x = _x_decomp_input(source, context)
        var aff = _rbf_affinity(x, config.gamma, routes.decomp, context)
        py_setitem(fields, nk[].k_affinity_matrix_, aff.public_array)
        py_setitem(fields, nk[].k_n_features_in_, PythonObject(Int(py=x.shape[1])))
        dense = py_present(aff.public_array)
        from_matrix = True
    elif affinity == "precomputed_nearest_neighbors":
        var affinity_room = _knn_affinity(fields, source, True, routes.base, context, True, config.fit.neighbors)
        dense = py_present(affinity_room.public_array)
        from_matrix = True
    var options = config.fit
    if produced:
        pass
    elif use_dense and Bool(dense):
        # the dense matrix straight to the device entry
        n = Int(py=dense.value().shape[0])
        _sc_check_shape(clusters, n, k)
        labels = family_array(Python.tuple(n), "<i4", context)
        embedding = family_array(Python.tuple(n, k), "<f4", context)
        if predicting:
            state.allocate(n, k, clusters, context)
        var request = SpectralFitRequest(
            SpectralInput(Int(py=classical_address(dense.value())), 0, 0, 0, SPECTRAL_DENSE, n, 0, 0),
            Int(py=classical_address(labels)), Int(py=classical_address(embedding)), state.view(), options)
        n_out = spectral_fit_run(connection.value(), request)
        _ = state
        _ = embedding
        _ = labels
        _ = dense
        _sc_refusal(n_out, "SpectralClustering")
    elif graph_affinity:
        # the COO route: a sparse precomputed input, or a binding without
        # the dense entry (the CPU-only host binding)
        var coo: _Coo
        if from_matrix:
            coo = _dense_coo(dense.value(), "SpectralClustering", routes.base, context)
        else:
            coo = _coo_triples(coo_source, "SpectralClustering", routes.base, context)
        n = coo.n
        var nnz = coo.count()
        if nnz == 0:
            raise Error("mojolearn SpectralClustering: the precomputed affinity "
                        + "matrix has no nonzero entries")
        var vp = FAMILY_F32P(unsafe_from_address=coo.value_address())
        if not host_all_finite_f32(vp, nnz):
            raise Error("mojolearn SpectralClustering: the precomputed affinity "
                        + "matrix has a non-finite entry (the implemented path refuses "
                        + "it by name, because sqrt of a non-finite degree is a "
                        + "NaN and no NaN may reach a recorded value)")
        if family_min_f32(FAMILY_F32P(unsafe_from_address=coo.value_address()), nnz) < 0:
            raise Error("mojolearn SpectralClustering: the precomputed affinity "
                        + "matrix has a negative entry (refused by name: sqrt of a "
                        + "negative degree is a NaN in cuVS too)")
        _sc_check_shape(clusters, n, k)
        labels = family_array(Python.tuple(n), "<i4", context)
        embedding = family_array(Python.tuple(n, k), "<f4", context)
        if predicting:
            state.allocate(n, k, clusters, context)
        var request = SpectralFitRequest(
            SpectralInput(0, coo.row_address(), coo.column_address(),
                coo.value_address(), SPECTRAL_GRAPH, n, 0, nnz),
            Int(py=classical_address(labels)), Int(py=classical_address(embedding)), state.view(), options)
        n_out = spectral_fit_run(connection.value(), request)
        _ = state
        _ = labels
        _ = embedding
        _ = coo
    else:
        # Lane a8d: the admission's finite test (see `_x_decomp_input`).
        var x = admit_family(source, 2, "X", context, finite=True,
                             refusal="mojolearn SpectralClustering: X contains NaN or "
                             + "infinity (the implemented path refuses it by name before any "
                             + "recorded stage)").array
        py_setitem(fields, nk[].k_input_copied_, PythonObject(not (x is source)))
        n = Int(py=x.shape[0])
        var d = Int(py=x.shape[1])
        _sc_check_shape(clusters, n, k)
        var neighbors = config.fit.neighbors
        if neighbors > n:
            raise Error("mojolearn SpectralClustering: n_neighbors="
                        + String(neighbors) + " exceeds n_samples=" + String(n))
        labels = family_array(Python.tuple(n), "<i4", context)
        embedding = family_array(Python.tuple(n, k), "<f4", context)
        if predicting:
            state.allocate(n, k, clusters, context)
        connection = SpectralConnection(metrics, context)
        var request = SpectralFitRequest(
            SpectralInput(Int(py=classical_address(x)), 0, 0, 0, SPECTRAL_DATASET, n, d, 0),
            Int(py=classical_address(labels)), Int(py=classical_address(embedding)), state.view(), options)
        n_out = spectral_fit_run(connection.value(), request)
        _ = state
        _ = x
        _ = labels
        _ = embedding
        if predicting:
            # Preserve the fitted copy of a borrowed input for later queries.
            py_setitem(fields, nk[].k__fit_X, cfb_borrow_copy(x, source, context))
        py_setitem(fields, nk[].k_n_features_in_, PythonObject(d))
        _ = x^
    if n_out != k:
        raise Error("RuntimeError: mojolearn SpectralClustering: the kernel returned " + String(n_out) + " "
                    + "embedding columns where " + String(k) + " were expected")
    py_setitem(fields, nk[].k_embedding_, embedding)
    py_setitem(fields, nk[].k_n_components_, PythonObject(k))
    _ = py_call(fields.pop, PythonObject("n_iter_assign_"), PythonObject(None))
    if config.assignment >= 0:
        # Preserve ENTRY_SPECTRAL_ASSIGN and its 30/20 restart/iteration
        # settings. Only its options and result envelope become native.
        var request = SpectralAssignmentRequest(Int(py=classical_address(embedding)),
            Int(py=embedding.size), n, k, config.assignment,
            config.fit.seed, 30, 20)
        var result = spectral_assignment_run(routes.cluster, request, context)
        _ = embedding
        if config.assignment == 0:
            py_setitem(fields, nk[].k_n_iter_assign_, PythonObject(result.storage[0].iterations))
        labels = spectral_assignment_pack(result, n, context)
        _ = embedding
    # A fit releases the last fit's resident prediction state at once (N326).
    cres_forget(model, CRES_SPECTRAL)
    py_setitem(fields, nk[].k_labels_, labels)
    if predicting:
        py_setitem(fields, nk[].k__pd_eigenvalues, state.eigenvalues)
        py_setitem(fields, nk[].k__pd_eigenvectors, state.eigenvectors)
        py_setitem(fields, nk[].k__pd_diag, state.diagonal)
        py_setitem(fields, nk[].k__pd_centroids, state.centroids)


def _sc_resident(model: PythonObject, binding: PythonObject, state: SpectralStatePins,
                 training: PythonObject, shape: SpectralQueryShape,
                 mut connection: Optional[SpectralConnection],
                 context: PythonObject) raises -> Optional[SpectralResidentLease]:
    """Take a typed native lease while this generation/storage/provider matches."""
    if (not fitted_is_freezable(state.eigenvalues, context)
            or not fitted_is_freezable(state.eigenvectors, context)
            or not fitted_is_freezable(state.diagonal, context)
            or not fitted_is_freezable(state.centroids, context)):
        return None
    if shape.affinity == 0 and not fitted_is_freezable(training, context):
        return None
    var dims: List[Int] = [shape.training_rows, shape.features, shape.components, shape.clusters, shape.affinity]
    connection = SpectralConnection(binding, context)
    var held = cres_spectral_lookup(model, binding, connection.value().provider, state, training, dims)
    if held:
        return held^
    if (connection.value().capabilities & SPECTRAL_CAP_RESIDENT) == 0:
        return None
    # Freeze before the same native resident copies; writable or rebound public
    # buffers retain the former per-call/rebuild behavior.
    _ = fitted_freeze(state.eigenvalues, context)
    _ = fitted_freeze(state.eigenvectors, context)
    _ = fitted_freeze(state.diagonal, context)
    _ = fitted_freeze(state.centroids, context)
    var training_address = 0
    if shape.affinity == 0:
        _ = fitted_freeze(training, context)
        training_address = Int(py=classical_address(training))
    var request = SpectralPrepareRequest(training_address, state.view(), shape)
    var lease = spectral_resident_prepare(connection.value(), state, training, request)
    var nbytes = 4 * (shape.training_rows * shape.components + shape.training_rows + shape.components
        + shape.clusters * shape.components + (shape.training_rows * shape.features if shape.affinity == 0 else 0))
    cres_spectral_store(model, binding, dims^, lease.copy(), nbytes)
    return lease^


def _sc_predict(model: PythonObject, binding: PythonObject, source: PythonObject,
                context: PythonObject) raises -> SpectralPredictionPins:
    """`SpectralClustering._predict_embedding`: (labels, embedding). The Nystrom extension of the fitted
    eigenvectors, then the nearest fitted k-means centroid."""
    var b = _builtins()
    if not family_raw_has(model, "labels_"):
        raise Error("mojolearn SpectralClustering.predict: this instance is not "
                    + "fitted yet; call fit first")
    var eigenvalues = family_raw_get_or(model, "_pd_eigenvalues", PythonObject(None))
    if eigenvalues is None:
        raise Error("mojolearn SpectralClustering.predict: prediction data was not "
                    + "stored. Fit with SpectralClustering(prediction_data=True), which "
                    + "keeps the eigenpairs, degrees and centroids the Nystrom extension "
                    + "needs (DEVIATION 2860)")
    var affinity = family_raw_get(model, "affinity")
    var nk = cluster_keys()
    var affinities = py_item(context, nk[].k_family_cfb_predict_affinities)
    if not py_contains(affinities, affinity):
        raise Error("mojolearn SpectralClustering.predict: affinity=" + _scalar_repr(affinity, context) + " "
                    + "has no out-of-sample rule; it must be one of "
                    + family_repr(py_call(b.list, affinities)))
    var k = _int(family_raw_get(model, "n_components_"))
    var diag = family_raw_get(model, "_pd_diag")
    var n_train = Int(py=diag.shape[0])
    var q = PythonObject(None)
    var training = PythonObject(None)
    var train_addr = 0
    var n_features = 0
    var code = 0
    if String(py=affinity) == "precomputed":
        # P1 (audit A #4): an exact SciPy container is densified natively;
        # the caller's `toarray()` is never called.
        var a = _densified(source, context)
        var shape = _shape_of(a)
        if len(shape) != 2 or not Bool(py=family_call(b.bool, family_call(native_loaded_module("operator").eq,
                                                                           shape[1], PythonObject(n_train)))):
            raise Error("mojolearn SpectralClustering.predict: with affinity="
                        + "'precomputed', X must be the affinity between the new rows "
                        + "and the " + String(n_train) + " training rows, shape (n_new, " + String(n_train) + "); "
                        + "got shape " + family_repr(shape))
        # Lane a8d: the admission's finite test (see `_x_decomp_input`).
        q = admit_family(a, 2, "X", context, finite=True,
                         refusal="mojolearn SpectralClustering.predict: the affinity has a "
                         + "non-finite entry; refused by name").array
        if Int(py=q.shape[0]) < 1:
            raise Error("mojolearn SpectralClustering.predict: X has no rows; refused by name")
        if family_min(q) < 0:
            raise Error("mojolearn SpectralClustering.predict: the affinity has a "
                        + "negative entry; refused by name, as the fit refuses one")
        code = 1
    else:
        var fitted = Int(py=family_raw_get(model, "n_features_in_"))
        q = _reference_queries(source, fitted, "SpectralClustering.predict", context)
        training = family_raw_get(model, "_fit_X")
        train_addr = Int(py=classical_address(training))
        n_features = fitted
    var nq = Int(py=q.shape[0])
    var labels = family_array(Python.tuple(nq), "<i4", context)
    var emb = family_array(Python.tuple(nq, k), "<f4", context)
    var n_clusters = _int(family_raw_get(model, "_n_clusters"))
    var n_neighbors = _int(family_raw_get(model, "_n_neighbors"))
    var shape = SpectralQueryShape(n_train, nq, n_features, k, n_clusters, n_neighbors, code)
    var state = SpectralStatePins()
    state.eigenvalues = eigenvalues
    state.eigenvectors = family_raw_get(model, "_pd_eigenvectors")
    state.diagonal = diag
    state.centroids = family_raw_get(model, "_pd_centroids")
    var connection: Optional[SpectralConnection] = None
    var held = _sc_resident(model, binding, state, training, shape, connection, context)
    if held:
        # The resident state (N326): only the queries cross. The lease owns
        # the handle until the synchronous query finishes, even after eviction.
        try:
            var request = SpectralResidentQueryRequest(held.value().native[].handle,
                Int(py=classical_address(q)), Int(py=classical_address(labels)),
                Int(py=classical_address(emb)), shape)
            spectral_resident_query(held.value(), request)
        finally:
            spectral_resident_retire(held.take())
            _ = q
            _ = emb
            _ = labels
        return SpectralPredictionPins(labels, emb)
    if not connection:
        connection = SpectralConnection(binding, context)
    var request = SpectralQueryRequest(Int(py=classical_address(q)), train_addr, state.view(),
        Int(py=classical_address(labels)), Int(py=classical_address(emb)), shape)
    spectral_query_run(connection.value(), request)
    _ = q
    _ = state
    _ = training
    _ = emb
    _ = labels
    return SpectralPredictionPins(labels, emb)


@fieldwise_init
struct _EmbeddingOptionPins(Movable):
    var components: PythonObject
    var affinity: PythonObject
    var seed: PythonObject
    var neighbors: PythonObject
    var gamma: PythonObject
    var solver: PythonObject
    var tolerance: PythonObject


def _se_validate_options(options: _EmbeddingOptionPins, context: PythonObject) raises -> SpectralEmbeddingConfig:
    """`SpectralEmbedding._validate_params`."""
    var b = _builtins()
    var n_components = options.components
    var affinity = options.affinity
    var random_state = options.seed
    var n_neighbors = options.neighbors
    var gamma = options.gamma
    var eigen_solver = options.solver
    var eigen_tol = options.tolerance
    var nk = cluster_keys()
    var affinities = py_item(context, nk[].k_family_cfb_se_affinities)
    if not py_contains(affinities, affinity):
        raise Error("mojolearn SpectralEmbedding: affinity=" + _scalar_repr(affinity, context) + " is "
                    + "refused; it must be one of " + family_repr(py_call(b.list, affinities)) + ". "
                    + "A callable affinity is Python code that no identity column "
                    + "can run.")
    var rbf = Bool(py=family_call(b.bool, family_call(native_loaded_module("operator").eq, affinity, PythonObject("rbf"))))
    if gamma is not None and not rbf:
        raise Error("NotImplementedError: mojolearn SpectralEmbedding: gamma is refused; it "
                    + "parameterizes an RBF affinity, and affinity is not 'rbf'")
    var gamma_value = Float64(1.0)
    if gamma is not None:
        gamma_value = Float64(py=_float(gamma))
        if not (gamma_value > 0):
            raise Error("mojolearn SpectralEmbedding: gamma must be positive")
    if eigen_solver is not None:
        raise Error("NotImplementedError: mojolearn SpectralEmbedding: eigen_solver=" + _scalar_repr(eigen_solver, context) + " "
                    + "is refused; there is one solver here, RAFT's thick-restart "
                    + "Lanczos. scikit-learn's 'arpack', 'lobpcg' and 'amg' are "
                    + "other algorithms, and running the Lanczos under their names "
                    + "would be a silent substitution (spectral/NOT_IMPLEMENTED.tsv)")
    var tolerance = Float64(Float32(1e-5))
    if family_is_str(eigen_tol):
        if String(py=eigen_tol) != "auto":
            raise Error("mojolearn SpectralEmbedding: eigen_tol=" + _scalar_repr(eigen_tol, context) + " "
                        + "must be 'auto' or a positive float")
    elif eigen_tol is not None:
        var tol = Float64(py=_float(eigen_tol))
        tolerance = tol
        if not (tol > 0.0) or tol != tol or isinf(tol):
            raise Error("mojolearn SpectralEmbedding: eigen_tol must be a positive "
                        + "finite float or 'auto', got " + _scalar_repr(eigen_tol, context) + " (0 disables the "
                        + "Lanczos convergence test, DEVIATION 890)")
    var components = _int(n_components)
    if components < 1:
        raise Error("mojolearn SpectralEmbedding: n_components must be at least 1")
    var neighbors = 0 if n_neighbors is None else _int(n_neighbors)
    if n_neighbors is not None and neighbors < 1:
        raise Error("mojolearn SpectralEmbedding: n_neighbors must be at least 1, "
                    + "or None for max(n_samples // 10, 1)")
    var seed = _seed(random_state, "SpectralEmbedding", False, context)
    return SpectralEmbeddingConfig(String(py=affinity), components, neighbors, seed,
        gamma_value, tolerance, n_neighbors is None, gamma is None)


def _se_validate(fields: PythonObject, context: PythonObject) raises -> SpectralEmbeddingConfig:
    var options = _EmbeddingOptionPins(family_field(fields, "n_components"),
        family_field(fields, "affinity"), family_field(fields, "random_state"),
        family_field(fields, "n_neighbors"), family_field(fields, "gamma"),
        family_field(fields, "eigen_solver"), family_field(fields, "eigen_tol"))
    var config = _se_validate_options(options, context)
    # Existing observable/private-name fields remain the archive/attribute
    # boundary. This fit consumes config directly, with no scalar readback.
    var nk = cluster_keys()
    py_setitem(fields, nk[].k__n_components, PythonObject(config.components))
    py_setitem(fields, nk[].k__n_neighbors, PythonObject(None) if config.automatic_neighbors else PythonObject(config.neighbors))
    py_setitem(fields, nk[].k__seed, PythonObject(config.seed))
    py_setitem(fields, nk[].k__gamma, PythonObject(None) if config.automatic_gamma else PythonObject(config.gamma))
    return config^


def _se_fit(model: PythonObject, routes: SpectralRoutePins, source: PythonObject, context: PythonObject) raises:
    """SpectralEmbedding fit over admitted source and provider pins."""
    var b = _builtins()
    var fields = py_fields(model)
    var config = _se_validate(fields, context)
    var drop_first = Bool(py=family_call(b.bool, family_raw_get(model, "_drop_first")))
    var norm_laplacian = Bool(py=family_call(b.bool, family_raw_get(model, "_norm_laplacian")))
    _ = _se_execute(config, drop_first, norm_laplacian, fields, routes, source, context, False)


def _se_execute(config: SpectralEmbeddingConfig, drop_first: Bool, norm_laplacian: Bool,
                 fields: PythonObject, routes: SpectralRoutePins, source: PythonObject,
                 context: PythonObject, admit_source: Bool) raises -> PythonObject:
    var metrics = routes.metrics
    var k = config.components
    # The one piece of arithmetic the reference keeps in Python
    # (spectral_embedding.pyx:294): the transform receives this number.
    var n_lanczos = k + 1 if drop_first else k
    var affinity = config.affinity.copy()
    var graph_affinity = affinity == "precomputed" or affinity == "rbf" or affinity == "precomputed_nearest_neighbors"
    var connection: Optional[SpectralConnection] = None
    var use_dense = False
    if graph_affinity:
        connection = SpectralConnection(metrics, context)
        use_dense = (connection.value().capabilities & SPECTRAL_CAP_DENSE) != 0
    if admit_source:
        # The standalone function does not pass through spectral_input. Admit
        # its supported native input before any shape/property operation,
        # after the same option/route checks. Exact sparse inputs stay allowed.
        classical_admit_value(source, context)
    if (affinity == "rbf" or affinity == "precomputed_nearest_neighbors") and use_dense and (
        connection.value().capabilities & SPECTRAL_CAP_AFFINITY
    ) != 0:
        # lane vdf-misc3: the affinity is built on the metrics image's own
        # device and the embedding reads it there (no n x n through the host).
        var produced = _se_device_affinity(config, drop_first, norm_laplacian, n_lanczos, fields,
                                           connection.value(), source, context)
        if produced:
            return produced.value()
    # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
    var dense = Optional[PythonObject]()
    var generated: Optional[SpectralAffinityRoom] = None
    var from_matrix = False
    if affinity == "precomputed_nearest_neighbors":
        generated = _knn_affinity(fields, source, False, routes.base, context, True,
            config.neighbors, config.automatic_neighbors, fields is not None)
        from_matrix = True
    if affinity == "precomputed" and use_dense and not _is_sparse(source):
        var shape = _shape_of(source)
        if not _square(shape):
            raise Error("mojolearn SpectralEmbedding: with affinity='precomputed', X "
                        + "must be a square affinity matrix or a SciPy sparse matrix, "
                        + "got shape " + family_repr(shape))
        dense = admit_family(source, 2, "X", context).array
    var nk = cluster_keys()
    if affinity == "rbf":
        var x = _x_decomp_input(source, context)
        var d = Int(py=x.shape[1])
        var gamma = config.gamma_for(d)
        if fields is not None:
            py_setitem(fields, nk[].k_gamma_, PythonObject(gamma))
        generated = _rbf_affinity(x, gamma, routes.decomp, context, fields is not None)
        if fields is not None:
            py_setitem(fields, nk[].k_affinity_matrix_, generated.value().public_array)
        if fields is not None:
            py_setitem(fields, nk[].k_n_features_in_, PythonObject(d))
        from_matrix = True
    var tolerance = config.tolerance
    var seed = config.seed
    var embedding = PythonObject(None)
    var n_out: Int
    if use_dense and (Bool(dense) or generated):
        var n = generated.value().rows if generated else Int(py=dense.value().shape[0])
        _se_check_shape(k, n, n_lanczos)
        if fields is not None:
            py_setitem(fields, nk[].k_n_neighbors_, PythonObject(config.neighbors_for(n)))
        embedding = family_array(Python.tuple(n, k), "<f4", context)
        var address = generated.value().address() if generated else Int(py=classical_address(dense.value()))
        var request = SpectralEmbeddingRequest(
            SpectralInput(address, 0, 0, 0, SPECTRAL_DENSE, n, 0, 0),
            Int(py=classical_address(embedding)), SpectralEmbeddingOptions(n_lanczos, k, 0,
                norm_laplacian, drop_first, seed, tolerance))
        n_out = spectral_embedding_run(connection.value(), request)
        _ = embedding
        _ = dense
        _ = generated
        _sc_refusal(n_out, "SpectralEmbedding")
    elif graph_affinity:
        var coo: _Coo
        if from_matrix:
            coo = _dense_coo_span(generated.value().address(), generated.value().rows,
                generated.value().rows, "SpectralEmbedding", routes.base, context)
            _ = generated
        else:
            coo = _coo_triples(source, "SpectralEmbedding", routes.base, context)
        var n = coo.n
        var nnz = coo.count()
        if nnz == 0:
            raise Error("mojolearn SpectralEmbedding: the precomputed affinity "
                        + "matrix has no nonzero entries")
        # the rbf affinity is exp of a finite value, finite and >= 0 by
        # construction (`exp_c` clamps), so its two scans are skipped
        if affinity == "precomputed":
            if not host_all_finite_f32(FAMILY_F32P(unsafe_from_address=coo.value_address()), nnz):
                raise Error("mojolearn SpectralEmbedding: the precomputed affinity "
                            + "matrix has a non-finite entry")
            if family_min_f32(FAMILY_F32P(unsafe_from_address=coo.value_address()), nnz) < 0:
                raise Error("mojolearn SpectralEmbedding: the precomputed affinity "
                            + "matrix has a negative entry (refused by name: sqrt of a "
                            + "negative degree is a NaN)")
        _se_check_shape(k, n, n_lanczos)
        if fields is not None:
            py_setitem(fields, nk[].k_n_neighbors_, PythonObject(config.neighbors_for(n)))
        embedding = family_array(Python.tuple(n, k), "<f4", context)
        var request = SpectralEmbeddingRequest(
            SpectralInput(0, coo.row_address(), coo.column_address(),
                coo.value_address(), SPECTRAL_GRAPH, n, 0, nnz),
            Int(py=classical_address(embedding)), SpectralEmbeddingOptions(n_lanczos, k, 0,
                norm_laplacian, drop_first, seed, tolerance))
        n_out = spectral_embedding_run(connection.value(), request)
        _ = embedding
        _ = coo
    else:
        # Lane a8d: the admission's finite test (see `_x_decomp_input`).
        var x = admit_family(source, 2, "X", context, finite=True,
                             refusal="mojolearn SpectralEmbedding: X contains NaN or infinity").array
        if fields is not None:
            py_setitem(fields, nk[].k_input_copied_, PythonObject(not (x is source)))
        var n = Int(py=x.shape[0])
        var d = Int(py=x.shape[1])
        _se_check_shape(k, n, n_lanczos)
        var neighbors = config.neighbors_for(n)
        if neighbors > n:
            raise Error("mojolearn SpectralEmbedding: n_neighbors=" + String(neighbors) + " "
                        + "exceeds n_samples=" + String(n))
        if fields is not None:
            py_setitem(fields, nk[].k_n_neighbors_, PythonObject(neighbors))
        embedding = family_array(Python.tuple(n, k), "<f4", context)
        connection = SpectralConnection(metrics, context)
        # Lane gd-resident-2: inside `resident()` the GPU image reads X from
        # its resident store in place (SPECTRAL_DATASET_STORE, the same
        # kernels on the same words): kept once per scope, a hit after.
        var store_slot = _se_scope_slot(connection.value(), metrics, source, x, n * d, context)
        var x_input = SpectralInput(Int(py=classical_address(x)), 0, 0, 0, SPECTRAL_DATASET, n, d, 0)
        if store_slot >= 0:
            x_input = SpectralInput(store_slot, 0, 0, 0, SPECTRAL_DATASET_STORE, n, d, 0)
        var request = SpectralEmbeddingRequest(
            x_input,
            Int(py=classical_address(embedding)), SpectralEmbeddingOptions(n_lanczos, k, neighbors,
                norm_laplacian, drop_first, seed, tolerance))
        n_out = spectral_embedding_run(connection.value(), request)
        _ = x
        _ = embedding
        if fields is not None:
            py_setitem(fields, nk[].k_n_features_in_, PythonObject(d))
        _ = x^
    if n_out != k:
        raise Error("RuntimeError: mojolearn SpectralEmbedding: the kernel returned " + String(n_out) + " "
                    + "embedding columns where " + String(k) + " were expected")
    if fields is not None:
        py_setitem(fields, nk[].k_embedding_, embedding)

    return embedding

def _se_scope_slot(connection: SpectralConnection, metrics: PythonObject, source: PythonObject,
                   x: PythonObject, words: Int, context: PythonObject) raises -> Int:
    """The metrics image's resident-store slot of the admitted X (`words`
    float32 words) inside the calling thread's `resident()` scope (lane
    gd-resident-2, 2026-11-25), or -1: outside a scope (one atomic load), an
    image without SPECTRAL_CAP_DEV_STORE (a host image), no published store,
    or an X the admission converted (a per-call copy is never kept). The
    slot belongs to the scope's cache of the metrics family; nothing to
    free. Was: the dataset route copied X into a host list and uploaded it
    on every call, inside a scope as outside."""
    var scope = FunctionScope(context)
    if not scope.active() or (connection.capabilities & SPECTRAL_CAP_DEV_STORE) == 0:
        return -1
    var store = metrics_store_find(connection.driver.metadata(SPECTRAL_MODE),
                                   device_vendor_code(String(py_call(metrics.metrics_vendor))))
    if not store:
        return -1
    return scope.put(store.value(), String(METRICS_STORE_PREFIX), source, x, Int(py=classical_address(x)), words)


def _se_device_affinity(config: SpectralEmbeddingConfig, drop_first: Bool, norm_laplacian: Bool,
                        n_lanczos: Int, fields: PythonObject, connection: SpectralConnection,
                        source: PythonObject, context: PythonObject) raises -> Optional[PythonObject]:
    """affinity 'rbf' / 'precomputed_nearest_neighbors' on an image with
    SPECTRAL_CAP_AFFINITY (lane vdf-misc3, 2026-11-24): the admitted X (rbf)
    or precomputed distances (knn) go to the metrics image once; it builds the
    n x n affinity on its device (rbf: the x_decomp cells; knn: the base
    image's kernels), copies it once into the estimator's public
    `affinity_matrix_` when there is one (the standalone function gets none)
    and runs the dense embedding on the same device buffer. Was: the affinity
    was built by the x_decomp (rbf) or base (knn) image, downloaded into host
    storage and uploaded again by this image. Returns None, having published
    nothing, when the shape refusal will fire (or n is 0): the caller then
    takes the former route, which raises in the former order. The decline
    tests the solver's own rule (`spectral_lanczos_fits`), so no shape the
    class refuses reaches the kernel."""
    var k = config.components
    var nk = cluster_keys()
    var options = SpectralEmbeddingOptions(n_lanczos, k, 0, norm_laplacian, drop_first,
                                           config.seed, config.tolerance)
    var embedding: PythonObject
    var n_out: Int
    if config.affinity == "rbf":
        var x = _x_decomp_input(source, context)
        var n = Int(py=x.shape[0])
        var d = Int(py=x.shape[1])
        if not spectral_lanczos_fits(n_lanczos, n):
            return None
        var gamma = config.gamma_for(d)
        var public = PythonObject(None)
        var out_address = 0
        if fields is not None:
            py_setitem(fields, nk[].k_gamma_, PythonObject(gamma))
            public = family_array(Python.tuple(n, n), "<f4", context)
            out_address = Int(py=classical_address(public))
            py_setitem(fields, nk[].k_affinity_matrix_, public)
            py_setitem(fields, nk[].k_n_features_in_, PythonObject(d))
            py_setitem(fields, nk[].k_n_neighbors_, PythonObject(config.neighbors_for(n)))
        embedding = family_array(Python.tuple(n, k), "<f4", context)
        var request = SpectralEmbeddingRequest(
            SpectralInput(Int(py=classical_address(x)), 0, 0, 0, SPECTRAL_RBF, n, d, 0, out_address, 0, gamma),
            Int(py=classical_address(embedding)), options)
        n_out = spectral_embedding_run(connection, request)
        _ = x
        _ = public
        _sc_refusal(n_out, "SpectralEmbedding")
    else:
        var admitted = _knn_admit(fields, source, False, context, True, config.neighbors,
                                  config.automatic_neighbors)
        var n = admitted.n
        if n == 0 or not spectral_lanczos_fits(n_lanczos, n):
            return None
        var status = SpectralAffinityStatus()
        var public = PythonObject(None)
        var out_address = 0
        if fields is not None:
            public = family_array(Python.tuple(n, n), "<f4", context)
            out_address = Int(py=classical_address(public))
        embedding = family_array(Python.tuple(n, k), "<f4", context)
        var request = SpectralEmbeddingRequest(
            SpectralInput(admitted.dense_address(), admitted.row_address(), admitted.column_address(),
                admitted.value_address(), SPECTRAL_KNN, n, admitted.k, admitted.nnz,
                out_address, status.address(), 0.0),
            Int(py=classical_address(embedding)), options)
        n_out = spectral_embedding_run(connection, request)
        _ = admitted  # the admitted arrays stay alive through the native call above
        _ = status  # the status words stay alive through the native call above
        if n_out == SPECTRAL_KNN_REFUSED:
            _knn_refusal(status, admitted.k)
        if fields is not None:
            py_setitem(fields, nk[].k_affinity_matrix_, public)
            py_setitem(fields, nk[].k_n_neighbors_, PythonObject(config.neighbors_for(n)))
        _ = public
        _sc_refusal(n_out, "SpectralEmbedding")
    if n_out != k:
        raise Error("RuntimeError: mojolearn SpectralEmbedding: the kernel returned " + String(n_out) + " "
                    + "embedding columns where " + String(k) + " were expected")
    if fields is not None:
        py_setitem(fields, nk[].k_embedding_, embedding)
    return embedding


def _spectral_embedding(args: PythonObject, routes: SpectralRoutePins, context: PythonObject) raises -> PythonObject:
    """Unpack the public spectral_embedding arguments; retain route pins separately."""
    var b = _builtins()
    var names: List[String] = ["norm_laplacian", "drop_first"]
    var nk = cluster_keys()
    for i in range(2):  # small-loop(flags: norm_laplacian, drop_first): 2
        var value = args[6 + i]
        if py_type(value) is not b.bool:
            var type_name = py_call(py_item(b.type.__dict__, nk[].k___name__).__get__, py_type(value))
            raise Error("TypeError: mojolearn spectral_embedding: " + names[i] + " must be a bool, got "
                        + String(type_name))
    # Public convenience arguments cross the same exact scalar admission in
    # constructor-schema order. No temporary estimator/dictionary is needed.
    host_inference_scalar(args[2], context, True)
    host_inference_scalar(args[3], context, True)
    host_inference_scalar(args[4], context, True)
    host_inference_scalar(args[5], context, True)
    var options = _EmbeddingOptionPins(args[2], args[3], args[4], args[5],
        PythonObject(None), PythonObject(None), PythonObject(None))
    var config = _se_validate_options(options, context)
    return _se_execute(config, Bool(py=args[7]), Bool(py=args[6]), PythonObject(None),
        routes, args[1], context, True)


def _cfb_spectral_algorithm_operation(model: PythonObject, operation: Int, args: PythonObject,
                           context: PythonObject, supplied_binding: PythonObject, source: PythonObject) raises -> PythonObject:
    """Fit/predict shells pass (X,); route selection belongs to this owner."""
    var nk = cluster_keys()
    if operation == CFB_OPN_FAMILY_CFB_SC_FIT or operation == CFB_OPN_FAMILY_CFB_SC_FIT_PREDICT:
        var routed = _routes(model, True, context)
        _sc_fit(model, routed, source, context)
        if operation == CFB_OPN_FAMILY_CFB_SC_FIT:
            return model
        return py_item(py_fields(model), nk[].k_labels_)
    if operation == CFB_OPN_FAMILY_CFB_SC_PREDICT:
        var mode = py_get(py_fields(model), nk[].k_numeric_mode)
        var binding = supplied_binding
        if binding is None:
            binding = host_estimator_binding(model, "_mojolearn_metrics", mode, context)
        if binding is None:
            binding = _routed(model, "metrics", "_mojolearn_metrics", mode, context)
        var result = _sc_predict(model, binding, source, context)
        return result.labels
    if operation == CFB_OPN_FAMILY_CFB_SC_PREDICT_EMBEDDING:
        var mode = py_get(py_fields(model), nk[].k_numeric_mode)
        var binding = supplied_binding
        if binding is None:
            binding = host_estimator_binding(model, "_mojolearn_metrics", mode, context)
        if binding is None:
            binding = _routed(model, "metrics", "_mojolearn_metrics", mode, context)
        var result = _sc_predict(model, binding, source, context)
        return Python.tuple(result.labels, result.embedding)
    if operation == CFB_OPN_FAMILY_CFB_SE_VALIDATE:
        _ = _se_validate(py_fields(model), context)
        return PythonObject(None)
    if operation == CFB_OPN_FAMILY_CFB_SE_KNN_AFFINITY:
        var base = _routed(model, "base", "_mojolearn", context["family_cfb_identical"], context)
        var result = _knn_affinity(py_fields(model), source, False, base, context)
        return result.public_array
    if operation == CFB_OPN_FAMILY_CFB_COO_TRIPLES:
        var base = spectral_umap_route(context, "base", PythonObject("identical"))
        var coo = _coo_triples(args[0], String(py=args[1]), base, context, True)
        return Python.tuple(coo.rows, coo.cols, coo.vals, PythonObject(coo.n))
    if operation == CFB_OPN_FAMILY_CFB_SE_FIT or operation == CFB_OPN_FAMILY_CFB_SE_FIT_TRANSFORM:
        var routed = _routes(model, False, context)
        _se_fit(model, routed, source, context)
        if operation == CFB_OPN_FAMILY_CFB_SE_FIT:
            return model
        return py_item(py_fields(model), nk[].k_embedding_)
    if operation == CFB_OPN_FAMILY_CFB_SPECTRAL_EMBEDDING:
        # args[8] is the call's `device` (lane vdf-misc), admitted first in the
        # shared words. The function has the device images and the CPU
        # identical images of its three groups connected beside a GPU set
        # (`_mojolearn_metrics`, `_mojolearn_x_decomp`, `_mojolearn`:
        # python/mojolearn/_spectral_impl.py `connect_bindings`), so "cpu"
        # beside a GPU set selects those (estimator_binding_device); every
        # other admitted value runs today's process connection.
        var device = function_device_default()
        if args[8] is not None:
            device = function_device_admit(args[8], "spectral_embedding", True, True, context, _builtins())
        var routed: SpectralRoutePins
        if device.host_beside:
            routed = SpectralRoutePins(
                estimator_binding_device(device, PythonObject(None), "_mojolearn_metrics", context, False),
                estimator_binding_device(device, PythonObject(None), "_mojolearn_x_decomp", context, False),
                estimator_binding_device(device, PythonObject("identical"), "_mojolearn", context, False),
                PythonObject(None))
        else:
            routed = SpectralRoutePins(
                spectral_umap_route(context, "metrics", PythonObject(None)),
                spectral_umap_route(context, "decomp", PythonObject(None)),
                spectral_umap_route(context, "base", PythonObject("identical")), PythonObject(None))
        return _spectral_embedding(args, routed, context)
    raise Error("RuntimeError: unknown spectral operation code " + String(operation))


def cfb_spectral_vendor_used(model: PythonObject, context: PythonObject) raises -> PythonObject:
    """`vendor_used()` of SpectralClustering and SpectralEmbedding: the
    vendor word of the x_decomp image the estimator's binding route names
    (`cpu` for a `device="cpu"` estimator beside a GPU set), the route every
    fit of these classes takes (`_routes`). Metadata, no input, in no loop."""
    var decomp = _routed(model, "decomp", "_mojolearn_x_decomp", PythonObject(None), context)
    return py_call(decomp.x_decomp_vendor)


def cfb_spectral_operation(model: PythonObject, operation: Int, args: PythonObject,
                           context: PythonObject) raises -> PythonObject:
    # `operation` is the leaf code of bindings/operation_names_cfb.mojo
    # (spectral block; lane bw2-cluster: it was the name, up to twelve String
    # compares here and eleven in the algorithm owner, plus a method String
    # built per call from two suffix tests).
    classical_connect_numpy(context)
    var b = _builtins()
    if py_type(args) is not b.tuple:
        raise Error("TypeError: spectral public arguments require an exact tuple")
    var supplied_binding = PythonObject(None)
    var source_index = 0
    if operation == CFB_OPN_FAMILY_CFB_SC_PREDICT and len(args) == 2:
        # The archive recipe already carries its admitted CPU owner.
        supplied_binding = args[0]
        source_index = 1
    var clustering = operation >= CFB_OPN_FAMILY_CFB_SC_FIT and operation <= CFB_OPN_FAMILY_CFB_SC_PREDICT_EMBEDDING
    if model is None:
        return _cfb_spectral_algorithm_operation(model, operation, args, context, supplied_binding, PythonObject(None))
    var fitting = (operation == CFB_OPN_FAMILY_CFB_SC_FIT or operation == CFB_OPN_FAMILY_CFB_SC_FIT_PREDICT
                   or operation == CFB_OPN_FAMILY_CFB_SE_FIT or operation == CFB_OPN_FAMILY_CFB_SE_FIT_TRANSFORM)
    var fields = spectral_admit(model, clustering, fitting, context)
    if operation == CFB_OPN_FAMILY_CFB_SE_VALIDATE:
        for name in ["_n_components", "_n_neighbors", "_seed", "_gamma"]:  # small-loop(private validation output fields)
            if py_contains(fields, name):
                native_release_admit(fields[name], context)
        return _cfb_spectral_algorithm_operation(model, operation, args, context, supplied_binding, PythonObject(None))
    var public_prediction = operation == CFB_OPN_FAMILY_CFB_SC_PREDICT
    if public_prediction:
        # The not-fitted text's method: the only operation here is predict
        # (the old per-call `method` String gave "predict" for it).
        classical_require_fitted(model, False, "predict", context)
    var nk = cluster_keys()
    if operation == CFB_OPN_FAMILY_CFB_SE_KNN_AFFINITY and py_contains(fields, nk[].k_affinity_matrix_):
        native_release_admit(py_item(fields, nk[].k_affinity_matrix_), context)
    var source = args[source_index]
    var name = classical_estimator_name(model)
    var prepared = spectral_input(source, name, context)
    var rows = Int(py=prepared.shape[0])
    var columns = Int(py=prepared.shape[1])
    var depth = Int(py=py_get(fields, nk[].k__mojolearn_fitting, PythonObject(0)))
    if depth == 0:
        if fitting:
            _target(args[1], rows, name, context)
        elif public_prediction:
            var width = py_get(fields, nk[].k_n_features_in_)
            if py_type(width) is b.int:
                classical_check_width(columns, Int(py=width), name)
            estimator_feature_check(model, fields, source, context)
    var initial_depth = estimator_fit_begin(fields) if fitting else 0
    var outside = not public_prediction and estimator_outside()
    var result = PythonObject(None)
    try:
        result = _cfb_spectral_algorithm_operation(model, operation, args, context, supplied_binding, prepared)
    except error:
        var refusal = _refusal(error, fitting, depth > 0, rows, columns, name)
        raise refusal^
    finally:
        if fitting:
            estimator_fit_end(model, initial_depth)
    if fitting:
        if not py_contains(fields, nk[].k_n_features_in_):
            py_setitem(fields, nk[].k_n_features_in_, PythonObject(columns))
        estimator_fit_complete(model, source, False, context)
    # predict has one result for every caller (Andrew, 2026-11-22, lane
    # predict-dtype): the labels as a NumPy view, with no caller test.
    if (((outside and fitting) or public_prediction)
            and py_type(result) is py_item(context, nk[].k_array_class) and py_get(context, nk[].k_ndarray) is not None):
        return native_array_ndarray_view(result, context)
    return result
