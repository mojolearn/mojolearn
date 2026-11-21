# SPDX-License-Identifier: Apache-2.0
# UNPACK of SVR `fit`, `predict` and `score`
# (bindings/svc_kernel_native_door.mojo sequences the operations): the
# parameter checks and the settings record (svm_settings_unpack), the admitted X and targets with their owners, gamma and
# the per-row bounds, the provider connection, the resident lease of the
# fitted state, and the typed requests RUN reads (core/svr_native_api.mojo:
# SvrFitRequest, SvrQueryRequest). Every check and every field read of the
# old `_svr_fit` / `_svr_predict` is here, in the same order and words; RUN
# sees addresses and plain words only.
from std.python import Python, PythonObject
from bindings.admitted_family_door import admit_family, admit_family_copied
from bindings.py_balanced import py_call, py_fields
from bindings.family_common_pack import family_array
from bindings.svc_kernel_query_unpack import svc_support_columns
from bindings.classical_arrays_door import classical_address
from bindings.family_common_unpack import family_repr
from core.svr_native_api import SvrFitPlan, SvrFitRequest, SvrQueryRequest
from bindings.svr_native_boundary_unpack import SvrConnection, svr_connection, SvrResidentPin, svr_resident_pin
from bindings.svc_common_unpack import svc_field, svc_has, svc_float, svc_int, svc_read_f32, svc_shape_of
from bindings.device_rows_door_run import DoorRowsStore
from providers.python.array_metadata_pyapi_door import ArrayMetadata, ArrayPin
from providers.python.neural_descriptor_pyapi_door import neural_descriptor_entry
from bindings.svm_keys_unpack import svm_keys
from bindings.svc_kernel_rows_unpack import _SVR_RESIDENT_SLOT, _SvmQuery, _addr, svm_device_ok, _svm_fit_admit, _svm_fit_entry, _svm_fit_host_rows, _svm_query, _svm_query_refuse, _svm_x_scan
from bindings.svc_kernel_params_unpack import _c_rows, _kernel_code, _resolve_gamma, svm_settings_unpack


# ---------------------------------------------------------------- fit


struct SvrFitUnpacked(Movable):
    """A fit: the typed request, the provider it runs on, the Mojo-owned
    worst-case result buffers (`plan`), and the Python owners of every
    buffer the request addresses (X through its pin, the admitted targets,
    the per-row bounds), held until PACK. `device_form`: the entry took X on
    the device and returned its finite and range words and the gamma it
    solved with. `copied` is the `input_copied_` flag PACK publishes."""
    var fields: PythonObject
    var x_pin: ArrayPin
    var x_meta: ArrayMetadata
    var targets: PythonObject
    var c_rows: PythonObject
    var connection: SvrConnection
    var plan: SvrFitPlan
    var request: SvrFitRequest
    var fold: DoorRowsStore
    var device_form: Bool
    var copied: Bool
    var gamma: Float64

    def __init__(out self, fields: PythonObject, var x_pin: ArrayPin, var x_meta: ArrayMetadata,
                 targets: PythonObject, c_rows: PythonObject, var connection: SvrConnection,
                 var plan: SvrFitPlan, request: SvrFitRequest, fold: DoorRowsStore, device_form: Bool,
                 copied: Bool, gamma: Float64):
        self.fields = fields
        self.x_pin = x_pin^
        self.x_meta = x_meta^
        self.targets = targets
        self.c_rows = c_rows
        self.connection = connection^
        self.plan = plan^
        self.request = request.copy()
        self.fold = fold
        self.device_form = device_form
        self.copied = copied
        self.gamma = gamma


def svr_fit_unpack(model: PythonObject, args: PythonObject, context: PythonObject,
                   native: PythonObject) raises -> SvrFitUnpacked:
    """`SVR.fit(X, y, sample_weight)`; args = (binding, X, y, sample_weight)."""
    var binding = args[0]
    var fields = py_fields(model)
    var hp = svm_settings_unpack(model, fields, False, context)
    # Lane a8a: the GPU binding's fit takes X on the device (a fold gathered
    # there, host words copied once; never a host List) and tests it there;
    # every other route tests the host words here. One wording.
    var entry = _svm_fit_entry(binding, native)
    var fold = DoorRowsStore.none()
    var x_pin = ArrayPin()
    var x_meta = _svm_fit_admit(args[1], "SVR", entry, context, fold, x_pin)
    var nk = svm_keys()
    var copied = True if fold.active() else admit_family_copied(x_meta, args[1])
    var n_rows = x_meta.shape[0]
    var n_cols = x_meta.shape[1]
    # `_as_targets`: y refused non-finite here, in scikit-learn's wording.
    var shape = svc_shape_of(args[2], context)
    if len(shape) != 1:
        raise Error("mojolearn SVR: y must be 1-D, got " + String(len(shape)) + "-D shape " + family_repr(shape))
    var targets = admit_family(args[2], 1, "y", context, False, True,
                               refusal="Input y contains NaN or infinity (SVR).").array
    if Int(py=targets.shape[0]) != n_rows:
        raise Error("mojolearn SVR: y has " + String(Int(py=targets.shape[0])) + " entries, X has " + String(n_rows) + " rows")
    var precomputed = hp.precomputed
    if precomputed and n_cols != n_rows:
        raise Error("mojolearn SVR: kernel='precomputed' needs the square n x n "
                    + "kernel matrix as X, got " + String(n_rows) + " x " + String(n_cols))
    var device_x = entry and not precomputed
    if fold.active() and not device_x:
        x_meta = _svm_fit_host_rows(args[1], context, x_pin)
        fold = DoorRowsStore.none()
    var gamma = 1.0
    var scale_gamma = False
    if device_x:
        var resolved = hp.device_gamma(n_cols)
        gamma = resolved[0]
        scale_gamma = resolved[1]
    else:
        _svm_x_scan(x_meta, "SVR")
        gamma = _resolve_gamma(hp, binding, x_meta, context, native)
    var c_rows = _c_rows(binding, hp.c, n_rows, args[3], PythonObject(None),
                         PythonObject(None), "SVR", context, native)
    var plan = SvrFitPlan(n_rows, n_cols)
    var bounds_address = 0 if c_rows is None else _addr(c_rows)
    var target_address = _addr(targets)
    var connection = svr_connection(binding)
    var device_form = device_x and native is None
    var request = SvrFitRequest(x_meta.source.addr, target_address,
        Int(plan.dual.unsafe_ptr()), Int(plan.support.unsafe_ptr()), Int(plan.vectors.unsafe_ptr()),
        Int(plan.info.unsafe_ptr()), bounds_address, n_rows, n_cols, hp.kernel,
        hp.max_iter, hp.nochange_steps, hp.degree,
        fold.base_id, fold.base_rows, fold.index_id, device_form,
        scale_gamma, native is not None, gamma, hp.c, hp.epsilon, hp.tol, hp.coef0)
    return SvrFitUnpacked(fields, x_pin^, x_meta^, targets, c_rows, connection^, plan^, request, fold,
                          device_form, copied, gamma)


# ---------------------------------------------------------------- predict


struct SvrQueryUnpacked(Movable):
    """A query: the typed request, the provider, the two status words the
    resident entry writes (`status`: X's finite and range words), and the
    Python owners of the buffers the request addresses: the admitted X
    (`q`), for kernel='precomputed' the int32 support indices and the buffer
    RUN gathers the cross-kernel's support columns into (`owners`;
    `gather_columns`, `gather_count`; the request's `x` is that buffer), the
    fitted dual coefficients and support vectors, and the
    resident lease of the fitted state. `resident`: the binding keeps the
    fitted state on the device and tests the query it uploads or gathers."""
    var q: _SvmQuery
    var owners: PythonObject
    var gather_columns: Int
    var gather_count: Int
    var dual: PythonObject
    var sv: PythonObject
    var connection: SvrConnection
    var lease: SvrResidentPin
    var status: List[Int32]
    var request: SvrQueryRequest
    var resident: Bool

    def __init__(out self, var q: _SvmQuery, owners: PythonObject, gather_columns: Int, gather_count: Int,
                 dual: PythonObject, sv: PythonObject,
                 var connection: SvrConnection, var lease: SvrResidentPin, var status: List[Int32],
                 request: SvrQueryRequest, resident: Bool):
        self.q = q^
        self.owners = owners
        self.gather_columns = gather_columns
        self.gather_count = gather_count
        self.dual = dual
        self.sv = sv
        self.connection = connection^
        self.lease = lease^
        self.status = status^
        self.request = request.copy()
        self.resident = resident


def svr_query_unpack(model: PythonObject, fields: PythonObject, binding: PythonObject, source: PythonObject,
                     context: PythonObject, candidate: Bool) raises -> SvrQueryUnpacked:
    """`SVR.predict`: the fitted test, X, the fitted state and the kernel
    parameters. The request's `dst` is 0: the door writes the address of the
    public result PACK allocates."""
    var nk = svm_keys()
    if not svc_has(fields, nk[].k_dual_coef_):
        raise Error("mojolearn SVR: call fit() first")
    # The B arm (the descriptor candidate) has no resident route.
    var resident = False
    if not candidate:
        resident = neural_descriptor_entry(binding, "svr_predict_resident", False) is not None
    var n_support = svc_int(svc_field(fields, nk[].k_n_support_, "n_support_"))
    var k = String(svc_field(fields, nk[].k__hp_kernel, "_hp_kernel"))
    var precomputed = k == "precomputed"
    var q_meta = _svm_query(source, "SVR", context,
                            svm_device_ok(resident, 4 if precomputed else 0, [n_support]))
    var d = svc_int(svc_field(fields, nk[].k_n_features_in_, "n_features_in_"))
    if q_meta.cols != d:
        raise Error("mojolearn SVR: X has " + String(q_meta.cols) + " features, fit saw " + String(d))
    var n_rows = q_meta.rows
    var dual = svc_field(fields, nk[].k_dual_coef_, "dual_coef_")
    var sv = svc_field(fields, nk[].k_support_vectors_, "support_vectors_")
    var n_features = d
    var owners = Python.list()
    var query_address = q_meta.address
    var gather_columns = 0
    var gather_count = 0
    if precomputed and n_support > 0:
        # the cross-kernel's columns at the support vectors (SVC's rule), gathered by RUN
        var columns = svc_support_columns(svc_field(fields, nk[].k_support_, "support_"), -1, owners, context)
        gather_columns = columns[0]
        gather_count = columns[1]
        var gathered = family_array(Python.tuple(PythonObject(n_rows), PythonObject(gather_count)), "<f4", context)
        _ = py_call(owners.append, gathered)
        query_address = _addr(gathered)
        n_features = n_support
        sv = dual
    var bias = svc_read_f32(svc_field(fields, nk[].k_intercept_, "intercept_"), 0)
    var kernel = _kernel_code(fields, context)
    var gamma = Float64(py=svc_field(fields, nk[].k__gamma, "_gamma"))
    var cache_mib = Float64(py=svc_field(fields, nk[].k__hp_cache_size, "_hp_cache_size"))
    var degree = 3
    var coef0 = Float64(0)
    if k == "poly" or k == "sigmoid":
        degree = svc_int(svc_field(fields, nk[].k__hp_degree, "_hp_degree"))
        coef0 = svc_float(svc_field(fields, nk[].k__hp_coef0, "_hp_coef0"))
    var connection = svr_connection(binding)
    var lease = SvrResidentPin(PythonObject(None), 0)
    if resident and not precomputed:
        var mirrors: List[PythonObject] = [dual, sv]
        lease = svr_resident_pin(model, _SVR_RESIDENT_SLOT, binding, connection, mirrors)
    var dual_address = Int(py=classical_address(dual)) if n_support > 0 else 0
    var vector_address = Int(py=classical_address(sv)) if n_support > 0 else 0
    # Native List, not Array: an Array local whose address escapes only as an Int is
    # neither initialised for nor read back from the callee in an optimized Mojo 1.2
    # build (reproduced on the Mac, smoke-triage C01 pattern).
    var status = List[Int32](length=2, fill=Int32(0))
    var request = SvrQueryRequest(query_address, dual_address, vector_address, 0,
        Int(status.unsafe_ptr()), n_rows, n_features, n_support, kernel, degree,
        lease.handle, 0, q_meta.store.base_id, q_meta.store.base_rows, q_meta.store.index_id,
        bias, gamma, cache_mib, coef0)
    return SvrQueryUnpacked(q_meta^, owners, gather_columns, gather_count, dual, sv, connection^, lease^, status^,
                            request, resident)


def svr_query_refused(unpacked: SvrQueryUnpacked) raises:
    """After RUN: the refusal of a query the resident entry flagged, in the
    one wording 'Input X contains NaN or infinity (SVR).' (the other routes
    tested X at admission)."""
    if unpacked.resident:
        _svm_query_refuse(Int(unpacked.status[0]), Int(unpacked.status[1]), unpacked.q, "SVR")


def svr_score_unpack(rows: Int, y: PythonObject, context: PythonObject) raises:
    """`SVR.score`: the shape checks of y against the predicted rows, made
    after the prediction, as before."""
    var shape = svc_shape_of(y, context)
    if len(shape) != 1:
        raise Error("mojolearn SVR: y must be 1-D, got " + String(len(shape)) + "-D shape " + family_repr(shape))
    if Int(py=shape[0]) != rows:
        raise Error("mojolearn SVR: y has " + String(Int(py=shape[0])) + " entries, X has " + String(rows) + " rows")
