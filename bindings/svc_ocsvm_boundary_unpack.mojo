# SPDX-License-Identifier: Apache-2.0
# UNPACK of OneClassSVM (the door is bindings/svc_ocsvm_native_door.mojo):
# the x_neighbors binding of the call, the parameter checks of the Python
# `fit` and query methods in their order and words, the admitted X, sample
# weights, dual coefficients and support columns with their Python owners,
# and the typed requests RUN reads (svm/svc_ocsvm_native_run.mojo:
# OcFitRequest, OcQueryRequest). Every parameter is read before RUN starts;
# RUN sees addresses and plain words only.
from providers.python.x_neighbors_descriptor_pyapi_door import xn_admit, xn_module, xn_is_module, xn_capabilities
from providers.python.estimator_binding_pyapi_door import estimator_binding
from std.python import PythonObject
from bindings.ocsvm_provider_unpack import OcConnection
from core.neighbors_native_driver import XnDriver
from core.ocsvm_resident_api import OcResidentDriver
from core.ocsvm_plan import oc_i32_address
from bindings.ocsvm_scalar_boundary_door import oc_dictionary, oc_name, oc_kernel_admit, oc_parameter_repr, oc_mode_value, oc_scalar_admit, oc_precomputed
from bindings.admitted_family_door import admit_family
from bindings.native_loaded_module_door import _loaded_fields
from providers.python.neural_descriptor_pyapi_door import neural_descriptor_entry
from bindings.py_balanced import py_type, py_get, py_fields, py_item
from bindings.classical_arrays_door import classical_address
from bindings.family_common_unpack import (
    family_min,
)
from bindings.family_common_pack import (
    family_call,
)
from core.admitted_convert import admitted_convert_vector
from bindings.gp_svm_records_door import ocsvm_coef_reused, ocsvm_coef_remember
from bindings.gp_svm_gp_state_door import gp_x_known
from bindings.svc_common_door import (
    svc_builtins, svc_field, svc_float, svc_int, svc_truthy,
)
from svm.svc_ocsvm_native_run import (
    OcFitRequest, OcQueryRequest, OC_SOLVE_GRAM, OC_SOLVE_RESIDENT, OC_SOLVE_DESCRIPTOR,
    OC_QUERY_FUSED_LABELS, OC_QUERY_KERNEL_PRODUCT, OC_QUERY_GRAM_PRECOMPUTED, OC_QUERY_GRAM_KERNEL,
)
from bindings.svm_keys_unpack import svm_keys


def oc_binding_unpack(model: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    # `source` is the x_neighbors module the shared door resolved from the
    # estimator's binding route. A binding facade handed in its place is
    # resolved through the same route (the estimator's device), never the
    # process set (before 2026-11-21 this branch took the process's module).
    oc_dictionary(context)
    var nk = svm_keys()
    var backend = py_item(context, nk[].k_backend_fields)
    oc_dictionary(backend)
    var binding = source
    if py_type(source) is py_item(backend, nk[].k__HostBinding) or py_type(source) is py_item(backend, nk[].k__NoGpuBinding):
        binding = estimator_binding(model, "_mojolearn_x_neighbors", context)
    return xn_admit(binding)


def _oc_kernel_code(kernel: PythonObject, context: PythonObject) raises -> Int:
    # The kernel-name table lookup. An array kernel (any rank) is an unhashable
    # key: keep CPython's TypeError (a caught Python error reaches Mojo as its
    # message only, which the door would otherwise raise as a ValueError).
    var code = PythonObject(None)
    var nk = svm_keys()
    try:
        code = py_item(context, nk[].k_family_svc_xn_kernels)[kernel]
    except e:
        raise Error("TypeError: " + String(e))
    return Int(py=code)


def _oc_float(value: PythonObject, context: PythonObject) raises -> Float64:
    oc_scalar_admit[True](value, context)
    return svc_float(value)


def _oc_int(value: PythonObject, context: PythonObject) raises -> Int:
    oc_scalar_admit[True](value, context)
    return svc_int(value)


def _a(value: PythonObject) raises -> PythonObject:
    return classical_address(value)


def _fast_tier_native(mut connection: OcConnection, binding: PythonObject, context: PythonObject) raises -> Bool:
    if not xn_is_module(binding):
        return False
    var module = xn_module(binding)
    var nk = svm_keys()
    var full_value = py_get(_loaded_fields(module), nk[].k___name__)
    if py_type(full_value) is not svc_builtins().str:
        return False
    var parts = String(full_value).split(".")
    var name = String(parts[len(parts) - 1])
    var prefix = String("mojolearn") if name == "_mojolearn" else String(name.removeprefix("_mojolearn_"))
    var getter = neural_descriptor_entry(module, prefix + "_numeric_mode", False)
    if getter is not None:
        var mode = oc_mode_value(context, connection.resident_driver().metadata(0))
        return mode is not None and oc_name(mode, "fast", context)
    if len(parts) < 2:
        return False
    return String(parts[len(parts) - 2]) == "fast"


# ---------------------------------------------------------------- fit


struct OcFitUnpacked(Movable):
    """A fit: the typed request, the drivers it runs on, and the Python
    owners of every buffer it addresses (the instance dictionary, X, the
    admitted sample weights), held until PACK."""
    var fields: PythonObject
    var x: PythonObject
    var weights: PythonObject
    var driver: XnDriver
    var resident: Optional[OcResidentDriver]
    var request: OcFitRequest

    def __init__(out self, fields: PythonObject, x: PythonObject, weights: PythonObject, driver: XnDriver):
        self.fields = fields
        self.x = x
        self.weights = weights
        self.driver = driver.copy()
        self.resident = Optional[OcResidentDriver]()
        self.request = OcFitRequest()


def oc_fit_unpack(model: PythonObject, binding: PythonObject, source: PythonObject, sample_weight: PythonObject,
                  context: PythonObject, mut connection: OcConnection) raises -> OcFitUnpacked:
    """`OneClassSVM.fit(X, sample_weight)`: every check and every parameter
    of the fit, in the order the fit always made them."""
    var b = svc_builtins()
    var fields = py_fields(model)
    var nk = svm_keys()
    var kernel = svc_field(fields, nk[].k_kernel, "kernel")
    if Bool(py=family_call(b.callable, kernel)):
        raise Error("NotImplementedError: OneClassSVM callable kernels require a native kernel name")
    if not oc_kernel_admit(kernel, context):
        raise Error("NotImplementedError: OneClassSVM: kernel=" + oc_parameter_repr(kernel, context) + " is not implemented")
    var nu = _oc_float(svc_field(fields, nk[].k_nu, "nu"), context)
    if not (0.0 < nu and nu <= 1.0):
        raise Error("nu must be in (0, 1]")
    var x = source if gp_x_known(source, context) else admit_family(source, 2, "X", context, True).array
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var precomputed = oc_precomputed(kernel, context)
    if precomputed and n != d:
        raise Error("Precomputed matrix must be a square matrix.")
    var wv = PythonObject(None)
    if sample_weight is not None:
        wv = admit_family(sample_weight, 1, "sample_weight", context, False).array
        if Int(py=wv.shape[0]) != n:
            raise Error("sample_weight and X have different numbers of samples")
        if n > 0 and family_min(wv) < 0:
            raise Error("negative sample_weight is not supported")
    var unpacked = OcFitUnpacked(fields, x, wv, connection.driver())
    unpacked.request.x = Int(py=_a(x))
    unpacked.request.rows = n
    unpacked.request.features = d
    unpacked.request.precomputed = precomputed
    unpacked.request.nu = nu
    if sample_weight is not None:
        unpacked.request.weighted = True
        unpacked.request.weights = Int(py=_a(wv))
    var resident_available = False
    if not precomputed:
        unpacked.request.degree = _oc_int(svc_field(fields, nk[].k_degree, "degree"), context)
        unpacked.request.coef0 = _oc_float(svc_field(fields, nk[].k_coef0, "coef0"), context)
        var gamma = svc_field(fields, nk[].k_gamma, "gamma")
        if gamma is None or oc_name(gamma, "auto", context):
            unpacked.request.gamma = 1.0 / Float64(d)
        elif oc_name(gamma, "scale", context):
            unpacked.request.gamma_scale = True  # 1 / (features * variance of X), read by RUN
        else:
            unpacked.request.gamma = _oc_float(gamma, context)
        if xn_is_module(binding):
            unpacked.request.resident_fast = _fast_tier_native(connection, binding, context)
            var entry = "x_neighbors_ocsvm_resident" if unpacked.request.resident_fast else "x_neighbors_ocsvm_resident_idn"
            resident_available = neural_descriptor_entry(xn_module(binding), entry, False) is not None
        else:
            resident_available = (xn_capabilities(binding) & 1) != 0
        if not resident_available:
            unpacked.request.kernel = _oc_kernel_code(kernel, context)
    var max_iter = _oc_int(svc_field(fields, nk[].k_max_iter, "max_iter"), context)
    unpacked.request.cap = 10000000 if max_iter < 0 else max_iter
    # Rounded to float32 by the solve (core/ocsvm_plan.mojo), as before.
    unpacked.request.tol = _oc_float(svc_field(fields, nk[].k_tol, "tol"), context)
    if resident_available:
        unpacked.request.kernel = _oc_kernel_code(kernel, context)
        if xn_is_module(binding):
            unpacked.request.solve = OC_SOLVE_RESIDENT
            unpacked.resident = Optional(connection.resident_driver())
        else:
            unpacked.request.solve = OC_SOLVE_DESCRIPTOR
    return unpacked^


# ---------------------------------------------------------------- queries


@fieldwise_init
struct OcIndexPin(Movable):
    var owner: PythonObject
    var words: List[Int32]
    var address: Int
    var count: Int


def _oc_indices(value: PythonObject) raises -> OcIndexPin:
    var typestr = String(value.dtype)
    var shape = value.shape
    if typestr == "<i4":
        # Original exact-int32 storage bypassed conversion and used shape[0].
        var count = Int(py=shape[0])
        return OcIndexPin(value, List[Int32](), Int(py=_a(value)), count)
    if len(shape) == 0:
        var kind = String("float") if typestr.startswith("<f") else String("int")
        raise Error("TypeError: '" + kind + "' object is not iterable")
    if len(shape) != 1:
        raise Error("TypeError: int() argument must be a string, a bytes-like object or a real number, not 'list'")
    var count = Int(py=shape[0])
    var words = List[Int32](length=count, fill=0)
    var address = oc_i32_address(words)
    _ = admitted_convert_vector(Int(py=_a(value)), typestr, address, "<i4", count)
    _ = value
    return OcIndexPin(value, words^, address, count)


def _ocsvm_labels_available(binding: PythonObject) raises -> Bool:
    if xn_is_module(binding):
        return neural_descriptor_entry(xn_module(binding), "x_neighbors_ocsvm_labels", False) is not None
    return (xn_capabilities(binding) & 16) != 0


struct OcQueryUnpacked(Movable):
    """A query: the typed request, the driver, and the Python owners of the
    buffers it addresses (X, the support vectors, the admitted dual
    coefficients, the support columns of a precomputed kernel)."""
    var q: PythonObject
    var sv: PythonObject
    var coef: PythonObject
    var support: OcIndexPin
    var driver: XnDriver
    var request: OcQueryRequest

    def __init__(out self, q: PythonObject, sv: PythonObject, coef: PythonObject, var support: OcIndexPin,
                 driver: XnDriver, request: OcQueryRequest):
        self.q = q
        self.sv = sv
        self.coef = coef
        self.support = support^
        self.driver = driver.copy()
        self.request = request.copy()


def oc_query_unpack(model: PythonObject, binding: PythonObject, source: PythonObject, labels: Bool,
                    offset: Bool, context: PythonObject, mut connection: OcConnection) raises -> OcQueryUnpacked:
    """score_samples / decision_function / predict: X, the fitted state and
    the kernel parameters of the path the query takes. `labels` asks for the
    fused labels when the binding has them (predict); `offset` reads
    `offset_` for the decision (decision_function, predict)."""
    var fields = py_fields(model)
    var q = source if gp_x_known(source, context) else admit_family(source, 2, "X", context, True).array
    var nk = svm_keys()
    var dual = svc_field(fields, nk[].k_dual_coef_, "dual_coef_")
    var n_sv = Int(py=dual.shape[1])
    var coef: PythonObject
    if ocsvm_coef_reused(model, dual):
        coef = dual
    else:
        coef = admit_family(dual, 2, "dual_coef_", context, False).array
        ocsvm_coef_remember(model, dual, coef)
    var kernel = svc_field(fields, nk[].k_kernel, "kernel")
    var precomputed = oc_precomputed(kernel, context)
    var nq = Int(py=q.shape[0])
    var sv = svc_field(fields, nk[].k_support_vectors_, "support_vectors_")
    var request = OcQueryRequest()
    var support = OcIndexPin(PythonObject(None), List[Int32](), 0, 0)
    request.rows = nq
    request.supports = n_sv
    if not precomputed and n_sv > 0 and nq > 0 and not svc_truthy(py_item(context, nk[].k_family_svc_xn_old_items)):
        request.degree = _oc_int(svc_field(fields, nk[].k_degree, "degree"), context)
        request.gamma = _oc_float(svc_field(fields, nk[].k__gamma, "_gamma"), context)
        request.coef0 = _oc_float(svc_field(fields, nk[].k_coef0, "coef0"), context)
        request.features = Int(py=q.shape[1])
        request.path = OC_QUERY_KERNEL_PRODUCT
        if labels and _ocsvm_labels_available(binding):
            request.offset = _oc_float(svc_field(fields, nk[].k_offset_, "offset_"), context)
            request.path = OC_QUERY_FUSED_LABELS
        request.query = Int(py=_a(q))
        request.references = Int(py=_a(sv))
        request.coefficients = Int(py=_a(coef))
        request.kernel = _oc_kernel_code(kernel, context)
    elif precomputed:
        support = _oc_indices(svc_field(fields, nk[].k_support_, "support_"))
        request.path = OC_QUERY_GRAM_PRECOMPUTED
        request.query = Int(py=_a(q))
        request.features = Int(py=q.shape[1])
        request.support = support.address
        request.support_count = support.count
        request.coefficients = Int(py=_a(coef))
    else:
        request.path = OC_QUERY_GRAM_KERNEL
        request.kernel = _oc_kernel_code(kernel, context)
        request.gamma = _oc_float(svc_field(fields, nk[].k__gamma, "_gamma"), context)
        request.coef0 = _oc_float(svc_field(fields, nk[].k_coef0, "coef0"), context)
        request.degree = _oc_int(svc_field(fields, nk[].k_degree, "degree"), context)
        request.features = Int(py=q.shape[1])
        request.reference_rows = Int(py=sv.shape[0])
        request.query = Int(py=_a(q))
        request.references = Int(py=_a(sv))
        request.coefficients = Int(py=_a(coef))
    if offset and request.path != OC_QUERY_FUSED_LABELS:
        request.offset = _oc_float(svc_field(fields, nk[].k_offset_, "offset_"), context)
    return OcQueryUnpacked(q, sv, coef, support^, connection.driver(), request)
