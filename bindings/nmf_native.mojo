# SPDX-License-Identifier: Apache-2.0
# NMF UNPACK/PACK: admit public options/storage, call the native descriptor,
# and publish public fitted state. Initialization and training stay in the
# existing x_decomp/nmf_owner kernels; no Python call envelopes reach RUN.
from std.python import Python, PythonObject
from std.python._cpython import ExternalFunction, PyObjectPtr
from core.shared_slot import SharedSlot
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_get, py_type, py_fields, py_setitem
from bindings.native_buffer_address import native_buffer_address
from bindings.admitted_input import admit_f32, AdmittedX, admitted_finite_refuse
from bindings.decomp_inputs import decomp_admit
from bindings.decomp_owner_common import decomp_attr, decomp_check, decomp_gemm_check, decomp_set_mode
from bindings.decomp_scalar_admission import decomp_scalar_admit, decomp_scalar_repr
from bindings.host_inference_scalars import host_inference_scalar
from bindings.nmf_descriptor_boundary import nmf_connection, nmf_run_fit, nmf_run_inverse
from core.nmf_native_state import NMFScratch, nmf_output_shape
from bindings.classical_arrays import classical_typed_native
from core.nmf_native_api import (
    NMFFitRequest, NMFFitResult, NMFInverseRequest, NMFRegularization, NMFOptions, nmf_regularization,
    NMF_CAP_FIT, NMF_CAP_INVERSE,
)
from x_decomp.nmf import NmfArgs
from x_decomp.nmf_owner import (
    NMF_INIT_RANDOM, NMF_INIT_NNDSVD, NMF_INIT_NNDSVDA, NMF_INIT_NNDSVDAR,
    NMF_INIT_GIVEN, NMF_TRANSFORM,
)


comptime _NMFCompare = ExternalFunction[
    "PyObject_RichCompareBool", def(PyObjectPtr, PyObjectPtr, Int32) thin abi("C") -> Int32]


struct NMFScalarAPI(Defaultable, Movable):
    var compare: _NMFCompare.type

    def __init__(out self):
        ref cpy = Python().cpython()
        self.compare = _NMFCompare.load(cpy.lib.borrow())


comptime NMF_SCALAR_API = SharedSlot[NMFScalarAPI, "MojolearnNMFScalarAPIV1"]


@fieldwise_init
struct NMFPackedFit(Movable):
    # Only public output storage pins remain Python objects. The execution
    # result, options and dimensions are native values.
    var w: PythonObject
    var h: PythonObject
    var count: Int
    var result: NMFFitResult


def _equal(value: PythonObject, other: PythonObject) raises -> Bool:
    # Both operands are exact admitted scalar storage, so their C comparison
    # slots cannot dispatch to user protocols. Never call operator.eq.
    var result = NMF_SCALAR_API.get_or_create_ptr()[].compare(value._obj_ptr, other._obj_ptr, Int32(2))
    _ = value
    _ = other
    if result < 0:
        ref cpy = Python().cpython()
        raise cpy.unsafe_get_error()
    return result != 0


def _name(value: PythonObject, name: String) raises -> Bool:
    # Caller admits the scalar first. The common exact-string path compares
    # native strings; native NumPy scalar/0-D comparisons keep their C slots.
    if py_type(value) is py_type(PythonObject("")):
        return String(value) == name
    return _equal(value, PythonObject(name))


def _int(value: PythonObject, context: PythonObject) raises -> Int:
    decomp_scalar_admit(value, context)
    return Int(py=py_call(native_loaded_module("builtins").int, value))


def _float(value: PythonObject, context: PythonObject) raises -> Float64:
    decomp_scalar_admit(value, context)
    return Float64(py=py_call(native_loaded_module("builtins").float, value))


def _truth(value: PythonObject, context: PythonObject) raises -> Bool:
    try:
        host_inference_scalar(value, context, True)
    except:
        decomp_scalar_admit(value, context)
    return Bool(py=value)


def _seed(value: PythonObject) raises -> Int:
    if value is None:
        return 0
    var b = native_loaded_module("builtins")
    if py_type(value) is not b.int:
        raise Error("TypeError: random_state must be None or an int (the counter-based RNG takes an integer seed)")
    return Int(py=py_call(b.int.__dict__["__and__"], value, PythonObject(0xFFFFFFFF)))


def _address(value: PythonObject, context: PythonObject) raises -> Int:
    if py_type(value) is context["array_class"]:
        return Int(py=value._addr)
    return native_buffer_address(value)


def _beta(fields: PythonObject, context: PythonObject) raises -> Float64:
    var loss = fields["beta_loss"]
    decomp_scalar_admit(loss, context)
    var cls = py_type(loss)
    var b = native_loaded_module("builtins")
    # The previous dictionary lookup refused these before value comparison.
    if cls is py_get(context, "ndarray"):
        raise Error("TypeError: unhashable type: 'numpy.ndarray'")
    if cls is b.bytearray:
        raise Error("TypeError: unhashable type: 'bytearray'")
    if _name(loss, "frobenius"):
        return 2.0
    if _name(loss, "kullback-leibler"):
        return 1.0
    if _name(loss, "itakura-saito"):
        return 0.0
    # Compare before Float64 conversion, as the original membership test did:
    # an extended-precision scalar adjacent to 2 must not round into admission.
    if _equal(loss, PythonObject(0.0)) or _equal(loss, PythonObject(1.0)) or _equal(loss, PythonObject(2.0)):
        return _float(loss, context)
    raise Error("beta_loss: 'frobenius', 'kullback-leibler' and 'itakura-saito' are carried; "
                + "a general float beta is not")


def _validate(fields: PythonObject, context: PythonObject) raises -> NMFOptions:
    var beta = _beta(fields, context)
    var solver = fields["solver"]
    decomp_scalar_admit(solver, context)
    var mu = _name(solver, "mu")
    if beta != 2.0 and not mu:
        raise Error("Invalid beta_loss parameter: solver 'cd' does not handle beta_loss other than 'frobenius'")
    if not _name(solver, "cd") and not mu:
        raise Error("Invalid solver parameter: got " + decomp_scalar_repr(solver, context) + " instead of one of {'cd', 'mu'}")
    return NMFOptions(beta, 1 if mu else 0)


def _regs(fields: PythonObject, n: Int, d: Int, context: PythonObject) raises -> NMFRegularization:
    var alpha_h = fields["alpha_H"]
    var a_w = _float(fields["alpha_W"], context)
    decomp_scalar_admit(alpha_h, context)
    var a_h = a_w if _name(alpha_h, "same") else _float(alpha_h, context)
    var l1 = _float(fields["l1_ratio"], context)
    return nmf_regularization(n, d, a_w, a_h, l1)


def _run(model: PythonObject, fields: PythonObject, admitted: AdmittedX, nc: Int, init: Int,
         w_address: Int, h_address: Int, seed: Int, update_h: Bool, options: NMFOptions,
         context: PythonObject, publish_w: Bool = True) raises -> NMFPackedFit:
    var connection = nmf_connection(model, fields, context, NMF_CAP_FIT, "x_decomp_nmf_fit")
    var n = admitted.rows
    var d = admitted.cols
    var regs = _regs(fields, n, d, context)
    var w = PythonObject(None)
    var h = PythonObject(None)
    var w_scratch = Optional[NMFScratch]()
    var h_scratch = Optional[NMFScratch]()
    var w_output: Int
    var h_output: Int
    if publish_w:
        w = classical_typed_native(nmf_output_shape(n, nc), "<f4", context["output_context"])
        w_output = _address(w, context)
    else:
        w_scratch = NMFScratch(n, nc)
        w_output = w_scratch.value().address()
    if update_h:
        h = classical_typed_native(nmf_output_shape(nc, d), "<f4", context["output_context"])
        h_output = _address(h, context)
    else:
        h_scratch = NMFScratch(nc, d)
        h_output = h_scratch.value().address()
    var max_iter = _int(fields["max_iter"], context)
    var shuffle = _truth(fields["shuffle"], context)
    var tol = _float(fields["tol"], context)
    var args = NmfArgs(options.beta, update_h, max_iter, tol, regs.l1w, regs.l1h, regs.l2w, regs.l2h, shuffle, seed)
    var request = NMFFitRequest(admitted.address, w_address if w_address >= 0 else h_output,
        h_address if h_address >= 0 else h_output, w_output, h_output,
        n, d, nc, init, options.solver, args^, not admitted.finite)
    var result = nmf_run_fit(connection, request)
    if not admitted.finite:
        admitted_finite_refuse(result.finite, "NMF")
    _ = w_scratch^
    _ = h_scratch^
    return NMFPackedFit(w, h, nc, result)


def nmf_fit_transform(model: PythonObject, source: PythonObject, w_given: PythonObject, h_given: PythonObject,
                      context: PythonObject, publish_w: Bool = True) raises -> PythonObject:
    var fields = py_fields(model)
    decomp_set_mode(fields, context)
    # TOMBSTONE: NMF admission host scan of a borrowed X deleted 2026-11-09 by lane a8d-device-x-decomp-manifold; code recoverable at 0ce88411d3.
    # Restore: experiments/removed/A8D_DECOMP_MANIFOLD_HOST_FINITE.patch.
    var admitted = decomp_admit(source, "NMF", context, defer_finite=True)
    var n = admitted.rows
    var d = admitted.cols
    var options = _validate(fields, context)
    var init_value = fields["init"]
    decomp_scalar_admit(init_value, context)
    var result: NMFPackedFit
    if _name(init_value, "custom"):
        var wm = decomp_admit(w_given, "NMF", context)
        var hm = decomp_admit(h_given, "NMF", context)
        var nc = hm.rows
        if wm.rows != n or wm.cols != nc or hm.cols != d:
            raise Error("NMF: W must be (n_samples, n_components) and H (n_components, n_features) for init='custom'")
        result = _run(model, fields, admitted, nc, NMF_INIT_GIVEN, wm.address, hm.address,
                      _seed(fields["random_state"]), True, options, context, publish_w)
        _ = wm^
        _ = hm^
    else:
        var requested = fields["n_components"]
        decomp_scalar_admit(requested, context)
        var nc = d if requested is None or _name(requested, "auto") else _int(requested, context)
        var smaller = min(n, d)
        var name = init_value
        if not _truth(init_value, context):
            name = PythonObject("nndsvda") if nc <= smaller else PythonObject("random")
        var code = -1
        if _name(name, "random"):
            code = NMF_INIT_RANDOM
        elif _name(name, "nndsvd"):
            code = NMF_INIT_NNDSVD
        elif _name(name, "nndsvda"):
            code = NMF_INIT_NNDSVDA
        elif _name(name, "nndsvdar"):
            code = NMF_INIT_NNDSVDAR
        if code < 0:
            raise Error("init=" + decomp_scalar_repr(name, context) + " is not supported here (random, nndsvd, nndsvda, nndsvdar)")
        if code != NMF_INIT_RANDOM and nc > smaller:
            raise Error("init = 'nndsvd' can only be used when n_components <= min(n_samples, n_features)")
        result = _run(model, fields, admitted, nc, code, -1, -1,
                      _seed(fields["random_state"]), True, options, context, publish_w)
    py_setitem(fields, "components_", result.h)
    py_setitem(fields, "n_components_", PythonObject(result.count))
    py_setitem(fields, "n_iter_", PythonObject(result.result.iterations))
    py_setitem(fields, "n_features_in_", PythonObject(d))
    py_setitem(fields, "reconstruction_err_", PythonObject(result.result.error))
    _ = admitted^
    return result.w


def nmf_fit(model: PythonObject, source: PythonObject, params: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if py_type(params) is not b.dict:
        raise Error("TypeError: NMF fit keywords require a native dictionary")
    var keys = py_call(b.list, params)
    for k in range(len(keys)):  # driver-loop: the caller's keyword names
        if py_type(keys[k]) is not b.str:
            raise Error("TypeError: NMF fit keywords require native string names")
        var key = String(keys[k])
        if key != "W" and key != "H" and key != "y":
            raise Error("TypeError: fit_transform() got an unexpected keyword argument " + decomp_scalar_repr(keys[k], context))
    _ = nmf_fit_transform(model, source, py_get(params, "W"), py_get(params, "H"), context, False)
    return model


def nmf_transform(model: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    decomp_check(model, fields, "components_")
    var admitted = decomp_admit(source, "NMF", context, defer_finite=True)
    var fitted = _int(decomp_attr(model, fields, "n_features_in_"), context)
    if admitted.cols != fitted:
        raise Error("X has " + String(admitted.cols) + " features, but NMF is expecting " + String(fitted))
    var options = _validate(fields, context)
    var h = admit_f32(fields["components_"], 2, "components_", context)
    var result = _run(model, fields, admitted, h.rows, NMF_TRANSFORM, -1, h.address,
                      _seed(fields["random_state"]), False, options, context)
    _ = admitted^
    _ = h^
    return result.w


def nmf_inverse_transform(model: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    decomp_check(model, fields, "components_")
    var connection = nmf_connection(model, fields, context, NMF_CAP_INVERSE, "x_decomp_nmf_inverse")
    var w = decomp_admit(source, "NMF", context)
    var h = admit_f32(fields["components_"], 2, "components_", context)
    decomp_gemm_check(w.cols, h.rows)
    var dst = classical_typed_native(nmf_output_shape(w.rows, h.cols), "<f4", context["output_context"])
    var request = NMFInverseRequest(w.address, h.address, _address(dst, context), w.rows, h.cols, w.cols)
    nmf_run_inverse(connection, request)
    _ = h^
    _ = w^
    return dst
