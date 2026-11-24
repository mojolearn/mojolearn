# SPDX-License-Identifier: Apache-2.0
# `mojolearn.matmul` and the low-bit functions of python/mojolearn/_linalg_impl.py
# (to_bf16, from_bf16, quantize/dequantize int8 and int15, matmul_bf16/int8/int15),
# one native call each (`family_lin_*`). The shell hands over the linalg binding
# `_load()` resolved; this owner makes the Python refusals in their order and
# words, the mode and profile read-backs (`numeric_mode`, `require_identical`,
# `_lowbit_binding`, `_int15_binding`, sharing the module's own caches), the
# operand layout (`_operand`, `as_*_c`) and outputs, and runs each operation
# through the binding's typed operations table ("mojolearn.linalg.v1",
# gemm/linalg_dispatch.mojo and gemm/host/linalg_dispatch.mojo: the device and
# the host image publish the same table) with the same address and parameter
# words in the same order. The same kernels on the same bytes: the bits do not
# move.
# UNPACK / RUN / PACK (lane dw-decomp, 2026-11-21): the functions below admit
# the Python arguments into AdmittedBuf pins and Int words (UNPACK), `lin_call`
# hands two List[Int] to the table with the GIL released (RUN holds no Python
# object), and the admitted output owner is returned (PACK). Each computing
# function takes `device=` as the last item of its tuple (lane vdf-linalg,
# `lin_device` below); the three read-backs (numeric_mode, profile,
# require_identical) compute nothing and take none.
# Defines read here: none. The text names MOJOLEARN_AFN_GEMM_EPILOGUE (the Apple
# fast epilogue arm, operation 15, refused in words by a build without it) and
# MOJOLEARN_NUMERIC_MODE (in a refusal's advice).
from providers.python.estimator_binding_pyapi_door import estimator_binding, estimator_binding_device
from providers.python.function_device_boundary_door import function_device_admit
from core.function_device import FunctionDevice, function_device_default
from std.python import Python, PythonObject
from std.os import getenv
from bindings.native_loaded_module_door import native_loaded_module
from bindings.classical_pandas_door import classical_admit_value
from bindings.host_inference_scalars_door import host_inference_scalar
from providers.python.neural_descriptor_pyapi_door import neural_descriptor_reuse, neural_descriptor_execute, neural_descriptor_entry
from bindings.py_balanced import py_call, py_get, py_setitem, py_type, py_fields
from bindings.family_common_unpack import family_repr, family_str, family_type_name, family_isinstance
from bindings.family_common_pack import family_call
from bindings.tensor_view import TensorView, shape_text
from bindings.linalg_ops_buffers_unpack import lin_truth, lin_dtype_name
from bindings.linalg_ops_buffers_run import lin_is_f32
from bindings.linalg_ops_lowbit_run import (
    LIN_OP_GEMM_FUSED, LIN_OPERATION_COUNT, LIN_OP_GEMM, LIN_OP_GEMM_BF16, LIN_OP_GEMM_INT8, LIN_OP_QUANTIZE_INT8, LIN_OP_DEQUANTIZE_INT8,
    LIN_OP_GEMM_INT15, LIN_OP_QUANTIZE_INT15, LIN_OP_DEQUANTIZE_INT15, LIN_OP_TO_BF16, LIN_OP_FROM_BF16,
    LIN_OP_FROM_F16, LIN_OP_LINALG_NUMERIC_MODE, LIN_OP_LINALG_PROFILE_VERSION, LIN_OP_LOWBIT_PROFILE_VERSION,
    LIN_OP_INT15_PROFILE_VERSION, LIN_OP_GEMM_DEV, LIN_OP_TO_BF16_DEV, LIN_OP_FROM_BF16_DEV, LIN_OP_QUANTIZE_INT8_DEV,
    LIN_OP_DEQUANTIZE_INT8_DEV, LIN_OP_GEMM_BF16_DEV, LIN_OP_GEMM_INT8_DEV)
from bindings.linalg_dev_arrays_unpack import (
    lin_train_device, lin_is_device_array, lin_any_device_array, lin_dev_device, lin_dev_entries, lin_dev_operand,
    lin_dev_out, lin_dev_new, lin_dev_handle, lin_dev_dtype, LinDevOperand,
)
from core.train_dev_pool import TRAIN_DEV_F32, TRAIN_DEV_I8, TRAIN_DEV_I32, TRAIN_DEV_U16
from bindings.admitted_input_unpack import AdmittedBuf, AdmittedTypedPin, admitted_typed_once, admitted_typed_c, admitted_typed_output, admitted_unfilled, admit_typed, admitted_shape_tuple
from bindings.linalg_keys_unpack import linalg_keys
from bindings.py_balanced import py_item
from bindings.operation_names_linops import (
    LINOPS_OPN_FAMILY_LIN_DEQUANTIZE_INT15, LINOPS_OPN_FAMILY_LIN_DEQUANTIZE_INT8, LINOPS_OPN_FAMILY_LIN_FROM_BF16,
    LINOPS_OPN_FAMILY_LIN_FROM_F16, LINOPS_OPN_FAMILY_LIN_MATMUL, LINOPS_OPN_FAMILY_LIN_MATMUL_BF16,
    LINOPS_OPN_FAMILY_LIN_MATMUL_FUSED, LINOPS_OPN_FAMILY_LIN_MATMUL_INT15, LINOPS_OPN_FAMILY_LIN_MATMUL_INT8,
    LINOPS_OPN_FAMILY_LIN_NUMERIC_MODE, LINOPS_OPN_FAMILY_LIN_PROFILE, LINOPS_OPN_FAMILY_LIN_QUANTIZE_INT15,
    LINOPS_OPN_FAMILY_LIN_QUANTIZE_INT8, LINOPS_OPN_FAMILY_LIN_REQUIRE_IDENTICAL, LINOPS_OPN_FAMILY_LIN_TO_BF16)

comptime OP_NN = 0
comptime OP_NT = 1
comptime OP_TN = 2


def lin_state(context: PythonObject) raises -> PythonObject:
    """python/mojolearn/_linalg_impl.py's module dictionary (its constants and caches)."""
    var nk = linalg_keys()
    return py_item(context, nk[].k_family_lin_state)


def lin_text(context: PythonObject, key: PythonObject) raises -> String:
    """A profile text of the module state by its held key (bw2-linalg: it
    built the key str from a name per call)."""
    var value = py_item(lin_state(context), key)
    if py_type(value) is not py_type(PythonObject("")):
        raise Error("TypeError: linalg profile metadata requires an exact string")
    return String(value)


def lin_attr(binding: PythonObject, name: String) raises -> PythonObject:
    """`getattr(binding, name, None)`."""
    return neural_descriptor_entry(binding, name, False)


def lin_call(context: PythonObject, op: Int, addresses: List[Int], parameters: List[Int]) raises -> Int:
    """One operation `op` (`LIN_OP_*`, bindings/linalg_ops_lowbit_run.mojo) of
    the connected linalg binding's operations table: buffer addresses and
    shape words in, the operation's count out. The GIL is released around the
    call; a refusal of the kernel's own admission propagates unchanged.
    `lin_connected_binding` connected the table for this operation.
    lane dw-decomp (2026-11-21): the Python-entry form is deleted (`_lin_apply`:
    a by-name entry of the binding called with a boxed address per buffer and
    a Python list of the shape words, which the provider path then read back
    word by word into these two lists). The host image has had the table
    since ne1-k-fail-b, so no binding is left without it."""
    var nk = linalg_keys()
    var provider = py_get(lin_state(context), nk[].k__native_provider)
    if provider is None:
        raise Error("RuntimeError: mojolearn.linalg: the loaded linalg extension has no native operations "
                    + "table (linalg_native_api_v1); rebuild it with bindings/build_linalg.sh")
    return neural_descriptor_execute(provider, op, addresses, parameters, List[Float64]())


def lin_backend_mode(context: PythonObject) raises -> String:
    var nk = linalg_keys()
    var mode = py_item(py_item(context, nk[].k_backend_fields), nk[].k__DEFAULT_MODE)
    if mode is None:
        raise Error("RuntimeError: linalg numeric mode must be selected before operation admission")
    return String(py=mode)


def lin_connected_binding(context: PythonObject) raises -> PythonObject:
    # Resolve the live import-connected module once per operation. Linalg's
    # own provider supplies its compile-mode/profile metadata immediately below.
    return lin_publish_binding(estimator_binding(PythonObject(None), "_mojolearn_linalg", context, False), context)


# ---------------------------------------------------------------------------
# `device=` of the linalg functions (lane vdf-linalg, 2026-11-24;
# core/function_device.mojo, providers/python/function_device_boundary_door.mojo).
#
# The linalg binding has both sides: the device kernels (`_mojolearn_linalg`,
# gemm/linalg_dispatch.mojo, decomposition/linalg_public_device.mojo) and the
# host image `_mojolearn_linalg_host` (host_surface's linalg family, the
# whole route on a CPU-only install), which publishes the same operations
# table. `device=None` is today's route exactly: the door's context and
# `lin_connected_binding`, unchanged. An explicit "cpu" beside a GPU set runs
# the host image connected at import (`family_binding_connections_cpu`) in
# its OWN module state (`lin_device_context`): the binding cache, the mode
# and profile caches and the operations-table provider of the host image
# never share a slot with the device image's, so a device="cpu" call on one
# thread can never read the device provider another thread published (and
# the reverse), and alternating calls do not reconnect either provider.
comptime LIN_HAS_GPU = True
comptime LIN_HAS_CPU = True
comptime _LIN_CPU_CONTEXT = "lin_cpu_context"


def lin_device(value: PythonObject, name: String, has_gpu: Bool, has_cpu: Bool,
               context: PythonObject) raises -> FunctionDevice:
    """UNPACK of a linalg function's `device` item: the shared admission,
    once per public call, before any work."""
    return function_device_admit(value, name, has_gpu, has_cpu, context, native_loaded_module("builtins"))


def lin_device_context(device: FunctionDevice, context: PythonObject) raises -> PythonObject:
    """The context one linalg call runs in: `context` itself for every
    device but an explicit "cpu" beside a GPU set (no read, no copy). For
    that one, a copy of the family context whose `family_lin_state` is the
    host image's own module state, built at the first such call and held in
    the family context (one dictionary read per later device="cpu" call)."""
    if not device.host_beside:
        return context
    var held = py_get(context, PythonObject(_LIN_CPU_CONTEXT))
    if held is not None:
        return held
    var nk = linalg_keys()
    var b = native_loaded_module("builtins")
    var scope = py_call(b.dict, context)
    var state = py_call(b.dict, py_item(context, nk[].k_family_lin_state))
    py_setitem(state, nk[].k__binding_cache, PythonObject(None))
    py_setitem(state, nk[].k__mode_binding, PythonObject(None))
    py_setitem(state, nk[].k__mode_cache, PythonObject(None))
    py_setitem(state, nk[].k__profile_cache, PythonObject(None))
    py_setitem(state, nk[].k__native_provider, PythonObject(None))
    py_setitem(scope, nk[].k_family_lin_state, state)
    py_setitem(context, PythonObject(_LIN_CPU_CONTEXT), scope)
    return scope


def lin_device_binding(device: FunctionDevice, context: PythonObject) raises -> PythonObject:
    """The linalg binding of one call: `lin_connected_binding` exactly for
    every device but an explicit "cpu" beside a GPU set; that one is the
    host image connected at import (identical mode only, refused in the
    estimators' words otherwise), published into the state of the context
    `lin_device_context` returned. It never runs the device image in its place."""
    if not device.host_beside:
        return lin_connected_binding(context)
    var binding = estimator_binding_device(device, PythonObject(None), "_mojolearn_linalg", context, False)
    return lin_publish_binding(binding, context)


def lin_publish_binding(binding: PythonObject, context: PythonObject) raises -> PythonObject:
    """Publish `binding` and its operations-table provider into the module
    state of `context` (the body `lin_connected_binding` always ran)."""
    var nk = linalg_keys()
    var state = lin_state(context)
    # Row N240: the module-state entries are written only when their value
    # changes (an unchanged binding and provider cost four identity reads,
    # not five dict writes per call). The binding and the provider factory
    # are still read live on every call.
    if py_get(state, nk[].k__binding_cache) is not binding:
        py_setitem(state, nk[].k__binding_cache, binding)
    if py_get(state, nk[].k__mode_binding) is not binding:
        py_setitem(state, nk[].k__mode_binding, binding)
        py_setitem(state, nk[].k__mode_cache, PythonObject(None))
        py_setitem(state, nk[].k__profile_cache, PythonObject(None))
    var factory = lin_attr(binding, "linalg_native_api_v1")
    var provider = PythonObject(None)
    if factory is not None:
        provider = neural_descriptor_reuse(py_get(state, nk[].k__native_provider), factory, "mojolearn.linalg.v1", LIN_OPERATION_COUNT)
    if py_get(state, nk[].k__native_provider) is not provider:
        py_setitem(state, nk[].k__native_provider, provider)
    return binding


def lin_mode(binding: PythonObject, context: PythonObject) raises -> String:
    """`numeric_mode()`: the loaded linalg binary's compiled mode, checked
    against the selected set and its profile version; cached in the
    module's `_mode_cache` / `_profile_cache` exactly as the Python did."""
    var nk = linalg_keys()
    var b = native_loaded_module("builtins")
    var state = lin_state(context)
    var cached = py_item(state, nk[].k__mode_cache)
    if cached is not None:
        return String(py=cached)
    var raw = PythonObject(lin_call(context, LIN_OP_LINALG_NUMERIC_MODE, List[Int](), List[Int]()))
    var compiled = String(py=py_get(py_item(py_item(context, nk[].k_backend_fields), nk[].k__CODE_MODE), raw, PythonObject("unknown")))
    var selected = lin_backend_mode(context)
    var build = lin_text(context, nk[].k__BUILD_SCRIPT)
    if compiled != selected:
        raise Error("RuntimeError: mojolearn.linalg: the loaded binary was compiled " + compiled
                    + " but this process selected the " + selected + " set -- a .so is in the wrong directory ("
                    + String(_lin_binary_path(binding)) + "); rebuild both sets with " + build)
    var version = lin_call(context, LIN_OP_LINALG_PROFILE_VERSION, List[Int](), List[Int]())
    var profile = py_item(state, nk[].k_PROFILE)
    var profile_version = Int(py=py_item(state, nk[].k_PROFILE_VERSION))
    var getter = lin_attr(binding, "linalg_numerical_profile")
    var actual = PythonObject(None)
    if getter is not None:
        actual = family_call(b.str, family_call(getter))
    elif version == profile_version:
        actual = profile
    var known = profile if version == profile_version else py_get(py_item(state, nk[].k__EXPERIMENT_PROFILES), PythonObject(version), PythonObject(None))
    var env_name = py_item(state, nk[].k__EXPECTED_PROFILE_ENV)
    var configured = String(getenv(String(py=env_name)))
    var expected = profile if configured == "" else PythonObject(configured)
    var ne = native_loaded_module("operator").ne
    if known is None or lin_truth(py_call(ne, actual, known)) or lin_truth(py_call(ne, actual, expected)):
        raise Error("RuntimeError: mojolearn.linalg: binary numerical profile " + family_repr(actual)
                    + " (version " + String(version) + ") does not match the supported expected profile "
                    + family_repr(expected) + ". The version names the arithmetic; unknown or stale "
                    + "profiles cannot be substituted. Experimental arms require their "
                    + "exact " + String(py=env_name) + " value and remain unqualified.")
    if version != profile_version and compiled != "identical":
        raise Error("RuntimeError: Experimental neural GEMM profiles require IDENTICAL mode")
    py_setitem(state, nk[].k__profile_cache, Python.tuple(actual, PythonObject(version)))
    py_setitem(state, nk[].k__mode_cache, PythonObject(compiled))
    return compiled


def lin_require_identical(binding: PythonObject, context: PythonObject) raises:
    """`require_identical()`."""
    var loaded = lin_mode(binding, context)
    if loaded != "identical":
        var nk = linalg_keys()
        raise Error("RuntimeError: mojolearn.linalg: this process loaded the " + loaded.upper()
                    + " build, which makes NO cross-vendor identity claim; " + lin_text(context, nk[].k_PROFILE)
                    + " is what the IDENTICAL build computes.\n"
                    + "  Select it with mojolearn.set_numeric_mode('identical') "
                    + "before the call, or set MOJOLEARN_NUMERIC_MODE=identical "
                    + "before importing mojolearn; "
                    + "the identical binary builds with\n      "
                    + "MOJOLEARN_NUMERIC_MODE=identical bash " + lin_text(context, nk[].k__BUILD_SCRIPT) + "\n"
                    + "  Or pass identical=False to say in the source that you want "
                    + "the " + loaded + " product and are making no identity claim about it.")


def lin_lowbit(binding: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_lowbit_binding()`: the identical build with the low-bit profile version this owner expects."""
    var nk = linalg_keys()
    lin_require_identical(binding, context)
    var v = lin_call(context, LIN_OP_LOWBIT_PROFILE_VERSION, List[Int](), List[Int]())
    var want = Int(py=py_item(lin_state(context), nk[].k_LOWBIT_PROFILE_VERSION))
    if v != want:
        raise Error("RuntimeError: mojolearn.linalg: the loaded extension implements low-bit "
                    + "profile version " + String(v) + ", this wrapper expects " + String(want))
    return binding


def lin_int15(binding: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_int15_binding()`."""
    var nk = linalg_keys()
    lin_require_identical(binding, context)
    var v = lin_call(context, LIN_OP_INT15_PROFILE_VERSION, List[Int](), List[Int]())
    var want = Int(py=py_item(lin_state(context), nk[].k_INT15_PROFILE_VERSION))
    if v != want:
        raise Error("RuntimeError: mojolearn.linalg: the loaded extension implements fifteen-bit "
                    + "profile version " + String(v) + ", this wrapper expects " + String(want))
    return binding


def lin_ints(*values: Int) raises -> PythonObject:
    """A Python list of ints (a params list). Read only by the decomposition
    door's two by-name entries (`svdvals`, `qr_r`,
    bindings/linalg_ops_decomp_door.mojo); nothing in this file builds one."""
    var out = Python.list()
    for i in range(len(values)):  # driver-loop: over (params: words of one params list): bounded by the call site
        _ = py_call(out.append, PythonObject(values[i]))
    return out


def lin_operand(x: PythonObject, name: String, context: PythonObject, admitted: Bool = False) raises -> AdmittedBuf:
    """`_operand(x, name)`: a float32 2-D buffer, refused by name otherwise,
    in C order (a layout copy at most, never a cast)."""
    # `admitted`: the family door already admitted this very argument (row
    # N237: lin_lowbit_operation and lin_decomp_operation admit every
    # operand before dispatch), so admitted_typed_once does not admit it again.
    var input_pin = AdmittedTypedPin()
    var input = admitted_typed_once(x, context, input_pin, admitted)
    return _lin_operand_admitted(input, input_pin^, name, context)


def _lin_operand_admitted(input: TensorView, var input_pin: AdmittedTypedPin, name: String, context: PythonObject) raises -> AdmittedBuf:
    var nk = linalg_keys()
    var profile = lin_text(context, nk[].k_PROFILE)
    var v = input.copy()
    if not lin_is_f32(v):
        raise Error("TypeError: mojolearn.linalg: " + name + " has dtype " + lin_dtype_name(input_pin.keep, v) + ", and "
                    + "only float32 is in this profile (" + profile + "; contract section 1 "
                    + "makes FP32 a hard requirement, and section 0.5 excludes FP16, "
                    + "BF16, TF32 and float64). Refused rather than cast, because a "
                    + "cast from float64 drops mantissa bits you may care about. "
                    + "Convert it yourself with np.asarray(" + name + ", dtype=np.float32) "
                    + "if that is what you want.")
    if len(v.shape) != 2:
        raise Error("mojolearn.linalg: " + name + " must be 2-D, got " + String(len(v.shape)) + "-D shape "
                    + shape_text(v.shape) + ". A vector product is OP_NT at n == 1 (contract 0.1); "
                    + "pass it as a 2-D array of shape (n, k) or (k, 1).")
    if v.count() == 0:
        raise Error("mojolearn.linalg: " + name + " has shape " + shape_text(v.shape) + " and no elements. "
                    + "Contract section 8 does specify the degenerate shapes (k == 0 "
                    + "writes +0.0 into every cell; m == 0 or n == 0 writes nothing), "
                    + "but no gate in this tree has run them through the Python "
                    + "surface, so they are refused here rather than answered "
                    + "unchecked.")
    return admitted_typed_c(input, input_pin^, "<f4", name, context)


def lin_operand_bits(x: PythonObject, name: String, context: PythonObject, admitted: Bool = False) raises -> AdmittedBuf:
    """`_operand_bits(x, name)`: a 2-D uint16 buffer of bf16 bits in C order."""
    # `admitted`: as lin_operand (row N237).
    var input_pin = AdmittedTypedPin()
    var input = admitted_typed_once(x, context, input_pin, admitted)
    return _lin_bits_admitted(input, input_pin^, name, context)


def _lin_bits_admitted(input: TensorView, var input_pin: AdmittedTypedPin, name: String, context: PythonObject) raises -> AdmittedBuf:
    var v = input.copy()
    var dtype = lin_dtype_name(input_pin.keep, v)
    if dtype != "uint16":
        raise Error("TypeError: mojolearn.linalg: " + name + " has dtype " + dtype + "; the "
                    + "bf16 profile takes bf16 BITS in a uint16 buffer, which "
                    + "to_bf16() produces. Refused rather than reinterpreted.")
    if len(v.shape) != 2:
        raise Error("mojolearn.linalg: " + name + " must be 2-D, got " + String(len(v.shape)) + "-D")
    if v.count() == 0:
        raise Error("mojolearn.linalg: " + name + " has no elements")
    return admitted_typed_c(input, input_pin^, "<u2", name, context)


# lane fix-output-zero-fill (2026-11-14): every new output in this door is
# admitted_unfilled (was admitted_zeros). Each kernel writes its whole output
# before it is returned: the GEMMs download all m*n cells (degenerate shapes
# are refused before any launch), the bf16/f16 conversions write n items,
# quantize writes every code and every row exponent, dequantize every cell.
# The zero fill of matmul's output was 80.8% of the gemm row (L40S profile
# 80a7f8aca3); 0.8.55 filled it too (_buffer.empty, array([0]) * size).
# No output bit changes: the zeros were overwritten before any read.
def lin_out(out_buf: PythonObject, m: Int, n: Int, who: String, context: PythonObject) raises -> AdmittedBuf:
    """`_out_or_new(out, m, n, who)`."""
    if out_buf is None:
        return admitted_unfilled([m, n], "<f4", context)
    var input_pin = AdmittedTypedPin()
    var input = admitted_typed_once(out_buf, context, input_pin)
    var v = input.copy()
    if not lin_is_f32(v):
        raise Error("TypeError: mojolearn.linalg." + who + ": out must be float32")
    if len(v.shape) != 2 or v.shape[0] != m or v.shape[1] != n:
        raise Error("mojolearn.linalg." + who + ": out has shape " + shape_text(v.shape) + ", want ("
                    + String(m) + ", " + String(n) + ")")
    if not v.c_contig or v.readonly:
        raise Error("mojolearn.linalg." + who + ": out must be C-contiguous and writable")
    return admitted_typed_output(input, input_pin^)


# ---------------------------------------------------------------------------
# the public functions


def lin_matmul_op(ta: Bool, tb: Bool, context: PythonObject) raises -> Int:
    """The GEMM op of matmul's two transpose flags (OP_NN, OP_NT or OP_TN), the both-transposed refusal in its
    words. Shared by the host-array door and the device-array door (lane vdf-linalg4)."""
    var nk = linalg_keys()
    var profile = lin_text(context, nk[].k_PROFILE)
    if ta and tb:
        raise Error("mojolearn.linalg.matmul: transpose_a=True with transpose_b=True "
                    + "is refused. " + profile + " has exactly three operations (contract "
                    + "section 0.1: OP_NN, OP_NT, OP_TN) and a.T @ b.T is not one of "
                    + "them. This function will not materialize a transpose to fake a "
                    + "fourth. If you want it, write the identity yourself:\n"
                    + "    np.ascontiguousarray(np.asarray(matmul(b, a)).T)\n"
                    + "which is (b @ a).T == a.T @ b.T, runs as OP_NN, and leaves the "
                    + "extra step visible in your source where you can price it.")
    return OP_NN if not ta and not tb else (OP_NT if tb else OP_TN)


def _lin_matmul(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`matmul(a, b, *, transpose_a, transpose_b, out, identical)`."""
    var binding = args[0]
    if lin_truth(args[6]):
        lin_require_identical(binding, context)
    else:
        _ = lin_mode(binding, context)
    var ta = lin_truth(args[3])
    var tb = lin_truth(args[4])
    var op = lin_matmul_op(ta, tb, context)
    var a = lin_operand(args[1], "a", context, True)
    var bb = lin_operand(args[2], "b", context, True)
    var m = a.dim(1) if ta else a.dim(0)
    var k = a.dim(0) if ta else a.dim(1)
    var n = bb.dim(0) if tb else bb.dim(1)
    var kb = bb.dim(1) if tb else bb.dim(0)
    if k != kb:
        raise Error("mojolearn.linalg.matmul: contracted extents disagree, a gives "
                    + "k=" + String(k) + " and b gives k=" + String(kb) + " (a.shape=" + a.text() + ", "
                    + "b.shape=" + bb.text() + ", transpose_a=" + ("True" if ta else "False") + ", "
                    + "transpose_b=" + ("True" if tb else "False") + "). The row-major shapes for each "
                    + "op are in this function's table and in contract section 0.1.")
    var out_obj = args[5]
    var out: AdmittedBuf
    if out_obj is None:
        out = admitted_unfilled([m, n], "<f4", context)
    else:
        var output_pin = AdmittedTypedPin()
        var output = admitted_typed_once(out_obj, context, output_pin)
        var v = output.copy()
        if not lin_is_f32(v):
            raise Error("TypeError: mojolearn.linalg.matmul: out has dtype " + lin_dtype_name(out_obj, v)
                        + ", and the profile's output is float32 (contract section 1). Refused rather than cast on "
                        + "the way out.")
        if len(v.shape) != 2 or v.shape[0] != m or v.shape[1] != n:
            raise Error("mojolearn.linalg.matmul: out has shape " + shape_text(v.shape) + ", want ("
                        + String(m) + ", " + String(n) + ")")
        if not v.c_contig:
            raise Error("mojolearn.linalg.matmul: out must be C-contiguous; the "
                        + "device writes it directly (contract section 2). A "
                        + "non-contiguous out cannot be written in place, and copying "
                        + "into it afterwards would make `out` a lie about where the "
                        + "result was produced.")
        if v.readonly:
            raise Error("mojolearn.linalg.matmul: out is read-only, refusing to "
                        + "write to it; the device writes `out` directly (contract "
                        + "section 2) and a read-only buffer handed to it is memory "
                        + "corruption, not an exception.")
        out = admitted_typed_output(output, output_pin^)
        _ = v  # keep the Python object alive through the native call above
    # ORDER MATCHES bindings/_mojolearn_linalg.mojo::gemm_binding: the output
    # address first, then a and b; params (m, n, k, op).
    _ = lin_call(context, LIN_OP_GEMM, [out.addr, a.addr, bb.addr], [m, n, k, op])
    _ = bb  # keep the Python object alive through the native call above
    _ = a  # keep the Python object alive through the native call above
    _ = a^
    _ = bb^
    return out.keep


def _lin_fused_bias(x: PythonObject, n: Int, context: PythonObject) raises -> AdmittedBuf:
    """The bias operand of `matmul_fused`: `n` float32 values in C order.
    The former shell reshaped a NumPy bias to (1, n) and refused any other
    count; the admitted buffer is read as its C-order values, so any rank
    holding exactly `n` float32 values is the same bias."""
    var input_pin = AdmittedTypedPin()
    var input = admitted_typed_once(x, context, input_pin, True)
    var v = input.copy()
    if not lin_is_f32(v):
        var nk = linalg_keys()
        raise Error("TypeError: mojolearn.linalg: bias has dtype " + lin_dtype_name(input_pin.keep, v) + ", and "
                    + "only float32 is in this profile (" + lin_text(context, nk[].k_PROFILE) + "; contract section 1 "
                    + "makes FP32 a hard requirement, and section 0.5 excludes FP16, "
                    + "BF16, TF32 and float64). Refused rather than cast, because a "
                    + "cast from float64 drops mantissa bits you may care about. "
                    + "Convert it yourself with np.asarray(bias, dtype=np.float32) "
                    + "if that is what you want.")
    if v.count() != n:
        raise Error("mojolearn.linalg_fast.matmul_fused: bias must hold n=" + String(n) + " values")
    return admitted_typed_c(input, input_pin^, "<f4", "bias", context)


def _lin_matmul_fused(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`linalg_fast.matmul_fused(a, b, *, bias, residual, activation,
    transpose_a, transpose_b, out)`: args are (binding, a, b, bias, residual,
    activation, transpose_a, transpose_b, out). The refusals of the former
    Python shell, in its order and words (lane py-classical-linalg)."""
    var binding = args[0]
    _ = lin_mode(binding, context)
    if lin_attr(binding, "gemm_fused") is None:
        raise Error("NotImplementedError: mojolearn.linalg_fast.matmul_fused: the loaded linalg binding has no "
                    + "gemm_fused entry; it is a FAST, Apple-GPU build with "
                    + "-D MOJOLEARN_AFN_GEMM_EPILOGUE (gemm/afn_apple_fast.mojo). Use "
                    + "mojolearn.linalg.matmul(..., identical=False) and apply the epilogue "
                    + "yourself.")
    var ta = lin_truth(args[6])
    var tb = lin_truth(args[7])
    if ta and tb:
        raise Error("mojolearn.linalg_fast.matmul_fused: transpose_a=True with "
                    + "transpose_b=True is refused (three operations: NN, NT, TN)")
    var op = OP_NN if not ta and not tb else (OP_NT if tb else OP_TN)
    var bias_obj = args[3]
    var residual_obj = args[4]
    var activation = args[5]
    if residual_obj is not None and activation is not None:
        raise Error("mojolearn.linalg_fast.matmul_fused: residual and activation "
                    + "cannot be combined; one epilogue per store")
    var epi = 0
    if activation is None:
        if bias_obj is not None or residual_obj is not None:
            epi = 2 if residual_obj is not None else 1
    else:
        var name = String("")
        if py_type(activation) is py_type(PythonObject("")):
            name = String(py=activation)
        if name != "silu" and name != "gelu":
            raise Error("mojolearn.linalg_fast.matmul_fused: activation must be None, "
                        + "'silu' or 'gelu', got " + family_repr(activation))
        epi = 3 if name == "silu" else 4
    if epi != 0 and bias_obj is None:
        raise Error("mojolearn.linalg_fast.matmul_fused: every epilogue needs bias")
    var a = lin_operand(args[1], "a", context, True)
    var bb = lin_operand(args[2], "b", context, True)
    var m = a.dim(1) if ta else a.dim(0)
    var k = a.dim(0) if ta else a.dim(1)
    var n = bb.dim(0) if tb else bb.dim(1)
    var kb = bb.dim(1) if tb else bb.dim(0)
    if k != kb:
        raise Error("mojolearn.linalg_fast.matmul_fused: contracted extents disagree, "
                    + "a gives k=" + String(k) + " and b gives k=" + String(kb))
    var bias = AdmittedBuf(PythonObject(None), 0, List[Int]())
    var residual = AdmittedBuf(PythonObject(None), 0, List[Int]())
    if epi != 0:
        bias = _lin_fused_bias(bias_obj, n, context)
    if epi == 2:
        residual = lin_operand(residual_obj, "residual", context, True)
        if residual.dim(0) != m or residual.dim(1) != n:
            raise Error("mojolearn.linalg_fast.matmul_fused: residual must be (" + String(m) + ", "
                        + String(n) + ")")
    var out_obj = args[8]
    var out: AdmittedBuf
    if out_obj is None:
        out = admitted_unfilled([m, n], "<f4", context)
    else:
        var output_pin = AdmittedTypedPin()
        var output = admitted_typed_once(out_obj, context, output_pin, True)
        var v = output.copy()
        if (not lin_is_f32(v) or len(v.shape) != 2 or v.shape[0] != m or v.shape[1] != n
                or not v.c_contig or v.readonly):
            raise Error("mojolearn.linalg_fast.matmul_fused: out must be a writable "
                        + "C-contiguous float32 (" + String(m) + ", " + String(n) + ") buffer")
        out = admitted_typed_output(output, output_pin^)
    # `params` is, in this exact order (mirrored word for word in
    # `gemm/linalg_fused_run.mojo::gemm_fused_native`): m, n, k, op, epi;
    # addresses c, a, b, bias, residual.
    _ = lin_call(context, LIN_OP_GEMM_FUSED, [out.addr, a.addr, bb.addr, bias.addr, residual.addr], [m, n, k, op, epi])
    _ = residual  # keep the Python object alive through the native call above
    _ = bias  # keep the Python object alive through the native call above
    _ = bb  # keep the Python object alive through the native call above
    _ = a  # keep the Python object alive through the native call above
    _ = residual^
    _ = bias^
    _ = a^
    _ = bb^
    return out.keep


def _lin_to_bf16(args: PythonObject, context: PythonObject) raises -> PythonObject:
    var a = admit_typed(args[1], "<f4", -1, "x", context, True)
    var n = a.count()
    if n == 0:
        raise Error("mojolearn.linalg.to_bf16: x has no elements")
    var out = admitted_unfilled(a.shape, "<u2", context)
    _ = lin_lowbit(args[0], context)
    _ = lin_call(context, LIN_OP_TO_BF16, [out.addr, a.addr], [n])
    _ = out  # keep the Python object alive through the native call above
    _ = a  # keep the Python object alive through the native call above
    _ = a^
    return out.keep


def _lin_from_bf16(args: PythonObject, context: PythonObject) raises -> PythonObject:
    var a = admit_typed(args[1], "<u2", -1, "bits", context, True)
    var n = a.count()
    if n == 0:
        raise Error("mojolearn.linalg.from_bf16: bits has no elements")
    var out = admitted_unfilled(a.shape, "<f4", context)
    _ = lin_lowbit(args[0], context)
    _ = lin_call(context, LIN_OP_FROM_BF16, [out.addr, a.addr], [n])
    _ = out  # keep the Python object alive through the native call above
    _ = a  # keep the Python object alive through the native call above
    _ = a^
    return out.keep


def _lin_from_f16(args: PythonObject, context: PythonObject) raises -> PythonObject:
    var a = admit_typed(args[1], "<u2", -1, "bits", context)
    var n = a.count()
    if n == 0:
        raise Error("mojolearn.linalg.from_f16: bits has no elements")
    var out = admitted_unfilled(a.shape, "<f4", context)
    _ = lin_lowbit(args[0], context)
    _ = lin_call(context, LIN_OP_FROM_F16, [out.addr, a.addr], [n])
    _ = out  # keep the Python object alive through the native call above
    _ = a  # keep the Python object alive through the native call above
    _ = a^
    return out.keep


def lin_quantize_int8(binding: PythonObject, a: AdmittedBuf, context: PythonObject) raises -> PythonObject:
    """`quantize_int8` of an admitted operand: (codes, exponents)."""
    var rows = a.dim(0)
    var cols = a.dim(1)
    var codes = admitted_unfilled([rows, cols], "<i1", context)
    var exps = admitted_unfilled([rows], "<i4", context)
    _ = lin_lowbit(binding, context)
    _ = lin_call(context, LIN_OP_QUANTIZE_INT8, [codes.addr, exps.addr, a.addr], [rows, cols])
    _ = exps  # keep the Python object alive through the native call above
    _ = codes  # keep the Python object alive through the native call above
    return Python.tuple(codes.keep, exps.keep)


def _lin_quantize_int8(args: PythonObject, context: PythonObject) raises -> PythonObject:
    var a = lin_operand(args[1], "x", context, True)
    var result = lin_quantize_int8(args[0], a, context)
    _ = a^
    return result


def _lin_dequantize_int8(args: PythonObject, context: PythonObject) raises -> PythonObject:
    var q = admit_typed(args[1], "<i1", 2, "codes", context, True)
    var e = admit_typed(args[2], "<i4", 1, "exponents", context, True)
    var rows = q.dim(0)
    var cols = q.dim(1)
    if e.rank() != 1 or e.dim(0) != rows:
        raise Error("mojolearn.linalg.dequantize_int8: exponents has shape " + e.text() + ", want ("
                    + String(rows) + ",)")
    var out = admitted_unfilled([rows, cols], "<f4", context)
    _ = lin_lowbit(args[0], context)
    _ = lin_call(context, LIN_OP_DEQUANTIZE_INT8, [out.addr, q.addr, e.addr], [rows, cols])
    _ = out  # keep the Python object alive through the native call above
    _ = q  # keep the Python object alive through the native call above
    _ = e  # keep the Python object alive through the native call above
    _ = q^
    _ = e^
    return out.keep


def _lin_int8_operand(binding: PythonObject, x: PythonObject, name: String, context: PythonObject,
                      mut q: AdmittedBuf, mut e: AdmittedBuf) raises:
    """`_int8_operand(x, name)`: a (codes, exponents) pair, or a float32
    matrix quantized by `quantize_int8`."""
    var b = native_loaded_module("builtins")
    if py_type(x) is b.tuple and len(x) == 2:
        q = admit_typed(x[0], "<i1", 2, name + " codes", context)
        e = admit_typed(x[1], "<i4", 1, name + " exponents", context)
        if e.rank() != 1 or e.dim(0) != q.dim(0):
            raise Error("mojolearn.linalg.matmul_int8: " + name + " exponents has shape " + e.text()
                        + ", want (" + String(q.dim(0)) + ",)")
        return
    var a = lin_operand(x, name, context)
    var pair = lin_quantize_int8(binding, a, context)
    _ = a^
    q = admit_typed(pair[0], "<i1", 2, name + " codes", context)
    e = admit_typed(pair[1], "<i4", 1, name + " exponents", context)


def _lin_matmul_int8(args: PythonObject, context: PythonObject) raises -> PythonObject:
    var binding = args[0]
    var qa = AdmittedBuf(PythonObject(None), 0, List[Int]())
    var ea = AdmittedBuf(PythonObject(None), 0, List[Int]())
    var qb = AdmittedBuf(PythonObject(None), 0, List[Int]())
    var eb = AdmittedBuf(PythonObject(None), 0, List[Int]())
    _lin_int8_operand(binding, args[1], "a", context, qa, ea)
    _lin_int8_operand(binding, args[2], "b", context, qb, eb)
    var m = qa.dim(0)
    var k = qa.dim(1)
    var n = qb.dim(0)
    var kb = qb.dim(1)
    if k != kb:
        raise Error("mojolearn.linalg.matmul_int8: contracted extents differ, a "
                    + "gives k=" + String(k) + " and b gives k=" + String(kb))
    var out = lin_out(args[3], m, n, "matmul_int8", context)
    # `params` is `[m, n, k]`, mirrored in `gemm_int8_binding`.
    _ = lin_lowbit(binding, context)
    _ = lin_call(context, LIN_OP_GEMM_INT8, [out.addr, qa.addr, ea.addr, qb.addr, eb.addr], [m, n, k])
    _ = out  # keep the Python object alive through the native call above
    _ = qa  # keep the Python object alive through the native call above
    _ = eb  # keep the Python object alive through the native call above
    _ = qb  # keep the Python object alive through the native call above
    _ = ea  # keep the Python object alive through the native call above
    _ = qa^
    _ = ea^
    _ = qb^
    _ = eb^
    return out.keep


def _lin_op_and_shape(a: AdmittedBuf, bb: AdmittedBuf, transpose_a: PythonObject, transpose_b: PythonObject,
                      who: String, mut op: Int, mut m: Int, mut n: Int, mut k: Int) raises:
    """`_op_and_shape(a.shape, b.shape, transpose_a, transpose_b, who)`."""
    var ta = lin_truth(transpose_a)
    var tb = lin_truth(transpose_b)
    if ta and tb:
        raise Error("mojolearn.linalg." + who + ": transpose_a and transpose_b together "
                    + "is not one of the contract's three operations (section 0.1); "
                    + "write (b @ a).T with the flags swapped, in your own source.")
    var kb: Int
    if ta:
        op = OP_TN
        k = a.dim(0)
        m = a.dim(1)
        kb = bb.dim(0)
        n = bb.dim(1)
    elif tb:
        op = OP_NT
        m = a.dim(0)
        k = a.dim(1)
        n = bb.dim(0)
        kb = bb.dim(1)
    else:
        op = OP_NN
        m = a.dim(0)
        k = a.dim(1)
        kb = bb.dim(0)
        n = bb.dim(1)
    if k != kb:
        raise Error("mojolearn.linalg." + who + ": contracted extents differ, a gives "
                    + "k=" + String(k) + " and b gives k=" + String(kb) + " (a.shape=" + a.text()
                    + ", b.shape=" + bb.text() + ", transpose_a=" + family_str(transpose_a)
                    + ", transpose_b=" + family_str(transpose_b) + ")")


def _lin_matmul_bf16(args: PythonObject, context: PythonObject) raises -> PythonObject:
    var binding = args[0]
    var input_pin = AdmittedTypedPin()
    var input = admitted_typed_once(args[1], context, input_pin)
    var a_bits = lin_dtype_name(input_pin.keep, input) == "uint16"
    var a = _lin_bits_admitted(input, input_pin^, "a", context) if a_bits else _lin_operand_admitted(input, input_pin^, "a", context)
    var bb = lin_operand_bits(args[2], "b", context, True)
    var op = 0
    var m = 0
    var n = 0
    var k = 0
    _lin_op_and_shape(a, bb, args[3], args[4], "matmul_bf16", op, m, n, k)
    var out = lin_out(args[5], m, n, "matmul_bf16", context)
    # params (mirrored in gemm_bf16_binding): m, n, k, op, a_bf16
    _ = lin_lowbit(binding, context)
    _ = lin_call(context, LIN_OP_GEMM_BF16, [out.addr, a.addr, bb.addr], [m, n, k, op, 1 if a_bits else 0])
    _ = out  # keep the Python object alive through the native call above
    _ = a  # keep the Python object alive through the native call above
    _ = bb  # keep the Python object alive through the native call above
    _ = a^
    _ = bb^
    return out.keep


def lin_quantize_int15(binding: PythonObject, a: AdmittedBuf, context: PythonObject) raises -> PythonObject:
    """`quantize_int15` of an admitted operand: (hi, lo, exponents)."""
    var rows = a.dim(0)
    var cols = a.dim(1)
    var hi = admitted_unfilled([rows, cols], "<i1", context)
    var lo = admitted_unfilled([rows, cols], "<i1", context)
    var exps = admitted_unfilled([rows], "<i4", context)
    # addrs (mirrored in quantize_int15_binding): hi, lo, exponents, x
    _ = lin_int15(binding, context)
    _ = lin_call(context, LIN_OP_QUANTIZE_INT15, [hi.addr, lo.addr, exps.addr, a.addr], [rows, cols])
    _ = exps  # keep the Python object alive through the native call above
    _ = lo  # keep the Python object alive through the native call above
    _ = hi  # keep the Python object alive through the native call above
    return Python.tuple(hi.keep, lo.keep, exps.keep)


def _lin_quantize_int15(args: PythonObject, context: PythonObject) raises -> PythonObject:
    var a = lin_operand(args[1], "x", context, True)
    var result = lin_quantize_int15(args[0], a, context)
    _ = a^
    return result


def _lin_int15_planes(x: PythonObject, name: String, who: String, context: PythonObject,
                      mut hi: AdmittedBuf, mut lo: AdmittedBuf, mut e: AdmittedBuf) raises:
    """`_int15_planes(x, name, who)`."""
    hi = admit_typed(x[0], "<i1", 2, name + " hi", context)
    lo = admit_typed(x[1], "<i1", 2, name + " lo", context)
    e = admit_typed(x[2], "<i4", 1, name + " exponents", context)
    if lo.rank() != hi.rank() or lo.dim(0) != hi.dim(0) or lo.dim(1) != hi.dim(1):
        raise Error("mojolearn.linalg." + who + ": " + name + " lo has shape " + lo.text() + ", hi has " + hi.text())
    if e.rank() != 1 or e.dim(0) != hi.dim(0):
        raise Error("mojolearn.linalg." + who + ": " + name + " exponents has shape " + e.text() + ", want ("
                    + String(hi.dim(0)) + ",)")


def _lin_dequantize_int15(args: PythonObject, context: PythonObject) raises -> PythonObject:
    var h = AdmittedBuf(PythonObject(None), 0, List[Int]())
    var l = AdmittedBuf(PythonObject(None), 0, List[Int]())
    var e = AdmittedBuf(PythonObject(None), 0, List[Int]())
    _lin_int15_planes(Python.tuple(args[1], args[2], args[3]), "operand", "dequantize_int15", context, h, l, e)
    var rows = h.dim(0)
    var cols = h.dim(1)
    var out = admitted_unfilled([rows, cols], "<f4", context)
    # addrs: y, hi, lo, exponents (mirrored in the binding)
    _ = lin_int15(args[0], context)
    _ = lin_call(context, LIN_OP_DEQUANTIZE_INT15, [out.addr, h.addr, l.addr, e.addr], [rows, cols])
    _ = out  # keep the Python object alive through the native call above
    _ = e  # keep the Python object alive through the native call above
    _ = l  # keep the Python object alive through the native call above
    _ = h  # keep the Python object alive through the native call above
    _ = h^
    _ = l^
    _ = e^
    return out.keep


def _lin_int15_operand(binding: PythonObject, x: PythonObject, name: String, context: PythonObject,
                       mut hi: AdmittedBuf, mut lo: AdmittedBuf, mut e: AdmittedBuf) raises:
    """`_int15_operand(x, name)`."""
    var b = native_loaded_module("builtins")
    if py_type(x) is b.tuple and len(x) == 3:
        _lin_int15_planes(x, name, "matmul_int15", context, hi, lo, e)
        return
    if py_type(x) is b.tuple:
        raise Error("TypeError: mojolearn.linalg.matmul_int15: " + name + " is a tuple of " + String(len(x))
                    + "; a fifteen-bit operand is (hi, lo, exponents) from quantize_int15, or a float32 matrix")
    var a = lin_operand(x, name, context)
    var planes = lin_quantize_int15(binding, a, context)
    _ = a^
    hi = admit_typed(planes[0], "<i1", 2, name + " hi", context)
    lo = admit_typed(planes[1], "<i1", 2, name + " lo", context)
    e = admit_typed(planes[2], "<i4", 1, name + " exponents", context)


def _lin_matmul_int15(args: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = linalg_keys()
    var binding = args[0]
    var ah = AdmittedBuf(PythonObject(None), 0, List[Int]())
    var al = AdmittedBuf(PythonObject(None), 0, List[Int]())
    var ea = AdmittedBuf(PythonObject(None), 0, List[Int]())
    var bh = AdmittedBuf(PythonObject(None), 0, List[Int]())
    var bl = AdmittedBuf(PythonObject(None), 0, List[Int]())
    var eb = AdmittedBuf(PythonObject(None), 0, List[Int]())
    _lin_int15_operand(binding, args[1], "a", context, ah, al, ea)
    _lin_int15_operand(binding, args[2], "b", context, bh, bl, eb)
    var m = ah.dim(0)
    var k = ah.dim(1)
    var n = bh.dim(0)
    var kb = bh.dim(1)
    if k != kb:
        raise Error("mojolearn.linalg.matmul_int15: contracted extents differ, a "
                    + "gives k=" + String(k) + " and b gives k=" + String(kb))
    var bound = py_item(lin_state(context), nk[].k_INT15_MAX_K)
    if py_type(bound) is not py_type(PythonObject(0)):
        raise Error("TypeError: linalg int15 extent metadata requires an exact integer")
    var max_k = Int(py=bound)
    if k > max_k:
        raise Error("mojolearn.linalg.matmul_int15: k=" + String(k) + " is above " + String(max_k) + ", the "
                    + "largest contracted extent " + lin_text(context, nk[].k_PROFILE_INT15) + " accepts (contract W-4)")
    var out = lin_out(args[3], m, n, "matmul_int15", context)
    # addrs (mirrored in gemm_int15_binding): c, a hi, a lo, a exponents,
    # b hi, b lo, b exponents; THE OUTPUT ADDRESS COMES FIRST
    _ = lin_int15(binding, context)
    _ = lin_call(context, LIN_OP_GEMM_INT15, [out.addr, ah.addr, al.addr, ea.addr, bh.addr, bl.addr, eb.addr], [m, n, k])
    _ = out  # keep the Python object alive through the native call above
    _ = eb  # keep the Python object alive through the native call above
    _ = bl  # keep the Python object alive through the native call above
    _ = bh  # keep the Python object alive through the native call above
    _ = ea  # keep the Python object alive through the native call above
    _ = al  # keep the Python object alive through the native call above
    _ = ah  # keep the Python object alive through the native call above
    _ = ah^
    _ = al^
    _ = ea^
    _ = bh^
    _ = bl^
    _ = eb^
    return out.keep


def _lin_public_name(operation: Int) -> String:
    """The public name a `device=` refusal of `operation` leads with."""
    if operation == LINOPS_OPN_FAMILY_LIN_MATMUL:
        return "mojolearn.linalg.matmul"
    if operation == LINOPS_OPN_FAMILY_LIN_MATMUL_FUSED:
        return "mojolearn.linalg_fast.matmul_fused"
    if operation == LINOPS_OPN_FAMILY_LIN_TO_BF16:
        return "mojolearn.linalg.to_bf16"
    if operation == LINOPS_OPN_FAMILY_LIN_FROM_BF16:
        return "mojolearn.linalg.from_bf16"
    if operation == LINOPS_OPN_FAMILY_LIN_FROM_F16:
        return "mojolearn.linalg.from_f16"
    if operation == LINOPS_OPN_FAMILY_LIN_QUANTIZE_INT8:
        return "mojolearn.linalg.quantize_int8"
    if operation == LINOPS_OPN_FAMILY_LIN_DEQUANTIZE_INT8:
        return "mojolearn.linalg.dequantize_int8"
    if operation == LINOPS_OPN_FAMILY_LIN_MATMUL_BF16:
        return "mojolearn.linalg.matmul_bf16"
    if operation == LINOPS_OPN_FAMILY_LIN_MATMUL_INT8:
        return "mojolearn.linalg.matmul_int8"
    if operation == LINOPS_OPN_FAMILY_LIN_QUANTIZE_INT15:
        return "mojolearn.linalg.quantize_int15"
    if operation == LINOPS_OPN_FAMILY_LIN_DEQUANTIZE_INT15:
        return "mojolearn.linalg.dequantize_int15"
    return "mojolearn.linalg.matmul_int15"


def lin_dev_operation(operation: Int) -> Bool:
    """The public operations with a device-array form (lane vdf-linalg4)."""
    return (operation == LINOPS_OPN_FAMILY_LIN_MATMUL or operation == LINOPS_OPN_FAMILY_LIN_TO_BF16
            or operation == LINOPS_OPN_FAMILY_LIN_FROM_BF16 or operation == LINOPS_OPN_FAMILY_LIN_QUANTIZE_INT8
            or operation == LINOPS_OPN_FAMILY_LIN_DEQUANTIZE_INT8 or operation == LINOPS_OPN_FAMILY_LIN_MATMUL_BF16
            or operation == LINOPS_OPN_FAMILY_LIN_MATMUL_INT8)


def lin_lowbit_dev(operation: Int, args: PythonObject, controls: PythonObject, binding: PythonObject,
                   held: PythonObject, context: PythonObject) raises -> PythonObject:
    """A linalg operation whose operands are TrainDeviceArrays: read in place, the result a new TrainDeviceArray."""
    if operation == LINOPS_OPN_FAMILY_LIN_MATMUL:
        return _lin_matmul_dev(args, controls, binding, held, context)
    if operation == LINOPS_OPN_FAMILY_LIN_TO_BF16 or operation == LINOPS_OPN_FAMILY_LIN_FROM_BF16:
        return _lin_bf16_convert_dev(operation == LINOPS_OPN_FAMILY_LIN_TO_BF16, args, binding, held, context)
    if operation == LINOPS_OPN_FAMILY_LIN_QUANTIZE_INT8:
        var entries = lin_dev_entries(args[1], held)
        var x = lin_dev_operand(args[1], "x", "quantize_int8", entries, held, TRAIN_DEV_F32, 2)
        return _lin_quantize_int8_dev(binding, x, entries, held, context)
    if operation == LINOPS_OPN_FAMILY_LIN_DEQUANTIZE_INT8:
        return _lin_dequantize_int8_dev(args, binding, held, context)
    if operation == LINOPS_OPN_FAMILY_LIN_MATMUL_BF16:
        return _lin_matmul_bf16_dev(args, controls, binding, held, context)
    if operation == LINOPS_OPN_FAMILY_LIN_MATMUL_INT8:
        return _lin_matmul_int8_dev(args, binding, held, context)
    raise Error("TypeError: mojolearn.linalg." + _lin_public_name(operation) + " takes host arrays only")


def _lin_matmul_dev(args: PythonObject, controls: PythonObject, binding: PythonObject, held: PythonObject,
                    context: PythonObject) raises -> PythonObject:
    """`matmul(a, b, *, transpose_a, transpose_b, out, identical)` on TrainDeviceArrays (lane vdf-linalg4): a and b
    read in place by the device entry (LIN_OP_GEMM_DEV, gemm/linalg_dev_arrays.mojo `gemm_dev_native`: the host entry's
    one kernel on the same words), the product a new TrainDeviceArray. The host door's refusals in its words; `out` is
    refused (the result is a new device array)."""
    if lin_truth(py_item(controls, PythonObject(6))):
        lin_require_identical(binding, context)
    else:
        _ = lin_mode(binding, context)
    var ta = lin_truth(py_item(controls, PythonObject(3)))
    var tb = lin_truth(py_item(controls, PythonObject(4)))
    var op = lin_matmul_op(ta, tb, context)
    lin_dev_out(args[5], "matmul")
    if not lin_is_device_array(args[1], held):
        raise Error("TypeError: mojolearn.linalg.matmul: device and host arrays are mixed; pass every operand "
                    + "through train_to_device, or none")
    var entries = lin_dev_entries(args[1], held)
    var a = lin_dev_operand(args[1], "a", "matmul", entries, held, TRAIN_DEV_F32, 2)
    var bb = lin_dev_operand(args[2], "b", "matmul", entries, held, TRAIN_DEV_F32, 2)
    var m = a.dim(1) if ta else a.dim(0)
    var k = a.dim(0) if ta else a.dim(1)
    var n = bb.dim(0) if tb else bb.dim(1)
    var kb = bb.dim(1) if tb else bb.dim(0)
    if k != kb:
        raise Error("mojolearn.linalg.matmul: contracted extents disagree, a gives "
                    + "k=" + String(k) + " and b gives k=" + String(kb) + " (a.shape=" + a.text() + ", "
                    + "b.shape=" + bb.text() + ", transpose_a=" + ("True" if ta else "False") + ", "
                    + "transpose_b=" + ("True" if tb else "False") + "). The row-major shapes for each "
                    + "op are in this function's table and in contract section 0.1.")
    var out = lin_dev_new(entries, held, [m, n], TRAIN_DEV_F32)
    # ORDER as LIN_OP_GEMM: the product's handle, then a and b; params (m, n, k, op).
    _ = lin_call(context, LIN_OP_GEMM_DEV, [lin_dev_handle(out), a.handle, bb.handle], [m, n, k, op])
    _ = args
    return out


def _lin_dev_first(args: PythonObject, held: PythonObject, who: String) raises -> PythonObject:
    """The first operand of a device-array call (a device array itself, or the first item of a (codes, exponents)
    pair): the binding the call's results belong to."""
    var b = native_loaded_module("builtins")
    var x = args[1]
    if py_type(x) is b.tuple and len(x) == 2:
        x = x[0]
    if not lin_is_device_array(x, held):
        raise Error("TypeError: mojolearn.linalg." + who + ": device and host arrays are mixed; pass every operand "
                    + "through train_to_device, or none")
    return x


def _lin_bf16_convert_dev(narrow: Bool, args: PythonObject, binding: PythonObject, held: PythonObject,
                          context: PythonObject) raises -> PythonObject:
    """`to_bf16(x)` / `from_bf16(bits)` on a TrainDeviceArray (lane vdf-linalg4): converted on the device into a new
    device array of the same shape (uint16 bf16 bits / float32), the host door's refusals in its words."""
    var who = String("to_bf16") if narrow else String("from_bf16")
    var entries = lin_dev_entries(args[1], held)
    var a = lin_dev_operand(args[1], "x" if narrow else "bits", who, entries, held,
                           TRAIN_DEV_F32 if narrow else TRAIN_DEV_U16, 0)
    var n = a.count()
    var out = lin_dev_new(entries, held, a.shape, TRAIN_DEV_U16 if narrow else TRAIN_DEV_F32)
    _ = lin_lowbit(binding, context)
    _ = lin_call(context, LIN_OP_TO_BF16_DEV if narrow else LIN_OP_FROM_BF16_DEV, [lin_dev_handle(out), a.handle], [n])
    _ = args
    return out


def _lin_quantize_int8_dev(binding: PythonObject, x: LinDevOperand, entries: PythonObject, held: PythonObject,
                           context: PythonObject) raises -> PythonObject:
    """`quantize_int8` of an admitted device operand: (codes, exponents) as new device arrays (int8 / int32)."""
    var rows = x.dim(0)
    var cols = x.dim(1)
    var codes = lin_dev_new(entries, held, [rows, cols], TRAIN_DEV_I8)
    var exps = lin_dev_new(entries, held, [rows], TRAIN_DEV_I32)
    _ = lin_lowbit(binding, context)
    _ = lin_call(context, LIN_OP_QUANTIZE_INT8_DEV, [lin_dev_handle(codes), lin_dev_handle(exps), x.handle],
                 [rows, cols])
    return Python.tuple(codes, exps)


def _lin_dequantize_int8_dev(args: PythonObject, binding: PythonObject, held: PythonObject,
                             context: PythonObject) raises -> PythonObject:
    """`dequantize_int8(codes, exponents)` on device arrays: a new float32 device array."""
    var entries = lin_dev_entries(_lin_dev_first(args, held, "dequantize_int8"), held)
    var q = lin_dev_operand(args[1], "codes", "dequantize_int8", entries, held, TRAIN_DEV_I8, 2)
    var e = lin_dev_operand(args[2], "exponents", "dequantize_int8", entries, held, TRAIN_DEV_I32, 1)
    var rows = q.dim(0)
    var cols = q.dim(1)
    if e.dim(0) != rows:
        raise Error("mojolearn.linalg.dequantize_int8: exponents has shape " + e.text() + ", want ("
                    + String(rows) + ",)")
    var out = lin_dev_new(entries, held, [rows, cols], TRAIN_DEV_F32)
    _ = lin_lowbit(binding, context)
    _ = lin_call(context, LIN_OP_DEQUANTIZE_INT8_DEV, [lin_dev_handle(out), q.handle, e.handle], [rows, cols])
    _ = args
    return out


def _lin_int8_dev_operand(binding: PythonObject, x: PythonObject, name: String, entries: PythonObject,
                          held: PythonObject, context: PythonObject, mut keep: List[PythonObject]) raises -> List[LinDevOperand]:
    """matmul_int8's operand on the device: a (codes, exponents) pair of device arrays, or a float32 device matrix
    quantized on the device first (its codes and exponents held in `keep` through the product)."""
    var b = native_loaded_module("builtins")
    var parts = List[LinDevOperand]()
    if py_type(x) is b.tuple and len(x) == 2:
        parts.append(lin_dev_operand(x[0], name + " codes", "matmul_int8", entries, held, TRAIN_DEV_I8, 2))
        parts.append(lin_dev_operand(x[1], name + " exponents", "matmul_int8", entries, held, TRAIN_DEV_I32, 1))
        if parts[1].dim(0) != parts[0].dim(0):
            raise Error("mojolearn.linalg.matmul_int8: " + name + " exponents has shape " + parts[1].text()
                        + ", want (" + String(parts[0].dim(0)) + ",)")
        return parts^
    var a = lin_dev_operand(x, name, "matmul_int8", entries, held, TRAIN_DEV_F32, 2)
    var pair = _lin_quantize_int8_dev(binding, a, entries, held, context)
    keep.append(pair)
    parts.append(lin_dev_operand(pair[0], name + " codes", "matmul_int8", entries, held, TRAIN_DEV_I8, 2))
    parts.append(lin_dev_operand(pair[1], name + " exponents", "matmul_int8", entries, held, TRAIN_DEV_I32, 1))
    return parts^


def _lin_matmul_int8_dev(args: PythonObject, binding: PythonObject, held: PythonObject,
                         context: PythonObject) raises -> PythonObject:
    """`matmul_int8(a, b, *, out)` on device arrays: a and b (pairs or float32 matrices) read in place, a . b^T a new
    float32 device array."""
    lin_dev_out(args[3], "matmul_int8")
    var entries = lin_dev_entries(_lin_dev_first(args, held, "matmul_int8"), held)
    var keep = List[PythonObject]()
    var pa = _lin_int8_dev_operand(binding, args[1], "a", entries, held, context, keep)
    var pb = _lin_int8_dev_operand(binding, args[2], "b", entries, held, context, keep)
    var m = pa[0].dim(0)
    var k = pa[0].dim(1)
    var n = pb[0].dim(0)
    var kb = pb[0].dim(1)
    if k != kb:
        raise Error("mojolearn.linalg.matmul_int8: contracted extents differ, a has k=" + String(k) + " and b has k="
                    + String(kb) + " (a codes " + pa[0].text() + ", b codes " + pb[0].text() + ")")
    var out = lin_dev_new(entries, held, [m, n], TRAIN_DEV_F32)
    _ = lin_lowbit(binding, context)
    _ = lin_call(context, LIN_OP_GEMM_INT8_DEV,
                 [lin_dev_handle(out), pa[0].handle, pa[1].handle, pb[0].handle, pb[1].handle], [m, n, k])
    _ = keep^
    _ = args
    return out


def _lin_matmul_bf16_dev(args: PythonObject, controls: PythonObject, binding: PythonObject, held: PythonObject,
                         context: PythonObject) raises -> PythonObject:
    """`matmul_bf16(a, b, *, transpose_a, transpose_b, out)` on device arrays: a float32 or bf16 bits, b bf16 bits,
    read in place; the product a new float32 device array."""
    lin_dev_out(args[5], "matmul_bf16")
    var entries = lin_dev_entries(_lin_dev_first(args, held, "matmul_bf16"), held)
    var a_code = lin_dev_dtype(py_fields(args[1]))
    var a_bits = a_code == TRAIN_DEV_U16
    var a = lin_dev_operand(args[1], "a", "matmul_bf16", entries, held, TRAIN_DEV_U16 if a_bits else TRAIN_DEV_F32, 2)
    var bb = lin_dev_operand(args[2], "b", "matmul_bf16", entries, held, TRAIN_DEV_U16, 2)
    var ta = lin_truth(py_item(controls, PythonObject(3)))
    var tb = lin_truth(py_item(controls, PythonObject(4)))
    if ta and tb:
        raise Error("mojolearn.linalg.matmul_bf16: transpose_a and transpose_b together are refused; the profile has "
                    + "OP_NN, OP_NT and OP_TN only")
    var op = OP_TN if ta else (OP_NT if tb else OP_NN)
    var m = a.dim(1) if ta else a.dim(0)
    var k = a.dim(0) if ta else a.dim(1)
    var n = bb.dim(0) if tb else bb.dim(1)
    var kb = bb.dim(1) if tb else bb.dim(0)
    if k != kb:
        raise Error("mojolearn.linalg.matmul_bf16: contracted extents differ, a gives k=" + String(k)
                    + " and b gives k=" + String(kb) + " (a.shape=" + a.text() + ", b.shape=" + bb.text() + ")")
    var out = lin_dev_new(entries, held, [m, n], TRAIN_DEV_F32)
    _ = lin_lowbit(binding, context)
    _ = lin_call(context, LIN_OP_GEMM_BF16_DEV, [lin_dev_handle(out), a.handle, bb.handle],
                 [m, n, k, op, 1 if a_bits else 0])
    _ = args
    return out


def lin_lowbit_operation(operation: Int, var args: PythonObject, var context: PythonObject,
                         has_device: Bool = True) raises -> PythonObject:
    # bw2-linalg: `operation` is a leaf code of bindings/operation_names_linops.mojo
    # (it was up to three String compares here and fifteen below).
    # Scalar controls may be NumPy scalar owners; exact numeric admission
    # precedes their native truth slots, including object-dtype rejection.
    # `has_device`: the public shells pass `device` as the last item of a
    # computing function's tuple (lane vdf-linalg); the one internal caller
    # that holds the name (the byte LM weight widening,
    # bindings/linalg_ops_native_unpack.mojo) passes none and runs today's route.
    var controls = Python.dict()
    if operation == LINOPS_OPN_FAMILY_LIN_MATMUL:
        for i in [3, 4, 6]:  # driver-loop: over (transpose and identity controls)
            host_inference_scalar(args[i], context, True)
            py_setitem(controls, PythonObject(i), PythonObject(lin_truth(args[i])))
    elif operation == LINOPS_OPN_FAMILY_LIN_MATMUL_FUSED:
        for i in [6, 7]:  # driver-loop: over (transpose controls)
            host_inference_scalar(args[i], context, True)
            py_setitem(controls, PythonObject(i), PythonObject(lin_truth(args[i])))
    elif operation == LINOPS_OPN_FAMILY_LIN_MATMUL_BF16:
        for i in [3, 4]:  # driver-loop: over (transpose controls)
            host_inference_scalar(args[i], context, True)
            py_setitem(controls, PythonObject(i), PythonObject(lin_truth(args[i])))
    var last = len(args)
    var device = function_device_default()
    # lane vdf-linalg4: TrainDeviceArray operands are read in place (bindings/linalg_dev_arrays_unpack.mojo). The
    # published class is read from the family context BEFORE an explicit "cpu" swaps it for the host image's copy;
    # only operations with a device form pay the one dictionary read.
    var held = lin_train_device(context) if has_device and lin_dev_operation(operation) else PythonObject(None)
    if has_device and operation != LINOPS_OPN_FAMILY_LIN_NUMERIC_MODE and operation != LINOPS_OPN_FAMILY_LIN_PROFILE \
            and operation != LINOPS_OPN_FAMILY_LIN_REQUIRE_IDENTICAL:
        # The one read of `device` (an identity compare for None), before any
        # work. matmul_fused is the Apple fast arm: no CPU image carries it.
        # None costs this one compare: no name is built, nothing else is read.
        last -= 1
        if args[last] is not None:
            device = lin_device(args[last], _lin_public_name(operation), LIN_HAS_GPU,
                                operation != LINOPS_OPN_FAMILY_LIN_MATMUL_FUSED, context)
            context = lin_device_context(device, context)
    if lin_any_device_array(args, 1, last, held, operation == LINOPS_OPN_FAMILY_LIN_MATMUL_INT8):
        # Device operands: refused with device="cpu" (they live on the GPU) before any work; never admitted as host
        # buffers, never copied.
        lin_dev_device(device, _lin_public_name(operation))
        return lin_lowbit_dev(operation, args, controls, lin_device_binding(device, context), held, context)
    var binding = lin_device_binding(device, context)
    var admitted = Python.list(binding)
    for i in range(1, last):  # driver-loop: over (public arguments admitted before input protocols)
        classical_admit_value(args[i], context)
        var value = py_get(controls, PythonObject(i), args[i])
        _ = py_call(admitted.append, value)
    args = admitted
    if operation == LINOPS_OPN_FAMILY_LIN_NUMERIC_MODE:
        return PythonObject(lin_mode(binding, context))
    if operation == LINOPS_OPN_FAMILY_LIN_PROFILE:
        return _lin_profile(binding, context)
    if operation == LINOPS_OPN_FAMILY_LIN_REQUIRE_IDENTICAL:
        lin_require_identical(binding, context)
        return PythonObject(None)
    if operation == LINOPS_OPN_FAMILY_LIN_MATMUL:
        return _lin_matmul(args, context)
    if operation == LINOPS_OPN_FAMILY_LIN_MATMUL_FUSED:
        return _lin_matmul_fused(args, context)
    if operation == LINOPS_OPN_FAMILY_LIN_TO_BF16:
        return _lin_to_bf16(args, context)
    if operation == LINOPS_OPN_FAMILY_LIN_FROM_BF16:
        return _lin_from_bf16(args, context)
    if operation == LINOPS_OPN_FAMILY_LIN_FROM_F16:
        return _lin_from_f16(args, context)
    if operation == LINOPS_OPN_FAMILY_LIN_QUANTIZE_INT8:
        return _lin_quantize_int8(args, context)
    if operation == LINOPS_OPN_FAMILY_LIN_DEQUANTIZE_INT8:
        return _lin_dequantize_int8(args, context)
    if operation == LINOPS_OPN_FAMILY_LIN_MATMUL_BF16:
        return _lin_matmul_bf16(args, context)
    if operation == LINOPS_OPN_FAMILY_LIN_MATMUL_INT8:
        return _lin_matmul_int8(args, context)
    if operation == LINOPS_OPN_FAMILY_LIN_QUANTIZE_INT15:
        return _lin_quantize_int15(args, context)
    if operation == LINOPS_OPN_FAMILY_LIN_DEQUANTIZE_INT15:
        return _lin_dequantize_int15(args, context)
    if operation == LINOPS_OPN_FAMILY_LIN_MATMUL_INT15:
        return _lin_matmul_int15(args, context)
    raise Error("RuntimeError: unknown linalg operation " + String(operation))


def _lin_binary_path(binding: PythonObject) raises -> PythonObject:
    var nk = linalg_keys()
    var path = py_get(py_fields(binding), nk[].k___file__)
    if py_type(path) is not py_type(PythonObject("")):
        raise Error("TypeError: linalg binary path requires an exact string")
    return path


def _lin_profile(binding: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = linalg_keys()
    # The same profile receipt previously assembled by the Python public method.
    # Reading mode first checks the current native provider/profile exactly once.
    var mode = lin_mode(binding, context)
    var state = lin_state(context)
    var profile = py_item(state, nk[].k__profile_cache)
    var actual = py_item(state, nk[].k_PROFILE) if profile is None else profile[0]
    var version = Int(py=py_item(state, nk[].k_PROFILE_VERSION) if profile is None else profile[1])
    var result = Python.dict()
    py_setitem(result, nk[].k_profile, actual)
    py_setitem(result, nk[].k_profile_version, version)
    py_setitem(result, nk[].k_numeric_mode, mode)
    py_setitem(result, nk[].k_identity_claimed, mode == "identical")
    py_setitem(result, nk[].k_dtype, "float32")
    py_setitem(result, nk[].k_ops, Python.tuple("OP_NN", "OP_NT", "OP_TN"))
    py_setitem(result, nk[].k_contract, "gemm/IDENTICAL_FP32_CONTRACT.md")
    py_setitem(result, nk[].k_kernel, "gemm/checks/gemm_identical.mojo")
    py_setitem(result, nk[].k_oracle, "gemm/host/gemm_oracle.mojo::gemm_oracle")
    py_setitem(result, nk[].k_certified_shapes, 62)
    py_setitem(result, nk[].k_certified_vendors, Python.tuple("Apple M4", "NVIDIA H100", "AMD MI325X"))
    py_setitem(result, nk[].k_certification_source, "gemm/README.md")
    py_setitem(result, nk[].k_binary, _lin_binary_path(binding))
    if version != Int(py=py_item(state, nk[].k_PROFILE_VERSION)):
        py_setitem(result, nk[].k_contract, "gemm/contract.mojo")
        py_setitem(result, nk[].k_identity_claimed, False)
        py_setitem(result, nk[].k_identity_required, True)
        py_setitem(result, nk[].k_certified_shapes, 0)
        py_setitem(result, nk[].k_certified_vendors, Python.tuple())
        py_setitem(result, nk[].k_certification_source, PythonObject(None))
        py_setitem(result, nk[].k_qualification, "UNVERIFIED_EXPERIMENT")
        py_setitem(result, nk[].k_experiment_source, "experiments/neural_identical_20261006/gemm_cnn.json")
    return result
