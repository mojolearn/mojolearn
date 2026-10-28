# SPDX-License-Identifier: Apache-2.0
# `mojolearn.matmul` and the low-bit functions of python/mojolearn/_linalg_impl.py
# (to_bf16, from_bf16, quantize/dequantize int8 and int15, matmul_bf16/int8/int15),
# one native call each (`family_lin_*`). The shell hands over the linalg binding
# `_load()` resolved; this owner makes the Python refusals in their order and
# words, the mode and profile read-backs (`numeric_mode`, `require_identical`,
# `_lowbit_binding`, `_int15_binding`, sharing the module's own caches), the
# operand layout (`_operand`, `as_*_c`) and outputs, and calls the same binding
# entries with the same address and params lists in the same order. The same
# kernels on the same bytes: the bits do not move.
from providers.python.estimator_binding_pyapi import estimator_binding
from std.python import Python, PythonObject
from std.os import getenv
from bindings.native_loaded_module import native_loaded_module
from bindings.classical_pandas import classical_admit_value
from providers.python.neural_descriptor_pyapi import neural_descriptor_reuse, neural_descriptor_execute, neural_descriptor_info, neural_descriptor_entry
from bindings.py_balanced import py_call, py_get, py_setitem, py_type
from bindings.family_common import family_call, family_repr, family_str, family_type_name, family_isinstance
from bindings.optimizer_public_door import TensorView, shape_text
from bindings.linalg_ops_buffers import (
    LinBuf, lin_truth, lin_view, lin_is_f32, lin_dtype_name, lin_new, lin_c_order, lin_typed, lin_tuple,
)

comptime OP_NN = 0
comptime OP_NT = 1
comptime OP_TN = 2


def lin_state(context: PythonObject) raises -> PythonObject:
    """python/mojolearn/_linalg_impl.py's module dictionary (its constants and caches)."""
    return context["family_lin_state"]


def lin_text(context: PythonObject, name: String) raises -> String:
    return String(py=lin_state(context)[name])


def lin_attr(binding: PythonObject, name: String) raises -> PythonObject:
    """`getattr(binding, name, None)`."""
    if py_type(binding) is py_type(native_loaded_module("builtins")):
        return neural_descriptor_entry(binding, name, False)
    return family_call(native_loaded_module("builtins").getattr, binding, PythonObject(name), PythonObject(None))


def lin_call(binding: PythonObject, context: PythonObject, name: String, *args: PythonObject) raises -> PythonObject:
    """`binding.name(*args)`; the binding's own exception propagates unchanged."""
    var op = -1
    if name == "gemm":
        op = 0
    if name == "gemm_bf16":
        op = 1
    if name == "gemm_int8":
        op = 2
    if name == "quantize_int8":
        op = 3
    if name == "dequantize_int8":
        op = 4
    if name == "gemm_int15":
        op = 5
    if name == "quantize_int15":
        op = 6
    if name == "dequantize_int15":
        op = 7
    if name == "to_bf16":
        op = 8
    if name == "from_bf16":
        op = 9
    if name == "from_f16":
        op = 10
    if name == "linalg_numeric_mode":
        op = 11
    if name == "linalg_profile_version":
        op = 12
    if name == "lowbit_profile_version":
        op = 13
    if name == "int15_profile_version":
        op = 14
    var state = lin_state(context)
    var provider = py_get(state, "_native_provider")
    if op >= 0 and provider is not None and py_get(state, "_native_provider_binding") is binding:
        var addresses = List[Int]()
        var parameters = List[Int]()
        if op < 11:
            var last = len(args) - 1
            for i in range(len(args[last])):  # small-loop(admitted shape words)
                parameters.append(Int(py=args[last][i]))
            if op >= 5 and op <= 7:
                for i in range(len(args[0])):  # small-loop(int15 buffer descriptors)
                    addresses.append(Int(py=args[0][i]))
            else:
                for i in range(last):  # small-loop(buffer descriptor addresses)
                    addresses.append(Int(py=args[i]))
        return PythonObject(neural_descriptor_execute(provider, op, addresses, parameters, List[Float64]()))
    # Older/CPU bindings retain their existing door and remain unqualified by
    # the native-provider runtime gate until that binding gains this ABI.
    var entry = family_call(native_loaded_module("builtins").getattr, binding, PythonObject(name))
    var arguments = Python.list()
    for i in range(len(args)):  # small-loop(args: positional arguments of one call): bounded by the call site
        _ = py_call(arguments.append, args[i])
    return _lin_apply(entry, arguments)


def _lin_apply(entry: PythonObject, arguments: PythonObject) raises -> PythonObject:
    var n = len(arguments)
    if n == 0:
        return family_call(entry)
    if n == 1:
        return family_call(entry, arguments[0])
    if n == 2:
        return family_call(entry, arguments[0], arguments[1])
    if n == 3:
        return family_call(entry, arguments[0], arguments[1], arguments[2])
    if n == 4:
        return family_call(entry, arguments[0], arguments[1], arguments[2], arguments[3])
    if n == 5:
        return family_call(entry, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4])
    return family_call(entry, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5])


def lin_backend_mode(context: PythonObject) raises -> String:
    var mode = context["backend_fields"]["_DEFAULT_MODE"]
    if mode is None:
        raise Error("RuntimeError: linalg numeric mode must be selected before operation admission")
    return String(py=mode)


def lin_connected_binding(context: PythonObject) raises -> PythonObject:
    var state = lin_state(context)
    # Resolve the live import-connected module once per operation. Linalg's
    # own provider supplies its compile-mode/profile metadata immediately below.
    var binding = estimator_binding(PythonObject(None), "_mojolearn_linalg", context, False)
    py_setitem(state, "_binding_cache", binding)
    if py_get(state, "_mode_binding") is not binding:
        py_setitem(state, "_mode_binding", binding)
        py_setitem(state, "_mode_cache", PythonObject(None))
        py_setitem(state, "_profile_cache", PythonObject(None))
    var factory = lin_attr(binding, "linalg_native_api_v1")
    var provider = PythonObject(None)
    if factory is not None:
        provider = neural_descriptor_reuse(py_get(state, "_native_provider"), factory, "mojolearn.linalg.v1", 15)
    py_setitem(state, "_native_provider", provider)
    py_setitem(state, "_native_provider_binding", binding)
    return binding


def lin_mode(binding: PythonObject, context: PythonObject) raises -> String:
    """`numeric_mode()`: the loaded linalg binary's compiled mode, checked
    against the selected set and its profile version; cached in the
    module's `_mode_cache` / `_profile_cache` exactly as the Python did."""
    var b = native_loaded_module("builtins")
    var state = lin_state(context)
    var cached = state["_mode_cache"]
    if cached is not None:
        return String(py=cached)
    var raw = family_call(b.int, lin_call(binding, context, "linalg_numeric_mode"))
    var compiled = String(py=py_get(context["backend_fields"]["_CODE_MODE"], raw, PythonObject("unknown")))
    var selected = lin_backend_mode(context)
    var build = lin_text(context, "_BUILD_SCRIPT")
    if compiled != selected:
        raise Error("RuntimeError: mojolearn.linalg: the loaded binary was compiled " + compiled
                    + " but this process selected the " + selected + " set -- a .so is in the wrong directory ("
                    + family_str(py_call(b.getattr, binding, "__file__")) + "); rebuild both sets with " + build)
    var version = Int(py=family_call(b.int, lin_call(binding, context, "linalg_profile_version")))
    var profile = state["PROFILE"]
    var profile_version = Int(py=state["PROFILE_VERSION"])
    var getter = lin_attr(binding, "linalg_numerical_profile")
    var actual = PythonObject(None)
    if getter is not None:
        actual = family_call(b.str, family_call(getter))
    elif version == profile_version:
        actual = profile
    var known = profile if version == profile_version else py_get(state["_EXPERIMENT_PROFILES"], PythonObject(version), PythonObject(None))
    var env_name = state["_EXPECTED_PROFILE_ENV"]
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
    py_setitem(state, PythonObject("_profile_cache"), Python.tuple(actual, PythonObject(version)))
    py_setitem(state, PythonObject("_mode_cache"), PythonObject(compiled))
    return compiled


def lin_require_identical(binding: PythonObject, context: PythonObject) raises:
    """`require_identical()`."""
    var loaded = lin_mode(binding, context)
    if loaded != "identical":
        raise Error("RuntimeError: mojolearn.linalg: this process loaded the " + loaded.upper()
                    + " build, which makes NO cross-vendor identity claim; " + lin_text(context, "PROFILE")
                    + " is what the IDENTICAL build computes.\n"
                    + "  Select it with mojolearn.set_numeric_mode('identical') "
                    + "before the call, or set MOJOLEARN_NUMERIC_MODE=identical "
                    + "before importing mojolearn; "
                    + "the identical binary builds with\n      "
                    + "MOJOLEARN_NUMERIC_MODE=identical bash " + lin_text(context, "_BUILD_SCRIPT") + "\n"
                    + "  Or pass identical=False to say in the source that you want "
                    + "the " + loaded + " product and are making no identity claim about it.")


def lin_lowbit(binding: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_lowbit_binding()`: the identical build with the low-bit profile version this owner expects."""
    lin_require_identical(binding, context)
    var v = 0
    if py_get(lin_state(context), "_native_provider") is not None:
        v = Int(py=lin_call(binding, context, "lowbit_profile_version"))
    else:
        var got = lin_attr(binding, "lowbit_profile_version")
        if got is None:
            raise Error("RuntimeError: mojolearn.linalg: the loaded linalg extension predates the "
                        + "low-bit profiles; rebuild it with bindings/build_linalg.sh")
        v = Int(py=family_call(native_loaded_module("builtins").int, family_call(got)))
    var want = Int(py=lin_state(context)["LOWBIT_PROFILE_VERSION"])
    if v != want:
        raise Error("RuntimeError: mojolearn.linalg: the loaded extension implements low-bit "
                    + "profile version " + String(v) + ", this wrapper expects " + String(want))
    return binding


def lin_int15(binding: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_int15_binding()`."""
    lin_require_identical(binding, context)
    var v = 0
    if py_get(lin_state(context), "_native_provider") is not None:
        v = Int(py=lin_call(binding, context, "int15_profile_version"))
    else:
        var got = lin_attr(binding, "int15_profile_version")
        if got is None:
            raise Error("RuntimeError: mojolearn.linalg: the loaded linalg extension predates the "
                        + "fifteen-bit profile; rebuild it with bindings/build_linalg.sh")
        v = Int(py=family_call(native_loaded_module("builtins").int, family_call(got)))
    var want = Int(py=lin_state(context)["INT15_PROFILE_VERSION"])
    if v != want:
        raise Error("RuntimeError: mojolearn.linalg: the loaded extension implements fifteen-bit "
                    + "profile version " + String(v) + ", this wrapper expects " + String(want))
    return binding


def lin_ints(*values: Int) raises -> PythonObject:
    """A Python list of ints (a params list)."""
    var out = Python.list()
    for i in range(len(values)):  # small-loop(params: words of one params list): bounded by the call site
        _ = py_call(out.append, PythonObject(values[i]))
    return out


def lin_operand(x: PythonObject, name: String, context: PythonObject) raises -> LinBuf:
    """`_operand(x, name)`: a float32 2-D buffer, refused by name otherwise,
    in C order (a layout copy at most, never a cast)."""
    var profile = lin_text(context, "PROFILE")
    var v = TensorView()
    if not lin_view(x, context, v):
        raise Error("TypeError: mojolearn.linalg: " + name + " is a " + family_type_name(x) + ", which does "
                    + "not support the buffer protocol, and only a float32 buffer is "
                    + "in this profile (" + profile + "; contract section 1 makes FP32 a "
                    + "hard requirement). Pass a float32 array -- a NumPy array, an "
                    + "array.array('f') or a mojolearn.Array; convert with "
                    + "np.asarray(" + name + ", dtype=np.float32) if that is what you want.")
    if not lin_is_f32(v):
        raise Error("TypeError: mojolearn.linalg: " + name + " has dtype " + lin_dtype_name(x, v) + ", and "
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
    return lin_c_order(x, v, "<f4", name, context)


def lin_operand_bits(x: PythonObject, name: String, context: PythonObject) raises -> LinBuf:
    """`_operand_bits(x, name)`: a 2-D uint16 buffer of bf16 bits in C order."""
    var v = TensorView()
    if not lin_view(x, context, v):
        raise Error("TypeError: mojolearn.linalg: " + name + " must be a uint16 buffer of bf16 bits "
                    + "(profile " + lin_text(context, "PROFILE_BF16") + "); got " + family_type_name(x))
    var dtype = lin_dtype_name(x, v)
    if dtype != "uint16":
        raise Error("TypeError: mojolearn.linalg: " + name + " has dtype " + dtype + "; the "
                    + "bf16 profile takes bf16 BITS in a uint16 buffer, which "
                    + "to_bf16() produces. Refused rather than reinterpreted.")
    if len(v.shape) != 2:
        raise Error("mojolearn.linalg: " + name + " must be 2-D, got " + String(len(v.shape)) + "-D")
    if v.count() == 0:
        raise Error("mojolearn.linalg: " + name + " has no elements")
    return lin_typed(x, "<u2", 2, name, context)


def lin_out(out_buf: PythonObject, m: Int, n: Int, who: String, context: PythonObject) raises -> LinBuf:
    """`_out_or_new(out, m, n, who)`."""
    if out_buf is None:
        return lin_new([m, n], "<f4", context)
    var v = TensorView()
    if not lin_view(out_buf, context, v):
        raise Error("TypeError: mojolearn: object does not support the buffer protocol")
    if not lin_is_f32(v):
        raise Error("TypeError: mojolearn.linalg." + who + ": out must be float32")
    if len(v.shape) != 2 or v.shape[0] != m or v.shape[1] != n:
        raise Error("mojolearn.linalg." + who + ": out has shape " + shape_text(v.shape) + ", want ("
                    + String(m) + ", " + String(n) + ")")
    if not v.c_contig or v.readonly:
        raise Error("mojolearn.linalg." + who + ": out must be C-contiguous and writable")
    var _keep_ret_v = LinBuf(out_buf, v.addr, v.shape.copy())
    _ = v  # keep the Python object alive through the native call above
    return _keep_ret_v^


# ---------------------------------------------------------------------------
# the public functions


def _lin_matmul(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`matmul(a, b, *, transpose_a, transpose_b, out, identical)`."""
    var binding = args[0]
    if lin_truth(args[6]):
        lin_require_identical(binding, context)
    else:
        _ = lin_mode(binding, context)
    var ta = lin_truth(args[3])
    var tb = lin_truth(args[4])
    var profile = lin_text(context, "PROFILE")
    if ta and tb:
        raise Error("mojolearn.linalg.matmul: transpose_a=True with transpose_b=True "
                    + "is refused. " + profile + " has exactly three operations (contract "
                    + "section 0.1: OP_NN, OP_NT, OP_TN) and a.T @ b.T is not one of "
                    + "them. This function will not materialize a transpose to fake a "
                    + "fourth. If you want it, write the identity yourself:\n"
                    + "    np.ascontiguousarray(np.asarray(matmul(b, a)).T)\n"
                    + "which is (b @ a).T == a.T @ b.T, runs as OP_NN, and leaves the "
                    + "extra step visible in your source where you can price it.")
    var op = OP_NN if not ta and not tb else (OP_NT if tb else OP_TN)
    var a = lin_operand(args[1], "a", context)
    var bb = lin_operand(args[2], "b", context)
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
    var out: LinBuf
    if out_obj is None:
        out = lin_new([m, n], "<f4", context)
    else:
        var v = TensorView()
        if not lin_view(out_obj, context, v):
            raise Error("TypeError: mojolearn.linalg.matmul: out must be a numpy array, a "
                        + "mojolearn.Array or any other writable float32 buffer "
                        + "(anything supporting the buffer protocol), got " + family_type_name(out_obj))
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
        out = LinBuf(out_obj, v.addr, v.shape.copy())
        _ = v  # keep the Python object alive through the native call above
    # ORDER MATCHES bindings/_mojolearn_linalg.mojo::gemm_binding: the output
    # address first, then a and b; params (m, n, k, op).
    _ = lin_call(binding, context, "gemm", PythonObject(out.addr), PythonObject(a.addr), PythonObject(bb.addr),
                 lin_ints(m, n, k, op))
    _ = bb  # keep the Python object alive through the native call above
    _ = a  # keep the Python object alive through the native call above
    _ = a^
    _ = bb^
    return out.keep


def _lin_to_bf16(args: PythonObject, context: PythonObject) raises -> PythonObject:
    var a = lin_typed(args[1], "<f4", -1, "x", context)
    var n = a.count()
    if n == 0:
        raise Error("mojolearn.linalg.to_bf16: x has no elements")
    var out = lin_new(a.shape, "<u2", context)
    _ = lin_call(lin_lowbit(args[0], context), context, "to_bf16", PythonObject(out.addr), PythonObject(a.addr), lin_ints(n))
    _ = out  # keep the Python object alive through the native call above
    _ = a  # keep the Python object alive through the native call above
    _ = a^
    return out.keep


def _lin_from_bf16(args: PythonObject, context: PythonObject) raises -> PythonObject:
    var a = lin_typed(args[1], "<u2", -1, "bits", context)
    var n = a.count()
    if n == 0:
        raise Error("mojolearn.linalg.from_bf16: bits has no elements")
    var out = lin_new(a.shape, "<f4", context)
    _ = lin_call(lin_lowbit(args[0], context), context, "from_bf16", PythonObject(out.addr), PythonObject(a.addr), lin_ints(n))
    _ = out  # keep the Python object alive through the native call above
    _ = a  # keep the Python object alive through the native call above
    _ = a^
    return out.keep


def _lin_from_f16(args: PythonObject, context: PythonObject) raises -> PythonObject:
    var a = lin_typed(args[1], "<u2", -1, "bits", context)
    var n = a.count()
    if n == 0:
        raise Error("mojolearn.linalg.from_f16: bits has no elements")
    var out = lin_new(a.shape, "<f4", context)
    _ = lin_call(lin_lowbit(args[0], context), context, "from_f16", PythonObject(out.addr), PythonObject(a.addr), lin_ints(n))
    _ = out  # keep the Python object alive through the native call above
    _ = a  # keep the Python object alive through the native call above
    _ = a^
    return out.keep


def lin_quantize_int8(binding: PythonObject, a: LinBuf, context: PythonObject) raises -> PythonObject:
    """`quantize_int8` of an admitted operand: (codes, exponents)."""
    var rows = a.dim(0)
    var cols = a.dim(1)
    var codes = lin_new([rows, cols], "<i1", context)
    var exps = lin_new([rows], "<i4", context)
    _ = lin_call(lin_lowbit(binding, context), context, "quantize_int8", PythonObject(codes.addr), PythonObject(exps.addr),
                 PythonObject(a.addr), lin_ints(rows, cols))
    _ = exps  # keep the Python object alive through the native call above
    _ = codes  # keep the Python object alive through the native call above
    return Python.tuple(codes.keep, exps.keep)


def _lin_quantize_int8(args: PythonObject, context: PythonObject) raises -> PythonObject:
    var a = lin_operand(args[1], "x", context)
    var result = lin_quantize_int8(args[0], a, context)
    _ = a^
    return result


def _lin_dequantize_int8(args: PythonObject, context: PythonObject) raises -> PythonObject:
    var q = lin_typed(args[1], "<i1", 2, "codes", context)
    var e = lin_typed(args[2], "<i4", 1, "exponents", context)
    var rows = q.dim(0)
    var cols = q.dim(1)
    if e.rank() != 1 or e.dim(0) != rows:
        raise Error("mojolearn.linalg.dequantize_int8: exponents has shape " + e.text() + ", want ("
                    + String(rows) + ",)")
    var out = lin_new([rows, cols], "<f4", context)
    _ = lin_call(lin_lowbit(args[0], context), context, "dequantize_int8", PythonObject(out.addr), PythonObject(q.addr),
                 PythonObject(e.addr), lin_ints(rows, cols))
    _ = out  # keep the Python object alive through the native call above
    _ = q  # keep the Python object alive through the native call above
    _ = e  # keep the Python object alive through the native call above
    _ = q^
    _ = e^
    return out.keep


def _lin_int8_operand(binding: PythonObject, x: PythonObject, name: String, context: PythonObject,
                      mut q: LinBuf, mut e: LinBuf) raises:
    """`_int8_operand(x, name)`: a (codes, exponents) pair, or a float32
    matrix quantized by `quantize_int8`."""
    var b = native_loaded_module("builtins")
    if family_isinstance(x, b.tuple) and len(x) == 2:
        q = lin_typed(x[0], "<i1", 2, name + " codes", context)
        e = lin_typed(x[1], "<i4", 1, name + " exponents", context)
        if e.rank() != 1 or e.dim(0) != q.dim(0):
            raise Error("mojolearn.linalg.matmul_int8: " + name + " exponents has shape " + e.text()
                        + ", want (" + String(q.dim(0)) + ",)")
        return
    var a = lin_operand(x, name, context)
    var pair = lin_quantize_int8(binding, a, context)
    _ = a^
    var codes = pair[0]
    var exps = pair[1]
    var v = TensorView()
    _ = lin_view(codes, context, v)
    q = LinBuf(codes, v.addr, v.shape.copy())
    _ = v  # keep the Python object alive through the native call above
    _ = lin_view(exps, context, v)
    e = LinBuf(exps, v.addr, v.shape.copy())
    _ = v  # keep the Python object alive through the native call above


def _lin_matmul_int8(args: PythonObject, context: PythonObject) raises -> PythonObject:
    var binding = args[0]
    var qa = LinBuf(PythonObject(None), 0, List[Int]())
    var ea = LinBuf(PythonObject(None), 0, List[Int]())
    var qb = LinBuf(PythonObject(None), 0, List[Int]())
    var eb = LinBuf(PythonObject(None), 0, List[Int]())
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
    _ = lin_call(lin_lowbit(binding, context), context, "gemm_int8", PythonObject(out.addr), PythonObject(qa.addr),
                 PythonObject(ea.addr), PythonObject(qb.addr), PythonObject(eb.addr), lin_ints(m, n, k))
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


def _lin_op_and_shape(a: LinBuf, bb: LinBuf, transpose_a: PythonObject, transpose_b: PythonObject,
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
    var a_bits = False
    var pa = TensorView()
    if lin_view(args[1], context, pa):
        a_bits = lin_dtype_name(args[1], pa) == "uint16"
    var a = lin_operand_bits(args[1], "a", context) if a_bits else lin_operand(args[1], "a", context)
    var bb = lin_operand_bits(args[2], "b", context)
    var op = 0
    var m = 0
    var n = 0
    var k = 0
    _lin_op_and_shape(a, bb, args[3], args[4], "matmul_bf16", op, m, n, k)
    var out = lin_out(args[5], m, n, "matmul_bf16", context)
    # params (mirrored in gemm_bf16_binding): m, n, k, op, a_bf16
    _ = lin_call(lin_lowbit(binding, context), context, "gemm_bf16", PythonObject(out.addr), PythonObject(a.addr),
                 PythonObject(bb.addr), lin_ints(m, n, k, op, 1 if a_bits else 0))
    _ = out  # keep the Python object alive through the native call above
    _ = a  # keep the Python object alive through the native call above
    _ = bb  # keep the Python object alive through the native call above
    _ = a^
    _ = bb^
    return out.keep


def lin_quantize_int15(binding: PythonObject, a: LinBuf, context: PythonObject) raises -> PythonObject:
    """`quantize_int15` of an admitted operand: (hi, lo, exponents)."""
    var rows = a.dim(0)
    var cols = a.dim(1)
    var hi = lin_new([rows, cols], "<i1", context)
    var lo = lin_new([rows, cols], "<i1", context)
    var exps = lin_new([rows], "<i4", context)
    # addrs (mirrored in quantize_int15_binding): hi, lo, exponents, x
    var addrs = Python.list(PythonObject(hi.addr), PythonObject(lo.addr), PythonObject(exps.addr), PythonObject(a.addr))
    _ = exps  # keep the Python object alive through the native call above
    _ = lo  # keep the Python object alive through the native call above
    _ = hi  # keep the Python object alive through the native call above
    _ = lin_call(lin_int15(binding, context), context, "quantize_int15", addrs, lin_ints(rows, cols))
    _ = exps  # keep the Python object alive through the native call above
    _ = lo  # keep the Python object alive through the native call above
    _ = hi  # keep the Python object alive through the native call above
    return Python.tuple(hi.keep, lo.keep, exps.keep)


def _lin_quantize_int15(args: PythonObject, context: PythonObject) raises -> PythonObject:
    var a = lin_operand(args[1], "x", context)
    var result = lin_quantize_int15(args[0], a, context)
    _ = a^
    return result


def _lin_int15_planes(x: PythonObject, name: String, who: String, context: PythonObject,
                      mut hi: LinBuf, mut lo: LinBuf, mut e: LinBuf) raises:
    """`_int15_planes(x, name, who)`."""
    hi = lin_typed(x[0], "<i1", 2, name + " hi", context)
    lo = lin_typed(x[1], "<i1", 2, name + " lo", context)
    e = lin_typed(x[2], "<i4", 1, name + " exponents", context)
    if lo.rank() != hi.rank() or lo.dim(0) != hi.dim(0) or lo.dim(1) != hi.dim(1):
        raise Error("mojolearn.linalg." + who + ": " + name + " lo has shape " + lo.text() + ", hi has " + hi.text())
    if e.rank() != 1 or e.dim(0) != hi.dim(0):
        raise Error("mojolearn.linalg." + who + ": " + name + " exponents has shape " + e.text() + ", want ("
                    + String(hi.dim(0)) + ",)")


def _lin_dequantize_int15(args: PythonObject, context: PythonObject) raises -> PythonObject:
    var h = LinBuf(PythonObject(None), 0, List[Int]())
    var l = LinBuf(PythonObject(None), 0, List[Int]())
    var e = LinBuf(PythonObject(None), 0, List[Int]())
    _lin_int15_planes(Python.tuple(args[1], args[2], args[3]), "operand", "dequantize_int15", context, h, l, e)
    var rows = h.dim(0)
    var cols = h.dim(1)
    var out = lin_new([rows, cols], "<f4", context)
    # addrs: y, hi, lo, exponents (mirrored in the binding)
    var addrs = Python.list(PythonObject(out.addr), PythonObject(h.addr), PythonObject(l.addr), PythonObject(e.addr))
    _ = out  # keep the Python object alive through the native call above
    _ = e  # keep the Python object alive through the native call above
    _ = l  # keep the Python object alive through the native call above
    _ = h  # keep the Python object alive through the native call above
    _ = lin_call(lin_int15(args[0], context), context, "dequantize_int15", addrs, lin_ints(rows, cols))
    _ = out  # keep the Python object alive through the native call above
    _ = e  # keep the Python object alive through the native call above
    _ = l  # keep the Python object alive through the native call above
    _ = h  # keep the Python object alive through the native call above
    _ = h^
    _ = l^
    _ = e^
    return out.keep


def _lin_int15_operand(binding: PythonObject, x: PythonObject, name: String, context: PythonObject,
                       mut hi: LinBuf, mut lo: LinBuf, mut e: LinBuf) raises:
    """`_int15_operand(x, name)`."""
    var b = native_loaded_module("builtins")
    if family_isinstance(x, b.tuple) and len(x) == 3:
        _lin_int15_planes(x, name, "matmul_int15", context, hi, lo, e)
        return
    if family_isinstance(x, b.tuple):
        raise Error("TypeError: mojolearn.linalg.matmul_int15: " + name + " is a tuple of " + String(len(x))
                    + "; a fifteen-bit operand is (hi, lo, exponents) from quantize_int15, or a float32 matrix")
    var a = lin_operand(x, name, context)
    var planes = lin_quantize_int15(binding, a, context)
    _ = a^
    var v = TensorView()
    _ = lin_view(planes[0], context, v)
    hi = LinBuf(planes[0], v.addr, v.shape.copy())
    _ = v  # keep the Python object alive through the native call above
    _ = lin_view(planes[1], context, v)
    lo = LinBuf(planes[1], v.addr, v.shape.copy())
    _ = v  # keep the Python object alive through the native call above
    _ = lin_view(planes[2], context, v)
    e = LinBuf(planes[2], v.addr, v.shape.copy())
    _ = v  # keep the Python object alive through the native call above


def _lin_matmul_int15(args: PythonObject, context: PythonObject) raises -> PythonObject:
    var binding = args[0]
    var ah = LinBuf(PythonObject(None), 0, List[Int]())
    var al = LinBuf(PythonObject(None), 0, List[Int]())
    var ea = LinBuf(PythonObject(None), 0, List[Int]())
    var bh = LinBuf(PythonObject(None), 0, List[Int]())
    var bl = LinBuf(PythonObject(None), 0, List[Int]())
    var eb = LinBuf(PythonObject(None), 0, List[Int]())
    _lin_int15_operand(binding, args[1], "a", context, ah, al, ea)
    _lin_int15_operand(binding, args[2], "b", context, bh, bl, eb)
    var m = ah.dim(0)
    var k = ah.dim(1)
    var n = bh.dim(0)
    var kb = bh.dim(1)
    if k != kb:
        raise Error("mojolearn.linalg.matmul_int15: contracted extents differ, a "
                    + "gives k=" + String(k) + " and b gives k=" + String(kb))
    var max_k = Int(py=lin_state(context)["INT15_MAX_K"])
    if k > max_k:
        raise Error("mojolearn.linalg.matmul_int15: k=" + String(k) + " is above " + String(max_k) + ", the "
                    + "largest contracted extent " + lin_text(context, "PROFILE_INT15") + " accepts (contract W-4)")
    var out = lin_out(args[3], m, n, "matmul_int15", context)
    # addrs (mirrored in gemm_int15_binding): c, a hi, a lo, a exponents,
    # b hi, b lo, b exponents; THE OUTPUT ADDRESS COMES FIRST
    var addrs = Python.list(PythonObject(out.addr), PythonObject(ah.addr), PythonObject(al.addr), PythonObject(ea.addr),
                            PythonObject(bh.addr), PythonObject(bl.addr), PythonObject(eb.addr))
    _ = out  # keep the Python object alive through the native call above
    _ = ea  # keep the Python object alive through the native call above
    _ = al  # keep the Python object alive through the native call above
    _ = ah  # keep the Python object alive through the native call above
    _ = eb  # keep the Python object alive through the native call above
    _ = bl  # keep the Python object alive through the native call above
    _ = bh  # keep the Python object alive through the native call above
    _ = lin_call(lin_int15(binding, context), context, "gemm_int15", addrs, lin_ints(m, n, k))
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


def lin_lowbit_operation(operation: String, var args: PythonObject, context: PythonObject) raises -> PythonObject:
    var binding = lin_connected_binding(context)
    var admitted = Python.list(binding)
    for i in range(1, len(args)):  # small-loop(public arguments admitted before input protocols)
        classical_admit_value(args[i], context)
        _ = py_call(admitted.append, args[i])
    args = admitted
    if operation == "family_lin_provider_info":
        return neural_descriptor_info(py_get(lin_state(context), "_native_provider"))
    if operation == "family_lin_matmul":
        return _lin_matmul(args, context)
    if operation == "family_lin_to_bf16":
        return _lin_to_bf16(args, context)
    if operation == "family_lin_from_bf16":
        return _lin_from_bf16(args, context)
    if operation == "family_lin_from_f16":
        return _lin_from_f16(args, context)
    if operation == "family_lin_quantize_int8":
        return _lin_quantize_int8(args, context)
    if operation == "family_lin_dequantize_int8":
        return _lin_dequantize_int8(args, context)
    if operation == "family_lin_matmul_bf16":
        return _lin_matmul_bf16(args, context)
    if operation == "family_lin_matmul_int8":
        return _lin_matmul_int8(args, context)
    if operation == "family_lin_quantize_int15":
        return _lin_quantize_int15(args, context)
    if operation == "family_lin_dequantize_int15":
        return _lin_dequantize_int15(args, context)
    if operation == "family_lin_matmul_int15":
        return _lin_matmul_int15(args, context)
    raise Error("RuntimeError: unknown linalg operation " + operation)
