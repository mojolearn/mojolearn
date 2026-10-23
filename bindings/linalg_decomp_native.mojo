# SPDX-License-Identifier: Apache-2.0
# The native owner of python/mojolearn/_expansion_decomp.py's dense linear
# algebra functions: randomized_svd and solve are each one call into
# linalg_decomp_c. The owner runs the steps the Python functions ran
# (`_rsvd_direct_input`, `_rsvd_core`, `_flip_u`; solve's gesv entry or
# lu_factor + lu_solve with their resident, refinement and host routes)
# through bindings/decomp_kit_native.mojo: the same x_decomp entries with the
# same words in the same order, so the bits are unchanged.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from std.memory import memcpy
from bindings.optimizer_public_words import PYTHON_ERROR_PENDING, optimizer_python_error
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type, py_get, py_setitem
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_arrays import classical_address, classical_host_f32, classical_wrap_array, classical_typed
from bindings.host_sparse_native import host_sparse_format
from bindings.decomp_kit_native import (
    DecompKit, DecompF32, decomp_int, decomp_float, decomp_seed, decomp_seed_word, decomp_mode, decomp_entries,
    decomp_input, XD_ABS, XD_ADDS, XD_SCALE, XD_GTS, XD_MUL, XD_ADD,
)


def _same_mode(a: PythonObject, b: PythonObject) raises -> Bool:
    # Whether two numeric modes name the same binding.
    var x = String(py=py_call(py_call(a.strip).lower))
    var y = String(py=py_call(py_call(b.strip).lower))
    return x == y


def _flip_u(mut kit: DecompKit, u: Int, vt: Int, context: PythonObject, mode: PythonObject) raises -> Tuple[Int, Int]:
    # `_flip_u`: svd_flip on U's columns, in the process-default mode's kit.
    var default = context["backend_fields"]["_DEFAULT_MODE"]
    if _same_mode(decomp_mode(mode, context), default):
        var sg = kit.absmax_signs(u, True)
        var u2 = kit.ew(XD_MUL, u, sg)
        var sgt = kit.transpose(sg)
        return (u2, kit.ew(XD_MUL, vt, sgt))
    # Another binding: the matrices move to it and back by host words.
    var other = DecompKit(decomp_entries(default, context), context["output_context"])
    try:
        var ou = other.input(kit.host(u), kit.nrows(u), kit.ncols(u))
        var ov = other.input(kit.host(vt), kit.nrows(vt), kit.ncols(vt))
        var sg = other.absmax_signs(ou, True)
        var u2 = other.ew(XD_MUL, ou, sg)
        var sgt = other.transpose(sg)
        var v2 = other.ew(XD_MUL, ov, sgt)
        var back_u = kit.input(other.host(u2), other.nrows(u2), other.ncols(u2))
        var back_v = kit.input(other.host(v2), other.nrows(v2), other.ncols(v2))
        return (back_u, back_v)
    finally:
        other.close()


def randomized_svd_call(args: PythonObject, context: PythonObject) raises -> PythonObject:
    # args: (M, n_components, n_oversamples, n_iter, power_iteration_normalizer,
    # transpose, flip_sign, random_state, numeric_mode)
    var b = Python.import_module("builtins")
    var mode = args[8]
    var kit = DecompKit(decomp_entries(mode, context), context["output_context"])
    try:
        var transpose_arg = args[5]
        var auto_transpose = Bool(py=py_call(b.isinstance, transpose_arg, b.str)) and String(py=transpose_arg) == "auto"
        var sparse = host_sparse_format(args[0]).byte_length() > 0
        var array = decomp_input(args[0], "M", context)
        var rows = Int(py=array.shape[0])
        var cols = Int(py=array.shape[1])
        var a = kit.input(array, rows, cols)
        # `_rsvd_direct_input` (w4 bit 2): a dense M goes up once from the
        # caller's buffer (RSVD_FAST_DEVSCAN: with the finiteness scan on the
        # device copy); a wide or explicitly transposed M keeps the host route.
        if (kit.flags("x_decomp_w4_flags") & 2) != 0 and not sparse and kit.resident:
            var host_transposes = (transpose_arg is PythonObject(True)) or (auto_transpose and rows < cols)
            if not host_transposes:
                if (kit.flags("x_decomp_s_flags") & 1) != 0:
                    var direct = kit.device_output(rows, cols)
                    var found = Int(py=py_call(kit.entries["x_decomp_dev_upload_scan"], kit.matrices[direct].device,
                                               classical_address(array), rows * cols))
                    if found >= 0:
                        raise Error("M: input must be finite; NaN/inf are unsupported")
                    a = direct
                else:
                    _ = kit.device(a)
        # `_rsvd_core`
        var normalizer = args[4]
        var known = False
        if Bool(py=py_call(b.isinstance, normalizer, b.str)):
            var name = String(py=normalizer)
            known = name == "auto" or name == "QR" or name == "LU" or name == "none"
        if not known:
            raise Error("power_iteration_normalizer must be 'auto', 'QR', 'LU' or 'none'")
        var n = kit.nrows(a)
        var d = kit.ncols(a)
        var kc = decomp_int(args[1])
        var nr = kc + decomp_int(args[2])
        var iterations: Int
        var n_iter = args[3]
        if Bool(py=py_call(b.isinstance, n_iter, b.str)) and String(py=n_iter) == "auto":
            iterations = 7 if Float64(kc) < 0.1 * Float64(min(n, d)) else 4
        else:
            iterations = decomp_int(n_iter)
        var transpose = (n < d) if auto_transpose else Bool(py=transpose_arg)
        if transpose:
            a = kit.transpose(a)
            var t = n
            n = d
            d = t
        if not (1 <= kc and kc <= min(n, d)):
            raise Error("n_components must be in [1, min(n_samples, n_features)]")
        nr = min(nr, min(d, n))
        var q = kit.rand(d, nr, decomp_seed_word(decomp_seed(args[7])), 30, 1)
        for _ in range(iterations):  # driver-loop: over the fixed randomized-SVD power iterations
            var aq = kit.mm(a, q)
            q = kit.orth(aq)
            var atq = kit.mm(a, q, ta=True)
            q = kit.orth(atq)
        var last = kit.mm(a, q)
        q = kit.orth(last)
        # The small SVD runs on the live columns of Q only (DEVIATION 5318).
        var absq = kit.ew(XD_ABS, q)
        var cs = kit.colsum(absq)
        var abscs = kit.ew(XD_ABS, cs)
        var nlive = kit.count_gt(abscs, 0.0)
        if not nlive:
            raise Error("randomized_svd: M is numerically zero")
        var full = kit.ncols(q)
        if nlive < full:
            var abscs2 = kit.ew(XD_ABS, cs)
            var live = kit.ew(XD_GTS, abscs2, scalar=0.0)
            var negated = kit.ew(XD_SCALE, live, scalar=-1.0)
            var dead = kit.ew(XD_ADDS, negated, scalar=1.0)
            var order = kit.order_small(dead)
            q = kit.take_cols_m(q, order, nlive)
        var bm = kit.mm(q, a, ta=True)
        var usv = kit.thin_svd(bm, min(kit.nrows(bm), kit.ncols(bm)), True)
        var uh = usv[0]
        var s = usv[1]
        var vt = usv[2]
        if kit.ncols(s) < full:
            s = kit.grown(s, 1, full)
            vt = kit.grown(vt, full, kit.ncols(vt))
            var uht = kit.transpose(uh)
            var grown = kit.grown(uht, full, kit.ncols(uht))
            uh = kit.transpose(grown)
        var u = kit.mm(q, uh)
        if Bool(py=args[6]):
            if not transpose:
                var flipped = _flip_u(kit, u, vt, context, mode)
                u = flipped[0]
                vt = flipped[1]
            else:
                var vtt = kit.transpose(vt)
                var ut = kit.transpose(u)
                var flipped = _flip_u(kit, vtt, ut, context, mode)
                u = kit.transpose(flipped[1])
                vt = kit.transpose(flipped[0])
        var out_u: Int
        var out_s = kit.cols_of(s, 0, kc)
        var out_vt: Int
        if transpose:
            var head = kit.rows_of(vt, 0, kc)
            out_u = kit.transpose(head)
            var left = kit.cols_of(u, 0, kc)
            out_vt = kit.transpose(left)
        else:
            out_u = kit.cols_of(u, 0, kc)
            out_vt = kit.rows_of(vt, 0, kc)
        return Python.tuple(kit.publish(out_u), kit.out_shape(out_s, Python.tuple(kc)), kit.publish(out_vt))
    finally:
        kit.close()


def linalg_decomp_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        if len(call) != 4:
            raise Error("TypeError: linalg_decomp_native_call requires model, operation, arguments and connected context")
        var b = Python.import_module("builtins")
        if py_type(call[1]) is not b.str or py_type(call[2]) is not b.tuple or py_type(call[3]) is not b.dict:
            raise Error("TypeError: native linear algebra requires an operation, an argument tuple and its context")
        var context = call[3]
        classical_connect_numpy(context)
        var operation = String(py=call[1])
        var result = PythonObject(None)
        if operation == "randomized_svd":
            result = randomized_svd_call(call[2], context)
        else:
            raise Error("unknown native linear algebra operation")
        return result.steal_data()
    except error:
        if String(error) == PYTHON_ERROR_PENDING:
            return optimizer_python_error(error^)
        if String(error).startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if String(error).startswith("NotImplementedError:"):
            return raise_python_exception(error_without_class(error, "NotImplementedError"), ExceptionType("PyExc_NotImplementedError"))
        if String(error).startswith("AttributeError:"):
            return raise_python_exception(error_without_class(error, "AttributeError"), ExceptionType("PyExc_AttributeError"))
        if String(error).startswith("ImportError:"):
            return raise_python_exception(error_without_class(error, "ImportError"), ExceptionType("PyExc_ImportError"))
        if String(error).startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
