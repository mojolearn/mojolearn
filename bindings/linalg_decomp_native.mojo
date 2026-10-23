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
from bindings.classical_validate import classical_view
from core.decomp_native_impl import decomp_piv_valid
from bindings.decomp_kit_native import (
    DecompKit, DecompF32, decomp_int, decomp_float, decomp_seed, decomp_seed_word, decomp_mode, decomp_entries,
    decomp_input, XD_ABS, XD_ADDS, XD_SCALE, XD_GTS, XD_MUL, XD_ADD, XD_SUB, XD_SQ, XD_SELECT, XD_RECIP,
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
        if nlive == 0:
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


def _is_vector(b: PythonObject) raises -> Bool:
    # `len(getattr(b, "shape", ())) == 1 or (not hasattr(b, "shape") and not isinstance(b[0], (list, tuple)))`
    var bi = Python.import_module("builtins")
    var shape = py_call(bi.getattr, b, PythonObject("shape"), Python.tuple())
    if len(shape) == 1:
        return True
    if Bool(py=py_call(bi.hasattr, b, PythonObject("shape"))):
        return False
    return not Bool(py=py_call(bi.isinstance, b[0], Python.tuple(bi.list, bi.tuple)))


def _rhs(mut kit: DecompKit, b: PythonObject, vec: Bool, context: PythonObject) raises -> Int:
    # `_M.from_input(_row_of(b), "b").T if vec else _M.from_input(b, "b")`.
    if not vec:
        var array = decomp_input(b, "b", context)
        return kit.input(array, Int(py=array.shape[0]), Int(py=array.shape[1]))
    var bi = Python.import_module("builtins")
    var row: PythonObject
    if py_call(bi.getattr, b, PythonObject("shape"), PythonObject(None)) is not None and Bool(py=py_call(bi.hasattr, b, PythonObject("reshape"))):
        row = py_call(b.reshape, 1, -1)
    else:
        row = Python.list(py_call(bi.list, b))
    var array = decomp_input(row, "b", context)
    var m = kit.input(array, 1, Int(py=array.shape[1]))
    return kit.transpose(m)


def _singular_warning(info: Int, stacklevel: Int) raises:
    var bi = Python.import_module("builtins")
    _ = py_call(Python.import_module("_warnings").warn,
                PythonObject("Diagonal number " + String(info) + " is exactly zero. Singular matrix."),
                bi.RuntimeWarning, stacklevel)


def _typed(shape: PythonObject, typestr: String, code: String, item: Int, context: PythonObject) raises -> PythonObject:
    var output = context["output_context"]
    var count = 1
    for k in range(len(shape)):  # driver-loop: over array rank metadata
        count *= Int(py=shape[k])
    var store = py_call(output["array_type"], PythonObject(code), py_call(output["native_bytes_type"], count * item))
    return classical_wrap_array(store, shape, typestr, "C", output)


def _resident_solve(mut kit: DecompKit, handles: List[Int], n: Int, b: PythonObject, refine: Bool,
                    context: PythonObject) raises -> PythonObject:
    # `_lu_solve_resident`: B up, the solve on the device factor, X down
    # (None for an empty system: the host route answers it).
    var vec = _is_vector(b)
    var array: PythonObject
    var rows: Int
    var nrhs: Int
    if vec:
        var view = classical_view(b, context, "b")
        var nd = Int(py=view.ndim)
        if nd != 1:
            raise Error("mojolearn: b must be 1-D, got " + String(nd) + "-D shape "
                        + String(py=py_call(Python.import_module("builtins").repr, view.shape)))
        rows = Int(py=view.shape[0])
        if rows != n:
            raise Error("b has " + String(rows) + " rows, the factorization has " + String(n))
        nrhs = 1
        if n == 0:
            return PythonObject(None)
        try:
            array = classical_host_f32(view, 1, "b", context)
        except error:
            if String(error) == "Input b contains NaN or infinity":
                raise Error("b: input must be finite; NaN/inf are unsupported")
            raise Error(String(error))
    else:
        array = decomp_input(b, "b", context)
        rows = Int(py=array.shape[0])
        if rows != n:
            raise Error("b has " + String(rows) + " rows, the factorization has " + String(n))
        nrhs = Int(py=array.shape[1])
    var out = _typed(Python.tuple(n) if vec else Python.tuple(n, nrhs), "<f4", "f", 4, context)
    _ = py_call(kit.entries["x_decomp_dev_lu_solve"], handles[1], handles[2], handles[0], classical_address(array),
                out._addr, Python.list(n, nrhs, 1 if refine else 0))
    return out


def solve_call(args: PythonObject, context: PythonObject) raises -> PythonObject:
    # args: (a, b, numeric_mode)
    var kit = DecompKit(decomp_entries(args[2], context), context["output_context"])
    try:
        var a_source = args[0]
        var b = args[1]
        if (kit.flags("x_decomp_idn_flags") & 2) != 0:
            # gesv: one binding entry, the factor and pivots resident.
            var a = _matrix(kit, a_source, "a", context)
            var n = kit.nrows(a)
            if n != kit.ncols(a):
                raise Error("expected a square matrix, got " + String(n) + " x " + String(kit.ncols(a)))
            var vec = _is_vector(b)
            var bm = _rhs(kit, b, vec, context)
            if kit.nrows(bm) != n:
                raise Error("b has " + String(kit.nrows(bm)) + " rows, the factorization has " + String(n))
            var x = kit.copy_host(bm)
            var info = kit.zeros(1, 1)
            var aa = kit.address(a)
            _ = py_call(kit.entries["x_decomp_lu_gesv"], aa, kit.address(x), kit.address(info),
                        Python.list(n, kit.ncols(bm)))
            kit.wrote_host(x)
            var singular = Int(kit.word(info))
            if singular != 0:
                _singular_warning(singular, 2)
            return kit.out_shape(x, Python.tuple(n)) if vec else kit.publish(x)
        # lu_factor(a), then lu_solve(pair, b)
        var sf = kit.flags("x_decomp_s_flags")
        var qf = kit.flags("x_decomp_qfix_flags")
        var lu_array: PythonObject
        var piv_array: PythonObject
        var qfix_a = -1
        var handles = List[Int]()
        if (((sf & 4) != 0 and (qf & 4) != 0) or (sf & 8) != 0) and host_sparse_format(a_source).byte_length() == 0:
            # `_lu_factor_resident`: A up once, the factor on the device, LU and piv down.
            var array = decomp_input(a_source, "a", context)
            var n = Int(py=array.shape[0])
            if n != Int(py=array.shape[1]):
                raise Error("expected a square matrix, got " + String(n) + " x " + String(Int(py=array.shape[1])))
            var da = kit.device_output(n, n)
            var dlu = kit.device_output(n, n)
            var dp = kit.device_output(n, 1)
            lu_array = _typed(Python.tuple(n, n), "<f4", "f", 4, context)
            piv_array = _typed(Python.tuple(n), "<i4", "i", 4, context)
            var info = _typed(Python.tuple(1), "<f4", "f", 4, context)
            var rc = decomp_int(py_call(kit.entries["x_decomp_dev_lu_factor"], classical_address(array), lu_array._addr,
                                        piv_array._addr, info._addr, kit.matrices[da].device,
                                        kit.matrices[dlu].device, kit.matrices[dp].device, Python.list(n)))
            if rc < 0:
                raise Error("a: input must be finite; NaN/inf are unsupported")
            var singular = Int(DecompF32(unsafe_from_address=Int(py=info._addr))[0])
            if singular != 0:
                _singular_warning(singular, 3)
            else:
                handles.append(kit.matrices[da].device)
                handles.append(kit.matrices[dlu].device)
                handles.append(kit.matrices[dp].device)
        else:
            var a = _matrix(kit, a_source, "a", context)
            var n = kit.nrows(a)
            if n != kit.ncols(a):
                raise Error("expected a square matrix, got " + String(n) + " x " + String(kit.ncols(a)))
            var factor = kit.lu(a)
            if factor[2] != 0:
                _singular_warning(factor[2], 2)
            lu_array = kit.publish(factor[0])
            piv_array = _typed(Python.tuple(n), "<i4", "i", 4, context)
            if n:
                memcpy(dest=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=piv_array._addr)),
                       src=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=py_call(factor[1].buffer_info)[0])),
                       count=n)
            if factor[2] == 0 and (qf & 4) != 0:
                qfix_a = a
        # lu_solve (trans 0)
        if len(handles) > 0 and (kit.flags("x_decomp_s_flags") & 12) != 0:
            var n = Int(py=lu_array.shape[0])
            var solved = _resident_solve(kit, handles, n, b, (kit.flags("x_decomp_s_flags") & 4) != 0, context)
            if solved is not None:
                return solved
        var l = _matrix(kit, lu_array, "lu", context)
        var n = kit.nrows(l)
        var piv = classical_host_i32(piv_array, context)
        if Int(py=piv.size) != n or not decomp_piv_valid(MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(piv))), n):
            raise Error("piv must hold one row index in [0, n) per row")
        var vec = _is_vector(b)
        var bm = _rhs(kit, b, vec, context)
        if kit.nrows(bm) != n:
            raise Error("b has " + String(kit.nrows(bm)) + " rows, the factorization has " + String(n))
        var store = py_call(context["output_context"]["array_type"], PythonObject("i"), py_call(context["output_context"]["native_bytes_type"], 4 * max(n, 1)))
        if n:
            memcpy(dest=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=py_call(store.buffer_info)[0])),
                   src=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(piv))), count=n)
        var x = kit.lu_solve(l, store, bm)
        if qfix_a >= 0 and kit.nrows(qfix_a) == n and kit.ncols(qfix_a) == n and (kit.flags("x_decomp_qfix_flags") & 4) != 0:
            # LU_QFIX: one refinement step, x += LU^-1 (B - A x)
            var residual = kit.lu_resid(qfix_a, x, bm)
            var step = kit.lu_solve(l, store, residual)
            x = kit.ew(XD_ADD, x, step)
        return kit.out_shape(x, Python.tuple(n)) if vec else kit.publish(x)
    finally:
        kit.close()


def _matrix(mut kit: DecompKit, source: PythonObject, name: String, context: PythonObject) raises -> Int:
    var array = decomp_input(source, name, context)
    return kit.input(array, Int(py=array.shape[0]), Int(py=array.shape[1]))


def classical_host_i32(source: PythonObject, context: PythonObject) raises -> PythonObject:
    # `as_i32_c(piv, ndim=1)` of the pivots this call made (an int32 Array).
    if Int(py=source.ndim) != 1 or String(py=source.dtype) != "<i4":
        raise Error("TypeError: piv must be a 1-D int32 array")
    return source


comptime LINALG_F32_EPS = 1.1920928955078125e-07
comptime LINALG_TS_MAX_N = 512


def _f32_input(source: PythonObject, name: String, ndim: Int, context: PythonObject) raises -> PythonObject:
    # `_f32_input`: a SciPy sparse input densified, a nonempty float32 C
    # array of `ndim` dimensions, non-finite values refused.
    if ndim == 2:
        return decomp_input(source, name, context)
    var view = classical_view(source, context, name)
    var nd = Int(py=view.ndim)
    var shape = String(py=py_call(Python.import_module("builtins").repr, view.shape))
    if nd != 1:
        raise Error("mojolearn: " + name + " must be 1-D, got " + String(nd) + "-D shape " + shape)
    if Int(py=view.shape[0]) == 0:
        raise Error("mojolearn: " + name + " is empty, shape " + shape + ": Found array with 0 sample(s) (shape="
                    + shape + ") while a minimum of 1 is required.")
    try:
        return classical_host_f32(view, 1, name, context)
    except error:
        if String(error) == "Input " + name + " contains NaN or infinity":
            raise Error(name + ": input must be finite; NaN/inf are unsupported")
        raise Error(String(error))


def _cut(mut kit: DecompKit, s: Int, rcond: Float64) raises -> Float64:
    # `_f32(k.word(S, 0) * rcond)`
    return Float64(Float32(kit.word(s, 0) * rcond))


def _pseudo_solve(mut kit: DecompKit, u: Int, s: Int, vt: Int, rhs: Int, cut: Float64) raises -> Int:
    # Vt^T ((U^T B) * (1 / S kept)^T): the values at or below cut dropped.
    var zero = kit.zeros(1, 1)
    var kept = kit.ew(XD_SELECT, s, s, zero, cut)
    var inverse = kit.ew(XD_RECIP, kept)
    var utb = kit.mm(u, rhs, ta=True)
    var inverse_t = kit.transpose(inverse)
    var scaled = kit.ew(XD_MUL, utb, inverse_t)
    return kit.mm(vt, scaled, ta=True)


def lstsq_call(args: PythonObject, context: PythonObject) raises -> PythonObject:
    # args: (a, b, rcond, numeric_mode)
    var kit = DecompKit(decomp_entries(args[3], context), context["output_context"])
    try:
        var b = args[1]
        var vec = _is_vector(b)
        var a_array = _f32_input(args[0], "a", 2, context)
        var m = Int(py=a_array.shape[0])
        var nn = Int(py=a_array.shape[1])
        var b_array = _f32_input(b, "b", 1 if vec else 2, context)
        var nrhs = 1 if vec else Int(py=b_array.shape[1])
        if Int(py=b_array.shape[0]) != m:
            raise Error("Incompatible dimensions")
        var rcond = LINALG_F32_EPS * Float64(max(m, nn)) if args[2] is None else decomp_float(args[2])
        var empty = _typed(Python.tuple(0), "<f4", "f", 4, context)
        if nn >= 1 and nrhs >= 1 and nn + nrhs <= LINALG_TS_MAX_N and m >= nn + nrhs:
            # `_tsqr_lstsq_core`: the blocked TSQR of [a | b], the SVD of its small R.
            var n = nn + nrhs
            var ra = kit.zeros(n, n)
            _ = py_call(kit.entries["x_decomp_tsqr_r"], classical_address(a_array), classical_address(b_array),
                        kit.address(ra), Python.list(m, nn, nrhs, 0))
            kit.wrote_host(ra)
            var top = kit.rows_of(ra, 0, nn)
            var r = kit.cols_of(top, 0, nn)
            var c = kit.cols_of(top, nn, n)
            var usv = kit.svd_tall(r)
            var cut = _cut(kit, usv[1], rcond)
            var rank = kit.count_gt(usv[1], cut)
            var x = _pseudo_solve(kit, usv[0], usv[1], usv[2], c, cut)
            var resid = empty
            if rank == nn and m > nn:
                var tail = kit.rows_of(ra, nn, n)
                var last = kit.cols_of(tail, nn, n)
                var squares = kit.ew(XD_SQ, last)
                resid = kit.out_shape(kit.colsum(squares), Python.tuple(nrhs))
            var solution = kit.out_shape(x, Python.tuple(nn)) if vec else kit.publish(x)
            return Python.tuple(solution, resid, rank, kit.out_shape(usv[1], Python.tuple(nn)))
        # The QR + one-sided Jacobi SVD of a (or of a^T).
        var a = kit.input(a_array, m, nn)
        var bm = kit.input(b_array, m, nrhs)
        var r = min(m, nn)
        var usv = kit.thin_svd(a, r, True)
        var cut = _cut(kit, usv[1], rcond) if r > 0 else Float64(0.0)
        var rank = kit.count_gt(usv[1], cut)
        var x = _pseudo_solve(kit, usv[0], usv[1], usv[2], bm, cut)
        var resid = empty
        if rank == nn and m > nn:
            var fitted = kit.mm(a, x)
            var diff = kit.ew(XD_SUB, bm, fitted)
            var squares = kit.ew(XD_SQ, diff)
            resid = kit.out_shape(kit.colsum(squares), Python.tuple(nrhs))
        var solution = kit.out_shape(x, Python.tuple(nn)) if vec else kit.publish(x)
        return Python.tuple(solution, resid, rank, kit.out_shape(usv[1], Python.tuple(r)))
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
        elif operation == "solve":
            result = solve_call(call[2], context)
        elif operation == "lstsq":
            result = lstsq_call(call[2], context)
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
