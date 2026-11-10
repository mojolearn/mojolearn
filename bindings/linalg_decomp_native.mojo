# SPDX-License-Identifier: Apache-2.0
# The native owner of python/mojolearn/_expansion_decomp.py's dense linear
# algebra functions: randomized_svd, solve, lstsq, lu_factor and lu_solve are
# each one call into linalg_decomp_c. randomized_svd is one compiled driver
# (x_decomp/randomized_svd_owner.mojo); the others run the steps the Python
# functions ran (solve's gesv entry or
# lu_factor + lu_solve with their resident, refinement and host routes)
# through bindings/decomp_kit_native.mojo: the same x_decomp entries with the
# same words in the same order, so the bits are unchanged.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from std.memory import memcpy
from bindings.optimizer_public_words import PYTHON_ERROR_PENDING, optimizer_python_error
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_setattr, py_contains
from bindings.admitted_input import admit_ids
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_arrays import classical_address, classical_wrap_array, classical_typed
from bindings.admitted_input import admit_x
from bindings.host_sparse_native import host_sparse_format
from bindings.classical_validate import classical_view
from core.decomp_native_impl import decomp_piv_valid
from bindings.decomp_kit_native import (
    DecompKit, DecompF32, decomp_int, decomp_float, decomp_seed, decomp_seed_word, decomp_mode, decomp_entries,
    decomp_input, decomp_freeze_array, XD_ABS, XD_ADDS, XD_SCALE, XD_GTS, XD_MUL, XD_ADD, XD_SUB, XD_SQ, XD_SELECT, XD_RECIP,
)


# TOMBSTONE: randomized_svd's per-step DecompKit sequence (`_rsvd_direct_input`, `_rsvd_core`, `_flip_u` in the process-default kit; about 40 py_calls into x_decomp per call) and `_same_mode` deleted 2026-11-08 by lane a3-admit-decomp-prep-metrics; code recoverable at f7d10d2b09.
# Restore: experiments/removed/A3_ADMISSION_HELPERS.patch. Now one driver,
# x_decomp/randomized_svd_owner.mojo (`x_decomp_randomized_svd`).


def randomized_svd_call(args: PythonObject, context: PythonObject) raises -> PythonObject:
    # args: (M, n_components, n_oversamples, n_iter, power_iteration_normalizer,
    # transpose, flip_sign, random_state, numeric_mode)
    # ONE compiled call (x_decomp/randomized_svd_owner.mojo): the sketch, the
    # power iterations, the small SVD and svd_flip run in the selected
    # mode's binding on resident matrices; M is admitted once here (the one
    # admission path: borrowed when C float32, else one parallel cast) and
    # uploaded once by the driver.
    var b = native_loaded_module("builtins")
    var entries = decomp_entries(args[8], context)
    var transpose_arg = args[5]
    var auto_transpose = Bool(py=py_call(b.isinstance, transpose_arg, b.str)) and String(py=transpose_arg) == "auto"
    # RSVD_FAST_DEVSCAN (x_decomp/s_linalg_fast.mojo, `x_decomp_s_flags` bit
    # 1; the FAST + Apple default, rab15: istella 490.1 -> 435.1 ms; deleted
    # 2026-11-08 by lane a3, restored 2026-11-09 by lane
    # r1-restore-measured-winners): M's finite test runs on its device copy
    # inside the one call, not as a host pass over M here.
    var flag_entry = py_get(entries, "x_decomp_s_flags")
    var devscan = flag_entry is not None and (Int(py=py_call(flag_entry)) & 1) != 0
    var array = decomp_input(args[0], "M", context, defer_finite=devscan)
    var rows = Int(py=array.shape[0])
    var cols = Int(py=array.shape[1])
    var normalizer = args[4]
    var known = False
    if Bool(py=py_call(b.isinstance, normalizer, b.str)):
        var name = String(py=normalizer)
        known = name == "auto" or name == "QR" or name == "LU" or name == "none"
    if not known:
        raise Error("power_iteration_normalizer must be 'auto', 'QR', 'LU' or 'none'")
    var kc = decomp_int(args[1])
    var oversamples = decomp_int(args[2])
    var n_iter = args[3]
    var auto_iterations = Bool(py=py_call(b.isinstance, n_iter, b.str)) and String(py=n_iter) == "auto"
    var iterations = 0 if auto_iterations else decomp_int(n_iter)
    var transpose = (rows < cols) if auto_transpose else Bool(py=transpose_arg)
    if not (1 <= kc and kc <= min(rows, cols)):
        raise Error("n_components must be in [1, min(n_samples, n_features)]")
    var seed = decomp_seed_word(decomp_seed(args[7]))
    if py_get(entries, "x_decomp_randomized_svd") is None:
        raise Error("NotImplementedError: mojolearn randomized_svd: this build of _mojolearn_x_decomp does not export "
                    + "x_decomp_randomized_svd; rebuild it with bindings/build_x_decomp.sh")
    var output = context["output_context"]
    var u = classical_typed(Python.tuple(rows, kc), "<f4", output)
    var s = classical_typed(Python.tuple(kc), "<f4", output)
    var vt = classical_typed(Python.tuple(kc, cols), "<f4", output)
    _ = py_call(entries["x_decomp_randomized_svd"], classical_address(array), classical_address(u),
                classical_address(s), classical_address(vt),
                Python.list(rows, cols, kc, oversamples, iterations, Int(auto_iterations), Int(transpose),
                            Int(Bool(py=args[6])), seed, Int(devscan)))
    _ = array  # keep the Python object alive through the native call above
    return Python.tuple(u, s, vt)


def _is_vector(b: PythonObject) raises -> Bool:
    # `len(getattr(b, "shape", ())) == 1 or (not hasattr(b, "shape") and not isinstance(b[0], (list, tuple)))`
    var bi = native_loaded_module("builtins")
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
    var bi = native_loaded_module("builtins")
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
                        + String(py=py_call(native_loaded_module("builtins").repr, view.shape)))
        rows = Int(py=view.shape[0])
        if rows != n:
            raise Error("b has " + String(rows) + " rows, the factorization has " + String(n))
        nrhs = 1
        if n == 0:
            return PythonObject(None)
        try:
            array = admit_x(view, 1, "b", context, admitted=True)
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
    _ = out  # keep the Python object alive through the native call above
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
            _ = a  # keep the Python object alive through the native call above
            _ = py_call(kit.entries["x_decomp_lu_gesv"], aa, kit.address(x), kit.address(info),
                        Python.list(n, kit.ncols(bm)))
            _ = info  # keep the Python object alive through the native call above
            _ = x  # keep the Python object alive through the native call above
            _ = a  # keep the Python object alive through the native call above
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
            _ = array  # keep the Python object alive through the native call above
            _ = info  # keep the Python object alive through the native call above
            if rc < 0:
                raise Error("a: input must be finite; NaN/inf are unsupported")
            _ = info  # keep the Python object alive through the native calls above
            _ = array  # keep the Python object alive through the native calls above
            var singular = Int(DecompF32(unsafe_from_address=Int(py=info._addr))[0])
            _ = info  # keep the Python object alive through the native call above
            if singular != 0:
                _singular_warning(singular, 3)
                _ = info  # keep the Python object alive through the native call above
            else:
                handles.append(kit.matrices[da].device)
                handles.append(kit.matrices[dlu].device)
                handles.append(kit.matrices[dp].device)
            _ = info  # keep the Python object alive through the native calls above
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
        var piv = _pivots(piv_array, context)
        if Int(py=piv.size) != n or not decomp_piv_valid(MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(piv))), n):
            raise Error("piv must hold one row index in [0, n) per row")
        _ = piv  # keep the Python object alive through the native calls above
        var vec = _is_vector(b)
        var bm = _rhs(kit, b, vec, context)
        if kit.nrows(bm) != n:
            raise Error("b has " + String(kit.nrows(bm)) + " rows, the factorization has " + String(n))
        var store = py_call(context["output_context"]["array_type"], PythonObject("i"), py_call(context["output_context"]["native_bytes_type"], 4 * max(n, 1)))
        if n:
            memcpy(dest=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=py_call(store.buffer_info)[0])),
                   src=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(piv))), count=n)
            _ = piv  # keep the Python object alive through the native call above
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


def _pivots(source: PythonObject, context: PythonObject) raises -> PythonObject:
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
    # The shape's repr names a refusal only; built on that path alone.
    if nd != 1:
        raise Error("mojolearn: " + name + " must be 1-D, got " + String(nd) + "-D shape "
                    + String(py=py_call(native_loaded_module("builtins").repr, view.shape)))
    if Int(py=view.shape[0]) == 0:
        var shape = String(py=py_call(native_loaded_module("builtins").repr, view.shape))
        raise Error("mojolearn: " + name + " is empty, shape " + shape + ": Found array with 0 sample(s) (shape="
                    + shape + ") while a minimum of 1 is required.")
    try:
        return admit_x(view, 1, name, context, admitted=True)
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
            _ = b_array  # keep the Python object alive through the native call above
            _ = a_array  # keep the Python object alive through the native call above
            _ = ra  # keep the Python object alive through the native call above
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


# ---------------------------------------------------------------- lu_factor / lu_solve
# The public `lu_factor(a)` and `lu_solve((lu, piv), b)`: the steps the
# Python functions ran (`_lu_factor_resident` / `_lu_solve_resident` on the
# device route, `_Kit.lu` / `_Kit.lu_solve` with LU_QFIX's refinement on the
# host route) through the same x_decomp entries with the same words in the
# same order, so the bits are unchanged. The pair keeps the Python
# `_expansion_decomp._LUPair` (a tuple; `_dev` holds the device A, LU and
# piv as `_DevBuf`s that return to the pool when the pair dies; `_qfix_a`
# this call's float32 copy of A).


def _decomp_module(mode_value: PythonObject, context: PythonObject) raises -> PythonObject:
    # `_Kit._raw()`: the x_decomp binding module of the mode decomp_entries admitted.
    var mode = decomp_mode(mode_value, context)
    var name = String(py=py_call(py_call(mode.strip).lower))
    return context["decomp_modules"][name]


def _device_buffer(module: PythonObject, count: Int, context: PythonObject) raises -> PythonObject:
    # `_DevBuf(raw, count)`: a pooled device buffer id, returned to the pool
    # when the holder dies; the holder is assembled from its slots.
    var bi = native_loaded_module("builtins")
    var holder = py_call(bi.object.__new__, context["lu_device_buffer"])
    py_setattr(holder, "b", module)
    py_setattr(holder, "n", PythonObject(count))
    py_setattr(holder, "id", py_call(module.x_decomp_dev_alloc, max(count, 1)))
    return holder


def _lu_pair(lu: PythonObject, piv: PythonObject, qfix_a: PythonObject, dev: PythonObject,
             context: PythonObject) raises -> PythonObject:
    # `_LUPair((lu, piv), qfix_a, dev=dev)`, assembled from its slots.
    var bi = native_loaded_module("builtins")
    var pair = py_call(bi.tuple.__new__, context["lu_pair_class"], Python.tuple(lu, piv))
    py_setattr(pair, "_qfix_a", qfix_a)
    py_setattr(pair, "_dev", dev)
    return pair


def lu_factor_call(args: PythonObject, context: PythonObject) raises -> PythonObject:
    # args: (a, numeric_mode)
    var kit = DecompKit(decomp_entries(args[1], context), context["output_context"])
    try:
        var a_source = args[0]
        var sf = kit.flags("x_decomp_s_flags")
        var routed = ((sf & 4) != 0 and (kit.flags("x_decomp_qfix_flags") & 4) != 0) or (sf & 8) != 0
        if routed and host_sparse_format(a_source).byte_length() == 0:
            # LU_FAST_RESIDENT (s_flags bit 4, FAST + Apple default) or
            # IDN_LU_RESIDENT_PAIR (bit 8, IDENTICAL default): `_lu_factor_resident`,
            # A up once, the factor on the device, LU and piv down; A, LU and
            # piv stay on the device in the returned pair for lu_solve.
            var array = decomp_input(a_source, "a", context)
            var n = Int(py=array.shape[0])
            if n != Int(py=array.shape[1]):
                raise Error("expected a square matrix, got " + String(n) + " x " + String(Int(py=array.shape[1])))
            var module = _decomp_module(args[1], context)
            var da = _device_buffer(module, n * n, context)
            var dlu = _device_buffer(module, n * n, context)
            var dp = _device_buffer(module, n, context)
            var lu_array = _typed(Python.tuple(n, n), "<f4", "f", 4, context)
            var piv_array = _typed(Python.tuple(n), "<i4", "i", 4, context)
            var info = _typed(Python.tuple(1), "<f4", "f", 4, context)
            var rc = decomp_int(py_call(kit.entries["x_decomp_dev_lu_factor"], classical_address(array), lu_array._addr,
                                        piv_array._addr, info._addr, da.id, dlu.id, dp.id, Python.list(n)))
            _ = array  # keep the Python object alive through the native call above
            _ = info  # keep the Python object alive through the native call above
            if rc < 0:
                raise Error("a: input must be finite; NaN/inf are unsupported")
            var singular = Int(DecompF32(unsafe_from_address=Int(py=info._addr))[0])
            _ = info  # keep the Python object alive through the native call above
            if singular != 0:
                # the device buffers return to the pool with their holders
                _singular_warning(singular, 2)
                return Python.tuple(lu_array, piv_array)
            # Hazard class A (p3 brief): lu_solve serves the device factor
            # while the pair holds these same objects, so they are read-only
            # (an in-place edit raises instead of diverging from the device
            # words; a pair rebuilt from edited copies misses by identity).
            decomp_freeze_array(lu_array)
            decomp_freeze_array(piv_array)
            return _lu_pair(lu_array, piv_array, PythonObject(None),
                            Python.tuple(da, dlu, dp, lu_array, piv_array), context)
        var a = _matrix(kit, a_source, "a", context)
        var n = kit.nrows(a)
        if n != kit.ncols(a):
            raise Error("expected a square matrix, got " + String(n) + " x " + String(kit.ncols(a)))
        var factor = kit.lu(a)
        if factor[2] != 0:
            _singular_warning(factor[2], 2)
        var lu_array = kit.publish(factor[0])
        var piv_array = _typed(Python.tuple(n), "<i4", "i", 4, context)
        if n:
            memcpy(dest=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=piv_array._addr)),
                   src=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=py_call(factor[1].buffer_info)[0])),
                   count=n)
            _ = piv_array  # keep the Python object alive through the native call above
        if factor[2] == 0 and (kit.flags("x_decomp_qfix_flags") & 4) != 0:
            # LU_QFIX (FAST default; -D MOJOLEARN_LU_QOLD off): the pair
            # carries this call's own float32 copy of A so lu_solve can refine
            return _lu_pair(lu_array, piv_array, kit.publish(a), PythonObject(None), context)
        return Python.tuple(lu_array, piv_array)
    finally:
        kit.close()


def _pivots_i32(source: PythonObject, context: PythonObject) raises -> PythonObject:
    # `as_i32_c(piv, ndim=1, name="piv")`: a fresh int32 Array of the pivots.
    # The one integer admission (lane a1b: was labels_view + labels_cast).
    return admit_ids(source, "piv", context, wide=False)


def _lu_solve_trans(mut kit: DecompKit, lu: Int, piv: PythonObject, b: Int, trans: Int) raises -> Int:
    # `_Kit.lu_solve(lu, piv, B, trans)`: on a host copy of B; trans 1 adds
    # the third word, as `p = [n, w, trans] if trans else [n, w]`.
    if trans == 0:
        return kit.lu_solve(lu, piv, b)
    var out = kit.copy_host(b)
    var al = kit.address(lu)
    var ao = kit.address(out)
    _ = py_call(kit.entries["x_decomp_lu_solve"], al, py_call(piv.buffer_info)[0], ao,
                Python.list(kit.nrows(lu), kit.ncols(b), trans))
    kit.wrote_host(out)
    return out


def lu_solve_call(args: PythonObject, context: PythonObject) raises -> PythonObject:
    # args: (lu_and_piv, b, trans, numeric_mode)
    var bi = native_loaded_module("builtins")
    var trans = args[2]
    if not py_contains(Python.tuple(0, 1, 2), trans):
        raise Error("trans must be 0, 1 or 2")
    # `lu, piv = lu_and_piv`, with Python's unpacking refusals
    var pair = args[0]
    var iterable = (Bool(py=py_call(bi.hasattr, pair, PythonObject("__iter__")))
                    or Bool(py=py_call(bi.hasattr, pair, PythonObject("__getitem__"))))
    if not iterable:
        raise Error("TypeError: cannot unpack non-iterable " + String(py=py_type(pair).__name__) + " object")
    var items = py_call(bi.tuple, pair)
    if len(items) < 2:
        raise Error("not enough values to unpack (expected 2, got " + String(len(items)) + ")")
    if len(items) > 2:
        raise Error("too many values to unpack (expected 2)")
    var lu = items[0]
    var piv = items[1]
    var kit = DecompKit(decomp_entries(args[3], context), context["output_context"])
    try:
        var transposed = Bool(py=py_call(bi.bool, trans))
        var dev = py_call(bi.getattr, pair, PythonObject("_dev"), PythonObject(None))
        if (dev is not None and not transposed and lu is dev[3] and piv is dev[4]
                and (kit.flags("x_decomp_s_flags") & 12) != 0):
            # LU_FAST_RESIDENT (bit 4, refinement) or IDN_LU_RESIDENT_PAIR
            # (bit 8, no refinement): the pair's device factor
            var handles = List[Int]()
            handles.append(Int(py=dev[0].id))
            handles.append(Int(py=dev[1].id))
            handles.append(Int(py=dev[2].id))
            var n = Int(py=lu.shape[0])
            var solved = _resident_solve(kit, handles, n, args[1], (kit.flags("x_decomp_s_flags") & 4) != 0, context)
            if solved is not None:
                return solved
        var l = _matrix(kit, lu, "lu", context)
        var n = kit.nrows(l)
        var pa = _pivots_i32(piv, context)
        if Int(py=pa.size) != n or not decomp_piv_valid(MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=pa._addr)), n):
            raise Error("piv must hold one row index in [0, n) per row")
        _ = pa  # keep the Python object alive through the native call above
        var b = args[1]
        var vec = _is_vector(b)
        var bm = _rhs(kit, b, vec, context)
        if kit.nrows(bm) != n:
            raise Error("b has " + String(kit.nrows(bm)) + " rows, the factorization has " + String(n))
        var store = pa._store
        var x = _lu_solve_trans(kit, l, store, bm, 1 if transposed else 0)
        var a0 = py_call(bi.getattr, pair, PythonObject("_qfix_a"), PythonObject(None))
        if (a0 is not None and not transposed and Int(py=a0.shape[0]) == n and Int(py=a0.shape[1]) == n
                and (kit.flags("x_decomp_qfix_flags") & 4) != 0):
            # LU_QFIX: one refinement step, x += LU^-1 (B - A x)
            var am = kit.input(a0, n, n)
            var residual = kit.lu_resid(am, x, bm)
            var step = kit.lu_solve(l, store, residual)
            x = kit.ew(XD_ADD, x, step)
        _ = a0  # keep the Python object alive through the native calls above
        _ = pa  # keep the Python object alive through the native calls above
        return kit.out_shape(x, Python.tuple(n)) if vec else kit.publish(x)
    finally:
        kit.close()


def linalg_decomp_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        if len(call) != 4:
            raise Error("TypeError: linalg_decomp_native_call requires model, operation, arguments and connected context")
        var b = native_loaded_module("builtins")
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
        elif operation == "lu_factor":
            result = lu_factor_call(call[2], context)
        elif operation == "lu_solve":
            result = lu_solve_call(call[2], context)
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
