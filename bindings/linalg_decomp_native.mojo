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
from std.memory import unsafe_memcpy
from bindings.optimizer_public_words import PYTHON_ERROR_PENDING, optimizer_python_error
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_setattr, py_contains
from bindings.admitted_input import admit_ids
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_arrays import classical_address, classical_wrap_array, classical_typed
from bindings.admitted_input import admit_x
from bindings.host_sparse_native_door import host_sparse_format
from bindings.classical_validate import classical_view
from core.decomp_native_impl import decomp_piv_valid
from bindings.decomp_kit_native import (
    DecompDoor, DecompF32, decomp_int, decomp_float, decomp_seed, decomp_seed_word, decomp_mode, decomp_entries,
    decomp_input, decomp_freeze_array, decomp_kit_unpack, decomp_kit_publish, decomp_kit_out_shape,
    decomp_door_borrow, decomp_door_input,
)
from bindings.decomp_kit_run import DKit, DK_HAS_S_FLAGS, DK_HAS_QFIX_FLAGS, DK_HAS_IDN_FLAGS
from bindings.linalg_decomp_run import (
    LstsqOut, ld_rsvd_run, ld_gesv_run, ld_dev_lu_factor_run, ld_device_buffers, ld_resident_solve_run,
    ld_lu_solve_run, ld_lstsq_run,
)
from std.python._cpython import GILReleased


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
    # The route word comes from the image's table (XD_S_FLAGS; the host
    # image answers 0, as a binding without `x_decomp_s_flags` read).
    var kit = decomp_kit_unpack(entries)
    var devscan = (kit.flags(DK_HAS_S_FLAGS) & 1) != 0
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
    var output = context["output_context"]
    var u = classical_typed(Python.tuple(rows, kc), "<f4", output)
    var s = classical_typed(Python.tuple(kc), "<f4", output)
    var vt = classical_typed(Python.tuple(kc, cols), "<f4", output)
    # XD_RANDOMIZED_SVD (was the Python entry x_decomp_randomized_svd)
    var addresses: List[Int] = [Int(py=classical_address(array)), Int(py=classical_address(u)),
                                Int(py=classical_address(s)), Int(py=classical_address(vt))]
    var params: List[Int] = [rows, cols, kc, oversamples, iterations, Int(auto_iterations), Int(transpose),
                             Int(Bool(py=args[6])), seed, Int(devscan)]
    var py = Python()
    with GILReleased(py):
        ld_rsvd_run(kit, addresses, params)
    _ = array  # keep the Python object alive through the native call above
    _ = u
    _ = s
    _ = vt
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


def _rhs(mut door: DecompDoor, mut kit: DKit, b: PythonObject, vec: Bool, context: PythonObject) raises -> Int:
    # `_M.from_input(_row_of(b), "b").T if vec else _M.from_input(b, "b")`
    # (the transpose of a row is a view: no operation runs).
    if not vec:
        return decomp_door_input(door, kit, b, "b", context)
    var bi = native_loaded_module("builtins")
    var row: PythonObject
    if py_call(bi.getattr, b, PythonObject("shape"), PythonObject(None)) is not None and Bool(py=py_call(bi.hasattr, b, PythonObject("reshape"))):
        row = py_call(b.reshape, 1, -1)
    else:
        row = Python.list(py_call(bi.list, b))
    var array = decomp_input(row, "b", context)
    var m = decomp_door_borrow(door, kit, array, 1, Int(py=array.shape[1]))
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


def _resident_solve(mut kit: DKit, da: Int, dlu: Int, dp: Int, n: Int, b: PythonObject, refine: Bool,
                    context: PythonObject) raises -> PythonObject:
    # `_lu_solve_resident`: UNPACK b, RUN B up, the solve on the device
    # factor, X down, PACK the Array X was written into (None for an empty
    # system: the host route answers it).
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
    var b_address = Int(py=classical_address(array))
    var out_address = Int(py=out._addr)
    var py = Python()
    with GILReleased(py):
        ld_resident_solve_run(kit, b_address, out_address, da, dlu, dp, n, nrhs, refine)
    _ = array  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    return out


def solve_call(args: PythonObject, context: PythonObject) raises -> PythonObject:
    # args: (a, b, numeric_mode)
    var entries = decomp_entries(args[2], context)
    var door = DecompDoor(entries, context["output_context"])
    var kit = decomp_kit_unpack(entries)
    try:
        var a_source = args[0]
        var b = args[1]
        var py = Python()
        if (kit.flags(DK_HAS_IDN_FLAGS) & 2) != 0:
            # gesv: one entry, the factor and pivots resident.
            var a = _matrix(door, kit, a_source, "a", context)
            var n = kit.nrows(a)
            if n != kit.ncols(a):
                raise Error("expected a square matrix, got " + String(n) + " x " + String(kit.ncols(a)))
            var vec = _is_vector(b)
            var bm = _rhs(door, kit, b, vec, context)
            if kit.nrows(bm) != n:
                raise Error("b has " + String(kit.nrows(bm)) + " rows, the factorization has " + String(n))
            var x = -1
            var singular = 0
            with GILReleased(py):
                var solved = ld_gesv_run(kit, a, bm)
                x = solved[0]
                singular = solved[1]
            if singular != 0:
                _singular_warning(singular, 2)
            return decomp_kit_out_shape(door, kit, x, Python.tuple(n)) if vec else decomp_kit_publish(door, kit, x)
        # lu_factor(a), then lu_solve(pair, b)
        var sf = kit.flags(DK_HAS_S_FLAGS)
        var qf = kit.flags(DK_HAS_QFIX_FLAGS)
        var lu_array: PythonObject
        var piv_array: PythonObject
        var qfix_a = -1
        var resident = False
        var da = -1
        var dlu = -1
        var dp = -1
        if (((sf & 4) != 0 and (qf & 4) != 0) or (sf & 8) != 0) and host_sparse_format(a_source).byte_length() == 0:
            # `_lu_factor_resident`: A up once, the factor on the device, LU and piv down.
            var array = decomp_input(a_source, "a", context)
            var n = Int(py=array.shape[0])
            if n != Int(py=array.shape[1]):
                raise Error("expected a square matrix, got " + String(n) + " x " + String(Int(py=array.shape[1])))
            lu_array = _typed(Python.tuple(n, n), "<f4", "f", 4, context)
            piv_array = _typed(Python.tuple(n), "<i4", "i", 4, context)
            var info = _typed(Python.tuple(1), "<f4", "f", 4, context)
            var a_address = Int(py=classical_address(array))
            var lu_address = Int(py=lu_array._addr)
            var piv_address = Int(py=piv_array._addr)
            var info_address = Int(py=info._addr)
            var rc = 0
            with GILReleased(py):
                var buffers = ld_device_buffers(kit, n)
                da = buffers[0]
                dlu = buffers[1]
                dp = buffers[2]
                rc = ld_dev_lu_factor_run(kit, a_address, lu_address, piv_address, info_address, da, dlu, dp, n)
            _ = array  # keep the Python object alive through the native call above
            if rc < 0:
                raise Error("a: input must be finite; NaN/inf are unsupported")
            var singular = Int(DecompF32(unsafe_from_address=info_address)[0])
            _ = info  # keep the Python object alive through the native call above
            if singular != 0:
                _singular_warning(singular, 3)
            else:
                resident = True
        else:
            var a = _matrix(door, kit, a_source, "a", context)
            var n = kit.nrows(a)
            if n != kit.ncols(a):
                raise Error("expected a square matrix, got " + String(n) + " x " + String(kit.ncols(a)))
            var lu = -1
            var piv = -1
            var info = 0
            with GILReleased(py):
                var factor = kit.lu(a)
                lu = factor[0]
                piv = factor[1]
                info = factor[2]
            if info != 0:
                _singular_warning(info, 2)
            lu_array = decomp_kit_publish(door, kit, lu)
            piv_array = _typed(Python.tuple(n), "<i4", "i", 4, context)
            if n:
                unsafe_memcpy(dest=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=piv_array._addr)),
                       src=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=kit.iaddr(piv)), count=n)
            if info == 0 and (qf & 4) != 0:
                qfix_a = a
        # lu_solve (trans 0)
        if resident and (sf & 12) != 0:
            var n = Int(py=lu_array.shape[0])
            var solved = _resident_solve(kit, da, dlu, dp, n, b, (sf & 4) != 0, context)
            if solved is not None:
                return solved
        var l = _matrix(door, kit, lu_array, "lu", context)
        var n = kit.nrows(l)
        var piv = _pivots(piv_array, context)
        var piv_address = Int(py=classical_address(piv))
        if Int(py=piv.size) != n or not decomp_piv_valid(MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=piv_address), n):
            raise Error("piv must hold one row index in [0, n) per row")
        door.keep(piv)
        var vec = _is_vector(b)
        var bm = _rhs(door, kit, b, vec, context)
        if kit.nrows(bm) != n:
            raise Error("b has " + String(kit.nrows(bm)) + " rows, the factorization has " + String(n))
        var refine = qfix_a >= 0 and kit.nrows(qfix_a) == n and kit.ncols(qfix_a) == n and (qf & 4) != 0
        var x = -1
        with GILReleased(py):
            x = ld_lu_solve_run(kit, l, piv_address, bm, 0, qfix_a if refine else -1)
        return decomp_kit_out_shape(door, kit, x, Python.tuple(n)) if vec else decomp_kit_publish(door, kit, x)
    finally:
        kit.close()
        door.keep_alive()  # the borrowed owners outlive every RUN read


def _matrix(mut door: DecompDoor, mut kit: DKit, source: PythonObject, name: String,
            context: PythonObject) raises -> Int:
    return decomp_door_input(door, kit, source, name, context)


def _pivots(source: PythonObject, context: PythonObject) raises -> PythonObject:
    # `as_i32_c(piv, ndim=1)` of the pivots this call made (an int32 Array).
    if Int(py=source.ndim) != 1 or String(py=source.dtype) != "<i4":
        raise Error("TypeError: piv must be a 1-D int32 array")
    return source


comptime LINALG_F32_EPS = 1.1920928955078125e-07


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


def lstsq_call(args: PythonObject, context: PythonObject) raises -> PythonObject:
    # args: (a, b, rcond, numeric_mode): one UNPACK, one RUN, one PACK.
    var entries = decomp_entries(args[3], context)
    var door = DecompDoor(entries, context["output_context"])
    var kit = decomp_kit_unpack(entries)
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
        var a = decomp_door_borrow(door, kit, a_array, m, nn)
        var bm = decomp_door_borrow(door, kit, b_array, m, nrhs)
        var out = LstsqOut(-1, -1, 0, -1)
        var py = Python()
        with GILReleased(py):
            out = ld_lstsq_run(kit, a, bm, rcond)
        var resid = _typed(Python.tuple(0), "<f4", "f", 4, context)
        if out.resid >= 0:
            resid = decomp_kit_out_shape(door, kit, out.resid, Python.tuple(nrhs))
        var solution = decomp_kit_out_shape(door, kit, out.x, Python.tuple(nn)) if vec else decomp_kit_publish(door, kit, out.x)
        return Python.tuple(solution, resid, out.rank, decomp_kit_out_shape(door, kit, out.s, Python.tuple(kit.ncols(out.s))))
    finally:
        kit.close()
        door.keep_alive()  # the borrowed owners outlive every RUN read


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


def _device_buffer(kit: DKit, module: PythonObject, count: Int, context: PythonObject) raises -> PythonObject:
    # `_DevBuf(raw, count)`: a pooled device buffer id, returned to the pool
    # when the holder dies; the holder is assembled from its slots. The id
    # comes from the image's table (XD_DEV_ALLOC; was a Python call of the
    # module's x_decomp_dev_alloc): the same pool the holder frees into.
    var bi = native_loaded_module("builtins")
    var holder = py_call(bi.object.__new__, context["lu_device_buffer"])
    py_setattr(holder, "b", module)
    py_setattr(holder, "n", PythonObject(count))
    py_setattr(holder, "id", PythonObject(kit.dev_alloc(max(count, 1))))
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
    var entries = decomp_entries(args[1], context)
    var door = DecompDoor(entries, context["output_context"])
    var kit = decomp_kit_unpack(entries)
    try:
        var a_source = args[0]
        var py = Python()
        var sf = kit.flags(DK_HAS_S_FLAGS)
        var routed = ((sf & 4) != 0 and (kit.flags(DK_HAS_QFIX_FLAGS) & 4) != 0) or (sf & 8) != 0
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
            var da = _device_buffer(kit, module, n * n, context)
            var dlu = _device_buffer(kit, module, n * n, context)
            var dp = _device_buffer(kit, module, n, context)
            var lu_array = _typed(Python.tuple(n, n), "<f4", "f", 4, context)
            var piv_array = _typed(Python.tuple(n), "<i4", "i", 4, context)
            var info = _typed(Python.tuple(1), "<f4", "f", 4, context)
            var a_address = Int(py=classical_address(array))
            var lu_address = Int(py=lu_array._addr)
            var piv_address = Int(py=piv_array._addr)
            var info_address = Int(py=info._addr)
            var ids: List[Int] = [Int(py=da.id), Int(py=dlu.id), Int(py=dp.id)]
            var rc = 0
            with GILReleased(py):
                rc = ld_dev_lu_factor_run(kit, a_address, lu_address, piv_address, info_address, ids[0], ids[1],
                                          ids[2], n)
            _ = array  # keep the Python object alive through the native call above
            if rc < 0:
                raise Error("a: input must be finite; NaN/inf are unsupported")
            var singular = Int(DecompF32(unsafe_from_address=info_address)[0])
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
        var a = _matrix(door, kit, a_source, "a", context)
        var n = kit.nrows(a)
        if n != kit.ncols(a):
            raise Error("expected a square matrix, got " + String(n) + " x " + String(kit.ncols(a)))
        var lu = -1
        var piv = -1
        var info = 0
        with GILReleased(py):
            var factor = kit.lu(a)
            lu = factor[0]
            piv = factor[1]
            info = factor[2]
        if info != 0:
            _singular_warning(info, 2)
        var lu_array = decomp_kit_publish(door, kit, lu)
        var piv_array = _typed(Python.tuple(n), "<i4", "i", 4, context)
        if n:
            unsafe_memcpy(dest=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=piv_array._addr)),
                   src=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=kit.iaddr(piv)), count=n)
            _ = piv_array  # keep the Python object alive through the native call above
        if info == 0 and (kit.flags(DK_HAS_QFIX_FLAGS) & 4) != 0:
            # LU_QFIX (FAST default; -D MOJOLEARN_LU_QOLD off): the pair
            # carries this call's own float32 copy of A so lu_solve can refine
            return _lu_pair(lu_array, piv_array, decomp_kit_publish(door, kit, a), PythonObject(None), context)
        return Python.tuple(lu_array, piv_array)
    finally:
        kit.close()
        door.keep_alive()  # the borrowed owners outlive every RUN read


def _pivots_i32(source: PythonObject, context: PythonObject) raises -> PythonObject:
    # `as_i32_c(piv, ndim=1, name="piv")`: a fresh int32 Array of the pivots.
    # The one integer admission (lane a1b: was labels_view + labels_cast).
    return admit_ids(source, "piv", context, wide=False)


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
    var entries = decomp_entries(args[3], context)
    var door = DecompDoor(entries, context["output_context"])
    var kit = decomp_kit_unpack(entries)
    try:
        var transposed = Bool(py=py_call(bi.bool, trans))
        var dev = py_call(bi.getattr, pair, PythonObject("_dev"), PythonObject(None))
        var sf = kit.flags(DK_HAS_S_FLAGS)
        if (dev is not None and not transposed and lu is dev[3] and piv is dev[4]
                and (sf & 12) != 0):
            # LU_FAST_RESIDENT (bit 4, refinement) or IDN_LU_RESIDENT_PAIR
            # (bit 8, no refinement): the pair's device factor (the pair's
            # holders own the buffers; `dev` is named below past the solve)
            var n = Int(py=lu.shape[0])
            var solved = _resident_solve(kit, Int(py=dev[0].id), Int(py=dev[1].id), Int(py=dev[2].id), n, args[1],
                                         (sf & 4) != 0, context)
            _ = dev  # keep the pair's device holders alive through the solve above
            if solved is not None:
                return solved
        var l = _matrix(door, kit, lu, "lu", context)
        var n = kit.nrows(l)
        var pa = _pivots_i32(piv, context)
        if Int(py=pa.size) != n or not decomp_piv_valid(MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=pa._addr)), n):
            raise Error("piv must hold one row index in [0, n) per row")
        _ = pa  # keep the Python object alive through the native call above
        var b = args[1]
        var vec = _is_vector(b)
        var bm = _rhs(door, kit, b, vec, context)
        if kit.nrows(bm) != n:
            raise Error("b has " + String(kit.nrows(bm)) + " rows, the factorization has " + String(n))
        door.keep(pa)
        var piv_address = Int(py=pa._addr)
        var a0 = py_call(bi.getattr, pair, PythonObject("_qfix_a"), PythonObject(None))
        var am = -1
        if (a0 is not None and not transposed and Int(py=a0.shape[0]) == n and Int(py=a0.shape[1]) == n
                and (kit.flags(DK_HAS_QFIX_FLAGS) & 4) != 0):
            # LU_QFIX: one refinement step, x += LU^-1 (B - A x), on the
            # pair's own float32 copy of A (borrowed as it is)
            am = decomp_door_borrow(door, kit, a0, n, n)
        var x = -1
        var py = Python()
        with GILReleased(py):
            x = ld_lu_solve_run(kit, l, piv_address, bm, 1 if transposed else 0, am)
        return decomp_kit_out_shape(door, kit, x, Python.tuple(n)) if vec else decomp_kit_publish(door, kit, x)
    finally:
        kit.close()
        door.keep_alive()  # the borrowed owners outlive every RUN read


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
