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
from bindings.native_loaded_module_door import native_loaded_module
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from std.memory import unsafe_memcpy
from bindings.optimizer_public_words_unpack import PYTHON_ERROR_PENDING, optimizer_python_error
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_setattr, py_contains
from bindings.admitted_input_unpack import admit_ids
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.classical_arrays_door import classical_address, classical_wrap_array, classical_typed
from bindings.admitted_input_unpack import admit_x
from bindings.host_sparse_native_door import host_sparse_format
from bindings.classical_validate_unpack import classical_view
from core.decomp_native_impl import decomp_piv_valid
from bindings.decomp_kit_native_door import (
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


def _matrix(mut door: DecompDoor, mut kit: DKit, source: PythonObject, name: String,
            context: PythonObject) raises -> Int:
    return decomp_door_input(door, kit, source, name, context)


def _pivots(source: PythonObject, context: PythonObject) raises -> PythonObject:
    # `as_i32_c(piv, ndim=1)` of the pivots this call made (an int32 Array).
    if Int(py=source.ndim) != 1 or String(py=source.dtype) != "<i4":
        raise Error("TypeError: piv must be a 1-D int32 array")
    return source


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


def _decomp_module(mode_value: PythonObject, context: PythonObject) raises -> PythonObject:
    # `_Kit._raw()`: the x_decomp binding module of the mode decomp_entries admitted.
    var mode = decomp_mode(mode_value, context)
    var name = String(py=py_call(py_call(mode.strip).lower))
    return context["decomp_modules"][name]


def _pivots_i32(source: PythonObject, context: PythonObject) raises -> PythonObject:
    # `as_i32_c(piv, ndim=1, name="piv")`: a fresh int32 Array of the pivots.
    # The one integer admission (lane a1b: was labels_view + labels_cast).
    return admit_ids(source, "piv", context, wide=False)
