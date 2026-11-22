# SPDX-License-Identifier: Apache-2.0
# RobustScaler owns its existing x_prep stage sequence and public lifecycle.
from std.os import getenv
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem, py_contains
# TOMBSTONE: P1_NATIVE_SKLEARN_TAGS (removed: native code importing sklearn.utils._tags and building its Tag dataclasses; the tags are built by the Python compatibility layer, _mode.sklearn_tags) deleted 2026-11-09 by lane p1-python-callbacks-worst; code recoverable at e68654a757.
# Restore: experiments/removed/P1_NATIVE_SKLEARN_TAGS.patch.
from bindings.classical_attributes_door import classical_connect_numpy, classical_check_layout, classical_getattribute
from bindings.classical_arrays_door import classical_typed, classical_is_array, classical_input
from bindings.native_numpy_metadata_unpack import native_numpy_dtype
from bindings.classical_warnings_unpack import classical_warning, _warning_fields
from providers.python.linear_metadata_boundary_door import classical_is_fitted
from bindings.classical_exceptions_door import classical_not_fitted_class
from bindings.causal_exception_unpack import causal_borrowed_item, causal_set_exception
from bindings.sequential_admission_door import _one_of, sequential_admit_value
from bindings.x_prep_native_inputs_door import _target
from bindings.scaler_native_door import _input, _ptr
from core.preprocessing_program import prep_program_stage as _stage
from x_prep.robust_native_run import _robust_fit_program
from bindings.p3_prep_fitted_unpack import prep_fitted_vector
from bindings.scaler_lifecycle_native_door import _entries, _public_result
from bindings.x_prep_native_owner_run import xprep_r3
from bindings.x_prep_native_owner_door import xprep_storage, xprep_take, xprep_values, xprep_run
from gemm.f16_widen import f16_bits_to_f32
from bindings.py_optional import py_or_none, py_present
from core.shared_slot import SharedSlot
from bindings.py_keys import py_key, py_op_add
from bindings.py_balanced import py_item


# ---- from bindings/robust_keys_unpack.mojo (inlined: its one importer is this file) ----
# SPDX-License-Identifier: Apache-2.0
# The robust family's held name keys and operation codes (bindings/py_keys.mojo
# has the form and its cost rule). Written by the boundary codemod of lane
# fx-boundary-core; a name the family starts to read is added in the three
# places below (field, None, fill), and the slot name keeps its V<n> until a
# wheel that holds the old layout could meet this one in a process (bump
# `V<n>` and the version together then: core/shared_slot.mojo).



struct RobustKeys(Defaultable, Movable):
    # Immutable interface names only: no model, state or user data.
    var ready: Bool
    var ops: PythonObject
    var k_ComplexWarning: PythonObject
    var k___dict__: PythonObject
    var k___mro__: PythonObject
    var k_modules: PythonObject
    var k_ndarray: PythonObject
    var k_numpy: PythonObject
    var k_scalar_types: PythonObject

    def __init__(out self):
        self.ready = False
        self.ops = PythonObject(None)
        self.k_ComplexWarning = PythonObject(None)
        self.k___dict__ = PythonObject(None)
        self.k___mro__ = PythonObject(None)
        self.k_modules = PythonObject(None)
        self.k_ndarray = PythonObject(None)
        self.k_numpy = PythonObject(None)
        self.k_scalar_types = PythonObject(None)


comptime _ROBUST_KEYS = SharedSlot[RobustKeys, "MojolearnRobustKeysV1"]


def _robust_keys_fill(nk: UnsafePointer[RobustKeys, MutUntrackedOrigin]) raises:
    # Once per process. `ready` is set last, so a fill that raised is run
    # again whole by the next call.
    nk[].k_ComplexWarning = py_key("ComplexWarning")
    nk[].k___dict__ = py_key("__dict__")
    nk[].k___mro__ = py_key("__mro__")
    nk[].k_modules = py_key("modules")
    nk[].k_ndarray = py_key("ndarray")
    nk[].k_numpy = py_key("numpy")
    nk[].k_scalar_types = py_key("scalar_types")
    var ops = Python.dict()
    nk[].ops = ops
    nk[].ready = True


@always_inline
def robust_keys() raises -> UnsafePointer[RobustKeys, MutUntrackedOrigin]:
    """The family's keys: one slot fetch and one flag test. Fetch once per
    function, outside loops."""
    var nk = _ROBUST_KEYS.get_or_create_ptr()
    if not nk[].ready:
        _robust_keys_fill(nk)
    return nk

def _truth(value: PythonObject, context: PythonObject) raises -> Bool:
    var nk = robust_keys()
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    if value is None or cls is b.bool or cls is b.int or cls is b.float or cls is b.complex or cls is b.str or cls is b.bytes or cls is b.list or cls is b.tuple or cls is b.dict or cls is b.range or _one_of(cls,py_item(context, nk[].k_scalar_types)):
        return Bool(py=value)
    raise Error("TypeError: RobustScaler options require native values; Python truth hooks cannot execute")


def _complex_warning(context: PythonObject) raises:
    var nk = robust_keys()
    var b = native_loaded_module("builtins")
    var module = py_get(py_item(context, nk[].k_modules),"numpy.exceptions")
    if module is None:
        module = py_get(py_item(context, nk[].k_modules),nk[].k_numpy)
    if module is None or py_type(module) is not py_type(native_loaded_module("sys")):
        raise Error("TypeError: complex quantiles require native NumPy warning metadata")
    var category = py_get(_warning_fields(py_fields(module)),nk[].k_ComplexWarning)
    if py_type(category) is not b.type:
        raise Error("TypeError: complex quantiles require a native warning category")
    var ancestry = py_call(py_item(b.type.__dict__, nk[].k___mro__).__get__,category)
    for i in range(len(ancestry)):  # driver-loop: admit warning class metadata before its native slot guard
        var mapping = py_call(py_item(b.type.__dict__, nk[].k___dict__).__get__,ancestry[i])
        var keys = py_call(b.list,mapping)
        for k in range(len(keys)):  # driver-loop: class keys may otherwise invoke Python equality during lookup
            if py_type(keys[k]) is not b.str:
                raise Error("TypeError: complex quantile warning keys require native strings")
    if py_type(py_call(b.type.__getattribute__,category,"__name__")) is not b.str:
        raise Error("TypeError: complex quantile warning names require native strings")
    classical_warning("Casting complex values to real discards the imaginary part",category,context,PythonObject(None))


def _number(value: PythonObject, context: PythonObject) raises -> Float64:
    var nk = robust_keys()
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    if cls is b.bool or cls is b.int or cls is b.float or cls is b.str or cls is b.bytes or _one_of(cls,py_item(context, nk[].k_scalar_types)):
        return Float64(py=py_call(b.float,value))
    if py_get(context,nk[].k_ndarray) is not None:
        var constructor = native_numpy_dtype(context)
        for code in ["S", "U", "g", "c8", "c16", "G"]:  # small-loop(params: supported native quantile scalar kinds)
            if cls is py_call(constructor,code).type:
                if code == "c8" or code == "c16" or code == "G":
                    _complex_warning(context)
                    return Float64(py=py_call(b.float,value.real))
                return Float64(py=py_call(b.float,value))
    raise Error("TypeError: RobustScaler quantiles require native numbers; Python conversion hooks cannot execute")


def _quantile_array(value: PythonObject, context: PythonObject) raises:
    # Exact ndarray alone is insufficient: user DTypes can own element
    # access. Match the descriptor's exact native class before dtype fields
    # or __getitem__. Object elements are admitted separately by _number.
    var dtype = value.dtype
    var constructor = native_numpy_dtype(context)
    var admitted = False
    for code in ["?","i1","u1","i2","u2","i4","u4","i8","u8","f2","f4","f8","g","c8","c16","G","S","U","O"]:  # small-loop(params: builtin native quantile descriptor kinds)
        if py_type(dtype) is py_type(py_call(constructor,code)):
            admitted = True
            break
    if not admitted:
        raise Error("TypeError: RobustScaler quantiles require native numeric, string or object storage; custom and structured dtypes cannot execute")
    if Int(py=value.ndim) != 1:
        raise Error("TypeError: RobustScaler quantile_range requires one-dimensional scalar metadata")
